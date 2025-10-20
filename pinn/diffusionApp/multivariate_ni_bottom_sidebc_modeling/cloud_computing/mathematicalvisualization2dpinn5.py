import os
import pickle
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import cdist
import re
import plotly.express as px

# ------------------------------
# Global Settings
# ------------------------------
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "mathtext.fontset": "dejavusans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10
})

SOLUTION_DIR = os.path.join(os.path.dirname(__file__), "pinn_solutions")
os.makedirs(SOLUTION_DIR, exist_ok=True)

DIFFUSION_TYPES = ['crossdiffusion', 'cu_selfdiffusion', 'ni_selfdiffusion']

# ------------------------------
# Colormap Definitions
# ------------------------------

def get_plotly_colormaps():
    """Get organized categories of Plotly colormaps"""
    categories = {
        "Sequential (Perceptually Uniform)": [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo'
        ],
        "Diverging": [
            'rdbu', 'rdylbu', 'spectral', 'balance', 'curl'
        ]
    }
    return categories

def get_matplotlib_colormaps():
    """Get organized categories of Matplotlib colormaps"""
    categories = {
        "Perceptually Uniform Sequential": [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis'
        ],
        "Diverging": [
            'coolwarm', 'RdYlBu', 'PiYG', 'Spectral', 'bwr'
        ]
    }
    return categories

# ------------------------------
# Utility Functions
# ------------------------------

@st.cache_data
def load_solutions(solution_dir):
    solutions, metadata, load_logs = [], [], []
    for fname in os.listdir(solution_dir):
        if not fname.endswith(".pkl"):
            continue
        filepath = os.path.join(solution_dir, fname)
        try:
            with open(filepath, "rb") as f:
                sol = pickle.load(f)
            required = ['params', 'X', 'Y', 'c1_preds', 'c2_preds', 'times']
            if not all(key in sol for key in required):
                continue
            match = re.match(r"solution_(\w+)_ly_([\d.]+)_tmax_([\d.]+)\.pkl", fname)
            if not match:
                continue
            raw_type, ly_val, _ = match.groups()
            diff_type = raw_type.lower()
            type_map = {
                'cross': 'crossdiffusion',
                'crossdiffusion': 'crossdiffusion',
                'cu_self': 'cu_selfdiffusion',
                'cu_selfdiffusion': 'cu_selfdiffusion',
                'ni_self': 'ni_selfdiffusion',
                'ni_selfdiffusion': 'ni_selfdiffusion'
            }
            diff_type = type_map.get(diff_type, diff_type)
            sol.update({
                'diffusion_type': diff_type,
                'Ly_parsed': float(ly_val),
                'filename': fname
            })
            solutions.append(sol)
        except Exception as e:
            load_logs.append(f"{fname}: ✗ Failed - {str(e)}")
    return solutions, metadata, load_logs


def compute_fluxes_and_grads(c1_preds, c2_preds, x_coords, y_coords, params):
    """Compute flux components and gradients."""
    D11, D12, D21, D22 = params['D11'], params['D12'], params['D21'], params['D22']
    dx, dy = x_coords[1]-x_coords[0], y_coords[1]-y_coords[0]
    J1_preds, J2_preds, grad_c1_y, grad_c2_y = [], [], [], []
    for c1, c2 in zip(c1_preds, c2_preds):
        grad_c1_x, grad_c1_y_i = np.gradient(c1, dx, axis=1), np.gradient(c1, dy, axis=0)
        grad_c2_x, grad_c2_y_i = np.gradient(c2, dx, axis=1), np.gradient(c2, dy, axis=0)
        J1_preds.append([-(D11*grad_c1_x + D12*grad_c2_x), -(D11*grad_c1_y_i + D12*grad_c2_y_i)])
        J2_preds.append([-(D21*grad_c1_x + D22*grad_c2_x), -(D21*grad_c1_y_i + D22*grad_c2_y_i)])
        grad_c1_y.append(grad_c1_y_i)
        grad_c2_y.append(grad_c2_y_i)
    return J1_preds, J2_preds, grad_c1_y, grad_c2_y


def detect_uphill(solution, time_index):
    """Detect uphill regions and compute maxima."""
    J1_y = solution['J1_preds'][time_index][1]
    grad_c1_y = solution['grad_c1_y'][time_index]
    J2_y = solution['J2_preds'][time_index][1]
    grad_c2_y = solution['grad_c2_y'][time_index]

    # Identify uphill (positive dot product)
    uphill_cu = J1_y * grad_c1_y > 0
    uphill_ni = J2_y * grad_c2_y > 0

    # Compute magnitude of J_i × ∇c_i
    uphill_product_cu = np.abs(J1_y * grad_c1_y) * uphill_cu
    uphill_product_ni = np.abs(J2_y * grad_c2_y) * uphill_ni

    # --- NEW: Compute maxima for Cu and Ni ---
    max_uphill_cu = np.max(uphill_product_cu)
    max_uphill_ni = np.max(uphill_product_ni)

    return uphill_cu, uphill_ni, uphill_product_cu, uphill_product_ni, max_uphill_cu, max_uphill_ni


# ------------------------------
# Plotting Functions
# ------------------------------

def plot_uphill_regions(solution, time_index, downsample=2, colorscale='RdBu', 
                        fig_width=10, fig_height=5, font_size=14):
    x_coords = solution['X'][:, 0]
    y_coords = solution['Y'][0, :]
    t_val = solution['times'][time_index]
    diff_type = solution['diffusion_type']

    uphill_cu, uphill_ni, uphill_prod_cu, uphill_prod_ni, max_cu, max_ni = detect_uphill(solution, time_index)

    ds = max(1, downsample)
    x_idx = np.arange(0, len(x_coords), ds)
    y_idx = np.arange(0, len(y_coords), ds)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[f"Cu Uphill (max={max_cu:.3e})", f"Ni Uphill (max={max_ni:.3e})"])

    for i, (uphill_prod, label) in enumerate([(uphill_prod_cu, "Cu"), (uphill_prod_ni, "Ni")]):
        z_plot = uphill_prod[np.ix_(y_idx, x_idx)]
        fig.add_trace(go.Heatmap(
            x=x_coords[x_idx],
            y=y_coords[y_idx],
            z=z_plot,
            colorscale=colorscale,
            colorbar=dict(title="|J·∇c|"),
            zsmooth='best'
        ), row=1, col=i+1)
        fig.update_xaxes(title_text="x (μm)", row=1, col=i+1)
        fig.update_yaxes(title_text="y (μm)", row=1, col=i+1)

    fig.update_layout(
        height=int(fig_height * 100),
        width=int(fig_width * 100),
        title=f"Uphill Diffusion Magnitude — {diff_type.replace('_', ' ')} @ t={t_val:.1f}s",
        template='plotly_white',
        font=dict(size=font_size)
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- NEW: Output maxima results below the plot ---
    st.markdown(f"**Max Uphill Cu (|J·∇c|):** {max_cu:.3e}")
    st.markdown(f"**Max Uphill Ni (|J·∇c|):** {max_ni:.3e}")

    return max_cu, max_ni


# ------------------------------
# Main App
# ------------------------------
def main():
    st.title("Theoretical Assessment of Diffusion Solutions")

    solutions, metadata, load_logs = load_solutions(SOLUTION_DIR)
    if not solutions:
        st.error("No valid solution files found.")
        return

    # Sidebar
    st.sidebar.header("Simulation Parameters")
    diff_type = st.sidebar.selectbox("Diffusion Type", DIFFUSION_TYPES)
    available_lys = sorted(set(s['Ly_parsed'] for s in solutions if s['diffusion_type'] == diff_type))
    ly_target = st.sidebar.select_slider("Ly (μm)", options=available_lys)
    time_index = st.sidebar.slider("Time Index", 0, 49, 49)
    downsample = st.sidebar.slider("Downsample", 1, 5, 2)
    colorscale = st.sidebar.selectbox("Colorscale", ['RdBu', 'Viridis', 'Plasma', 'Inferno', 'Cividis'])

    # Select solution
    solution = next((s for s in solutions if s['diffusion_type'] == diff_type and abs(s['Ly_parsed'] - ly_target) < 1e-4), None)
    if solution is None:
        st.error(f"No solution for {diff_type} with Ly={ly_target}")
        return

    # Compute fluxes and gradients
    J1, J2, grad_c1, grad_c2 = compute_fluxes_and_grads(solution['c1_preds'], solution['c2_preds'],
                                                       solution['X'][:, 0], solution['Y'][0, :],
                                                       solution['params'])
    solution.update({'J1_preds': J1, 'J2_preds': J2, 'grad_c1_y': grad_c1, 'grad_c2_y': grad_c2})

    # Uphill detection and maxima output
    st.subheader("Uphill Diffusion Detection and Maxima")
    max_cu, max_ni = plot_uphill_regions(solution, time_index, downsample, colorscale)

    st.success(f"✅ Max Uphill Cu = {max_cu:.3e} | Max Uphill Ni = {max_ni:.3e}")

if __name__ == "__main__":
    main()
