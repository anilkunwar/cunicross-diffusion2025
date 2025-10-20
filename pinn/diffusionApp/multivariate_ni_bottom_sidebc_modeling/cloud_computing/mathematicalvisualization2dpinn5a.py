import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

# ------------------------------
# Global Settings
# ------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 150
})

SOLUTION_DIR = os.path.join(os.path.dirname(__file__), "pinn_solutions")
os.makedirs(SOLUTION_DIR, exist_ok=True)
DIFFUSION_TYPES = ['crossdiffusion', 'cu_selfdiffusion', 'ni_selfdiffusion']

# ------------------------------
# Utility Functions
# ------------------------------

@st.cache_data
def load_solutions(solution_dir):
    """Load valid .pkl PINN solution files."""
    solutions = []
    for fname in os.listdir(solution_dir):
        if not fname.endswith(".pkl"):
            continue
        filepath = os.path.join(solution_dir, fname)
        try:
            with open(filepath, "rb") as f:
                sol = pickle.load(f)
            match = re.match(r"solution_(\w+)_ly_([\d.]+)_tmax_([\d.]+)\.pkl", fname)
            if not match:
                continue
            raw_type, ly_val, _ = match.groups()
            type_map = {
                'cross': 'crossdiffusion',
                'crossdiffusion': 'crossdiffusion',
                'cu_self': 'cu_selfdiffusion',
                'cu_selfdiffusion': 'cu_selfdiffusion',
                'ni_self': 'ni_selfdiffusion',
                'ni_selfdiffusion': 'ni_selfdiffusion'
            }
            diff_type = type_map.get(raw_type.lower(), raw_type)
            sol.update({
                'diffusion_type': diff_type,
                'Ly_parsed': float(ly_val),
                'filename': fname
            })
            solutions.append(sol)
        except Exception:
            continue
    return solutions


def compute_fluxes_and_grads(c1_preds, c2_preds, x_coords, y_coords, params):
    """Compute fluxes and gradients along x and y."""
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
    """Detect uphill diffusion and compute maxima."""
    J1_y = solution['J1_preds'][time_index][1]
    grad_c1_y = solution['grad_c1_y'][time_index]
    J2_y = solution['J2_preds'][time_index][1]
    grad_c2_y = solution['grad_c2_y'][time_index]

    uphill_cu = J1_y * grad_c1_y > 0
    uphill_ni = J2_y * grad_c2_y > 0

    uphill_product_cu = np.abs(J1_y * grad_c1_y) * uphill_cu
    uphill_product_ni = np.abs(J2_y * grad_c2_y) * uphill_ni

    max_uphill_cu = np.max(uphill_product_cu)
    max_uphill_ni = np.max(uphill_product_ni)

    return uphill_cu, uphill_ni, uphill_product_cu, uphill_product_ni, max_uphill_cu, max_uphill_ni


# ------------------------------
# Plotting Functions
# ------------------------------

def plot_flux_vs_gradient(solution, time_index, fig_height=400, fig_width=900):
    """Plot flux vs gradient with uphill overlay."""
    J1_y = solution['J1_preds'][time_index][1].flatten()
    J2_y = solution['J2_preds'][time_index][1].flatten()
    grad_c1_y = solution['grad_c1_y'][time_index].flatten()
    grad_c2_y = solution['grad_c2_y'][time_index].flatten()
    uphill_cu = J1_y * grad_c1_y > 0
    uphill_ni = J2_y * grad_c2_y > 0

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Cu Flux vs ∇c", "Ni Flux vs ∇c"))

    # Cu subplot
    fig.add_trace(go.Scatter(
        x=grad_c1_y, y=J1_y,
        mode='markers', name="Cu Diffusion",
        marker=dict(color='gray', size=5, opacity=0.6)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=grad_c1_y[uphill_cu], y=J1_y[uphill_cu],
        mode='markers', name="Uphill Region",
        marker=dict(color='red', size=6, symbol='diamond')
    ), row=1, col=1)

    # Ni subplot
    fig.add_trace(go.Scatter(
        x=grad_c2_y, y=J2_y,
        mode='markers', name="Ni Diffusion",
        marker=dict(color='gray', size=5, opacity=0.6)
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=grad_c2_y[uphill_ni], y=J2_y[uphill_ni],
        mode='markers', name="Uphill Region",
        marker=dict(color='blue', size=6, symbol='diamond')
    ), row=1, col=2)

    fig.update_xaxes(title_text="∇c (1/μm)", row=1, col=1)
    fig.update_yaxes(title_text="J_y (a.u.)", row=1, col=1)
    fig.update_xaxes(title_text="∇c (1/μm)", row=1, col=2)
    fig.update_yaxes(title_text="J_y (a.u.)", row=1, col=2)

    fig.update_layout(
        height=fig_height,
        width=fig_width,
        template='simple_white',
        title=dict(text="Flux vs Concentration Gradient", x=0.5),
        legend=dict(orientation='h', y=-0.2),
        font=dict(family="Serif", size=14)
    )
    st.plotly_chart(fig, use_container_width=False)


def plot_uphill_heatmap(solution, time_index, colorscale='RdBu', downsample=2):
    """Plot uphill diffusion magnitudes as heatmaps."""
    x_coords = solution['X'][:, 0]
    y_coords = solution['Y'][0, :]
    uphill_cu, uphill_ni, uphill_prod_cu, uphill_prod_ni, max_cu, max_ni = detect_uphill(solution, time_index)

    ds = max(1, downsample)
    x_idx, y_idx = np.arange(0, len(x_coords), ds), np.arange(0, len(y_coords), ds)
    diff_type = solution['diffusion_type']
    t_val = solution['times'][time_index]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[f"Cu Uphill (max={max_cu:.2e})", f"Ni Uphill (max={max_ni:.2e})"])

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

    fig.update_layout(
        height=500,
        width=900,
        title=f"Uphill Diffusion Map — {diff_type.replace('_', ' ')} @ t={t_val:.1f}s",
        template='plotly_white',
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=False)

    return max_cu, max_ni


# ------------------------------
# Main App
# ------------------------------
def main():
    st.title("📈 Advanced Diffusion Analysis with Uphill Detection")

    solutions = load_solutions(SOLUTION_DIR)
    if not solutions:
        st.warning("No valid PINN solutions found in 'pinn_solutions/'.")
        return

    # Sidebar
    st.sidebar.header("Controls")
    diff_type = st.sidebar.selectbox("Select Diffusion Type", DIFFUSION_TYPES)
    ly_values = sorted(set(s['Ly_parsed'] for s in solutions if s['diffusion_type'] == diff_type))
    ly_target = st.sidebar.select_slider("Select Ly (μm)", options=ly_values)
    time_index = st.sidebar.slider("Select Time Index", 0, 49, 49)
    downsample = st.sidebar.slider("Downsample", 1, 5, 2)
    colorscale = st.sidebar.selectbox("Heatmap Colorscale", ['RdBu', 'Viridis', 'Plasma', 'Cividis'])

    # Filter solution
    solution = next((s for s in solutions if s['diffusion_type'] == diff_type and abs(s['Ly_parsed'] - ly_target) < 1e-4), None)
    if not solution:
        st.error("No matching solution found.")
        return

    # Compute fluxes
    J1, J2, grad_c1, grad_c2 = compute_fluxes_and_grads(solution['c1_preds'], solution['c2_preds'],
                                                       solution['X'][:, 0], solution['Y'][0, :], solution['params'])
    solution.update({'J1_preds': J1, 'J2_preds': J2, 'grad_c1_y': grad_c1, 'grad_c2_y': grad_c2})

    st.subheader("1️⃣ Flux vs Concentration Gradient")
    plot_flux_vs_gradient(solution, time_index)

    st.subheader("2️⃣ Uphill Diffusion Heatmap")
    max_cu, max_ni = plot_uphill_heatmap(solution, time_index, colorscale, downsample)
    st.success(f"Max Uphill Cu = {max_cu:.3e} | Max Uphill Ni = {max_ni:.3e}")

    # ---- DataFrame Summary ----
    st.subheader("3️⃣ Summary Table: Max Uphill Magnitudes Across All Simulations")

    summary_data = []
    for s in solutions:
        J1, J2, grad_c1, grad_c2 = compute_fluxes_and_grads(s['c1_preds'], s['c2_preds'],
                                                            s['X'][:, 0], s['Y'][0, :], s['params'])
        s.update({'J1_preds': J1, 'J2_preds': J2, 'grad_c1_y': grad_c1, 'grad_c2_y': grad_c2})
        _, _, _, _, max_cu_i, max_ni_i = detect_uphill(s, time_index)
        summary_data.append({
            "Diffusion Type": s['diffusion_type'],
            "Ly (μm)": s['Ly_parsed'],
            "Max Uphill Cu (|J·∇c|)": max_cu_i,
            "Max Uphill Ni (|J·∇c|)": max_ni_i
        })

    df_summary = pd.DataFrame(summary_data).sort_values(by=["Diffusion Type", "Ly (μm)"])
    st.dataframe(df_summary, use_container_width=True, height=350)

    st.download_button("💾 Download Summary CSV", df_summary.to_csv(index=False), "uphill_summary.csv", "text/csv")


if __name__ == "__main__":
    main()
