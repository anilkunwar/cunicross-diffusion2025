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
# Visual / Global Settings
# ------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 14,
    "figure.dpi": 150
})

SOLUTION_DIR = os.path.join(os.path.dirname(__file__), "pinn_solutions")
os.makedirs(SOLUTION_DIR, exist_ok=True)
DIFFUSION_TYPES = ['crossdiffusion', 'cu_selfdiffusion', 'ni_selfdiffusion']

# ------------------------------
# I/O: load solutions
# ------------------------------
@st.cache_data
def load_solutions(solution_dir):
    solutions = []
    for fname in os.listdir(solution_dir):
        if fname.endswith(".pkl"):
            try:
                with open(os.path.join(solution_dir, fname), "rb") as f:
                    sol = pickle.load(f)
                match = re.match(r"solution_(\w+)_ly_([\d.]+)_tmax_([\d.]+)\.pkl", fname)
                if not match:
                    continue
                raw_type, ly_val, _ = match.groups()
                type_map = {
                    'cross': 'crossdiffusion',
                    'cu_self': 'cu_selfdiffusion',
                    'ni_self': 'ni_selfdiffusion'
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

# ------------------------------
# Compute fluxes and gradients
# ------------------------------
def compute_fluxes_and_grads(c1_preds, c2_preds, x_coords, y_coords, params):
    """
    c1_preds, c2_preds: lists or arrays of shape (Nt, Ny, Nx) or (Nt, Nx, Ny)
    This function expects the same orientation used elsewhere (here we follow user's convention).
    Returns lists of J1_preds (per time) and J2_preds etc.
    J*_preds contains [Jx, Jy] arrays; for our purposes Jy is under index 1.
    """
    D11, D12, D21, D22 = params['D11'], params['D12'], params['D21'], params['D22']
    # assume x_coords and y_coords are 1D arrays; dx,dy from first two entries
    dx = x_coords[1] - x_coords[0]
    dy = y_coords[1] - y_coords[0]

    J1_preds, J2_preds, grad_c1_y, grad_c2_y = [], [], [], []
    for c1, c2 in zip(c1_preds, c2_preds):
        # gradient w.r.t y (axis=0 if data is (Ny, Nx) or (rows,cols))
        grad_c1_y_i = np.gradient(c1, dy, axis=0)
        grad_c2_y_i = np.gradient(c2, dy, axis=0)
        # compute flux y-component using cross terms
        J1_y = -(D11 * grad_c1_y_i + D12 * grad_c2_y_i)
        J2_y = -(D21 * grad_c1_y_i + D22 * grad_c2_y_i)
        # store J as [None, Jy] to keep same interface as your previous code
        J1_preds.append([None, J1_y])
        J2_preds.append([None, J2_y])
        grad_c1_y.append(grad_c1_y_i)
        grad_c2_y.append(grad_c2_y_i)

    return J1_preds, J2_preds, grad_c1_y, grad_c2_y

# ------------------------------
# Detect uphill: non-absolute maxima
# ------------------------------
def detect_uphill(solution, time_index):
    """
    Returns:
      uphill_cu_mask (bool array),
      uphill_ni_mask (bool array),
      uphill_prod_cu_pos (float array with positive values where uphill else 0),
      uphill_prod_ni_pos,
      max_pos_cu (scalar, 0 if none),
      max_pos_ni
    """
    J1_y = solution['J1_preds'][time_index][1]   # shape (Ny, Nx)
    grad_c1_y = solution['grad_c1_y'][time_index]
    J2_y = solution['J2_preds'][time_index][1]
    grad_c2_y = solution['grad_c2_y'][time_index]

    # elementwise product
    prod_cu = J1_y * grad_c1_y
    prod_ni = J2_y * grad_c2_y

    # uphill masks (strictly positive indicates uphill)
    uphill_cu = prod_cu > 0
    uphill_ni = prod_ni > 0

    # non-absolute uphill product: keep only positive values, zero elsewhere
    uphill_prod_cu_pos = np.where(uphill_cu, prod_cu, 0.0)
    uphill_prod_ni_pos = np.where(uphill_ni, prod_ni, 0.0)

    # maxima of non-absolute (positive) product (0 if none)
    if np.any(uphill_cu):
        max_pos_cu = float(np.max(uphill_prod_cu_pos))
    else:
        max_pos_cu = 0.0

    if np.any(uphill_ni):
        max_pos_ni = float(np.max(uphill_prod_ni_pos))
    else:
        max_pos_ni = 0.0

    # uphill fraction (useful)
    total_cells = prod_cu.size
    frac_uphill_cu = float(np.count_nonzero(uphill_cu) / total_cells)
    frac_uphill_ni = float(np.count_nonzero(uphill_ni) / total_cells)

    return (uphill_cu, uphill_ni,
            uphill_prod_cu_pos, uphill_prod_ni_pos,
            max_pos_cu, max_pos_ni,
            frac_uphill_cu, frac_uphill_ni)

# ------------------------------
# Plotting utilities
# ------------------------------
def plot_flux_vs_gradient(solution, time_index):
    J1_y = solution['J1_preds'][time_index][1].flatten()
    J2_y = solution['J2_preds'][time_index][1].flatten()
    grad_c1_y = solution['grad_c1_y'][time_index].flatten()
    grad_c2_y = solution['grad_c2_y'][time_index].flatten()

    uphill_cu = J1_y * grad_c1_y > 0
    uphill_ni = J2_y * grad_c2_y > 0

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Cu Flux vs ∇c", "Ni Flux vs ∇c"))

    fig.add_trace(go.Scatter(x=grad_c1_y, y=J1_y, mode='markers',
                             name="Cu (all)", marker=dict(color='gray', opacity=0.5, size=5)), row=1, col=1)
    if uphill_cu.any():
        fig.add_trace(go.Scatter(x=grad_c1_y[uphill_cu], y=J1_y[uphill_cu],
                                 mode='markers', name="Uphill (Cu)", marker=dict(color='red', size=6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=grad_c2_y, y=J2_y, mode='markers',
                             name="Ni (all)", marker=dict(color='gray', opacity=0.5, size=5)), row=1, col=2)
    if uphill_ni.any():
        fig.add_trace(go.Scatter(x=grad_c2_y[uphill_ni], y=J2_y[uphill_ni],
                                 mode='markers', name="Uphill (Ni)", marker=dict(color='blue', size=6)), row=1, col=2)

    fig.update_layout(height=420, width=940, template='simple_white',
                      title="Flux vs Concentration Gradient (Fickian vs Uphill)",
                      font=dict(family="Serif", size=14))
    st.plotly_chart(fig, use_container_width=False)

    st.caption(
        "🧠 **Physical Meaning:** The flux–gradient curve compares y-component of flux (J_y) vs local gradient ∇c_y. "
        "Points where J·∇c > 0 (highlighted) indicate **uphill flow**, i.e. flux directed against the concentration gradient — "
        "a signature of cross-diffusion or non-Fickian coupling. The maxima reported below are the largest **positive** (non-absolute) "
        "local values of J·∇c, which quantify the strongest uphill driving at that time."
    )

def plot_uphill_regions_heatmap(solution, time_index, colorscale='RdBu', fig_width=10, fig_height=5):
    x_coords = solution['X'][:,0] if solution['X'].ndim==2 else solution['X']
    y_coords = solution['Y'][0,:] if solution['Y'].ndim==2 else solution['Y']
    t_val = solution['times'][time_index]
    (uphill_cu, uphill_ni,
     uphill_prod_cu_pos, uphill_prod_ni_pos,
     max_pos_cu, max_pos_ni,
     frac_cu, frac_ni) = detect_uphill(solution, time_index)

    # Plot with Plotly heatmap for positive (uphill) product only
    fig = make_subplots(rows=1, cols=2, subplot_titles=[f"Cu Uphill (max={max_pos_cu:.3e})", f"Ni Uphill (max={max_pos_ni:.3e})"])

    # we pass z as 2D arrays
    z1 = uphill_prod_cu_pos
    z2 = uphill_prod_ni_pos

    fig.add_trace(go.Heatmap(x=x_coords, y=y_coords, z=z1, colorscale=colorscale, colorbar=dict(title="J·∇c")), row=1, col=1)
    fig.add_trace(go.Heatmap(x=x_coords, y=y_coords, z=z2, colorscale=colorscale, colorbar=dict(title="J·∇c")), row=1, col=2)

    fig.update_layout(height=int(fig_height*100), width=int(fig_width*100),
                      title=f"Uphill Diffusion (positive J·∇c) @ t={t_val:.2f}s",
                      template='plotly_white', font=dict(size=14))
    st.plotly_chart(fig, use_container_width=True)

    # Show numeric summary for this time
    st.markdown(f"- **Max (positive) J·∇c (Cu):** {max_pos_cu:.3e}")
    st.markdown(f"- **Max (positive) J·∇c (Ni):** {max_pos_ni:.3e}")
    st.markdown(f"- **Uphill fraction (Cu):** {frac_cu*100:.2f}% of grid points")
    st.markdown(f"- **Uphill fraction (Ni):** {frac_ni*100:.2f}% of grid points")

    return max_pos_cu, max_pos_ni, frac_cu, frac_ni

def plot_uphill_over_time(solution):
    """
    Plots the time evolution of the positive maxima (non-absolute) of J·∇c for Cu and Ni.
    """
    times = np.array(solution['times'])
    max_pos_cu_list = []
    max_pos_ni_list = []
    frac_cu_list = []
    frac_ni_list = []

    Nt = len(times)
    for t_idx in range(Nt):
        (_, _, _, _, max_pos_cu, max_pos_ni, frac_cu, frac_ni) = detect_uphill(solution, t_idx)
        max_pos_cu_list.append(max_pos_cu)
        max_pos_ni_list.append(max_pos_ni)
        frac_cu_list.append(frac_cu)
        frac_ni_list.append(frac_ni)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=max_pos_cu_list, mode='lines+markers', name='Max positive J·∇c (Cu)', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=times, y=max_pos_ni_list, mode='lines+markers', name='Max positive J·∇c (Ni)', line=dict(color='blue')))

    fig.update_layout(height=420, width=940,
                      title="Temporal Evolution of Global Positive Max (J·∇c)",
                      xaxis_title="Time (s)",
                      yaxis_title="Max positive J·∇c",
                      template="simple_white", font=dict(family="Serif", size=14))
    st.plotly_chart(fig, use_container_width=False)

    st.caption(
        "🧠 **Meaning:** This plot shows the largest *positive* local values of J·∇c over time (i.e. strongest uphill drivers). "
        "Peaks indicate times when uphill transport is locally most intense. The fraction plots below show the fraction of grid "
        "cells that are uphill at each time (a measure of spatial extent)."
    )

    # Also show uphill fraction evolution as small subplots
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=times, y=frac_cu_list, mode='lines', name='Uphill fraction Cu', line=dict(color='salmon')))
    fig2.add_trace(go.Scatter(x=times, y=frac_ni_list, mode='lines', name='Uphill fraction Ni', line=dict(color='lightblue')))
    fig2.update_layout(height=260, width=940, title="Uphill fraction vs Time", xaxis_title="Time (s)", yaxis_title="Fraction of grid points")
    st.plotly_chart(fig2, use_container_width=False)

    return np.array(max_pos_cu_list), np.array(max_pos_ni_list), np.array(frac_cu_list), np.array(frac_ni_list)

# ------------------------------
# Summary DataFrame across all solutions
# ------------------------------
@st.cache_data
def compute_summary_dataframe(all_solutions, time_index_for_summary=0):
    """
    For each solution: compute max positive J·∇c for Cu & Ni (at specified time index)
    and the uphill fraction.
    """
    rows = []
    for s in all_solutions:
        # compute fluxes if missing
        if 'J1_preds' not in s:
            J1, J2, grad_c1, grad_c2 = compute_fluxes_and_grads(s['c1_preds'], s['c2_preds'], s['X'][:,0], s['Y'][0,:], s['params'])
            s.update({'J1_preds': J1, 'J2_preds': J2, 'grad_c1_y': grad_c1, 'grad_c2_y': grad_c2})
        try:
            (_, _, _, _, max_pos_cu, max_pos_ni, frac_cu, frac_ni) = detect_uphill(s, time_index_for_summary)
        except Exception:
            max_pos_cu, max_pos_ni, frac_cu, frac_ni = 0.0, 0.0, 0.0, 0.0
        rows.append({
            "filename": s.get('filename', ''),
            "diffusion_type": s.get('diffusion_type', ''),
            "Ly (μm)": s.get('Ly_parsed', np.nan),
            "time_index": time_index_for_summary,
            "max_pos_JdotGrad_Cu": max_pos_cu,
            "max_pos_JdotGrad_Ni": max_pos_ni,
            "uphill_frac_Cu": frac_cu,
            "uphill_frac_Ni": frac_ni
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(['diffusion_type', 'Ly (μm)'])
    return df

# ------------------------------
# Main App
# ------------------------------
def main():
    st.title("📈 Diffusion Analysis: Uphill (positive) J·∇c maxima")

    # Load
    solutions = load_solutions(SOLUTION_DIR)
    if not solutions:
        st.warning("No PINN solution pickles found in 'pinn_solutions/'.")
        return

    # Sidebar controls
    st.sidebar.header("Controls")
    diff_type = st.sidebar.selectbox("Diffusion Type", DIFFUSION_TYPES)
    ly_values = sorted(set(s['Ly_parsed'] for s in solutions if s['diffusion_type'] == diff_type))
    if not ly_values:
        st.error("No solutions for selected diffusion type.")
        return
    ly_target = st.sidebar.select_slider("Select Ly (μm)", options=ly_values, value=ly_values[0])
    time_index = st.sidebar.slider("Select Time Index", 0, 49, 0)
    downsample = st.sidebar.slider("Downsample heatmap", 1, 5, 1)
    colorscale = st.sidebar.selectbox("Heatmap colorscale", ['RdBu', 'Viridis', 'Plasma', 'Cividis'])

    # pick solution
    solution = next((s for s in solutions if s['diffusion_type']==diff_type and abs(s['Ly_parsed']-ly_target)<1e-8), None)
    if solution is None:
        st.error("No matching solution for the chosen Ly and diffusion type.")
        return

    # compute fluxes if missing
    if 'J1_preds' not in solution:
        J1, J2, grad_c1, grad_c2 = compute_fluxes_and_grads(solution['c1_preds'], solution['c2_preds'],
                                                           solution['X'][:,0], solution['Y'][0,:], solution['params'])
        solution.update({'J1_preds': J1, 'J2_preds': J2, 'grad_c1_y': grad_c1, 'grad_c2_y': grad_c2})

    st.subheader("1️⃣ Flux vs Gradient (instantaneous)")
    plot_flux_vs_gradient(solution, time_index)

    st.subheader("2️⃣ Uphill diffusion heatmap (positive J·∇c only)")
    max_pos_cu, max_pos_ni, frac_cu, frac_ni = plot_uphill_regions_heatmap(solution, time_index, colorscale, fig_width=10, fig_height=5)

    st.subheader("3️⃣ Temporal evolution of global positive maxima")
    max_pos_cu_series, max_pos_ni_series, frac_cu_series, frac_ni_series = plot_uphill_over_time(solution)

    # Summary table across all solutions (use current time_index for consistency)
    st.subheader("4️⃣ Summary table (positive maxima at selected time index)")
    df_summary = compute_summary_dataframe(solutions, time_index_for_summary=time_index)
    st.dataframe(df_summary.style.format({
        "max_pos_JdotGrad_Cu": "{:.3e}",
        "max_pos_JdotGrad_Ni": "{:.3e}",
        "uphill_frac_Cu": "{:.2%}",
        "uphill_frac_Ni": "{:.2%}"
    }), use_container_width=True)

    st.download_button("Download summary CSV", df_summary.to_csv(index=False), file_name="uphill_summary.csv", mime="text/csv")

if __name__ == "__main__":
    main()
