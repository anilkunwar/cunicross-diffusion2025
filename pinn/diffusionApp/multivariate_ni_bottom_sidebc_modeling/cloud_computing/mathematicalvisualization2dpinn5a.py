import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 14,
    "figure.dpi": 150
})

SOLUTION_DIR = os.path.join(os.path.dirname(__file__), "pinn_solutions")
os.makedirs(SOLUTION_DIR, exist_ok=True)
DIFFUSION_TYPES = ['crossdiffusion', 'cu_selfdiffusion', 'ni_selfdiffusion']

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


def compute_fluxes_and_grads(c1_preds, c2_preds, x_coords, y_coords, params):
    D11, D12, D21, D22 = params['D11'], params['D12'], params['D21'], params['D22']
    dx, dy = x_coords[1]-x_coords[0], y_coords[1]-y_coords[0]
    J1_preds, J2_preds, grad_c1_y, grad_c2_y = [], [], [], []
    for c1, c2 in zip(c1_preds, c2_preds):
        grad_c1_y_i = np.gradient(c1, dy, axis=0)
        grad_c2_y_i = np.gradient(c2, dy, axis=0)
        J1_preds.append([None, -(D11*grad_c1_y_i + D12*grad_c2_y_i)])
        J2_preds.append([None, -(D21*grad_c1_y_i + D22*grad_c2_y_i)])
        grad_c1_y.append(grad_c1_y_i)
        grad_c2_y.append(grad_c2_y_i)
    return J1_preds, J2_preds, grad_c1_y, grad_c2_y


def detect_uphill(solution, time_index):
    J1_y = solution['J1_preds'][time_index][1]
    grad_c1_y = solution['grad_c1_y'][time_index]
    J2_y = solution['J2_preds'][time_index][1]
    grad_c2_y = solution['grad_c2_y'][time_index]

    uphill_cu = J1_y * grad_c1_y > 0
    uphill_ni = J2_y * grad_c2_y > 0

    uphill_prod_cu = np.abs(J1_y * grad_c1_y) * uphill_cu
    uphill_prod_ni = np.abs(J2_y * grad_c2_y) * uphill_ni

    return uphill_cu, uphill_ni, uphill_prod_cu, uphill_prod_ni, np.max(uphill_prod_cu), np.max(uphill_prod_ni)


def plot_flux_vs_gradient(solution, time_index):
    J1_y = solution['J1_preds'][time_index][1].flatten()
    J2_y = solution['J2_preds'][time_index][1].flatten()
    grad_c1_y = solution['grad_c1_y'][time_index].flatten()
    grad_c2_y = solution['grad_c2_y'][time_index].flatten()
    uphill_cu = J1_y * grad_c1_y > 0
    uphill_ni = J2_y * grad_c2_y > 0

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Cu Flux vs ∇c", "Ni Flux vs ∇c"))

    fig.add_trace(go.Scatter(x=grad_c1_y, y=J1_y, mode='markers',
                             name="Cu Diffusion", marker=dict(color='gray', opacity=0.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=grad_c1_y[uphill_cu], y=J1_y[uphill_cu],
                             mode='markers', name="Uphill Cu", marker=dict(color='red', size=6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=grad_c2_y, y=J2_y, mode='markers',
                             name="Ni Diffusion", marker=dict(color='gray', opacity=0.6)), row=1, col=2)
    fig.add_trace(go.Scatter(x=grad_c2_y[uphill_ni], y=J2_y[uphill_ni],
                             mode='markers', name="Uphill Ni", marker=dict(color='blue', size=6)), row=1, col=2)

    fig.update_layout(height=400, width=900, template='simple_white',
                      title="Flux vs Concentration Gradient (Fickian vs Uphill Diffusion)",
                      font=dict(family="Serif", size=14))
    st.plotly_chart(fig, use_container_width=False)

    st.caption(
        "🧠 **Physical Meaning:** The flux–gradient relationship indicates whether diffusion follows Fick's law. "
        "Regions where the red or blue points deviate into the 'uphill' zone (J·∇c > 0) signify mass transport "
        "against the concentration gradient — a hallmark of cross-diffusion or chemical coupling effects."
    )


def plot_uphill_over_time(solution):
    """Plot time evolution of global max |J·∇c| for Cu and Ni."""
    times = solution['times']
    max_cu, max_ni = [], []

    for t_idx in range(len(times)):
        _, _, _, _, max_cu_i, max_ni_i = detect_uphill(solution, t_idx)
        max_cu.append(max_cu_i)
        max_ni.append(max_ni_i)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=max_cu, mode='lines+markers',
                             name='Cu', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=times, y=max_ni, mode='lines+markers',
                             name='Ni', line=dict(color='blue')))

    fig.update_layout(height=400, width=900,
                      title="Temporal Evolution of Global Max |J·∇c|",
                      xaxis_title="Time (s)",
                      yaxis_title="Max |J·∇c| (a.u.)",
                      template="simple_white",
                      font=dict(family="Serif", size=14))
    st.plotly_chart(fig, use_container_width=False)
    st.caption(
        "🧠 **Physical Meaning:** This plot tracks the strongest local uphill diffusion intensity over time. "
        "Peaks indicate moments when cross-diffusion coupling is most active, revealing transient dominance "
        "of non-Fickian diffusion mechanisms."
    )


# ------------------------------
# Main
# ------------------------------
def main():
    st.title("📈 Diffusion Analysis: Uphill vs Fickian Behavior")

    solutions = load_solutions(SOLUTION_DIR)
    if not solutions:
        st.warning("No PINN solutions found in 'pinn_solutions/'.")
        return

    st.sidebar.header("Controls")
    diff_type = st.sidebar.selectbox("Diffusion Type", DIFFUSION_TYPES)
    ly_values = sorted(set(s['Ly_parsed'] for s in solutions if s['diffusion_type'] == diff_type))
    ly_target = st.sidebar.select_slider("Select Ly (μm)", options=ly_values)
    time_index = st.sidebar.slider("Select Time Index", 0, 49, 25)

    solution = next((s for s in solutions if s['diffusion_type'] == diff_type and abs(s['Ly_parsed'] - ly_target) < 1e-4), None)
    if not solution:
        st.error("No matching solution found.")
        return

    J1, J2, grad_c1, grad_c2 = compute_fluxes_and_grads(solution['c1_preds'], solution['c2_preds'],
                                                       solution['X'][:, 0], solution['Y'][0, :], solution['params'])
    solution.update({'J1_preds': J1, 'J2_preds': J2, 'grad_c1_y': grad_c1, 'grad_c2_y': grad_c2})

    st.subheader("1️⃣ Flux vs Concentration Gradient (Instantaneous Behavior)")
    plot_flux_vs_gradient(solution, time_index)

    st.subheader("2️⃣ Temporal Evolution of Global Max |J·∇c|")
    plot_uphill_over_time(solution)


if __name__ == "__main__":
    main()
