import os
import pickle
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import cdist

# Directory containing .pkl solution files
SOLUTION_DIR = "pinn_solutions"  # Adjust this path to the actual directory containing your .pkl files

# Diffusion types
DIFFUSION_TYPES = ['crossdiffusion', 'cu_selfdiffusion', 'ni_selfdiffusion']

@st.cache_data
def load_solutions(solution_dir):
    solutions = []
    load_logs = []
    metadata = []

    for fname in os.listdir(solution_dir):
        if not fname.endswith(".pkl"):
            load_logs.append(f"{fname}: Skipped - not a .pkl file.")
            continue

        filepath = os.path.join(solution_dir, fname)
        try:
            with open(filepath, "rb") as f:
                sol = pickle.load(f)

            required = ['params', 'X', 'Y', 'c1_preds', 'c2_preds', 'times']
            if not all(key in sol for key in required):
                load_logs.append(f"{fname}: Failed - missing keys: {set(required) - set(sol.keys())}")
                continue

            match = re.match(r"solution_(\w+)_ly_([\d.]+)_tmax_([\d.]+)\.pkl", fname)
            if not match:
                load_logs.append(f"{fname}: Failed - invalid filename format.")
                continue

            diff_type, ly_val, t_max = match.groups()
            ly_val = float(ly_val)
            t_max = float(t_max)

            if diff_type not in DIFFUSION_TYPES:
                load_logs.append(f"{fname}: Failed - unknown diffusion type '{diff_type}'.")
                continue

            c1_preds = sol['c1_preds']
            c2_preds = sol['c2_preds']
            if not (isinstance(c1_preds, list) and isinstance(c2_preds, list) and len(c1_preds) == len(c2_preds)):
                load_logs.append(f"{fname}: Failed - invalid c1_preds/c2_preds structure.")
                continue

            if c1_preds[0].shape == (50, 50):
                sol['orientation_note'] = "Already rows=y, cols=x"
            else:
                c1_preds = [c.T for c in c1_preds]
                c2_preds = [c.T for c in c2_preds]
                sol['orientation_note'] = "Transposed to rows=y, cols=x"

            sol['c1_preds'] = c1_preds
            sol['c2_preds'] = c2_preds
            sol['diffusion_type'] = diff_type
            sol['Ly_parsed'] = ly_val

            solutions.append(sol)
            metadata.append({'type': diff_type, 'Ly': ly_val, 'filename': fname})
            load_logs.append(f"{fname}: Loaded [{diff_type}, Ly={ly_val:.1f}, t_max={t_max:.1f}]")

        except Exception as e:
            load_logs.append(f"{fname}: Load failed - {str(e)}")

    return solutions, metadata, load_logs

def compute_fluxes_and_grads(c1_preds, c2_preds, x_coords, y_coords, params):
    D11, D12, D21, D22 = params['D11'], params['D12'], params['D21'], params['D22']
    dx = x_coords[1] - x_coords[0]
    dy = y_coords[1] - y_coords[0]
    
    J1_preds = []
    J2_preds = []
    grad_c1_y_preds = []
    grad_c2_y_preds = []
    for c1, c2 in zip(c1_preds, c2_preds):
        grad_c1_x = np.gradient(c1, dx, axis=1)
        grad_c1_y = np.gradient(c1, dy, axis=0)
        grad_c2_x = np.gradient(c2, dx, axis=1)
        grad_c2_y = np.gradient(c2, dy, axis=0)
        
        J1_x = -(D11 * grad_c1_x + D12 * grad_c2_x)
        J1_y = -(D11 * grad_c1_y + D12 * grad_c2_y)
        J2_x = -(D21 * grad_c1_x + D22 * grad_c2_x)
        J2_y = -(D21 * grad_c1_y + D22 * grad_c2_y)
        
        J1_preds.append([J1_x, J1_y])
        J2_preds.append([J2_x, J2_y])
        grad_c1_y_preds.append(grad_c1_y)
        grad_c2_y_preds.append(grad_c2_y)
    
    return J1_preds, J2_preds, grad_c1_y_preds, grad_c2_y_preds

@st.cache_data
def load_and_process_solution(solutions, diff_type, ly_target, tolerance=1e-4):
    exact = [s for s in solutions if s['diffusion_type'] == diff_type and abs(s['Ly_parsed'] - ly_target) < tolerance]
    if exact:
        solution = exact[0]
        solution['interpolated'] = False
    else:
        solution = attention_weighted_interpolation(solutions, [s['Ly_parsed'] for s in solutions], ly_target, diff_type)
    
    if solution:
        J1_preds, J2_preds, grad_c1_y, grad_c2_y = compute_fluxes_and_grads(
            solution['c1_preds'], solution['c2_preds'],
            solution['X'][:, 0], solution['Y'][0, :], solution['params']
        )
        solution['J1_preds'] = J1_preds
        solution['J2_preds'] = J2_preds
        solution['grad_c1_y'] = grad_c1_y
        solution['grad_c2_y'] = grad_c2_y
    return solution

def attention_weighted_interpolation(solutions, lys, ly_target, diff_type, sigma=2.5):
    matching = [s for s in solutions if s['diffusion_type'] == diff_type]
    if not matching:
        return None

    lys = np.array([s['Ly_parsed'] for s in matching])
    weights = get_interpolation_weights(lys, ly_target, sigma)

    Lx = matching[0]['params']['Lx']
    t_max = matching[0]['params']['t_max']
    x_coords = np.linspace(0, Lx, 50)
    y_coords = np.linspace(0, ly_target, 50)
    times = np.linspace(0, t_max, 50)

    c1_interp = np.zeros((len(times), 50, 50))
    c2_interp = np.zeros((len(times), 50, 50))

    for sol, w in zip(matching, weights):
        X_sol = sol['X'][:, 0]
        Y_sol = sol['Y'][0, :] * (ly_target / sol['params']['Ly'])
        for t_idx in range(len(times)):
            interp_c1 = RegularGridInterpolator(
                (Y_sol, X_sol), sol['c1_preds'][t_idx],
                method='linear', bounds_error=False, fill_value=0
            )
            interp_c2 = RegularGridInterpolator(
                (Y_sol, X_sol), sol['c2_preds'][t_idx],
                method='linear', bounds_error=False, fill_value=0
            )
            X_target, Y_target = np.meshgrid(x_coords, y_coords, indexing='ij')
            points = np.column_stack([Y_target.ravel(), X_target.ravel()])
            c1_interp[t_idx] += w * interp_c1(points).reshape(50, 50)
            c2_interp[t_idx] += w * interp_c2(points).reshape(50, 50)

    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    param_set = matching[0]['params'].copy()
    param_set['Ly'] = ly_target

    return {
        'params': param_set,
        'X': X,
        'Y': Y,
        'c1_preds': list(c1_interp),
        'c2_preds': list(c2_interp),
        'times': times,
        'diffusion_type': diff_type,
        'interpolated': True,
        'used_lys': lys.tolist(),
        'attention_weights': weights.tolist(),
        'orientation_note': "rows=y, cols=x"
    }

def get_interpolation_weights(lys, ly_target, sigma=2.5):
    lys = np.array(lys).reshape(-1, 1)
    target = np.array([[ly_target]])
    distances = cdist(target, lys).flatten()
    weights = np.exp(-(distances**2) / (2 * sigma**2))
    weights /= weights.sum() + 1e-10
    return weights

def detect_uphill(solution, time_index):
    J1_y = solution['J1_preds'][time_index][1]
    grad_c1_y = solution['grad_c1_y'][time_index]
    J2_y = solution['J2_preds'][time_index][1]
    grad_c2_y = solution['grad_c2_y'][time_index]
    
    uphill_cu = J1_y * grad_c1_y > 0
    uphill_ni = J2_y * grad_c2_y > 0
    
    return uphill_cu, uphill_ni

def plot_uphill_regions(solution, time_index, downsample):
    x_coords = solution['X'][:, 0]
    y_coords = solution['Y'][0, :]
    t_val = solution['times'][time_index]
    Lx = solution['params']['Lx']
    Ly = solution['params']['Ly']
    diff_type = solution['diffusion_type']

    ds = max(1, downsample)
    x_indices = np.unique(np.linspace(0, len(x_coords)-1, num=len(x_coords)//ds, dtype=int))
    y_indices = np.unique(np.linspace(0, len(y_coords)-1, num=len(y_coords)//ds, dtype=int))

    x_ds = x_coords[x_indices]
    y_ds = y_coords[y_indices]
    
    uphill_cu, uphill_ni = detect_uphill(solution, time_index)
    uphill_cu_ds = uphill_cu[np.ix_(y_indices, x_indices)]
    uphill_ni_ds = uphill_ni[np.ix_(y_indices, x_indices)]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Cu Uphill Regions", "Ni Uphill Regions"))

    fig.add_trace(go.Heatmap(
        x=x_ds, y=y_ds, z=uphill_cu_ds.astype(float), colorscale='RdBu',
        colorbar=dict(title='Uphill (1/0)', x=0.45), zsmooth='best'
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        x=x_ds, y=y_ds, z=uphill_ni_ds.astype(float), colorscale='RdBu',
        colorbar=dict(title='Uphill (1/0)', x=1.02), zsmooth='best'
    ), row=1, col=2)

    fig.update_layout(
        height=500,
        title=f"Uphill Diffusion Regions: {diff_type.replace('_', ' ')} @ t={t_val:.1f}s",
        showlegend=False,
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_flux_vs_gradient(solution, time_index):
    x_idx = 25  # center x
    y_coords = solution['Y'][0, :]
    t_val = solution['times'][time_index]
    diff_type = solution['diffusion_type']

    J1_y_center = solution['J1_preds'][time_index][1][:, x_idx]
    grad_c1_y_center = solution['grad_c1_y'][time_index][:, x_idx]
    J2_y_center = solution['J2_preds'][time_index][1][:, x_idx]
    grad_c2_y_center = solution['grad_c2_y'][time_index][:, x_idx]

    sns.set_context("paper")
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=300)

    ax1.plot(y_coords, -grad_c1_y_center, label='-∇C_Cu', linewidth=2)
    ax1.plot(y_coords, J1_y_center, label='J_Cu', linewidth=2, linestyle='--')
    ax1.set_xlabel('y (μm)', fontsize=14)
    ax1.set_ylabel('Flux / -Gradient', fontsize=14)
    ax1.set_title(f'Cu Flux vs Gradient @ t={t_val:.1f}s', fontsize=16)
    ax1.legend(fontsize=12)

    ax2.plot(y_coords, -grad_c2_y_center, label='-∇C_Ni', linewidth=2)
    ax2.plot(y_coords, J2_y_center, label='J_Ni', linewidth=2, linestyle='--')
    ax2.set_xlabel('y (μm)', fontsize=14)
    ax2.set_ylabel('Flux / -Gradient', fontsize=14)
    ax2.set_title(f'Ni Flux vs Gradient @ t={t_val:.1f}s', fontsize=16)
    ax2.legend(fontsize=12)

    plt.suptitle(f"Flux vs Gradient: {diff_type.replace('_', ' ')}", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    st.pyplot(fig)
    plt.close()

def main():
    st.title("Theoretical Assessment of Diffusion Solutions")

    st.markdown(r"""
    ### Applicable Theories
    1. **Fick's First Law (Extended for Multicomponent Systems)**:
       \[ \mathbf{J}_i = - \sum_{j} D_{ij} \nabla C_j \]
       - For self-diffusion: \( D_{ij} = 0 \) for \( i \neq j \), so \( \mathbf{J}_i = -D_{ii} \nabla C_i \).
       - For cross-diffusion: Off-diagonal terms couple fluxes, potentially causing uphill diffusion where \( \mathbf{J}_i \cdot \nabla C_i > 0 \).

    2. **Continuity Equation**:
       \[ \frac{\partial C_i}{\partial t} = - \nabla \cdot \mathbf{J}_i \]
       - Governs concentration evolution; in cross-diffusion, coupling affects profile shapes (e.g., non-monotonic in cross vs. monotonic in self).
    """)

    solutions, metadata, load_logs = load_solutions(SOLUTION_DIR)

    if not solutions:
        st.error("No valid solution files found.")
        return

    st.sidebar.header("Parameters")
    diff_type = st.sidebar.selectbox("Diffusion Type", DIFFUSION_TYPES, format_func=lambda x: x.replace('_', ' ').title())
    available_lys = sorted(set(s['Ly_parsed'] for s in solutions if s['diffusion_type'] == diff_type))
    ly_target = st.sidebar.select_slider("Ly (μm)", options=available_lys, value=available_lys[0] if available_lys else 50.0)
    time_index = st.sidebar.slider("Time Index", 0, 49, 49)
    downsample = st.sidebar.slider("Downsample", 1, 5, 2)

    solution = load_and_process_solution(solutions, diff_type, ly_target)

    if solution:
        st.subheader("Uphill Diffusion Detection")
        st.markdown(r"""
        Uphill diffusion occurs when \( \mathbf{J}_y \cdot \nabla_y C > 0 \), i.e., species diffuses against its gradient due to cross-terms.
        Expected in cross-diffusion, absent in self-diffusion.
        """)
        plot_uphill_regions(solution, time_index, downsample)

        st.subheader("Flux vs Gradient Comparison")
        st.markdown(r"""
        Plots \( \mathbf{J}_y \) vs \( -\nabla_y C \) along central line.
        In self-diffusion, they align proportionally; in cross, deviations indicate coupling.
        """)
        plot_flux_vs_gradient(solution, time_index)
    else:
        st.error(f"No solution for {diff_type}, Ly={ly_target:.1f}")

if __name__ == "__main__":
    main()
