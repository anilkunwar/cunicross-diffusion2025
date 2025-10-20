import os
import pickle
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import io
import zipfile
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns

# Directory containing .pkl solution files
SOLUTION_DIR = os.path.join(os.path.dirname(__file__), "pinn_solutions")

# Diffusion types and their boundary conditions
DIFFUSION_TYPES = {
    'crossdiffusion': {'C_CU_TOP': 1.59e-3, 'C_NI_BOTTOM': 4.0e-4},
    'cu_selfdiffusion': {'C_CU_TOP': 1.59e-3, 'C_NI_BOTTOM': 0.0},
    'ni_selfdiffusion': {'C_CU_TOP': 0.0, 'C_NI_BOTTOM': 4.0e-4}
}

@st.cache_data
def load_solutions(solution_dir):
    solutions = []
    load_logs = []
    metadata = []  # Store type, Ly, filename

    for fname in os.listdir(solution_dir):
        if fname.endswith(".pkl"):
            filepath = os.path.join(solution_dir, fname)
            try:
                with open(filepath, "rb") as f:
                    sol = pickle.load(f)

                # Validate required keys
                required = ['params', 'X', 'Y', 'c1_preds', 'c2_preds', 'times']
                if not all(key in sol for key in required):
                    load_logs.append(f"{fname}: Missing required keys.")
                    continue

                # Parse diffusion type and Ly from filename
                parts = fname.replace('.pkl', '').split('_')
                diff_type = None
                ly_val = None
                for i, part in enumerate(parts):
                    if part in DIFFUSION_TYPES and i + 2 < len(parts) and parts[i+1] == 'ly':
                        diff_type = part
                        try:
                            ly_val = float(parts[i+2])
                        except:
                            pass
                        break

                if not diff_type or not ly_val:
                    load_logs.append(f"{fname}: Could not parse diffusion type or Ly.")
                    continue

                # Fix orientation: ensure c1_preds/c2_preds are (y,x) i.e., shape (50,50) with rows=y
                c1_preds = sol['c1_preds']
                c2_preds = sol['c2_preds']
                if isinstance(c1_preds[0], np.ndarray) and c1_preds[0].shape == (50, 50):
                    # Already (y,x) → good
                    pass
                elif c1_preds[0].shape == (50, 50):
                    # Transposed: (x,y) → transpose
                    c1_preds = [c.T for c in c1_preds]
                    c2_preds = [c.T for c in c2_preds]
                    sol['orientation_note'] = "Transposed during load: now rows=y, cols=x"
                else:
                    load_logs.append(f"{fname}: Unexpected array shape.")
                    continue

                sol['c1_preds'] = c1_preds
                sol['c2_preds'] = c2_preds
                sol['diffusion_type'] = diff_type
                sol['Ly_parsed'] = ly_val

                solutions.append(sol)
                metadata.append({'type': diff_type, 'Ly': ly_val, 'filename': fname})
                load_logs.append(f"{fname}: Loaded [{diff_type}, Ly={ly_val:.1f}]")

            except Exception as e:
                load_logs.append(f"{fname}: Load failed - {str(e)}")

    return solutions, metadata, load_logs

def compute_fluxes(c1_preds, c2_preds, x_coords, y_coords, params):
    """Compute fluxes J1 (Cu), J2 (Ni) using finite differences."""
    D11, D12, D21, D22 = params['D11'], params['D12'], params['D21'], params['D22']
    dx = x_coords[1] - x_coords[0]
    dy = y_coords[1] - y_coords[0]

    J1_preds, J2_preds = [], []
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

    return J1_preds, J2_preds

def get_interpolation_weights(lys, ly_target, sigma=2.5):
    lys = np.array(lys).reshape(-1, 1)
    target = np.array([[ly_target]])
    distances = cdist(target, lys).flatten()
    weights = np.exp(-(distances**2) / (2 * sigma**2))
    weights /= weights.sum()
    return weights

@st.cache_data
def interpolate_solution(solutions, target_Ly, target_type, sigma=2.5):
    """Interpolate solution for target Ly and diffusion type."""
    matching = [s for s in solutions if s['diffusion_type'] == target_type]
    if not matching:
        return None

    lys = np.array([s['params']['Ly'] for s in matching])
    weights = get_interpolation_weights(lys, target_Ly, sigma)

    Lx = matching[0]['params']['Lx']
    t_max = matching[0]['params']['t_max']
    x_coords = np.linspace(0, Lx, 50)
    y_coords = np.linspace(0, target_Ly, 50)
    times = np.linspace(0, t_max, 50)

    c1_interp = np.zeros((len(times), 50, 50))
    c2_interp = np.zeros((len(times), 50, 50))

    for sol, w in zip(matching, weights):
        X_sol = sol['X'][:, 0]
        Y_sol = sol['Y'][0, :] * (target_Ly / sol['params']['Ly'])
        for t_idx in range(len(times)):
            interp_c1 = RegularGridInterpolator((X_sol, Y_sol), sol['c1_preds'][t_idx],
                                                method='linear', bounds_error=False, fill_value=0)
            interp_c2 = RegularGridInterpolator((X_sol, Y_sol), sol['c2_preds'][t_idx],
                                                method='linear', bounds_error=False, fill_value=0)
            X_target, Y_target = np.meshgrid(x_coords, y_coords, indexing='ij')
            points = np.column_stack([X_target.ravel(), Y_target.ravel()])
            c1_interp[t_idx] += w * interp_c1(points).reshape(50, 50)
            c2_interp[t_idx] += w * interp_c2(points).reshape(50, 50)

    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    params = matching[0]['params'].copy()
    params['Ly'] = target_Ly

    J1_preds, J2_preds = compute_fluxes(c1_interp, c2_interp, x_coords, y_coords, params)

    return {
        'params': params,
        'X': X, 'Y': Y,
        'c1_preds': list(c1_interp),
        'c2_preds': list(c2_interp),
        'J1_preds': J1_preds,
        'J2_preds': J2_preds,
        'times': times,
        'diffusion_type': target_type,
        'interpolated': True,
        'used_lys': lys.tolist(),
        'weights': weights.tolist()
    }

@st.cache_data
def get_solution(solutions, ly_target, diff_type):
    """Get exact or interpolated solution."""
    exact = [s for s in solutions if s['diffusion_type'] == diff_type and abs(s['params']['Ly'] - ly_target) < 1e-3]
    if exact:
        sol = exact[0]
        if 'J1_preds' not in sol:
            J1, J2 = compute_fluxes(sol['c1_preds'], sol['c2_preds'],
                                    sol['X'][:, 0], sol['Y'][0, :], sol['params'])
            sol['J1_preds'], sol['J2_preds'] = J1, J2
        sol['interpolated'] = False
        return sol
    else:
        return interpolate_solution(solutions, ly_target, diff_type)

def plot_concentration_heatmap(solution, time_index, downsample=2):
    x = solution['X'][:, 0][::downsample]
    y = solution['Y'][0, :][::downsample]
    c1 = solution['c1_preds'][time_index][::downsample, ::downsample]
    c2 = solution['c2_preds'][time_index][::downsample, ::downsample]
    t = solution['times'][time_index]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=(f"Cu @ t={t:.1f}s", f"Ni @ t={t:.1f}s"))

    fig.add_trace(go.Heatmap(z=c1, x=x, y=y, colorscale='Viridis',
                             colorbar=dict(title='Cu (mol/cm³)', x=0.45)), row=1, col=1)
    fig.add_trace(go.Heatmap(z=c2, x=x, y=y, colorscale='Magma',
                             colorbar=dict(title='Ni (mol/cm³)', x=1.02)), row=1, col=2)

    fig.update_layout(height=500, title=f"Concentration [{solution['diffusion_type']}, Ly={solution['params']['Ly']:.1f}μm]")
    fig.update_xaxes(title_text="x (μm)", row=1, col=1)
    fig.update_yaxes(title_text="y (μm)", row=1, col=1)
    st.plotly_chart(fig, use_container_width=True)

def plot_flux_quiver(solution, time_index, downsample=3):
    x = solution['X'][:, 0][::downsample]
    y = solution['Y'][0, :][::downsample]
    J1 = solution['J1_preds'][time_index]
    J2 = solution['J2_preds'][time_index]
    J1_x = J1[0][::downsample, ::downsample]
    J1_y = J1[1][::downsample, ::downsample]
    J2_x = J2[0][::downsample, ::downsample]
    J2_y = J2[1][::downsample, ::downsample]
    t = solution['times'][time_index]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=(f"Cu Flux", f"Ni Flux"))

    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(color=np.log10(np.sqrt(J1_x**2 + J1_y**2)+1e-12),
                                                                 colorscale='Viridis', size=8),
                             text=[f"J={np.sqrt(J1_x[i,j]**2+J1_y[i,j]**2):.2e}" for i in range(J1_x.shape[0]) for j in range(J1_x.shape[1])]),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=x.repeat(downsample), y=np.tile(y, downsample), mode='markers'), row=1, col=1)

    # Quiver plot using annotations
    scale = 1.0
    for i in range(0, len(x), 2):
        for j in range(0, len(y), 2):
            fig.add_annotation(x=x[i], y=y[j], ax=x[i] + scale*J1_x[i,j], ay=y[j] + scale*J1_y[i,j],
                               xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor='white')

    fig.update_layout(height=500, title=f"Flux Quiver @ t={t:.1f}s")
    st.plotly_chart(fig, use_container_width=True)

def plot_central_line_profiles(solution, time_index):
    """Plot Cu and Ni concentration along central vertical line (x = Lx/2)."""
    Lx = solution['params']['Lx']
    x_center_idx = len(solution['X'][:, 0]) // 2
    y_coords = solution['Y'][0, :]
    c1_center = solution['c1_preds'][time_index][:, x_center_idx]
    c2_center = solution['c2_preds'][time_index][:, x_center_idx]
    t = solution['times'][time_index]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    ax1.plot(c1_center, y_coords, 'o-', label='Cu', color='tab:blue')
    ax1.set_xlabel('Cu Concentration (mol/cm³)')
    ax1.set_ylabel('y (μm)')
    ax1.set_title(f'Cu @ x={Lx/2:.1f}μm, t={t:.1f}s')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(c2_center, y_coords, 's-', label='Ni', color='tab:red')
    ax2.set_xlabel('Ni Concentration (mol/cm³)')
    ax2.set_title(f'Ni @ x={Lx/2:.1f}μm, t={t:.1f}s')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def plot_center_point_evolution(solutions, target_Ly, diff_type):
    """Plot concentration at center point (x=Lx/2, y=Ly/2) over time."""
    sol = get_solution(solutions, target_Ly, diff_type)
    if not sol:
        st.warning("No solution for center point plot.")
        return

    Lx = sol['params']['Lx']
    Ly = sol['params']['Ly']
    x_idx = len(sol['X'][:, 0]) // 2
    y_idx = len(sol['Y'][0, :]) // 2

    times = sol['times']
    c1_center = [c[y_idx, x_idx] for c in sol['c1_preds']]
    c2_center = [c[y_idx, x_idx] for c in sol['c2_preds']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    ax1.plot(times, c1_center, 'o-', color='tab:blue')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Cu Concentration')
    ax1.set_title(f'Center Point Cu (x={Lx/2:.1f}, y={Ly/2:.1f})')
    ax1.grid(True, alpha=0.3)

    ax2.plot(times, c2_center, 's-', color='tab:red')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Ni Concentration')
    ax2.set_title(f'Center Point Ni')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"{diff_type.replace('_', ' ').title()}, Ly={Ly:.1f}μm")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def main():
    st.title("PINN Diffusion Post-Processor: Cross vs. Self Diffusion")

    solutions, metadata, load_logs = load_solutions(SOLUTION_DIR)

    if not solutions:
        st.error("No valid .pkl files found. Check filenames and directory.")
        st.write("Expected: `solution_crossdiffusion_ly_90.0_tmax_200.pkl`")
        return

    # Sidebar
    st.sidebar.header("Simulation Selector")
    diff_type = st.sidebar.selectbox("Diffusion Type", options=list(DIFFUSION_TYPES.keys()),
                                     format_func=lambda x: x.replace('_', ' ').title())
    ly_target = st.sidebar.number_input("Domain Height Ly (μm)", 50.0, 100.0, 90.0, 0.1)

    solution = get_solution(solutions, ly_target, diff_type)
    if not solution:
        st.error(f"No solution for {diff_type}, Ly={ly_target}")
        return

    st.success(f"Loaded: **{diff_type}**, Ly={solution['params']['Ly']:.1f}μm "
               f"{'(Interpolated)' if solution.get('interpolated') else ''}")

    time_index = st.slider("Time Step", 0, len(solution['times'])-1, len(solution['times'])//2)
    downsample = st.slider("Downsample Grid", 1, 5, 2)

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Concentration", "Flux", "Central Line", "Center Point Evolution"])

    with tab1:
        st.subheader("2D Concentration Fields")
        plot_concentration_heatmap(solution, time_index, downsample)

    with tab2:
        st.subheader("Flux Fields (Quiver + Magnitude)")
        plot_flux_quiver(solution, time_index, downsample)

    with tab3:
        st.subheader("Central Line Profile (x = Lx/2)")
        plot_central_line_profiles(solution, time_index)

    with tab4:
        st.subheader("Center Point Time Evolution")
        plot_center_point_evolution(solutions, ly_target, diff_type)

    # Download
    st.subheader("Download Data")
    csv_data, fname = download_data(solution, time_index)
    st.download_button("Download CSV (Selected Time)", csv_data, fname, "text/csv")

    zip_data, zipname = download_data(solution, time_index, all_times=True)
    st.download_button("Download ZIP (All Times)", zip_data, zipname, "application/zip")

def download_data(solution, time_index, all_times=False):
    X = solution['X']
    Y = solution['Y']
    if not all_times:
        df = pd.DataFrame({
            'x': X.flatten(),
            'y': Y.flatten(),
            'Cu': np.array(solution['c1_preds'][time_index]).flatten(),
            'Ni': np.array(solution['c2_preds'][time_index]).flatten(),
            'J_Cu_x': solution['J1_preds'][time_index][0].flatten(),
            'J_Cu_y': solution['J1_preds'][time_index][1].flatten(),
            'J_Ni_x': solution['J2_preds'][time_index][0].flatten(),
            'J_Ni_y': solution['J2_preds'][time_index][1].flatten(),
        })
        csv = df.to_csv(index=False).encode()
        return csv, f"data_{solution['diffusion_type']}_ly_{solution['params']['Ly']:.1f}_t_{solution['times'][time_index]:.1f}.csv"
    else:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
            for t_idx, t in enumerate(solution['times']):
                df = pd.DataFrame({
                    't': t, 'x': X.flatten(), 'y': Y.flatten(),
                    'Cu': np.array(solution['c1_preds'][t_idx]).flatten(),
                    'Ni': np.array(solution['c2_preds'][t_idx]).flatten(),
                })
                z.writestr(f"t_{t:.1f}s.csv", df.to_csv(index=False))
        return buffer.getvalue(), f"all_times_{solution['diffusion_type']}_ly_{solution['params']['Ly']:.1f}.zip"

if __name__ == "__main__":
    main()
