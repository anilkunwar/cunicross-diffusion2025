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
import re

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
    metadata = []

    for fname in os.listdir(solution_dir):
        if not fname.endswith(".pkl"):
            load_logs.append(f"{fname}: Skipped - not a .pkl file.")
            continue

        filepath = os.path.join(solution_dir, fname)
        try:
            with open(filepath, "rb") as f:
                sol = pickle.load(f)

            # Validate required keys
            required = ['params', 'X', 'Y', 'c1_preds', 'c2_preds', 'times']
            if not all(key in sol for key in required):
                load_logs.append(f"{fname}: Failed - missing keys: {set(required) - set(sol.keys())}")
                continue

            # Parse diffusion type and Ly from filename
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

            # Validate array shapes and fix orientation
            c1_preds = sol['c1_preds']
            c2_preds = sol['c2_preds']
            if not (isinstance(c1_preds, list) and isinstance(c2_preds, list) and len(c1_preds) == len(c2_preds)):
                load_logs.append(f"{fname}: Failed - invalid c1_preds/c2_preds structure.")
                continue

            if c1_preds[0].shape == (50, 50):
                # Already (y,x) as expected
                pass
            elif c1_preds[0].shape == (50, 50):
                # Transpose to (y,x)
                c1_preds = [c.T for c in c1_preds]
                c2_preds = [c.T for c in c2_preds]
                sol['orientation_note'] = "Transposed to rows=y, cols=x"
            else:
                load_logs.append(f"{fname}: Failed - unexpected array shape {c1_preds[0].shape}.")
                continue

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

def compute_fluxes(c1_preds, c2_preds, x_coords, y_coords, params):
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

@st.cache_data
def interpolate_solution(solutions, target_Ly, target_type, sigma=2.5):
    matching = [s for s in solutions if s['diffusion_type'] == target_type]
    if not matching:
        return None

    lys = np.array([s['params']['Ly'] for s in matching])
    if not lys.size:
        return None

    weights = get_interpolation_weights(lys, target_Ly, sigma=2.5)

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

def get_interpolation_weights(lys, ly_target, sigma=2.5):
    lys = np.array(lys).reshape(-1, 1)
    target = np.array([[ly_target]])
    distances = cdist(target, lys).flatten()
    weights = np.exp(-(distances**2) / (2 * sigma**2))
    weights /= weights.sum() + 1e-10
    return weights

@st.cache_data
def get_solution(solutions, ly_target, diff_type):
    exact = [s for s in solutions if s['diffusion_type'] == diff_type and abs(s['params']['Ly'] - ly_target) < 1e-3]
    if exact:
        sol = exact[0]
        if 'J1_preds' not in sol:
            J1, J2 = compute_fluxes(sol['c1_preds'], sol['c2_preds'],
                                    sol['X'][:, 0], sol['Y'][0, :], sol['params'])
            sol['J1_preds'], sol['J2_preds'] = J1, J2
        sol['interpolated'] = False
        return sol
    return interpolate_solution(solutions, ly_target, diff_type)

def plot_concentration_heatmap(solution, time_index, downsample=2):
    x = solution['X'][:, 0][::downsample]
    y = solution['Y'][0, :][::downsample]
    c1 = solution['c1_preds'][time_index][::downsample, ::downsample]
    c2 = solution['c2_preds'][time_index][::downsample, ::downsample]
    t = solution['times'][time_index]
    Lx, Ly = solution['params']['Lx'], solution['params']['Ly']

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=(f"Cu @ t={t:.1f}s", f"Ni @ t={t:.1f}s"))
    fig.add_trace(go.Heatmap(z=c1, x=x, y=y, colorscale='Viridis',
                             colorbar=dict(title='Cu (mol/cm³)', x=0.45)), row=1, col=1)
    fig.add_trace(go.Heatmap(z=c2, x=x, y=y, colorscale='Magma',
                             colorbar=dict(title='Ni (mol/cm³)', x=1.02)), row=1, col=2)

    fig.update_layout(height=500, title=f"Concentration [{solution['diffusion_type'].replace('_', ' ')}, Ly={Ly:.1f}μm]")
    fig.update_xaxes(title_text="x (μm)", range=[0, Lx], row=1, col=1)
    fig.update_yaxes(title_text="y (μm)", range=[0, Ly], row=1, col=1)
    fig.update_xaxes(title_text="x (μm)", range=[0, Lx], row=1, col=2)
    fig.update_yaxes(title_text="y (μm)", range=[0, Ly], row=1, col=2)
    st.plotly_chart(fig, use_container_width=True)

def plot_flux_quiver(solution, time_index, downsample=2):
    x = solution['X'][:, 0][::downsample]
    y = solution['Y'][0, :][::downsample]
    J1_x = solution['J1_preds'][time_index][0][::downsample, ::downsample]
    J1_y = solution['J1_preds'][time_index][1][::downsample, ::downsample]
    J2_x = solution['J2_preds'][time_index][0][::downsample, ::downsample]
    J2_y = solution['J2_preds'][time_index][1][::downsample, ::downsample]
    t = solution['times'][time_index]
    Lx, Ly = solution['params']['Lx'], solution['params']['Ly']

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Cu Flux Magnitude", "Ni Flux Magnitude",
                                       "Cu J_x", "Ni J_x"),
                        vertical_spacing=0.15)

    # Magnitude plots
    J1_mag = np.sqrt(J1_x**2 + J1_y**2)
    J2_mag = np.sqrt(J2_x**2 + J2_y**2)

    fig.add_trace(go.Heatmap(z=np.log10(np.maximum(J1_mag, 1e-10)), x=x, y=y, colorscale='Viridis',
                             colorbar=dict(title='Log Cu Flux', x=0.45, y=0.85, len=0.4)), row=1, col=1)
    fig.add_trace(go.Heatmap(z=np.log10(np.maximum(J2_mag, 1e-10)), x=x, y=y, colorscale='Magma',
                             colorbar=dict(title='Log Ni Flux', x=1.02, y=0.85, len=0.4)), row=1, col=2)
    fig.add_trace(go.Heatmap(z=J1_x, x=x, y=y, colorscale='RdBu', zmid=0,
                             colorbar=dict(title='Cu J_x', x=0.45, y=0.35, len=0.4)), row=2, col=1)
    fig.add_trace(go.Heatmap(z=J2_x, x=x, y=y, colorscale='RdBu', zmid=0,
                             colorbar=dict(title='Ni J_x', x=1.02, y=0.35, len=0.4)), row=2, col=2)

    # Quiver annotations
    scale = 0.1 * Lx
    max_J1 = np.max(J1_mag) + 1e-9
    max_J2 = np.max(J2_mag) + 1e-9
    for i in range(0, len(x), 2):
        for j in range(0, len(y), 2):
            if J1_mag[i,j] > 1e-10:
                fig.add_annotation(x=x[i], y=y[j], ax=x[i] + scale*J1_x[i,j]/max_J1, ay=y[j] + scale*J1_y[i,j]/max_J1,
                                  xref="x", yref="y", axref="x", ayref="y",
                                  showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor='white', row=1, col=1)
            if J2_mag[i,j] > 1e-10:
                fig.add_annotation(x=x[i], y=y[j], ax=x[i] + scale*J2_x[i,j]/max_J2, ay=y[j] + scale*J2_y[i,j]/max_J2,
                                  xref="x2", yref="y2", axref="x2", ayref="y2",
                                  showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor='white', row=1, col=2)

    fig.update_layout(height=800, title=f"Flux Fields @ t={t:.1f}s [{solution['diffusion_type'].replace('_', ' ')}, Ly={Ly:.1f}μm]")
    for row, col, xref in [(1,1,'x'), (1,2,'x2'), (2,1,'x3'), (2,2,'x4')]:
        fig.update_xaxes(title_text="x (μm)", range=[0, Lx], row=row, col=col)
        fig.update_yaxes(title_text="y (μm)", range=[0, Ly], row=row, col=col)
    st.plotly_chart(fig, use_container_width=True)

def plot_central_line_profiles(solutions, time_index, ly_target, diff_types):
    """Compare central line profiles across diffusion types."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    colors = plt.cm.tab10(np.linspace(0, 1, len(diff_types)))

    for diff_type, color in zip(diff_types, colors):
        sol = get_solution(solutions, ly_target, diff_type)
        if not sol:
            continue

        x_idx = len(sol['X'][:, 0]) // 2
        y_coords = sol['Y'][0, :]
        c1_center = sol['c1_preds'][time_index][:, x_idx]
        c2_center = sol['c2_preds'][time_index][:, x_idx]
        t = sol['times'][time_index]

        label = f"{diff_type.replace('_', ' ')} (Ly={ly_target:.1f})"
        ax1.plot(c1_center, y_coords, label=label, color=color)
        ax2.plot(c2_center, y_coords, label=label, color=color)

    ax1.set_xlabel('Cu Concentration (mol/cm³)')
    ax1.set_ylabel('y (μm)')
    ax1.set_title(f'Cu @ x=30μm, t={t:.1f}s')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_xlabel('Ni Concentration (mol/cm³)')
    ax2.set_title(f'Ni @ x=30μm, t={t:.1f}s')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def plot_center_point_evolution(solutions, ly_target, diff_types):
    """Compare center point evolution across diffusion types."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    colors = plt.cm.tab10(np.linspace(0, 1, len(diff_types)))

    for diff_type, color in zip(diff_types, colors):
        sol = get_solution(solutions, ly_target, diff_type)
        if not sol:
            continue

        x_idx = len(sol['X'][:, 0]) // 2
        y_idx = len(sol['Y'][0, :]) // 2
        times = sol['times']
        c1_center = [c[y_idx, x_idx] for c in sol['c1_preds']]
        c2_center = [c[y_idx, x_idx] for c in sol['c2_preds']]

        label = f"{diff_type.replace('_', ' ')} (Ly={ly_target:.1f})"
        ax1.plot(times, c1_center, label=label, color=color)
        ax2.plot(times, c2_center, label=label, color=color)

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Cu Concentration (mol/cm³)')
    ax1.set_title(f'Center Point Cu (x=30, y={ly_target/2:.1f})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Ni Concentration (mol/cm³)')
    ax2.set_title(f'Center Point Ni (x=30, y={ly_target/2:.1f})')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

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

def main():
    st.title("PINN Diffusion Post-Processor: Cross vs. Self Diffusion")

    solutions, metadata, load_logs = load_solutions(SOLUTION_DIR)

    if not solutions:
        st.error("No valid .pkl files found in pinn_solutions directory.")
        st.write("Expected files:")
        st.write("- solution_crossdiffusion_ly_90.0_tmax_200.pkl")
        st.write("- solution_crossdiffusion_ly_50.0_tmax_200.pkl")
        st.write("- solution_cu_selfdiffusion_ly_90.0_tmax_200.pkl")
        st.write("- solution_cu_selfdiffusion_ly_50.0_tmax_200.pkl")
        st.write("- solution_ni_selfdiffusion_ly_50.0_tmax_200.pkl")
        st.subheader("Load Log")
        for log in load_logs:
            st.write(log)
        return

    # Sidebar
    st.sidebar.header("Simulation Selector")
    diff_type = st.sidebar.selectbox("Diffusion Type", options=list(DIFFUSION_TYPES.keys()),
                                     format_func=lambda x: x.replace('_', ' ').title())
    ly_target = st.sidebar.selectbox("Domain Height Ly (μm)", options=[50.0, 90.0], format_func=lambda x: f"{x:.1f}")

    solution = get_solution(solutions, ly_target, diff_type)
    if not solution:
        st.error(f"No solution for {diff_type}, Ly={ly_target}. Check load log.")
        st.subheader("Load Log")
        for log in load_logs:
            st.write(log)
        return

    st.success(f"Loaded: **{diff_type.replace('_', ' ')}**, Ly={solution['params']['Ly']:.1f}μm "
               f"{'(Interpolated)' if solution.get('interpolated') else ''}")

    time_index = st.slider("Time Step", 0, len(solution['times'])-1, len(solution['times'])//2)
    downsample = st.slider("Downsample Grid", 1, 5, 2)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Concentration", "Flux", "Central Line", "Center Point Evolution"])

    with tab1:
        st.subheader("2D Concentration Fields")
        plot_concentration_heatmap(solution, time_index, downsample)

    with tab2:
        st.subheader("Flux Fields")
        plot_flux_quiver(solution, time_index, downsample)

    with tab3:
        st.subheader("Central Line Profile (x = Lx/2)")
        diff_types = st.multiselect("Compare Diffusion Types", options=list(DIFFUSION_TYPES.keys()),
                                    default=[diff_type], format_func=lambda x: x.replace('_', ' ').title())
        plot_central_line_profiles(solutions, time_index, ly_target, diff_types)

    with tab4:
        st.subheader("Center Point Time Evolution")
        diff_types = st.multiselect("Compare Diffusion Types", options=list(DIFFUSION_TYPES.keys()),
                                    default=[diff_type], format_func=lambda x: x.replace('_', ' ').title())
        plot_center_point_evolution(solutions, ly_target, diff_types)

    # Download
    st.subheader("Download Data")
    col1, col2 = st.columns(2)
    with col1:
        csv_data, fname = download_data(solution, time_index)
        st.download_button("Download CSV (Selected Time)", csv_data, fname, "text/csv")
    with col2:
        zip_data, zipname = download_data(solution, time_index, all_times=True)
        st.download_button("Download ZIP (All Times)", zip_data, zipname, "application/zip")

if __name__ == "__main__":
    main()
