import os
import pickle
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import matplotlib as mpl

# Configure Matplotlib for publication-quality figures
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['legend.frameon'] = True
mpl.rcParams['legend.framealpha'] = 0.8
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.alpha'] = 0.3

SOLUTION_DIR = os.path.join(os.path.dirname(__file__), "pinn_solutions")

# Available colormaps for selection
COLORMAPS = [
    'viridis', 'magma', 'plasma', 'inferno', 'hot',
    'coolwarm', 'RdBu', 'seismic', 'Blues', 'Reds'
]

@st.cache_data
def load_solutions(solution_dir):
    solutions = []
    params_list = []
    load_logs = []
    lys = []
    c_cus = []
    c_nis = []
    for fname in os.listdir(solution_dir):
        if fname.endswith(".pkl"):
            try:
                with open(os.path.join(solution_dir, fname), "rb") as f:
                    sol = pickle.load(f)
                required_keys = ['params', 'X', 'Y', 'c1_preds', 'c2_preds', 'times']
                if all(key in sol for key in required_keys):
                    if (np.any(np.isnan(sol['c1_preds'])) or np.any(np.isnan(sol['c2_preds'])) or
                        np.all(sol['c1_preds'] == 0) or np.all(sol['c2_preds'] == 0)):
                        load_logs.append(f"{fname}: Skipped - Invalid data (NaNs or all zeros).")
                        continue
                    c1_min, c1_max = np.min(sol['c1_preds'][0]), np.max(sol['c1_preds'][0])
                    c2_min, c2_max = np.min(sol['c2_preds'][0]), np.max(sol['c2_preds'][0])
                    solutions.append(sol)
                    param_tuple = (sol['params']['Ly'], sol['params']['C_Cu'], sol['params']['C_Ni'])
                    params_list.append(param_tuple)
                    lys.append(sol['params']['Ly'])
                    c_cus.append(sol['params']['C_Cu'])
                    c_nis.append(sol['params']['C_Ni'])
                    load_logs.append(
                        f"{fname}: Loaded. Cu: {c1_min:.2e} to {c1_max:.2e}, Ni: {c2_min:.2e} to {c2_max:.2e}, "
                        f"Ly={param_tuple[0]:.1f}, C_Cu={param_tuple[1]:.1e}, C_Ni={param_tuple[2]:.1e}"
                    )
                else:
                    missing_keys = [key for key in required_keys if key not in sol]
                    load_logs.append(f"{fname}: Skipped - Missing keys: {missing_keys}")
            except Exception as e:
                load_logs.append(f"{fname}: Skipped - Failed to load: {str(e)}")
    if len(solutions) < 1:
        load_logs.append("Error: No valid solutions loaded. Interpolation will fail.")
    else:
        load_logs.append(f"Loaded {len(solutions)} solutions. Expected 32.")
    return solutions, params_list, lys, c_cus, c_nis, load_logs

def attention_weighted_interpolation(solutions, params_list, ly_target, c_cu_target, c_ni_target, sigma_ly=0.2, sigma_ccu=0.2, sigma_cni=0.2):
    if not solutions or not params_list:
        raise ValueError("No solutions or parameters available for interpolation.")
    
    lys = np.array([p[0] for p in params_list])
    c_cus = np.array([p[1] for p in params_list])
    c_nis = np.array([p[2] for p in params_list])
    
    st.write(f"Debug: lys shape={lys.shape}, c_cus shape={c_cus.shape}, c_nis shape={c_nis.shape}")
    
    if not (lys.shape == c_cus.shape == c_nis.shape):
        raise ValueError(f"Parameter array shapes mismatch: lys={lys.shape}, c_cus={c_cus.shape}, c_nis={c_nis.shape}")
    
    ly_norm = (lys - 30.0) / (120.0 - 30.0)
    c_cu_norm = (c_cus - 1.5e-3) / (2.9e-3 - 1.5e-3)
    c_ni_norm = (c_nis - 4.0e-4) / (1.8e-3 - 4.0e-4)
    
    target_ly_norm = (ly_target - 30.0) / (120.0 - 30.0)
    target_c_cu_norm = (c_cu_target - 1.5e-3) / (2.9e-3 - 1.5e-3)
    target_c_ni_norm = (c_ni_target - 4.0e-4) / (1.8e-3 - 4.0e-4)
    
    scaled_distances = np.sqrt(
        ((ly_norm - target_ly_norm) / sigma_ly)**2 +
        ((c_cu_norm - target_c_cu_norm) / sigma_ccu)**2 +
        ((c_ni_norm - target_c_ni_norm) / sigma_cni)**2
    )
    weights = np.exp(-scaled_distances**2 / 2)
    weights_sum = weights.sum()
    if weights_sum == 0:
        raise ValueError("Interpolation weights sum to zero. Check parameter ranges or sigma values.")
    weights /= weights_sum
    
    Lx = solutions[0]['params']['Lx']
    t_max = solutions[0]['params']['t_max']
    x_coords = np.linspace(0, Lx, 50)
    y_coords = np.linspace(0, ly_target, 50)
    times = np.linspace(0, t_max, 50)
    
    c1_interp = np.zeros((len(times), 50, 50))
    c2_interp = np.zeros((len(times), 50, 50))
    
    for weight, solution in zip(weights, solutions):
        X_sol = solution['X'][:, 0]
        Y_sol = solution['Y'][0, :] * (ly_target / solution['params']['Ly'])
        
        for t_idx in range(len(times)):
            interp_c1 = RegularGridInterpolator(
                (X_sol, Y_sol), solution['c1_preds'][t_idx],
                method='linear', bounds_error=False, fill_value=0
            )
            interp_c2 = RegularGridInterpolator(
                (X_sol, Y_sol), solution['c2_preds'][t_idx],
                method='linear', bounds_error=False, fill_value=0
            )
            X_target, Y_target = np.meshgrid(x_coords, y_coords, indexing='ij')
            points = np.stack([X_target.flatten(), Y_target.flatten()], axis=1)
            c1_interp[t_idx] += weight * interp_c1(points).reshape(50, 50)
            c2_interp[t_idx] += weight * interp_c2(points).reshape(50, 50)
    
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    param_set = solutions[0]['params'].copy()
    param_set['Ly'] = ly_target
    param_set['C_Cu'] = c_cu_target
    param_set['C_Ni'] = c_ni_target
    
    return {
        'params': param_set,
        'X': X,
        'Y': Y,
        'c1_preds': list(c1_interp),
        'c2_preds': list(c2_interp),
        'times': times,
        'interpolated': True,
        'attention_weights': weights.tolist()
    }

@st.cache_data
def load_and_interpolate_solution(solutions, params_list, ly_target, c_cu_target, c_ni_target, tolerance_ly=0.1, tolerance_c=1e-5):
    for sol, params in zip(solutions, params_list):
        ly, c_cu, c_ni = params
        if (abs(ly - ly_target) < tolerance_ly and
            abs(c_cu - c_cu_target) < tolerance_c and
            abs(c_ni - c_ni_target) < tolerance_c):
            sol['interpolated'] = False
            return sol
    if not solutions:
        raise ValueError("No solutions available for interpolation.")
    return attention_weighted_interpolation(solutions, params_list, ly_target, c_cu_target, c_ni_target)

def plot_2d_concentration(solution, time_index, output_dir="figures", cmap_cu='viridis', cmap_ni='magma'):
    x_coords = solution['X'][:, 0]
    y_coords = solution['Y'][0, :]
    t_val = solution['times'][time_index]
    Lx = solution['params']['Lx']
    Ly = solution['params']['Ly']
    c1 = solution['c1_preds'][time_index]
    c2 = solution['c2_preds'][time_index]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    
    # Cu heatmap
    im1 = ax1.imshow(
        c1,
        origin='lower',
        extent=[0, Lx, 0, Ly],
        cmap=cmap_cu,
        vmin=0,
        vmax=np.max(c1)
    )
    ax1.set_xlabel('x (μm)')
    ax1.set_ylabel('y (μm)')
    ax1.set_title(f'Cu Concentration, t = {t_val:.1f} s')
    ax1.grid(True)
    cb1 = fig.colorbar(im1, ax=ax1, label='Cu Conc. (mol/cc)', format='%.1e')
    cb1.ax.tick_params(labelsize=10)
    
    # Ni heatmap
    im2 = ax2.imshow(
        c2,
        origin='lower',
        extent=[0, Lx, 0, Ly],
        cmap=cmap_ni,
        vmin=0,
        vmax=np.max(c2)
    )
    ax2.set_xlabel('x (μm)')
    ax2.set_ylabel('y (μm)')
    ax2.set_title(f'Ni Concentration, t = {t_val:.1f} s')
    ax2.grid(True)
    cb2 = fig.colorbar(im2, ax=ax2, label='Ni Conc. (mol/cc)', format='%.1e')
    cb2.ax.tick_params(labelsize=10)
    
    param_text = f"$L_y$ = {Ly:.1f} μm, $C_{{Cu}}$ = {solution['params']['C_Cu']:.1e}, $C_{{Ni}}$ = {solution['params']['C_Ni']:.1e}"
    if solution.get('interpolated', False):
        param_text += " (Interpolated)"
    fig.suptitle(f'Concentration Profiles\n{param_text}', fontsize=14)
    
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"conc_2d_t_{t_val:.1f}_ly_{Ly:.1f}_ccu_{solution['params']['C_Cu']:.1e}_cni_{solution['params']['C_Ni']:.1e}"
    plt.savefig(os.path.join(output_dir, f"{base_filename}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f"{base_filename}.pdf"), bbox_inches='tight')
    plt.close()
    
    return fig, base_filename

def plot_centerline_curves(solution, time_indices, sidebar_metric='mean_cu', output_dir="figures"):
    x_coords = solution['X'][:, 0]
    y_coords = solution['Y'][0, :]
    Lx = solution['params']['Lx']
    Ly = solution['params']['Ly']
    center_idx = 25  # x = Lx/2
    times = solution['times']
    
    # Prepare sidebar data
    if sidebar_metric == 'loss' and 'loss' in solution:
        sidebar_data = solution['loss'][:len(times)]
        sidebar_label = 'Loss'
    elif sidebar_metric == 'mean_cu':
        sidebar_data = [np.mean(c1) for c1 in solution['c1_preds']]
        sidebar_label = 'Mean Cu Conc. (mol/cc)'
    else:  # mean_ni
        sidebar_data = [np.mean(c2) for c2 in solution['c2_preds']]
        sidebar_label = 'Mean Ni Conc. (mol/cc)'
    
    fig = plt.figure(figsize=(8, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.05, 0.5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[3])
    
    # Centerline curves
    colors = cm.viridis(np.linspace(0, 1, len(time_indices)))
    for idx, t_idx in enumerate(time_indices):
        t_val = times[t_idx]
        c1 = solution['c1_preds'][t_idx][:, center_idx]
        c2 = solution['c2_preds'][t_idx][:, center_idx]
        ax1.plot(y_coords, c1, label=f't = {t_val:.1f} s', color=colors[idx])
        ax2.plot(y_coords, c2, label=f't = {t_val:.1f} s', color=colors[idx])
    
    ax1.set_xlabel('y (μm)')
    ax1.set_ylabel('Cu Conc. (mol/cc)')
    ax1.set_title(f'Cu at x = {Lx/2:.1f} μm')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    ax2.set_xlabel('y (μm)')
    ax2.set_ylabel('Ni Conc. (mol/cc)')
    ax2.set_title(f'Ni at x = {Lx/2:.1f} μm')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Sidebar plot
    ax3.plot(sidebar_data, times, 'k-')
    ax3.set_xlabel(sidebar_label, fontsize=10)
    ax3.set_ylabel('Time (s)')
    ax3.set_title('Metric vs. Time', fontsize=12)
    ax3.grid(True)
    ax3.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    param_text = f"$L_y$ = {Ly:.1f} μm, $C_{{Cu}}$ = {solution['params']['C_Cu']:.1e}, $C_{{Ni}}$ = {solution['params']['C_Ni']:.1e}"
    if solution.get('interpolated', False):
        param_text += " (Interpolated)"
    fig.suptitle(f'Centerline Concentration Profiles\n{param_text}', fontsize=14)
    
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"conc_centerline_ly_{Ly:.1f}_ccu_{solution['params']['C_Cu']:.1e}_cni_{solution['params']['C_Ni']:.1e}"
    plt.savefig(os.path.join(output_dir, f"{base_filename}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f"{base_filename}.pdf"), bbox_inches='tight')
    plt.close()
    
    return fig, base_filename

def plot_parameter_sweep(solutions, params_list, selected_params, time_index, sidebar_metric='mean_cu', output_dir="figures"):
    Lx = solutions[0]['params']['Lx']
    center_idx = 25  # x = Lx/2
    t_val = solutions[0]['times'][time_index]
    
    # Prepare sidebar data
    sidebar_data = []
    sidebar_labels = []
    for sol, params in zip(solutions, params_list):
        if params in selected_params:
            if sidebar_metric == 'loss' and 'loss' in sol:
                sidebar_data.append(sol['loss'][time_index])
            elif sidebar_metric == 'mean_cu':
                sidebar_data.append(np.mean(sol['c1_preds'][time_index]))
            else:  # mean_ni
                sidebar_data.append(np.mean(sol['c2_preds'][time_index]))
            ly, c_cu, c_ni = params
            sidebar_labels.append(f'$L_y$={ly:.1f}, $C_{{Cu}}$={c_cu:.1e}, $C_{{Ni}}$={c_ni:.1e}')
    
    fig = plt.figure(figsize=(8, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.05, 0.5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[3])
    
    # Parameter sweep curves
    colors = cm.tab10(np.linspace(0, 1, 10))
    for idx, (sol, params) in enumerate(zip(solutions, params_list)):
        ly, c_cu, c_ni = params
        if params in selected_params:
            y_coords = sol['Y'][0, :]
            c1 = sol['c1_preds'][time_index][:, center_idx]
            c2 = sol['c2_preds'][time_index][:, center_idx]
            label = f'$L_y$={ly:.1f}, $C_{{Cu}}$={c_cu:.1e}, $C_{{Ni}}$={c_ni:.1e}'
            ax1.plot(y_coords, c1, label=label, color=colors[idx % len(colors)])
            ax2.plot(y_coords, c2, label=label, color=colors[idx % len(colors)])
    
    ax1.set_xlabel('y (μm)')
    ax1.set_ylabel('Cu Conc. (mol/cc)')
    ax1.set_title(f'Cu at x = {Lx/2:.1f} μm, t = {t_val:.1f} s')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    ax2.set_xlabel('y (μm)')
    ax2.set_ylabel('Ni Conc. (mol/cc)')
    ax2.set_title(f'Ni at x = {Lx/2:.1f} μm, t = {t_val:.1f} s')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Sidebar bar plot
    ax3.barh(range(len(sidebar_data)), sidebar_data, color='gray', edgecolor='black')
    ax3.set_yticks(range(len(sidebar_data)))
    ax3.set_yticklabels(sidebar_labels, fontsize=8)
    ax3.set_xlabel('Mean Cu Conc. (mol/cc)' if sidebar_metric == 'mean_cu' else 'Mean Ni Conc. (mol/cc)' if sidebar_metric == 'mean_ni' else 'Loss')
    ax3.set_title('Metric per Parameter', fontsize=12)
    ax3.grid(True, axis='x')
    ax3.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    fig.suptitle('Concentration Profiles for Parameter Sweep', fontsize=14)
    
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"conc_sweep_t_{t_val:.1f}"
    plt.savefig(os.path.join(output_dir, f"{base_filename}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f"{base_filename}.pdf"), bbox_inches='tight')
    plt.close()
    
    return fig, base_filename

def main():
    st.title("Publication-Quality Concentration Profiles")
    
    solutions, params_list, lys, c_cus, c_nis, load_logs = load_solutions(SOLUTION_DIR)
    
    if load_logs:
        with st.expander("Load Log"):
            for log in load_logs:
                st.write(log)
    
    if not solutions:
        st.error("No valid solution files found in pinn_solutions directory. Please check the directory and file contents.")
        return
    
    st.write(f"Loaded {len(solutions)} solutions. Unique Ly: {len(set(lys))}, C_Cu: {len(set(c_cus))}, C_Ni: {len(set(c_nis))}")
    
    lys = sorted(set(lys))
    c_cus = sorted(set(c_cus))
    c_nis = sorted(set(c_nis))
    
    st.subheader("Select Parameters")
    ly_choice = st.selectbox("Domain Height (Ly, μm)", options=lys, format_func=lambda x: f"{x:.1f}")
    c_cu_choice = st.selectbox("Cu Boundary Concentration (mol/cc)", options=c_cus, format_func=lambda x: f"{x:.1e}")
    c_ni_choice = st.selectbox("Ni Boundary Concentration (mol/cc)", options=c_nis, format_func=lambda x: f"{x:.1e}")
    
    use_custom_params = st.checkbox("Use Custom Parameters for Interpolation", value=False)
    if use_custom_params:
        ly_target = st.number_input(
            "Custom Ly (μm)",
            min_value=30.0,
            max_value=120.0,
            value=ly_choice,
            step=0.1,
            format="%.1f"
        )
        c_cu_target = st.number_input(
            "Custom C_Cu (mol/cc)",
            min_value=1.5e-3,
            max_value=2.9e-3,
            value=c_cu_choice,
            step=0.1e-3,
            format="%.1e"
        )
        c_ni_target = st.number_input(
            "Custom C_Ni (mol/cc)",
            min_value=4.0e-4,
            max_value=1.8e-3,
            value=c_ni_choice,
            step=0.1e-4,
            format="%.1e"
        )
    else:
        ly_target, c_cu_target, c_ni_target = ly_choice, c_cu_choice, c_ni_choice
    
    st.subheader("Visualization Settings")
    cmap_cu = st.selectbox("Cu Heatmap Colormap", options=COLORMAPS, index=COLORMAPS.index('viridis'))
    cmap_ni = st.selectbox("Ni Heatmap Colormap", options=COLORMAPS, index=COLORMAPS.index('magma'))
    sidebar_metric = st.selectbox("Sidebar Metric for Curves", options=['mean_cu', 'mean_ni', 'loss'], index=0)
    
    try:
        solution = load_and_interpolate_solution(solutions, params_list, ly_target, c_cu_target, c_ni_target)
    except Exception as e:
        st.error(f"Failed to load or interpolate solution: {str(e)}")
        return
    
    st.subheader("Solution Details")
    st.write(f"$L_y$ = {solution['params']['Ly']:.1f} μm")
    st.write(f"$C_{{Cu}}$ = {solution['params']['C_Cu']:.1e} mol/cc")
    st.write(f"$C_{{Ni}}$ = {solution['params']['C_Ni']:.1e} mol/cc")
    if solution.get('interpolated', False):
        st.write("**Status**: Interpolated solution")
    else:
        st.write("**Status**: Exact solution")
    
    st.subheader("2D Concentration Heatmaps")
    time_index = st.slider("Select Time Index for Heatmaps", 0, len(solution['times'])-1, len(solution['times'])-1)
    fig_2d, filename_2d = plot_2d_concentration(solution, time_index, cmap_cu=cmap_cu, cmap_ni=cmap_ni)
    st.pyplot(fig_2d)
    st.download_button(
        label="Download 2D Plot as PNG",
        data=open(os.path.join("figures", f"{filename_2d}.png"), "rb").read(),
        file_name=f"{filename_2d}.png",
        mime="image/png"
    )
    st.download_button(
        label="Download 2D Plot as PDF",
        data=open(os.path.join("figures", f"{filename_2d}.pdf"), "rb").read(),
        file_name=f"{filename_2d}.pdf",
        mime="application/pdf"
    )
    
    st.subheader("Centerline Concentration Curves")
    time_indices = st.multiselect(
        "Select Time Indices for Curves",
        options=list(range(len(solution['times']))),
        default=[0, len(solution['times'])//4, len(solution['times'])//2, 3*len(solution['times'])//4, len(solution['times'])-1],
        format_func=lambda x: f"t = {solution['times'][x]:.1f} s"
    )
    if time_indices:
        fig_curves, filename_curves = plot_centerline_curves(solution, time_indices, sidebar_metric=sidebar_metric)
        st.pyplot(fig_curves)
        st.download_button(
            label="Download Centerline Plot as PNG",
            data=open(os.path.join("figures", f"{filename_curves}.png"), "rb").read(),
            file_name=f"{filename_curves}.png",
            mime="image/png"
        )
        st.download_button(
            label="Download Centerline Plot as PDF",
            data=open(os.path.join("figures", f"{filename_curves}.pdf"), "rb").read(),
            file_name=f"{filename_curves}.pdf",
            mime="application/pdf"
        )
    
    st.subheader("Parameter Sweep Curves")
    param_options = [(ly, c_cu, c_ni) for ly, c_cu, c_ni in params_list]
    param_labels = [f"$L_y$={ly:.1f}, $C_{{Cu}}$={c_cu:.1e}, $C_{{Ni}}$={c_ni:.1e}" for ly, c_cu, c_ni in param_options]
    default_params = param_options[:min(4, len(param_options))]
    selected_labels = st.multiselect(
        "Select Parameter Combinations",
        options=param_labels,
        default=[param_labels[param_options.index(p)] for p in default_params],
        format_func=lambda x: x
    )
    selected_params = [param_options[param_labels.index(label)] for label in selected_labels]
    sweep_time_index = st.slider("Select Time Index for Sweep", 0, len(solution['times'])-1, len(solution['times'])-1)
    
    if selected_params:
        fig_sweep, filename_sweep = plot_parameter_sweep(solutions, params_list, selected_params, sweep_time_index, sidebar_metric=sidebar_metric)
        st.pyplot(fig_sweep)
        st.download_button(
            label="Download Sweep Plot as PNG",
            data=open(os.path.join("figures", f"{filename_sweep}.png"), "rb").read(),
            file_name=f"{filename_sweep}.png",
            mime="image/png"
        )
        st.download_button(
            label="Download Sweep Plot as PDF",
            data=open(os.path.join("figures", f"{filename_sweep}.pdf"), "rb").read(),
            file_name=f"{filename_sweep}.pdf",
            mime="application/pdf"
        )

if __name__ == "__main__":
    main()
