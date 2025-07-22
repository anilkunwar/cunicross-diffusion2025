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
    "viridis", "plasma", "inferno", "magma", "cividis",
    "Greys", "Purples", "Blues", "Greens", "Oranges", "Reds",
    "YlOrBr", "YlOrRd", "OrRd", "PuRd", "RdPu", "BuPu", "GnBu",
    "PuBu", "YlGnBu", "PuBuGn", "BuGn", "YlGn",
    "cubehelix", "binary", "gist_yarg", "gist_gray", "gray", "bone",
    "pink", "spring", "summer", "autumn", "winter",
    "PiYG", "PRGn", "BrBG", "PuOr", "RdGy", "RdBu", "RdYlBu", "RdYlGn",
    "Spectral", "coolwarm", "bwr", "seismic",
    "twilight", "twilight_shifted", "hsv",
    "Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1", "Set2", "Set3",
    "tab10", "tab20", "tab20b", "tab20c",
    "flag", "prism", "ocean", "gist_earth", "terrain", "gist_stern", "gnuplot",
    "gnuplot2", "CMRmap", "cubehelix", "brg", "gist_rainbow", "rainbow",
    "jet", "nipy_spectral", "gist_ncar",
    "viridis_r", "plasma_r", "inferno_r", "magma_r", "cividis_r", "Greys_r",
    "Purples_r", "Blues_r", "Greens_r", "Oranges_r", "Reds_r", "YlOrBr_r",
    "YlOrRd_r", "OrRd_r", "PuRd_r", "RdPu_r", "BuPu_r", "GnBu_r", "PuBu_r",
    "YlGnBu_r", "PuBuGn_r", "BuGn_r", "YlGn_r", "twilight_r", "twilight_shifted_r",
    "hsv_r", "Spectral_r", "coolwarm_r", "bwr_r", "seismic_r", "RdBu_r",
    "PiYG_r", "PRGn_r", "BrBG_r", "PuOr_r", "RdGy_r", "RdYlBu_r", "RdYlGn_r",
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
                if not all(key in sol for key in required_keys):
                    missing_keys = [key for key in required_keys if key not in sol]
                    load_logs.append(f"{fname}: Skipped - Missing keys: {missing_keys}")
                    continue
                
                # Check if c1_preds and c2_preds are lists
                if not (isinstance(sol['c1_preds'], list) and isinstance(sol['c2_preds'], list)):
                    load_logs.append(f"{fname}: Skipped - c1_preds or c2_preds is not a list. Types: c1_preds={type(sol['c1_preds'])}, c2_preds={type(sol['c2_preds'])}")
                    continue
                
                # Convert lists to NumPy arrays if necessary and validate
                try:
                    c1_preds = [np.array(pred, dtype=float) if not isinstance(pred, np.ndarray) else pred for pred in sol['c1_preds']]
                    c2_preds = [np.array(pred, dtype=float) if not isinstance(pred, np.ndarray) else pred for pred in sol['c2_preds']]
                    sol['c1_preds'] = c1_preds
                    sol['c2_preds'] = c2_preds
                    
                    # Validate data
                    if (any(np.any(np.isnan(pred)) for pred in c1_preds) or 
                        any(np.any(np.isnan(pred)) for pred in c2_preds) or
                        any(np.all(pred == 0) for pred in c1_preds) or 
                        any(np.all(pred == 0) for pred in c2_preds) or
                        any(np.any(pred < 0) for pred in c1_preds) or 
                        any(np.any(pred < 0) for pred in c2_preds)):
                        load_logs.append(f"{fname}: Skipped - Invalid data (NaNs, all zeros, or negative values).")
                        continue
                except Exception as e:
                    load_logs.append(f"{fname}: Skipped - Failed to process c1_preds/c2_preds: {str(e)}")
                    continue
                
                c1_min, c1_max = np.min(c1_preds[0]), np.max(c1_preds[0])
                c2_min, c2_max = np.min(c2_preds[0]), np.max(c2_preds[0])
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
            except Exception as e:
                load_logs.append(f"{fname}: Skipped - Failed to load: {str(e)}")
                continue
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

def plot_2d_concentration(solution, time_index, output_dir="figures", cmap_cu='viridis', cmap_ni='magma', vmin_cu=None, vmax_cu=None, vmin_ni=None, vmax_ni=None):
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
        vmin=vmin_cu if vmin_cu is not None else 0,
        vmax=vmax_cu if vmax_cu is not None else np.max(c1)
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
        vmin=vmin_ni if vmin_ni is not None else 0,
        vmax=vmax_ni if vmax_ni is not None else np.max(c2)
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

def plot_centerline_curves(
    solution, time_indices, sidebar_metric='mean_cu', output_dir="figures",
    label_size=12, title_size=14, tick_label_size=10, legend_loc='upper right',
    curve_colormap='viridis', axis_linewidth=1.5, tick_major_width=1.5,
    tick_major_length=4.0, fig_width=8.0, fig_height=6.0, curve_linewidth=1.0,
    grid_alpha=0.3, grid_linestyle='--', legend_frameon=True, legend_framealpha=0.8
):
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
    
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.05, 0.5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[3])
    
    # Centerline curves
    colors = cm.get_cmap(curve_colormap)(np.linspace(0, 1, len(time_indices)))
    for idx, t_idx in enumerate(time_indices):
        t_val = times[t_idx]
        c1 = solution['c1_preds'][t_idx][:, center_idx]
        c2 = solution['c2_preds'][t_idx][:, center_idx]
        ax1.plot(y_coords, c1, label=f't = {t_val:.1f} s', color=colors[idx], linewidth=curve_linewidth)
        ax2.plot(y_coords, c2, label=f't = {t_val:.1f} s', color=colors[idx], linewidth=curve_linewidth)
    
    # Axis styling
    for ax in [ax1, ax2, ax3]:
        for spine in ax.spines.values():
            spine.set_linewidth(axis_linewidth)
        ax.tick_params(
            axis='both',
            which='major',
            width=tick_major_width,
            length=tick_major_length,
            labelsize=tick_label_size
        )
        ax.grid(True, linestyle=grid_linestyle, alpha=grid_alpha)
    
    # Legend placement
    legend_positions = {
        'upper right': {'loc': 'upper right', 'bbox': None},
        'upper left': {'loc': 'upper left', 'bbox': None},
        'lower right': {'loc': 'lower right', 'bbox': None},
        'lower left': {'loc': 'lower left', 'bbox': None},
        'center': {'loc': 'center', 'bbox': None},
        'best': {'loc': 'best', 'bbox': None},
        'right': {'loc': 'center left', 'bbox': (1.05, 0.5)},
        'left': {'loc': 'center right', 'bbox': (-0.05, 0.5)},
        'above': {'loc': 'lower center', 'bbox': (0.5, 1.05)},
        'below': {'loc': 'upper center', 'bbox': (0.5, -0.05)}
    }
    legend_params = legend_positions.get(legend_loc, {'loc': 'upper right', 'bbox': None})
    
    ax1.set_xlabel('y (μm)', fontsize=label_size)
    ax1.set_ylabel('Cu Conc. (mol/cc)', fontsize=label_size)
    ax1.set_title(f'Cu at x = {Lx/2:.1f} μm', fontsize=title_size)
    ax1.legend(fontsize=8, loc=legend_params['loc'], bbox_to_anchor=legend_params['bbox'],
               frameon=legend_frameon, framealpha=legend_framealpha)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    ax2.set_xlabel('y (μm)', fontsize=label_size)
    ax2.set_ylabel('Ni Conc. (mol/cc)', fontsize=label_size)
    ax2.set_title(f'Ni at x = {Lx/2:.1f} μm', fontsize=title_size)
    ax2.legend(fontsize=8, loc=legend_params['loc'], bbox_to_anchor=legend_params['bbox'],
               frameon=legend_frameon, framealpha=legend_framealpha)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Sidebar plot
    ax3.plot(sidebar_data, times, 'k-', linewidth=curve_linewidth)
    ax3.set_xlabel(sidebar_label, fontsize=label_size)
    ax3.set_ylabel('Time (s)', fontsize=label_size)
    ax3.set_title('Metric vs. Time', fontsize=title_size)
    ax3.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    param_text = f"$L_y$ = {Ly:.1f} μm, $C_{{Cu}}$ = {solution['params']['C_Cu']:.1e}, $C_{{Ni}}$ = {solution['params']['C_Ni']:.1e}"
    if solution.get('interpolated', False):
        param_text += " (Interpolated)"
    fig.suptitle(f'Centerline Concentration Profiles\n{param_text}', fontsize=title_size)
    
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"conc_centerline_ly_{Ly:.1f}_ccu_{solution['params']['C_Cu']:.1e}_cni_{solution['params']['C_Ni']:.1e}"
    plt.savefig(os.path.join(output_dir, f"{base_filename}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f"{base_filename}.pdf"), bbox_inches='tight')
    plt.close()
    
    return fig, base_filename

def plot_parameter_sweep(
    solutions, params_list, selected_params, time_index, sidebar_metric='mean_cu', output_dir="figures",
    label_size=12, title_size=14, tick_label_size=10, legend_loc='upper right',
    curve_colormap='tab10', axis_linewidth=1.5, tick_major_width=1.5,
    tick_major_length=4.0, fig_width=8.0, fig_height=6.0, curve_linewidth=1.0,
    grid_alpha=0.3, grid_linestyle='--', legend_frameon=True, legend_framealpha=0.8
):
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
            label = f'$L_y$={ly:.1f}, $C_{{Cu}}$={c_cu:.1e}, $C_{{Ni}}$={c_ni:.1e}'
            if sol.get('interpolated', False):
                label += " (Interpolated)"
            sidebar_labels.append(label)
    
    # Create figure with custom size
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.05, 0.5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[3])
    
    # Parameter sweep curves
    colors = cm.get_cmap(curve_colormap)(np.linspace(0, 1, len(selected_params)))
    for idx, (sol, params) in enumerate(zip(solutions, params_list)):
        ly, c_cu, c_ni = params
        if params in selected_params:
            y_coords = sol['Y'][0, :]
            c1 = sol['c1_preds'][time_index][:, center_idx]
            c2 = sol['c2_preds'][time_index][:, center_idx]
            label = f'$L_y$={ly:.1f}, $C_{{Cu}}$={c_cu:.1e}, $C_{{Ni}}$={c_ni:.1e}'
            if sol.get('interpolated', False):
                label += " (Interpolated)"
            ax1.plot(y_coords, c1, label=label, color=colors[idx], linewidth=curve_linewidth)
            ax2.plot(y_coords, c2, label=label, color=colors[idx], linewidth=curve_linewidth)
    
    # Axis styling
    for ax in [ax1, ax2, ax3]:
        for spine in ax.spines.values():
            spine.set_linewidth(axis_line
