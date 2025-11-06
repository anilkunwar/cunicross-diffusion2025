import os
import pickle
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import matplotlib as mpl
import torch
import torch.nn as nn
import re
from matplotlib.colors import Normalize
from io import BytesIO

# =============================================================================
# CONFIGURATION
# =============================================================================
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

# =============================================================================
# LOAD SOLUTIONS
# =============================================================================
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
                    # Parse diffusion type
                    match = re.match(r"solution_(\w+)_ly_([\d.]+)_tmax_([\d.]+)\.pkl", fname)
                    if match:
                        raw_type = match.group(1)
                        type_map = {
                            'cross': 'crossdiffusion',
                            'cu_self': 'cu_selfdiffusion',
                            'ni_self': 'ni_selfdiffusion'
                        }
                        diff_type = type_map.get(raw_type.lower(), 'crossdiffusion')
                        sol['diffusion_type'] = diff_type
                    else:
                        sol['diffusion_type'] = 'crossdiffusion'
                    load_logs.append(
                        f"{fname}: Loaded. Cu: {c1_min:.2e} to {c1_max:.2e}, Ni: {c2_min:.2e} to {c2_max:.2e}, "
                        f"Ly={param_tuple[0]:.1f}, C_Cu={param_tuple[1]:.1e}, C_Ni={param_tuple[2]:.1e}, Type={sol['diffusion_type']}"
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

# =============================================================================
# INTERPOLATOR
# =============================================================================
class MultiParamAttentionInterpolator(nn.Module):
    def __init__(self, sigma=0.2, num_heads=4, d_head=8):
        super().__init__()
        self.sigma = sigma
        self.num_heads = num_heads
        self.d_head = d_head
        self.W_q = nn.Linear(3, self.num_heads * self.d_head)
        self.W_k = nn.Linear(3, self.num_heads * self.d_head)

    def forward(self, solutions, params_list, ly_target, c_cu_target, c_ni_target):
        if not solutions or not params_list:
            raise ValueError("No solutions or parameters available for interpolation.")

        lys = np.array([p[0] for p in params_list])
        c_cus = np.array([p[1] for p in params_list])
        c_nis = np.array([p[2] for p in params_list])

        ly_norm = (lys - 30.0) / (120.0 - 30.0)
        c_cu_norm = (c_cus - 0.0) / (2.9e-3 - 0.0)
        c_ni_norm = (c_nis - 0.0) / (1.8e-3 - 0.0)

        target_ly_norm = (ly_target - 30.0) / (120.0 - 30.0)
        target_c_cu_norm = (c_cu_target - 0.0) / (2.9e-3 - 0.0)
        target_c_ni_norm = (c_ni_target - 0.0) / (1.8e-3 - 0.0)

        params_tensor = torch.tensor(np.stack([ly_norm, c_cu_norm, c_ni_norm], axis=1), dtype=torch.float32)
        target_params_tensor = torch.tensor([[target_ly_norm, target_c_cu_norm, target_c_ni_norm]], dtype=torch.float32)

        queries = self.W_q(target_params_tensor)
        keys = self.W_k(params_tensor)

        queries = queries.view(1, self.num_heads, self.d_head)
        keys = keys.view(len(params_list), self.num_heads, self.d_head)

        attn_logits = torch.einsum('nhd,mhd->nmh', keys, queries) / np.sqrt(self.d_head)
        attn_weights = torch.softmax(attn_logits, dim=0)
        attn_weights = attn_weights.mean(dim=2).squeeze(1)

        scaled_distances = torch.sqrt(
            ((torch.tensor(ly_norm) - target_ly_norm) / self.sigma)**2 +
            ((torch.tensor(c_cu_norm) - target_c_cu_norm) / self.sigma)**2 +
            ((torch.tensor(c_ni_norm) - target_c_ni_norm) / self.sigma)**2
        )
        spatial_weights = torch.exp(-scaled_distances**2 / 2)
        spatial_weights /= spatial_weights.sum()

        combined_weights = attn_weights * spatial_weights
        combined_weights /= combined_weights.sum()

        return self._physics_aware_interpolation(solutions, combined_weights.detach().numpy(), ly_target, c_cu_target, c_ni_target)

    def _physics_aware_interpolation(self, solutions, weights, ly_target, c_cu_target, c_ni_target):
        Lx = solutions[0]['params']['Lx']
        t_max = solutions[0]['params']['t_max']
        x_coords = np.linspace(0, Lx, 50)
        y_coords = np.linspace(0, ly_target, 50)
        times = np.linspace(0, t_max, 50)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        c1_interp = np.zeros((len(times), 50, 50))
        c2_interp = np.zeros((len(times), 50, 50))

        for t_idx in range(len(times)):
            for sol, weight in zip(solutions, weights):
                scale_factor = ly_target / sol['params']['Ly']
                Y_scaled = sol['Y'][0, :] * scale_factor
                interp_c1 = RegularGridInterpolator(
                    (sol['X'][:, 0], Y_scaled), sol['c1_preds'][t_idx],
                    method='linear', bounds_error=False, fill_value=0
                )
                interp_c2 = RegularGridInterpolator(
                    (sol['X'][:, 0], Y_scaled), sol['c2_preds'][t_idx],
                    method='linear', bounds_error=False, fill_value=0
                )
                points = np.stack([X.flatten(), Y.flatten()], axis=1)
                c1_interp[t_idx] += weight * interp_c1(points).reshape(50, 50)
                c2_interp[t_idx] += weight * interp_c2(points).reshape(50, 50)

        c1_interp[:, :, 0] = c_cu_target
        c2_interp[:, :, -1] = c_ni_target

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

    interpolator = MultiParamAttentionInterpolator(sigma=0.2)
    return interpolator(solutions, params_list, ly_target, c_cu_target, c_ni_target)

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================
def plot_2d_concentration(solution, time_index, output_dir="figures", cmap_cu='viridis', cmap_ni='magma', vmin_cu=None, vmax_cu=None, vmin_ni=None, vmax_ni=None):
    x_coords = solution['X'][:, 0] if solution['X'].ndim == 2 else solution['X']
    y_coords = solution['Y'][0, :] if solution['Y'].ndim == 2 else solution['Y']
    t_val = solution['times'][time_index]
    Lx = solution['params']['Lx']
    Ly = solution['params']['Ly']
    c1 = solution['c1_preds'][time_index]
    c2 = solution['c2_preds'][time_index]

    cu_min = vmin_cu if vmin_cu is not None else 0
    cu_max = vmax_cu if vmax_cu is not None else np.max(c1)
    ni_min = vmin_ni if vmin_ni is not None else 0
    ni_max = vmax_ni if vmax_ni is not None else np.max(c2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    im1 = ax1.imshow(c1, origin='lower', extent=[0, Lx, 0, Ly], cmap=cmap_cu, vmin=cu_min, vmax=cu_max)
    ax1.set_xlabel('x (μm)')
    ax1.set_ylabel('y (μm)')
    ax1.set_title(f'Cu Concentration, t = {t_val:.1f} s')
    ax1.grid(True)
    cb1 = fig.colorbar(im1, ax=ax1, label='Cu Conc. (mol/cc)', format='%.1e')
    cb1.ax.tick_params(labelsize=10)

    im2 = ax2.imshow(c2, origin='lower', extent=[0, Lx, 0, Ly], cmap=cmap_ni, vmin=ni_min, vmax=ni_max)
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
    fig.subplots_adjust(wspace=0.3)

    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"conc_2d_t_{t_val:.1f}_ly_{Ly:.1f}_ccu_{solution['params']['C_Cu']:.1e}_cni_{solution['params']['C_Ni']:.1e}"
    plt.savefig(os.path.join(output_dir, f"{base_filename}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f"{base_filename}.pdf"), bbox_inches='tight')
    plt.close()
    return fig, base_filename

def plot_centerline_curves(solution, time_indices, sidebar_metric='mean_cu', output_dir="figures",
                           label_size=12, title_size=14, tick_label_size=10, legend_loc='upper right',
                           curve_colormap='viridis', axis_linewidth=1.5, tick_major_width=1.5,
                           tick_major_length=4.0, fig_width=8.0, fig_height=6.0, curve_linewidth=1.0,
                           grid_alpha=0.3, grid_linestyle='--', legend_frameon=True, legend_framealpha=0.8):
    x_coords = solution['X'][:, 0] if solution['X'].ndim == 2 else solution['X']
    y_coords = solution['Y'][0, :] if solution['Y'].ndim == 2 else solution['Y']
    Lx = solution['params']['Lx']
    Ly = solution['params']['Ly']
    center_idx = len(x_coords) // 2
    times = solution['times']

    if sidebar_metric == 'loss' and 'loss' in solution:
        sidebar_data = solution['loss'][:len(times)]
        sidebar_label = 'Loss'
    elif sidebar_metric == 'mean_cu':
        sidebar_data = [np.mean(c1) for c1 in solution['c1_preds']]
        sidebar_label = 'Mean Cu Conc. (mol/cc)'
    else:
        sidebar_data = [np.mean(c2) for c2 in solution['c2_preds']]
        sidebar_label = 'Mean Ni Conc. (mol/cc)'

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.05, 0.5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[3])

    colors = cm.get_cmap(curve_colormap)(np.linspace(0, 1, len(time_indices)))
    for idx, t_idx in enumerate(time_indices):
        t_val = times[t_idx]
        c1 = solution['c1_preds'][t_idx][:, center_idx]
        c2 = solution['c2_preds'][t_idx][:, center_idx]
        ax1.plot(y_coords, c1, label=f't = {t_val:.1f} s', color=colors[idx], linewidth=curve_linewidth)
        ax2.plot(y_coords, c2, label=f't = {t_val:.1f} s', color=colors[idx], linewidth=curve_linewidth)

    for ax in [ax1, ax2, ax3]:
        for spine in ax.spines.values():
            spine.set_linewidth(axis_linewidth)
        ax.tick_params(axis='both', which='major', width=tick_major_width, length=tick_major_length, labelsize=tick_label_size)
        ax.grid(True, linestyle=grid_linestyle, alpha=grid_alpha)

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
    ax1.set_title(f'Cu at x = {x_coords[center_idx]:.1f} μm', fontsize=title_size)
    ax1.legend(fontsize=8, loc=legend_params['loc'], bbox_to_anchor=legend_params['bbox'],
               frameon=legend_frameon, framealpha=legend_framealpha)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    ax2.set_xlabel('y (μm)', fontsize=label_size)
    ax2.set_ylabel('Ni Conc. (mol/cc)', fontsize=label_size)
    ax2.set_title(f'Ni at x = {x_coords[center_idx]:.1f} μm', fontsize=title_size)
    ax2.legend(fontsize=8, loc=legend_params['loc'], bbox_to_anchor=legend_params['bbox'],
               frameon=legend_frameon, framealpha=legend_framealpha)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    ax3.plot(sidebar_data, times, 'k-', linewidth=curve_linewidth)
    ax3.set_xlabel(sidebar_label, fontsize=label_size)
    ax3.set_ylabel('Time (s)', fontsize=label_size)
    ax3.set_title('Metric vs. Time', fontsize=title_size)
    ax3.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    param_text = f"$L_y$ = {Ly:.1f} μm, $C_{{Cu}}$ = {solution['params']['C_Cu']:.1e}, $C_{{Ni}}$ = {solution['params']['C_Ni']:.1e}"
    if solution.get('interpolated', False):
        param_text += " (Interpolated)"
    fig.suptitle(f'Centerline Concentration Profiles\n{param_text}', fontsize=title_size)

    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"conc_centerline_ly_{Ly:.1f}_ccu_{solution['params']['C_Cu']:.1e}_cni_{solution['params']['C_Ni']:.1e}"
    plt.savefig(os.path.join(output_dir, f"{base_filename}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f"{base_filename}.pdf"), dpi=300, bbox_inches='tight')
    plt.close()
    return fig, base_filename

def plot_parameter_sweep(solutions, params_list, selected_params, time_index, sidebar_metric='mean_cu', output_dir="figures",
                         label_size=12, title_size=14, tick_label_size=10, legend_loc='upper right',
                         curve_colormap='tab10', axis_linewidth=1.5, tick_major_width=1.5,
                         tick_major_length=4.0, fig_width=8.0, fig_height=6.0, curve_linewidth=1.0,
                         grid_alpha=0.3, grid_linestyle='--', legend_frameon=True, legend_framealpha=0.8):
    Lx = solutions[0]['params']['Lx']
    center_idx = 25
    t_val = solutions[0]['times'][time_index]

    sidebar_data = []
    sidebar_labels = []
    for sol, params in zip(solutions, params_list):
        if params in selected_params:
            if sidebar_metric == 'loss' and 'loss' in sol:
                sidebar_data.append(sol['loss'][time_index])
            elif sidebar_metric == 'mean_cu':
                sidebar_data.append(np.mean(sol['c1_preds'][time_index]))
            else:
                sidebar_data.append(np.mean(sol['c2_preds'][time_index]))
            ly, c_cu, c_ni = params
            label = f'$L_y$={ly:.1f}, $C_{{Cu}}$={c_cu:.1e}, $C_{{Ni}}$={c_ni:.1e}'
            if sol.get('interpolated', False):
                label += " (Interpolated)"
            sidebar_labels.append(label)

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.05, 0.5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[3])

    colors = cm.get_cmap(curve_colormap)(np.linspace(0, 1, len(selected_params)))
    for idx, (sol, params) in enumerate(zip(solutions, params_list)):
        ly, c_cu, c_ni = params
        if params in selected_params:
            y_coords = sol['Y'][0, :] if sol['Y'].ndim == 2 else sol['Y']
            c1 = sol['c1_preds'][time_index][:, center_idx]
            c2 = sol['c2_preds'][time_index][:, center_idx]
            label = f'$L_y$={ly:.1f}, $C_{{Cu}}$={c_cu:.1e}, $C_{{Ni}}$={c_ni:.1e}'
            if sol.get('interpolated', False):
                label += " (Interpolated)"
            ax1.plot(y_coords, c1, label=label, color=colors[idx], linewidth=curve_linewidth)
            ax2.plot(y_coords, c2, label=label, color=colors[idx], linewidth=curve_linewidth)

    for ax in [ax1, ax2, ax3]:
        for spine in ax.spines.values():
            spine.set_linewidth(axis_linewidth)
        ax.tick_params(axis='both', which='major', width=tick_major_width, length=tick_major_length, labelsize=tick_label_size)
        ax.grid(True, linestyle=grid_linestyle, alpha=grid_alpha)

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
    ax1.set_title(f'Cu at x = {Lx/2:.1f} μm, t = {t_val:.1f} s', fontsize=title_size)
    ax1.legend(fontsize=8, loc=legend_params['loc'], bbox_to_anchor=legend_params['bbox'],
               frameon=legend_frameon, framealpha=legend_framealpha)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    ax2.set_xlabel('y (μm)', fontsize=label_size)
    ax2.set_ylabel('Ni Conc. (mol/cc)', fontsize=label_size)
    ax2.set_title(f'Ni at x = {Lx/2:.1f} μm, t = {t_val:.1f} s', fontsize=title_size)
    ax2.legend(fontsize=8, loc=legend_params['loc'], bbox_to_anchor=legend_params['bbox'],
               frameon=legend_frameon, framealpha=legend_framealpha)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    ax3.barh(range(len(sidebar_data)), sidebar_data, color='gray', edgecolor='black')
    ax3.set_yticks(range(len(sidebar_data)))
    ax3.set_yticklabels(sidebar_labels, fontsize=tick_label_size)
    ax3.set_xlabel(
        'Mean Cu Conc. (mol/cc)' if sidebar_metric == 'mean_cu' else 'Mean Ni Conc. (mol/cc)' if sidebar_metric == 'mean_ni' else 'Loss',
        fontsize=label_size
    )
    ax3.set_title('Metric per Parameter', fontsize=title_size)
    ax3.grid(True, axis='x', linestyle=grid_linestyle, alpha=grid_alpha)
    ax3.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    fig.suptitle('Concentration Profiles for Parameter Sweep', fontsize=title_size)

    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"conc_sweep_t_{t_val:.1f}"
    plt.savefig(os.path.join(output_dir, f"{base_filename}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f"{base_filename}.pdf"), bbox_inches='tight')
    plt.close()
    return fig, base_filename

def compute_fluxes_and_grads(c1_preds, c2_preds, X, Y, params):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)
    if X.shape != Y.shape:
        raise ValueError("X and Y must have same shape")
    x_coords = X[:, 0]
    y_coords = Y[0, :]
    dy = y_coords[1] - y_coords[0] if len(y_coords) > 1 else 1.0
    if dy <= 0:
        dy = 1.0

    D11 = params.get('D11', 1.0)
    D12 = params.get('D12', 0.0)
    D21 = params.get('D21', 0.0)
    D22 = params.get('D22', 1.0)

    J1_list, J2_list, grad1_list, grad2_list = [], [], [], []
    for c1, c2 in zip(c1_preds, c2_preds):
        c1 = np.asarray(c1)
        c2 = np.asarray(c2)
        if c1.shape != X.shape or c2.shape != X.shape:
            raise ValueError(f"Concentration shape {c1.shape} != grid {X.shape}")
        grad_c1_y = np.gradient(c1, dy, axis=0)
        grad_c2_y = np.gradient(c2, dy, axis=0)
        J1_y = -(D11 * grad_c1_y + D12 * grad_c2_y)
        J2_y = -(D21 * grad_c1_y + D22 * grad_c2_y)
        J1_list.append([None, J1_y])
        J2_list.append([None, J2_y])
        grad1_list.append(grad_c1_y)
        grad2_list.append(grad_c2_y)
    return J1_list, J2_list, grad1_list, grad2_list

def detect_uphill(solution, time_index):
    J1_y = solution['J1_preds'][time_index][1]
    grad_c1_y = solution['grad_c1_y'][time_index]
    J2_y = solution['J2_preds'][time_index][1]
    grad_c2_y = solution['grad_c2_y'][time_index]
    prod_cu = J1_y * grad_c1_y
    prod_ni = J2_y * grad_c2_y
    uphill_cu = prod_cu > 0
    uphill_ni = prod_ni > 0
    uphill_prod_cu_pos = np.where(uphill_cu, prod_cu, 0.0)
    uphill_prod_ni_pos = np.where(uphill_ni, prod_ni, 0.0)
    max_pos_cu = float(np.max(uphill_prod_cu_pos)) if np.any(uphill_cu) else 0.0
    max_pos_ni = float(np.max(uphill_prod_ni_pos)) if np.any(uphill_ni) else 0.0
    total_cells = prod_cu.size
    frac_uphill_cu = float(np.count_nonzero(uphill_cu) / total_cells)
    frac_uphill_ni = float(np.count_nonzero(uphill_ni) / total_cells)
    avg_pos_cu = np.mean(uphill_prod_cu_pos[uphill_prod_cu_pos > 0]) if np.any(uphill_cu) else 0.0
    avg_pos_ni = np.mean(uphill_prod_ni_pos[uphill_prod_ni_pos > 0]) if np.any(uphill_ni) else 0.0
    total_intensity_cu = np.sum(uphill_prod_cu_pos)
    total_intensity_ni = np.sum(uphill_prod_ni_pos)
    return (uphill_cu, uphill_ni,
            uphill_prod_cu_pos, uphill_prod_ni_pos,
            max_pos_cu, max_pos_ni,
            frac_uphill_cu, frac_uphill_ni,
            avg_pos_cu, avg_pos_ni,
            total_intensity_cu, total_intensity_ni)

def fig_to_bytes(fig, fmt='png'):
    buf = BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches='tight')
    buf.seek(0)
    return buf

def plot_flux_vs_gradient(solution, time_index, figsize=(6,3), marker_size=12, linewidth=1.2,
                          label_fontsize=12, title_fontsize=14, scatter_alpha=0.6,
                          marker_edgewidth=0.2, output_dir="figures"):
    J1_y = np.array(solution['J1_preds'][time_index][1]).flatten()
    J2_y = np.array(solution['J2_preds'][time_index][1]).flatten()
    grad_c1_y = np.array(solution['grad_c1_y'][time_index]).flatten()
    grad_c2_y = np.array(solution['grad_c2_y'][time_index]).flatten()
    uphill_cu = J1_y * grad_c1_y > 0
    uphill_ni = J2_y * grad_c2_y > 0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(grad_c1_y, J1_y, s=marker_size, alpha=scatter_alpha, edgecolors='none', label='Cu (all)')
    if uphill_cu.any():
        axes[0].scatter(grad_c1_y[uphill_cu], J1_y[uphill_cu], s=marker_size*1.1,
                        edgecolors='k', linewidths=marker_edgewidth, label='Uphill (Cu)')
    axes[0].set_xlabel('∇c (y)', fontsize=label_fontsize)
    axes[0].set_ylabel('J_y', fontsize=label_fontsize)
    axes[0].set_title('Cu Flux vs ∇c', fontsize=title_fontsize)
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.3)

    axes[1].scatter(grad_c2_y, J2_y, s=marker_size, alpha=scatter_alpha, edgecolors='none', label='Ni (all)')
    if uphill_ni.any():
        axes[1].scatter(grad_c2_y[uphill_ni], J2_y[uphill_ni], s=marker_size*1.1,
                        edgecolors='k', linewidths=marker_edgewidth, label='Uphill (Ni)')
    axes[1].set_xlabel('∇c (y)', fontsize=label_fontsize)
    axes[1].set_ylabel('J_y', fontsize=label_fontsize)
    axes[1].set_title('Ni Flux vs ∇c', fontsize=title_fontsize)
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.3)

    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=label_fontsize-2)
        ax.legend(fontsize=label_fontsize-2)

    fig.subplots_adjust(wspace=0.3)
    Ly = solution['params']['Ly']
    t_val = solution['times'][time_index]
    base_filename = f"flux_vs_grad_t_{t_val:.1f}_ly_{Ly:.1f}_ccu_{solution['params']['C_Cu']:.1e}_cni_{solution['params']['C_Ni']:.1e}"
    os.makedirs(output_dir, exist_ok=True)
    for ext in ('.png', '.pdf'):
        fig.savefig(os.path.join(output_dir, base_filename + ext), dpi=300, bbox_inches='tight')
    plt.close()
    return fig, base_filename

def plot_uphill_heatmap(solution, time_index, cmap='viridis', vmin=None, vmax=None,
                        figsize=(14, 6), colorbar=True, cbar_label='J·∇c',
                        label_fontsize=13, title_fontsize=15, downsample=1,
                        output_dir="figures"):
    x_coords = solution['X'][:, 0] if solution['X'].ndim == 2 else solution['X']
    y_coords = solution['Y'][0, :] if solution['Y'].ndim == 2 else solution['Y']
    t_val = solution['times'][time_index]

    (uphill_cu, uphill_ni,
     uphill_prod_cu_pos, uphill_prod_ni_pos,
     max_pos_cu, max_pos_ni,
     frac_cu, frac_ni,
     avg_pos_cu, avg_pos_ni,
     total_intensity_cu, total_intensity_ni) = detect_uphill(solution, time_index)

    z1 = uphill_prod_cu_pos[::downsample, ::downsample]
    z2 = uphill_prod_ni_pos[::downsample, ::downsample]
    x_ds = x_coords[::downsample]
    y_ds = y_coords[::downsample]

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=False)

    im1 = axes[0].imshow(z1, origin='lower', aspect='equal',
                         extent=(x_ds[0], x_ds[-1], y_ds[0], y_ds[-1]),
                         cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bilinear')
    axes[0].set_title(f'Cu Uphill\n(max={max_pos_cu:.2e}, avg={avg_pos_cu:.2e})',
                      fontsize=title_fontsize, pad=12)
    axes[0].set_xlabel('x (μm)', fontsize=label_fontsize, labelpad=8)
    axes[0].set_ylabel('y (μm)', fontsize=label_fontsize, labelpad=8)
    axes[0].tick_params(axis='both', which='major', labelsize=label_fontsize-2)

    im2 = axes[1].imshow(z2, origin='lower', aspect='equal',
                         extent=(x_ds[0], x_ds[-1], y_ds[0], y_ds[-1]),
                         cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bilinear')
    axes[1].set_title(f'Ni Uphill\n(max={max_pos_ni:.2e}, avg={avg_pos_ni:.2e})',
                      fontsize=title_fontsize, pad=12)
    axes[1].set_xlabel('x (μm)', fontsize=label_fontsize, labelpad=8)
    axes[1].set_ylabel('', fontsize=label_fontsize, labelpad=8)
    axes[1].tick_params(axis='both', which='major', labelsize=label_fontsize-2)

    for ax in axes:
        ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.3)

    if colorbar:
        cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), orientation='vertical',
                            fraction=0.046, pad=0.08, shrink=0.8)
        cbar.set_label(cbar_label, fontsize=label_fontsize, labelpad=10)
        cbar.ax.tick_params(labelsize=label_fontsize-2)

    fig.suptitle(f'Uphill Diffusion (positive J·∇c) @ t = {t_val:.2f} s',
                 fontsize=title_fontsize + 2, y=0.96)
    fig.subplots_adjust(left=0.08, right=0.88, top=0.88, bottom=0.12, wspace=0.40)

    Ly = solution['params']['Ly']
    base_filename = f"uphill_heatmap_t_{t_val:.1f}_ly_{Ly:.1f}_ccu_{solution['params']['C_Cu']:.1e}_cni_{solution['params']['C_Ni']:.1e}"
    os.makedirs(output_dir, exist_ok=True)
    for ext in ('.png', '.pdf'):
        fig.savefig(os.path.join(output_dir, base_filename + ext), dpi=300, bbox_inches='tight')
    plt.close(fig)

    return (fig, max_pos_cu, max_pos_ni, frac_cu, frac_ni,
            avg_pos_cu, avg_pos_ni, total_intensity_cu, total_intensity_ni, base_filename)

def plot_uphill_over_time(solution, figsize=(8,3), linewidth=1.6, marker_size=6,
                          label_fontsize=12, title_fontsize=14, output_dir="figures"):
    times = np.array(solution['times'])
    max_pos_cu_list, max_pos_ni_list = [], []
    frac_cu_list, frac_ni_list = [], []
    avg_pos_cu_list, avg_pos_ni_list = [], []
    total_intensity_cu_list, total_intensity_ni_list = [], []

    for t_idx in range(len(times)):
        (_, _, _, _, max_pos_cu, max_pos_ni, frac_cu, frac_ni,
         avg_pos_cu, avg_pos_ni, total_intensity_cu, total_intensity_ni) = detect_uphill(solution, t_idx)
        max_pos_cu_list.append(max_pos_cu)
        max_pos_ni_list.append(max_pos_ni)
        frac_cu_list.append(frac_cu)
        frac_ni_list.append(frac_ni)
        avg_pos_cu_list.append(avg_pos_cu)
        avg_pos_ni_list.append(avg_pos_ni)
        total_intensity_cu_list.append(total_intensity_cu)
        total_intensity_ni_list.append(total_intensity_ni)

    # Max
    fig_max, ax = plt.subplots(figsize=figsize)
    ax.plot(times, max_pos_cu_list, marker='o', markersize=marker_size, linewidth=linewidth, label='Max (Cu)')
    ax.plot(times, max_pos_ni_list, marker='s', markersize=marker_size, linewidth=linewidth, label='Max (Ni)')
    ax.set_xlabel('Time (s)', fontsize=label_fontsize)
    ax.set_ylabel('Max positive J·∇c', fontsize=label_fontsize)
    ax.set_title('Max Uphill Intensity', fontsize=title_fontsize)
    ax.legend(fontsize=label_fontsize-2)
    ax.grid(True, linestyle='--', linewidth=0.3)
    fig_max.tight_layout()

    # Average
    fig_avg, ax = plt.subplots(figsize=figsize)
    ax.plot(times, avg_pos_cu_list, marker='o', markersize=marker_size, linewidth=linewidth, label='Avg (Cu)')
    ax.plot(times, avg_pos_ni_list, marker='s', markersize=marker_size, linewidth=linewidth, label='Avg (Ni)')
    ax.set_xlabel('Time (s)', fontsize=label_fontsize)
    ax.set_ylabel('Avg positive J·∇c', fontsize=label_fontsize)
    ax.set_title('Average Uphill Intensity', fontsize=title_fontsize)
    ax.legend(fontsize=label_fontsize-2)
    ax.grid(True, linestyle='--', linewidth=0.3)
    fig_avg.tight_layout()

    # Fraction
    fig_frac, ax = plt.subplots(figsize=(8,2.5))
    ax.plot(times, frac_cu_list, label='Cu', linewidth=linewidth)
    ax.plot(times, frac_ni_list, label='Ni', linewidth=linewidth)
    ax.set_xlabel('Time (s)', fontsize=label_fontsize)
    ax.set_ylabel('Uphill Fraction', fontsize=label_fontsize)
    ax.set_title('Uphill Fraction Over Time', fontsize=title_fontsize-2)
    ax.legend(fontsize=label_fontsize-2)
    ax.grid(True, linestyle='--', linewidth=0.3)
    fig_frac.tight_layout()

    # Total Intensity
    fig_int, ax = plt.subplots(figsize=figsize)
    ax.plot(times, total_intensity_cu_list, marker='o', markersize=marker_size, linewidth=linewidth, label='Total (Cu)')
    ax.plot(times, total_intensity_ni_list, marker='s', markersize=marker_size, linewidth=linewidth, label='Total (Ni)')
    ax.set_xlabel('Time (s)', fontsize=label_fontsize)
    ax.set_ylabel('Total positive J·∇c', fontsize=label_fontsize)
    ax.set_title('Total Uphill Intensity', fontsize=title_fontsize)
    ax.legend(fontsize=label_fontsize-2)
    ax.grid(True, linestyle='--', linewidth=0.3)
    fig_int.tight_layout()

    Ly = solution['params']['Ly']
    base_filename = f"uphill_over_time_ly_{Ly:.1f}_ccu_{solution['params']['C_Cu']:.1e}_cni_{solution['params']['C_Ni']:.1e}"
    os.makedirs(output_dir, exist_ok=True)
    for f, name in zip([fig_max, fig_avg, fig_frac, fig_int], ["_max", "_avg", "_frac", "_intensity"]):
        for ext in ('.png', '.pdf'):
            f.savefig(os.path.join(output_dir, base_filename + name + ext), dpi=300, bbox_inches='tight')
        plt.close(f)

    return fig_max, fig_avg, fig_frac, fig_int, base_filename

@st.cache_data
def compute_summary_dataframe(all_solutions, time_index_for_summary=0):
    rows = []
    for s in all_solutions:
        if 'J1_preds' not in s:
            x_coords = s['X'][:, 0] if s['X'].ndim == 2 else s['X']
            y_coords = s['Y'][0, :] if s['Y'].ndim == 2 else s['Y']
            J1, J2, grad_c1, grad_c2 = compute_fluxes_and_grads(s['c1_preds'], s['c2_preds'], x_coords, y_coords, s['params'])
            s['J1_preds'] = J1
            s['J2_preds'] = J2
            s['grad_c1_y'] = grad_c1
            s['grad_c2_y'] = grad_c2
        try:
            (_, _, _, _, max_pos_cu, max_pos_ni, frac_cu, frac_ni, avg_pos_cu, avg_pos_ni, total_intensity_cu, total_intensity_ni) = detect_uphill(s, time_index_for_summary)
        except:
            max_pos_cu = max_pos_ni = frac_cu = frac_ni = avg_pos_cu = avg_pos_ni = total_intensity_cu = total_intensity_ni = 0.0
        rows.append({
            "Ly (μm)": s['params']['Ly'],
            "C_Cu": s['params']['C_Cu'],
            "C_Ni": s['params']['C_Ni'],
            "diffusion_type": s.get('diffusion_type', 'crossdiffusion'),
            "max_pos_JdotGrad_Cu": max_pos_cu,
            "max_pos_JdotGrad_Ni": max_pos_ni,
            "uphill_frac_Cu": frac_cu,
            "uphill_frac_Ni": frac_ni,
            "avg_pos_JdotGrad_Cu": avg_pos_cu,
            "avg_pos_JdotGrad_Ni": avg_pos_ni,
            "total_intensity_Cu": total_intensity_cu,
            "total_intensity_Ni": total_intensity_ni
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(['diffusion_type', 'Ly (μm)'])
    return df

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    st.title("Publication-Quality Concentration Profiles with Uphill Diffusion Analysis")

    solutions, params_list, lys, c_cus, c_nis, load_logs = load_solutions(SOLUTION_DIR)

    if load_logs:
        with st.expander("Load Log"):
            for log in load_logs:
                st.write(log)

    if not solutions:
        st.error("No valid solution files found in pinn_solutions directory.")
        return

    st.write(f"Loaded {len(solutions)} solutions. Unique Ly: {len(set(lys))}, C_Cu: {len(set(c_cus))}, C_Ni: {len(set(c_nis))}")

    lys = sorted(set(lys))
    c_cus = sorted(set(c_cus))
    c_nis = sorted(set(c_nis))

    st.subheader("Select Parameters for Single Solution")
    ly_choice = st.selectbox("Domain Height (Ly, μm)", options=lys, format_func=lambda x: f"{x:.1f}")
    c_cu_choice = st.selectbox("Cu Boundary Concentration (mol/cc)", options=c_cus, format_func=lambda x: f"{x:.1e}")
    c_ni_choice = st.selectbox("Ni Boundary Concentration (mol/cc)", options=c_nis, format_func=lambda x: f"{x:.1e}")

    use_custom_params = st.checkbox("Use Custom Parameters for Interpolation", value=False)
    if use_custom_params:
        ly_target = st.number_input("Custom Ly (μm)", 30.0, 120.0, ly_choice, 0.1, format="%.1f")
        c_cu_target = st.number_input("Custom C_Cu (mol/cc)", 0.0, 2.9e-3, max(c_cu_choice, 1.5e-3), 0.1e-3, format="%.1e")
        c_ni_target = st.number_input("Custom C_Ni (mol/cc)", 0.0, 1.8e-3, max(c_ni_choice, 1.0e-4), 0.1e-4, format="%.1e")
    else:
        ly_target, c_cu_target, c_ni_target = ly_choice, c_cu_choice, c_ni_choice

    st.subheader("Visualization Settings")
    cmap_cu = st.selectbox("Cu Heatmap Colormap", options=COLORMAPS, index=COLORMAPS.index('viridis'))
    cmap_ni = st.selectbox("Ni Heatmap Colormap", options=COLORMAPS, index=COLORMAPS.index('magma'))
    sidebar_metric = st.selectbox("Sidebar Metric for Curves", options=['mean_cu', 'mean_ni', 'loss'], index=0)

    st.subheader("Color Scale Limits")
    use_custom_scale = st.checkbox("Use custom color scale limits", value=False)
    custom_cu_min = custom_cu_max = custom_ni_min = custom_ni_max = None
    if use_custom_scale:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Cu Concentration Limits**")
            custom_cu_min = st.number_input("Cu Min", value=0.0, format="%.2e", key="cu_min")
            custom_cu_max = st.number_input("Cu Max", value=1.0, format="%.2e", key="cu_max")
        with col2:
            st.write("**Ni Concentration Limits**")
            custom_ni_min = st.number_input("Ni Min", value=0.0, format="%.2e", key="ni_min")
            custom_ni_max = st.number_input("Ni Max", value=1.0, format="%.2e", key="ni_max")

    if use_custom_scale:
        if custom_cu_min >= custom_cu_max or custom_ni_min >= custom_ni_max:
            st.error("Min must be less than max.")
            return

    with st.expander("Figure Customization"):
        label_size = st.slider("Axis Label Size", 8, 20, 12, 1)
        title_size = st.slider("Title Size", 10, 24, 14, 1)
        tick_label_size = st.slider("Tick Label Size", 6, 16, 10, 1)
        legend_loc = st.selectbox("Legend Location", ['upper right','upper left','lower right','lower left','center','best','right','left','above','below'], index=0)
        curve_colormap = st.selectbox("Curve Colormap", ['viridis','plasma','inferno','magma','tab10','Set1','Set2'], index=4)
        axis_linewidth = st.slider("Axis Line Width", 0.5, 3.0, 1.5, 0.1)
        tick_major_width = st.slider("Tick Major Width", 0.5, 3.0, 1.5, 0.1)
        tick_major_length = st.slider("Tick Major Length", 2.0, 10.0, 4.0, 0.5)
        fig_width = st.slider("Figure Width (inches)", 4.0, 12.0, 8.0, 0.5)
        fig_height = st.slider("Figure Height (inches)", 3.0, 8.0, 6.0, 0.5)
        curve_linewidth = st.slider("Curve Line Width", 0.5, 3.0, 1.0, 0.1)
        grid_alpha = st.slider("Grid Opacity", 0.0, 1.0, 0.3, 0.1)
        grid_linestyle = st.selectbox("Grid Line Style", ['--','-','-',':','-.'], index=0)
        legend_frameon = st.checkbox("Show Legend Frame", value=True)
        legend_framealpha = st.slider("Legend Frame Opacity", 0.0, 1.0, 0.8, 0.1)

    st.subheader("Uphill Diffusion Visualization Settings")
    cmap_uphill = st.selectbox("Uphill Heatmap Colormap", options=COLORMAPS, index=COLORMAPS.index('viridis'))
    use_custom_uphill_scale = st.checkbox("Use custom uphill color scale limits", value=False)
    custom_uphill_min = custom_uphill_max = None
    if use_custom_uphill_scale:
        custom_uphill_min = st.number_input("Uphill Min", value=0.0, format="%.2e", key="uphill_min")
        custom_uphill_max = st.number_input("Uphill Max", value=1e-6, format="%.2e", key="uphill_max")
    marker_size = st.slider("Scatter Marker Size (Flux vs Grad)", 1, 50, 12)
    linewidth = st.slider("Line Width (Temporal Plots)", 0.2, 5.0, 1.2)
    downsample = st.slider("Downsample Heatmap (Uphill)", 1, 8, 1)

    try:
        solution = load_and_interpolate_solution(solutions, params_list, ly_target, c_cu_target, c_ni_target)
    except Exception as e:
        st.error(f"Failed to load or interpolate solution: {e}")
        return

    if 'J1_preds' not in solution:
        st.info("Computing fluxes and gradients...")
        try:
            J1, J2, grad_c1, grad_c2 = compute_fluxes_and_grads(
                solution['c1_preds'], solution['c2_preds'], solution['X'], solution['Y'], solution['params']
            )
            solution['J1_preds'] = J1
            solution['J2_preds'] = J2
            solution['grad_c1_y'] = grad_c1
            solution['grad_c2_y'] = grad_c2
        except Exception as e:
            st.error(f"Failed to compute fluxes: {e}")
            return

    st.subheader("Solution Details")
    st.write(f"$L_y$ = {solution['params']['Ly']:.1f} μm")
    st.write(f"$C_{{Cu}}$ = {solution['params']['C_Cu']:.1e} mol/cc")
    st.write(f"$C_{{Ni}}$ = {solution['params']['C_Ni']:.1e} mol/cc")
    st.write(f"Diffusion Type: {solution.get('diffusion_type','crossdiffusion')}")
    st.write("**Status**: Interpolated solution" if solution.get('interpolated',False) else "**Status**: Exact solution")

    st.subheader("2D Concentration Heatmaps")
    time_index = st.slider("Select Time Index for Heatmaps", 0, len(solution['times'])-1, len(solution['times'])-1)
    fig_2d, filename_2d = plot_2d_concentration(
        solution, time_index, cmap_cu=cmap_cu, cmap_ni=cmap_ni,
        vmin_cu=custom_cu_min if use_custom_scale else None,
        vmax_cu=custom_cu_max if use_custom_scale else None,
        vmin_ni=custom_ni_min if use_custom_scale else None,
        vmax_ni=custom_ni_max if use_custom_scale else None
    )
    st.pyplot(fig_2d)
    for ext in ('png','pdf'):
        path = os.path.join("figures", f"{filename_2d}.{ext}")
        st.download_button(f"Download 2D Plot as {ext.upper()}", data=open(path,"rb").read(), file_name=f"{filename_2d}.{ext}", mime=f"image/{ext}" if ext=='png' else "application/pdf")

    st.subheader("Centerline Concentration Curves")
    time_indices = st.multiselect("Select Time Indices for Curves", options=list(range(len(solution['times']))),
                                  default=[0, len(solution['times'])//4, len(solution['times'])//2, 3*len(solution['times'])//4, len(solution['times'])-1],
                                  format_func=lambda i: f"t = {solution['times'][i]:.1f} s")
    if time_indices:
        fig_curves, filename_curves = plot_centerline_curves(
            solution, time_indices, sidebar_metric=sidebar_metric,
            label_size=label_size, title_size=title_size, tick_label_size=tick_label_size,
            legend_loc=legend_loc, curve_colormap=curve_colormap,
            axis_linewidth=axis_linewidth, tick_major_width=tick_major_width,
            tick_major_length=tick_major_length, fig_width=fig_width, fig_height=fig_height,
            curve_linewidth=curve_linewidth, grid_alpha=grid_alpha, grid_linestyle=grid_linestyle,
            legend_frameon=legend_frameon, legend_framealpha=legend_framealpha
        )
        st.pyplot(fig_curves)
        for ext in ('png','pdf'):
            path = os.path.join("figures", f"{filename_curves}.{ext}")
            st.download_button(f"Download Centerline Plot as {ext.upper()}", data=open(path,"rb").read(), file_name=f"{filename_curves}.{ext}", mime=f"image/{ext}" if ext=='png' else "application/pdf")

    st.subheader("Uphill Diffusion Analysis")

    st.markdown("**1) Flux vs Gradient**")
    fig_fg, filename_fg = plot_flux_vs_gradient(solution, time_index, figsize=(fig_width, fig_height/2), marker_size=marker_size, linewidth=linewidth)
    st.pyplot(fig_fg)
    for ext in ('png','pdf'):
        buf = fig_to_bytes(fig_fg, fmt=ext)
        st.download_button(f"Download Flux vs Gradient as {ext.upper()}", data=buf.read(), file_name=f"{filename_fg}.{ext}", mime=f"image/{ext}" if ext=='png' else "application/pdf")

    st.markdown("**2) Uphill Heatmaps**")
    fig_hm, max_cu, max_ni, frac_cu, frac_ni, avg_cu, avg_ni, tot_cu, tot_ni, filename_hm = plot_uphill_heatmap(
        solution, time_index, cmap=cmap_uphill,
        vmin=custom_uphill_min if use_custom_uphill_scale else None,
        vmax=custom_uphill_max if use_custom_uphill_scale else None,
        figsize=(fig_width+2, fig_height), downsample=downsample
    )
    st.pyplot(fig_hm)
    for ext in ('png','pdf'):
        buf = fig_to_bytes(fig_hm, fmt=ext)
        st.download_button(f"Download Uphill Heatmaps as {ext.upper()}", data=buf.read(), file_name=f"{filename_hm}.{ext}", mime=f"image/{ext}" if ext=='png' else "application/pdf")
    st.markdown(f"- **Max (Cu):** {max_cu:.3e} | **Max (Ni):** {max_ni:.3e}")
    st.markdown(f"- **Avg (Cu):** {avg_cu:.3e} | **Avg (Ni):** {avg_ni:.3e}")
    st.markdown(f"- **Total (Cu):** {tot_cu:.3e} | **Total (Ni):** {tot_ni:.3e}")
    st.markdown(f"- **Fraction (Cu):** {frac_cu*100:.2f}% | **Fraction (Ni):** {frac_ni*100:.2f}%")

    st.markdown("**3) Temporal Evolution of Uphill Metrics**")
    fig_max, fig_avg, fig_frac, fig_int, filename_time = plot_uphill_over_time(
        solution, figsize=(fig_width, fig_height/2), linewidth=linewidth, marker_size=max(4, int(marker_size/2))
    )
    for f, name in zip([fig_max, fig_avg, fig_frac, fig_int], ["Max", "Average", "Fraction", "Total Intensity"]):
        st.pyplot(f)
        for ext in ('png','pdf'):
            buf = fig_to_bytes(f, fmt=ext)
            st.download_button(f"Download {name} Evolution as {ext.upper()}", data=buf.read(), file_name=f"{filename_time}_{name.lower().replace(' ', '_')}.{ext}", mime=f"image/{ext}" if ext=='png' else "application/pdf")

    st.subheader("Parameter Sweep Curves")
    with st.expander("Add Custom Parameter Combinations for Sweep"):
        n_custom = st.number_input("Number of Custom Sets", 0, 5, 0, 1)
        custom_params = []
        for i in range(n_custom):
            st.write(f"**Custom Set {i+1}**")
            ly_c = st.number_input(f"Ly {i+1}", 30.0, 120.0, ly_choice, 0.1, format="%.1f", key=f"ly_c_{i}")
            cu_c = st.number_input(f"C_Cu {i+1}", 0.0, 2.9e-3, max(c_cu_choice,1.5e-3), 0.1e-3, format="%.1e", key=f"cu_c_{i}")
            ni_c = st.number_input(f"C_Ni {i+1}", 0.0, 1.8e-3, max(c_ni_choice,1.0e-4), 0.1e-4, format="%.1e", key=f"ni_c_{i}")
            custom_params.append((ly_c, cu_c, ni_c))

    param_options = [(ly, c_cu, c_ni) for ly, c_cu, c_ni in params_list]
    param_labels = [f"$L_y$={ly:.1f}, $C_{{Cu}}$={c_cu:.1e}, $C_{{Ni}}$={c_ni:.1e}" for ly, c_cu, c_ni in param_options]
    default_labels = [param_labels[param_options.index(p)] for p in param_options[:min(4,len(param_options))]]
    selected_labels = st.multiselect("Select Exact Parameter Combinations", options=param_labels, default=default_labels)
    selected_params = [param_options[param_labels.index(l)] for l in selected_labels]
    selected_params.extend(custom_params)

    sweep_solutions = []
    sweep_params_list = []
    for ly, c_cu, c_ni in selected_params:
        try:
            sol = load_and_interpolate_solution(solutions, params_list, ly, c_cu, c_ni)
            if 'J1_preds' not in sol:
                J1, J2, grad_c1, grad_c2 = compute_fluxes_and_grads(sol['c1_preds'], sol['c2_preds'], sol['X'], sol['Y'], sol['params'])
                sol['J1_preds'] = J1
                sol['J2_preds'] = J2
                sol['grad_c1_y'] = grad_c1
                sol['grad_c2_y'] = grad_c2
            sweep_solutions.append(sol)
            sweep_params_list.append((ly, c_cu, c_ni))
        except Exception as e:
            st.warning(f"Skip Ly={ly:.1f}, C_Cu={c_cu:.1e}, C_Ni={c_ni:.1e}: {e}")

    sweep_time_index = st.slider("Select Time Index for Sweep", 0, len(solution['times'])-1, len(solution['times'])-1)
    if sweep_solutions:
        fig_sweep, filename_sweep = plot_parameter_sweep(
            sweep_solutions, sweep_params_list, selected_params, sweep_time_index, sidebar_metric=sidebar_metric,
            label_size=label_size, title_size=title_size, tick_label_size=tick_label_size, legend_loc=legend_loc,
            curve_colormap=curve_colormap, axis_linewidth=axis_linewidth, tick_major_width=tick_major_width,
            tick_major_length=tick_major_length, fig_width=fig_width, fig_height=fig_height,
            curve_linewidth=curve_linewidth, grid_alpha=grid_alpha, grid_linestyle=grid_linestyle,
            legend_frameon=legend_frameon, legend_framealpha=legend_framealpha
        )
        st.pyplot(fig_sweep)
        for ext in ('png','pdf'):
            path = os.path.join("figures", f"{filename_sweep}.{ext}")
            st.download_button(f"Download Sweep Plot as {ext.upper()}", data=open(path,"rb").read(), file_name=f"{filename_sweep}.{ext}", mime=f"image/{ext}" if ext=='png' else "application/pdf")
    else:
        st.info("No solutions selected for parameter sweep.")

    st.subheader("Summary Table (all loaded solutions)")
    summary_time_idx = st.slider("Time index for summary metrics", 0, len(solutions[0]['times'])-1, len(solutions[0]['times'])-1)
    df_summary = compute_summary_dataframe(solutions, time_index_for_summary=summary_time_idx)
    st.dataframe(df_summary.style.format({
        "Ly (μm)": "{:.1f}",
        "C_Cu": "{:.1e}",
        "C_Ni": "{:.1e}",
        "max_pos_JdotGrad_Cu": "{:.2e}",
        "max_pos_JdotGrad_Ni": "{:.2e}",
        "uphill_frac_Cu": "{:.3f}",
        "uphill_frac_Ni": "{:.3f}",
        "avg_pos_JdotGrad_Cu": "{:.2e}",
        "avg_pos_JdotGrad_Ni": "{:.2e}",
        "total_intensity_Cu": "{:.2e}",
        "total_intensity_Ni": "{:.2e}"
    }))

    st.success("All visualizations generated! Files saved in `figures/` folder.")

if __name__ == "__main__":
    main()
