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
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

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

# Fixed boundary conditions from PINN formulation
C_CU_TOP = 0.0
C_CU_BOTTOM = 1.6e-3
C_NI_TOP = 1.25e-3
C_NI_BOTTOM = 0.0

def compute_boundary_modes(solutions, time_index=-1):
    """Compute the most frequent (mode) boundary values across all solutions"""
    if not solutions:
        return {}
    
    # Extract boundary values from all solutions at the specified time index
    top_cu_values = []
    bottom_ni_values = []
    left_cu_values = []
    left_ni_values = []
    right_cu_values = []
    right_ni_values = []
    
    for sol in solutions:
        c1 = sol['c1_preds'][time_index]  # Cu concentrations
        c2 = sol['c2_preds'][time_index]  # Ni concentrations
        
        # Top boundary (y=Ly) - Cu and Ni values
        top_cu_values.extend(c1[:, -1])
        top_ni_values = c2[:, -1]
        
        # Bottom boundary (y=0) - Cu and Ni values
        bottom_cu_values = c1[:, 0]
        bottom_ni_values.extend(c2[:, 0])
        
        # Left boundary (x=0) - both Cu and Ni
        left_cu_values.extend(c1[0, :])
        left_ni_values.extend(c2[0, :])
        
        # Right boundary (x=Lx) - both Cu and Ni
        right_cu_values.extend(c1[-1, :])
        right_ni_values.extend(c2[-1, :])
    
    # Convert to numpy arrays
    top_cu_values = np.array(top_cu_values)
    top_ni_values = np.array(top_ni_values)
    bottom_cu_values = np.array(bottom_cu_values)
    bottom_ni_values = np.array(bottom_ni_values)
    left_cu_values = np.array(left_cu_values)
    left_ni_values = np.array(left_ni_values)
    right_cu_values = np.array(right_cu_values)
    right_ni_values = np.array(right_ni_values)
    
    # Compute modes using scipy stats
    def safe_mode(values, default=0.0):
        """Safely compute mode, handling edge cases"""
        if len(values) == 0:
            return default
        try:
            mode_result = stats.mode(values, nan_policy='omit')
            if np.isscalar(mode_result.mode):
                return float(mode_result.mode)
            else:
                return float(mode_result.mode[0])
        except:
            # Fallback: use median if mode fails
            return float(np.median(values))
    
    boundary_modes = {
        'top_cu': safe_mode(top_cu_values),
        'top_ni': safe_mode(top_ni_values),
        'bottom_cu': safe_mode(bottom_cu_values),
        'bottom_ni': safe_mode(bottom_ni_values),
        'left_cu': safe_mode(left_cu_values),
        'left_ni': safe_mode(left_ni_values),
        'right_cu': safe_mode(right_cu_values),
        'right_ni': safe_mode(right_ni_values),
        'stats': {
            'top_cu_mean': float(np.mean(top_cu_values)),
            'top_cu_std': float(np.std(top_cu_values)),
            'top_ni_mean': float(np.mean(top_ni_values)),
            'top_ni_std': float(np.std(top_ni_values)),
            'bottom_cu_mean': float(np.mean(bottom_cu_values)),
            'bottom_cu_std': float(np.std(bottom_cu_values)),
            'bottom_ni_mean': float(np.mean(bottom_ni_values)),
            'bottom_ni_std': float(np.std(bottom_ni_values)),
            'left_cu_mean': float(np.mean(left_cu_values)),
            'left_cu_std': float(np.std(left_cu_values)),
            'left_ni_mean': float(np.mean(left_ni_values)),
            'left_ni_std': float(np.std(left_ni_values)),
            'right_cu_mean': float(np.mean(right_cu_values)),
            'right_cu_std': float(np.std(right_cu_values)),
            'right_ni_mean': float(np.mean(right_ni_values)),
            'right_ni_std': float(np.std(right_ni_values)),
        }
    }
    
    return boundary_modes

def validate_boundary_conditions(solution, tolerance=1e-6):
    """Validate that boundary conditions are properly satisfied"""
    validation_results = {
        'top_bc_cu': True,
        'top_bc_ni': True,
        'bottom_bc_cu': True,
        'bottom_bc_ni': True,
        'left_flux_cu': True,
        'left_flux_ni': True,
        'right_flux_cu': True,
        'right_flux_ni': True,
        'initial_condition': True,
        'details': []
    }
    
    # Check last time step for steady state
    t_idx = -1
    c1 = solution['c1_preds'][t_idx]
    c2 = solution['c2_preds'][t_idx]
    
    # Check top boundary (y=Ly) - Cu = 0, Ni = 1.25e-3
    top_cu_values = c1[:, -1]
    top_ni_values = c2[:, -1]
    top_cu_std = np.std(top_cu_values)
    top_ni_std = np.std(top_ni_values)
    top_cu_mean = np.mean(top_cu_values)
    top_ni_mean = np.mean(top_ni_values)
    if top_cu_std > tolerance or abs(top_cu_mean - C_CU_TOP) > tolerance:
        validation_results['top_bc_cu'] = False
        validation_results['details'].append(f"Top BC Cu not constant at {C_CU_TOP:.1e}: mean={top_cu_mean:.2e}, std={top_cu_std:.2e}")
    if top_ni_std > tolerance or abs(top_ni_mean - C_NI_TOP) > tolerance:
        validation_results['top_bc_ni'] = False
        validation_results['details'].append(f"Top BC Ni not constant at {C_NI_TOP:.1e}: mean={top_ni_mean:.2e}, std={top_ni_std:.2e}")
    
    # Check bottom boundary (y=0) - Cu = 1.6e-3, Ni = 0
    bottom_cu_values = c1[:, 0]
    bottom_ni_values = c2[:, 0]
    bottom_cu_std = np.std(bottom_cu_values)
    bottom_ni_std = np.std(bottom_ni_values)
    bottom_cu_mean = np.mean(bottom_cu_values)
    bottom_ni_mean = np.mean(bottom_ni_values)
    if bottom_cu_std > tolerance or abs(bottom_cu_mean - C_CU_BOTTOM) > tolerance:
        validation_results['bottom_bc_cu'] = False
        validation_results['details'].append(f"Bottom BC Cu not constant at {C_CU_BOTTOM:.1e}: mean={bottom_cu_mean:.2e}, std={bottom_cu_std:.2e}")
    if bottom_ni_std > tolerance or abs(bottom_ni_mean - C_NI_BOTTOM) > tolerance:
        validation_results['bottom_bc_ni'] = False
        validation_results['details'].append(f"Bottom BC Ni not constant at {C_NI_BOTTOM:.1e}: mean={bottom_ni_mean:.2e}, std={bottom_ni_std:.2e}")
    
    # Check left boundary (x=0) - zero flux for both
    left_flux_cu = np.mean(np.abs(c1[1, :] - c1[0, :]))
    left_flux_ni = np.mean(np.abs(c2[1, :] - c2[0, :]))
    if left_flux_cu > tolerance:
        validation_results['left_flux_cu'] = False
        validation_results['details'].append(f"Left flux Cu violation: {left_flux_cu:.2e}")
    if left_flux_ni > tolerance:
        validation_results['left_flux_ni'] = False
        validation_results['details'].append(f"Left flux Ni violation: {left_flux_ni:.2e}")
    
    # Check right boundary (x=Lx) - zero flux for both
    right_flux_cu = np.mean(np.abs(c1[-1, :] - c1[-2, :]))
    right_flux_ni = np.mean(np.abs(c2[-1, :] - c2[-2, :]))
    if right_flux_cu > tolerance:
        validation_results['right_flux_cu'] = False
        validation_results['details'].append(f"Right flux Cu violation: {right_flux_cu:.2e}")
    if right_flux_ni > tolerance:
        validation_results['right_flux_ni'] = False
        validation_results['details'].append(f"Right flux Ni violation: {right_flux_ni:.2e}")
    
    # Check initial condition (t=0)
    c1_initial = solution['c1_preds'][0]
    c2_initial = solution['c2_preds'][0]
    initial_cu_mean = np.mean(np.abs(c1_initial))
    initial_ni_mean = np.mean(np.abs(c2_initial))
    if initial_cu_mean > tolerance:
        validation_results['initial_condition'] = False
        validation_results['details'].append(f"Initial condition Cu not zero: mean={initial_cu_mean:.2e}")
    if initial_ni_mean > tolerance:
        validation_results['initial_condition'] = False
        validation_results['details'].append(f"Initial condition Ni not zero: mean={initial_ni_mean:.2e}")
    
    return validation_results

def enforce_boundary_conditions(solution):
    """Enforce boundary conditions consistent with PINN formulation"""
    for t_idx in range(len(solution['times'])):
        c1 = solution['c1_preds'][t_idx]
        c2 = solution['c2_preds'][t_idx]
        
        # Top boundary (y=Ly): Cu = 0, Ni = 1.25e-3
        c1[:, -1] = C_CU_TOP
        c2[:, -1] = C_NI_TOP
        
        # Bottom boundary (y=0): Cu = 1.6e-3, Ni = 0
        c1[:, 0] = C_CU_BOTTOM
        c2[:, 0] = C_NI_BOTTOM
        
        # Left boundary (x=0): zero flux (Neumann)
        c1[0, :] = c1[1, :]
        c2[0, :] = c2[1, :]
        
        # Right boundary (x=Lx): zero flux (Neumann)
        c1[-1, :] = c1[-2, :]
        c2[-1, :] = c2[-2, :]
        
        solution['c1_preds'][t_idx] = c1
        solution['c2_preds'][t_idx] = c2
    
    # Enforce initial condition (t=0): c1 = c2 = 0 everywhere
    solution['c1_preds'][0] = np.zeros_like(solution['c1_preds'][0])
    solution['c2_preds'][0] = np.zeros_like(solution['c2_preds'][0])
    
    return solution

def compute_fluxes(solution, time_index):
    """Compute concentration gradients (fluxes) in x and y directions"""
    x_coords = solution['X'][:, 0]
    y_coords = solution['Y'][0, :]
    c1 = solution['c1_preds'][time_index]  # Cu concentration
    c2 = solution['c2_preds'][time_index]  # Ni concentration
    dx = x_coords[1] - x_coords[0]
    dy = y_coords[1] - y_coords[0]
    
    # Compute gradients using np.gradient
    grad_c1_x, grad_c1_y = np.gradient(c1, dx, dy, axis=(1, 0))
    grad_c2_x, grad_c2_y = np.gradient(c2, dx, dy, axis=(1, 0))
    
    return {
        'cu_flux_x': grad_c1_x,  # dC_Cu/dx
        'cu_flux_y': grad_c1_y,  # dC_Cu/dy
        'ni_flux_x': grad_c2_x,  # dC_Ni/dx
        'ni_flux_y': grad_c2_y   # dC_Ni/dy
    }

def plot_boundary_profiles(solution, time_index, output_dir="figures"):
    """Plot concentration profiles along all boundaries for debugging"""
    x_coords = solution['X'][:, 0]
    y_coords = solution['Y'][0, :]
    t_val = solution['times'][time_index]
    Lx = solution['params']['Lx']
    Ly = solution['params']['Ly']
    c1 = solution['c1_preds'][time_index]
    c2 = solution['c2_preds'][time_index]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    
    # Top boundary (y=Ly)
    axes[0,0].plot(x_coords, c1[:, -1], 'b-', label='Cu', linewidth=2)
    axes[0,0].plot(x_coords, c2[:, -1], 'r-', label='Ni', linewidth=2)
    axes[0,0].set_xlabel('x (μm)')
    axes[0,0].set_ylabel('Concentration (mol/cc)')
    axes[0,0].set_title(f'Top Boundary (y=Ly)\nCu = {C_CU_TOP:.1e}, Ni = {C_NI_TOP:.1e}')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    
    # Bottom boundary (y=0)
    axes[0,1].plot(x_coords, c1[:, 0], 'b-', label='Cu', linewidth=2)
    axes[0,1].plot(x_coords, c2[:, 0], 'r-', label='Ni', linewidth=2)
    axes[0,1].set_xlabel('x (μm)')
    axes[0,1].set_ylabel('Concentration (mol/cc)')
    axes[0,1].set_title(f'Bottom Boundary (y=0)\nCu = {C_CU_BOTTOM:.1e}, Ni = {C_NI_BOTTOM:.1e}')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    
    # Left boundary (x=0)
    axes[1,0].plot(y_coords, c1[0, :], 'b-', linewidth=2, label='Cu')
    axes[1,0].plot(y_coords, c2[0, :], 'r-', linewidth=2, label='Ni')
    axes[1,0].set_xlabel('y (μm)')
    axes[1,0].set_ylabel('Concentration (mol/cc)')
    axes[1,0].set_title('Left Boundary (x=0)\nShould show zero flux (flat)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    
    # Right boundary (x=Lx)
    axes[1,1].plot(y_coords, c1[-1, :], 'b-', linewidth=2, label='Cu')
    axes[1,1].plot(y_coords, c2[-1, :], 'r-', linewidth=2, label='Ni')
    axes[1,1].set_xlabel('y (μm)')
    axes[1,1].set_ylabel('Concentration (mol/cc)')
    axes[1,1].set_title('Right Boundary (x=Lx)\nShould show zero flux (flat)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    
    param_text = f"$L_y$ = {Ly:.1f} μm, t = {t_val:.1f} s"
    if solution.get('interpolated', False):
        param_text += " (Interpolated)"
    fig.suptitle(f'Boundary Condition Profiles\n{param_text}', fontsize=14)
    
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"boundary_profiles_t_{t_val:.1f}_ly_{Ly:.1f}"
    plt.savefig(os.path.join(output_dir, f"{base_filename}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f"{base_filename}.pdf"), bbox_inches='tight')
    plt.close()
    return fig, base_filename

@st.cache_data
def load_solutions(solution_dir):
    solutions = []
    params_list = []
    load_logs = []
    lys = []
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
                    
                    # Enforce boundary conditions consistent with PINN
                    sol = enforce_boundary_conditions(sol)
                    
                    # Validate boundary conditions
                    bc_validation = validate_boundary_conditions(sol)
                    bc_status = "✓" if all([bc_validation['top_bc_cu'], bc_validation['top_bc_ni'],
                                           bc_validation['bottom_bc_cu'], bc_validation['bottom_bc_ni'],
                                           bc_validation['left_flux_cu'], bc_validation['left_flux_ni'],
                                           bc_validation['right_flux_cu'], bc_validation['right_flux_ni'],
                                           bc_validation['initial_condition']]) else "✗"
                    
                    c1_min, c1_max = np.min(sol['c1_preds'][0]), np.max(sol['c1_preds'][0])
                    c2_min, c2_max = np.min(sol['c2_preds'][0]), np.max(sol['c2_preds'][0])
                    solutions.append(sol)
                    param_tuple = (sol['params']['Ly'],)
                    params_list.append(param_tuple)
                    lys.append(sol['params']['Ly'])
                    load_logs.append(
                        f"{fname}: {bc_status} Loaded. Cu: {c1_min:.2e} to {c1_max:.2e}, Ni: {c2_min:.2e} to {c2_max:.2e}, "
                        f"Ly={param_tuple[0]:.1f}"
                    )
                    if not all([bc_validation['top_bc_cu'], bc_validation['top_bc_ni'],
                               bc_validation['bottom_bc_cu'], bc_validation['bottom_bc_ni'],
                               bc_validation['left_flux_cu'], bc_validation['left_flux_ni'],
                               bc_validation['right_flux_cu'], bc_validation['right_flux_ni'],
                               bc_validation['initial_condition']]):
                        load_logs.append(f"     BC violations: {', '.join(bc_validation['details'])}")
                else:
                    missing_keys = [key for key in required_keys if key not in sol]
                    load_logs.append(f"{fname}: Skipped - Missing keys: {missing_keys}")
            except Exception as e:
                load_logs.append(f"{fname}: Skipped - Failed to load: {str(e)}")
    if len(solutions) < 1:
        load_logs.append("Error: No valid solutions loaded. Interpolation will fail.")
    else:
        load_logs.append(f"Loaded {len(solutions)} solutions.")
    return solutions, params_list, lys, load_logs

class MultiParamAttentionInterpolator(nn.Module):
    def __init__(self, sigma=0.2, num_heads=4, d_head=8):
        super().__init__()
        self.sigma = sigma
        self.num_heads = num_heads
        self.d_head = d_head
        self.W_q = nn.Linear(1, self.num_heads * self.d_head)  # Query projection (only Ly)
        self.W_k = nn.Linear(1, self.num_heads * self.d_head)  # Key projection

    def forward(self, solutions, params_list, ly_target):
        if not solutions or not params_list:
            raise ValueError("No solutions or parameters available for interpolation.")

        # Extract and normalize Ly parameter
        lys = np.array([p[0] for p in params_list])
        ly_norm = (lys - 30.0) / (120.0 - 30.0)
        target_ly_norm = (ly_target - 30.0) / (120.0 - 30.0)

        # Combine normalized parameters into tensors
        params_tensor = torch.tensor(ly_norm, dtype=torch.float32).reshape(-1, 1)  # [N, 1]
        target_params_tensor = torch.tensor([target_ly_norm], dtype=torch.float32).reshape(1, 1)  # [1, 1]

        # Project to query/key space
        queries = self.W_q(target_params_tensor)  # [1, num_heads * d_head]
        keys = self.W_k(params_tensor)  # [N, num_heads * d_head]

        # Reshape for multi-head attention
        queries = queries.view(1, self.num_heads, self.d_head)  # [1, num_heads, d_head]
        keys = keys.view(len(params_list), self.num_heads, self.d_head)  # [N, num_heads, d_head]

        # Scaled dot-product attention
        attn_logits = torch.einsum('nhd,mhd->nmh', keys, queries) / np.sqrt(self.d_head)  # [N, 1, num_heads]
        attn_weights = torch.softmax(attn_logits, dim=0)  # [N, 1, num_heads]
        attn_weights = attn_weights.mean(dim=2).squeeze(1)  # [N], average across heads

        # Spatial weights (Gaussian-like for locality)
        scaled_distances = torch.sqrt(
            ((torch.tensor(ly_norm) - target_ly_norm) / self.sigma)**2
        )
        spatial_weights = torch.exp(-scaled_distances**2 / 2)
        spatial_weights /= spatial_weights.sum()  # Normalize

        # Combine attention and spatial weights
        combined_weights = attn_weights * spatial_weights
        combined_weights /= combined_weights.sum()  # Normalize

        return self._physics_aware_interpolation(solutions, combined_weights.detach().numpy(), ly_target)

    def _physics_aware_interpolation(self, solutions, weights, ly_target):
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

        # Enforce boundary conditions consistent with PINN
        for t_idx in range(len(times)):
            # Top boundary (y=Ly): Cu = 0, Ni = 1.25e-3
            c1_interp[t_idx, :, -1] = C_CU_TOP
            c2_interp[t_idx, :, -1] = C_NI_TOP
            
            # Bottom boundary (y=0): Cu = 1.6e-3, Ni = 0
            c1_interp[t_idx, :, 0] = C_CU_BOTTOM
            c2_interp[t_idx, :, 0] = C_NI_BOTTOM
            
            # Left boundary (x=0): zero flux
            c1_interp[t_idx, 0, :] = c1_interp[t_idx, 1, :]
            c2_interp[t_idx, 0, :] = c2_interp[t_idx, 1, :]
            
            # Right boundary (x=Lx): zero flux
            c1_interp[t_idx, -1, :] = c1_interp[t_idx, -2, :]
            c2_interp[t_idx, -1, :] = c2_interp[t_idx, -2, :]

        # Enforce initial condition (t=0): c1 = c2 = 0 everywhere
        c1_interp[0] = np.zeros_like(c1_interp[0])
        c2_interp[0] = np.zeros_like(c2_interp[0])

        param_set = solutions[0]['params'].copy()
        param_set['Ly'] = ly_target
        param_set['C_Cu'] = C_CU_BOTTOM  # For display purposes
        param_set['C_Ni'] = C_NI_TOP     # For display purposes

        interpolated_solution = {
            'params': param_set,
            'X': X,
            'Y': Y,
            'c1_preds': list(c1_interp),
            'c2_preds': list(c2_interp),
            'times': times,
            'interpolated': True,
            'attention_weights': weights.tolist()
        }

        # Enforce BCs and IC one more time to be sure
        interpolated_solution = enforce_boundary_conditions(interpolated_solution)
        
        return interpolated_solution

@st.cache_data
def load_and_interpolate_solution(solutions, params_list, ly_target, tolerance_ly=0.1):
    for sol, params in zip(solutions, params_list):
        ly = params[0]
        if abs(ly - ly_target) < tolerance_ly:
            sol['interpolated'] = False
            # Ensure BCs are enforced on exact solutions
            return enforce_boundary_conditions(sol)
    if not solutions:
        raise ValueError("No solutions available for interpolation.")
    interpolator = MultiParamAttentionInterpolator(sigma=0.2)
    return interpolator(solutions, params_list, ly_target)

def plot_2d_concentration(solution, time_index, output_dir="figures", cmap_cu='viridis', cmap_ni='magma', vmin_cu=None, vmax_cu=None, vmin_ni=None, vmax_ni=None):
    x_coords = solution['X'][:, 0]
    y_coords = solution['Y'][0, :]
    t_val = solution['times'][time_index]
    Lx = solution['params']['Lx']
    Ly = solution['params']['Ly']
    c1 = solution['c1_preds'][time_index]
    c2 = solution['c2_preds'][time_index]

    # Apply custom limits or auto-scale
    cu_min = vmin_cu if vmin_cu is not None else 0
    cu_max = vmax_cu if vmax_cu is not None else np.max(c1)
    ni_min = vmin_ni if vmin_ni is not None else 0
    ni_max = vmax_ni if vmax_ni is not None else np.max(c2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    # Cu heatmap
    im1 = ax1.imshow(
        c1,
        origin='lower',
        extent=[0, Lx, 0, Ly],
        cmap=cmap_cu,
        vmin=cu_min,
        vmax=cu_max
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
        vmin=ni_min,
        vmax=ni_max
    )
    ax2.set_xlabel('x (μm)')
    ax2.set_ylabel('y (μm)')
    ax2.set_title(f'Ni Concentration, t = {t_val:.1f} s')
    ax2.grid(True)
    cb2 = fig.colorbar(im2, ax=ax2, label='Ni Conc. (mol/cc)', format='%.1e')
    cb2.ax.tick_params(labelsize=10)

    param_text = f"$L_y$ = {Ly:.1f} μm"
    if solution.get('interpolated', False):
        param_text += " (Interpolated)"
    fig.suptitle(f'Concentration Profiles\n{param_text}', fontsize=14)

    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"conc_2d_t_{t_val:.1f}_ly_{Ly:.1f}"
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
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    ax2.set_xlabel('y (μm)', fontsize=label_size)
    ax2.set_ylabel('Ni Conc. (mol/cc)', fontsize=label_size)
    ax2.set_title(f'Ni at x = {Lx/2:.1f} μm', fontsize=title_size)
    ax2.legend(fontsize=8, loc=legend_params['loc'], bbox_to_anchor=legend_params['bbox'],
               frameon=legend_frameon, framealpha=legend_framealpha)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # Sidebar plot
    ax3.plot(sidebar_data, times, 'k-', linewidth=curve_linewidth)
    ax3.set_xlabel(sidebar_label, fontsize=label_size)
    ax3.set_ylabel('Time (s)', fontsize=label_size)
    ax3.set_title('Metric vs. Time', fontsize=title_size)
    ax3.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    param_text = f"$L_y$ = {Ly:.1f} μm"
    if solution.get('interpolated', False):
        param_text += " (Interpolated)"
    fig.suptitle(f'Centerline Concentration Profiles\n{param_text}', fontsize=title_size)

    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"conc_centerline_ly_{Ly:.1f}"
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
            ly = params[0]
            label = f'$L_y$={ly:.1f}'
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
        ly = params[0]
        if params in selected_params:
            y_coords = sol['Y'][0, :]
            c1 = sol['c1_preds'][time_index][:, center_idx]
            c2 = sol['c2_preds'][time_index][:, center_idx]
            label = f'$L_y$={ly:.1f}'
            if sol.get('interpolated', False):
                label += " (Interpolated)"
            ax1.plot(y_coords, c1, label=label, color=colors[idx], linewidth=curve_linewidth)
            ax2.plot(y_coords, c2, label=label, color=colors[idx], linewidth=curve_linewidth)

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

    # Sidebar bar plot
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

def main():
    st.title("Publication-Quality Concentration Profiles with Boundary Condition Validation")

    # Load solutions
    solutions, params_list, lys, load_logs = load_solutions(SOLUTION_DIR)

    # Display load logs
    if load_logs:
        with st.expander("Load Log"):
            for log in load_logs:
                st.write(log)

    # Check if solutions were loaded
    if not solutions:
        st.error("No valid solution files found in pinn_solutions directory. Please check the directory and file contents.")
        return

    st.write(f"Loaded {len(solutions)} solutions. Unique Ly: {len(set(lys))}")

    # Compute boundary modes
    boundary_modes = compute_boundary_modes(solutions)
    
    # Display boundary mode information
    st.subheader("Most Frequent Boundary Values Across All Solutions")
    if boundary_modes:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Top Boundary (Cu)", 
                f"{boundary_modes['top_cu']:.2e} mol/cc",
                f"±{boundary_modes['stats']['top_cu_std']:.1e}"
            )
            st.metric(
                "Top Boundary (Ni)", 
                f"{boundary_modes['top_ni']:.2e} mol/cc",
                f"±{boundary_modes['stats']['top_ni_std']:.1e}"
            )
        
        with col2:
            st.metric(
                "Bottom Boundary (Cu)", 
                f"{boundary_modes['bottom_cu']:.2e} mol/cc",
                f"±{boundary_modes['stats']['bottom_cu_std']:.1e}"
            )
            st.metric(
                "Bottom Boundary (Ni)", 
                f"{boundary_modes['bottom_ni']:.2e} mol/cc",
                f"±{boundary_modes['stats']['bottom_ni_std']:.1e}"
            )
        
        with col3:
            st.metric(
                "Left Boundary (Cu)", 
                f"{boundary_modes['left_cu']:.2e} mol/cc",
                f"±{boundary_modes['stats']['left_cu_std']:.1e}"
            )
            st.metric(
                "Left Boundary (Ni)", 
                f"{boundary_modes['left_ni']:.2e} mol/cc",
                f"±{boundary_modes['stats']['left_ni_std']:.1e}"
            )
        
        with col4:
            st.metric(
                "Right Boundary (Cu)", 
                f"{boundary_modes['right_cu']:.2e} mol/cc",
                f"±{boundary_modes['stats']['right_cu_std']:.1e}"
            )
            st.metric(
                "Right Boundary (Ni)", 
                f"{boundary_modes['right_ni']:.2e} mol/cc",
                f"±{boundary_modes['stats']['right_ni_std']:.1e}"
            )
        
        # Verbal summary
        st.info(
            f"**Boundary Value Summary:** The most common values across all loaded solutions are: "
            f"**{boundary_modes['top_cu']:.2e} mol/cc** for Cu and **{boundary_modes['top_ni']:.2e} mol/cc** for Ni at the top boundary, "
            f"**{boundary_modes['bottom_cu']:.2e} mol/cc** for Cu and **{boundary_modes['bottom_ni']:.2e} mol/cc** for Ni at the bottom boundary, "
            f"**{boundary_modes['left_cu']:.2e} mol/cc** for Cu and **{boundary_modes['left_ni']:.2e} mol/cc** for Ni at the left boundary, "
            f"**{boundary_modes['right_cu']:.2e} mol/cc** for Cu and **{boundary_modes['right_ni']:.2e} mol/cc** for Ni at the right boundary."
        )

    # Sort unique parameters
    lys = sorted(set(lys))

    # Parameter selection for single solution
    st.subheader("Select Parameters for Single Solution")
    ly_choice = st.selectbox("Domain Height (Ly, μm)", options=lys, format_func=lambda x: f"{x:.1f}")

    # Custom parameters for interpolation
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
    else:
        ly_target = ly_choice

    # Boundary condition validation section
    st.subheader("Boundary Condition Validation")
    if st.checkbox("Show Boundary Condition Validation", value=True):
        try:
            # Load or interpolate solution
            solution = load_and_interpolate_solution(solutions, params_list, ly_target)
            
            # Validate boundary conditions
            bc_validation = validate_boundary_conditions(solution)
            
            # Display validation results
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Boundary Condition Status:**")
                status_color = "green" if bc_validation['top_bc_cu'] else "red"
                st.markdown(f"- Top BC (Cu = {C_CU_TOP:.1e}): <span style='color:{status_color}'>{'✓ Satisfied' if bc_validation['top_bc_cu'] else '✗ Violated'}</span>", unsafe_allow_html=True)
                status_color = "green" if bc_validation['top_bc_ni'] else "red"
                st.markdown(f"- Top BC (Ni = {C_NI_TOP:.1e}): <span style='color:{status_color}'>{'✓ Satisfied' if bc_validation['top_bc_ni'] else '✗ Violated'}</span>", unsafe_allow_html=True)
                status_color = "green" if bc_validation['bottom_bc_cu'] else "red"
                st.markdown(f"- Bottom BC (Cu = {C_CU_BOTTOM:.1e}): <span style='color:{status_color}'>{'✓ Satisfied' if bc_validation['bottom_bc_cu'] else '✗ Violated'}</span>", unsafe_allow_html=True)
                status_color = "green" if bc_validation['bottom_bc_ni'] else "red"
                st.markdown(f"- Bottom BC (Ni = {C_NI_BOTTOM:.1e}): <span style='color:{status_color}'>{'✓ Satisfied' if bc_validation['bottom_bc_ni'] else '✗ Violated'}</span>", unsafe_allow_html=True)
            
            with col2:
                status_color = "green" if bc_validation['left_flux_cu'] and bc_validation['left_flux_ni'] else "red"
                st.markdown(f"- Left Flux: <span style='color:{status_color}'>{'✓ Satisfied' if bc_validation['left_flux_cu'] and bc_validation['left_flux_ni'] else '✗ Violated'}</span>", unsafe_allow_html=True)
                status_color = "green" if bc_validation['right_flux_cu'] and bc_validation['right_flux_ni'] else "red"
                st.markdown(f"- Right Flux: <span style='color:{status_color}'>{'✓ Satisfied' if bc_validation['right_flux_cu'] and bc_validation['right_flux_ni'] else '✗ Violated'}</span>", unsafe_allow_html=True)
                status_color = "green" if bc_validation['initial_condition'] else "red"
                st.markdown(f"- Initial Condition (c1=c2=0): <span style='color:{status_color}'>{'✓ Satisfied' if bc_validation['initial_condition'] else '✗ Violated'}</span>", unsafe_allow_html=True)
            
            if bc_validation['details']:
                st.warning("Boundary condition issues detected:")
                for detail in bc_validation['details']:
                    st.write(f"  - {detail}")
            
            # Plot boundary profiles
            bc_time_index = st.slider("Select Time Index for Boundary Check", 0, len(solution['times'])-1, len(solution['times'])-1, key="bc_time")
            if st.button("Generate Boundary Profile Plots"):
                fig_bc, filename_bc = plot_boundary_profiles(solution, bc_time_index)
                st.pyplot(fig_bc)
                st.download_button(
                    label="Download Boundary Profiles as PNG",
                    data=open(os.path.join("figures", f"{filename_bc}.png"), "rb").read(),
                    file_name=f"{filename_bc}.png",
                    mime="image/png"
                )
                st.download_button(
                    label="Download Boundary Profiles as PDF",
                    data=open(os.path.join("figures", f"{filename_bc}.pdf"), "rb").read(),
                    file_name=f"{filename_bc}.pdf",
                    mime="application/pdf"
                )
                
        except Exception as e:
            st.error(f"Failed during boundary validation: {str(e)}")

    # Visualization settings
    st.subheader("Visualization Settings")
    cmap_cu = st.selectbox("Cu Heatmap Colormap", options=COLORMAPS, index=COLORMAPS.index('viridis'))
    cmap_ni = st.selectbox("Ni Heatmap Colormap", options=COLORMAPS, index=COLORMAPS.index('magma'))
    sidebar_metric = st.selectbox("Sidebar Metric for Curves", options=['mean_cu', 'mean_ni', 'loss'], index=0)

    # Color scale limits
    st.subheader("Color Scale Limits")
    use_custom_scale = st.checkbox("Use custom color scale limits", value=False)
    custom_cu_min, custom_cu_max, custom_ni_min, custom_ni_max = None, None, None, None
    if use_custom_scale:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Cu Concentration Limits**")
            custom_cu_min = st.number_input("Cu Min", value=0.0, format="%.2e", key="cu_min")
            custom_cu_max = st.number_input("Cu Max", value=float(np.max([sol['c1_preds'] for sol in solutions])), format="%.2e", key="cu_max")
        with col2:
            st.write("**Ni Concentration Limits**")
            custom_ni_min = st.number_input("Ni Min", value=0.0, format="%.2e", key="ni_min")
            custom_ni_max = st.number_input("Ni Max", value=float(np.max([sol['c2_preds'] for sol in solutions])), format="%.2e", key="ni_max")

    # Validate color scale limits
    if custom_cu_min is not None and custom_cu_max is not None and custom_cu_min >= custom_cu_max:
        st.error("Cu minimum concentration must be less than maximum concentration.")
        return
    if custom_ni_min is not None and custom_ni_max is not None and custom_ni_min >= custom_ni_max:
        st.error("Ni minimum concentration must be less than maximum concentration.")
        return

    # Figure customization controls
    with st.expander("Figure Customization"):
        label_size = st.slider("Axis Label Size", min_value=8, max_value=20, value=12, step=1)
        title_size = st.slider("Title Size", min_value=10, max_value=24, value=14, step=1)
        tick_label_size = st.slider("Tick Label Size", min_value=6, max_value=16, value=10, step=1)
        legend_loc = st.selectbox(
            "Legend Location",
            options=['upper right', 'upper left', 'lower right', 'lower left', 'center', 'best',
                     'right', 'left', 'above', 'below'],
            index=0
        )
        curve_colormap = st.selectbox(
            "Curve Colormap",
            options=['viridis', 'plasma', 'inferno', 'magma', 'tab10', 'Set1', 'Set2'],
            index=4
        )
        axis_linewidth = st.slider("Axis Line Width", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
        tick_major_width = st.slider("Tick Major Width", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
        tick_major_length = st.slider("Tick Major Length", min_value=2.0, max_value=10.0, value=4.0, step=0.5)
        fig_width = st.slider("Figure Width (inches)", min_value=4.0, max_value=12.0, value=8.0, step=0.5)
        fig_height = st.slider("Figure Height (inches)", min_value=3.0, max_value=8.0, value=6.0, step=0.5)
        curve_linewidth = st.slider("Curve Line Width", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
        grid_alpha = st.slider("Grid Opacity", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
        grid_linestyle = st.selectbox("Grid Line Style", options=['--', '-', ':', '-.'], index=0)
        legend_frameon = st.checkbox("Show Legend Frame", value=True)
        legend_framealpha = st.slider("Legend Frame Opacity", min_value=0.0, max_value=1.0, value=0.8, step=0.1)

    # Load or interpolate single solution
    try:
        solution = load_and_interpolate_solution(solutions, params_list, ly_target)
    except Exception as e:
        st.error(f"Failed to load or interpolate solution: {str(e)}")
        return

    # Display solution details
    st.subheader("Solution Details")
    st.write(f"$L_y$ = {solution['params']['Ly']:.1f} μm")
    st.write(f"$C_{{Cu}}$ (bottom) = {C_CU_BOTTOM:.1e} mol/cc, $C_{{Cu}}$ (top) = {C_CU_TOP:.1e} mol/cc")
    st.write(f"$C_{{Ni}}$ (bottom) = {C_NI_BOTTOM:.1e} mol/cc, $C_{{Ni}}$ (top) = {C_NI_TOP:.1e} mol/cc")
    if solution.get('interpolated', False):
        st.write("**Status**: Interpolated solution")
    else:
        st.write("**Status**: Exact solution")

    # 2D Concentration Heatmaps
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

    # Centerline Concentration Curves
    st.subheader("Centerline Concentration Curves")
    time_indices = st.multiselect(
        "Select Time Indices for Curves",
        options=list(range(len(solution['times']))),
        default=[0, len(solution['times'])//4, len(solution['times'])//2, 3*len(solution['times'])//4, len(solution['times'])-1],
        format_func=lambda x: f"t = {solution['times'][x]:.1f} s"
    )
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

    # Parameter Sweep Curves
    st.subheader("Parameter Sweep Curves")
    with st.expander("Add Custom Parameter Combinations for Sweep"):
        num_custom_params = st.number_input("Number of Custom Parameter Sets", min_value=0, max_value=5, value=0, step=1)
        custom_params = []
        for i in range(num_custom_params):
            st.write(f"Custom Parameter Set {i+1}")
            ly_custom = st.number_input(
                f"Custom Ly (μm) {i+1}",
                min_value=30.0,
                max_value=120.0,
                value=ly_choice,
                step=0.1,
                format="%.1f",
                key=f"ly_custom_{i}"
            )
            custom_params.append((ly_custom,))

    # Combine exact and custom parameters
    param_options = [(ly,) for ly in lys]
    param_labels = [f"$L_y$={ly:.1f}" for ly in lys]
    default_params = param_options[:min(4, len(param_options))]
    selected_labels = st.multiselect(
        "Select Exact Parameter Combinations",
        options=param_labels,
        default=[param_labels[param_options.index(p)] for p in default_params],
        format_func=lambda x: x
    )
    selected_params = [param_options[param_labels.index(label)] for label in selected_labels]
    selected_params.extend(custom_params)

    # Generate solutions for selected parameters (exact or interpolated)
    sweep_solutions = []
    sweep_params_list = []
    for params in selected_params:
        ly = params[0]
        try:
            sol = load_and_interpolate_solution(solutions, params_list, ly)
            sweep_solutions.append(sol)
            sweep_params_list.append(params)
        except Exception as e:
            st.warning(f"Failed to load or interpolate solution for Ly={ly:.1f}: {str(e)}")

    # Plot parameter sweep
    sweep_time_index = st.slider("Select Time Index for Sweep", 0, len(solution['times'])-1, len(solution['times'])-1)
    if sweep_solutions and sweep_params_list:
        fig_sweep, filename_sweep = plot_parameter_sweep(
            sweep_solutions, sweep_params_list, sweep_params_list, sweep_time_index, sidebar_metric=sidebar_metric,
            label_size=label_size, title_size=title_size, tick_label_size=tick_label_size,
            legend_loc=legend_loc, curve_colormap=curve_colormap,
            axis_linewidth=axis_linewidth, tick_major_width=tick_major_width,
            tick_major_length=tick_major_length, fig_width=fig_width, fig_height=fig_height,
            curve_linewidth=curve_linewidth, grid_alpha=grid_alpha, grid_linestyle=grid_linestyle,
            legend_frameon=legend_frameon, legend_framealpha=legend_framealpha
        )
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

    # New section for advanced visualizations
    st.subheader("Advanced Visualizations")
    adv_vis_type = st.selectbox("Select Visualization Type", options=['Sunburst Chart', 'Radar Chart', 'Polar Chart'], index=0)
    
    # Select time index for advanced visualization
    adv_time_index = st.slider("Select Time Index for Advanced Visualization", 0, len(solution['times'])-1, len(solution['times'])-1)
    
    if adv_vis_type == 'Sunburst Chart':
        fig_sunburst, filename_sunburst = plot_sunburst_chart(solution, adv_time_index)
        st.plotly_chart(fig_sunburst, use_container_width=True)
        st.download_button(
            label="Download Sunburst Chart as HTML",
            data=open(os.path.join("figures", f"{filename_sunburst}.html"), "rb").read(),
            file_name=f"{filename_sunburst}.html",
            mime="text/html"
        )
        st.download_button(
            label="Download Sunburst Chart as PNG",
            data=open(os.path.join("figures", f"{filename_sunburst}.png"), "rb").read(),
            file_name=f"{filename_sunburst}.png",
            mime="image/png"
        )
    
    elif adv_vis_type == 'Radar Chart':
        adv_time_indices = st.multiselect(
            "Select Time Indices for Radar Chart",
            options=list(range(len(solution['times']))),
            default=[0, len(solution['times'])//4, len(solution['times'])//2, 3*len(solution['times'])//4, len(solution['times'])-1],
            format_func=lambda x: f"t = {solution['times'][x]:.1f} s"
        )
        if adv_time_indices:
            fig_radar, filename_radar = plot_radar_chart(solution, adv_time_indices)
            st.plotly_chart(fig_radar, use_container_width=True)
            st.download_button(
                label="Download Radar Chart as HTML",
                data=open(os.path.join("figures", f"{filename_radar}.html"), "rb").read(),
                file_name=f"{filename_radar}.html",
                mime="text/html"
            )
            st.download_button(
                label="Download Radar Chart as PNG",
                data=open(os.path.join("figures", f"{filename_radar}.png"), "rb").read(),
                file_name=f"{filename_radar}.png",
                mime="image/png"
            )
    
    elif adv_vis_type == 'Polar Chart':
        fig_polar, filename_polar = plot_polar_chart(solution, adv_time_index)
        st.plotly_chart(fig_polar, use_container_width=True)
        st.download_button(
            label="Download Polar Chart as HTML",
            data=open(os.path.join("figures", f"{filename_polar}.html"), "rb").read(),
            file_name=f"{filename_polar}.html",
            mime="text/html"
        )
        st.download_button(
            label="Download Polar Chart as PNG",
            data=open(os.path.join("figures", f"{filename_polar}.png"), "rb").read(),
            file_name=f"{filename_polar}.png",
            mime="image/png"
        )

if __name__ == "__main__":
    main()
