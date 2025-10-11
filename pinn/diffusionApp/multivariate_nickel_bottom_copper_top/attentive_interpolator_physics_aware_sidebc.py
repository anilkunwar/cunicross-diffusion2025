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

def validate_boundary_conditions(solution, tolerance=1e-6):
    """Validate that boundary conditions are properly satisfied"""
    validation_results = {
        'top_bc_cu': True,
        'bottom_bc_ni': True, 
        'left_flux_cu': True,
        'left_flux_ni': True,
        'right_flux_cu': True,
        'right_flux_ni': True,
        'details': []
    }
    
    Lx = solution['params']['Lx']
    Ly = solution['params']['Ly']
    c_cu_target = solution['params']['C_Cu']
    c_ni_target = solution['params']['C_Ni']
    
    # Check last time step for steady state
    t_idx = -1
    c1 = solution['c1_preds'][t_idx]
    c2 = solution['c2_preds'][t_idx]
    
    # Check top boundary (y=0) - Cu should be constant
    top_cu_values = c1[:, 0]
    top_cu_std = np.std(top_cu_values)
    if top_cu_std > tolerance:
        validation_results['top_bc_cu'] = False
        validation_results['details'].append(f"Top BC Cu not constant: std={top_cu_std:.2e}")
    
    # Check bottom boundary (y=Ly) - Ni should be constant  
    bottom_ni_values = c2[:, -1]
    bottom_ni_std = np.std(bottom_ni_values)
    if bottom_ni_std > tolerance:
        validation_results['bottom_bc_ni'] = False
        validation_results['details'].append(f"Bottom BC Ni not constant: std={bottom_ni_std:.2e}")
    
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
    
    return validation_results

def enforce_boundary_conditions(solution):
    """Enforce all boundary conditions on a solution"""
    c_cu_target = solution['params']['C_Cu']
    c_ni_target = solution['params']['C_Ni']
    
    for t_idx in range(len(solution['times'])):
        c1 = solution['c1_preds'][t_idx]
        c2 = solution['c2_preds'][t_idx]
        
        # Top boundary: Cu = C_Cu
        c1[:, 0] = c_cu_target
        
        # Bottom boundary: Ni = C_Ni  
        c2[:, -1] = c_ni_target
        
        # Left boundary: zero flux (Neumann)
        c1[0, :] = c1[1, :]
        c2[0, :] = c2[1, :]
        
        # Right boundary: zero flux (Neumann)
        c1[-1, :] = c1[-2, :]
        c2[-1, :] = c2[-2, :]
        
        solution['c1_preds'][t_idx] = c1
        solution['c2_preds'][t_idx] = c2
    
    return solution

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
    
    # Top boundary (y=0)
    axes[0,0].plot(x_coords, c1[:, 0], 'b-', label='Cu', linewidth=2)
    axes[0,0].plot(x_coords, c2[:, 0], 'r-', label='Ni', linewidth=2)
    axes[0,0].set_xlabel('x (μm)')
    axes[0,0].set_ylabel('Concentration (mol/cc)')
    axes[0,0].set_title(f'Top Boundary (y=0)\nCu should be constant = {solution["params"]["C_Cu"]:.1e}')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    
    # Bottom boundary (y=Ly)
    axes[0,1].plot(x_coords, c1[:, -1], 'b-', linewidth=2)
    axes[0,1].plot(x_coords, c2[:, -1], 'r-', linewidth=2)
    axes[0,1].set_xlabel('x (μm)')
    axes[0,1].set_ylabel('Concentration (mol/cc)')
    axes[0,1].set_title(f'Bottom Boundary (y=Ly)\nNi should be constant = {solution["params"]["C_Ni"]:.1e}')
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
    
    param_text = f"$L_y$ = {Ly:.1f} μm, $C_{{Cu}}$ = {solution['params']['C_Cu']:.1e}, $C_{{Ni}}$ = {solution['params']['C_Ni']:.1e}, t = {t_val:.1f} s"
    if solution.get('interpolated', False):
        param_text += " (Interpolated)"
    fig.suptitle(f'Boundary Condition Profiles\n{param_text}', fontsize=14)
    
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"boundary_profiles_t_{t_val:.1f}_ly_{Ly:.1f}_ccu_{solution['params']['C_Cu']:.1e}_cni_{solution['params']['C_Ni']:.1e}"
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
                    
                    # Enforce boundary conditions on loaded solutions
                    sol = enforce_boundary_conditions(sol)
                    
                    # Validate boundary conditions
                    bc_validation = validate_boundary_conditions(sol)
                    bc_status = "✓" if all(bc_validation.values()) else "✗"
                    
                    c1_min, c1_max = np.min(sol['c1_preds'][0]), np.max(sol['c1_preds'][0])
                    c2_min, c2_max = np.min(sol['c2_preds'][0]), np.max(sol['c2_preds'][0])
                    solutions.append(sol)
                    param_tuple = (sol['params']['Ly'], sol['params']['C_Cu'], sol['params']['C_Ni'])
                    params_list.append(param_tuple)
                    lys.append(sol['params']['Ly'])
                    c_cus.append(sol['params']['C_Cu'])
                    c_nis.append(sol['params']['C_Ni'])
                    load_logs.append(
                        f"{fname}: {bc_status} Loaded. Cu: {c1_min:.2e} to {c1_max:.2e}, Ni: {c2_min:.2e} to {c2_max:.2e}, "
                        f"Ly={param_tuple[0]:.1f}, C_Cu={param_tuple[1]:.1e}, C_Ni={param_tuple[2]:.1e}"
                    )
                    if not all(bc_validation.values()):
                        load_logs.append(f"     BC violations: {', '.join(bc_validation['details'])}")
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

class MultiParamAttentionInterpolator(nn.Module):
    def __init__(self, sigma=0.2, num_heads=4, d_head=8):
        super().__init__()
        self.sigma = sigma
        self.num_heads = num_heads
        self.d_head = d_head
        self.W_q = nn.Linear(3, self.num_heads * self.d_head)  # Query projection
        self.W_k = nn.Linear(3, self.num_heads * self.d_head)  # Key projection

    def forward(self, solutions, params_list, ly_target, c_cu_target, c_ni_target):
        if not solutions or not params_list:
            raise ValueError("No solutions or parameters available for interpolation.")

        # Extract and normalize parameters
        lys = np.array([p[0] for p in params_list])
        c_cus = np.array([p[1] for p in params_list])
        c_nis = np.array([p[2] for p in params_list])
        if not (lys.shape == c_cus.shape == c_nis.shape):
            raise ValueError(f"Parameter array shapes mismatch: lys={lys.shape}, c_cus={c_cus.shape}, c_nis={c_nis.shape}")

        ly_norm = (lys - 30.0) / (120.0 - 30.0)
        c_cu_norm = (c_cus - 0.0) / (2.9e-3 - 0.0)  # Updated to allow C_Cu = 0
        c_ni_norm = (c_nis - 0.0) / (1.8e-3 - 0.0)  # Updated to allow C_Ni = 0
        target_ly_norm = (ly_target - 30.0) / (120.0 - 30.0)
        target_c_cu_norm = (c_cu_target - 0.0) / (2.9e-3 - 0.0)
        target_c_ni_norm = (c_ni_target - 0.0) / (1.8e-3 - 0.0)

        # Combine normalized parameters into tensors
        params_tensor = torch.tensor(np.stack([ly_norm, c_cu_norm, c_ni_norm], axis=1), dtype=torch.float32)  # [N, 3]
        target_params_tensor = torch.tensor([[target_ly_norm, target_c_cu_norm, target_c_ni_norm]], dtype=torch.float32)  # [1, 3]

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
            ((torch.tensor(ly_norm) - target_ly_norm) / self.sigma)**2 +
            ((torch.tensor(c_cu_norm) - target_c_cu_norm) / self.sigma)**2 +
            ((torch.tensor(c_ni_norm) - target_c_ni_norm) / self.sigma)**2
        )
        spatial_weights = torch.exp(-scaled_distances**2 / 2)
        spatial_weights /= spatial_weights.sum()  # Normalize

        # Combine attention and spatial weights
        combined_weights = attn_weights * spatial_weights
        combined_weights /= combined_weights.sum()  # Normalize

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
                    method='cubic', bounds_error=False, fill_value=None
                )
                interp_c2 = RegularGridInterpolator(
                    (sol['X'][:, 0], Y_scaled), sol['c2_preds'][t_idx],
                    method='cubic', bounds_error=False, fill_value=None
                )
                points = np.stack([X.flatten(), Y.flatten()], axis=1)
                c1_interp[t_idx] += weight * interp_c1(points).reshape(50, 50)
                c2_interp[t_idx] += weight * interp_c2(points).reshape(50, 50)

        # Enforce boundary conditions
        for t_idx in range(len(times)):
            # Top boundary: Cu = C_Cu
            c1_interp[t_idx, :, 0] = c_cu_target
            
            # Bottom boundary: Ni = C_Ni
            c2_interp[t_idx, :, -1] = c_ni_target
            
            # Left boundary: zero flux
            c1_interp[t_idx, 0, :] = c1_interp[t_idx, 1, :]
            c2_interp[t_idx, 0, :] = c2_interp[t_idx, 1, :]
            
            # Right boundary: zero flux
            c1_interp[t_idx, -1, :] = c1_interp[t_idx, -2, :]
            c2_interp[t_idx, -1, :] = c2_interp[t_idx, -2, :]

        param_set = solutions[0]['params'].copy()
        param_set['Ly'] = ly_target
        param_set['C_Cu'] = c_cu_target
        param_set['C_Ni'] = c_ni_target

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

        # Enforce BCs one more time to be sure
        interpolated_solution = enforce_boundary_conditions(interpolated_solution)
        
        return interpolated_solution

@st.cache_data
def load_and_interpolate_solution(solutions, params_list, ly_target, c_cu_target, c_ni_target, tolerance_ly=0.1, tolerance_c=1e-5):
    for sol, params in zip(solutions, params_list):
        ly, c_cu, c_ni = params
        if (abs(ly - ly_target) < tolerance_ly and
                abs(c_cu - c_cu_target) < tolerance_c and
                abs(c_ni - c_ni_target) < tolerance_c):
            sol['interpolated'] = False
            # Ensure BCs are enforced on exact solutions too
            return enforce_boundary_conditions(sol)
    if not solutions:
        raise ValueError("No solutions available for interpolation.")
    interpolator = MultiParamAttentionInterpolator(sigma=0.2)
    return interpolator(solutions, params_list, ly_target, c_cu_target, c_ni_target)

# [Rest of your existing functions: plot_2d_concentration, plot_centerline_curves, plot_parameter_sweep, main remain the same]

def main():
    st.title("Publication-Quality Concentration Profiles with Boundary Condition Validation")

    # Load solutions
    solutions, params_list, lys, c_cus, c_nis, load_logs = load_solutions(SOLUTION_DIR)

    # Display load logs
    if load_logs:
        with st.expander("Load Log"):
            for log in load_logs:
                st.write(log)

    # Check if solutions were loaded
    if not solutions:
        st.error("No valid solution files found in pinn_solutions directory. Please check the directory and file contents.")
        return

    st.write(f"Loaded {len(solutions)} solutions. Unique Ly: {len(set(lys))}, C_Cu: {len(set(c_cus))}, C_Ni: {len(set(c_nis))}")

    # Sort unique parameters
    lys = sorted(set(lys))
    c_cus = sorted(set(c_cus))
    c_nis = sorted(set(c_nis))

    # Parameter selection for single solution
    st.subheader("Select Parameters for Single Solution")
    ly_choice = st.selectbox("Domain Height (Ly, μm)", options=lys, format_func=lambda x: f"{x:.1f}")
    c_cu_choice = st.selectbox("Cu Boundary Concentration (mol/cc)", options=c_cus, format_func=lambda x: f"{x:.1e}")
    c_ni_choice = st.selectbox("Ni Boundary Concentration (mol/cc)", options=c_nis, format_func=lambda x: f"{x:.1e}")

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
        c_cu_target = st.number_input(
            "Custom C_Cu (mol/cc)",
            min_value=0.0,  # Allow self-diffusion
            max_value=2.9e-3,
            value=max(c_cu_choice, 1.5e-3),
            step=0.1e-3,
            format="%.1e"
        )
        c_ni_target = st.number_input(
            "Custom C_Ni (mol/cc)",
            min_value=0.0,  # Allow self-diffusion
            max_value=1.8e-3,
            value=max(c_ni_choice, 1.0e-4),
            step=0.1e-4,
            format="%.1e"
        )
    else:
        ly_target, c_cu_target, c_ni_target = ly_choice, c_cu_choice, c_ni_choice

    # Boundary condition validation section
    st.subheader("Boundary Condition Validation")
    if st.checkbox("Show Boundary Condition Validation", value=True):
        try:
            # Load or interpolate solution
            solution = load_and_interpolate_solution(solutions, params_list, ly_target, c_cu_target, c_ni_target)
            
            # Validate boundary conditions
            bc_validation = validate_boundary_conditions(solution)
            
            # Display validation results
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Boundary Condition Status:**")
                status_color = "green" if bc_validation['top_bc_cu'] else "red"
                st.markdown(f"- Top BC (Cu): <span style='color:{status_color}'>{'✓ Satisfied' if bc_validation['top_bc_cu'] else '✗ Violated'}</span>", unsafe_allow_html=True)
                
                status_color = "green" if bc_validation['bottom_bc_ni'] else "red"
                st.markdown(f"- Bottom BC (Ni): <span style='color:{status_color}'>{'✓ Satisfied' if bc_validation['bottom_bc_ni'] else '✗ Violated'}</span>", unsafe_allow_html=True)
                
            with col2:
                status_color = "green" if bc_validation['left_flux_cu'] and bc_validation['left_flux_ni'] else "red"
                st.markdown(f"- Left Flux: <span style='color:{status_color}'>{'✓ Satisfied' if bc_validation['left_flux_cu'] and bc_validation['left_flux_ni'] else '✗ Violated'}</span>", unsafe_allow_html=True)
                
                status_color = "green" if bc_validation['right_flux_cu'] and bc_validation['right_flux_ni'] else "red"
                st.markdown(f"- Right Flux: <span style='color:{status_color}'>{'✓ Satisfied' if bc_validation['right_flux_cu'] and bc_validation['right_flux_ni'] else '✗ Violated'}</span>", unsafe_allow_html=True)
            
            if bc_validation['details']:
                st.warning("Boundary condition issues detected:")
                for detail in bc_validation['details']:
                    st.write(f"  - {detail}")
            
            # Plot boundary profiles
            bc_time_index = st.slider("Select Time Index for Boundary Check", 0, len(solution['times'])-1, len(solution['times'])-1)
            if st.button("Generate Boundary Profile Plots"):
                fig_bc, filename_bc = plot_boundary_profiles(solution, bc_time_index)
                st.pyplot(fig_bc)
                st.download_button(
                    label="Download Boundary Profiles as PNG",
                    data=open(os.path.join("figures", f"{filename_bc}.png"), "rb").read(),
                    file_name=f"{filename_bc}.png",
                    mime="image/png"
                )
                
        except Exception as e:
            st.error(f"Failed during boundary validation: {str(e)}")

    # [Rest of your existing main function continues...]
    # Visualization settings, color scales, figure customization, etc.

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
        solution = load_and_interpolate_solution(solutions, params_list, ly_target, c_cu_target, c_ni_target)
    except Exception as e:
        st.error(f"Failed to load or interpolate solution: {str(e)}")
        return

    # Display solution details
    st.subheader("Solution Details")
    st.write(f"$L_y$ = {solution['params']['Ly']:.1f} μm")
    st.write(f"$C_{{Cu}}$ = {solution['params']['C_Cu']:.1e} mol/cc")
    st.write(f"$C_{{Ni}}$ = {solution['params']['C_Ni']:.1e} mol/cc")
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

    # [Rest of your existing main function continues...]

if __name__ == "__main__":
    main()
