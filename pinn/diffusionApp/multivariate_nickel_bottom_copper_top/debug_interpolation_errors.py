import os
import pickle
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
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

# Fixed boundary conditions from PINN formulation
C_CU_BOTTOM = 1.6e-3
C_CU_TOP = 0.0
C_NI_BOTTOM = 0.0
C_NI_TOP = 1.25e-3

@st.cache_data
def load_solutions(solution_dir):
    """Load all solutions, checking for validity"""
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
                    solutions.append(sol)
                    param_tuple = (sol['params']['Ly'],)
                    params_list.append(param_tuple)
                    lys.append(sol['params']['Ly'])
                    load_logs.append(
                        f"{fname}: Loaded. Ly={sol['params']['Ly']:.1f}"
                    )
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

def find_solution_by_ly(solutions, params_list, target_ly, tolerance=1.0):
    """Find solution closest to target Ly value"""
    for sol, params in zip(solutions, params_list):
        ly = params[0]
        if abs(ly - target_ly) < tolerance:
            return sol, ly
    return None, None

def find_alternative_solution(solutions, params_list, exclude_ly, target_ly):
    """Find a solution for interpolation, excluding the target Ly"""
    min_diff = float('inf')
    best_sol = None
    best_ly = None
    for sol, params in zip(solutions, params_list):
        ly = params[0]
        if abs(ly - exclude_ly) > 1e-6:  # Exclude the target Ly
            diff = abs(ly - target_ly)
            if diff < min_diff:
                min_diff = diff
                best_sol = sol
                best_ly = ly
    return best_sol, best_ly

def enforce_boundary_conditions(sol):
    """Enforce PINN boundary conditions on a solution"""
    for t_idx in range(len(sol['times'])):
        c1 = sol['c1_preds'][t_idx]
        c2 = sol['c2_preds'][t_idx]
        # Dirichlet: Top (y=0) and bottom (y=Ly)
        c1[:, 0] = C_CU_TOP
        c1[:, -1] = C_CU_BOTTOM
        c2[:, 0] = C_NI_TOP
        c2[:, -1] = C_NI_BOTTOM
        # Neumann: Left (x=0) and right (x=Lx)
        c1[0, :] = c1[1, :]
        c1[-1, :] = c1[-2, :]
        c2[0, :] = c2[1, :]
        c2[-1, :] = c2[-2, :]
        sol['c1_preds'][t_idx] = c1
        sol['c2_preds'][t_idx] = c2
    return sol

def validate_boundary_conditions(sol, tolerance=1e-6):
    """Validate boundary conditions against PINN specifications"""
    results = {
        'top_bc_cu': True,
        'top_bc_ni': True,
        'bottom_bc_cu': True,
        'bottom_bc_ni': True,
        'left_flux_cu': True,
        'left_flux_ni': True,
        'right_flux_cu': True,
        'right_flux_ni': True,
        'details': []
    }
    t_idx = -1  # Check last time step
    c1 = sol['c1_preds'][t_idx]
    c2 = sol['c2_preds'][t_idx]
    
    # Top boundary (y=0): Cu=0, Ni=1.25e-3
    top_cu_mean = np.mean(c1[:, 0])
    top_ni_mean = np.mean(c2[:, 0])
    if abs(top_cu_mean - C_CU_TOP) > tolerance:
        results['top_bc_cu'] = False
        results['details'].append(f"Top Cu: {top_cu_mean:.2e} != {C_CU_TOP:.2e}")
    if abs(top_ni_mean - C_NI_TOP) > tolerance:
        results['top_bc_ni'] = False
        results['details'].append(f"Top Ni: {top_ni_mean:.2e} != {C_NI_TOP:.2e}")
    
    # Bottom boundary (y=Ly): Cu=1.6e-3, Ni=0
    bottom_cu_mean = np.mean(c1[:, -1])
    bottom_ni_mean = np.mean(c2[:, -1])
    if abs(bottom_cu_mean - C_CU_BOTTOM) > tolerance:
        results['bottom_bc_cu'] = False
        results['details'].append(f"Bottom Cu: {bottom_cu_mean:.2e} != {C_CU_BOTTOM:.2e}")
    if abs(bottom_ni_mean - C_NI_BOTTOM) > tolerance:
        results['bottom_bc_ni'] = False
        results['details'].append(f"Bottom Ni: {bottom_ni_mean:.2e} != {C_NI_BOTTOM:.2e}")
    
    # Side boundaries: Zero flux (Neumann)
    left_flux_cu = np.mean(np.abs(c1[1, :] - c1[0, :]))
    left_flux_ni = np.mean(np.abs(c2[1, :] - c2[0, :]))
    right_flux_cu = np.mean(np.abs(c1[-1, :] - c1[-2, :]))
    right_flux_ni = np.mean(np.abs(c2[-1, :] - c2[-2, :]))
    if left_flux_cu > tolerance:
        results['left_flux_cu'] = False
        results['details'].append(f"Left flux Cu: {left_flux_cu:.2e}")
    if left_flux_ni > tolerance:
        results['left_flux_ni'] = False
        results['details'].append(f"Left flux Ni: {left_flux_ni:.2e}")
    if right_flux_cu > tolerance:
        results['right_flux_cu'] = False
        results['details'].append(f"Right flux Cu: {right_flux_cu:.2e}")
    if right_flux_ni > tolerance:
        results['right_flux_ni'] = False
        results['details'].append(f"Right flux Ni: {right_flux_ni:.2e}")
    
    results['valid'] = all([
        results['top_bc_cu'], results['top_bc_ni'],
        results['bottom_bc_cu'], results['bottom_bc_ni'],
        results['left_flux_cu'], results['left_flux_ni'],
        results['right_flux_cu'], results['right_flux_ni']
    ])
    return results

def interpolate_solution_with_boundary_masking(sol1, sol2, ly_target):
    """Interpolate only interior points, apply PINN boundary conditions"""
    if sol1 is None or sol2 is None:
        return None
    
    weight1 = (sol2['params']['Ly'] - ly_target) / (sol2['params']['Ly'] - sol1['params']['Ly'])
    weight2 = 1 - weight1
    
    X = sol1['X']
    y_coords = np.linspace(0, ly_target, sol1['Y'].shape[1])
    Y_target = np.tile(y_coords, (X.shape[0], 1))
    
    c1_interp = []
    c2_interp = []
    
    for t_idx in range(len(sol1['times'])):
        scale1 = ly_target / sol1['params']['Ly']
        scale2 = ly_target / sol2['params']['Ly']
        Y1_scaled = sol1['Y'][0, :] * scale1
        Y2_scaled = sol2['Y'][0, :] * scale2
        
        # Interpolate interior points only
        interp1_c1 = RegularGridInterpolator(
            (sol1['X'][:, 0], Y1_scaled), sol1['c1_preds'][t_idx],
            method='linear', bounds_error=False, fill_value=0
        )
        interp2_c1 = RegularGridInterpolator(
            (sol2['X'][:, 0], Y2_scaled), sol2['c1_preds'][t_idx],
            method='linear', bounds_error=False, fill_value=0
        )
        interp1_c2 = RegularGridInterpolator(
            (sol1['X'][:, 0], Y1_scaled), sol1['c2_preds'][t_idx],
            method='linear', bounds_error=False, fill_value=0
        )
        interp2_c2 = RegularGridInterpolator(
            (sol2['X'][:, 0], Y2_scaled), sol2['c2_preds'][t_idx],
            method='linear', bounds_error=False, fill_value=0
        )
        
        x_interior = X[1:-1, 0]
        y_interior = y_coords[1:-1]
        X_interior, Y_interior = np.meshgrid(x_interior, y_interior, indexing='ij')
        points_interior = np.stack([X_interior.flatten(), Y_interior.flatten()], axis=1)
        
        c1 = weight1 * interp1_c1(points_interior) + weight2 * interp2_c1(points_interior)
        c2 = weight1 * interp1_c2(points_interior) + weight2 * interp2_c2(points_interior)
        c1 = c1.reshape(X_interior.shape)
        c2 = c2.reshape(X_interior.shape)
        
        c1_full = np.zeros_like(X)
        c2_full = np.zeros_like(X)
        c1_full[1:-1, 1:-1] = c1
        c2_full[1:-1, 1:-1] = c2
        
        # Apply PINN boundary conditions
        c1_full[:, 0] = C_CU_TOP
        c1_full[:, -1] = C_CU_BOTTOM
        c2_full[:, 0] = C_NI_TOP
        c2_full[:, -1] = C_NI_BOTTOM
        c1_full[0, :] = c1_full[1, :]
        c1_full[-1, :] = c1_full[-2, :]
        c2_full[0, :] = c2_full[1, :]
        c2_full[-1, :] = c2_full[-2, :]
        
        c1_interp.append(c1_full)
        c2_interp.append(c2_full)
    
    return {
        'params': {'Lx': sol1['params']['Lx'], 'Ly': ly_target, 't_max': sol1['params']['t_max']},
        'X': X,
        'Y': Y_target,
        'c1_preds': c1_interp,
        'c2_preds': c2_interp,
        'times': sol1['times'],
        'interpolated': True,
        'method': 'linear_boundary_masking',
        'source_ly1': sol1['params']['Ly'],
        'source_ly2': sol2['params']['Ly'],
        'weights': (weight1, weight2)
    }

def compute_errors(sol_ref, sol_interp, time_idx):
    """Compute L2 errors and boundary-specific errors"""
    c1_ref = sol_ref['c1_preds'][time_idx]
    c2_ref = sol_ref['c2_preds'][time_idx]
    c1_interp = sol_interp['c1_preds'][time_idx]
    c2_interp = sol_interp['c2_preds'][time_idx]
    
    l2_error_cu = np.mean((c1_ref[1:-1, 1:-1] - c1_interp[1:-1, 1:-1])**2)
    l2_error_ni = np.mean((c2_ref[1:-1, 1:-1] - c2_interp[1:-1, 1:-1])**2)
    
    top_error_cu = np.mean(np.abs(c1_ref[:, 0] - c1_interp[:, 0]))
    top_error_ni = np.mean(np.abs(c2_ref[:, 0] - c2_interp[:, 0]))
    bottom_error_cu = np.mean(np.abs(c1_ref[:, -1] - c1_interp[:, -1]))
    bottom_error_ni = np.mean(np.abs(c2_ref[:, -1] - c2_interp[:, -1]))
    
    return {
        'l2_error_cu': l2_error_cu,
        'l2_error_ni': l2_error_ni,
        'top_error_cu': top_error_cu,
        'top_error_ni': top_error_ni,
        'bottom_error_cu': bottom_error_cu,
        'bottom_error_ni': bottom_error_ni
    }

def plot_2d_comparison(sol_ref, sol_interp, time_indices, ly_value, output_dir="figures"):
    """Plot 2D concentration distributions for PINN vs interpolated solutions"""
    Lx = sol_ref['params']['Lx']
    Ly = sol_ref['params']['Ly']
    
    for t_idx in time_indices:
        t_val = sol_ref['times'][t_idx]
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # Cu PINN
        im1 = axes[0, 0].imshow(sol_ref['c1_preds'][t_idx], origin='lower', 
                               extent=[0, Lx, 0, Ly], cmap='viridis',
                               vmin=0, vmax=C_CU_BOTTOM)
        axes[0, 0].set_title(f'Cu PINN (Ly={ly_value:.0f} μm, t={t_val:.1f} s)')
        axes[0, 0].set_xlabel('x (μm)')
        axes[0, 0].set_ylabel('y (μm)')
        axes[0, 0].grid(True, alpha=0.3)
        cb1 = fig.colorbar(im1, ax=axes[0, 0], label='Cu Conc. (mol/cc)', format='%.1e')
        
        # Cu Interpolated
        im2 = axes[0, 1].imshow(sol_interp['c1_preds'][t_idx], origin='lower', 
                               extent=[0, Lx, 0, Ly], cmap='viridis',
                               vmin=0, vmax=C_CU_BOTTOM)
        axes[0, 1].set_title(f'Cu Interpolated (Ly={ly_value:.0f} μm, t={t_val:.1f} s)')
        axes[0, 1].set_xlabel('x (μm)')
        axes[0, 1].set_ylabel('y (μm)')
        axes[0, 1].grid(True, alpha=0.3)
        cb2 = fig.colorbar(im2, ax=axes[0, 1], label='Cu Conc. (mol/cc)', format='%.1e')
        
        # Ni PINN
        im3 = axes[1, 0].imshow(sol_ref['c2_preds'][t_idx], origin='lower', 
                               extent=[0, Lx, 0, Ly], cmap='magma',
                               vmin=0, vmax=C_NI_TOP)
        axes[1, 0].set_title(f'Ni PINN (Ly={ly_value:.0f} μm, t={t_val:.1f} s)')
        axes[1, 0].set_xlabel('x (μm)')
        axes[1, 0].set_ylabel('y (μm)')
        axes[1, 0].grid(True, alpha=0.3)
        cb3 = fig.colorbar(im3, ax=axes[1, 0], label='Ni Conc. (mol/cc)', format='%.1e')
        
        # Ni Interpolated
        im4 = axes[1, 1].imshow(sol_interp['c2_preds'][t_idx], origin='lower', 
                               extent=[0, Lx, 0, Ly], cmap='magma',
                               vmin=0, vmax=C_NI_TOP)
        axes[1, 1].set_title(f'Ni Interpolated (Ly={ly_value:.0f} μm, t={t_val:.1f} s)')
        axes[1, 1].set_xlabel('x (μm)')
        axes[1, 1].set_ylabel('y (μm)')
        axes[1, 1].grid(True, alpha=0.3)
        cb4 = fig.colorbar(im4, ax=axes[1, 1], label='Ni Conc. (mol/cc)', format='%.1e')
        
        fig.suptitle(f'2D Concentration Profiles (Ly={ly_value:.0f} μm)', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        os.makedirs(output_dir, exist_ok=True)
        filename = f"2d_profiles_ly_{ly_value:.0f}_t_{t_val:.1f}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        yield fig, filename

def main():
    st.title("Debug Interpolation: 2D Concentration Profiles with Boundary Masking")
    
    # Load all solutions
    solutions, params_list, lys, load_logs = load_solutions(SOLUTION_DIR)
    
    if load_logs:
        with st.expander("Load Log"):
            for log in load_logs:
                st.write(log)
    
    if len(solutions) < 2:
        st.error("Need at least 2 solutions for interpolation. Please check pinn_solutions directory.")
        st.stop()
    
    # Find 30 μm and 120 μm solutions
    sol_30, ly_30 = find_solution_by_ly(solutions, params_list, 30.0)
    sol_120, ly_120 = find_solution_by_ly(solutions, params_list, 120.0)
    
    if sol_30 is None or sol_120 is None:
        st.error("Could not find solutions for Ly=30 μm or Ly=120 μm. Available Ly values: " + str(sorted(set(lys))))
        st.stop()
    
    # Find alternative solutions for interpolation
    sol_alt_30, ly_alt_30 = find_alternative_solution(solutions, params_list, ly_30, 60.0)
    sol_alt_120, ly_alt_120 = find_alternative_solution(solutions, params_list, ly_120, 90.0)
    
    if sol_alt_30 is None or sol_alt_120 is None:
        st.error("Could not find alternative solutions for interpolation. Available Ly values: " + str(sorted(set(lys))))
        st.stop()
    
    st.success(f"Loaded PINN solutions: Ly={ly_30:.1f} μm (with alt Ly={ly_alt_30:.1f} μm) and Ly={ly_120:.1f} μm (with alt Ly={ly_alt_120:.1f} μm)")
    
    # Enforce boundary conditions
    sol_30 = enforce_boundary_conditions(sol_30)
    sol_120 = enforce_boundary_conditions(sol_120)
    sol_alt_30 = enforce_boundary_conditions(sol_alt_30)
    sol_alt_120 = enforce_boundary_conditions(sol_alt_120)
    
    # Generate interpolated solutions
    st.subheader("Interpolation with Boundary Masking")
    with st.spinner("Interpolating..."):
        sol_interp_30 = interpolate_solution_with_boundary_masking(sol_30, sol_alt_30, 30.0)
        sol_interp_120 = interpolate_solution_with_boundary_masking(sol_120, sol_alt_120, 120.0)
    
    # Validate boundary conditions
    st.subheader("Boundary Condition Validation")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        bc_30 = validate_boundary_conditions(sol_30)
        st.metric("30 μm PINN BC", "✓" if bc_30['valid'] else "✗", f"{len(bc_30['details'])} issues")
    
    with col2:
        bc_120 = validate_boundary_conditions(sol_120)
        st.metric("120 μm PINN BC", "✓" if bc_120['valid'] else "✗", f"{len(bc_120['details'])} issues")
    
    with col3:
        bc_interp_30 = validate_boundary_conditions(sol_interp_30) if sol_interp_30 else {'valid': False, 'details': ['Not generated']}
        st.metric("30 μm Interp BC", "✓" if bc_interp_30['valid'] else "✗", f"{len(bc_interp_30['details'])} issues")
    
    with col4:
        bc_interp_120 = validate_boundary_conditions(sol_interp_120) if sol_interp_120 else {'valid': False, 'details': ['Not generated']}
        st.metric("120 μm Interp BC", "✓" if bc_interp_120['valid'] else "✗", f"{len(bc_interp_120['details'])} issues")
    
    # 2D profile comparison
    st.subheader("2D Concentration Profiles")
    time_indices = st.multiselect(
        "Select Time Indices for Comparison",
        options=list(range(len(sol_30['times']))),
        default=[0, len(sol_30['times'])//4, len(sol_30['times'])//2, 3*len(sol_30['times'])//4, len(sol_30['times'])-1],
        format_func=lambda x: f"t = {sol_30['times'][x]:.1f} s"
    )
    
    if time_indices:
        # 30 μm comparison
        if sol_interp_30:
            st.write("**Ly=30 μm: PINN vs Interpolated**")
            for fig, filename in plot_2d_comparison(sol_30, sol_interp_30, time_indices, 30.0):
                st.pyplot(fig)
                st.download_button(
                    label=f"Download 30 μm Plot (t={sol_30['times'][time_indices[0]]:.1f} s) as PNG",
                    data=open(os.path.join("figures", filename), "rb").read(),
                    file_name=filename,
                    mime="image/png"
                )
        
        # 120 μm comparison
        if sol_interp_120:
            st.write("**Ly=120 μm: PINN vs Interpolated**")
            for fig, filename in plot_2d_comparison(sol_120, sol_interp_120, time_indices, 120.0):
                st.pyplot(fig)
                st.download_button(
                    label=f"Download 120 μm Plot (t={sol_120['times'][time_indices[0]]:.1f} s) as PNG",
                    data=open(os.path.join("figures", filename), "rb").read(),
                    file_name=filename,
                    mime="image/png"
                )
    
    # Error metrics
    st.subheader("Error Metrics")
    if sol_interp_30:
        st.write("**30 μm Interpolation Errors**")
        for t_idx in time_indices:
            errors = compute_errors(sol_30, sol_interp_30, t_idx)
            st.write(f"t = {sol_30['times'][t_idx]:.1f} s: L2 Cu: {errors['l2_error_cu']:.2e}, L2 Ni: {errors['l2_error_ni']:.2e}")
    
    if sol_interp_120:
        st.write("**120 μm Interpolation Errors**")
        for t_idx in time_indices:
            errors = compute_errors(sol_120, sol_interp_120, t_idx)
            st.write(f"t = {sol_120['times'][t_idx]:.1f} s: L2 Cu: {errors['l2_error_cu']:.2e}, L2 Ni: {errors['l2_error_ni']:.2e}")
    
    # Boundary condition issues
    st.subheader("Boundary Condition Issues")
    st.write("**30 μm PINN**")
    for issue in bc_30['details']:
        st.write(f"• {issue}")
    st.write("**120 μm PINN**")
    for issue in bc_120['details']:
        st.write(f"• {issue}")
    if sol_interp_30:
        st.write("**30 μm Interpolated**")
        for issue in bc_interp_30['details']:
            st.write(f"• {issue}")
    if sol_interp_120:
        st.write("**120 μm Interpolated**")
        for issue in bc_interp_120['details']:
            st.write(f"• {issue}")
    
    # Clarification on previous curves
    st.subheader("Note on Previous Curve Plots")
    st.write("The 'curve' plots in the previous code were **centerline concentration profiles** plotted along **x = Lx/2** (midpoint in the x-direction, typically x=100 μm if Lx=200 μm) as a function of y (from 0 to Ly). These showed Cu and Ni concentrations along the y-direction at the domain's x-midpoint.")

if __name__ == "__main__":
    main()
