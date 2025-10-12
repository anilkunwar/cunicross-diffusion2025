import os
import pickle
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import matplotlib as mpl
from scipy import stats
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

def simple_interpolate_solution(sol1, sol2, ly_target):
    """Simple linear interpolation between two solutions"""
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
        
        points = np.stack([X.flatten(), Y_target.flatten()], axis=1)
        
        c1 = weight1 * interp1_c1(points) + weight2 * interp2_c1(points)
        c2 = weight1 * interp1_c2(points) + weight2 * interp2_c2(points)
        
        c1 = c1.reshape(X.shape)
        c2 = c2.reshape(X.shape)
        
        # Enforce boundary conditions
        c1[:, 0] = C_CU_TOP
        c1[:, -1] = C_CU_BOTTOM
        c2[:, 0] = C_NI_TOP
        c2[:, -1] = C_NI_BOTTOM
        c1[0, :] = c1[1, :]
        c1[-1, :] = c1[-2, :]
        c2[0, :] = c2[1, :]
        c2[-1, :] = c2[-2, :]
        
        c1_interp.append(c1)
        c2_interp.append(c2)
    
    return {
        'params': {'Lx': sol1['params']['Lx'], 'Ly': ly_target, 't_max': sol1['params']['t_max']},
        'X': X,
        'Y': Y_target,
        'c1_preds': c1_interp,
        'c2_preds': c2_interp,
        'times': sol1['times'],
        'interpolated': True,
        'method': 'linear'
    }

def attention_interpolate_solution(solutions, params_list, ly_target):
    """Attention-based interpolation using all solutions"""
    class SimpleAttentionInterpolator:
        def __init__(self, sigma=0.2):
            self.sigma = sigma
        
        def interpolate(self, solutions, params_list, ly_target):
            lys = np.array([p[0] for p in params_list])
            distances = np.abs(lys - ly_target)
            weights = np.exp(-(distances / self.sigma)**2)
            weights /= np.sum(weights)
            
            Lx = solutions[0]['params']['Lx']
            t_max = solutions[0]['params']['t_max']
            x_coords = np.linspace(0, Lx, 50)
            y_coords = np.linspace(0, ly_target, 50)
            X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
            
            c1_interp = np.zeros((len(solutions[0]['times']), 50, 50))
            c2_interp = np.zeros((len(solutions[0]['times']), 50, 50))
            
            for t_idx in range(len(solutions[0]['times'])):
                for i, (sol, weight) in enumerate(zip(solutions, weights)):
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
                
                # Enforce boundary conditions
                c1_interp[t_idx, :, 0] = C_CU_TOP
                c1_interp[t_idx, :, -1] = C_CU_BOTTOM
                c2_interp[t_idx, :, 0] = C_NI_TOP
                c2_interp[t_idx, :, -1] = C_NI_BOTTOM
                c1_interp[t_idx, 0, :] = c1_interp[t_idx, 1, :]
                c1_interp[t_idx, -1, :] = c1_interp[t_idx, -2, :]
                c2_interp[t_idx, 0, :] = c2_interp[t_idx, 1, :]
                c2_interp[t_idx, -1, :] = c2_interp[t_idx, -2, :]
            
            return {
                'params': {'Lx': Lx, 'Ly': ly_target, 't_max': t_max},
                'X': X,
                'Y': Y,
                'c1_preds': list(c1_interp),
                'c2_preds': list(c2_interp),
                'times': solutions[0]['times'],
                'interpolated': True,
                'method': 'attention',
                'weights': weights.tolist()
            }
    
    interpolator = SimpleAttentionInterpolator()
    return interpolator.interpolate(solutions, params_list, ly_target)

def compute_errors(sol_ref, sol_interp, time_idx):
    """Compute L2 errors and boundary-specific errors"""
    c1_ref = sol_ref['c1_preds'][time_idx]
    c2_ref = sol_ref['c2_preds'][time_idx]
    c1_interp = sol_interp['c1_preds'][time_idx]
    c2_interp = sol_interp['c2_preds'][time_idx]
    
    # Overall L2 error
    l2_error_cu = np.mean((c1_ref - c1_interp)**2)
    l2_error_ni = np.mean((c2_ref - c2_interp)**2)
    
    # Boundary-specific errors
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

def plot_comparison(sol_30, sol_120, sol_linear, sol_attention, time_idx):
    """Plot centerline profiles and boundary errors"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    center_idx = sol_30['X'].shape[0] // 2
    
    # Cu profiles
    axes[0, 0].plot(sol_30['Y'][0, :], sol_30['c1_preds'][time_idx][center_idx, :], 'b-', label='30 μm PINN')
    axes[0, 0].plot(sol_120['Y'][0, :], sol_120['c1_preds'][time_idx][center_idx, :], 'g-', label='120 μm PINN')
    if sol_linear:
        axes[0, 0].plot(sol_linear['Y'][0, :], sol_linear['c1_preds'][time_idx][center_idx, :], 'r--', label='Linear Interp')
    if sol_attention:
        axes[0, 0].plot(sol_attention['Y'][0, :], sol_attention['c1_preds'][time_idx][center_idx, :], 'm--', label='Attention Interp')
    axes[0, 0].set_title('Cu Concentration Profiles')
    axes[0, 0].set_xlabel('y (μm)')
    axes[0, 0].set_ylabel('Concentration (mol/cc)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Ni profiles
    axes[0, 1].plot(sol_30['Y'][0, :], sol_30['c2_preds'][time_idx][center_idx, :], 'b-', label='30 μm PINN')
    axes[0, 1].plot(sol_120['Y'][0, :], sol_120['c2_preds'][time_idx][center_idx, :], 'g-', label='120 μm PINN')
    if sol_linear:
        axes[0, 1].plot(sol_linear['Y'][0, :], sol_linear['c1_preds'][time_idx][center_idx, :], 'r--', label='Linear Interp')
    if sol_attention:
        axes[0, 1].plot(sol_attention['Y'][0, :], sol_attention['c2_preds'][time_idx][center_idx, :], 'm--', label='Attention Interp')
    axes[0, 1].set_title('Ni Concentration Profiles')
    axes[0, 1].set_xlabel('y (μm)')
    axes[0, 1].set_ylabel('Concentration (mol/cc)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Boundary errors
    if sol_linear and sol_30['params']['Ly'] == sol_linear['params']['Ly']:
        errors_linear_30 = compute_errors(sol_30, sol_linear, time_idx)
        axes[1, 0].bar(['Top Cu', 'Bottom Cu', 'Top Ni', 'Bottom Ni'],
                        [errors_linear_30['top_error_cu'], errors_linear_30['bottom_error_cu'],
                         errors_linear_30['top_error_ni'], errors_linear_30['bottom_error_ni']],
                        color='r', alpha=0.5, label='Linear vs 30 μm')
    if sol_attention and sol_30['params']['Ly'] == sol_attention['params']['Ly']:
        errors_attention_30 = compute_errors(sol_30, sol_attention, time_idx)
        axes[1, 0].bar(['Top Cu', 'Bottom Cu', 'Top Ni', 'Bottom Ni'],
                        [errors_attention_30['top_error_cu'], errors_attention_30['bottom_error_cu'],
                         errors_attention_30['top_error_ni'], errors_attention_30['bottom_error_ni']],
                        color='m', alpha=0.5, label='Attention vs 30 μm')
    axes[1, 0].set_title('Boundary Errors vs 30 μm PINN')
    axes[1, 0].set_ylabel('Mean Absolute Error')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    if sol_linear and sol_120['params']['Ly'] == sol_linear['params']['Ly']:
        errors_linear_120 = compute_errors(sol_120, sol_linear, time_idx)
        axes[1, 1].bar(['Top Cu', 'Bottom Cu', 'Top Ni', 'Bottom Ni'],
                        [errors_linear_120['top_error_cu'], errors_linear_120['bottom_error_cu'],
                         errors_linear_120['top_error_ni'], errors_linear_120['bottom_error_ni']],
                        color='r', alpha=0.5, label='Linear vs 120 μm')
    if sol_attention and sol_120['params']['Ly'] == sol_attention['params']['Ly']:
        errors_attention_120 = compute_errors(sol_120, sol_attention, time_idx)
        axes[1, 1].bar(['Top Cu', 'Bottom Cu', 'Top Ni', 'Bottom Ni'],
                        [errors_attention_120['top_error_cu'], errors_attention_120['bottom_error_cu'],
                         errors_attention_120['top_error_ni'], errors_attention_120['bottom_error_ni']],
                        color='m', alpha=0.5, label='Attention vs 120 μm')
    axes[1, 1].set_title('Boundary Errors vs 120 μm PINN')
    axes[1, 1].set_ylabel('Mean Absolute Error')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    return fig

def main():
    st.title("Debug Interpolation Errors: PINN vs Interpolated Solutions")
    
    # Load all solutions
    solutions, params_list, lys, load_logs = load_solutions(SOLUTION_DIR)
    
    if load_logs:
        with st.expander("Load Log"):
            for log in load_logs:
                st.write(log)
    
    if len(solutions) < 2:
        st.error("Need at least 2 solutions for comparison. Please check pinn_solutions directory.")
        st.stop()
    
    # Find 30 μm and 120 μm solutions
    sol_30, ly_30 = find_solution_by_ly(solutions, params_list, 30.0)
    sol_120, ly_120 = find_solution_by_ly(solutions, params_list, 120.0)
    
    if sol_30 is None or sol_120 is None:
        st.error("Could not find solutions for Ly=30 μm or Ly=120 μm. Available Ly values: " + str(sorted(set(lys))))
        st.stop()
    
    st.success(f"Loaded PINN solutions: Ly={ly_30:.1f} μm and Ly={ly_120:.1f} μm")
    
    # Enforce boundary conditions on reference solutions
    sol_30 = enforce_boundary_conditions(sol_30)
    sol_120 = enforce_boundary_conditions(sol_120)
    
    # Select interpolation target
    st.subheader("Interpolation Target")
    ly_target = st.number_input("Target Ly for Interpolation (μm)", 
                                min_value=30.0, max_value=120.0, value=75.0, step=0.1, format="%.1f")
    
    # Generate interpolated solutions
    if st.button("Generate Interpolated Solutions"):
        with st.spinner("Interpolating..."):
            sol_linear = simple_interpolate_solution(sol_30, sol_120, ly_target)
            sol_attention = attention_interpolate_solution(solutions, params_list, ly_target)
        
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
            bc_linear = validate_boundary_conditions(sol_linear) if sol_linear else {'valid': False, 'details': ['Not generated']}
            st.metric("Linear Interp BC", "✓" if bc_linear['valid'] else "✗", f"{len(bc_linear['details'])} issues")
        
        with col4:
            bc_attention = validate_boundary_conditions(sol_attention) if sol_attention else {'valid': False, 'details': ['Not generated']}
            st.metric("Attention Interp BC", "✓" if bc_attention['valid'] else "✗", f"{len(bc_attention['details'])} issues")
        
        # Compare profiles
        st.subheader("Profile and Error Comparison")
        time_idx = st.slider("Select Time Index for Comparison", 0, len(sol_30['times'])-1, len(sol_30['times'])-1)
        
        if sol_linear or sol_attention:
            fig = plot_comparison(sol_30, sol_120, sol_linear, sol_attention, time_idx)
            st.pyplot(fig)
        
        # Detailed boundary issues
        st.subheader("Boundary Condition Issues")
        st.write("**30 μm PINN**")
        for issue in bc_30['details']:
            st.write(f"• {issue}")
        st.write("**120 μm PINN**")
        for issue in bc_120['details']:
            st.write(f"• {issue}")
        if sol_linear:
            st.write("**Linear Interpolation**")
            for issue in bc_linear['details']:
                st.write(f"• {issue}")
        if sol_attention:
            st.write("**Attention Interpolation**")
            for issue in bc_attention['details']:
                st.write(f"• {issue}")
        
        # Error metrics
        st.subheader("Error Metrics")
        if sol_linear and sol_30['params']['Ly'] == sol_linear['params']['Ly']:
            errors_linear_30 = compute_errors(sol_30, sol_linear, time_idx)
            st.write(f"**Linear vs 30 μm PINN**: L2 Cu: {errors_linear_30['l2_error_cu']:.2e}, L2 Ni: {errors_linear_30['l2_error_ni']:.2e}")
        if sol_linear and sol_120['params']['Ly'] == sol_linear['params']['Ly']:
            errors_linear_120 = compute_errors(sol_120, sol_linear, time_idx)
            st.write(f"**Linear vs 120 μm PINN**: L2 Cu: {errors_linear_120['l2_error_cu']:.2e}, L2 Ni: {errors_linear_120['l2_error_ni']:.2e}")
        if sol_attention and sol_30['params']['Ly'] == sol_attention['params']['Ly']:
            errors_attention_30 = compute_errors(sol_30, sol_attention, time_idx)
            st.write(f"**Attention vs 30 μm PINN**: L2 Cu: {errors_attention_30['l2_error_cu']:.2e}, L2 Ni: {errors_attention_30['l2_error_ni']:.2e}")
        if sol_attention and sol_120['params']['Ly'] == sol_attention['params']['Ly']:
            errors_attention_120 = compute_errors(sol_120, sol_attention, time_idx)
            st.write(f"**Attention vs 120 μm PINN**: L2 Cu: {errors_attention_120['l2_error_cu']:.2e}, L2 Ni: {errors_attention_120['l2_error_ni']:.2e}")
        
        # Recommendations
        st.subheader("Recommendations for Fixing Boundary Issues")
        recommendations = [
            "1. **Boundary Masking**: Interpolate only interior points, apply PINN BCs (Cu: 1.6e-3 bottom/0 top, Ni: 0 bottom/1.25e-3 top) post-interpolation.",
            "2. **Coordinate Normalization**: Normalize y to [0,1] before interpolation to align boundaries, then denormalize.",
            "3. **Physics-Constrained Weights**: Modify attention to downweight boundary regions or use separate boundary interpolation.",
            "4. **Boundary Loss**: Add boundary condition residual to interpolation objective.",
            "5. **Hybrid Approach**: Use PINN solutions near boundaries, interpolate only in interior with smooth blending."
        ]
        for rec in recommendations:
            st.write(f"• {rec}")

if __name__ == "__main__":
    main()
