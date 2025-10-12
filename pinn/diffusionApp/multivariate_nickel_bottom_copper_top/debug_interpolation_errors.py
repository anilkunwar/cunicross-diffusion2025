import os
import pickle
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import matplotlib as mpl
from scipy import stats

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

def load_specific_solutions(solution_dir, ly_values=[30.0, 120.0]):
    solutions = {}
    load_logs = []
    for ly in ly_values:
        fname = f"solution_ly_{ly:.1f}.pkl"  # Assume naming convention, adjust if different
        file_path = os.path.join(solution_dir, fname)
        if os.path.exists(file_path):
            try:
                with open(file_path, "rb") as f:
                    sol = pickle.load(f)
                required_keys = ['params', 'X', 'Y', 'c1_preds', 'c2_preds', 'times']
                if all(key in sol for key in required_keys):
                    solutions[ly] = sol
                    load_logs.append(f"{fname}: Loaded successfully.")
                else:
                    missing_keys = [key for key in required_keys if key not in sol]
                    load_logs.append(f"{fname}: Skipped - Missing keys: {missing_keys}")
            except Exception as e:
                load_logs.append(f"{fname}: Skipped - Failed to load: {str(e)}")
        else:
            load_logs.append(f"{fname}: Skipped - File not found")
    return solutions, load_logs

def interpolate_solution(sol_low, sol_high, ly_target):
    # Simple linear interpolation between two solutions for demonstration
    weight_high = (ly_target - sol_low['params']['Ly']) / (sol_high['params']['Ly'] - sol_low['params']['Ly'])
    weight_low = 1 - weight_high
    
    X = sol_low['X']  # Assume same X grid
    y_coords = np.linspace(0, ly_target, sol_low['Y'].shape[1])
    Y = np.meshgrid(sol_low['Y'][0, :], y_coords, indexing='ij')[1]  # Wait, need to fix
    Y = np.tile(y_coords, (X.shape[0], 1))
    
    c1_interp = []
    c2_interp = []
    for t_idx in range(len(sol_low['times'])):
        # Scale y for low and high
        scale_low = ly_target / sol_low['params']['Ly']
        scale_high = ly_target / sol_high['params']['Ly']
        
        Y_low = sol_low['Y'][0, :] * scale_low
        Y_high = sol_high['Y'][0, :] * scale_high
        
        interp_low_c1 = RegularGridInterpolator(
            (sol_low['X'][:, 0], Y_low), sol_low['c1_preds'][t_idx],
            method='linear', bounds_error=False, fill_value=0
        )
        interp_high_c1 = RegularGridInterpolator(
            (sol_high['X'][:, 0], Y_high), sol_high['c1_preds'][t_idx],
            method='linear', bounds_error=False, fill_value=0
        )
        
        interp_low_c2 = RegularGridInterpolator(
            (sol_low['X'][:, 0], Y_low), sol_low['c2_preds'][t_idx],
            method='linear', bounds_error=False, fill_value=0
        )
        interp_high_c2 = RegularGridInterpolator(
            (sol_high['X'][:, 0], Y_high), sol_high['c2_preds'][t_idx],
            method='linear', bounds_error=False, fill_value=0
        )
        
        points = np.stack([X.flatten(), Y.flatten()], axis=1)
        
        c1 = weight_low * interp_low_c1(points) + weight_high * interp_high_c1(points)
        c2 = weight_low * interp_low_c2(points) + weight_high * interp_high_c2(points)
        
        c1_interp.append(c1.reshape(X.shape))
        c2_interp.append(c2.reshape(X.shape))
    
    interpolated_solution = {
        'params': {'Lx': sol_low['params']['Lx'], 'Ly': ly_target, 't_max': sol_low['params']['t_max']},
        'X': X,
        'Y': Y,
        'c1_preds': c1_interp,
        'c2_preds': c2_interp,
        'times': sol_low['times'],
        'interpolated': True
    }
    
    return interpolated_solution

def compare_profiles(sol_low, sol_high, sol_interp, time_index):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    center_idx = sol_low['X'].shape[0] // 2
    
    # Cu profiles
    ax1.plot(sol_low['Y'][0, :], sol_low['c1_preds'][time_index][center_idx, :], label='30 μm (PINN)')
    ax1.plot(sol_high['Y'][0, :], sol_high['c1_preds'][time_index][center_idx, :], label='120 μm (PINN)')
    ax1.plot(sol_interp['Y'][0, :], sol_interp['c1_preds'][time_index][center_idx, :], label='Interpolated')
    ax1.set_title('Cu Concentration Profiles')
    ax1.set_xlabel('y (μm)')
    ax1.set_ylabel('Concentration')
    ax1.legend()
    ax1.grid(True)
    
    # Ni profiles
    ax2.plot(sol_low['Y'][0, :], sol_low['c2_preds'][time_index][center_idx, :], label='30 μm (PINN)')
    ax2.plot(sol_high['Y'][0, :], sol_high['c2_preds'][time_index][center_idx, :], label='120 μm (PINN)')
    ax2.plot(sol_interp['Y'][0, :], sol_interp['c2_preds'][time_index][center_idx, :], label='Interpolated')
    ax2.set_title('Ni Concentration Profiles')
    ax2.set_xlabel('y (μm)')
    ax2.set_ylabel('Concentration')
    ax2.legend()
    ax2.grid(True)
    
    return fig

def analyze_differences(sol_low, sol_high, sol_interp):
    differences = []
    
    # Compare boundary conditions
    for name, sol in [("30 μm PINN", sol_low), ("120 μm PINN", sol_high), ("Interpolated", sol_interp)]:
        bc_validation = validate_boundary_conditions(sol)
        differences.append(f"{name}: {bc_validation['details'] if bc_validation['details'] else 'No issues'}")
    
    # Propose solutions
    proposals = [
        "1. Enforce fixed boundary values in interpolation code to match PINN: Cu bottom 1.6e-3, top 0; Ni bottom 0, top 1.25e-3",
        "2. Use boundary-aware interpolation, e.g., interpolate interior only and set boundaries fixed",
        "3. Add boundary loss terms to the attention mechanism if possible",
        "4. Normalize y-coordinates during interpolation to preserve boundary locations",
        "5. Use physics-constrained interpolation that incorporates zero-flux conditions"
    ]
    
    return differences, proposals

def main():
    st.title("Debug Code: Comparison between PINN Solutions and Attention Interpolations")

    st.subheader("Load Specific Solutions")
    solutions, load_logs = load_specific_solutions(SOLUTION_DIR)
    
    if load_logs:
        with st.expander("Load Log"):
            for log in load_logs:
                st.write(log)

    if len(solutions) < 2:
        st.error("Could not load both 30 μm and 120 μm solutions. Please check the pinn_solutions directory.")
        return

    sol_30 = solutions[30.0]
    sol_120 = solutions[120.0]

    st.subheader("Interpolation")
    ly_target = st.number_input("Target Ly for Interpolation (μm)", min_value=30.0, max_value=120.0, value=75.0, step=0.1, format="%.1f")
    sol_interp = interpolate_solution(sol_30, sol_120, ly_target)

    st.subheader("Compare Concentration Profiles")
    time_index = st.slider("Select Time Index for Comparison", 0, len(sol_30['times'])-1, len(sol_30['times'])-1)
    
    fig = compare_profiles(sol_30, sol_120, sol_interp, time_index)
    st.pyplot(fig)

    st.subheader("Boundary Condition Differences")
    differences, proposals = analyze_differences(sol_30, sol_120, sol_interp)
    for diff in differences:
        st.write(diff)

    st.subheader("Proposed Solutions for Boundary Issues")
    for proposal in proposals:
        st.write(proposal)

if __name__ == "__main__":
    main()
