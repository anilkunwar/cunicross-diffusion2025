import os
import pickle
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import cdist
from matplotlib import rcParams

# Disable LaTeX globally to prevent errors
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "mathtext.fontset": "dejavusans"
})

# Directory containing .pkl solution files (ensure exists)
SOLUTION_DIR = os.path.join(os.path.dirname(__file__), "pinn_solutions")
os.makedirs(SOLUTION_DIR, exist_ok=True)

# Diffusion types
DIFFUSION_TYPES = ['crossdiffusion', 'cu_selfdiffusion', 'ni_selfdiffusion']

# ------------------------------
# Utility functions
# ------------------------------

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
                missing = set(required) - set(sol.keys())
                load_logs.append(f"{fname}: Missing keys {missing}")
                continue

            # Parse diffusion type and Ly from filename - more flexible pattern
            patterns = [
                r"solution_(\w+)_ly_([\d.]+)_tmax_([\d.]+)\.pkl",
                r"solution_([a-zA-Z_]+)_ly_([\d.]+)_tmax_([\d.]+)\.pkl",
                r"(\w+)_ly_([\d.]+)_tmax_([\d.]+)\.pkl"
            ]
            
            match = None
            for pattern in patterns:
                match = re.match(pattern, fname)
                if match:
                    break
            
            if not match:
                load_logs.append(f"{fname}: Invalid filename format - {fname}")
                continue

            diff_type, ly_val, t_max = match.groups()
            ly_val = float(ly_val)
            t_max = float(t_max)

            # Normalize diffusion type
            diff_type = diff_type.lower()
            if diff_type not in DIFFUSION_TYPES:
                # Try to map variations
                type_map = {
                    'cross': 'crossdiffusion',
                    'cu_self': 'cu_selfdiffusion', 
                    'ni_self': 'ni_selfdiffusion',
                    'cucross': 'crossdiffusion',
                    'nicross': 'crossdiffusion',
                    'self_cu': 'cu_selfdiffusion',
                    'self_ni': 'ni_selfdiffusion'
                }
                diff_type = type_map.get(diff_type, diff_type)
                
            if diff_type not in DIFFUSION_TYPES:
                load_logs.append(f"{fname}: Unknown diffusion type '{diff_type}'. Known: {DIFFUSION_TYPES}")
                continue

            # Handle array orientation - more robust approach
            c1_preds = sol['c1_preds']
            c2_preds = sol['c2_preds']
            
            # Ensure we have lists of arrays
            if not (isinstance(c1_preds, list) and isinstance(c2_preds, list)):
                load_logs.append(f"{fname}: c1_preds/c2_preds are not lists")
                continue

            if len(c1_preds) == 0:
                load_logs.append(f"{fname}: Empty predictions")
                continue

            # Check and fix orientation
            first_shape = c1_preds[0].shape
            if len(first_shape) != 2:
                load_logs.append(f"{fname}: Invalid prediction shape {first_shape}")
                continue

            # Standardize to (50, 50) shape if needed
            if first_shape != (50, 50):
                try:
                    if first_shape == (2500,):
                        # Reshape flattened arrays
                        c1_preds = [c.reshape(50, 50) for c in c1_preds]
                        c2_preds = [c.reshape(50, 50) for c in c2_preds]
                        sol['orientation_note'] = "Reshaped from flattened"
                    else:
                        # Transpose to get (50, 50)
                        c1_preds = [c.T for c in c1_preds]
                        c2_preds = [c.T for c in c2_preds]
                        sol['orientation_note'] = f"Transposed from {first_shape} to (50, 50)"
                except Exception as e:
                    load_logs.append(f"{fname}: Shape adjustment failed - {str(e)}")
                    continue
            else:
                sol['orientation_note'] = "Already (50, 50)"

            # Update the solution
            sol.update({
                'c1_preds': c1_preds, 
                'c2_preds': c2_preds,
                'diffusion_type': diff_type, 
                'Ly_parsed': ly_val,
                'filename': fname
            })

            solutions.append(sol)
            metadata.append({
                'type': diff_type, 
                'Ly': ly_val, 
                'filename': fname,
                'shape': c1_preds[0].shape
            })
            load_logs.append(f"{fname}: ✓ Loaded [{diff_type}, Ly={ly_val:.1f}]")

        except Exception as e:
            load_logs.append(f"{fname}: ✗ Failed - {str(e)}")

    return solutions, metadata, load_logs

def compute_fluxes_and_grads(c1_preds, c2_preds, x_coords, y_coords, params):
    D11, D12, D21, D22 = params['D11'], params['D12'], params['D21'], params['D22']
    dx, dy = x_coords[1]-x_coords[0], y_coords[1]-y_coords[0]
    J1_preds, J2_preds, grad_c1_y, grad_c2_y = [], [], [], []
    for c1, c2 in zip(c1_preds, c2_preds):
        grad_c1_x, grad_c1_y_i = np.gradient(c1, dx, axis=1), np.gradient(c1, dy, axis=0)
        grad_c2_x, grad_c2_y_i = np.gradient(c2, dx, axis=1), np.gradient(c2, dy, axis=0)
        J1_preds.append([-(D11*grad_c1_x + D12*grad_c2_x), -(D11*grad_c1_y_i + D12*grad_c2_y_i)])
        J2_preds.append([-(D21*grad_c1_x + D22*grad_c2_x), -(D21*grad_c1_y_i + D22*grad_c2_y_i)])
        grad_c1_y.append(grad_c1_y_i)
        grad_c2_y.append(grad_c2_y_i)
    return J1_preds, J2_preds, grad_c1_y, grad_c2_y

@st.cache_data
def load_and_process_solution(solutions, diff_type, ly_target, tol=1e-4):
    exact = [s for s in solutions if s['diffusion_type']==diff_type and abs(s['Ly_parsed']-ly_target)<tol]
    if exact:
        solution = exact[0]
        solution['interpolated'] = False
    else:
        solution = attention_weighted_interpolation(solutions, [s['Ly_parsed'] for s in solutions], ly_target, diff_type)
    if solution:
        J1, J2, grad_c1, grad_c2 = compute_fluxes_and_grads(
            solution['c1_preds'], solution['c2_preds'],
            solution['X'][:,0], solution['Y'][0,:], solution['params'])
        solution.update({'J1_preds': J1, 'J2_preds': J2,
                         'grad_c1_y': grad_c1, 'grad_c2_y': grad_c2})
    return solution

def attention_weighted_interpolation(solutions, lys, ly_target, diff_type, sigma=2.5):
    matching = [s for s in solutions if s['diffusion_type']==diff_type]
    if not matching: return None
    lys = np.array([s['Ly_parsed'] for s in matching])
    weights = get_interpolation_weights(lys, ly_target, sigma)
    Lx, t_max = matching[0]['params']['Lx'], matching[0]['params']['t_max']
    x_coords = np.linspace(0,Lx,50)
    y_coords = np.linspace(0,ly_target,50)
    times = np.linspace(0,t_max,50)
    c1_interp = np.zeros((len(times),50,50))
    c2_interp = np.zeros((len(times),50,50))
    for sol,w in zip(matching, weights):
        X_sol = sol['X'][:,0]
        Y_sol = sol['Y'][0,:]*(ly_target/sol['params']['Ly'])
        for t_idx in range(len(times)):
            interp_c1 = RegularGridInterpolator((Y_sol,X_sol), sol['c1_preds'][t_idx], method='linear', bounds_error=False, fill_value=0)
            interp_c2 = RegularGridInterpolator((Y_sol,X_sol), sol['c2_preds'][t_idx], method='linear', bounds_error=False, fill_value=0)
            X_target,Y_target = np.meshgrid(x_coords,y_coords,indexing='ij')
            points = np.column_stack([Y_target.ravel(), X_target.ravel()])
            c1_interp[t_idx] += w*interp_c1(points).reshape(50,50)
            c2_interp[t_idx] += w*interp_c2(points).reshape(50,50)
    X,Y = np.meshgrid(x_coords,y_coords,indexing='ij')
    param_set = matching[0]['params'].copy()
    param_set['Ly'] = ly_target
    return {'params': param_set,'X': X,'Y': Y,'c1_preds': list(c1_interp),'c2_preds': list(c2_interp),
            'times': times,'diffusion_type': diff_type,'interpolated': True,
            'used_lys': lys.tolist(),'attention_weights': weights.tolist(),'orientation_note':"rows=y, cols=x"}

def get_interpolation_weights(lys, ly_target, sigma=2.5):
    lys = np.array(lys).reshape(-1,1)
    distances = cdist(np.array([[ly_target]]), lys).flatten()
    weights = np.exp(-(distances**2)/(2*sigma**2))
    weights /= weights.sum()+1e-10
    return weights

def detect_uphill(solution, time_index):
    J1_y = solution['J1_preds'][time_index][1]
    grad_c1_y = solution['grad_c1_y'][time_index]
    J2_y = solution['J2_preds'][time_index][1]
    grad_c2_y = solution['grad_c2_y'][time_index]
    return J1_y*grad_c1_y>0, J2_y*grad_c2_y>0

# ------------------------------
# Plot customization
# ------------------------------
def get_plot_customization():
    st.sidebar.header("Plot Styling Options")
    color_cu = st.sidebar.color_picker("Cu Curve Color", "#1f77b4")
    color_ni = st.sidebar.color_picker("Ni Curve Color", "#ff7f0e")
    linewidth = st.sidebar.slider("Line Width", 1, 5, 2)
    linestyle = st.sidebar.selectbox("Line Style", ["solid","dashed","dotted"],0)
    font_size = st.sidebar.slider("Font Size", 8, 24, 14)
    dpi = st.sidebar.slider("DPI (Matplotlib)",100,600,300)
    colorscale = st.sidebar.selectbox("Heatmap Colorscale", ["RdBu","Viridis","Cividis"],0)
    return color_cu,color_ni,linewidth,linestyle,font_size,dpi,colorscale

def plot_flux_vs_gradient(solution,time_index,color_cu,color_ni,linewidth,linestyle,font_size,dpi):
    x_idx = 25
    y_coords = solution['Y'][0,:]
    t_val = solution['times'][time_index]
    diff_type = solution['diffusion_type']

    J1_y_center = solution['J1_preds'][time_index][1][:,x_idx]
    grad_c1_y_center = solution['grad_c1_y'][time_index][:,x_idx]
    J2_y_center = solution['J2_preds'][time_index][1][:,x_idx]
    grad_c2_y_center = solution['grad_c2_y'][time_index][:,x_idx]

    # Use matplotlib's built-in mathtext instead of LaTeX
    plt.rcParams.update({
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size+2,
        "legend.fontsize": font_size-2,
        "mathtext.fontset": "dejavusans"
    })
    
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,6),dpi=dpi)
    
    # Use matplotlib's mathtext (without LaTeX)
    ax1.plot(y_coords,-grad_c1_y_center,label=r'$-\nabla C_{Cu}$',color=color_cu,linewidth=linewidth,linestyle=linestyle)
    ax1.plot(y_coords,J1_y_center,label=r'$J_{Cu}$',color=color_cu,linewidth=linewidth,linestyle='--')
    ax1.set_xlabel('y (μm)')
    ax1.set_ylabel('Flux / -Gradient')
    ax1.set_title(f'Cu Flux vs Gradient @ t={t_val:.1f}s')
    ax1.legend()
    
    ax2.plot(y_coords,-grad_c2_y_center,label=r'$-\nabla C_{Ni}$',color=color_ni,linewidth=linewidth,linestyle=linestyle)
    ax2.plot(y_coords,J2_y_center,label=r'$J_{Ni}$',color=color_ni,linewidth=linewidth,linestyle='--')
    ax2.set_xlabel('y (μm)')
    ax2.set_ylabel('Flux / -Gradient')
    ax2.set_title(f'Ni Flux vs Gradient @ t={t_val:.1f}s')
    ax2.legend()
    
    plt.suptitle(f"Flux vs Gradient: {diff_type.replace('_',' ')}",fontsize=font_size+4)
    plt.tight_layout(rect=[0,0,1,0.95])
    st.pyplot(fig)
    plt.close()

def plot_uphill_regions(solution,time_index,downsample,colorscale='RdBu'):
    x_coords = solution['X'][:,0]
    y_coords = solution['Y'][0,:]
    t_val = solution['times'][time_index]
    diff_type = solution['diffusion_type']
    ds = max(1,downsample)
    x_indices = np.unique(np.linspace(0,len(x_coords)-1,num=len(x_coords)//ds,dtype=int))
    y_indices = np.unique(np.linspace(0,len(y_coords)-1,num=len(y_coords)//ds,dtype=int))
    x_ds, y_ds = x_coords[x_indices], y_coords[y_indices]

    uphill_cu, uphill_ni = detect_uphill(solution,time_index)
    uphill_cu_ds = uphill_cu[np.ix_(y_indices,x_indices)]
    uphill_ni_ds = uphill_ni[np.ix_(y_indices,x_indices)]

    fig = make_subplots(rows=1,cols=2,subplot_titles=("Cu Uphill Regions","Ni Uphill Regions"))
    for i,(z,title) in enumerate(zip([uphill_cu_ds,uphill_ni_ds],["Cu Uphill","Ni Uphill"])):
        fig.add_trace(go.Heatmap(
            x=x_ds,y=y_ds,z=z.astype(float),colorscale=colorscale,
            colorbar=dict(title='Uphill (1/0)'),zsmooth='best'
        ),row=1,col=i+1)
    fig.update_layout(height=500,title=f"Uphill Diffusion Regions: {diff_type.replace('_',' ')} @ t={t_val:.1f}s",template='plotly_white')
    st.plotly_chart(fig,use_container_width=True)

# ------------------------------
# Main App
# ------------------------------
def main():
    st.title("Theoretical Assessment of Diffusion Solutions")

    st.markdown("""
    ### Applicable Theories
    1. **Fick's First Law (Extended for Multicomponent Systems)**:
       J_i = - Σ D_ij ∇ C_j
       - Self-diffusion: D_ij = 0 for i ≠ j
       - Cross-diffusion: Off-diagonal terms may cause uphill diffusion (J_i · ∇ C_i > 0)
    2. **Continuity Equation**:
       ∂C_i/∂t = - ∇ · J_i
    """)

    solutions, metadata, load_logs = load_solutions(SOLUTION_DIR)
    
    # DEBUG: Show what was loaded
    with st.expander("🔍 Debug: File Loading Results", expanded=False):
        st.subheader("Load Summary")
        
        if solutions:
            loaded_types = set([s['diffusion_type'] for s in solutions])
            st.success(f"✅ Successfully loaded {len(solutions)} solution files")
            st.write(f"**Loaded diffusion types:** {loaded_types}")
            
            # Show breakdown by type
            for diff_type in DIFFUSION_TYPES:
                type_solutions = [s for s in solutions if s['diffusion_type'] == diff_type]
                if type_solutions:
                    lys = sorted(set([s['Ly_parsed'] for s in type_solutions]))
                    st.write(f"- **{diff_type}**: {len(type_solutions)} files, Ly values: {lys}")
                else:
                    st.write(f"- **{diff_type}**: ❌ No files loaded")
        else:
            st.error("❌ No solutions loaded!")
        
        st.subheader("Detailed Load Logs")
        log_container = st.container(height=300)
        for log in load_logs:
            if "✓" in log or "Loaded" in log:
                log_container.success(log)
            elif "Skipped" in log:
                log_container.warning(log)
            elif "Failed" in log or "✗" in log:
                log_container.error(log)
            else:
                log_container.info(log)

    if not solutions:
        st.error("No valid solution files found in the pinn_solutions directory.")
        st.info("""
        **Troubleshooting tips:**
        - Check that .pkl files exist in the `pinn_solutions` folder
        - Verify filenames follow pattern: `solution_[type]_ly_[value]_tmax_[value].pkl`
        - Supported types: crossdiffusion, cu_selfdiffusion, ni_selfdiffusion
        - Example: `solution_crossdiffusion_ly_50.0_tmax_100.0.pkl`
        """)
        return

    # Sidebar parameters
    st.sidebar.header("Simulation Parameters")
    
    # Get available diffusion types that actually have solutions
    available_types = sorted(set(s['diffusion_type'] for s in solutions))
    if not available_types:
        st.error("No diffusion types available despite loading solutions. Check debug info.")
        return
        
    diff_type = st.sidebar.selectbox("Diffusion Type", available_types, 
                                    format_func=lambda x: x.replace('_',' ').title())
    
    # Get available Ly values for the selected diffusion type
    available_lys = sorted(set(s['Ly_parsed'] for s in solutions if s['diffusion_type'] == diff_type))
    if not available_lys:
        st.error(f"No Ly values available for {diff_type}. Check debug info.")
        return
        
    ly_target = st.sidebar.select_slider("Ly (μm)", options=available_lys, 
                                        value=available_lys[0] if available_lys else 50.0)
    
    time_index = st.sidebar.slider("Time Index", 0, 49, 49)
    downsample = st.sidebar.slider("Downsample", 1, 5, 2)

    # Plot customization
    color_cu, color_ni, linewidth, linestyle, font_size, dpi, colorscale = get_plot_customization()

    solution = load_and_process_solution(solutions, diff_type, ly_target)
    if solution:
        st.subheader("Uphill Diffusion Detection")
        st.markdown("Uphill occurs when J_y · ∇_y C > 0")
        plot_uphill_regions(solution, time_index, downsample, colorscale)

        st.subheader("Flux vs Gradient Comparison")
        st.markdown("Plots J_y vs -∇_y C along the central line.")
        plot_flux_vs_gradient(solution, time_index, color_cu, color_ni, linewidth, linestyle, font_size, dpi)
        
        # Show solution info
        with st.expander("Solution Information"):
            st.write(f"**Diffusion type:** {solution['diffusion_type']}")
            st.write(f"**Ly:** {solution['params']['Ly']} μm")
            st.write(f"**Lx:** {solution['params']['Lx']} μm") 
            st.write(f"**Time range:** {solution['times'][0]:.1f}s to {solution['times'][-1]:.1f}s")
            st.write(f"**Array shape:** {solution['c1_preds'][0].shape}")
            st.write(f"**Orientation:** {solution.get('orientation_note', 'Not specified')}")
            if solution.get('interpolated'):
                st.write("**Note:** This solution was interpolated from multiple files")
                st.write(f"**Source Ly values:** {solution.get('used_lys', [])}")
    else:
        st.error(f"No solution available for {diff_type} with Ly={ly_target:.1f}")

if __name__ == "__main__":
    main()
