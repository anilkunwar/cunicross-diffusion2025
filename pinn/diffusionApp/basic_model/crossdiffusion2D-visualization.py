# visualize.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pickle
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import cdist

# Configuration
#SOLUTION_DIR = "pinn_solutions"
SOLUTION_DIR = os.path.join(os.path.dirname(__file__), "pinn_solutions")
MAX_CACHE_SIZE = 10  # Adjust based on available memory
DOWNSAMPLE_FACTOR = 2  # Reduce rendering resolution

# Initialize session state
if 'solution_cache' not in st.session_state:
    st.session_state.solution_cache = {}
if 'current_solution' not in st.session_state:
    st.session_state.current_solution = None
if 'last_params' not in st.session_state:
    st.session_state.last_params = None

@st.cache_data(max_entries=MAX_CACHE_SIZE, show_spinner=False)
def load_solution_file(file_path):
    """Load and optimize a single solution file"""
    try:
        with open(file_path, 'rb') as f:
            solution = pickle.load(f)
            # Optimize data types
            return {
                'params': solution['params'],
                'X': solution['X'].astype(np.float32),
                'Y': solution['Y'].astype(np.float32),
                'c1_preds': [c.astype(np.float32) for c in solution['c1_preds']],
                'c2_preds': [c.astype(np.float32) for c in solution['c2_preds']],
                'times': solution['times'].astype(np.float32)
            }
    except Exception as e:
        st.error(f"Error loading {file_path}: {str(e)}")
        return None

def get_cache_key(params):
    """Generate unique cache key for parameters"""
    return f"{params['D11']:.6f}-{params['D12']:.6f}-{params['D21']:.6f}-{params['D22']:.6f}-{params['Lx']:.1f}-{params['t_max']:.1f}"

def filter_solutions(solution_files, target_params):
    """Efficiently filter solutions using filename parsing"""
    valid_files = []
    for fname in solution_files:
        parts = fname.split('_')
        try:
            if (
                abs(float(parts[4]) - target_params['D11']) < 1e-6 and
                abs(float(parts[6]) - target_params['D12']) < 1e-6 and
                abs(float(parts[8]) - target_params['D21']) < 1e-6 and
                abs(float(parts[10]) - target_params['D22']) < 1e-6 and
                abs(float(parts[12]) - target_params['Lx']) < 1e-2 and
                abs(float(parts[14].split('.')[0]) - target_params['t_max']) < 1e-2
            ):
                valid_files.append(fname)
        except:
            continue
    return valid_files

def load_relevant_solutions(target_params):
    """Load solutions with cache management"""
    cache_key = get_cache_key(target_params)
    
    # Return cached solutions if available
    if cache_key in st.session_state.solution_cache:
        return st.session_state.solution_cache[cache_key]
    
    # Find matching files
    solution_files = os.listdir(SOLUTION_DIR)
    valid_files = filter_solutions(solution_files, target_params)
    
    # Load solutions with cache
    solutions = []
    for fname in valid_files:
        file_path = os.path.join(SOLUTION_DIR, fname)
        solution = load_solution_file(file_path)
        if solution:
            solutions.append(solution)
    
    # Update cache
    if len(st.session_state.solution_cache) >= MAX_CACHE_SIZE:
        oldest_key = next(iter(st.session_state.solution_cache))
        del st.session_state.solution_cache[oldest_key]
    
    st.session_state.solution_cache[cache_key] = solutions
    return solutions

def create_interpolated_solution(solutions, ly_target):
    """Create interpolated solution with attention weighting"""
    lys = np.array([s['params']['Ly'] for s in solutions])
    solution_coords = lys.reshape(-1, 1)
    target_coord = np.array([[ly_target]])
    
    # Calculate Gaussian weights
    distances = cdist(target_coord, solution_coords).flatten()
    sigma = 2.5  # Kernel bandwidth
    weights = np.exp(-(distances**2)/(2*sigma**2))
    weights /= weights.sum()
    
    # Interpolation logic
    Lx = solutions[0]['params']['Lx']
    t_max = solutions[0]['params']['t_max']
    x_coords = np.linspace(0, Lx, 50, dtype=np.float32)
    y_coords = np.linspace(0, ly_target, 50, dtype=np.float32)
    times = np.linspace(0, t_max, 50, dtype=np.float32)
    
    c1_interp = np.zeros((len(times), 50, 50), dtype=np.float32)
    c2_interp = np.zeros((len(times), 50, 50), dtype=np.float32)
    
    for weight, solution in zip(weights, solutions):
        X_sol = solution['X'][:,0]
        Y_sol = solution['Y'][0,:] * (ly_target / solution['params']['Ly'])
        
        for t_idx in range(len(times)):
            interp_c1 = RegularGridInterpolator(
                (X_sol, Y_sol), solution['c1_preds'][t_idx],
                method='linear', bounds_error=False, fill_value=None
            )
            interp_c2 = RegularGridInterpolator(
                (X_sol, Y_sol), solution['c2_preds'][t_idx],
                method='linear', bounds_error=False, fill_value=None
            )
            
            X_target, Y_target = np.meshgrid(x_coords, y_coords, indexing='ij')
            points = np.stack([X_target.flatten(), Y_target.flatten()], axis=1)
            
            c1_interp[t_idx] += weight * interp_c1(points).reshape(50, 50)
            c2_interp[t_idx] += weight * interp_c2(points).reshape(50, 50)
    
    return {
        'params': {
            **solutions[0]['params'],
            'Ly': ly_target
        },
        'X': np.meshgrid(x_coords, y_coords, indexing='ij')[0],
        'Y': np.meshgrid(x_coords, y_coords, indexing='ij')[1],
        'c1_preds': c1_interp,
        'c2_preds': c2_interp,
        'times': times,
        'interpolated': True
    }

def get_solution(target_params, ly_target):
    """Main solution retrieval with caching"""
    current_params = (*target_params.values(), ly_target)
    
    # Return cached solution if parameters match
    if (st.session_state.last_params == current_params and 
        st.session_state.current_solution):
        return st.session_state.current_solution
    
    # Load relevant solutions
    solutions = load_relevant_solutions(target_params)
    
    if not solutions:
        st.error("No matching solutions found")
        return None
    
    # Find exact match or interpolate
    lys = np.array([s['params']['Ly'] for s in solutions])
    exact_match = np.isclose(lys, ly_target, atol=1e-4)
    
    if np.any(exact_match):
        solution = solutions[np.argmax(exact_match)]
        solution['interpolated'] = False
    else:
        solution = create_interpolated_solution(solutions, ly_target)
        st.info(f"Interpolated from {len(solutions)} solutions")
    
    # Update cache
    st.session_state.last_params = current_params
    st.session_state.current_solution = solution
    return solution
    
def plot_solution(solution, downsample):
    time_index = st.session_state.get('time_index', 0)
    
    # Extract original coordinates
    x_coords = solution['X'][:, 0]
    y_coords = solution['Y'][0, :]
    
    # Downsample with boundary inclusion
    ds = max(1, downsample)
    x_indices = np.concatenate([
        [0], np.arange(1, len(x_coords)-1, ds), [len(x_coords)-1]
    ])
    y_indices = np.concatenate([
        [0], np.arange(1, len(y_coords)-1, ds), [len(y_coords)-1]
    ])
    x_ds = x_coords[x_indices]
    y_ds = y_coords[y_indices]
    c1 = solution['c1_preds'][time_index][np.ix_(x_indices, y_indices)]
    c2 = solution['c2_preds'][time_index][np.ix_(x_indices, y_indices)]
    
    # Create meshgrid for plotting
    X, Y = np.meshgrid(x_ds, y_ds, indexing='ij')
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        f"Copper @ {solution['times'][time_index]:.1f}s",
        f"Nickel @ {solution['times'][time_index]:.1f}s"
    ))
    
    fig.add_trace(go.Contour(
        z=c1,  # No transposition
        x=x_ds,
        y=y_ds,
        colorscale='Viridis',
        colorbar=dict(title='Cu (mol/cm³)', x=0.45)
    ), row=1, col=1)
    
    fig.add_trace(go.Contour(
        z=c2,  # No transposition
        x=x_ds,
        y=y_ds,
        colorscale='Cividis',
        colorbar=dict(title='Ni (mol/cm³)', x=1.02)
    ), row=1, col=2)
    
    fig.update_layout(
        height=600,
        margin=dict(l=20, r=20),
        title=f"Domain: {solution['params']['Lx']}μm × {solution['params']['Ly']}μm"
    )
    fig.update_xaxes(title_text="x (μm)", range=[0, solution['params']['Lx']], row=1, col=1)
    fig.update_yaxes(title_text="y (μm)", range=[0, solution['params']['Ly']], row=1, col=1)
    fig.update_xaxes(title_text="x (μm)", range=[0, solution['params']['Lx']], row=1, col=2)
    fig.update_yaxes(title_text="y (μm)", range=[0, solution['params']['Ly']], row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    

def main():
    st.set_page_config(layout="wide", page_title="PINN Diffusion Visualizer")
    st.title("Attention Mechanism assisted PINN model for study of size effect on Ni and Cu Cross-Diffusion in Solder ")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Parameters")
        with st.form("params_form"):
            D11 = st.number_input("D11 (Cu self-diffusion)", 0.001, 1.0, 0.006, 0.0001, "%.6f")
            D12 = st.number_input("D12 (Cu cross-diffusion)", 0.0, 1.0, 0.00427, 0.0001, "%.6f")
            D21 = st.number_input("D21 (Ni cross-diffusion)", 0.0, 1.0, 0.003697, 0.0001, "%.6f")
            D22 = st.number_input("D22 (Ni self-diffusion)", 0.001, 1.0, 0.0054, 0.0001, "%.6f")
            Lx = st.number_input("Width (μm)", 1.0, 100.0, 60.0, 1.0)
            t_max = st.number_input("Time (s)", 1.0, 3600.0, 200.0, 10.0)
            ly_target = st.number_input("Height (μm)", 50.0, 100.0, 60.0, 0.1)
            st.form_submit_button("Update")
        
        # Performance controls
        with st.expander("Performance Settings"):
            downsample = st.slider("Detail Level", 1, 4, DOWNSAMPLE_FACTOR)
            if st.button("Clear Cache"):
                st.session_state.solution_cache.clear()
                st.session_state.current_solution = None
                st.rerun()
    
    # Main display
    target_params = {
        'D11': D11,
        'D12': D12,
        'D21': D21,
        'D22': D22,
        'Lx': Lx,
        't_max': t_max
    }
    
    solution = get_solution(target_params, ly_target)
    
    if solution:
        # Time slider
        time_index = st.slider(
            "Time Instance", 0, len(solution['times'])-1, 0,
            key="time_index"
        )
        
        # Display solution info
        st.subheader("Solution Details")
        col1, col2 = st.columns(2)
        col1.metric("Domain Size", f"{solution['params']['Lx']} × {solution['params']['Ly']} μm")
        col2.metric("Simulation Time", f"{solution['params']['t_max']} s")
        
        # Plot solution
        plot_solution(solution, downsample)

if __name__ == "__main__":
    main()
