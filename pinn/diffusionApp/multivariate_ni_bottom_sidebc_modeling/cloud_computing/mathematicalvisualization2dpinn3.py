import os
import pickle
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import cdist
import re
import plotly.express as px

# ------------------------------
# Global Settings
# ------------------------------
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "mathtext.fontset": "dejavusans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10
})

SOLUTION_DIR = os.path.join(os.path.dirname(__file__), "pinn_solutions")
os.makedirs(SOLUTION_DIR, exist_ok=True)

DIFFUSION_TYPES = ['crossdiffusion', 'cu_selfdiffusion', 'ni_selfdiffusion']

# ------------------------------
# Colormap Definitions
# ------------------------------

def get_plotly_colormaps():
    """Get organized categories of Plotly colormaps"""
    categories = {
        "Sequential (Perceptually Uniform)": [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo'
        ],
        "Sequential": [
            'blues', 'greens', 'reds', 'purples', 'oranges', 'greys',
            'ylorbr', 'ylgnbu', 'reds', 'hot', 'pink', 'deep',
            'dense', 'gray', 'burg', 'burgyl'
        ],
        "Diverging": [
            'rdbu', 'picnic', 'portland', 'rdgy', 'rdylbu', 'rdylgn',
            'spectral', 'tealrose', 'temps', 'tropic', 'balance', 'curl',
            'brbg', 'prgn', 'piyg', 'puor', 'rdbu_r', 'rdylbu_r'
        ],
        "Cyclical": [
            'twilight', 'icefire', 'phase', 'hsv'
        ],
        "Qualitative": [
            'plotly3', 'pastel', 'dark24', 'light24', 'vivid'
        ],
        "Earth & Terrain": [
            'earth', 'armyrose', 'fall', 'geyser', 'haline',
            'oxy', 'solar', 'speed', 'tempo', 'thermal', 'algae',
            'matter', 'deep', 'dense'
        ]
    }
    return categories

def get_matplotlib_colormaps():
    """Get organized categories of Matplotlib colormaps"""
    categories = {
        "Perceptually Uniform Sequential": [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis'
        ],
        "Sequential": [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'
        ],
        "Sequential (2)": [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper'
        ],
        "Diverging": [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'
        ],
        "Cyclical": [
            'twilight', 'twilight_shifted', 'hsv'
        ],
        "Qualitative": [
            'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
            'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c'
        ],
        "Miscellaneous": [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral'
        ]
    }
    return categories

# ------------------------------
# Utility Functions
# ------------------------------

@st.cache_data
def load_solutions(solution_dir):
    solutions, metadata, load_logs = [], [], []
    for fname in os.listdir(solution_dir):
        if not fname.endswith(".pkl"): 
            continue
        filepath = os.path.join(solution_dir, fname)
        try:
            with open(filepath, "rb") as f:
                sol = pickle.load(f)
            required = ['params', 'X', 'Y', 'c1_preds', 'c2_preds', 'times']
            if not all(key in sol for key in required):
                load_logs.append(f"{fname}: Missing keys {set(required)-set(sol.keys())}")
                continue

            # --- FIXED: parse filename correctly ---
            match = re.match(r"solution_(\w+)_ly_([\d.]+)_tmax_([\d.]+)\.pkl", fname)
            if not match:
                load_logs.append(f"{fname}: Invalid filename format")
                continue

            raw_type, ly_val, _ = match.groups()
            ly_val = float(ly_val)

            # Normalize diffusion type
            diff_type = raw_type.lower()
            type_map = {
                'crossdiffusion': 'crossdiffusion',
                'cu_selfdiffusion': 'cu_selfdiffusion',
                'ni_selfdiffusion': 'ni_selfdiffusion',
                'cross': 'crossdiffusion',
                'cu_self': 'cu_selfdiffusion',
                'ni_self': 'ni_selfdiffusion'
            }
            diff_type = type_map.get(diff_type, diff_type)
            if diff_type not in DIFFUSION_TYPES:
                load_logs.append(f"{fname}: Unknown diffusion type '{diff_type}'")
                continue
            # --- END FIX ---

            # Ensure predictions are 50x50
            c1_preds, c2_preds = sol['c1_preds'], sol['c2_preds']
            if c1_preds[0].shape != (50,50):
                if c1_preds[0].shape == (2500,):
                    c1_preds = [c.reshape(50,50) for c in c1_preds]
                    c2_preds = [c.reshape(50,50) for c in c2_preds]
                    sol['orientation_note'] = "Reshaped from flattened"
                else:
                    c1_preds = [c.T for c in c1_preds]
                    c2_preds = [c.T for c in c2_preds]
                    sol['orientation_note'] = f"Transposed from {c1_preds[0].shape}"
            else:
                sol['orientation_note'] = "Already (50,50)"
            sol.update({'c1_preds': c1_preds,'c2_preds':c2_preds,'diffusion_type':diff_type,'Ly_parsed':ly_val,'filename':fname})
            solutions.append(sol)
            metadata.append({'type':diff_type,'Ly':ly_val,'filename':fname,'shape':c1_preds[0].shape})
            load_logs.append(f"{fname}: ✓ Loaded [{diff_type}, Ly={ly_val:.1f}]")
        except Exception as e:
            load_logs.append(f"{fname}: ✗ Failed - {str(e)}")
    return solutions, metadata, load_logs

def compute_fluxes_and_grads(c1_preds, c2_preds, x_coords, y_coords, params):
    D11,D12,D21,D22 = params['D11'],params['D12'],params['D21'],params['D22']
    dx, dy = x_coords[1]-x_coords[0], y_coords[1]-y_coords[0]
    J1_preds,J2_preds,grad_c1_y,grad_c2_y = [],[],[],[]
    for c1,c2 in zip(c1_preds,c2_preds):
        grad_c1_x, grad_c1_y_i = np.gradient(c1, dx, axis=1), np.gradient(c1, dy, axis=0)
        grad_c2_x, grad_c2_y_i = np.gradient(c2, dx, axis=1), np.gradient(c2, dy, axis=0)
        J1_preds.append([-(D11*grad_c1_x + D12*grad_c2_x), -(D11*grad_c1_y_i + D12*grad_c2_y_i)])
        J2_preds.append([-(D21*grad_c1_x + D22*grad_c2_x), -(D21*grad_c1_y_i + D22*grad_c2_y_i)])
        grad_c1_y.append(grad_c1_y_i)
        grad_c2_y.append(grad_c2_y_i)
    return J1_preds,J2_preds,grad_c1_y,grad_c2_y

def detect_uphill(solution,time_index):
    J1_y = solution['J1_preds'][time_index][1]
    grad_c1_y = solution['grad_c1_y'][time_index]
    J2_y = solution['J2_preds'][time_index][1]
    grad_c2_y = solution['grad_c2_y'][time_index]
    return J1_y*grad_c1_y>0, J2_y*grad_c2_y>0

# ------------------------------
# Streamlit Plotting Functions
# ------------------------------

def get_plot_customization():
    st.sidebar.header("Plot Styling Options")
    color_cu = st.sidebar.color_picker("Cu Line Color", "#1f77b4")
    color_ni = st.sidebar.color_picker("Ni Line Color", "#ff7f0e")
    line_width = st.sidebar.slider("Line Width", 1, 10, 2)
    line_style = st.sidebar.selectbox("Line Style", ["solid","dashed","dotted"],0)
    fig_width = st.sidebar.slider("Figure Width", 6, 20, 14)
    fig_height = st.sidebar.slider("Figure Height", 4, 12, 6)
    font_size = st.sidebar.slider("Font Size", 8, 24, 14)
    tick_len = st.sidebar.slider("Tick Length", 2, 20, 6)
    tick_width = st.sidebar.slider("Tick Width", 1, 5, 1)
    legend_font_size = st.sidebar.slider("Legend Font Size", 6, 20, 12)
    grid_on = st.sidebar.checkbox("Show Grid", True)
    
    # New options for matplotlib styling
    axis_label_size = st.sidebar.slider("Axis Label Size", 8, 20, 12)
    tick_label_size = st.sidebar.slider("Tick Label Size", 6, 16, 10)
    spine_width = st.sidebar.slider("Axis Spine Width", 0.5, 3.0, 1.0)
    
    return (color_cu, color_ni, line_width, line_style, fig_width, fig_height, 
            font_size, tick_len, tick_width, legend_font_size, grid_on,
            axis_label_size, tick_label_size, spine_width)

def plot_flux_vs_gradient(solution, time_index, color_cu, color_ni, line_width, line_style,
                          fig_width, fig_height, font_size, tick_len, tick_width, 
                          legend_font_size, grid_on, axis_label_size, tick_label_size, spine_width):
    x_idx = solution['X'].shape[0]//2
    y_coords = solution['Y'][0,:]
    t_val = solution['times'][time_index]

    J1_y = solution['J1_preds'][time_index][1][:,x_idx]
    grad_c1_y = solution['grad_c1_y'][time_index][:,x_idx]
    J2_y = solution['J2_preds'][time_index][1][:,x_idx]
    grad_c2_y = solution['grad_c2_y'][time_index][:,x_idx]

    fig, axes = plt.subplots(1,2,figsize=(fig_width,fig_height))
    
    # Apply styling to both axes
    for ax in axes:
        # Set spine width
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)
        # Set tick parameters
        ax.tick_params(axis='both', which='major', length=tick_len, width=tick_width, 
                       labelsize=tick_label_size)
        ax.tick_params(axis='both', which='minor', length=tick_len/2, width=tick_width/2)
        if grid_on: 
            ax.grid(True, alpha=0.3)
    
    # Cu
    axes[0].plot(y_coords, -grad_c1_y, color=color_cu, lw=line_width, linestyle=line_style,label=r"$-\nabla C_{Cu}$")
    axes[0].plot(y_coords, J1_y, color=color_cu, lw=line_width, linestyle='--',label=r"$J_{Cu}$")
    axes[0].fill_between(y_coords,0,J1_y,where=(J1_y*-grad_c1_y>0),color='red',alpha=0.3,label='Uphill')
    axes[0].set_xlabel("y (μm)", fontsize=axis_label_size)
    axes[0].set_ylabel("Flux / -Gradient", fontsize=axis_label_size)
    axes[0].set_title(f"Cu Flux vs Gradient @ t={t_val:.1f}s", fontsize=font_size+2)
    axes[0].legend(fontsize=legend_font_size)

    # Ni
    axes[1].plot(y_coords, -grad_c2_y, color=color_ni, lw=line_width, linestyle=line_style,label=r"$-\nabla C_{Ni}$")
    axes[1].plot(y_coords, J2_y, color=color_ni, lw=line_width, linestyle='--',label=r"$J_{Ni}$")
    axes[1].fill_between(y_coords,0,J2_y,where=(J2_y*-grad_c2_y>0),color='red',alpha=0.3,label='Uphill')
    axes[1].set_xlabel("y (μm)", fontsize=axis_label_size)
    axes[1].set_ylabel("Flux / -Gradient", fontsize=axis_label_size)
    axes[1].set_title(f"Ni Flux vs Gradient @ t={t_val:.1f}s", fontsize=font_size+2)
    axes[1].legend(fontsize=legend_font_size)

    plt.suptitle(f"Flux vs Gradient: {solution['diffusion_type'].replace('_',' ')}", fontsize=font_size+4)
    plt.tight_layout(rect=[0,0,1,0.95])
    st.pyplot(fig)
    plt.close()

def get_colormap_selector(plot_type):
    """Create a colormap selector based on plot type"""
    if plot_type == "Plotly":
        plotly_categories = get_plotly_colormaps()
        
        st.sidebar.subheader("Plotly Colormap Selection")
        selected_category = st.sidebar.selectbox(
            "Colormap Category",
            list(plotly_categories.keys()),
            key="plotly_category"
        )
        
        available_cmaps = plotly_categories[selected_category]
        default_idx = available_cmaps.index('rdbu') if 'rdbu' in available_cmaps else 0
        
        selected_cmap = st.sidebar.selectbox(
            "Plotly Colorscale",
            available_cmaps,
            index=default_idx,
            key="plotly_cmap"
        )
        
        # Add option to reverse colormap
        reverse = st.sidebar.checkbox("Reverse Colormap", key="plotly_reverse")
        if reverse:
            selected_cmap = selected_cmap + '_r'
            
    else:  # Matplotlib
        mpl_categories = get_matplotlib_colormaps()
        
        st.sidebar.subheader("Matplotlib Colormap Selection")
        selected_category = st.sidebar.selectbox(
            "Colormap Category",
            list(mpl_categories.keys()),
            key="mpl_category"
        )
        
        available_cmaps = mpl_categories[selected_category]
        default_idx = available_cmaps.index('viridis') if 'viridis' in available_cmaps else 0
        
        selected_cmap = st.sidebar.selectbox(
            "Matplotlib Colormap",
            available_cmaps,
            index=default_idx,
            key="mpl_cmap"
        )
        
        # Add option to reverse colormap
        reverse = st.sidebar.checkbox("Reverse Colormap", key="mpl_reverse")
        if reverse:
            selected_cmap = selected_cmap + '_r'
    
    return selected_cmap

def plot_uphill_regions(solution, time_index, downsample=2, colorscale='RdBu', 
                        fig_width=10, fig_height=5, font_size=14):
    x_coords = solution['X'][:,0]  # Shape: (50,)
    y_coords = solution['Y'][0,:]  # Shape: (50,)
    t_val = solution['times'][time_index]
    diff_type = solution['diffusion_type']

    ds = max(1, downsample)
    x_idx = np.arange(0, len(x_coords), ds)
    y_idx = np.arange(0, len(y_coords), ds)

    uphill_cu, uphill_ni = detect_uphill(solution, time_index)
    
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=["Cu Uphill Magnitude", "Ni Uphill Magnitude"])
    
    for i, uphill in enumerate([uphill_cu, uphill_ni]):
        J = solution['J1_preds'][time_index][1] if i==0 else solution['J2_preds'][time_index][1]
        
        # Downsample the data
        x_plot = x_coords[x_idx]
        y_plot = y_coords[y_idx]
        z_plot = np.abs(J[np.ix_(y_idx, x_idx)]) * uphill[np.ix_(y_idx, x_idx)]
        
        fig.add_trace(go.Heatmap(
            x=x_plot,  # Correct x coordinates
            y=y_plot,  # Correct y coordinates  
            z=z_plot,
            colorscale=colorscale,
            colorbar=dict(title="|J|"),
            zsmooth='best'
        ), row=1, col=i+1)
        
        # Update axes labels
        fig.update_xaxes(title_text="x (μm)", row=1, col=i+1)
        fig.update_yaxes(title_text="y (μm)", row=1, col=i+1)
    
    fig.update_layout(
        height=int(fig_height*100), 
        width=int(fig_width*100),
        title=f"Uphill Diffusion Magnitude: {diff_type.replace('_',' ')} @ t={t_val:.1f}s",
        template='plotly_white', 
        font=dict(size=font_size)
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_uphill_regions_matplotlib(solution, time_index, downsample=2, colorscale='viridis',
                                  fig_width=10, fig_height=5, font_size=14,
                                  axis_label_size=12, tick_label_size=10, spine_width=1.0):
    x_coords = solution['X'][:,0]
    y_coords = solution['Y'][0,:]
    t_val = solution['times'][time_index]
    diff_type = solution['diffusion_type']

    ds = max(1, downsample)
    x_idx = np.arange(0, len(x_coords), ds)
    y_idx = np.arange(0, len(y_coords), ds)

    uphill_cu, uphill_ni = detect_uphill(solution, time_index)
    
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    
    for i, (uphill, ax) in enumerate(zip([uphill_cu, uphill_ni], axes)):
        J = solution['J1_preds'][time_index][1] if i==0 else solution['J2_preds'][time_index][1]
        
        # Downsample the data
        x_plot = x_coords[x_idx]
        y_plot = y_coords[y_idx]
        z_plot = np.abs(J[np.ix_(y_idx, x_idx)]) * uphill[np.ix_(y_idx, x_idx)]
        
        # Create heatmap
        im = ax.imshow(z_plot, 
                      extent=[x_plot.min(), x_plot.max(), y_plot.min(), y_plot.max()],
                      origin='lower', 
                      aspect='auto',
                      cmap=colorscale)
        
        # Set labels and title
        ax.set_xlabel("x (μm)", fontsize=axis_label_size)
        ax.set_ylabel("y (μm)", fontsize=axis_label_size)
        component = "Cu" if i == 0 else "Ni"
        ax.set_title(f"{component} Uphill Magnitude", fontsize=font_size)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("|J|", fontsize=axis_label_size)
        cbar.ax.tick_params(labelsize=tick_label_size)
        
        # Style the axes
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)
        ax.tick_params(axis='both', which='major', labelsize=tick_label_size)
    
    plt.suptitle(f"Uphill Diffusion Magnitude: {diff_type.replace('_',' ')} @ t={t_val:.1f}s", 
                 fontsize=font_size+2)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    st.pyplot(fig)
    plt.close()

# ------------------------------
# Main App
# ------------------------------
def main():
    st.title("Theoretical Assessment of Diffusion Solutions")
    solutions, metadata, load_logs = load_solutions(SOLUTION_DIR)

    with st.expander("🔍 Debug: File Loading Results"):
        st.subheader("Detailed Load Logs")
        for log in load_logs:
            if "✓" in log: st.success(log)
            elif "Skipped" in log: st.warning(log)
            elif "✗" in log: st.error(log)
            else: st.info(log)

    if not solutions: 
        st.error("No valid solution files found!")
        return

    # Sidebar Parameters
    st.sidebar.header("Simulation Parameters")
    available_types = sorted(set(s['diffusion_type'] for s in solutions))
    diff_type = st.sidebar.selectbox("Diffusion Type", available_types, format_func=lambda x: x.replace('_',' ').title())
    available_lys = sorted(set(s['Ly_parsed'] for s in solutions if s['diffusion_type']==diff_type))
    ly_target = st.sidebar.select_slider("Ly (μm)", options=available_lys, value=available_lys[0])
    time_index = st.sidebar.slider("Time Index", 0, 49, 49)
    downsample = st.sidebar.slider("Downsample", 1, 5, 2)
    
    # Plot type selection
    plot_type = st.sidebar.radio("Heatmap Type", ["Plotly", "Matplotlib"], index=0)
    
    # Get colormap based on plot type
    selected_colormap = get_colormap_selector(plot_type)
    
    # Get other styling options
    (color_cu, color_ni, line_width, line_style, fig_width, fig_height, 
     font_size, tick_len, tick_width, legend_font_size, grid_on,
     axis_label_size, tick_label_size, spine_width) = get_plot_customization()

    # Select solution
    solution = next((s for s in solutions if s['diffusion_type']==diff_type and abs(s['Ly_parsed']-ly_target)<1e-4), None)
    if solution is None:
        st.error(f"No solution for {diff_type} with Ly={ly_target}")
        return

    # Compute fluxes and gradients
    J1,J2,grad_c1,grad_c2 = compute_fluxes_and_grads(solution['c1_preds'], solution['c2_preds'],
                                                     solution['X'][:,0], solution['Y'][0,:], solution['params'])
    solution.update({'J1_preds':J1,'J2_preds':J2,'grad_c1_y':grad_c1,'grad_c2_y':grad_c2})

    # Display selected colormap info
    st.sidebar.markdown("---")
    st.sidebar.info(f"Selected: **{selected_colormap}**")

    # Uphill
    st.subheader("Uphill Diffusion Detection")
    if plot_type == "Plotly":
        plot_uphill_regions(solution, time_index, downsample, selected_colormap, fig_width, fig_height, font_size)
    else:
        plot_uphill_regions_matplotlib(solution, time_index, downsample, selected_colormap, 
                                      fig_width, fig_height, font_size,
                                      axis_label_size, tick_label_size, spine_width)

    # Flux vs Gradient
    st.subheader("Flux vs Gradient Comparison")
    plot_flux_vs_gradient(solution, time_index, color_cu, color_ni, line_width, line_style,
                          fig_width, fig_height, font_size, tick_len, tick_width, 
                          legend_font_size, grid_on, axis_label_size, tick_label_size, spine_width)

    # Solution Info
    with st.expander("Solution Information"):
        st.write(f"**Diffusion type:** {solution['diffusion_type']}")
        st.write(f"**Ly:** {solution['params']['Ly']} μm")
        st.write(f"**Lx:** {solution['params']['Lx']} μm") 
        st.write(f"**Time range:** {solution['times'][0]:.1f}s to {solution['times'][-1]:.1f}s")
        st.write(f"**Array shape:** {solution['c1_preds'][0].shape}")
        st.write(f"**Orientation:** {solution.get('orientation_note', 'Not specified')}")

if __name__=="__main__":
    main()
