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

# ------------------------------
# Global Settings
# ------------------------------
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "mathtext.fontset": "dejavusans"
})

# Directory containing .pkl solution files (ensure exists)
SOLUTION_DIR = os.path.join(os.path.dirname(__file__), "pinn_solutions")
os.makedirs(SOLUTION_DIR, exist_ok=True)

DIFFUSION_TYPES = ['crossdiffusion', 'cu_selfdiffusion', 'ni_selfdiffusion']

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
    color_cu = st.sidebar.color_picker("Cu Curve Color", "#1f77b4")
    color_ni = st.sidebar.color_picker("Ni Curve Color", "#ff7f0e")
    linewidth = st.sidebar.slider("Line Width",1,5,2)
    linestyle = st.sidebar.selectbox("Line Style",["solid","dashed","dotted"],0)
    font_size = st.sidebar.slider("Font Size",8,24,14)
    dpi = st.sidebar.slider("DPI (Matplotlib)",100,600,300)
    colorscale = st.sidebar.selectbox("Heatmap Colorscale",["RdBu","Viridis","Cividis"],0)
    return color_cu,color_ni,linewidth,linestyle,font_size,dpi,colorscale

def plot_flux_vs_gradient(solution,time_index,color_cu,color_ni,linewidth,linestyle,font_size,dpi):
    x_idx = solution['X'].shape[0]//2
    y_coords = solution['Y'][0,:]
    t_val = solution['times'][time_index]

    J1_y = solution['J1_preds'][time_index][1][:,x_idx]
    grad_c1_y = solution['grad_c1_y'][time_index][:,x_idx]
    J2_y = solution['J2_preds'][time_index][1][:,x_idx]
    grad_c2_y = solution['grad_c2_y'][time_index][:,x_idx]

    fig, axes = plt.subplots(1,2,figsize=(14,6),dpi=dpi)
    plt.rcParams.update({"font.size": font_size})
    
    # Cu
    axes[0].plot(y_coords, -grad_c1_y, color=color_cu,lw=linewidth,linestyle=linestyle,label=r"$-\nabla C_{Cu}$")
    axes[0].plot(y_coords, J1_y, color=color_cu,lw=linewidth,linestyle='--',label=r"$J_{Cu}$")
    axes[0].fill_between(y_coords,0,J1_y,where=(J1_y*-grad_c1_y>0),color='red',alpha=0.3,label='Uphill')
    axes[0].set_xlabel("y (μm)")
    axes[0].set_ylabel("Flux / -Gradient")
    axes[0].set_title(f"Cu Flux vs Gradient @ t={t_val:.1f}s")
    axes[0].legend()
    axes[0].grid(True)

    # Ni
    axes[1].plot(y_coords, -grad_c2_y, color=color_ni,lw=linewidth,linestyle=linestyle,label=r"$-\nabla C_{Ni}$")
    axes[1].plot(y_coords, J2_y, color=color_ni,lw=linewidth,linestyle='--',label=r"$J_{Ni}$")
    axes[1].fill_between(y_coords,0,J2_y,where=(J2_y*-grad_c2_y>0),color='red',alpha=0.3,label='Uphill')
    axes[1].set_xlabel("y (μm)")
    axes[1].set_ylabel("Flux / -Gradient")
    axes[1].set_title(f"Ni Flux vs Gradient @ t={t_val:.1f}s")
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle(f"Flux vs Gradient: {solution['diffusion_type'].replace('_',' ')}",fontsize=font_size+4)
    plt.tight_layout(rect=[0,0,1,0.95])
    st.pyplot(fig)
    plt.close()

def plot_uphill_regions(solution,time_index,downsample=2,colorscale='RdBu'):
    x_coords = solution['X'][:,0]
    y_coords = solution['Y'][0,:]
    t_val = solution['times'][time_index]
    diff_type = solution['diffusion_type']

    ds = max(1,downsample)
    x_idx = np.arange(0,len(x_coords),ds)
    y_idx = np.arange(0,len(y_coords),ds)

    uphill_cu, uphill_ni = detect_uphill(solution,time_index)
    
    fig = make_subplots(rows=1,cols=2,subplot_titles=["Cu Uphill Magnitude","Ni Uphill Magnitude"])
    for i,uphill in enumerate([uphill_cu,uphill_ni]):
        z = np.zeros_like(uphill,dtype=float)
        J = solution['J1_preds'][time_index][1] if i==0 else solution['J2_preds'][time_index][1]
        z[np.ix_(y_idx,x_idx)] = np.abs(J[np.ix_(y_idx,x_idx)]*uphill[np.ix_(y_idx,x_idx)])
        fig.add_trace(go.Heatmap(
            x=x_coords[x_idx],y=y_coords[y_idx],z=z,
            colorscale=colorscale,colorbar=dict(title="|J|"),zsmooth='best'
        ),row=1,col=i+1)
    fig.update_layout(height=500,width=1000,title=f"Uphill Diffusion Magnitude: {diff_type.replace('_',' ')} @ t={t_val:.1f}s",template='plotly_white')
    st.plotly_chart(fig,use_container_width=True)

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
    diff_type = st.sidebar.selectbox("Diffusion Type",available_types,format_func=lambda x:x.replace('_',' ').title())
    available_lys = sorted(set(s['Ly_parsed'] for s in solutions if s['diffusion_type']==diff_type))
    ly_target = st.sidebar.select_slider("Ly (μm)", options=available_lys,value=available_lys[0])
    time_index = st.sidebar.slider("Time Index",0,49,49)
    downsample = st.sidebar.slider("Downsample",1,5,2)
    color_cu,color_ni,linewidth,linestyle,font_size,dpi,colorscale = get_plot_customization()

    # Select solution
    solution = next((s for s in solutions if s['diffusion_type']==diff_type and abs(s['Ly_parsed']-ly_target)<1e-4), None)
    if solution:
        # Compute fluxes if not already
        if 'J1_preds' not in solution:
            J1,J2,grad1,grad2 = compute_fluxes_and_grads(solution['c1_preds'],solution['c2_preds'],
                                                         solution['X'][:,0],solution['Y'][0,:],
                                                         solution['params'])
            solution.update({'J1_preds':J1,'J2_preds':J2,'grad_c1_y':grad1,'grad_c2_y':grad2})

        st.subheader("Uphill Diffusion Detection")
        st.markdown("Regions where $J_y \\cdot \\nabla_y C > 0$ indicate uphill diffusion")
        plot_uphill_regions(solution,time_index,downsample,colorscale)

        st.subheader("Flux vs Gradient Comparison")
        st.markdown("Central line comparison of $J_y$ vs $-\\nabla_y C$")
        plot_flux_vs_gradient(solution,time_index,color_cu,color_ni,linewidth,linestyle,font_size,dpi)

        with st.expander("Solution Information"):
            st.write(f"**Diffusion type:** {solution['diffusion_type']}")
            st.write(f"**Ly:** {solution['params']['Ly']} μm")
            st.write(f"**Lx:** {solution['params']['Lx']} μm")
            st.write(f"**Time range:** {solution['times'][0]:.1f}s - {solution['times'][-1]:.1f}s")
            st.write(f"**Array shape:** {solution['c1_preds'][0].shape}")
            st.write(f"**Orientation:** {solution.get('orientation_note','Not specified')}")
    else:
        st.error(f"No solution available for {diff_type} with Ly={ly_target:.1f}")

if __name__=="__main__":
    main()
