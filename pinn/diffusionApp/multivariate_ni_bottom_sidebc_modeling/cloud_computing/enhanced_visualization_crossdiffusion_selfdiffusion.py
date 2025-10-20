# app.py
import os
import pickle
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import io
import zipfile
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
import re

# --- Page config & Streamlit helper UI ---
st.set_page_config(page_title="Cu/Ni Cross-Diffusion Viewer", layout="wide")

# Directory containing .pkl solution files
SOLUTION_DIR = os.path.join(os.path.dirname(__file__), "pinn_solutions")
os.makedirs(SOLUTION_DIR, exist_ok=True)

st.title("🧠 Cu–Ni Cross-Diffusion PINN Visualization")
st.caption("Automatically loads all `.pkl` solutions from the `pinn_solutions/` folder")

num_files = len([f for f in os.listdir(SOLUTION_DIR) if f.endswith('.pkl')])
st.info(f"📁 **Solution directory:** `{SOLUTION_DIR}` — {num_files} file(s) found")

if st.button("🔄 Reload Solutions"):
    st.cache_data.clear()
    st.experimental_rerun()

# Diffusion types and Plotly colorscales
DIFFUSION_TYPES = ['crossdiffusion', 'cu_selfdiffusion', 'ni_selfdiffusion']
COLORSCALES = ['viridis','magma','cividis','plasma','jet','inferno','blues','reds','greens']

# ---------------------------- Utility Functions ----------------------------
@st.cache_data
def load_solutions(solution_dir):
    solutions, metadata, load_logs = [], [], []
    for fname in os.listdir(solution_dir):
        if not fname.endswith(".pkl"):
            load_logs.append(f"{fname}: Skipped - not a .pkl file.")
            continue
        try:
            with open(os.path.join(solution_dir, fname), "rb") as f:
                sol = pickle.load(f)
            # Check required keys
            required = ['params', 'X', 'Y', 'c1_preds', 'c2_preds', 'times']
            if not all(k in sol for k in required):
                load_logs.append(f"{fname}: Failed - missing keys {set(required) - set(sol.keys())}")
                continue
            # Parse diffusion type and Ly
            match = re.match(r"solution_(\w+)_ly_([\d.]+)_tmax_([\d.]+)\.pkl", fname)
            if not match:
                load_logs.append(f"{fname}: Failed - invalid filename format")
                continue
            diff_type, ly_val, t_max = match.groups()
            ly_val, t_max = float(ly_val), float(t_max)
            if diff_type not in DIFFUSION_TYPES:
                load_logs.append(f"{fname}: Failed - unknown diffusion type '{diff_type}'")
                continue
            # Ensure orientation
            c1_preds = sol['c1_preds']
            c2_preds = sol['c2_preds']
            if c1_preds[0].shape != (50,50):
                c1_preds = [c.T for c in c1_preds]
                c2_preds = [c.T for c in c2_preds]
                sol['orientation_note'] = "Transposed to rows=y, cols=x"
            else:
                sol['orientation_note'] = "Rows=y, cols=x"
            sol.update({'c1_preds':c1_preds, 'c2_preds':c2_preds, 'diffusion_type':diff_type, 'Ly_parsed':ly_val})
            solutions.append(sol)
            metadata.append({'type':diff_type, 'Ly':ly_val, 'filename':fname})
            load_logs.append(f"{fname}: Loaded [{diff_type}, Ly={ly_val:.1f}, t_max={t_max:.1f}]")
        except Exception as e:
            load_logs.append(f"{fname}: Load failed - {str(e)}")
    return solutions, metadata, load_logs

@st.cache_data
def compute_fluxes(c1_preds, c2_preds, x_coords, y_coords, params):
    D11, D12, D21, D22 = params['D11'], params['D12'], params['D21'], params['D22']
    dx, dy = x_coords[1]-x_coords[0], y_coords[1]-y_coords[0]
    J1_preds, J2_preds = [], []
    for c1, c2 in zip(c1_preds, c2_preds):
        grad_c1_x = np.gradient(c1, dx, axis=1)
        grad_c1_y = np.gradient(c1, dy, axis=0)
        grad_c2_x = np.gradient(c2, dx, axis=1)
        grad_c2_y = np.gradient(c2, dy, axis=0)
        J1_preds.append([-(D11*grad_c1_x + D12*grad_c2_x), -(D11*grad_c1_y + D12*grad_c2_y)])
        J2_preds.append([-(D21*grad_c1_x + D22*grad_c2_x), -(D21*grad_c1_y + D22*grad_c2_y)])
    return J1_preds, J2_preds

@st.cache_data
def get_interpolation_weights(lys, ly_target, sigma=2.5):
    lys = np.array(lys).reshape(-1,1)
    dist = cdist(np.array([[ly_target]]), lys).flatten()
    weights = np.exp(-(dist**2)/(2*sigma**2))
    return weights/weights.sum()

@st.cache_data
def attention_weighted_interpolation(solutions, lys, ly_target, diff_type, sigma=2.5):
    matching = [s for s in solutions if s['diffusion_type']==diff_type]
    if not matching: return None
    lys = np.array([s['Ly_parsed'] for s in matching])
    weights = get_interpolation_weights(lys, ly_target, sigma)
    Lx = matching[0]['params']['Lx']; t_max = matching[0]['params']['t_max']
    x_coords = np.linspace(0,Lx,50); y_coords = np.linspace(0,ly_target,50); times=np.linspace(0,t_max,50)
    c1_interp, c2_interp = np.zeros((50,50,50)), np.zeros((50,50,50))
    c1_interp = np.zeros((len(times),50,50))
    c2_interp = np.zeros((len(times),50,50))
    for sol,w in zip(matching, weights):
        X_sol = sol['X'][:,0]
        Y_sol = sol['Y'][0,:]*(ly_target/sol['params']['Ly'])
        for t_idx,t in enumerate(times):
            interp_c1 = RegularGridInterpolator((Y_sol,X_sol),sol['c1_preds'][t_idx],bounds_error=False,fill_value=0)
            interp_c2 = RegularGridInterpolator((Y_sol,X_sol),sol['c2_preds'][t_idx],bounds_error=False,fill_value=0)
            X_target,Y_target = np.meshgrid(x_coords,y_coords,indexing='ij')
            points = np.column_stack([Y_target.ravel(),X_target.ravel()])
            c1_interp[t_idx] += w*interp_c1(points).reshape(50,50).T
            c2_interp[t_idx] += w*interp_c2(points).reshape(50,50).T
    param_set = matching[0]['params'].copy(); param_set['Ly']=ly_target
    J1_preds,J2_preds = compute_fluxes(c1_interp,c2_interp,x_coords,y_coords,param_set)
    X,Y = np.meshgrid(x_coords,y_coords,indexing='ij')
    return {'params':param_set,'X':X,'Y':Y,'c1_preds':list(c1_interp),'c2_preds':list(c2_interp),
            'J1_preds':J1_preds,'J2_preds':J2_preds,'times':times,'diffusion_type':diff_type,
            'interpolated':True,'used_lys':lys.tolist(),'attention_weights':weights.tolist(),
            'orientation_note':'rows=y, cols=x'}

@st.cache_data
def load_and_interpolate_solution(solutions,diff_type,ly_target,tol=1e-4):
    exact = [s for s in solutions if s['diffusion_type']==diff_type and abs(s['Ly_parsed']-ly_target)<tol]
    if exact:
        s = exact[0]; s['interpolated']=False
        if 'J1_preds' not in s: s['J1_preds'],s['J2_preds']=compute_fluxes(s['c1_preds'],s['c2_preds'],s['X'][:,0],s['Y'][0,:],s['params'])
        return s
    return attention_weighted_interpolation(solutions,[s['Ly_parsed'] for s in solutions],ly_target,diff_type)

# ---------------------------- Main UI ----------------------------
def main():
    st.sidebar.header("Simulation Parameters")
    solutions, metadata, load_logs = load_solutions(SOLUTION_DIR)
    if load_logs:
        st.subheader("Solution Load Log")
        st.selectbox("View load status for solutions", load_logs, index=0)

    if not solutions:
        st.error("No valid solution files found in pinn_solutions directory.")
        return

    diff_type = st.sidebar.selectbox("Select Diffusion Type", DIFFUSION_TYPES, format_func=lambda x:x.replace('_',' ').title())
    available_lys = sorted({s['Ly_parsed'] for s in solutions if s['diffusion_type']==diff_type})
    if len(available_lys)<2: st.sidebar.error(f"Need at least 2 Ly for {diff_type}"); return
    ly_values = st.sidebar.multiselect("Select Two Ly Values",available_lys,default=available_lys[:2],format_func=lambda x:f"{x:.1f}",max_selections=2)
    time_index = st.sidebar.slider("Select Time",0,len(solutions[0]['times'])-1,len(solutions[0]['times'])-1)
    downsample = st.sidebar.slider("Detail Level",1,5,2)
    cu_colormap = st.sidebar.selectbox("Cu Colormap",COLORSCALES,index=COLORSCALES.index('viridis'))
    ni_colormap = st.sidebar.selectbox("Ni Colormap",COLORSCALES,index=COLORSCALES.index('magma'))

    tab1, tab2, tab3, tab4 = st.tabs(["Concentration","Flux Comparison","Central Line","Center Point"])
    # --- Tab 1: Concentration ---
    with tab1:
        st.subheader("Concentration Fields")
        for ly in ly_values:
            solution = load_and_interpolate_solution(solutions,diff_type,ly)
            if solution: plot_solution(solution,time_index,downsample,title_suffix=f"[{diff_type.replace('_',' ')}, Ly={ly:.1f}]",cu_colormap=cu_colormap,ni_colormap=ni_colormap)
            else: st.error(f"No solution for Ly={ly:.1f}")

    # --- Tab 2: Flux Comparison ---
    with tab2:
        st.subheader("Flux Fields Comparison")
        plot_flux_comparison(solutions,diff_type,ly_values,time_index,downsample)

    # --- Tab 3: Central Line ---
    with tab3:
        st.subheader("Central Line Profiles Comparison")
        plot_line_comparison(solutions,diff_type,ly_values,time_index)

    # --- Tab 4: Center Point ---
    with tab4:
        st.subheader("Center Point Comparison")
        center_concentrations = compute_center_concentrations(solutions,diff_type,ly_values)
        if center_concentrations: plot_center_concentrations(center_concentrations,diff_type)
        else: st.error("Could not compute center concentrations")

    # --- Download ---
    st.subheader("Download Data")
    for ly in ly_values:
        solution = load_and_interpolate_solution(solutions,diff_type,ly)
        if not solution: continue
        col1,col2 = st.columns(2)
        with col1:
            data_bytes, filename = download_data(solution,time_index,all_times=False)
            st.download_button(f"Download CSV (t={solution['times'][time_index]:.1f}s, Ly={ly:.1f})",data_bytes,filename,"text/csv")
        with col2:
            data_bytes, filename = download_data(solution,time_index,all_times=True)
            st.download_button(f"Download ZIP (All Times, Ly={ly:.1f})",data_bytes,filename,"application/zip")

if __name__=="__main__":
    main()
