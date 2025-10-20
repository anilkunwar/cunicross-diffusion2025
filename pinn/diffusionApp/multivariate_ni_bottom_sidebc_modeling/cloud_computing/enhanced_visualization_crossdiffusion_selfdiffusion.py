# app_publication_complete.py
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

# --- Page config ---
st.set_page_config(
    page_title="Cu/Ni Cross-Diffusion Publication Visualizer", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Directory setup
SOLUTION_DIR = os.path.join(os.path.dirname(__file__), "pinn_solutions")
os.makedirs(SOLUTION_DIR, exist_ok=True)

st.title("🎨 **Cu–Ni Cross-Diffusion Publication Visualizer**")
st.caption("🔬 Publication-quality • **50μm vs 90μm VISIBLY DISTINCT** • Live styling")

num_files = len([f for f in os.listdir(SOLUTION_DIR) if f.endswith('.pkl')])
st.info(f"📁 **{num_files} solution file(s)** loaded from `{SOLUTION_DIR}`")

if st.button("🔄 **Reload Solutions**", type="primary"):
    st.cache_data.clear()
    st.rerun()

# --- COMPLETE PUBLICATION CONTROLS ---
st.sidebar.markdown("## 🎨 **Publication Controls**")

# LAYOUT & FONTS
st.sidebar.markdown("### 📐 **Layout & Fonts**")
font_size = st.sidebar.slider("**Font Size**", 8, 28, 16, 1)
title_size = st.sidebar.slider("**Title Size**", 12, 32, 20, 1)
tick_size = st.sidebar.slider("**Tick Size**", 6, 20, 12, 1)
label_interval = st.sidebar.slider("**Tick Interval**", 10, 50, 20, 5)

# LINE STYLING
st.sidebar.markdown("### 📈 **Line Styling**")
line_width = st.sidebar.slider("**Line Width**", 0.5, 8.0, 2.5, 0.1)
line_alpha = st.sidebar.slider("**Line Opacity**", 0.3, 1.0, 0.9, 0.05)
line_styles = ["solid", "dashed", "dotted", "dashdot"]
line_style_50 = st.sidebar.selectbox("**Ly=50μm Style**", line_styles, index=0)
line_style_90 = st.sidebar.selectbox("**Ly=90μm Style**", line_styles, index=1)

# COLORS - DISTINCT FOR 50μm vs 90μm
st.sidebar.markdown("### 🌈 **Domain Colors**")
ly50_color = st.sidebar.color_picker("**Ly=50μm** (Warm)", "#FF4757")  # Red
ly90_color = st.sidebar.color_picker("**Ly=90μm** (Cool)", "#1E90FF")  # Blue

# DOMAIN ENHANCEMENTS
st.sidebar.markdown("### 📏 **Domain Visibility**")
show_domain_labels = st.sidebar.checkbox("**Show Domain Labels**", value=True)
domain_label_size = st.sidebar.slider("**Label Size**", 10, 24, 16, 1)
border_thickness = st.sidebar.slider("**Border Thickness**", 2, 8, 4, 1)

# EXPORT
st.sidebar.markdown("### 💾 **Export Quality**")
export_dpi = st.sidebar.slider("**Figure DPI**", 150, 1200, 600, 50)
transparent_bg = st.sidebar.checkbox("**Transparent BG**", value=False)

# Diffusion types
DIFFUSION_TYPES = ['crossdiffusion', 'cu_selfdiffusion', 'ni_selfdiffusion']

# DOMAIN DIFFERENTIATION - CRITICAL FOR VISIBILITY
DOMAIN_STYLES = {
    50.0: {'colormap': 'plasma', 'border': '#FF4757', 'label': '50×100 μm²', 'color': '#FF4757'},
    90.0: {'colormap': 'viridis', 'border': '#1E90FF', 'label': '90×100 μm²', 'color': '#1E90FF'}
}

@st.cache_data
def load_solutions(solution_dir):
    """Load and validate all .pkl solution files."""
    solutions = []
    load_logs = []
    metadata = []

    for fname in os.listdir(solution_dir):
        if not fname.endswith(".pkl"):
            continue

        filepath = os.path.join(solution_dir, fname)
        try:
            with open(filepath, "rb") as f:
                sol = pickle.load(f)

            required = ['params', 'X', 'Y', 'c1_preds', 'c2_preds', 'times']
            if not all(key in sol for key in required):
                load_logs.append(f"❌ {fname}: Missing keys")
                continue

            match = re.match(r"solution_(\w+)_ly_([\d.]+)_tmax_([\d.]+)\.pkl", fname)
            if not match:
                load_logs.append(f"❌ {fname}: Bad filename")
                continue

            diff_type, ly_val, t_max = match.groups()
            ly_val = float(ly_val)
            t_max = float(t_max)

            if diff_type not in DIFFUSION_TYPES:
                load_logs.append(f"❌ {fname}: Unknown type")
                continue

            # Fix orientation
            c1_preds = sol['c1_preds']
            c2_preds = sol['c2_preds']
            if c1_preds[0].shape == (50, 50):
                sol['orientation_note'] = "rows=y, cols=x"
            else:
                c1_preds = [c.T for c in c1_preds]
                c2_preds = [c.T for c in c2_preds]
                sol['orientation_note'] = "Transposed"

            sol['c1_preds'] = c1_preds
            sol['c2_preds'] = c2_preds
            sol['diffusion_type'] = diff_type
            sol['Ly_parsed'] = ly_val

            solutions.append(sol)
            metadata.append({'type': diff_type, 'Ly': ly_val, 'filename': fname})
            load_logs.append(f"✅ {fname}: Ly={ly_val:.0f}μm")

        except Exception as e:
            load_logs.append(f"❌ {fname}: {str(e)}")

    return solutions, metadata, load_logs

def compute_fluxes(c1_preds, c2_preds, x_coords, y_coords, params):
    """Compute J1, J2 from concentrations."""
    D11, D12, D21, D22 = params['D11'], params['D12'], params['D21'], params['D22']
    dx, dy = x_coords[1] - x_coords[0], y_coords[1] - y_coords[0]
    
    J1_preds, J2_preds = [], []
    for c1, c2 in zip(c1_preds, c2_preds):
        grad_c1_x, grad_c1_y = np.gradient(c1, dx, axis=1), np.gradient(c1, dy, axis=0)
        grad_c2_x, grad_c2_y = np.gradient(c2, dx, axis=1), np.gradient(c2, dy, axis=0)
        
        J1_preds.append([
            -(D11 * grad_c1_x + D12 * grad_c2_x),
            -(D11 * grad_c1_y + D12 * grad_c2_y)
        ])
        J2_preds.append([
            -(D21 * grad_c1_x + D22 * grad_c2_x),
            -(D21 * grad_c1_y + D22 * grad_c2_y)
        ])
    
    return J1_preds, J2_preds

def get_interpolation_weights(lys, ly_target, sigma=2.5):
    """Attention-weighted interpolation weights."""
    lys = np.array(lys).reshape(-1, 1)
    target = np.array([[ly_target]])
    distances = cdist(target, lys).flatten()
    weights = np.exp(-(distances**2) / (2 * sigma**2))
    return weights / (weights.sum() + 1e-10)

@st.cache_data
def attention_weighted_interpolation(solutions, lys, ly_target, diff_type, sigma=2.5):
    """Interpolate solution for target Ly."""
    matching = [s for s in solutions if s['diffusion_type'] == diff_type]
    if not matching:
        return None

    lys = np.array([s['Ly_parsed'] for s in matching])
    weights = get_interpolation_weights(lys, ly_target, sigma)

    Lx = matching[0]['params']['Lx']
    t_max = matching[0]['params']['t_max']
    x_coords = np.linspace(0, Lx, 50)
    y_coords = np.linspace(0, ly_target, 50)
    times = np.linspace(0, t_max, 50)

    c1_interp = np.zeros((len(times), 50, 50))
    c2_interp = np.zeros((len(times), 50, 50))

    for sol, w in zip(matching, weights):
        X_sol = sol['X'][:, 0]
        Y_sol = sol['Y'][0, :] * (ly_target / sol['Ly_parsed'])
        for t_idx in range(len(times)):
            interp_c1 = RegularGridInterpolator((Y_sol, X_sol), sol['c1_preds'][t_idx], 
                                              method='linear', bounds_error=False, fill_value=0)
            interp_c2 = RegularGridInterpolator((Y_sol, X_sol), sol['c2_preds'][t_idx], 
                                              method='linear', bounds_error=False, fill_value=0)
            
            X_target, Y_target = np.meshgrid(x_coords, y_coords, indexing='ij')
            points = np.column_stack([Y_target.ravel(), X_target.ravel()])
            c1_interp[t_idx] += w * interp_c1(points).reshape(50, 50).T
            c2_interp[t_idx] += w * interp_c2(points).reshape(50, 50).T

    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    params = matching[0]['params'].copy()
    params['Ly'] = ly_target
    J1_preds, J2_preds = compute_fluxes(c1_interp, c2_interp, x_coords, y_coords, params)

    return {
        'params': params, 'X': X, 'Y': Y, 'c1_preds': list(c1_interp), 'c2_preds': list(c2_interp),
        'J1_preds': J1_preds, 'J2_preds': J2_preds, 'times': times, 'diffusion_type': diff_type,
        'interpolated': True, 'Ly_parsed': ly_target
    }

@st.cache_data
def load_and_interpolate_solution(solutions, diff_type, ly_target, tolerance=1e-4):
    """Load exact or interpolate solution."""
    exact = [s for s in solutions if s['diffusion_type'] == diff_type and abs(s['Ly_parsed'] - ly_target) < tolerance]
    if exact:
        solution = exact[0].copy()
        solution['interpolated'] = False
        if 'J1_preds' not in solution:
            J1, J2 = compute_fluxes(
                solution['c1_preds'], solution['c2_preds'],
                solution['X'][:, 0], solution['Y'][0, :], solution['params']
            )
            solution['J1_preds'], solution['J2_preds'] = J1, J2
        solution['Ly_parsed'] = ly_target
        return solution
    
    return attention_weighted_interpolation(solutions, [s['Ly_parsed'] for s in solutions], ly_target, diff_type)

def plot_publication_concentrations(solution, time_index, downsample, **style_params):
    """Publication-quality concentration plots with DISTINCT domain visualization."""
    x_coords, y_coords = solution['X'][:, 0], solution['Y'][0, :]
    t_val, Lx, Ly = solution['times'][time_index], solution['params']['Lx'], solution['params']['Ly']
    diff_type = solution['diffusion_type'].replace('_', ' ').title()

    # STYLE PARAMS
    font_size, title_size, tick_size, label_interval = style_params['font_size'], style_params['title_size'], style_params['tick_size'], style_params['label_interval']
    show_domain_labels, domain_label_size, border_thickness = style_params['show_domain_labels'], style_params['domain_label_size'], style_params['border_thickness']

    # Downsample
    ds = max(1, downsample)
    x_indices = np.unique(np.linspace(0, 49, num=50//ds, dtype=int))
    y_indices = np.unique(np.linspace(0, 49, num=50//ds, dtype=int))
    x_ds, y_ds = x_coords[x_indices], y_coords[y_indices]
    c1, c2 = solution['c1_preds'][time_index][np.ix_(y_indices, x_indices)], solution['c2_preds'][time_index][np.ix_(y_indices, x_indices)]

    # DOMAIN STYLE - CRITICAL FOR VISIBILITY
    domain_info = DOMAIN_STYLES.get(round(Ly, 1), DOMAIN_STYLES[50.0])
    border_color = domain_info['border']

    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=(f"**Cu Concentration**<br><sub>t = {t_val:.1f} s</sub>", 
                                     f"**Ni Concentration**<br><sub>t = {t_val:.1f} s</sub>"))

    # CONCENTRATION HEATMAPS
    fig.add_trace(go.Heatmap(x=x_ds, y=y_ds, z=c1, colorscale=domain_info['colormap'],
                           colorbar=dict(title="**Cu**", titlefont=dict(size=font_size), tickfont=dict(size=tick_size)),
                           zsmooth='best', hovertemplate="%{z:.3f}<br>x: %{x:.1f}<br>y: %{y:.1f}<extra></extra>"), row=1, col=1)
    
    fig.add_trace(go.Heatmap(x=x_ds, y=y_ds, z=c2, colorscale='magma' if Ly < 70 else 'viridis',
                           colorbar=dict(title="**Ni**", titlefont=dict(size=font_size), tickfont=dict(size=tick_size)),
                           zsmooth='best', hovertemplate="%{z:.3f}<br>x: %{x:.1f}<br>y: %{y:.1f}<extra></extra>"), row=1, col=2)

    # **ENHANCED DOMAIN BORDERS** - MAKE 50μm vs 90μm OBVIOUS
    for col in [1, 2]:
        xref, yref = f'x{col}', f'y{col}'
        
        # THICK COLORED BORDER
        fig.add_shape(type='rect', x0=0, y0=0, x1=Lx, y1=Ly, xref=xref, yref=yref,
                     fillcolor=None, line=dict(color=border_color, width=border_thickness))
        
        # TOP INTERFACE (BLACK)
        fig.add_shape(type='line', x0=0, y0=Ly, x1=Lx, y1=Ly, xref=xref, yref=yref,
                     line=dict(color='black', width=3))
        
        # SUBTLE GRID
        for x in x_ds[::3]: fig.add_shape(type='line', x0=x, y0=0, x1=x, y1=Ly, xref=xref, yref=yref,
                                         line=dict(color='white', width=1, dash='dot'))
        for y in y_ds[::3]: fig.add_shape(type='line', x0=0, y0=y, x1=Lx, y1=y, xref=xref, yref=yref,
                                         line=dict(color='white', width=1, dash='dot'))

    # **DOMAIN LABEL** - SUPER VISIBLE
    if show_domain_labels:
        label_text = f"<b>{domain_info['label']}</b>"
        fig.add_annotation(x=0.98*Lx, y=0.02*Ly, xref=f'x1', yref=f'y1', text=label_text,
                          showarrow=False, font=dict(size=domain_label_size+2, color=border_color, family="Arial Black"),
                          bgcolor="rgba(255,255,255,0.9)", bordercolor=border_color, borderwidth=2)

    fig.update_layout(
        height=500, title=dict(text=f"**{diff_type}**", font=dict(size=title_size+2, family="Arial Black")),
        font=dict(size=font_size), showlegend=False, template='plotly_white'
    )

    # FORMATTED AXES
    for col in [1, 2]:
        fig.update_xaxes(title="**x (μm)**", titlefont=dict(size=font_size), tickfont=dict(size=tick_size),
                        tickmode='linear', tick0=0, dtick=label_interval, row=1, col=col)
        fig.update_yaxes(title="**y (μm)**", titlefont=dict(size=font_size), tickfont=dict(size=tick_size),
                        tickmode='linear', tick0=0, dtick=label_interval, range=[0, Ly], row=1, col=col)

    return fig

def plot_publication_lines(solutions, diff_type, ly_values, time_index, **style_params):
    """Publication-quality line profiles."""
    if len(ly_values) != 2: return st.error("Select **exactly 2 Ly values**")

    sol1 = load_and_interpolate_solution(solutions, diff_type, ly_values[0])
    sol2 = load_and_interpolate_solution(solutions, diff_type, ly_values[1])
    if not sol1 or not sol2: return

    # MATPLOTLIB PUBLICATION STYLE
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=export_dpi, facecolor='white')

    # STYLE PARAMS
    colors = [ly50_color, ly90_color]
    styles = [line_style_50, line_style_90]
    t_val = sol1['times'][time_index]
    x_center = 25  # Center column

    for i, (sol, color, style) in enumerate(zip([sol1, sol2], colors, styles)):
        y_coords = sol['Y'][0, :]
        c1_center = sol['c1_preds'][time_index][:, x_center]
        c2_center = sol['c2_preds'][time_index][:, x_center]
        label = f'**Ly = {sol["Ly_parsed"]:.0f} μm**'

        # ENHANCED LINES
        ax1.plot(c1_center, y_coords, color=color, linewidth=line_width, 
                linestyle=style, alpha=line_alpha, label=label, zorder=10)
        ax2.plot(c2_center, y_coords, color=color, linewidth=line_width, 
                linestyle=style, alpha=line_alpha, label=label, zorder=10)

        # DOMAIN HEIGHT MARKER
        ax1.axhline(y=sol['params']['Ly'], color=color, linestyle=':', alpha=0.8, linewidth=3, label=f'Domain Top')

    # PUBLICATION FORMATTING
    for ax in [ax1, ax2]:
        ax.set_xlabel('**Concentration (mol/cm³)**', fontsize=font_size+1, fontweight='bold')
        ax.set_ylabel('**y (μm)**', fontsize=font_size+1, fontweight='bold')
        ax.tick_params(labelsize=tick_size)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=font_size-1, frameon=True, fancybox=True, shadow=True)

    ax1.set_title(f'**Cu Profile** • t = {t_val:.1f} s', fontsize=title_size, fontweight='bold', pad=20)
    ax2.set_title(f'**Ni Profile** • t = {t_val:.1f} s', fontsize=title_size, fontweight='bold', pad=20)
    
    plt.suptitle(f'**Central Line Profiles: {diff_type.replace("_", " ").title()}**', 
                fontsize=title_size+2, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    st.pyplot(fig)
    plt.close()

def main():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔬 **Simulation Controls**")
    
    solutions, metadata, load_logs = load_solutions(SOLUTION_DIR)
    
    if not solutions:
        st.error("❌ **No .pkl files found!**")
        st.info("**Place files like:** `solution_crossdiffusion_ly_50.0_tmax_200.pkl`")
        return

    # CONTROLS
    diff_type = st.sidebar.selectbox("**Diffusion Type**", DIFFUSION_TYPES,
                                   format_func=lambda x: x.replace('_', ' ').title())
    
    available_lys = sorted(set(s['Ly_parsed'] for s in solutions if s['diffusion_type'] == diff_type))
    if len(available_lys) < 2:
        st.error(f"❌ Need **2+ Ly values** for {diff_type}")
        return
        
    ly_values = st.sidebar.multiselect("**Select Ly Values**", available_lys,
                                     default=available_lys[:2], max_selections=2,
                                     format_func=lambda x: f"**{x:.0f} μm**")
    
    if len(ly_values) != 2:
        st.warning("⚠️ **Select EXACTLY 2 Ly values**")
        return

    time_index = st.sidebar.slider("**Time Step**", 0, 49, 49)
    downsample = st.sidebar.slider("**Detail**", 1, 5, 2)

    # PACKAGE STYLE PARAMS
    style_params = {
        'font_size': font_size, 'title_size': title_size, 'tick_size': tick_size, 'label_interval': label_interval,
        'show_domain_labels': show_domain_labels, 'domain_label_size': domain_label_size, 'border_thickness': border_thickness
    }

    # TABS
    tab1, tab2 = st.tabs(["🌈 **Concentrations**", "📈 **Line Profiles**"])

    with tab1:
        st.markdown("## **Concentration Fields** - **50μm vs 90μm**")
        for ly in ly_values:
            solution = load_and_interpolate_solution(solutions, diff_type, ly)
            if solution:
                fig = plot_publication_concentrations(solution, time_index, downsample, **style_params)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"❌ No solution for Ly={ly}")

    with tab2:
        st.markdown("## **Central Line Profiles**")
        plot_publication_lines(solutions, diff_type, ly_values, time_index, **style_params)

    # STATUS
    st.sidebar.markdown("---")
    st.sidebar.success(f"✅ **Loaded: {len(solutions)} solutions**")
    st.sidebar.info(f"**Selected:** {diff_type} • Ly={ly_values}")

if __name__ == "__main__":
    main()
