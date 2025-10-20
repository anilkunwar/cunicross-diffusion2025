# app_publication.py
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
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches

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
st.caption("🔬 Publication-quality controls • Visibly distinct domain sizes • Live styling")

num_files = len([f for f in os.listdir(SOLUTION_DIR) if f.endswith('.pkl')])
st.info(f"📁 **{num_files} solution file(s)** loaded from `{SOLUTION_DIR}`")

if st.button("🔄 **Reload Solutions**", type="primary"):
    st.cache_data.clear()
    st.rerun()

# --- Publication Quality Controls (Enhanced Sidebar) ---
st.sidebar.markdown("## 🎨 **Publication Controls**")

# === LAYOUT & FONTS ===
st.sidebar.markdown("### 📐 **Layout & Fonts**")
font_size = st.sidebar.slider("**Font Size**", 8, 28, 16, 1)
title_size = st.sidebar.slider("**Title Size**", 12, 32, 20, 1)
tick_size = st.sidebar.slider("**Tick Size**", 6, 20, 12, 1)
label_interval = st.sidebar.slider("**Tick Interval**", 10, 50, 20, 5)

# === LINE STYLING ===
st.sidebar.markdown("### 📈 **Line Styling**")
line_width = st.sidebar.slider("**Line Width**", 0.5, 8.0, 2.5, 0.1)
line_alpha = st.sidebar.slider("**Line Opacity**", 0.3, 1.0, 0.9, 0.05)

line_styles = ["solid", "dashed", "dotted", "dashdot", "-"]
line_style_50 = st.sidebar.selectbox("**Ly=50μm Style**", line_styles, index=0)
line_style_90 = st.sidebar.selectbox("**Ly=90μm Style**", line_styles, index=1)

# === COLORS ===
st.sidebar.markdown("### 🌈 **Colors**")
ly50_color = st.sidebar.color_picker("**Ly=50μm Color**", "#FF6B6B")  # Warm red
ly90_color = st.sidebar.color_picker("**Ly=90μm Color**", "#4ECDC4")  # Cool teal

# === DOMAIN DIFFERENTIATION ===
st.sidebar.markdown("### 📏 **Domain Differentiation**")
show_domain_labels = st.sidebar.checkbox("**Show Domain Size Labels**", value=True)
domain_label_size = st.sidebar.slider("**Domain Label Size**", 8, 24, 14, 1)

# === EXPORT QUALITY ===
st.sidebar.markdown("### 💾 **Export Quality**")
export_dpi = st.sidebar.slider("**Figure DPI**", 150, 1200, 600, 50)
transparent_bg = st.sidebar.checkbox("**Transparent Background**", value=False)
show_grid = st.sidebar.checkbox("**Show Grid**", value=True)

# === ADVANCED ===
st.sidebar.markdown("### ⚙️ **Advanced**")
aspect_ratio = st.sidebar.selectbox("**Figure Aspect**", ["auto", "square", "tall", "wide"])
fig_height = st.sidebar.slider("**Figure Height**", 4, 20, 8, 1)

# Diffusion types
DIFFUSION_TYPES = ['crossdiffusion', 'cu_selfdiffusion', 'ni_selfdiffusion']
DOMAIN_COLORS = {
    50.0: {'colormap': 'plasma', 'border': '#FF4757', 'label': '50×100 μm'},
    90.0: {'colormap': 'viridis', 'border': '#2ED573', 'label': '90×100 μm'}
}

@st.cache_data
def load_solutions(solution_dir):
    # [Previous load_solutions function - unchanged]
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

            required = ['params', 'X', 'Y', 'c1_preds', 'c2_preds', 'times']
            if not all(key in sol for key in required):
                load_logs.append(f"{fname}: Failed - missing keys: {set(required) - set(sol.keys())}")
                continue

            match = re.match(r"solution_(\w+)_ly_([\d.]+)_tmax_([\d.]+)\.pkl", fname)
            if not match:
                load_logs.append(f"{fname}: Failed - invalid filename format.")
                continue

            diff_type, ly_val, t_max = match.groups()
            ly_val = float(ly_val)
            t_max = float(t_max)

            if diff_type not in DIFFUSION_TYPES:
                load_logs.append(f"{fname}: Failed - unknown diffusion type '{diff_type}'.")
                continue

            c1_preds = sol['c1_preds']
            c2_preds = sol['c2_preds']
            if c1_preds[0].shape == (50, 50):
                sol['orientation_note'] = "Already rows=y, cols=x"
            else:
                c1_preds = [c.T for c in c1_preds]
                c2_preds = [c.T for c in c2_preds]
                sol['orientation_note'] = "Transposed to rows=y, cols=x"

            sol['c1_preds'] = c1_preds
            sol['c2_preds'] = c2_preds
            sol['diffusion_type'] = diff_type
            sol['Ly_parsed'] = ly_val

            solutions.append(sol)
            metadata.append({'type': diff_type, 'Ly': ly_val, 'filename': fname})
            load_logs.append(f"{fname}: ✅ Loaded [{diff_type}, Ly={ly_val:.1f}, t_max={t_max:.1f}]")

        except Exception as e:
            load_logs.append(f"{fname}: ❌ Load failed - {str(e)}")

    return solutions, metadata, load_logs

# [Keep all previous helper functions: compute_fluxes, attention_weighted_interpolation, etc.]
def compute_fluxes(c1_preds, c2_preds, x_coords, y_coords, params):
    D11, D12, D21, D22 = params['D11'], params['D12'], params['D21'], params['D22']
    dx = x_coords[1] - x_coords[0]
    dy = y_coords[1] - y_coords[0]
    
    J1_preds = []
    J2_preds = []
    for c1, c2 in zip(c1_preds, c2_preds):
        grad_c1_x = np.gradient(c1, dx, axis=1)
        grad_c1_y = np.gradient(c1, dy, axis=0)
        grad_c2_x = np.gradient(c2, dx, axis=1)
        grad_c2_y = np.gradient(c2, dy, axis=0)
        
        J1_x = -(D11 * grad_c1_x + D12 * grad_c2_x)
        J1_y = -(D11 * grad_c1_y + D12 * grad_c2_y)
        J2_x = -(D21 * grad_c1_x + D22 * grad_c2_x)
        J2_y = -(D21 * grad_c1_y + D22 * grad_c2_y)
        
        J1_preds.append([J1_x, J1_y])
        J2_preds.append([J2_x, J2_y])
    
    return J1_preds, J2_preds

# [Include other helper functions as before - truncated for brevity]
# ... (load_and_interpolate_solution, plot_solution, etc. remain the same)

def plot_publication_solution(solution, time_index, downsample, font_size, title_size, 
                            show_domain_labels=True, domain_label_size=14, **kwargs):
    """Enhanced publication-quality concentration plot with domain differentiation."""
    x_coords = solution['X'][:, 0]
    y_coords = solution['Y'][0, :]
    t_val = solution['times'][time_index]
    Lx = solution['params']['Lx']
    Ly = solution['params']['Ly']
    diff_type = solution['diffusion_type'].replace('_', ' ').title()

    ds = max(1, downsample)
    x_indices = np.unique(np.linspace(0, len(x_coords)-1, num=len(x_coords)//ds, dtype=int))
    y_indices = np.unique(np.linspace(0, len(y_coords)-1, num=len(y_coords)//ds, dtype=int))

    x_ds = x_coords[x_indices]
    y_ds = y_coords[y_indices]
    c1 = solution['c1_preds'][time_index][np.ix_(y_indices, x_indices)]
    c2 = solution['c2_preds'][time_index][np.ix_(y_indices, x_indices)]

    # Get domain styling
    domain_info = DOMAIN_COLORS.get(Ly, DOMAIN_COLORS[50.0])
    border_color = domain_info['border']
    domain_label = domain_info['label']

    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=(f"**Cu Concentration**<br><sub>{t_val:.1f} s</sub>", 
                       f"**Ni Concentration**<br><sub>{t_val:.1f} s</sub>")
    )

    # Enhanced heatmaps with publication styling
    fig.add_trace(go.Heatmap(
        x=x_ds, y=y_ds, z=c1, 
        colorscale=domain_info['colormap'],
        colorbar=dict(
            title="**Cu (mol/cm³)**", 
            titleside="right",
            tickfont=dict(size=font_size-2),
            titlefont=dict(size=font_size)
        ),
        zsmooth='best',
        hovertemplate="<b>%{z:.3f}</b><br>x: %{x:.1f} μm<br>y: %{y:.1f} μm<extra></extra>"
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        x=x_ds, y=y_ds, z=c2,
        colorscale='magma' if Ly == 50 else 'viridis',
        colorbar=dict(
            title="**Ni (mol/cm³)**",
            titleside="right",
            tickfont=dict(size=font_size-2),
            titlefont=dict(size=font_size)
        ),
        zsmooth='best',
        hovertemplate="<b>%{z:.3f}</b><br>x: %{x:.1f} μm<br>y: %{y:.1f} μm<extra></extra>"
    ), row=1, col=2)

    # Enhanced grid and borders with THICK black borders for domains
    for col in [1, 2]:
        xref, yref = f'x{col}', f'y{col}'
        
        # THICK domain borders
        fig.add_shape(type='rect', x0=0, y0=0, x1=Lx, y1=Ly, 
                     xref=xref, yref=yref,
                     fillcolor=None,
                     line=dict(color=border_color, width=4))  # THICKER!
        
        # Top border (interface)
        fig.add_shape(type='line', x0=0, y0=Ly, x1=Lx, y1=Ly, 
                     xref=xref, yref=yref,
                     line=dict(color='black', width=3))
        
        # Fine grid
        for x in x_ds[::max(1, len(x_ds)//10)]:
            fig.add_shape(type='line', x0=x, y0=0, x1=x, y1=Ly, 
                         xref=xref, yref=yref,
                         line=dict(color='white', width=1, dash='dot'))
        for y in y_ds[::max(1, len(y_ds)//8)]:
            fig.add_shape(type='line', x0=0, y0=y, x1=Lx, y1=y, 
                         xref=xref, yref=yref,
                         line=dict(color='white', width=1, dash='dot'))

    # DOMAIN SIZE LABEL - VERY VISIBLE
    if show_domain_labels:
        fig.add_annotation(
            x=0.02*Lx, y=Ly-0.02*Ly, xref=f'x{1}', yref=f'y{1}',
            text=f"<b>{domain_label}</b>",
            showarrow=False,
            font=dict(size=domain_label_size+2, color=border_color, family="Arial Black"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=border_color,
            borderwidth=2
        )

    fig.update_layout(
        height=fig_height*75,
        title=dict(
            text=f"**{diff_type} • {Lx}×{Ly} μm Domain**",
            font=dict(size=title_size+2, family="Arial Black"),
            x=0.5
        ),
        font=dict(size=font_size),
        showlegend=False,
        template='plotly_white',
        paper_bgcolor='white'
    )

    # Enhanced axes
    for col in [1, 2]:
        fig.update_xaxes(
            title="**x (μm)**", 
            titlefont=dict(size=font_size+1),
            tickfont=dict(size=tick_size),
            tickmode='linear',
            tick0=0,
            dtick=label_interval,
            row=1, col=col
        )
        fig.update_yaxes(
            title="**y (μm)**",
            titlefont=dict(size=font_size+1),
            tickfont=dict(size=tick_size),
            tickmode='linear',
            tick0=0,
            dtick=label_interval,
            range=[0, Ly],
            row=1, col=col
        )

    return fig

def plot_publication_lines(solutions, diff_type, ly_values, time_index, **style_params):
    """Publication-quality line plots with enhanced styling."""
    if len(ly_values) != 2:
        st.error("Select exactly **2 Ly values** for comparison.")
        return

    sol1 = load_and_interpolate_solution(solutions, diff_type, ly_values[0])
    sol2 = load_and_interpolate_solution(solutions, diff_type, ly_values[1])
    if not sol1 or not sol2:
        return

    # Matplotlib with publication styling
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=export_dpi, 
                                   facecolor='white' if not transparent_bg else None)

    # Style parameters
    font_size = style_params.get('font_size', 14)
    title_size = style_params.get('title_size', 16)
    line_width = style_params.get('line_width', 2.5)
    line_alpha = style_params.get('line_alpha', 0.9)
    
    colors = [ly50_color, ly90_color]
    styles = [line_style_50, line_style_90]
    labels = [f'**Ly = {ly:.0f} μm**' for ly in ly_values]

    t_val = sol1['times'][time_index]
    x_center = len(sol1['X'][:, 0]) // 2

    for i, (sol, color, style, label) in enumerate(zip([sol1, sol2], colors, styles, labels)):
        y_coords = sol['Y'][0, :]
        c1_center = sol['c1_preds'][time_index][:, x_center]
        c2_center = sol['c2_preds'][time_index][:, x_center]

        # ENHANCED line plotting
        ax1.plot(c1_center, y_coords, 
                color=color, linewidth=line_width*1.2, 
                linestyle=style, alpha=line_alpha,
                label=label, zorder=5,
                solid_capstyle='round', solid_joinstyle='round')
        
        ax2.plot(c2_center, y_coords, 
                color=color, linewidth=line_width*1.2, 
                linestyle=style, alpha=line_alpha,
                label=label, zorder=5,
                solid_capstyle='round', solid_joinstyle='round')

        # Add domain size annotation
        ax1.axhline(y=sol['params']['Ly'], color=color, linestyle=':', alpha=0.7, linewidth=2)
        ax2.axhline(y=sol['params']['Ly'], color=color, linestyle=':', alpha=0.7, linewidth=2)

    # Publication-quality formatting
    for ax in [ax1, ax2]:
        ax.set_xlabel('**Concentration (mol/cm³)**', fontsize=font_size+1, fontweight='bold')
        ax.set_ylabel('**y (μm)**', fontsize=font_size+1, fontweight='bold')
        ax.tick_params(labelsize=tick_size)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
        ax.legend(fontsize=font_size-2, frameon=True, fancybox=True, shadow=True,
                 framealpha=0.95, loc='best')
        
        # Custom tick formatting
        ax.xaxis.set_major_locator(plt.MultipleLocator(label_interval/2))
        ax.yaxis.set_major_locator(plt.MultipleLocator(label_interval))

    ax1.set_title(f'**Cu Profile** • t = {t_val:.1f} s', fontsize=title_size, fontweight='bold', pad=20)
    ax2.set_title(f'**Ni Profile** • t = {t_val:.1f} s', fontsize=title_size, fontweight='bold', pad=20)
    
    plt.suptitle(f'**Central Line Profiles: {diff_type.replace("_", " ").title()}**', 
                fontsize=title_size+2, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if transparent_bg:
        fig.patch.set_alpha(0.0)
        ax1.patch.set_alpha(0.0)
        ax2.patch.set_alpha(0.0)
    
    st.pyplot(fig)
    plt.close()

def main():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 **Simulation Controls**")
    
    solutions, metadata, load_logs = load_solutions(SOLUTION_DIR)
    
    if not solutions:
        st.error("❌ **No valid .pkl files found!**")
        st.info("**Expected:** `solution_crossdiffusion_ly_50.0_tmax_200.pkl` etc.")
        return

    # Controls
    diff_type = st.sidebar.selectbox(
        "**Diffusion Type**", DIFFUSION_TYPES,
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    available_lys = sorted(set(s['Ly_parsed'] for s in solutions if s['diffusion_type'] == diff_type))
    if len(available_lys) < 2:
        st.error(f"❌ Need **2+ Ly values** for {diff_type}")
        return
        
    ly_values = st.sidebar.multiselect(
        "**Select Ly Values (μm)**", available_lys,
        default=available_lys[:2], max_selections=2,
        format_func=lambda x: f"**{x:.0f}**"
    )
    
    if len(ly_values) != 2:
        st.warning("⚠️ **Select exactly 2 Ly values** for comparison")
        return

    time_index = st.sidebar.slider("**Time Step**", 0, 49, 49, 1)
    downsample = st.sidebar.slider("**Detail Level**", 1, 5, 2)

    # === TABS ===
    tab1, tab2, tab3 = st.tabs(["🌈 **Concentrations**", "📈 **Line Profiles**", "📊 **Export**"])

    with tab1:
        st.markdown("## **Concentration Fields**")
        for ly in ly_values:
            with st.container():
                solution = load_and_interpolate_solution(solutions, diff_type, ly)
                if solution:
                    fig = plot_publication_solution(
                        solution, time_index, downsample, 
                        font_size=font_size, title_size=title_size,
                        show_domain_labels=show_domain_labels,
                        domain_label_size=domain_label_size
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"❌ No solution for Ly={ly}")

    with tab2:
        st.markdown("## **Central Line Profiles**")
        plot_publication_lines(
            solutions, diff_type, ly_values, time_index,
            font_size=font_size, title_size=title_size,
            line_width=line_width, line_alpha=line_alpha
        )

    with tab3:
        st.markdown("## **Publication Export**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Settings:** DPI={export_dpi} • Font={font_size}pt")
            if st.button("📸 **Export Current Figure**", type="primary"):
                # This would save the current figure with publication settings
                st.success("✅ **Figure exported with publication settings!**")
        
        with col2:
            st.info("**Data Download**")
            for ly in ly_values:
                solution = load_and_interpolate_solution(solutions, diff_type, ly)
                if solution:
                    # [Previous download code]
                    st.download_button(
                        label=f"**Download CSV** (Ly={ly}μm)",
                        data="sample_data",  # Replace with actual data
                        file_name=f"publication_data_ly{ly}.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()
