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
# Directory containing .pkl solution files (ensure exists)
SOLUTION_DIR = os.path.join(os.path.dirname(__file__), "pinn_solutions")
os.makedirs(SOLUTION_DIR, exist_ok=True)
st.title("🧠 Cu–Ni Cross-Diffusion PINN Visualization")
st.caption("Automatically loads all `.pkl` solutions from the `pinn_solutions/` folder")
num_files = len([f for f in os.listdir(SOLUTION_DIR) if f.endswith('.pkl')])
st.info(f"📁 **Solution directory:** `{SOLUTION_DIR}` — {num_files} file(s) found")
if st.button("🔄 Reload Solutions"):
    st.cache_data.clear()
    st.experimental_rerun()
# Diffusion types
DIFFUSION_TYPES = ['crossdiffusion', 'cu_selfdiffusion', 'ni_selfdiffusion']
# List of Plotly colorscales
COLORSCALES = ['aggrnyl', 'agsunset', 'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'darkmint', 'electric', 'emrld', 'gnbu', 'greens', 'greys', 'hot', 'inferno', 'jet', 'magenta', 'magma', 'mint', 'orrd', 'oranges', 'oryel', 'peach', 'pinkyl', 'plasma', 'plotly3', 'pubu', 'pubugn', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu', 'rdpu', 'redor', 'reds', 'sunset', 'sunsetdark', 'teal', 'tealgrn', 'turbo', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr', 'ylorrd', 'algae', 'amp', 'deep', 'dense', 'gray', 'haline', 'ice', 'matter', 'solar', 'speed', 'tempo', 'thermal', 'turbid', 'armyrose', 'brbg', 'earth', 'fall', 'geyser', 'prgn', 'piyg', 'picnic', 'portland', 'puor', 'rdgy', 'rdylbu', 'rdylgn', 'spectral', 'tealrose', 'temps', 'tropic', 'balance', 'curl', 'delta', 'oxy', 'edge', 'hsv', 'icefire', 'phase', 'twilight', 'mrybm', 'mygbm']
@st.cache_data
def load_solutions(solution_dir):
    solutions = []
    load_logs = []
    metadata = [] # Store diffusion type, Ly, filename
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
                load_logs.append(f"{fname}: Failed - missing keys: {set(required) - set(sol.keys())}")
                continue
            # Parse diffusion type and Ly from filename
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
            # Validate and fix array orientation
            c1_preds = sol['c1_preds']
            c2_preds = sol['c2_preds']
            if not (isinstance(c1_preds, list) and isinstance(c2_preds, list) and len(c1_preds) == len(c2_preds)):
                load_logs.append(f"{fname}: Failed - invalid c1_preds/c2_preds structure.")
                continue
            if c1_preds[0].shape == (50, 50):
                # Assume already (y,x) based on training code
                sol['orientation_note'] = "Already rows=y, cols=x"
            else:
                # Transpose if necessary (though unlikely)
                c1_preds = [c.T for c in c1_preds]
                c2_preds = [c.T for c in c2_preds]
                sol['orientation_note'] = "Transposed to rows=y, cols=x"
            sol['c1_preds'] = c1_preds
            sol['c2_preds'] = c2_preds
            sol['diffusion_type'] = diff_type
            sol['Ly_parsed'] = ly_val
            solutions.append(sol)
            metadata.append({'type': diff_type, 'Ly': ly_val, 'filename': fname})
            load_logs.append(f"{fname}: Loaded [{diff_type}, Ly={ly_val:.1f}, t_max={t_max:.1f}]")
        except Exception as e:
            load_logs.append(f"{fname}: Load failed - {str(e)}")
    return solutions, metadata, load_logs
def compute_fluxes(c1_preds, c2_preds, x_coords, y_coords, params):
    """Compute fluxes J1 and J2 from concentrations using finite differences."""
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
@st.cache_data
def attention_weighted_interpolation(solutions, lys, ly_target, diff_type, sigma=2.5):
    """Interpolate solutions for a specific diffusion type and Ly."""
    matching = [s for s in solutions if s['diffusion_type'] == diff_type]
    if not matching:
        return None
    lys = np.array([s['Ly_parsed'] for s in matching])
    if not lys.size:
        return None
    weights = get_interpolation_weights(lys, ly_target, sigma)
    Lx = matching[0]['params']['Lx']
    t_max = matching[0]['params']['t_max']
    x_coords = np.linspace(0, Lx, 50)
    y_coords = np.linspace(0, ly_target, 50)
    times = np.linspace(0, t_max, 50)
    c1_interp = np.zeros((len(times), 50, 50))
    c2_interp = np.zeros((len(times), 50, 50))
    for sol, w in zip(matching, weights):
        X_sol = sol['X'][:, 0] # x_coords
        Y_sol = sol['Y'][0, :] * (ly_target / sol['params']['Ly']) # scaled y_coords
        for t_idx in range(len(times)):
            # Fix: orientation rows=y cols=x, so grid (y, x)
            interp_c1 = RegularGridInterpolator(
                (Y_sol, X_sol), sol['c1_preds'][t_idx],
                method='linear', bounds_error=False, fill_value=0
            )
            interp_c2 = RegularGridInterpolator(
                (Y_sol, X_sol), sol['c2_preds'][t_idx],
                method='linear', bounds_error=False, fill_value=0
            )
            X_target, Y_target = np.meshgrid(x_coords, y_coords, indexing='ij')
            # Points in order (y, x) to match grid
            points = np.column_stack([Y_target.ravel(), X_target.ravel()])
            # Reshape and transpose to match rows=y cols=x
            c1_interp[t_idx] += w * interp_c1(points).reshape(50, 50).T
            c2_interp[t_idx] += w * interp_c2(points).reshape(50, 50).T
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    param_set = matching[0]['params'].copy()
    param_set['Ly'] = ly_target
    J1_preds, J2_preds = compute_fluxes(c1_interp, c2_interp, x_coords, y_coords, param_set)
    return {
        'params': param_set,
        'X': X,
        'Y': Y,
        'c1_preds': list(c1_interp),
        'c2_preds': list(c2_interp),
        'J1_preds': J1_preds,
        'J2_preds': J2_preds,
        'times': times,
        'diffusion_type': diff_type,
        'interpolated': True,
        'used_lys': lys.tolist(),
        'attention_weights': weights.tolist(),
        'orientation_note': "c1_preds and c2_preds are arrays of shape (50,50) where rows (i) correspond to y-coordinates and columns (j) correspond to x-coordinates."
    }
def get_interpolation_weights(lys, ly_target, sigma=2.5):
    lys = np.array(lys).reshape(-1, 1)
    target = np.array([[ly_target]])
    distances = cdist(target, lys).flatten()
    weights = np.exp(-(distances**2) / (2 * sigma**2))
    weights /= weights.sum() + 1e-10
    return weights
@st.cache_data
def load_and_interpolate_solution(solutions, diff_type, ly_target, tolerance=1e-4):
    """Load or interpolate solution for target diffusion type and Ly."""
    exact = [s for s in solutions if s['diffusion_type'] == diff_type and abs(s['Ly_parsed'] - ly_target) < tolerance]
    if exact:
        solution = exact[0]
        solution['interpolated'] = False
        if 'J1_preds' not in solution or 'J2_preds' not in solution:
            J1_preds, J2_preds = compute_fluxes(
                solution['c1_preds'], solution['c2_preds'],
                solution['X'][:, 0], solution['Y'][0, :], solution['params']
            )
            solution['J1_preds'] = J1_preds
            solution['J2_preds'] = J2_preds
        return solution
    return attention_weighted_interpolation(solutions, [s['Ly_parsed'] for s in solutions], ly_target, diff_type)
def plot_solution(solution, time_index, downsample, title_suffix="", cu_colormap='viridis', ni_colormap='magma', font_size=12, x_tick_interval=10, y_tick_interval=10, show_grid=True, grid_thickness=0.5, border_thickness=1, height_multiplier=5, width_multiplier=5):
    """Plot concentration fields for a single solution."""
    x_coords = solution['X'][:, 0]
    y_coords = solution['Y'][0, :]
    t_val = solution['times'][time_index]
    Lx = solution['params']['Lx']
    Ly = solution['params']['Ly']
    ds = max(1, downsample)
    x_indices = np.unique(np.linspace(0, len(x_coords)-1, num=len(x_coords)//ds, dtype=int))
    y_indices = np.unique(np.linspace(0, len(y_coords)-1, num=len(y_coords)//ds, dtype=int))
    x_ds = x_coords[x_indices]
    y_ds = y_coords[y_indices]
    c1 = solution['c1_preds'][time_index][np.ix_(y_indices, x_indices)] # rows=y, cols=x
    c2 = solution['c2_preds'][time_index][np.ix_(y_indices, x_indices)]
    fig = make_subplots(rows=1, cols=2,
                       subplot_titles=(f"Cu @ {t_val:.1f}s", f"Ni @ {t_val:.1f}s"),
                       horizontal_spacing=0.20) # Increased horizontal spacing between subplots
    fig.add_trace(go.Heatmap(
        x=x_ds, y=y_ds, z=c1, colorscale=cu_colormap,
        colorbar=dict(title='Cu Conc', x=0.45), zsmooth='best'
    ), row=1, col=1)
    fig.add_trace(go.Heatmap(
        x=x_ds, y=y_ds, z=c2, colorscale=ni_colormap,
        colorbar=dict(title='Ni Conc', x=1.02), zsmooth='best'
    ), row=1, col=2)
    if show_grid:
        for col, xref, yref in [(1, 'x', 'y'), (2, 'x2', 'y2')]:
            for x in np.arange(0, Lx + x_tick_interval, x_tick_interval):
                fig.add_shape(type='line', x0=x, y0=0, x1=x, y1=Ly, xref=xref, yref=yref,
                              line=dict(color='gray', width=grid_thickness, dash='dot'))
            for y in np.arange(0, Ly + y_tick_interval, y_tick_interval):
                fig.add_shape(type='line', x0=0, y0=y, x1=Lx, y1=y, xref=xref, yref=yref,
                              line=dict(color='gray', width=grid_thickness, dash='dot'))
    for col, xref, yref in [(1, 'x', 'y'), (2, 'x2', 'y2')]:
        fig.add_shape(type='line', x0=0, y0=Ly, x1=Lx, y1=Ly, xref=xref, yref=yref,
                      line=dict(color='black', width=border_thickness))
        fig.add_shape(type='rect', x0=0, y0=0, x1=Lx, y1=Ly, xref=xref, yref=yref,
                      line=dict(color='black', width=border_thickness))
    height = int(height_multiplier * Ly)
    width = int(width_multiplier * Lx * 2) # *2 for two subplots
    fig.update_layout(
        height=height,
        width=width,
        title=f"Concentration Fields: {Lx}μm × {Ly}μm {title_suffix}",
        showlegend=False,
        template='plotly_white',
        font=dict(size=font_size)
    )
    fig.update_xaxes(title_text="x (μm)", range=[0, Lx], gridcolor='white', zeroline=False, row=1, col=1, dtick=x_tick_interval)
    fig.update_yaxes(title_text="y (μm)", range=[0, Ly], gridcolor='white', zeroline=False, row=1, col=1, dtick=y_tick_interval)
    fig.update_xaxes(title_text="x (μm)", range=[0, Lx], gridcolor='white', zeroline=False, row=1, col=2, dtick=x_tick_interval)
    fig.update_yaxes(title_text="y (μm)", range=[0, Ly], gridcolor='white', zeroline=False, row=1, col=2, dtick=y_tick_interval)
    st.plotly_chart(fig, use_container_width=False)
def create_flux_fig(sol, Ly, diff_type, t_val, time_index, downsample=2,
                    font_size=12, x_tick_interval=10, y_tick_interval=10,
                    show_grid=True, grid_thickness=0.5, border_thickness=1.5,
                    arrow_thickness=1.5):
    """Create a single flux figure with TRUE physical aspect ratio."""
    Lx = float(sol['params']['Lx'])
    Ly = float(Ly)

    x_coords = sol['X'][:, 0]
    y_coords = sol['Y'][0, :]

    ds = max(1, downsample)
    x_idx = np.unique(np.linspace(0, len(x_coords)-1, max(2, len(x_coords)//ds), dtype=int))
    y_idx = np.unique(np.linspace(0, len(y_coords)-1, max(2, len(y_coords)//ds), dtype=int))
    x_ds = x_coords[x_idx]
    y_ds = y_coords[y_idx]

    # Extract downsampled fields (rows=y, cols=x)
    J1_x = sol['J1_preds'][time_index][0][np.ix_(y_idx, x_idx)]
    J1_y = sol['J1_preds'][time_index][1][np.ix_(y_idx, x_idx)]
    J2_x = sol['J2_preds'][time_index][0][np.ix_(y_idx, x_idx)]
    J2_y = sol['J2_preds'][time_index][1][np.ix_(y_idx, x_idx)]
    c1 = sol['c1_preds'][time_index][np.ix_(y_idx, x_idx)]
    c2 = sol['c2_preds'][time_index][np.ix_(y_idx, x_idx)]

    J1_mag = np.sqrt(J1_x**2 + J1_y**2)
    J2_mag = np.sqrt(J2_x**2 + J2_y**2)

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Cu Flux Magnitude (log)", "Ni Flux Magnitude (log)",
            "Cu J₁x",                   "Ni J₂x",
            "Cu J₁y",                   "Ni J₂y"
        ),
        vertical_spacing=0.09,
        horizontal_spacing=0.12
    )

    # === Heatmaps + Contours ===
    fig.add_trace(go.Heatmap(z=np.log10(np.maximum(J1_mag, 1e-12)), x=x_ds, y=y_ds,
                             colorscale='viridis', showscale=True,
                             colorbar=dict(title="log|J<sub>Cu</sub>|", x=1.02, len=0.28, y=0.83)), row=1, col=1)
    fig.add_trace(go.Contour(z=c1, x=x_ds, y=y_ds, showscale=False, line_width=1, opacity=0.4,
                             contours_coloring='lines'), row=1, col=1)

    fig.add_trace(go.Heatmap(z=np.log10(np.maximum(J2_mag, 1e-12)), x=x_ds, y=y_ds,
                             colorscale='cividis', showscale=True,
                             colorbar=dict(title="log|J<sub>Ni</sub>|", x=1.27, len=0.28, y=0.83)), row=1, col=2)
    fig.add_trace(go.Contour(z=c2, x=x_ds, y=y_ds, showscale=False, line_width=1, opacity=0.4,
                             contours_coloring='lines'), row=1, col=2)

    for data, row, col in [(J1_x, 2, 1), (J2_x, 2, 2), (J1_y, 3, 1), (J2_y, 3, 2)]:
        fig.add_trace(go.Heatmap(z=data, x=x_ds, y=y_ds, colorscale='RdBu', zmid=0,
                                 colorbar=dict(x=1.02 + (col-1)*0.25, len=0.28,
                                               y=0.5 - (row-2)*0.35)), row=row, col=col)

    # === Flux arrows (only on magnitude plots) ===
    scale = 0.18 * Lx
    stride = max(1, len(x_ds) // 9)
    arrows = []
    for i in range(0, len(x_ds), stride):
        for j in range(0, len(y_ds), stride):
            if J1_mag[j, i] > 1e-11:
                arrows.append(dict(
                    x=x_ds[i], y=y_ds[j],
                    ax=x_ds[i] + scale * J1_x[j,i] / (J1_mag.max() + 1e-15),
                    ay=y_ds[j] + scale * J1_y[j,i] / (J1_mag.max() + 1e-15),
                    xref="x", yref="y", axref="x", ayref="y",
                    arrowhead=2, arrowsize=1.4, arrowwidth=arrow_thickness, arrowcolor="white"
                ))
            if J2_mag[j, i] > 1e-11:
                arrows.append(dict(
                    x=x_ds[i], y=y_ds[j],
                    ax=x_ds[i] + scale * J2_x[j,i] / (J2_mag.max() + 1e-15),
                    ay=y_ds[j] + scale * J2_y[j,i] / (J2_mag.max() + 1e-15),
                    xref="x2", yref="y2", axref="x2", ayref="y2",
                    arrowhead=2, arrowsize=1.4, arrowwidth=arrow_thickness, arrowcolor="white"
                ))

    # === TRUE PHYSICAL ASPECT RATIO ===
    physical_ratio = Ly / Lx  # e.g., 50/60 ≈ 0.833 → y shorter than x
    base_width = 1100
    fig_height = int(base_width * physical_ratio * 3.1)  # 3 rows + spacing

    fig.update_layout(
        height=fig_height,
        width=base_width,
        title=f"Flux Fields — {diff_type.replace('_', ' ')} | t = {t_val:.1f}s | Domain: {Lx}×{Ly} μm (True Physical Ratio)",
        template="plotly_white",
        font=dict(size=font_size),
        margin=dict(l=70, r=220, t=110, b=60),
        annotations=arrows
    )

    # === Enforce correct aspect ratio on EVERY subplot ===
    subplot_map = [
        (1,1,"x","y"), (1,2,"x2","y2"),
        (2,1,"x3","y3"), (2,2,"x4","y4"),
        (3,1,"x5","y5"), (3,2,"x6","y6")
    ]

    for row, col, xname, yname in subplot_map:
        fig.update_xaxes(
            title="x (μm)" if row == 3 else None,
            range=[0, Lx], dtick=x_tick_interval,
            showgrid=show_grid, gridcolor="lightgray", gridwidth=grid_thickness,
            zeroline=False, row=row, col=col
        )
        fig.update_yaxes(
            title="y (μm)" if col == 1 else None,
            range=[0, Ly], dtick=y_tick_interval,
            showgrid=show_grid, gridcolor="lightgray", gridwidth=grid_thickness,
            zeroline=False,
            scaleanchor=xname,
            scaleratio=physical_ratio,   # ← THIS IS THE KEY
            constrain="domain",
            row=row, col=col
        )
        # Domain border
        fig.add_shape(type="rect",
                      x0=0, y0=0, x1=Lx, y1=Ly,
                      line=dict(color="black", width=border_thickness),
                      xref=xname, yref=yname, row=row, col=col)

    return fig
def plot_flux_comparison(solutions, diff_type, ly_values, time_index, downsample,
                         font_size=12, x_tick_interval=10, y_tick_interval=10,
                         show_grid=True, grid_thickness=0.5, border_thickness=1.5,
                         arrow_thickness=1.5):
    """Side-by-side flux comparison with TRUE physical aspect ratio."""
    if len(ly_values) != 2:
        st.error("Please select exactly two Ly values.")
        return

    # Convert to float once and for all
    ly_values = [float(ly) for ly in ly_values]

    sol1 = load_and_interpolate_solution(solutions, diff_type, ly_values[0])
    sol2 = load_and_interpolate_solution(solutions, diff_type, ly_values[1])
    if not (sol1 and sol2):
        st.error("Failed to load one or both solutions.")
        return

    t_val = sol1['times'][time_index]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### **Ly = {ly_values[0]:.1f} μm**")
        fig1 = create_flux_fig(sol1, ly_values[0], diff_type, t_val, time_index,
                               downsample, font_size, x_tick_interval, y_tick_interval,
                               show_grid, grid_thickness, border_thickness, arrow_thickness)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown(f"### **Ly = {ly_values[1]:.1f} μm**")
        fig2 = create_flux_fig(sol2, ly_values[1], diff_type, t_val, time_index,
                               downsample, font_size, x_tick_interval, y_tick_interval,
                               show_grid, grid_thickness, border_thickness, arrow_thickness)
        st.plotly_chart(fig2, use_container_width=True)

    st.success("True physical aspect ratio applied: 60 μm (x) > 50 μm (y) | 90 μm (y) > 60 μm (x)")
def plot_line_comparison(solutions, diff_type, ly_values, time_index, line_thickness=2, label_font_size=12, tick_font_size=10, conc_x_tick_interval=0.0005, line_y_tick_interval=10, spine_thickness=1.5, color_ly1='#1f77b4', color_ly2='#ff7f0e', fig_width=12, fig_height=6, legend_loc='upper right', show_grid=True, cu_x_label='Cu Concentration (mol/cm³)', cu_y_label='y (μm)', ni_x_label='Ni Concentration (mol/cm³)', ni_y_label='y (μm)', legend_label1='', legend_label2='', rotate_ticks=False):
    """Plot central line profiles for two Ly values for a given diffusion type."""
    if len(ly_values) != 2:
        st.error("Please select exactly two Ly values for comparison.")
        return
    sol1 = load_and_interpolate_solution(solutions, diff_type, ly_values[0])
    sol2 = load_and_interpolate_solution(solutions, diff_type, ly_values[1])
    if not sol1 or not sol2:
        st.error(f"Could not load solutions for {diff_type}, Ly={ly_values}")
        return
    sns.set_context("paper")
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=300)
    colors = [color_ly1, color_ly2]
    labels = [legend_label1 or f'Ly = {ly_values[0]:.1f} μm', legend_label2 or f'Ly = {ly_values[1]:.1f} μm']
    c1_min = min(min(sol1['c1_preds'][time_index][:, len(sol1['X'][:, 0]) // 2]), min(sol2['c1_preds'][time_index][:, len(sol2['X'][:, 0]) // 2]))
    c1_max = max(max(sol1['c1_preds'][time_index][:, len(sol1['X'][:, 0]) // 2]), max(sol2['c1_preds'][time_index][:, len(sol2['X'][:, 0]) // 2]))
    c2_min = min(min(sol1['c2_preds'][time_index][:, len(sol1['X'][:, 0]) // 2]), min(sol2['c2_preds'][time_index][:, len(sol2['X'][:, 0]) // 2]))
    c2_max = max(max(sol1['c2_preds'][time_index][:, len(sol1['X'][:, 0]) // 2]), max(sol2['c2_preds'][time_index][:, len(sol2['X'][:, 0]) // 2]))
    for idx, (sol, Ly, color, label, linestyle) in enumerate(zip([sol1, sol2], ly_values, colors, labels, ['-', '--'])):
        x_idx = len(sol['X'][:, 0]) // 2 # x = center index
        y_coords = sol['Y'][0, :]
        c1_center = sol['c1_preds'][time_index][:, x_idx]
        c2_center = sol['c2_preds'][time_index][:, x_idx]
        t_val = sol['times'][time_index]
        ax1.plot(c1_center, y_coords, label=label, linestyle=linestyle, linewidth=line_thickness, color=color)
        ax2.plot(c2_center, y_coords, label=label, linestyle=linestyle, linewidth=line_thickness, color=color)
    ax1.set_xlabel(cu_x_label, fontsize=label_font_size)
    ax1.set_ylabel(cu_y_label, fontsize=label_font_size)
    ax1.set_title(f'Cu @ x=center, t={t_val:.1f}s', fontsize=label_font_size + 2)
    ax1.legend(fontsize=label_font_size - 2, loc=legend_loc)
    ax1.tick_params(labelsize=tick_font_size)
    ax1.set_xticks(np.arange(c1_min, c1_max + conc_x_tick_interval, conc_x_tick_interval))
    ax1.set_yticks(np.arange(0, max(ly_values) + line_y_tick_interval, line_y_tick_interval))
    ax1.grid(show_grid)
    for spine in ax1.spines.values():
        spine.set_linewidth(spine_thickness)
    if rotate_ticks:
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    ax2.set_xlabel(ni_x_label, fontsize=label_font_size)
    ax2.set_ylabel(ni_y_label, fontsize=label_font_size)
    ax2.set_title(f'Ni @ x=center, t={t_val:.1f}s', fontsize=label_font_size + 2)
    ax2.legend(fontsize=label_font_size - 2, loc=legend_loc)
    ax2.tick_params(labelsize=tick_font_size)
    ax2.set_xticks(np.arange(c2_min, c2_max + conc_x_tick_interval, conc_x_tick_interval))
    ax2.set_yticks(np.arange(0, max(ly_values) + line_y_tick_interval, line_y_tick_interval))
    ax2.grid(show_grid)
    for spine in ax2.spines.values():
        spine.set_linewidth(spine_thickness)
    if rotate_ticks:
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    plt.suptitle(f"Central Line Profiles: {diff_type.replace('_', ' ')}", fontsize=label_font_size + 4)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    st.pyplot(fig)
    plt.close()
def compute_center_concentrations(solutions, diff_type, ly_values):
    """Compute Cu and Ni concentrations and flux magnitudes at the center point for given Ly values."""
    center_concentrations = []
    center_idx = 25 # Approximate center for 50x50 grid
    for ly in ly_values:
        sol = load_and_interpolate_solution(solutions, diff_type, ly)
        if not sol:
            continue
        times = sol['times']
        c1_center = []
        c2_center = []
        J1_mag_center = []
        J2_mag_center = []
        for ti in range(len(times)):
            c1_center.append(sol['c1_preds'][ti][center_idx, center_idx])
            c2_center.append(sol['c2_preds'][ti][center_idx, center_idx])
            J1_x = sol['J1_preds'][ti][0][center_idx, center_idx]
            J1_y = sol['J1_preds'][ti][1][center_idx, center_idx]
            J2_x = sol['J2_preds'][ti][0][center_idx, center_idx]
            J2_y = sol['J2_preds'][ti][1][center_idx, center_idx]
            J1_mag_center.append(np.sqrt(J1_x**2 + J1_y**2))
            J2_mag_center.append(np.sqrt(J2_x**2 + J2_y**2))
        center_concentrations.append({
            'Ly': ly,
            'times': times,
            'c1_center': np.array(c1_center),
            'c2_center': np.array(c2_center),
            'J1_mag_center': np.array(J1_mag_center),
            'J2_mag_center': np.array(J2_mag_center)
        })
    return center_concentrations
def plot_center_concentrations(center_concentrations, diff_type, line_thickness=2, label_font_size=12, tick_font_size=10, center_time_tick_interval=50, center_conc_y_tick_interval=0.0001, center_flux_y_tick_interval=0.0001, spine_thickness=1.5, color_ly1='#1f77b4', color_ly2='#ff7f0e', fig_width=12, fig_height=12, legend_loc='upper right', show_grid=True, cu_conc_x_label='Time (s)', cu_conc_y_label='Cu Concentration (mol/cm³)', ni_conc_x_label='Time (s)', ni_conc_y_label='Ni Concentration (mol/cm³)', cu_flux_x_label='Time (s)', cu_flux_y_label='Cu Flux Magnitude', ni_flux_x_label='Time (s)', ni_flux_y_label='Ni Flux Magnitude', legend_label1='', legend_label2='', rotate_ticks=False):
    """Plot center point concentrations and flux magnitudes for two Ly values."""
    sns.set_context("paper")
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    colors = [color_ly1, color_ly2]
    labels = [legend_label1 or f'Ly = {center_concentrations[0]["Ly"]:.1f} μm', legend_label2 or f'Ly = {center_concentrations[1]["Ly"]:.1f} μm']
    c1_min = min(min(center_concentrations[0]['c1_center']), min(center_concentrations[1]['c1_center']))
    c1_max = max(max(center_concentrations[0]['c1_center']), max(center_concentrations[1]['c1_center']))
    c2_min = min(min(center_concentrations[0]['c2_center']), min(center_concentrations[1]['c2_center']))
    c2_max = max(max(center_concentrations[0]['c2_center']), max(center_concentrations[1]['c2_center']))
    j1_min = min(min(center_concentrations[0]['J1_mag_center']), min(center_concentrations[1]['J1_mag_center']))
    j1_max = max(max(center_concentrations[0]['J1_mag_center']), max(center_concentrations[1]['J1_mag_center']))
    j2_min = min(min(center_concentrations[0]['J2_mag_center']), min(center_concentrations[1]['J2_mag_center']))
    j2_max = max(max(center_concentrations[0]['J2_mag_center']), max(center_concentrations[1]['J2_mag_center']))
    for conc, color, label in zip(center_concentrations, colors, labels):
        ax1.plot(conc['times'], conc['c1_center'], label=label, linewidth=line_thickness, color=color)
        ax2.plot(conc['times'], conc['c2_center'], label=label, linewidth=line_thickness, color=color)
        ax3.plot(conc['times'], conc['J1_mag_center'], label=label, linewidth=line_thickness, color=color)
        ax4.plot(conc['times'], conc['J2_mag_center'], label=label, linewidth=line_thickness, color=color)
    ax1.set_xlabel(cu_conc_x_label, fontsize=label_font_size)
    ax1.set_ylabel(cu_conc_y_label, fontsize=label_font_size)
    ax1.set_title('Cu Concentration at Center', fontsize=label_font_size + 2)
    ax1.legend(fontsize=label_font_size - 2, loc=legend_loc)
    ax1.tick_params(labelsize=tick_font_size)
    ax1.set_xticks(np.arange(0, max(conc['times']) + center_time_tick_interval, center_time_tick_interval))
    ax1.set_yticks(np.arange(c1_min, c1_max + center_conc_y_tick_interval, center_conc_y_tick_interval))
    ax1.grid(show_grid)
    for spine in ax1.spines.values():
        spine.set_linewidth(spine_thickness)
    if rotate_ticks:
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    ax2.set_xlabel(ni_conc_x_label, fontsize=label_font_size)
    ax2.set_ylabel(ni_conc_y_label, fontsize=label_font_size)
    ax2.set_title('Ni Concentration at Center', fontsize=label_font_size + 2)
    ax2.legend(fontsize=label_font_size - 2, loc=legend_loc)
    ax2.tick_params(labelsize=tick_font_size)
    ax2.set_xticks(np.arange(0, max(conc['times']) + center_time_tick_interval, center_time_tick_interval))
    ax2.set_yticks(np.arange(c2_min, c2_max + center_conc_y_tick_interval, center_conc_y_tick_interval))
    ax2.grid(show_grid)
    for spine in ax2.spines.values():
        spine.set_linewidth(spine_thickness)
    if rotate_ticks:
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    ax3.set_xlabel(cu_flux_x_label, fontsize=label_font_size)
    ax3.set_ylabel(cu_flux_y_label, fontsize=label_font_size)
    ax3.set_title('Cu Flux Magnitude at Center', fontsize=label_font_size + 2)
    ax3.legend(fontsize=label_font_size - 2, loc=legend_loc)
    ax3.tick_params(labelsize=tick_font_size)
    ax3.set_xticks(np.arange(0, max(conc['times']) + center_time_tick_interval, center_time_tick_interval))
    ax3.set_yticks(np.arange(j1_min, j1_max + center_flux_y_tick_interval, center_flux_y_tick_interval))
    ax3.grid(show_grid)
    if any(np.any(conc['J1_mag_center'] > 0) for conc in center_concentrations):
        ax3.set_yscale('log')
    for spine in ax3.spines.values():
        spine.set_linewidth(spine_thickness)
    if rotate_ticks:
        plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
    ax4.set_xlabel(ni_flux_x_label, fontsize=label_font_size)
    ax4.set_ylabel(ni_flux_y_label, fontsize=label_font_size)
    ax4.set_title('Ni Flux Magnitude at Center', fontsize=label_font_size + 2)
    ax4.legend(fontsize=label_font_size - 2, loc=legend_loc)
    ax4.tick_params(labelsize=tick_font_size)
    ax4.set_xticks(np.arange(0, max(conc['times']) + center_time_tick_interval, center_time_tick_interval))
    ax4.set_yticks(np.arange(j2_min, j2_max + center_flux_y_tick_interval, center_flux_y_tick_interval))
    ax4.grid(show_grid)
    if any(np.any(conc['J2_mag_center'] > 0) for conc in center_concentrations):
        ax4.set_yscale('log')
    for spine in ax4.spines.values():
        spine.set_linewidth(spine_thickness)
    if rotate_ticks:
        plt.setp(ax4.get_xticklabels(), rotation=45, ha="right")
    plt.suptitle(f"Center Point Evolution: {diff_type.replace('_', ' ')} (center)", fontsize=label_font_size + 4)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    st.pyplot(fig)
    plt.close(fig)
def download_data(solution, time_index, all_times=False):
    """Generate CSV or ZIP file for download."""
    x_coords = solution['X'][:, 0]
    y_coords = solution['Y'][0, :]
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    if not all_times:
        t_val = solution['times'][time_index]
        c1 = solution['c1_preds'][time_index]
        c2 = solution['c2_preds'][time_index]
        J1_x = solution['J1_preds'][time_index][0]
        J1_y = solution['J1_preds'][time_index][1]
        J2_x = solution['J2_preds'][time_index][0]
        J2_y = solution['J2_preds'][time_index][1]
        df = pd.DataFrame({
            'x': X.flatten(), 'y': Y.flatten(),
            'c1': c1.flatten(), 'c2': c2.flatten(),
            'J1_x': J1_x.flatten(), 'J1_y': J1_y.flatten(),
            'J2_x': J2_x.flatten(), 'J2_y': J2_y.flatten()
        })
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode('utf-8')
        return csv_bytes, f"data_{solution['diffusion_type']}_ly_{solution['params']['Ly']:.1f}_t_{t_val:.1f}s.csv"
    else:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for ti, t_val in enumerate(solution['times']):
                c1 = solution['c1_preds'][ti]
                c2 = solution['c2_preds'][ti]
                J1_x = solution['J1_preds'][ti][0]
                J1_y = solution['J1_preds'][ti][1]
                J2_x = solution['J2_preds'][ti][0]
                J2_y = solution['J2_preds'][ti][1]
                df = pd.DataFrame({
                    't': t_val * np.ones(X.size),
                    'x': X.flatten(), 'y': Y.flatten(),
                    'c1': c1.flatten(), 'c2': c2.flatten(),
                    'J1_x': J1_x.flatten(), 'J1_y': J1_y.flatten(),
                    'J2_x': J2_x.flatten(), 'J2_y': J2_y.flatten()
                })
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                zip_file.writestr(f"data_t_{t_val:.1f}s.csv", csv_buffer.getvalue())
        zip_buffer.seek(0)
        return zip_buffer.getvalue(), f"data_{solution['diffusion_type']}_ly_{solution['params']['Ly']:.1f}_all_times.zip"
def main():
    st.sidebar.header("Simulation Parameters")
    solutions, metadata, load_logs = load_solutions(SOLUTION_DIR)

    # Optional: Show loading log
    if load_logs:
        with st.expander("Solution Load Log", expanded=False):
            selected_log = st.selectbox("View load status", load_logs, index=0)
            st.code(selected_log, language="text")

    if not solutions:
        st.error("No valid solution files found in `pinn_solutions` directory.")
        st.info("""
        Expected files (examples):
        - `solution_crossdiffusion_ly_50.0_tmax_200.pkl`
        - `solution_crossdiffusion_ly_90.0_tmax_200.pkl`
        - `solution_cu_selfdiffusion_ly_50.0_tmax_200.pkl`
        - etc.
        """)
        return

    # Diffusion type selector
    diff_type = st.sidebar.selectbox(
        "Select Diffusion Type",
        options=DIFFUSION_TYPES,
        format_func=lambda x: x.replace('_', ' ').title()
    )

    # Extract available Ly values for this diffusion type
    available_lys_raw = sorted({
        s['Ly_parsed'] for s in solutions 
        if s['diffusion_type'] == diff_type
    })

    if len(available_lys_raw) < 2:
        st.sidebar.error(f"Not enough Ly values for {diff_type}. Need at least two.")
        st.stop()

    # Convert to float once and for all
    available_lys = [float(ly) for ly in available_lys_raw]
    ly_display = [f"{ly:.1f} μm" for ly in available_lys]

    # Let user pick two Ly values
    selected_labels = st.sidebar.multiselect(
        "Select Two Ly Values for Comparison",
        options=ly_display,
        default=ly_display[:2],
        max_selections=2
    )

    if len(selected_labels) != 2:
        st.sidebar.warning("Please select **exactly two** Ly values.")
        st.stop()

    # Convert back: "50.0 μm" → 50.0
    ly_values = [
        float(label.replace('μm', '').strip()) for label in selected_labels
    ]

    # Time & detail
    max_time_idx = len(solutions[0]['times']) - 1
    time_index = st.sidebar.slider("Select Time Index", 0, max_time_idx, max_time_idx)
    downsample = st.sidebar.slider("Visualization Detail Level (higher = faster)", 1, 6, 2)

    # === Visualization Options ===
    st.sidebar.header("Plot Customization")
    font_size = st.sidebar.slider("Font Size", 8, 20, 13)
    x_tick_interval = st.sidebar.slider("X Tick Interval (μm)", 5, 30, 10)
    y_tick_interval = st.sidebar.slider("Y Tick Interval (μm)", 5, 30, 10)
    show_grid = st.sidebar.checkbox("Show Grid", value=True)
    grid_thickness = st.sidebar.slider("Grid Thickness", 0.1, 2.0, 0.5, 0.1)
    border_thickness = st.sidebar.slider("Domain Border Thickness", 0.5, 4.0, 1.5, 0.5)
    arrow_thickness = st.sidebar.slider("Flux Arrow Thickness", 0.5, 3.0, 1.2, 0.1)

    # Colormaps
    cu_colormap = st.sidebar.selectbox("Cu Colormap", options=COLORSCALES, index=COLORSCALES.index('viridis'))
    ni_colormap = st.sidebar.selectbox("Ni Colormap", options=COLORSCALES, index=COLORSCALES.index('plasma'))

    # Line plot styling
    color_ly1 = st.sidebar.color_picker("Color for First Ly", "#2ca02c")
    color_ly2 = st.sidebar.color_picker("Color for Second Ly", "#d62728")
    line_thickness = st.sidebar.slider("Line Thickness (Profiles)", 1.5, 4.0, 2.5, 0.5)
    fig_width = st.sidebar.slider("Matplotlib Figure Width (inches)", 8, 20, 14)
    fig_height = st.sidebar.slider("Matplotlib Figure Height (inches)", 5, 12, 7)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Concentration Fields",
        "Flux Fields Comparison",
        "Central Line Profiles",
        "Center Point Evolution"
    ])

    with tab1:
        st.subheader("Concentration Fields (True Physical Aspect Ratio)")
        for ly in ly_values:
            sol = load_and_interpolate_solution(solutions, diff_type, ly)
            if sol:
                st.write(f"**Ly = {ly:.1f} μm**")
                plot_solution(
                    sol, time_index, downsample,
                    title_suffix=f"Ly = {ly:.1f} μm",
                    cu_colormap=cu_colormap,
                    ni_colormap=ni_colormap,
                    font_size=font_size,
                    x_tick_interval=x_tick_interval,
                    y_tick_interval=y_tick_interval,
                    show_grid=show_grid,
                    grid_thickness=grid_thickness,
                    border_thickness=border_thickness
                )
            else:
                st.error(f"Failed to load solution for Ly = {ly:.1f}")

    with tab2:
        st.subheader("Flux Fields Comparison — True Physical Geometry")
        st.info("60 μm horizontal is **longer** than 50 μm vertical | 90 μm vertical is **taller** than 60 μm horizontal")
        
        plot_flux_comparison(
            solutions=solutions,
            diff_type=diff_type,
            ly_values=ly_values,           # ← Now guaranteed floats
            time_index=time_index,
            downsample=downsample,
            font_size=font_size,
            x_tick_interval=x_tick_interval,
            y_tick_interval=y_tick_interval,
            show_grid=show_grid,
            grid_thickness=grid_thickness,
            border_thickness=border_thickness,
            arrow_thickness=arrow_thickness
        )

    with tab3:
        st.subheader("Central Vertical Line Profiles (x = 30 μm)")
        plot_line_comparison(
            solutions=solutions,
            diff_type=diff_type,
            ly_values=ly_values,
            time_index=time_index,
            line_thickness=line_thickness,
            color_ly1=color_ly1,
            color_ly2=color_ly2,
            fig_width=fig_width,
            fig_height=fig_height
        )

    with tab4:
        st.subheader("Center Point (x=30, y=Ly/2) Evolution Over Time")
        center_data = compute_center_concentrations(solutions, diff_type, ly_values)
        if center_data:
            plot_center_concentrations(
                center_data, diff_type,
                line_thickness=line_thickness,
                color_ly1=color_ly1,
                color_ly2=color_ly2,
                fig_width=fig_width,
                fig_height=fig_height
            )
        else:
            st.error("Could not extract center point data.")

    # === Download Section ===
    st.markdown("---")
    st.subheader("Download Simulation Data")
    for ly in ly_values:
        sol = load_and_interpolate_solution(solutions, diff_type, ly)
        if not sol:
            continue
        t_val = sol['times'][time_index]
        st.write(f"**Ly = {ly:.1f} μm** | t = {t_val:.1f} s")

        col1, col2 = st.columns(2)
        with col1:
            csv_bytes, csv_name = download_data(sol, time_index, all_times=False)
            st.download_button(
                label=f"Download CSV (Current Time)",
                data=csv_bytes,
                file_name=csv_name,
                mime="text/csv"
            )
        with col2:
            zip_bytes, zip_name = download_data(sol, time_index, all_times=True)
            st.download_button(
                label=f"Download ZIP (All Times)",
                data=zip_bytes,
                file_name=zip_name,
                mime="application/zip"
            )


if __name__ == "__main__":
    main()
