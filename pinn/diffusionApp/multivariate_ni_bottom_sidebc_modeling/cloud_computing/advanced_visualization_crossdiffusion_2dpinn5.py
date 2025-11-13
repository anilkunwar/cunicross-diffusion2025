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
def create_flux_fig(sol, Ly, diff_type, t_val, time_index, downsample, font_size=12, x_tick_interval=10, y_tick_interval=10, show_grid=True, grid_thickness=0.5, border_thickness=1, arrow_thickness=1, height_multiplier=5, width_multiplier=5):
    """Create flux figure for a single Ly value."""
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Cu Flux Mag", "Ni Flux Mag",
            "Cu J_1x", "Ni J_2x",
            "Cu J_1y", "Ni J_2y"
        ),
        vertical_spacing=0.15, # increased vertical spacing
        horizontal_spacing=0.20 # increased horizontal spacing
    )
    annotations_all = []
    x_coords = sol['X'][:, 0]
    y_coords = sol['Y'][0, :]
    Lx = sol['params']['Lx']
    ds = max(1, downsample)
    x_indices = np.unique(np.linspace(0, len(x_coords)-1, num=max(2, len(x_coords)//ds), dtype=int))
    y_indices = np.unique(np.linspace(0, len(y_coords)-1, num=max(2, len(y_coords)//ds), dtype=int))
    x_ds = x_coords[x_indices]
    y_ds = y_coords[y_indices]
    X_ds, Y_ds = np.meshgrid(x_ds, y_ds, indexing='ij')
    J1_x = sol['J1_preds'][time_index][0][np.ix_(y_indices, x_indices)]
    J1_y = sol['J1_preds'][time_index][1][np.ix_(y_indices, x_indices)]
    J2_x = sol['J2_preds'][time_index][0][np.ix_(y_indices, x_indices)]
    J2_y = sol['J2_preds'][time_index][1][np.ix_(y_indices, x_indices)]
    c1 = sol['c1_preds'][time_index][np.ix_(y_indices, x_indices)]
    c2 = sol['c2_preds'][time_index][np.ix_(y_indices, x_indices)]
    # Flux magnitudes (log for display)
    J1_magnitude = np.sqrt(J1_x**2 + J1_y**2)
    J2_magnitude = np.sqrt(J2_x**2 + J2_y**2)
    # Heatmap for log flux magnitude (Cu)
    fig.add_trace(go.Heatmap(
        x=x_ds, y=y_ds, z=np.log10(np.maximum(J1_magnitude, 1e-10)),
        colorscale='viridis',
        colorbar=dict(title=dict(text='Log |JCu|', side='top', font=dict(size=font_size - 2)), x=1.05, len=0.25, y=0.85),
        zsmooth='best', hovertemplate='x: %{x:.1f} μm<br>y: %{y:.1f} μm<br>Flux: %{z:.2e}'
    ), row=1, col=1)
    # Overlay contour of concentration
    fig.add_trace(go.Contour(
        z=c1, x=x_ds, y=y_ds, colorscale='blues', showscale=False, opacity=0.35,
        contours=dict(showlabels=False),
        line=dict(width=1)
    ), row=1, col=1)
    # Heatmap for log flux magnitude (Ni)
    fig.add_trace(go.Heatmap(
        x=x_ds, y=y_ds, z=np.log10(np.maximum(J2_magnitude, 1e-10)),
        colorscale='cividis',
        colorbar=dict(title=dict(text='Log |JNi|', side='top', font=dict(size=font_size - 2)), x=1.3, len=0.25, y=0.85),
        zsmooth='best', hovertemplate='x: %{x:.1f} μm<br>y: %{y:.1f} μm<br>Flux: %{z:.2e}'
    ), row=1, col=2)
    fig.add_trace(go.Contour(
        z=c2, x=x_ds, y=y_ds, colorscale='reds', showscale=False, opacity=0.35,
        contours=dict(showlabels=False),
        line=dict(width=1)
    ), row=1, col=2)
    # Jx components (row 2)
    fig.add_trace(go.Heatmap(
        x=x_ds, y=y_ds, z=J1_x, colorscale='rdbu', zmid=0,
        colorbar=dict(title=dict(text='Cu J_1x', side='top', font=dict(size=font_size - 2)), x=1.05, len=0.25, y=0.5),
        zsmooth='best', hovertemplate='x: %{x:.1f} μm<br>y: %{y:.1f} μm<br>J_1x: %{z:.2e}'
    ), row=2, col=1)
    fig.add_trace(go.Contour(
        z=c1, x=x_ds, y=y_ds, colorscale='blues', showscale=False, opacity=0.25, line=dict(width=1)
    ), row=2, col=1)
    fig.add_trace(go.Heatmap(
        x=x_ds, y=y_ds, z=J2_x, colorscale='rdbu', zmid=0,
        colorbar=dict(title=dict(text='Ni J_2x', side='top', font=dict(size=font_size - 2)), x=1.3, len=0.25, y=0.5),
        zsmooth='best', hovertemplate='x: %{x:.1f} μm<br>y: %{y:.1f} μm<br>J_2x: %{z:.2e}'
    ), row=2, col=2)
    fig.add_trace(go.Contour(
        z=c2, x=x_ds, y=y_ds, colorscale='reds', showscale=False, opacity=0.25, line=dict(width=1)
    ), row=2, col=2)
    # Jy components (row 3)
    fig.add_trace(go.Heatmap(
        x=x_ds, y=y_ds, z=J1_y, colorscale='rdbu', zmid=0,
        colorbar=dict(title=dict(text='Cu J_1y', side='top', font=dict(size=font_size - 2)), x=1.05, len=0.25, y=0.15),
        zsmooth='best', hovertemplate='x: %{x:.1f} μm<br>y: %{y:.1f} μm<br>J_1y: %{z:.2e}'
    ), row=3, col=1)
    fig.add_trace(go.Contour(
        z=c1, x=x_ds, y=y_ds, colorscale='blues', showscale=False, opacity=0.25, line=dict(width=1)
    ), row=3, col=1)
    fig.add_trace(go.Heatmap(
        x=x_ds, y=y_ds, z=J2_y, colorscale='rdbu', zmid=0,
        colorbar=dict(title=dict(text='Ni J_2y', side='top', font=dict(size=font_size - 2)), x=1.3, len=0.25, y=0.15),
        zsmooth='best', hovertemplate='x: %{x:.1f} μm<br>y: %{y:.1f} μm<br>J_2y: %{z:.2e}'
    ), row=3, col=2)
    fig.add_trace(go.Contour(
        z=c2, x=x_ds, y=y_ds, colorscale='reds', showscale=False, opacity=0.25, line=dict(width=1)
    ), row=3, col=2)
    # Add vector annotations (quiver-like arrows) but convert to annotation arrows to avoid overlap with colorbars.
    scale = 0.12 * Lx
    # sample stride for annotations to keep them readable
    stride = max(1, len(x_ds) // 10)
    for i in range(0, len(x_ds), stride):
        for j in range(0, len(y_ds), stride):
            if J1_magnitude[j, i] > 1e-12:
                annotations_all.append(dict(
                    x=x_ds[i], y=y_ds[j],
                    ax=x_ds[i] + scale * (J1_x[j, i] / (np.max(J1_magnitude) + 1e-12)),
                    ay=y_ds[j] + scale * (J1_y[j, i] / (np.max(J1_magnitude) + 1e-12)),
                    xref="x", yref="y",
                    axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=arrow_thickness, arrowcolor='white'
                ))
            if J2_magnitude[j, i] > 1e-12:
                annotations_all.append(dict(
                    x=x_ds[i], y=y_ds[j],
                    ax=x_ds[i] + scale * (J2_x[j, i] / (np.max(J2_magnitude) + 1e-12)),
                    ay=y_ds[j] + scale * (J2_y[j, i] / (np.max(J2_magnitude) + 1e-12)),
                    xref="x2", yref="y2",
                    axref="x2", ayref="y2",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=arrow_thickness, arrowcolor='white'
                ))
    # Add grid lines and border shapes per subplot
    subplot_refs = [
        (1, 1, 'x', 'y'), (1, 2, 'x2', 'y2'),
        (2, 1, 'x3', 'y3'), (2, 2, 'x4', 'y4'),
        (3, 1, 'x5', 'y5'), (3, 2, 'x6', 'y6')
    ]
    if show_grid:
        for row, col, xref, yref in subplot_refs:
            for x in np.arange(0, Lx + x_tick_interval, x_tick_interval):
                fig.add_shape(type='line', x0=x, y0=0, x1=x, y1=Ly, xref=xref, yref=yref,
                              line=dict(color='gray', width=grid_thickness, dash='dot'))
            for y in np.arange(0, Ly + y_tick_interval, y_tick_interval):
                fig.add_shape(type='line', x0=0, y0=y, x1=Lx, y1=y, xref=xref, yref=yref,
                              line=dict(color='gray', width=grid_thickness, dash='dot'))
    for row, col, xref, yref in subplot_refs:
        fig.add_shape(type='line', x0=0, y0=Ly, x1=Lx, y1=Ly, xref=xref, yref=yref,
                      line=dict(color='black', width=border_thickness))
        fig.add_shape(type='rect', x0=0, y0=0, x1=Lx, y1=Ly, xref=xref, yref=yref,
                      line=dict(color='black', width=border_thickness))
    height = int(3 * height_multiplier * Ly)
    width = int(width_multiplier * Lx * 2)
    fig.update_layout(
        height=height,
        width=width,
        margin=dict(l=30, r=250, t=150, b=30), # Increased right and top margin for colorbars and titles
        title=f"Flux Fields: {diff_type.replace('_', ' ')} @ t={t_val:.1f}s, Ly={Ly:.1f}μm",
        annotations=annotations_all,
        showlegend=False,
        template='plotly_white',
        font=dict(size=font_size)
    )
    # axis formatting for all subplots
    for row in range(1, 4):
        for col in range(1, 3):
            fig.update_xaxes(title_text="x (μm)", range=[0, Lx], gridcolor='white', zeroline=False, row=row, col=col, dtick=x_tick_interval)
            fig.update_yaxes(title_text="y (μm)", range=[0, Ly], gridcolor='white', zeroline=False, row=row, col=col, dtick=y_tick_interval)
    return fig
def plot_flux_comparison(solutions, diff_type, ly_values, time_index, downsample, font_size=12, x_tick_interval=10, y_tick_interval=10, show_grid=True, grid_thickness=0.5, border_thickness=1, arrow_thickness=1, height_multiplier=5, width_multiplier=5):
    """Plot flux fields for two Ly values for a given diffusion type (enhanced spacing/colorbar handling)."""
    if len(ly_values) != 2:
        st.error("Please select exactly two Ly values for comparison.")
        return
    sol1 = load_and_interpolate_solution(solutions, diff_type, ly_values[0])
    sol2 = load_and_interpolate_solution(solutions, diff_type, ly_values[1])
    if not sol1 or not sol2:
        st.error(f"Could not load solutions for {diff_type}, Ly={ly_values}")
        return
    t_val = sol1['times'][time_index]
    col1, col2 = st.columns(2)
    with col1:
        fig1 = create_flux_fig(sol1, ly_values[0], diff_type, t_val, time_index, downsample, font_size, x_tick_interval, y_tick_interval, show_grid, grid_thickness, border_thickness, arrow_thickness, height_multiplier, width_multiplier)
        st.plotly_chart(fig1, use_container_width=False)
    with col2:
        fig2 = create_flux_fig(sol2, ly_values[1], diff_type, t_val, time_index, downsample, font_size, x_tick_interval, y_tick_interval, show_grid, grid_thickness, border_thickness, arrow_thickness, height_multiplier, width_multiplier)
        st.plotly_chart(fig2, use_container_width=False)
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
    if load_logs:
        st.subheader("Solution Load Log")
        selected_log = st.selectbox("View load status for solutions", load_logs, index=0)
        st.write(selected_log)
    if not solutions:
        st.error("No valid solution files found in pinn_solutions directory.")
        st.write("Expected files:")
        st.write("- solution_crossdiffusion_ly_50.0_tmax_200.pkl")
        st.write("- solution_crossdiffusion_ly_90.0_tmax_200.pkl")
        st.write("- solution_cu_selfdiffusion_ly_50.0_tmax_200.pkl")
        st.write("- solution_cu_selfdiffusion_ly_90.0_tmax_200.pkl")
        st.write("- solution_ni_selfdiffusion_ly_50.0_tmax_200.pkl")
        return
    diff_type = st.sidebar.selectbox(
        "Select Diffusion Type",
        options=DIFFUSION_TYPES,
        format_func=lambda x: x.replace('_', ' ').title()
    )
    available_lys = sorted(set(s['Ly_parsed'] for s in solutions if s['diffusion_type'] == diff_type))
    if len(available_lys) < 2:
        st.sidebar.error(f"Not enough Ly values for {diff_type}. Need at least two.")
        return
    ly_values = st.sidebar.multiselect(
        "Select Two Ly Values for Comparison (μm)",
        options=available_lys,
        default=available_lys[:2] if len(available_lys) >= 2 else available_lys,
        format_func=lambda x: f"{x:.1f}",
        max_selections=2
    )
    time_index = st.sidebar.slider("Select Time", 0, len(solutions[0]['times'])-1, len(solutions[0]['times'])-1)
    #downsample = st.sidebar.slider("Detail Level", 1, 5, 2)
    downsample = st.sidebar.slider("Detail Level", 1, 15, 2)
    st.sidebar.header("Visualization Options")
    cu_colormap = st.sidebar.selectbox("Cu Colormap", options=COLORSCALES, index=COLORSCALES.index('viridis'))
    ni_colormap = st.sidebar.selectbox("Ni Colormap", options=COLORSCALES, index=COLORSCALES.index('magma'))
    st.sidebar.header("Plot Customization")
    font_size = st.sidebar.slider("Font Size", 8, 24, 12)
    x_tick_interval = st.sidebar.slider("X Tick Interval (μm)", 5, 50, 10)
    y_tick_interval = st.sidebar.slider("Y Tick Interval (μm)", 5, 50, 10)
    show_grid = st.sidebar.checkbox("Show Grid", value=True)
    grid_thickness = st.sidebar.slider("Grid Line Thickness", 0.1, 2.0, 0.5, step=0.1)
    border_thickness = st.sidebar.slider("Border Thickness", 0.5, 5.0, 1.0, step=0.5)
    arrow_thickness = st.sidebar.slider("Arrow Thickness (Flux)", 0.5, 3.0, 1.0, step=0.5)
    line_thickness = st.sidebar.slider("Line Thickness (Curves)", 1.0, 5.0, 2.0, step=0.5)
    label_font_size = st.sidebar.slider("Label Font Size", 8, 24, 12)
    tick_font_size = st.sidebar.slider("Tick Font Size", 6, 18, 10)
    spine_thickness = st.sidebar.slider("Spine Thickness", 0.5, 3.0, 1.5, step=0.5)
    color_ly1 = st.sidebar.color_picker("Line Color for First Ly", "#1f77b4")
    color_ly2 = st.sidebar.color_picker("Line Color for Second Ly", "#ff7f0e")
    fig_width = st.sidebar.slider("Figure Width", 6, 20, 12)
    fig_height = st.sidebar.slider("Figure Height", 4, 15, 6)
    legend_loc = st.sidebar.selectbox("Legend Location", options=['upper left', 'upper right', 'lower left', 'lower right', 'best'], index=1)
    rotate_ticks = st.sidebar.checkbox("Rotate Tick Labels", value=False)
    size_multiplier = st.sidebar.slider("Size Multiplier", 1, 10, 5)
    cu_x_label = st.sidebar.text_input("Cu X Label", "Cu Concentration (mol/cm³)")
    cu_y_label = st.sidebar.text_input("Cu Y Label", "y (μm)")
    ni_x_label = st.sidebar.text_input("Ni X Label", "Ni Concentration (mol/cm³)")
    ni_y_label = st.sidebar.text_input("Ni Y Label", "y (μm)")
    legend_label1 = st.sidebar.text_input("Legend Label 1", "")
    legend_label2 = st.sidebar.text_input("Legend Label 2", "")
    cu_conc_x_label = st.sidebar.text_input("Cu Conc X Label", "Time (s)")
    cu_conc_y_label = st.sidebar.text_input("Cu Conc Y Label", "Cu Concentration (mol/cm³)")
    ni_conc_x_label = st.sidebar.text_input("Ni Conc X Label", "Time (s)")
    ni_conc_y_label = st.sidebar.text_input("Ni Conc Y Label", "Ni Concentration (mol/cm³)")
    cu_flux_x_label = st.sidebar.text_input("Cu Flux X Label", "Time (s)")
    cu_flux_y_label = st.sidebar.text_input("Cu Flux Y Label", "Cu Flux Magnitude")
    ni_flux_x_label = st.sidebar.text_input("Ni Flux X Label", "Time (s)")
    ni_flux_y_label = st.sidebar.text_input("Ni Flux Y Label", "Ni Flux Magnitude")
    line_conc_x_tick_interval = st.sidebar.slider("Line Conc X Tick Interval (mol/cm³)", 0.0001, 0.001, 0.0005, step=0.0001, format="%.4f")
    line_y_tick_interval = st.sidebar.slider("Line Y Tick Interval (μm)", 5, 50, 10)
    center_time_tick_interval = st.sidebar.slider("Center Time Tick Interval (s)", 10, 100, 50)
    center_conc_y_tick_interval = st.sidebar.slider("Center Conc Y Tick Interval (mol/cm³)", 0.0001, 0.001, 0.0001, step=0.0001, format="%.4f")
    center_flux_y_tick_interval = st.sidebar.slider("Center Flux Y Tick Interval", 0.0001, 0.001, 0.0001, step=0.0001, format="%.4f")
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Concentration", "Flux Comparison", "Central Line Comparison", "Center Point Comparison"])
    with tab1:
        st.subheader("Concentration Fields")
        for ly in ly_values:
            solution = load_and_interpolate_solution(solutions, diff_type, ly)
            if solution:
                plot_solution(solution, time_index, downsample, title_suffix=f"[{diff_type.replace('_', ' ')}, Ly={ly:.1f}]", cu_colormap=cu_colormap, ni_colormap=ni_colormap, font_size=font_size, x_tick_interval=x_tick_interval, y_tick_interval=y_tick_interval, show_grid=show_grid, grid_thickness=grid_thickness, border_thickness=border_thickness, height_multiplier=size_multiplier, width_multiplier=size_multiplier)
            else:
                st.error(f"No solution for {diff_type}, Ly={ly:.1f}")
    with tab2:
        st.subheader("Flux Fields Comparison")
        plot_flux_comparison(solutions, diff_type, ly_values, time_index, downsample, font_size=font_size, x_tick_interval=x_tick_interval, y_tick_interval=y_tick_interval, show_grid=show_grid, grid_thickness=grid_thickness, border_thickness=border_thickness, arrow_thickness=arrow_thickness, height_multiplier=size_multiplier, width_multiplier=size_multiplier)
    with tab3:
        st.subheader("Central Line Profiles Comparison")
        plot_line_comparison(solutions, diff_type, ly_values, time_index, line_thickness=line_thickness, label_font_size=label_font_size, tick_font_size=tick_font_size, conc_x_tick_interval=line_conc_x_tick_interval, line_y_tick_interval=line_y_tick_interval, spine_thickness=spine_thickness, color_ly1=color_ly1, color_ly2=color_ly2, fig_width=fig_width, fig_height=fig_height, legend_loc=legend_loc, show_grid=show_grid, cu_x_label=cu_x_label, cu_y_label=cu_y_label, ni_x_label=ni_x_label, ni_y_label=ni_y_label, legend_label1=legend_label1, legend_label2=legend_label2, rotate_ticks=rotate_ticks)
    with tab4:
        st.subheader("Center Point Concentration and Flux Magnitude Comparison")
        center_concentrations = compute_center_concentrations(solutions, diff_type, ly_values)
        if center_concentrations:
            plot_center_concentrations(center_concentrations, diff_type, line_thickness=line_thickness, label_font_size=label_font_size, tick_font_size=tick_font_size, center_time_tick_interval=center_time_tick_interval, center_conc_y_tick_interval=center_conc_y_tick_interval, center_flux_y_tick_interval=center_flux_y_tick_interval, spine_thickness=spine_thickness, color_ly1=color_ly1, color_ly2=color_ly2, fig_width=fig_width, fig_height=fig_height, legend_loc=legend_loc, show_grid=show_grid, cu_conc_x_label=cu_conc_x_label, cu_conc_y_label=cu_conc_y_label, ni_conc_x_label=ni_conc_x_label, ni_conc_y_label=ni_conc_y_label, cu_flux_x_label=cu_flux_x_label, cu_flux_y_label=cu_flux_y_label, ni_flux_x_label=ni_flux_x_label, ni_flux_y_label=ni_flux_y_label, legend_label1=legend_label1, legend_label2=legend_label2, rotate_ticks=rotate_ticks)
        else:
            st.error(f"Could not compute center concentrations for {diff_type}, Ly={ly_values}")
    # Download
    st.subheader("Download Data")
    for ly in ly_values:
        solution = load_and_interpolate_solution(solutions, diff_type, ly)
        if not solution:
            continue
        st.write(f"Data for Ly = {ly:.1f} μm")
        col1, col2 = st.columns(2)
        with col1:
            data_bytes, filename = download_data(solution, time_index, all_times=False)
            st.download_button(
                label=f"Download CSV (t={solution['times'][time_index]:.1f}s, Ly={ly:.1f})",
                data=data_bytes,
                file_name=filename,
                mime="text/csv"
            )
        with col2:
            data_bytes, filename = download_data(solution, time_index, all_times=True)
            st.download_button(
                label=f"Download ZIP (All Times, Ly={ly:.1f})",
                data=data_bytes,
                file_name=filename,
                mime="application/zip"
            )
if __name__ == "__main__":
    main()
