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

# Directory containing .pkl solution files
SOLUTION_DIR = os.path.join(os.path.dirname(__file__), "pinn_solutions")

# Diffusion types
DIFFUSION_TYPES = ['crossdiffusion', 'cu_selfdiffusion', 'ni_selfdiffusion']

# List of Plotly colorscales
COLORSCALES = ['aggrnyl', 'agsunset', 'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'darkmint', 'electric', 'emrld', 'gnbu', 'greens', 'greys', 'hot', 'inferno', 'jet', 'magenta', 'magma', 'mint', 'orrd', 'oranges', 'oryel', 'peach', 'pinkyl', 'plasma', 'plotly3', 'pubu', 'pubugn', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu', 'rdpu', 'redor', 'reds', 'sunset', 'sunsetdark', 'teal', 'tealgrn', 'turbo', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr', 'ylorrd', 'algae', 'amp', 'deep', 'dense', 'gray', 'haline', 'ice', 'matter', 'solar', 'speed', 'tempo', 'thermal', 'turbid', 'armyrose', 'brbg', 'earth', 'fall', 'geyser', 'prgn', 'piyg', 'picnic', 'portland', 'puor', 'rdgy', 'rdylbu', 'rdylgn', 'spectral', 'tealrose', 'temps', 'tropic', 'balance', 'curl', 'delta', 'oxy', 'edge', 'hsv', 'icefire', 'phase', 'twilight', 'mrybm', 'mygbm']

@st.cache_data
def load_solutions(solution_dir):
    solutions = []
    load_logs = []
    metadata = []  # Store diffusion type, Ly, filename

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
        X_sol = sol['X'][:, 0]  # x_coords
        Y_sol = sol['Y'][0, :] * (ly_target / sol['params']['Ly'])  # scaled y_coords
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

def plot_solution(solution, time_index, downsample, title_suffix="", cu_colormap='viridis', ni_colormap='magma'):
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
    c1 = solution['c1_preds'][time_index][np.ix_(y_indices, x_indices)]  # rows=y, cols=x
    c2 = solution['c2_preds'][time_index][np.ix_(y_indices, x_indices)]

    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Cu @ {t_val:.1f}s", f"Ni @ {t_val:.1f}s"))

    fig.add_trace(go.Heatmap(
        x=x_ds, y=y_ds, z=c1, colorscale=cu_colormap,
        colorbar=dict(title='Cu Conc', x=0.45), zsmooth='best'
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        x=x_ds, y=y_ds, z=c2, colorscale=ni_colormap,
        colorbar=dict(title='Ni Conc', x=1.02), zsmooth='best'
    ), row=1, col=2)

    for col, xref, yref in [(1, 'x', 'y'), (2, 'x2', 'y2')]:
        for x in x_ds[::2]:
            fig.add_shape(type='line', x0=x, y0=0, x1=x, y1=Ly, xref=xref, yref=yref,
                          line=dict(color='gray', width=0.5, dash='dot'))
        for y in y_ds[::2]:
            fig.add_shape(type='line', x0=0, y0=y, x1=Lx, y1=y, xref=xref, yref=yref,
                          line=dict(color='gray', width=0.5, dash='dot'))
        fig.add_shape(type='line', x0=0, y0=Ly, x1=Lx, y1=Ly, xref=xref, yref=yref,
                      line=dict(color='black', width=2))
        fig.add_shape(type='rect', x0=0, y0=0, x1=Lx, y1=Ly, xref=xref, yref=yref,
                      line=dict(color='black', width=1))

    fig.update_layout(
        height=500,
        title=f"Concentration Fields: {Lx}μm × {Ly}μm {title_suffix}",
        showlegend=False,
        template='plotly_white'
    )
    fig.update_xaxes(title_text="x (μm)", range=[0, Lx], gridcolor='white', zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text="y (μm)", range=[0, Ly], gridcolor='white', zeroline=False, row=1, col=1)
    fig.update_xaxes(title_text="x (μm)", range=[0, Lx], gridcolor='white', zeroline=False, row=1, col=2)
    fig.update_yaxes(title_text="y (μm)", range=[0, Ly], gridcolor='white', zeroline=False, row=1, col=2)

    st.plotly_chart(fig, use_container_width=True)

def plot_flux_comparison(solutions, diff_type, ly_values, time_index, downsample):
    """Plot flux fields for two Ly values for a given diffusion type."""
    if len(ly_values) != 2:
        st.error("Please select exactly two Ly values for comparison.")
        return

    sol1 = load_and_interpolate_solution(solutions, diff_type, ly_values[0])
    sol2 = load_and_interpolate_solution(solutions, diff_type, ly_values[1])
    if not sol1 or not sol2:
        st.error(f"Could not load solutions for {diff_type}, Ly={ly_values}")
        return

    fig = make_subplots(
        rows=3, cols=4,
        subplot_titles=(
            f"Cu Flux Quiver, Ly={ly_values[0]:.1f}", f"Ni Flux Quiver, Ly={ly_values[0]:.1f}",
            f"Cu Flux Quiver, Ly={ly_values[1]:.1f}", f"Ni Flux Quiver, Ly={ly_values[1]:.1f}",
            f"Cu J_1x, Ly={ly_values[0]:.1f}", f"Ni J_2x, Ly={ly_values[0]:.1f}",
            f"Cu J_1x, Ly={ly_values[1]:.1f}", f"Ni J_2x, Ly={ly_values[1]:.1f}",
            f"Cu J_1y, Ly={ly_values[0]:.1f}", f"Ni J_2y, Ly={ly_values[0]:.1f}",
            f"Cu J_1y, Ly={ly_values[1]:.1f}", f"Ni J_2y, Ly={ly_values[1]:.1f}"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    for idx, (sol, Ly) in enumerate([(sol1, ly_values[0]), (sol2, ly_values[1])]):
        x_coords = sol['X'][:, 0]
        y_coords = sol['Y'][0, :]
        t_val = sol['times'][time_index]
        Lx = sol['params']['Lx']

        ds = max(1, downsample)
        x_indices = np.unique(np.linspace(0, len(x_coords)-1, num=len(x_coords)//ds, dtype=int))
        y_indices = np.unique(np.linspace(0, len(y_coords)-1, num=len(y_coords)//ds, dtype=int))

        x_ds = x_coords[x_indices]
        y_ds = y_coords[y_indices]
        X_ds, Y_ds = np.meshgrid(x_ds, y_ds, indexing='ij')

        J1_x = sol['J1_preds'][time_index][0][np.ix_(y_indices, x_indices)]
        J1_y = sol['J1_preds'][time_index][1][np.ix_(y_indices, x_indices)]
        J2_x = sol['J2_preds'][time_index][0][np.ix_(y_indices, x_indices)]
        J2_y = sol['J2_preds'][time_index][1][np.ix_(y_indices, x_indices)]
        c1 = sol['c1_preds'][time_index][np.ix_(y_indices, x_indices)]
        c2 = sol['c2_preds'][time_index][np.ix_(y_indices, x_indices)]

        # Quiver plots
        J1_magnitude = np.sqrt(J1_x**2 + J1_y**2)
        max_J1 = np.max(J1_magnitude) + 1e-9
        fig.add_trace(go.Heatmap(
            x=x_ds, y=y_ds, z=np.log10(np.maximum(J1_magnitude, 1e-10)),
            colorscale='viridis', colorbar=dict(title='Log Cu Flux Mag', x=0.22+0.55*idx, len=0.3, y=0.85),
            zsmooth='best', hovertemplate='x: %{x:.1f} μm<br>y: %{y:.1f} μm<br>Flux: %{z:.2e}'
        ), row=1, col=1+idx*2)

        scale = 0.1 * Lx
        annotations_cu = []
        for i in range(0, len(x_ds), 2):
            for j in range(0, len(y_ds), 2):
                if J1_magnitude[j, i] > 1e-10:  # Adjust index for rows=y, cols=x
                    annotations_cu.append(dict(
                        x=x_ds[i], y=y_ds[j], xref=f"x{1+idx*2}", yref=f"y{1+idx*2}",
                        ax=x_ds[i] + scale * J1_x[j, i] / max_J1, ay=y_ds[j] + scale * J1_y[j, i] / max_J1,
                        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.2, arrowcolor='white'
                    ))

        fig.add_trace(go.Contour(
            z=c1, x=x_ds, y=y_ds, colorscale='blues', showscale=False, opacity=0.3,
            contours=dict(showlabels=True, start=np.min(c1), end=np.max(c1), size=(np.max(c1)-np.min(c1))/8),
            line=dict(width=1)
        ), row=1, col=1+idx*2)

        J2_magnitude = np.sqrt(J2_x**2 + J2_y**2)
        max_J2 = np.max(J2_magnitude) + 1e-9
        fig.add_trace(go.Heatmap(
            x=x_ds, y=y_ds, z=np.log10(np.maximum(J2_magnitude, 1e-10)),
            colorscale='cividis', colorbar=dict(title='Log Ni Flux Mag', x=0.45+0.55*idx, len=0.3, y=0.85),
            zsmooth='best', hovertemplate='x: %{x:.1f} μm<br>y: %{y:.1f} μm<br>Flux: %{z:.2e}'
        ), row=1, col=2+idx*2)

        annotations_ni = []
        for i in range(0, len(x_ds), 2):
            for j in range(0, len(y_ds), 2):
                if J2_magnitude[j, i] > 1e-10:
                    annotations_ni.append(dict(
                        x=x_ds[i], y=y_ds[j], xref=f"x{2+idx*2}", yref=f"y{2+idx*2}",
                        ax=x_ds[i] + scale * J2_x[j, i] / max_J2, ay=y_ds[j] + scale * J2_y[j, i] / max_J2,
                        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.2, arrowcolor='white'
                    ))

        fig.add_trace(go.Contour(
            z=c2, x=x_ds, y=y_ds, colorscale='reds', showscale=False, opacity=0.3,
            contours=dict(showlabels=True, start=np.min(c2), end=np.max(c2), size=(np.max(c2)-np.min(c2))/8),
            line=dict(width=1)
        ), row=1, col=2+idx*2)

        # J_x plots
        fig.add_trace(go.Heatmap(
            x=x_ds, y=y_ds, z=J1_x, colorscale='rdbu', zmid=0,
            colorbar=dict(title='Cu J_1x', x=0.22+0.55*idx, len=0.3, y=0.5),
            zsmooth='best', hovertemplate='x: %{x:.1f} μm<br>y: %{y:.1f} μm<br>J_1x: %{z:.2e}'
        ), row=2, col=1+idx*2)

        fig.add_trace(go.Contour(
            z=c1, x=x_ds, y=y_ds, colorscale='blues', showscale=False, opacity=0.3,
            contours=dict(showlabels=True, start=np.min(c1), end=np.max(c1), size=(np.max(c1)-np.min(c1))/8),
            line=dict(width=1)
        ), row=2, col=1+idx*2)

        fig.add_trace(go.Heatmap(
            x=x_ds, y=y_ds, z=J2_x, colorscale='rdbu', zmid=0,
            colorbar=dict(title='Ni J_2x', x=0.45+0.55*idx, len=0.3, y=0.5),
            zsmooth='best', hovertemplate='x: %{x:.1f} μm<br>y: %{y:.1f} μm<br>J_2x: %{z:.2e}'
        ), row=2, col=2+idx*2)

        fig.add_trace(go.Contour(
            z=c2, x=x_ds, y=y_ds, colorscale='reds', showscale=False, opacity=0.3,
            contours=dict(showlabels=True, start=np.min(c2), end=np.max(c2), size=(np.max(c2)-np.min(c2))/8),
            line=dict(width=1)
        ), row=2, col=2+idx*2)

        # J_y plots
        fig.add_trace(go.Heatmap(
            x=x_ds, y=y_ds, z=J1_y, colorscale='rdbu', zmid=0,
            colorbar=dict(title='Cu J_1y', x=0.22+0.55*idx, len=0.3, y=0.15),
            zsmooth='best', hovertemplate='x: %{x:.1f} μm<br>y: %{y:.1f} μm<br>J_1y: %{z:.2e}'
        ), row=3, col=1+idx*2)

        fig.add_trace(go.Contour(
            z=c1, x=x_ds, y=y_ds, colorscale='blues', showscale=False, opacity=0.3,
            contours=dict(showlabels=True, start=np.min(c1), end=np.max(c1), size=(np.max(c1)-np.min(c1))/8),
            line=dict(width=1)
        ), row=3, col=1+idx*2)

        fig.add_trace(go.Heatmap(
            x=x_ds, y=y_ds, z=J2_y, colorscale='rdbu', zmid=0,
            colorbar=dict(title='Ni J_2y', x=0.45+0.55*idx, len=0.3, y=0.15),
            zsmooth='best', hovertemplate='x: %{x:.1f} μm<br>y: %{y:.1f} μm<br>J_2y: %{z:.2e}'
        ), row=3, col=2+idx*2)

        fig.add_trace(go.Contour(
            z=c2, x=x_ds, y=y_ds, colorscale='reds', showscale=False, opacity=0.3,
            contours=dict(showlabels=True, start=np.min(c2), end=np.max(c2), size=(np.max(c2)-np.min(c2))/8),
            line=dict(width=1)
        ), row=3, col=2+idx*2)

        for row, col, xref, yref in [
            (1, 1+idx*2, f'x{1+idx*2}', f'y{1+idx*2}'), (1, 2+idx*2, f'x{2+idx*2}', f'y{2+idx*2}'),
            (2, 1+idx*2, f'x{5+idx*2}', f'y{5+idx*2}'), (2, 2+idx*2, f'x{6+idx*2}', f'y{6+idx*2}'),
            (3, 1+idx*2, f'x{9+idx*2}', f'y{9+idx*2}'), (3, 2+idx*2, f'x{10+idx*2}', f'y{10+idx*2}')
        ]:
            for x in x_ds[::2]:
                fig.add_shape(type='line', x0=x, y0=0, x1=x, y1=Ly, xref=xref, yref=yref,
                              line=dict(color='gray', width=0.5, dash='dot'))
            for y in y_ds[::2]:
                fig.add_shape(type='line', x0=0, y0=y, x1=Lx, y1=y, xref=xref, yref=yref,
                              line=dict(color='gray', width=0.5, dash='dot'))
            fig.add_shape(type='line', x0=0, y0=Ly, x1=Lx, y1=Ly, xref=xref, yref=yref,
                          line=dict(color='black', width=2))
            fig.add_shape(type='rect', x0=0, y0=0, x1=Lx, y1=Ly, xref=xref, yref=yref,
                          line=dict(color='black', width=1))

    fig.update_layout(
        height=1000,
        margin=dict(l=20, r=20, t=100, b=20),
        title=f"Flux Fields Comparison: {diff_type.replace('_', ' ')} @ t={t_val:.1f}s",
        annotations=annotations_cu + annotations_ni,
        showlegend=False,
        template='plotly_white'
    )

    for row, col, xref in [(1,1,'x1'), (1,2,'x2'), (1,3,'x3'), (1,4,'x4'),
                           (2,1,'x5'), (2,2,'x6'), (2,3,'x7'), (2,4,'x8'),
                           (3,1,'x9'), (3,2,'x10'), (3,3,'x11'), (3,4,'x12')]:
        fig.update_xaxes(title_text="x (μm)", range=[0, Lx], gridcolor='white', zeroline=False, row=row, col=col)
        fig.update_yaxes(title_text="y (μm)", range=[0, max(ly_values)], gridcolor='white', zeroline=False, row=row, col=col)

    st.plotly_chart(fig, use_container_width=True)

def plot_line_comparison(solutions, diff_type, ly_values, time_index):
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=300)

    for sol, Ly, linestyle in [(sol1, ly_values[0], '-'), (sol2, ly_values[1], '--')]:
        x_idx = len(sol['X'][:, 0]) // 2  # x = Lx/2
        y_coords = sol['Y'][0, :]
        c1_center = sol['c1_preds'][time_index][:, x_idx]
        c2_center = sol['c2_preds'][time_index][:, x_idx]
        t_val = sol['times'][time_index]

        ax1.plot(c1_center, y_coords, label=f'Ly = {Ly:.1f} μm', linestyle=linestyle, linewidth=2)
        ax2.plot(c2_center, y_coords, label=f'Ly = {Ly:.1f} μm', linestyle=linestyle, linewidth=2)

    ax1.set_xlabel('Cu Concentration (mol/cm³)', fontsize=14)
    ax1.set_ylabel('y (μm)', fontsize=14)
    ax1.set_title(f'Cu @ x=30μm, t={t_val:.1f}s', fontsize=16)
    ax1.legend(fontsize=12)

    ax2.set_xlabel('Ni Concentration (mol/cm³)', fontsize=14)
    ax2.set_ylabel('y (μm)', fontsize=14)
    ax2.set_title(f'Ni @ x=30μm, t={t_val:.1f}s', fontsize=16)
    ax2.legend(fontsize=12)

    plt.suptitle(f"Central Line Profiles: {diff_type.replace('_', ' ')}", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    st.pyplot(fig)
    plt.close()

def compute_center_concentrations(solutions, diff_type, ly_values):
    """Compute Cu and Ni concentrations and flux magnitudes at the center point for given Ly values."""
    center_concentrations = []
    center_idx = 25  # Approximate center for 50x50 grid

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

def plot_center_concentrations(center_concentrations, diff_type):
    """Plot center point concentrations and flux magnitudes for two Ly values."""
    sns.set_context("paper")
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(12, 12), dpi=300)
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    colors = plt.cm.tab20([0, 0.5])

    for conc, color in zip(center_concentrations, colors):
        label = f'Ly = {conc["Ly"]:.1f} μm'
        ax1.plot(conc['times'], conc['c1_center'], label=label, linewidth=2, color=color)
        ax2.plot(conc['times'], conc['c2_center'], label=label, linewidth=2, color=color)
        ax3.plot(conc['times'], conc['J1_mag_center'], label=label, linewidth=2, color=color)
        ax4.plot(conc['times'], conc['J2_mag_center'], label=label, linewidth=2, color=color)

    ax1.set_xlabel('Time (s)', fontsize=14)
    ax1.set_ylabel('Cu Concentration (mol/cm³)', fontsize=14)
    ax1.set_title('Cu Concentration at Center', fontsize=16)
    ax1.legend(fontsize=12)

    ax2.set_xlabel('Time (s)', fontsize=14)
    ax2.set_ylabel('Ni Concentration (mol/cm³)', fontsize=14)
    ax2.set_title('Ni Concentration at Center', fontsize=16)
    ax2.legend(fontsize=12)

    ax3.set_xlabel('Time (s)', fontsize=14)
    ax3.set_ylabel('Cu Flux Magnitude', fontsize=14)
    ax3.set_title('Cu Flux Magnitude at Center', fontsize=16)
    ax3.legend(fontsize=12)
    ax3.set_yscale('log') if np.any(conc['J1_mag_center'] > 0) else None

    ax4.set_xlabel('Time (s)', fontsize=14)
    ax4.set_ylabel('Ni Flux Magnitude', fontsize=14)
    ax4.set_title('Ni Flux Magnitude at Center', fontsize=16)
    ax4.legend(fontsize=12)
    ax4.set_yscale('log') if np.any(conc['J2_mag_center'] > 0) else None

    plt.suptitle(f"Center Point Evolution: {diff_type.replace('_', ' ')} (x=30μm, y=Ly/2)", fontsize=18)
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
    st.title("Cross-Diffusion 2D Visualization with Ly Comparison")

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

    # Sidebar
    st.sidebar.header("Simulation Parameters")
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
    downsample = st.sidebar.slider("Detail Level", 1, 5, 2)

    st.sidebar.header("Visualization Options")
    cu_colormap = st.sidebar.selectbox("Cu Colormap", options=COLORSCALES, index=COLORSCALES.index('viridis'))
    ni_colormap = st.sidebar.selectbox("Ni Colormap", options=COLORSCALES, index=COLORSCALES.index('magma'))

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Concentration", "Flux Comparison", "Central Line Comparison", "Center Point Comparison"])

    with tab1:
        st.subheader("Concentration Fields")
        for ly in ly_values:
            solution = load_and_interpolate_solution(solutions, diff_type, ly)
            if solution:
                plot_solution(solution, time_index, downsample, title_suffix=f"[{diff_type.replace('_', ' ')}, Ly={ly:.1f}]", cu_colormap=cu_colormap, ni_colormap=ni_colormap)
            else:
                st.error(f"No solution for {diff_type}, Ly={ly:.1f}")

    with tab2:
        st.subheader("Flux Fields Comparison")
        plot_flux_comparison(solutions, diff_type, ly_values, time_index, downsample)

    with tab3:
        st.subheader("Central Line Profiles Comparison")
        plot_line_comparison(solutions, diff_type, ly_values, time_index)

    with tab4:
        st.subheader("Center Point Concentration and Flux Magnitude Comparison")
        center_concentrations = compute_center_concentrations(solutions, diff_type, ly_values)
        if center_concentrations:
            plot_center_concentrations(center_concentrations, diff_type)
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
