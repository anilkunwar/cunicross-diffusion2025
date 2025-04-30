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

SOLUTION_DIR = "./pinn_solutions"

@st.cache_data
def load_solutions(solution_dir):
    solutions = []
    load_logs = []
    lys = []
    c_cus = []
    c_nis = []
    for fname in os.listdir(solution_dir):
        if fname.endswith(".pkl"):
            try:
                with open(os.path.join(solution_dir, fname), "rb") as f:
                    sol = pickle.load(f)
                if all(key in sol for key in ['params', 'X', 'Y', 'c1_preds', 'c2_preds', 'times']):
                    if 'orientation_note' in sol:
                        log_message = f"{fname}: Loaded with orientation note: {sol['orientation_note']}"
                    else:
                        log_message = f"{fname}: Loaded, no orientation note. Assuming c1_preds/c2_preds rows=y, columns=x (transposed)."
                    solutions.append(sol)
                    lys.append(sol['params']['Ly'])
                    c_cus.append(sol['params']['C_Cu'])
                    c_nis.append(sol['params']['C_Ni'])
                    load_logs.append(log_message)
                else:
                    load_logs.append(f"{fname}: Failed to load - missing required keys.")
            except Exception as e:
                load_logs.append(f"{fname}: Failed to load - {str(e)}.")
    return solutions, lys, c_cus, c_nis, load_logs

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

def attention_weighted_interpolation(solutions, lys, c_cus, c_nis, ly_target, c_cu_target, c_ni_target, sigma_ly=0.2, sigma_ccu=0.2, sigma_cni=0.2):
    """Attention-based interpolation across Ly, C_Cu, and C_Ni."""
    lys = np.array(lys)
    c_cus = np.array(c_cus)
    c_nis = np.array(c_nis)
    
    # Normalize coordinates
    ly_norm = (lys - 30.0) / (120.0 - 30.0)
    c_cu_norm = (c_cus - 1.5e-3) / (2.9e-3 - 1.5e-3)
    c_ni_norm = (c_nis - 4.0e-4) / (1.8e-3 - 4.0e-4)
    
    target_ly_norm = (ly_target - 30.0) / (120.0 - 30.0)
    target_c_cu_norm = (c_cu_target - 1.5e-3) / (2.9e-3 - 1.5e-3)
    target_c_ni_norm = (c_ni_target - 4.0e-4) / (1.8e-3 - 4.0e-4)
    
    # Solution coordinates in normalized 3D space
    solution_coords = np.stack([ly_norm, c_cu_norm, c_ni_norm], axis=1)
    target_coord = np.array([[target_ly_norm, target_c_cu_norm, target_c_ni_norm]])
    
    # Compute distances in normalized space
    distances = cdist(target_coord, solution_coords).flatten()
    # Scale distances by sigmas
    scaled_distances = np.sqrt(
        ((ly_norm - target_ly_norm) / sigma_ly)**2 +
        ((c_cu_norm - target_c_cu_norm) / sigma_ccu)**2 +
        ((c_ni_norm - target_c_ni_norm) / sigma_cni)**2
    )
    weights = np.exp(-scaled_distances**2 / 2)
    weights /= weights.sum()
    
    Lx = solutions[0]['params']['Lx']
    t_max = solutions[0]['params']['t_max']
    x_coords = np.linspace(0, Lx, 50)
    y_coords = np.linspace(0, ly_target, 50)
    times = np.linspace(0, t_max, 50)
    
    c1_interp = np.zeros((len(times), 50, 50))
    c2_interp = np.zeros((len(times), 50, 50))
    
    for idx, (weight, solution) in enumerate(zip(weights, solutions)):
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
    
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    param_set = solutions[0]['params'].copy()
    param_set['Ly'] = ly_target
    param_set['C_Cu'] = c_cu_target
    param_set['C_Ni'] = c_ni_target
    
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
        'interpolated': True,
        'attention_weights': weights.tolist(),
        'used_lys': lys.tolist(),
        'used_c_cus': c_cus.tolist(),
        'used_c_nis': c_nis.tolist(),
        'orientation_note': "c1_preds and c2_preds are arrays of shape (50,50) where rows (i) correspond to y-coordinates and columns (j) correspond to x-coordinates due to transpose."
    }

@st.cache_data
def load_and_interpolate_solution(solutions, lys, c_cus, c_nis, ly_target, c_cu_target, c_ni_target, tolerance_ly=0.1, tolerance_c=1e-5):
    """Load or interpolate solution for target Ly, C_Cu, C_Ni."""
    lys = np.array(lys)
    c_cus = np.array(c_cus)
    c_nis = np.array(c_nis)
    
    # Check for exact match
    matches = (
        (np.abs(lys - ly_target) < tolerance_ly) &
        (np.abs(c_cus - c_cu_target) < tolerance_c) &
        (np.abs(c_nis - c_ni_target) < tolerance_c)
    )
    exact_match_indices = np.where(matches)[0]
    
    if exact_match_indices.size > 0:
        solution = solutions[exact_match_indices[0]]
        solution['interpolated'] = False
        if 'J1_preds' in solution and isinstance(solution['J1_preds'][0], tuple):
            solution['J1_preds'] = [[J_x, J_y] for J_x, J_y in solution['J1_preds']]
        if 'J2_preds' in solution and isinstance(solution['J2_preds'][0], tuple):
            solution['J2_preds'] = [[J_x, J_y] for J_x, J_y in solution['J2_preds']]
        if 'J1_preds' not in solution or 'J2_preds' not in solution:
            J1_preds, J2_preds = compute_fluxes(
                solution['c1_preds'], solution['c2_preds'],
                solution['X'][:,0], solution['Y'][0,:], solution['params']
            )
            solution['J1_preds'] = J1_preds
            solution['J2_preds'] = J2_preds
        return solution
    else:
        return attention_weighted_interpolation(solutions, lys, c_cus, c_nis, ly_target, c_cu_target, c_ni_target)

@st.cache_data
def compute_center_concentrations(solutions, lys, c_cus, c_nis):
    """Compute Cu and Ni concentrations at the center point for all solutions."""
    center_concentrations = []
    Lx = solutions[0]['params']['Lx']
    center_x = Lx / 2  # 30.0 μm
    center_idx = 25  # Closest grid index for x=30.0, y=Ly/2 (50x50 grid)
    
    for sol, ly, c_cu, c_ni in zip(solutions, lys, c_cus, c_nis):
        center_y = ly / 2
        times = sol['times']
        c1_center = []
        c2_center = []
        for c1, c2 in zip(sol['c1_preds'], sol['c2_preds']):
            c1_center.append(c1[center_idx, center_idx])  # c1_preds: rows=y, cols=x
            c2_center.append(c2[center_idx, center_idx])
        center_concentrations.append({
            'Ly': ly,
            'C_Cu': c_cu,
            'C_Ni': c_ni,
            'times': times,
            'c1_center': np.array(c1_center),
            'c2_center': np.array(c2_center)
        })
    
    return center_concentrations

def compute_interpolated_center_concentrations(solution, ly_target):
    """Compute interpolated Cu and Ni concentrations at the center point."""
    Lx = solution['params']['Lx']
    center_x = Lx / 2  # 30.0 μm
    center_y = ly_target / 2
    times = solution['times']
    
    X = solution['X'][:, 0]
    Y = solution['Y'][0, :]
    
    c1_center = []
    c2_center = []
    for t_idx, c1, c2 in zip(range(len(times)), solution['c1_preds'], solution['c2_preds']):
        interp_c1 = RegularGridInterpolator(
            (X, Y), c1,
            method='linear', bounds_error=False, fill_value=None
        )
        interp_c2 = RegularGridInterpolator(
            (X, Y), c2,
            method='linear', bounds_error=False, fill_value=None
        )
        point = np.array([[center_x, center_y]])
        c1_center.append(interp_c1(point)[0])
        c2_center.append(interp_c2(point)[0])
    
    return np.array(c1_center), np.array(c2_center)

def plot_center_concentrations(center_concentrations, selected_params, solution, ly_target, c_cu_target, c_ni_target, is_interpolated):
    """Generate publication-quality plots for center point concentrations."""
    plt.style.use('ggplot')
    colors = plt.cm.tab20(np.linspace(0, 1, len(center_concentrations)))
    
    # Cu Plot
    fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    for conc, color in zip(center_concentrations, colors):
        param_key = (conc['Ly'], conc['C_Cu'], conc['C_Ni'])
        if param_key in selected_params:
            label = f'Ly={conc["Ly"]:.1f}, C_Cu={conc["C_Cu"]:.1e}, C_Ni={conc["C_Ni"]:.1e}'
            ax1.plot(conc['times'], conc['c1_center'], label=label, linewidth=1.5, color=color)
    
    if is_interpolated:
        c1_center, _ = compute_interpolated_center_concentrations(solution, ly_target)
        label = f'ML Predicted (Ly={ly_target:.1f}, C_Cu={c_cu_target:.1e}, C_Ni={c_ni_target:.1e})'
        ax1.plot(solution['times'], c1_center, label=label, linewidth=2.5, color='black', linestyle='--')
    
    ax1.set_xlabel('Time (s)', fontsize=14)
    ax1.set_ylabel('Cu Concentration (mol/cc)', fontsize=14)
    ax1.set_title(f'Cu Concentration at Center (x={solution["params"]["Lx"]/2:.1f} μm, y=Ly/2) vs. Time', fontsize=16)
    ax1.grid(True, which="both", ls="--", alpha=0.7)
    ax1.legend(fontsize=10, loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    
    # Ni Plot
    fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=300)
    for conc, color in zip(center_concentrations, colors):
        param_key = (conc['Ly'], conc['C_Cu'], conc['C_Ni'])
        if param_key in selected_params:
            label = f'Ly={conc["Ly"]:.1f}, C_Cu={conc["C_Cu"]:.1e}, C_Ni={conc["C_Ni"]:.1e}'
            ax2.plot(conc['times'], conc['c2_center'], label=label, linewidth=1.5, color=color)
    
    if is_interpolated:
        _, c2_center = compute_interpolated_center_concentrations(solution, ly_target)
        label = f'ML Predicted (Ly={ly_target:.1f}, C_Cu={c_cu_target:.1e}, C_Ni={c_ni_target:.1e})'
        ax2.plot(solution['times'], c2_center, label=label, linewidth=2.5, color='black', linestyle='--')
    
    ax2.set_xlabel('Time (s)', fontsize=14)
    ax2.set_ylabel('Ni Concentration (mol/cc)', fontsize=14)
    ax2.set_title(f'Ni Concentration at Center (x={solution["params"]["Lx"]/2:.1f} μm, y=Ly/2) vs. Time', fontsize=16)
    ax2.grid(True, which="both", ls="--", alpha=0.7)
    ax2.legend(fontsize=10, loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    
    return fig1, fig2

def plot_solution(solution, time_index, downsample):
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
    c1 = solution['c1_preds'][time_index][np.ix_(x_indices, y_indices)]
    c2 = solution['c2_preds'][time_index][np.ix_(x_indices, y_indices)]
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Cu @ {t_val:.1f}s", f"Ni @ {t_val:.1f}s"))
    
    fig.add_trace(go.Heatmap(
        x=x_ds,
        y=y_ds,
        z=c1,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='Cu Conc (mol/cc)', x=0.45),
        zsmooth='best'
    ), row=1, col=1)
    
    fig.add_trace(go.Heatmap(
        x=x_ds,
        y=y_ds,
        z=c2,
        colorscale='Magma',
        showscale=True,
        colorbar=dict(title='Ni Conc (mol/cc)', x=1.02),
        zsmooth='best'
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
        title=f"Concentration Fields: {Lx}μm × {Ly}μm",
        showlegend=False
    )
    fig.update_xaxes(title_text="x (μm)", range=[0, Lx], gridcolor='white', zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text="y (μm)", range=[0, Ly], gridcolor='white', zeroline=False, row=1, col=1)
    fig.update_xaxes(title_text="x (μm)", range=[0, Lx], gridcolor='white', zeroline=False, row=1, col=2)
    fig.update_yaxes(title_text="y (μm)", range=[0, Ly], gridcolor='white', zeroline=False, row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)

def plot_flux(solution, time_index, downsample):
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
    X_ds, Y_ds = np.meshgrid(x_ds, y_ds, indexing='ij')
    
    J1_x = solution['J1_preds'][time_index][0][np.ix_(x_indices, y_indices)]
    J1_y = solution['J1_preds'][time_index][1][np.ix_(x_indices, y_indices)]
    J2_x = solution['J2_preds'][time_index][0][np.ix_(x_indices, y_indices)]
    J2_y = solution['J2_preds'][time_index][1][np.ix_(x_indices, y_indices)]
    c1 = solution['c1_preds'][time_index][np.ix_(x_indices, y_indices)]
    c2 = solution['c2_preds'][time_index][np.ix_(x_indices, y_indices)]
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            f"Cu Flux Quiver @ {t_val:.1f}s",
            f"Ni Flux Quiver @ {t_val:.1f}s",
            f"Cu Flux J_1x @ {t_val:.1f}s",
            f"Ni Flux J_2x @ {t_val:.1f}s",
            f"Cu Flux J_1y @ {t_val:.1f}s",
            f"Ni Flux J_2y @ {t_val:.1f}s"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    J1_magnitude = np.sqrt(J1_x**2 + J1_y**2)
    max_J1 = np.max(J1_magnitude) + 1e-9
    
    fig.add_trace(go.Heatmap(
        x=x_ds,
        y=y_ds,
        z=np.log10(np.maximum(J1_magnitude, 1e-10)),
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='Log Cu Flux Mag', x=0.45, len=0.3, y=0.85),
        zsmooth='best',
        hovertemplate='x: %{x:.1f} μm<br>y: %{y:.1f} μm<br>Flux: %{z:.2e}'
    ), row=1, col=1)
    
    scale = 0.1 * Lx
    annotations_cu = []
    for i in range(0, len(x_ds), 2):
        for j in range(0, len(y_ds), 2):
            if J1_magnitude[i, j] > 1e-10:
                annotations_cu.append(dict(
                    x=x_ds[i],
                    y=y_ds[j],
                    xref="x",
                    yref="y",
                    ax=x_ds[i] + scale * J1_x[i, j] / max_J1,
                    ay=y_ds[j] + scale * J1_y[i, j] / max_J1,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1.2,
                    arrowcolor='white'
                ))
    
    fig.add_trace(go.Contour(
        z=c1,
        x=x_ds,
        y=y_ds,
        colorscale='Blues',
        showscale=False,
        opacity=0.3,
        contours=dict(
            showlabels=True,
            start=np.min(c1),
            end=np.max(c1),
            size=(np.max(c1)-np.min(c1))/8
        ),
        line=dict(width=1)
    ), row=1, col=1)
    
    J2_magnitude = np.sqrt(J2_x**2 + J2_y**2)
    max_J2 = np.max(J2_magnitude) + 1e-9
    
    fig.add_trace(go.Heatmap(
        x=x_ds,
        y=y_ds,
        z=np.log10(np.maximum(J2_magnitude, 1e-10)),
        colorscale='Cividis',
        showscale=True,
        colorbar=dict(title='Log Ni Flux Mag', x=1.02, len=0.3, y=0.85),
        zsmooth='best',
        hovertemplate='x: %{x:.1f} μm<br>y: %{y:.1f} μm<br>Flux: %{z:.2e}'
    ), row=1, col=2)
    
    annotations_ni = []
    for i in range(0, len(x_ds), 2):
        for j in range(0, len(y_ds), 2):
            if J2_magnitude[i, j] > 1e-10:
                annotations_ni.append(dict(
                    x=x_ds[i],
                    y=y_ds[j],
                    xref="x2",
                    yref="y2",
                    ax=x_ds[i] + scale * J2_x[i, j] / max_J2,
                    ay=y_ds[j] + scale * J2_y[i, j] / max_J2,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1.2,
                    arrowcolor='white'
                ))
    
    fig.add_trace(go.Contour(
        z=c2,
        x=x_ds,
        y=y_ds,
        colorscale='Reds',
        showscale=False,
        opacity=0.3,
        contours=dict(
            showlabels=True,
            start=np.min(c2),
            end=np.max(c2),
            size=(np.max(c2)-np.min(c2))/8
        ),
        line=dict(width=1)
    ), row=1, col=2)
    
    fig.add_trace(go.Heatmap(
        x=x_ds,
        y=y_ds,
        z=J1_x,
        colorscale='RdBu',
        showscale=True,
        colorbar=dict(title='Cu J_1x', x=0.45, len=0.3, y=0.5),
        zsmooth='best',
        zmid=0,
        hovertemplate='x: %{x:.1f} μm<br>y: %{y:.1f} μm<br>J_1x: %{z:.2e}'
    ), row=2, col=1)
    
    fig.add_trace(go.Contour(
        z=c1,
        x=x_ds,
        y=y_ds,
        colorscale='Blues',
        showscale=False,
        opacity=0.3,
        contours=dict(
            showlabels=True,
            start=np.min(c1),
            end=np.max(c1),
            size=(np.max(c1)-np.min(c1))/8
        ),
        line=dict(width=1)
    ), row=2, col=1)
    
    fig.add_trace(go.Heatmap(
        x=x_ds,
        y=y_ds,
        z=J2_x,
        colorscale='RdBu',
        showscale=True,
        colorbar=dict(title='Ni J_2x', x=1.02, len=0.3, y=0.5),
        zsmooth='best',
        zmid=0,
        hovertemplate='x: %{x:.1f} μm<br>y: %{y:.1f} μm<br>J_2x: %{z:.2e}'
    ), row=2, col=2)
    
    fig.add_trace(go.Contour(
        z=c2,
        x=x_ds,
        y=y_ds,
        colorscale='Reds',
        showscale=False,
        opacity=0.3,
        contours=dict(
            showlabels=True,
            start=np.min(c2),
            end=np.max(c2),
            size=(np.max(c2)-np.min(c2))/8
        ),
        line=dict(width=1)
    ), row=2, col=2)
    
    fig.add_trace(go.Heatmap(
        x=x_ds,
        y=y_ds,
        z=J1_y,
        colorscale='RdBu',
        showscale=True,
        colorbar=dict(title='Cu J_1y', x=0.45, len=0.3, y=0.15),
        zsmooth='best',
        zmid=0,
        hovertemplate='x: %{x:.1f} μm<br>y: %{y:.1f} μm<br>J_1y: %{z:.2e}'
    ), row=3, col=1)
    
    fig.add_trace(go.Contour(
        z=c1,
        x=x_ds,
        y=y_ds,
        colorscale='Blues',
        showscale=False,
        opacity=0.3,
        contours=dict(
            showlabels=True,
            start=np.min(c1),
            end=np.max(c1),
            size=(np.max(c1)-np.min(c1))/8
        ),
        line=dict(width=1)
    ), row=3, col=1)
    
    fig.add_trace(go.Heatmap(
        x=x_ds,
        y=y_ds,
        z=J2_y,
        colorscale='RdBu',
        showscale=True,
        colorbar=dict(title='Ni J_2y', x=1.02, len=0.3, y=0.15),
        zsmooth='best',
        zmid=0,
        hovertemplate='x: %{x:.1f} μm<br>y: %{y:.1f} μm<br>J_2y: %{z:.2e}'
    ), row=3, col=2)
    
    fig.add_trace(go.Contour(
        z=c2,
        x=x_ds,
        y=y_ds,
        colorscale='Reds',
        showscale=False,
        opacity=0.3,
        contours=dict(
            showlabels=True,
            start=np.min(c2),
            end=np.max(c2),
            size=(np.max(c2)-np.min(c2))/8
        ),
        line=dict(width=1)
    ), row=3, col=2)
    
    for row, col, xref, yref in [
        (1, 1, 'x', 'y'), (1, 2, 'x2', 'y2'),
        (2, 1, 'x3', 'y3'), (2, 2, 'x4', 'y4'),
        (3, 1, 'x5', 'y5'), (3, 2, 'x6', 'y6')
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
        title=f"Flux Fields: {Lx}μm × {Ly}μm",
        annotations=annotations_cu + annotations_ni,
        showlegend=False
    )
    
    for row, col, xref in [(1, 1, 'x'), (1, 2, 'x2'), (2, 1, 'x3'), (2, 2, 'x4'), (3, 1, 'x5'), (3, 2, 'x6')]:
        fig.update_xaxes(title_text="x (μm)", range=[0, Lx], gridcolor='white', zeroline=False, row=row, col=col)
        fig.update_yaxes(title_text="y (μm)", range=[0, Ly], gridcolor='white', zeroline=False, row=row, col=col)
    
    st.plotly_chart(fig, use_container_width=True)

def download_data(solution, time_index, all_times=False):
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
            'x': X.flatten(),
            'y': Y.flatten(),
            'c1': c1.flatten(),
            'c2': c2.flatten(),
            'J1_x': J1_x.flatten(),
            'J1_y': J1_y.flatten(),
            'J2_x': J2_x.flatten(),
            'J2_y': J2_y.flatten()
        })
        
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode('utf-8')
        return csv_bytes, f"data_t_{t_val:.1f}s.csv"
    
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
                    'x': X.flatten(),
                    'y': Y.flatten(),
                    'c1': c1.flatten(),
                    'c2': c2.flatten(),
                    'J1_x': J1_x.flatten(),
                    'J1_y': J1_y.flatten(),
                    'J2_x': J2_x.flatten(),
                    'J2_y': J2_y.flatten()
                })
                
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                zip_file.writestr(f"data_t_{t_val:.1f}s.csv", csv_buffer.getvalue())
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue(), "data_all_times.zip"

def main():
    st.title("Cross-Diffusion 2D Visualization with Interpolation")
    
    solutions, lys, c_cus, c_nis, load_logs = load_solutions(SOLUTION_DIR)
    
    if load_logs:
        st.subheader("Solution Load Log")
        selected_log = st.selectbox("View load status for solutions", load_logs, index=0)
        st.write(selected_log)
    else:
        st.warning("No solution files found in pinn_solutions directory.")
    
    if not solutions:
        st.error("No valid solution files found in pinn_solutions directory.")
        return
    
    st.subheader("Select Parameters")
    ly_target = st.number_input(
        "Domain Height (Ly, μm)",
        min_value=30.0,
        max_value=120.0,
        value=60.0,
        step=0.1,
        format="%.1f"
    )
    c_cu_target = st.number_input(
        "Cu Boundary Concentration (mol/cc)",
        min_value=1.5e-3,
        max_value=2.9e-3,
        value=2.0e-3,
        step=0.1e-3,
        format="%.1e"
    )
    c_ni_target = st.number_input(
        "Ni Boundary Concentration (mol/cc)",
        min_value=4.0e-4,
        max_value=1.8e-3,
        value=1.0e-3,
        step=0.1e-4,
        format="%.1e"
    )
    
    try:
        solution = load_and_interpolate_solution(solutions, lys, c_cus, c_nis, ly_target, c_cu_target, c_ni_target)
    except Exception as e:
        st.error(f"Failed to load or interpolate solution: {str(e)}")
        return
    
    st.subheader("Solution Details")
    st.write(f"Ly = {solution['params']['Ly']:.1f} μm")
    st.write(f"C_Cu = {solution['params']['C_Cu']:.1e} mol/cc")
    st.write(f"C_Ni = {solution['params']['C_Ni']:.1e} mol/cc")
    if solution.get('interpolated', False):
        st.write("**Status**: Interpolated solution")
        st.write(f"Used Ly values: {', '.join(f'{ly:.1f}' for ly in solution['used_lys'])}")
        st.write(f"Used C_Cu values: {', '.join(f'{c:.1e}' for c in solution['used_c_cus'])}")
        st.write(f"Used C_Ni values: {', '.join(f'{c:.1e}' for c in solution['used_c_nis'])}")
        st.write(f"Attention weights: {', '.join(f'{w:.3f}' for w in solution['attention_weights'])}")
    else:
        st.write("**Status**: Exact solution")
    
    st.info("Note: Training loss plots are available in the pinn_solutions/ directory (e.g., loss_plot_ly_XX.X_ccu_X.Xe-X_cni_X.Xe-X.png).")
    
    max_time = solution['times'][-1]
    time_index = st.slider("Select Time", 0, len(solution['times'])-1, len(solution['times'])-1)
    downsample = st.slider("Detail Level", 1, 5, 2)
    
    show_flux = st.checkbox("Show Flux Fields", value=False)
    show_center_plots = st.checkbox("Show Center Concentration Plots", value=False)
    
    plot_solution(solution, time_index, downsample)
    
    if show_flux:
        if 'J1_preds' not in solution or 'J2_preds' not in solution:
            st.error("Flux data not available in the selected solution.")
        else:
            plot_flux(solution, time_index, downsample)
    
    if show_center_plots:
        st.subheader("Center Point Concentration Plots")
        center_concentrations = compute_center_concentrations(solutions, lys, c_cus, c_nis)
        available_params = [(conc['Ly'], conc['C_Cu'], conc['C_Ni']) for conc in center_concentrations]
        param_labels = [f"Ly={ly:.1f}, C_Cu={c_cu:.1e}, C_Ni={c_ni:.1e}" for ly, c_cu, c_ni in available_params]
        
        selected_labels = st.multiselect(
            "Select parameter combinations to plot",
            options=param_labels,
            default=param_labels[:min(10, len(param_labels))],  # Default to first 10 or fewer
            format_func=lambda x: x
        )
        selected_params = [available_params[param_labels.index(label)] for label in selected_labels]
        
        is_interpolated = solution.get('interpolated', False)
        fig1, fig2 = plot_center_concentrations(
            center_concentrations, selected_params, solution, ly_target, c_cu_target, c_ni_target, is_interpolated
        )
        st.pyplot(fig1)
        st.pyplot(fig2)
        plt.close(fig1)
        plt.close(fig2)
    
    st.subheader("Download Data")
    col1, col2 = st.columns(2)
    
    with col1:
        data_bytes, filename = download_data(solution, time_index, all_times=False)
        st.download_button(
            label="Download Data for Selected Time",
            data=data_bytes,
            file_name=filename,
            mime="text/csv"
        )
    
    with col2:
        data_bytes, filename = download_data(solution, time_index, all_times=True)
        st.download_button(
            label="Download Data for All Times",
            data=data_bytes,
            file_name=filename,
            mime="application/zip"
        )

if __name__ == "__main__":
    main()
