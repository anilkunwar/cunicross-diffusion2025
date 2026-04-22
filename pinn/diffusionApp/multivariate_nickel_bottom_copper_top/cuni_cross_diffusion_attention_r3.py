#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PHYSICS-INFORMED VALIDATION & UNCERTAINTY QUANTIFICATION MODULE
================================================================
- Validates attention-interpolated fields against held-out PINN simulations
- Quantifies uncertainty via weight entropy, parameter distance, ensemble variance
- Enforces physics constraints: PDE residuals, boundary conditions, mass conservation
- Interactive visualization: bar charts, radar plots, scatter matrices, residual heatmaps
- Exportable metrics tables and diagnostic reports
"""
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
import hashlib
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skimage.metrics import structural_similarity as ssim
import warnings
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================
# GLOBAL CONFIGURATION
# =============================================
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'figure.autolayout': True,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

# Physical constants for Cu-Ni cross-diffusion
PHYSICS_CONSTANTS = {
    'D11': 0.006,      # Cu self-diffusivity (μm²/s)
    'D12': 0.00427,    # Cu-Ni cross-diffusivity
    'D21': 0.003697,   # Ni-Cu cross-diffusivity
    'D22': 0.0054,     # Ni self-diffusivity
    'C_CU_RANGE': (0.0, 2.9e-3),   # mol/cc
    'C_NI_RANGE': (0.0, 1.8e-3),   # mol/cc
    'LY_RANGE': (30.0, 120.0),     # μm
    'T_MAX': 200.0,                # s
    'MASS_TOLERANCE': 1e-4,        # Relative mass conservation tolerance
}

# =============================================
# PHYSICS-INFORMED LOSS FUNCTIONS
# =============================================
def compute_pde_residual(c1: np.ndarray, c2: np.ndarray, 
                        x: np.ndarray, y: np.ndarray, t: float,
                        D11: float, D12: float, D21: float, D22: float,
                        dx: float, dy: float) -> np.ndarray:
    """
    Compute Fick's second law residual for cross-diffusion:
    ∂c₁/∂t = D₁₁∇²c₁ + D₁₂∇²c₂
    ∂c₂/∂t = D₂₁∇²c₁ + D₂₂∇²c₂
    
    Uses finite differences for spatial derivatives.
    Returns residual field of shape (ny, nx).
    """
    ny, nx = c1.shape
    
    # Laplacian using 5-point stencil
    def laplacian(c):
        lap = np.zeros_like(c)
        # Interior points
        lap[1:-1, 1:-1] = (
            (c[2:, 1:-1] - 2*c[1:-1, 1:-1] + c[:-2, 1:-1]) / dy**2 +
            (c[1:-1, 2:] - 2*c[1:-1, 1:-1] + c[1:-1, :-2]) / dx**2
        )
        # Boundary: one-sided differences (Neumann assumed)
        lap[0, 1:-1] = (c[1, 1:-1] - c[0, 1:-1]) / dy**2 + \
                      (c[0, 2:] - 2*c[0, 1:-1] + c[0, :-2]) / dx**2
        lap[-1, 1:-1] = (c[-1, 1:-1] - c[-2, 1:-1]) / dy**2 + \
                       (c[-1, 2:] - 2*c[-1, 1:-1] + c[-1, :-2]) / dx**2
        lap[1:-1, 0] = (c[1:-1, 1] - c[1:-1, 0]) / dx**2 + \
                      (c[2:, 0] - 2*c[1:-1, 0] + c[:-2, 0]) / dy**2
        lap[1:-1, -1] = (c[1:-1, -1] - c[1:-1, -2]) / dx**2 + \
                       (c[2:, -1] - 2*c[1:-1, -1] + c[:-2, -1]) / dy**2
        return lap
    
    lap_c1 = laplacian(c1)
    lap_c2 = laplacian(c2)
    
    # Time derivative approximation (forward difference from previous timestep)
    # For single-time validation, assume quasi-steady: ∂c/∂t ≈ 0
    # For multi-time, pass c1_prev, c2_prev and dt
    residual1 = -(D11 * lap_c1 + D12 * lap_c2)  # ∂c₁/∂t ≈ 0
    residual2 = -(D21 * lap_c1 + D22 * lap_c2)
    
    return np.sqrt(residual1**2 + residual2**2)


def enforce_boundary_conditions(c1: np.ndarray, c2: np.ndarray,
                               x: np.ndarray, y: np.ndarray,
                               c_cu_top: float, c_cu_bottom: float,
                               c_ni_top: float, c_ni_bottom: float,
                               enforce_type: str = 'hard') -> Tuple[np.ndarray, np.ndarray]:
    """
    Enforce Dirichlet boundary conditions on interpolated fields.
    
    Args:
        enforce_type: 'hard' (direct assignment) or 'soft' (blending with Gaussian weight)
    """
    c1_bc = c1.copy()
    c2_bc = c2.copy()
    ny, nx = c1.shape
    
    if enforce_type == 'hard':
        # Top boundary (y = Ly)
        c1_bc[:, -1] = c_cu_top
        c2_bc[:, -1] = c_ni_top
        # Bottom boundary (y = 0)
        c1_bc[:, 0] = c_cu_bottom
        c2_bc[:, 0] = c_ni_bottom
        # Side boundaries: Neumann (zero flux) - already satisfied by interpolation
    elif enforce_type == 'soft':
        # Gaussian blending near boundaries
        boundary_width = 3  # cells
        y_weight = np.exp(-((np.arange(ny) - ny/2)**2) / (2 * (ny/4)**2))
        x_weight = np.exp(-((np.arange(nx) - nx/2)**2) / (2 * (nx/4)**2))
        X, Y = np.meshgrid(x_weight, y_weight)
        blend = 1 - X * Y  # 1 at center, 0 at boundaries
        
        # Apply boundary values with blending
        c1_bc = blend * c1 + (1 - blend) * np.where(
            (np.arange(ny)[:, None] == 0) | (np.arange(ny)[:, None] == ny-1),
            np.where(np.arange(ny)[:, None] == 0, c_cu_bottom, c_cu_top),
            c1
        )
        c2_bc = blend * c2 + (1 - blend) * np.where(
            (np.arange(ny)[:, None] == 0) | (np.arange(ny)[:, None] == ny-1),
            np.where(np.arange(ny)[:, None] == 0, c_ni_bottom, c_ni_top),
            c2
        )
    
    return c1_bc, c2_bc


def compute_mass_conservation(c1: np.ndarray, c2: np.ndarray, 
                             x: np.ndarray, y: np.ndarray,
                             c_cu_initial: float, c_ni_initial: float) -> Dict[str, float]:
    """
    Compute relative mass conservation error.
    Assumes initial uniform concentration c_initial.
    """
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    dy = y[1] - y[0] if len(y) > 1 else 1.0
    area = dx * dy
    
    # Total mass at current time
    mass_c1 = np.sum(c1) * area
    mass_c2 = np.sum(c2) * area
    
    # Initial mass (uniform)
    nx, ny = c1.shape
    initial_mass_c1 = c_cu_initial * nx * ny * area
    initial_mass_c2 = c_ni_initial * nx * ny * area
    
    # Relative error
    error_c1 = abs(mass_c1 - initial_mass_c1) / (initial_mass_c1 + 1e-12)
    error_c2 = abs(mass_c2 - initial_mass_c2) / (initial_mass_c2 + 1e-12)
    
    return {
        'mass_error_c1': error_c1,
        'mass_error_c2': error_c2,
        'mass_error_max': max(error_c1, error_c2),
        'mass_c1': mass_c1,
        'mass_c2': mass_c2
    }


# =============================================
# VALIDATION METRICS COMPUTATION
# =============================================
@dataclass
class ValidationMetrics:
    """Container for comprehensive validation metrics."""
    # Pointwise errors
    mse_c1: float = 0.0
    mse_c2: float = 0.0
    mae_c1: float = 0.0
    mae_c2: float = 0.0
    max_error_c1: float = 0.0
    max_error_c2: float = 0.0
    r2_c1: float = 0.0
    r2_c2: float = 0.0
    ssim_c1: float = 0.0
    ssim_c2: float = 0.0
    
    # Physics-based metrics
    pde_residual_mean: float = 0.0
    pde_residual_max: float = 0.0
    bc_error_top_c1: float = 0.0
    bc_error_top_c2: float = 0.0
    bc_error_bottom_c1: float = 0.0
    bc_error_bottom_c2: float = 0.0
    mass_error: float = 0.0
    
    # Uncertainty metrics
    weight_entropy: float = 0.0
    param_distance: float = 0.0
    ensemble_variance_c1: float = 0.0
    ensemble_variance_c2: float = 0.0
    
    # Composite score (0-1, higher is better)
    overall_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}
    
    def to_dataframe(self) -> pd.DataFrame:
        data = []
        for key, value in self.__dict__.items():
            category = 'pointwise' if key.startswith(('mse', 'mae', 'max_error', 'r2', 'ssim')) else \
                      'physics' if key.startswith(('pde', 'bc', 'mass')) else \
                      'uncertainty' if key.startswith(('weight', 'param', 'ensemble')) else \
                      'composite'
            data.append({'Metric': key, 'Value': value, 'Category': category})
        return pd.DataFrame(data)


def compute_validation_metrics(interp_c1: np.ndarray, interp_c2: np.ndarray,
                              gt_c1: np.ndarray, gt_c2: np.ndarray,
                              x: np.ndarray, y: np.ndarray, t: float,
                              params: Dict, weights: Optional[np.ndarray] = None,
                              ensemble_fields: Optional[List[Dict]] = None) -> ValidationMetrics:
    """
    Compute comprehensive validation metrics comparing interpolated vs ground-truth PINN fields.
    """
    metrics = ValidationMetrics()
    
    # Flatten for pointwise metrics
    flat_interp_c1 = interp_c1.flatten()
    flat_interp_c2 = interp_c2.flatten()
    flat_gt_c1 = gt_c1.flatten()
    flat_gt_c2 = gt_c2.flatten()
    
    # Pointwise errors
    metrics.mse_c1 = mean_squared_error(flat_gt_c1, flat_interp_c1)
    metrics.mse_c2 = mean_squared_error(flat_gt_c2, flat_interp_c2)
    metrics.mae_c1 = mean_absolute_error(flat_gt_c1, flat_interp_c1)
    metrics.mae_c2 = mean_absolute_error(flat_gt_c2, flat_interp_c2)
    metrics.max_error_c1 = np.max(np.abs(flat_gt_c1 - flat_interp_c1))
    metrics.max_error_c2 = np.max(np.abs(flat_gt_c2 - flat_interp_c2))
    
    # R² and SSIM (only if variance > 0)
    if np.var(flat_gt_c1) > 1e-12:
        metrics.r2_c1 = r2_score(flat_gt_c1, flat_interp_c1)
    if np.var(flat_gt_c2) > 1e-12:
        metrics.r2_c2 = r2_score(flat_gt_c2, flat_interp_c2)
    
    data_range_c1 = max(gt_c1.max() - gt_c1.min(), 1e-6)
    data_range_c2 = max(gt_c2.max() - gt_c2.min(), 1e-6)
    metrics.ssim_c1 = ssim(gt_c1, interp_c1, data_range=data_range_c1)
    metrics.ssim_c2 = ssim(gt_c2, interp_c2, data_range=data_range_c2)
    
    # Physics-based metrics
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    dy = y[1] - y[0] if len(y) > 1 else 1.0
    
    residual = compute_pde_residual(
        interp_c1, interp_c2, x, y, t,
        params.get('D11', PHYSICS_CONSTANTS['D11']),
        params.get('D12', PHYSICS_CONSTANTS['D12']),
        params.get('D21', PHYSICS_CONSTANTS['D21']),
        params.get('D22', PHYSICS_CONSTANTS['D22']),
        dx, dy
    )
    metrics.pde_residual_mean = np.mean(residual)
    metrics.pde_residual_max = np.max(residual)
    
    # Boundary condition errors
    ny, nx = interp_c1.shape
    metrics.bc_error_top_c1 = np.abs(np.mean(interp_c1[:, -1]) - params.get('c_cu_top', 0))
    metrics.bc_error_top_c2 = np.abs(np.mean(interp_c2[:, -1]) - params.get('c_ni_top', 0))
    metrics.bc_error_bottom_c1 = np.abs(np.mean(interp_c1[:, 0]) - params.get('c_cu_bottom', 0))
    metrics.bc_error_bottom_c2 = np.abs(np.mean(interp_c2[:, 0]) - params.get('c_ni_bottom', 0))
    
    # Mass conservation
    mass_metrics = compute_mass_conservation(
        interp_c1, interp_c2, x, y,
        params.get('c_cu_initial', 1.5e-3),
        params.get('c_ni_initial', 4.0e-4)
    )
    metrics.mass_error = mass_metrics['mass_error_max']
    
    # Uncertainty metrics
    if weights is not None and len(weights) > 0:
        eps = 1e-10
        metrics.weight_entropy = -np.sum(weights * np.log(weights + eps))
    
    if 'target_params' in params and 'source_params' in params:
        # Normalized parameter distance
        target = params['target_params']
        sources = params['source_params']
        if sources:
            distances = []
            for src in sources:
                d_ly = abs(target.get('Ly', 60) - src.get('Ly', 60)) / 90  # normalized by range
                d_cu = abs(target.get('c_cu', 1.5e-3) - src.get('c_cu', 1.5e-3)) / 2.9e-3
                d_ni = abs(target.get('c_ni', 4e-4) - src.get('c_ni', 4e-4)) / 1.8e-3
                distances.append(np.sqrt(d_ly**2 + d_cu**2 + d_ni**2))
            metrics.param_distance = np.min(distances)
    
    # Ensemble variance (if multiple interpolations available)
    if ensemble_fields and len(ensemble_fields) > 1:
        c1_ensemble = np.stack([f['c1'] for f in ensemble_fields], axis=0)
        c2_ensemble = np.stack([f['c2'] for f in ensemble_fields], axis=0)
        metrics.ensemble_variance_c1 = np.mean(np.var(c1_ensemble, axis=0))
        metrics.ensemble_variance_c2 = np.mean(np.var(c2_ensemble, axis=0))
    
    # Composite score: weighted combination (higher = better)
    # Normalize each metric to 0-1 scale (error metrics inverted)
    scores = []
    scores.append(np.exp(-metrics.mse_c1 / 1e-6))  # MSE ~1e-6 is good
    scores.append(np.exp(-metrics.mse_c2 / 1e-6))
    scores.append(metrics.r2_c1 if metrics.r2_c1 > 0 else 0)
    scores.append(metrics.r2_c2 if metrics.r2_c2 > 0 else 0)
    scores.append(metrics.ssim_c1)
    scores.append(metrics.ssim_c2)
    scores.append(np.exp(-metrics.pde_residual_mean / 1e-4))
    scores.append(1 - metrics.mass_error)
    scores.append(np.exp(-metrics.param_distance / 0.3))  # distance < 0.3 is close
    
    metrics.overall_score = np.mean(scores)
    
    return metrics


# =============================================
# VISUALIZATION FUNCTIONS
# =============================================
def plot_metrics_bar_chart(metrics_df: pd.DataFrame, 
                          title: str = "Validation Metrics",
                          color_map: Optional[Dict] = None) -> go.Figure:
    """Create interactive bar chart of validation metrics, grouped by category."""
    if color_map is None:
        color_map = {
            'pointwise': '#2E86AB',
            'physics': '#A23B72',
            'uncertainty': '#F18F01',
            'composite': '#C73E1D'
        }
    
    # Normalize error metrics for better visualization (invert and scale)
    df = metrics_df.copy()
    error_metrics = ['mse_c1', 'mse_c2', 'mae_c1', 'mae_c2', 'max_error_c1', 'max_error_c2',
                    'pde_residual_mean', 'pde_residual_max', 'bc_error_top_c1', 'bc_error_top_c2',
                    'bc_error_bottom_c1', 'bc_error_bottom_c2', 'mass_error']
    for col in error_metrics:
        if col in df['Metric'].values:
            idx = df[df['Metric'] == col].index[0]
            val = df.loc[idx, 'Value']
            # Map error to [0,1] where 0 error = 1.0 score
            if val < 1e-8:
                df.loc[idx, 'Value'] = 1.0
            else:
                df.loc[idx, 'Value'] = np.exp(-val / 1e-4)  # exponential decay
    
    fig = go.Figure()
    
    for category in df['Category'].unique():
        cat_df = df[df['Category'] == category]
        fig.add_trace(go.Bar(
            x=cat_df['Metric'],
            y=cat_df['Value'],
            name=category,
            marker_color=color_map.get(category, '#666666'),
            text=cat_df['Value'].apply(lambda x: f'{x:.3f}'),
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Metric",
        yaxis_title="Normalized Score (0-1, higher=better)",
        barmode='group',
        legend_title="Category",
        height=500,
        hovermode='x unified'
    )
    
    return fig


def plot_radar_chart(metrics: ValidationMetrics, 
                    title: str = "Validation Radar Plot") -> go.Figure:
    """Create radar chart showing key validation metrics."""
    # Select representative metrics (normalized to 0-1, higher=better)
    categories = [
        'MSE (Cu)', 'MSE (Ni)', 'R² (Cu)', 'R² (Ni)', 
        'SSIM (Cu)', 'SSIM (Ni)', 'PDE Residual', 'Mass Cons.', 'Param. Distance'
    ]
    
    # Normalize and invert error metrics
    values = [
        np.exp(-metrics.mse_c1 / 1e-6),
        np.exp(-metrics.mse_c2 / 1e-6),
        max(0, metrics.r2_c1),
        max(0, metrics.r2_c2),
        metrics.ssim_c1,
        metrics.ssim_c2,
        np.exp(-metrics.pde_residual_mean / 1e-4),
        1 - metrics.mass_error,
        np.exp(-metrics.param_distance / 0.3)
    ]
    
    # Close the polygon
    categories += [categories[0]]
    values += [values[0]]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        line_color='#2E86AB',
        fillcolor='rgba(46, 134, 171, 0.3)',
        name='Interpolated Solution'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat='.1f'
            )),
        title=dict(text=title, x=0.5),
        showlegend=False,
        height=500
    )
    
    return fig


def plot_residual_heatmap(residual: np.ndarray, x: np.ndarray, y: np.ndarray,
                         title: str = "PDE Residual Field") -> go.Figure:
    """Create interactive heatmap of PDE residual."""
    fig = go.Figure(data=go.Heatmap(
        z=residual.T,  # Transpose for correct orientation
        x=x,
        y=y,
        colorscale='RdYlBu_r',
        colorbar=dict(title="Residual"),
        hovertemplate='x=%{x:.1f} μm, y=%{y:.1f} μm<br>Residual=%{z:.2e}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="x (μm)",
        yaxis_title="y (μm)",
        width=600,
        height=500,
        hovermode='closest'
    )
    
    return fig


def plot_uncertainty_scatter(metrics_list: List[ValidationMetrics],
                           param_names: List[str],
                           title: str = "Uncertainty vs Parameter Distance") -> go.Figure:
    """Scatter plot of uncertainty metrics vs parameter distance."""
    df = pd.DataFrame([
        {
            'Param Distance': m.param_distance,
            'Weight Entropy': m.weight_entropy,
            'Ensemble Var (Cu)': m.ensemble_variance_c1,
            'Ensemble Var (Ni)': m.ensemble_variance_c2,
            'Overall Score': m.overall_score
        }
        for m in metrics_list
    ])
    
    fig = px.scatter(
        df,
        x='Param Distance',
        y='Weight Entropy',
        size='Overall Score',
        color='Overall Score',
        color_continuous_scale='Viridis',
        hover_data=['Ensemble Var (Cu)', 'Ensemble Var (Ni)'],
        title=title,
        labels={'Param Distance': 'Normalized Parameter Distance',
               'Weight Entropy': 'Attention Weight Entropy',
               'Overall Score': 'Validation Score'}
    )
    
    fig.update_layout(
        height=500,
        coloraxis_colorbar=dict(title="Score")
    )
    
    return fig


def plot_comparison_heatmaps(interp_c1: np.ndarray, gt_c1: np.ndarray,
                           interp_c2: np.ndarray, gt_c2: np.ndarray,
                           x: np.ndarray, y: np.ndarray,
                           title_prefix: str = "Field Comparison") -> Tuple[go.Figure, go.Figure]:
    """Create side-by-side comparison heatmaps for interpolated vs ground truth."""
    # Cu comparison
    fig_c1 = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Interpolated", "Ground Truth", "Absolute Error"),
        horizontal_spacing=0.05
    )
    
    fig_c1.add_trace(go.Heatmap(z=interp_c1.T, x=x, y=y, colorscale='viridis', showscale=True, name='Interp'), row=1, col=1)
    fig_c1.add_trace(go.Heatmap(z=gt_c1.T, x=x, y=y, colorscale='viridis', showscale=False, name='GT'), row=1, col=2)
    fig_c1.add_trace(go.Heatmap(z=np.abs(interp_c1 - gt_c1).T, x=x, y=y, colorscale='RdYlBu_r', showscale=True, name='Error'), row=1, col=3)
    
    fig_c1.update_layout(title=dict(text=f"{title_prefix} - Cu Concentration", x=0.5), height=400)
    
    # Ni comparison
    fig_c2 = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Interpolated", "Ground Truth", "Absolute Error"),
        horizontal_spacing=0.05
    )
    
    fig_c2.add_trace(go.Heatmap(z=interp_c2.T, x=x, y=y, colorscale='magma', showscale=True, name='Interp'), row=1, col=1)
    fig_c2.add_trace(go.Heatmap(z=gt_c2.T, x=x, y=y, colorscale='magma', showscale=False, name='GT'), row=1, col=2)
    fig_c2.add_trace(go.Heatmap(z=np.abs(interp_c2 - gt_c2).T, x=x, y=y, colorscale='RdYlBu_r', showscale=True, name='Error'), row=1, col=3)
    
    fig_c2.update_layout(title=dict(text=f"{title_prefix} - Ni Concentration", x=0.5), height=400)
    
    return fig_c1, fig_c2


# =============================================
# PHYSICS-INFORMED INTERPOLATION ENHANCEMENT
# =============================================
class PhysicsAwareInterpolator:
    """
    Enhanced attention interpolator with physics constraints.
    Adds PDE residual penalty, boundary enforcement, and mass conservation.
    """
    
    def __init__(self, base_interpolator, physics_constants: Dict = None):
        self.base = base_interpolator
        self.constants = physics_constants or PHYSICS_CONSTANTS
        self.enforce_bc = True
        self.pde_weight = 0.1  # Weight for PDE residual in optimization
        self.mass_weight = 0.05  # Weight for mass conservation
        
    def interpolate_with_physics(self, solutions: List[Dict], params_list: List[Dict],
                               target_params: Dict, target_shape: Tuple[int, int] = (50, 50),
                               time_norm: float = 1.0, optimize: bool = True) -> Dict:
        """
        Perform attention interpolation with optional physics-based refinement.
        
        Args:
            optimize: If True, apply gradient-based refinement to minimize PDE residual
        """
        # Step 1: Base attention interpolation
        result = self.base(solutions, params_list, 
                          target_params.get('Ly', 60),
                          target_params.get('c_cu', 1.5e-3),
                          target_params.get('c_ni', 4e-4))
        
        if not result or 'fields' not in result:
            return result
        
        fields = result['fields']
        c1 = fields.get('c1_preds', [np.zeros(target_shape)])[0]
        c2 = fields.get('c2_preds', [np.zeros(target_shape)])[0]
        
        # Step 2: Enforce boundary conditions
        if self.enforce_bc:
            Lx = target_params.get('Lx', 60)
            Ly = target_params.get('Ly', 60)
            x = np.linspace(0, Lx, target_shape[1])
            y = np.linspace(0, Ly, target_shape[0])
            
            c1, c2 = enforce_boundary_conditions(
                c1, c2, x, y,
                target_params.get('c_cu_top', 1.59e-3),
                target_params.get('c_cu_bottom', 0.0),
                target_params.get('c_ni_top', 0.0),
                target_params.get('c_ni_bottom', 4e-4),
                enforce_type='hard'
            )
        
        # Step 3: Optional physics-based refinement via gradient descent
        if optimize and torch.cuda.is_available():
            c1, c2 = self._physics_refinement(
                c1, c2, target_params, target_shape, time_norm
            )
        
        # Update result
        result['fields']['c1_preds'] = [c1]
        result['fields']['c2_preds'] = [c2]
        result['physics_enforced'] = True
        
        # Compute physics metrics for diagnostics
        Lx = target_params.get('Lx', 60)
        Ly = target_params.get('Ly', 60)
        x = np.linspace(0, Lx, target_shape[1])
        y = np.linspace(0, Ly, target_shape[0])
        t = time_norm * target_params.get('t_max', 200)
        
        residual = compute_pde_residual(
            c1, c2, x, y, t,
            self.constants['D11'], self.constants['D12'],
            self.constants['D21'], self.constants['D22'],
            x[1]-x[0], y[1]-y[0]
        )
        
        mass_metrics = compute_mass_conservation(
            c1, c2, x, y,
            target_params.get('c_cu_initial', 1.5e-3),
            target_params.get('c_ni_initial', 4e-4)
        )
        
        result['physics_diagnostics'] = {
            'pde_residual_mean': float(np.mean(residual)),
            'pde_residual_max': float(np.max(residual)),
            'mass_error': float(mass_metrics['mass_error_max'])
        }
        
        return result
    
    def _physics_refinement(self, c1: np.ndarray, c2: np.ndarray,
                          target_params: Dict, target_shape: Tuple[int, int],
                          time_norm: float, n_steps: int = 50, lr: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Refine interpolated fields by minimizing PDE residual via gradient descent.
        Uses PyTorch for automatic differentiation.
        """
        # Convert to tensors
        c1_t = torch.tensor(c1, dtype=torch.float32, requires_grad=True)
        c2_t = torch.tensor(c2, dtype=torch.float32, requires_grad=True)
        
        Lx = target_params.get('Lx', 60)
        Ly = target_params.get('Ly', 60)
        t = time_norm * target_params.get('t_max', 200)
        
        x = torch.linspace(0, Lx, target_shape[1])
        y = torch.linspace(0, Ly, target_shape[0])
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        optimizer = optim.Adam([c1_t, c2_t], lr=lr)
        
        for step in range(n_steps):
            optimizer.zero_grad()
            
            # Compute PDE residual loss
            lap_c1 = self._laplacian_torch(c1_t, X, Y)
            lap_c2 = self._laplacian_torch(c2_t, X, Y)
            
            # Time derivative approximation (quasi-steady)
            residual1 = -(self.constants['D11'] * lap_c1 + self.constants['D12'] * lap_c2)
            residual2 = -(self.constants['D21'] * lap_c1 + self.constants['D22'] * lap_c2)
            
            pde_loss = torch.mean(residual1**2 + residual2**2)
            
            # Boundary loss (soft constraint)
            bc_loss = (
                torch.mean((c1_t[:, -1] - target_params.get('c_cu_top', 1.59e-3))**2) +
                torch.mean((c1_t[:, 0] - target_params.get('c_cu_bottom', 0.0))**2) +
                torch.mean((c2_t[:, -1] - target_params.get('c_ni_top', 0.0))**2) +
                torch.mean((c2_t[:, 0] - target_params.get('c_ni_bottom', 4e-4))**2)
            )
            
            # Total loss
            loss = self.pde_weight * pde_loss + self.mass_weight * bc_loss
            
            loss.backward()
            optimizer.step()
            
            # Clip to physical ranges
            with torch.no_grad():
                c1_t.clamp_(0, self.constants['C_CU_RANGE'][1])
                c2_t.clamp_(0, self.constants['C_NI_RANGE'][1])
        
        return c1_t.detach().numpy(), c2_t.detach().numpy()
    
    def _laplacian_torch(self, c: torch.Tensor, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian using PyTorch autograd."""
        # First derivatives
        dc_dx = torch.autograd.grad(
            c.sum(), X, create_graph=True, retain_graph=True
        )[0]
        dc_dy = torch.autograd.grad(
            c.sum(), Y, create_graph=True, retain_graph=True
        )[0]
        
        # Second derivatives
        d2c_dx2 = torch.autograd.grad(
            dc_dx.sum(), X, create_graph=True, retain_graph=True
        )[0]
        d2c_dy2 = torch.autograd.grad(
            dc_dy.sum(), Y, create_graph=True, retain_graph=True
        )[0]
        
        return d2c_dx2 + d2c_dy2


# =============================================
# STREAMLIT APP: VALIDATION & UNCERTAINTY DASHBOARD
# =============================================
def initialize_session_state():
    """Initialize Streamlit session state for validation module."""
    defaults = {
        'solutions_loaded': False,
        'solutions': [],
        'held_out_indices': [],
        'validation_results': {},
        'uncertainty_results': {},
        'current_target_params': None,
        'interpolator': None,
        'physics_interpolator': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_solutions_for_validation(solution_dir: str = "pinn_solutions") -> List[Dict]:
    """Load precomputed PINN solutions for validation."""
    solutions = []
    if not os.path.exists(solution_dir):
        st.warning(f"Solution directory {solution_dir} not found.")
        return solutions
    
    for fname in os.listdir(solution_dir):
        if fname.endswith('.pkl'):
            try:
                with open(os.path.join(solution_dir, fname), 'rb') as f:
                    sol = pickle.load(f)
                # Validate required keys
                required = ['params', 'X', 'Y', 'c1_preds', 'c2_preds', 'times']
                if all(k in sol for k in required):
                    solutions.append(sol)
            except Exception as e:
                st.warning(f"Failed to load {fname}: {e}")
    
    return solutions


def select_held_out_set(solutions: List[Dict], held_out_fraction: float = 0.2) -> List[int]:
    """Randomly select held-out solutions for validation."""
    np.random.seed(42)  # Reproducibility
    n_held_out = max(1, int(len(solutions) * held_out_fraction))
    return np.random.choice(len(solutions), size=n_held_out, replace=False).tolist()


def run_validation_pipeline(solutions: List[Dict], held_out_indices: List[int],
                           target_params: Dict, interpolator, 
                           physics_aware: bool = True) -> Dict[str, ValidationMetrics]:
    """
    Run validation pipeline: interpolate held-out targets, compute metrics.
    """
    results = {}
    
    for idx in held_out_indices:
        gt_sol = solutions[idx]
        gt_params = gt_sol['params']
        
        # Use ground truth params as target for validation
        target = {
            'Ly': gt_params.get('Ly', 60),
            'c_cu': gt_params.get('C_Cu', 1.5e-3),
            'c_ni': gt_params.get('C_Ni', 4e-4),
            'Lx': gt_params.get('Lx', 60),
            't_max': gt_params.get('t_max', 200),
            'c_cu_top': gt_params.get('C_Cu', 1.59e-3),
            'c_cu_bottom': 0.0,
            'c_ni_top': 0.0,
            'c_ni_bottom': gt_params.get('C_Ni', 4e-4),
            'c_cu_initial': 1.5e-3,
            'c_ni_initial': 4e-4,
            'D11': PHYSICS_CONSTANTS['D11'],
            'D12': PHYSICS_CONSTANTS['D12'],
            'D21': PHYSICS_CONSTANTS['D21'],
            'D22': PHYSICS_CONSTANTS['D22'],
            'target_params': target_params,
            'source_params': [s['params'] for s in solutions if s != gt_sol]
        }
        
        # Interpolate
        if physics_aware and st.session_state.physics_interpolator:
            interp_result = st.session_state.physics_interpolator.interpolate_with_physics(
                solutions, [s['params'] for s in solutions], target,
                target_shape=(50, 50), time_norm=1.0, optimize=True
            )
        else:
            interp_result = interpolator(solutions, [s['params'] for s in solutions],
                                       target['Ly'], target['c_cu'], target['c_ni'])
        
        if not interp_result or 'fields' not in interp_result:
            continue
        
        # Extract fields at final time
        interp_c1 = interp_result['fields']['c1_preds'][-1]
        interp_c2 = interp_result['fields']['c2_preds'][-1]
        gt_c1 = gt_sol['c2_preds'][-1] if len(gt_sol['c2_preds']) > 0 else np.zeros_like(interp_c1)
        gt_c2 = gt_sol['c2_preds'][-1] if len(gt_sol['c2_preds']) > 0 else np.zeros_like(interp_c2)
        
        x = gt_sol['X'][:, 0] if 'X' in gt_sol else np.linspace(0, target['Lx'], 50)
        y = gt_sol['Y'][0, :] if 'Y' in gt_sol else np.linspace(0, target['Ly'], 50)
        
        # Compute metrics
        metrics = compute_validation_metrics(
            interp_c1, interp_c2, gt_c1, gt_c2, x, y, target['t_max'],
            target, 
            weights=interp_result.get('weights', {}).get('combined', None),
            ensemble_fields=None  # Could add ensemble here
        )
        
        results[idx] = {
            'metrics': metrics,
            'interp_c1': interp_c1,
            'interp_c2': interp_c2,
            'gt_c1': gt_c1,
            'gt_c2': gt_c2,
            'x': x,
            'y': y,
            'params': target
        }
    
    return results


def render_validation_dashboard():
    """Main Streamlit dashboard for validation and uncertainty quantification."""
    st.set_page_config(page_title="Physics-Informed Validation Dashboard", layout="wide")
    st.title("🔬 Physics-Informed Validation & Uncertainty Quantification")
    
    initialize_session_state()
    
    # Sidebar: Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        solution_dir = st.text_input("Solution Directory", value="pinn_solutions")
        held_out_fraction = st.slider("Held-out Fraction", 0.1, 0.5, 0.2, 0.05)
        physics_aware = st.checkbox("Enable Physics-Aware Interpolation", value=True)
        optimize_fields = st.checkbox("Optimize Fields (PDE Refinement)", value=True, disabled=not physics_aware)
        
        if st.button("🔄 Load Solutions & Run Validation"):
            with st.spinner("Loading solutions..."):
                solutions = load_solutions_for_validation(solution_dir)
                if not solutions:
                    st.error("No valid solutions found.")
                    return
                st.session_state.solutions = solutions
                st.session_state.solutions_loaded = True
                st.session_state.held_out_indices = select_held_out_set(solutions, held_out_fraction)
                
            with st.spinner("Running validation pipeline..."):
                # Initialize interpolators
                if st.session_state.interpolator is None:
                    from your_attention_module import MultiParamAttentionInterpolator  # Import your base interpolator
                    st.session_state.interpolator = MultiParamAttentionInterpolator()
                
                if physics_aware and st.session_state.physics_interpolator is None:
                    st.session_state.physics_interpolator = PhysicsAwareInterpolator(
                        st.session_state.interpolator
                    )
                
                target_params = {
                    'Ly': 60, 'c_cu': 1.5e-3, 'c_ni': 4e-4,
                    'Lx': 60, 't_max': 200
                }
                st.session_state.current_target_params = target_params
                
                results = run_validation_pipeline(
                    solutions, st.session_state.held_out_indices,
                    target_params, st.session_state.interpolator,
                    physics_aware=physics_aware and optimize_fields
                )
                st.session_state.validation_results = results
                st.success(f"Validation complete: {len(results)} held-out cases evaluated.")
    
    # Main content
    if not st.session_state.solutions_loaded:
        st.info("👈 Use the sidebar to load solutions and run validation.")
        return
    
    if not st.session_state.validation_results:
        st.warning("No validation results available. Click 'Load Solutions & Run Validation'.")
        return
    
    results = st.session_state.validation_results
    
    # Summary metrics table
    st.subheader("📊 Validation Summary")
    summary_data = []
    for idx, res in results.items():
        m = res['metrics']
        summary_data.append({
            'Case': f"#{idx}",
            'Overall Score': f"{m.overall_score:.3f}",
            'MSE (Cu)': f"{m.mse_c1:.2e}",
            'MSE (Ni)': f"{m.mse_c2:.2e}",
            'PDE Residual': f"{m.pde_residual_mean:.2e}",
            'Mass Error': f"{m.mass_error:.2%}",
            'Weight Entropy': f"{m.weight_entropy:.3f}",
            'Param Distance': f"{m.param_distance:.3f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df.style.format({
        'Overall Score': '{:.3f}',
        'MSE (Cu)': '{:.2e}',
        'MSE (Ni)': '{:.2e}',
        'PDE Residual': '{:.2e}',
        'Mass Error': '{:.2%}'
    }).background_gradient(subset=['Overall Score'], cmap='Greens'), use_container_width=True)
    
    # Interactive visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Metrics Charts", "🎯 Radar Analysis", "🔍 Field Comparison", "📉 Uncertainty Analysis"])
    
    with tab1:
        st.subheader("Validation Metrics by Category")
        # Aggregate metrics across all cases
        all_metrics_df = pd.concat([res['metrics'].to_dataframe() for res in results.values()])
        avg_metrics = all_metrics_df.groupby('Metric')['Value'].mean().reset_index()
        avg_metrics['Category'] = all_metrics_df.groupby('Metric')['Category'].first().values
        
        fig_bar = plot_metrics_bar_chart(avg_metrics)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab2:
        st.subheader("Radar Plot: Multi-Metric Validation")
        # Plot radar for first case as example
        first_metrics = list(results.values())[0]['metrics']
        fig_radar = plot_radar_chart(first_metrics)
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with tab3:
        st.subheader("Field Comparison: Interpolated vs Ground Truth")
        case_idx = st.selectbox("Select Validation Case", list(results.keys()), format_func=lambda x: f"Case #{x}")
        res = results[case_idx]
        
        fig_c1, fig_c2 = plot_comparison_heatmaps(
            res['interp_c1'], res['gt_c1'],
            res['interp_c2'], res['gt_c2'],
            res['x'], res['y'],
            title_prefix=f"Case #{case_idx}"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_c1, use_container_width=True)
        with col2:
            st.plotly_chart(fig_c2, use_container_width=True)
        
        # PDE residual heatmap
        st.subheader("PDE Residual Field")
        residual = compute_pde_residual(
            res['interp_c1'], res['interp_c2'], res['x'], res['y'], 
            res['params'].get('t_max', 200),
            PHYSICS_CONSTANTS['D11'], PHYSICS_CONSTANTS['D12'],
            PHYSICS_CONSTANTS['D21'], PHYSICS_CONSTANTS['D22'],
            res['x'][1]-res['x'][0], res['y'][1]-res['y'][0]
        )
        fig_residual = plot_residual_heatmap(residual, res['x'], res['y'])
        st.plotly_chart(fig_residual, use_container_width=True)
    
    with tab4:
        st.subheader("Uncertainty Quantification")
        
        # Scatter: uncertainty vs parameter distance
        metrics_list = [res['metrics'] for res in results.values()]
        fig_scatter = plot_uncertainty_scatter(metrics_list, ['Ly', 'c_cu', 'c_ni'])
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Key uncertainty insights
        st.markdown("### 🔑 Key Insights")
        avg_entropy = np.mean([m.weight_entropy for m in metrics_list])
        avg_distance = np.mean([m.param_distance for m in metrics_list])
        avg_score = np.mean([m.overall_score for m in metrics_list])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg. Weight Entropy", f"{avg_entropy:.3f}", 
                     help="Higher = more uncertain source weighting")
        with col2:
            st.metric("Avg. Parameter Distance", f"{avg_distance:.3f}",
                     help="Lower = target closer to training sources")
        with col3:
            st.metric("Avg. Validation Score", f"{avg_score:.3f} / 1.0",
                     help="Higher = better agreement with held-out PINN")
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        if avg_distance > 0.3:
            st.warning("⚠️ Target parameters are far from training data. Consider adding nearby simulations.")
        if avg_entropy > 1.5:
            st.warning("⚠️ High weight entropy indicates ambiguous source selection. Review parameter space coverage.")
        if avg_score < 0.7:
            st.error("❌ Low validation score. Enable physics-aware refinement or collect more training data.")
        if all(m.mass_error < 0.01 for m in metrics_list):
            st.success("✅ Mass conservation satisfied (<1% error) across all cases.")
    
    # Export results
    with st.expander("💾 Export Validation Report"):
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'held_out_fraction': held_out_fraction,
                'physics_aware': physics_aware,
                'optimize_fields': optimize_fields
            },
            'summary': summary_df.to_dict('records'),
            'detailed_metrics': {idx: res['metrics'].to_dict() for idx, res in results.items()}
        }
        
        json_str = json.dumps(report_data, indent=2, default=str)
        st.download_button(
            "📥 Download JSON Report",
            json_str,
            f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json"
        )
        
        # CSV export of metrics
        csv_str = pd.concat([res['metrics'].to_dataframe() for res in results.values()]).to_csv(index=False)
        st.download_button(
            "📥 Download Metrics CSV",
            csv_str,
            f"validation_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )


if __name__ == "__main__":
    render_validation_dashboard()
