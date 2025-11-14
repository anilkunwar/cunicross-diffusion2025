import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import zipfile
import io
import matplotlib as mpl
import logging
import pyvista as pv
import hashlib

# Ensure output directory exists
OUTPUT_DIR = '/tmp/pinn_solutions'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure Matplotlib
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['figure.dpi'] = 300

# Configure logging
logging.basicConfig(level=logging.INFO, filename=os.path.join(OUTPUT_DIR, 'training.log'), filemode='a')
logger = logging.getLogger(__name__)

# Fixed parameters
C_CU_TOP = 0.0 #2.85e-3    # Top boundary (y=Ly): Cu-rich
C_CU_BOTTOM = 1.0e-3      # Bottom (y=0): Cu-poor
#C_CU_BOTTOM = 2.85e-3     # Bottom (y=0): Cu-rich
C_NI_TOP = 0.0     #  Top (y=Ly): Ni-poor
C_NI_BOTTOM = 0.0      # Bottom (y=0): Ni-poor
Ly = 30.0             # Domain height (μm)
Lx = 60.0             # Domain width (μm)
D11 = 0.006
D12 = 0.00427
D21 = 0.003697
D22 = 0.0054
T_max = 200.0
epochs = 5000
lr = 1e-3

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Helper function for cache key
def get_cache_key(*args):
    key_string = "_".join(str(arg) for arg in args)
    return hashlib.md5(key_string.encode()).hexdigest()

class SmoothSigmoid(nn.Module):
    def __init__(self, slope=1.0):
        super().__init__()
        self.k = slope
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return self.scale * 1 / (1 + torch.exp(-self.k * x))

class DualScaledPINN(nn.Module):
    def __init__(self, D11, D12, D21, D22, Lx, Ly, T_max, C_Cu, C_Ni):
        super().__init__()
        self.D11 = D11
        self.D12 = D12
        self.D21 = D21
        self.D22 = D22
        self.Lx = Lx
        self.Ly = Ly
        self.T_max = T_max
        self.C_Cu = C_Cu
        self.C_Ni = C_Ni
        
        self.C_Cu_norm = (C_Cu - 1.5e-3) / (2.9e-3 - 1.5e-3)
        self.C_Ni_norm = (C_Ni - 4.0e-4) / (1.8e-3 - 4.0e-4)
        
        self.shared_net = nn.Sequential(
            nn.Linear(5, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh()
        )
        
        self.cu_head = nn.Sequential(
            nn.Linear(128, 1),
            SmoothSigmoid(slope=0.5),
            nn.Linear(1, 1, bias=False),
        )
        self.ni_head = nn.Sequential(
            nn.Linear(128, 1),
            SmoothSigmoid(slope=0.5),
            nn.Linear(1, 1, bias=False),
        )
        
        self.cu_head[2].weight.data.fill_(C_Cu)
        self.ni_head[2].weight.data.fill_(C_Ni)

    def forward(self, x, y, t):
        x_norm = x / self.Lx
        y_norm = y / self.Ly
        t_norm = t / self.T_max
        C_Cu_input = torch.full_like(x, self.C_Cu_norm)
        C_Ni_input = torch.full_like(x, self.C_Ni_norm)
        
        inputs = torch.cat([x_norm, y_norm, t_norm, C_Cu_input, C_Ni_input], dim=1)
        features = self.shared_net(inputs)
        cu = self.cu_head(features)
        ni = self.ni_head(features)
        return torch.cat([cu, ni], dim=1)

def laplacian(c, x, y):
    c_x = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c),
                            create_graph=True, retain_graph=True)[0]
    c_y = torch.autograd.grad(c, y, grad_outputs=torch.ones_like(c),
                            create_graph=True, retain_graph=True)[0]
    
    c_xx = torch.autograd.grad(c_x, x, grad_outputs=torch.ones_like(c_x),
                             create_graph=True, retain_graph=True)[0]
    c_yy = torch.autograd.grad(c_y, y, grad_outputs=torch.ones_like(c_y),
                             create_graph=True, retain_graph=True)[0]
    return c_xx + c_yy

def physics_loss(model, x, y, t):
    c_pred = model(x, y, t)
    c1_pred, c2_pred = c_pred[:, 0:1], c_pred[:, 1:2]
    
    c1_t = torch.autograd.grad(c1_pred, t, grad_outputs=torch.ones_like(c1_pred),
                             create_graph=True, retain_graph=True)[0]
    c2_t = torch.autograd.grad(c2_pred, t, grad_outputs=torch.ones_like(c2_pred),
                             create_graph=True, retain_graph=True)[0]
    
    lap_c1 = laplacian(c1_pred, x, y)
    lap_c2 = laplacian(c2_pred, x, y)
    
    residual1 = c1_t - (model.D11 * lap_c1 + model.D12 * lap_c2)
    residual2 = c2_t - (model.D21 * lap_c1 + model.D22 * lap_c2)
    return torch.mean(residual1**2 + residual2**2)

def boundary_loss_bottom(model):
    num = 200
    x = torch.rand(num, 1, requires_grad=True) * model.Lx
    y = torch.zeros(num, 1, requires_grad=True)
    t = torch.rand(num, 1, requires_grad=True) * model.T_max
    
    c_pred = model(x, y, t)
    return (torch.mean((c_pred[:, 0] - C_CU_BOTTOM)**2) + 
            torch.mean((c_pred[:, 1] - C_NI_BOTTOM)**2))

def boundary_loss_top(model):
    num = 200
    x = torch.rand(num, 1, requires_grad=True) * model.Lx
    y = torch.full((num, 1), model.Ly, requires_grad=True)
    t = torch.rand(num, 1, requires_grad=True) * model.T_max
    
    c_pred = model(x, y, t)
    return (torch.mean((c_pred[:, 0] - C_CU_TOP)**2) + 
            torch.mean((c_pred[:, 1] - C_NI_TOP)**2))

def boundary_loss_sides(model):
    num = 200
    x_left = torch.zeros(num, 1, dtype=torch.float32, requires_grad=True)
    y_left = torch.rand(num, 1, requires_grad=True) * model.Ly
    t_left = torch.rand(num, 1, requires_grad=True) * model.T_max
    c_left = model(x_left, y_left, t_left)
    
    x_right = torch.full((num, 1), float(model.Lx), dtype=torch.float32, requires_grad=True)
    y_right = torch.rand(num, 1, requires_grad=True) * model.Ly
    t_right = torch.rand(num, 1, requires_grad=True) * model.T_max
    c_right = model(x_right, y_right, t_right)
    
    try:
        grad_cu_x_left = torch.autograd.grad(
            c_left[:, 0], x_left,
            grad_outputs=torch.ones_like(c_left[:, 0]),
            create_graph=True, retain_graph=True
        )[0]
        
        grad_ni_x_left = torch.autograd.grad(
            c_left[:, 1], x_left,
            grad_outputs=torch.ones_like(c_left[:, 1]),
            create_graph=True, retain_graph=True
        )[0]
        
        grad_cu_x_right = torch.autograd.grad(
            c_right[:, 0], x_right,
            grad_outputs=torch.ones_like(c_right[:, 0]),
            create_graph=True, retain_graph=True
        )[0]
        
        grad_ni_x_right = torch.autograd.grad(
            c_right[:, 1], x_right,
            grad_outputs=torch.ones_like(c_right[:, 1]),
            create_graph=True, retain_graph=True
        )[0]
        
        grad_cu_x_left = grad_cu_x_left if grad_cu_x_left is not None else torch.zeros_like(c_left[:, 0])
        grad_ni_x_left = grad_ni_x_left if grad_ni_x_left is not None else torch.zeros_like(c_left[:, 1])
        grad_cu_x_right = grad_cu_x_right if grad_cu_x_right is not None else torch.zeros_like(c_right[:, 0])
        grad_ni_x_right = grad_ni_x_right if grad_ni_x_right is not None else torch.zeros_like(c_right[:, 1])
        
        return (torch.mean(grad_cu_x_left**2) + 
                torch.mean(grad_ni_x_left**2) + 
                torch.mean(grad_cu_x_right**2) + 
                torch.mean(grad_ni_x_right**2))
    
    except RuntimeError as e:
        logger.error(f"Gradient computation failed in boundary_loss_sides: {str(e)}")
        st.error(f"Gradient computation failed: {str(e)}")
        return torch.tensor(1e-6, requires_grad=True)

def initial_loss(model):
    num = 500
    x = torch.rand(num, 1, requires_grad=True) * model.Lx
    y = torch.rand(num, 1, requires_grad=True) * model.Ly
    t = torch.zeros(num, 1, requires_grad=True)
    return torch.mean(model(x, y, t)**2)

def validate_boundary_conditions(solution, tolerance=1e-6):
    results = {
        'top_bc_cu': True,
        'top_bc_ni': True,
        'bottom_bc_cu': True,
        'bottom_bc_ni': True,
        'left_flux_cu': True,
        'left_flux_ni': True,
        'right_flux_cu': True,
        'right_flux_ni': True,
        'details': []
    }
    t_idx = -1
    c1 = solution['c1_preds'][t_idx]
    c2 = solution['c2_preds'][t_idx]
    
    top_cu_mean = np.mean(c1[:, -1])
    top_ni_mean = np.mean(c2[:, -1])
    if abs(top_cu_mean - C_CU_TOP) > tolerance:
        results['top_bc_cu'] = False
        results['details'].append(f"Top Cu: {top_cu_mean:.2e} != {C_CU_TOP:.2e}")
    if abs(top_ni_mean - C_NI_BOTTOM) > tolerance:
        results['top_bc_ni'] = False
        results['details'].append(f"Top Ni: {top_ni_mean:.2e} != {C_NI_BOTTOM:.2e}")
    
    bottom_cu_mean = np.mean(c1[:, 0])
    bottom_ni_mean = np.mean(c2[:, 0])
    if abs(bottom_cu_mean - C_CU_BOTTOM) > tolerance:
        results['bottom_bc_cu'] = False
        results['details'].append(f"Bottom Cu: {bottom_cu_mean:.2e} != {C_CU_BOTTOM:.2e}")
    if abs(bottom_ni_mean - C_NI_TOP) > tolerance:
        results['bottom_bc_ni'] = False
        results['details'].append(f"Bottom Ni: {bottom_ni_mean:.2e} != {C_NI_TOP:.2e}")
    
    left_flux_cu = np.mean(np.abs(c1[1, :] - c1[0, :]))
    left_flux_ni = np.mean(np.abs(c2[1, :] - c2[0, :]))
    right_flux_cu = np.mean(np.abs(c1[-1, :] - c1[-2, :]))
    right_flux_ni = np.mean(np.abs(c2[-1, :] - c2[-2, :]))
    if left_flux_cu > tolerance:
        results['left_flux_cu'] = False
        results['details'].append(f"Left flux Cu: {left_flux_cu:.2e}")
    if left_flux_ni > tolerance:
        results['left_flux_ni'] = False
        results['details'].append(f"Left flux Ni: {left_flux_ni:.2e}")
    if right_flux_cu > tolerance:
        results['right_flux_cu'] = False
        results['details'].append(f"Right flux Cu: {right_flux_cu:.2e}")
    if right_flux_ni > tolerance:
        results['right_flux_ni'] = False
        results['details'].append(f"Right flux Ni: {right_flux_ni:.2e}")
    
    results['valid'] = all([
        results['top_bc_cu'], results['top_bc_ni'],
        results['bottom_bc_cu'], results['bottom_bc_ni'],
        results['left_flux_cu'], results['left_flux_ni'],
        results['right_flux_cu'], results['right_flux_ni']
    ])
    return results

@st.cache_data(ttl=3600, show_spinner=False)
def plot_losses(loss_history, output_dir, _hash):
    epochs = np.array(loss_history['epochs'])
    total_loss = np.array(loss_history['total'])
    physics_loss = np.array(loss_history['physics'])
    bottom_loss = np.array(loss_history['bottom'])
    top_loss = np.array(loss_history['top'])
    sides_loss = np.array(loss_history['sides'])
    initial_loss = np.array(loss_history['initial'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, total_loss, label='Total Loss', linewidth=2, color='black')
    plt.plot(epochs, physics_loss, label='Physics Loss', linewidth=1.5, linestyle='--', color='blue')
    plt.plot(epochs, bottom_loss, label='Bottom Boundary Loss', linewidth=1.5, linestyle='-.', color='red')
    plt.plot(epochs, top_loss, label='Top Boundary Loss', linewidth=1.5, linestyle=':', color='green')
    plt.plot(epochs, sides_loss, label='Sides Boundary Loss', linewidth=1.5, linestyle='-', color='purple')
    plt.plot(epochs, initial_loss, label='Initial Condition Loss', linewidth=1.5, linestyle='--', color='orange')
    
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss for Ly = {Ly:.1f} μm')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f'loss_plot_ly_{Ly:.1f}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved loss plot to {plot_filename}")
    return plot_filename

@st.cache_data(ttl=3600, show_spinner=False)
def plot_2d_profiles(solution, time_idx, output_dir, _hash):
    t_val = solution['times'][time_idx]
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    im1 = plt.imshow(solution['c1_preds'][time_idx], origin='lower', 
                     extent=[0, Lx, 0, Ly], cmap='viridis',
                     vmin=0, vmax=C_CU_TOP)
    plt.title(f'Cu Concentration (t={t_val:.1f} s)')
    plt.xlabel('x (μm)')
    plt.ylabel('y (μm)')
    plt.grid(True, alpha=0.3)
    plt.colorbar(im1, label='Cu Conc. (mol/cc)', format='%.1e')
    
    plt.subplot(1, 2, 2)
    im2 = plt.imshow(solution['c2_preds'][time_idx], origin='lower', 
                     extent=[0, Lx, 0, Ly], cmap='magma',
                     vmin=0, vmax=C_NI_TOP)
    plt.title(f'Ni Concentration (t={t_val:.1f} s)')
    plt.xlabel('x (μm)')
    plt.ylabel('y (μm)')
    plt.grid(True, alpha=0.3)
    plt.colorbar(im2, label='Ni Conc. (mol/cc)', format='%.1e')
    
    plt.suptitle(f'2D Profiles (Ly={Ly:.0f} μm)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f'profile_ly_{Ly:.1f}_t_{t_val:.1f}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved profile plot to {plot_filename}")
    return plot_filename

@st.cache_resource(ttl=3600, show_spinner=False)
def train_model(D11, D12, D21, D22, Lx, Ly, T_max, C_Cu, C_Ni, epochs, lr, output_dir, _hash):
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Starting training with Ly={Ly}, C_Cu={C_Cu}, C_Ni={C_Ni}, epochs={epochs}, lr={lr}")
    model = DualScaledPINN(D11, D12, D21, D22, Lx, Ly, T_max, C_Cu, C_Ni)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    x_pde = torch.rand(1000, 1, requires_grad=True) * Lx
    y_pde = torch.rand(1000, 1, requires_grad=True) * Ly
    t_pde = torch.rand(1000, 1, requires_grad=True) * T_max
    
    loss_history = {
        'epochs': [],
        'total': [],
        'physics': [],
        'bottom': [],
        'top': [],
        'sides': [],
        'initial': []
    }
    
    progress = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        phys_loss = physics_loss(model, x_pde, y_pde, t_pde)
        bot_loss = boundary_loss_bottom(model)
        top_loss = boundary_loss_top(model)
        side_loss = boundary_loss_sides(model)
        init_loss = initial_loss(model)
        
        loss = (10 * phys_loss + 100 * bot_loss + 100 * top_loss + 
                100 * side_loss + 100 * init_loss)
        
        try:
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        except RuntimeError as e:
            logger.error(f"Backward pass failed at epoch {epoch + 1}: {str(e)}")
            st.error(f"Training failed at epoch {epoch + 1}: {str(e)}")
            return None, None
        
        if (epoch + 1) % 100 == 0:
            loss_history['epochs'].append(epoch + 1)
            loss_history['total'].append(loss.item())
            loss_history['physics'].append(10 * phys_loss.item())
            loss_history['bottom'].append(100 * bot_loss.item())
            loss_history['top'].append(100 * top_loss.item())
            loss_history['sides'].append(100 * side_loss.item())
            loss_history['initial'].append(100 * init_loss.item())
            
            progress.progress((epoch + 1) / epochs)
            status_text.text(
                f"Epoch {epoch + 1}/{epochs}, Total Loss: {loss.item():.6f}, "
                f"Physics: {10 * phys_loss.item():.6f}, Bottom: {100 * bot_loss.item():.6f}, "
                f"Top: {100 * top_loss.item():.6f}, Sides: {100 * side_loss.item():.6f}, "
                f"Initial: {100 * init_loss.item():.6f}"
            )
    
    progress.progress(1.0)
    status_text.text("Training completed!")
    logger.info("Training completed successfully")
    
    return model, loss_history

@st.cache_data(ttl=3600, show_spinner=False)
def evaluate_model(_model, times, Lx, Ly, D11, D12, D21, D22, _hash):
    x = torch.linspace(0, Lx, 50, requires_grad=False)
    y = torch.linspace(0, Ly, 50, requires_grad=False)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    c1_preds, c2_preds = [], []
    for t_val in times:
        t = torch.full((X.numel(), 1), t_val, requires_grad=False)
        c_pred = _model(X.reshape(-1,1), Y.reshape(-1,1), t)
        try:
            c1 = c_pred[:,0].detach().numpy().reshape(50,50).T  # [y,x] for matplotlib
            c2 = c_pred[:,1].detach().numpy().reshape(50,50).T  # [y,x] for matplotlib
        except RuntimeError as e:
            logger.error(f"Failed to convert concentration predictions to NumPy: {str(e)}")
            raise e
        
        c1_preds.append(c1)
        c2_preds.append(c2)
    
    return X.numpy(), Y.numpy(), c1_preds, c2_preds

@st.cache_data(ttl=3600, show_spinner=False)
def generate_and_save_solution(_model, times, param_set, output_dir, _hash):
    os.makedirs(output_dir, exist_ok=True)
    if _model is None:
        logger.error("Model is None, cannot generate solution")
        return None, None
    
    try:
        X, Y, c1_preds, c2_preds = evaluate_model(
            _model, times, param_set['Lx'], param_set['Ly'],
            param_set['D11'], param_set['D12'], param_set['D21'], param_set['D22'], _hash
        )
    except RuntimeError as e:
        logger.error(f"evaluate_model failed: {str(e)}")
        st.error(f"evaluate_model failed: {str(e)}")
        return None, None
    
    solution = {
        'params': param_set,
        'X': X,
        'Y': Y,
        'c1_preds': c1_preds,
        'c2_preds': c2_preds,
        'times': times,
        'loss_history': {},
        'orientation_note': 'c1_preds and c2_preds are arrays of shape (50,50) where rows (i) correspond to y-coordinates and columns (j) correspond to x-coordinates for matplotlib.'
    }
    
    solution_filename = os.path.join(output_dir, 
        f"solution_cu_selfdiffusion_ly_{param_set['Ly']:.1f}_tmax_{param_set['t_max']:.1f}.pkl")
    
    try:
        with open(solution_filename, 'wb') as f:
            pickle.dump(solution, f)
        logger.info(f"Saved solution to {solution_filename}")
    except Exception as e:
        logger.error(f"Failed to save solution: {str(e)}")
        st.error(f"Failed to save solution: {str(e)}")
        return None, None
    
    return solution_filename, solution

@st.cache_data(ttl=3600, show_spinner=False)
def generate_vts_time_series(solution, output_dir, _hash):
    os.makedirs(output_dir, exist_ok=True)
    Lx = solution['params']['Lx']
    Ly = solution['params']['Ly']
    times = solution['times']
    
    vts_files = []
    nx, ny = 50, 50
    
    for t_idx, t_val in enumerate(times):
        c1_xy = solution['c1_preds'][t_idx].T  # [x,y] for VTK
        c2_xy = solution['c2_preds'][t_idx].T  # [x,y] for VTK
        
        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        z = np.zeros((nx, ny))
        grid = pv.StructuredGrid()
        X, Y = np.meshgrid(x, y, indexing='ij')
        points = np.stack([X.ravel(), Y.ravel(), z.ravel()], axis=1)
        grid.points = points
        grid.dimensions = (nx, ny, 1)
        
        grid.point_data['Cu_Concentration'] = c1_xy.ravel()
        grid.point_data['Ni_Concentration'] = c2_xy.ravel()
        
        vts_filename = os.path.join(output_dir, 
            f'concentration_ly_{Ly:.1f}_t_{t_val:.1f}.vts')
        
        try:
            grid.save(vts_filename)
            vts_files.append((t_val, vts_filename))
            logger.info(f"Saved VTS file to {vts_filename}")
        except Exception as e:
            logger.error(f"Failed to save VTS file for t={t_val:.1f}: {str(e)}")
            st.error(f"Failed to save VTS file for t={t_val:.1f}: {str(e)}")
    
    pvd_filename = os.path.join(output_dir, 
        f'concentration_time_series_ly_{Ly:.1f}.pvd')
    
    try:
        pvd_content = ['<?xml version="1.0"?>']
        pvd_content.append('<VTKFile type="Collection" version="0.1">')
        pvd_content.append('  <Collection>')
        
        for t_val, vts_file in vts_files:
            relative_path = os.path.basename(vts_file)
            pvd_content.append(f'    <DataSet timestep="{t_val}" group="" part="0" file="{relative_path}"/>')
        
        pvd_content.append('  </Collection>')
        pvd_content.append('</VTKFile>')
        
        with open(pvd_filename, 'w') as f:
            f.write('\n'.join(pvd_content))
        
        logger.info(f"Saved PVD collection file to {pvd_filename}")
    except Exception as e:
        logger.error(f"Failed to create PVD file: {str(e)}")
        st.error(f"Failed to create PVD file: {str(e)}")
        pvd_filename = None
    
    return vts_files, pvd_filename

@st.cache_data(ttl=3600, show_spinner=False)
def generate_vtu_time_series(solution, output_dir, _hash):
    os.makedirs(output_dir, exist_ok=True)
    Lx = solution['params']['Lx']
    Ly = solution['params']['Ly']
    times = solution['times']
    
    nx, ny = 50, 50
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    z = np.zeros((nx, ny))
    X, Y = np.meshgrid(x, y, indexing='ij')
    points = np.stack([X.ravel(), Y.ravel(), z.ravel()], axis=1)
    
    # Define cells (quads for a 2D grid)
    cells = []
    cell_types = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            idx = i + j * nx
            cell = [4, idx, idx + 1, idx + nx + 1, idx + nx]  # Quad: bottom-left, bottom-right, top-right, top-left
            cells.extend(cell)
            cell_types.append(pv.CellType.QUAD)
    
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    
    for t_idx, t_val in enumerate(times):
        c1_xy = solution['c1_preds'][t_idx].T  # [x,y] for VTK
        c2_xy = solution['c2_preds'][t_idx].T  # [x,y] for VTK
        grid.point_data[f'Cu_Concentration_t{t_val:.1f}'] = c1_xy.ravel()
        grid.point_data[f'Ni_Concentration_t{t_val:.1f}'] = c2_xy.ravel()
    
    vtu_filename = os.path.join(output_dir, 
        f'concentration_time_series_ly_{Ly:.1f}.vtu')
    
    try:
        grid.save(vtu_filename)
        logger.info(f"Saved VTU file to {vtu_filename}")
    except Exception as e:
        logger.error(f"Failed to save VTU file: {str(e)}")
        st.error(f"Failed to save VTU file: {str(e)}")
        return None
    
    return vtu_filename

@st.cache_data(ttl=3600, show_spinner=False)
def create_zip_file(_files, output_dir, _hash):
    os.makedirs(output_dir, exist_ok=True)
    zip_buffer = io.BytesIO()
    try:
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in _files:
                if os.path.exists(file_path):
                    zip_file.write(file_path, os.path.basename(file_path))
                else:
                    logger.warning(f"File not found for zipping: {file_path}")
        
        zip_filename = os.path.join(output_dir, f'pinn_solutions_cu_selfdiffusion_ly_{Ly:.1f}.zip')
        with open(zip_filename, 'wb') as f:
            f.write(zip_buffer.getvalue())
        logger.info(f"Created ZIP file: {zip_filename}")
        return zip_filename
    except Exception as e:
        logger.error(f"Failed to create ZIP file: {str(e)}")
        st.error(f"Failed to create ZIP file: {str(e)}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_file_bytes(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return f.read()
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def train_and_generate_solution(_model, loss_history, output_dir, _hash_key):
    os.makedirs(output_dir, exist_ok=True)
    
    if _model is None or loss_history is None:
        return None, None
    
    times = np.linspace(0, T_max, 50)
    param_set = {
        'D11': D11, 'D12': D12, 'D21': D21, 'D22': D22,
        'Lx': Lx, 'Ly': Ly, 't_max': T_max,
        'C_Cu': C_CU_TOP, 'C_Ni': C_NI_TOP,
        'epochs': epochs
    }
    
    solution_filename, solution = generate_and_save_solution(
        _model, times, param_set, output_dir, _hash_key
    )
    
    if solution is None:
        return None, None
    
    solution['loss_history'] = loss_history
    
    loss_plot_filename = plot_losses(loss_history, output_dir, _hash_key)
    profile_plot_filename = plot_2d_profiles(solution, -1, output_dir, _hash_key)
    vts_files, pvd_file = generate_vts_time_series(solution, output_dir, _hash_key)
    vtu_file = generate_vtu_time_series(solution, output_dir, _hash_key)
    
    return solution, {
        'solution_file': solution_filename,
        'loss_plot': loss_plot_filename,
        'profile_plot': profile_plot_filename,
        'vts_files': vts_files,
        'pvd_file': pvd_file,
        'vtu_file': vtu_file
    }

def initialize_session_state():
    if 'training_complete' not in st.session_state:
        st.session_state.training_complete = False
    if 'solution_data' not in st.session_state:
        st.session_state.solution_data = None
    if 'file_data' not in st.session_state:
        st.session_state.file_data = {}
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'current_hash' not in st.session_state:
        st.session_state.current_hash = None

def store_solution_in_session(_hash_key, solution, file_info, model):
    st.session_state.training_complete = True
    st.session_state.solution_data = solution
    st.session_state.file_data = file_info
    st.session_state.model = model
    st.session_state.current_hash = _hash_key

def main():
    #st.title("PINN Simulation: Cu-Ni Diffusion")
    st.title("2D PINN Simulation: Cu Self-Diffusion for Liquid Solder Height = 50 micrometers")
    
    initialize_session_state()
    current_hash = get_cache_key(Ly, C_CU_TOP, C_NI_TOP, epochs, lr)
    
    # Check for cached results
    if st.session_state.training_complete and st.session_state.current_hash == current_hash:
        solution = st.session_state.solution_data
        file_info = st.session_state.file_data
        model = st.session_state.model
        st.info("Displaying cached results.")
    else:
        solution = None
        file_info = {}
        model = None
        st.warning("No results available. Click 'Run Simulation' to generate results.")
    
    # Run simulation only when button is clicked
    if st.button("Run Simulation"):
        try:
            with st.spinner("Running simulation..."):
                model, loss_history = train_model(
                    D11, D12, D21, D22, Lx, Ly, T_max, C_CU_TOP, C_NI_TOP, epochs, lr, OUTPUT_DIR, current_hash
                )
                
                if model is None or loss_history is None:
                    st.error("Simulation failed!")
                    return
                
                solution, file_info = train_and_generate_solution(
                    model, loss_history, OUTPUT_DIR, current_hash
                )
                
                if solution is None:
                    st.error("Solution generation failed!")
                    return
                
                store_solution_in_session(current_hash, solution, file_info, model)
                st.success("Simulation completed successfully!")
        
        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}")
            st.error(f"Simulation failed: {str(e)}")
            return
    
    # Display results only if available
    if solution and file_info:
        with st.expander("Training Logs", expanded=False):
            log_file = os.path.join(OUTPUT_DIR, 'training.log')
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    st.text(f.read())
        
        st.subheader("Training Loss")
        st.image(file_info['loss_plot'])
        
        st.subheader("Boundary Condition Validation")
        bc_results = validate_boundary_conditions(solution)
        st.metric("Boundary Conditions", "✓" if bc_results['valid'] else "✗", 
                f"{len(bc_results['details'])} issues")
        with st.expander("Boundary Condition Details"):
            for issue in bc_results['details']:
                st.write(f"• {issue}")
        
        st.subheader("2D Concentration Profiles (Final Time Step)")
        st.image(file_info['profile_plot'])
        
        st.subheader("Download Files")
        solution_filename = file_info.get('solution_file')
        if solution_filename and os.path.exists(solution_filename):
            solution_data = get_file_bytes(solution_filename)
            if solution_data:
                st.download_button(
                    label="Download Solution (.pkl)",
                    data=solution_data,
                    file_name=os.path.basename(solution_filename),
                    mime="application/octet-stream"
                )
        
        for file_type, file_path in [
            ("Loss Plot", file_info['loss_plot']),
            ("2D Profile Plot", file_info['profile_plot'])
        ]:
            if os.path.exists(file_path):
                file_data = get_file_bytes(file_path)
                if file_data:
                    st.download_button(
                        label=f"Download {file_type} (.png)",
                        data=file_data,
                        file_name=os.path.basename(file_path),
                        mime="image/png"
                    )
        
        st.subheader("Download Time Series Files")
        if file_info.get('pvd_file') and os.path.exists(file_info['pvd_file']):
            pvd_data = get_file_bytes(file_info['pvd_file'])
            if pvd_data:
                st.download_button(
                    label="Download VTS Time Series (.pvd + .vts)",
                    data=pvd_data,
                    file_name=os.path.basename(file_info['pvd_file']),
                    mime="application/xml",
                    help="Download the PVD collection file. Keep all .vts files in the same folder."
                )
        
        if file_info.get('vtu_file') and os.path.exists(file_info['vtu_file']):
            vtu_data = get_file_bytes(file_info['vtu_file'])
            if vtu_data:
                st.download_button(
                    label="Download VTU Time Series (.vtu)",
                    data=vtu_data,
                    file_name=os.path.basename(file_info['vtu_file']),
                    mime="application/xml",
                    help="Single VTU file with all timesteps."
                )
        
        st.subheader("Download Individual Time Steps")
        for t_val, vts_file in file_info.get('vts_files', []):
            if os.path.exists(vts_file):
                vts_data = get_file_bytes(vts_file)
                if vts_data:
                    st.download_button(
                        label=f"Download Time = {t_val:.1f} s (.vts)",
                        data=vts_data,
                        file_name=os.path.basename(vts_file),
                        mime="application/xml"
                    )
        
        st.subheader("Download All Files as ZIP")
        if st.button("Generate ZIP File"):
            with st.spinner("Creating ZIP file..."):
                files_to_zip = [
                    file_info['loss_plot'], 
                    file_info['profile_plot']
                ]
                if solution_filename:
                    files_to_zip.append(solution_filename)
                for _, vts_file in file_info.get('vts_files', []):
                    files_to_zip.append(vts_file)
                if file_info.get('pvd_file'):
                    files_to_zip.append(file_info['pvd_file'])
                if file_info.get('vtu_file'):
                    files_to_zip.append(file_info['vtu_file'])
                
                zip_filename = create_zip_file(files_to_zip, OUTPUT_DIR, (Ly,))
                
                if zip_filename and os.path.exists(zip_filename):
                    zip_data = get_file_bytes(zip_filename)
                    if zip_data:
                        st.download_button(
                            label="Download All Files (.zip)",
                            data=zip_data,
                            file_name=os.path.basename(zip_filename),
                            mime="application/zip"
                        )

if __name__ == "__main__":
    main()
