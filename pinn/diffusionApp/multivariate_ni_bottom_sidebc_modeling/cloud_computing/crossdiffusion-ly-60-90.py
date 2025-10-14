import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import zipfile
import io
import matplotlib as mpl
import logging
import pyvista as pv

# Ensure output directory exists
OUTPUT_DIR = '/tmp/pinn_solutions'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure Matplotlib for publication-quality figures
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['legend.frameon'] = True
mpl.rcParams['legend.framealpha'] = 0.8
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.alpha'] = 0.3

# Configure logging
logging.basicConfig(level=logging.INFO, filename=os.path.join(OUTPUT_DIR, 'training.log'), filemode='a')
logger = logging.getLogger(__name__)

# Fixed boundary conditions
C_CU_BOTTOM = 1.6e-3  # Top boundary (y=Ly): Cu-rich
C_CU_TOP = 0.0        # Bottom (y=0): Cu-poor
C_NI_BOTTOM = 0.0     # Top: Ni-poor
C_NI_TOP = 1.25e-3    # Bottom: Ni-rich

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

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
    return (torch.mean((c_pred[:, 0] - C_CU_TOP)**2) + 
            torch.mean((c_pred[:, 1] - C_NI_TOP)**2))

def boundary_loss_top(model):
    num = 200
    x = torch.rand(num, 1, requires_grad=True) * model.Lx
    y = torch.full((num, 1), model.Ly, requires_grad=True)
    t = torch.rand(num, 1, requires_grad=True) * model.T_max
    
    c_pred = model(x, y, t)
    return (torch.mean((c_pred[:, 0] - C_CU_BOTTOM)**2) + 
            torch.mean((c_pred[:, 1] - C_NI_BOTTOM)**2))

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
    
    top_cu_mean = np.mean(c1[:, 0])
    top_ni_mean = np.mean(c2[:, 0])
    if abs(top_cu_mean - C_CU_TOP) > tolerance:
        results['top_bc_cu'] = False
        results['details'].append(f"Top Cu: {top_cu_mean:.2e} != {C_CU_TOP:.2e}")
    if abs(top_ni_mean - C_NI_TOP) > tolerance:
        results['top_bc_ni'] = False
        results['details'].append(f"Top Ni: {top_ni_mean:.2e} != {C_NI_TOP:.2e}")
    
    bottom_cu_mean = np.mean(c1[:, -1])
    bottom_ni_mean = np.mean(c2[:, -1])
    if abs(bottom_cu_mean - C_CU_BOTTOM) > tolerance:
        results['bottom_bc_cu'] = False
        results['details'].append(f"Bottom Cu: {bottom_cu_mean:.2e} != {C_CU_BOTTOM:.2e}")
    if abs(bottom_ni_mean - C_NI_BOTTOM) > tolerance:
        results['bottom_bc_ni'] = False
        results['details'].append(f"Bottom Ni: {bottom_ni_mean:.2e} != {C_NI_BOTTOM:.2e}")
    
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

@st.cache_resource
def plot_losses(loss_history, Ly, C_Cu, C_Ni, output_dir, _hash):
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
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title(f'Training Loss for Ly = {Ly:.1f} μm, C_Cu = {C_Cu:.1e}, C_Ni = {C_Ni:.1e}', fontsize=16)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f'loss_plot_ly_{Ly:.1f}_ccu_{C_Cu:.1e}_cni_{C_Ni:.1e}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved loss plot to {plot_filename}")
    return plot_filename

@st.cache_resource
def plot_2d_profiles(solution, time_idx, output_dir, _hash):
    Lx = solution['params']['Lx']
    Ly = solution['params']['Ly']
    t_val = solution['times'][time_idx]
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    im1 = plt.imshow(solution['c1_preds'][time_idx], origin='lower', 
                     extent=[0, Lx, 0, Ly], cmap='viridis',
                     vmin=0, vmax=C_CU_BOTTOM)
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
    plot_filename = os.path.join(output_dir, f'profile_ly_{Ly:.1f}_t_{t_val:.1f}_ccu_{solution['params']['C_Cu']:.1e}_cni_{solution['params']['C_Ni']:.1e}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved profile plot to {plot_filename}")
    return plot_filename

@st.cache_resource
def plot_side_gradients(_model, solution, time_idx, output_dir, _hash):
    Lx = solution['params']['Lx']
    Ly = solution['params']['Ly']
    t_val = solution['times'][time_idx]
    
    y = torch.linspace(0, Ly, 50, requires_grad=True)
    t = torch.full((50, 1), t_val, requires_grad=True)
    
    x_left = torch.zeros(50, 1, requires_grad=True)
    c_left = _model(x_left, y.reshape(-1, 1), t)
    try:
        grad_cu_x_left = torch.autograd.grad(c_left[:, 0], x_left,
                                            grad_outputs=torch.ones_like(c_left[:, 0]),
                                            create_graph=True)[0]
        grad_ni_x_left = torch.autograd.grad(c_left[:, 1], x_left,
                                            grad_outputs=torch.ones_like(c_left[:, 1]),
                                            create_graph=True)[0]
        grad_cu_x_left_np = grad_cu_x_left.detach().numpy()
        grad_ni_x_left_np = grad_ni_x_left.detach().numpy()
    except Exception as e:
        logger.error(f"Gradient computation failed in plot_side_gradients (left): {str(e)}")
        grad_cu_x_left_np = np.zeros(50)
        grad_ni_x_left_np = np.zeros(50)
    
    x_right = torch.full((50, 1), Lx, requires_grad=True)
    c_right = _model(x_right, y.reshape(-1, 1), t)
    try:
        grad_cu_x_right = torch.autograd.grad(c_right[:, 0], x_right,
                                             grad_outputs=torch.ones_like(c_right[:, 0]),
                                             create_graph=True)[0]
        grad_ni_x_right = torch.autograd.grad(c_right[:, 1], x_right,
                                             grad_outputs=torch.ones_like(c_right[:, 1]),
                                             create_graph=True)[0]
        grad_cu_x_right_np = grad_cu_x_right.detach().numpy()
        grad_ni_x_right_np = grad_ni_x_right.detach().numpy()
    except Exception as e:
        logger.error(f"Gradient computation failed in plot_side_gradients (right): {str(e)}")
        grad_cu_x_right_np = np.zeros(50)
        grad_ni_x_right_np = np.zeros(50)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(y.detach().numpy(), grad_cu_x_left_np, label='Cu (x=0)', color='blue')
    plt.plot(y.detach().numpy(), grad_ni_x_left_np, label='Ni (x=0)', color='red')
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title(f'Gradients at Left Boundary (t={t_val:.1f} s)')
    plt.xlabel('y (μm)')
    plt.ylabel('∂c/∂x')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(y.detach().numpy(), grad_cu_x_right_np, label='Cu (x=Lx)', color='blue')
    plt.plot(y.detach().numpy(), grad_ni_x_right_np, label='Ni (x=Lx)', color='red')
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title(f'Gradients at Right Boundary (t={t_val:.1f} s)')
    plt.xlabel('y (μm)')
    plt.ylabel('∂c/∂x')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.suptitle(f'Side Boundary Gradients (Ly={Ly:.0f} μm)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f'gradients_ly_{Ly:.1f}_t_{t_val:.1f}_ccu_{solution['params']['C_Cu']:.1e}_cni_{solution['params']['C_Ni']:.1e}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved gradient plot to {plot_filename}")
    return plot_filename

@st.cache_resource
def train_pinn_cached(D11, D12, D21, D22, Lx, Ly, T_max, C_Cu, C_Ni, epochs, lr, output_dir, _hash):
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Starting training with Ly={Ly}, C_Cu={C_Cu}, C_Ni={C_Ni}, epochs={epochs}, lr={lr}")
    model = DualScaledPINN(D11, D12, D21, D22, Lx, Ly, T_max, C_Cu, C_Ni)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)
    
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
        
        scheduler.step(loss)
        
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

@st.cache_resource
def evaluate_model(_model, times, Lx, Ly, D11, D12, D21, D22, _hash):
    x = torch.linspace(0, Lx, 50, requires_grad=False)
    y = torch.linspace(0, Ly, 50, requires_grad=False)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    c1_preds, c2_preds, J1_preds, J2_preds = [], [], [], []
    for t_val in times:
        t = torch.full((X.numel(), 1), t_val, requires_grad=False)
        c_pred = _model(X.reshape(-1,1), Y.reshape(-1,1), t)
        try:
            c1 = c_pred[:,0].detach().numpy().reshape(50,50).T
            c2 = c_pred[:,1].detach().numpy().reshape(50,50).T
        except RuntimeError as e:
            logger.error(f"Failed to convert concentration predictions to NumPy: {str(e)}")
            raise e
        
        c1_preds.append(c1)
        c2_preds.append(c2)
        
        X_np, Y_np = X.numpy(), Y.numpy()
        X_torch = torch.tensor(X_np, dtype=torch.float32, requires_grad=True).reshape(-1, 1)
        Y_torch = torch.tensor(Y_np, dtype=torch.float32, requires_grad=True).reshape(-1, 1)
        t_torch = torch.full((X_torch.numel(), 1), t_val, dtype=torch.float32, requires_grad=True)
        
        c_pred = _model(X_torch, Y_torch, t_torch)
        c1_pred, c2_pred = c_pred[:, 0:1], c_pred[:, 1:2]
        
        try:
            grad_c1_x = torch.autograd.grad(c1_pred, X_torch, 
                                            grad_outputs=torch.ones_like(c1_pred),
                                            create_graph=True)[0]
            grad_c1_y = torch.autograd.grad(c1_pred, Y_torch,
                                            grad_outputs=torch.ones_like(c1_pred),
                                            create_graph=True)[0]
            grad_c2_x = torch.autograd.grad(c2_pred, X_torch,
                                            grad_outputs=torch.ones_like(c2_pred),
                                            create_graph=True)[0]
            grad_c2_y = torch.autograd.grad(c2_pred, Y_torch,
                                            grad_outputs=torch.ones_like(c2_pred),
                                            create_graph=True)[0]
            
            J1_x = -D11 * grad_c1_x.detach().numpy() - D12 * grad_c2_x.detach().numpy()
            J1_y = -D11 * grad_c1_y.detach().numpy() - D12 * grad_c2_y.detach().numpy()
            J2_x = -D21 * grad_c1_x.detach().numpy() - D22 * grad_c2_x.detach().numpy()
            J2_y = -D21 * grad_c1_y.detach().numpy() - D22 * grad_c2_y.detach().numpy()
        except RuntimeError as e:
            logger.error(f"Failed to compute fluxes: {str(e)}")
            raise e
        
        J1_preds.append((J1_x.reshape(X_np.shape), J1_y.reshape(X_np.shape)))
        J2_preds.append((J2_x.reshape(X_np.shape), J2_y.reshape(X_np.shape)))
    
    return X_np, Y_np, c1_preds, c2_preds, J1_preds, J2_preds

@st.cache_resource
def generate_and_save_solution(_model, times, param_set, output_dir, _hash):
    os.makedirs(output_dir, exist_ok=True)
    if _model is None:
        logger.error("Model is None, cannot generate solution")
        return None, None
    
    try:
        X, Y, c1_preds, c2_preds, J1_preds, J2_preds = evaluate_model(
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
        'J1_preds': J1_preds,
        'J2_preds': J2_preds,
        'times': times,
        'loss_history': {},
        'orientation_note': 'c1_preds and c2_preds are arrays of shape (50,50) where rows (i) correspond to y-coordinates and columns (j) correspond to x-coordinates due to transpose.'
    }
    
    solution_filename = os.path.join(output_dir, 
        f"solution_ly_{param_set['Ly']:.1f}_ccu_{param_set['C_Cu']:.1e}_cni_{param_set['C_Ni']:.1e}_d11_{param_set['D11']:.6f}_d12_{param_set['D12']:.6f}_d21_{param_set['D21']:.6f}_d22_{param_set['D22']:.6f}_lx_{param_set['Lx']:.1f}_tmax_{param_set['t_max']:.1f}.pkl")
    
    try:
        with open(solution_filename, 'wb') as f:
            pickle.dump(solution, f)
        logger.info(f"Saved solution to {solution_filename}")
    except Exception as e:
        logger.error(f"Failed to save solution: {str(e)}")
        st.error(f"Failed to save solution: {str(e)}")
        return None, None
    
    return solution_filename, solution

@st.cache_resource
def generate_vts_files(solution, output_dir, _hash):
    os.makedirs(output_dir, exist_ok=True)
    Lx = solution['params']['Lx']
    Ly = solution['params']['Ly']
    times = solution['times']
    c1_preds = solution['c1_preds']
    c2_preds = solution['c2_preds']
    
    vts_files = []
    nx, ny = 50, 50  # Grid dimensions
    for t_idx, t_val in enumerate(times):
        # Create structured grid
        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        z = np.zeros((nx, ny))  # 2D grid, z=0
        grid = pv.StructuredGrid()
        X, Y = np.meshgrid(x, y, indexing='ij')
        points = np.stack([X.ravel(), Y.ravel(), z.ravel()], axis=1)
        grid.points = points
        grid.dimensions = (nx, ny, 1)
        
        # Add concentration data
        grid.point_data['Cu_Concentration'] = c1_preds[t_idx].ravel()
        grid.point_data['Ni_Concentration'] = c2_preds[t_idx].ravel()
        
        # Save VTS file
        vts_filename = os.path.join(output_dir, 
            f'concentration_ly_{solution["params"]["Ly"]:.1f}_t_{t_val:.1f}_ccu_{solution["params"]["C_Cu"]:.1e}_cni_{solution["params"]["C_Ni"]:.1e}.vts')
        try:
            grid.save(vts_filename)
            vts_files.append(vts_filename)
            logger.info(f"Saved VTS file to {vts_filename}")
        except Exception as e:
            logger.error(f"Failed to save VTS file for t={t_val:.1f}: {str(e)}")
            st.error(f"Failed to save VTS file for t={t_val:.1f}: {str(e)}")
    
    return vts_files

@st.cache_resource
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
    
        zip_filename = os.path.join(output_dir, f'pinn_solutions_ly_{_hash[0]:.1f}_ccu_{_hash[1]:.1e}_cni_{_hash[2]:.1e}.zip')
        with open(zip_filename, 'wb') as f:
            f.write(zip_buffer.getvalue())
        logger.info(f"Created ZIP file: {zip_filename}")
        return zip_filename
    except Exception as e:
        logger.error(f"Failed to create ZIP file: {str(e)}")
        st.error(f"Failed to create ZIP file: {str(e)}")
        return None

def main():
    st.title("PINN Training and Visualization App")
    
    st.sidebar.header("Model Parameters")
    Ly = st.sidebar.selectbox("Ly (μm)", [30.0, 50.0, 90.0, 120.0], index=0)
    
    # Text inputs for C_Cu and C_Ni with validation
    C_Cu_input = st.sidebar.text_input("C_Cu (mol/cc, 0 to 3.0e-3)", value="1.6e-3")
    C_Ni_input = st.sidebar.text_input("C_Ni (mol/cc, 0 to 1.8e-3)", value="1.25e-3")
    
    try:
        C_Cu = float(C_Cu_input)
        if not (0.0 <= C_Cu <= 3.0e-3):
            st.sidebar.error("C_Cu must be between 0 and 3.0e-3 mol/cc")
            return
    except ValueError:
        st.sidebar.error("Invalid input for C_Cu. Enter a number (e.g., 1.6e-3)")
        return
    
    try:
        C_Ni = float(C_Ni_input)
        if not (0.0 <= C_Ni <= 1.8e-3):
            st.sidebar.error("C_Ni must be between 0 and 1.8e-3 mol/cc")
            return
    except ValueError:
        st.sidebar.error("Invalid input for C_Ni. Enter a number (e.g., 1.25e-3)")
        return
    
    epochs = st.sidebar.slider("Epochs", 1000, 10000, 5000, 1000)
    lr = st.sidebar.slider("Learning Rate", 1e-4, 1e-2, 1e-3, 1e-4, format="%.4f")
    
    D11 = 0.006
    D12 = 0.00427
    D21 = 0.003697
    D22 = 0.0054
    Lx = 60.0
    t_max = 200.0
    
    output_dir = OUTPUT_DIR
    hash_key = (Ly, C_Cu, C_Ni, epochs, lr)
    
    param_set = {
        'D11': D11, 'D12': D12, 'D21': D21, 'D22': D22,
        'Lx': Lx, 'Ly': float(Ly), 't_max': t_max,
        'C_Cu': float(C_Cu), 'C_Ni': float(C_Ni),
        'epochs': epochs
    }
    
    if st.button("Train PINN Model"):
        try:
            with st.spinner("Training model..."):
                model, loss_history = train_pinn_cached(
                    D11, D12, D21, D22, Lx, Ly, t_max, C_Cu, C_Ni, epochs, lr, output_dir, hash_key
                )
            
            if model is None or loss_history is None:
                st.error("Training failed, model or loss history is None")
                return
            
            times = np.linspace(0, t_max, 50)
            solution_filename, solution = generate_and_save_solution(model, times, param_set, output_dir, hash_key)
            
            if solution_filename is None or solution is None:
                st.error("Failed to generate solution")
                return
            
            solution['loss_history'] = loss_history
            loss_plot_filename = plot_losses(loss_history, Ly, C_Cu, C_Ni, output_dir, hash_key)
            profile_plot_filename = plot_2d_profiles(solution, -1, output_dir, hash_key)
            gradient_plot_filename = plot_side_gradients(model, solution, -1, output_dir, hash_key)
            vts_files = generate_vts_files(solution, output_dir, hash_key)
            
            st.success(f"Model trained successfully! Saved to {solution_filename}")
            
            with st.expander("Training Logs"):
                log_file = os.path.join(output_dir, 'training.log')
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        st.text(f.read())
            
            st.subheader("Training Loss")
            st.image(loss_plot_filename)
            
            st.subheader("Boundary Condition Validation")
            bc_results = validate_boundary_conditions(solution)
            st.metric("Boundary Conditions", "✓" if bc_results['valid'] else "✗", 
                     f"{len(bc_results['details'])} issues")
            with st.expander("Boundary Condition Details"):
                for issue in bc_results['details']:
                    st.write(f"• {issue}")
            
            st.subheader("2D Concentration Profiles (Final Time Step)")
            st.image(profile_plot_filename)
            
            st.subheader("Side Boundary Gradients (Final Time Step)")
            st.image(gradient_plot_filename)
            
            st.subheader("Download Individual Files")
            files_to_download = [
                (solution_filename, "Solution (.pkl)", "application/octet-stream"),
                (loss_plot_filename, "Loss Plot (.png)", "image/png"),
                (profile_plot_filename, "2D Profile Plot (.png)", "image/png"),
                (gradient_plot_filename, "Gradient Plot (.png)", "image/png")
            ]
            
            for file_path, label, mime in files_to_download:
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        st.download_button(
                            label=label,
                            data=f,
                            file_name=os.path.basename(file_path),
                            mime=mime
                        )
                else:
                    st.warning(f"File not found: {file_path}")
            
            st.subheader("Download VTS Files")
            for vts_file in vts_files:
                if os.path.exists(vts_file):
                    with open(vts_file, 'rb') as f:
                        t_val = float(os.path.basename(vts_file).split('_t_')[1].split('_')[0])
                        st.download_button(
                            label=f"Concentration Profile t={t_val:.1f} s (.vts)",
                            data=f,
                            file_name=os.path.basename(vts_file),
                            mime="application/xml"
                        )
                else:
                    st.warning(f"VTS file not found: {vts_file}")
            
            st.subheader("Download All Files as ZIP")
            files_to_zip = [solution_filename, loss_plot_filename, profile_plot_filename, gradient_plot_filename] + vts_files
            zip_filename = create_zip_file(files_to_zip, output_dir, hash_key)
            if zip_filename and os.path.exists(zip_filename):
                with open(zip_filename, 'rb') as f:
                    st.download_button(
                        label="Download All Files (.zip)",
                        data=f,
                        file_name=os.path.basename(zip_filename),
                        mime="application/zip"
                    )
            else:
                st.error("Failed to create ZIP file")
        
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            st.error(f"Training pipeline failed: {str(e)}")
    
    st.sidebar.markdown("""
    **Notes for Streamlit Cloud:**
    - Files are saved to `/tmp/pinn_solutions`.
    - Use the 'Train PINN Model' button to start training.
    - Enter C_Cu (0 to 3.0e-3) and C_Ni (0 to 1.8e-3) as numbers (e.g., 1.6e-3).
    - Download individual `.pkl`, `.png`, or `.vts` files, or all as a ZIP.
    - All operations are cached to prevent crashes and redundant file generation.
    - Check logs for debugging information.
    - VTS files can be visualized in ParaView.
    """)

if __name__ == "__main__":
    main()
