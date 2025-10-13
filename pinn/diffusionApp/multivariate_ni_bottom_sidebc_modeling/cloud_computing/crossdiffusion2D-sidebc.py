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
logging.basicConfig(level=logging.INFO, filename='/tmp/pinn_solutions/training.log', filemode='a')
logger = logging.getLogger(__name__)

# Fixed boundary conditions
C_CU_BOTTOM = 1.6e-3
C_CU_TOP = 0.0
C_NI_BOTTOM = 0.0
C_NI_TOP = 1.25e-3

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

# FIXED: Cleaner, robust side BC enforcement (pure gradient MSE, no scaling/clamping, full requires_grad)
def boundary_loss_sides(model):
    num = 200
    # Left side (x=0) with requires_grad on all inputs
    x_left = torch.zeros(num, 1, dtype=torch.float32, requires_grad=True)
    y_left = torch.rand(num, 1, requires_grad=True) * model.Ly
    t_left = torch.rand(num, 1, requires_grad=True) * model.T_max
    c_left = model(x_left, y_left, t_left)
    
    # Right side (x=Lx) - independent sampling
    x_right = torch.full((num, 1), float(model.Lx), dtype=torch.float32, requires_grad=True)
    y_right = torch.rand(num, 1, requires_grad=True) * model.Ly
    t_right = torch.rand(num, 1, requires_grad=True) * model.T_max
    c_right = model(x_right, y_right, t_right)
    
    try:
        # Gradients for Cu and Ni on left
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
        
        # Gradients on right
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
        
        # Fallback to zeros if grad fails (robustness)
        grad_cu_x_left = grad_cu_x_left if grad_cu_x_left is not None else torch.zeros_like(c_left[:, 0])
        grad_ni_x_left = grad_ni_x_left if grad_ni_x_left is not None else torch.zeros_like(c_left[:, 1])
        grad_cu_x_right = grad_cu_x_right if grad_cu_x_right is not None else torch.zeros_like(c_right[:, 0])
        grad_ni_x_right = grad_ni_x_right if grad_ni_x_right is not None else torch.zeros_like(c_right[:, 1])
        
        # Pure MSE on gradients (no Lx scaling or clamping - keeps loss in network's normalized space)
        return (torch.mean(grad_cu_x_left**2) + 
                torch.mean(grad_ni_x_left**2) +
                torch.mean(grad_cu_x_right**2) + 
                torch.mean(grad_ni_x_right**2))
    
    except RuntimeError as e:
        logger.error(f"Gradient computation failed in boundary_loss_sides: {str(e)}")
        st.error(f"Gradient computation failed: {str(e)}")
        # Return small require_grad tensor to continue training
        return torch.tensor(1e-6, requires_grad=True)

def initial_loss(model):
    num = 500
    x = torch.rand(num, 1, requires_grad=True) * model.Lx
    y = torch.rand(num, 1, requires_grad=True) * model.Ly
    t = torch.zeros(num, 1, requires_grad=True)
    return torch.mean(model(x, y, t)**2)

# ... (validate_boundary_conditions, plot_losses, plot_2d_profiles, plot_side_gradients remain unchanged)

# Rest of the code (train_pinn_cached, evaluate_model, etc.) unchanged - paste as-is

if __name__ == "__main__":
    main()
