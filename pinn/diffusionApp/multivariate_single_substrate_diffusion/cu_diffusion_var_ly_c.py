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
C_CU_TOP = 0.0
Lx = 60.0  # Domain width (μm)
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

class ScaledPINN(nn.Module):
    def __init__(self, D11, Lx, Ly, T_max, C_Cu_bottom):
        super().__init__()
        self.D11 = D11
        self.Lx = Lx
        self.Ly = Ly
        self.T_max = T_max
        self.C_Cu_bottom = C_Cu_bottom
        
        self.net = nn.Sequential(
            nn.Linear(3, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1),
            SmoothSigmoid(slope=0.5),
            nn.Linear(1, 1, bias=False),
        )
        
        self.net[6].weight.data.fill_(C_Cu_bottom)

    def forward(self, x, y, t):
        x_norm = x / self.Lx
        y_norm = y / self.Ly
        t_norm = t / self.T_max
        inputs = torch.cat([x_norm, y_norm, t_norm], dim=1)
        return self.net(inputs)

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
    
    c_t = torch.autograd.grad(c_pred, t, grad_outputs=torch.ones_like(c_pred),
                              create_graph=True, retain_graph=True)[0]
    
    lap_c = laplacian(c_pred, x, y)
    
    residual = c_t - (model.D11 * lap_c)
    return torch.mean(residual**2)

def boundary_loss_bottom(model):
    num = 200
    x = torch.rand(num, 1, requires_grad=True) * model.Lx
    y = torch.zeros(num, 1, requires_grad=True)
    t = torch.rand(num, 1, requires_grad=True) * model.T_max
    
    c_pred = model(x, y, t)
    return torch.mean((c_pred - model.C_Cu_bottom)**2)

def boundary_loss_top(model):
    num = 200
    x = torch.rand(num, 1, requires_grad=True) * model.Lx
    y = torch.full((num, 1), model.Ly, requires_grad=True)
    t = torch.rand(num, 1, requires_grad=True) * model.T_max
    
    c_pred = model(x, y, t)
    return torch.mean((c_pred - C_CU_TOP)**2)

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
        grad_x_left = torch.autograd.grad(
            c_left, x_left,
            grad_outputs=torch.ones_like(c_left),
            create_graph=True, retain_graph=True
        )[0]
        
        grad_x_right = torch.autograd.grad(
            c_right, x_right,
            grad_outputs=torch.ones_like(c_right),
            create_graph=True, retain_graph=True
        )[0]
        
        grad_x_left = grad_x_left if grad_x_left is not None else torch.zeros_like(c_left)
        grad_x_right = grad_x_right if grad_x_right is not None else torch.zeros_like(c_right)
        
        return (torch.mean(grad_x_left**2) + torch.mean(grad_x_right**2))
    
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
        'top_bc': True,
        'bottom_bc': True,
        'left_flux': True,
        'right_flux': True,
        'details': []
    }
    t_idx = -1
    c = solution['c_preds'][t_idx]
    
    top_mean = np.mean(c[:, -1])
    if abs(top_mean - C_CU_TOP) > tolerance:
        results['top_bc'] = False
        results['details'].append(f"Top: {top_mean:.2e} != {C_CU_TOP:.2e}")
    
    bottom_mean = np.mean(c[:, 0])
    if abs(bottom_mean - solution['params']['C_Cu_bottom']) > tolerance:
        results['bottom_bc'] = False
        results['details'].append(f"Bottom: {bottom_mean:.2e} != {solution['params']['C_Cu_bottom']:.2e}")
    
    left_flux = np.mean(np.abs(c[1, :] - c[0, :]))
    right_flux = np.mean(np.abs(c[-1, :] - c[-2, :]))
    if left_flux > tolerance:
        results['left_flux'] = False
        results['details'].append(f"Left flux: {left_flux:.2e}")
    if right_flux > tolerance:
        results['right_flux'] = False
        results['details'].append(f"Right flux: {right_flux:.2e}")
    
    results['valid'] = all([
        results['top_bc'], results['bottom_bc'],
        results['left_flux'], results['right_flux']
    ])
    return results

@st.cache_data(ttl=3600, show_spinner=False)
def plot_losses(loss_history, output_dir, _hash, Ly, C_Cu_bottom):
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
    plt.title(f'Training Loss for Ly = {Ly:.1f} μm, c1 = {C_Cu_bottom:.1e}')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f'loss_plot_ly_{Ly:.1f}_c1_{C_Cu_bottom:.1e}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved loss plot to {plot_filename}")
    return plot_filename

@st.cache_data(ttl=3600, show_spinner=False)
def plot_2d_profiles(solution, time_idx, output_dir, _hash):
    t_val = solution['times'][time_idx]
    Ly = solution['params']['Ly']
    C_Cu_bottom = solution['params']['C_Cu_bottom']
    
    plt.figure(figsize=(6, 5))
    im = plt.imshow(solution['c_preds'][time_idx], origin='lower',
                    extent=[0, Lx, 0, Ly], cmap='viridis',
                    vmin=0, vmax=C_Cu_bottom)
    plt.title(f'Cu Concentration (t={t_val:.1f} s)')
    plt.xlabel('x (μm)')
    plt.ylabel('y (μm)')
    plt.grid(True, alpha=0.3)
    plt.colorbar(im, label='Cu Conc. (mol/cc)', format='%.1e')
    
    plt.suptitle(f'2D Profile (Ly={Ly:.0f} μm, c1={C_Cu_bottom:.1e})', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f'profile_ly_{Ly:.1f}_c1_{C_Cu_bottom:.1e}_t_{t_val:.1f}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved profile plot to {plot_filename}")
    return plot_filename

@st.cache_resource(ttl=3600, show_spinner=False)
def train_model(D11, Lx, Ly, T_max, C_Cu_bottom, epochs, lr, output_dir, _hash):
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Starting training with Ly={Ly}, C_Cu_bottom={C_Cu_bottom}, epochs={epochs}, lr={lr}")
    model = ScaledPINN(D11, Lx, Ly, T_max, C_Cu_bottom)
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
def evaluate_model(_model, times, Lx, Ly, _hash):
    x = torch.linspace(0, Lx, 50, requires_grad=False)
    y = torch.linspace(0, Ly, 50, requires_grad=False)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    c_preds = []
    for t_val in times:
        t = torch.full((X.numel(), 1), t_val, requires_grad=False)
        c_pred = _model(X.reshape(-1,1), Y.reshape(-1,1), t)
        try:
            c = c_pred.detach().numpy().reshape(50,50).T  # [y,x] for matplotlib
        except RuntimeError as e:
            logger.error(f"Failed to convert concentration predictions to NumPy: {str(e)}")
            raise e
        
        c_preds.append(c)
    
    return X.numpy(), Y.numpy(), c_preds

@st.cache_data(ttl=3600, show_spinner=False)
def generate_and_save_solution(_model, times, param_set, output_dir, _hash):
    os.makedirs(output_dir, exist_ok=True)
    if _model is None:
        logger.error("Model is None, cannot generate solution")
        return None, None
    
    try:
        X, Y, c_preds = evaluate_model(
            _model, times, param_set['Lx'], param_set['Ly'], _hash
        )
    except RuntimeError as e:
        logger.error(f"evaluate_model failed: {str(e)}")
        st.error(f"evaluate_model failed: {str(e)}")
        return None, None
    
    solution = {
        'params': param_set,
        'X': X,
        'Y': Y,
        'c_preds': c_preds,
        'times': times,
        'loss_history': {},
        'orientation_note': 'c_preds are arrays of shape (50,50) where rows (i) correspond to y-coordinates and columns (j) correspond to x-coordinates for matplotlib.'
    }
    
    Ly = param_set['Ly']
    C_Cu_bottom = param_set['C_Cu_bottom']
    solution_filename = os.path.join(output_dir,
        f"solution_cu_selfdiffusion_ly_{Ly:.1f}_c1_{C_Cu_bottom:.1e}.pkl")
    
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
    C_Cu_bottom = solution['params']['C_Cu_bottom']
    times = solution['times']
    
    vts_files = []
    nx, ny = 50, 50
    
    for t_idx, t_val in enumerate(times):
        c_xy = solution['c_preds'][t_idx].T  # [x,y] for VTK
        
        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        z = np.zeros((nx, ny))
        grid = pv.StructuredGrid()
        X, Y = np.meshgrid(x, y, indexing='ij')
        points = np.stack([X.ravel(), Y.ravel(), z.ravel()], axis=1)
        grid.points = points
        grid.dimensions = (nx, ny, 1)
        
        grid.point_data['Cu_Concentration'] = c_xy.ravel()
        
        vts_filename = os.path.join(output_dir,
            f'concentration_ly_{Ly:.1f}_c1_{C_Cu_bottom:.1e}_t_{t_val:.1f}.vts')
        
        try:
            grid.save(vts_filename)
            vts_files.append((t_val, vts_filename))
            logger.info(f"Saved VTS file to {vts_filename}")
        except Exception as e:
            logger.error(f"Failed to save VTS file for t={t_val:.1f}: {str(e)}")
            st.error(f"Failed to save VTS file for t={t_val:.1f}: {str(e)}")
    
    pvd_filename = os.path.join(output_dir,
        f'concentration_time_series_ly_{Ly:.1f}_c1_{C_Cu_bottom:.1e}.pvd')
    
    try:
        pvd_content = ['<?xml version="1.0"?>']
        pvd_content.append('<VTKFile type="Collection" version="0.1">')
        pvd_content.append(' <Collection>')
        
        for t_val, vts_file in vts_files:
            relative_path = os.path.basename(vts_file)
            pvd_content.append(f' <DataSet timestep="{t_val}" group="" part="0" file="{relative_path}"/>')
        
        pvd_content.append(' </Collection>')
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
    C_Cu_bottom = solution['params']['C_Cu_bottom']
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
        c_xy = solution['c_preds'][t_idx].T  # [x,y] for VTK
        grid.point_data[f'Cu_Concentration_t{t_val:.1f}'] = c_xy.ravel()
    
    vtu_filename = os.path.join(output_dir,
        f'concentration_time_series_ly_{Ly:.1f}_c1_{C_Cu_bottom:.1e}.vtu')
    
    try:
        grid.save(vtu_filename)
        logger.info(f"Saved VTU file to {vtu_filename}")
    except Exception as e:
        logger.error(f"Failed to save VTU file: {str(e)}")
        st.error(f"Failed to save VTU file: {str(e)}")
        return None
    
    return vtu_filename

@st.cache_data(ttl=3600, show_spinner=False)
def create_zip_file(_files, output_dir, _hash, Ly, C_Cu_bottom):
    os.makedirs(output_dir, exist_ok=True)
    zip_buffer = io.BytesIO()
    try:
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in _files:
                if os.path.exists(file_path):
                    zip_file.write(file_path, os.path.basename(file_path))
                else:
                    logger.warning(f"File not found for zipping: {file_path}")
        
        zip_filename = os.path.join(output_dir, f'pinn_solutions_cu_selfdiffusion_ly_{Ly:.1f}_c1_{C_Cu_bottom:.1e}.zip')
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
def train_and_generate_solution(_model, loss_history, output_dir, _hash_key, Ly, C_Cu_bottom):
    os.makedirs(output_dir, exist_ok=True)
    
    if _model is None or loss_history is None:
        return None, None
    
    times = np.linspace(0, T_max, 50)
    param_set = {
        'D11': _model.D11,
        'Lx': Lx, 'Ly': Ly, 't_max': T_max,
        'C_Cu_bottom': C_Cu_bottom,
        'epochs': epochs
    }
    
    solution_filename, solution = generate_and_save_solution(
        _model, times, param_set, output_dir, _hash_key
    )
    
    if solution is None:
        return None, None
    
    solution['loss_history'] = loss_history
    
    loss_plot_filename = plot_losses(loss_history, output_dir, _hash_key, Ly, C_Cu_bottom)
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
    st.title("2D PINN Simulation: Cu Self-Diffusion")
    
    Ly = st.number_input("Domain height Ly (μm)", min_value=30.0, max_value=90.0, value=30.0, step=1.0)
    D11 = st.number_input("Diffusion coefficient D11 (μm²/s)", value=0.006, step=0.0001, format="%.4f")
    c1 = st.number_input("Bottom Cu concentration c1 (mol/cc)", min_value=1.0e-4, max_value=6.0e-4, value=1.0e-4, step=1e-5, format="%.1e")
    
    initialize_session_state()
    current_hash = get_cache_key(Ly, c1, epochs, lr, D11)
    
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
                    D11, Lx, Ly, T_max, c1, epochs, lr, OUTPUT_DIR, current_hash
                )
                
                if model is None or loss_history is None:
                    st.error("Simulation failed!")
                    return
                
                solution, file_info = train_and_generate_solution(
                    model, loss_history, OUTPUT_DIR, current_hash, Ly, c1
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
        
        st.subheader("2D Concentration Profile (Final Time Step)")
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
                
                zip_filename = create_zip_file(files_to_zip, OUTPUT_DIR, (Ly,), Ly, c1)
                
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
