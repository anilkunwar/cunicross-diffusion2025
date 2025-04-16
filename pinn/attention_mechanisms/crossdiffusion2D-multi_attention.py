import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pickle
import os
#from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import cdist

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
    def __init__(self, D11, D12, D21, D22, Lx, Ly, T_max):
        super().__init__()
        self.D11 = D11
        self.D12 = D12
        self.D21 = D21
        self.D22 = D22
        self.Lx = Lx
        self.Ly = Ly
        self.T_max = T_max
        
        self.shared_net = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh()
        )
        
        self.cu_head = nn.Sequential(
            nn.Linear(64, 1),
            SmoothSigmoid(slope=0.5),
            nn.Linear(1, 1, bias=False),
        )
        self.ni_head = nn.Sequential(
            nn.Linear(64, 1),
            SmoothSigmoid(slope=0.5),
            nn.Linear(1, 1, bias=False),
        )
        
        self.cu_head[2].weight.data.fill_(1.6e-3)
        self.ni_head[2].weight.data.fill_(1.25e-3)

    def forward(self, x, y, t):
        x_norm = x / self.Lx
        y_norm = y / self.Ly
        t_norm = t / self.T_max
        
        features = self.shared_net(torch.cat([x_norm, y_norm, t_norm], dim=1))
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
    num = 100
    x = torch.rand(num, 1) * model.Lx
    y = torch.zeros(num, 1)
    t = torch.rand(num, 1) * model.T_max
    
    c_pred = model(x, y, t)
    return (torch.mean((c_pred[:, 0] - 1.6e-3)**2) + 
            torch.mean((c_pred[:, 1] - 0.0)**2))

def boundary_loss_top(model):
    num = 100
    x = torch.rand(num, 1) * model.Lx
    y = torch.full((num, 1), model.Ly)
    t = torch.rand(num, 1) * model.T_max
    
    c_pred = model(x, y, t)
    return (torch.mean((c_pred[:, 0] - 0.0)**2) + 
            torch.mean((c_pred[:, 1] - 1.25e-3)**2))

def boundary_loss_sides(model):
    num = 100
    x_left = torch.zeros(num, 1, dtype=torch.float32, requires_grad=True)
    y_left = torch.rand(num, 1) * model.Ly
    t_left = torch.rand(num, 1) * model.T_max
    c_left = model(x_left, y_left, t_left)
    
    x_right = torch.full((num, 1), float(model.Lx), 
                        dtype=torch.float32, requires_grad=True)
    c_right = model(x_right, y_left, t_left)
    
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
    
    return (torch.mean(grad_cu_x_left**2) + 
            torch.mean(grad_ni_x_left**2) +
            torch.mean(grad_cu_x_right**2) + 
            torch.mean(grad_ni_x_right**2))

def initial_loss(model):
    num = 500
    x = torch.rand(num, 1) * model.Lx
    y = torch.rand(num, 1) * model.Ly
    t = torch.zeros(num, 1)
    return torch.mean(model(x, y, t)**2)

@st.cache_resource
def train_PINN(D11, D12, D21, D22, Lx, Ly, T_max, epochs=2000, lr=0.001):
    model = DualScaledPINN(D11, D12, D21, D22, Lx, Ly, T_max)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    x_pde = torch.rand(1000, 1, requires_grad=True) * Lx
    y_pde = torch.rand(1000, 1, requires_grad=True) * Ly
    t_pde = torch.rand(1000, 1, requires_grad=True) * T_max
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = 10*physics_loss(model, x_pde, y_pde, t_pde)
        loss += 100*(boundary_loss_bottom(model) + boundary_loss_top(model))
        loss += 50*boundary_loss_sides(model) + 100*initial_loss(model)
        
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
    return model

@st.cache_data
def evaluate_model(_model, times, Lx, Ly):
    x = torch.linspace(0, Lx, 50)
    y = torch.linspace(0, Ly, 50)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    c1_preds, c2_preds = [], []
    for t_val in times:
        t = torch.full((X.numel(), 1), t_val)
        c_pred = _model(X.reshape(-1,1), Y.reshape(-1,1), t)
        c1 = c_pred[:,0].detach().numpy().reshape(50,50).T
        c2 = c_pred[:,1].detach().numpy().reshape(50,50).T
        c1_preds.append(c1)
        c2_preds.append(c2)
    
    return X.numpy(), Y.numpy(), c1_preds, c2_preds

def generate_parameter_sets(D11, D12, D21, D22, Lx, t_max, epochs):
    Ly_range = np.arange(50, 101, 5)  # 50, 55, ..., 100 (11 values)
    params = []
    for Ly in Ly_range:
        param_set = {
            'D11': D11,
            'D12': D12,
            'D21': D21,
            'D22': D22,
            'Lx': Lx,
            'Ly': float(Ly),
            't_max': t_max,
            'epochs': epochs
        }
        params.append(param_set)
    return params

def train_and_save_solutions(D11, D12, D21, D22, Lx, t_max, epochs, output_dir="pinn_solutions"):
    os.makedirs(output_dir, exist_ok=True)
    params = generate_parameter_sets(D11, D12, D21, D22, Lx, t_max, epochs)
    times = np.linspace(0, t_max, 50)  # 50 frames
    
    for idx, param_set in enumerate(params):
        st.write(f"Training model {idx+1}/{len(params)} for Ly={param_set['Ly']:.1f} μm...")
        with st.spinner(f"Training with Ly={param_set['Ly']:.1f} μm..."):
            model = train_PINN(
                param_set['D11'], param_set['D12'],
                param_set['D21'], param_set['D22'],
                param_set['Lx'], param_set['Ly'],
                param_set['t_max'],
                epochs=param_set['epochs']
            )
            X, Y, c1_preds, c2_preds = evaluate_model(
                model, times, param_set['Lx'], param_set['Ly']
            )
            
            solution = {
                'params': param_set,
                'X': X,
                'Y': Y,
                'c1_preds': c1_preds,
                'c2_preds': c2_preds,
                'times': times
            }
            filename = os.path.join(output_dir, 
                f"solution_ly_{param_set['Ly']:.1f}_d11_{D11:.6f}_d12_{D12:.6f}_d21_{D21:.6f}_d22_{D22:.6f}_lx_{Lx:.1f}_tmax_{t_max:.1f}.pkl")
            with open(filename, 'wb') as f:
                pickle.dump(solution, f)
            st.write(f"Saved solution {idx+1} to {filename}")
    
    return len(params)



# Transformer-inspired attention mechanism
class LyAttentionInterpolator(nn.Module):
    def __init__(self, sigma=5.0):
        super().__init__()
        self.sigma = sigma
        self.num_heads = 4  # Number of attention heads
        self.d_head = 8      # Dimension per head
        
        # Projection layers
        self.W_q = nn.Linear(1, self.num_heads * self.d_head)
        self.W_k = nn.Linear(1, self.num_heads * self.d_head)

    def forward(self, solutions, lys, ly_target):
        """Transformer-style attention interpolation"""
        # Convert to tensors
        lys_tensor = torch.tensor(lys, dtype=torch.float32).unsqueeze(-1)  # [N, 1]
        ly_target_tensor = torch.tensor([ly_target], dtype=torch.float32).unsqueeze(-1)  # [1, 1]
        
        # Project to query/key space
        queries = self.W_q(ly_target_tensor)  # [1, num_heads * d_head]
        keys = self.W_k(lys_tensor)           # [N, num_heads * d_head]
        
        # Reshape for multi-head attention
        queries = queries.view(1, self.num_heads, self.d_head)  # [1, num_heads, d_head]
        keys = keys.view(len(lys), self.num_heads, self.d_head)  # [N, num_heads, d_head]
        
        # Scaled dot-product attention
        attn_logits = torch.einsum('nhd,mhd->nmh', keys, queries) / np.sqrt(self.d_head)
        attn_weights = torch.softmax(attn_logits, dim=1)  # [N, 1, num_heads]
        
        # Combine spatial and attention weights
        spatial_weights = torch.exp(-(torch.tensor(ly_target) - torch.tensor(lys))**2/(2*self.sigma**2))
        combined_weights = (attn_weights.mean(dim=-1).squeeze() * spatial_weights)
        combined_weights /= combined_weights.sum()
        
        return self._physics_aware_interpolation(solutions, combined_weights.detach().numpy(), ly_target)

    def _physics_aware_interpolation(self, solutions, weights, ly_target):
        """Guaranteed to return dictionary with required keys"""
        Lx = solutions[0]['params']['Lx']
        x_grid = np.linspace(0, Lx, 50)
        y_grid = np.linspace(0, ly_target, 50)
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        
        c1_interp = np.zeros((50, 50, 50))
        c2_interp = np.zeros((50, 50, 50))
        
        for t_idx in range(50):
            for sol, weight in zip(solutions, weights):
                scale_factor = ly_target / sol['params']['Ly']
                Y_scaled = sol['Y'][0,:] * scale_factor
                
                interp_c1 = RegularGridInterpolator(
                    (sol['X'][:,0], Y_scaled),
                    sol['c1_preds'][t_idx],
                    bounds_error=False,
                    fill_value=0.0
                )
                interp_c2 = RegularGridInterpolator(
                    (sol['X'][:,0], Y_scaled),
                    sol['c2_preds'][t_idx],
                    bounds_error=False,
                    fill_value=0.0
                )
                
                c1_interp[t_idx] += weight * interp_c1((X, Y))
                c2_interp[t_idx] += weight * interp_c2((X, Y))
        
        # Enforce boundary conditions
        c1_interp[:, :, 0] = 1.6e-3  # Cu bottom
        c2_interp[:, :, -1] = 1.25e-3  # Ni top
        
        return {
            'params': {**solutions[0]['params'], 'Ly': ly_target},
            'X': X, 
            'Y': Y,
            'c1_preds': list(c1_interp),
            'c2_preds': list(c2_interp),
            'times': np.linspace(0, solutions[0]['params']['t_max'], 50), # Update to mitigate TypeError: 'LyAttentionInterpolator' object is not subscriptable
            'attention_weights': weights,
            'interpolated': True
        }
# Use of Gaussian Attention Weights for Interpolation
# Not used in the current model
def attention_weighted_interpolation(solutions, lys, ly_target, sigma=5.0):
    """Attention-based interpolation using multiple Ly solutions.
    
    Args:
        solutions: List of solution dictionaries
        lys: List of corresponding Ly values
        ly_target: Target Ly value
        sigma: Attention bandwidth (controls neighborhood focus)
    
    Returns:
        Interpolated solution with attention weights
    """
    # Convert to numpy arrays
    lys = np.array(lys)
    solution_coords = np.array(lys).reshape(-1, 1)
    target_coord = np.array([[ly_target]])
    
    # Calculate attention weights using Gaussian kernel
    distances = cdist(target_coord, solution_coords).flatten()
    weights = np.exp(-(distances**2)/(2*sigma**2))
    weights /= weights.sum()  # Normalize
    
    # Get common grid dimensions
    Lx = solutions[0]['params']['Lx']
    t_max = solutions[0]['params']['t_max']
    x_coords = np.linspace(0, Lx, 50)
    y_coords = np.linspace(0, ly_target, 50)
    times = np.linspace(0, t_max, 50)
    
    # Initialize arrays for accumulation
    c1_interp = np.zeros((len(times), 50, 50))
    c2_interp = np.zeros((len(times), 50, 50))
    
    # Process each solution
    for idx, (weight, solution) in enumerate(zip(weights, solutions)):
        # Create interpolators for this solution
        X_sol = solution['X'][:,0]
        Y_sol = solution['Y'][0,:] * (ly_target / solution['params']['Ly'])  # Scale Y
        
        for t_idx in range(len(times)):
            # Interpolate Cu
            interp_c1 = RegularGridInterpolator(
                (X_sol, Y_sol), solution['c1_preds'][t_idx],
                method='linear', bounds_error=False, fill_value=None
            )
            # Interpolate Ni
            interp_c2 = RegularGridInterpolator(
                (X_sol, Y_sol), solution['c2_preds'][t_idx],
                method='linear', bounds_error=False, fill_value=None
            )
            
            # Create target grid
            X_target, Y_target = np.meshgrid(x_coords, y_coords, indexing='ij')
            points = np.stack([X_target.flatten(), Y_target.flatten()], axis=1)
            
            # Apply interpolation and accumulate
            c1_interp[t_idx] += weight * interp_c1(points).reshape(50, 50)
            c2_interp[t_idx] += weight * interp_c2(points).reshape(50, 50)
    
    # Create final solution structure
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    param_set = solutions[0]['params'].copy()
    param_set['Ly'] = ly_target
    
    return {
        'params': param_set,
        'X': X, 'Y': Y,
        'c1_preds': list(c1_interp),
        'c2_preds': list(c2_interp),
        'times': times,
        'interpolated': True,
        'attention_weights': weights.tolist(),
        'used_lys': lys.tolist()
    }

def load_and_visualize_solution(D11, D12, D21, D22, Lx, t_max, ly_target, output_dir="pinn_solutions"):
    # Get all solution files
    solution_files = [f for f in os.listdir(output_dir) if f.startswith('solution_ly_') and f.endswith('.pkl')]
    
    if not solution_files:
        st.error("No solutions found. Please train models first.")
        return

    # Initialize variables
    valid_solutions = []
    valid_lys = []
    
    # Load and validate solutions
    for fname in solution_files:
        try:
            # Extract parameters from filename
            parts = fname.split('_')
            file_params = {
                'Ly': float(parts[2]),
                'D11': float(parts[4]),
                'D12': float(parts[6]),
                'D21': float(parts[8]),
                'D22': float(parts[10]),
                'Lx': float(parts[12]),
                't_max': float(parts[14].split('.pkl')[0])
            }
            
            # Parameter validation
            if (abs(file_params['D11'] - D11) < 1e-6 and
                abs(file_params['D12'] - D12) < 1e-6 and
                abs(file_params['D21'] - D21) < 1e-6 and
                abs(file_params['D22'] - D22) < 1e-6 and
                abs(file_params['Lx'] - Lx) < 1e-2 and
                abs(file_params['t_max'] - t_max) < 1e-2):
                
                with open(os.path.join(output_dir, fname), 'rb') as f:
                    solution = pickle.load(f)
                    valid_solutions.append(solution)
                    valid_lys.append(file_params['Ly'])
                    
        except Exception as e:
            st.warning(f"Skipped {fname}: {str(e)}")
            continue

    if not valid_solutions:
        st.error("No valid solutions found matching current parameters")
        return

    # Convert to numpy array
    valid_lys = np.array(valid_lys)
    ly_target = round(ly_target, 1)

    # Find exact match index using vectorized operations
    tolerance = 1e-4
    exact_match_indices = np.where(np.abs(valid_lys - ly_target) < tolerance)[0]

    if exact_match_indices.size > 0:
        # Use first exact match found
        solution = valid_solutions[exact_match_indices[0]]
        solution['interpolated'] = False
    else:
        # Proceed with attention-based interpolation
        try:
            interpolator = LyAttentionInterpolator(sigma=sigma)
            solution = interpolator(valid_solutions, valid_lys, ly_target)
            st.info(f"Interpolated using transformer-like attention over {len(valid_solutions)} solutions")
        except Exception as e:
            st.error(f"Interpolation failed: {str(e)}")
            return

    # Add type validation (should be OUTSIDE the else block)
    if not isinstance(solution, dict):
        st.error("Invalid solution format - expected dictionary")
        return

    # Explicit key access (should be OUTSIDE the else block)
    X = solution['X']
    Y = solution['Y']
    c1_preds = solution['c1_preds']
    c2_preds = solution['c2_preds']
    times = solution['times']
    param_set = solution['params']
    
    # Visualization code (should be OUTSIDE the else block)
    st.subheader("Simulation Parameters")
    st.json(param_set)
    if solution.get('interpolated', False):
        st.write("**Note**: This is an interpolated result.")
    else:
        st.write("**Note**: This is an exact solution from saved data.")
    
    time_index = st.slider(
        "Select Time Instance (seconds)",
        min_value=0,
        max_value=len(times)-1,
        value=0,
        format="%.2f s"
    )
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Copper @ t={times[time_index]:.2f}s",
            f"Nickel @ t={times[time_index]:.2f}s"
        ),
        horizontal_spacing=0.15
    )
    
    global_max_cu = max(np.max(c) for c in c1_preds) or 1e-6
    global_max_ni = max(np.max(c) for c in c2_preds) or 1e-6
    
    x_coords = X[:,0]
    y_coords = Y[0,:]
    
    fig.add_trace(
        go.Contour(
            z=c1_preds[time_index],
            x=x_coords,
            y=y_coords,
            colorscale='Viridis',
            zmin=0,
            zmax=global_max_cu,
            colorbar=dict(title='Cu (mol/cm³)', x=0.45)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Contour(
            z=c2_preds[time_index],
            x=x_coords,
            y=y_coords,
            colorscale='Cividis',
            zmin=0,
            zmax=global_max_ni,
            colorbar=dict(title='Ni (mol/cm³)', x=1.02)
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=600,
        width=1200,
        margin=dict(l=50, r=50, t=80, b=50),
        xaxis_title="X (μm)",
        yaxis_title="Y (μm)",
        annotations=[
            dict(
                text=f"Spatial Domain: {param_set['Lx']:.1f}μm × {param_set['Ly']:.1f}μm",
                x=0.5,
                y=1.15,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14)
            )
        ],
        hovermode='x unified'
    )
    
    fig.update_yaxes(autorange=True)
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    st.title("Attention Mechanism assisted PINN model for study of size effect on Ni and Cu Cross-Diffusion in Solder ")
    
    # Sidebar for parameter inputs
    with st.sidebar:
        st.header("Simulation Parameters")
        D11 = st.number_input("D11 (Cu self-diffusion)", 
                             min_value=0.001, max_value=1.0, value=0.006, step=0.0001, format="%.6f")
        D12 = st.number_input("D12 (Cu cross-diffusion)", 
                             min_value=0.0, max_value=1.0, value=0.00427, step=0.0001, format="%.6f")
        D21 = st.number_input("D21 (Ni cross-diffusion)", 
                             min_value=0.0, max_value=1.0, value=0.003697, step=0.0001, format="%.6f")
        D22 = st.number_input("D22 (Ni self-diffusion)", 
                             min_value=0.001, max_value=1.0, value=0.0054, step=0.0001, format="%.6f")
        Lx = st.number_input("Domain Width (μm)", 
                            min_value=1.0, max_value=100.0, value=60.0, step=1.0)
        t_max = st.number_input("Simulation Time (s)", 
                               min_value=1.0, max_value=3600.0, value=200.0, step=10.0)
        epochs = st.number_input("Training Epochs", 
                                min_value=100, max_value=10000, value=2000, step=100, format="%d")
        ly_target = st.number_input("Target Ly (μm)", 
                                   min_value=50.0, max_value=100.0, value=60.0, step=0.1)
        sigma = st.slider("Attention Bandwidth (σ)", 1.0, 10.0, 5.0, 0.5,
                     help="Controls neighborhood focus for interpolation")                          
        
        # Store parameters in session state
        st.session_state.D11 = D11
        st.session_state.D12 = D12
        st.session_state.D21 = D21
        st.session_state.D22 = D22
        st.session_state.Lx = Lx
        st.session_state.t_max = t_max
        st.session_state.epochs = epochs
        st.session_state.ly_target = ly_target
    
    output_dir = "pinn_solutions"
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Train and Save Solutions"):
            with st.spinner("Training models..."):
                train_PINN.clear()
                evaluate_model.clear()
                num_saved = train_and_save_solutions(D11, D12, D21, D22, Lx, t_max, epochs, output_dir)
                st.success(f"Saved {num_saved} solutions to {output_dir}/")
    
    with col2:
        if st.button("Visualize Solution"):
            st.session_state.show_visualization = True
    
    if st.session_state.get('show_visualization', False):
        load_and_visualize_solution(D11, D12, D21, D22, Lx, t_max, ly_target, output_dir)
