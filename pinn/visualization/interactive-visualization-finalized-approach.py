import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

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
def train_PINN(D11, D12, D21, D22, Lx=100, Ly=100, T_max=10, epochs=2000, lr=0.001):
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
def evaluate_model(_model, times, Lx=100, Ly=100):
    x = torch.linspace(0, Lx, 100)
    y = torch.linspace(0, Ly, 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    c1_preds, c2_preds = [], []
    boundary_checks = {
        'c1_bottom': [], 'c1_top': [],
        'c2_bottom': [], 'c2_top': []
    }
    
    for t_val in times:
        t = torch.full((X.numel(), 1), t_val)
        c_pred = _model(X.reshape(-1,1), Y.reshape(-1,1), t)
        # Transpose to align with Plotly's (y, x) convention
        c1 = c_pred[:,0].detach().numpy().reshape(100,100).T
        c2 = c_pred[:,1].detach().numpy().reshape(100,100).T
        c1_preds.append(c1)
        c2_preds.append(c2)
        
        # Boundary checks for first time step
        if t_val == times[0]:
            boundary_checks['c1_bottom'] = c1[:,0]   # y=0
            boundary_checks['c1_top'] = c1[:,-1]     # y=Ly
            boundary_checks['c2_bottom'] = c2[:,0]   # y=0
            boundary_checks['c2_top'] = c2[:,-1]     # y=Ly
    
    return X.numpy(), Y.numpy(), c1_preds, c2_preds, boundary_checks

def create_plotly_plot(X, Y, c1_list, c2_list, times, Lx, Ly):
    """Interactive visualization with correct y-axis alignment"""
    st.markdown("## Concentration Dynamics Visualization")
    
    # Coordinates (1D arrays)
    x_coords = X[:,0]  # Shape (100,)
    y_coords = Y[0,:]  # Shape (100,)
    
    # Global maxima for consistent color scaling
    global_max_cu = max(np.max(c) for c in c1_list) or 1e-6
    global_max_ni = max(np.max(c) for c in c2_list) or 1e-6
    
    # Time slider
    time_index = st.slider(
        "Select Time Instance (seconds)",
        min_value=0,
        max_value=len(times)-1,
        value=0,
        format="%.2f s"
    )
    
    # Create figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Copper @ t={times[time_index]:.2f}s", 
            f"Nickel @ t={times[time_index]:.2f}s"
        ),
        horizontal_spacing=0.15
    )

    # Copper contour
    fig.add_trace(
        go.Contour(
            z=c1_list[time_index],
            x=x_coords,
            y=y_coords,
            colorscale='Viridis',
            zmin=0,
            zmax=global_max_cu,
            colorbar=dict(title='Cu (mol/cm³)', x=0.45)
        ),
        row=1, col=1
    )

    # Nickel contour
    fig.add_trace(
        go.Contour(
            z=c2_list[time_index],
            x=x_coords,
            y=y_coords,
            colorscale='Cividis',
            zmin=0,
            zmax=global_max_ni,
            colorbar=dict(title='Ni (mol/cm³)', x=1.02)
        ),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        height=600,
        width=1200,
        margin=dict(l=50, r=50, t=80, b=50),
        xaxis_title="X (μm)",
        yaxis_title="Y (μm)",
        annotations=[
            dict(
                text=f"Spatial Domain: {Lx:.0f}μm × {Ly:.0f}μm",
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
    
    # Ensure y-axis has y=0 at bottom, y=Ly at top
    fig.update_yaxes(autorange=True)
    
    st.plotly_chart(fig, use_container_width=True)

# Main application
if __name__ == "__main__":
    st.title("Cross-Diffusion PINN Visualization")
    
    # User inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        Lx = st.number_input("Domain Length (X) [μm]", 
                           min_value=10.0, max_value=500.0, 
                           value=100.0, step=10.0)
    with col2:
        Ly = st.number_input("Domain Length (Y) [μm]",
                           min_value=10.0, max_value=500.0,
                           value=100.0, step=10.0)
    with col3:
        T_max = st.number_input("Simulation Time [s]",
                              min_value=1.0, max_value=1000.0,
                              value=10.0, step=10.0)

    # Fixed diffusion coefficients
    D11, D12 = 1e-4, 5e-5
    D21, D22 = 5e-5, 1e-4
    times = np.linspace(0, T_max, 20)

    # Model training
    if st.button("Train Model"):
        # Clear cache to ensure fresh evaluation
        evaluate_model.clear()
        with st.spinner(f"Training PINN for {Lx:.0f}μm × {Ly:.0f}μm domain..."):
            model = train_PINN(D11, D12, D21, D22, Lx, Ly, T_max)
            st.session_state.model = model
            
    # Display results
    if 'model' in st.session_state:
        # Unpack all return values from evaluate_model
        X, Y, c1_preds, c2_preds, boundary_checks = evaluate_model(
            st.session_state.model, 
            times,
            Lx=Lx,
            Ly=Ly
        )
        
        # Display boundary checks on Streamlit screen
        st.subheader("Boundary Condition Checks (at t=0)")
        st.write("Copper (c1) at bottom (y=0): Mean = {:.2e}, Expected ≈ 1.6e-3".format(
            np.mean(boundary_checks['c1_bottom'])))
        st.write("Copper (c1) at top (y=Ly): Mean = {:.2e}, Expected ≈ 0.0".format(
            np.mean(boundary_checks['c1_top'])))
        st.write("Nickel (c2) at bottom (y=0): Mean = {:.2e}, Expected ≈ 0.0".format(
            np.mean(boundary_checks['c2_bottom'])))
        st.write("Nickel (c2) at top (y=Ly): Mean = {:.2e}, Expected ≈ 1.25e-3".format(
            np.mean(boundary_checks['c2_top'])))
        
        create_plotly_plot(X, Y, c1_preds, c2_preds, times, Lx, Ly)
