import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import meshio
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import os
import zipfile
from tempfile import TemporaryDirectory
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pyvista as pv
from stpyvista import stpyvista
import time

matplotlib.use('Agg')


################################################################################
# defintion of the physics (mathematical formulation)
################################################################################

##################################################################################################################################################
# the sigmoid function in the output layer has proved to be consistent
# it (after scaling with factor A) ensures the solution of composition to be in the range [0,A] where A is the boundary concentration of Ni or Cu
###################################################################################################################################################
class SmoothSigmoid(nn.Module):
    def __init__(self, slope=1.0):
        super().__init__()
        self.k = slope  # smaller = smoother
        self.scale = nn.Parameter(torch.tensor(1.0))  # Learnable scale factor

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
        
        # Shared feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh()
        )
        
        # Separate output heads with physics-informed scaling
        self.cu_head = nn.Sequential(
            nn.Linear(64, 1),
            SmoothSigmoid(slope=0.5),
            nn.Linear(1, 1, bias=False),  # Scaling layer
        )
        self.ni_head = nn.Sequential(
            nn.Linear(64, 1),
            SmoothSigmoid(slope=0.5),
            nn.Linear(1, 1, bias=False),
        )
        
        # Initialize scaling weights
        self.cu_head[2].weight.data.fill_(1.6e-3)  # Cu max concentration
        self.ni_head[2].weight.data.fill_(1.25e-3) # Ni max concentration

    def forward(self, x, y, t):
        # Normalize inputs to [0,1] range
        x_norm = x / self.Lx
        y_norm = y / self.Ly
        t_norm = t / self.T_max
        
        features = self.shared_net(torch.cat([x_norm, y_norm, t_norm], dim=1))
        cu = self.cu_head(features)
        ni = self.ni_head(features)
        return torch.cat([cu, ni], dim=1)

def laplacian(c, x, y):
    # Compute first derivatives
    c_x = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c),
                              create_graph=True, retain_graph=True)[0]
    c_y = torch.autograd.grad(c, y, grad_outputs=torch.ones_like(c),
                              create_graph=True, retain_graph=True)[0]
    
    # Compute second derivatives
    c_xx = torch.autograd.grad(c_x, x, grad_outputs=torch.ones_like(c_x),
                               create_graph=True, retain_graph=True)[0]
    c_yy = torch.autograd.grad(c_y, y, grad_outputs=torch.ones_like(c_y),
                               create_graph=True, retain_graph=True)[0]
    
    return c_xx + c_yy

def physics_loss(model, x, y, t):
    c_pred = model(x, y, t)
    c1_pred = c_pred[:, 0:1]
    c2_pred = c_pred[:, 1:2]
    
    # Compute time derivatives
    c1_t = torch.autograd.grad(c1_pred, t, 
                              grad_outputs=torch.ones_like(c1_pred),
                              create_graph=True, retain_graph=True)[0]
    c2_t = torch.autograd.grad(c2_pred, t,
                              grad_outputs=torch.ones_like(c2_pred),
                              create_graph=True, retain_graph=True)[0]
    
    # Compute laplacians
    lap_c1 = laplacian(c1_pred, x, y)  # Remove model.Lx and model.Ly
    lap_c2 = laplacian(c2_pred, x, y)  # Remove model.Lx and model.Ly
    
    # Cross-diffusion PDE residuals (negative sign becomes positive in the residual)
    residual1 = c1_t + (model.D11 * lap_c1 + model.D12 * lap_c2)
    residual2 = c2_t + (model.D21 * lap_c1 + model.D22 * lap_c2)
    
    return torch.mean(residual1**2 + residual2**2)

def boundary_loss_bottom(model):
    # Bottom boundary (y=0) - Dirichlet for both components
    num = 100
    x = torch.rand(num, 1) * model.Lx
    y = torch.zeros(num, 1)
    t = torch.rand(num, 1) * model.T_max
    
    c_pred = model(x, y, t)
    cu_loss = torch.mean((c_pred[:, 0] - 1.6e-3)**2)  # Cu concentration
    ni_loss = torch.mean((c_pred[:, 1] - 0.0)**2)     # Ni concentration
    
    return cu_loss + ni_loss

def boundary_loss_top(model):
    # Top boundary (y=Ly) - Dirichlet for both components
    num = 100
    x = torch.rand(num, 1) * model.Lx
    y = torch.full((num, 1), model.Ly)
    t = torch.rand(num, 1) * model.T_max
    
    c_pred = model(x, y, t)
    cu_loss = torch.mean((c_pred[:, 0] - 0.0)**2)      # Cu concentration
    ni_loss = torch.mean((c_pred[:, 1] - 1.25e-3)**2)  # Ni concentration
    
    return cu_loss + ni_loss

def boundary_loss_sides(model):
    # Neumann BC (zero flux) on left/right boundaries
    num = 100
    
    # Left boundary
    x_left = torch.zeros(num, 1, requires_grad=True)  # Enable gradients
    y_left = torch.rand(num, 1) * model.Ly
    t_left = torch.rand(num, 1) * model.T_max
    c_left = model(x_left, y_left, t_left)
    
    # Right boundary
    x_right = torch.full((num, 1), model.Lx, requires_grad=True)  # Enable gradients
    y_right = torch.rand(num, 1) * model.Ly
    t_right = torch.rand(num, 1) * model.T_max
    c_right = model(x_right, y_right, t_right)
    
    # Compute gradients for both components
    grad_cu_x_left = torch.autograd.grad(c_left[:, 0], x_left,
                                         grad_outputs=torch.ones_like(c_left[:, 0]),
                                         create_graph=True, retain_graph=True)[0]
    grad_ni_x_left = torch.autograd.grad(c_left[:, 1], x_left,
                                         grad_outputs=torch.ones_like(c_left[:, 1]),
                                         create_graph=True, retain_graph=True)[0]
    
    grad_cu_x_right = torch.autograd.grad(c_right[:, 0], x_right,
                                          grad_outputs=torch.ones_like(c_right[:, 0]),
                                          create_graph=True, retain_graph=True)[0]
    grad_ni_x_right = torch.autograd.grad(c_right[:, 1], x_right,
                                          grad_outputs=torch.ones_like(c_right[:, 1]),
                                          create_graph=True, retain_graph=True)[0]
    
    return (torch.mean(grad_cu_x_left**2) + torch.mean(grad_ni_x_left**2) +
            torch.mean(grad_cu_x_right**2) + torch.mean(grad_ni_x_right**2))

def initial_loss(model):
    num = 500
    x = torch.rand(num, 1) * model.Lx
    y = torch.rand(num, 1) * model.Ly
    t = torch.zeros(num, 1)
    
    c_pred = model(x, y, t)
    return torch.mean(c_pred**2)  # Initial condition: c=0 everywhere
    
###############################################################################
# Execution of the physics-iinformed neural network model for cross-diffusion
##################################################################################
def train_PINN(D11, D12, D21, D22, Lx, Ly, T_max, epochs=2000, lr=0.001):
    model = DualScaledPINN(D11, D12, D21, D22, Lx, Ly, T_max)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Collocation points in physical space
    x_pde = torch.rand(1000, 1, requires_grad=True) * Lx
    y_pde = torch.rand(1000, 1, requires_grad=True) * Ly
    t_pde = torch.rand(1000, 1, requires_grad=True) * T_max
    
    loss_history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Compute loss components
        loss_pde = physics_loss(model, x_pde, y_pde, t_pde)
        loss_bottom = boundary_loss_bottom(model)
        loss_top = boundary_loss_top(model)
        loss_sides = boundary_loss_sides(model)
        loss_ic = initial_loss(model)
        
        # Adaptive loss weighting
        total_loss = (10*loss_pde + 
                     100*(loss_bottom + loss_top) + 
                     50*loss_sides + 
                     100*loss_ic)
        
        total_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 100 == 0:
            loss_history.append(total_loss.item())
            st.write(f"Epoch {epoch}: Loss = {total_loss.item():.2e}")
    
    return model, loss_history



def evaluate_model(model, times, Lx, Ly):
    # Generate grid in physical coordinates
    x = torch.linspace(0, Lx, 100)
    y = torch.linspace(0, Ly, 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    c1_predictions = []
    c2_predictions = []
    for t_val in times:
        t = torch.full((X.numel(), 1), t_val)
        # Model outputs already scaled and bounded
        c_pred = model(X.reshape(-1,1), Y.reshape(-1,1), t)
        c1 = c_pred[:, 0].detach().numpy().reshape(100, 100)
        c2 = c_pred[:, 1].detach().numpy().reshape(100, 100)
        c1_predictions.append(c1)
        c2_predictions.append(c2)
    
    # Return physical coordinates directly
    return X.numpy(), Y.numpy(), c1_predictions, c2_predictions

def compute_flux(model, X, Y, t_val, D11, D12, D21, D22, Lx, Ly):
    # Inputs in physical coordinates, model handles normalization
    X_torch = torch.tensor(X, requires_grad=True).reshape(-1, 1)
    Y_torch = torch.tensor(Y, requires_grad=True).reshape(-1, 1)
    t = torch.full((X_torch.numel(), 1), t_val, requires_grad=True)
    
    # Get predictions (already properly scaled)
    c_pred = model(X_torch, Y_torch, t)
    c1_pred = c_pred[:, 0:1]
    c2_pred = c_pred[:, 1:2]
    
    # Compute gradients with respect to physical coordinates
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
    
    # Calculate flux components using physical gradients
    J1_x = -D11 * grad_c1_x.detach().numpy() - D12 * grad_c2_x.detach().numpy()
    J1_y = -D11 * grad_c1_y.detach().numpy() - D12 * grad_c2_y.detach().numpy()
    J2_x = -D21 * grad_c1_x.detach().numpy() - D22 * grad_c2_x.detach().numpy()
    J2_y = -D21 * grad_c1_y.detach().numpy() - D22 * grad_c2_y.detach().numpy()
    
    return (J1_x.reshape(100, 100), J1_y.reshape(100, 100)), \
           (J2_x.reshape(100, 100), J2_y.reshape(100, 100))

#####################################################################################
# The following is the post-procesing part
#####################################################################################
def export_to_vtu(X, Y, c1_list, c2_list, times, Lx, Ly, filename="simulation"):
    """Export simulation data to VTU files (one per time step) with flux vectors."""
    os.makedirs("vtu_output", exist_ok=True)
    
    for i, t in enumerate(times):
        # Compute flux vectors for this time step
        (J1_x, J1_y), (J2_x, J2_y) = compute_flux(model, X, Y, t, D11, D12, D21, D22, Lx, Ly)
        
        # Create points (physical coordinates)
        points = np.column_stack((
            X.flatten(),  # Already in physical coordinates
            Y.flatten(),
            np.zeros_like(X.flatten())  # z=0 (2D data)
        ))
        
        # Create cells (quadrilaterals for structured grid)
        cells = []
        nx, ny = X.shape
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                v0 = ix + iy * nx
                v1 = (ix + 1) + iy * nx
                v2 = (ix + 1) + (iy + 1) * nx
                v3 = ix + (iy + 1) * nx
                cells.append([v0, v1, v2, v3])
        
        # Data fields (point data for better visualization)
        point_data = {
            "Cu_Concentration": c1_list[i].flatten(),
            "Ni_Concentration": c2_list[i].flatten(),
            "Cu_Flux_x": J1_x.flatten(),
            "Cu_Flux_y": J1_y.flatten(),
            "Ni_Flux_x": J2_x.flatten(),
            "Ni_Flux_y": J2_y.flatten(),
            "Cu_Flux_Magnitude": np.sqrt(J1_x**2 + J1_y**2).flatten(),
            "Ni_Flux_Magnitude": np.sqrt(J2_x**2 + J2_y**2).flatten()
        }
        
        # Write to VTU
        mesh = meshio.Mesh(
            points=points,
            cells=[("quad", np.array(cells))],
            point_data=point_data,
        )
        mesh.write(f"vtu_output/{filename}_t{t:.2f}.vtu")
    
    # Zip all VTU files for download
    with zipfile.ZipFile(f"{filename}_vtu.zip", "w") as zipf:
        for file in os.listdir("vtu_output"):
            zipf.write(os.path.join("vtu_output", file), file)
    return f"{filename}_vtu.zip"

########################################################################################
def export_to_exodus(X, Y, c1_list, c2_list, times, Lx, Ly, filename="simulation"):
    points = np.column_stack((
        X.flatten(),
        Y.flatten(),
        np.zeros_like(X.flatten())
    ))
    
    nx, ny = X.shape
    cells = []
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            v0 = ix + iy * nx
            v1 = (ix + 1) + iy * nx
            v2 = (ix + 1) + (iy + 1) * nx
            v3 = ix + (iy + 1) * nx
            cells.append([v0, v1, v2, v3])
    cells = np.array(cells)
    
    time_steps = np.array(times)
    
    flux_data = [
        compute_flux(model, X, Y, t, D11, D12, D21, D22, Lx, Ly)
        for t in times
    ]
    
    point_data = {
        "Cu_Concentration": [c.flatten() for c in c1_list],
        "Ni_Concentration": [c.flatten() for c in c2_list],
        "Cu_Flux_x": [J1_x.flatten() for (J1_x, J1_y), _ in flux_data],
        "Cu_Flux_y": [J1_y.flatten() for (J1_x, J1_y), _ in flux_data],
        "Ni_Flux_x": [J2_x.flatten() for _, (J2_x, J2_y) in flux_data],
        "Ni_Flux_y": [J2_y.flatten() for _, (J2_x, J2_y) in flux_data]
    }
    
    mesh = meshio.Mesh(
        points=points,
        cells=[("quad", cells)],
        point_data=point_data,
    )
    mesh.write(f"{filename}.e", file_format="exodus")
    return f"{filename}.e"

def create_animation(X, Y, c1_list, c2_list, D_matrix, Lx, Ly, times, cmap, temp_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    vmin1, vmax1 = 0, max(np.max(c1) for c1 in c1_list)
    vmin2, vmax2 = 0, max(np.max(c2) for c2 in c2_list)
    
    cont1 = ax1.contourf(X, Y, c1_list[0], levels=50, cmap=cmap, vmin=vmin1, vmax=vmax1)
    cont2 = ax2.contourf(X, Y, c2_list[0], levels=50, cmap=cmap, vmin=vmin2, vmax=vmax2)
    plt.colorbar(cont1, ax=ax1, label='Cu Concentration')
    plt.colorbar(cont2, ax=ax2, label='Ni Concentration')
    
    ax1.set_title(f"Cu Diffusion (D11={D_matrix[0][0]}, D12={D_matrix[0][1]})")
    ax2.set_title(f"Ni Diffusion (D21={D_matrix[1][0]}, D22={D_matrix[1][1]})")
    
    def update(frame):
        ax1.clear()
        ax2.clear()
        cont1 = ax1.contourf(X, Y, c1_list[frame], levels=50, cmap=cmap, vmin=vmin1, vmax=vmax1)
        cont2 = ax2.contourf(X, Y, c2_list[frame], levels=50, cmap=cmap, vmin=vmin2, vmax=vmax2)
        ax1.set_xlabel("x (μm)")
        ax1.set_ylabel("y (μm)")
        ax2.set_xlabel("x (μm)")
        ax1.set_title(f"Cu @ t={times[frame]:.2f}s")
        ax2.set_title(f"Ni @ t={times[frame]:.2f}s")
        return cont1, cont2
    
    ani = animation.FuncAnimation(fig, update, frames=len(times), interval=100)
    plt.close()
    
    ani_path = os.path.join(temp_dir, "animation.gif")
    ani.save(ani_path, writer="pillow", fps=15)
    return ani_path

def create_interactive_plot(X, Y, c1_list, c2_list, times):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Cu Concentration", "Ni Concentration"))
    
    c1_list = [np.maximum(c1, 0) for c1 in c1_list]
    c2_list = [np.maximum(c2, 0) for c2 in c2_list]
    
    fig.add_trace(go.Contour(z=c1_list[0], x=X[:,0], y=Y[0,:], colorscale="Viridis", zmin=0), row=1, col=1)
    fig.add_trace(go.Contour(z=c2_list[0], x=X[:,0], y=Y[0,:], colorscale="Viridis", zmin=0), row=1, col=2)
    
    frames = []
    for i, t in enumerate(times):
        frame = go.Frame(
            data=[
                go.Contour(z=c1_list[i], x=X[:,0], y=Y[0,:], zmin=0),
                go.Contour(z=c2_list[i], x=X[:,0], y=Y[0,:], zmin=0)
            ],
            name=f"frame{i}"
        )
        frames.append(frame)
    
    fig.frames = frames
    
    sliders = [{
        "steps": [
            {"args": [[f"frame{i}"]], "label": f"t={t:.2f}s", "method": "animate"}
            for i, t in enumerate(times)
        ],
        "active": 0,
        "y": 0,
        "x": 0.1,
        "len": 0.9,
    }]
    
    fig.update_layout(
        sliders=sliders,
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]}
            ]
        }],
        width=1000,
        height=500,
        title_text="Cross-Diffusion Dynamics Over Time"
    )
    
    fig.update_xaxes(title_text="x (μm)", row=1, col=1)
    fig.update_xaxes(title_text="x (μm)", row=1, col=2)
    fig.update_yaxes(title_text="y (μm)", row=1, col=1)
    fig.update_yaxes(title_text="y (μm)", row=1, col=2)
    
    return fig


def visualize_with_pyvista(vtu_files):
    """Create an interactive PyVista visualization in Streamlit"""
    if not vtu_files:
        st.warning("No VTU files available for visualization")
        return
    
    # Create plotter
    plotter = pv.Plotter(window_size=[800, 600])
    
    # Load first time step
    mesh = pv.read(vtu_files[0])
    
    # Find concentration range
    c_max = max(mesh["Cu_Concentration"].max(), 1e-6)  # Avoid division by zero
    
    # Add concentration surface
    plotter.add_mesh(
        mesh,
        scalars="Cu_Concentration",
        clim=[0, c_max],
        cmap="viridis",
        show_edges=False,
        name="concentration"
    )
    
    # Add flux vectors if available
    if all(field in mesh.array_names for field in ["Cu_Flux_x", "Cu_Flux_y"]):
        vectors = np.column_stack((mesh["Cu_Flux_x"], mesh["Cu_Flux_y"], np.zeros_like(mesh["Cu_Flux_x"])))
        plotter.add_arrows(
            mesh.points,
            vectors,
            mag=0.1*c_max,
            color="black",
            name="flux"
        )
    
    # Configure plotter
    plotter.view_xy()
    plotter.background_color = "white"
    plotter.add_scalar_bar(title="Cu Concentration", vertical=True)
    
    # Add time slider if multiple files
    if len(vtu_files) > 1:
        plotter.add_slider_widget(
            lambda value: update_pyvista_time_step(plotter, vtu_files, value),
            [0, len(vtu_files)-1],
            value=0,
            title="Time Step",
            style="modern",
            pointa=(0.25, 0.9),
            pointb=(0.75, 0.9)
        )
    
    # Show in Streamlit
    stpyvista(plotter)

def update_pyvista_time_step(plotter, vtu_files, time_step):
    """Update visualization for new time step"""
    time_step = int(time_step)
    mesh = pv.read(vtu_files[time_step])
    
    # Update concentration
    plotter.update_scalars(mesh["Cu_Concentration"], render=False)
    
    # Update flux vectors
    if "flux" in plotter.renderer._actors:
        plotter.remove_actor("flux")
        if all(field in mesh.array_names for field in ["Cu_Flux_x", "Cu_Flux_y"]):
            vectors = np.column_stack((mesh["Cu_Flux_x"], mesh["Cu_Flux_y"], np.zeros_like(mesh["Cu_Flux_x"])))
            plotter.add_arrows(
                mesh.points,
                vectors,
                mag=0.1*mesh["Cu_Concentration"].max(),
                color="black",
                name="flux"
            )
    
    plotter.render()

#  Streamlit app's export section (after generating VTU files):
st.subheader("Interactive 3D Visualization")
#vtu_files = sorted([os.path.join("vtu_output", f) for f in os.listdir("vtu_output") if f.endswith(".vtu")])

if st.checkbox("Show PyVista Visualization", value=False):
    with st.spinner("Loading 3D visualization..."):
        visualize_with_pyvista(vtu_files)
        


# Streamlit interface
st.title("Cross-Diffusion Simulation: Cu-Ni in Sn-2.5Ag Alloy")

with st.sidebar:
    st.header("Diffusion Parameters")
    D11 = st.number_input("D11 (Cu self-diffusion, μm²/s)", 0.001, 1.0, 0.006)
    D12 = st.number_input("D12 (Cu cross-diffusion, μm²/s)", 0.0, 1.0, 0.00427)
    D21 = st.number_input("D21 (Ni cross-diffusion, μm²/s)", 0.0, 1.0, 0.003697)
    D22 = st.number_input("D22 (Ni self-diffusion, μm²/s)", 0.001, 1.0, 0.0054)
    Lx = st.number_input("Domain width (μm)", 1.0, 100.0, 60.0)
    Ly = st.number_input("Domain height (μm)", 1.0, 100.0, 90.0)
    epochs = st.slider("Training epochs", 100, 10000, 2000)
    t_max = st.number_input("Simulation time (s)", 1.0, 3600.0, 200.0)
    num_frames = st.slider("Animation frames", 10, 200, 50)
    cmap = st.selectbox("Color map", plt.colormaps())
    specific_time = st.number_input("Specific time for analysis (s)", 0.0, t_max, t_max/2)


                        
if st.button("Start Simulation"):
    with st.spinner("Training neural network..."):
        model, losses = train_PINN(D11, D12, D21, D22, Lx, Ly, epochs)
        
        # Loss history plot
        fig, ax = plt.subplots()
        ax.semilogy(losses)
        ax.set_title("Training Loss History")
        st.pyplot(fig)
        
        # Evaluate model for all time frames
        times = np.linspace(0, t_max, num_frames)
        X, Y, c1_preds, c2_preds = evaluate_model(model, times, Lx, Ly)
        
        # Animation (GIF)
        with TemporaryDirectory() as temp_dir:
            ani_path = create_animation(X, Y, c1_preds, c2_preds, 
                                      [[D11, D12], [D21, D22]], Lx, Ly, 
                                      times, cmap, temp_dir)
            
            st.header("Cross-Diffusion Dynamics (Animation)")
            st.image(ani_path)
            
            with open(ani_path, "rb") as f:
                st.download_button(
                    "Download Animation",
                    f.read(),
                    "cross_diffusion.gif",
                    "image/gif"
                )
        
        # VTU Generation and Visualization Section
        st.header("3D Visualization Export")
        
        with st.spinner("Generating VTU files for 3D visualization..."):
            try:
                # Ensure directory exists and is empty
                os.makedirs("vtu_output", exist_ok=True)
                
                # Clear existing files safely
                for f in os.listdir("vtu_output"):
                    file_path = os.path.join("vtu_output", f)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        st.error(f"Error deleting {file_path}: {e}")
                
                # Generate VTU files
                vtu_zip_path = export_to_vtu(X, Y, c1_preds, c2_preds, times, Lx, Ly)
                
                # Get list of generated files with error handling
                try:
                    vtu_files = sorted([
                        os.path.join("vtu_output", f) 
                        for f in os.listdir("vtu_output") 
                        if f.endswith('.vtu')
                    ])
                except FileNotFoundError:
                    st.error("VTU output directory not found after generation")
                    vtu_files = []

                if not vtu_files:
                    st.error("Failed to generate VTU files")
                else:
                    st.success(f"Generated {len(vtu_files)} VTU files")
                    
                    # Download option
                    with open(vtu_zip_path, "rb") as f:
                        st.download_button(
                            "Download VTU Files",
                            f.read(),
                            "simulation_vtu.zip",
                            "application/zip"
                        )
                    
                    # Interactive Visualization
                    st.subheader("Interactive 3D Visualization")
                    if st.checkbox("Enable Live 3D Visualization", help="May require significant resources"):
                        try:
                            # Initialize plotter with error handling
                            plotter = pv.Plotter(window_size=[800, 600])
                            
                            # Load first time step with validation
                            if os.path.exists(vtu_files[0]):
                                mesh = pv.read(vtu_files[0])
                            else:
                                raise FileNotFoundError(f"First VTU file not found: {vtu_files[0]}")
                            
                            # Configure visualization
                            plotter.add_mesh(
                                mesh,
                                scalars="Cu_Concentration",
                                cmap="viridis",
                                clim=[0, np.max(c1_preds)],
                                show_edges=False
                            )
                            
                            # Add flux vectors if available
                            if all(f in mesh.array_names for f in ["Cu_Flux_x", "Cu_Flux_y"]):
                                plotter.add_arrows(
                                    mesh.points,
                                    np.column_stack((
                                        mesh["Cu_Flux_x"],
                                        mesh["Cu_Flux_y"],
                                        np.zeros_like(mesh["Cu_Flux_x"])
                                    )),
                                    mag=0.1,
                                    color="red"
                                )
                            
                            # Add time slider if multiple time steps
                            if len(vtu_files) > 1:
                                def update_mesh(time_idx):
                                    try:
                                        new_mesh = pv.read(vtu_files[int(time_idx)])
                                        plotter.update_scalars(new_mesh["Cu_Concentration"])
                                        plotter.add_text(
                                            f"Time: {times[int(time_idx)]:.2f}s", 
                                            position="upper_right",
                                            font_size=12
                                        )
                                    except Exception as e:
                                        st.error(f"Error loading time step {time_idx}: {e}")
                                
                                plotter.add_slider_widget(
                                    update_mesh,
                                    [0, len(vtu_files)-1],
                                    value=0,
                                    title="Time Step",
                                    pointa=(0.25, 0.9),
                                    pointb=(0.75, 0.9)
                                )
                            
                            plotter.view_xy()
                            plotter.background_color = "white"
                            
                            # Display in Streamlit
                            with st.spinner("Rendering 3D visualization..."):
                                stpyvista(plotter)
                                
                        except Exception as e:
                            st.error(f"3D visualization failed: {str(e)}")
                            st.warning("Please download the VTU files and view them in ParaView for full functionality")

            except Exception as main_error:
                st.error(f"Main VTU generation error: {str(main_error)}")

st.markdown("""
**Boundary Conditions:**
- Cu (c1): Fixed concentration (1.5e-3) at bottom boundary (y=0) and (0) at top boundary (y=1)
- Ni (c2): Fixed concentration (0) at bottom boundary (y=0) and (1.2e-3) at top boundary (y=1)
- Zero flux (∂c/∂x = 0) at side boundaries
- Initial condition: c1 = c2 = 0 everywhere at t=0
""")
