import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import zipfile
from tempfile import TemporaryDirectory

matplotlib.use('Agg')

############################################################
# Using sigmoid to ensure that the output solution for c >=0 
# Using linear layer can cause the solution to attain a negative value
############################################################
class SmoothSigmoid(nn.Module):
    def __init__(self, slope=1.0):
        super().__init__()
        self.k = slope  # smaller = smoother

    def forward(self, x):
        return 1 / (1 + torch.exp(-self.k * x))


class PINN(nn.Module):
    def __init__(self, D, Lx, Ly, T_max):
        super(PINN, self).__init__()
        self.D = D
        self.Lx = Lx
        self.Ly = Ly
        self.T_max = T_max
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
            #nn.Sigmoid()  # maps to (0,1)
            SmoothSigmoid(slope=0.5) 
        )

    def forward(self, x, y, t):
        # Scale inputs to [0,1] range
        x_scaled = x / self.Lx
        y_scaled = y / self.Ly
        t_scaled = t / self.T_max
        #return torch.nn.functional.softplus(self.net(torch.cat([x_scaled, y_scaled, t_scaled], dim=1))) # alters the physics and so it is incorrect
        #return self.net(torch.cat([x_scaled, y_scaled, t_scaled], dim=1))
        #log_c = self.net(torch.cat([x_scaled, y_scaled, t_scaled], dim=1))
        #return torch.exp(log_c)  # Exponentiate to ensure positivity
        return 1.59e-3 *self.net(torch.cat([x_scaled, y_scaled, t_scaled], dim=1))

def laplacian(c, x, y):
    c_x = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c), create_graph=True, retain_graph=True)[0]
    c_y = torch.autograd.grad(c, y, grad_outputs=torch.ones_like(c), create_graph=True, retain_graph=True)[0]
    
    c_xx = torch.autograd.grad(c_x, x, grad_outputs=torch.ones_like(c_x), create_graph=True, retain_graph=True)[0]
    c_yy = torch.autograd.grad(c_y, y, grad_outputs=torch.ones_like(c_y), create_graph=True, retain_graph=True)[0]
    
    return c_xx + c_yy


def physics_loss(model, x, y, t):
    c_pred = model(x, y, t)
    c_t = torch.autograd.grad(c_pred, t, grad_outputs=torch.ones_like(c_pred), create_graph=True, retain_graph=True)[0]
    c_lap = laplacian(c_pred, x, y)
    #return torch.mean((c_t - model.D * c_lap) ** 2)
    return torch.mean((c_t + model.D * c_lap) ** 2)

def boundary_loss_bottom(model):
    # Dirichlet BC at bottom (y=0)
    num = 100
    x = torch.rand(num, 1) * model.Lx
    y = torch.zeros(num, 1)
    t = torch.rand(num, 1) * model.T_max
    c_pred = model(x, y, t)
    #c_pred = torch.clamp(c_pred, min=0)  # Enforce non-negativity but not needed at the bottom
    return torch.mean((c_pred - 1.59e-3)**2)

def boundary_loss_top(model):
    # Dirichlet BC at top (y=Ly) or Neumann BC
    num = 100
    x = torch.rand(num, 1) * model.Lx
    y = torch.full((num, 1), model.Ly) # for Dirichlet no grad required
    #y = torch.full((num, 1), model.Ly, requires_grad=True)  # Enable gradients for Neumann BC
    t = torch.rand(num, 1) * model.T_max
    c_pred = model(x, y, t)
    #c_pred = torch.clamp(c_pred, min=0)  # Enforce non-negativity
    return torch.mean(c_pred**2)  #Dirichlet
    #return torch.mean((c_pred - 1.00e-8)**2)  # a small number to prevent the solutions from going negative
    #grad_c_pred = torch.autograd.grad(c_pred, y, torch.ones_like(c_pred), create_graph=True, retain_graph=True)[0]
    #return torch.mean(grad_c_pred**2) #Neumann

def boundary_loss_sides(model):
    num = 100
    
    # Left boundary
    x_left = torch.zeros(num, 1, requires_grad=True)
    y_left = torch.rand(num, 1) * model.Ly
    t_left = torch.rand(num, 1) * model.T_max
    c_left = model(x_left, y_left, t_left)
    grad_cx_left = torch.autograd.grad(c_left, x_left, torch.ones_like(c_left), create_graph=True, retain_graph=True)[0]
    
    # Right boundary
    x_right = torch.full((num, 1), model.Lx, requires_grad=True)
    y_right = torch.rand(num, 1) * model.Ly
    t_right = torch.rand(num, 1) * model.T_max
    c_right = model(x_right, y_right, t_right)
    grad_cx_right = torch.autograd.grad(c_right, x_right, torch.ones_like(c_right), create_graph=True, retain_graph=True)[0]
    
    return torch.mean(grad_cx_left**2) + torch.mean(grad_cx_right**2)

def initial_loss(model):
    num = 500
    x = torch.rand(num, 1) * model.Lx
    y = torch.rand(num, 1) * model.Ly
    t = torch.zeros(num, 1)
    c_pred = model(x, y, t)
    #c_pred = torch.clamp(c_pred, min=0)  # Enforce non-negativity. not advisable to use in the model as it challenges the physical consistency
    #return torch.mean(c_pred**2)
    return torch.mean((c_pred - 0.0)**2)  # a small number to prevent the solutions from going negative

def non_negativity_loss(model, x, y, t):
    c_pred = model(x, y, t)
    return torch.mean(torch.clamp(-c_pred, min=0)**2)  # Penalize negative values to ensure the physics
    
def train_PINN(D, Lx, Ly, T_max, epochs=2000, lr=0.001):
    model = PINN(D, Lx, Ly, T_max)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Collocation points in physical coordinates
    x_pde = torch.rand(1000, 1, requires_grad=True) * Lx
    y_pde = torch.rand(1000, 1, requires_grad=True) * Ly
    t_pde = torch.rand(1000, 1, requires_grad=True) * T_max
    
    loss_history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        loss_pde = physics_loss(model, x_pde, y_pde, t_pde)
        loss_bottom = boundary_loss_bottom(model)
        loss_top = boundary_loss_top(model)
        loss_sides = boundary_loss_sides(model)
        loss_ic = initial_loss(model)
        
        # for non-negativity of composition 
        #loss_non_neg = non_negativity_loss(model, x_pde, y_pde, t_pde) # somehow alters the physics and so not recommended
        #loss = loss_pde + 10*(loss_bottom + loss_top) + 5*loss_sides + 10*loss_ic + 10*loss_non_neg
        #loss = loss_pde + 10*(loss_bottom + loss_top) + 5*loss_sides + 10*loss_ic
        loss = 10*loss_pde + 5*loss_bottom + loss_top + loss_sides + loss_ic
        #loss.backward()
        loss.backward(retain_graph=True)  # Retain graph for further backward passes
        optimizer.step()
        
        if epoch % 100 == 0:
            loss_history.append(loss.item())
    
    return model, loss_history

def evaluate_model(model, times, Lx, Ly):
    x = torch.linspace(0, Lx, 100)
    y = torch.linspace(0, Ly, 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    predictions = []
    for t_val in times:
        t = torch.full((X.numel(), 1), t_val)
        c_pred = model(X.reshape(-1,1), Y.reshape(-1,1), t)
        predictions.append(c_pred.detach().numpy().reshape(100, 100))
    
    return X.numpy(), Y.numpy(), predictions

def create_animation(X, Y, C_list, D, Lx, Ly, times, cmap, temp_dir):
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(X, Y, C_list[0], levels=50, cmap=cmap)
    plt.colorbar(contour, label='Concentration')
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")
    ax.set_title(f"Diffusion in {Lx}x{Ly} μm domain, D={D} μm²/s\nTime={times[0]:.2f}s")
    
    def update(frame):
        ax.clear()
        contour = ax.contourf(X, Y, C_list[frame], levels=50, cmap=cmap)
        ax.set_title(f"Diffusion in {Lx}x{Ly} μm domain, D={D} μm²/s\nTime={times[frame]:.2f}s")
        return contour
    
    ani = animation.FuncAnimation(fig, update, frames=len(times), interval=100)
    plt.close()
    
    ani_path = os.path.join(temp_dir, "animation.gif")
    ani.save(ani_path, writer="pillow", fps=5)
    #ani.save(ani_path, writer="pillow", fps=15)
    return ani_path

# Streamlit interface
st.title("Modeling 2D Diffusion Process ( Cu in Sn-2.5Ag Liquid) with Physics-Informed Neural Networks")

with st.sidebar:
    st.header("Simulation Parameters")
    Lx = st.number_input("Domain width (Lx, μm)", 1.0, 100.0, 60.0)
    Ly = st.number_input("Domain height (Ly, μm)", 1.0, 100.0, 90.0)
    D = st.number_input("Diffusion coefficient (D, μm²/s)", 0.001, 1.0, 0.006)
    t_max = st.number_input("Simulation time (seconds)", 1.0, 3600.0, 500.0)
    epochs = st.slider("Training epochs", 100, 10000, 2000)
    num_frames = st.slider("Animation frames", 10, 500, 500)
    cmap = st.selectbox("Color map", plt.colormaps())

if st.button("Run Simulation"):
    with st.spinner("Training neural network..."):
        model, losses = train_PINN(D, Lx, Ly, t_max, epochs)
        
        fig, ax = plt.subplots()
        ax.semilogy(losses)
        ax.set_title("Training Loss History")
        ax.set_xlabel("Epochs (x100)")
        ax.set_ylabel("Loss (log scale)")
        st.pyplot(fig)
        
        times = np.linspace(0, t_max, num_frames)
        X, Y, predictions = evaluate_model(model, times, Lx, Ly)
        
        with TemporaryDirectory() as temp_dir:
            ani_path = create_animation(X, Y, predictions, D, Lx, Ly, times, cmap, temp_dir)
            
            st.header("Diffusion Process Animation")
            st.image(ani_path)
            
            with open(ani_path, "rb") as f:
                st.download_button(
                    "Download Animation",
                    f.read(),
                    "diffusion_simulation.gif",
                    "image/gif"
                )

st.markdown("""
**Boundary Conditions:**
- Bottom boundary (y=0): Fixed concentration c=1.5e-3
- Top boundary (y=Ly): c=0
- Side boundaries (x=0/Lx): Zero flux (∂c/∂x=0)
- Initial condition: c=0 everywhere at t=0
""")
