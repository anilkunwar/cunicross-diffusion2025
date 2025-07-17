import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
        
        # Normalize boundary concentrations for input
        self.C_Cu_norm = (C_Cu - 1.5e-3) / (2.9e-3 - 1.5e-3)
        self.C_Ni_norm = (C_Ni - 4.0e-4) / (1.8e-3 - 4.0e-4)
        
        self.shared_net = nn.Sequential(
            nn.Linear(5, 128), nn.Tanh(),  # Inputs: x, y, t, C_Cu, C_Ni
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
        
        # Initialize head weights to approximate boundary conditions
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
    num = 100
    x = torch.rand(num, 1) * model.Lx
    y = torch.zeros(num, 1)
    t = torch.rand(num, 1) * model.T_max
    
    c_pred = model(x, y, t)
    return (torch.mean((c_pred[:, 0] - 0.0)**2) + 
            torch.mean((c_pred[:, 1] - model.C_Ni)**2))

def boundary_loss_top(model):
    num = 100
    x = torch.rand(num, 1) * model.Lx
    y = torch.full((num, 1), model.Ly)
    t = torch.rand(num, 1) * model.T_max
    
    c_pred = model(x, y, t)
    return (torch.mean((c_pred[:, 0] - model.C_Cu)**2) + 
            torch.mean((c_pred[:, 1] - 0.0)**2))

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

def plot_losses(loss_history, Ly, C_Cu, C_Ni, output_dir):
    """Generate publishable-quality loss plot."""
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
    
    plot_filename = os.path.join(output_dir, f'loss_plot_ly_{Ly:.1f}_ccu_{C_Cu:.1e}_cni_{C_Ni:.1e}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved loss plot to {plot_filename}")

def train_PINN(D11, D12, D21, D22, Lx, Ly, T_max, C_Cu, C_Ni, epochs=5000, lr=0.001, output_dir="pinn_solutions"):
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
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        phys_loss = physics_loss(model, x_pde, y_pde, t_pde)
        bot_loss = boundary_loss_bottom(model)
        top_loss = boundary_loss_top(model)
        side_loss = boundary_loss_sides(model)
        init_loss = initial_loss(model)
        
        loss = (10 * phys_loss + 100 * bot_loss + 100 * top_loss + 
                50 * side_loss + 100 * init_loss)
        
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        scheduler.step(loss)
        
        if (epoch + 1) % 100 == 0:
            loss_history['epochs'].append(epoch + 1)
            loss_history['total'].append(loss.item())
            loss_history['physics'].append(10 * phys_loss.item())
            loss_history['bottom'].append(100 * bot_loss.item())
            loss_history['top'].append(100 * top_loss.item())
            loss_history['sides'].append(50 * side_loss.item())
            loss_history['initial'].append(100 * init_loss.item())
            
            if (epoch + 1) % 500 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Total Loss: {loss.item():.6f}, "
                      f"Physics: {10 * phys_loss.item():.6f}, "
                      f"Bottom: {100 * bot_loss.item():.6f}, "
                      f"Top: {100 * top_loss.item():.6f}, "
                      f"Sides: {50 * side_loss.item():.6f}, "
                      f"Initial: {100 * init_loss.item():.6f}")
    
    plot_losses(loss_history, Ly, C_Cu, C_Ni, output_dir)
    
    return model, loss_history

def compute_flux(model, X, Y, t_val, D11, D12, D21, D22):
    X_torch = torch.tensor(X, dtype=torch.float32, requires_grad=True).reshape(-1, 1)
    Y_torch = torch.tensor(Y, dtype=torch.float32, requires_grad=True).reshape(-1, 1)
    t = torch.full((X_torch.numel(), 1), t_val, dtype=torch.float32, requires_grad=True)
    
    c_pred = model(X_torch, Y_torch, t)
    c1_pred, c2_pred = c_pred[:, 0:1], c_pred[:, 1:2]
    
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
    return (J1_x.reshape(X.shape), J1_y.reshape(X.shape)), \
           (J2_x.reshape(X.shape), J2_y.reshape(X.shape))

def evaluate_model(model, times, Lx, Ly, D11, D12, D21, D22):
    x = torch.linspace(0, Lx, 50)
    y = torch.linspace(0, Ly, 50)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    c1_preds, c2_preds, J1_preds, J2_preds = [], [], [], []
    for t_val in times:
        t = torch.full((X.numel(), 1), t_val)
        c_pred = model(X.reshape(-1,1), Y.reshape(-1,1), t)
        c1 = c_pred[:,0].detach().numpy().reshape(50,50).T
        c2 = c_pred[:,1].detach().numpy().reshape(50,50).T
        c1_preds.append(c1)
        c2_preds.append(c2)
        
        (J1_x, J1_y), (J2_x, J2_y) = compute_flux(model, X.numpy(), Y.numpy(), t_val, D11, D12, D21, D22)
        J1_preds.append((J1_x, J1_y))
        J2_preds.append((J2_x, J2_y))
    
    return X.numpy(), Y.numpy(), c1_preds, c2_preds, J1_preds, J2_preds

def generate_parameter_sets(D11, D12, D21, D22, Lx, t_max, epochs):
    Ly_range = [30.0, 120.0]
    C_Cu_range = [1.5e-3, 1.9e-3, 2.5e-3, 2.9e-3]
    C_Ni_range = [4.0e-4, 8.0e-4, 1.2e-3, 1.8e-3]
    
    params = []
    for Ly in Ly_range:
        for C_Cu in C_Cu_range:
            for C_Ni in C_Ni_range:
                param_set = {
                    'D11': D11,
                    'D12': D12,
                    'D21': D21,
                    'D22': D22,
                    'Lx': Lx,
                    'Ly': float(Ly),
                    't_max': t_max,
                    'C_Cu': float(C_Cu),
                    'C_Ni': float(C_Ni),
                    'epochs': epochs
                }
                params.append(param_set)
    return params

def train_and_save_solutions(D11, D12, D21, D22, Lx, t_max, epochs, output_dir="pinn_solutions"):
    os.makedirs(output_dir, exist_ok=True)
    params = generate_parameter_sets(D11, D12, D21, D22, Lx, t_max, epochs)
    times = np.linspace(0, t_max, 50)
    
    for idx, param_set in enumerate(params):
        print(f"Training model {idx + 1}/{len(params)} for Ly={param_set['Ly']:.1f} μm, "
              f"C_Cu={param_set['C_Cu']:.1e}, C_Ni={param_set['C_Ni']:.1e}...")
        model, loss_history = train_PINN(
            param_set['D11'], param_set['D12'],
            param_set['D21'], param_set['D22'],
            param_set['Lx'], param_set['Ly'],
            param_set['t_max'],
            param_set['C_Cu'], param_set['C_Ni'],
            epochs=param_set['epochs'],
            output_dir=output_dir
        )
        
        X, Y, c1_preds, c2_preds, J1_preds, J2_preds = evaluate_model(
            model, times, param_set['Lx'], param_set['Ly'],
            param_set['D11'], param_set['D12'], param_set['D21'], param_set['D22']
        )
        
        solution = {
            'params': param_set,
            'X': X,
            'Y': Y,
            'c1_preds': c1_preds,
            'c2_preds': c2_preds,
            'J1_preds': J1_preds,
            'J2_preds': J2_preds,
            'times': times,
            'loss_history': loss_history,
            'orientation_note': 'c1_preds and c2_preds are arrays of shape (50,50) where rows (i) correspond to y-coordinates and columns (j) correspond to x-coordinates due to transpose.'
        }
        solution_filename = os.path.join(output_dir, 
            f"solution_ly_{param_set['Ly']:.1f}_ccu_{param_set['C_Cu']:.1e}_cni_{param_set['C_Ni']:.1e}_d11_{D11:.6f}_d12_{D12:.6f}_d21_{D21:.6f}_d22_{D22:.6f}_lx_{Lx:.1f}_tmax_{t_max:.1f}.pkl")
        with open(solution_filename, 'wb') as f:
            pickle.dump(solution, f)
        
        print(f"Saved solution {idx + 1} to {solution_filename}")
    
    return len(params)

if __name__ == "__main__":
    D11 = 0.006
    D12 = 0.00427
    D21 = 0.003697
    D22 = 0.0054
    Lx = 60.0
    t_max = 200.0
    epochs = 5000
    
    num_saved = train_and_save_solutions(D11, D12, D21, D22, Lx, t_max, epochs)
    print(f"Saved {num_saved} solutions to pinn_solutions/")
