import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os

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

def train_PINN(D11, D12, D21, D22, Lx, Ly, T_max, epochs=2000, lr=0.001):
    model = DualScaledPINN(D11, D12, D21, D22, Lx, Ly, T_max)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    x_pde = torch.rand(1000, 1, requires_grad=True) * Lx
    y_pde = torch.rand(1000, 1, requires_grad=True) * Ly
    t_pde = torch.rand(1000, 1, requires_grad=True) * T_max
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = 10 * physics_loss(model, x_pde, y_pde, t_pde)
        loss += 100 * (boundary_loss_bottom(model) + boundary_loss_top(model))
        loss += 50 * boundary_loss_sides(model) + 100 * initial_loss(model)
        
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")
    
    return model

def evaluate_model(model, times, Lx, Ly):
    x = torch.linspace(0, Lx, 50)
    y = torch.linspace(0, Ly, 50)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    c1_preds, c2_preds = [], []
    for t_val in times:
        t = torch.full((X.numel(), 1), t_val)
        c_pred = model(X.reshape(-1,1), Y.reshape(-1,1), t)
        c1 = c_pred[:,0].detach().numpy().reshape(50,50).T
        c2 = c_pred[:,1].detach().numpy().reshape(50,50).T
        c1_preds.append(c1)
        c2_preds.append(c2)
    
    return X.numpy(), Y.numpy(), c1_preds, c2_preds

def generate_parameter_sets(D11, D12, D21, D22, Lx, t_max, epochs):
    Ly_range = np.arange(50, 101, 5)  # 50, 55, ..., 100
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
    times = np.linspace(0, t_max, 50)
    
    for idx, param_set in enumerate(params):
        print(f"Training model {idx + 1}/{len(params)} for Ly={param_set['Ly']:.1f} μm...")
        model = train_PINN(
            param_set['D11'], param_set['D12'],
            param_set['D21'], param_set['D22'],
            param_set['Lx'], param_set['Ly'],
            param_set['t_max'],
            epochs=param_set['epochs']
        )
        
        # Evaluate model and generate solutions
        X, Y, c1_preds, c2_preds = evaluate_model(model, times, param_set['Lx'], param_set['Ly'])
        
        # Save the solution
        solution = {
            'params': param_set,
            'X': X,
            'Y': Y,
            'c1_preds': c1_preds,
            'c2_preds': c2_preds,
            'times': times
        }
        solution_filename = os.path.join(output_dir, 
            f"solution_ly_{param_set['Ly']:.1f}_d11_{D11:.6f}_d12_{D12:.6f}_d21_{D21:.6f}_d22_{D22:.6f}_lx_{Lx:.1f}_tmax_{t_max:.1f}.pkl")
        with open(solution_filename, 'wb') as f:
            pickle.dump(solution, f)
        
        print(f"Saved solution {idx + 1} to {solution_filename}")
    
    return len(params)

if __name__ == "__main__":
    # Example parameters
    D11 = 0.006
    D12 = 0.00427
    D21 = 0.003697
    D22 = 0.0054
    Lx = 60.0
    t_max = 200.0
    epochs = 2000
    
    num_saved = train_and_save_solutions(D11, D12, D21, D22, Lx, t_max, epochs)
    print(f"Saved {num_saved} solutions to pinn_solutions/")
