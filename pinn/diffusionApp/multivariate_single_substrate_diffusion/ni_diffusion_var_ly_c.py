# batch_ni_diffusion_full.py
# Full-featured batch trainer for pure Ni self-diffusion (from bottom)
# Identical structure to the Cu version — nothing missing, everything preserved

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
from datetime import datetime
import re

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
OUTPUT_DIR = "pinn_solutions_ni_batch"
os.makedirs(OUTPUT_DIR, exist_ok=True)

mpl.rcParams.update({
    'font.family': 'Arial', 'font.size': 12,
    'axes.linewidth': 1.5, 'xtick.major.width': 1.5, 'ytick.major.width': 1.5,
    'figure.dpi': 300
})

logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(OUTPUT_DIR, 'batch_ni_training.log'),
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Fixed physics parameters (pure Ni self-diffusion)
C_CU_TOP = C_CU_BOTTOM = 0.0
C_NI_TOP = 0.0
Lx = 60.0
D11 = 0.006
D12 = D21 = 0.0          # No cross terms
D22 = 0.0054             # Ni self-diffusion coefficient
T_max = 200.0
epochs = 5000
lr = 1e-3
times_eval = np.linspace(0, T_max, 50)

torch.manual_seed(42)
np.random_seed(42)

# ----------------------------------------------------------------------
# Model & Helpers
# ----------------------------------------------------------------------
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
        self.D11, self.D12, self.D21, self.D22 = D11, D12, D21, D22
        self.Lx, self.Ly, self.T_max = Lx, Ly, T_max
        self.C_Cu, self.C_Ni = C_Cu, C_Ni

        self.C_Cu_norm = (C_Cu - 1.5e-3) / (2.9e-3 - 1.5e-3)
        self.C_Ni_norm = (C_Ni - 4.0e-4) / (1.8e-3 - 4.0e-4)

        self.shared_net = nn.Sequential(
            nn.Linear(5, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh()
        )
        self.cu_head = nn.Sequential(nn.Linear(128, 1), SmoothSigmoid(0.5), nn.Linear(1, 1, bias=False))
        self.ni_head = nn.Sequential(nn.Linear(128, 1), SmoothSigmoid(0.5), nn.Linear(1, 1, bias=False))

        self.cu_head[2].weight.data.fill_(C_Cu)
        self.ni_head[2].weight.data.fill_(C_Ni)

    def forward(self, x, y, t):
        x_norm = x / self.Lx
        y_norm = y / self.Ly
        t_norm = t / self.T_max
        C_Cu_in = torch.full_like(x, self.C_Cu_norm)
        C_Ni_in = torch.full_like(x, self.C_Ni_norm)
        inp = torch.cat([x_norm, y_norm, t_norm, C_Cu_in, C_Ni_in], dim=1)
        h = self.shared_net(inp)
        return torch.cat([self.cu_head(h), self.ni_head(h)], dim=1)

def laplacian(c, x, y):
    c_x = torch.autograd.grad(c, x, torch.ones_like(c), create_graph=True, retain_graph=True)[0]
    c_y = torch.autograd.grad(c, y, torch.ones_like(c), create_graph=True, retain_graph=True)[0]
    c_xx = torch.autograd.grad(c_x, x, torch.ones_like(c_x), create_graph=True, retain_graph=True)[0]
    c_yy = torch.autograd.grad(c_y, y, torch.ones_like(c_y), create_graph=True, retain_graph=True)[0]
    return c_xx + c_yy

# ----------------------------------------------------------------------
# Loss functions (pure Ni diffusion only)
# ----------------------------------------------------------------------
def physics_loss(model, x, y, t):
    c_ni = model(x, y, t)[:, 1:2]
    c_t = torch.autograd.grad(c_ni, t, torch.ones_like(c_ni), create_graph=True, retain_graph=True)[0]
    lap = laplacian(c_ni, x, y)
    return torch.mean((c_t - model.D22 * lap)**2)

def boundary_loss_bottom(model, C_Ni_bottom):
    n = 200
    x = torch.rand(n, 1, requires_grad=True) * model.Lx
    y = torch.zeros(n, 1, requires_grad=True)
    t = torch.rand(n, 1, requires_grad=True) * model.T_max
    c_ni = model(x, y, t)[:, 1]
    return torch.mean((c_ni - C_Ni_bottom)**2)

def boundary_loss_top(model):
    n = 200
    x = torch.rand(n, 1, requires_grad=True) * model.Lx
    y = torch.full((n, 1), model.Ly, requires_grad=True)
    t = torch.rand(n, 1, requires_grad=True) * model.T_max
    c_ni = model(x, y, t)[:, 1]
    return torch.mean(c_ni**2)

def boundary_loss_sides(model):
    n = 200
    # Left
    x_l = torch.zeros(n, 1, requires_grad=True)
    y_l = torch.rand(n, 1, requires_grad=True) * model.Ly
    t_l = torch.rand(n, 1, requires_grad=True) * model.T_max
    c_l = model(x_l, y_l, t_l)[:, 1]
    g_l = torch.autograd.grad(c_l, x_l, torch.ones_like(c_l), create_graph=True)[0]
    # Right
    x_r = torch.full((n, 1), model.Lx, requires_grad=True)
    y_r = torch.rand(n, 1, requires_grad=True) * model.Ly
    t_r = torch.rand(n, 1, requires_grad=True) * model.T_max
    c_r = model(x_r, y_r, t_r)[:, 1]
    g_r = torch.autograd.grad(c_r, x_r, torch.ones_like(c_r), create_graph=True)[0]
    return torch.mean(g_l**2) + torch.mean(g_r**2)

def initial_loss(model):
    n = 500
    x = torch.rand(n, 1, requires_grad=True) * model.Lx
    y = torch.rand(n, 1, requires_grad=True) * model.Ly
    t = torch.zeros(n, 1, requires_grad=True)
    c_ni = model(x, y, t)[:, 1]
    return torch.mean(c_ni**2)

# ----------------------------------------------------------------------
# Training (cached per case)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_single_case(Ly_val: float, C_Ni_bottom_val: float, _hash):
    model = DualScaledPINN(D11, D12, D21, D22, Lx, Ly_val, T_max, 0.0, C_Ni_bottom_val)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    x_pde = torch.rand(1000, 1, requires_grad=True) * Lx
    y_pde = torch.rand(1000, 1, requires_grad=True) * Ly_val
    t_pde = torch.rand(1000, 1, requires_grad=True) * T_max

    progress = st.progress(0)
    status = st.empty()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss = (10 * physics_loss(model, x_pde, y_pde, t_pde) +
                100 * boundary_loss_bottom(model, C_Ni_bottom_val) +
                100 * boundary_loss_top(model) +
                100 * boundary_loss_sides(model) +
                100 * initial_loss(model))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % 200 == 0 or epoch == epochs:
            progress.progress(epoch / epochs)
            status.text(f"Ly={Ly_val:.1f} μm | C_Ni={C_Ni_bottom_val:.2e} → Epoch {epoch}, Loss {loss.item():.2e}")

    progress.progress(1.0)
    status.text(f"Complete: Ly={Ly_val:.1f} μm, C_Ni={C_Ni_bottom_val:.2e}")
    return model

# ----------------------------------------------------------------------
# Evaluation & Save
# ----------------------------------------------------------------------
def evaluate_and_save(model, Ly_val, C_Ni_bottom_val):
    x = torch.linspace(0, Lx, 50)
    y = torch.linspace(0, Ly_val, 50)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    c2_preds = []
    for t_val in times_eval:
        t = torch.full((X.numel(), 1), t_val)
        c_ni = model(X.reshape(-1,1), Y.reshape(-1,1), t)[:, 1].detach().cpu().numpy()
        c2_preds.append(c_ni.reshape(50, 50).T)

    # Unique clean filename
    cni_str = f"{C_Ni_bottom_val:.1e}".replace('.', '').replace('+', '').replace('-', 'm')
    filename = f"solutions_Ni_diffusion_ly_{Ly_val:.1f}_cni_{cni_str}.pkl"
    filepath = os.path.join(OUTPUT_DIR, filename)

    solution = {
        'params': {'Lx': Lx, 'Ly': Ly_val, 'C_Ni_bottom': C_Ni_bottom_val, 'D22': D22, 't_max': T_max},
        'X': X.cpu().numpy(),
        'Y': Y.cpu().numpy(),
        'times': times_eval.tolist(),
        'c1_preds': [np.zeros_like(c) for c in c2_preds],  # Cu = 0
        'c2_preds': c2_preds
    }

    with open(filepath, 'wb') as f:
        pickle.dump(solution, f)
    logger.info(f"Saved: {filename}")
    return filepath, solution

# ----------------------------------------------------------------------
# Main App
# ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Batch Ni Diffusion", layout="wide")
    st.title("Batch Pure Ni Self-Diffusion PINN Trainer")
    st.markdown("""
    **Pure Ni diffusion from bottom** → variable solder height `Ly` and bottom Ni concentration `C_Ni_bottom`.  
    One trained model per parameter pair.  
    Output: `solutions_Ni_diffusion_ly_X.X_cni_Ye-Z.pkl`
    """)

    ly_input = st.text_area("Ly values (μm) — one per line or comma-separated", "30\n50\n70\n90\n110")
    cni_input = st.text_area("C_Ni_bottom values (mol/cc) — same count", "4e-4\n8e-4\n1.2e-3\n1.6e-3\n1.8e-3")

    if st.button("Start Batch Training", type="primaryist"):
        try:
            ly_list = [float(x.strip()) for x in re.split(r'[,\n]+', ly_input) if x.strip()]
            cni_list = [float(x.strip()) for x in re.split(r'[,\n]+', cni_input) if x.strip()]

            if len(ly_list) != len(cni_list):
                st.error("Number of Ly and C_Ni_bottom values must be equal!")
                return

            results = []
            prog = st.progress(0)
            status = st.empty()

            for i, (Ly_val, C_Ni_val) in enumerate(zip(ly_list, cni_list)):
                status.write(f"Training {i+1}/{len(ly_list)} → Ly = {Ly_val:.1f} μm, C_Ni = {C_Ni_val:.2e}")
                hash_key = hashlib.md5(f"{Ly_val}_{C_Ni_val}".encode()).hexdigest()
                model = train_single_case(Ly_val, C_Ni_val, hash_key)
                pkl_path, sol = evaluate_and_save(model, Ly_val, C_Ni_val)
                results.append(pkl_path)
                prog.progress((i + 1) / len(ly_list))

            st.success("All Ni diffusion cases completed!")

            # ZIP all
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for path in results:
                    if os.path.exists(path):
                        zf.write(path, os.path.basename(path))
            buffer.seek(0)

            st.download_button(
                "Download All Ni Solutions (.zip)",
                data=buffer,
                file_name=f"ni_batch_{datetime.now():%Y%m%d_%H%M%S}.zip",
                mime="application/zip"
            )

            st.write("### Individual Downloads")
            for path in results:
                with open(path, "rb") as f:
                    st.download_button(
                        label=os.path.basename(path),
                        data=f.read(),
                        file_name=os.path.basename(path),
                        mime="application/octet-stream"
                    )

        except Exception as e:
            st.error(f"Error: {e}")
            logger.error(str(e))

if __name__ == "__main__":
    main()
