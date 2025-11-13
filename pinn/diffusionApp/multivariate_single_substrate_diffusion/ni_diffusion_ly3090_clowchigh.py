# batch_ni_diffusion.py
# Batch trainer for pure Ni self-diffusion from bottom (variable Ly + C_Ni_bottom)

import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import zipfile
import io
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

logging.basicConfig(level=logging.INFO,
                    filename=os.path.join(OUTPUT_DIR, 'batch_ni_training.log'),
                    filemode='a')
logger = logging.getLogger(__name__)

# Fixed physics (pure Ni self-diffusion)
C_CU_TOP = C_CU_BOTTOM = 0.0
C_NI_TOP = 0.0
Lx = 60.0
D11 = 0.006          # Cu self (unused)
D22 = 0.0054         # Ni self-diffusion coefficient
D12 = D21 = 0.0      # No coupling
T_max = 200.0
epochs = 5000
lr = 1e-3
times_eval = np.linspace(0, T_max, 50)

# ----------------------------------------------------------------------
# Model (same architecture, but only Ni matters)
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

# ----------------------------------------------------------------------
# Loss functions (pure Ni diffusion)
# ----------------------------------------------------------------------
def laplacian(c, x, y):
    c_x = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c), create_graph=True, retain_graph=True)[0]
    c_y = torch.autograd.grad(c, y, grad_outputs=torch.ones_like(c), create_graph=True, retain_graph=True)[0]
    c_xx = torch.autograd.grad(c_x, x, grad_outputs=torch.ones_like(c_x), create_graph=True, retain_graph=True)[0]
    c_yy = torch.autograd.grad(c_y, y, grad_outputs=torch.ones_like(c_y), create_graph=True, retain_graph=True)[0]
    return c_xx + c_yy

def physics_loss(model, x, y, t):
    c = model(x, y, t)[:, 1:2]  # Only Ni (index 1)
    c_t = torch.autograd.grad(c, t, grad_outputs=torch.ones_like(c), create_graph=True, retain_graph=True)[0]
    lap_c = laplacian(c, x, y)
    return torch.mean((c_t - model.D22 * lap_c)**2)

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
    return torch.mean(c_ni**2)  # C_Ni_top = 0

def boundary_loss_sides(model):
    n = 200
    # Left & Right flux = 0 for Ni
    x_left = torch.zeros(n, 1, requires_grad=True)
    y_left = torch.rand(n, 1, requires_grad=True) * model.Ly
    t_left = torch.rand(n, 1, requires_grad=True) * model.T_max
    c_ni_left = model(x_left, y_left, t_left)[:, 1]
    grad_left = torch.autograd.grad(c_ni_left, x_left, torch.ones_like(c_ni_left), create_graph=True)[0]

    x_right = torch.full((n, 1), model.Lx, requires_grad=True)
    y_right = torch.rand(n, 1, requires_grad=True) * model.Ly
    t_right = torch.rand(n, 1, requires_grad=True) * model.T_max
    c_ni_right = model(x_right, y_right, t_right)[:, 1]
    grad_right = torch.autograd.grad(c_ni_right, x_right, torch.ones_like(c_ni_right), create_graph=True)[0]

    return torch.mean(grad_left**2) + torch.mean(grad_right**2)

def initial_loss(model):
    n = 500
    x = torch.rand(n, 1, requires_grad=True) * model.Lx
    y = torch.rand(n, 1, requires_grad=True) * model.Ly
    t = torch.zeros(n, 1, requires_grad=True)
    c_ni = model(x, y, t)[:, 1]
    return torch.mean(c_ni**2)

# ----------------------------------------------------------------------
# Single case training
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_single_ni_case(Ly_val, C_Ni_bottom_val, _hash):
    model = DualScaledPINN(D11, D12, D21, D22, Lx, Ly_val, T_max, C_CU_BOTTOM, C_Ni_bottom_val)
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
            status.text(f"Ly={Ly_val:.1f} μm | C_Ni_bottom={C_Ni_bottom_val:.2e} → Epoch {epoch}, Loss: {loss.item():.2e}")

    progress.progress(1.0)
    status.text(f"Complete: Ly={Ly_val:.1f} μm, C_Ni={C_Ni_bottom_val:.2e}")
    return model

# ----------------------------------------------------------------------
# Evaluate & save
# ----------------------------------------------------------------------
def evaluate_and_save_ni(model, Ly_val, C_Ni_bottom_val):
    x = torch.linspace(0, Lx, 50)
    y = torch.linspace(0, Ly_val, 50)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    c2_preds = []  # Ni concentration
    for t_val in times_eval:
        t = torch.full((X.numel(), 1), t_val)
        c_ni = model(X.reshape(-1,1), Y.reshape(-1,1), t)[:, 1].detach().numpy()
        c2_preds.append(c_ni.reshape(50, 50).T)

    # Unique filename
    cni_str = f"{C_Ni_bottom_val:.1e}".replace(".", "").replace("+", "")
    filename = f"solutions_Ni_diffusion_ly_{Ly_val:.1f}_cni_{cni_str}.pkl"
    filepath = os.path.join(OUTPUT_DIR, filename)

    solution = {
        'params': {
            'Lx': Lx, 'Ly': Ly_val,
            'C_Cu': 0.0, 'C_Ni': C_Ni_bottom_val,
            'D11': D11, 'D12': D12, 'D21': D21, 'D22': D22,
            't_max': T_max
        },
        'X': X.numpy(),
        'Y': Y.numpy(),
        'times': times_eval.tolist(),
        'c1_preds': [np.zeros_like(c) for c in c2_preds],  # Cu = 0
        'c2_preds': c2_preds
    }

    with open(filepath, "wb") as f:
        pickle.dump(solution, f)
    logger.info(f"Saved Ni solution: {filename}")
    return filepath, solution

# ----------------------------------------------------------------------
# Main App
# ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Batch Ni Diffusion Trainer", layout="wide")
    st.title("Batch Pure Ni Self-Diffusion PINN Trainer")
    st.markdown("""
    Enter arrays of **solder height (Ly)** and **bottom Ni concentration (C_Ni_bottom)**.  
    One model will be trained per pair.  
    Output files: `solutions_Ni_diffusion_ly_X_cni_Y.pkl`
    """)

    ly_text = st.text_area("Ly values (μm) — one per line or comma-separated", "30\n50\n70\n90\n110")
    cni_text = st.text_area("C_Ni_bottom values (mol/cc) — same number of entries", "4e-4\n8e-4\n1.2e-3\n1.6e-3\n1.8e-3")

    if st.button("Start Batch Training", type="primary"):
        try:
            ly_list = [float(x) for x in re.split(r'[,\n]+', ly_text.strip()) if x.strip()]
            cni_list = [float(x) for x in re.split(r'[,\n]+', cni_text.strip()) if x.strip()]

            if len(ly_list) != len(cni_list):
                st.error("Number of Ly and C_Ni_bottom values must match!")
                return
            if not ly_list:
                st.error("No valid input.")
                return

            st.success(f"Training {len(ly_list)} Ni self-diffusion cases...")

            results = []
            prog_bar = st.progress(0)
            status = st.empty()

            for idx, (Ly_val, C_Ni_val) in enumerate(zip(ly_list, cni_list)):
                status.write(f"Training {idx+1}/{len(ly_list)} → Ly = {Ly_val} μm, C_Ni = {C_Ni_val:.2e}")
                hash_key = hashlib.md5(f"{Ly_val}_{C_Ni_val}".encode()).hexdigest()
                model = train_single_ni_case(Ly_val, C_Ni_val, hash_key)
                pkl_path, sol = evaluate_and_save_ni(model, Ly_val, C_Ni_val)
                results.append((pkl_path, sol))
                prog_bar.progress((idx + 1) / len(ly_list))

            st.success("All Ni trainings completed!")

            # ZIP all results
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for path, _ in results:
                    if os.path.exists(path):
                        zf.write(path, os.path.basename(path))
            zip_buffer.seek(0)

            st.download_button(
                "Download All Ni Solutions (.zip)",
                data=zip_buffer,
                file_name=f"ni_diffusion_batch_{datetime.now():%Y%m%d_%H%M%S}.zip",
                mime="application/zip"
            )

            st.write("### Individual Files")
            for path, _ in results:
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
