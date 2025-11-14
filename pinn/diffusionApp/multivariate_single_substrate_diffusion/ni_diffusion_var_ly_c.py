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

# === CONFIGURATION ===
OUTPUT_DIR = '/tmp/pinn_solutions'
os.makedirs(OUTPUT_DIR, exist_ok=True)

mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['figure.dpi'] = 300

logging.basicConfig(level=logging.INFO, filename=os.path.join(OUTPUT_DIR, 'training.log'), filemode='a')
logger = logging.getLogger(__name__)

# Fixed parameters
C_CU_TOP = 0.0      # Top: Cu-poor
C_CU_BOTTOM = 0.0   # Bottom: Cu-poor
C_NI_TOP = 0.0      # Top: Ni-poor
Lx = 60.0           # Domain width (μm)
D11 = 0.006
D12 = 0.00427
D21 = 0.003697
D22 = 0.0054
T_max = 200.0
epochs = 5000
lr = 1e-3

torch.manual_seed(42)
np.random.seed(42)

# === USER INPUTS ===
st.sidebar.header("Simulation Parameters")

Ly = st.sidebar.number_input(
    "Domain Height (Ly) [μm]",
    min_value=10.0,
    max_value=200.0,
    value=50.0,
    step=5.0,
    help="Height of the solder joint domain (y-direction)."
)

C_NI_BOTTOM = st.sidebar.number_input(
    "Bottom Ni Concentration (C_Ni at y=0) [mol/cc]",
    min_value=1.0e-6,
    max_value=1.0e-3,
    value=4.0e-4,
    step=1.0e-5,
    format="%.2e",
    help="Ni concentration at the bottom boundary (y = 0)."
)

# === CACHE KEY ===
def get_cache_key(*args):
    key_string = "_".join(str(arg) for arg in args)
    return hashlib.md5(key_string.encode()).hexdigest()

# === MODEL & FUNCTIONS (unchanged except boundary updates) ===
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

# === BOUNDARY LOSS FUNCTIONS (updated to use C_NI_BOTTOM) ===
def boundary_loss_bottom(model):
    num = 200
    x = torch.rand(num, 1, requires_grad=True) * model.Lx
    y = torch.zeros(num, 1, requires_grad=True)
    t = torch.rand(num, 1, requires_grad=True) * model.T_max
   
    c_pred = model(x, y, t)
    return (torch.mean((c_pred[:, 0] - C_CU_BOTTOM)**2) +
            torch.mean((c_pred[:, 1] - C_NI_BOTTOM)**2))

def boundary_loss_top(model):
    num = 200
    x = torch.rand(num, 1, requires_grad=True) * model.Lx
    y = torch.full((num, 1), model.Ly, requires_grad=True)
    t = torch.rand(num, 1, requires_grad=True) * model.T_max
   
    c_pred = model(x, y, t)
    return (torch.mean((c_pred[:, 0] - C_CU_TOP)**2) +
            torch.mean((c_pred[:, 1] - C_NI_TOP)**2))

# === BOUNDARY VALIDATION (FIXED: swapped Ni top/bottom) ===
def validate_boundary_conditions(solution, tolerance=1e-6):
    results = {
        'top_bc_cu': True, 'top_bc_ni': True,
        'bottom_bc_cu': True, 'bottom_bc_ni': True,
        'left_flux_cu': True, 'left_flux_ni': True,
        'right_flux_cu': True, 'right_flux_ni': True,
        'details': []
    }
    t_idx = -1
    c1 = solution['c1_preds'][t_idx]  # Cu
    c2 = solution['c2_preds'][t_idx]  # Ni

    # Top boundary (y = Ly)
    top_cu_mean = np.mean(c1[:, -1])
    top_ni_mean = np.mean(c2[:, -1])
    if abs(top_cu_mean - C_CU_TOP) > tolerance:
        results['top_bc_cu'] = False
        results['details'].append(f"Top Cu: {top_cu_mean:.2e} ≠ {C_CU_TOP:.2e}")
    if abs(top_ni_mean - C_NI_TOP) > tolerance:
        results['top_bc_ni'] = False
        results['details'].append(f"Top Ni: {top_ni_mean:.2e} ≠ {C_NI_TOP:.2e}")

    # Bottom boundary (y = 0)
    bottom_cu_mean = np.mean(c1[:, 0])
    bottom_ni_mean = np.mean(c2[:, 0])
    if abs(bottom_cu_mean - C_CU_BOTTOM) > tolerance:
        results['bottom_bc_cu'] = False
        results['details'].append(f"Bottom Cu: {bottom_cu_mean:.2e} ≠ {C_CU_BOTTOM:.2e}")
    if abs(bottom_ni_mean - C_NI_BOTTOM) > tolerance:
        results['bottom_bc_ni'] = False
        results['details'].append(f"Bottom Ni: {bottom_ni_mean:.2e} ≠ {C_NI_BOTTOM:.2e}")

    # Side flux (approximate zero gradient)
    left_flux_cu = np.mean(np.abs(c1[1, :] - c1[0, :]))
    left_flux_ni = np.mean(np.abs(c2[1, :] - c2[0, :]))
    right_flux_cu = np.mean(np.abs(c1[-1, :] - c1[-2, :]))
    right_flux_ni = np.mean(np.abs(c2[-1, :] - c2[-2, :]))

    if left_flux_cu > tolerance:
        results['left_flux_cu'] = False
        results['details'].append(f"Left flux Cu: {left_flux_cu:.2e}")
    if left_flux_ni > tolerance:
        results['left_flux_ni'] = False
        results['details'].append(f"Left flux Ni: {left_flux_ni:.2e}")
    if right_flux_cu > tolerance:
        results['right_flux_cu'] = False
        results['details'].append(f"Right flux Cu: {right_flux_cu:.2e}")
    if right_flux_ni > tolerance:
        results['right_flux_ni'] = False
        results['details'].append(f"Right flux Ni: {right_flux_ni:.2e}")

    results['valid'] = all([
        results['top_bc_cu'], results['top_bc_ni'],
        results['bottom_bc_cu'], results['bottom_bc_ni'],
        results['left_flux_cu'], results['left_flux_ni'],
        results['right_flux_cu'], results['right_flux_ni']
    ])
    return results

# === CACHING & TRAINING (updated cache key) ===
@st.cache_resource(ttl=3600, show_spinner=False)
def train_model(D11, D12, D21, D22, Lx, Ly, T_max, C_Cu, C_Ni, epochs, lr, output_dir, _hash):
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Training: Ly={Ly}, C_Ni_bottom={C_Ni}, epochs={epochs}")
    model = DualScaledPINN(D11, D12, D21, D22, Lx, Ly, T_max, C_Cu, C_Ni)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    x_pde = torch.rand(1000, 1, requires_grad=True) * Lx
    y_pde = torch.rand(1000, 1, requires_grad=True) * Ly
    t_pde = torch.rand(1000, 1, requires_grad=True) * T_max

    loss_history = {'epochs': [], 'total': [], 'physics': [], 'bottom': [], 'top': [], 'sides': [], 'initial': []}
    progress = st.progress(0)
    status_text = st.empty()

    for epoch in range(epochs):
        optimizer.zero_grad()
        phys_loss = physics_loss(model, x_pde, y_pde, t_pde)
        bot_loss = boundary_loss_bottom(model)
        top_loss = boundary_loss_top(model)
        side_loss = boundary_loss_sides(model)
        init_loss = initial_loss(model)

        loss = (10 * phys_loss + 100 * bot_loss + 100 * top_loss + 100 * side_loss + 100 * init_loss)

        try:
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        except RuntimeError as e:
            logger.error(f"Backward failed at epoch {epoch + 1}: {str(e)}")
            st.error(f"Training failed: {str(e)}")
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
            status_text.text(f"Epoch {epoch + 1}: Loss={loss.item():.2e}")

    progress.progress(1.0)
    status_text.text("Training complete!")
    return model, loss_history

# === MAIN APP ===
def main():
    st.title("2D PINN Simulation: Ni Self-Diffusion in Liquid Solder")

    initialize_session_state()
    current_hash = get_cache_key(Ly, C_NI_BOTTOM, epochs, lr)

    if st.session_state.training_complete and st.session_state.current_hash == current_hash:
        solution = st.session_state.solution_data
        file_info = st.session_state.file_data
        model = st.session_state.model
        st.success("Loaded from cache.")
    else:
        solution = None
        file_info = {}
        model = None

    if st.button("Run Simulation", type="primary"):
        with st.spinner("Training PINN model..."):
            model, loss_history = train_model(
                D11, D12, D21, D22, Lx, Ly, T_max,
                C_CU_TOP, C_NI_BOTTOM, epochs, lr, OUTPUT_DIR, current_hash
            )
            if model is None:
                st.error("Training failed.")
                return

            solution, file_info = train_and_generate_solution(model, loss_history, OUTPUT_DIR, current_hash)
            if solution is None:
                st.error("Solution generation failed.")
                return

            store_solution_in_session(current_hash, solution, file_info, model)
            st.success("Simulation completed!")

    if solution and file_info:
        st.subheader("Training Loss")
        st.image(file_info['loss_plot'])

        st.subheader("Boundary Conditions")
        bc = validate_boundary_conditions(solution)
        st.write(f"**Valid:** {'Yes' if bc['valid'] else 'No'}")
        if bc['details']:
            with st.expander("Issues"):
                for d in bc['details']:
                    st.write(f"• {d}")

        st.subheader("Final Concentration Profiles")
        st.image(file_info['profile_plot'])

        st.subheader("Downloads")
        cols = st.columns(3)
        with cols[0]:
            if file_info.get('solution_file'):
                st.download_button("Solution (.pkl)", get_file_bytes(file_info['solution_file']), "solution.pkl")
        with cols[1]:
            if file_info.get('pvd_file'):
                st.download_button("VTS Series (.pvd)", get_file_bytes(file_info['pvd_file']), "series.pvd")
        with cols[2]:
            if file_info.get('vtu_file'):
                st.download_button("VTU File", get_file_bytes(file_info['vtu_file']), "series.vtu")

        if st.button("Generate ZIP"):
            with st.spinner("Zipping..."):
                zip_path = create_zip_file([
                    file_info['loss_plot'], file_info['profile_plot'],
                    file_info['solution_file'], file_info['pvd_file'], file_info['vtu_file']
                ] + [f for _, f in file_info.get('vts_files', [])], OUTPUT_DIR, current_hash)
                if zip_path:
                    st.download_button("Download All (.zip)", get_file_bytes(zip_path), os.path.basename(zip_path))

if __name__ == "__main__":
    main()
