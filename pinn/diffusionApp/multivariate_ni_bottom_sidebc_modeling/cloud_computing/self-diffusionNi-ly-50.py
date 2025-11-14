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

# === USER INPUTS AT THE TOP ===
st.set_page_config(page_title="PINN Ni Diffusion", layout="wide")
st.title("2D PINN Simulation: Ni Self-Diffusion in Liquid Solder")

# User inputs
col1, col2 = st.columns(2)
with col1:
    Ly = st.number_input("Domain Height Ly (μm)", min_value=1.0, max_value=200.0, value=50.0, step=5.0)
with col2:
    C_NI_BOTTOM = st.number_input("Bottom Ni Concentration (mol/cc)", min_value=1e-6, max_value=1e-3, value=4.0e-4, format="%.2e")

# Fixed parameters (can be made input later if needed)
C_CU_TOP = 0.0
C_CU_BOTTOM = 0.0
C_NI_TOP = 0.0  # Top is Ni-poor
Lx = 60.0
D11 = 0.006
D12 = 0.00427
D21 = 0.003697
D22 = 0.0054
T_max = 200.0
epochs = 5000
lr = 1e-3

# === OUTPUT DIR & LOGGING ===
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

# === CACHE KEY WITH USER INPUTS ===
def get_cache_key(*args):
    key_string = "_".join(str(arg) for arg in args)
    return hashlib.md5(key_string.encode()).hexdigest()

# === UPDATE ALL FILE-NAMING FUNCTIONS TO USE Ly AND C_NI_BOTTOM ===
def format_ni_conc(conc):
    return f"{conc:.2e}".replace("+", "").replace("e-0", "e-")

NI_BOTTOM_STR = format_ni_conc(C_NI_BOTTOM)

# === MODIFIED PLOT & FILE FUNCTIONS ===
@st.cache_data(ttl=3600, show_spinner=False)
def plot_losses(loss_history, output_dir, _hash):
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
    plt.title(f'Training Loss (Ly={Ly:.1f} μm, C_Ni_bottom={C_NI_BOTTOM:.2e})')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend(loc='upper right')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f'loss_plot_Ly{Ly:.1f}_NiBottom{NI_BOTTOM_STR}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved loss plot to {plot_filename}")
    return plot_filename

@st.cache_data(ttl=3600, show_spinner=False)
def plot_2d_profiles(solution, time_idx, output_dir, _hash):
    t_val = solution['times'][time_idx]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    im1 = plt.imshow(solution['c1_preds'][time_idx], origin='lower',
                     extent=[0, Lx, 0, Ly], cmap='viridis',
                     vmin=0, vmax=C_CU_TOP)
    plt.title(f'Cu Concentration (t={t_val:.1f} s)')
    plt.xlabel('x (μm)')
    plt.ylabel('y (μm)')
    plt.grid(True, alpha=0.3)
    plt.colorbar(im1, label='Cu Conc. (mol/cc)', format='%.1e')

    plt.subplot(1, 2, 2)
    im2 = plt.imshow(solution['c2_preds'][time_idx], origin='lower',
                     extent=[0, Lx, 0, Ly], cmap='magma',
                     vmin=0, vmax=C_NI_TOP)
    plt.title(f'Ni Concentration (t={t_val:.1f} s)')
    plt.xlabel('x (μm)')
    plt.ylabel('y (μm)')
    plt.grid(True, alpha=0.3)
    plt.colorbar(im2, label='Ni Conc. (mol/cc)', format='%.1e')

    plt.suptitle(f'2D Profiles (Ly={Ly:.0f} μm, C_Ni_bottom={C_NI_BOTTOM:.2e})', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f'profile_Ly{Ly:.1f}_NiBottom{NI_BOTTOM_STR}_t{t_val:.1f}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved profile plot to {plot_filename}")
    return plot_filename

@st.cache_resource(ttl=3600, show_spinner=False)
def train_model(D11, D12, D21, D22, Lx, Ly, T_max, C_Cu, C_Ni, epochs, lr, output_dir, _hash):
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Starting training with Ly={Ly}, C_Ni_bottom={C_Ni}, epochs={epochs}, lr={lr}")
    model = DualScaledPINN(D11, D12, D21, D22, Lx, Ly, T_max, C_Cu, C_Ni)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    x_pde = torch.rand(1000, 1, requires_grad=True) * Lx
    y_pde = torch.rand(1000, 1, requires_grad=True) * Ly
    t_pde = torch.rand(1000, 1, requires_grad=True) * T_max

    loss_history = {
        'epochs': [], 'total': [], 'physics': [], 'bottom': [], 'top': [], 'sides': [], 'initial': []
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
                f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}"
            )

    progress.progress(1.0)
    status_text.text("Training completed!")
    return model, loss_history

@st.cache_data(ttl=3600, show_spinner=False)
def generate_and_save_solution(_model, times, param_set, output_dir, _hash):
    os.makedirs(output_dir, exist_ok=True)
    if _model is None:
        return None, None

    X, Y, c1_preds, c2_preds = evaluate_model(
        _model, times, param_set['Lx'], param_set['Ly'],
        param_set['D11'], param_set['D12'], param_set['D21'], param_set['D22'], _hash
    )

    solution = {
        'params': param_set,
        'X': X, 'Y': Y,
        'c1_preds': c1_preds, 'c2_preds': c2_preds,
        'times': times,
        'loss_history': {},
        'orientation_note': 'c1_preds and c2_preds are arrays of shape (50,50) where rows (i) correspond to y-coordinates and columns (j) correspond to x-coordinates for matplotlib.'
    }

    solution_filename = os.path.join(output_dir,
        f"solution_Ly{Ly:.1f}_NiBottom{NI_BOTTOM_STR}.pkl")

    with open(solution_filename, 'wb') as f:
        pickle.dump(solution, f)
    logger.info(f"Saved solution to {solution_filename}")
    return solution_filename, solution

@st.cache_data(ttl=3600, show_spinner=False)
def generate_vts_time_series(solution, output_dir, _hash):
    os.makedirs(output_dir, exist_ok=True)
    Lx = solution['params']['Lx']
    Ly = solution['params']['Ly']
    times = solution['times']
    vts_files = []
    nx, ny = 50, 50

    for t_idx, t_val in enumerate(times):
        c1_xy = solution['c1_preds'][t_idx].T
        c2_xy = solution['c2_preds'][t_idx].T

        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        z = np.zeros((nx, ny))
        grid = pv.StructuredGrid()
        X, Y = np.meshgrid(x, y, indexing='ij')
        points = np.stack([X.ravel(), Y.ravel(), z.ravel()], axis=1)
        grid.points = points
        grid.dimensions = (nx, ny, 1)
        grid.point_data['Cu_Concentration'] = c1_xy.ravel()
        grid.point_data['Ni_Concentration'] = c2_xy.ravel()

        vts_filename = os.path.join(output_dir,
            f'concentration_Ly{Ly:.1f}_NiBottom{NI_BOTTOM_STR}_t{t_val:.1f}.vts')
        grid.save(vts_filename)
        vts_files.append((t_val, vts_filename))

    pvd_filename = os.path.join(output_dir,
        f'concentration_time_series_Ly{Ly:.1f}_NiBottom{NI_BOTTOM_STR}.pvd')

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

    return vts_files, pvd_filename

@st.cache_data(ttl=3600, show_spinner=False)
def generate_vtu_time_series(solution, output_dir, _hash):
    os.makedirs(output_dir, exist_ok=True)
    nx, ny = 50, 50
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    z = np.zeros((nx, ny))
    X, Y = np.meshgrid(x, y, indexing='ij')
    points = np.stack([X.ravel(), Y.ravel(), z.ravel()], axis=1)

    cells = []
    cell_types = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            idx = i + j * nx
            cell = [4, idx, idx + 1, idx + nx + 1, idx + nx]
            cells.extend(cell)
            cell_types.append(pv.CellType.QUAD)

    grid = pv.UnstructuredGrid(cells, cell_types, points)

    for t_idx, t_val in enumerate(solution['times']):
        c1_xy = solution['c1_preds'][t_idx].T
        c2_xy = solution['c2_preds'][t_idx].T
        grid.point_data[f'Cu_Concentration_t{t_val:.1f}'] = c1_xy.ravel()
        grid.point_data[f'Ni_Concentration_t{t_val:.1f}'] = c2_xy.ravel()

    vtu_filename = os.path.join(output_dir,
        f'concentration_time_series_Ly{Ly:.1f}_NiBottom{NI_BOTTOM_STR}.vtu')
    grid.save(vtu_filename)
    return vtu_filename

@st.cache_data(ttl=3600, show_spinner=False)
def create_zip_file(_files, output_dir, _hash):
    os.makedirs(output_dir, exist_ok=True)
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in _files:
            if os.path.exists(file_path):
                zip_file.write(file_path, os.path.basename(file_path))

    zip_filename = os.path.join(output_dir, f'pinn_solution_Ly{Ly:.1f}_NiBottom{NI_BOTTOM_STR}.zip')
    with open(zip_filename, 'wb') as f:
        f.write(zip_buffer.getvalue())
    logger.info(f"Created ZIP file: {zip_filename}")
    return zip_filename

# === MAIN FUNCTION ===
def main():
    initialize_session_state()
    current_hash = get_cache_key(Ly, C_NI_BOTTOM, epochs, lr)

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

    if st.button("Run Simulation"):
        with st.spinner("Running simulation..."):
            model, loss_history = train_model(
                D11, D12, D21, D22, Lx, Ly, T_max, C_CU_TOP, C_NI_BOTTOM, epochs, lr, OUTPUT_DIR, current_hash
            )
            if model is None:
                st.error("Simulation failed!")
                return

            times = np.linspace(0, T_max, 50)
            param_set = {
                'D11': D11, 'D12': D12, 'D21': D21, 'D22': D22,
                'Lx': Lx, 'Ly': Ly, 't_max': T_max,
                'C_Cu': C_CU_TOP, 'C_Ni': C_NI_BOTTOM,
                'epochs': epochs
            }

            solution_filename, solution = generate_and_save_solution(model, times, param_set, OUTPUT_DIR, current_hash)
            if solution is None:
                st.error("Solution generation failed!")
                return

            solution['loss_history'] = loss_history
            loss_plot_filename = plot_losses(loss_history, OUTPUT_DIR, current_hash)
            profile_plot_filename = plot_2d_profiles(solution, -1, OUTPUT_DIR, current_hash)
            vts_files, pvd_file = generate_vts_time_series(solution, OUTPUT_DIR, current_hash)
            vtu_file = generate_vtu_time_series(solution, OUTPUT_DIR, current_hash)

            file_info = {
                'solution_file': solution_filename,
                'loss_plot': loss_plot_filename,
                'profile_plot': profile_plot_filename,
                'vts_files': vts_files,
                'pvd_file': pvd_file,
                'vtu_file': vtu_file
            }

            store_solution_in_session(current_hash, solution, file_info, model)
            st.success("Simulation completed successfully!")

    if solution and file_info:
        st.subheader("Training Loss")
        st.image(file_info['loss_plot'])

        st.subheader("2D Concentration Profiles (Final Time Step)")
        st.image(file_info['profile_plot'])

        st.subheader("Download All Files as ZIP")
        if st.button("Generate ZIP File"):
            with st.spinner("Creating ZIP file..."):
                files_to_zip = [
                    file_info['solution_file'], file_info['loss_plot'], file_info['profile_plot'],
                    file_info['pvd_file'], file_info['vtu_file']
                ] + [v[1] for v in file_info['vts_files']]
                zip_filename = create_zip_file(files_to_zip, OUTPUT_DIR, current_hash)
                if zip_filename and os.path.exists(zip_filename):
                    zip_data = get_file_bytes(zip_filename)
                    st.download_button(
                        label=f"Download All (Ly={Ly:.1f}μm, Ni={C_NI_BOTTOM:.2e}).zip",
                        data=zip_data,
                        file_name=os.path.basename(zip_filename),
                        mime="application/zip"
                    )

        # Individual downloads
        for label, key in [
            ("Solution (.pkl)", 'solution_file'),
            ("Loss Plot", 'loss_plot'),
            ("Profile Plot", 'profile_plot'),
            ("PVD Collection", 'pvd_file'),
            ("VTU Time Series", 'vtu_file')
        ]:
            path = file_info.get(key)
            if path and os.path.exists(path):
                data = get_file_bytes(path)
                st.download_button(f"Download {label}", data=data, file_name=os.path.basename(path), mime="application/octet-stream")

        st.subheader("Individual Time Steps (.vts)")
        for t_val, vts_file in file_info.get('vts_files', []):
            if os.path.exists(vts_file):
                data = get_file_bytes(vts_file)
                st.download_button(f"t = {t_val:.1f}s", data=data, file_name=os.path.basename(vts_file))

if __name__ == "__main__":
    main()
