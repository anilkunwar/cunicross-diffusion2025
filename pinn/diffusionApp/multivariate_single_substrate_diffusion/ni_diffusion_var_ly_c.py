# --------------------------------------------------------------
#  2D PINN – Ni Self-Diffusion (Soft Constraints + Gradient Viz)
# --------------------------------------------------------------
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

# --------------------------------------------------------------
#  Output / logging
# --------------------------------------------------------------
OUTPUT_DIR = '/tmp/pinn_solutions'
os.makedirs(OUTPUT_DIR, exist_ok=True)

mpl.rcParams.update({
    'font.family': 'Arial', 'font.size': 12,
    'axes.linewidth': 1.5, 'xtick.major.width': 1.5, 'ytick.major.width': 1.5,
    'figure.dpi': 300
})
logging.basicConfig(level=logging.INFO,
                    filename=os.path.join(OUTPUT_DIR, 'training.log'),
                    filemode='a')
logger = logging.getLogger(__name__)

# --------------------------------------------------------------
#  Fixed / default parameters
# --------------------------------------------------------------
C_NI_TOP = 0.0                 # Top boundary (y = Ly)
Lx = 60.0                      # Domain width (µm)
T_max = 200.0
epochs = 5000
lr = 1e-3

torch.manual_seed(42)
np.random.seed(42)

def get_cache_key(*args):
    return hashlib.md5("_".join(str(a) for a in args).encode()).hexdigest()

# --------------------------------------------------------------
#  Model (soft constraints)
# --------------------------------------------------------------
class ScaledPINN(nn.Module):
    def __init__(self, D, Lx, Ly, T_max, C_bottom):
        super().__init__()
        self.D = D
        self.Lx, self.Ly, self.T_max = Lx, Ly, T_max
        self.C_bottom = C_bottom

        self.net = nn.Sequential(
            nn.Linear(3, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x, y, t):
        x_n = x / self.Lx
        y_n = y / self.Ly
        t_n = t / self.T_max
        inp = torch.cat([x_n, y_n, t_n], dim=1)
        return self.net(inp)

# --------------------------------------------------------------
#  PDE & BC losses
# --------------------------------------------------------------
def laplacian(c, x, y):
    cx = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c),
                             create_graph=True, retain_graph=True)[0]
    cy = torch.autograd.grad(c, y, grad_outputs=torch.ones_like(c),
                             create_graph=True, retain_graph=True)[0]
    cxx = torch.autograd.grad(cx, x, grad_outputs=torch.ones_like(cx),
                              create_graph=True, retain_graph=True)[0]
    cyy = torch.autograd.grad(cy, y, grad_outputs=torch.ones_like(cy),
                              create_graph=True, retain_graph=True)[0]
    return cxx + cyy

def physics_loss(model, x, y, t):
    c = model(x, y, t)
    ct = torch.autograd.grad(c, t, grad_outputs=torch.ones_like(c),
                             create_graph=True, retain_graph=True)[0]
    lap = laplacian(c, x, y)
    res = ct - model.D * lap
    return torch.mean(res**2)

def boundary_loss_bottom(model):
    n = 200
    x = torch.rand(n, 1, requires_grad=True) * model.Lx
    y = torch.zeros(n, 1, requires_grad=True)
    t = torch.rand(n, 1, requires_grad=True) * model.T_max
    c = model(x, y, t)
    return torch.mean((c - model.C_bottom)**2)

def boundary_loss_top(model):
    n = 200
    x = torch.rand(n, 1, requires_grad=True) * model.Lx
    y = torch.full((n, 1), model.Ly, requires_grad=True)
    t = torch.rand(n, 1, requires_grad=True) * model.T_max
    c = model(x, y, t)
    return torch.mean((c - C_NI_TOP)**2)

def boundary_loss_sides(model):
    n = 200
    xl = torch.zeros(n, 1, requires_grad=True)
    yl = torch.rand(n, 1, requires_grad=True) * model.Ly
    tl = torch.rand(n, 1, requires_grad=True) * model.T_max
    cl = model(xl, yl, tl)

    xr = torch.full((n, 1), model.Lx, requires_grad=True)
    yr = torch.rand(n, 1, requires_grad=True) * model.Ly
    tr = torch.rand(n, 1, requires_grad=True) * model.T_max
    cr = model(xr, yr, tr)

    try:
        gl = torch.autograd.grad(cl, xl, grad_outputs=torch.ones_like(cl),
                                 create_graph=True, retain_graph=True)[0]
        gr = torch.autograd.grad(cr, xr, grad_outputs=torch.ones_like(cr),
                                 create_graph=True, retain_graph=True)[0]
        gl = gl if gl is not None else torch.zeros_like(cl)
        gr = gr if gr is not None else torch.zeros_like(cr)
        return torch.mean(gl**2) + torch.mean(gr**2)
    except RuntimeError as e:
        logger.error(f"Side gradient failed: {e}")
        st.error(f"Gradient failed: {e}")
        return torch.tensor(1e-6, requires_grad=True)

def initial_loss(model):
    n = 500
    x = torch.rand(n, 1, requires_grad=True) * model.Lx
    y = torch.rand(n, 1, requires_grad=True) * model.Ly
    t = torch.zeros(n, 1, requires_grad=True)
    return torch.mean(model(x, y, t)**2)

# --------------------------------------------------------------
#  Validation
# --------------------------------------------------------------
def validate_boundary_conditions(solution, tolerance=1e-6):
    results = {'top_bc': True, 'bottom_bc': True,
               'left_flux': True, 'right_flux': True, 'details': []}
    c = solution['c_preds'][-1]
    top_mean = np.mean(c[:, -1])
    bottom_mean = np.mean(c[:, 0])
    left_flux = np.mean(np.abs(c[1, :] - c[0, :]))
    right_flux = np.mean(np.abs(c[-1, :] - c[-2, :]))

    if abs(top_mean - C_NI_TOP) > tolerance:
        results['top_bc'] = False
        results['details'].append(f"Top: {top_mean:.2e} != {C_NI_TOP:.2e}")
    if abs(bottom_mean - solution['params']['C_bottom']) > tolerance:
        results['bottom_bc'] = False
        results['details'].append(f"Bottom: {bottom_mean:.2e} != {solution['params']['C_bottom']:.2e}")
    if left_flux > tolerance:
        results['left_flux'] = False
        results['details'].append(f"Left flux: {left_flux:.2e}")
    if right_flux > tolerance:
        results['right_flux'] = False
        results['details'].append(f"Right flux: {right_flux:.2e}")

    results['valid'] = all([results[k] for k in ['top_bc', 'bottom_bc', 'left_flux', 'right_flux']])
    return results

# --------------------------------------------------------------
#  Plotting
# --------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def plot_losses(loss_history, output_dir, _hash, Ly, C_bottom):
    e = np.array(loss_history['epochs'])
    plt.figure(figsize=(10,6))
    plt.plot(e, loss_history['total'], label='Total', lw=2, color='k')
    plt.plot(e, loss_history['physics'], '--', label='Physics', lw=1.5, color='b')
    plt.plot(e, loss_history['bottom'], '-.', label='Bottom BC', lw=1.5, color='r')
    plt.plot(e, loss_history['top'], ':', label='Top BC', lw=1.5, color='g')
    plt.plot(e, loss_history['sides'], '-', label='Sides BC', lw=1.5, color='purple')
    plt.plot(e, loss_history['initial'], '--', label='IC', lw=1.5, color='orange')
    plt.yscale('log'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title(f'Training Loss (Ni, Ly={Ly:.1f} μm, c={C_bottom:.1e})')
    plt.grid(True, which='both', ls='--', alpha=0.7); plt.legend(); plt.tight_layout()
    fn = os.path.join(output_dir, f'loss_ni_soft_ly_{Ly:.1f}_c_{C_bottom:.1e}.png')
    plt.savefig(fn, dpi=300, bbox_inches='tight'); plt.close()
    return fn

@st.cache_data(ttl=3600, show_spinner=False)
def plot_2d_profiles(solution, time_idx, output_dir, _hash):
    t = solution['times'][time_idx]
    c = solution['c_preds'][time_idx]
    Ly = solution['params']['Ly']
    C_bottom = solution['params']['C_bottom']
    plt.figure(figsize=(6,5))
    im = plt.imshow(c, origin='lower', extent=[0, Lx, 0, Ly],
                    cmap='magma', vmin=0, vmax=C_bottom)
    plt.title(f'Ni Concentration (t={t:.1f} s)')
    plt.xlabel('x (µm)'); plt.ylabel('y (µm)'); plt.colorbar(im, label='mol/cc')
    plt.grid(alpha=0.3); plt.tight_layout()
    fn = os.path.join(output_dir, f'ni_profile_soft_ly_{Ly:.1f}_c_{C_bottom:.1e}_t_{t:.1f}.png')
    plt.savefig(fn, dpi=300, bbox_inches='tight'); plt.close()
    return fn

# NEW: Vertical profile (gradient evolution)
@st.cache_data(ttl=3600, show_spinner=False)
def plot_vertical_profile(solution, output_dir, _hash):
    times = solution['times'][::10]  # every 10th frame
    y = np.linspace(0, solution['params']['Ly'], 50)
    plt.figure(figsize=(8,5))
    for i, t in enumerate(times):
        c_center = solution['c_preds'][i*10][:, 25]  # middle x
        plt.plot(c_center, y, label=f't={t:.0f}s')
    plt.xlabel('Ni Concentration (mol/cc)'); plt.ylabel('y (µm)')
    plt.title('Vertical Profile (Center Line) – Diffusion Front')
    plt.legend(); plt.grid(alpha=0.4); plt.tight_layout()
    fn = os.path.join(output_dir, f'ni_vertical_profile_ly_{solution["params"]["Ly"]:.1f}_c_{solution["params"]["C_bottom"]:.1e}.png')
    plt.savefig(fn, dpi=300, bbox_inches='tight'); plt.close()
    return fn

# NEW: Gradient magnitude |∇c|
@st.cache_data(ttl=3600, show_spinner=False)
def plot_gradient_magnitude(solution, time_idx, output_dir, _hash):
    t = solution['times'][time_idx]
    c = solution['c_preds'][time_idx]
    Ly = solution['params']['Ly']
    C_bottom = solution['params']['C_bottom']

    # Compute gradient using numpy
    grad_y, grad_x = np.gradient(c, 1.0)  # dy, dx
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    plt.figure(figsize=(6,5))
    im = plt.imshow(grad_mag, origin='lower', extent=[0, Lx, 0, Ly],
                    cmap='hot', vmin=0, vmax=np.max(grad_mag))
    plt.title(f'|∇c| – Gradient Magnitude (t={t:.1f} s)')
    plt.xlabel('x (µm)'); plt.ylabel('y (µm)')
    plt.colorbar(im, label='|∇c| (mol/cc/µm)')
    plt.grid(alpha=0.3); plt.tight_layout()
    fn = os.path.join(output_dir, f'ni_gradient_mag_ly_{Ly:.1f}_c_{C_bottom:.1e}_t_{t:.1f}.png')
    plt.savefig(fn, dpi=300, bbox_inches='tight'); plt.close()
    return fn

# --------------------------------------------------------------
#  Training
# --------------------------------------------------------------
@st.cache_resource(ttl=3600, show_spinner=False)
def train_model(D, Lx, Ly, T_max, C_bottom, epochs, lr, output_dir, _hash):
    model = ScaledPINN(D, Lx, Ly, T_max, C_bottom)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    x_pde = torch.rand(1000, 1, requires_grad=True) * Lx
    y_pde = torch.rand(1000, 1, requires_grad=True) * Ly
    t_pde = torch.rand(1000, 1, requires_grad=True) * T_max

    loss_history = {
        'epochs': [], 'total': [], 'physics': [], 'bottom': [],
        'top': [], 'sides': [], 'initial': []
    }

    progress = st.progress(0)
    status_text = st.empty()

    for epoch in range(epochs):
        optimizer.zero_grad()
        phys = physics_loss(model, x_pde, y_pde, t_pde)
        bot = boundary_loss_bottom(model)
        top = boundary_loss_top(model)
        side = boundary_loss_sides(model)
        init = initial_loss(model)

        loss = 10*phys + 100*bot + 100*top + 100*side + 100*init
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            loss_history['epochs'].append(epoch + 1)
            loss_history['total'].append(loss.item())
            loss_history['physics'].append(10*phys.item())
            loss_history['bottom'].append(100*bot.item())
            loss_history['top'].append(100*top.item())
            loss_history['sides'].append(100*side.item())
            loss_history['initial'].append(100*init.item())
            progress.progress((epoch + 1) / epochs)
            status_text.text(f"Epoch {epoch+1}/{epochs} – Loss: {loss.item():.2e}")

    progress.progress(1.0)
    status_text.text("Training complete!")
    return model, loss_history

# --------------------------------------------------------------
#  Evaluation & Save
# --------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def evaluate_model(_model, times, Lx, Ly, _hash):
    x = torch.linspace(0, Lx, 50)
    y = torch.linspace(0, Ly, 50)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    c_preds = []
    for t_val in times:
        t = torch.full((X.numel(), 1), t_val)
        c = _model(X.reshape(-1,1), Y.reshape(-1,1), t).detach().numpy()
        c_preds.append(c.reshape(50,50).T)
    return X.numpy(), Y.numpy(), c_preds

@st.cache_data(ttl=3600, show_spinner=False)
def generate_and_save_solution(_model, times, param_set, output_dir, _hash):
    X, Y, c_preds = evaluate_model(_model, times, param_set['Lx'], param_set['Ly'], _hash)
    solution = {
        'params': param_set,
        'X': X, 'Y': Y, 'c_preds': c_preds, 'times': times,
        'orientation_note': 'c_preds: rows=y, cols=x'
    }
    Ly = param_set['Ly']
    C_bottom = param_set['C_bottom']
    fn = os.path.join(output_dir, f"solution_ni_soft_ly_{Ly:.1f}_c_{C_bottom:.1e}.pkl")
    with open(fn, 'wb') as f: pickle.dump(solution, f)
    return fn, solution

# --------------------------------------------------------------
#  VTK Export (unchanged)
# --------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def generate_vts_time_series(solution, output_dir, _hash):
    Lx, Ly = solution['params']['Lx'], solution['params']['Ly']
    C_bottom = solution['params']['C_bottom']
    times = solution['times']
    vts_files = []
    nx, ny = 50, 50
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    z = np.zeros((nx, ny))
    X, Y = np.meshgrid(x, y, indexing='ij')
    points = np.stack([X.ravel(), Y.ravel(), z.ravel()], axis=1)

    for t_idx, t_val in enumerate(times):
        c_xy = solution['c_preds'][t_idx].T
        grid = pv.StructuredGrid()
        grid.points = points
        grid.dimensions = (nx, ny, 1)
        grid.point_data['Ni_Concentration'] = c_xy.ravel()
        vts_fn = os.path.join(output_dir, f'ni_conc_ly_{Ly:.1f}_c_{C_bottom:.1e}_t_{t_val:.1f}.vts')
        grid.save(vts_fn)
        vts_files.append((t_val, vts_fn))

    pvd_fn = os.path.join(output_dir, f'ni_time_series_ly_{Ly:.1f}_c_{C_bottom:.1e}.pvd')
    with open(pvd_fn, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1">\n')
        f.write(' <Collection>\n')
        for t, vts in vts_files:
            f.write(f'  <DataSet timestep="{t}" file="{os.path.basename(vts)}"/>\n')
        f.write(' </Collection>\n</VTKFile>')
    return vts_files, pvd_fn

@st.cache_data(ttl=3600, show_spinner=False)
def generate_vtu_time_series(solution, output_dir, _hash):
    Lx, Ly = solution['params']['Lx'], solution['params']['Ly']
    C_bottom = solution['params']['C_bottom']
    times = solution['times']
    nx, ny = 50, 50
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    z = np.zeros((nx, ny))
    X, Y = np.meshgrid(x, y, indexing='ij')
    points = np.stack([X.ravel(), Y.ravel(), z.ravel()], axis=1)
    cells = []
    cell_types = []
    for j in range(ny-1):
        for i in range(nx-1):
            idx = i + j*nx
            cells.extend([4, idx, idx+1, idx+nx+1, idx+nx])
            cell_types.append(pv.CellType.QUAD)
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    for t_idx, t_val in enumerate(times):
        c_xy = solution['c_preds'][t_idx].T
        grid.point_data[f'Ni_t{t_val:.1f}'] = c_xy.ravel()
    vtu_fn = os.path.join(output_dir, f'ni_time_series_ly_{Ly:.1f}_c_{C_bottom:.1e}.vtu')
    grid.save(vtu_fn)
    return vtu_fn

# --------------------------------------------------------------
#  ZIP & Download
# --------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def create_zip_file(_files, output_dir, _hash, Ly, C_bottom):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in _files:
            if os.path.exists(f):
                z.write(f, os.path.basename(f))
    zip_fn = os.path.join(output_dir, f'ni_pinn_ly_{Ly:.1f}_c_{C_bottom:.1e}.zip')
    with open(zip_fn, 'wb') as f:
        f.write(zip_buffer.getvalue())
    return zip_fn

@st.cache_data(ttl=3600, show_spinner=False)
def get_file_bytes(fp):
    return open(fp, 'rb').read() if os.path.exists(fp) else None

# --------------------------------------------------------------
#  Main App
# --------------------------------------------------------------
def main():
    st.title("2D PINN: Ni Self-Diffusion (Soft + Gradient Viz)")

    Ly = st.number_input("Domain height Ly (μm)", 30.0, 90.0, 30.0, 1.0)
    D = st.number_input("D (μm²/s)", value=0.0054, format="%.5f")
    c_bottom = st.number_input("Bottom Ni conc. (mol/cc)", 1e-4, 6e-4, 1e-4, 1e-5, format="%.1e")

    key = get_cache_key(Ly, c_bottom, D, epochs, lr)

    if 'solution' not in st.session_state:
        st.session_state.solution = None
        st.session_state.files = {}

    if st.button("Run Simulation"):
        with st.spinner("Training..."):
            model, hist = train_model(D, Lx, Ly, T_max, c_bottom, epochs, lr, OUTPUT_DIR, key)
            times = np.linspace(0, T_max, 50)
            param_set = {'D': D, 'Lx': Lx, 'Ly': Ly, 't_max': T_max, 'C_bottom': c_bottom}
            sol_file, sol = generate_and_save_solution(model, times, param_set, OUTPUT_DIR, key)

            loss_plot = plot_losses(hist, OUTPUT_DIR, key, Ly, c_bottom)
            conc_plot = plot_2d_profiles(sol, -1, OUTPUT_DIR, key)
            vert_plot = plot_vertical_profile(sol, OUTPUT_DIR, key)
            grad_plot = plot_gradient_magnitude(sol, -1, OUTPUT_DIR, key)
            vts_files, pvd_file = generate_vts_time_series(sol, OUTPUT_DIR, key)
            vtu_file = generate_vtu_time_series(sol, OUTPUT_DIR, key)

            st.session_state.solution = sol
            st.session_state.files = {
                'solution': sol_file, 'loss': loss_plot, 'conc': conc_plot,
                'vert': vert_plot, 'grad': grad_plot,
                'vts_files': vts_files, 'pvd': pvd_file, 'vtu': vtu_file
            }
            st.success("Done!")

    if st.session_state.solution:
        sol = st.session_state.solution
        f = st.session_state.files

        st.subheader("Training Loss")
        st.image(f['loss'])

        st.subheader("Boundary Check")
        bc = validate_boundary_conditions(sol)
        st.metric("BCs", "Pass" if bc['valid'] else "Fail", f"{len(bc['details'])} issues")
        with st.expander("Details"):
            for d in bc['details']: st.caption(d)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Final Concentration")
            st.image(f['conc'])
        with col2:
            st.subheader("|∇c| Gradient Map")
            st.image(f['grad'])

        st.subheader("Vertical Profile (Center)")
        st.image(f['vert'])

        st.subheader("Downloads")
        for label, key in [
            ("Solution .pkl", 'solution'),
            ("Loss Plot", 'loss'),
            ("2D Conc.", 'conc'),
            ("Gradient Map", 'grad'),
            ("Vertical Profile", 'vert')
        ]:
            data = get_file_bytes(f[key])
            if data:
                st.download_button(label, data, os.path.basename(f[key]))

        st.subheader("VTK Time Series")
        if st.button("Download .pvd + .vts"):
            pvd_data = get_file_bytes(f['pvd'])
            st.download_button("PVD", pvd_data, os.path.basename(f['pvd']))
        if st.button("Download .vtu"):
            vtu_data = get_file_bytes(f['vtu'])
            st.download_button("VTU", vtu_data, os.path.basename(f['vtu']))

        if st.button("Download ZIP"):
            zip_fn = create_zip_file([
                f['solution'], f['loss'], f['conc'], f['vert'], f['grad'],
                f['pvd'], f['vtu']
            ] + [v[1] for v in f['vts_files']], OUTPUT_DIR, key, Ly, c_bottom)
            zip_data = get_file_bytes(zip_fn)
            st.download_button("All Files (.zip)", zip_data, os.path.basename(zip_fn))

if __name__ == "__main__":
    main()
