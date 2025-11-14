# --------------------------------------------------------------
#  Cu-only PINN – diffusion front now MOVES
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
#  Physical parameters – **GRADIENT IS THE KEY**
# --------------------------------------------------------------
C_CU_BOTTOM = 1.59e-3      # Cu-rich at the *bottom* (y = 0)
C_CU_TOP    = 0.0          # Cu-poor at the *top*    (y = Ly)
Ly = 50.0                  # domain height (µm)
Lx = 60.0                  # domain width  (µm)
D_CU = 0.006               # diffusion coefficient (µm²/s)
T_max = 200.0
epochs = 5000
lr = 1e-3

torch.manual_seed(42)
np.random.seed(42)

def get_cache_key(*args):
    return hashlib.md5("_".join(str(a) for a in args).encode()).hexdigest()

# --------------------------------------------------------------
#  Simple MLP (soft constraints)
# --------------------------------------------------------------
class CuPINN(nn.Module):
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
#  PDE helpers
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

def bc_bottom(model):
    n = 200
    x = torch.rand(n, 1, requires_grad=True) * model.Lx
    y = torch.zeros(n, 1, requires_grad=True)
    t = torch.rand(n, 1, requires_grad=True) * model.T_max
    c = model(x, y, t)
    return torch.mean((c - model.C_bottom)**2)

def bc_top(model):
    n = 200
    x = torch.rand(n, 1, requires_grad=True) * model.Lx
    y = torch.full((n, 1), model.Ly, requires_grad=True)
    t = torch.rand(n, 1, requires_grad=True) * model.T_max
    c = model(x, y, t)
    return torch.mean((c - C_CU_TOP)**2)

def bc_sides(model):
    n = 200
    # left
    xl = torch.zeros(n, 1, requires_grad=True)
    yl = torch.rand(n, 1, requires_grad=True) * model.Ly
    tl = torch.rand(n, 1, requires_grad=True) * model.T_max
    cl = model(xl, yl, tl)
    # right
    xr = torch.full((n, 1), model.Lx, requires_grad=True)
    yr = torch.rand(n, 1, requires_grad=True) * model.Ly
    tr = torch.rand(n, 1, requires_grad=True) * model.T_max
    cr = model(xr, yr, tr)

    grad_l = torch.autograd.grad(cl, xl, grad_outputs=torch.ones_like(cl),
                                 create_graph=True, retain_graph=True)[0]
    grad_r = torch.autograd.grad(cr, xr, grad_outputs=torch.ones_like(cr),
                                 create_graph=True, retain_graph=True)[0]
    return torch.mean(grad_l**2) + torch.mean(grad_r**2)

def initial_loss(model):
    n = 500
    x = torch.rand(n, 1, requires_grad=True) * model.Lx
    y = torch.rand(n, 1, requires_grad=True) * model.Ly
    t = torch.zeros(n, 1, requires_grad=True)
    return torch.mean(model(x, y, t)**2)

# --------------------------------------------------------------
#  Validation (checks that BCs are satisfied)
# --------------------------------------------------------------
def validate(solution, tol=1e-6):
    c = solution['c_preds'][-1]                 # final time step
    bottom = np.mean(c[:, 0])
    top    = np.mean(c[:, -1])
    left   = np.mean(np.abs(c[1, :] - c[0, :]))
    right  = np.mean(np.abs(c[-1, :] - c[-2, :]))
    ok = (abs(bottom - C_CU_BOTTOM) < tol and
          abs(top    - C_CU_TOP)    < tol and
          left < tol and right < tol)
    return ok, [f"Bottom {bottom:.2e} (target {C_CU_BOTTOM})",
                f"Top    {top:.2e} (target {C_CU_TOP})",
                f"Left flux {left:.2e}", f"Right flux {right:.2e}"]

# --------------------------------------------------------------
#  Plotting helpers
# --------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def plot_loss(hist, out_dir, _h):
    plt.figure(figsize=(10,6))
    e = np.array(hist['epochs'])
    plt.plot(e, hist['total'], label='Total', lw=2, color='k')
    plt.plot(e, hist['physics'], '--', label='Physics', lw=1.5, color='b')
    plt.plot(e, hist['bottom'], '-.', label='Bottom BC', lw=1.5, color='r')
    plt.plot(e, hist['top'], ':', label='Top BC', lw=1.5, color='g')
    plt.plot(e, hist['sides'], '-', label='Sides BC', lw=1.5, color='purple')
    plt.plot(e, hist['initial'], '--', label='IC', lw=1.5, color='orange')
    plt.yscale('log'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Training loss (Cu self-diffusion)'); plt.grid(True, which='both', ls='--', alpha=0.7)
    plt.legend(); plt.tight_layout()
    fn = os.path.join(out_dir, 'loss_cu.png')
    plt.savefig(fn, dpi=300, bbox_inches='tight'); plt.close()
    return fn

@st.cache_data(ttl=3600, show_spinner=False)
def plot_2d(solution, tidx, out_dir, _h):
    t = solution['times'][tidx]
    c = solution['c_preds'][tidx]
    plt.figure(figsize=(6,5))
    im = plt.imshow(c, origin='lower', extent=[0, Lx, 0, Ly],
                    cmap='viridis', vmin=0, vmax=C_CU_BOTTOM)
    plt.title(f'Cu concentration (t={t:.1f} s)')
    plt.xlabel('x (µm)'); plt.ylabel('y (µm)'); plt.colorbar(im, label='mol/cc')
    plt.grid(alpha=0.3); plt.tight_layout()
    fn = os.path.join(out_dir, f'cu_2d_t{t:.1f}.png')
    plt.savefig(fn, dpi=300, bbox_inches='tight'); plt.close()
    return fn

# --------------------------------------------------------------
#  NEW: vertical profile (gradient) plot
# --------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def plot_vertical_profiles(solution, out_dir, _h):
    times = solution['times']
    y = np.linspace(0, Ly, 50)
    plt.figure(figsize=(8,5))
    for i, t in enumerate(times[::10]):          # every 10-th frame
        c = solution['c_preds'][i][:, 25]        # middle column (x≈Lx/2)
        plt.plot(c, y, label=f't={t:.0f}s')
    plt.xlabel('Cu concentration (mol/cc)'); plt.ylabel('y (µm)')
    plt.title('Vertical concentration profile (center line)')
    plt.legend(); plt.grid(alpha=0.4); plt.tight_layout()
    fn = os.path.join(out_dir, 'cu_vertical_profiles.png')
    plt.savefig(fn, dpi=300, bbox_inches='tight'); plt.close()
    return fn

# --------------------------------------------------------------
#  Training
# --------------------------------------------------------------
@st.cache_resource(ttl=3600, show_spinner=False)
def train(_hash):
    model = CuPINN(D_CU, Lx, Ly, T_max, C_CU_BOTTOM)
    opt = optim.Adam(model.parameters(), lr=lr)

    xp = torch.rand(1000,1,requires_grad=True)*Lx
    yp = torch.rand(1000,1,requires_grad=True)*Ly
    tp = torch.rand(1000,1,requires_grad=True)*T_max

    hist = {'epochs':[], 'total':[], 'physics':[], 'bottom':[],
            'top':[], 'sides':[], 'initial':[]}
    prog = st.progress(0); stat = st.empty()

    for e in range(epochs):
        opt.zero_grad()
        Lphy = physics_loss(model, xp, yp, tp)
        Lbot = bc_bottom(model)
        Ltop = bc_top(model)
        Lsid = bc_sides(model)
        Lini = initial_loss(model)

        loss = 10*Lphy + 100*Lbot + 100*Ltop + 100*Lsid + 100*Lini
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (e+1) % 100 == 0:
            hist['epochs'].append(e+1)
            hist['total'].append(loss.item())
            hist['physics'].append(10*Lphy.item())
            hist['bottom'].append(100*Lbot.item())
            hist['top'].append(100*Ltop.item())
            hist['sides'].append(100*Lsid.item())
            hist['initial'].append(100*Lini.item())
            prog.progress((e+1)/epochs)
            stat.text(f'Epoch {e+1}/{epochs} – loss {loss.item():.2e}')

    prog.progress(1.0); stat.text('  ('Training finished')
    return model, hist

# --------------------------------------------------------------
#  Evaluation & saving
# --------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def evaluate(_model, times):
    x = torch.linspace(0, Lx, 50)
    y = torch.linspace(0, Ly, 50)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    c_preds = []
    for t in times:
        c = _model(X.reshape(-1,1), Y.reshape(-1,1),
                   torch.full((X.numel(),1), t)).detach().numpy()
        c_preds.append(c.reshape(50,50).T)      # (y,x) for imshow
    return X.numpy(), Y.numpy(), c_preds

@st.cache_data(ttl=3600, show_spinner=False)
def save_solution(_model, times, out_dir, _h):
    X, Y, c_preds = evaluate(_model, times)
    sol = {
        'params': {'D':D_CU, 'Lx':Lx, 'Ly':Ly, 't_max':T_max,
                   'C_bottom':C_CU_BOTTOM, 'C_top':C_CU_TOP},
        'X':X, 'Y':Y, 'c_preds':c_preds, 'times':times,
        'orientation_note':'rows = y, cols = x'
    }
    fn = os.path.join(out_dir, f'cu_solution_ly{Ly:.0f}.pkl')
    with open(fn, 'wb') as f: pickle.dump(sol, f)
    return fn, sol

# --------------------------------------------------------------
#  VTK export (unchanged, just renamed)
# --------------------------------------------------------------
# (copy-paste the generate_vts_time_series / generate_vtu_time_series
#  functions from your original script – they work unchanged)
# --------------------------------------------------------------
#  (for brevity they are omitted here; just keep the ones you already have)

# --------------------------------------------------------------
#  Streamlit UI
# --------------------------------------------------------------
def main():
    st.title('2D PINN – Cu Self-Diffusion (moving front)')

    # ---- show the gradient that drives diffusion ----
    st.markdown('### Diffusion driving gradient')
    st.write(f'**Bottom BC** (y = 0) : `{C_CU_BOTTOM:.2e}` mol/cc')
    st.write(f'**Top BC**    (y = Ly): `{C_CU_TOP:.2e}` mol/cc')
    st.write(f'**Gradient exists** → **Diffusion front will move**')

    key = get_cache_key(Ly, C_CU_BOTTOM, C_CU_TOP, epochs, lr)

    if st.button('Run Simulation'):
        with st.spinner('Training…'):
            model, loss_hist = train(key)
            times = np.linspace(0, T_max, 50)
            sol_file, solution = save_solution(model, times, OUTPUT_DIR, key)

            loss_plot   = plot_loss(loss_hist, OUTPUT_DIR, key)
            prof_plot   = plot_2d(solution, -1, OUTPUT_DIR, key)
            vert_plot   = plot_vertical_profiles(solution, OUTPUT_DIR, key)

            # ---- store in session for later display ----
            st.session_state.solution = solution
            st.session_state.files = {
                'solution': sol_file,
                'loss': loss_plot,
                'profile': prof_plot,
                'vertical': vert_plot
            }
            st.success('Done!')

    # ---- display results -------------------------------------------------
    if 'solution' in st.session_state:
        sol = st.session_state.solution
        f   = st.session_state.files

        st.subheader('Training loss')
        st.image(f['loss'])

        ok, msgs = validate(sol)
        st.subheader('Boundary-check')
        st.write('All BCs satisfied' if ok else 'Warning: BCs not perfect')
        for m in msgs: st.caption(m)

        st.subheader('2-D concentration (final time)')
        st.image(f['profile'])

        st.subheader('Vertical concentration profile (center line)')
        st.image(f['vertical'])

        # ---- downloads ----------------------------------------------------
        st.subheader('Downloads')
        for label, path in [('Solution .pkl', f['solution']),
                            ('Loss plot', f['loss']),
                            ('2-D final', f['profile']),
                            ('Vertical profiles', f['vertical'])]:
            data = open(path, 'rb').read()
            st.download_button(label, data, os.path.basename(path))

        # (VTK / ZIP sections can be added exactly as in your original script)

if __name__ == '__main__':
    main()
