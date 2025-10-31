import os
import pickle
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import matplotlib as mpl
import torch
import torch.nn as nn
import re
from matplotlib.colors import Normalize
from io import BytesIO

# -----------------------------
# Matplotlib Publication Style
# -----------------------------
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['legend.frameon'] = True
mpl.rcParams['legend.framealpha'] = 0.8
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.alpha'] = 0.3

SOLUTION_DIR = os.path.join(os.path.dirname(__file__), "pinn_solutions")
os.makedirs(SOLUTION_DIR, exist_ok=True)

# All available colormaps
COLORMAPS = sorted([c for c in plt.colormaps() if not c.endswith('_r')])

# -----------------------------
# Load Solutions
# -----------------------------
@st.cache_data
def load_solutions(solution_dir):
    solutions = []
    params_list = []
    load_logs = []
    lys, c_cus, c_nis = [], [], []
    for fname in os.listdir(solution_dir):
        if fname.endswith(".pkl"):
            path = os.path.join(solution_dir, fname)
            try:
                with open(path, "rb") as f:
                    sol = pickle.load(f)
                required = ['params', 'X', 'Y', 'c1_preds', 'c2_preds', 'times']
                if not all(k in sol for k in required):
                    load_logs.append(f"{fname}: Missing keys")
                    continue
                if np.any(np.isnan(sol['c1_preds'])) or np.any(np.isnan(sol['c2_preds'])):
                    load_logs.append(f"{fname}: Contains NaN")
                    continue

                solutions.append(sol)
                p = sol['params']
                param_tuple = (p['Ly'], p['C_Cu'], p['C_Ni'])
                params_list.append(param_tuple)
                lys.append(p['Ly'])
                c_cus.append(p['C_Cu'])
                c_nis.append(p['C_Ni'])

                # Parse diffusion type
                match = re.match(r"solution_(\w+)_ly_([\d.]+)_tmax_([\d.]+)\.pkl", fname)
                diff_type = 'crossdiffusion'
                if match:
                    raw = match.group(1).lower()
                    diff_type = {'cross': 'crossdiffusion', 'cu_self': 'cu_selfdiffusion', 'ni_self': 'ni_selfdiffusion'}.get(raw, diff_type)
                sol['diffusion_type'] = diff_type
                sol['filename'] = fname

                load_logs.append(f"{fname}: Loaded (Ly={p['Ly']:.1f}, C_Cu={p['C_Cu']:.1e}, C_Ni={p['C_Ni']:.1e})")
            except Exception as e:
                load_logs.append(f"{fname}: Error - {str(e)}")
    return solutions, params_list, lys, c_cus, c_nis, load_logs

# -----------------------------
# Attention Interpolator
# -----------------------------
class MultiParamAttentionInterpolator(nn.Module):
    def __init__(self, sigma=0.2, num_heads=4, d_head=8):
        super().__init__()
        self.sigma = sigma
        self.num_heads = num_heads
        self.d_head = d_head
        self.W_q = nn.Linear(3, num_heads * d_head)
        self.W_k = nn.Linear(3, num_heads * d_head)

    def forward(self, solutions, params_list, ly_target, c_cu_target, c_ni_target):
        if len(solutions) == 0:
            raise ValueError("No source solutions.")

        # Normalize parameters
        lys = np.array([p[0] for p in params_list])
        c_cus = np.array([p[1] for p in params_list])
        c_nis = np.array([p[2] for p in params_list])

        ly_norm = (lys - 30.0) / (120.0 - 30.0)
        c_cu_norm = c_cus / 2.9e-3
        c_ni_norm = c_nis / 1.8e-3
        target_ly_norm = (ly_target - 30.0) / (120.0 - 30.0)
        target_c_cu_norm = c_cu_target / 2.9e-3
        target_c_ni_norm = c_ni_target / 1.8e-3

        params_tensor = torch.tensor(np.stack([ly_norm, c_cu_norm, c_ni_norm], axis=1), dtype=torch.float32)
        target_tensor = torch.tensor([[target_ly_norm, target_c_cu_norm, target_c_ni_norm]], dtype=torch.float32)

        # Attention
        q = self.W_q(target_tensor).view(1, self.num_heads, self.d_head)
        k = self.W_k(params_tensor).view(len(params_list), self.num_heads, self.d_head)
        attn = torch.einsum('nhd,mhd->nmh', k, q) / np.sqrt(self.d_head)
        attn_weights = torch.softmax(attn, dim=0).mean(dim=2).squeeze(1)

        # Gaussian locality
        dists = torch.sqrt(
            ((params_tensor[:,0] - target_ly_norm)/self.sigma)**2 +
            ((params_tensor[:,1] - target_c_cu_norm)/self.sigma)**2 +
            ((params_tensor[:,2] - target_c_ni_norm)/self.sigma)**2
        )
        spatial_weights = torch.exp(-dists**2 / 2)
        spatial_weights /= spatial_weights.sum()

        # Combine
        weights = attn_weights * spatial_weights
        weights = weights / weights.sum()
        weights = weights.detach().cpu().numpy()

        return self._interpolate(solutions, weights, ly_target, c_cu_target, c_ni_target)

    def _interpolate(self, solutions, weights, ly_target, c_cu_target, c_ni_target):
        Lx = solutions[0]['params']['Lx']
        t_max = solutions[0]['params']['t_max']
        x_grid = np.linspace(0, Lx, 50)
        y_grid = np.linspace(0, ly_target, 50)
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        times = np.linspace(0, t_max, 50)

        c1_interp = np.zeros((len(times), 50, 50))
        c2_interp = np.zeros((len(times), 50, 50))

        for t_idx in range(len(times)):
            for sol, w in zip(solutions, weights):
                scale = ly_target / sol['params']['Ly']
                Ys = sol['Y'][0, :] * scale
                interp_c1 = RegularGridInterpolator((sol['X'][:,0], Ys), sol['c1_preds'][t_idx], method='linear', bounds_error=False, fill_value=0)
                interp_c2 = RegularGridInterpolator((sol['X'][:,0], Ys), sol['c2_preds'][t_idx], method='linear', bounds_error=False, fill_value=0)
                pts = np.stack([X.flatten(), Y.flatten()], axis=1)
                c1_interp[t_idx] += w * interp_c1(pts).reshape(50, 50)
                c2_interp[t_idx] += w * interp_c2(pts).reshape(50, 50)

        # Enforce BCs
        c1_interp[:, :, 0] = c_cu_target
        c2_interp[:, :, -1] = c_ni_target

        param_copy = solutions[0]['params'].copy()
        param_copy.update({'Ly': ly_target, 'C_Cu': c_cu_target, 'C_Ni': c_ni_target})

        return {
            'params': param_copy,
            'X': X, 'Y': Y,
            'c1_preds': list(c1_interp),
            'c2_preds': list(c2_interp),
            'times': times,
            'interpolated': True,
            'attention_weights': weights.tolist()
        }

@st.cache_data
def load_and_interpolate_solution(solutions, params_list, ly_target, c_cu_target, c_ni_target, tol_ly=0.1, tol_c=1e-5):
    for sol, p in zip(solutions, params_list):
        if (abs(p[0] - ly_target) < tol_ly and
            abs(p[1] - c_cu_target) < tol_c and
            abs(p[2] - c_ni_target) < tol_c):
            sol['interpolated'] = False
            return sol
    interpolator = MultiParamAttentionInterpolator()
    return interpolator(solutions, params_list, ly_target, c_cu_target, c_ni_target)

# -----------------------------
# Flux & Uphill Detection
# -----------------------------
def compute_fluxes_and_grads(c1_preds, c2_preds, x_coords, y_coords, params):
    D11 = params.get('D11', 1.0)
    D12 = params.get('D12', 0.0)
    D21 = params.get('D21', 0.0)
    D22 = params.get('D22', 1.0)
    dy = y_coords[1] - y_coords[0]

    J1_list, J2_list, grad1_list, grad2_list = [], [], [], []
    for c1, c2 in zip(c1_preds, c2_preds):
        g1 = np.gradient(c1, dy, axis=0)
        g2 = np.gradient(c2, dy, axis=0)
        J1y = -(D11 * g1 + D12 * g2)
        J2y = -(D21 * g1 + D22 * g2)
        J1_list.append([None, J1y])
        J2_list.append([None, J2y])
        grad1_list.append(g1)
        grad2_list.append(g2)
    return J1_list, J2_list, grad1_list, grad2_list

def detect_uphill(solution, t_idx):
    J1y = solution['J1_preds'][t_idx][1]
    g1 = solution['grad_c1_y'][t_idx]
    J2y = solution['J2_preds'][t_idx][1]
    g2 = solution['grad_c2_y'][t_idx]

    prod_cu = J1y * g1
    prod_ni = J2y * g2
    uphill_cu = prod_cu > 0
    uphill_ni = prod_ni > 0

    pos_cu = np.where(uphill_cu, prod_cu, 0)
    pos_ni = np.where(uphill_ni, prod_ni, 0)

    max_cu = float(np.max(pos_cu)) if np.any(uphill_cu) else 0.0
    max_ni = float(np.max(pos_ni)) if np.any(uphill_ni) else 0.0
    frac_cu = np.count_nonzero(uphill_cu) / prod_cu.size
    frac_ni = np.count_nonzero(uphill_ni) / prod_ni.size
    avg_cu = np.mean(pos_cu[pos_cu > 0]) if np.any(uphill_cu) else 0.0
    avg_ni = np.mean(pos_ni[pos_ni > 0]) if np.any(uphill_ni) else 0.0
    total_cu = np.sum(pos_cu)
    total_ni = np.sum(pos_ni)

    return (uphill_cu, uphill_ni, pos_cu, pos_ni,
            max_cu, max_ni, frac_cu, frac_ni,
            avg_cu, avg_ni, total_cu, total_ni)

# -----------------------------
# Plotting Utilities
# -----------------------------
def fig_to_bytes(fig, fmt='png'):
    buf = BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches='tight', dpi=300)
    buf.seek(0)
    return buf

def plot_2d_concentration(solution, t_idx, cmap_cu='viridis', cmap_ni='plasma', vmin_cu=None, vmax_cu=None, vmin_ni=None, vmax_ni=None):
    x = solution['X'][:,0]
    y = solution['Y'][0,:]
    Lx, Ly = solution['params']['Lx'], solution['params']['Ly']
    c1 = solution['c1_preds'][t_idx]
    c2 = solution['c2_preds'][t_idx]
    t = solution['times'][t_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    im1 = ax1.imshow(c1, origin='lower', extent=[0, Lx, 0, Ly], cmap=cmap_cu, vmin=vmin_cu, vmax=vmax_cu)
    ax1.set_title(f'Cu @ t = {t:.1f} s')
    ax1.set_xlabel('x (μm)'); ax1.set_ylabel('y (μm)')
    plt.colorbar(im1, ax=ax1, label='mol/cc')

    im2 = ax2.imshow(c2, origin='lower', extent=[0, Lx, 0, Ly], cmap=cmap_ni, vmin=vmin_ni, vmax=vmax_ni)
    ax2.set_title(f'Ni @ t = {t:.1f} s')
    ax2.set_xlabel('x (μm)')
    plt.colorbar(im2, ax=ax2, label='mol/cc')

    status = "Interpolated" if solution.get('interpolated') else "Exact"
    fig.suptitle(f'2D Concentration Profiles ({status})')
    return fig

def plot_flux_vs_gradient(solution, t_idx):
    J1y = solution['J1_preds'][t_idx][1].flatten()
    g1 = solution['grad_c1_y'][t_idx].flatten()
    J2y = solution['J2_preds'][t_idx][1].flatten()
    g2 = solution['grad_c2_y'][t_idx].flatten()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.scatter(g1, J1y, s=10, alpha=0.6, label='All')
    ax1.scatter(g1[J1y*g1 > 0], J1y[J1y*g1 > 0], s=15, edgecolors='k', label='Uphill')
    ax1.set_xlabel('∇c₁'); ax1.set_ylabel('J₁'); ax1.set_title('Cu')
    ax1.legend(); ax1.grid(True)

    ax2.scatter(g2, J2y, s=10, alpha=0.6, label='All')
    ax2.scatter(g2[J2y*g2 > 0], J2y[J2y*g2 > 0], s=15, edgecolors='k', label='Uphill')
    ax2.set_xlabel('∇c₂'); ax2.set_ylabel('J₂'); ax2.set_title('Ni')
    ax2.legend(); ax2.grid(True)

    fig.suptitle('Flux vs Gradient')
    return fig

def plot_uphill_heatmap(solution, t_idx, cmap='viridis', vmin=None, vmax=None):
    x = solution['X'][:,0]; y = solution['Y'][0,:]
    (_, _, pos_cu, pos_ni, max_cu, max_ni, _, _, avg_cu, avg_ni, _, _) = detect_uphill(solution, t_idx)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    im1 = ax1.imshow(pos_cu, origin='lower', extent=[x[0], x[-1], y[0], y[-1]], cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.set_title(f'Cu Uphill (max={max_cu:.2e})'); ax1.set_xlabel('x'); ax1.set_ylabel('y')
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(pos_ni, origin='lower', extent=[x[0], x[-1], y[0], y[-1]], cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.set_title(f'Ni Uphill (max={max_ni:.2e})'); ax2.set_xlabel('x')
    plt.colorbar(im2, ax=ax2)

    fig.suptitle('Uphill Diffusion Heatmap (J·∇c > 0)')
    return fig

# -----------------------------
# Main App
# -----------------------------
def main():
    st.title("Attention-Interpolated Diffusion + Uphill Analysis")

    solutions, params_list, lys, c_cus, c_nis, logs = load_solutions(SOLUTION_DIR)
    with st.expander("Load Log"):
        for log in logs: st.write(log)

    if not solutions:
        st.error("No valid solutions found.")
        return

    lys = sorted(set(lys)); c_cus = sorted(set(c_cus)); c_nis = sorted(set(c_nis))

    st.subheader("Parameter Selection")
    ly = st.selectbox("Ly (μm)", lys, format_func=lambda x: f"{x:.1f}")
    c_cu = st.selectbox("C_Cu", c_cus, format_func=lambda x: f"{x:.1e}")
    c_ni = st.selectbox("C_Ni", c_nis, format_func=lambda x: f"{x:.1e}")

    use_custom = st.checkbox("Use Custom Parameters")
    if use_custom:
        ly = st.number_input("Ly", 30.0, 120.0, ly, 0.1)
        c_cu = st.number_input("C_Cu", 0.0, 2.9e-3, c_cu, 1e-5)
        c_ni = st.number_input("C_Ni", 0.0, 1.8e-3, c_ni, 1e-5)

    try:
        sol = load_and_interpolate_solution(solutions, params_list, ly, c_cu, c_ni)
    except Exception as e:
        st.error(f"Interpolation failed: {e}")
        return

    # Compute fluxes if missing
    if 'J1_preds' not in sol:
        J1, J2, g1, g2 = compute_fluxes_and_grads(sol['c1_preds'], sol['c2_preds'], sol['X'][:,0], sol['Y'][0,:], sol['params'])
        sol.update({'J1_preds': J1, 'J2_preds': J2, 'grad_c1_y': g1, 'grad_c2_y': g2})

    st.write(f"**Status**: {'Interpolated' if sol.get('interpolated') else 'Exact'} Solution")

    t_idx = st.slider("Time Index", 0, len(sol['times'])-1, len(sol['times'])-1)
    t_val = sol['times'][t_idx]

    col1, col2 = st.columns(2)
    with col1:
        st.write("**2D Concentrations**")
        fig = plot_2d_concentration(sol, t_idx)
        st.pyplot(fig)
        st.download_button("Download PNG", fig_to_bytes(fig), "conc_2d.png", "image/png")

    with col2:
        st.write("**Flux vs Gradient**")
        fig = plot_flux_vs_gradient(sol, t_idx)
        st.pyplot(fig)
        st.download_button("Download PNG", fig_to_bytes(fig), "flux_grad.png", "image/png")

    st.write("**Uphill Heatmap**")
    fig = plot_uphill_heatmap(sol, t_idx)
    st.pyplot(fig)
    st.download_button("Download PNG", fig_to_bytes(fig), "uphill.png", "image/png")

    # Metrics
    (_, _, _, _, max_cu, max_ni, frac_cu, frac_ni, avg_cu, avg_ni, _, _) = detect_uphill(sol, t_idx)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Max Uphill (Cu)", f"{max_cu:.2e}")
        st.metric("Uphill Fraction (Cu)", f"{frac_cu:.1%}")
    with col2:
        st.metric("Max Uphill (Ni)", f"{max_ni:.2e}")
        st.metric("Uphill Fraction (Ni)", f"{frac_ni:.1%}")

if __name__ == "__main__":
    main()
