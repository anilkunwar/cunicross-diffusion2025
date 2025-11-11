import os
import pickle
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn as nn
import re
from matplotlib.colors import Normalize, LogNorm
from io import BytesIO
from scipy.interpolate import RegularGridInterpolator

# ----------------------------------------------------------------------
# Matplotlib style
# ----------------------------------------------------------------------
mpl.rcParams.update({
    'font.family': 'Arial', 'font.size': 12,
    'axes.linewidth': 1.5, 'xtick.major.width': 1.5, 'ytick.major.width': 1.5,
    'axes.titlesize': 14, 'axes.labelsize': 12, 'legend.fontsize': 10,
    'figure.dpi': 300, 'legend.frameon': True, 'legend.framealpha': 0.8,
    'grid.linestyle': '--', 'grid.alpha': 0.3
})

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTION_DIR = os.path.join(SCRIPT_DIR, "pinn_solutions")
FIGURE_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Load solutions
# ----------------------------------------------------------------------
@st.cache_data
def load_solutions(solution_dir):
    solutions, params_list, load_logs = [], [], []
    lys, c_cus, c_nis = [], [], []
    for fname in os.listdir(solution_dir):
        if not fname.endswith(".pkl"):
            continue
        path = os.path.join(solution_dir, fname)
        try:
            with open(path, "rb") as f:
                sol = pickle.load(f)
            req = ['params', 'X', 'Y', 'c1_preds', 'c2_preds', 'times']
            if not all(k in sol for k in req):
                raise ValueError("Missing keys")

            p = sol['params']
            param_tuple = (p['Ly'], p['C_Cu'], p['C_Ni'])
            sol['filename'] = fname
            solutions.append(sol)
            params_list.append(param_tuple)
            lys.append(p['Ly'])
            c_cus.append(p['C_Cu'])
            c_nis.append(p['C_Ni'])
            load_logs.append(f"{fname}: OK")

        except Exception as e:
            load_logs.append(f"{fname}: {e}")
    load_logs.append(f"Loaded {len(solutions)} solutions.")
    return solutions, params_list, lys, c_cus, c_nis, load_logs


# ----------------------------------------------------------------------
# Attention-based interpolator
# ----------------------------------------------------------------------
class MultiParamAttentionInterpolator(nn.Module):
    def __init__(self, sigma=0.2, num_heads=4, d_head=8):
        super().__init__()
        self.sigma = sigma
        self.num_heads = num_heads
        self.d_head = d_head
        self.W_q = nn.Linear(3, self.num_heads * self.d_head)
        self.W_k = nn.Linear(3, self.num_heads * self.d_head)

    def forward(self, solutions, params_list, ly_target, c_cu_target, c_ni_target):
        lys = np.array([p[0] for p in params_list])
        c_cus = np.array([p[1] for p in params_list])
        c_nis = np.array([p[2] for p in params_list])

        ly_norm = (lys - 30.0) / (120.0 - 30.0)
        c_cu_norm = c_cus / 2.9e-3
        c_ni_norm = c_nis / 1.8e-3

        tgt_ly_norm = (ly_target - 30.0) / (120.0 - 30.0)
        tgt_c_cu_norm = c_cu_target / 2.9e-3
        tgt_c_ni_norm = c_ni_target / 1.8e-3

        params_tensor = torch.tensor(np.stack([ly_norm, c_cu_norm, c_ni_norm], axis=1), dtype=torch.float32)
        target_tensor = torch.tensor([[tgt_ly_norm, tgt_c_cu_norm, tgt_c_ni_norm]], dtype=torch.float32)

        Q = self.W_q(target_tensor).view(1, self.num_heads, self.d_head)
        K = self.W_k(params_tensor).view(-1, self.num_heads, self.d_head)

        attn = torch.einsum('nhd,mhd->nmh', K, Q) / np.sqrt(self.d_head)
        attn_w = torch.softmax(attn, dim=0).mean(dim=2).squeeze(1)

        dist = torch.sqrt(
            ((torch.tensor(ly_norm) - tgt_ly_norm) / self.sigma) ** 2 +
            ((torch.tensor(c_cu_norm) - tgt_c_cu_norm) / self.sigma) ** 2 +
            ((torch.tensor(c_ni_norm) - tgt_c_ni_norm) / self.sigma) ** 2
        )
        spatial_w = torch.exp(-dist ** 2 / 2)
        spatial_w = spatial_w / (spatial_w.sum() + 1e-12)

        w = attn_w * spatial_w
        w = w / (w.sum() + 1e-12)

        return self._physics_aware_interpolation(solutions, w.detach().numpy(),
                                                ly_target, c_cu_target, c_ni_target)

    def _physics_aware_interpolation(self, solutions, weights,
                                     ly_target, c_cu_target, c_ni_target):
        Lx = solutions[0]['params']['Lx']
        x = np.linspace(0, Lx, 50)
        y = np.linspace(0, ly_target, 50)
        times = np.linspace(0, 200.0, 50)
        X, Y = np.meshgrid(x, y, indexing='ij')
        c1 = np.zeros((len(times), 50, 50))
        c2 = np.zeros((len(times), 50, 50))

        for t_idx, t in enumerate(times):
            for sol, w in zip(solutions, weights):
                src_times = sol['times']
                t_src = min(int(np.round(t / src_times[-1] * (len(src_times) - 1))), len(src_times) - 1)
                scale = ly_target / sol['params']['Ly']
                Ysrc = sol['Y'][0, :] * scale
                try:
                    interp_c1 = RegularGridInterpolator((sol['X'][:, 0], Ysrc), sol['c1_preds'][t_src],
                                                        method='linear', bounds_error=False, fill_value=0)
                    interp_c2 = RegularGridInterpolator((sol['X'][:, 0], Ysrc), sol['c2_preds'][t_src],
                                                        method='linear', bounds_error=False, fill_value=0)
                    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
                    c1[t_idx] += w * interp_c1(pts).reshape(50, 50)
                    c2[t_idx] += w * interp_c2(pts).reshape(50, 50)
                except:
                    continue

        c1[:, :, 0] = c_cu_target
        c2[:, :, -1] = c_ni_target
        param_set = solutions[0]['params'].copy()
        param_set.update({'Ly': ly_target, 'C_Cu': c_cu_target, 'C_Ni': c_ni_target})
        return {'params': param_set, 'X': X, 'Y': Y, 'times': times,
                'c1_preds': list(c1), 'c2_preds': list(c2), 'interpolated': True}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def get_center_conc(solution, ly_fraction=0.5):
    Lx, Ly = solution['params']['Lx'], solution['params']['Ly']
    ix = np.argmin(np.abs(solution['X'][:, 0] - Lx / 2))
    iy = np.argmin(np.abs(solution['Y'][0, :] - Ly * ly_fraction))
    cu = np.array([c1[ix, iy] for c1 in solution['c1_preds']])
    ni = np.array([c2[ix, iy] for c2 in solution['c2_preds']])
    return cu, ni


# ----------------------------------------------------------------------
# Sunburst matrices
# ----------------------------------------------------------------------
LY_SPOKES = [30, 40, 50, 60, 70, 80, 90, 100]
N_TIME = 50

def build_sunburst_matrices(solutions, params_list, interpolator, ccu, cni, ly_frac):
    cu_mat = np.zeros((N_TIME, len(LY_SPOKES)))
    ni_mat = np.zeros((N_TIME, len(LY_SPOKES)))
    prog = st.progress(0)
    for j, ly in enumerate(LY_SPOKES):
        sol = interpolator(solutions, params_list, ly, ccu, cni)
        cu, ni = get_center_conc(sol, ly_frac)
        cu_mat[:, j] = cu
        ni_mat[:, j] = ni
        prog.progress((j + 1) / len(LY_SPOKES))
    prog.empty()
    return cu_mat, ni_mat


# ----------------------------------------------------------------------
# Sunburst plot
# ----------------------------------------------------------------------
def build_sunburst_matrices(sols, params, interpolator, c_cu_target, c_ni_target, ly_fraction):
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='polar'))
    theta_edges = np.linspace(0, 2 * np.pi, len(LY_SPOKES) + 1)
    r_edges = np.linspace(0, 1, N_TIME + 1)
    if ly_dir == "top→bottom":
        r_edges = r_edges[::-1]

    Theta, R = np.meshgrid(theta_edges, r_edges)
    norm = LogNorm(vmin=max(vmin, 1e-9), vmax=vmax) if log_scale else Normalize(vmin=vmin, vmax=vmax)
    im = ax.pcolormesh(Theta, R, data, cmap=cmap, norm=norm, shading='auto')

    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    ax.set_xticks(theta_centers)
    ax.set_xticklabels([f"{ly}" for ly in LY_SPOKES], fontsize=13, fontweight='bold')

    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(['0', '50', '100', '150', '200'], fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(True, color='w', linewidth=1.2, alpha=0.7)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=25)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.08)
    cbar.set_label('Concentration (mol/cc)', fontsize=13)

    plt.tight_layout()
    png = os.path.join(FIGURE_DIR, f"{fname}.png")
    plt.savefig(png, dpi=300, bbox_inches='tight')
    plt.close()
    return fig, png


# ----------------------------------------------------------------------
# Radar chart
# ----------------------------------------------------------------------
def plot_radar(cu_row, ni_row, t_val, fname):
    angles = np.linspace(0, 2 * np.pi, len(LY_SPOKES), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection='polar'))
    ax.plot(angles, np.concatenate([cu_row, [cu_row[0]]]), 'o-', linewidth=2, label='Cu')
    ax.fill(angles, np.concatenate([cu_row, [cu_row[0]]]), alpha=0.25)
    ax.plot(angles, np.concatenate([ni_row, [ni_row[0]]]), 's--', linewidth=2, label='Ni')
    ax.fill(angles, np.concatenate([ni_row, [ni_row[0]]]), alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"{ly}" for ly in LY_SPOKES], fontsize=12)
    ax.set_ylim(0, max(cu_row.max(), ni_row.max()) * 1.1)
    ax.set_title(f"t = {t_val:.1f} s", fontsize=15, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax.grid(True)
    return fig


# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Sunburst + Radar Concentration Visualizer", layout="wide")
    st.title("Sunburst & Radar Visualization of Cu/Ni Concentration")

    st.markdown("""
    ### 🔍 Physical Meaning
    - **Spokes (Angular Axis):** Represent different *film thicknesses (Ly)* — each spoke corresponds to a distinct Ly value.  
    - **Radial Length (Radius):** Represents *time evolution* of the center-point concentration.  
    - The **Sunburst chart** thus shows how concentration evolves with time (radially) for various Ly values (angularly).  
    - The **Radar chart** is a snapshot at a specific time — comparing concentrations across Ly values.  
    """)

    sols, params, _, _, _, logs = load_solutions(SOLUTION_DIR)
    with st.expander("Load log"):
        for l in logs: st.write(l)
    if not sols:
        st.stop()

    interpolator = MultiParamAttentionInterpolator()

    st.sidebar.header("⚙️ Controls")
    cmap = st.sidebar.selectbox("Colormap", ["plasma", "viridis", "inferno", "magma", "cividis"])
    log_scale = st.sidebar.selectbox("Scale Type", ["Linear", "Logarithmic"]) == "Logarithmic"
    c_cu_target = st.sidebar.number_input("Boundary Cu Concentration (mol/cc)", 1e-6, 5e-3, 1e-3, format="%.1e")
    c_ni_target = st.sidebar.number_input("Boundary Ni Concentration (mol/cc)", 1e-6, 5e-3, 1e-4, format="%.1e")
    ly_center_choice = st.sidebar.selectbox("Center Sampling (Ly fraction)", {"Ly/2": 0.5, "Ly/3": 1/3, "Ly/4": 1/4})
    ly_dir = st.sidebar.radio("Ly Direction", ["top→bottom", "bottom→top"])

    cu_mat, ni_mat = build_sunburst_matrices(
        sols, params, interpolator, c_cu_target, c_ni_target, ly_center_choice
    )



    col1, col2 = st.columns(2)
    with col1:
        fig_cu, _ = plot_sunburst(cu_mat, "Cu Concentration (Center)", cmap,
                                  0, c_cu_target, log_scale, ly_dir, "sunburst_cu")
        st.pyplot(fig_cu)
    with col2:
        fig_ni, _ = plot_sunburst(ni_mat, "Ni Concentration (Center)", cmap,
                                  0, c_ni_target, log_scale, ly_dir, "sunburst_ni")
        st.pyplot(fig_ni)

    st.divider()
    st.subheader("📈 Time-dependent Radar Chart")

    t_idx = st.slider("Select Time Index", 0, N_TIME - 1, N_TIME // 2)
    t_val = np.linspace(0, 200, N_TIME)[t_idx]
    cu_row = cu_mat[t_idx]
    ni_row = ni_mat[t_idx]
    fig_radar = plot_radar(cu_row, ni_row, t_val, f"radar_t{t_val:.0f}")
    st.pyplot(fig_radar)

    st.caption("Note: x=30 µm (fixed), Ly varies around the circle.")


if __name__ == "__main__":
    main()
