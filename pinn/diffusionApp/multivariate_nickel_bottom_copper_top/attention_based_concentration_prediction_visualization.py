# sunburst_centerpoint_interpolation.py
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.interpolate import RegularGridInterpolator
from matplotlib.colors import Normalize
import streamlit as st
import re

# ----------------------------------------------------------------------
# 1. CONFIG – locate the .pkl folder next to this file
# ----------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PINN_DIR   = os.path.join(SCRIPT_DIR, "pinn_solutions")   # <-- same as first code
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "sunburst_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sunburst geometry
LY_VALUES     = [30, 40, 50, 60, 70, 80, 90, 100]   # 8 spokes
T_MAX         = 200.0
N_TIME_STEPS  = 50
N_LY          = len(LY_VALUES)

C_CU_TARGET = 1.0e-3
C_NI_TARGET = 1.0e-4

# ----------------------------------------------------------------------
# 2. LOAD SOLUTIONS – **exactly the same as the first script**
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Loading PINN solutions...")
def load_solutions(solution_dir):
    solutions = []
    params_list = []
    load_logs = []
    lys, c_cus, c_nis = [], [], []

    if not os.path.isdir(solution_dir):
        load_logs.append(f"Directory not found: {solution_dir}")
        st.error("\n".join(load_logs))
        return solutions, params_list, lys, c_cus, c_nis, load_logs

    files = [f for f in os.listdir(solution_dir) if f.endswith(".pkl")]
    load_logs.append(f"Found {len(files)} .pkl files.")

    for fname in files:
        path = os.path.join(solution_dir, fname)
        try:
            with open(path, "rb") as f:
                sol = pickle.load(f)

            req = ['params', 'X', 'Y', 'c1_preds', 'c2_preds', 'times']
            if not all(k in sol for k in req):
                load_logs.append(f"{fname}: missing keys")
                continue

            if np.any(np.isnan(sol['c1_preds'])) or np.any(np.isnan(sol['c2_preds'])):
                load_logs.append(f"{fname}: NaNs")
                continue

            # ---- parameter tuple (exactly as first code) ----
            p = sol['params']
            param_tuple = (p['Ly'], p['C_Cu'], p['C_Ni'])
            solutions.append(sol)
            params_list.append(param_tuple)
            lys.append(p['Ly'])
            c_cus.append(p['C_Cu'])
            c_nis.append(p['C_Ni'])

            # diffusion-type from filename (same logic)
            m = re.search(r"ly_([\d.]+)_ccu_([\deE.-]+)_cni_([\deE.-]+)", fname)
            diff_type = 'crossdiffusion'
            if m:
                ly, cu, ni = map(float, m.groups())
                if cu == 0 and ni > 0:   diff_type = 'ni_selfdiffusion'
                if ni == 0 and cu > 0:   diff_type = 'cu_selfdiffusion'
            sol['diffusion_type'] = diff_type

            load_logs.append(
                f"{fname}: Ly={p['Ly']:.1f} Cu={p['C_Cu']:.1e} Ni={p['C_Ni']:.1e}"
            )
        except Exception as e:
            load_logs.append(f"{fname}: {e}")

    load_logs.append(f"Loaded {len(solutions)} solutions.")
    return solutions, params_list, lys, c_cus, c_nis, load_logs


# ----------------------------------------------------------------------
# 3. ATTENTION INTERPOLATOR – **copy-paste from the first script**
# ----------------------------------------------------------------------
class MultiParamAttentionInterpolator(nn.Module):
    """Exact copy of the interpolator used in the first (working) app."""
    def __init__(self, sigma=0.2, num_heads=4, d_head=8):
        super().__init__()
        self.sigma = sigma
        self.num_heads = num_heads
        self.d_head = d_head
        self.W_q = nn.Linear(3, self.num_heads * self.d_head)
        self.W_k = nn.Linear(3, self.num_heads * self.d_head)

    def forward(self, solutions, params_list, ly_target, c_cu_target, c_ni_target):
        # ----- 1. min-max normalisation (exactly as first code) -----
        lys   = np.array([p[0] for p in params_list])
        c_cus = np.array([p[1] for p in params_list])
        c_nis = np.array([p[2] for p in params_list])

        ly_norm   = (lys   - 30.0) / (120.0 - 30.0)
        c_cu_norm = (c_cus - 0.0) / (2.9e-3 - 0.0)   # allows C_Cu = 0
        c_ni_norm = (c_nis - 0.0) / (1.8e-3 - 0.0)   # allows C_Ni = 0

        tgt_ly_norm   = (ly_target   - 30.0) / (120.0 - 30.0)
        tgt_c_cu_norm = (c_cu_target - 0.0) / (2.9e-3 - 0.0)
        tgt_c_ni_norm = (c_ni_target - 0.0) / (1.8e-3 - 0.0)

        # ----- 2. multi-head attention -----
        params_tensor = torch.tensor(np.stack([ly_norm, c_cu_norm, c_ni_norm], axis=1),
                                    dtype=torch.float32)               # [N,3]
        target_tensor = torch.tensor([[tgt_ly_norm, tgt_c_cu_norm, tgt_c_ni_norm]],
                                    dtype=torch.float32)               # [1,3]

        Q = self.W_q(target_tensor).view(1, self.num_heads, self.d_head)   # [1,H,D]
        K = self.W_k(params_tensor).view(-1, self.num_heads, self.d_head) # [N,H,D]

        attn_logits = torch.einsum('nhd,mhd->nmh', K, Q) / np.sqrt(self.d_head)
        attn_weights = torch.softmax(attn_logits, dim=0)          # [N,1,H]
        attn_weights = attn_weights.mean(dim=2).squeeze(1)        # [N]

        # ----- 3. Gaussian spatial weighting -----
        dist = torch.sqrt(
            ((torch.tensor(ly_norm)   - tgt_ly_norm)   / self.sigma)**2 +
            ((torch.tensor(c_cu_norm) - tgt_c_cu_norm) / self.sigma)**2 +
            ((torch.tensor(c_ni_norm) - tgt_c_ni_norm) / self.sigma)**2
        )
        spatial_weights = torch.exp(-dist**2 / 2)
        spatial_weights = spatial_weights / (spatial_weights.sum() + 1e-12)

        # ----- 4. combine & normalise -----
        combined = attn_weights * spatial_weights
        combined = combined / (combined.sum() + 1e-12)

        # ----- 5. physics-aware interpolation (identical to first code) -----
        return self._physics_aware_interpolation(
            solutions, combined.detach().numpy(),
            ly_target, c_cu_target, c_ni_target
        )

    # ------------------------------------------------------------------
    def _physics_aware_interpolation(self, solutions, weights,
                                     ly_target, c_cu_target, c_ni_target):
        Lx     = solutions[0]['params']['Lx']
        t_max  = solutions[0]['params']['t_max']
        x      = np.linspace(0, Lx, 50)
        y      = np.linspace(0, ly_target, 50)
        times  = np.linspace(0, min(t_max, T_MAX), N_TIME_STEPS)

        X, Y = np.meshgrid(x, y, indexing='ij')
        c1_interp = np.zeros((len(times), 50, 50))
        c2_interp = np.zeros((len(times), 50, 50))

        for t_idx, t in enumerate(times):
            # ---- map target time to *exact* index in each source solution ----
            for sol, w in zip(solutions, weights):
                src_times = sol['times']
                t_idx_src = min(
                    int(np.round(t / src_times[-1] * (len(src_times) - 1))),
                    len(src_times) - 1
                )
                scale = ly_target / sol['params']['Ly']
                Y_src = sol['Y'][0, :] * scale

                # ---- linear interpolation on the scaled source grid ----
                try:
                    interp_c1 = RegularGridInterpolator(
                        (sol['X'][:, 0], Y_src), sol['c1_preds'][t_idx_src],
                        method='linear', bounds_error=False, fill_value=0)
                    interp_c2 = RegularGridInterpolator(
                        (sol['X'][:, 0], Y_src), sol['c2_preds'][t_idx_src],
                        method='linear', bounds_error=False, fill_value=0)

                    pts = np.stack([X.flatten(), Y.flatten()], axis=1)
                    c1_interp[t_idx] += w * interp_c1(pts).reshape(50, 50)
                    c2_interp[t_idx] += w * interp_c2(pts).reshape(50, 50)
                except Exception:
                    # silent skip – same behaviour as the first script
                    continue

        # ---- enforce BCs (identical to first code) ----
        c1_interp[:, :, 0]  = c_cu_target      # Cu at y = 0
        c2_interp[:, :, -1] = c_ni_target      # Ni at y = Ly

        param_set = solutions[0]['params'].copy()
        param_set.update({'Ly': ly_target, 'C_Cu': c_cu_target, 'C_Ni': c_ni_target})

        return {
            'params': param_set,
            'X': X, 'Y': Y,
            'times': times,
            'c1_preds': list(c1_interp),
            'c2_preds': list(c2_interp),
            'interpolated': True,
            'attention_weights': weights.tolist()
        }


# ----------------------------------------------------------------------
# 4. CENTER-POINT EXTRACTION
# ----------------------------------------------------------------------
def get_center_concentration(solution):
    Lx, Ly = solution['params']['Lx'], solution['params']['Ly']
    x_center, y_center = Lx / 2, Ly / 2
    ix = np.argmin(np.abs(solution['X'][:, 0] - x_center))
    iy = np.argmin(np.abs(solution['Y'][0, :] - y_center))
    cu = [c1[ix, iy] for c1 in solution['c1_preds']]
    ni = [c2[ix, iy] for c2 in solution['c2_preds']]
    return np.array(cu), np.array(ni)


# ----------------------------------------------------------------------
# 5. BUILD SUNBURST MATRICES
# ----------------------------------------------------------------------
def build_sunburst_matrices(solutions, params_list, interpolator):
    cu_mat = np.zeros((N_TIME_STEPS, N_LY))
    ni_mat = np.zeros((N_TIME_STEPS, N_LY))

    prog = st.progress(0)
    for j, ly in enumerate(LY_VALUES):
        sol = interpolator(solutions, params_list,
                           ly, C_CU_TARGET, C_NI_TARGET)
        cu, ni = get_center_concentration(sol)
        cu_mat[:, j] = cu
        ni_mat[:, j] = ni
        prog.progress((j + 1) / N_LY)
    prog.empty()
    return cu_mat, ni_mat


# ----------------------------------------------------------------------
# 6. POLAR SUNBURST PLOT
# ----------------------------------------------------------------------
def plot_sunburst_polar(data, title, cmap, vmin, vmax, fname):
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # --- Grid: 50 time steps × 8 Ly values ---
    theta = np.linspace(0, 2*np.pi, N_LY, endpoint=False)  # 8 points, no repeat
    r     = np.linspace(0, 1, N_TIME_STEPS + 1)           # 51 edges → 50 cells
    Theta, R = np.meshgrid(theta, r)

    # --- C must be (50, 8) — one smaller than mesh in each direction ---
    C = data  # Shape: (50, 8)

    norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.pcolormesh(Theta, R, C, cmap=cmap, norm=norm, shading='auto')

    # --- Labels ---
    ax.set_xticks(theta)
    ax.set_xticklabels([f"{ly}" for ly in LY_VALUES], fontsize=14, fontweight='bold')

    r_labels = [0, 50, 100, 150, 200]
    r_ticks  = np.array(r_labels) / T_MAX
    ax.set_yticks(r_ticks)
    ax.set_yticklabels([f"{int(t)} s" for t in r_labels], fontsize=12)

    ax.set_ylim(0, 1)
    ax.grid(True, color='white', linewidth=1.5, alpha=0.7)
    ax.set_title(title, fontsize=18, fontweight='bold', pad=30)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Concentration (mol/cc)', fontsize=14)

    plt.tight_layout()
    png = os.path.join(OUTPUT_DIR, f"{fname}.png")
    pdf = os.path.join(OUTPUT_DIR, f"{fname}.pdf")
    plt.savefig(png, dpi=300, bbox_inches='tight')
    plt.savefig(pdf, bbox_inches='tight')
    plt.close()
    return fig, png, pdf

# ----------------------------------------------------------------------
# 7. STREAMLIT UI
# ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Sunburst Centerpoint", layout="wide")
    st.title("Interpolation & Sunburst Visualization of Centerpoint Concentrations")

    sols, params, _, _, _, logs = load_solutions(PINN_DIR)

    with st.expander("Load Logs"):
        for l in logs:
            st.write(l)

    if not sols:
        st.stop()

    interpolator = MultiParamAttentionInterpolator(sigma=0.2)

    cu_data, ni_data = build_sunburst_matrices(sols, params, interpolator)

    col1, col2 = st.columns(2)

    with col1:
        fig_cu, png_cu, pdf_cu = plot_sunburst_polar(
            cu_data, "Cu Concentration at Center (x=30, y=Ly/2)",
            cmap='plasma', vmin=0, vmax=1.5e-3, fname="sunburst_cu_center"
        )
        st.pyplot(fig_cu)
        with open(png_cu, "rb") as f:
            st.download_button("Cu (PNG)", f, "sunburst_cu_center.png", "image/png")
        with open(pdf_cu, "rb") as f:
            st.download_button("Cu (PDF)", f, "sunburst_cu_center.pdf", "application/pdf")

    with col2:
        fig_ni, png_ni, pdf_ni = plot_sunburst_polar(
            ni_data, "Ni Concentration at Center (x=30, y=Ly/2)",
            cmap='viridis', vmin=0, vmax=3e-4, fname="sunburst_ni_center"
        )
        st.pyplot(fig_ni)
        with open(png_ni, "rb") as f:
            st.download_button("Ni (PNG)", f, "sunburst_ni_center.png", "image/png")
        with open(pdf_ni, "rb") as f:
            st.download_button("Ni (PDF)", f, "sunburst_ni_center.pdf", "application/pdf")

    st.success(f"Sunburst figures saved to `{OUTPUT_DIR}/`")


if __name__ == "__main__":
    main()
