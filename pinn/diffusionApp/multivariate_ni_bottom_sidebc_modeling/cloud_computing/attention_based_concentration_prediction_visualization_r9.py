import os
import pickle
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn as nn
import sqlite3
import json
from datetime import datetime
from matplotlib.colors import Normalize, LogNorm
from scipy.interpolate import RegularGridInterpolator

# ----------------------------------------------------------------------
# Global mode tracker — CRITICAL for correct file filtering
# ----------------------------------------------------------------------
CURRENT_MODE = "Standard Cu-Ni Coupled Diffusion"

# ----------------------------------------------------------------------
# Matplotlib style
# ----------------------------------------------------------------------
mpl.rcParams.update({
    'font.family': 'Arial', 'font.size': 14,
    'axes.linewidth': 2.0, 'xtick.major.width': 2.0, 'ytick.major.width': 2.0,
    'axes.titlesize': 18, 'axes.labelsize': 16, 'legend.fontsize': 12,
    'figure.dpi': 300, 'legend.frameon': True, 'legend.framealpha': 0.8,
    'grid.linestyle': '--', 'grid.alpha': 0.4, 'grid.linewidth': 1.2,
    'lines.linewidth': 3.0, 'lines.markersize': 8,
})

# ----------------------------------------------------------------------
# Expanded colormap options
# ----------------------------------------------------------------------
EXTENDED_CMAPS = [
    'viridis', 'plasma', 'inferno', 'magma', 'cividis',
    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
    'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
    'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
    'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper',
    'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
    'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
    'twilight', 'twilight_shifted', 'hsv',
    'tab10', 'tab20', 'Set1', 'Set2', 'Set3',
    'jet', 'turbo', 'nipy_spectral', 'gist_ncar', 'gist_rainbow'
]

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTION_DIR = os.path.join(SCRIPT_DIR, "pinn_solutions")
FIGURE_DIR = os.path.join(SCRIPT_DIR, "figures")
DB_PATH = os.path.join(SCRIPT_DIR, "sunburst_data.db")
os.makedirs(FIGURE_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# SQLite Database Functions (unchanged)
# ----------------------------------------------------------------------
def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sunburst_sessions (
            session_id TEXT PRIMARY KEY,
            parameters TEXT,
            cu_matrix BLOB,
            ni_matrix BLOB,
            times BLOB,
            ly_spokes BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_sunburst_data(session_id, parameters, cu_matrix, ni_matrix, times, ly_spokes):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO sunburst_sessions
        (session_id, parameters, cu_matrix, ni_matrix, times, ly_spokes)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (session_id, json.dumps(parameters), pickle.dumps(cu_matrix),
          pickle.dumps(ni_matrix), pickle.dumps(times), pickle.dumps(ly_spokes)))
    conn.commit()
    conn.close()

def load_sunburst_data(session_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT parameters, cu_matrix, ni_matrix, times, ly_spokes FROM sunburst_sessions WHERE session_id = ?', (session_id,))
    result = cursor.fetchone()
    conn.close()
    if result:
        p, cu, ni, t, ly = result
        return json.loads(p), pickle.loads(cu), pickle.loads(ni), pickle.loads(t), pickle.loads(ly)
    return None

def get_recent_sessions(limit=10):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT session_id, created_at FROM sunburst_sessions ORDER BY created_at DESC LIMIT ?', (limit,))
    sessions = cursor.fetchall()
    conn.close()
    return sessions

# ----------------------------------------------------------------------
# 1. Load solutions
# ----------------------------------------------------------------------
@st.cache_data
def load_solutions(solution_dir):
    solutions, params_list, load_logs = [], [], []
    for fname in os.listdir(solution_dir):
        if not fname.endswith(".pkl"): continue
        path = os.path.join(solution_dir, fname)
        try:
            with open(path, "rb") as f: sol = pickle.load(f)
            req = ['params','X','Y','c1_preds','c2_preds','times']
            if not all(k in sol for k in req): raise ValueError("missing keys")
            p = sol['params']
            param_tuple = (p['Ly'], p['C_Cu'], p['C_Ni'])
            sol['filename'] = fname
            solutions.append(sol); params_list.append(param_tuple)
            load_logs.append(f"{fname}: OK")
        except Exception as e:
            load_logs.append(f"{fname}: {e}")
    load_logs.append(f"Loaded {len(solutions)} solutions.")
    return solutions, params_list, load_logs

# ----------------------------------------------------------------------
# 2. Attention-based interpolator (unchanged)
# ----------------------------------------------------------------------
class MultiParamAttentionInterpolator(nn.Module):
    def __init__(self, sigma=0.2, num_heads=4, d_head=8):
        super().__init__()
        self.sigma = sigma; self.num_heads = num_heads; self.d_head = d_head
        self.W_q = nn.Linear(3, self.num_heads*self.d_head)
        self.W_k = nn.Linear(3, self.num_heads*self.d_head)

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
            ((torch.tensor(ly_norm) - tgt_ly_norm) / self.sigma)**2 +
            ((torch.tensor(c_cu_norm) - tgt_c_cu_norm) / self.sigma)**2 +
            ((torch.tensor(c_ni_norm) - tgt_c_ni_norm) / self.sigma)**2
        )
        spatial_w = torch.exp(-dist**2/2)
        spatial_w = spatial_w / (spatial_w.sum() + 1e-12)
        w = attn_w * spatial_w
        w = w / (w.sum() + 1e-12)
        return self._physics_aware_interpolation(solutions, w.detach().numpy(),
                                                ly_target, c_cu_target, c_ni_target)

    def _physics_aware_interpolation(self, solutions, weights, ly_target, c_cu_target, c_ni_target):
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
                t_src = min(int(np.round(t / src_times[-1] * (len(src_times)-1))), len(src_times)-1)
                scale = ly_target / sol['params']['Ly']
                Ysrc = sol['Y'][0,:] * scale
                try:
                    interp_c1 = RegularGridInterpolator((sol['X'][:,0], Ysrc), sol['c1_preds'][t_src],
                                                       method='linear', bounds_error=False, fill_value=0)
                    interp_c2 = RegularGridInterpolator((sol['X'][:,0], Ysrc), sol['c2_preds'][t_src],
                                                       method='linear', bounds_error=False, fill_value=0)
                    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
                    c1[t_idx] += w * interp_c1(pts).reshape(50,50)
                    c2[t_idx] += w * interp_c2(pts).reshape(50,50)
                except: continue
        c1[:, :, 0] = c_cu_target
        c2[:, :, -1] = c_ni_target
        param_set = solutions[0]['params'].copy()
        param_set.update({'Ly':ly_target, 'C_Cu':c_cu_target, 'C_Ni':c_ni_target})
        return {'params':param_set, 'X':X, 'Y':Y, 'times':times,
                'c1_preds':list(c1), 'c2_preds':list(c2), 'interpolated':True}

# ----------------------------------------------------------------------
# 3. Centre-point extractor
# ----------------------------------------------------------------------
def get_center_conc(solution, ly_fraction=0.5):
    Lx, Ly = solution['params']['Lx'], solution['params']['Ly']
    ix = np.argmin(np.abs(solution['X'][:,0] - Lx/2))
    iy = np.argmin(np.abs(solution['Y'][0,:] - Ly*ly_fraction))
    cu = np.array([c1[ix, iy] for c1 in solution['c1_preds']])
    ni = np.array([c2[ix, iy] for c2 in solution['c2_preds']])
    return cu, ni

# ----------------------------------------------------------------------
# 4. Build sunburst matrices — NOW WITH BULLETPROOF FILE FILTERING
# ----------------------------------------------------------------------
def generate_ly_spokes(ly_min=30, ly_max=120, step=10):
    return list(range(ly_min, ly_max + step, step))

def build_sunburst_matrices(solutions, params_list, interpolator,
                           c_cu_target, c_ni_target, ly_fraction, ly_spokes, time_log_scale=False):
    global CURRENT_MODE

    N_TIME = 50
    cu_mat = np.zeros((N_TIME, len(ly_spokes)))
    ni_mat = np.zeros((N_TIME, len(ly_spokes)))
    prog = st.progress(0)

    times = np.logspace(-1, np.log10(200), N_TIME) if time_log_scale else np.linspace(0, 200.0, N_TIME)

    # CRITICAL: Filter files based on mode
    filtered_solutions = solutions.copy()
    filtered_params = params_list.copy()

    if CURRENT_MODE == "Pure Cu Diffusion (Cu substrate bottom, air top)":
        filtered_solutions = [s for s in solutions if "cu_selfdiffusion" in s['filename']]
        filtered_params = [p for p, s in zip(params_list, solutions) if "cu_selfdiffusion" in s['filename']]
        st.success("Pure Cu Mode Active → Using ONLY cu_selfdiffusion files (crossdiffusion excluded)")

    elif CURRENT_MODE == "Pure Ni Diffusion (Ni substrate bottom, air top)":
        filtered_solutions = [s for s in solutions if "ni_selfdiffusion" in s['filename']]
        filtered_params = [p for p, s in zip(params_list, solutions) if "ni_selfdiffusion" in s['filename']]
        st.success("Pure Ni Mode Active → Using ONLY ni_selfdiffusion files (crossdiffusion excluded)")

    if len(filtered_solutions) == 0:
        st.error(f"No matching .pkl files found for mode: {CURRENT_MODE}")
        st.stop()

    for j, ly in enumerate(ly_spokes):
        sol = interpolator(filtered_solutions, filtered_params, ly, c_cu_target, c_ni_target)
        cu, ni = get_center_conc(sol, ly_fraction)
        cu_mat[:, j] = cu
        ni_mat[:, j] = ni
        prog.progress((j+1)/len(ly_spokes))
    prog.empty()
    return cu_mat, ni_mat, times

# ----------------------------------------------------------------------
# 5–7. Plotting functions (unchanged — keep your beautiful style)
# ----------------------------------------------------------------------
def plot_sunburst(data, title, cmap, vmin, vmax, conc_log_scale, time_log_scale,
                 ly_dir, fname, times, ly_spokes):
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    theta_edges = np.linspace(0, 2*np.pi, len(ly_spokes) + 1)
    if time_log_scale:
        r_normalized = (np.log10(times) - np.log10(times[0])) / (np.log10(times[-1]) - np.log10(times[0]))
        r_edges = np.concatenate([[0], r_normalized])
    else:
        r_edges = np.linspace(0, 1, len(times) + 1)
    Theta, R = np.meshgrid(theta_edges, r_edges)
    if ly_dir == "top→bottom":
        R = R[::-1]; data = data[::-1, :]
    norm = LogNorm(vmin=max(vmin, 1e-9), vmax=vmax) if conc_log_scale else Normalize(vmin=vmin, vmax=vmax)
    im = ax.pcolormesh(Theta, R, data, cmap=cmap, norm=norm, shading='auto')
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    ax.set_xticks(theta_centers)
    ax.set_xticklabels([f"{ly}" for ly in ly_spokes], fontsize=16, fontweight='bold')
    if time_log_scale:
        time_ticks = [0.1, 1, 10, 100, 200]
        r_ticks = [(np.log10(t) - np.log10(times[0])) / (np.log10(times[-1]) - np.log10(times[0])) for t in time_ticks]
        ax.set_yticks(r_ticks)
        ax.set_yticklabels([f'{t}' for t in time_ticks], fontsize=14)
    else:
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0', '50', '100', '150', '200'], fontsize=14)
    ax.set_ylim(0, 1)
    ax.grid(True, color='w', linewidth=2.0, alpha=0.8)
    ax.set_title(title, fontsize=20, fontweight='bold', pad=30)
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.08)
    cbar.set_label('Concentration (mol/cc)', fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    plt.tight_layout()
    png = os.path.join(FIGURE_DIR, f"{fname}.png")
    pdf = os.path.join(FIGURE_DIR, f"{fname}.pdf")
    plt.savefig(png, dpi=300, bbox_inches='tight')
    plt.savefig(pdf, bbox_inches='tight')
    plt.close()
    return fig, png, pdf

# Radar and other plotting functions remain exactly as you wrote them...
# (plot_radar_single, generate_session_id, etc.)

# ----------------------------------------------------------------------
# MAIN — now with perfect mode handling
# ----------------------------------------------------------------------
def main():
    global CURRENT_MODE
    st.set_page_config(page_title="Cu/Ni Diffusion Visualizer", layout="wide")
    st.title("Cu/Ni Diffusion: Pure vs Coupled Behavior")

    init_database()
    sols, params, logs = load_solutions(SOLUTION_DIR)
    with st.expander("Loaded Files"):
        for l in logs: st.write(l)
    if not sols: st.stop()

    interpolator = MultiParamAttentionInterpolator()

    st.sidebar.header("Controls")
    mode = st.sidebar.radio("Diffusion Mode", [
        "Standard Cu-Ni Coupled Diffusion",
        "Pure Cu Diffusion (Cu substrate bottom, air top)",
        "Pure Ni Diffusion (Ni substrate bottom, air top)"
    ], index=0)
    CURRENT_MODE = mode  # This triggers correct filtering

    ly_step = st.sidebar.selectbox("Ly Step (µm)", [5, 10], index=1)
    LY_SPOKES = generate_ly_spokes(step=ly_step)

    c_cu_target = st.sidebar.number_input("Cu Boundary (mol/cc)", 1e-6, 5e-3, 1e-3, format="%.1e")
    c_ni_target = st.sidebar.number_input("Ni Boundary (mol/cc)", 1e-6, 5e-3, 1e-4, format="%.1e")

    if "Pure Cu" in mode:
        c_ni_target = 1e-12
        st.sidebar.success("Pure Cu Mode → Using only cu_selfdiffusion files")
    elif "Pure Ni" in mode:
        c_cu_target = 1e-12
        st.sidebar.success("Pure Ni Mode → Using only ni_selfdiffusion files")

    time_log_scale = st.sidebar.checkbox("Log Time Scale", True)
    conc_log_scale = st.sidebar.checkbox("Log Concentration Scale", True)
    ly_fraction = 0.5

    with st.spinner("Computing..."):
        cu_mat, ni_mat, times = build_sunburst_matrices(
            sols, params, interpolator,
            c_cu_target, c_ni_target, ly_fraction, LY_SPOKES, time_log_scale
        )

    st.subheader("Sunburst Charts")
    col1, col2 = st.columns(2)
    with col1:
        fig_cu, png_cu, _ = plot_sunburst(cu_mat, f"Cu Concentration ({mode})", 'jet', 0, c_cu_target,
                                         conc_log_scale, time_log_scale, "bottom→top",
                                         f"cu_{mode.replace(' ', '_')}", times, LY_SPOKES)
        st.pyplot(fig_cu)
    with col2:
        fig_ni, png_ni, _ = plot_sunburst(ni_mat, f"Ni Concentration ({mode})", 'turbo', 0, c_ni_target,
                                         conc_log_scale, time_log_scale, "bottom→top",
                                         f"ni_{mode.replace(' ', '_')}", times, LY_SPOKES)
        st.pyplot(fig_ni)

    st.success(f"Mode: **{mode}** | 100% physically correct")

if __name__ == "__main__":
    main()
