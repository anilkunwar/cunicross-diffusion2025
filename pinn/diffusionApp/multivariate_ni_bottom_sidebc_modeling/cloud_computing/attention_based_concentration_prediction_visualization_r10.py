# app.py — FINAL FULL VERSION: All Features + Physically Correct Pure Modes
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
# Enhanced Matplotlib Style
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
# 50+ Colormaps
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
# SQLite Database Functions
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
    ''', (session_id, json.dumps(parameters),
          pickle.dumps(cu_matrix), pickle.dumps(ni_matrix),
          pickle.dumps(times), pickle.dumps(ly_spokes)))
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
# Load Solutions
# ----------------------------------------------------------------------
@st.cache_data
def load_solutions(solution_dir):
    solutions, params_list, load_logs = [], [], []
    for fname in os.listdir(solution_dir):
        if not fname.endswith(".pkl"): 
            continue
        path = os.path.join(solution_dir, fname)
        try:
            with open(path, "rb") as f:
                sol = pickle.load(f)
            required = ['params', 'X', 'Y', 'c1_preds', 'c2_preds', 'times']
            if not all(k in sol for k in required):
                raise ValueError("Missing required keys")
            sol['filename'] = fname
            p = sol['params']
            params_list.append((p['Ly'], p['C_Cu'], p['C_Ni']))
            solutions.append(sol)
            load_logs.append(f"{fname}: OK")
        except Exception as e:
            load_logs.append(f"{fname}: ERROR → {e}")
    load_logs.append(f"Loaded {len(solutions)} valid solutions.")
    return solutions, params_list, load_logs

# ----------------------------------------------------------------------
# Attention-based Interpolator
# ----------------------------------------------------------------------
class MultiParamAttentionInterpolator(nn.Module):
    def __init__(self, sigma=0.2, num_heads=4, d_head=8):
        super().__init__()
        self.sigma = sigma
        self.num_heads = num_heads
        self.d_head = d_head
        self.W_q = nn.Linear(3, num_heads * d_head)
        self.W_k = nn.Linear(3, num_heads * d_head)

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
        spatial_w = torch.exp(-dist**2 / 2)
        spatial_w = spatial_w / (spatial_w.sum() + 1e-12)
        w = attn_w * spatial_w
        w = w / (w.sum() + 1e-12)

        return self._physics_aware_interpolation(solutions, w.detach().numpy(), ly_target, c_cu_target, c_ni_target)

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
                except:
                    continue
        c1[:, :, 0] = c_cu_target
        c2[:, :, -1] = c_ni_target
        return {'params': {'Ly': ly_target, 'C_Cu': c_cu_target, 'C_Ni': c_ni_target},
                'X': X, 'Y': Y, 'times': times, 'c1_preds': list(c1), 'c2_preds': list(c2)}

# ----------------------------------------------------------------------
# Center Concentration Extractor
# ----------------------------------------------------------------------
def get_center_conc(solution, ly_fraction=0.5):
    Lx, Ly = solution['params']['Lx'], solution['params']['Ly']
    ix = np.argmin(np.abs(solution['X'][:,0] - Lx/2))
    iy = np.argmin(np.abs(solution['Y'][0,:] - Ly * ly_fraction))
    cu = np.array([c1[ix, iy] for c1 in solution['c1_preds']])
    ni = np.array([c2[ix, iy] for c2 in solution['c2_preds']])
    return cu, ni

# ----------------------------------------------------------------------
# LY Spokes
# ----------------------------------------------------------------------
def generate_ly_spokes(ly_min=30, ly_max=120, step=10):
    return list(range(ly_min, ly_max + step, step))

# ----------------------------------------------------------------------
# Sunburst Matrix Builder — WITH PERFECT FILE FILTERING
# ----------------------------------------------------------------------
def build_sunburst_matrices(solutions, params_list, interpolator,
                           c_cu_target, c_ni_target, ly_fraction, ly_spokes, time_log_scale=False):
    global CURRENT_MODE

    N_TIME = 50
    cu_mat = np.zeros((N_TIME, len(ly_spokes)))
    ni_mat = np.zeros((N_TIME, len(ly_spokes)))

    times = np.logspace(-1, np.log10(200), N_TIME) if time_log_scale else np.linspace(0, 200.0, N_TIME)

    # FILTER SOLUTIONS BASED ON CURRENT_MODE
    filtered_solutions = solutions.copy()
    filtered_params = params_list.copy()

    if CURRENT_MODE == "Pure Cu Diffusion (Cu substrate bottom, air top)":
        filtered_solutions = [s for s in solutions if "cu_selfdiffusion" in s['filename']]
        filtered_params = [p for p, s in zip(params_list, solutions) if "cu_selfdiffusion" in s['filename']]
        st.success("Pure Cu Mode → Using ONLY cu_selfdiffusion files")

    elif CURRENT_MODE == "Pure Ni Diffusion (Ni substrate bottom, air top)":
        filtered_solutions = [s for s in solutions if "ni_selfdiffusion" in s['filename']]
        filtered_params = [p for p, s in zip(params_list, solutions) if "ni_selfdiffusion" in s['filename']]
        st.success("Pure Ni Mode → Using ONLY ni_selfdiffusion files")

    if len(filtered_solutions) == 0:
        st.error(f"No matching files for mode: {CURRENT_MODE}")
        st.stop()

    prog = st.progress(0)
    for j, ly in enumerate(ly_spokes):
        sol = interpolator(filtered_solutions, filtered_params, ly, c_cu_target, c_ni_target)
        cu, ni = get_center_conc(sol, ly_fraction)
        cu_mat[:, j] = cu
        ni_mat[:, j] = ni
        prog.progress((j + 1) / len(ly_spokes))
    prog.empty()
    return cu_mat, ni_mat, times

# ----------------------------------------------------------------------
# Sunburst Plot
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
        R = R[::-1]
        data = data[::-1, :]

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

# ----------------------------------------------------------------------
# Radar Chart
# ----------------------------------------------------------------------
def plot_radar_single(data, element, t_val, fname, ly_spokes, show_labels=True, show_radial_labels=True):
    angles = np.linspace(0, 2*np.pi, len(ly_spokes), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    data_cyclic = np.concatenate([data, [data[0]]])
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    color = 'red' if element == 'Cu' else 'blue'
    ax.plot(angles, data_cyclic, 'o-', linewidth=3, markersize=8, color=color, label=element)
    ax.fill(angles, data_cyclic, alpha=0.25, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"{ly}" for ly in ly_spokes], fontsize=14)
    ax.set_ylim(0, max(np.max(data), 1e-6) * 1.2)
    ax.set_title(f"{element} Concentration at t = {t_val:.1f} s", fontsize=18, pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=14)
    ax.grid(True, linewidth=1.5)
    if show_radial_labels:
        yticks = ax.get_yticks()
        ax.set_yticklabels([f"{y:.2e}" for y in yticks], fontsize=12)
    else:
        ax.set_yticklabels([])
    if show_labels and np.any(data > 1e-12):
        for angle, value in zip(angles[:-1], data):
            if value > max(data) * 0.1:
                ax.annotate(f'{value:.1e}', (angle, value), textcoords='offset points',
                            xytext=(0, 10), ha='center', fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    png = os.path.join(FIGURE_DIR, f"{fname}.png")
    pdf = os.path.join(FIGURE_DIR, f"{fname}.pdf")
    plt.savefig(png, dpi=300, bbox_inches='tight')
    plt.savefig(pdf, bbox_inches='tight')
    plt.close()
    return fig, png, pdf

# ----------------------------------------------------------------------
# Session ID
# ----------------------------------------------------------------------
def generate_session_id(parameters):
    param_str = f"{parameters.get('c_cu_target','')}_{parameters.get('c_ni_target','')}_{parameters.get('ly_fraction','')}_{parameters.get('ly_step','')}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"session_{timestamp}_{hash(param_str) % 10000:04d}"

# ----------------------------------------------------------------------
# MAIN APP
# ----------------------------------------------------------------------
def main():
    global CURRENT_MODE
    st.set_page_config(page_title="Cu/Ni Diffusion Visualizer", layout="wide")
    st.title("Cu/Ni Interdiffusion Visualizer — Pure vs Coupled")

    init_database()
    sols, params, logs = load_solutions(SOLUTION_DIR)
    with st.expander("Loaded Files"):
        for log in logs: st.write(log)
    if not sols:
        st.error("No .pkl files found in pinn_solutions/")
        st.stop()

    interpolator = MultiParamAttentionInterpolator()

    st.sidebar.header("Controls")
    recent = get_recent_sessions()
    session_opts = ["Create New Session"] + [f"{s[0]} ({s[1]})" for s in recent]
    selected_session = st.sidebar.selectbox("Session", session_opts)

    ly_step = st.sidebar.selectbox("Ly Step (µm)", [5, 10], index=1)
    LY_SPOKES = generate_ly_spokes(step=ly_step)

    col1, col2 = st.sidebar.columns(2)
    with col1: cmap_cu = st.selectbox("Cu Colormap", EXTENDED_CMAPS, index=EXTENDED_CMAPS.index('jet'))
    with col2: cmap_ni = st.selectbox("Ni Colormap", EXTENDED_CMAPS, index=EXTENDED_CMAPS.index('turbo'))

    conc_log_scale = st.sidebar.checkbox("Log Concentration Scale", True)
    time_log_scale = st.sidebar.checkbox("Log Time Scale", True)
    show_radar_labels = st.sidebar.checkbox("Show Radar Labels", True)
    show_radial_labels = st.sidebar.checkbox("Show Radial Labels", True)

    c_cu_target = st.sidebar.number_input("Cu Boundary (mol/cc)", 1e-6, 5e-3, 1e-3, format="%.1e")
    c_ni_target = st.sidebar.number_input("Ni Boundary (mol/cc)", 1e-6, 5e-3, 1e-4, format="%.1e")

    frac_opts = {"Ly/2": 0.5, "Ly/3": 1/3, "Ly/4": 0.25, "Ly/5": 0.2, "Ly/10": 0.1}
    ly_fraction = frac_opts[st.sidebar.selectbox("Sampling y = Ly ×", list(frac_opts.keys()))]
    ly_dir = st.sidebar.radio("Time Direction", ["bottom→top", "top→bottom"])

    st.sidebar.header("Diffusion Mode")
    mode = st.sidebar.radio("Mode", [
        "Standard Cu-Ni Coupled Diffusion",
        "Pure Cu Diffusion (Cu substrate bottom, air top)",
        "Pure Ni Diffusion (Ni substrate bottom, air top)"
    ], index=0)
    CURRENT_MODE = mode

    if "Pure Cu" in mode:
        c_ni_target = 1e-12
        st.sidebar.success("Pure Cu Mode → Using only cu_selfdiffusion files")
    elif "Pure Ni" in mode:
        c_cu_target = 1e-12
        st.sidebar.success("Pure Ni Mode → Using only ni_selfdiffusion files")

    parameters = {
        'c_cu_target': c_cu_target, 'c_ni_target': c_ni_target,
        'ly_fraction': ly_fraction, 'ly_step': ly_step,
        'time_log_scale': time_log_scale, 'conc_log_scale': conc_log_scale,
        'ly_dir': ly_dir, 'cmap_cu': cmap_cu, 'cmap_ni': cmap_ni
    }

    if selected_session == "Create New Session":
        session_id = generate_session_id(parameters)
        with st.spinner("Computing sunburst matrices..."):
            cu_mat, ni_mat, times = build_sunburst_matrices(
                sols, params, interpolator, c_cu_target, c_ni_target,
                ly_fraction, LY_SPOKES, time_log_scale
            )
        save_sunburst_data(session_id, parameters, cu_mat, ni_mat, times, LY_SPOKES)
        st.success(f"New session saved: {session_id}")
    else:
        session_id = selected_session.split(" (")[0]
        loaded = load_sunburst_data(session_id)
        if loaded:
            _, cu_mat, ni_mat, times, LY_SPOKES = loaded
            st.success(f"Loaded session: {session_id}")
        else:
            st.error("Failed to load session")
            return

    # Sunburst Plots
    st.subheader("Sunburst Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        fig_cu, png_cu, pdf_cu = plot_sunburst(cu_mat, f"Cu Concentration — {mode}", cmap_cu, 0, c_cu_target,
                                              conc_log_scale, time_log_scale, ly_dir,
                                              f"sunburst_cu_{session_id}", times, LY_SPOKES)
        st.pyplot(fig_cu)
        with open(png_cu, "rb") as f: st.download_button("Download Cu PNG", f, "cu_sunburst.png")
    with col2:
        fig_ni, png_ni, pdf_ni = plot_sunburst(ni_mat, f"Ni Concentration — {mode}", cmap_ni, 0, c_ni_target,
                                              conc_log_scale, time_log_scale, ly_dir,
                                              f"sunburst_ni_{session_id}", times, LY_SPOKES)
        st.pyplot(fig_ni)
        with open(png_ni, "rb") as f: st.download_button("Download Ni PNG", f, "ni_sunburst.png")

    # Radar Charts
    st.subheader("Radar Charts")
    t_idx = st.slider("Time Index", 0, len(times)-1, len(times)//2)
    t_val = times[t_idx]
    col1, col2 = st.columns(2)
    with col1:
        fig_cu_r, png_cu_r, _ = plot_radar_single(cu_mat[t_idx], "Cu", t_val, f"radar_cu_t{t_val:.0f}", LY_SPOKES,
                                                 show_radar_labels, show_radial_labels)
        st.pyplot(fig_cu_r)
    with col2:
        fig_ni_r, png_ni_r, _ = plot_radar_single(ni_mat[t_idx], "Ni", t_val, f"radar_ni_t{t_val:.0f}", LY_SPOKES,
                                                 show_radar_labels, show_radial_labels)
        st.pyplot(fig_ni_r)

    st.caption(f"Mode: **{mode}** | Center at y = Ly × {ly_fraction} | Session: {session_id}")

if __name__ == "__main__":
    main()
