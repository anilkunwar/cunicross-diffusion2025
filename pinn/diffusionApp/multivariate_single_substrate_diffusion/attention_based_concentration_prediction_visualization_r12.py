# app.py — FINAL, FULLY ROBUST, NO ERRORS, ALL FEATURES
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
from scipy.interpolate import RegularGridInterpolator, interp1d
# ----------------------------------------------------------------------
# Global mode
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
# SQLite Database
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
# Extract data
# ----------------------------------------------------------------------
def extract_params_from_filename(filename):
    """
    Extract parameters from filename pattern:
    - solution_cu_selfdiffusion_ly_30.0_c_cu_1e-3_tmax_200.0.pkl
    - solution_ni_selfdiffusion_ly_60.0_c_ni_6e-4_tmax_200.0.pkl
   
    Returns: dict with Ly, C_Cu, C_Ni, and element type
    """
    import re
   
    # Initialize default values
    params = {
        'Ly': 60.0, # default
        'C_Cu': 1e-3, # default
        'C_Ni': 1e-4, # default
        'element': 'unknown',
        'tmax': 200.0
    }
   
    try:
        # Extract element type
        if 'cu_selfdiffusion' in filename.lower():
            params['element'] = 'cu'
        elif 'ni_selfdiffusion' in filename.lower():
            params['element'] = 'ni'
        elif 'coupled' in filename.lower():
            params['element'] = 'coupled'
       
        # Extract Ly (geometry length)
        ly_match = re.search(r'ly_([\d.]+)', filename.lower())
        if ly_match:
            params['Ly'] = float(ly_match.group(1))
       
        # Extract Cu concentration
        cu_match = re.search(r'c_cu_([\d.e+-]+)', filename.lower())
        if cu_match:
            conc_str = cu_match.group(1).replace('e-', 'e-').replace('e+', 'e+')
            params['C_Cu'] = float(conc_str)
       
        # Extract Ni concentration
        ni_match = re.search(r'c_ni_([\d.e+-]+)', filename.lower())
        if ni_match:
            conc_str = ni_match.group(1).replace('e-', 'e-').replace('e+', 'e+')
            params['C_Ni'] = float(conc_str)
           
        # Extract tmax if present
        tmax_match = re.search(r'tmax_([\d.]+)', filename.lower())
        if tmax_match:
            params['tmax'] = float(tmax_match.group(1))
           
    except Exception as e:
        print(f"Warning: Could not parse filename {filename}: {e}")
   
    return params
# ----------------------------------------------------------------------
# Load Solutions — SAFE
# ----------------------------------------------------------------------
@st.cache_data
def load_solutions(solution_dir):
    solutions, params_list, load_logs = [], [], []
    for fname in os.listdir(solution_dir):
        if not fname.endswith(".pkl"): continue
        path = os.path.join(solution_dir, fname)
        try:
            with open(path, "rb") as f:
                sol = pickle.load(f)
            required = ['params', 'X', 'Y', 'c1_preds', 'c2_preds', 'times']
            if not all(k in sol for k in required):
                raise ValueError("Missing keys")
           
            # Extract parameters from filename and update solution params
            file_params = extract_params_from_filename(fname)
           
            # Update solution parameters with values from filename
            p = sol['params']
            p['Ly'] = file_params['Ly']
            p['element_from_filename'] = file_params['element']
           
            # Update concentrations based on element type
            if file_params['element'] == 'cu':
                p['C_Cu'] = file_params['C_Cu']
                p['C_Ni'] = 0.0 # For pure Cu diffusion
            elif file_params['element'] == 'ni':
                p['C_Ni'] = file_params['C_Ni']
                p['C_Cu'] = 0.0 # For pure Ni diffusion
            else: # coupled or unknown
                p['C_Cu'] = file_params.get('C_Cu', p.get('C_Cu', 1e-3))
                p['C_Ni'] = file_params.get('C_Ni', p.get('C_Ni', 1e-4))
           
            sol['filename'] = fname
            solutions.append(sol)
            params_list.append((p['Ly'], p['C_Cu'], p['C_Ni']))
            load_logs.append(f"{fname}: Ly={file_params['Ly']}, C_Cu={file_params['C_Cu']:.2e}, C_Ni={file_params['C_Ni']:.2e}, element={file_params['element']}")
           
        except Exception as e:
            load_logs.append(f"{fname}: FAILED → {e}")
    load_logs.append(f"Loaded {len(solutions)} valid solutions.")
    return solutions, params_list, load_logs
# ----------------------------------------------------------------------
# Display Extracted Parameters
# ----------------------------------------------------------------------
def display_extracted_parameters(solutions):
    """Display table of extracted parameters from filenames"""
    if not solutions:
        return
   
    st.subheader("Extracted Parameters from Filenames")
   
    data = []
    for sol in solutions:
        fname = sol.get('filename', 'unknown')
        params = sol['params']
        data.append({
            'Filename': fname,
            'Ly (um)': params['Ly'],
            'C_Cu (mol/cc)': f"{params.get('C_Cu', 0):.2e}",
            'C_Ni (mol/cc)': f"{params.get('C_Ni', 0):.2e}",
            'Element': params.get('element_from_filename', 'unknown')
        })
   
    st.table(data)
# ----------------------------------------------------------------------
# Attention Interpolator — FIXED RETURN
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
        if len(solutions) == 0:
            st.error("No solutions to interpolate from!")
            st.stop()
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
        return self._physics_aware_interpolation(solutions, w.detach().numpy(),
                                                ly_target, c_cu_target, c_ni_target)
    def _physics_aware_interpolation(self, solutions, weights, ly_target, c_cu_target, c_ni_target):
        if len(solutions) == 0:
            return None
        Lx = solutions[0]['params'].get('Lx', 100.0)
        x = np.linspace(0, Lx, 50)
        y = np.linspace(0, ly_target, 50)
        times = np.linspace(0, 200.0, 50)
        X, Y = np.meshgrid(x, y, indexing='ij')
        c1 = np.zeros((len(times), 50, 50))
        c2 = np.zeros((len(times), 50, 50))
        for t_idx, t in enumerate(times):
            for sol, w in zip(solutions, weights):
                if w < 1e-8: continue
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
        # Enforce boundary conditions
        c1[:, :, 0] = c_cu_target
        c2[:, :, -1] = c_ni_target
        param_set = solutions[0]['params'].copy()
        param_set.update({'Ly': ly_target, 'C_Cu': c_cu_target, 'C_Ni': c_ni_target, 'Lx': Lx})
        return {
            'params': param_set,
            'X': X, 'Y': Y,
            'times': times,
            'c1_preds': list(c1),
            'c2_preds': list(c2),
            'interpolated': True
        }
# ----------------------------------------------------------------------
# Safe Center Extractor
# ----------------------------------------------------------------------
def get_center_conc(solution, ly_fraction=0.5, ly_current=None, temporal_bias_factor=0.1):
    if solution is None or 'params' not in solution:
        return np.zeros(50), np.zeros(50)
    params = solution['params']
    Lx = params.get('Lx', 100.0)
    Ly = params.get('Ly', 60.0)
    try:
        ix = np.argmin(np.abs(solution['X'][:,0] - Lx/2))
        iy = np.argmin(np.abs(solution['Y'][0,:] - Ly * ly_fraction))
        cu_raw = np.array([c1[ix, iy] for c1 in solution['c1_preds']])
        ni_raw = np.array([c2[ix, iy] for c2 in solution['c2_preds']])

        # === APPLY TEMPORAL BIAS BASED ON Ly INCREASE ===
        if ly_current is not None and temporal_bias_factor > 0:
            # Reference Ly = 30 μm (smallest in range)
            ly_ref = 30.0
            delay_scale = 1.0 + temporal_bias_factor * (ly_current - ly_ref) / 10.0
            delay_scale = max(delay_scale, 1.0)  # no speedup

            # Stretch time axis → slower rise
            times = solution.get('times', np.linspace(0, 200, len(cu_raw)))
            t_stretched = times * delay_scale

            # Re-interpolate concentrations onto original time grid
            # Clamp to avoid extrapolation
            cu_interp = interp1d(t_stretched, cu_raw, kind='linear',
                                 bounds_error=False, fill_value=(cu_raw[0], cu_raw[-1]))
            ni_interp = interp1d(t_stretched, ni_raw, kind='linear',
                                 bounds_error=False, fill_value=(ni_raw[0], ni_raw[-1]))

            t_original = np.linspace(0, 200, len(cu_raw))
            cu = cu_interp(t_original)
            ni = ni_interp(t_original)
        else:
            cu, ni = cu_raw, ni_raw

        return cu, ni
    except Exception as e:
        st.warning(f"Center extraction failed: {e}")
        return np.zeros(50), np.zeros(50)
# ----------------------------------------------------------------------
# Sunburst Matrix Builder — BULLETPROOF
# ----------------------------------------------------------------------
def build_sunburst_matrices(solutions, params_list, interpolator,
                           c_cu_target, c_ni_target, ly_fraction, ly_spokes, 
                           time_log_scale=False, temporal_bias_factor=0.1):
    global CURRENT_MODE
    N_TIME = 50
    cu_mat = np.zeros((N_TIME, len(ly_spokes)))
    ni_mat = np.zeros((N_TIME, len(ly_spokes)))
    times = np.logspace(-1, np.log10(200), N_TIME) if time_log_scale else np.linspace(0, 200.0, N_TIME)
    # Filter solutions safely
    filtered_solutions = solutions.copy()
    filtered_params = params_list.copy()
    if CURRENT_MODE == "Pure Cu Diffusion (Cu substrate bottom, air top)":
        filtered_solutions = [s for s in solutions if "cu_selfdiffusion" in s.get('filename', '')]
        filtered_params = [p for p, s in zip(params_list, solutions) if "cu_selfdiffusion" in s.get('filename', '')]
        st.success("Pure Cu Mode Active — Using only cu_selfdiffusion files")
    elif CURRENT_MODE == "Pure Ni Diffusion (Ni substrate bottom, air top)":
        filtered_solutions = [s for s in solutions if "ni_selfdiffusion" in s.get('filename', '')]
        filtered_params = [p for p, s in zip(params_list, solutions) if "ni_selfdiffusion" in s.get('filename', '')]
        st.success("Pure Ni Mode Active — Using only ni_selfdiffusion files")
    if len(filtered_solutions) == 0:
        st.error(f"No matching files found for mode: {CURRENT_MODE}")
        st.stop()
    prog = st.progress(0)
    for j, ly in enumerate(ly_spokes):
        sol = interpolator(filtered_solutions, filtered_params, ly, c_cu_target, c_ni_target)
        cu, ni = get_center_conc(sol, ly_fraction, ly_current=ly, temporal_bias_factor=temporal_bias_factor)
        cu_mat[:, j] = cu
        ni_mat[:, j] = ni
        prog.progress((j + 1) / len(ly_spokes))
    prog.empty()
    return cu_mat, ni_mat, times
# ----------------------------------------------------------------------
# Plotting Functions (unchanged — your beautiful versions)
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
        ticks = [0.1, 1, 10, 100, 200]
        r_ticks = [(np.log10(t) - np.log10(times[0])) / (np.log10(times[-1]) - np.log10(times[0])) for t in ticks]
        ax.set_yticks(r_ticks)
        ax.set_yticklabels([f'{t}' for t in ticks], fontsize=14)
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
    ax.set_title(f"{element} at t = {t_val:.1f} s", fontsize=18, pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=14)
    ax.grid(True, linewidth=1.5)
    if show_radial_labels:
        ax.set_yticklabels([f"{y:.2e}" for y in ax.get_yticks()], fontsize=12)
    if show_labels:
        for a, v in zip(angles[:-1], data):
            if v > max(data) * 0.1:
                ax.annotate(f'{v:.1e}', (a, v), xytext=(0, 10), textcoords='offset points',
                            ha='center', fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    png = os.path.join(FIGURE_DIR, f"{fname}.png")
    plt.savefig(png, dpi=300, bbox_inches='tight')
    plt.close()
    return fig, png, None
def generate_session_id(parameters):
    s = f"{parameters.get('c_cu_target',0)}_{parameters.get('c_ni_target',0)}_{parameters.get('ly_fraction',0.5)}"
    return f"session_{datetime.now():%Y%m%d_%H%M%S}_{hash(s)%10000:04d}"
# ----------------------------------------------------------------------
# MAIN — FULLY ELABORATE
# ----------------------------------------------------------------------
def main():
    global CURRENT_MODE
    st.set_page_config(page_title="Cu/Ni Diffusion Visualizer", layout="wide")
    st.title("Cu/Ni Interdiffusion — Pure vs Coupled Modes")
    init_database()
    sols, params, logs = load_solutions(SOLUTION_DIR)
    with st.expander("Loaded Files"): [st.write(l) for l in logs]
    if not sols: st.stop()
    # Display extracted parameters
    with st.expander("Extracted Parameters from Filenames"):
        display_extracted_parameters(sols)
   
    with st.expander("Loaded Files"):
        [st.write(l) for l in logs]
    interpolator = MultiParamAttentionInterpolator()
    # Sidebar
    st.sidebar.header("Controls")
    sessions = get_recent_sessions()
    opts = ["Create New Session"] + [f"{s[0]} ({s[1]})" for s in sessions]
    selected = st.sidebar.selectbox("Session", opts)
    ly_step = st.sidebar.selectbox("Ly Step", [5, 10], index=1)
    LY_SPOKES = list(range(30, 121, ly_step))
    col1, col2 = st.sidebar.columns(2)
    cmap_cu = col1.selectbox("Cu Map", EXTENDED_CMAPS, EXTENDED_CMAPS.index('jet'))
    cmap_ni = col2.selectbox("Ni Map", EXTENDED_CMAPS, EXTENDED_CMAPS.index('turbo'))
    conc_log = st.sidebar.checkbox("Log Conc", True)
    time_log = st.sidebar.checkbox("Log Time", True)
    show_labels = st.sidebar.checkbox("Radar Labels", True)
    show_radial = st.sidebar.checkbox("Radial Labels", True)
    c_cu_target = st.sidebar.number_input("Cu Boundary", 1e-6, 5e-3, 1e-3, format="%.1e")
    c_ni_target = st.sidebar.number_input("Ni Boundary", 1e-6, 5e-3, 1e-4, format="%.1e")
    frac = st.sidebar.selectbox("y = Ly ×", ["Ly/2", "Ly/3", "Ly/4", "Ly/5"], index=0)
    ly_fraction = {"Ly/2": 0.5, "Ly/3": 1/3, "Ly/4": 0.25, "Ly/5": 0.2}[frac]
    ly_dir = st.sidebar.radio("Time Flow", ["bottom→top", "top→bottom"])
    st.sidebar.header("Advanced Bias")
    temporal_bias_factor = st.sidebar.slider(
        "Temporal Delay Bias (per 10μm Ly increase)", 
        min_value=0.0, max_value=0.02, value=0.01, step=0.005,
        help="Higher value = slower centerline concentration rise for larger Ly. 0 = no bias."
    )
    st.sidebar.header("Mode")
    mode = st.sidebar.radio("Diffusion Mode", [
        "Standard Cu-Ni Coupled Diffusion",
        "Pure Cu Diffusion (Cu substrate bottom, air top)",
        "Pure Ni Diffusion (Ni substrate bottom, air top)"
    ])
    CURRENT_MODE = mode
    if "Pure Cu" in mode: c_ni_target = 1e-12
    if "Pure Ni" in mode: c_cu_target = 1e-12
    if selected == "Create New Session":
        session_id = generate_session_id({'c_cu_target': c_cu_target, 'c_ni_target': c_ni_target})
        with st.spinner("Computing..."):
            cu_mat, ni_mat, times = build_sunburst_matrices(
                sols, params, interpolator, c_cu_target, c_ni_target,
                ly_fraction, LY_SPOKES, time_log, temporal_bias_factor
            )
        save_sunburst_data(session_id, {}, cu_mat, ni_mat, times, LY_SPOKES)
        st.success(f"Saved: {session_id}")
    else:
        session_id = selected.split(" (")[0]
        data = load_sunburst_data(session_id)
        if data:
            _, cu_mat, ni_mat, times, LY_SPOKES = data
            st.success(f"Loaded: {session_id}")
        else:
            st.error("Load failed"); return
    st.subheader("Sunburst Charts")
    c1, c2 = st.columns(2)
    with c1:
        f, p, _ = plot_sunburst(cu_mat, f"Cu — {mode}", cmap_cu, 0, c_cu_target or 1e-3,
                                conc_log, time_log, ly_dir, f"cu_{session_id}", times, LY_SPOKES)
        st.pyplot(f)
    with c2:
        f, p, _ = plot_sunburst(ni_mat, f"Ni — {mode}", cmap_ni, 0, c_ni_target or 1e-3,
                                conc_log, time_log, ly_dir, f"ni_{session_id}", times, LY_SPOKES)
        st.pyplot(f)
    st.subheader("Radar Charts")
    t_idx = st.slider("Time", 0, len(times)-1, 25)
    c1, c2 = st.columns(2)
    with c1:
        f, _, _ = plot_radar_single(cu_mat[t_idx], "Cu", times[t_idx], f"radar_cu", LY_SPOKES, show_labels, show_radial)
        st.pyplot(f)
    with c2:
        f, _, _ = plot_radar_single(ni_mat[t_idx], "Ni", times[t_idx], f"radar_ni", LY_SPOKES, show_labels, show_radial)
        st.pyplot(f)
if __name__ == "__main__":
    main()
