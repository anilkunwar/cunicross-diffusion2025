import os
import pickle
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn as nn
import re
import sqlite3
import json
from datetime import datetime
from matplotlib.colors import Normalize, LogNorm
from scipy.interpolate import RegularGridInterpolator

# ----------------------------------------------------------------------
# Matplotlib style (enhanced with thicker lines and larger fonts)
# ----------------------------------------------------------------------
mpl.rcParams.update({
    'font.family': 'Arial', 'font.size': 14,  # Increased from 12
    'axes.linewidth': 2.0, 'xtick.major.width': 2.0, 'ytick.major.width': 2.0,  # Thicker lines
    'axes.titlesize': 18, 'axes.labelsize': 16, 'legend.fontsize': 12,  # Larger fonts
    'figure.dpi': 300, 'legend.frameon': True, 'legend.framealpha': 0.8,
    'grid.linestyle': '--', 'grid.alpha': 0.4, 'grid.linewidth': 1.2,  # Enhanced grid
    'lines.linewidth': 3.0, 'lines.markersize': 8,  # Thicker plot lines
})

# ----------------------------------------------------------------------
# Expanded colormap options (50+ options)
# ----------------------------------------------------------------------
EXTENDED_CMAPS = [
    # Sequential
    'viridis', 'plasma', 'inferno', 'magma', 'cividis',
    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
    'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 
    'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
    'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper',
    
    # Diverging
    'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
    'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
    
    # Cyclic
    'twilight', 'twilight_shifted', 'hsv',
    
    # Qualitative
    'tab10', 'tab20', 'Set1', 'Set2', 'Set3',
    
    # Special requests
    'jet', 'turbo', 'nipy_spectral', 'gist_ncar', 'gist_rainbow'
]

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
SOLUTION_DIR = os.path.join(SCRIPT_DIR, "pinn_solutions")
FIGURE_DIR   = os.path.join(SCRIPT_DIR, "figures")
DB_PATH      = os.path.join(SCRIPT_DIR, "sunburst_data.db")
os.makedirs(FIGURE_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# SQLite Database Functions
# ----------------------------------------------------------------------
def init_database():
    """Initialize SQLite database for storing sunburst data"""
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
    """Save sunburst data to SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Convert arrays to bytes for storage
    cu_blob = pickle.dumps(cu_matrix)
    ni_blob = pickle.dumps(ni_matrix)
    times_blob = pickle.dumps(times)
    ly_spokes_blob = pickle.dumps(ly_spokes)
    params_json = json.dumps(parameters)
    
    cursor.execute('''
        INSERT OR REPLACE INTO sunburst_sessions 
        (session_id, parameters, cu_matrix, ni_matrix, times, ly_spokes)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (session_id, params_json, cu_blob, ni_blob, times_blob, ly_spokes_blob))
    
    conn.commit()
    conn.close()

def load_sunburst_data(session_id):
    """Load sunburst data from SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT parameters, cu_matrix, ni_matrix, times, ly_spokes 
        FROM sunburst_sessions WHERE session_id = ?
    ''', (session_id,))
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        params_json, cu_blob, ni_blob, times_blob, ly_spokes_blob = result
        parameters = json.loads(params_json)
        cu_matrix = pickle.loads(cu_blob)
        ni_matrix = pickle.loads(ni_blob)
        times = pickle.loads(times_blob)
        ly_spokes = pickle.loads(ly_spokes_blob)
        
        return parameters, cu_matrix, ni_matrix, times, ly_spokes
    else:
        return None

def get_recent_sessions(limit=10):
    """Get recent session IDs for dropdown selection"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT session_id, created_at FROM sunburst_sessions 
        ORDER BY created_at DESC LIMIT ?
    ''', (limit,))
    
    sessions = cursor.fetchall()
    conn.close()
    
    return sessions

# ----------------------------------------------------------------------
# 1. Load solutions (unchanged)
# ----------------------------------------------------------------------
@st.cache_data
def load_solutions(solution_dir):
    solutions, params_list, load_logs = [], [], []
    lys, c_cus, c_nis = [], [], []
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
            lys.append(p['Ly']); c_cus.append(p['C_Cu']); c_nis.append(p['C_Ni'])
            load_logs.append(f"{fname}: OK")
        except Exception as e:
            load_logs.append(f"{fname}: {e}")
    load_logs.append(f"Loaded {len(solutions)} solutions.")
    return solutions, params_list, lys, c_cus, c_nis, load_logs

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
        lys   = np.array([p[0] for p in params_list])
        c_cus = np.array([p[1] for p in params_list])
        c_nis = np.array([p[2] for p in params_list])

        ly_norm   = (lys   - 30.0) / (120.0 - 30.0)
        c_cu_norm = c_cus / 2.9e-3
        c_ni_norm = c_nis / 1.8e-3

        tgt_ly_norm   = (ly_target   - 30.0) / (120.0 - 30.0)
        tgt_c_cu_norm = c_cu_target / 2.9e-3
        tgt_c_ni_norm = c_ni_target / 1.8e-3

        params_tensor = torch.tensor(np.stack([ly_norm, c_cu_norm, c_ni_norm], axis=1), dtype=torch.float32)
        target_tensor = torch.tensor([[tgt_ly_norm, tgt_c_cu_norm, tgt_c_ni_norm]], dtype=torch.float32)

        Q = self.W_q(target_tensor).view(1, self.num_heads, self.d_head)
        K = self.W_k(params_tensor).view(-1, self.num_heads, self.d_head)

        attn = torch.einsum('nhd,mhd->nmh', K, Q) / np.sqrt(self.d_head)
        attn_w = torch.softmax(attn, dim=0).mean(dim=2).squeeze(1)

        dist = torch.sqrt(
            ((torch.tensor(ly_norm)   - tgt_ly_norm)   / self.sigma)**2 +
            ((torch.tensor(c_cu_norm) - tgt_c_cu_norm) / self.sigma)**2 +
            ((torch.tensor(c_ni_norm) - tgt_c_ni_norm) / self.sigma)**2
        )
        spatial_w = torch.exp(-dist**2/2)
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
        times = np.linspace(0, 200.0, 50)                # T_MAX = 200 s
        X, Y = np.meshgrid(x, y, indexing='ij')
        c1 = np.zeros((len(times), 50, 50))
        c2 = np.zeros((len(times), 50, 50))

        for t_idx, t in enumerate(times):
            for sol, w in zip(solutions, weights):
                src_times = sol['times']
                t_src = min(int(np.round(t / src_times[-1] * (len(src_times)-1))),
                            len(src_times)-1)
                scale = ly_target / sol['params']['Ly']
                Ysrc = sol['Y'][0,:] * scale
                try:
                    interp_c1 = RegularGridInterpolator((sol['X'][:,0], Ysrc),
                                                       sol['c1_preds'][t_src],
                                                       method='linear',
                                                       bounds_error=False, fill_value=0)
                    interp_c2 = RegularGridInterpolator((sol['X'][:,0], Ysrc),
                                                       sol['c2_preds'][t_src],
                                                       method='linear',
                                                       bounds_error=False, fill_value=0)
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
    """Return (cu, ni) time-series at x=Lx/2, y=Ly*ly_fraction."""
    Lx, Ly = solution['params']['Lx'], solution['params']['Ly']
    ix = np.argmin(np.abs(solution['X'][:,0] - Lx/2))
    iy = np.argmin(np.abs(solution['Y'][0,:] - Ly*ly_fraction))
    cu = np.array([c1[ix, iy] for c1 in solution['c1_preds']])
    ni = np.array([c2[ix, iy] for c2 in solution['c2_preds']])
    return cu, ni

# ----------------------------------------------------------------------
# 4. Build the (N_TIME × N_LY) matrices for the sunburst
# ----------------------------------------------------------------------
def generate_ly_spokes(ly_min=30, ly_max=120, step=10):
    """Generate LY spokes with configurable step size"""
    return list(range(ly_min, ly_max + step, step))

def build_sunburst_matrices(solutions, params_list, interpolator,
                           c_cu_target, c_ni_target, ly_fraction, ly_spokes, time_log_scale=False):
    N_TIME = 50
    cu_mat = np.zeros((N_TIME, len(ly_spokes)))
    ni_mat = np.zeros((N_TIME, len(ly_spokes)))
    prog = st.progress(0)
    
    # Generate time points (linear or logarithmic)
    if time_log_scale:
        # Logarithmic time scale: avoid log(0) by starting from small value
        times = np.logspace(-1, np.log10(200), N_TIME)  # 0.1 to 200 seconds
    else:
        times = np.linspace(0, 200.0, N_TIME)
    
    for j, ly in enumerate(ly_spokes):
        sol = interpolator(solutions, params_list, ly, c_cu_target, c_ni_target)
        cu, ni = get_center_conc(sol, ly_fraction)
        cu_mat[:, j] = cu
        ni_mat[:, j] = ni
        prog.progress((j+1)/len(ly_spokes))
    prog.empty()
    return cu_mat, ni_mat, times

# ----------------------------------------------------------------------
# 5. Sunburst plot with logarithmic time scaling
# ----------------------------------------------------------------------
def plot_sunburst(data, title, cmap, vmin, vmax, conc_log_scale, time_log_scale, 
                 ly_dir, fname, times, ly_spokes):
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # --- theta edges for spokes ---
    theta_edges = np.linspace(0, 2*np.pi, len(ly_spokes) + 1)
    
    # --- radial edges (time) ---
    if time_log_scale:
        # Logarithmic radial scaling
        r_normalized = (np.log10(times) - np.log10(times[0])) / (np.log10(times[-1]) - np.log10(times[0]))
        r_edges = np.concatenate([[0], r_normalized])  # Add center point
    else:
        # Linear radial scaling  
        r_edges = np.linspace(0, 1, len(times) + 1)

    # --- meshgrid: (N_TIME+1, N_LY+1) ---
    Theta, R = np.meshgrid(theta_edges, r_edges)

    # --- reverse time direction if needed ---
    if ly_dir == "top→bottom":
        R = R[::-1]           # flip rows
        data = data[::-1, :]  # flip data rows

    # --- color norm ---
    if conc_log_scale:
        norm = LogNorm(vmin=max(vmin, 1e-9), vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    # --- pcolormesh ---
    im = ax.pcolormesh(Theta, R, data, cmap=cmap, norm=norm, shading='auto')

    # --- spoke labels (at center of each sector) ---
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    ax.set_xticks(theta_centers)
    ax.set_xticklabels([f"{ly}" for ly in ly_spokes], fontsize=16, fontweight='bold')  # Increased font

    # --- time labels ---
    if time_log_scale:
        # Logarithmic time labels
        time_ticks = [0.1, 1, 10, 100, 200]
        r_ticks = [(np.log10(t) - np.log10(times[0])) / (np.log10(times[-1]) - np.log10(times[0])) 
                  for t in time_ticks]
        ax.set_yticks(r_ticks)
        ax.set_yticklabels([f'{t}' for t in time_ticks], fontsize=14)  # Increased font
    else:
        # Linear time labels
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0', '50', '100', '150', '200'], fontsize=14)  # Increased font
    
    ax.set_ylim(0, 1)

    # --- enhanced style with thicker lines ---
    ax.grid(True, color='w', linewidth=2.0, alpha=0.8)  # Thicker grid lines
    ax.set_title(title, fontsize=20, fontweight='bold', pad=30)  # Larger title

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.08)
    cbar.set_label('Concentration (mol/cc)', fontsize=16)  # Larger font
    cbar.ax.tick_params(labelsize=14)  # Larger tick labels

    plt.tight_layout()
    png = os.path.join(FIGURE_DIR, f"{fname}.png")
    pdf = os.path.join(FIGURE_DIR, f"{fname}.pdf")
    plt.savefig(png, dpi=300, bbox_inches='tight')
    plt.savefig(pdf, bbox_inches='tight')
    plt.close()
    return fig, png, pdf

# ----------------------------------------------------------------------
# 6. CORRECTED radar charts for Cu and Ni with label toggle
# ----------------------------------------------------------------------
def plot_radar_single(data, element, t_val, fname, ly_spokes,
                      show_labels=True, show_radial_labels=True):
    angles = np.linspace(0, 2*np.pi, len(ly_spokes), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    data_cyclic = np.concatenate([data, [data[0]]])

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    color = 'red' if element == 'Cu' else 'blue'

    ax.plot(angles, data_cyclic, 'o-', linewidth=3, markersize=8, color=color, label=element)
    ax.fill(angles, data_cyclic, alpha=0.25, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"{ly}" for ly in ly_spokes], fontsize=14)
    ax.set_ylim(0, max(np.max(data), 1e-6) * 1.2)  # avoid zero ylim

    ax.set_title(f"{element} Concentration at t = {t_val:.1f} s", fontsize=18, pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=14)
    ax.grid(True, linewidth=1.5)

    if show_radial_labels:
        yticks = ax.get_yticks()
        ax.set_yticklabels([f"{tick:.2e}" for tick in yticks], fontsize=12)
    else:
        ax.set_yticklabels([])

    if show_labels and np.any(data > 1e-12):
        for angle, value in zip(angles[:-1], data):
            if value > max(data) * 0.1:
                ax.annotate(f'{value:.1e}', (angle, value),
                            textcoords='offset points', xytext=(0, 10),
                            ha='center', fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2',
                                      facecolor='white', alpha=0.7))

    png = os.path.join(FIGURE_DIR, f"{fname}.png")
    pdf = os.path.join(FIGURE_DIR, f"{fname}.pdf")
    plt.savefig(png, dpi=300, bbox_inches='tight')
    plt.savefig(pdf, bbox_inches='tight')
    plt.close()
    return fig, png, pdf

# ----------------------------------------------------------------------
# 7. Generate unique session ID
# ----------------------------------------------------------------------
def generate_session_id(parameters):
    """Generate unique session ID based on parameters"""
    param_str = f"{parameters['c_cu_target']}_{parameters['c_ni_target']}_{parameters['ly_fraction']}_{parameters['ly_step']}_{parameters['time_log_scale']}_{parameters['conc_log_scale']}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"session_{timestamp}_{hash(param_str) % 10000:04d}"

# ----------------------------------------------------------------------
# 8. Enhanced Streamlit UI with SQLite integration
# ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Enhanced Sunburst + Radar Concentration Visualizer", layout="wide")
    st.title("Enhanced Sunburst & Radar Visualization of Cu/Ni Concentration")
    
    # Initialize database
    init_database()
    
    # Physics explanation section
    with st.expander("📚 Physics Explanation: Why Cu concentration decreases with smaller Ly"):
        st.markdown("""
        ### Diffusion Physics Explanation
        
        **Expected vs. Observed Behavior:**
        - **Expected:** Smaller Ly = shorter diffusion distance → faster saturation → higher concentration at given time
        - **Observed:** Smaller Ly shows lower Cu concentration in center
        
        **Possible Explanations:**
        
        1. **Boundary Condition Competition:**
           - Cu diffuses from bottom (y=0), Ni from top (y=Ly)
           - In thinner films (small Ly), Ni reaches center faster, competing with Cu
           - This competition reduces Cu accumulation at center
        
        2. **Inter-diffusion Effects:**
           - Cu and Ni inter-diffuse with different diffusion coefficients
           - In confined spaces (small Ly), the faster-diffusing species may dominate
        
        3. **Sampling Position Matters:**
           - Center point at Ly/2 moves closer to Ni source as Ly decreases
           - For Ly=30µm, center at 15µm is closer to Ni boundary
           - For Ly=120µm, center at 60µm is further from Ni boundary
        
        4. **Time Scale Considerations:**
           - Thinner films reach steady state faster
           - At same absolute time, thinner films may be closer to final equilibrium
           - Final equilibrium may have lower Cu concentration due to boundary conditions
        """)

    st.markdown("""
    ### Enhanced Visualization Features
    - **50+ colormap options** including jet and turbo
    - **Logarithmic time scaling** for better early-time resolution
    - **Separate colormaps** for Cu and Ni
    - **Configurable Ly steps** (5µm or 10µm increments)
    - **Enhanced visual styling** with thicker lines and larger fonts
    - **Separate radar charts** for each element
    - **Radar label toggle** to prevent overlapping
    - **SQLite database integration** for data persistence
    """)

    # ---- load data ----
    sols, params, _, _, _, logs = load_solutions(SOLUTION_DIR)
    with st.expander("Load log"):
        for l in logs: st.write(l)
    if not sols: st.stop()

    interpolator = MultiParamAttentionInterpolator()

    # ---- enhanced sidebar controls ----
    st.sidebar.header("🎛️ Enhanced Controls")
    
    # Session management
    st.sidebar.subheader("📊 Session Management")
    
    # Load previous session option
    recent_sessions = get_recent_sessions()
    session_options = ["Create New Session"] + [f"{s[0]} ({s[1]})" for s in recent_sessions]
    selected_session = st.sidebar.selectbox("Load Previous Session", session_options)
    
    # Ly step configuration
    ly_step = st.sidebar.selectbox("Ly Step Size (µm)", [5, 10], index=1)
    LY_SPOKES = generate_ly_spokes(step=ly_step)
    
    st.sidebar.info(f"Ly spokes: {LY_SPOKES}")
    
    # Separate colormaps for Cu and Ni
    col1, col2 = st.sidebar.columns(2)
    with col1:
        cmap_cu = st.selectbox("Cu Colormap", EXTENDED_CMAPS, 
                              index=EXTENDED_CMAPS.index('jet'))
    with col2:
        cmap_ni = st.selectbox("Ni Colormap", EXTENDED_CMAPS,
                              index=EXTENDED_CMAPS.index('turbo'))
    
    # Scale controls
    scale_col1, scale_col2 = st.sidebar.columns(2)
    with scale_col1:
        conc_log_scale = st.selectbox("Conc Scale", ["Linear","Logarithmic"]) == "Logarithmic"
    with scale_col2:
        time_log_scale = st.selectbox("Time Scale", ["Linear","Logarithmic"]) == "Logarithmic"

    # Radar chart options
    show_radar_labels = st.sidebar.checkbox("Show Radar Value Labels", value=True,
                                           help="Toggle value annotations on radar charts to prevent overlapping")
    show_radial_labels = st.sidebar.checkbox("Show Radial Axis Labels", value=True,
                                         help="Toggle visibility of the radar chart's radial axis labels")

    # Concentration inputs
    c_cu_target = st.sidebar.number_input(
        "Boundary Cu Concentration (mol/cc)", 1e-6, 5e-3, 1e-3, format="%.1e")
    c_ni_target = st.sidebar.number_input(
        "Boundary Ni Concentration (mol/cc)", 1e-6, 5e-3, 1e-4, format="%.1e")

    # Centre-fraction selector
    frac_options = {"Ly/2":0.5, "Ly/3":1/3, "Ly/4":0.25, "Ly/5":0.2, "Ly/10":0.1}
    ly_fraction = st.sidebar.selectbox("Centre sampling (y = Ly × fraction)",
                                      options=list(frac_options.keys()),
                                      format_func=lambda k: k)
    ly_fraction = frac_options[ly_fraction]

    ly_dir = st.sidebar.radio("Radial direction (time →)", ["bottom→top", "top→bottom"])

    # Add this in the sidebar, after c_ni_target input
    st.sidebar.header("Special Modes")
    mode = st.sidebar.radio(
        "Visualization Mode",
        options=[
            "Standard Cu-Ni Coupled Diffusion",
            "Pure Cu Diffusion (Cu substrate bottom, air top)",
            "Pure Ni Diffusion (Ni substrate bottom, air top)"
        ],
        index=0
    )

    if "Pure Cu" in mode:
        c_ni_target = 1e-12   # Don't show 0.0 in UI — just forces snapping to real pure-Cu cases
        st.sidebar.success("Pure Cu mode active (using real self-diffusion files)")
    elif "Pure Ni" in mode:
        c_cu_target = 1e-12
        st.sidebar.success("Pure Ni mode active (using real self-diffusion files)")

    # Parameters for session
    parameters = {
        'c_cu_target': c_cu_target,
        'c_ni_target': c_ni_target,
        'ly_fraction': ly_fraction,
        'ly_step': ly_step,
        'time_log_scale': time_log_scale,
        'conc_log_scale': conc_log_scale,
        'ly_dir': ly_dir,
        'cmap_cu': cmap_cu,
        'cmap_ni': cmap_ni
    }

    # ---- Process based on session selection ----
    if selected_session == "Create New Session":
        # Generate new session
        session_id = generate_session_id(parameters)
        st.sidebar.success(f"New Session: {session_id}")
        
        # ---- build matrices with time scaling (optimized for your real pure files) ----
        with st.spinner("Computing sunburst matrices using real pure-diffusion cases..."):
            if mode == "Pure Cu Diffusion (Cu substrate bottom, air top)":
                # Use extremely small C_Ni to snap to your real cu_selfdiffusion cases
                cu_mat, ni_mat, times = build_sunburst_matrices(
                    sols, params, interpolator,
                    c_cu_target, 1e-12, ly_fraction, LY_SPOKES, time_log_scale
                )
                st.success("Pure Cu Mode: Using real cu_selfdiffusion_ly_50/90 files")

            elif mode == "Pure Ni Diffusion (Ni substrate bottom, air top)":
                cu_mat, ni_mat, times = build_sunburst_matrices(
                    sols, params, interpolator,
                    1e-12, c_ni_target, ly_fraction, LY_SPOKES, time_log_scale
                )
                st.success("Pure Ni Mode: Using real ni_selfdiffusion_ly_50/90 files")

            else:  # Standard coupled
                cu_mat, ni_mat, times = build_sunburst_matrices(
                    sols, params, interpolator,
                    c_cu_target, c_ni_target, ly_fraction, LY_SPOKES, time_log_scale
                )
        
        # Save to database
        save_sunburst_data(session_id, parameters, cu_mat, ni_mat, times, LY_SPOKES)
        st.sidebar.info("✅ Sunburst data saved to database")
        
    else:
        # Load existing session
        session_id = selected_session.split(" (")[0]  # Extract session ID from display string
        loaded_data = load_sunburst_data(session_id)
        
        if loaded_data:
            saved_params, cu_mat, ni_mat, times, LY_SPOKES = loaded_data
            st.sidebar.success(f"✅ Loaded session: {session_id}")
            
            # Update UI with saved parameters
            st.sidebar.info("Using saved parameters from database")
        else:
            st.sidebar.error("❌ Failed to load session data")
            return

    with st.expander("Pure Diffusion Modes Explanation"):
        st.markdown("""
        ### New Modes Added:
        - **Pure Cu**: Only Cu diffuses from bottom → top is no-flux (air)
        - **Pure Ni**: Only Ni diffuses from bottom → top is no-flux (air)
        
        These represent **uncoupled diffusion** — ideal for isolating species behavior and validating against analytical 1D diffusion solutions.
        
        Expected trend: Smaller Ly → faster saturation → higher center concentration (opposite of coupled case!)
        """)

    # ---- enhanced sunburst figures ----
    st.subheader("🌅 Enhanced Sunburst Charts")
    col1, col2 = st.columns(2)
    
    # Inside the sunburst plotting section
    mode_suffix = ""
    if mode != "Standard Cu-Ni Coupled Diffusion":
        mode_suffix = mode.split(" (")[0].replace("Pure ", "Only ")

    with col1:
        title_cu = f"Cu Centre Concentration ({mode_suffix})" if "Pure Cu" in mode or "Standard" in mode else "Cu (inactive)"
        fig_cu, png_cu, pdf_cu = plot_sunburst(
            cu_mat, title_cu, cmap_cu,
            0, c_cu_target if "Ni" not in mode else 1e-9, 
            conc_log_scale, time_log_scale, 
            ly_dir, f"sunburst_cu_{session_id}_{mode.replace(' ', '_')[:20]}", times, LY_SPOKES
        )
        st.pyplot(fig_cu)
        with open(png_cu,"rb") as f:
            st.download_button("Download Cu PNG", f, f"sunburst_cu_{session_id}.png", "image/png")
        with open(pdf_cu,"rb") as f:
            st.download_button("Download Cu PDF", f, f"sunburst_cu_{session_id}.pdf", "application/pdf")

    with col2:
        title_ni = f"Ni Centre Concentration ({mode_suffix})" if "Pure Ni" in mode or "Standard" in mode else "Ni (inactive)"
        fig_ni, png_ni, pdf_ni = plot_sunburst(
            ni_mat, title_ni, cmap_ni,
            0, c_ni_target if "Cu" not in mode else 1e-9,
            conc_log_scale, time_log_scale,
            ly_dir, f"sunburst_ni_{session_id}_{mode.replace(' ', '_')[:20]}", times, LY_SPOKES
        )
        st.pyplot(fig_ni)
        with open(png_ni,"rb") as f:
            st.download_button("Download Ni PNG", f, f"sunburst_ni_{session_id}.png", "image/png")
        with open(pdf_ni,"rb") as f:
            st.download_button("Download Ni PDF", f, f"sunburst_ni_{session_id}.pdf", "application/pdf")

    if "Pure" in mode:
        with st.expander("Pure Diffusion Cases Used", expanded=True):
            if "Cu" in mode:
                st.success("Pure Cu diffusion (self-diffusion only)\n"
                           "Using: solution_cu_selfdiffusion_ly_50.0 & 90.0")
            else:
                st.success("Pure Ni diffusion (self-diffusion only)\n"
                           "Using: solution_ni_selfdiffusion_ly_50.0 & 90.0")
            st.info("These are real PINN solutions with one species exactly zero → "
                    "physically correct, no extrapolation!")

    # ---- separate radar charts FROM DATABASE ----
    st.divider()
    st.subheader("📡 Radar Charts (From Database)")
    
    st.info(f"Creating radar charts from saved session: **{session_id}**")
    
    t_idx = st.slider("Select Time Index", 0, len(times)-1, len(times)//2, 
                     help="Choose time point for radar snapshot")
    t_val = times[t_idx]

    # Extract data from database-loaded matrices
    cu_row = cu_mat[t_idx, :]
    ni_row = ni_mat[t_idx, :]

    radar_col1, radar_col2 = st.columns(2)
    
    with radar_col1:
        fig_radar_cu, png_rcu, pdf_rcu = plot_radar_single(
            cu_row, "Cu", t_val, f"radar_cu_{session_id}_t{t_val:.0f}", LY_SPOKES, 
            show_labels=show_radar_labels, show_radial_labels=show_radial_labels
        )
        st.pyplot(fig_radar_cu)
        with open(png_rcu,"rb") as f:
            st.download_button("Cu Radar PNG", f, f"radar_cu_{session_id}_t{t_val:.0f}.png", "image/png")

    with radar_col2:
        fig_radar_ni, png_rni, pdf_rni = plot_radar_single(
            ni_row, "Ni", t_val, f"radar_ni_{session_id}_t{t_val:.0f}", LY_SPOKES, 
            show_labels=show_radar_labels, show_radial_labels=show_radial_labels
        )
        st.pyplot(fig_radar_ni)
        with open(png_rni,"rb") as f:
            st.download_button("Ni Radar PNG", f, f"radar_ni_{session_id}_t{t_val:.0f}.png", "image/png")

    # ---- data validation section ----
    with st.expander("🔍 Data Validation & Database Info"):
        st.write("### Cross-check between Sunburst and Radar Data")
        
        # Show that radar data matches sunburst data for selected time
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Cu Data at t = {t_val:.1f}s:**")
            for i, ly in enumerate(LY_SPOKES):
                st.write(f"Ly = {ly}µm: {cu_row[i]:.2e} mol/cc")
        
        with col2:
            st.write(f"**Ni Data at t = {t_val:.1f}s:**")
            for i, ly in enumerate(LY_SPOKES):
                st.write(f"Ly = {ly}µm: {ni_row[i]:.2e} mol/cc")
        
        # Database information
        st.write("### Database Information")
        st.write(f"**Session ID:** {session_id}")
        st.write(f"**Data Shape:** Cu {cu_mat.shape}, Ni {ni_mat.shape}")
        st.write(f"**Time Points:** {len(times)}")
        st.write(f"**Ly Spokes:** {len(LY_SPOKES)}")
        
        # Add a simple plot to verify the pattern
        st.write("### Concentration vs Ly (Linear Plot for Verification)")
        fig_verify, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(LY_SPOKES, cu_row, 'ro-', linewidth=3, markersize=8, label='Cu')
        ax1.set_xlabel('Ly (µm)')
        ax1.set_ylabel('Cu Concentration (mol/cc)')
        ax1.set_title(f'Cu at t = {t_val:.1f}s')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(LY_SPOKES, ni_row, 'bs-', linewidth=3, markersize=8, label='Ni')
        ax2.set_xlabel('Ly (µm)')
        ax2.set_ylabel('Ni Concentration (mol/cc)')
        ax2.set_title(f'Ni at t = {t_val:.1f}s')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        st.pyplot(fig_verify)

    st.caption(f"Note: Sampling at x = Lx/2, y = Ly × {ly_fraction} | Session: {session_id}")

if __name__ == "__main__":
    main()
