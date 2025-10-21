import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from io import BytesIO
import re

# ------------------------------
# Publication-style Matplotlib Streamlit App
# ------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 14,
    "figure.dpi": 300,  # higher DPI for publication quality
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
})

SOLUTION_DIR = os.path.join(os.path.dirname(__file__), "pinn_solutions")
os.makedirs(SOLUTION_DIR, exist_ok=True)
DIFFUSION_TYPES = ['crossdiffusion', 'cu_selfdiffusion', 'ni_selfdiffusion']

# ------------------------------
# Utility: get many colormaps (matplotlib provides >50)
# ------------------------------
ALL_CMAPS = sorted(list(mpl.colormaps()))

# ------------------------------
# I/O: load solutions (unchanged)
# ------------------------------
@st.cache_data
def load_solutions(solution_dir):
    solutions = []
    for fname in os.listdir(solution_dir):
        if fname.endswith(".pkl"):
            try:
                with open(os.path.join(solution_dir, fname), "rb") as f:
                    sol = pickle.load(f)
                match = re.match(r"solution_(\w+)_ly_([\d.]+)_tmax_([\d.]+)\.pkl", fname)
                if not match:
                    continue
                raw_type, ly_val, _ = match.groups()
                type_map = {
                    'cross': 'crossdiffusion',
                    'cu_self': 'cu_selfdiffusion',
                    'ni_self': 'ni_selfdiffusion'
                }
                diff_type = type_map.get(raw_type.lower(), raw_type)
                sol.update({
                    'diffusion_type': diff_type,
                    'Ly_parsed': float(ly_val),
                    'filename': fname
                })
                solutions.append(sol)
            except Exception:
                continue
    return solutions

# ------------------------------
# Compute fluxes and gradients (same as user)
# ------------------------------
def compute_fluxes_and_grads(c1_preds, c2_preds, x_coords, y_coords, params):
    D11, D12, D21, D22 = params['D11'], params['D12'], params['D21'], params['D22']
    dx = x_coords[1] - x_coords[0]
    dy = y_coords[1] - y_coords[0]

    J1_preds, J2_preds, grad_c1_y, grad_c2_y = [], [], [], []
    for c1, c2 in zip(c1_preds, c2_preds):
        grad_c1_y_i = np.gradient(c1, dy, axis=0)
        grad_c2_y_i = np.gradient(c2, dy, axis=0)
        J1_y = -(D11 * grad_c1_y_i + D12 * grad_c2_y_i)
        J2_y = -(D21 * grad_c1_y_i + D22 * grad_c2_y_i)
        J1_preds.append([None, J1_y])
        J2_preds.append([None, J2_y])
        grad_c1_y.append(grad_c1_y_i)
        grad_c2_y.append(grad_c2_y_i)

    return J1_preds, J2_preds, grad_c1_y, grad_c2_y

# ------------------------------
# Detect uphill (same logic)
# ------------------------------
def detect_uphill(solution, time_index):
    J1_y = solution['J1_preds'][time_index][1]
    grad_c1_y = solution['grad_c1_y'][time_index]
    J2_y = solution['J2_preds'][time_index][1]
    grad_c2_y = solution['grad_c2_y'][time_index]

    prod_cu = J1_y * grad_c1_y
    prod_ni = J2_y * grad_c2_y

    uphill_cu = prod_cu > 0
    uphill_ni = prod_ni > 0

    uphill_prod_cu_pos = np.where(uphill_cu, prod_cu, 0.0)
    uphill_prod_ni_pos = np.where(uphill_ni, prod_ni, 0.0)

    max_pos_cu = float(np.max(uphill_prod_cu_pos)) if np.any(uphill_cu) else 0.0
    max_pos_ni = float(np.max(uphill_prod_ni_pos)) if np.any(uphill_ni) else 0.0

    total_cells = prod_cu.size
    frac_uphill_cu = float(np.count_nonzero(uphill_cu) / total_cells)
    frac_uphill_ni = float(np.count_nonzero(uphill_ni) / total_cells)

    return (uphill_cu, uphill_ni,
            uphill_prod_cu_pos, uphill_prod_ni_pos,
            max_pos_cu, max_pos_ni,
            frac_uphill_cu, frac_uphill_ni)

# ------------------------------
# Matplotlib plotting utilities (publication-ready)
# All figures can be edited via sidebar controls and downloaded
# ------------------------------

def fig_to_bytes(fig, fmt='png'):
    buf = BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches='tight')
    buf.seek(0)
    return buf


def plot_flux_vs_gradient_matplotlib(solution, time_index,
                                     figsize=(6,3), marker_size=12, linewidth=1.2,
                                     label_fontsize=12, title_fontsize=14, scatter_alpha=0.6,
                                     marker_edgewidth=0.2):
    J1_y = solution['J1_preds'][time_index][1].flatten()
    J2_y = solution['J2_preds'][time_index][1].flatten()
    grad_c1_y = solution['grad_c1_y'][time_index].flatten()
    grad_c2_y = solution['grad_c2_y'][time_index].flatten()

    uphill_cu = J1_y * grad_c1_y > 0
    uphill_ni = J2_y * grad_c2_y > 0

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot Cu
    axes[0].scatter(grad_c1_y, J1_y, s=marker_size, alpha=scatter_alpha, edgecolors='none', label='Cu (all)')
    if uphill_cu.any():
        axes[0].scatter(grad_c1_y[uphill_cu], J1_y[uphill_cu], s=marker_size*1.1,
                        edgecolors='k', linewidths=marker_edgewidth, label='Uphill (Cu)')
    
    axes[0].set_xlabel('∇c (y)', fontsize=label_fontsize)
    axes[0].set_ylabel('J_y', fontsize=label_fontsize)
    axes[0].set_title('Cu Flux vs ∇c', fontsize=title_fontsize)
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.3)
    
    # Dynamic tick spacing to avoid overlap (FIXED)
    from matplotlib.ticker import MaxNLocator
    axes[0].xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
    axes[0].yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))

    # Plot Ni
    axes[1].scatter(grad_c2_y, J2_y, s=marker_size, alpha=scatter_alpha, edgecolors='none', label='Ni (all)')
    if uphill_ni.any():
        axes[1].scatter(grad_c2_y[uphill_ni], J2_y[uphill_ni], s=marker_size*1.1,
                        edgecolors='k', linewidths=marker_edgewidth, label='Uphill (Ni)')
    
    axes[1].set_xlabel('∇c (y)', fontsize=label_fontsize)
    axes[1].set_ylabel('J_y', fontsize=label_fontsize)
    axes[1].set_title('Ni Flux vs ∇c', fontsize=title_fontsize)
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.3)
    
    # Dynamic tick spacing for Ni subplot (FIXED)
    axes[1].xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
    axes[1].yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))

    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=label_fontsize-2)
        ax.legend(fontsize=label_fontsize-2)

    fig.tight_layout()
    return fig




def plot_uphill_heatmap_matplotlib(solution, time_index, cmap='viridis', vmin=None, vmax=None,
                                   figsize=(10,4), colorbar=True, cbar_label='J·∇c',
                                   label_fontsize=12, title_fontsize=14, downsample=1,
                                   cbar_pad=0.05, cbar_width=0.03):
    x_coords = solution['X'][:,0] if solution['X'].ndim==2 else solution['X']
    y_coords = solution['Y'][0,:] if solution['Y'].ndim==2 else solution['Y']
    t_val = solution['times'][time_index]
    (uphill_cu, uphill_ni,
     uphill_prod_cu_pos, uphill_prod_ni_pos,
     max_pos_cu, max_pos_ni,
     frac_cu, frac_ni) = detect_uphill(solution, time_index)

    # optionally downsample for speed
    z1 = uphill_prod_cu_pos[::downsample, ::downsample]
    z2 = uphill_prod_ni_pos[::downsample, ::downsample]
    x_ds = x_coords[::downsample]
    y_ds = y_coords[::downsample]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, z, name, max_val in zip(axes, [z1, z2], ['Cu', 'Ni'], [max_pos_cu, max_pos_ni]):
        im = ax.imshow(z, origin='lower', aspect='auto',
                       extent=(x_ds[0], x_ds[-1], y_ds[0], y_ds[-1]),
                       cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f'{name} Uphill (max={max_val:.3e})', fontsize=title_fontsize)
        ax.set_xlabel('x (μm)', fontsize=label_fontsize)
        if name == 'Cu':
            ax.set_ylabel('y (μm)', fontsize=label_fontsize)
        else:
            ax.set_ylabel('')

        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size=cbar_width, pad=cbar_pad)
            fig.colorbar(im, cax=cax, label=cbar_label).ax.tick_params(labelsize=label_fontsize-2)

    fig.suptitle(f'Uphill Diffusion (positive J·∇c) @ t={t_val:.2f}s', fontsize=title_fontsize+1)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig, max_pos_cu, max_pos_ni, frac_cu, frac_ni



def plot_uphill_over_time_matplotlib(solution, figsize=(8,3), linewidth=1.6, marker_size=6,
                                     label_fontsize=12, title_fontsize=14):
    times = np.array(solution['times'])
    max_pos_cu_list = []
    max_pos_ni_list = []
    frac_cu_list = []
    frac_ni_list = []

    Nt = len(times)
    for t_idx in range(Nt):
        (_, _, _, _, max_pos_cu, max_pos_ni, frac_cu, frac_ni) = detect_uphill(solution, t_idx)
        max_pos_cu_list.append(max_pos_cu)
        max_pos_ni_list.append(max_pos_ni)
        frac_cu_list.append(frac_cu)
        frac_ni_list.append(frac_ni)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(times, max_pos_cu_list, marker='o', markersize=marker_size, linewidth=linewidth, label='Max positive J·∇c (Cu)')
    ax.plot(times, max_pos_ni_list, marker='s', markersize=marker_size, linewidth=linewidth, label='Max positive J·∇c (Ni)')
    ax.set_xlabel('Time (s)', fontsize=label_fontsize)
    ax.set_ylabel('Max positive J·∇c', fontsize=label_fontsize)
    ax.set_title('Temporal Evolution of Global Positive Max (J·∇c)', fontsize=title_fontsize)
    ax.legend(fontsize=label_fontsize-2)
    ax.grid(True, which='both', linestyle='--', linewidth=0.3)
    fig.tight_layout()

    # small subplot for fraction evolution
    fig2, ax2 = plt.subplots(figsize=(8,2.5))
    ax2.plot(times, frac_cu_list, label='Uphill fraction Cu', linewidth=linewidth)
    ax2.plot(times, frac_ni_list, label='Uphill fraction Ni', linewidth=linewidth)
    ax2.set_xlabel('Time (s)', fontsize=label_fontsize)
    ax2.set_ylabel('Fraction of grid points', fontsize=label_fontsize)
    ax2.set_title('Uphill fraction vs Time', fontsize=title_fontsize-2)
    ax2.legend(fontsize=label_fontsize-2)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.3)
    fig2.tight_layout()

    return fig, fig2

# ------------------------------
# Summary DataFrame across all solutions (unchanged)
# ------------------------------
@st.cache_data
def compute_summary_dataframe(all_solutions, time_index_for_summary=0):
    rows = []
    for s in all_solutions:
        if 'J1_preds' not in s:
            J1, J2, grad_c1, grad_c2 = compute_fluxes_and_grads(s['c1_preds'], s['c2_preds'], s['X'][:,0], s['Y'][0,:], s['params'])
            s.update({'J1_preds': J1, 'J2_preds': J2, 'grad_c1_y': grad_c1, 'grad_c2_y': grad_c2})
        try:
            (_, _, _, _, max_pos_cu, max_pos_ni, frac_cu, frac_ni) = detect_uphill(s, time_index_for_summary)
        except Exception:
            max_pos_cu, max_pos_ni, frac_cu, frac_ni = 0.0, 0.0, 0.0, 0.0
        rows.append({
            "filename": s.get('filename', ''),
            "diffusion_type": s.get('diffusion_type', ''),
            "Ly (μm)": s.get('Ly_parsed', np.nan),
            "time_index": time_index_for_summary,
            "max_pos_JdotGrad_Cu": max_pos_cu,
            "max_pos_JdotGrad_Ni": max_pos_ni,
            "uphill_frac_Cu": frac_cu,
            "uphill_frac_Ni": frac_ni
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(['diffusion_type', 'Ly (μm)'])
    return df

# ------------------------------
# Main App (Streamlit UI for editing visualization parameters)
# ------------------------------
def main():
    st.title("📈 Diffusion Analysis — Matplotlib Publication Figures")

    solutions = load_solutions(SOLUTION_DIR)
    if not solutions:
        st.warning("No PINN solution pickles found in 'pinn_solutions/'. Add .pkl files named like solution_cross_ly_...pkl")
        return

    st.sidebar.header("Controls")
    diff_type = st.sidebar.selectbox("Diffusion Type", DIFFUSION_TYPES)
    ly_values = sorted(set(s['Ly_parsed'] for s in solutions if s['diffusion_type'] == diff_type))
    if not ly_values:
        st.error("No solutions for selected diffusion type.")
        return
    ly_target = st.sidebar.select_slider("Select Ly (μm)", options=ly_values, value=ly_values[0])
    
    # Find maximum time index from all solutions
    max_time_index = max(len(sol['times']) for sol in solutions if 'times' in sol) - 1 if solutions else 49
    
    # Create slider with adjustable step interval
    st.sidebar.markdown("---")
    st.sidebar.subheader("Time Selection")
    step_interval = st.sidebar.selectbox(
        "Time step interval", 
        options=[1, 2, 5, 10, 20],
        index=2,  # Default to step of 5
        help="Increase this value to jump through time points faster"
    )
    
    time_index = st.sidebar.slider(
        "Select Time Index", 
        0, 
        max_time_index, 
        0, 
        step=step_interval,
        help=f"Select time point (0 to {max_time_index}) with step size {step_interval}"
    )

    # pick solution
    solution = next((s for s in solutions if s['diffusion_type']==diff_type and abs(s['Ly_parsed']-ly_target)<1e-8), None)
    if solution is None:
        st.error("No matching solution for the chosen Ly and diffusion type.")
        return

    # compute fluxes if missing
    if 'J1_preds' not in solution:
        J1, J2, grad_c1, grad_c2 = compute_fluxes_and_grads(solution['c1_preds'], solution['c2_preds'], solution['X'][:,0], solution['Y'][0,:], solution['params'])
        solution.update({'J1_preds': J1, 'J2_preds': J2, 'grad_c1_y': grad_c1, 'grad_c2_y': grad_c2})

    st.sidebar.markdown("---")
    st.sidebar.subheader("Figure options")
    fig_width = st.sidebar.number_input("Figure width (in)", value=8.0, min_value=2.0, max_value=20.0, step=0.5)
    fig_height = st.sidebar.number_input("Figure height (in)", value=4.0, min_value=2.0, max_value=20.0, step=0.5)
    cmap_choice = st.sidebar.selectbox("Colormap", ALL_CMAPS, index=ALL_CMAPS.index('viridis') if 'viridis' in ALL_CMAPS else 0)
    vmin = st.sidebar.text_input("Colorbar vmin (leave blank for auto)", value="")
    vmax = st.sidebar.text_input("Colorbar vmax (leave blank for auto)", value="")
    downsample = st.sidebar.slider("Downsample heatmap (integer)", 1, 8, 1)

    marker_size = st.sidebar.slider("Scatter marker size", 1, 50, 8)
    linewidth = st.sidebar.slider("Line width", 0.2, 5.0, 1.2)
    dpi = st.sidebar.selectbox("Output DPI", [150, 200, 300, 600], index=2)

    st.header("1) Flux vs Gradient (matplotlib)")
    fig_fg = plot_flux_vs_gradient_matplotlib(solution, time_index, figsize=(fig_width, fig_height/2),
                                              marker_size=marker_size, linewidth=linewidth)
    st.pyplot(fig_fg)

    # provide download buttons
    png_buf = fig_to_bytes(fig_fg, fmt='png')
    svg_buf = fig_to_bytes(fig_fg, fmt='svg')
    st.download_button("Download figure (PNG)", data=png_buf, file_name="flux_vs_gradient.png", mime="image/png")
    st.download_button("Download figure (SVG)", data=svg_buf, file_name="flux_vs_gradient.svg", mime="image/svg+xml")

    st.header("2) Uphill heatmaps (matplotlib)")
    try:
        vmin_val = float(vmin) if vmin.strip() != '' else None
    except ValueError:
        vmin_val = None
    try:
        vmax_val = float(vmax) if vmax.strip() != '' else None
    except ValueError:
        vmax_val = None

    fig_hm, max_pos_cu, max_pos_ni, frac_cu, frac_ni = plot_uphill_heatmap_matplotlib(solution, time_index,
                                                                                      cmap=cmap_choice, vmin=vmin_val, vmax=vmax_val,
                                                                                      figsize=(fig_width, fig_height), downsample=downsample)
    st.pyplot(fig_hm)
    png_buf2 = fig_to_bytes(fig_hm, fmt='png')
    st.download_button("Download heatmaps (PNG)", data=png_buf2, file_name="uphill_heatmaps.png", mime="image/png")

    st.markdown(f"- **Max (positive) J·∇c (Cu):** {max_pos_cu:.3e}")
    st.markdown(f"- **Max (positive) J·∇c (Ni):** {max_pos_ni:.3e}")
    st.markdown(f"- **Uphill fraction (Cu):** {frac_cu*100:.2f}%")
    st.markdown(f"- **Uphill fraction (Ni):** {frac_ni*100:.2f}%")

    st.header("3) Temporal evolution (matplotlib)")
    fig_time, fig_frac = plot_uphill_over_time_matplotlib(solution, figsize=(fig_width, fig_height/2), linewidth=linewidth, marker_size=max(4, int(marker_size/2)))
    st.pyplot(fig_time)
    st.pyplot(fig_frac)
    st.download_button("Download time evolution (PNG)", data=fig_to_bytes(fig_time, fmt='png'), file_name='time_evolution.png', mime='image/png')

    st.header("4) Summary table and CSV")
    df_summary = compute_summary_dataframe(solutions, time_index_for_summary=time_index)
    st.dataframe(df_summary.style.format({
        "max_pos_JdotGrad_Cu": "{:.3e}",
        "max_pos_JdotGrad_Ni": "{:.3e}",
        "uphill_frac_Cu": "{:.2%}",
        "uphill_frac_Ni": "{:.2%}"
    }), use_container_width=True)
    st.download_button("Download summary CSV", df_summary.to_csv(index=False), file_name="uphill_summary.csv", mime="text/csv")

if __name__ == '__main__':
    main()
