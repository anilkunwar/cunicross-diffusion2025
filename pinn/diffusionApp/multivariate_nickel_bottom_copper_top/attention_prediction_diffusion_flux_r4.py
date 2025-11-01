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

# Configure Matplotlib for publication-quality figures
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

# Available colormaps
COLORMAPS = [
    "viridis", "plasma", "inferno", "magma", "cividis",
    "Greys", "Purples", "Blues", "Greens", "Oranges", "Reds",
    "YlOrBr", "YlOrRd", "OrRd", "PuRd", "RdPu", "BuPu", "GnBu",
    "PuBu", "YlGnBu", "PuBuGn", "BuGn", "YlGn",
    "cubehelix", "binary", "gist_yarg", "gist_gray", "gray", "bone",
    "pink", "spring", "summer", "autumn", "winter",
    "PiYG", "PRGn", "BrBG", "PuOr", "RdGy", "RdBu", "RdYlBu", "RdYlGn",
    "Spectral", "coolwarm", "bwr", "seismic",
    "twilight", "twilight_shifted", "hsv",
    "Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1", "Set2", "Set3",
    "tab10", "tab20", "tab20b", "tab20c",
    "flag", "prism", "ocean", "gist_earth", "terrain", "gist_stern", "gnuplot",
    "gnuplot2", "CMRmap", "cubehelix", "brg", "gist_rainbow", "rainbow",
    "jet", "nipy_spectral", "gist_ncar",
    "viridis_r", "plasma_r", "inferno_r", "magma_r", "cividis_r", "Greys_r",
    "Purples_r", "Blues_r", "Greens_r", "Oranges_r", "Reds_r", "YlOrBr_r",
    "YlOrRd_r", "OrRd_r", "PuRd_r", "RdPu_r", "BuPu_r", "GnBu_r", "PuBu_r",
    "YlGnBu_r", "PuBuGn_r", "BuGn_r", "YlGn_r", "twilight_r", "twilight_shifted_r",
    "hsv_r", "Spectral_r", "coolwarm_r", "bwr_r", "seismic_r", "RdBu_r",
    "PiYG_r", "PRGn_r", "BrBG_r", "PuOr_r", "RdGy_r", "RdYlBu_r", "RdYlGn_r",
]

@st.cache_data
def load_solutions(solution_dir):
    solutions = []
    params_list = []
    load_logs = []
    lys = []
    c_cus = []
    c_nis = []
    for fname in os.listdir(solution_dir):
        if fname.endswith(".pkl"):
            try:
                with open(os.path.join(solution_dir, fname), "rb") as f:
                    sol = pickle.load(f)
                required_keys = ['params', 'X', 'Y', 'c1_preds', 'c2_preds', 'times']
                if all(key in sol for key in required_keys):
                    if (np.any(np.isnan(sol['c1_preds'])) or np.any(np.isnan(sol['c2_preds'])) or
                            np.all(sol['c1_preds'] == 0) or np.all(sol['c2_preds'] == 0)):
                        load_logs.append(f"{fname}: Skipped - Invalid data (NaNs or all zeros).")
                        continue
                    c1_min, c1_max = np.min(sol['c1_preds'][0]), np.max(sol['c1_preds'][0])
                    c2_min, c2_max = np.min(sol['c2_preds'][0]), np.max(sol['c2_preds'][0])
                    solutions.append(sol)
                    param_tuple = (sol['params']['Ly'], sol['params']['C_Cu'], sol['params']['C_Ni'])
                    params_list.append(param_tuple)
                    lys.append(sol['params']['Ly'])
                    c_cus.append(sol['params']['C_Cu'])
                    c_nis.append(sol['params']['C_Ni'])
                    # Parse diffusion type
                    match = re.match(r"solution_(\w+)_ly_([\d.]+)_tmax_([\d.]+)\.pkl", fname)
                    if match:
                        raw_type, _, _ = match.groups()
                        type_map = {
                            'cross': 'crossdiffusion',
                            'cu_self': 'cu_selfdiffusion',
                            'ni_self': 'ni_selfdiffusion'
                        }
                        diff_type = type_map.get(raw_type.lower(), 'crossdiffusion')
                        sol['diffusion_type'] = diff_type
                    else:
                        sol['diffusion_type'] = 'crossdiffusion'
                    load_logs.append(
                        f"{fname}: Loaded. Cu: {c1_min:.2e} to {c1_max:.2e}, Ni: {c2_min:.2e} to {c2_max:.2e}, "
                        f"Ly={param_tuple[0]:.1f}, C_Cu={param_tuple[1]:.1e}, C_Ni={param_tuple[2]:.1e}, Type={sol['diffusion_type']}"
                    )
                else:
                    missing_keys = [key for key in required_keys if key not in sol]
                    load_logs.append(f"{fname}: Skipped - Missing keys: {missing_keys}")
            except Exception as e:
                load_logs.append(f"{fname}: Skipped - Failed to load: {str(e)}")
    if len(solutions) < 1:
        load_logs.append("Error: No valid solutions loaded. Interpolation will fail.")
    else:
        load_logs.append(f"Loaded {len(solutions)} solutions. Expected 32.")
    return solutions, params_list, lys, c_cus, c_nis, load_logs

class MultiParamAttentionInterpolator(nn.Module):
    def __init__(self, sigma=0.2, num_heads=4, d_head=8):
        super().__init__()
        self.sigma = sigma
        self.num_heads = num_heads
        self.d_head = d_head
        self.W_q = nn.Linear(3, self.num_heads * self.d_head)
        self.W_k = nn.Linear(3, self.num_heads * self.d_head)

    def forward(self, solutions, params_list, ly_target, c_cu_target, c_ni_target):
        if not solutions or not params_list:
            raise ValueError("No solutions or parameters available for interpolation.")

        lys = np.array([p[0] for p in params_list])
        c_cus = np.array([p[1] for p in params_list])
        c_nis = np.array([p[2] for p in params_list])

        ly_norm = (lys - 30.0) / (120.0 - 30.0)
        c_cu_norm = (c_cus - 0.0) / (2.9e-3 - 0.0)
        c_ni_norm = (c_nis - 0.0) / (1.8e-3 - 0.0)
        target_ly_norm = (ly_target - 30.0) / (120.0 - 30.0)
        target_c_cu_norm = (c_cu_target - 0.0) / (2.9e-3 - 0.0)
        target_c_ni_norm = (c_ni_target - 0.0) / (1.8e-3 - 0.0)

        params_tensor = torch.tensor(np.stack([ly_norm, c_cu_norm, c_ni_norm], axis=1), dtype=torch.float32)
        target_params_tensor = torch.tensor([[target_ly_norm, target_c_cu_norm, target_c_ni_norm]], dtype=torch.float32)

        queries = self.W_q(target_params_tensor)
        keys = self.W_k(params_tensor)

        queries = queries.view(1, self.num_heads, self.d_head)
        keys = keys.view(len(params_list), self.num_heads, self.d_head)

        attn_logits = torch.einsum('nhd,mhd->nmh', keys, queries) / np.sqrt(self.d_head)
        attn_weights = torch.softmax(attn_logits, dim=0)
        attn_weights = attn_weights.mean(dim=2).squeeze(1)

        scaled_distances = torch.sqrt(
            ((torch.tensor(ly_norm) - target_ly_norm) / self.sigma)**2 +
            ((torch.tensor(c_cu_norm) - target_c_cu_norm) / self.sigma)**2 +
            ((torch.tensor(c_ni_norm) - target_c_ni_norm) / self.sigma)**2
        )
        spatial_weights = torch.exp(-scaled_distances**2 / 2)
        spatial_weights /= spatial_weights.sum()

        combined_weights = attn_weights * spatial_weights
        combined_weights /= combined_weights.sum()

        return self._physics_aware_interpolation(solutions, combined_weights.detach().numpy(), ly_target, c_cu_target, c_ni_target)

    def _physics_aware_interpolation(self, solutions, weights, ly_target, c_cu_target, c_ni_target):
        Lx = solutions[0]['params']['Lx']
        t_max = solutions[0]['params']['t_max']
        x_coords = np.linspace(0, Lx, 50)
        y_coords = np.linspace(0, ly_target, 50)
        times = np.linspace(0, t_max, 50)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        c1_interp = np.zeros((len(times), 50, 50))
        c2_interp = np.zeros((len(times), 50, 50))

        for t_idx in range(len(times)):
            for sol, weight in zip(solutions, weights):
                scale_factor = ly_target / sol['params']['Ly']
                Y_scaled = sol['Y'][0, :] * scale_factor
                interp_c1 = RegularGridInterpolator(
                    (sol['X'][:, 0], Y_scaled), sol['c1_preds'][t_idx],
                    method='linear', bounds_error=False, fill_value=0
                )
                interp_c2 = RegularGridInterpolator(
                    (sol['X'][:, 0], Y_scaled), sol['c2_preds'][t_idx],
                    method='linear', bounds_error=False, fill_value=0
                )
                points = np.stack([X.flatten(), Y.flatten()], axis=1)
                c1_interp[t_idx] += weight * interp_c1(points).reshape(50, 50)
                c2_interp[t_idx] += weight * interp_c2(points).reshape(50, 50)

        c1_interp[:, :, 0] = c_cu_target
        c2_interp[:, :, -1] = c_ni_target

        param_set = solutions[0]['params'].copy()
        param_set['Ly'] = ly_target
        param_set['C_Cu'] = c_cu_target
        param_set['C_Ni'] = c_ni_target

        return {
            'params': param_set,
            'X': X,
            'Y': Y,
            'c1_preds': list(c1_interp),
            'c2_preds': list(c2_interp),
            'times': times,
            'interpolated': True,
            'attention_weights': weights.tolist()
        }

@st.cache_data
def load_and_interpolate_solution(solutions, params_list, ly_target, c_cu_target, c_ni_target, tolerance_ly=0.1, tolerance_c=1e-5):
    for sol, params in zip(solutions, params_list):
        ly, c_cu, c_ni = params
        if (abs(ly - ly_target) < tolerance_ly and
                abs(c_cu - c_cu_target) < tolerance_c and
                abs(c_ni - c_ni_target) < tolerance_c):
            sol['interpolated'] = False
            return sol
    if not solutions:
        raise ValueError("No solutions available for interpolation.")
    interpolator = MultiParamAttentionInterpolator(sigma=0.2)
    return interpolator(solutions, params_list, ly_target, c_cu_target, c_ni_target)

# [Include ALL plotting functions: plot_2d_concentration, plot_centerline_curves, etc.]
# ... (same as in your original code — keep them unchanged)

# === OPTIMAL LENGTH FOR MAX/MIN UPHILL DIFFUSION (FIXED) ===
st.subheader("Optimal Domain Length for Uphill Diffusion")

metric = st.selectbox("Uphill Metric", [
    "Max positive J·∇c (Cu)", "Max positive J·∇c (Ni)",
    "Total uphill intensity (Cu)", "Total uphill intensity (Ni)",
    "Uphill fraction (Cu)", "Uphill fraction (Ni)"
], index=2)
extremum = st.radio("Find", ["Maximum", "Minimum"])
opt_time_idx = st.slider("Time for Optimization", 0, len(solutions[0]['times'])-1, len(solutions[0]['times'])//2, key="opt_time")

# Precompute fluxes
st.info("Precomputing fluxes and gradients for all solutions...")
progress_bar = st.progress(0)
valid_solutions = []
for i, sol in enumerate(solutions):
    progress_bar.progress((i + 1) / len(solutions))
    try:
        if 'J1_preds' not in sol:
            X = sol['X'][:, 0] if sol['X'].ndim == 2 else sol['X']
            Y = sol['Y'][0, :] if sol['Y'].ndim == 2 else sol['Y']
            J1, J2, grad1, grad2 = compute_fluxes_and_grads(
                sol['c1_preds'], sol['c2_preds'], X, Y, sol['params']
            )
            sol['J1_preds'] = J1
            sol['J2_preds'] = J2
            sol['grad_c1_y'] = grad1
            sol['grad_c2_y'] = grad2
        valid_solutions.append(sol)
    except Exception as e:
        st.warning(f"Failed to compute fluxes for Ly={sol['params']['Ly']}: {e}")
progress_bar.empty()

if not valid_solutions:
    st.error("No solutions with valid fluxes. Check your data.")
    st.stop()

# Extract metric
metrics = []
lys_valid = []
c_cu_list = []
c_ni_list = []

metric_map = {
    "Max positive J·∇c (Cu)": lambda s: detect_uphill(s, opt_time_idx)[4],
    "Max positive J·∇c (Ni)": lambda s: detect_uphill(s, opt_time_idx)[5],
    "Total uphill intensity (Cu)": lambda s: detect_uphill(s, opt_time_idx)[10],
    "Total uphill intensity (Ni)": lambda s: detect_uphill(s, opt_time_idx)[11],
    "Uphill fraction (Cu)": lambda s: detect_uphill(s, opt_time_idx)[6],
    "Uphill fraction (Ni)": lambda s: detect_uphill(s, opt_time_idx)[7],
}

for sol in valid_solutions:
    try:
        value = metric_map[metric](sol)
        metrics.append(value)
        lys_valid.append(sol['params']['Ly'])
        c_cu_list.append(sol['params']['C_Cu'])
        c_ni_list.append(sol['params']['C_Ni'])
    except:
        continue

if not metrics:
    st.warning("No valid data for optimization.")
else:
    metrics = np.array(metrics)
    lys_valid = np.array(lys_valid)
    idx_opt = np.argmax(metrics) if extremum == "Maximum" else np.argmin(metrics)
    Ly_opt = lys_valid[idx_opt]
    val_opt = metrics[idx_opt]
    c_cu_opt = c_cu_list[idx_opt]
    c_ni_opt = c_ni_list[idx_opt]

    st.success(f"**Optimal \(L_y\) = {Ly_opt:.1f} μm**  \n"
               f"→ **{metric} = {val_opt:.3e}** ({extremum})  \n"
               f"at \(C_{{Cu}} = {c_cu_opt:.1e}\), \(C_{{Ni}} = {c_ni_opt:.1e}\)")

    # Plot metric vs Ly
    fig_opt, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(lys_valid, metrics, c=c_cu_list, cmap='viridis', alpha=0.7, edgecolors='k', linewidth=0.5)
    ax.plot(Ly_opt, val_opt, 'r*', markersize=15, label=f'Optimal ({extremum})')
    ax.set_xlabel('$L_y$ (μm)')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} vs Domain Length')
    ax.grid(True, alpha=0.3)
    ax.legend()
    cbar = fig_opt.colorbar(scatter, ax=ax)
    cbar.set_label('$C_{Cu}$ (mol/cc)')
    st.pyplot(fig_opt)

    if st.button(f"Show Uphill Heatmap at \(L_y\) = {Ly_opt:.1f} μm"):
        with st.spinner("Interpolating optimal solution..."):
            try:
                opt_sol = load_and_interpolate_solution(
                    solutions, params_list, Ly_opt, c_cu_opt, c_ni_opt
                )
                X = opt_sol['X'][:, 0] if opt_sol['X'].ndim == 2 else opt_sol['X']
                Y = opt_sol['Y'][0, :] if opt_sol['Y'].ndim == 2 else opt_sol['Y']
                J1, J2, grad1, grad2 = compute_fluxes_and_grads(
                    opt_sol['c1_preds'], opt_sol['c2_preds'], X, Y, opt_sol['params']
                )
                opt_sol['J1_preds'], opt_sol['J2_preds'] = J1, J2
                opt_sol['grad_c1_y'], opt_sol['grad_c2_y'] = grad1, grad2

                fig_hm, *_ = plot_uphill_heatmap(opt_sol, opt_time_idx, cmap='viridis')
                st.pyplot(fig_hm)
            except Exception as e:
                st.error(f"Failed to interpolate optimal solution: {e}")

if __name__ == "__main__":
    main()
