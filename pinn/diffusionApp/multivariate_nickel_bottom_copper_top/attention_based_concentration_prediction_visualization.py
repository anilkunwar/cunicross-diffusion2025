import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.interpolate import RegularGridInterpolator
import pickle
from matplotlib.colors import Normalize
import matplotlib as mpl

# === CONFIG ===
PINN_DIR = "pinn_solutions"
OUTPUT_DIR = "sunburst_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Target Ly values for 8 spokes
LY_VALUES = [30, 40, 50, 60, 70, 80, 90, 100]
T_MAX = 200.0  # r = 1 corresponds to t = 200 s
N_TIME_STEPS = 50  # radial resolution
N_LY = len(LY_VALUES)

# Fixed boundary conditions for interpolation
C_CU_TARGET = 1.0e-3   # mol/cc
C_NI_TARGET = 1.0e-4   # mol/cc

# === 1. Load all PINN solutions ===
def load_solutions(dir_path):
    solutions = []
    params_list = []
    for fname in os.listdir(dir_path):
        if fname.endswith(".pkl"):
            path = os.path.join(dir_path, fname)
            with open(path, "rb") as f:
                sol = pickle.load(f)
            if all(k in sol for k in ['params', 'X', 'Y', 'c1_preds', 'c2_preds', 'times']):
                solutions.append(sol)
                params_list.append((sol['params']['Ly'], sol['params']['C_Cu'], sol['params']['C_Ni']))
    return solutions, params_list

# === 2. Transformer-Inspired Attention Interpolator ===
class AttentionInterpolator(nn.Module):
    def __init__(self, sigma=0.25, num_heads=4, d_head=8):
        super().__init__()
        self.sigma = sigma
        self.num_heads = num_heads
        self.W_q = nn.Linear(3, num_heads * d_head)
        self.W_k = nn.Linear(3, num_heads * d_head)

    def forward(self, solutions, params_list, ly_target, c_cu_target, c_ni_target):
        # Normalize parameters
        lys = torch.tensor([p[0] for p in params_list], dtype=torch.float32)
        c_cus = torch.tensor([p[1] for p in params_list], dtype=torch.float32)
        c_nis = torch.tensor([p[2] for p in params_list], dtype=torch.float32)

        ly_norm = (lys - 30.0) / (120.0 - 30.0)
        c_cu_norm = c_cus / 2.9e-3
        c_ni_norm = c_nis / 1.8e-3

        target_ly_norm = (ly_target - 30.0) / (120.0 - 30.0)
        target_c_cu_norm = c_cu_target / 2.9e-3
        target_c_ni_norm = c_ni_target / 1.8e-3

        params_tensor = torch.stack([ly_norm, c_cu_norm, c_ni_norm], dim=1)  # [N, 3]
        target_tensor = torch.tensor([[target_ly_norm, target_c_cu_norm, target_c_ni_norm]])

        # Multi-head attention
        Q = self.W_q(target_tensor).view(1, self.num_heads, -1)
        K = self.W_k(params_tensor).view(-1, self.num_heads, Q.size(-1))
        attn = torch.einsum('nhd,mhd->nmh', K, Q) / (Q.size(-1) ** 0.5)
        attn_weights = torch.softmax(attn, dim=0).mean(dim=2).squeeze(1)  # [N]

        # Gaussian locality
        dist = torch.sqrt(
            ((ly_norm - target_ly_norm) / self.sigma) ** 2 +
            ((c_cu_norm - target_c_cu_norm) / self.sigma) ** 2 +
            ((c_ni_norm - target_c_ni_norm) / self.sigma) ** 2
        )
        spatial_weights = torch.exp(-dist ** 2 / 2)
        spatial_weights /= spatial_weights.sum()

        # Combine
        weights = attn_weights * spatial_weights
        weights /= weights.sum()

        return self._interpolate(solutions, weights.numpy(), ly_target)

    def _interpolate(self, solutions, weights, ly_target):
        Lx = solutions[0]['params']['Lx']
        t_max = solutions[0]['params']['t_max']
        x = np.linspace(0, Lx, 50)
        y = np.linspace(0, ly_target, 50)
        times = np.linspace(0, min(t_max, T_MAX), N_TIME_STEPS)

        X, Y = np.meshgrid(x, y, indexing='ij')
        c1_interp = np.zeros((len(times), 50, 50))
        c2_interp = np.zeros((len(times), 50, 50))

        for t_idx, t in enumerate(times):
            for sol, w in zip(solutions, weights):
                t_idx_sol = min(int(t / t_max * len(sol['times'])), len(sol['times']) - 1)
                scale = ly_target / sol['params']['Ly']
                Y_scaled = sol['Y'][0, :] * scale

                interp_c1 = RegularGridInterpolator(
                    (sol['X'][:, 0], Y_scaled), sol['c1_preds'][t_idx_sol],
                    method='linear', bounds_error=False, fill_value=0
                )
                interp_c2 = RegularGridInterpolator(
                    (sol['X'][:, 0], Y_scaled), sol['c2_preds'][t_idx_sol],
                    method='linear', bounds_error=False, fill_value=0
                )

                pts = np.stack([X.flatten(), Y.flatten()], axis=1)
                c1_interp[t_idx] += w * interp_c1(pts).reshape(50, 50)
                c2_interp[t_idx] += w * interp_c2(pts).reshape(50, 50)

        # Enforce BCs
        c1_interp[:, :, 0] = C_CU_TARGET
        c2_interp[:, :, -1] = C_NI_TARGET

        return {
            'X': X, 'Y': Y, 'times': times,
            'c1_preds': c1_interp, 'c2_preds': c2_interp,
            'params': {'Ly': ly_target, 'C_Cu': C_CU_TARGET, 'C_Ni': C_NI_TARGET}
        }

# === 3. Extract Centerpoint Concentration ===
def get_center_concentration(solution):
    x_center = solution['params']['Lx'] / 2
    y_center = solution['params']['Ly'] / 2
    x_idx = np.argmin(np.abs(solution['X'][:, 0] - x_center))
    y_idx = np.argmin(np.abs(solution['Y'][0, :] - y_center))
    cu = [c1[x_idx, y_idx] for c1 in solution['c1_preds']]
    ni = [c2[x_idx, y_idx] for c2 in solution['c2_preds']]
    return np.array(cu), np.array(ni)

# === 4. Build Sunburst Data ===
def build_sunburst_matrices(solutions, params_list, interpolator):
    cu_matrix = np.zeros((N_TIME_STEPS, N_LY))
    ni_matrix = np.zeros((N_TIME_STEPS, N_LY))

    for j, ly in enumerate(LY_VALUES):
        print(f"Interpolating Ly = {ly} μm...")
        sol = interpolator(solutions, params_list, ly, C_CU_TARGET, C_NI_TARGET)
        cu_center, ni_center = get_center_concentration(sol)
        cu_matrix[:, j] = cu_center
        ni_matrix[:, j] = ni_center

    return cu_matrix, ni_matrix

# === 5. Sunburst Polar Heatmap ===
def plot_sunburst_polar(data, title, cmap, vmin, vmax, filename):
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Meshgrid in polar: theta = Ly, r = t
    theta = np.linspace(0, 2*np.pi, N_LY + 1)
    r = np.linspace(0, 1, N_TIME_STEPS + 1)
    Theta, R = np.meshgrid(theta, r)

    # Repeat data for closed circle
    Z = np.hstack([data, data[:, 0:1]])

    norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.pcolormesh(Theta, R, Z, cmap=cmap, norm=norm, shading='auto')

    # Spoke labels
    theta_labels = [f"{ly}" for ly in LY_VALUES]
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(theta_labels, fontsize=14, fontweight='bold')

    # Radial labels: time
    r_labels = [0, 50, 100, 150, 200]
    r_ticks = np.array(r_labels) / T_MAX
    ax.set_yticks(r_ticks)
    ax.set_yticklabels([f"{int(t)} s" for t in r_labels], fontsize=12)

    ax.set_ylim(0, 1)
    ax.grid(True, color='white', linewidth=1.5, alpha=0.7)
    ax.set_title(title, fontsize=18, fontweight='bold', pad=30)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Concentration (mol/cc)', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}.pdf"), bbox_inches='tight')
    plt.close()

# === MAIN ===
if __name__ == "__main__":
    print("Loading PINN solutions...")
    solutions, params_list = load_solutions(PINN_DIR)

    print("Initializing attention interpolator...")
    interpolator = AttentionInterpolator(sigma=0.25, num_heads=4)

    print("Building sunburst data...")
    cu_data, ni_data = build_sunburst_matrices(solutions, params_list, interpolator)

    print("Plotting sunburst charts...")
    plot_sunburst_polar(
        cu_data, "Cu Concentration at Center (x=30, y=Ly/2)",
        cmap='plasma', vmin=0, vmax=1.5e-3,
        filename="sunburst_cu_center"
    )
    plot_sunburst_polar(
        ni_data, "Ni Concentration at Center (x=30, y=Ly/2)",
        cmap='viridis', vmin=0, vmax=3e-4,
        filename="sunburst_ni_center"
    )

    print(f"Sunburst plots saved to {OUTPUT_DIR}/")
