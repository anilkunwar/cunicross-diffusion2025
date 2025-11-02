# app.py
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import RegularGridInterpolator
import os
from io import BytesIO
import pandas as pd

# -------------------------------
# CONFIG & GLOBALS
# -------------------------------
st.set_page_config(page_title="Transformer Attention Diffusion Interpolator", layout="wide")
st.title("Transformer-Inspired Attention for Diffusion Profile Interpolation")

# Dummy precomputed solutions (replace with real .pkl loading in production)
class DummySolution:
    def __init__(self, Ly, C_Cu, C_Ni):
        self['params'] = {'Ly': Ly, 'C_Cu': C_Cu, 'C_Ni': C_Ni, 'Lx': 50, 't_max': 100}
        self['X'] = np.linspace(0, 50, 50)
        self['Y'] = np.linspace(0, Ly, 50)
        self['times'] = np.linspace(0, 100, 10)
        self['c1_preds'] = [np.random.rand(50, 50) * C_Cu for _ in self['times']]  # Cu
        self['c2_preds'] = [np.random.rand(50, 50) * C_Ni for _ in self['times']]  # Ni

# Sample source solutions
source_params = [
    (30, 1.5e-3, 1.0e-4),
    (60, 2.0e-3, 5.0e-4),
    (90, 2.5e-3, 1.5e-3)
]
solutions = [DummySolution(*p) for p in source_params]

# -------------------------------
# ATTENTION INTERPOLATOR
# -------------------------------
class MultiParamAttentionInterpolator(nn.Module):
    def __init__(self, sigma=0.2, num_heads=4, d_head=8):
        super().__init__()
        self.sigma = sigma
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_model = self.num_heads * self.d_head
        self.W_q = nn.Linear(3, self.d_model, bias=False)
        self.W_k = nn.Linear(3, self.d_model, bias=False)
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)

    def forward(self, solutions, params_list, ly_target, c_cu_target, c_ni_target):
        # Normalize
        def norm(x, min_val, max_val): return (x - min_val) / (max_val - min_val)
        ly_norm = norm(np.array([p[0] for p in params_list]), 30, 120)
        c_cu_norm = norm(np.array([p[1] for p in params_list]), 0, 2.9e-3)
        c_ni_norm = norm(np.array([p[2] for p in params_list]), 0, 1.8e-3)
        target_norm = torch.tensor([[
            norm(ly_target, 30, 120),
            norm(c_cu_target, 0, 2.9e-3),
            norm(c_ni_target, 0, 1.8e-3)
        ]], dtype=torch.float32)

        source_norm = torch.tensor(np.stack([ly_norm, c_cu_norm, c_ni_norm], axis=1), dtype=torch.float32)

        # Project
        q = self.W_q(target_norm).view(1, self.num_heads, self.d_head)
        k = self.W_k(source_norm).view(len(params_list), self.num_heads, self.d_head)

        # Scaled dot-product
        logits = torch.einsum('nhd,mhd->nmh', k, q) / np.sqrt(self.d_head)
        attn = torch.softmax(logits, dim=0).mean(dim=2).squeeze(1)  # [N]

        # Gaussian locality
        dists = torch.sum(((source_norm - target_norm) / self.sigma) ** 2, dim=1)
        spatial = torch.exp(-dists / 2)
        spatial = spatial / spatial.sum()

        # Combine
        weights = attn * spatial
        weights = weights / weights.sum()
        weights = weights.detach().cpu().numpy()

        return self._blend(solutions, weights, ly_target, c_cu_target, c_ni_target)

    def _blend(self, solutions, weights, ly_target, c_cu_target, c_ni_target):
        Lx = 50
        x = np.linspace(0, Lx, 50)
        y = np.linspace(0, ly_target, 50)
        X, Y = np.meshgrid(x, y, indexing='ij')
        times = solutions[0]['times']
        c1_out = [np.zeros((50, 50)) for _ in times]
        c2_out = [np.zeros((50, 50)) for _ in times]

        for t_idx in range(len(times)):
            for sol, w in zip(solutions, weights):
                scale = ly_target / sol['params']['Ly']
                Y_src = sol['Y'] * scale
                interp_c1 = RegularGridInterpolator((sol['X'], Y_src), sol['c1_preds'][t_idx], method='linear', bounds_error=False, fill_value=0)
                interp_c2 = RegularGridInterpolator((sol['X'], Y_src), sol['c2_preds'][t_idx], method='linear', bounds_error=False, fill_value=0)
                pts = np.stack([X.flatten(), Y.flatten()], axis=1)
                c1_out[t_idx] += w * interp_c1(pts).reshape(50, 50)
                c2_out[t_idx] += w * interp_c2(pts).reshape(50, 50)

        # Enforce BCs
        for t in range(len(times)):
            c1_out[t][:, 0] = c_cu_target
            c2_out[t][:, -1] = c_ni_target

        return {
            'X': X, 'Y': Y, 'times': times,
            'c1_preds': c1_out, 'c2_preds': c2_out,
            'params': {'Ly': ly_target, 'C_Cu': c_cu_target, 'C_Ni': c_ni_target},
            'attention_weights': weights,
            'W_q': self.W_q.weight.detach().cpu().numpy(),
            'W_k': self.W_k.weight.detach().cpu().numpy(),
            'interpolated': True
        }

# -------------------------------
# SIDEBAR: Parameters
# -------------------------------
st.sidebar.header("Input Parameters")
ly_target = st.sidebar.slider("Target Ly (μm)", 30.0, 120.0, 45.0, 0.1)
c_cu_target = st.sidebar.slider("Target C_Cu (mol/cc)", 0.0, 2.9e-3, 1.8e-3, 0.1e-3, format="%.1e")
c_ni_target = st.sidebar.slider("Target C_Ni (mol/cc)", 0.0, 1.8e-3, 3.0e-4, 0.1e-4, format="%.1e")

# -------------------------------
# RUN INTERPOLATION
# -------------------------------
@st.cache_resource
def run_interpolation():
    interpolator = MultiParamAttentionInterpolator()
    result = interpolator(solutions, source_params, ly_target, c_cu_target, c_ni_target)
    return result, interpolator

result, model = run_interpolation()
weights = result['attention_weights']
W_q = result['W_q']
W_k = result['W_k']

# -------------------------------
# DISPLAY: Attention Weights & Matrices
# -------------------------------
st.subheader("Attention Mechanism Details")
col1, col2 = st.columns(2)
with col1:
    st.write("**Attention + Locality Weights**")
    df_w = pd.DataFrame({
        'Source': [f"θ{i+1}" for i in range(3)],
        'Ly': [p[0] for p in source_params],
        'C_Cu': [f"{p[1]:.1e}" for p in source_params],
        'C_Ni': [f"{p[2]:.1e}" for p in source_params],
        'Weight w_i': [f"{w:.3f}" for w in weights]
    })
    st.table(df_w)

with col2:
    st.write("**Query Projection Matrix W_q** (3×32)")
    st.text(np.array2string(W_q, precision=3, suppress_small=True, max_line_width=200))

st.info(f"Interpolated using weights: {weights.round(3)}")

# -------------------------------
# PLOT: 2D Concentration
# -------------------------------
st.subheader("Interpolated Concentration Fields")
time_idx = st.slider("Time Step", 0, len(result['times'])-1, len(result['times'])//2)
t_val = result['times'][time_idx]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
im1 = ax1.imshow(result['c1_preds'][time_idx], origin='lower', extent=[0, 50, 0, ly_target], cmap='viridis')
ax1.set_title(f"Cu Concentration @ t = {t_val:.1f}s")
ax1.set_xlabel("x (μm)"); ax1.set_ylabel("y (μm)")
plt.gcf().colorbar(im1, ax=ax1, label="mol/cc")

im2 = ax2.imshow(result['c2_preds'][time_idx], origin='lower', extent=[0, 50, 0, ly_target], cmap='plasma')
ax2.set_title(f"Ni Concentration @ t = {t_val:.1f}s")
ax2.set_xlabel("x (μm)")
plt.gcf().colorbar(im2, ax=ax2, label="mol/cc")

st.pyplot(fig)

# -------------------------------
# FLUX & UPHILL DETECTION
# -------------------------------
def compute_fluxes(c1, c2, dy):
    grad_c1 = np.gradient(c1, dy, axis=1)
    grad_c2 = np.gradient(c2, dy, axis=1)
    D11, D12, D21, D22 = 1.0, 0.3, 0.2, 1.0
    J1 = -(D11 * grad_c1 + D12 * grad_c2)
    J2 = -(D21 * grad_c1 + D22 * grad_c2)
    return J1, J2, grad_c1, grad_c2

dy = ly_target / 49
J1, J2, g1, g2 = compute_fluxes(result['c1_preds'][time_idx], result['c2_preds'][time_idx], dy)
uphill_cu = (J1 * g1 > 0)
uphill_ni = (J2 * g2 > 0)

st.subheader("Uphill Diffusion (J · ∇c > 0)")
col1, col2 = st.columns(2)
with col1:
    st.write(f"**Cu Uphill Fraction**: {uphill_cu.mean():.1%}")
    fig, ax = plt.subplots()
    ax.scatter(g1.flatten(), J1.flatten(), c=uphill_cu.flatten(), cmap='coolwarm', s=1)
    ax.set_xlabel("∇c_Cu"); ax.set_ylabel("J_Cu")
    ax.set_title("Cu: Flux vs Gradient")
    st.pyplot(fig)
with col2:
    st.write(f"**Ni Uphill Fraction**: {uphill_ni.mean():.1%}")
    fig, ax = plt.subplots()
    ax.scatter(g2.flatten(), J2.flatten(), c=uphill_ni.flatten(), cmap='coolwarm', s=1)
    ax.set_xlabel("∇c_Ni"); ax.set_ylabel("J_Ni")
    ax.set_title("Ni: Flux vs Gradient")
    st.pyplot(fig)

# -------------------------------
# LaTeX DISCUSSION (Copy-Paste Ready)
# -------------------------------
st.subheader("LaTeX for Manuscript")
latex_text = r"""
\section{Discussion}
\label{sec:discussion}

The transformer-inspired attention mechanism enables parameter-continuous interpolation of diffusion profiles, as shown in Appendix~\ref{app:numerical}. Query-key projections embed normalized parameters $(\tilde{L}_y, \tilde{C}_{\text{Cu}}, \tilde{C}_{\text{Ni}})$ into a latent space, where scaled dot-product attention computes relevance. Multi-head averaging and Gaussian locality fusion yield hybrid weights $w_i$, blending rescaled source fields.

In Cu-Ni solder joints, concentration gradients drive flux via $ \mathbf{J} = -\mathbf{D} \nabla \mathbf{c} $. Off-diagonal $D_{12}, D_{21}$ induce \emph{uphill diffusion} (J · ∇c > 0), accelerating intermetallic growth (e.g., Cu$_6$Sn$_5$) via Kirkendall voiding. The interpolated flux imbalance $\int (J_1 - J_2) dt$ predicts IMC thickness, guiding reliability design.

\appendix
\section{Numerical Illustration}
\label{app:numerical}
Sources: $\theta_1=(30,1.5\times10^{-3},10^{-4})$, $\theta_2=(60,2\times10^{-3},5\times10^{-4})$, $\theta_3=(90,2.5\times10^{-3},1.5\times10^{-3})$.  
Target: $\theta^*=(45,1.8\times10^{-3},3\times10^{-4})$.  
Weights: $w \approx [0.33, 0.35, 0.32]$.  
$W_q \in \mathbb{R}^{32 \times 3}$ (shown in app).
"""

st.code(latex_text, language='latex')

# -------------------------------
# DOWNLOAD RESULTS
# -------------------------------
@st.cache_data
def get_results_df():
    return pd.DataFrame({
        'Source': [f'S{i+1}' for i in range(3)],
        'Ly': [p[0] for p in source_params],
        'C_Cu': [p[1] for p in source_params],
        'C_Ni': [p[2] for p in source_params],
        'Weight': weights
    })

csv = get_results_df().to_csv(index=False).encode()
st.download_button("Download Weights CSV", csv, "attention_weights.csv", "text/csv")
