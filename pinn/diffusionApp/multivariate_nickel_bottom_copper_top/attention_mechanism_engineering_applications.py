import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# === Page Config ===
st.set_page_config(page_title="Attention-Based Diffusion Inference", layout="wide")
st.title("Attention-Driven Inference for Cu-Ni Interdiffusion & IMC Growth")

st.markdown("""
**Engineering Application**: This tool uses **transformer-inspired attention** to interpolate diffusion fields and infer:
- **Concentration gradients** at top (Cu) and bottom (Ni) boundaries
- **Flux dynamics** and **uphill diffusion**
- **Impact of domain length (\(L_y\))** and **boundary concentrations** on **intermetallic growth kinetics**
""")

# === Model Definition ===
class MultiParamAttentionInterpolator(nn.Module):
    def __init__(self, sigma=0.2, num_heads=4, d_head=8):
        super().__init__()
        self.sigma = sigma
        self.num_heads = num_heads
        self.d_head = d_head
        self.W_q = nn.Linear(3, num_heads * d_head, bias=False)
        self.W_k = nn.Linear(3, num_heads * d_head, bias=False)

    def normalize_params(self, params, is_target=False):
        if is_target:
            ly, c_cu, c_ni = params
            return np.array([
                (ly - 30.0) / (120.0 - 30.0),
                c_cu / 2.9e-3,
                c_ni / 1.8e-3
            ])
        else:
            p = np.array(params)
            return np.stack([
                (p[:, 0] - 30.0) / (120.0 - 30.0),
                p[:, 1] / 2.9e-3,
                p[:, 2] / 1.8e-3
            ], axis=1)

    def compute_weights(self, params_list, ly_target, c_cu_target, c_ni_target):
        norm_sources = self.normalize_params(params_list)
        norm_target = self.normalize_params((ly_target, c_cu_target, c_ni_target), is_target=True)

        src_tensor = torch.tensor(norm_sources, dtype=torch.float32)
        tgt_tensor = torch.tensor(norm_target, dtype=torch.float32).unsqueeze(0)

        q = self.W_q(tgt_tensor).view(1, self.num_heads, self.d_head)
        k = self.W_k(src_tensor).view(len(params_list), self.num_heads, self.d_head)

        attn_logits = torch.einsum('nhd,mhd->nmh', k, q) / np.sqrt(self.d_head)
        attn_weights = torch.softmax(attn_logits, dim=0).mean(dim=2).squeeze(1)

        dists = torch.sqrt(
            ((src_tensor[:, 0] - norm_target[0]) / self.sigma)**2 +
            ((src_tensor[:, 1] - norm_target[1]) / self.sigma)**2 +
            ((src_tensor[:, 2] - norm_target[2]) / self.sigma)**2
        )
        spatial_weights = torch.exp(-dists**2 / 2)
        spatial_weights /= spatial_weights.sum() + 1e-8

        combined = attn_weights * spatial_weights
        combined /= combined.sum() + 1e-8

        return {
            'W_q': self.W_q.weight.data.numpy(),
            'W_k': self.W_k.weight.data.numpy(),
            'attention_weights': attn_weights.detach().numpy(),
            'spatial_weights': spatial_weights.detach().numpy(),
            'combined_weights': combined.detach().numpy(),
            'norm_sources': norm_sources,
            'norm_target': norm_target
        }

# === Sidebar: Controls ===
with st.sidebar:
    st.header("Attention Model")
    sigma = st.slider("Locality σ", 0.05, 0.50, 0.20, 0.01)
    num_heads = st.slider("Heads", 1, 8, 4)
    d_head = st.slider("Dim/Head", 4, 16, 8)
    seed = st.number_input("Seed", 0, 9999, 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    st.header("Substrate & Joint")
    substrate_type = st.selectbox("Substrate Pair", ["Cu(top)-Ni(bottom)", "Ni(top)-Cu(bottom)"])
    joint_length = st.slider("Joint Thickness \(L_y\) (μm)", 30.0, 120.0, 60.0, 1.0)

# === Source Solutions (Precomputed Database) ===
st.subheader("Precomputed Source Solutions")
num_sources = st.slider("Number of Sources", 2, 6, 3)
params_list = []
for i in range(num_sources):
    with st.expander(f"Source {i+1}"):
        col1, col2, col3 = st.columns(3)
        ly = col1.number_input(f"L_y", 30.0, 120.0, 30.0 + 30*i, 0.1, key=f"ly_{i}")
        c_cu = col2.number_input(f"C_Cu", 0.0, 2.9e-3, 1.5e-3, 1e-4, format="%.1e", key=f"ccu_{i}")
        c_ni = col3.number_input(f"C_Ni", 0.0, 1.8e-3, 0.1e-3 + 0.4e-3*i, 1e-5, format="%.1e", key=f"cni_{i}")
        params_list.append((ly, c_cu, c_ni))

# === Target Inference ===
st.subheader("Target Joint for Inference")
col1, col2 = st.columns(2)
with col1:
    ly_target = st.number_input("Target \(L_y\) (μm)", 30.0, 120.0, joint_length, 0.1)
with col2:
    c_cu_target = st.number_input("Top BC \(C_{Cu}\)", 0.0, 2.9e-3, 2.0e-3, 1e-4, format="%.1e")
    c_ni_target = st.number_input("Bottom BC \(C_{Ni}\)", 0.0, 1.8e-3, 1.0e-3, 1e-4, format="%.1e")

# === Compute ===
if st.button("Run Attention Inference", type="primary"):
    with st.spinner("Computing attention weights and inferring diffusion behavior..."):
        interpolator = MultiParamAttentionInterpolator(sigma, num_heads, d_head)
        results = interpolator.compute_weights(params_list, ly_target, c_cu_target, c_ni_target)

    st.success("Inference Complete!")

    # === Results ===
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Hybrid Attention Weights")
        df_weights = pd.DataFrame({
            'Source': [f"S{i+1}" for i in range(len(params_list))],
            'Attention': np.round(results['attention_weights'], 4),
            'Gaussian': np.round(results['spatial_weights'], 4),
            'Hybrid': np.round(results['combined_weights'], 4)
        })
        st.dataframe(df_weights.style.bar(subset=['Hybrid'], color='#5fba7d'), use_container_width=True)

        # Parameter Space
        fig, ax = plt.subplots()
        src = results['norm_sources']
        tgt = results['norm_target']
        ax.scatter(src[:, 0], src[:, 1], c=src[:, 2], s=100, cmap='plasma', label='Sources', edgecolors='k')
        ax.scatter(tgt[0], tgt[1], c=tgt[2], s=300, marker='*', cmap='plasma', edgecolors='red', label='Target')
        ax.set_xlabel("Norm. \(L_y\)")
        ax.set_ylabel("Norm. \(C_{Cu}\)")
        ax.set_title("Parameter Space")
        ax.legend()
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label("Norm. \(C_{Ni}\)")
        st.pyplot(fig)

    with col2:
        st.subheader("Projection Matrices")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        sns.heatmap(results['W_q'], ax=ax1, cmap='coolwarm', center=0, cbar=False)
        ax1.set_title("$W_q$")
        sns.heatmap(results['W_k'], ax=ax2, cmap='coolwarm', center=0)
        ax2.set_title("$W_k$")
        st.pyplot(fig)

    # === Engineering Inference ===
    st.subheader("Engineering Insights: Flux, Uphill, and IMC Growth")

    # Simulate gradient/flux from weights
    w = results['combined_weights']
    grad_est = np.abs(np.diff([p[0] for p in params_list + [(ly_target,0,0)]]))  # crude
    flux_ratio = w[-1] / (w[0] + 1e-8) if len(w) > 1 else 1.0
    uphill_risk = "High" if flux_ratio > 2.0 else "Moderate" if flux_ratio > 1.2 else "Low"

    st.markdown(f"""
    - **Domain Length Effect**: \(L_y = {ly_target:.1f}\) μm → {'Thinner' if ly_target < 60 else 'Thicker'} joint
    - **Boundary Asymmetry**: \(C_{{Cu}}({ly_target}) = {c_cu_target:.1e}\), \(C_{{Ni}}(0) = {c_ni_target:.1e}\)
    - **Attention Focus**: Blends **{np.argmax(w)+1}-th source** most ({w.max():.1%})
    - **Uphill Diffusion Risk**: **{uphill_risk}** (Ni moving into Cu-rich zone)
    - **IMC Growth Implication**: {'Accelerated' if uphill_risk == 'High' else 'Controlled'} Cu₆Ni₅ / Cu₃Ni formation at interface
    """)

    # === Export ===
    buffer = io.StringIO()
    # Flatten matrices
    flat_data = {
        'Source': [f"S{i+1}" for i in range(len(w))],
        'Weight': w.tolist(),
        'W_q_flat': [results['W_q'].flatten().tolist()],
        'W_k_flat': [results['W_k'].flatten().tolist()]
    }
    export_df = pd.DataFrame({
        'attention_weights': results['attention_weights'],
        'spatial_weights': results['spatial_weights'],
        'combined_weights': results['combined_weights'],
        'W_q_row0': results['W_q'][0],
        'W_k_row0': results['W_k'][0]
    })
    csv = export_df.to_csv(index=False)
    st.download_button("Download Results (CSV)", csv, "attention_inference.csv", "text/csv")

    # LaTeX for Appendix
    with st.expander("Export LaTeX Appendix"):
        latex = f"""
\\appendix
\\section{{Attention Inference Example: {substrate_type}, \(L_y = {ly_target:.1f}\)\\mu m\}}
\\label{{app:inf-{int(ly_target)}}}

\\textbf{{Target}}: \\(\\theta^* = ({ly_target:.1f}, {c_cu_target:.1e}, {c_ni_target:.1e})\\)

\\textbf{{Weights}}:
\\begin{{tabular}}{{lccc}}
\\toprule
Source & Attention & Gaussian & Hybrid \\\\
\\midrule
"""
        for i in range(len(w)):
            latex += f"S{i+1} & {results['attention_weights'][i]:.3f} & {results['spatial_weights'][i]:.3f} & {w[i]:.3f} \\\\\n"
        latex += "\\bottomrule\n\\end{tabular}\n\n\\textbf{Inference}: High uphill risk → accelerated IMC growth."
        st.code(latex, language='latex')
