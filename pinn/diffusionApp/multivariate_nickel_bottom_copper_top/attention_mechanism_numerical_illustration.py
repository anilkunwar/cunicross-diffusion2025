import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# Set page config
st.set_page_config(page_title="Attention Interpolator Demo", layout="wide")
st.title("Transformer-Inspired Attention Interpolator for Diffusion Fields")

st.markdown("""
Interactive demonstration of the **multi-parameter attention mechanism** used to interpolate 
diffusion concentration profiles across domain length \(L_y\) and boundary conditions \(C_{\\text{Cu}}\), \(C_{\\text{Ni}}\).
""")

# ——————————————————————————————————————————————————————
# 1. Model Definition (same as your code)
# ——————————————————————————————————————————————————————
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
            ly_norm = (ly - 30.0) / (120.0 - 30.0)
            c_cu_norm = c_cu / 2.9e-3
            c_ni_norm = c_ni / 1.8e-3
            return np.array([ly_norm, c_cu_norm, c_ni_norm])
        else:
            params = np.array(params)
            ly_norm = (params[:, 0] - 30.0) / (120.0 - 30.0)
            c_cu_norm = params[:, 1] / 2.9e-3
            c_ni_norm = params[:, 2] / 1.8e-3
            return np.stack([ly_norm, c_cu_norm, c_ni_norm], axis=1)

    def compute_weights(self, params_list, ly_target, c_cu_target, c_ni_target):
        norm_sources = self.normalize_params(params_list)
        norm_target = self.normalize_params((ly_target, c_cu_target, c_ni_target), is_target=True)

        params_tensor = torch.tensor(norm_sources, dtype=torch.float32)
        target_tensor = torch.tensor(norm_target, dtype=torch.float32).unsqueeze(0)

        queries = self.W_q(target_tensor).view(1, self.num_heads, self.d_head)
        keys = self.W_k(params_tensor).view(len(params_list), self.num_heads, self.d_head)

        attn_logits = torch.einsum('nhd,mhd->nmh', keys, queries) / np.sqrt(self.d_head)
        attn_weights = torch.softmax(attn_logits, dim=0).mean(dim=2).squeeze(1)

        scaled_distances = torch.sqrt(
            ((params_tensor[:, 0] - norm_target[0]) / self.sigma)**2 +
            ((params_tensor[:, 1] - norm_target[1]) / self.sigma)**2 +
            ((params_tensor[:, 2] - norm_target[2]) / self.sigma)**2
        )
        spatial_weights = torch.exp(-scaled_distances**2 / 2)
        spatial_weights /= spatial_weights.sum() + 1e-8

        combined_weights = attn_weights * spatial_weights
        combined_weights /= combined_weights.sum() + 1e-8

        return {
            'W_q': self.W_q.weight.data.numpy(),
            'W_k': self.W_k.weight.data.numpy(),
            'attention_weights': attn_weights.detach().numpy(),
            'spatial_weights': spatial_weights.detach().numpy(),
            'combined_weights': combined_weights.detach().numpy(),
            'norm_sources': norm_sources,
            'norm_target': norm_target
        }

# ——————————————————————————————————————————————————————
# 2. Sidebar: User Input
# ——————————————————————————————————————————————————————
with st.sidebar:
    st.header("Source Solutions (Precomputed)")
    num_sources = st.slider("Number of Source Solutions", 2, 6, 3)

    params_list = []
    for i in range(num_sources):
        st.subheader(f"Source {i+1}")
        ly = st.number_input(f"L_y (μm)", 30.0, 120.0, 30.0 + i*30.0, 0.1, key=f"ly_{i}")
        c_cu = st.number_input(f"C_Cu (mol/cc)", 0.0, 2.9e-3, 1.5e-3 + i*0.5e-3, 0.1e-3, format="%.2e", key=f"ccu_{i}")
        c_ni = st.number_input(f"C_Ni (mol/cc)", 0.0, 1.8e-3, 0.1e-3 + i*0.4e-3, 0.1e-4, format="%.2e", key=f"cni_{i}")
        params_list.append((ly, c_cu, c_ni))

    st.header("Target Parameters (Interpolation)")
    ly_target = st.number_input("Target L_y (μm)", 30.0, 120.0, 45.0, 0.1)
    c_cu_target = st.number_input("Target C_Cu (mol/cc)", 0.0, 2.9e-3, 1.8e-3, 0.1e-3, format="%.2e")
    c_ni_target = st.number_input("Target C_Ni (mol/cc)", 0.0, 1.8e-3, 3.0e-4, 0.1e-4, format="%.2e")

    sigma = st.slider("Gaussian σ (locality)", 0.05, 0.5, 0.2, 0.01)
    seed = st.number_input("Random Seed (for W_q, W_k)", 0, 9999, 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

# ——————————————————————————————————————————————————————
# 3. Run Computation
# ——————————————————————————————————————————————————————
@st.cache_resource
def get_interpolator(_sigma):
    return MultiParamAttentionInterpolator(sigma=_sigma)

interpolator = get_interpolator(sigma)
results = interpolator.compute_weights(params_list, ly_target, c_cu_target, c_ni_target)

# ——————————————————————————————————————————————————————
# 4. Display Results
# ——————————————————————————————————————————————————————
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Parameter Space (Normalized)")
    fig, ax = plt.subplots(figsize=(6, 5))
    norm_sources = results['norm_sources']
    norm_target = results['norm_target']
    scatter = ax.scatter(norm_sources[:, 0], norm_sources[:, 1], c=norm_sources[:, 2], cmap='viridis', s=120, edgecolors='k', label='Sources')
    ax.scatter(norm_target[0], norm_target[1], c=norm_target[2], cmap='viridis', s=200, marker='*', edgecolors='red', linewidth=2, label='Target')
    ax.set_xlabel('Normalized $L_y$')
    ax.set_ylabel('Normalized $C_{Cu}$')
    ax.set_title('Parameter Space')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Normalized $C_{Ni}$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Attention + Locality Weights")
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(params_list))
    width = 0.25
    ax.bar(x - width, results['attention_weights'], width, label='Attention (ᾱ)', color='skyblue')
    ax.bar(x, results['spatial_weights'], width, label='Gaussian (s̄)', color='lightcoral')
    ax.bar(x + width, results['combined_weights'], width, label='Hybrid (w)', color='gold')
    ax.set_xlabel('Source Index')
    ax.set_ylabel('Weight')
    ax.set_title('Weight Contributions')
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{i+1}" for i in range(len(params_list))])
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    st.pyplot(fig)
    plt.close()

# ——————————————————————————————————————————————————————
# 5. Projection Matrices (Heatmaps)
# ——————————————————————————————————————————————————————
st.subheader("Learned Projection Matrices \(W_q\), \(W_k\) (32×3)")

w_q = results['W_q']
w_k = results['W_k']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(w_q, ax=ax1, cmap='coolwarm', center=0, cbar_kws={'label': 'Weight'}, annot=False)
ax1.set_title('$W_q$ (Query Projection)')
ax1.set_xlabel('Input Dim (L_y, C_Cu, C_Ni)')
ax1.set_ylabel('Output Dim (32)')

sns.heatmap(w_k, ax=ax2, cmap='coolwarm', center=0, cbar_kws={'label': 'Weight'}, annot=False)
ax2.set_title('$W_k$ (Key Projection)')
ax2.set_xlabel('Input Dim')
ax2.set_ylabel('Output Dim')

st.pyplot(fig)
plt.close()

# ——————————————————————————————————————————————————————
# 6. Flowchart (Mermaid)
# ——————————————————————————————————————————————————————
st.subheader("Attention Pipeline Flowchart")
mermaid_code = f"""
graph LR
    subgraph Input
        A[Sources θ_i] --> N[Normalize → θ̃_i ∈ [0,1]^3]
        B[Target θ*] --> N
    end
    N --> Q[Query W_q → q]
    N --> K[Keys W_k → k_i]
    Q & K --> A1[Scaled Dot-Product]
    A1 --> S1[Softmax per head]
    S1 --> A2[Avg over {interpolator.num_heads} heads → ᾱ]
    N --> D[Distance ||θ̃_i - θ̃*||]
    D --> G[Gaussian exp(-d²/2σ²), σ={sigma}]
    G --> S2[Normalize → s̄]
    A2 & S2 --> F[Fuse w_i = (ᾱ_i s̄_i) / Σ]
    F --> B1[Blend Fields]
    B1 --> BC[Enforce BCs]
    BC --> O[Output c*(x,y,t)]
"""
st.markdown(f"```mermaid\n{mermaid_code}\n```")

# ——————————————————————————————————————————————————————
# 7. Appendix-Ready LaTeX Output
# ——————————————————————————————————————————————————————
with st.expander("Export the Attention Weights"):
    latex_code = f"""
\\appendix
\\section{{Numerical Example of Attention Weights}}
\\label{{app:attention_example}}

\\textbf{{Sources:}}
\\begin{{align*}}
&\\theta_1 = ({params_list[0][0]}, {params_list[0][1]:.2e}, {params_list[0][2]:.2e}) \\quad
\\tilde{{\\theta}}_1 = ({norm_sources[0,0]:.3f}, {norm_sources[0,1]:.3f}, {norm_sources[0,2]:.3f}) \\\\
&\\theta_2 = ({params_list[1][0]}, {params_list[1][1]:.2e}, {params_list[1][2]:.2e}) \\quad
\\tilde{{\\theta}}_2 = ({norm_sources[1,0]:.3f}, {norm_sources[1,1]:.3f}, {norm_sources[1,2]:.3f}) \\\\
&\\theta_3 = ({params_list[2][0]}, {params_list[2][1]:.2e}, {params_list[2][2]:.2e}) \\quad
\\tilde{{\\theta}}_3 = ({norm_sources[2,0]:.3f}, {norm_sources[2,1]:.3f}, {norm_sources[2,2]:.3f})
\\end{{align*}}

\\textbf{{Target:}} \\(\\theta^* = ({ly_target}, {c_cu_target:.2e}, {c_ni_target:.2e}) \\rightarrow 
\\tilde{{\\theta}}^* = ({norm_target[0]:.3f}, {norm_target[1]:.3f}, {norm_target[2]:.3f})\\)

\\textbf{{Weights:}}
\\begin{{center}}
\\begin{{tabular}}{{cccc}}
\\toprule
Source & Attention (ᾱ) & Gaussian (s̄) & Hybrid (w) \\\\
\\midrule
1 & {results['attention_weights'][0]:.3f} & {results['spatial_weights'][0]:.3f} & {results['combined_weights'][0]:.3f} \\\\
2 & {results['attention_weights'][1]:.3f} & {results['spatial_weights'][1]:.3f} & {results['combined_weights'][1]:.3f} \\\\
3 & {results['attention_weights'][2]:.3f} & {results['spatial_weights'][2]:.3f} & {results['combined_weights'][2]:.3f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{center}}
"""
    st.code(latex_code, language='latex')
    st.download_button("Download LaTeX", latex_code, file_name="attention_appendix.tex", mime="text/plain")

# ——————————————————————————————————————————————————————
# 8. Footer
# ——————————————————————————————————————————————————————
st.markdown("---")
st.caption("Developed for manuscript: *Attention-Based Interpolation of Diffusion Profiles in Cu-Ni Joints*")
