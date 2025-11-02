import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Attention Weights Calculator", layout="wide")
st.title("Transformer-Inspired Attention Weights Calculator")

st.markdown("""
This app computes real projection matrices (\(W_q, W_k\)), attention weights, Gaussian spatial weights, and combined hybrid weights for multi-parameter interpolation.
Enter source solutions and target parameters below. Seed controls random initialization of \(W_q, W_k\).
""")

# Sidebar for global params
with st.sidebar:
    st.header("Model Parameters")
    sigma = st.number_input("Gaussian σ", 0.05, 1.0, 0.2, 0.01)
    num_heads = st.number_input("Number of Heads", 1, 8, 4, 1)
    d_head = st.number_input("Dim per Head", 4, 16, 8, 1)
    seed = st.number_input("Random Seed", 0, 9999, 42, 1)
    torch.manual_seed(seed)
    np.random.seed(seed)

# Input for sources
st.header("Source Solutions")
num_sources = st.number_input("Number of Sources", 1, 10, 3, 1)
params_list = []
for i in range(num_sources):
    col1, col2, col3 = st.columns(3)
    with col1:
        ly = st.number_input(f"L_y {i+1} (μm)", 30.0, 120.0, 30.0 + 30.0 * i, 0.1)
    with col2:
        c_cu = st.number_input(f"C_Cu {i+1} (mol/cc)", 0.0, 2.9e-3, 1.5e-3 + 0.5e-3 * i, 1e-4, format="%.1e")
    with col3:
        c_ni = st.number_input(f"C_Ni {i+1} (mol/cc)", 0.0, 1.8e-3, 1.0e-4 + 0.5e-4 * i, 1e-5, format="%.1e")
    params_list.append((ly, c_cu, c_ni))

# Target params
st.header("Target Parameters")
col1, col2, col3 = st.columns(3)
with col1:
    ly_target = st.number_input("Target L_y (μm)", 30.0, 120.0, 45.0, 0.1)
with col2:
    c_cu_target = st.number_input("Target C_Cu (mol/cc)", 0.0, 2.9e-3, 1.8e-3, 1e-4, format="%.1e")
with col3:
    c_ni_target = st.number_input("Target C_Ni (mol/cc)", 0.0, 1.8e-3, 3.0e-4, 1e-5, format="%.1e")

# Compute button
if st.button("Compute Weights"):
    class MultiParamAttentionInterpolator(nn.Module):
        def __init__(self, sigma, num_heads, d_head):
            super().__init__()
            self.sigma = sigma
            self.num_heads = num_heads
            self.d_head = d_head
            self.W_q = nn.Linear(3, num_heads * d_head)
            self.W_k = nn.Linear(3, num_heads * d_head)

        def normalize_params(self, params, is_target=False):
            lys = params[0] if is_target else params[:, 0]
            c_cus = params[1] if is_target else params[:, 1]
            c_nis = params[2] if is_target else params[:, 2]
            ly_norm = (lys - 30.0) / (120.0 - 30.0)
            c_cu_norm = c_cus / 2.9e-3
            c_ni_norm = c_nis / 1.8e-3
            if is_target:
                return np.array([ly_norm, c_cu_norm, c_ni_norm])
            else:
                return np.stack([ly_norm, c_cu_norm, c_ni_norm], axis=1)

        def compute_weights(self, params_list, ly_target, c_cu_target, c_ni_target):
            params_np = np.array(params_list)
            norm_sources = self.normalize_params(params_np)
            norm_target = self.normalize_params((ly_target, c_cu_target, c_ni_target), is_target=True)

            params_tensor = torch.tensor(norm_sources, dtype=torch.float32)
            target_tensor = torch.tensor(norm_target, dtype=torch.float32).unsqueeze(0)

            queries = self.W_q(target_tensor).view(1, self.num_heads, self.d_head)
            keys = self.W_k(params_tensor).view(len(params_list), self.num_heads, self.d_head)

            attn_logits = torch.einsum('nhd,mhd->nmh', keys, queries) / np.sqrt(self.d_head)
            attn_weights = torch.softmax(attn_logits, dim=0).mean(dim=2).squeeze(1)

            scaled_distances = torch.sqrt(
                ((torch.tensor(norm_sources[:, 0]) - norm_target[0]) / self.sigma)**2 +
                ((torch.tensor(norm_sources[:, 1]) - norm_target[1]) / self.sigma)**2 +
                ((torch.tensor(norm_sources[:, 2]) - norm_target[2]) / self.sigma)**2
            )
            spatial_weights = torch.exp(-scaled_distances**2 / 2)
            spatial_weights /= spatial_weights.sum()

            combined_weights = attn_weights * spatial_weights
            combined_weights /= combined_weights.sum()

            return {
                'W_q': self.W_q.weight.data.numpy(),
                'W_k': self.W_k.weight.data.numpy(),
                'attention_weights': attn_weights.detach().numpy(),
                'spatial_weights': spatial_weights.detach().numpy(),
                'combined_weights': combined_weights.detach().numpy()
            }

    interpolator = MultiParamAttentionInterpolator(sigma=sigma, num_heads=num_heads, d_head=d_head)
    results = interpolator.compute_weights(params_list, ly_target, c_cu_target, c_ni_target)

    # Display results
    st.subheader("Attention Weights")
    df_weights = pd.DataFrame({
        'Source': range(1, len(params_list) + 1),
        'Attention (ᾱ)': results['attention_weights'],
        'Gaussian (s̄)': results['spatial_weights'],
        'Hybrid (w)': results['combined_weights']
    })
    st.dataframe(df_weights.style.format("{:.3f}"))

    st.subheader("Projection Matrices")
    col1, col2 = st.columns(2)
    with col1:
        st.write("W_q (Query Projection)")
        fig_q, ax_q = plt.subplots(figsize=(5, 6))
        sns.heatmap(results['W_q'], ax=ax_q, cmap='viridis', annot=False, cbar=True)
        ax_q.set_xlabel("Input Dimensions")
        ax_q.set_ylabel("Output Dimensions")
        st.pyplot(fig_q)

    with col2:
        st.write("W_k (Key Projection)")
        fig_k, ax_k = plt.subplots(figsize=(5, 6))
        sns.heatmap(results['W_k'], ax=ax_k, cmap='viridis', annot=False, cbar=True)
        ax_k.set_xlabel("Input Dimensions")
        ax_k.set_ylabel("Output Dimensions")
        st.pyplot(fig_k)

# Download results
df_export = pd.DataFrame(results)
st.download_button("Download Results (CSV)", df_export.to_csv(index=False), "attention_weights.csv", mime="text/csv")
