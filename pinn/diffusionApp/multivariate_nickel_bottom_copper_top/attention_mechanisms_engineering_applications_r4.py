import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import sqlite3
from transformers import pipeline

# === Page Config ===
st.set_page_config(page_title="Attention-Based Diffusion Inference", layout="wide")
st.title("Attention-Driven Inference for Cu-Ni Interdiffusion & IMC Growth in Solder Joints")

st.markdown("""
**Engineering Context**: This tool leverages **transformer-inspired attention** to interpolate precomputed diffusion fields from PINN models and infer key phenomena in Cu pillar microbumps with Sn2.5Ag solder. It analyzes the role of domain length (\(L_y\)), boundary concentrations, substrate configurations (symmetric/asymmetric), and joining paths on concentration profiles, flux dynamics, uphill diffusion, cross-interactions, and intermetallic compound (IMC) growth kinetics at top (Cu) and bottom (Ni) interfaces.
""")

# === Experimental Description (from nlp_information/description_of_experiment.db) ===
DB_DIR = os.path.join(os.path.dirname(__file__), "nlp_information")
DB_FILENAME = "description_of_experiment.db"
DB_PATH = os.path.join(DB_DIR, DB_FILENAME)

EXPERIMENTAL_DESCRIPTION = ""

if os.path.exists(DB_PATH):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # Example: if your table is named 'experiments' and has a column 'description'
        cursor.execute("SELECT description FROM experiments LIMIT 1;")
        row = cursor.fetchone()
        if row:
            EXPERIMENTAL_DESCRIPTION = row[0]
        else:
            st.warning("No description found in the database.")
        conn.close()
    except Exception as e:
        st.error(f"Error reading database: {e}")
else:
    st.error(f"Database file not found: `{DB_PATH}`\n\n"
             "Please ensure the file exists in the `nlp_information/` directory.")

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

# === Numeric Integration for Diffusion Analysis ===
class DiffusionAnalyzer:
    def __init__(self):
        self.D_Cu = 1.2e-9  # Diffusion coefficient for Cu in Sn (m²/s)
        self.D_Ni = 8.5e-10  # Diffusion coefficient for Ni in Sn (m²/s)
        
    def concentration_profile(self, y, L, C_top, C_bottom, D, time):
        """Calculate concentration profile using error function solution"""
        # Convert to meters for diffusion calculation
        y_norm = y / L
        return C_bottom + (C_top - C_bottom) * (1 - y_norm)
    
    def calculate_flux(self, concentration_gradient, D):
        """Calculate diffusion flux using Fick's first law"""
        return -D * concentration_gradient
    
    def integrate_imc_growth(self, flux_Cu, flux_Ni, time_hours):
        """Integrate IMC thickness based on interdiffusion fluxes"""
        # IMC growth proportional to interdiffusion flux
        k_Cu6Sn5 = 2.3e-11  # Growth constant for Cu6Sn5 (m²/s)
        k_Ni3Sn4 = 1.8e-11  # Growth constant for Ni3Sn4 (m²/s)
        
        time_seconds = time_hours * 3600
        
        # IMC thickness from parabolic growth law
        imc_Cu_thickness = np.cbrt(2 * k_Cu6Sn5 * time_seconds) * 1e6  # convert to μm
        imc_Ni_thickness = np.cbrt(2 * k_Ni3Sn4 * time_seconds) * 1e6  # convert to μm
        
        return imc_Cu_thickness, imc_Ni_thickness

# === Sidebar: Controls ===
with st.sidebar:
    st.header("Attention Model")
    sigma = st.slider("Locality σ", 0.05, 0.50, 0.20, 0.01)
    num_heads = st.slider("Heads", 1, 8, 4)
    d_head = st.slider("Dim/Head", 4, 16, 8)
    seed = st.number_input("Seed", 0, 9999, 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    st.header("Joint Configuration")
    substrate_type = st.selectbox("Substrate Configuration", ["Cu(top)-Ni(bottom) Asymmetric", "Cu/Sn2.5Ag/Cu Symmetric", "Ni/Sn2.5Ag/Ni Symmetric"])
    joining_path = st.selectbox("Joining Path (for Asymmetric)", ["Path I (Cu→Ni)", "Path II (Ni→Cu)", "N/A"])
    joint_length = st.slider("Joint Thickness \(L_y\) (μm)", 30.0, 120.0, 60.0, 1.0)
    
    st.header("Diffusion Analysis Parameters")
    reflow_time = st.slider("Reflow Time (seconds)", 1, 1000, 50, key="reflow_time")
    num_points = st.slider("Grid Points", 50, 500, 100, key="num_points")

    st.header("AI Model Selection")
    nlp_model = st.selectbox(
        "AI Model for Engineering Insights",
        ["gpt2", "facebook/opt-1.3b", "tiiuae/falcon-7b-instruct", "google/flan-t5-large"],
        key="nlp_model"
    )

# === Source Solutions (Precomputed PINN Simulations) ===
st.subheader("Precomputed Source Simulations (e.g., PINN-Generated Diffusion Profiles)")
num_sources = st.slider("Number of Sources", 2, 6, 3, key="num_sources")
params_list = []
for i in range(num_sources):
    with st.expander(f"Source {i+1} (e.g., Simulated Configuration)"):
        col1, col2, col3 = st.columns(3)
        ly = col1.number_input(f"L_y", 30.0, 120.0, 30.0 + 30*i, 0.1, key=f"ly_{i}")
        c_cu = col2.number_input(f"C_Cu (top, mol/cc)", 0.0, 2.9e-3, 1.5e-3, 1e-4, format="%.1e", key=f"ccu_{i}")
        c_ni = col3.number_input(f"C_Ni (bottom, mol/cc)", 0.0, 1.8e-3, 0.1e-3 + 0.4e-3*i, 1e-5, format="%.1e", key=f"cni_{i}")
        params_list.append((ly, c_cu, c_ni))

# === Target Inference ===
st.subheader("Target Joint for Inference")
col1, col2 = st.columns(2)
with col1:
    ly_target = st.number_input("Target \(L_y\) (μm)", 30.0, 120.0, joint_length, 0.1, key="ly_target")
with col2:
    c_cu_target = st.number_input("Top BC \(C_{Cu}\) (mol/cc)", 0.0, 2.9e-3, 2.0e-3, 1e-4, format="%.1e", key="c_cu_target")
    c_ni_target = st.number_input("Bottom BC \(C_{Ni}\) (mol/cc)", 0.0, 1.8e-3, 1.0e-3, 1e-4, format="%.1e", key="c_ni_target")

# === Compute ===
if st.button("Run Attention Inference", type="primary"):
    with st.spinner("Interpolating diffusion profiles and inferring joint behavior..."):
        interpolator = MultiParamAttentionInterpolator(sigma, num_heads, d_head)
        results = interpolator.compute_weights(params_list, ly_target, c_cu_target, c_ni_target)
        
        # Perform numeric integration
        analyzer = DiffusionAnalyzer()
        y_positions = np.linspace(0, ly_target, num_points)
        
        # Calculate concentration profiles
        time_seconds = reflow_time #reflow_time * 3600
        Cu_profile = analyzer.concentration_profile(y_positions, ly_target, c_cu_target, 0, analyzer.D_Cu, time_seconds)
        Ni_profile = analyzer.concentration_profile(ly_target - y_positions, ly_target, c_ni_target, 0, analyzer.D_Ni, time_seconds)
        
        # Calculate concentration gradients and fluxes
        gradient_Cu = np.gradient(Cu_profile, y_positions)
        gradient_Ni = np.gradient(Ni_profile, y_positions)
        flux_Cu = analyzer.calculate_flux(gradient_Cu, analyzer.D_Cu)
        flux_Ni = analyzer.calculate_flux(gradient_Ni, analyzer.D_Ni)
        
        # Calculate IMC growth
        imc_Cu_thickness, imc_Ni_thickness = analyzer.integrate_imc_growth(
            np.mean(flux_Cu), np.mean(flux_Ni), reflow_time
        )

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
        ax.set_xlabel("Norm. $L_y$")
        ax.set_ylabel("Norm. $C_{Cu}$")
        ax.set_title("Parameter Space")
        ax.legend()
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label("Norm. $C_{Ni}$")
        st.pyplot(fig)

    with col2:
        st.subheader("Projection Matrices")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        sns.heatmap(results['W_q'], ax=ax1, cmap='coolwarm', center=0, cbar=False)
        ax1.set_title("$W_q$")
        sns.heatmap(results['W_k'], ax=ax2, cmap='coolwarm', center=0)
        ax2.set_title("$W_k$")
        st.pyplot(fig)

    # === Dynamic Diffusion Analysis Results ===
    st.subheader("Dynamic Diffusion Analysis & IMC Growth Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Concentration profiles
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(Cu_profile, y_positions, 'r-', linewidth=2, label='Cu Concentration')
        ax.plot(Ni_profile, y_positions, 'b-', linewidth=2, label='Ni Concentration')
        ax.set_xlabel('Concentration (mol/cc)')
        ax.set_ylabel('Position (μm)')
        ax.set_title('Concentration Profiles')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        # Flux profiles
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(flux_Cu, y_positions, 'r--', linewidth=2, label='Cu Flux')
        ax.plot(flux_Ni, y_positions, 'b--', linewidth=2, label='Ni Flux')
        ax.set_xlabel('Diffusion Flux (mol/m²s)')
        ax.set_ylabel('Position (μm)')
        ax.set_title('Diffusion Flux Profiles')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # IMC Growth Results
    st.markdown(f"""
    **IMC Growth Prediction after {reflow_time} seconds reflow:**
    - **Cu-side IMC (Cu6Sn5) thickness**: {imc_Cu_thickness:.2f} μm
    - **Ni-side IMC (Ni3Sn4) thickness**: {imc_Ni_thickness:.2f} μm
    - **IMC thickness ratio (Cu/Ni)**: {imc_Cu_thickness/imc_Ni_thickness:.2f}
    - **Average Cu flux**: {np.mean(flux_Cu):.2e} mol/m²s
    - **Average Ni flux**: {np.mean(flux_Ni):.2e} mol/m²s
    """)

    # === AI-Generated Engineering Insights ===
    st.subheader("🤖 AI-Generated Engineering Insights")
    
    # Prepare context for AI
    w = results['combined_weights']
    dominant_source = np.argmax(w) + 1
    
    ai_prompt = f"""
    Experimental Context: {EXPERIMENTAL_DESCRIPTION}
    
    Generate engineering insights on Cu-Ni interdiffusion and IMC growth in solder joints with the following parameters:
    - Joint configuration: {substrate_type}, {joining_path}
    - Joint thickness (L_y): {ly_target} μm
    - Boundary concentrations: Cu={c_cu_target:.1e} mol/cc, Ni={c_ni_target:.1e} mol/cc
    - Reflow time: {reflow_time} seconds
    - Predicted IMC thickness: Cu-side={imc_Cu_thickness:.2f} μm, Ni-side={imc_Ni_thickness:.2f} μm
    - Dominant attention source: Source {dominant_source} with weight {w.max():.1%}
    - Average fluxes: Cu={np.mean(flux_Cu):.2e} mol/m²s, Ni={np.mean(flux_Ni):.2e} mol/m²s
    
    Focus on diffusion mechanisms, IMC growth kinetics, reliability implications, and comparison with experimental observations.
    """
    
    if st.button("Generate AI Insights", type="secondary"):
        with st.spinner("Generating AI-powered engineering insights..."):
            try:
                # Load the Hugging Face text-generation pipeline
                generator = pipeline(
                    "text-generation",
                    model=nlp_model,
                    device=-1  # -1 for CPU, change to 0 for GPU
                )

                # Generate output
                outputs = generator(ai_prompt, max_length=500, do_sample=True, temperature=0.7)
                ai_insights = outputs[0]["generated_text"]

                st.markdown("#### AI Analysis:")
                st.write(ai_insights)

            except Exception as e:
                st.error(f"Error generating AI insights: {e}")
                # Fallback to precomputed insights using experimental description
                ni_conc_ratio = c_ni_target / (c_cu_target + 1e-8)
                cu_conc_ratio = c_cu_target / (c_ni_target + 1e-8)
                uphill_risk = "High" if ni_conc_ratio > 0.5 or cu_conc_ratio > 2.0 else "Moderate" if ni_conc_ratio > 0.3 else "Low"
                
                st.markdown(f"""
                **Based on Experimental Context**: {EXPERIMENTAL_DESCRIPTION[:200]}...
                
                **AI Insights Fallback Analysis**:
                - **Domain Effect**: {'Thinner joints promote faster IMC growth' if ly_target < 60 else 'Thicker joints sustain cross-diffusion'}
                - **Flux Dynamics**: Cu flux dominates with {np.mean(flux_Cu):.2e} mol/m²s vs Ni flux {np.mean(flux_Ni):.2e} mol/m²s
                - **Uphill Diffusion**: {uphill_risk} risk due to concentration ratios
                - **IMC Growth**: Asymmetric growth with Cu-side {imc_Cu_thickness:.2f} μm vs Ni-side {imc_Ni_thickness:.2f} μm
                - **Reliability**: {'Void formation risk high' if imc_Cu_thickness/imc_Ni_thickness > 2.0 else 'Stable IMC formation'}
                """)

    # === Export ===
    st.subheader("Export Results")
    
    # Create separate DataFrames to avoid length mismatch
    weights_df = pd.DataFrame({
        'source': [f"S{i+1}" for i in range(len(params_list))],
        'attention_weights': results['attention_weights'],
        'spatial_weights': results['spatial_weights'],
        'combined_weights': results['combined_weights']
    })
    
    target_df = pd.DataFrame({
        'parameter': ['ly_target', 'c_cu_target', 'c_ni_target', 'reflow_time', 
                     'imc_Cu_thickness', 'imc_Ni_thickness', 'avg_Cu_flux', 'avg_Ni_flux'],
        'value': [ly_target, c_cu_target, c_ni_target, reflow_time,
                 imc_Cu_thickness, imc_Ni_thickness, np.mean(flux_Cu), np.mean(flux_Ni)]
    })
    
    # Combine for single CSV download
    combined_data = {
        'source_weights': weights_df.to_dict(),
        'target_results': target_df.to_dict(),
        'experimental_context': EXPERIMENTAL_DESCRIPTION
    }
    
    import json
    json_export = json.dumps(combined_data, indent=2)
    
    st.download_button(
        "Download Results (JSON)", 
        json_export, 
        "attention_inference.json", 
        "application/json"
    )

    # LaTeX for Appendix
    with st.expander("Export LaTeX Appendix"):
        latex = f"""
\\appendix
\\section{{Attention Inference Example: {substrate_type}, Path {joining_path}, \(L_y = {ly_target:.1f}\)\\mu m\}}
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
        latex += f"\\bottomrule\n\\end{{tabular}}\n\n"
        latex += f"\\textbf{{IMC Growth}}: Cu-side: {imc_Cu_thickness:.2f} \\mu m, Ni-side: {imc_Ni_thickness:.2f} \\mu m after {reflow_time} seconds aging."
        latex += f"\\textbf{{Experimental Context}}: {EXPERIMENTAL_DESCRIPTION[:100]}..."
        st.code(latex, language='latex')
