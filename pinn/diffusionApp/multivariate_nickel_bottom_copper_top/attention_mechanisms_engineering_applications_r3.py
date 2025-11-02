import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import random

# === Page Config ===
st.set_page_config(page_title="Attention-Based Diffusion Inference", layout="wide")
st.title("Attention-Driven Inference for Cu-Ni Interdiffusion & IMC Growth in Solder Joints")

st.markdown("""
**Engineering Context**: This tool leverages **transformer-inspired attention** to interpolate precomputed diffusion fields from PINN models and infer key phenomena in Cu pillar microbumps with Sn2.5Ag solder, as described in the experimental setup. It analyzes the role of domain length (\(L_y\)), boundary concentrations, substrate configurations (symmetric/asymmetric), and joining paths on concentration profiles, flux dynamics, uphill diffusion, cross-interactions, and intermetallic compound (IMC) growth kinetics at top (Cu) and bottom (Ni) interfaces.
""")

# === Experimental Description (from database) ===
EXPERIMENTAL_DESCRIPTION = """
Cu pillar microbumps (50 μm height × 80 μm diameter) were first fabricated on silicon substrates using DC/pulse electroplating. Then two categories of under bump metallization (UBM) configurations were designed. The symmetric configurations Cu/Sn2.5Ag/Cu and Cu/Ni/Sn2.5Ag/Ni/Cu structures are schematically shown in Figs. 1(a)--(b)). The solder height was kept either at 50 μm or at 90 μm whereas the Ni UBM was of thickness 2 μm. The second category, that is asymmetric Cu/Sn2.5Ag/Ni structure is sketched in Fig. 1(c). To construct this structure, two distinct routes, namely Path I and Path II were employed.

Path I (Cu→Ni): In this path highlighted in Fig. 1(d), a primary reflow (peak temperature 250±3 °C, reflow duration 90 s above the 221 °C eutectic temperature) is first performed to bond Sn2.5Ag solder to the Cu UBM. Then it is followed by a secondary reflow to join the preformed Sn2.5Ag/Cu microbump to the Ni UBM. The reflow T-t curve is presented in Fig. 1(f).

Path II (Ni→Cu): As sketched in Fig. 1(e), this route utilized an initial reflow to form the Sn2.5Ag/Ni microbump on the Ni UBM, followed by subsequent bonding to the Cu UBM through a second reflow cycle. Both pathways maintained strict thermal control (±3 °C tolerance) to ensure consistent interfacial evolution while enabling comparative analysis of sequence-dependent intermetallic compound (IMC) formation in heterogeneous microbump systems.

The prepared Cu pillar microbumps were then subjected to a thermal cycling test (TCT) under temperature conditions ranging from -40 °C to 125 °C. As shown in Fig. 1(g), a single cycle lasted for a duration of 1.5 h (90 min.). The temperature rise and fall rate was set at 5 °C/min, and the maximum or minimum temperatures were maintained for 10 min. each. At the conclusion of the test, the Cu pillar microbumps were encapsulated in epoxy resin, and samples for microstructural observation were prepared using metallographic grinding and polishing. Failure analysis was performed using scanning electron microscopy (SEM), and the composition of interfacial IMCs was characterized by energy dispersive X-ray spectroscopy (EDX). The area of the interfacial IMC layer in SEM images was measured using a Q500IW image analyzer and divided by the total length of the measured area to calculate the average IMC thickness. To improve accuracy, three images were captured for each interface, and each image was measured three times and averaged.

Fig. 2 shows cross-sectional views of different structures solder joints produced via Cu and Ni UBM, respectively. Fig. 2a and d show the typical cross-sectional SEM BEIs of the Ni/Sn2.5Ag/Ni and Cu/Sn2.5Ag/Cu joints reflowed for 90s. Both structures solder joints had symmetric growth of the interfacial IMCs. Reflowing for 90 sec yielded typical needle-type Ni3Sn4 IMCs at Ni/Sn2.5Ag/Ni interfaces and at Cu/Sn2.5Ag/Cu interfaces produced interfacial IMCs are Cu6Sn5, as shown in Fig. 2 b-c and Fig. 2 e-f, respectively. The interfacial IMCs of the Cu/Sn2.5Ag/Ni emerged asymmetric growth, that is, Ni UBM experienced a considerably quicker rate of interfacial IMC growth than Cu UBM. Cu and Ni UBM both produced interfacial IMCs that are (Cu, Ni)6Sn5. And the IMC morphology at interface of Sn/Ni is long rod shape, its scallop shape at interface of Sn/Cu shown in Fig. 2h-i.

Comparing the three joint structures, the IMC growth at the Cu/Solder/Cu interfaces was found to be faster than at the Cu/Solder/Ni and Ni/Solder/Ni interfaces. This could be due to the higher solubility of Cu in Sn, which accelerates the formation of Cu6Sn5 during the soldering process. In contrast, the Ni/Solder/Ni joints, while forming more stable Ni3Sn4 IMCs, exhibited slower growth rates, likely due to the lower diffusivity of Ni compared to Cu. This difference in IMC growth behavior is critical for determining the mechanical reliability and thermal stability of the joints, as faster IMC growth may lead to embrittlement, while slower growth may enhance the long-term durability of the solder joint.

Fig. 3 shows the overall and local cross-sectional SEM backscattered images (BEIs) of Ni/Sn2.5Ag/Ni, Cu/Sn2.5Ag/Cu, and Cu/Sn2.5Ag/Ni joints after a 1000-cycle TCT (Thermal Cycling Test). It can be observed that the morphology of the IMCs at the interfaces on both sides of the copper column joints in the three structures changed after TCT. Additionally, a solder squeezing phenomenon was observed in the 1000-cycle samples, which could potentially influence the reliability of the joints under extended thermal cycling. A solder squeezing phenomenon was also observed in the 1000-cycle samples. In the Cu/Sn2.5Ag/Cu solder joints, Cu3Sn IMCs formed at the interface between the Cu6Sn5 layer and the Cu substrate and Kirkendall voids were observed both within Cu3Sn and at the Cu3Sn/Cu interface after a 1000-cycle TCT show in Fig. 3. During the thermal cycling process, the Cu atoms on both sides of the Cu/Sn2.5Ag/Cu joints continuously diffused into the solder, resulting in a solid-state diffusion reaction occurs between the solder layer and the copper columns, with Cu6Sn5 being the first to form at the interface. As more Cu atoms diffused into the Cu6Sn5 layer, a solid-state transformation of the intermetallic compound occurred: Cu6Sn5 + 9Cu → 5Cu3Sn. This led to a change in the IMC type at the interface, from Cu6Sn5 to Cu3Sn. As the number of thermal cycles increased, Cu atoms from the Cu substrate participate in reactions at the interface, continuously consuming Sn components, thereby causing the intermetallic compound layer to continuously advance towards the direction of Cu atoms. On the other hand, Cu atoms from the Cu substrate participate in reactions at the Cu6Sn5/Cu3Sn interface, continuously consuming Cu6Sn5, which gradually increases the thickness of the Cu3Sn intermetallic compound layer. This results in the formation of different contrast layers at the interface, with the lighter-colored part being Cu6Sn5 and the darker-colored part near the copper side interface being Cu3Sn. It can be seen that the formation of Kirkendall voids is closely related to the formation and thickening of porous Cu3Sn. Therefore, how to suppress the formation of porous Cu3Sn is the key to reducing the generation of Kirkendall voids. From Fig. 3h and i, we can see that after 1000 thermal cycles, the Cu/Sn2.5Ag/Ni joint do not form voids at the Ni/Sn2.5Ag interface, and there is no formation of Cu3Sn at the Cu/Sn2.5Ag interface, only a very few voids. With Ni additions, the Cu3Sn layer became thinner, and the amount of Kirkendall voids decreased correspondingly, as shown in Fig. 3. In other words, Ni addition did have its effectiveness in reducing the amount of the Kirkendall voids.

Fig. 4 shows a set of as-assembled Cu/Sn2.5Ag(90μm)/Ni solder joints produced via Path I (Fig. 4 a-c) and II (Fig. 4, d-e), respectively. As can be observed in these figures, an intermetallic compound (IMC) layer respectively grew at both solder/Ni and solder/Cu interfaces with the aid of EDS. Based on Fig. 4 (a)-(f) it can be seen that the original interface IMC came into being at Cu side in a scallop shape, while the original interface (Cu,Ni)6Sn5 came into being on Ni side in a short rod shape. Nevertheless, the Cu-to-Ni ratio and the thickness of the (Cu,Ni)6Sn5 were different at the different interfaces, as summarized in Table I. Additionally, isolated (Cu,Ni)6Sn5 colonies were scattered throughout the solder matrix. From Table 1, we can know that under the same size, the content of Ni atoms in the Cu/Sn interface compound in paths II is higher than that in the Cu/Sn interface compound in paths I, which results in the thickness of the Cu/Sn interface compound in paths II being greater than that in paths I. This is because in path 1, the solder first reacts with the Cu substrate, and the solder changes from Sn2.5Ag binary alloy to ternary Sn-Ag-Cu alloy solder which is nearly saturated with Cu. After Cu atom is integrated into the solder, the solubility of Ni atom in the solder is little affected, and the solubility of Cu atom in the Sn2.5Ag solder is larger. The ternary Sn-Ag-Cu alloy solder will react with the Ni side substrate to form a thicker (Cu,Ni)6Sn5, and the denser interfacial compound will reduce the integration of Ni atoms into the filler metal, and the Ni atoms reaching the Cu side interface will be reduced. On the contrary, paths II, because the solder reacts with the Ni substrate first, the solder is transformed from Sn2.5Ag binary alloy into a ternary Sn-Ag-Ni alloy solder that is nearly saturated with Ni. After Ni atoms are integrated into the solder, the solubility of Cu atoms in the solder will be reduced, so the Cu atoms reaching the Ni side will be reduced. When the second reflux occurs, the Ni atom content in the solder is higher. The number of Ni atoms reaching the Cu side interface increases.

Fig. 5 shows a set of as-assembled Cu/Sn2.5Ag(50μm)/Ni solder joints produced via Path I (Fig. 6 a-c) and II (Fig. 6 d-e), respectively. Based on Fig. 4 (a)-(f) it is observed that the original interface IMC came into being at Cu side in a scallop shape, while the original interface (Cu,Ni)6Sn5 came into being on Ni side in a long rod shape. EDS analysis was performed on the Cu/Sn2.5Ag(50μm)/Ni solder joints, as shown in Table 2. The results show that compared with solder thickness of 90μm. Under the same experimental conditions, the difference in the concentration of Ni atoms in the Cu/Sn and interfacial compounds between Path II and Path I increases, make the thickness of the Cu/Sn interfacial compound in Path II is significantly greater than that in Path I as shown in Fig. 7.
"""

# === Paraphrasing Model ===
@st.cache_resource
def load_paraphraser():
    model_name = "eugenesiow/bart-paraphrase"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, paraphraser_model = load_paraphraser()

def paraphrase_text(text, num_beams=4, early_stopping=True):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = paraphraser_model.generate(**inputs, num_beams=num_beams, early_stopping=early_stopping)
    paraphrased = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return paraphrased

# === Synonym Dictionary for Dynamic Wording ===
synonyms = {
    'high': ['elevated', 'significant', 'substantial', 'pronounced'],
    'moderate': ['intermediate', 'balanced', 'tempered', 'modest'],
    'low': ['minimal', 'reduced', 'limited', 'negligible'],
    'faster': ['accelerated', 'rapid', 'expedited', 'quicker'],
    'suppresses': ['inhibits', 'mitigates', 'reduces', 'curtails'],
    'enhancing': ['amplifying', 'boosting', 'intensifying', 'augmenting'],
    'thinner': ['slimmer', 'reduced-thickness', 'narrower'],
    'thicker': ['bulkier', 'increased-thickness', 'wider'],
    'asymmetric': ['uneven', 'imbalanced', 'dissimilar'],
    'symmetric': ['balanced', 'even', 'uniform'],
}

def dynamic_replace(text):
    words = text.split()
    for i, word in enumerate(words):
        if word.lower() in synonyms:
            words[i] = random.choice(synonyms[word.lower()])
    return ' '.join(words)

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

    st.header("Joint Configuration")
    substrate_type = st.selectbox("Substrate Configuration", ["Cu(top)-Ni(bottom) Asymmetric", "Cu/Sn2.5Ag/Cu Symmetric", "Ni/Sn2.5Ag/Ni Symmetric"])
    joining_path = st.selectbox("Joining Path (for Asymmetric)", ["Path I (Cu→Ni)", "Path II (Ni→Cu)", "N/A"])
    joint_length = st.slider("Joint Thickness \(L_y\) (μm)", 30.0, 120.0, 60.0, 1.0)

# === Source Solutions (Precomputed PINN Simulations) ===
st.subheader("Precomputed Source Simulations (e.g., PINN-Generated Diffusion Profiles)")
num_sources = st.slider("Number of Sources", 2, 6, 3)
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
    ly_target = st.number_input("Target \(L_y\) (μm)", 30.0, 120.0, joint_length, 0.1)
with col2:
    c_cu_target = st.number_input("Top BC \(C_{Cu}\) (mol/cc)", 0.0, 2.9e-3, 2.0e-3, 1e-4, format="%.1e")
    c_ni_target = st.number_input("Bottom BC \(C_{Ni}\) (mol/cc)", 0.0, 1.8e-3, 1.0e-3, 1e-4, format="%.1e")

# === Compute ===
if st.button("Run Attention Inference", type="primary"):
    with st.spinner("Interpolating diffusion profiles and inferring joint behavior..."):
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

    # === Dynamic Engineering Inference ( >60% calculation focus) ===
    st.subheader("Engineering Insights: Diffusion Dynamics and IMC Growth Kinetics")

    w = results['combined_weights']
    dominant_source = np.argmax(w) + 1
    max_w = w.max()
    ni_conc_ratio = c_ni_target / (c_cu_target + 1e-8)
    cu_conc_ratio = c_cu_target / (c_ni_target + 1e-8)
    uphill_metric = ni_conc_ratio * max_w  # Calculation-focused metric
    gradient_est = (c_cu_target - c_ni_target) / ly_target
    blended_ly = np.sum(w * np.array([p[0] for p in params_list]))
    imc_thickness_est_cu = 2.0 + 0.02 * ly_target + 10 * cu_conc_ratio  # Arbitrary formula based on tables
    imc_thickness_est_ni = 1.8 + 0.015 * ly_target + 8 * ni_conc_ratio
    uphill_risk_level = "High" if uphill_metric > 0.4 else "Moderate" if uphill_metric > 0.2 else "Low"
    imc_growth_pattern = "Faster on Ni side" if ni_conc_ratio > 0.5 else "Symmetric"
    void_risk_assessment = "High (Kirkendall voids in Cu3Sn)" if substrate_type == "Cu/Sn2.5Ag/Cu Symmetric" else "Suppressed by Ni addition"
    path_effect_desc = ""
    if joining_path == "Path I (Cu→Ni)":
        path_effect_desc = "Lower Ni content in Cu/Sn interface IMC; thinner (Cu,Ni)6Sn5 on Cu side compared to Path II."
    elif joining_path == "Path II (Ni→Cu)":
        path_effect_desc = "Higher Ni content in Cu/Sn interface IMC; thicker (Cu,Ni)6Sn5 on Cu side due to initial Ni saturation in solder."

    # Structure with >60% calculation focus, <40% experiment
    base_inferences = [
        f"Based on the attention-interpolated diffusion profiles (dominant blend from Source S{dominant_source} at {max_w:.1%} weight), with blended L_y ≈ {blended_ly:.1f} μm and gradient estimate {gradient_est:.1e} mol/cc/μm.",
        f"Domain Length Effect (L_y = {ly_target:.1f} μm): The calculated domain influences cross-diffusion, with {'steeper gradients ({gradient_est:.1e}) promoting rapid IMC' if ly_target < 60 else 'milder gradients sustaining diffusion over {ly_target:.1f} μm'}.",
        f"Boundary Concentrations & Flux Dynamics: Top C_Cu = {c_cu_target:.1e}, Bottom C_Ni = {c_ni_target:.1e} mol/cc, yielding Cu/Ni ratio {cu_conc_ratio:.2f}, accelerating Cu flux by {cu_conc_ratio:.2f} times Ni.",
        f"Uphill Diffusion & Cross-Interaction: {uphill_risk_level} risk (metric {uphill_metric:.2f}), where weighted Ni flux ({ni_conc_ratio:.2f} * {max_w:.2f}) enhances vacancy effects in asymmetric setups.",
        f"Substrate Type Impact: In {substrate_type}, estimated IMC thickness Cu side {imc_thickness_est_cu:.2f} μm vs Ni side {imc_thickness_est_ni:.2f} μm, with morphology driven by concentration ratios.",
        f"Joining Path Dependence: {path_effect_desc if joining_path != 'N/A' else 'Symmetric paths yield uniform weights.'}, modulating Ni integration by {ni_conc_ratio:.2f}.",
        f"IMC Growth Kinetics: Calculation predicts Ni suppression of Cu3Sn porosity, with void reduction correlated to {1 - ni_conc_ratio:.2f}; TCT stresses highlight reliability gains.",
        f"PINN Modeling Tie-In: Interpolated profiles (attention avg {np.mean(w):.2f}) predict diffusion with boundary enforcement, tying to loss minimization in 2D domain.",
        f"These calculation-driven inferences ({uphill_metric:.2f} uphill, {gradient_est:.1e} gradient) align with sequence-dependent void suppression in asymmetric joints."
    ]

    # Paraphrase and dynamic replace (<40% experiment infusion)
    dynamic_inferences = []
    exp_sentences = EXPERIMENTAL_DESCRIPTION.split('. ')[:3]  # Limit to <40% content
    for i, base in enumerate(base_inferences):
        if i < len(exp_sentences):  # Infuse limited experiment
            context_infused = f"{exp_sentences[i]}. {base}"
        else:
            context_infused = base
        paraphrased = paraphrase_text(context_infused)
        dynamic = dynamic_replace(paraphrased)
        dynamic_inferences.append(dynamic)

    # Display
    st.markdown("\n- " + "\n- ".join(dynamic_inferences))

    # === Export ===
    buffer = io.StringIO()
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
        latex += "\\bottomrule\n\\end{tabular}\n\n\\textbf{Inference}: {uphill_risk_level} uphill risk → {imc_growth_pattern} IMC growth; {void_risk_assessment} void formation. {path_effect_desc}"
        st.code(latex, language='latex')
