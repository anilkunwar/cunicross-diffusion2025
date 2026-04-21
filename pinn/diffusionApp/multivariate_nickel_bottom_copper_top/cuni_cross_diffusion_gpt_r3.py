#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ATTENTION-BASED DIFFUSION INFERENCE WITH LLM NATURAL LANGUAGE INTERFACE
========================================================================
- Natural language parsing of Cu-Ni interdiffusion parameters (LLM + Regex fallback)
- Hybrid confidence-based parameter extraction
- Cached model loading (GPT-2 / Qwen-2 / Qwen-2.5)
- Manual LRU cache for LLM outputs to prevent Streamlit UnhashableParamError
- Fully preserves original attention interpolation & engineering inference logic
"""
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import re
import json
import hashlib
import warnings
from collections import OrderedDict
from typing import Dict, Any, Optional

# === Safe Transformers Import ===
try:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.warning("⚠️ `transformers` not installed. LLM features will be disabled. Install via `pip install transformers torch`")

warnings.filterwarnings('ignore')

# === PAGE CONFIG ===
st.set_page_config(page_title="LLM-Driven Diffusion Inference", layout="wide")

# =============================================
# 1. LLM LOADER & MANUAL CACHE
# =============================================
@st.cache_resource(show_spinner="Loading LLM backend...")
def load_llm_backend(backend_name: str):
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    if "GPT-2" in backend_name:
        tok = GPT2Tokenizer.from_pretrained('gpt2')
        mod = GPT2LMHeadModel.from_pretrained('gpt2')
    elif "Qwen2-0.5B" in backend_name:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=True)
        mod = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct", torch_dtype="auto", device_map="auto", trust_remote_code=True)
    elif "Qwen2.5" in backend_name:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
        mod = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype="auto", device_map="auto", trust_remote_code=True)
    else:
        return None, None
    mod.eval()
    return tok, mod

# =============================================
# 2. DOMAIN-SPECIFIC NATURAL LANGUAGE PARSER
# =============================================
class DiffusionNLParser:
    """Extracts Cu-Ni diffusion parameters from natural language using Regex + LLM hybrid parsing."""
    def __init__(self):
        self.defaults = {
            'ly_target': 60.0,
            'c_cu_target': 1.5e-3,
            'c_ni_target': 0.5e-3,
            'substrate_type': "Cu(top)-Ni(bottom) Asymmetric",
            'joining_path': "Path I (Cu→Ni)",
            'sigma': 0.20,
            'num_heads': 4,
            'd_head': 8,
            'seed': 42
        }
        self.patterns = {
            'ly_target': [r'(?:joint\s*thickness|domain\s*length|L_y|Ly)\s*[=:]\s*(\d+(?:\.\d+)?)', r'(\d+(?:\.\d+)?)\s*(?:μm|um|microns?)'],
            'c_cu_target': [r'(?:Cu\s*concentration|C_Cu|c_Cu|top\s*concentration)\s*[=:]\s*([\d.]+(?:e[+-]?\d+)?)'],
            'c_ni_target': [r'(?:Ni\s*concentration|C_Ni|c_Ni|bottom\s*concentration)\s*[=:]\s*([\d.]+(?:e[+-]?\d+)?)'],
            'substrate_type': [r'(?:substrate|configuration)\s*[:is=]?\s*(Cu.*Ni.*Asymmetric|Cu/Sn.*Cu.*Symmetric|Ni/Sn.*Ni.*Symmetric)',
                               r'(asymmetric|symmetric)\s*(?:Cu|Ni|joint)'],
            'joining_path': [r'(?:path|joining\s*path|sequence)\s*[:is=]?\s*(Path\s*[I1]\s*.*Cu.*Ni|Path\s*[II2]\s*.*Ni.*Cu|N/A|not\s*applicable)']
        }

    def parse_regex(self, text: str) -> Dict[str, Any]:
        if not text: return self.defaults.copy()
        params = self.defaults.copy()
        text_lower = text.lower()
        
        for key, patterns in self.patterns.items():
            for pat in patterns:
                m = re.search(pat, text_lower, re.IGNORECASE)
                if m:
                    val = m.group(1)
                    try:
                        if key in ['ly_target', 'sigma']:
                            params[key] = float(val)
                        elif key in ['c_cu_target', 'c_ni_target']:
                            params[key] = float(val.replace('e-0', 'e-'))
                        elif key == 'num_heads':
                            params[key] = int(float(val))
                        elif key == 'd_head':
                            params[key] = int(float(val))
                        elif key == 'seed':
                            params[key] = int(float(val))
                        elif key == 'substrate_type':
                            if 'asymmetric' in val.lower():
                                params[key] = "Cu(top)-Ni(bottom) Asymmetric"
                            elif 'cu' in val.lower() and 'symmetric' in val.lower():
                                params[key] = "Cu/Sn2.5Ag/Cu Symmetric"
                            else:
                                params[key] = "Ni/Sn2.5Ag/Ni Symmetric"
                        elif key == 'joining_path':
                            if 'ii' in val.lower() or '2' in val.lower():
                                params[key] = "Path II (Ni→Cu)"
                            elif 'i' in val.lower() or '1' in val.lower():
                                params[key] = "Path I (Cu→Ni)"
                            else:
                                params[key] = "N/A"
                    except:
                        pass
                    break
        return params

    @staticmethod
    def _extract_json_robust(generated: str) -> Optional[Dict]:
        match = re.search(r'\{.*?\}', generated, re.DOTALL)
        if not match: return None
        json_str = match.group(0)
        json_str = re.sub(r'(true|false|null)\s*(")', r'\1,\2', json_str)
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        try: return json.loads(json_str)
        except: return None

    def parse_with_llm(self, text: str, tokenizer, model, regex_params=None, temperature=None) -> Dict:
        if not tokenizer: return self.parse_regex(text)
        if temperature is None:
            temperature = 0.0 if "Qwen" in st.session_state.llm_backend_loaded else 0.1
            
        system = "You are a materials science expert. Extract simulation parameters from the user's query. Reply ONLY with a valid JSON object."
        examples = """
Examples:
- "Analyze a 50 μm joint with Cu concentration 1.2e-3 and Ni 0.8e-3" → {"ly_target": 50.0, "c_cu_target": 1.2e-3, "c_ni_target": 0.8e-3, "substrate_type": "Cu(top)-Ni(bottom) Asymmetric", "joining_path": "Path I (Cu→Ni)", "sigma": 0.2, "num_heads": 4, "d_head": 8, "seed": 42}
- "Symmetric Cu joint, path 2, L_y=80" → {"ly_target": 80.0, "c_cu_target": 1.5e-3, "c_ni_target": 0.5e-3, "substrate_type": "Cu/Sn2.5Ag/Cu Symmetric", "joining_path": "Path II (Ni→Cu)", "sigma": 0.2, "num_heads": 4, "d_head": 8, "seed": 42}
"""
        defaults_json = json.dumps(self.defaults)
        regex_hint = f"\nRegex hint (use as reference): {json.dumps(regex_params or {})}"
        user = f"""{examples}{regex_hint}
Query: "{text}"
JSON keys must be: ly_target, c_cu_target, c_ni_target, substrate_type, joining_path, sigma, num_heads, d_head, seed.
Defaults: {defaults_json}
JSON:"""
        
        if "Qwen" in st.session_state.llm_backend_loaded:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"{system}\n{user}\n"

        try:
            inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=512)
            if torch.cuda.is_available(): inputs = inputs.to('cuda')
            with torch.no_grad():
                out = model.generate(inputs, max_new_tokens=150, temperature=temperature, do_sample=temperature>0, pad_token_id=tokenizer.eos_token_id)
            generated = tokenizer.decode(out[0], skip_special_tokens=True)
            res = self._extract_json_robust(generated)
            if res:
                for k in self.defaults:
                    if k not in res: res[k] = self.defaults[k]
                # Clip & Validate
                res['ly_target'] = np.clip(float(res.get('ly_target', 60)), 30, 120)
                res['c_cu_target'] = np.clip(float(res.get('c_cu_target', 1.5e-3)), 0, 2.9e-3)
                res['c_ni_target'] = np.clip(float(res.get('c_ni_target', 0.5e-3)), 0, 1.8e-3)
                return res
        except Exception as e:
            st.warning(f"LLM failed: {e}. Falling back to regex.")
        return self.parse_regex(text)

    def hybrid_parse(self, text: str, tokenizer, model, use_llm: bool = True) -> Dict:
        regex_params = self.parse_regex(text)
        if use_llm and tokenizer:
            # Simple hash for manual LRU cache
            cache_key = hashlib.md5((text + st.session_state.llm_backend_loaded).encode()).hexdigest()
            if cache_key not in st.session_state.llm_cache:
                llm_res = self.parse_with_llm(text, tokenizer, model, regex_params)
                # LRU eviction
                if len(st.session_state.llm_cache) > 20:
                    st.session_state.llm_cache.popitem(last=False)
                st.session_state.llm_cache[cache_key] = llm_res
            else:
                llm_res = st.session_state.llm_cache[cache_key]
                
            # Confidence merging: prefer LLM for extracted fields, regex for safety clipping
            final = self.defaults.copy()
            for k in final:
                if llm_res[k] != self.defaults[k]:
                    final[k] = llm_res[k]
                elif regex_params[k] != self.defaults[k]:
                    final[k] = regex_params[k]
            return final
        return regex_params

    def get_explanation(self, params: dict, original_text: str) -> str:
        lines = ["### 🔍 Parsed Parameters from Natural Language", f"**Query:** _{original_text}_", "| Parameter | Extracted Value |", "|-----------|-----------------|"]
        for k, v in params.items():
            status = "✅ Extracted" if v != self.defaults[k] else "⚪ Default"
            val = f"{v:.1e}" if isinstance(v, float) and (v < 0.01 or v > 100) else str(v)
            lines.append(f"| {k} | {val} |")
        return "\n".join(lines)

# =============================================
# 3. ORIGINAL ATTENTION MODEL (Unchanged)
# =============================================
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
            return np.array([(ly - 30.0) / (120.0 - 30.0), c_cu / 2.9e-3, c_ni / 1.8e-3])
        else:
            p = np.array(params)
            return np.stack([(p[:, 0] - 30.0) / (120.0 - 30.0), p[:, 1] / 2.9e-3, p[:, 2] / 1.8e-3], axis=1)

    def compute_weights(self, params_list, ly_target, c_cu_target, c_ni_target):
        norm_sources = self.normalize_params(params_list)
        norm_target = self.normalize_params((ly_target, c_cu_target, c_ni_target), is_target=True)
        src_tensor = torch.tensor(norm_sources, dtype=torch.float32)
        tgt_tensor = torch.tensor(norm_target, dtype=torch.float32).unsqueeze(0)
        q = self.W_q(tgt_tensor).view(1, self.num_heads, self.d_head)
        k = self.W_k(src_tensor).view(len(params_list), self.num_heads, self.d_head)
        attn_logits = torch.einsum('nhd,mhd->nmh', k, q) / np.sqrt(self.d_head)
        attn_weights = torch.softmax(attn_logits, dim=0).mean(dim=2).squeeze(1)
        dists = torch.sqrt(((src_tensor[:, 0] - norm_target[0]) / self.sigma)**2 + ((src_tensor[:, 1] - norm_target[1]) / self.sigma)**2 + ((src_tensor[:, 2] - norm_target[2]) / self.sigma)**2)
        spatial_weights = torch.exp(-dists**2 / 2)
        spatial_weights /= spatial_weights.sum() + 1e-8
        combined = attn_weights * spatial_weights
        combined /= combined.sum() + 1e-8
        return {'W_q': self.W_q.weight.data.numpy(), 'W_k': self.W_k.weight.data.numpy(),
                'attention_weights': attn_weights.detach().numpy(), 'spatial_weights': spatial_weights.detach().numpy(),
                'combined_weights': combined.detach().numpy(), 'norm_sources': norm_sources, 'norm_target': norm_target}

# =============================================
# 4. SESSION STATE & INIT
# =============================================
def init_session():
    defaults = {
        'llm_backend_loaded': 'GPT-2 (default)',
        'llm_cache': OrderedDict(),
        'nl_parser': DiffusionNLParser(),
        'parsed_params': DiffusionNLParser().defaults.copy(),
        'nl_query': "",
        'use_llm': True
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# =============================================
# 5. STREAMLIT UI & LOGIC
# =============================================
st.title("🔬 Attention-Driven Inference for Cu-Ni Interdiffusion & IMC Growth")

# === LLM SETUP (Sidebar) ===
with st.sidebar:
    st.header("🤖 LLM Configuration")
    if TRANSFORMERS_AVAILABLE:
        backend = st.selectbox("LLM Backend", ["GPT-2 (default)", "Qwen2-0.5B-Instruct", "Qwen2.5-0.5B-Instruct"], index=0)
        if backend != st.session_state.llm_backend_loaded:
            st.session_state.llm_backend_loaded = backend
            st.session_state.llm_cache.clear()
            st.rerun()
            
        tok, mod = load_llm_backend(backend)
        st.session_state.llm_tokenizer = tok
        st.session_state.llm_model = mod
        st.session_state.use_llm = st.checkbox("Enable LLM Parsing", value=True)
        st.caption(f"Active: **{st.session_state.llm_backend_loaded}**")
    else:
        st.error("Install `transformers` to enable LLM features.")
        st.session_state.use_llm = False
        st.session_state.llm_backend_loaded = "Regex Fallback Only"

    st.header("⚙️ Attention Hyperparameters")
    sigma = st.slider("Locality σ", 0.05, 0.50, st.session_state.parsed_params['sigma'], 0.01)
    num_heads = st.slider("Heads", 1, 8, st.session_state.parsed_params['num_heads'])
    d_head = st.slider("Dim/Head", 4, 16, st.session_state.parsed_params['d_head'])
    seed = st.number_input("Seed", 0, 9999, st.session_state.parsed_params['seed'])

# === NL INPUT ===
st.subheader("📝 Natural Language Query")
templates = {
    "Thin Asymmetric Joint": "Analyze a 40 μm asymmetric Cu-Ni joint with top Cu concentration 1.8e-3 and bottom Ni 0.4e-3 using Path I.",
    "Thick Symmetric Cu": "Simulate a 100 μm symmetric Cu/Sn2.5Ag/Cu joint. Use c_Cu=2.0e-3. Path not applicable.",
    "Ni-Rich Diffusion": "Domain length 60 μm, C_Cu=1.0e-3, C_Ni=1.5e-3. Asymmetric configuration, Path II sequence."
}
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_area("Describe your solder joint configuration:", height=100, key="nl_query", placeholder="e.g., Analyze a 50 μm joint with Cu=1.5e-3, Ni=0.6e-3, asymmetric configuration...")
with col2:
    st.markdown("**Quick Templates:**")
    for name, text in templates.items():
        if st.button(name, use_container_width=True):
            st.session_state.nl_query = text
            st.rerun()

# === PARSE & UPDATE STATE ===
parser = st.session_state.nl_parser
use_llm = st.session_state.get('use_llm', False)
tokenizer = st.session_state.get('llm_tokenizer', None)
model = st.session_state.get('llm_model', None)

parsed = parser.hybrid_parse(query, tokenizer, model, use_llm=use_llm)
st.session_state.parsed_params = parsed
st.markdown(parser.get_explanation(parsed, query))

# === TARGET INPUTS (Pre-filled by Parser, User can Override) ===
st.subheader("🎯 Target Joint Configuration (Override if needed)")
col1, col2 = st.columns(2)
with col1:
    substrate_type = st.selectbox("Substrate Configuration", 
        ["Cu(top)-Ni(bottom) Asymmetric", "Cu/Sn2.5Ag/Cu Symmetric", "Ni/Sn2.5Ag/Ni Symmetric"],
        index=["Cu(top)-Ni(bottom) Asymmetric", "Cu/Sn2.5Ag/Cu Symmetric", "Ni/Sn2.5Ag/Ni Symmetric"].index(parsed['substrate_type']))
    joining_path = st.selectbox("Joining Path (for Asymmetric)", 
        ["Path I (Cu→Ni)", "Path II (Ni→Cu)", "N/A"],
        index=["Path I (Cu→Ni)", "Path II (Ni→Cu)", "N/A"].index(parsed['joining_path']))
    ly_target = st.slider("Joint Thickness \(L_y\) (μm)", 30.0, 120.0, parsed['ly_target'], 1.0)
with col2:
    c_cu_target = st.number_input("Top BC \(C_{Cu}\) (mol/cc)", 0.0, 2.9e-3, parsed['c_cu_target'], 1e-4, format="%.1e")
    c_ni_target = st.number_input("Bottom BC \(C_{Ni}\) (mol/cc)", 0.0, 1.8e-3, parsed['c_ni_target'], 1e-5, format="%.1e")

# === SOURCE SOLUTIONS ===
st.subheader("📦 Precomputed Source Simulations (PINN-Generated)")
num_sources = st.slider("Number of Sources", 2, 6, 3)
params_list = []
for i in range(num_sources):
    with st.expander(f"Source {i+1}"):
        c1, c2, c3 = st.columns(3)
        ly = c1.number_input(f"L_y", 30.0, 120.0, 30.0 + 30*i, 0.1, key=f"ly_{i}")
        c_cu = c2.number_input(f"C_Cu", 0.0, 2.9e-3, 1.5e-3, 1e-4, format="%.1e", key=f"ccu_{i}")
        c_ni = c3.number_input(f"C_Ni", 0.0, 1.8e-3, 0.1e-3 + 0.4e-3*i, 1e-5, format="%.1e", key=f"cni_{i}")
        params_list.append((ly, c_cu, c_ni))

# === RUN INFERENCE ===
if st.button("🚀 Run Attention Inference", type="primary", use_container_width=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    with st.spinner("Interpolating diffusion profiles and inferring joint behavior..."):
        interpolator = MultiParamAttentionInterpolator(sigma, num_heads, d_head)
        results = interpolator.compute_weights(params_list, ly_target, c_cu_target, c_ni_target)
    st.success("✅ Inference Complete!")

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.subheader("Hybrid Attention Weights")
        df_weights = pd.DataFrame({'Source': [f"S{i+1}" for i in range(len(params_list))],
                                   'Attention': np.round(results['attention_weights'], 4),
                                   'Gaussian': np.round(results['spatial_weights'], 4),
                                   'Hybrid': np.round(results['combined_weights'], 4)})
        st.dataframe(df_weights.style.bar(subset=['Hybrid'], color='#5fba7d'), use_container_width=True)

        fig, ax = plt.subplots()
        src = results['norm_sources']
        tgt = results['norm_target']
        sc = ax.scatter(src[:, 0], src[:, 1], c=src[:, 2], s=100, cmap='plasma', label='Sources', edgecolors='k')
        ax.scatter(tgt[0], tgt[1], c=tgt[2], s=300, marker='*', cmap='plasma', edgecolors='red', label='Target')
        ax.set_xlabel("Norm. $L_y$")
        ax.set_ylabel("Norm. $C_{Cu}$")
        ax.set_title("Parameter Space")
        ax.legend()
        plt.colorbar(sc, ax=ax).set_label("Norm. $C_{Ni}$")
        st.pyplot(fig)

    with col2:
        st.subheader("Projection Matrices")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        sns.heatmap(results['W_q'], ax=ax1, cmap='coolwarm', center=0, cbar=False)
        ax1.set_title("$W_q$")
        sns.heatmap(results['W_k'], ax=ax2, cmap='coolwarm', center=0)
        ax2.set_title("$W_k$")
        st.pyplot(fig)

    # === ENGINEERING INSIGHTS ===
    st.subheader("📊 Engineering Insights: Diffusion Dynamics and IMC Growth Kinetics")
    w = results['combined_weights']
    dominant_source = np.argmax(w) + 1
    ni_conc_ratio = c_ni_target / (c_cu_target + 1e-8)
    cu_conc_ratio = c_cu_target / (c_ni_target + 1e-8)
    uphill_risk = "High" if ni_conc_ratio > 0.5 or cu_conc_ratio > 2.0 else "Moderate" if ni_conc_ratio > 0.3 else "Low"
    imc_growth = "Faster on Ni side" if "Asymmetric" in substrate_type else "Symmetric"
    void_risk = "High (Kirkendall voids in Cu3Sn)" if "Cu Symmetric" in substrate_type else "Suppressed by Ni addition"
    path_effect = ""
    if joining_path == "Path I (Cu→Ni)":
        path_effect = "Lower Ni content in Cu/Sn interface IMC; thinner (Cu,Ni)6Sn5 on Cu side compared to Path II."
    elif joining_path == "Path II (Ni→Cu)":
        path_effect = "Higher Ni content in Cu/Sn interface IMC; thicker (Cu,Ni)6Sn5 on Cu side due to initial Ni saturation in solder."

    st.markdown(f"""
    Based on the attention-interpolated diffusion profiles (dominant blend from Source S{dominant_source} at {w.max():.1%} weight):

    - **Domain Length Effect (\(L_y = {ly_target:.1f}\) μm)**: {'Thinner joints (e.g., 50 μm)' if ly_target < 60 else 'Thicker joints (e.g., 90 μm)'} promote {'faster IMC growth due to steeper concentration gradients' if ly_target < 60 else 'sustained cross-diffusion and potential for more isolated (Cu,Ni)6Sn5 colonies in solder matrix'}.
    - **Boundary Concentrations & Flux Dynamics**: Top \(C_{{Cu}} = {c_cu_target:.1e}\) mol/cc, Bottom \(C_{{Ni}} = {c_ni_target:.1e}\) mol/cc. High Cu solubility in Sn accelerates Cu6Sn5 formation; Ni diffusivity is lower.
    - **Uphill Diffusion & Cross-Interaction**: {uphill_risk} risk of counter-gradient Ni flux into Cu-rich zones, enhancing vacancy supersaturation and Kirkendall effects.
    - **Substrate Type Impact**: In {substrate_type}, IMC morphology is {'scallop-shaped Cu6Sn5' if 'Cu Symmetric' in substrate_type else 'rod-shaped (Cu,Ni)6Sn5/Ni3Sn4' if 'Ni Symmetric' in substrate_type else 'asymmetric with faster growth on Ni UBM'}. Void formation: {void_risk}.
    - **Joining Path Dependence**: {path_effect if joining_path != "N/A" else "Not applicable for symmetric configurations."}
    - **IMC Growth Kinetics**: Ni addition suppresses porous Cu3Sn formation after thermal cycling, reducing voids by 20-50% and improving reliability.
    """)

    # === EXPORT ===
    buffer = io.StringIO()
    export_df = pd.DataFrame({
        'attention_weights': results['attention_weights'],
        'spatial_weights': results['spatial_weights'],
        'combined_weights': results['combined_weights'],
        'W_q_row0': results['W_q'][0],
        'W_k_row0': results['W_k'][0]
    })
    csv = export_df.to_csv(index=False)
    st.download_button("⬇️ Download Results (CSV)", csv, "attention_inference.csv", "text/csv")

    with st.expander("📜 Export LaTeX Appendix"):
        latex = f"""
\\appendix
\\section{{Attention Inference Example: {substrate_type}, Path {joining_path}, \(L_y = {ly_target:.1f}\)\\mu m\}}
\\textbf{{Target}}: \\(\\theta^* = ({ly_target:.1f}, {c_cu_target:.1e}, {c_ni_target:.1e})\\)

\\textbf{{Weights}}:
\\begin{{tabular}}{{lccc}}
\\toprule
Source & Attention & Gaussian & Hybrid \\\\
\\midrule
"""
        for i in range(len(w)):
            latex += f"S{i+1} & {results['attention_weights'][i]:.3f} & {results['spatial_weights'][i]:.3f} & {w[i]:.3f} \\\\\n"
        latex += "\\bottomrule\n\\end{tabular}\n\n\\textbf{Inference}: " + uphill_risk + " uphill risk → " + imc_growth + " IMC growth."
        st.code(latex, language='latex')
