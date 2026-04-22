#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ATTENTION-BASED CU-NI INTERDIFFUSION VISUALIZER WITH LLM NATURAL LANGUAGE INTERFACE
====================================================================================
- Natural language parsing of diffusion parameters (regex + GPT-2/Qwen hybrid)
- Multi-head attention with spatial locality for physics-aware interpolation
- Publication-quality 2D heatmaps, centerline curves, and parameter sweeps
- Full figure customization and PNG/PDF export
- Cached LLM loading, robust JSON extraction, and confidence-based fallbacks
- EXTRACTABLE CONCENTRATION FEATURES: profiles, gradients, integrated totals, CSV export
"""
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
import json
import hashlib
from collections import OrderedDict
from typing import Dict, Any, Optional, List, Tuple

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

# Available colormaps for selection
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

# =============================================
# LLM IMPORT WITH GRACEFUL FALLBACK
# =============================================
try:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.warning("⚠️ `transformers` not installed. LLM features will be disabled. Install via `pip install transformers torch`")

# =============================================
# NATURAL LANGUAGE PARSER (Hybrid Regex + LLM)
# =============================================
class DiffusionNLParser:
    """Extracts Cu-Ni diffusion parameters from natural language using regex + LLM hybrid parsing."""
    
    def __init__(self):
        self.defaults = {
            'ly_target': 60.0,
            'c_cu_target': 1.5e-3,
            'c_ni_target': 0.5e-3,
            'sigma': 0.20,
        }
        # Flexible patterns: delimiter (= or :) is optional, simple "Cu 1.2e-3" works
        self.patterns = {
            'ly_target': [
                r'(?:joint\s*thickness|domain\s*length|L_y|Ly)\s*[=:]?\s*(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*(?:μm|um|microns?)',
            ],
            'c_cu_target': [
                r'(?:Cu\s*concentration|C_Cu|c_Cu|top\s*concentration)\s*[=:]?\s*([\d.]+(?:e[+-]?\d+)?)',
                r'Cu\s*[=:]?\s*([\d.]+(?:e[+-]?\d+)?)',
            ],
            'c_ni_target': [
                r'(?:Ni\s*concentration|C_Ni|c_Ni|bottom\s*concentration)\s*[=:]?\s*([\d.]+(?:e[+-]?\d+)?)',
                r'Ni\s*[=:]?\s*([\d.]+(?:e[+-]?\d+)?)',
            ],
        }

    def parse_regex(self, text: str) -> Dict[str, Any]:
        """Extract parameters using flexible regex patterns."""
        if not text:
            return self.defaults.copy()
        params = self.defaults.copy()
        text_lower = text.lower()
        
        for key, patterns in self.patterns.items():
            for pat in patterns:
                match = re.search(pat, text_lower, re.IGNORECASE)
                if match:
                    val = match.group(1)
                    try:
                        if key in ['ly_target']:
                            params[key] = float(val)
                        elif key in ['c_cu_target', 'c_ni_target']:
                            # Python's float handles scientific notation natively
                            params[key] = float(val)
                    except (ValueError, TypeError):
                        pass
                    break
        # Clip to valid ranges
        params['ly_target'] = np.clip(params['ly_target'], 30.0, 120.0)
        params['c_cu_target'] = np.clip(params['c_cu_target'], 0.0, 2.9e-3)
        params['c_ni_target'] = np.clip(params['c_ni_target'], 0.0, 1.8e-3)
        return params

    @staticmethod
    def _extract_json_robust(generated: str) -> Optional[Dict]:
        """Robustly extract JSON from LLM output with repair attempts."""
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        match = re.search(json_pattern, generated, re.DOTALL)
        if not match:
            match = re.search(r'\{.*?\}', generated, re.DOTALL)
        if not match:
            return None
        json_str = match.group(0)
        json_str = re.sub(r'(true|false|null)\s*(")', r'\1,\2', json_str)
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

    def parse_with_llm(self, text: str, tokenizer, model, regex_params: Dict = None, temperature: float = None) -> Dict:
        """Use LLM (GPT-2 or Qwen) to extract parameters from natural language."""
        if not tokenizer or not model:
            return self.parse_regex(text)
        
        if temperature is None:
            backend = st.session_state.get('llm_backend_loaded', 'GPT-2')
            temperature = 0.0 if "Qwen" in backend else 0.1
        
        system = "You are a materials science expert. Extract simulation parameters from the user's query. Reply ONLY with a valid JSON object."
        examples = """
Examples:
- "Analyze a 50 μm joint with Cu concentration 1.2e-3 and Ni 0.8e-3" → {"ly_target": 50.0, "c_cu_target": 1.2e-3, "c_ni_target": 0.8e-3, "sigma": 0.2}
- "Domain length 80, C_Cu=2.0e-3, C_Ni=1.0e-3" → {"ly_target": 80.0, "c_cu_target": 2.0e-3, "c_ni_target": 1.0e-3, "sigma": 0.2}
- "Ly=45um, top Cu=1.5e-3 mol/cc, bottom Ni=0.3e-3" → {"ly_target": 45.0, "c_cu_target": 1.5e-3, "c_ni_target": 0.3e-3, "sigma": 0.2}
"""
        defaults_json = json.dumps(self.defaults)
        regex_hint = f"\nRegex hint (use as reference): {json.dumps(regex_params or {})}" if regex_params else ""
        
        user = f"""{examples}{regex_hint}
Query: "{text}"
JSON keys must be: ly_target (float, 30-120 μm), c_cu_target (float, 0-2.9e-3 mol/cc), c_ni_target (float, 0-1.8e-3 mol/cc), sigma (float, 0.05-0.5).
Defaults: {defaults_json}
JSON:"""
        
        if "Qwen" in st.session_state.get('llm_backend_loaded', ''):
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"{system}\n{user}\n"
        
        try:
            inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=200,
                    temperature=temperature,
                    do_sample=(temperature > 0),
                    pad_token_id=tokenizer.eos_token_id
                )
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            result = self._extract_json_robust(generated)
            if result:
                for key in self.defaults:
                    if key not in result:
                        result[key] = self.defaults[key]
                result['ly_target'] = np.clip(float(result.get('ly_target', 60)), 30, 120)
                result['c_cu_target'] = np.clip(float(result.get('c_cu_target', 1.5e-3)), 0, 2.9e-3)
                result['c_ni_target'] = np.clip(float(result.get('c_ni_target', 0.5e-3)), 0, 1.8e-3)
                if regex_params:
                    for key in ['ly_target', 'c_cu_target', 'c_ni_target']:
                        if key in regex_params and key in result:
                            if abs(result[key] - regex_params[key]) > 1e-4:
                                result[key] = regex_params[key]
                return result
        except Exception as e:
            st.warning(f"LLM parsing failed: {e}. Falling back to regex.")
        return self.parse_regex(text)

    def hybrid_parse(self, text: str, tokenizer, model, use_llm: bool = True) -> Dict:
        """Run regex first, then optionally LLM, and merge based on confidence."""
        regex_params = self.parse_regex(text)
        if use_llm and tokenizer and model:
            cache_key = hashlib.md5((text + st.session_state.get('llm_backend_loaded', '')).encode()).hexdigest()
            if 'llm_cache' not in st.session_state:
                st.session_state.llm_cache = OrderedDict()
            if cache_key in st.session_state.llm_cache:
                llm_params = st.session_state.llm_cache[cache_key]
            else:
                llm_params = self.parse_with_llm(text, tokenizer, model, regex_params)
                if len(st.session_state.llm_cache) > 20:
                    st.session_state.llm_cache.popitem(last=False)
                st.session_state.llm_cache[cache_key] = llm_params
            final = self.defaults.copy()
            for key in final:
                if llm_params[key] != self.defaults[key]:
                    final[key] = llm_params[key]
                elif regex_params[key] != self.defaults[key]:
                    final[key] = regex_params[key]
            return final
        return regex_params

    def get_explanation(self, params: dict, original_text: str) -> str:
        """Generate a markdown table explaining parsed parameters."""
        lines = ["### 🔍 Parsed Parameters from Natural Language", f"**Query:** _{original_text}_", "| Parameter | Extracted Value | Status |", "|-----------|-----------------|--------|"]
        for key, val in params.items():
            if key == 'sigma':
                continue
            status = "✅ Extracted" if val != self.defaults[key] else "⚪ Default"
            if isinstance(val, float):
                val_str = f"{val:.1e}" if (val < 0.01 or val > 100) else f"{val:.3f}"
            else:
                val_str = str(val)
            lines.append(f"| {key} | {val_str} | {status} |")
        return "\n".join(lines)


# =============================================
# UNIFIED LLM LOADER WITH CACHING
# =============================================
@st.cache_resource(show_spinner="Loading LLM backend...")
def load_llm(backend_name: str):
    """Load tokenizer and model for specified backend. Cached forever per backend."""
    if not TRANSFORMERS_AVAILABLE:
        return None, None, "Regex Fallback Only"
    
    if "GPT-2" in backend_name:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
    elif "Qwen2-0.5B" in backend_name:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct", torch_dtype="auto", device_map="auto", trust_remote_code=True)
    elif "Qwen2.5" in backend_name:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype="auto", device_map="auto", trust_remote_code=True)
    else:
        return None, None, "Unknown Backend"
    
    model.eval()
    return tokenizer, model, backend_name


# =============================================
# ORIGINAL SOLUTION LOADING (UNCHANGED)
# =============================================
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
                    load_logs.append(
                        f"{fname}: Loaded. Cu: {c1_min:.2e} to {c1_max:.2e}, Ni: {c2_min:.2e} to {c2_max:.2e}, "
                        f"Ly={param_tuple[0]:.1f}, C_Cu={param_tuple[1]:.1e}, C_Ni={param_tuple[2]:.1e}"
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


# =============================================
# ORIGINAL ATTENTION INTERPOLATOR (UNCHANGED)
# =============================================
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
        if not (lys.shape == c_cus.shape == c_nis.shape):
            raise ValueError(f"Parameter array shapes mismatch: lys={lys.shape}, c_cus={c_cus.shape}, c_nis={c_nis.shape}")

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


# =============================================
# ORIGINAL INTERPOLATION WRAPPER (UNCHANGED)
# =============================================
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


# =============================================
# NEW: EXTRACTABLE CONCENTRATION FEATURES
# =============================================
def compute_concentration_features(solution, time_index: int, y_positions: Optional[List[float]] = None):
    """
    Extract quantitative concentration features from a solution at a given time index.
    
    Features:
    - Centerline (x = Lx/2) concentration profiles vs y for Cu and Ni.
    - Concentration at specified y-positions (default: 0, Ly/4, Ly/2, 3Ly/4, Ly).
    - Integrated concentration (total moles per unit x-length) for Cu and Ni.
    - Maximum concentration gradient (dC/dy) and its y-location for Cu and Ni.
    
    Returns a dictionary with pandas DataFrames and scalars.
    """
    if time_index < 0 or time_index >= len(solution['times']):
        raise ValueError(f"time_index {time_index} out of range (0-{len(solution['times'])-1})")
    
    X = solution['X']
    Y = solution['Y']
    Lx = solution['params']['Lx']
    Ly = solution['params']['Ly']
    c1 = solution['c1_preds'][time_index]   # shape (nx, ny)
    c2 = solution['c2_preds'][time_index]
    
    # Centerline index (x = Lx/2)
    center_idx = np.argmin(np.abs(X[:, 0] - Lx/2))
    y_coords = Y[0, :]
    
    # 1. Centerline profiles
    centerline_cu = c1[center_idx, :]
    centerline_ni = c2[center_idx, :]
    
    # 2. Concentrations at specific y positions
    if y_positions is None:
        y_positions = [0.0, Ly/4, Ly/2, 3*Ly/4, Ly]
    y_indices = [np.argmin(np.abs(y_coords - y)) for y in y_positions]
    conc_at_y = {
        'y (μm)': y_positions,
        'Cu (mol/cc)': [c1[center_idx, idx] for idx in y_indices],
        'Ni (mol/cc)': [c2[center_idx, idx] for idx in y_indices]
    }
    df_conc_at_y = pd.DataFrame(conc_at_y)
    
    # 3. Integrated concentration (trapezoidal rule over y, assuming unit x-length)
    dy = y_coords[1] - y_coords[0]
    integrated_cu = np.trapz(centerline_cu, dx=dy)   # mol/cc * μm -> mol/μm per unit x
    integrated_ni = np.trapz(centerline_ni, dx=dy)
    
    # 4. Concentration gradients (dC/dy) along centerline
    grad_cu = np.gradient(centerline_cu, dy)
    grad_ni = np.gradient(centerline_ni, dy)
    max_grad_cu_idx = np.argmax(np.abs(grad_cu))
    max_grad_ni_idx = np.argmax(np.abs(grad_ni))
    max_grad_cu = grad_cu[max_grad_cu_idx]
    max_grad_ni = grad_ni[max_grad_ni_idx]
    y_max_grad_cu = y_coords[max_grad_cu_idx]
    y_max_grad_ni = y_coords[max_grad_ni_idx]
    
    features = {
        'time': solution['times'][time_index],
        'Ly': Ly,
        'centerline_y': y_coords,
        'centerline_cu': centerline_cu,
        'centerline_ni': centerline_ni,
        'conc_at_y_table': df_conc_at_y,
        'integrated_cu': integrated_cu,
        'integrated_ni': integrated_ni,
        'max_grad_cu': max_grad_cu,
        'max_grad_ni': max_grad_ni,
        'y_max_grad_cu': y_max_grad_cu,
        'y_max_grad_ni': y_max_grad_ni,
        'gradient_cu': grad_cu,
        'gradient_ni': grad_ni,
    }
    return features


def format_features_for_download(features: dict) -> pd.DataFrame:
    """Create a DataFrame from the scalar features for easy CSV export."""
    data = {
        'Feature': [
            'Time (s)', 'Ly (μm)', 'Integrated Cu (mol/μm per x-unit)', 'Integrated Ni (mol/μm per x-unit)',
            'Max |dCu/dy| (mol/cc/μm)', 'y of max |dCu/dy| (μm)', 'Max |dNi/dy| (mol/cc/μm)', 'y of max |dNi/dy| (μm)'
        ],
        'Value': [
            features['time'], features['Ly'], features['integrated_cu'], features['integrated_ni'],
            features['max_grad_cu'], features['y_max_grad_cu'], features['max_grad_ni'], features['y_max_grad_ni']
        ]
    }
    return pd.DataFrame(data)


# =============================================
# ORIGINAL PLOTTING FUNCTIONS (UNCHANGED)
# =============================================
def plot_2d_concentration(solution, time_index, output_dir="figures", cmap_cu='viridis', cmap_ni='magma', vmin_cu=None, vmax_cu=None, vmin_ni=None, vmax_ni=None):
    x_coords = solution['X'][:, 0]
    y_coords = solution['Y'][0, :]
    t_val = solution['times'][time_index]
    Lx = solution['params']['Lx']
    Ly = solution['params']['Ly']
    c1 = solution['c1_preds'][time_index]
    c2 = solution['c2_preds'][time_index]

    cu_min = vmin_cu if vmin_cu is not None else 0
    cu_max = vmax_cu if vmax_cu is not None else np.max(c1)
    ni_min = vmin_ni if vmin_ni is not None else 0
    ni_max = vmax_ni if vmax_ni is not None else np.max(c2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    im1 = ax1.imshow(
        c1,
        origin='lower',
        extent=[0, Lx, 0, Ly],
        cmap=cmap_cu,
        vmin=cu_min,
        vmax=cu_max
    )
    ax1.set_xlabel('x (μm)')
    ax1.set_ylabel('y (μm)')
    ax1.set_title(f'Cu Concentration, t = {t_val:.1f} s')
    ax1.grid(True)
    cb1 = fig.colorbar(im1, ax=ax1, label='Cu Conc. (mol/cc)', format='%.1e')
    cb1.ax.tick_params(labelsize=10)

    im2 = ax2.imshow(
        c2,
        origin='lower',
        extent=[0, Lx, 0, Ly],
        cmap=cmap_ni,
        vmin=ni_min,
        vmax=ni_max
    )
    ax2.set_xlabel('x (μm)')
    ax2.set_ylabel('y (μm)')
    ax2.set_title(f'Ni Concentration, t = {t_val:.1f} s')
    ax2.grid(True)
    cb2 = fig.colorbar(im2, ax=ax2, label='Ni Conc. (mol/cc)', format='%.1e')
    cb2.ax.tick_params(labelsize=10)

    param_text = f"$L_y$ = {Ly:.1f} μm, $C_{{Cu}}$ = {solution['params']['C_Cu']:.1e}, $C_{{Ni}}$ = {solution['params']['C_Ni']:.1e}"
    if solution.get('interpolated', False):
        param_text += " (Interpolated)"
    fig.suptitle(f'Concentration Profiles\n{param_text}', fontsize=14)

    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"conc_2d_t_{t_val:.1f}_ly_{Ly:.1f}_ccu_{solution['params']['C_Cu']:.1e}_cni_{solution['params']['C_Ni']:.1e}"
    plt.savefig(os.path.join(output_dir, f"{base_filename}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f"{base_filename}.pdf"), bbox_inches='tight')
    plt.close()
    return fig, base_filename


def plot_centerline_curves(
        solution, time_indices, sidebar_metric='mean_cu', output_dir="figures",
        label_size=12, title_size=14, tick_label_size=10, legend_loc='upper right',
        curve_colormap='viridis', axis_linewidth=1.5, tick_major_width=1.5,
        tick_major_length=4.0, fig_width=8.0, fig_height=6.0, curve_linewidth=1.0,
        grid_alpha=0.3, grid_linestyle='--', legend_frameon=True, legend_framealpha=0.8
):
    x_coords = solution['X'][:, 0]
    y_coords = solution['Y'][0, :]
    Lx = solution['params']['Lx']
    Ly = solution['params']['Ly']
    center_idx = 25
    times = solution['times']

    if sidebar_metric == 'loss' and 'loss' in solution:
        sidebar_data = solution['loss'][:len(times)]
        sidebar_label = 'Loss'
    elif sidebar_metric == 'mean_cu':
        sidebar_data = [np.mean(c1) for c1 in solution['c1_preds']]
        sidebar_label = 'Mean Cu Conc. (mol/cc)'
    else:
        sidebar_data = [np.mean(c2) for c2 in solution['c2_preds']]
        sidebar_label = 'Mean Ni Conc. (mol/cc)'

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.05, 0.5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[3])

    colors = cm.get_cmap(curve_colormap)(np.linspace(0, 1, len(time_indices)))
    for idx, t_idx in enumerate(time_indices):
        t_val = times[t_idx]
        c1 = solution['c1_preds'][t_idx][:, center_idx]
        c2 = solution['c2_preds'][t_idx][:, center_idx]
        ax1.plot(y_coords, c1, label=f't = {t_val:.1f} s', color=colors[idx], linewidth=curve_linewidth)
        ax2.plot(y_coords, c2, label=f't = {t_val:.1f} s', color=colors[idx], linewidth=curve_linewidth)

    for ax in [ax1, ax2, ax3]:
        for spine in ax.spines.values():
            spine.set_linewidth(axis_linewidth)
        ax.tick_params(
            axis='both',
            which='major',
            width=tick_major_width,
            length=tick_major_length,
            labelsize=tick_label_size
        )
        ax.grid(True, linestyle=grid_linestyle, alpha=grid_alpha)

    legend_positions = {
        'upper right': {'loc': 'upper right', 'bbox': None},
        'upper left': {'loc': 'upper left', 'bbox': None},
        'lower right': {'loc': 'lower right', 'bbox': None},
        'lower left': {'loc': 'lower left', 'bbox': None},
        'center': {'loc': 'center', 'bbox': None},
        'best': {'loc': 'best', 'bbox': None},
        'right': {'loc': 'center left', 'bbox': (1.05, 0.5)},
        'left': {'loc': 'center right', 'bbox': (-0.05, 0.5)},
        'above': {'loc': 'lower center', 'bbox': (0.5, 1.05)},
        'below': {'loc': 'upper center', 'bbox': (0.5, -0.05)}
    }
    legend_params = legend_positions.get(legend_loc, {'loc': 'upper right', 'bbox': None})

    ax1.set_xlabel('y (μm)', fontsize=label_size)
    ax1.set_ylabel('Cu Conc. (mol/cc)', fontsize=label_size)
    ax1.set_title(f'Cu at x = {Lx/2:.1f} μm', fontsize=title_size)
    ax1.legend(fontsize=8, loc=legend_params['loc'], bbox_to_anchor=legend_params['bbox'],
               frameon=legend_frameon, framealpha=legend_framealpha)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    ax2.set_xlabel('y (μm)', fontsize=label_size)
    ax2.set_ylabel('Ni Conc. (mol/cc)', fontsize=label_size)
    ax2.set_title(f'Ni at x = {Lx/2:.1f} μm', fontsize=title_size)
    ax2.legend(fontsize=8, loc=legend_params['loc'], bbox_to_anchor=legend_params['bbox'],
               frameon=legend_frameon, framealpha=legend_framealpha)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    ax3.plot(sidebar_data, times, 'k-', linewidth=curve_linewidth)
    ax3.set_xlabel(sidebar_label, fontsize=label_size)
    ax3.set_ylabel('Time (s)', fontsize=label_size)
    ax3.set_title('Metric vs. Time', fontsize=title_size)
    ax3.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    param_text = f"$L_y$ = {Ly:.1f} μm, $C_{{Cu}}$ = {solution['params']['C_Cu']:.1e}, $C_{{Ni}}$ = {solution['params']['C_Ni']:.1e}"
    if solution.get('interpolated', False):
        param_text += " (Interpolated)"
    fig.suptitle(f'Centerline Concentration Profiles\n{param_text}', fontsize=title_size)

    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"conc_centerline_ly_{Ly:.1f}_ccu_{solution['params']['C_Cu']:.1e}_cni_{solution['params']['C_Ni']:.1e}"
    plt.savefig(os.path.join(output_dir, f"{base_filename}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f"{base_filename}.pdf"), bbox_inches='tight')
    plt.close()
    return fig, base_filename


def plot_parameter_sweep(
        solutions, params_list, selected_params, time_index, sidebar_metric='mean_cu', output_dir="figures",
        label_size=12, title_size=14, tick_label_size=10, legend_loc='upper right',
        curve_colormap='tab10', axis_linewidth=1.5, tick_major_width=1.5,
        tick_major_length=4.0, fig_width=8.0, fig_height=6.0, curve_linewidth=1.0,
        grid_alpha=0.3, grid_linestyle='--', legend_frameon=True, legend_framealpha=0.8
):
    Lx = solutions[0]['params']['Lx']
    center_idx = 25
    t_val = solutions[0]['times'][time_index]

    sidebar_data = []
    sidebar_labels = []
    for sol, params in zip(solutions, params_list):
        if params in selected_params:
            if sidebar_metric == 'loss' and 'loss' in sol:
                sidebar_data.append(sol['loss'][time_index])
            elif sidebar_metric == 'mean_cu':
                sidebar_data.append(np.mean(sol['c1_preds'][time_index]))
            else:
                sidebar_data.append(np.mean(sol['c2_preds'][time_index]))
            ly, c_cu, c_ni = params
            label = f'$L_y$={ly:.1f}, $C_{{Cu}}$={c_cu:.1e}, $C_{{Ni}}$={c_ni:.1e}'
            if sol.get('interpolated', False):
                label += " (Interpolated)"
            sidebar_labels.append(label)

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.05, 0.5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[3])

    colors = cm.get_cmap(curve_colormap)(np.linspace(0, 1, len(selected_params)))
    for idx, (sol, params) in enumerate(zip(solutions, params_list)):
        ly, c_cu, c_ni = params
        if params in selected_params:
            y_coords = sol['Y'][0, :]
            c1 = sol['c1_preds'][time_index][:, center_idx]
            c2 = sol['c2_preds'][time_index][:, center_idx]
            label = f'$L_y$={ly:.1f}, $C_{{Cu}}$={c_cu:.1e}, $C_{{Ni}}$={c_ni:.1e}'
            if sol.get('interpolated', False):
                label += " (Interpolated)"
            ax1.plot(y_coords, c1, label=label, color=colors[idx], linewidth=curve_linewidth)
            ax2.plot(y_coords, c2, label=label, color=colors[idx], linewidth=curve_linewidth)

    for ax in [ax1, ax2, ax3]:
        for spine in ax.spines.values():
            spine.set_linewidth(axis_linewidth)
        ax.tick_params(
            axis='both',
            which='major',
            width=tick_major_width,
            length=tick_major_length,
            labelsize=tick_label_size
        )
        ax.grid(True, linestyle=grid_linestyle, alpha=grid_alpha)

    legend_positions = {
        'upper right': {'loc': 'upper right', 'bbox': None},
        'upper left': {'loc': 'upper left', 'bbox': None},
        'lower right': {'loc': 'lower right', 'bbox': None},
        'lower left': {'loc': 'lower left', 'bbox': None},
        'center': {'loc': 'center', 'bbox': None},
        'best': {'loc': 'best', 'bbox': None},
        'right': {'loc': 'center left', 'bbox': (1.05, 0.5)},
        'left': {'loc': 'center right', 'bbox': (-0.05, 0.5)},
        'above': {'loc': 'lower center', 'bbox': (0.5, 1.05)},
        'below': {'loc': 'upper center', 'bbox': (0.5, -0.05)}
    }
    legend_params = legend_positions.get(legend_loc, {'loc': 'upper right', 'bbox': None})

    ax1.set_xlabel('y (μm)', fontsize=label_size)
    ax1.set_ylabel('Cu Conc. (mol/cc)', fontsize=label_size)
    ax1.set_title(f'Cu at x = {Lx/2:.1f} μm, t = {t_val:.1f} s', fontsize=title_size)
    ax1.legend(fontsize=8, loc=legend_params['loc'], bbox_to_anchor=legend_params['bbox'],
               frameon=legend_frameon, framealpha=legend_framealpha)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    ax2.set_xlabel('y (μm)', fontsize=label_size)
    ax2.set_ylabel('Ni Conc. (mol/cc)', fontsize=label_size)
    ax2.set_title(f'Ni at x = {Lx/2:.1f} μm, t = {t_val:.1f} s', fontsize=title_size)
    ax2.legend(fontsize=8, loc=legend_params['loc'], bbox_to_anchor=legend_params['bbox'],
               frameon=legend_frameon, framealpha=legend_framealpha)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    ax3.barh(range(len(sidebar_data)), sidebar_data, color='gray', edgecolor='black')
    ax3.set_yticks(range(len(sidebar_data)))
    ax3.set_yticklabels(sidebar_labels, fontsize=tick_label_size)
    ax3.set_xlabel(
        'Mean Cu Conc. (mol/cc)' if sidebar_metric == 'mean_cu' else 'Mean Ni Conc. (mol/cc)' if sidebar_metric == 'mean_ni' else 'Loss',
        fontsize=label_size
    )
    ax3.set_title('Metric per Parameter', fontsize=title_size)
    ax3.grid(True, axis='x', linestyle=grid_linestyle, alpha=grid_alpha)
    ax3.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    fig.suptitle('Concentration Profiles for Parameter Sweep', fontsize=title_size)

    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"conc_sweep_t_{t_val:.1f}"
    plt.savefig(os.path.join(output_dir, f"{base_filename}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f"{base_filename}.pdf"), bbox_inches='tight')
    plt.close()
    return fig, base_filename


# =============================================
# SESSION STATE INITIALIZATION
# =============================================
def initialize_session_state():
    defaults = {
        'nl_parser': DiffusionNLParser(),
        'llm_backend_loaded': 'GPT-2 (default)',
        'llm_cache': OrderedDict(),
        'parsed_params': None,
        'nl_query': "",
        'use_llm': True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================
# MAIN APP WITH LLM INTEGRATION AND FEATURE EXTRACTION
# =============================================
def main():
    st.set_page_config(
        page_title="LLM-Enhanced Cu-Ni Diffusion Visualizer",
        layout="wide",
        page_icon="🔬"
    )
    
    st.title("🔬 Attention-Based Cu-Ni Interdiffusion with Natural Language Interface")
    
    initialize_session_state()
    
    # =========================================
    # SIDEBAR: LLM CONFIGURATION
    # =========================================
    with st.sidebar:
        st.header("🤖 LLM Configuration")
        if TRANSFORMERS_AVAILABLE:
            backend_choice = st.selectbox(
                "LLM Backend",
                ["GPT-2 (default)", "Qwen2-0.5B-Instruct", "Qwen2.5-0.5B-Instruct"],
                index=0
            )
            if backend_choice != st.session_state.llm_backend_loaded:
                st.session_state.llm_backend_loaded = backend_choice
                st.session_state.llm_cache.clear()
                st.rerun()
            
            tokenizer, model, active_backend = load_llm(backend_choice)
            st.session_state.llm_tokenizer = tokenizer
            st.session_state.llm_model = model
            st.session_state.use_llm = st.checkbox("Enable LLM Parsing", value=True)
            st.caption(f"Active: **{active_backend}**")
        else:
            st.error("Install `transformers` to enable LLM features.")
            st.session_state.use_llm = False
            st.session_state.llm_backend_loaded = "Regex Fallback Only"
        
        st.header("⚙️ Interpolation Hyperparameters")
        sigma = st.slider("Locality σ", 0.05, 0.50, 0.20, 0.01)
        num_heads = st.slider("Attention Heads", 1, 8, 4)
        d_head = st.slider("Dim/Head", 4, 16, 8)
        seed = st.number_input("Random Seed", 0, 9999, 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # =========================================
    # MAIN: NATURAL LANGUAGE INPUT
    # =========================================
    st.subheader("📝 Describe Your Solder Joint Configuration")
    
    templates = {
        "Thin Asymmetric Joint": "Analyze a 40 μm asymmetric Cu-Ni joint with top Cu concentration 1.8e-3 and bottom Ni 0.4e-3.",
        "Thick Symmetric Cu": "Simulate a 100 μm symmetric Cu/Sn2.5Ag/Cu joint. Use c_Cu=2.0e-3.",
        "Ni-Rich Diffusion": "Domain length 60 μm, C_Cu=1.0e-3, C_Ni=1.5e-3. Asymmetric configuration.",
        "Self-Diffusion Baseline": "Ly=75 μm, C_Cu=0, C_Ni=0 for self-diffusion reference.",
    }
    
    col1, col2 = st.columns([3, 1])
    with col1:
        nl_query = st.text_area(
            "Enter natural language query:",
            height=100,
            placeholder="e.g., Analyze a 50 μm joint with Cu=1.5e-3, Ni=0.6e-3, asymmetric configuration...",
            key="nl_query"
        )
    with col2:
        st.markdown("**Quick Templates:**")
        for name, text in templates.items():
            if st.button(name, use_container_width=True):
                st.session_state.nl_query = text
                st.rerun()
    
    parser = st.session_state.nl_parser
    use_llm = st.session_state.get('use_llm', False)
    tokenizer = st.session_state.get('llm_tokenizer', None)
    model = st.session_state.get('llm_model', None)
    
    if nl_query:
        parsed = parser.hybrid_parse(nl_query, tokenizer, model, use_llm=use_llm)
        st.session_state.parsed_params = parsed
        st.markdown(parser.get_explanation(parsed, nl_query))
    else:
        parsed = parser.defaults.copy()
        st.session_state.parsed_params = parsed
    
    # =========================================
    # PARAMETER SELECTION (Pre-filled by Parser)
    # =========================================
    st.subheader("🎯 Target Parameters (Override if Needed)")
    
    solutions, params_list, lys, c_cus, c_nis, load_logs = load_solutions(SOLUTION_DIR)
    
    if load_logs:
        with st.expander("📋 Load Log"):
            for log in load_logs:
                st.write(log)
    
    if not solutions:
        st.error("No valid solution files found in pinn_solutions directory.")
        return
    
    lys = sorted(set(lys))
    c_cus = sorted(set(c_cus))
    c_nis = sorted(set(c_nis))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        ly_choice = st.selectbox(
            "Domain Height (Ly, μm)",
            options=lys,
            format_func=lambda x: f"{x:.1f}",
            index=min(lys.index(parsed['ly_target']), len(lys)-1) if parsed['ly_target'] in lys else 0
        )
    with col2:
        c_cu_choice = st.selectbox(
            "Cu Boundary Concentration (mol/cc)",
            options=c_cus,
            format_func=lambda x: f"{x:.1e}",
            index=min(c_cus.index(parsed['c_cu_target']), len(c_cus)-1) if parsed['c_cu_target'] in c_cus else 0
        )
    with col3:
        c_ni_choice = st.selectbox(
            "Ni Boundary Concentration (mol/cc)",
            options=c_nis,
            format_func=lambda x: f"{x:.1e}",
            index=min(c_nis.index(parsed['c_ni_target']), len(c_nis)-1) if parsed['c_ni_target'] in c_nis else 0
        )
    
    use_custom_params = st.checkbox("Use Custom Parameters for Interpolation", value=False)
    if use_custom_params:
        ly_target = st.number_input(
            "Custom Ly (μm)",
            min_value=30.0,
            max_value=120.0,
            value=parsed['ly_target'],
            step=0.1,
            format="%.1f"
        )
        c_cu_target = st.number_input(
            "Custom C_Cu (mol/cc)",
            min_value=0.0,
            max_value=2.9e-3,
            value=parsed['c_cu_target'],
            step=0.1e-3,
            format="%.1e"
        )
        c_ni_target = st.number_input(
            "Custom C_Ni (mol/cc)",
            min_value=0.0,
            max_value=1.8e-3,
            value=parsed['c_ni_target'],
            step=0.1e-4,
            format="%.1e"
        )
    else:
        ly_target, c_cu_target, c_ni_target = ly_choice, c_cu_choice, c_ni_choice
    
    # =========================================
    # VISUALIZATION SETTINGS
    # =========================================
    st.subheader("🎨 Visualization Settings")
    cmap_cu = st.selectbox("Cu Heatmap Colormap", options=COLORMAPS, index=COLORMAPS.index('viridis'))
    cmap_ni = st.selectbox("Ni Heatmap Colormap", options=COLORMAPS, index=COLORMAPS.index('magma'))
    sidebar_metric = st.selectbox("Sidebar Metric for Curves", options=['mean_cu', 'mean_ni', 'loss'], index=0)
    
    st.subheader("🎯 Color Scale Limits")
    use_custom_scale = st.checkbox("Use custom color scale limits", value=False)
    custom_cu_min, custom_cu_max, custom_ni_min, custom_ni_max = None, None, None, None
    if use_custom_scale:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Cu Concentration Limits**")
            custom_cu_min = st.number_input("Cu Min", value=0.0, format="%.2e", key="cu_min")
            custom_cu_max = st.number_input("Cu Max", value=float(np.max([np.max(sol['c1_preds']) for sol in solutions])), format="%.2e", key="cu_max")
        with col2:
            st.write("**Ni Concentration Limits**")
            custom_ni_min = st.number_input("Ni Min", value=0.0, format="%.2e", key="ni_min")
            custom_ni_max = st.number_input("Ni Max", value=float(np.max([np.max(sol['c2_preds']) for sol in solutions])), format="%.2e", key="ni_max")
    
    if custom_cu_min is not None and custom_cu_max is not None and custom_cu_min >= custom_cu_max:
        st.error("Cu minimum concentration must be less than maximum concentration.")
        return
    if custom_ni_min is not None and custom_ni_max is not None and custom_ni_min >= custom_ni_max:
        st.error("Ni minimum concentration must be less than maximum concentration.")
        return
    
    with st.expander("🔧 Figure Customization"):
        label_size = st.slider("Axis Label Size", min_value=8, max_value=20, value=12, step=1)
        title_size = st.slider("Title Size", min_value=10, max_value=24, value=14, step=1)
        tick_label_size = st.slider("Tick Label Size", min_value=6, max_value=16, value=10, step=1)
        legend_loc = st.selectbox(
            "Legend Location",
            options=['upper right', 'upper left', 'lower right', 'lower left', 'center', 'best',
                     'right', 'left', 'above', 'below'],
            index=0
        )
        curve_colormap = st.selectbox(
            "Curve Colormap",
            options=['viridis', 'plasma', 'inferno', 'magma', 'tab10', 'Set1', 'Set2'],
            index=4
        )
        axis_linewidth = st.slider("Axis Line Width", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
        tick_major_width = st.slider("Tick Major Width", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
        tick_major_length = st.slider("Tick Major Length", min_value=2.0, max_value=10.0, value=4.0, step=0.5)
        fig_width = st.slider("Figure Width (inches)", min_value=4.0, max_value=12.0, value=8.0, step=0.5)
        fig_height = st.slider("Figure Height (inches)", min_value=3.0, max_value=8.0, value=6.0, step=0.5)
        curve_linewidth = st.slider("Curve Line Width", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
        grid_alpha = st.slider("Grid Opacity", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
        grid_linestyle = st.selectbox("Grid Line Style", options=['--', '-', ':', '-.'], index=0)
        legend_frameon = st.checkbox("Show Legend Frame", value=True)
        legend_framealpha = st.slider("Legend Frame Opacity", min_value=0.0, max_value=1.0, value=0.8, step=0.1)
    
    # =========================================
    # LOAD OR INTERPOLATE SOLUTION
    # =========================================
    try:
        solution = load_and_interpolate_solution(solutions, params_list, ly_target, c_cu_target, c_ni_target)
    except Exception as e:
        st.error(f"Failed to load or interpolate solution: {str(e)}")
        return
    
    # Display solution details
    st.subheader("📊 Solution Details")
    st.write(f"$L_y$ = {solution['params']['Ly']:.1f} μm")
    st.write(f"$C_{{Cu}}$ = {solution['params']['C_Cu']:.1e} mol/cc")
    st.write(f"$C_{{Ni}}$ = {solution['params']['C_Ni']:.1e} mol/cc")
    if solution.get('interpolated', False):
        st.write("**Status**: ⚡ Interpolated via attention mechanism")
        weights = solution.get('attention_weights', [])
        if weights:
            st.write("**Source Weights:**")
            weight_df = pd.DataFrame({
                'Source': [f"S{i+1}" for i in range(len(weights))],
                'Weight': [f"{w:.4f}" for w in weights]
            })
            st.dataframe(weight_df.style.bar(subset=['Weight'], color='#5fba7d'), use_container_width=True)
    else:
        st.write("**Status**: ✅ Exact precomputed solution")
    
    # =========================================
    # 2D CONCENTRATION HEATMAPS
    # =========================================
    st.subheader("🗺️ 2D Concentration Heatmaps")
    time_index = st.slider("Select Time Index for Heatmaps", 0, len(solution['times'])-1, len(solution['times'])-1)
    fig_2d, filename_2d = plot_2d_concentration(
        solution, time_index, cmap_cu=cmap_cu, cmap_ni=cmap_ni,
        vmin_cu=custom_cu_min if use_custom_scale else None,
        vmax_cu=custom_cu_max if use_custom_scale else None,
        vmin_ni=custom_ni_min if use_custom_scale else None,
        vmax_ni=custom_ni_max if use_custom_scale else None
    )
    st.pyplot(fig_2d)
    st.download_button(
        label="⬇️ Download 2D Plot as PNG",
        data=open(os.path.join("figures", f"{filename_2d}.png"), "rb").read(),
        file_name=f"{filename_2d}.png",
        mime="image/png"
    )
    st.download_button(
        label="⬇️ Download 2D Plot as PDF",
        data=open(os.path.join("figures", f"{filename_2d}.pdf"), "rb").read(),
        file_name=f"{filename_2d}.pdf",
        mime="application/pdf"
    )
    
    # =========================================
    # EXTRACTABLE CONCENTRATION FEATURES
    # =========================================
    st.subheader("📊 Extractable Concentration Features")
    # Use the same time index as heatmaps (or allow separate selection)
    feature_time_idx = st.number_input(
        "Time index for feature extraction",
        min_value=0, max_value=len(solution['times'])-1,
        value=time_index, step=1, key="feature_time_idx"
    )
    try:
        features = compute_concentration_features(solution, feature_time_idx)
        # Display key features
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Integrated Cu (mol/μm per x-unit)", f"{features['integrated_cu']:.3e}")
            st.metric("Max |dCu/dy| (mol/cc/μm)", f"{features['max_grad_cu']:.3e} at y={features['y_max_grad_cu']:.1f} μm")
        with col2:
            st.metric("Integrated Ni (mol/μm per x-unit)", f"{features['integrated_ni']:.3e}")
            st.metric("Max |dNi/dy| (mol/cc/μm)", f"{features['max_grad_ni']:.3e} at y={features['y_max_grad_ni']:.1f} μm")
        
        # Show concentration at key y positions
        st.write("**Concentrations at Selected y‑positions (centerline):**")
        st.dataframe(features['conc_at_y_table'], use_container_width=True)
        
        # Download buttons for feature data
        df_scalar = format_features_for_download(features)
        csv_scalar = df_scalar.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Scalar Features as CSV",
            data=csv_scalar,
            file_name=f"concentration_features_scalar_t{features['time']:.1f}.csv",
            mime="text/csv"
        )
        
        # Also allow download of full centerline profiles
        df_centerline = pd.DataFrame({
            'y (μm)': features['centerline_y'],
            'Cu (mol/cc)': features['centerline_cu'],
            'Ni (mol/cc)': features['centerline_ni']
        })
        csv_centerline = df_centerline.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Centerline Profiles as CSV",
            data=csv_centerline,
            file_name=f"centerline_profiles_t{features['time']:.1f}.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.warning(f"Feature extraction failed: {e}")
    
    # =========================================
    # CENTERLINE CONCENTRATION CURVES
    # =========================================
    st.subheader("📈 Centerline Concentration Curves")
    time_indices = st.multiselect(
        "Select Time Indices for Curves",
        options=list(range(len(solution['times']))),
        default=[0, len(solution['times'])//4, len(solution['times'])//2, 3*len(solution['times'])//4, len(solution['times'])-1],
        format_func=lambda x: f"t = {solution['times'][x]:.1f} s"
    )
    if time_indices:
        fig_curves, filename_curves = plot_centerline_curves(
            solution, time_indices, sidebar_metric=sidebar_metric,
            label_size=label_size, title_size=title_size, tick_label_size=tick_label_size,
            legend_loc=legend_loc, curve_colormap=curve_colormap,
            axis_linewidth=axis_linewidth, tick_major_width=tick_major_width,
            tick_major_length=tick_major_length, fig_width=fig_width, fig_height=fig_height,
            curve_linewidth=curve_linewidth, grid_alpha=grid_alpha, grid_linestyle=grid_linestyle,
            legend_frameon=legend_frameon, legend_framealpha=legend_framealpha
        )
        st.pyplot(fig_curves)
        st.download_button(
            label="⬇️ Download Centerline Plot as PNG",
            data=open(os.path.join("figures", f"{filename_curves}.png"), "rb").read(),
            file_name=f"{filename_curves}.png",
            mime="image/png"
        )
        st.download_button(
            label="⬇️ Download Centerline Plot as PDF",
            data=open(os.path.join("figures", f"{filename_curves}.pdf"), "rb").read(),
            file_name=f"{filename_curves}.pdf",
            mime="application/pdf"
        )
    
    # =========================================
    # PARAMETER SWEEP CURVES
    # =========================================
    st.subheader("🔄 Parameter Sweep Curves")
    with st.expander("➕ Add Custom Parameter Combinations for Sweep"):
        num_custom_params = st.number_input("Number of Custom Parameter Sets", min_value=0, max_value=5, value=0, step=1)
        custom_params = []
        for i in range(num_custom_params):
            st.write(f"Custom Parameter Set {i+1}")
            ly_custom = st.number_input(
                f"Custom Ly (μm) {i+1}",
                min_value=30.0,
                max_value=120.0,
                value=ly_choice,
                step=0.1,
                format="%.1f",
                key=f"ly_custom_{i}"
            )
            c_cu_custom = st.number_input(
                f"Custom C_Cu (mol/cc) {i+1}",
                min_value=0.0,
                max_value=2.9e-3,
                value=max(c_cu_choice, 1.5e-3),
                step=0.1e-3,
                format="%.1e",
                key=f"c_cu_custom_{i}"
            )
            c_ni_custom = st.number_input(
                f"Custom C_Ni (mol/cc) {i+1}",
                min_value=0.0,
                max_value=1.8e-3,
                value=max(c_ni_choice, 1.0e-4),
                step=0.1e-4,
                format="%.1e",
                key=f"c_ni_custom_{i}"
            )
            custom_params.append((ly_custom, c_cu_custom, c_ni_custom))
    
    param_options = [(ly, c_cu, c_ni) for ly, c_cu, c_ni in params_list]
    param_labels = [f"$L_y$={ly:.1f}, $C_{{Cu}}$={c_cu:.1e}, $C_{{Ni}}$={c_ni:.1e}" for ly, c_cu, c_ni in param_options]
    default_params = param_options[:min(4, len(param_options))]
    selected_labels = st.multiselect(
        "Select Exact Parameter Combinations",
        options=param_labels,
        default=[param_labels[param_options.index(p)] for p in default_params],
        format_func=lambda x: x
    )
    selected_params = [param_options[param_labels.index(label)] for label in selected_labels]
    selected_params.extend(custom_params)
    
    sweep_solutions = []
    sweep_params_list = []
    for params in selected_params:
        ly, c_cu, c_ni = params
        try:
            sol = load_and_interpolate_solution(solutions, params_list, ly, c_cu, c_ni)
            sweep_solutions.append(sol)
            sweep_params_list.append(params)
        except Exception as e:
            st.warning(f"Failed to load or interpolate solution for Ly={ly:.1f}, C_Cu={c_cu:.1e}, C_Ni={c_ni:.1e}: {str(e)}")
    
    sweep_time_index = st.slider("Select Time Index for Sweep", 0, len(solution['times'])-1, len(solution['times'])-1)
    if sweep_solutions and sweep_params_list:
        fig_sweep, filename_sweep = plot_parameter_sweep(
            sweep_solutions, sweep_params_list, sweep_params_list, sweep_time_index, sidebar_metric=sidebar_metric,
            label_size=label_size, title_size=title_size, tick_label_size=tick_label_size,
            legend_loc=legend_loc, curve_colormap=curve_colormap,
            axis_linewidth=axis_linewidth, tick_major_width=tick_major_width,
            tick_major_length=tick_major_length, fig_width=fig_width, fig_height=fig_height,
            curve_linewidth=curve_linewidth, grid_alpha=grid_alpha, grid_linestyle=grid_linestyle,
            legend_frameon=legend_frameon, legend_framealpha=legend_framealpha
        )
        st.pyplot(fig_sweep)
        st.download_button(
            label="⬇️ Download Sweep Plot as PNG",
            data=open(os.path.join("figures", f"{filename_sweep}.png"), "rb").read(),
            file_name=f"{filename_sweep}.png",
            mime="image/png"
        )
        st.download_button(
            label="⬇️ Download Sweep Plot as PDF",
            data=open(os.path.join("figures", f"{filename_sweep}.pdf"), "rb").read(),
            file_name=f"{filename_sweep}.pdf",
            mime="application/pdf"
        )


if __name__ == "__main__":
    main()
