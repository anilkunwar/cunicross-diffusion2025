#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UNIFIED CU-NI INTERDIFFUSION VISUALIZER WITH VALIDATION & UNCERTAINTY
=====================================================================
Features:
- Attention-based interpolation with physics constraints (PDE residual, BC enforcement)
- Natural language parameter extraction (regex + LLM hybrid)
- Publication-quality 2D heatmaps, centerline curves, parameter sweeps
- Held-out PINN validation: MSE, MAE, R², SSIM, PDE residual, mass error
- Uncertainty quantification: weight entropy, parameter distance, ensemble variance
- Interactive validation dashboard: bar charts, radar plots, scatter matrices
- Export metrics as JSON/CSV
"""
import os
import pickle
import json
import hashlib
import warnings
from collections import OrderedDict
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skimage.metrics import structural_similarity as ssim
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
import torch.optim as optim
import re

# Suppress warnings
warnings.filterwarnings('ignore')

# =============================================
# GLOBAL CONFIGURATION
# =============================================
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['figure.dpi'] = 300
SOLUTION_DIR = os.path.join(os.path.dirname(__file__), "pinn_solutions")

# Physical constants for Cu-Ni cross-diffusion
PHYSICS_CONSTANTS = {
    'D11': 0.006,       # Cu self-diffusivity (μm²/s)
    'D12': 0.00427,     # Cu-Ni cross-diffusivity
    'D21': 0.003697,    # Ni-Cu cross-diffusivity
    'D22': 0.0054,      # Ni self-diffusivity
    'C_CU_RANGE': (0.0, 2.9e-3),
    'C_NI_RANGE': (0.0, 1.8e-3),
    'LY_RANGE': (30.0, 120.0),
    'T_MAX': 200.0,
    'MASS_TOLERANCE': 1e-4,
}

# Colormaps for plotting
COLORMAPS = [
    "viridis", "plasma", "inferno", "magma", "cividis", "Greys", "Purples",
    "Blues", "Greens", "Oranges", "Reds", "YlOrBr", "YlOrRd", "OrRd", "PuRd",
    "RdPu", "BuPu", "GnBu", "PuBu", "YlGnBu", "PuBuGn", "BuGn", "YlGn",
    "cubehelix", "binary", "gist_yarg", "gist_gray", "gray", "bone", "pink",
    "spring", "summer", "autumn", "winter", "PiYG", "PRGn", "BrBG", "PuOr",
    "RdGy", "RdBu", "RdYlBu", "RdYlGn", "Spectral", "coolwarm", "bwr", "seismic",
    "twilight", "twilight_shifted", "hsv", "Pastel1", "Pastel2", "Paired",
    "Accent", "Dark2", "Set1", "Set2", "Set3", "tab10", "tab20", "tab20b", "tab20c",
    "flag", "prism", "ocean", "gist_earth", "terrain", "gist_stern", "gnuplot",
    "gnuplot2", "CMRmap", "brg", "gist_rainbow", "rainbow", "jet", "nipy_spectral",
    "gist_ncar"
]

# =============================================
# LLM IMPORT WITH GRACEFUL FALLBACK
# =============================================
try:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.warning("⚠️ `transformers` not installed. LLM features disabled.")

# =============================================
# NATURAL LANGUAGE PARSER (HYBRID REGEX + LLM)
# =============================================
class DiffusionNLParser:
    def __init__(self):
        self.defaults = {
            'ly_target': 60.0,
            'c_cu_target': 1.5e-3,
            'c_ni_target': 0.5e-3,
            'sigma': 0.20,
        }
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
                        if key == 'ly_target':
                            params[key] = float(val)
                        else:
                            params[key] = float(val)
                    except (ValueError, TypeError):
                        pass
                    break
        params['ly_target'] = np.clip(params['ly_target'], 30.0, 120.0)
        params['c_cu_target'] = np.clip(params['c_cu_target'], 0.0, 2.9e-3)
        params['c_ni_target'] = np.clip(params['c_ni_target'], 0.0, 1.8e-3)
        return params

    @staticmethod
    def _extract_json_robust(generated: str) -> Optional[Dict]:
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
"""
        defaults_json = json.dumps(self.defaults)
        regex_hint = f"\nRegex hint: {json.dumps(regex_params or {})}" if regex_params else ""
        user = f"""{examples}{regex_hint}
Query: "{text}"
JSON keys: ly_target (float, 30-120), c_cu_target (float, 0-2.9e-3), c_ni_target (float, 0-1.8e-3), sigma (0.05-0.5)
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
                outputs = model.generate(inputs, max_new_tokens=200, temperature=temperature, do_sample=(temperature>0), pad_token_id=tokenizer.eos_token_id)
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
            st.warning(f"LLM parsing failed: {e}")
        return self.parse_regex(text)

    def hybrid_parse(self, text: str, tokenizer, model, use_llm: bool = True) -> Dict:
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
        lines = ["### 🔍 Parsed Parameters", f"**Query:** _{original_text}_", "| Parameter | Value | Status |", "|-----------|-------|--------|"]
        for key, val in params.items():
            if key == 'sigma':
                continue
            status = "✅ Extracted" if val != self.defaults[key] else "⚪ Default"
            val_str = f"{val:.1e}" if isinstance(val, float) and (val<0.01 or val>100) else f"{val:.3f}" if isinstance(val, float) else str(val)
            lines.append(f"| {key} | {val_str} | {status} |")
        return "\n".join(lines)

# =============================================
# UNIFIED LLM LOADER
# =============================================
@st.cache_resource(show_spinner="Loading LLM...")
def load_llm(backend_name: str):
    if not TRANSFORMERS_AVAILABLE:
        return None, None, "Regex Only"
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
        return None, None, "Unknown"
    model.eval()
    return tokenizer, model, backend_name

# =============================================
# LOAD PRECOMPUTED PINN SOLUTIONS
# =============================================
@st.cache_data
def load_solutions(solution_dir):
    solutions = []
    params_list = []
    load_logs = []
    lys, c_cus, c_nis = [], [], []
    for fname in os.listdir(solution_dir):
        if fname.endswith(".pkl"):
            try:
                with open(os.path.join(solution_dir, fname), "rb") as f:
                    sol = pickle.load(f)
                required = ['params', 'X', 'Y', 'c1_preds', 'c2_preds', 'times']
                if all(k in sol for k in required):
                    if np.any(np.isnan(sol['c1_preds'])) or np.all(sol['c1_preds']==0):
                        load_logs.append(f"{fname}: Skipped (invalid data)")
                        continue
                    solutions.append(sol)
                    param_tuple = (sol['params']['Ly'], sol['params']['C_Cu'], sol['params']['C_Ni'])
                    params_list.append(param_tuple)
                    lys.append(sol['params']['Ly'])
                    c_cus.append(sol['params']['C_Cu'])
                    c_nis.append(sol['params']['C_Ni'])
                    load_logs.append(f"{fname}: Loaded. Ly={param_tuple[0]:.1f}, C_Cu={param_tuple[1]:.1e}, C_Ni={param_tuple[2]:.1e}")
                else:
                    load_logs.append(f"{fname}: Skipped - missing keys")
            except Exception as e:
                load_logs.append(f"{fname}: Error - {str(e)}")
    if not solutions:
        load_logs.append("ERROR: No valid solutions found.")
    return solutions, params_list, lys, c_cus, c_nis, load_logs

# =============================================
# ATTENTION INTERPOLATOR (BASE)
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
        if not solutions:
            raise ValueError("No solutions")
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
        target_tensor = torch.tensor([[target_ly_norm, target_c_cu_norm, target_c_ni_norm]], dtype=torch.float32)

        queries = self.W_q(target_tensor).view(1, self.num_heads, self.d_head)
        keys = self.W_k(params_tensor).view(len(params_list), self.num_heads, self.d_head)
        attn_logits = torch.einsum('nhd,mhd->nmh', keys, queries) / np.sqrt(self.d_head)
        attn_weights = torch.softmax(attn_logits, dim=0).mean(dim=2).squeeze(1)

        scaled_dist = torch.sqrt(((torch.tensor(ly_norm)-target_ly_norm)/self.sigma)**2 +
                                 ((torch.tensor(c_cu_norm)-target_c_cu_norm)/self.sigma)**2 +
                                 ((torch.tensor(c_ni_norm)-target_c_ni_norm)/self.sigma)**2)
        spatial_weights = torch.exp(-scaled_dist**2/2)
        spatial_weights /= spatial_weights.sum()
        combined = attn_weights * spatial_weights
        combined /= combined.sum()
        return self._interpolate(solutions, combined.detach().numpy(), ly_target, c_cu_target, c_ni_target)

    def _interpolate(self, solutions, weights, ly_target, c_cu_target, c_ni_target):
        Lx = solutions[0]['params']['Lx']
        t_max = solutions[0]['params']['t_max']
        x_coords = np.linspace(0, Lx, 50)
        y_coords = np.linspace(0, ly_target, 50)
        times = np.linspace(0, t_max, 50)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        c1_interp = np.zeros((len(times), 50, 50))
        c2_interp = np.zeros((len(times), 50, 50))

        for t_idx in range(len(times)):
            for sol, w in zip(solutions, weights):
                scale = ly_target / sol['params']['Ly']
                Y_scaled = sol['Y'][0, :] * scale
                interp1 = RegularGridInterpolator((sol['X'][:,0], Y_scaled), sol['c1_preds'][t_idx], method='linear', bounds_error=False, fill_value=0)
                interp2 = RegularGridInterpolator((sol['X'][:,0], Y_scaled), sol['c2_preds'][t_idx], method='linear', bounds_error=False, fill_value=0)
                points = np.stack([X.flatten(), Y.flatten()], axis=1)
                c1_interp[t_idx] += w * interp1(points).reshape(50,50)
                c2_interp[t_idx] += w * interp2(points).reshape(50,50)

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
# PHYSICS-INFORMED INTERPOLATION ENHANCEMENT
# =============================================
class PhysicsAwareInterpolator:
    def __init__(self, base_interpolator, constants=None):
        self.base = base_interpolator
        self.constants = constants or PHYSICS_CONSTANTS
        self.enforce_bc = True
        self.pde_weight = 0.1
        self.mass_weight = 0.05

    def interpolate_with_physics(self, solutions, params_list, target_params, optimize=True):
        result = self.base(solutions, params_list, target_params['Ly'], target_params['c_cu'], target_params['c_ni'])
        if not result:
            return result

        c1 = np.array(result['c1_preds'][-1])
        c2 = np.array(result['c2_preds'][-1])
        Lx = target_params.get('Lx', 60)
        Ly = target_params.get('Ly', 60)
        x = np.linspace(0, Lx, 50)
        y = np.linspace(0, Ly, 50)

        # Hard boundary enforcement
        if self.enforce_bc:
            c1[:, 0] = target_params.get('c_cu_bottom', 0.0)
            c1[:, -1] = target_params.get('c_cu_top', 1.59e-3)
            c2[:, 0] = target_params.get('c_ni_bottom', 4e-4)
            c2[:, -1] = target_params.get('c_ni_top', 0.0)

        if optimize and torch.cuda.is_available():
            c1, c2 = self._pde_refinement(c1, c2, target_params, x, y)

        result['c1_preds'][-1] = c1
        result['c2_preds'][-1] = c2
        result['physics_enforced'] = True

        # Compute diagnostic metrics
        residual = self._compute_pde_residual(c1, c2, x, y, target_params.get('t_max', 200))
        mass_err = self._mass_conservation_error(c1, c2, x, y, target_params.get('c_cu_initial', 1.5e-3), target_params.get('c_ni_initial', 4e-4))
        result['physics_diagnostics'] = {'pde_residual_mean': float(np.mean(residual)), 'mass_error': float(mass_err)}
        return result

    def _pde_refinement(self, c1, c2, target_params, x, y, n_steps=50, lr=1e-4):
        c1_t = torch.tensor(c1, dtype=torch.float32, requires_grad=True)
        c2_t = torch.tensor(c2, dtype=torch.float32, requires_grad=True)
        X, Y = torch.meshgrid(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), indexing='ij')
        optimizer = optim.Adam([c1_t, c2_t], lr=lr)
        for _ in range(n_steps):
            optimizer.zero_grad()
            lap1 = self._laplacian_torch(c1_t, X, Y)
            lap2 = self._laplacian_torch(c2_t, X, Y)
            residual1 = -(self.constants['D11']*lap1 + self.constants['D12']*lap2)
            residual2 = -(self.constants['D21']*lap1 + self.constants['D22']*lap2)
            pde_loss = torch.mean(residual1**2 + residual2**2)
            bc_loss = (torch.mean((c1_t[:,0] - target_params.get('c_cu_bottom',0))**2) +
                       torch.mean((c1_t[:,-1] - target_params.get('c_cu_top',1.59e-3))**2) +
                       torch.mean((c2_t[:,0] - target_params.get('c_ni_bottom',4e-4))**2) +
                       torch.mean((c2_t[:,-1] - target_params.get('c_ni_top',0))**2))
            loss = self.pde_weight * pde_loss + self.mass_weight * bc_loss
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                c1_t.clamp_(0, self.constants['C_CU_RANGE'][1])
                c2_t.clamp_(0, self.constants['C_NI_RANGE'][1])
        return c1_t.detach().numpy(), c2_t.detach().numpy()

    def _laplacian_torch(self, c, X, Y):
        dc_dx = torch.autograd.grad(c.sum(), X, create_graph=True, retain_graph=True)[0]
        dc_dy = torch.autograd.grad(c.sum(), Y, create_graph=True, retain_graph=True)[0]
        d2c_dx2 = torch.autograd.grad(dc_dx.sum(), X, create_graph=True, retain_graph=True)[0]
        d2c_dy2 = torch.autograd.grad(dc_dy.sum(), Y, create_graph=True, retain_graph=True)[0]
        return d2c_dx2 + d2c_dy2

    def _compute_pde_residual(self, c1, c2, x, y, t):
        dx, dy = x[1]-x[0], y[1]-y[0]
        nx, ny = len(x), len(y)
        def laplacian(c):
            lap = np.zeros_like(c)
            lap[1:-1,1:-1] = ((c[2:,1:-1] - 2*c[1:-1,1:-1] + c[:-2,1:-1])/dy**2 +
                              (c[1:-1,2:] - 2*c[1:-1,1:-1] + c[1:-1,:-2])/dx**2)
            # Simplified boundaries
            return lap
        lap1 = laplacian(c1)
        lap2 = laplacian(c2)
        resid1 = -(self.constants['D11']*lap1 + self.constants['D12']*lap2)
        resid2 = -(self.constants['D21']*lap1 + self.constants['D22']*lap2)
        return np.sqrt(resid1**2 + resid2**2)

    def _mass_conservation_error(self, c1, c2, x, y, init_c1, init_c2):
        dx, dy = x[1]-x[0], y[1]-y[0]
        area = dx*dy
        mass1 = np.sum(c1)*area
        mass2 = np.sum(c2)*area
        init_mass1 = init_c1 * c1.size * area
        init_mass2 = init_c2 * c2.size * area
        return max(abs(mass1 - init_mass1)/(init_mass1+1e-12), abs(mass2 - init_mass2)/(init_mass2+1e-12))

# =============================================
# VALIDATION METRICS AND DATACLASS
# =============================================
@dataclass
class ValidationMetrics:
    mse_c1: float = 0.0
    mse_c2: float = 0.0
    mae_c1: float = 0.0
    mae_c2: float = 0.0
    max_error_c1: float = 0.0
    max_error_c2: float = 0.0
    r2_c1: float = 0.0
    r2_c2: float = 0.0
    ssim_c1: float = 0.0
    ssim_c2: float = 0.0
    pde_residual_mean: float = 0.0
    pde_residual_max: float = 0.0
    bc_error_top_c1: float = 0.0
    bc_error_top_c2: float = 0.0
    bc_error_bottom_c1: float = 0.0
    bc_error_bottom_c2: float = 0.0
    mass_error: float = 0.0
    weight_entropy: float = 0.0
    param_distance: float = 0.0
    ensemble_variance_c1: float = 0.0
    ensemble_variance_c2: float = 0.0
    overall_score: float = 0.0

    def to_dataframe(self) -> pd.DataFrame:
        data = []
        for k,v in self.__dict__.items():
            cat = 'pointwise' if k in ['mse_c1','mse_c2','mae_c1','mae_c2','max_error_c1','max_error_c2','r2_c1','r2_c2','ssim_c1','ssim_c2'] else \
                  'physics' if k in ['pde_residual_mean','pde_residual_max','bc_error_top_c1','bc_error_top_c2','bc_error_bottom_c1','bc_error_bottom_c2','mass_error'] else \
                  'uncertainty' if k in ['weight_entropy','param_distance','ensemble_variance_c1','ensemble_variance_c2'] else 'composite'
            data.append({'Metric':k, 'Value':v, 'Category':cat})
        return pd.DataFrame(data)

def compute_validation_metrics(interp_c1, interp_c2, gt_c1, gt_c2, x, y, t, params, weights=None) -> ValidationMetrics:
    m = ValidationMetrics()
    flat_i1, flat_i2 = interp_c1.flatten(), interp_c2.flatten()
    flat_g1, flat_g2 = gt_c1.flatten(), gt_c2.flatten()
    m.mse_c1 = mean_squared_error(flat_g1, flat_i1)
    m.mse_c2 = mean_squared_error(flat_g2, flat_i2)
    m.mae_c1 = mean_absolute_error(flat_g1, flat_i1)
    m.mae_c2 = mean_absolute_error(flat_g2, flat_i2)
    m.max_error_c1 = np.max(np.abs(flat_g1 - flat_i1))
    m.max_error_c2 = np.max(np.abs(flat_g2 - flat_i2))
    if np.var(flat_g1) > 1e-12:
        m.r2_c1 = r2_score(flat_g1, flat_i1)
    if np.var(flat_g2) > 1e-12:
        m.r2_c2 = r2_score(flat_g2, flat_i2)
    dr_c1 = gt_c1.max() - gt_c1.min()
    dr_c2 = gt_c2.max() - gt_c2.min()
    m.ssim_c1 = ssim(gt_c1, interp_c1, data_range=dr_c1) if dr_c1>0 else 1.0
    m.ssim_c2 = ssim(gt_c2, interp_c2, data_range=dr_c2) if dr_c2>0 else 1.0
    # PDE residual (simplified)
    dx, dy = x[1]-x[0], y[1]-y[0]
    def lap(c):
        lap = np.zeros_like(c)
        lap[1:-1,1:-1] = ((c[2:,1:-1]-2*c[1:-1,1:-1]+c[:-2,1:-1])/dy**2 + (c[1:-1,2:]-2*c[1:-1,1:-1]+c[1:-1,:-2])/dx**2)
        return lap
    lap1 = lap(interp_c1)
    lap2 = lap(interp_c2)
    resid1 = -(PHYSICS_CONSTANTS['D11']*lap1 + PHYSICS_CONSTANTS['D12']*lap2)
    resid2 = -(PHYSICS_CONSTANTS['D21']*lap1 + PHYSICS_CONSTANTS['D22']*lap2)
    resid = np.sqrt(resid1**2 + resid2**2)
    m.pde_residual_mean = float(np.mean(resid))
    m.pde_residual_max = float(np.max(resid))
    # BC errors
    m.bc_error_top_c1 = np.abs(np.mean(interp_c1[:,-1]) - params.get('c_cu_top',1.59e-3))
    m.bc_error_top_c2 = np.abs(np.mean(interp_c2[:,-1]) - params.get('c_ni_top',0))
    m.bc_error_bottom_c1 = np.abs(np.mean(interp_c1[:,0]) - params.get('c_cu_bottom',0))
    m.bc_error_bottom_c2 = np.abs(np.mean(interp_c2[:,0]) - params.get('c_ni_bottom',4e-4))
    # Mass error
    area = dx*dy
    mass1 = np.sum(interp_c1)*area
    mass2 = np.sum(interp_c2)*area
    init_mass1 = params.get('c_cu_initial',1.5e-3)*interp_c1.size*area
    init_mass2 = params.get('c_ni_initial',4e-4)*interp_c2.size*area
    m.mass_error = max(abs(mass1-init_mass1)/(init_mass1+1e-12), abs(mass2-init_mass2)/(init_mass2+1e-12))
    # Uncertainty
    if weights is not None and len(weights)>0:
        eps=1e-10
        m.weight_entropy = -np.sum(weights*np.log(weights+eps))
    if 'target_params' in params and 'source_params' in params:
        target = params['target_params']
        sources = params['source_params']
        if sources:
            dists = []
            for src in sources:
                d_ly = abs(target.get('Ly',60)-src.get('Ly',60))/90
                d_cu = abs(target.get('c_cu',1.5e-3)-src.get('c_cu',1.5e-3))/2.9e-3
                d_ni = abs(target.get('c_ni',4e-4)-src.get('c_ni',4e-4))/1.8e-3
                dists.append(np.sqrt(d_ly**2+d_cu**2+d_ni**2))
            m.param_distance = min(dists) if dists else 1.0
    # Overall score
    scores = [np.exp(-m.mse_c1/1e-6), np.exp(-m.mse_c2/1e-6), max(0,m.r2_c1), max(0,m.r2_c2), m.ssim_c1, m.ssim_c2,
              np.exp(-m.pde_residual_mean/1e-4), 1-m.mass_error, np.exp(-m.param_distance/0.3)]
    m.overall_score = float(np.mean(scores))
    return m

# =============================================
# PLOTTING FUNCTIONS (PLOTLY)
# =============================================
def plot_metrics_bar_chart(metrics_df, title="Validation Metrics"):
    df = metrics_df.copy()
    error_metrics = ['mse_c1','mse_c2','mae_c1','mae_c2','max_error_c1','max_error_c2','pde_residual_mean','pde_residual_max','mass_error']
    for col in error_metrics:
        if col in df['Metric'].values:
            idx = df[df['Metric']==col].index[0]
            val = df.loc[idx,'Value']
            df.loc[idx,'Value'] = np.exp(-val/1e-4) if val>1e-8 else 1.0
    fig = go.Figure()
    for cat in df['Category'].unique():
        sub = df[df['Category']==cat]
        fig.add_trace(go.Bar(x=sub['Metric'], y=sub['Value'], name=cat, text=sub['Value'].apply(lambda x:f'{x:.3f}'), textposition='auto'))
    fig.update_layout(title=dict(text=title, x=0.5), xaxis_title="Metric", yaxis_title="Normalized Score (0-1)", barmode='group', height=500)
    return fig

def plot_radar_chart(metrics, title="Radar Plot"):
    categories = ['MSE Cu','MSE Ni','R² Cu','R² Ni','SSIM Cu','SSIM Ni','PDE Res.','Mass Cons.','Param Dist.']
    values = [np.exp(-metrics.mse_c1/1e-6), np.exp(-metrics.mse_c2/1e-6), max(0,metrics.r2_c1), max(0,metrics.r2_c2),
              metrics.ssim_c1, metrics.ssim_c2, np.exp(-metrics.pde_residual_mean/1e-4), 1-metrics.mass_error,
              np.exp(-metrics.param_distance/0.3)]
    categories += [categories[0]]; values += [values[0]]
    fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself', line_color='#2E86AB', fillcolor='rgba(46,134,171,0.3)'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), title=dict(text=title, x=0.5), height=500)
    return fig

def plot_comparison_heatmaps(interp_c1, gt_c1, interp_c2, gt_c2, x, y, title_prefix=""):
    fig_c1 = make_subplots(rows=1, cols=3, subplot_titles=("Interpolated","Ground Truth","Absolute Error"))
    fig_c1.add_trace(go.Heatmap(z=interp_c1.T, x=x, y=y, colorscale='viridis'), row=1, col=1)
    fig_c1.add_trace(go.Heatmap(z=gt_c1.T, x=x, y=y, colorscale='viridis', showscale=False), row=1, col=2)
    fig_c1.add_trace(go.Heatmap(z=np.abs(interp_c1-gt_c1).T, x=x, y=y, colorscale='RdYlBu_r'), row=1, col=3)
    fig_c1.update_layout(title=f"{title_prefix} - Cu", height=400)
    fig_c2 = make_subplots(rows=1, cols=3, subplot_titles=("Interpolated","Ground Truth","Absolute Error"))
    fig_c2.add_trace(go.Heatmap(z=interp_c2.T, x=x, y=y, colorscale='magma'), row=1, col=1)
    fig_c2.add_trace(go.Heatmap(z=gt_c2.T, x=x, y=y, colorscale='magma', showscale=False), row=1, col=2)
    fig_c2.add_trace(go.Heatmap(z=np.abs(interp_c2-gt_c2).T, x=x, y=y, colorscale='RdYlBu_r'), row=1, col=3)
    fig_c2.update_layout(title=f"{title_prefix} - Ni", height=400)
    return fig_c1, fig_c2

def plot_uncertainty_scatter(metrics_list):
    df = pd.DataFrame([{'Param Distance': m.param_distance, 'Weight Entropy': m.weight_entropy, 'Overall Score': m.overall_score} for m in metrics_list])
    fig = px.scatter(df, x='Param Distance', y='Weight Entropy', size='Overall Score', color='Overall Score',
                     color_continuous_scale='Viridis', title="Uncertainty vs Parameter Distance",
                     labels={'Param Distance':'Normalized distance','Weight Entropy':'Attention entropy'})
    fig.update_layout(height=500)
    return fig

# =============================================
# MAIN STREAMLIT APP
# =============================================
def main():
    st.set_page_config(page_title="Cu-Ni Diffusion: Validation & Uncertainty", layout="wide")
    st.title("🔬 Physics‑Informed Cu‑Ni Interdiffusion with Validation & Uncertainty")

    # Initialize session state
    if 'nl_parser' not in st.session_state:
        st.session_state.nl_parser = DiffusionNLParser()
    if 'llm_backend_loaded' not in st.session_state:
        st.session_state.llm_backend_loaded = "GPT-2 (default)"
    if 'llm_cache' not in st.session_state:
        st.session_state.llm_cache = OrderedDict()
    if 'parsed_params' not in st.session_state:
        st.session_state.parsed_params = None
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    if 'params_list' not in st.session_state:
        st.session_state.params_list = []
    if 'validation_results' not in st.session_state:
        st.session_state.validation_results = {}
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = None
    if 'physics_interpolator' not in st.session_state:
        st.session_state.physics_interpolator = None

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        # LLM options
        if TRANSFORMERS_AVAILABLE:
            backend = st.selectbox("LLM Backend", ["GPT-2 (default)", "Qwen2-0.5B-Instruct", "Qwen2.5-0.5B-Instruct"])
            if backend != st.session_state.llm_backend_loaded:
                st.session_state.llm_backend_loaded = backend
                st.session_state.llm_cache.clear()
                st.rerun()
            tokenizer, model, _ = load_llm(backend)
            st.session_state.llm_tokenizer = tokenizer
            st.session_state.llm_model = model
            use_llm = st.checkbox("Enable LLM Parsing", value=True)
        else:
            use_llm = False
            st.session_state.llm_tokenizer = None
            st.session_state.llm_model = None
            st.info("LLM disabled. Install transformers to enable.")

        st.header("🎯 Target Parameters")
        ly_input = st.number_input("Ly (μm)", 30.0, 120.0, 60.0, 0.1)
        c_cu_input = st.number_input("C_Cu (mol/cc)", 0.0, 2.9e-3, 1.5e-3, 1e-5, format="%.1e")
        c_ni_input = st.number_input("C_Ni (mol/cc)", 0.0, 1.8e-3, 0.5e-3, 1e-5, format="%.1e")

        st.header("🔬 Validation Settings")
        held_out_frac = st.slider("Held-out fraction", 0.1, 0.5, 0.2, 0.05)
        physics_aware = st.checkbox("Physics-aware interpolation", value=True)
        optimize_fields = st.checkbox("Optimize fields (PDE refinement)", value=True, disabled=not physics_aware)

        if st.button("🚀 Run Validation", type="primary"):
            with st.spinner("Loading solutions..."):
                solutions, params_list, _, _, _, logs = load_solutions(SOLUTION_DIR)
                if not solutions:
                    st.error("No solutions found.")
                    st.stop()
                st.session_state.solutions = solutions
                st.session_state.params_list = params_list
                with st.expander("Load logs"):
                    for log in logs:
                        st.write(log)

            with st.spinner("Running validation..."):
                # Initialize interpolators
                if st.session_state.interpolator is None:
                    st.session_state.interpolator = MultiParamAttentionInterpolator()
                if physics_aware and st.session_state.physics_interpolator is None:
                    st.session_state.physics_interpolator = PhysicsAwareInterpolator(st.session_state.interpolator)

                # Randomly select held-out indices
                np.random.seed(42)
                n_held = max(1, int(len(solutions)*held_out_frac))
                held_indices = np.random.choice(len(solutions), n_held, replace=False).tolist()

                results = {}
                for idx in held_indices:
                    gt = solutions[idx]
                    target = {
                        'Ly': gt['params']['Ly'],
                        'c_cu': gt['params']['C_Cu'],
                        'c_ni': gt['params']['C_Ni'],
                        'Lx': gt['params']['Lx'],
                        't_max': gt['params']['t_max'],
                        'c_cu_top': gt['params']['C_Cu'],
                        'c_cu_bottom': 0.0,
                        'c_ni_top': 0.0,
                        'c_ni_bottom': gt['params']['C_Ni'],
                        'c_cu_initial': 1.5e-3,
                        'c_ni_initial': 4e-4,
                        'target_params': gt['params'],
                        'source_params': [s['params'] for s in solutions if s != gt]
                    }
                    if physics_aware and st.session_state.physics_interpolator:
                        interp_res = st.session_state.physics_interpolator.interpolate_with_physics(
                            solutions, params_list, target, optimize=optimize_fields
                        )
                    else:
                        interp_res = st.session_state.interpolator(solutions, params_list, target['Ly'], target['c_cu'], target['c_ni'])

                    if not interp_res:
                        continue
                    interp_c1 = np.array(interp_res['c1_preds'][-1])
                    interp_c2 = np.array(interp_res['c2_preds'][-1])
                    gt_c1 = np.array(gt['c1_preds'][-1])
                    gt_c2 = np.array(gt['c2_preds'][-1])
                    x = gt['X'][:,0]
                    y = gt['Y'][0,:]
                    t = gt['params']['t_max']
                    weights = interp_res.get('attention_weights', None)
                    metrics = compute_validation_metrics(interp_c1, interp_c2, gt_c1, gt_c2, x, y, t, target, weights)
                    results[idx] = {'metrics': metrics, 'interp_c1': interp_c1, 'interp_c2': interp_c2, 'gt_c1': gt_c1, 'gt_c2': gt_c2, 'x': x, 'y': y, 'params': target}
                st.session_state.validation_results = results
                st.success(f"Validation completed on {len(results)} held-out cases.")

    # Main area
    if not st.session_state.validation_results:
        st.info("👈 Configure settings and click 'Run Validation' to start.")
        return

    results = st.session_state.validation_results
    st.subheader("📊 Validation Summary")
    summary_data = []
    for idx, res in results.items():
        m = res['metrics']
        summary_data.append({
            'Case': f"#{idx}",
            'Overall Score': f"{m.overall_score:.3f}",
            'MSE Cu': f"{m.mse_c1:.2e}",
            'MSE Ni': f"{m.mse_c2:.2e}",
            'PDE Residual': f"{m.pde_residual_mean:.2e}",
            'Mass Error': f"{m.mass_error:.2%}",
            'Weight Entropy': f"{m.weight_entropy:.3f}",
            'Param Distance': f"{m.param_distance:.3f}"
        })
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

    # Tabs for visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Metrics Charts", "🎯 Radar Analysis", "🔍 Field Comparison", "📉 Uncertainty"])
    with tab1:
        # Aggregate metrics across all cases
        all_metrics = pd.concat([res['metrics'].to_dataframe() for res in results.values()])
        avg_metrics = all_metrics.groupby('Metric')['Value'].mean().reset_index()
        avg_metrics['Category'] = all_metrics.groupby('Metric')['Category'].first().values
        fig_bar = plot_metrics_bar_chart(avg_metrics)
        st.plotly_chart(fig_bar, use_container_width=True)
    with tab2:
        first_metrics = list(results.values())[0]['metrics']
        fig_radar = plot_radar_chart(first_metrics)
        st.plotly_chart(fig_radar, use_container_width=True)
    with tab3:
        case_idx = st.selectbox("Select case", list(results.keys()), format_func=lambda x: f"Case #{x}")
        res = results[case_idx]
        fig_c1, fig_c2 = plot_comparison_heatmaps(res['interp_c1'], res['gt_c1'], res['interp_c2'], res['gt_c2'], res['x'], res['y'], title_prefix=f"Case #{case_idx}")
        st.plotly_chart(fig_c1, use_container_width=True)
        st.plotly_chart(fig_c2, use_container_width=True)
    with tab4:
        metrics_list = [res['metrics'] for res in results.values()]
        fig_scatter = plot_uncertainty_scatter(metrics_list)
        st.plotly_chart(fig_scatter, use_container_width=True)
        # Insights
        avg_entropy = np.mean([m.weight_entropy for m in metrics_list])
        avg_dist = np.mean([m.param_distance for m in metrics_list])
        avg_score = np.mean([m.overall_score for m in metrics_list])
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg. Weight Entropy", f"{avg_entropy:.3f}")
        col2.metric("Avg. Parameter Distance", f"{avg_dist:.3f}")
        col3.metric("Avg. Validation Score", f"{avg_score:.3f}")
        if avg_dist > 0.3:
            st.warning("⚠️ Target parameters far from training data. Consider adding nearby simulations.")
        if avg_entropy > 1.5:
            st.warning("⚠️ High weight entropy → ambiguous source selection. Review parameter space coverage.")
        if avg_score < 0.7:
            st.error("❌ Low validation score. Enable physics-aware refinement or collect more training data.")

    # Export
    with st.expander("💾 Export Report"):
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': {'held_out_fraction': held_out_frac, 'physics_aware': physics_aware, 'optimize_fields': optimize_fields},
            'summary': summary_data,
            'detailed_metrics': {idx: res['metrics'].__dict__ for idx, res in results.items()}
        }
        json_str = json.dumps(report, indent=2, default=str)
        st.download_button("Download JSON Report", json_str, f"validation_{datetime.now():%Y%m%d_%H%M%S}.json", "application/json")
        csv_str = pd.concat([res['metrics'].to_dataframe() for res in results.values()]).to_csv(index=False)
        st.download_button("Download Metrics CSV", csv_str, f"metrics_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")

if __name__ == "__main__":
    main()
