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
- Proper solution loading from pinn_solutions/ directory (CoreShellGPT pattern)
"""
import os
import re
import json
import pickle
import hashlib
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
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
from collections import OrderedDict
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================
# 1. GLOBAL CONFIGURATION & MATPLOTLIB SETUP
# =============================================
mpl.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'grid.linestyle': '--',
    'grid.alpha': 0.3,
    'image.cmap': 'viridis'
})

# Physical constants for Cu-Ni cross-diffusion
PHYSICS_CONSTANTS = {
    'D11': 0.006,       # Cu self-diffusivity (μm²/s)
    'D12': 0.00427,     # Cu-Ni cross-diffusivity
    'D21': 0.003697,    # Ni-Cu cross-diffusivity
    'D22': 0.0054,      # Ni self-diffusivity
    'C_CU_RANGE': (0.0, 2.9e-3),   # mol/cc
    'C_NI_RANGE': (0.0, 1.8e-3),   # mol/cc
    'LY_RANGE': (30.0, 120.0),     # μm
    'T_MAX': 200.0,                # s
    'MASS_TOLERANCE': 1e-4,        # Relative mass conservation tolerance
}

# Directory for precomputed PINN solutions
SOLUTION_DIR = os.path.join(os.path.dirname(__file__), "pinn_solutions")
os.makedirs(SOLUTION_DIR, exist_ok=True)
os.makedirs("figures", exist_ok=True)

# Available colormaps for visualization
COLORMAPS = [
    "viridis", "plasma", "inferno", "magma", "cividis",
    "Greys", "Purples", "Blues", "Greens", "Oranges", "Reds",
    "YlOrBr", "YlOrRd", "OrRd", "PuRd", "RdPu", "BuPu", "GnBu",
    "PuBu", "YlGnBu", "PuBuGn", "BuGn", "YlGn",
    "coolwarm", "bwr", "seismic", "RdBu", "Spectral",
    "tab10", "Set1", "Set2", "Pastel1"
]

# =============================================
# 2. LLM IMPORT WITH GRACEFUL FALLBACK
# =============================================
try:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.warning("⚠️ `transformers` not installed. LLM features will be disabled. Install via `pip install transformers torch`")

# =============================================
# 3. NATURAL LANGUAGE PARSER (Hybrid Regex + LLM)
# =============================================
class DiffusionNLParser:
    """
    Extracts Cu-Ni diffusion parameters from natural language using a hybrid approach:
    1. Regex-based deterministic extraction (fast, reliable)
    2. LLM-based semantic extraction (GPT-2/Qwen fallback)
    3. Confidence-aware merging with manual LRU caching
    """
    def __init__(self):
        self.defaults = {
            'ly_target': 60.0,
            'c_cu_target': 1.5e-3,
            'c_ni_target': 0.5e-3,
            'sigma': 0.20,
            'num_heads': 4,
            'd_head': 8,
            'seed': 42
        }
        # Flexible patterns with optional delimiters [=:]? and simple fallbacks
        self.patterns = {
            'ly_target': [
                r'(?:joint\s*thickness|domain\s*length|L_y|Ly)\s*[=:]?\s*(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*(?:μm|um|microns?)',
            ],
            'c_cu_target': [
                r'(?:Cu\s*concentration|C_Cu|c_Cu|top\s*concentration)\s*[=:]?\s*([\d.]+(?:e[+-]?\d+)?)',
                r'Cu\s*[=:]?\s*([\d.]+(?:e[+-]?\d+)?)',  # Fallback: "Cu 1.26e-3"
            ],
            'c_ni_target': [
                r'(?:Ni\s*concentration|C_Ni|c_Ni|bottom\s*concentration)\s*[=:]?\s*([\d.]+(?:e[+-]?\d+)?)',
                r'Ni\s*[=:]?\s*([\d.]+(?:e[+-]?\d+)?)',  # Fallback: "Ni 0.8e-3"
            ],
        }

    def parse_regex(self, text: str) -> Dict[str, Any]:
        """Extract parameters using robust regex patterns with native scientific notation handling."""
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
                            # Python's float() natively handles scientific notation
                            params[key] = float(val)
                        elif key in ['num_heads', 'd_head', 'seed']:
                            params[key] = int(float(val))
                        elif key == 'sigma':
                            params[key] = float(val)
                    except (ValueError, TypeError):
                        pass
                    break  # Stop at first match for this key
        
        # Hard clip to valid physical ranges to prevent interpolation divergence
        params['ly_target'] = np.clip(params['ly_target'], 30.0, 120.0)
        params['c_cu_target'] = np.clip(params['c_cu_target'], 0.0, 2.9e-3)
        params['c_ni_target'] = np.clip(params['c_ni_target'], 0.0, 1.8e-3)
        params['sigma'] = np.clip(params['sigma'], 0.05, 0.5)
        return params

    @staticmethod
    def _extract_json_robust(generated: str) -> Optional[Dict]:
        """Robustly extract JSON from LLM output with structural repair attempts."""
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        match = re.search(json_pattern, generated, re.DOTALL)
        if not match:
            match = re.search(r'\{.*?\}', generated, re.DOTALL)
        if not match:
            return None
        json_str = match.group(0)
        # Repair common LLM JSON artifacts
        json_str = re.sub(r'(true|false|null)\s*(")', r'\1,\2', json_str)
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

    def parse_with_llm(self, text: str, tokenizer, model, regex_params: Dict = None, temperature: float = None) -> Dict:
        """Use LLM (GPT-2 or Qwen) to extract parameters with few-shot prompting."""
        if not tokenizer or not model:
            return self.parse_regex(text)
        
        backend = st.session_state.get('llm_backend_loaded', 'GPT-2')
        temperature = 0.0 if "Qwen" in backend else (temperature if temperature is not None else 0.1)
        
        system = "You are a materials science expert. Extract simulation parameters from the user's query. Reply ONLY with a valid JSON object."
        examples = """
Examples:
- "Analyze a 50 μm joint with Cu concentration 1.2e-3 and Ni 0.8e-3" → {"ly_target": 50.0, "c_cu_target": 1.2e-3, "c_ni_target": 0.8e-3, "sigma": 0.2}
- "Domain length 80, C_Cu=2.0e-3, C_Ni=1.0e-3" → {"ly_target": 80.0, "c_cu_target": 2.0e-3, "c_ni_target": 1.0e-3, "sigma": 0.2}
"""
        defaults_json = json.dumps(self.defaults)
        regex_hint = f"\nRegex hint (use as reference): {json.dumps(regex_params or {})}" if regex_params else ""
        
        user = f"""{examples}{regex_hint}
Query: "{text}"
JSON keys must be: ly_target (float, 30-120 μm), c_cu_target (float, 0-2.9e-3 mol/cc), c_ni_target (float, 0-1.8e-3 mol/cc), sigma (float, 0.05-0.5).
Defaults: {defaults_json}
JSON:"""
        
        # Qwen uses chat templates; GPT-2 uses raw prompt
        if "Qwen" in backend:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"{system}\n{user}\n"
        
        try:
            inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
            with torch.no_grad():
                out = model.generate(inputs, max_new_tokens=150, temperature=temperature, do_sample=(temperature>0), pad_token_id=tokenizer.eos_token_id)
            generated = tokenizer.decode(out[0], skip_special_tokens=True)
            res = self._extract_json_robust(generated)
            if res:
                for k in self.defaults:
                    if k not in res: res[k] = self.defaults[k]
                # Clip & Validate
                res['ly_target'] = np.clip(float(res.get('ly_target', 60)), 30, 120)
                res['c_cu_target'] = np.clip(float(res.get('c_cu_target', 1.5e-3)), 0, 2.9e-3)
                res['c_ni_target'] = np.clip(float(res.get('c_ni_target', 0.5e-3)), 0, 1.8e-3)
                res['sigma'] = np.clip(float(res.get('sigma', 0.2)), 0.05, 0.5)
                # Confidence merge: prefer regex if mismatch is significant
                if regex_params:
                    for k in ['ly_target', 'c_cu_target', 'c_ni_target']:
                        if k in regex_params and abs(res[k] - regex_params[k]) > 1e-4:
                            res[k] = regex_params[k]
                return res
        except Exception as e:
            st.warning(f"LLM parsing failed: {e}. Falling back to regex.")
        return self.parse_regex(text)

    def hybrid_parse(self, text: str, tokenizer, model, use_llm: bool = True) -> Dict:
        """Run regex first, optionally LLM, and merge with confidence-based fallbacks."""
        regex_params = self.parse_regex(text)
        if use_llm and tokenizer and model:
            cache_key = hashlib.md5((text + st.session_state.get('llm_backend_loaded', '')).encode()).hexdigest()
            if 'llm_cache' not in st.session_state:
                st.session_state.llm_cache = OrderedDict()
            if cache_key in st.session_state.llm_cache:
                llm_res = st.session_state.llm_cache[cache_key]
            else:
                llm_res = self.parse_with_llm(text, tokenizer, model, regex_params)
                # LRU eviction
                if len(st.session_state.llm_cache) > 20:
                    st.session_state.llm_cache.popitem(last=False)
                st.session_state.llm_cache[cache_key] = llm_res
            
            # Merge logic
            final = self.defaults.copy()
            for k in final:
                if llm_res[k] != self.defaults[k]:
                    final[k] = llm_res[k]
                elif regex_params[k] != self.defaults[k]:
                    final[k] = regex_params[k]
            return final
        return regex_params

    def get_explanation(self, params: dict, original_text: str) -> str:
        """Generate a diagnostic markdown table of extracted parameters."""
        lines = ["### 🔍 Parsed Parameters from Natural Language", f"**Query:** _{original_text}_", "| Parameter | Extracted Value | Valid Range | Status |", "|-----------|-----------------|-------------|--------|"]
        for key, val in params.items():
            if key == 'sigma': continue
            status = "✅ Extracted" if val != self.defaults[key] else "⚪ Default"
            if key == 'ly_target':
                val_str, rng = f"{val:.3f} μm", "30.0–120.0 μm"
            elif key in ['c_cu_target', 'c_ni_target']:
                val_str, rng = f"{val:.1e} mol/cc", f"0.0–{2.9e-3 if key=='c_cu_target' else 1.8e-3:.1e} mol/cc"
            else:
                val_str, rng = str(val), "N/A"
            lines.append(f"| {key} | {val_str} | {rng} | {status} |")
        return "\n".join(lines)

# =============================================
# 4. UNIFIED LLM LOADER WITH CACHING
# =============================================
@st.cache_resource(show_spinner="Loading LLM backend...")
def load_llm(backend_name: str):
    """Load tokenizer and model for specified backend. Cached forever per backend."""
    if not TRANSFORMERS_AVAILABLE:
        return None, None, "Regex Fallback Only"
    
    try:
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
    except Exception as e:
        st.error(f"Failed to load {backend_name}: {e}")
        return None, None, "Load Failed"

# =============================================
# 5. ENHANCED SOLUTION LOADER (CoreShellGPT Pattern)
# =============================================
class EnhancedSolutionLoader:
    """
    Loads PKL files from pinn_solutions directory, parsing filenames as fallback.
    Follows the CoreShellGPT pattern for robust solution management.
    """
    def __init__(self, solutions_dir: str = SOLUTION_DIR):
        self.solutions_dir = solutions_dir
        self._ensure_directory()
        self.cache = {}
    
    def _ensure_directory(self):
        """Create solutions directory if it doesn't exist."""
        os.makedirs(self.solutions_dir, exist_ok=True)
    
    def scan_solutions(self) -> List[Dict[str, Any]]:
        """Scan directory for PKL files and return file info."""
        import glob
        all_files = []
        for ext in ['*.pkl', '*.pickle']:
            pattern = os.path.join(self.solutions_dir, ext)
            files = glob.glob(pattern)
            all_files.extend(files)
        all_files.sort(key=os.path.getmtime, reverse=True)
        
        file_info = []
        for file_path in all_files:
            try:
                info = {
                    'path': file_path,
                    'filename': os.path.basename(file_path),
                    'size': os.path.getsize(file_path),
                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path)),
                    'format': 'pkl'
                }
                file_info.append(info)
            except:
                continue
        return file_info
    
    def parse_filename(self, filename: str) -> Dict[str, Any]:
        """Parse simulation parameters from filename pattern."""
        params = {}
        
        # Extract Ly (domain height)
        ly_match = re.search(r'Ly[_-]?([0-9.]+)', filename, re.IGNORECASE)
        if ly_match:
            params['Ly'] = float(ly_match.group(1))
        
        # Extract C_Cu (top boundary concentration)
        ccu_match = re.search(r'C_Cu[_-]?([0-9.eE+-]+)', filename, re.IGNORECASE)
        if ccu_match:
            params['C_Cu'] = float(ccu_match.group(1))
        
        # Extract C_Ni (bottom boundary concentration)
        cni_match = re.search(r'C_Ni[_-]?([0-9.eE+-]+)', filename, re.IGNORECASE)
        if cni_match:
            params['C_Ni'] = float(cni_match.group(1))
        
        # Extract Lx if present
        lx_match = re.search(r'Lx[_-]?([0-9.]+)', filename, re.IGNORECASE)
        if lx_match:
            params['Lx'] = float(lx_match.group(1))
        
        # Extract t_max if present
        tmax_match = re.search(r't[_-]?max[_-]?([0-9.]+)', filename, re.IGNORECASE)
        if tmax_match:
            params['t_max'] = float(tmax_match.group(1))
        
        return params
    
    def _ensure_2d(self, arr):
        """Ensure array is 2D for visualization."""
        if arr is None:
            return np.zeros((1, 1))
        if torch.is_tensor(arr):
            arr = arr.cpu().numpy()
        if arr.ndim == 3:
            mid = arr.shape[0] // 2
            return arr[mid, :, :]
        elif arr.ndim == 1:
            n = int(np.sqrt(arr.size))
            return arr[:n*n].reshape(n, n)
        return arr
    
    def _convert_tensors(self, data):
        """Convert PyTorch tensors to NumPy arrays recursively."""
        if isinstance(data, dict):
            for key, value in data.items():
                if torch.is_tensor(value):
                    data[key] = value.cpu().numpy()
                elif isinstance(value, (dict, list)):
                    self._convert_tensors(value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if torch.is_tensor(item):
                    data[i] = item.cpu().numpy()
                elif isinstance(item, (dict, list)):
                    self._convert_tensors(item)
    
    def read_simulation_file(self, file_path: str) -> Optional[Dict]:
        """Read and standardize a simulation file."""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Standardize structure
            standardized = {
                'params': {},
                'X': None,
                'Y': None,
                'c1_preds': [],
                'c2_preds': [],
                'times': [],
                'metadata': {
                    'filename': os.path.basename(file_path),
                    'loaded_at': datetime.now().isoformat(),
                }
            }
            
            if isinstance(data, dict):
                # Extract parameters
                if 'params' in data and isinstance(data['params'], dict):
                    standardized['params'].update(data['params'])
                if 'parameters' in data and isinstance(data['parameters'], dict):
                    standardized['params'].update(data['parameters'])
                
                # Extract fields
                if 'X' in data:
                    standardized['X'] = self._ensure_2d(data['X'])
                if 'Y' in data:
                    standardized['Y'] = self._ensure_2d(data['Y'])
                if 'c1_preds' in data:
                    standardized['c1_preds'] = [self._ensure_2d(c) for c in data['c1_preds']]
                if 'c2_preds' in data:
                    standardized['c2_preds'] = [self._ensure_2d(c) for c in data['c2_preds']]
                if 'times' in data:
                    standardized['times'] = data['times']
                
                # Fallback: parse from filename
                if not standardized['params']:
                    parsed = self.parse_filename(os.path.basename(file_path))
                    standardized['params'].update(parsed)
                    st.sidebar.info(f"Parsed parameters from filename: {os.path.basename(file_path)}")
                
                # Set defaults for missing params
                params = standardized['params']
                params.setdefault('Ly', 60.0)
                params.setdefault('C_Cu', 1.5e-3)
                params.setdefault('C_Ni', 0.5e-3)
                params.setdefault('Lx', 60.0)
                params.setdefault('t_max', 200.0)
                
                # Validate required fields
                if not standardized['c1_preds'] or not standardized['c2_preds']:
                    st.sidebar.warning(f"No concentration fields in {os.path.basename(file_path)}")
                    return None
                
                # Convert tensors
                self._convert_tensors(standardized)
                
                return standardized
            else:
                st.sidebar.warning(f"Unexpected data format in {os.path.basename(file_path)}")
                return None
                
        except Exception as e:
            st.sidebar.error(f"Error loading {os.path.basename(file_path)}: {e}")
            return None
    
    def load_all_solutions(self, use_cache: bool = True, max_files: Optional[int] = None) -> Tuple[List[Dict], List[str]]:
        """Load all valid solutions from directory."""
        solutions = []
        load_logs = []
        
        file_info = self.scan_solutions()
        if max_files:
            file_info = file_info[:max_files]
        
        if not file_info:
            load_logs.append(f"⚠️ No PKL files found in {self.solutions_dir}")
            return solutions, load_logs
        
        for item in file_info:
            cache_key = item['filename']
            
            # Check cache
            if use_cache and cache_key in self.cache:
                solutions.append(self.cache[cache_key])
                load_logs.append(f"✓ {item['filename']} (from cache)")
                continue
            
            # Load file
            sol = self.read_simulation_file(item['path'])
            if sol:
                self.cache[cache_key] = sol
                solutions.append(sol)
                params = sol['params']
                load_logs.append(f"✓ {item['filename']}: Ly={params.get('Ly', 'N/A'):.1f}μm, C_Cu={params.get('C_Cu', 'N/A'):.1e}, C_Ni={params.get('C_Ni', 'N/A'):.1e}")
            else:
                load_logs.append(f"✗ {item['filename']}: Failed to load")
        
        load_logs.append(f"\n📊 Summary: Loaded {len(solutions)}/{len(file_info)} solutions")
        return solutions, load_logs

# =============================================
# 6. ATTENTION-BASED INTERPOLATOR (PYTORCH)
# =============================================
class MultiParamAttentionInterpolator(nn.Module):
    """
    Hybrid Spatial-Parameter Attention Interpolator.
    Combines learned query/key projections with Gaussian spatial locality to blend 
    precomputed PINN diffusion fields.
    """
    def __init__(self, sigma=0.2, num_heads=4, d_head=8):
        super().__init__()
        self.sigma = sigma
        self.num_heads = num_heads
        self.d_head = d_head
        self.W_q = nn.Linear(3, self.num_heads * self.d_head, bias=False)
        self.W_k = nn.Linear(3, self.num_heads * self.d_head, bias=False)

    def normalize_params(self, params: Union[Tuple, List, np.ndarray], is_target: bool = False) -> np.ndarray:
        """Min-Max normalize parameters to [0,1] for stable attention computation."""
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

    def compute_weights(self, params_list: List[Tuple], ly_target: float, c_cu_target: float, c_ni_target: float) -> Dict:
        """Compute hybrid attention + spatial weights for source blending."""
        if not params_list:
            raise ValueError("Empty parameter list provided for interpolation.")
            
        norm_sources = self.normalize_params(params_list)
        norm_target = self.normalize_params((ly_target, c_cu_target, c_ni_target), is_target=True)

        src_tensor = torch.tensor(norm_sources, dtype=torch.float32)
        tgt_tensor = torch.tensor(norm_target, dtype=torch.float32).unsqueeze(0)

        q = self.W_q(tgt_tensor).view(1, self.num_heads, self.d_head)
        k = self.W_k(src_tensor).view(len(params_list), self.num_heads, self.d_head)

        # Scaled dot-product attention
        attn_logits = torch.einsum('nhd,mhd->nmh', k, q) / np.sqrt(self.d_head)
        attn_weights = torch.softmax(attn_logits, dim=0).mean(dim=2).squeeze(1)

        # Gaussian spatial locality weights
        dists = torch.sqrt(
            ((src_tensor[:, 0] - norm_target[0]) / self.sigma)**2 +
            ((src_tensor[:, 1] - norm_target[1]) / self.sigma)**2 +
            ((src_tensor[:, 2] - norm_target[2]) / self.sigma)**2
        )
        spatial_weights = torch.exp(-dists**2 / 2)
        spatial_weights /= spatial_weights.sum() + 1e-8

        # Hybrid combination
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

    def _physics_aware_interpolation(self, solutions, weights, ly_target, c_cu_target, c_ni_target):
        """Interpolate fields with physics-aware post-processing."""
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

        # Enforce boundary conditions
        c1_interp[:, :, 0] = c_cu_target  # Cu at y=0
        c2_interp[:, :, -1] = c_ni_target  # Ni at y=Ly

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

    def forward(self, solutions, params_list, ly_target, c_cu_target, c_ni_target):
        """Main forward pass: compute weights and interpolate."""
        weights_result = self.compute_weights(params_list, ly_target, c_cu_target, c_ni_target)
        return self._physics_aware_interpolation(
            solutions, 
            weights_result['combined_weights'], 
            ly_target, c_cu_target, c_ni_target
        )

# =============================================
# 7. PHYSICS-INFORMED LOSS FUNCTIONS
# =============================================
def compute_pde_residual(c1: np.ndarray, c2: np.ndarray, 
                        x: np.ndarray, y: np.ndarray, t: float,
                        D11: float, D12: float, D21: float, D22: float,
                        dx: float, dy: float) -> np.ndarray:
    """
    Compute Fick's second law residual for cross-diffusion:
    ∂c₁/∂t = D₁₁∇²c₁ + D₁₂∇²c₂
    ∂c₂/∂t = D₂₁∇²c₁ + D₂₂∇²c₂
    
    Uses finite differences for spatial derivatives.
    Returns residual field of shape (ny, nx).
    """
    ny, nx = c1.shape
    
    # Laplacian using 5-point stencil
    def laplacian(c):
        lap = np.zeros_like(c)
        # Interior points
        lap[1:-1, 1:-1] = (
            (c[2:, 1:-1] - 2*c[1:-1, 1:-1] + c[:-2, 1:-1]) / dy**2 +
            (c[1:-1, 2:] - 2*c[1:-1, 1:-1] + c[1:-1, :-2]) / dx**2
        )
        # Boundary: one-sided differences (Neumann assumed)
        lap[0, 1:-1] = (c[1, 1:-1] - c[0, 1:-1]) / dy**2 + \
                      (c[0, 2:] - 2*c[0, 1:-1] + c[0, :-2]) / dx**2
        lap[-1, 1:-1] = (c[-1, 1:-1] - c[-2, 1:-1]) / dy**2 + \
                       (c[-1, 2:] - 2*c[-1, 1:-1] + c[-1, :-2]) / dx**2
        lap[1:-1, 0] = (c[1:-1, 1] - c[1:-1, 0]) / dx**2 + \
                      (c[2:, 0] - 2*c[1:-1, 0] + c[:-2, 0]) / dy**2
        lap[1:-1, -1] = (c[1:-1, -1] - c[1:-1, -2]) / dx**2 + \
                       (c[2:, -1] - 2*c[1:-1, -1] + c[:-2, -1]) / dy**2
        return lap
    
    lap_c1 = laplacian(c1)
    lap_c2 = laplacian(c2)
    
    # Time derivative approximation (forward difference from previous timestep)
    # For single-time validation, assume quasi-steady: ∂c/∂t ≈ 0
    residual1 = -(D11 * lap_c1 + D12 * lap_c2)  # ∂c₁/∂t ≈ 0
    residual2 = -(D21 * lap_c1 + D22 * lap_c2)
    
    return np.sqrt(residual1**2 + residual2**2)


def enforce_boundary_conditions(c1: np.ndarray, c2: np.ndarray,
                               x: np.ndarray, y: np.ndarray,
                               c_cu_top: float, c_cu_bottom: float,
                               c_ni_top: float, c_ni_bottom: float,
                               enforce_type: str = 'hard') -> Tuple[np.ndarray, np.ndarray]:
    """
    Enforce Dirichlet boundary conditions on interpolated fields.
    
    Args:
        enforce_type: 'hard' (direct assignment) or 'soft' (blending with Gaussian weight)
    """
    c1_bc = c1.copy()
    c2_bc = c2.copy()
    ny, nx = c1.shape
    
    if enforce_type == 'hard':
        # Top boundary (y = Ly)
        c1_bc[:, -1] = c_cu_top
        c2_bc[:, -1] = c_ni_top
        # Bottom boundary (y = 0)
        c1_bc[:, 0] = c_cu_bottom
        c2_bc[:, 0] = c_ni_bottom
        # Side boundaries: Neumann (zero flux) - already satisfied by interpolation
    elif enforce_type == 'soft':
        # Gaussian blending near boundaries
        boundary_width = 3  # cells
        y_weight = np.exp(-((np.arange(ny) - ny/2)**2) / (2 * (ny/4)**2))
        x_weight = np.exp(-((np.arange(nx) - nx/2)**2) / (2 * (nx/4)**2))
        X, Y = np.meshgrid(x_weight, y_weight)
        blend = 1 - X * Y  # 1 at center, 0 at boundaries
        
        # Apply boundary values with blending
        c1_bc = blend * c1 + (1 - blend) * np.where(
            (np.arange(ny)[:, None] == 0) | (np.arange(ny)[:, None] == ny-1),
            np.where(np.arange(ny)[:, None] == 0, c_cu_bottom, c_cu_top),
            c1
        )
        c2_bc = blend * c2 + (1 - blend) * np.where(
            (np.arange(ny)[:, None] == 0) | (np.arange(ny)[:, None] == ny-1),
            np.where(np.arange(ny)[:, None] == 0, c_ni_bottom, c_ni_top),
            c2
        )
    
    return c1_bc, c2_bc


def compute_mass_conservation(c1: np.ndarray, c2: np.ndarray, 
                             x: np.ndarray, y: np.ndarray,
                             c_cu_initial: float, c_ni_initial: float) -> Dict[str, float]:
    """
    Compute relative mass conservation error.
    Assumes initial uniform concentration c_initial.
    """
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    dy = y[1] - y[0] if len(y) > 1 else 1.0
    area = dx * dy
    
    # Total mass at current time
    mass_c1 = np.sum(c1) * area
    mass_c2 = np.sum(c2) * area
    
    # Initial mass (uniform)
    nx, ny = c1.shape
    initial_mass_c1 = c_cu_initial * nx * ny * area
    initial_mass_c2 = c_ni_initial * nx * ny * area
    
    # Relative error
    error_c1 = abs(mass_c1 - initial_mass_c1) / (initial_mass_c1 + 1e-12)
    error_c2 = abs(mass_c2 - initial_mass_c2) / (initial_mass_c2 + 1e-12)
    
    return {
        'mass_error_c1': error_c1,
        'mass_error_c2': error_c2,
        'mass_error_max': max(error_c1, error_c2),
        'mass_c1': mass_c1,
        'mass_c2': mass_c2
    }


# =============================================
# 8. VALIDATION METRICS COMPUTATION
# =============================================
@dataclass
class ValidationMetrics:
    """Container for comprehensive validation metrics."""
    # Pointwise errors
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
    
    # Physics-based metrics
    pde_residual_mean: float = 0.0
    pde_residual_max: float = 0.0
    bc_error_top_c1: float = 0.0
    bc_error_top_c2: float = 0.0
    bc_error_bottom_c1: float = 0.0
    bc_error_bottom_c2: float = 0.0
    mass_error: float = 0.0
    
    # Uncertainty metrics
    weight_entropy: float = 0.0
    param_distance: float = 0.0
    ensemble_variance_c1: float = 0.0
    ensemble_variance_c2: float = 0.0
    
    # Composite score (0-1, higher is better)
    overall_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}
    
    def to_dataframe(self) -> pd.DataFrame:
        data = []
        for key, value in self.__dict__.items():
            category = 'pointwise' if key.startswith(('mse', 'mae', 'max_error', 'r2', 'ssim')) else \
                      'physics' if key.startswith(('pde', 'bc', 'mass')) else \
                      'uncertainty' if key.startswith(('weight', 'param', 'ensemble')) else \
                      'composite'
            data.append({'Metric': key, 'Value': value, 'Category': category})
        return pd.DataFrame(data)


def compute_validation_metrics(interp_c1: np.ndarray, interp_c2: np.ndarray,
                              gt_c1: np.ndarray, gt_c2: np.ndarray,
                              x: np.ndarray, y: np.ndarray, t: float,
                              params: Dict, weights: Optional[np.ndarray] = None,
                              ensemble_fields: Optional[List[Dict]] = None) -> ValidationMetrics:
    """
    Compute comprehensive validation metrics comparing interpolated vs ground-truth PINN fields.
    """
    metrics = ValidationMetrics()
    
    # Flatten for pointwise metrics
    flat_interp_c1 = interp_c1.flatten()
    flat_interp_c2 = interp_c2.flatten()
    flat_gt_c1 = gt_c1.flatten()
    flat_gt_c2 = gt_c2.flatten()
    
    # Pointwise errors
    metrics.mse_c1 = mean_squared_error(flat_gt_c1, flat_interp_c1)
    metrics.mse_c2 = mean_squared_error(flat_gt_c2, flat_interp_c2)
    metrics.mae_c1 = mean_absolute_error(flat_gt_c1, flat_interp_c1)
    metrics.mae_c2 = mean_absolute_error(flat_gt_c2, flat_interp_c2)
    metrics.max_error_c1 = np.max(np.abs(flat_gt_c1 - flat_interp_c1))
    metrics.max_error_c2 = np.max(np.abs(flat_gt_c2 - flat_interp_c2))
    
    # R² and SSIM (only if variance > 0)
    if np.var(flat_gt_c1) > 1e-12:
        metrics.r2_c1 = r2_score(flat_gt_c1, flat_interp_c1)
    if np.var(flat_gt_c2) > 1e-12:
        metrics.r2_c2 = r2_score(flat_gt_c2, flat_interp_c2)
    
    # SSIM: safe handling of constant fields
    data_range_c1 = max(gt_c1.max() - gt_c1.min(), 1e-6)
    data_range_c2 = max(gt_c2.max() - gt_c2.min(), 1e-6)
    if data_range_c1 > 1e-8:
        metrics.ssim_c1 = ssim(gt_c1, interp_c1, data_range=data_range_c1)
    else:
        metrics.ssim_c1 = 1.0 if np.allclose(gt_c1, interp_c1) else 0.0
    if data_range_c2 > 1e-8:
        metrics.ssim_c2 = ssim(gt_c2, interp_c2, data_range=data_range_c2)
    else:
        metrics.ssim_c2 = 1.0 if np.allclose(gt_c2, interp_c2) else 0.0
    
    # Physics-based metrics
    dx, dy = (x[1]-x[0] if len(x) > 1 else 1.0), (y[1]-y[0] if len(y) > 1 else 1.0)
    
    if dx > 0 and dy > 0 and interp_c1.shape[0] > 2 and interp_c1.shape[1] > 2:
        residual = compute_pde_residual(
            interp_c1, interp_c2, x, y, t,
            params.get('D11', PHYSICS_CONSTANTS['D11']),
            params.get('D12', PHYSICS_CONSTANTS['D12']),
            params.get('D21', PHYSICS_CONSTANTS['D21']),
            params.get('D22', PHYSICS_CONSTANTS['D22']),
            dx, dy
        )
        metrics.pde_residual_mean = np.mean(residual)
        metrics.pde_residual_max = np.max(residual)
    else:
        metrics.pde_residual_mean = 0.0
        metrics.pde_residual_max = 0.0
    
    # Boundary condition errors
    ny, nx = interp_c1.shape
    metrics.bc_error_top_c1 = np.abs(np.mean(interp_c1[:, -1]) - params.get('c_cu_top', 0))
    metrics.bc_error_top_c2 = np.abs(np.mean(interp_c2[:, -1]) - params.get('c_ni_top', 0))
    metrics.bc_error_bottom_c1 = np.abs(np.mean(interp_c1[:, 0]) - params.get('c_cu_bottom', 0))
    metrics.bc_error_bottom_c2 = np.abs(np.mean(interp_c2[:, 0]) - params.get('c_ni_bottom', 0))
    
    # Mass conservation
    mass_metrics = compute_mass_conservation(
        interp_c1, interp_c2, x, y,
        params.get('c_cu_initial', 1.5e-3),
        params.get('c_ni_initial', 4.0e-4)
    )
    metrics.mass_error = mass_metrics['mass_error_max']
    
    # Uncertainty metrics
    if weights is not None and len(weights) > 0:
        eps = 1e-10
        # Convert to numpy array and ensure proper dtype
        weights_arr = np.asarray(weights, dtype=np.float64).flatten()
        # Clip to avoid log(0) and re-normalize
        weights_arr = np.clip(weights_arr, eps, 1.0)
        weights_arr = weights_arr / (np.sum(weights_arr) + eps)
        metrics.weight_entropy = float(-np.sum(weights_arr * np.log(weights_arr + eps)))
    
    if 'target_params' in params and 'source_params' in params:
        # Normalized parameter distance
        target = params['target_params']
        sources = params['source_params']
        if sources and len(sources) > 0:
            distances = []
            for src in sources:
                d_ly = abs(target.get('Ly', 60) - src.get('Ly', 60)) / 90  # normalized by range
                d_cu = abs(target.get('C_Cu', 1.5e-3) - src.get('C_Cu', 1.5e-3)) / 2.9e-3
                d_ni = abs(target.get('C_Ni', 4e-4) - src.get('C_Ni', 4e-4)) / 1.8e-3
                distances.append(np.sqrt(d_ly**2 + d_cu**2 + d_ni**2))
            metrics.param_distance = float(min(distances)) if distances else 1.0
    
    # Ensemble variance (if multiple interpolations available)
    if ensemble_fields and len(ensemble_fields) > 1:
        c1_ensemble = np.stack([f['c1'] for f in ensemble_fields], axis=0)
        c2_ensemble = np.stack([f['c2'] for f in ensemble_fields], axis=0)
        metrics.ensemble_variance_c1 = np.mean(np.var(c1_ensemble, axis=0))
        metrics.ensemble_variance_c2 = np.mean(np.var(c2_ensemble, axis=0))
    
    # Composite score: weighted combination (higher = better)
    # Normalize each metric to 0-1 scale (error metrics inverted)
    scores = []
    scores.append(np.exp(-metrics.mse_c1 / 1e-6))  # MSE ~1e-6 is good
    scores.append(np.exp(-metrics.mse_c2 / 1e-6))
    scores.append(metrics.r2_c1 if metrics.r2_c1 > 0 else 0)
    scores.append(metrics.r2_c2 if metrics.r2_c2 > 0 else 0)
    scores.append(metrics.ssim_c1)
    scores.append(metrics.ssim_c2)
    scores.append(np.exp(-metrics.pde_residual_mean / 1e-4))
    scores.append(1 - metrics.mass_error)
    scores.append(np.exp(-metrics.param_distance / 0.3))  # distance < 0.3 is close
    
    metrics.overall_score = float(np.mean(scores))
    
    return metrics


# =============================================
# 9. VISUALIZATION FUNCTIONS (PLOTLY)
# =============================================
def plot_metrics_bar_chart(metrics_df: pd.DataFrame, 
                          title: str = "Validation Metrics",
                          color_map: Optional[Dict] = None) -> go.Figure:
    """Create interactive bar chart of validation metrics, grouped by category."""
    if color_map is None:
        color_map = {
            'pointwise': '#2E86AB',
            'physics': '#A23B72',
            'uncertainty': '#F18F01',
            'composite': '#C73E1D'
        }
    
    # Normalize error metrics for better visualization (invert and scale)
    df = metrics_df.copy()
    error_metrics = ['mse_c1', 'mse_c2', 'mae_c1', 'mae_c2', 'max_error_c1', 'max_error_c2',
                    'pde_residual_mean', 'pde_residual_max', 'bc_error_top_c1', 'bc_error_top_c2',
                    'bc_error_bottom_c1', 'bc_error_bottom_c2', 'mass_error']
    for col in error_metrics:
        if col in df['Metric'].values:
            idx = df[df['Metric'] == col].index[0]
            val = df.loc[idx, 'Value']
            # Map error to [0,1] where 0 error = 1.0 score
            if val < 1e-8:
                df.loc[idx, 'Value'] = 1.0
            else:
                df.loc[idx, 'Value'] = np.exp(-val / 1e-4)  # exponential decay
    
    fig = go.Figure()
    
    for category in df['Category'].unique():
        cat_df = df[df['Category'] == category]
        fig.add_trace(go.Bar(
            x=cat_df['Metric'],
            y=cat_df['Value'],
            name=category,
            marker_color=color_map.get(category, '#666666'),
            text=cat_df['Value'].apply(lambda x: f'{x:.3f}'),
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Metric",
        yaxis_title="Normalized Score (0-1, higher=better)",
        barmode='group',
        legend_title="Category",
        height=500,
        hovermode='x unified'
    )
    
    return fig


def plot_radar_chart(metrics: ValidationMetrics, 
                    title: str = "Validation Radar Plot") -> go.Figure:
    """Create radar chart showing key validation metrics."""
    # Select representative metrics (normalized to 0-1, higher=better)
    categories = [
        'MSE (Cu)', 'MSE (Ni)', 'R² (Cu)', 'R² (Ni)', 
        'SSIM (Cu)', 'SSIM (Ni)', 'PDE Residual', 'Mass Cons.', 'Param. Distance'
    ]
    
    # Normalize and invert error metrics
    values = [
        np.exp(-metrics.mse_c1 / 1e-6),
        np.exp(-metrics.mse_c2 / 1e-6),
        max(0, metrics.r2_c1),
        max(0, metrics.r2_c2),
        metrics.ssim_c1,
        metrics.ssim_c2,
        np.exp(-metrics.pde_residual_mean / 1e-4),
        1 - metrics.mass_error,
        np.exp(-metrics.param_distance / 0.3)
    ]
    
    # Close the polygon
    categories += [categories[0]]
    values += [values[0]]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        line_color='#2E86AB',
        fillcolor='rgba(46, 134, 171, 0.3)',
        name='Interpolated Solution'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat='.1f'
            )),
        title=dict(text=title, x=0.5),
        showlegend=False,
        height=500
    )
    
    return fig


def plot_residual_heatmap(residual: np.ndarray, x: np.ndarray, y: np.ndarray,
                         title: str = "PDE Residual Field") -> go.Figure:
    """Create interactive heatmap of PDE residual."""
    fig = go.Figure(data=go.Heatmap(
        z=residual.T,  # Transpose for correct orientation
        x=x,
        y=y,
        colorscale='RdYlBu_r',
        colorbar=dict(title="Residual"),
        hovertemplate='x=%{x:.1f} μm, y=%{y:.1f} μm<br>Residual=%{z:.2e}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="x (μm)",
        yaxis_title="y (μm)",
        width=600,
        height=500,
        hovermode='closest'
    )
    
    return fig


def plot_uncertainty_scatter(metrics_list: List[ValidationMetrics],
                           param_names: List[str],
                           title: str = "Uncertainty vs Parameter Distance") -> go.Figure:
    """Scatter plot of uncertainty metrics vs parameter distance."""
    df = pd.DataFrame([
        {
            'Param Distance': m.param_distance,
            'Weight Entropy': m.weight_entropy,
            'Ensemble Var (Cu)': m.ensemble_variance_c1,
            'Ensemble Var (Ni)': m.ensemble_variance_c2,
            'Overall Score': m.overall_score
        }
        for m in metrics_list
    ])
    
    fig = px.scatter(
        df,
        x='Param Distance',
        y='Weight Entropy',
        size='Overall Score',
        color='Overall Score',
        color_continuous_scale='Viridis',
        hover_data=['Ensemble Var (Cu)', 'Ensemble Var (Ni)'],
        title=title,
        labels={'Param Distance': 'Normalized Parameter Distance',
               'Weight Entropy': 'Attention Weight Entropy',
               'Overall Score': 'Validation Score'}
    )
    
    fig.update_layout(
        height=500,
        coloraxis_colorbar=dict(title="Score")
    )
    
    return fig


def plot_comparison_heatmaps(interp_c1: np.ndarray, gt_c1: np.ndarray,
                           interp_c2: np.ndarray, gt_c2: np.ndarray,
                           x: np.ndarray, y: np.ndarray,
                           title_prefix: str = "Field Comparison") -> Tuple[go.Figure, go.Figure]:
    """Create side-by-side comparison heatmaps for interpolated vs ground truth."""
    # Cu comparison
    fig_c1 = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Interpolated", "Ground Truth", "Absolute Error"),
        horizontal_spacing=0.05
    )
    
    fig_c1.add_trace(go.Heatmap(z=interp_c1.T, x=x, y=y, colorscale='viridis', showscale=True, name='Interp'), row=1, col=1)
    fig_c1.add_trace(go.Heatmap(z=gt_c1.T, x=x, y=y, colorscale='viridis', showscale=False, name='GT'), row=1, col=2)
    fig_c1.add_trace(go.Heatmap(z=np.abs(interp_c1 - gt_c1).T, x=x, y=y, colorscale='RdYlBu_r', showscale=True, name='Error'), row=1, col=3)
    
    fig_c1.update_layout(title=dict(text=f"{title_prefix} - Cu Concentration", x=0.5), height=400)
    
    # Ni comparison
    fig_c2 = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Interpolated", "Ground Truth", "Absolute Error"),
        horizontal_spacing=0.05
    )
    
    fig_c2.add_trace(go.Heatmap(z=interp_c2.T, x=x, y=y, colorscale='magma', showscale=True, name='Interp'), row=1, col=1)
    fig_c2.add_trace(go.Heatmap(z=gt_c2.T, x=x, y=y, colorscale='magma', showscale=False, name='GT'), row=1, col=2)
    fig_c2.add_trace(go.Heatmap(z=np.abs(interp_c2 - gt_c2).T, x=x, y=y, colorscale='RdYlBu_r', showscale=True, name='Error'), row=1, col=3)
    
    fig_c2.update_layout(title=dict(text=f"{title_prefix} - Ni Concentration", x=0.5), height=400)
    
    return fig_c1, fig_c2


# =============================================
# 10. PHYSICS-INFORMED INTERPOLATION ENHANCEMENT
# =============================================
class PhysicsAwareInterpolator:
    """
    Enhanced attention interpolator with physics constraints.
    Adds PDE residual penalty, boundary enforcement, and mass conservation.
    """
    
    def __init__(self, base_interpolator, physics_constants: Dict = None):
        self.base = base_interpolator
        self.constants = physics_constants or PHYSICS_CONSTANTS
        self.enforce_bc = True
        self.pde_weight = 0.1  # Weight for PDE residual in optimization
        self.mass_weight = 0.05  # Weight for mass conservation
        
    def interpolate_with_physics(self, solutions: List[Dict], params_list: List[Dict],
                               target_params: Dict, target_shape: Tuple[int, int] = (50, 50),
                               time_norm: float = 1.0, optimize: bool = True) -> Dict:
        """
        Perform attention interpolation with optional physics-based refinement.
        
        Args:
            optimize: If True, apply gradient-based refinement to minimize PDE residual
        """
        # Step 1: Base attention interpolation
        result = self.base(solutions, params_list, 
                          target_params.get('Ly', 60),
                          target_params.get('c_cu', 1.5e-3),
                          target_params.get('c_ni', 4e-4))
        
        if not result or 'fields' not in result:
            return result
        
        fields = result['fields']
        c1 = fields.get('c1_preds', [np.zeros(target_shape)])[0]
        c2 = fields.get('c2_preds', [np.zeros(target_shape)])[0]
        
        # Step 2: Enforce boundary conditions
        if self.enforce_bc:
            Lx = target_params.get('Lx', 60)
            Ly = target_params.get('Ly', 60)
            x = np.linspace(0, Lx, target_shape[1])
            y = np.linspace(0, Ly, target_shape[0])
            
            c1, c2 = enforce_boundary_conditions(
                c1, c2, x, y,
                target_params.get('c_cu_top', 1.59e-3),
                target_params.get('c_cu_bottom', 0.0),
                target_params.get('c_ni_top', 0.0),
                target_params.get('c_ni_bottom', 4e-4),
                enforce_type='hard'
            )
        
        # Step 3: Optional physics-based refinement via gradient descent
        if optimize and torch.cuda.is_available():
            c1, c2 = self._physics_refinement(
                c1, c2, target_params, target_shape, time_norm
            )
        
        # Update result
        result['fields']['c1_preds'] = [c1]
        result['fields']['c2_preds'] = [c2]
        result['physics_enforced'] = True
        
        # Compute physics metrics for diagnostics
        Lx = target_params.get('Lx', 60)
        Ly = target_params.get('Ly', 60)
        x = np.linspace(0, Lx, target_shape[1])
        y = np.linspace(0, Ly, target_shape[0])
        t = time_norm * target_params.get('t_max', 200)
        
        residual = compute_pde_residual(
            c1, c2, x, y, t,
            self.constants['D11'], self.constants['D12'],
            self.constants['D21'], self.constants['D22'],
            x[1]-x[0], y[1]-y[0]
        )
        
        mass_metrics = compute_mass_conservation(
            c1, c2, x, y,
            target_params.get('c_cu_initial', 1.5e-3),
            target_params.get('c_ni_initial', 4e-4)
        )
        
        result['physics_diagnostics'] = {
            'pde_residual_mean': float(np.mean(residual)),
            'pde_residual_max': float(np.max(residual)),
            'mass_error': float(mass_metrics['mass_error_max'])
        }
        
        return result
    
    def _physics_refinement(self, c1: np.ndarray, c2: np.ndarray,
                          target_params: Dict, target_shape: Tuple[int, int],
                          time_norm: float, n_steps: int = 50, lr: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Refine interpolated fields by minimizing PDE residual via gradient descent.
        Uses PyTorch for automatic differentiation.
        """
        # Convert to tensors
        c1_t = torch.tensor(c1, dtype=torch.float32, requires_grad=True)
        c2_t = torch.tensor(c2, dtype=torch.float32, requires_grad=True)
        
        Lx = target_params.get('Lx', 60)
        Ly = target_params.get('Ly', 60)
        t = time_norm * target_params.get('t_max', 200)
        
        x = torch.linspace(0, Lx, target_shape[1])
        y = torch.linspace(0, Ly, target_shape[0])
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        optimizer = optim.Adam([c1_t, c2_t], lr=lr)
        
        for step in range(n_steps):
            optimizer.zero_grad()
            
            # Compute PDE residual loss
            lap_c1 = self._laplacian_torch(c1_t, X, Y)
            lap_c2 = self._laplacian_torch(c2_t, X, Y)
            
            # Time derivative approximation (quasi-steady)
            residual1 = -(self.constants['D11'] * lap_c1 + self.constants['D12'] * lap_c2)
            residual2 = -(self.constants['D21'] * lap_c1 + self.constants['D22'] * lap_c2)
            
            pde_loss = torch.mean(residual1**2 + residual2**2)
            
            # Boundary loss (soft constraint)
            bc_loss = (
                torch.mean((c1_t[:, -1] - target_params.get('c_cu_top', 1.59e-3))**2) +
                torch.mean((c1_t[:, 0] - target_params.get('c_cu_bottom', 0.0))**2) +
                torch.mean((c2_t[:, -1] - target_params.get('c_ni_top', 0.0))**2) +
                torch.mean((c2_t[:, 0] - target_params.get('c_ni_bottom', 4e-4))**2)
            )
            
            # Total loss
            loss = self.pde_weight * pde_loss + self.mass_weight * bc_loss
            
            loss.backward()
            optimizer.step()
            
            # Clip to physical ranges
            with torch.no_grad():
                c1_t.clamp_(0, self.constants['C_CU_RANGE'][1])
                c2_t.clamp_(0, self.constants['C_NI_RANGE'][1])
        
        return c1_t.detach().numpy(), c2_t.detach().numpy()
    
    def _laplacian_torch(self, c: torch.Tensor, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian using PyTorch autograd."""
        # First derivatives
        dc_dx = torch.autograd.grad(
            c.sum(), X, create_graph=True, retain_graph=True
        )[0]
        dc_dy = torch.autograd.grad(
            c.sum(), Y, create_graph=True, retain_graph=True
        )[0]
        
        # Second derivatives
        d2c_dx2 = torch.autograd.grad(
            dc_dx.sum(), X, create_graph=True, retain_graph=True
        )[0]
        d2c_dy2 = torch.autograd.grad(
            dc_dy.sum(), Y, create_graph=True, retain_graph=True
        )[0]
        
        return d2c_dx2 + d2c_dy2


# =============================================
# 11. SESSION STATE INITIALIZATION
# =============================================
def initialize_session_state():
    """Initialize Streamlit session state with robust defaults."""
    defaults = {
        'nl_parser': DiffusionNLParser(),
        'llm_backend_loaded': 'GPT-2 (default)',
        'llm_cache': OrderedDict(),
        'parsed_params': DiffusionNLParser().defaults.copy(),
        'nl_query': "",
        'use_llm': True,
        'solutions_loaded': False,
        'validation_results': {},
        'uncertainty_results': {},
        'interpolator': None,
        'physics_interpolator': None,
        'solution_loader': None,
        'solutions': [],
        'params_list': []
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# =============================================
# 12. MAIN STREAMLIT APPLICATION
# =============================================
def main():
    st.set_page_config(page_title="Physics-Informed Validation Dashboard", layout="wide")
    st.title("🔬 Physics-Informed Validation & Uncertainty Quantification")
    
    initialize_session_state()
    
    # === SIDEBAR: CONFIGURATION ===
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Solution loading (CoreShellGPT pattern)
        st.header("📁 Data Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Load Solutions", width='stretch'):
                with st.spinner("Loading solutions from pinn_solutions/..."):
                    if st.session_state.solution_loader is None:
                        st.session_state.solution_loader = EnhancedSolutionLoader(SOLUTION_DIR)
                    
                    solutions, logs = st.session_state.solution_loader.load_all_solutions()
                    st.session_state.solutions = solutions
                    
                    # Extract params list for interpolation
                    params_list = []
                    for sol in solutions:
                        p = sol['params']
                        params_list.append((p['Ly'], p['C_Cu'], p['C_Ni']))
                    st.session_state.params_list = params_list
                    
                    st.session_state.solutions_loaded = len(solutions) > 0
                    
                    # Show load logs
                    with st.expander("📋 Load Logs", expanded=True):
                        for log in logs:
                            st.write(log)
                    
                    if not solutions:
                        st.error("❌ No valid solutions found in pinn_solutions/")
                        st.info("💡 Place .pkl files with keys: params, X, Y, c1_preds, c2_preds, times")
        
        with col2:
            if st.button("🗑️ Clear Cache", width='stretch'):
                st.session_state.solutions = []
                st.session_state.params_list = []
                st.session_state.solutions_loaded = False
                st.session_state.validation_results = {}
                st.success("Cache cleared")
        
        # LLM options
        st.header("🤖 LLM Configuration")
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
        
        # Run validation button
        if st.button("🚀 Run Validation", type="primary", width='stretch'):
            if not st.session_state.solutions_loaded:
                st.error("❌ Please load solutions first!")
                st.stop()
            
            with st.spinner("Running validation pipeline..."):
                # Initialize interpolators
                if st.session_state.interpolator is None:
                    st.session_state.interpolator = MultiParamAttentionInterpolator()
                if physics_aware and st.session_state.physics_interpolator is None:
                    st.session_state.physics_interpolator = PhysicsAwareInterpolator(st.session_state.interpolator)
                
                # Randomly select held-out indices
                np.random.seed(42)
                n_held = max(1, int(len(st.session_state.solutions)*held_out_frac))
                held_indices = np.random.choice(len(st.session_state.solutions), n_held, replace=False).tolist()
                
                results = {}
                for idx in held_indices:
                    gt = st.session_state.solutions[idx]
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
                        'source_params': [s['params'] for s in st.session_state.solutions if s != gt]
                    }
                    if physics_aware and st.session_state.physics_interpolator:
                        interp_res = st.session_state.physics_interpolator.interpolate_with_physics(
                            st.session_state.solutions, st.session_state.params_list, target, optimize=optimize_fields
                        )
                    else:
                        interp_res = st.session_state.interpolator(st.session_state.solutions, st.session_state.params_list, target['Ly'], target['c_cu'], target['c_ni'])
                    
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
                st.success(f"✅ Validation completed on {len(results)} held-out cases.")
    
    # === MAIN CONTENT ===
    if not st.session_state.solutions_loaded:
        st.info("👈 Use the sidebar to load solutions from `pinn_solutions/` and run validation.")
        st.markdown("""
        ### 📁 Expected File Format
        Place `.pkl` files in the `pinn_solutions/` directory with the following structure:
        ```python
        {
            'params': {'Ly': 60.0, 'C_Cu': 1.5e-3, 'C_Ni': 0.5e-3, 'Lx': 60.0, 't_max': 200.0},
            'X': np.ndarray (50,),  # x-coordinates
            'Y': np.ndarray (50,),  # y-coordinates
            'c1_preds': [np.ndarray (50,50), ...],  # Cu concentration at each time
            'c2_preds': [np.ndarray (50,50), ...],  # Ni concentration at each time
            'times': [t0, t1, ..., tN]  # time values
        }
        ```
        """)
        return
    
    if not st.session_state.validation_results:
        st.warning("⚠️ No validation results available. Click 'Run Validation' in the sidebar.")
        return
    
    results = st.session_state.validation_results
    
    # Summary metrics table
    st.subheader("📊 Validation Summary")
    summary_data = []
    for idx, res in results.items():
        m = res['metrics']
        summary_data.append({
            'Case': f"#{idx}",
            'Overall Score': m.overall_score,
            'MSE (Cu)': m.mse_c1,
            'MSE (Ni)': m.mse_c2,
            'PDE Residual': m.pde_residual_mean,
            'Mass Error': m.mass_error,
            'Weight Entropy': m.weight_entropy,
            'Param Distance': m.param_distance
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(
        summary_df.style.format({
            'Overall Score': '{:.3f}',
            'MSE (Cu)': '{:.2e}',
            'MSE (Ni)': '{:.2e}',
            'PDE Residual': '{:.2e}',
            'Mass Error': '{:.2%}',
            'Weight Entropy': '{:.3f}',
            'Param Distance': '{:.3f}'
        }).background_gradient(subset=['Overall Score'], cmap='Greens'),
        width='stretch'
    )
    
    # Interactive visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Metrics Charts", "🎯 Radar Analysis", "🔍 Field Comparison", "📉 Uncertainty Analysis"])
    
    with tab1:
        st.subheader("Validation Metrics by Category")
        # Aggregate metrics across all cases
        all_metrics_df = pd.concat([res['metrics'].to_dataframe() for res in results.values()])
        avg_metrics = all_metrics_df.groupby('Metric')['Value'].mean().reset_index()
        avg_metrics['Category'] = all_metrics_df.groupby('Metric')['Category'].first().values
        
        fig_bar = plot_metrics_bar_chart(avg_metrics)
        st.plotly_chart(fig_bar, use_container_width=True)  # Plotly's use_container_width is not deprecated
        # Actually plotly's use_container_width is fine, but we can change to config={'responsive': True}
        # For consistency, we keep as is; Streamlit deprecation only for its own widgets.
    
    with tab2:
        st.subheader("Radar Plot: Multi-Metric Validation")
        # Plot radar for first case as example
        first_metrics = list(results.values())[0]['metrics']
        fig_radar = plot_radar_chart(first_metrics)
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with tab3:
        st.subheader("Field Comparison: Interpolated vs Ground Truth")
        case_idx = st.selectbox("Select Validation Case", list(results.keys()), format_func=lambda x: f"Case #{x}")
        res = results[case_idx]
        
        fig_c1, fig_c2 = plot_comparison_heatmaps(
            res['interp_c1'], res['gt_c1'],
            res['interp_c2'], res['gt_c2'],
            res['x'], res['y'],
            title_prefix=f"Case #{case_idx}"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_c1, use_container_width=True)
        with col2:
            st.plotly_chart(fig_c2, use_container_width=True)
        
        # PDE residual heatmap
        st.subheader("PDE Residual Field")
        residual = compute_pde_residual(
            res['interp_c1'], res['interp_c2'], res['x'], res['y'], 
            res['params'].get('t_max', 200),
            PHYSICS_CONSTANTS['D11'], PHYSICS_CONSTANTS['D12'],
            PHYSICS_CONSTANTS['D21'], PHYSICS_CONSTANTS['D22'],
            res['x'][1]-res['x'][0], res['y'][1]-res['y'][0]
        )
        fig_residual = plot_residual_heatmap(residual, res['x'], res['y'])
        st.plotly_chart(fig_residual, use_container_width=True)
    
    with tab4:
        st.subheader("Uncertainty Quantification")
        
        # Scatter: uncertainty vs parameter distance
        metrics_list = [res['metrics'] for res in results.values()]
        fig_scatter = plot_uncertainty_scatter(metrics_list, ['Ly', 'c_cu', 'c_ni'])
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Key uncertainty insights
        st.markdown("### 🔑 Key Insights")
        avg_entropy = np.mean([m.weight_entropy for m in metrics_list])
        avg_distance = np.mean([m.param_distance for m in metrics_list])
        avg_score = np.mean([m.overall_score for m in metrics_list])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg. Weight Entropy", f"{avg_entropy:.3f}", 
                     help="Higher = more uncertain source weighting")
        with col2:
            st.metric("Avg. Parameter Distance", f"{avg_distance:.3f}",
                     help="Lower = target closer to training sources")
        with col3:
            st.metric("Avg. Validation Score", f"{avg_score:.3f} / 1.0",
                     help="Higher = better agreement with held-out PINN")
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        if avg_distance > 0.3:
            st.warning("⚠️ Target parameters are far from training data. Consider adding nearby simulations.")
        if avg_entropy > 1.5:
            st.warning("⚠️ High weight entropy indicates ambiguous source selection. Review parameter space coverage.")
        if avg_score < 0.7:
            st.error("❌ Low validation score. Enable physics-aware refinement or collect more training data.")
        if all(m.mass_error < 0.01 for m in metrics_list):
            st.success("✅ Mass conservation satisfied (<1% error) across all cases.")
    
    # Export results
    with st.expander("💾 Export Validation Report"):
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'held_out_fraction': held_out_frac,
                'physics_aware': physics_aware,
                'optimize_fields': optimize_fields
            },
            'summary': summary_df.to_dict('records'),
            'detailed_metrics': {idx: res['metrics'].to_dict() for idx, res in results.items()}
        }
        
        json_str = json.dumps(report_data, indent=2, default=str)
        st.download_button(
            "📥 Download JSON Report",
            json_str,
            f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json"
        )
        
        # CSV export of metrics
        csv_str = pd.concat([res['metrics'].to_dataframe() for res in results.values()]).to_csv(index=False)
        st.download_button(
            "📥 Download Metrics CSV",
            csv_str,
            f"validation_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )


if __name__ == "__main__":
    main()
