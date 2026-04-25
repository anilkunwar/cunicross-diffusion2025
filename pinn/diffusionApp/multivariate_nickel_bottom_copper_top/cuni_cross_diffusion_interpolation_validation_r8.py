#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UNIFIED CU-NI INTERDIFFUSION VISUALIZER WITH VALIDATION & UNCERTAINTY
=====================================================================
PRODUCTION-READY VERSION: All syntax errors fixed + robustness improvements
- ✅ Fixed incomplete 'if' statements in EnhancedSolutionLoader
- ✅ Fixed SavedValidationRun.from_dict signature (no type shadowing)
- ✅ Added indexing='ij' to torch.meshgrid in _physics_refinement
- ✅ Added empty-state handling in plot_multi_case_bar_chart
- ✅ Defensive checks for missing target_params keys
- ✅ Retains all customization + metric selection features
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
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging
import colorsys
import glob as glob_module

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================
# COLOR UTILITIES - FIX FOR STREAMLIT COLOR_PICKER
# =============================================

CSS_COLOR_TO_HEX = {
    'black': '#000000', 'white': '#FFFFFF', 'red': '#FF0000', 'green': '#008000',
    'blue': '#0000FF', 'yellow': '#FFFF00', 'cyan': '#00FFFF', 'magenta': '#FF00FF',
    'gray': '#808080', 'grey': '#808080', 'silver': '#C0C0C0', 'maroon': '#800000',
    'olive': '#808000', 'lime': '#00FF00', 'aqua': '#00FFFF', 'teal': '#008080',
    'navy': '#000080', 'fuchsia': '#FF00FF', 'purple': '#800080', 'orange': '#FFA500',
    'aliceblue': '#F0F8FF', 'antiquewhite': '#FAEBD7', 'aquamarine': '#7FFFD4',
    'azure': '#F0FFFF', 'beige': '#F5F5DC', 'bisque': '#FFE4C4', 'blanchedalmond': '#FFEBCD',
    'blueviolet': '#8A2BE2', 'brown': '#A52A2A', 'burlywood': '#DEB887', 'cadetblue': '#5F9EA0',
    'chartreuse': '#7FFF00', 'chocolate': '#D2691E', 'coral': '#FF7F50', 'cornflowerblue': '#6495ED',
    'cornsilk': '#FFF8DC', 'crimson': '#DC143C', 'darkblue': '#00008B', 'darkcyan': '#008B8B',
    'darkgoldenrod': '#B8860B', 'darkgray': '#A9A9A9', 'darkgrey': '#A9A9A9', 'darkgreen': '#006400',
    'darkkhaki': '#BDB76B', 'darkmagenta': '#8B008B', 'darkolivegreen': '#556B2F',
    'darkorange': '#FF8C00', 'darkorchid': '#9932CC', 'darkred': '#8B0000', 'darksalmon': '#E9967A',
    'darkseagreen': '#8FBC8F', 'darkslateblue': '#483D8B', 'darkslategray': '#2F4F4F',
    'darkslategrey': '#2F4F4F', 'darkturquoise': '#00CED1', 'darkviolet': '#9400D3',
    'deeppink': '#FF1493', 'deepskyblue': '#00BFFF', 'dimgray': '#696969', 'dimgrey': '#696969',
    'dodgerblue': '#1E90FF', 'firebrick': '#B22222', 'floralwhite': '#FFFAF0',
    'forestgreen': '#228B22', 'gainsboro': '#DCDCDC', 'ghostwhite': '#F8F8FF',
    'gold': '#FFD700', 'goldenrod': '#DAA520', 'greenyellow': '#ADFF2F', 'honeydew': '#F0FFF0',
    'hotpink': '#FF69B4', 'indianred': '#CD5C5C', 'indigo': '#4B0082', 'ivory': '#FFFFF0',
    'khaki': '#F0E68C', 'lavender': '#E6E6FA', 'lavenderblush': '#FFF0F5', 'lawngreen': '#7CFC00',
    'lemonchiffon': '#FFFACD', 'lightblue': '#ADD8E6', 'lightcoral': '#F08080',
    'lightcyan': '#E0FFFF', 'lightgoldenrodyellow': '#FAFAD2', 'lightgray': '#D3D3D3',
    'lightgrey': '#D3D3D3', 'lightgreen': '#90EE90', 'lightpink': '#FFB6C1',
    'lightsalmon': '#FFA07A', 'lightseagreen': '#20B2AA', 'lightskyblue': '#87CEFA',
    'lightslategray': '#778899', 'lightslategrey': '#778899', 'lightsteelblue': '#B0C4DE',
    'lightyellow': '#FFFFE0', 'limegreen': '#32CD32', 'linen': '#FAF0E6',
    'mediumaquamarine': '#66CDAA', 'mediumblue': '#0000CD', 'mediumorchid': '#BA55D3',
    'mediumpurple': '#9370DB', 'mediumseagreen': '#3CB371', 'mediumslateblue': '#7B68EE',
    'mediumspringgreen': '#00FA9A', 'mediumturquoise': '#48D1CC', 'mediumvioletred': '#C71585',
    'midnightblue': '#191970', 'mintcream': '#F5FFFA', 'mistyrose': '#FFE4E1',
    'moccasin': '#FFE4B5', 'navajowhite': '#FFDEAD', 'oldlace': '#FDF5E6',
    'olivedrab': '#6B8E23', 'orangered': '#FF4500', 'orchid': '#DA70D6',
    'palegoldenrod': '#EEE8AA', 'palegreen': '#98FB98', 'paleturquoise': '#AFEEEE',
    'palevioletred': '#DB7093', 'papayawhip': '#FFEFD5', 'peachpuff': '#FFDAB9',
    'peru': '#CD853F', 'pink': '#FFC0CB', 'plum': '#DDA0DD', 'powderblue': '#B0E0E6',
    'rosybrown': '#BC8F8F', 'royalblue': '#4169E1', 'saddlebrown': '#8B4513',
    'salmon': '#FA8072', 'sandybrown': '#F4A460', 'seagreen': '#2E8B57',
    'seashell': '#FFF5EE', 'sienna': '#A0522D', 'skyblue': '#87CEEB', 'slateblue': '#6A5ACD',
    'slategray': '#708090', 'slategrey': '#708090', 'snow': '#FFFAFA', 'springgreen': '#00FF7F',
    'steelblue': '#4682B4', 'tan': '#D2B48C', 'thistle': '#D8BFD8', 'tomato': '#FF6347',
    'turquoise': '#40E0D0', 'violet': '#EE82EE', 'wheat': '#F5DEB3', 'whitesmoke': '#F5F5F5',
    'yellowgreen': '#9ACD32', 'rebeccapurple': '#663399'
}


def validate_hex_color(color_value: str, default: str = '#D3D3D3') -> str:
    if not isinstance(color_value, str):
        logger.warning(f"Color value is not a string: {color_value}, using default {default}")
        return default
    
    color_value = color_value.strip()
    
    if re.match(r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{8})$', color_value):
        return color_value.upper()
    
    color_lower = color_value.lower()
    if color_lower in CSS_COLOR_TO_HEX:
        hex_color = CSS_COLOR_TO_HEX[color_lower]
        logger.info(f"Converted CSS color '{color_value}' to hex '{hex_color}'")
        return hex_color
    
    try:
        import matplotlib.colors as mcolors
        if color_lower in mcolors.CSS4_COLORS:
            hex_color = mcolors.CSS4_COLORS[color_lower]
            if not hex_color.startswith('#'):
                hex_color = '#' + hex_color
            if len(hex_color) == 4:
                hex_color = '#' + ''.join([c*2 for c in hex_color[1:]])
            logger.info(f"Converted matplotlib color '{color_value}' to hex '{hex_color}'")
            return hex_color.upper()
    except ImportError:
        pass
    
    try:
        rgb_match = re.match(r'rgba?\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*[\d.]+)?\s*\)', color_value.lower())
        if rgb_match:
            r, g, b = [int(x) for x in rgb_match.groups()[:3]]
            hex_color = '#{:02X}{:02X}{:02X}'.format(min(255, max(0, r)), 
                                                      min(255, max(0, g)), 
                                                      min(255, max(0, b)))
            logger.info(f"Converted RGB string '{color_value}' to hex '{hex_color}'")
            return hex_color
    except Exception as e:
        logger.warning(f"Failed to parse RGB string '{color_value}': {e}")
    
    logger.warning(f"Could not convert color '{color_value}' to hex format, using default '{default}'")
    return default


def safe_color_picker(label: str, key: Optional[str] = None, 
                     default: str = '#D3D3D3', help: Optional[str] = None) -> str:
    validated_default = validate_hex_color(default)
    
    current_value = None
    if key and key in st.session_state:
        current_value = st.session_state[key]
        if current_value:
            current_value = validate_hex_color(current_value, validated_default)
    
    try:
        result = st.color_picker(
            label=label,
            value=current_value if current_value else validated_default,
            key=key,
            help=help
        )
        return validate_hex_color(result, validated_default)
    except Exception as e:
        logger.error(f"Error in color_picker '{label}': {e}")
        st.warning(f"⚠️ Color picker error: {str(e)[:100]}... Using default color.")
        return validated_default


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

PHYSICS_CONSTANTS = {
    'D11': 0.006,
    'D12': 0.00427,
    'D21': 0.003697,
    'D22': 0.0054,
    'C_CU_RANGE': (0.0, 2.9e-3),
    'C_NI_RANGE': (0.0, 1.8e-3),
    'LY_RANGE': (30.0, 120.0),
    'T_MAX': 200.0,
    'MASS_TOLERANCE': 1e-4,
}

SOLUTION_DIR = os.path.join(os.path.dirname(__file__), "pinn_solutions")
os.makedirs(SOLUTION_DIR, exist_ok=True)
os.makedirs("figures", exist_ok=True)
SAVED_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "saved_validation_results")
os.makedirs(SAVED_RESULTS_DIR, exist_ok=True)

COLORMAPS = [
    "viridis", "plasma", "inferno", "magma", "cividis",
    "Greys", "Purples", "Blues", "Greens", "Oranges", "Reds",
    "YlOrBr", "YlOrRd", "OrRd", "PuRd", "RdPu", "BuPu", "GnBu",
    "PuBu", "YlGnBu", "PuBuGn", "BuGn", "YlGn",
    "coolwarm", "bwr", "seismic", "RdBu", "Spectral",
    "tab10", "Set1", "Set2", "Pastel1"
]

# =============================================
# METRIC SELECTION CONFIGURATION - USER CUSTOMIZABLE ASPECTS
# =============================================
ALL_VISUAL_METRICS = {
    "MSE (Cu)": "mse_c1",
    "MSE (Ni)": "mse_c2",
    "MAE (Cu)": "mae_c1",
    "MAE (Ni)": "mae_c2",
    "Max Error (Cu)": "max_error_c1",
    "Max Error (Ni)": "max_error_c2",
    "R² (Cu)": "r2_c1",
    "R² (Ni)": "r2_c2",
    "SSIM (Cu)": "ssim_c1",
    "SSIM (Ni)": "ssim_c2",
    "PDE Residual (Mean)": "pde_residual_mean",
    "PDE Residual (Max)": "pde_residual_max",
    "BC Error Top (Cu)": "bc_error_top_c1",
    "BC Error Top (Ni)": "bc_error_top_c2",
    "BC Error Bottom (Cu)": "bc_error_bottom_c1",
    "BC Error Bottom (Ni)": "bc_error_bottom_c2",
    "Mass Conservation": "mass_error",
    "Weight Entropy": "weight_entropy",
    "Param. Distance": "param_distance",
    "Ensemble Var (Cu)": "ensemble_variance_c1",
    "Ensemble Var (Ni)": "ensemble_variance_c2",
    "Overall Score": "overall_score"
}

# Default metrics shown in radar charts (curated subset for clarity)
DEFAULT_RADAR_METRICS = [
    "MSE (Cu)", "MSE (Ni)", "R² (Cu)", "R² (Ni)", 
    "SSIM (Cu)", "SSIM (Ni)", "PDE Residual (Mean)", 
    "Mass Conservation", "Param. Distance"
]

# Default metrics shown in bar charts (all pointwise + composite)
DEFAULT_BAR_METRICS = [
    "MSE (Cu)", "MSE (Ni)", "MAE (Cu)", "MAE (Ni)",
    "R² (Cu)", "R² (Ni)", "SSIM (Cu)", "SSIM (Ni)",
    "PDE Residual (Mean)", "Mass Conservation", 
    "Param. Distance", "Overall Score"
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
# 3. NATURAL LANGUAGE PARSER
# =============================================
class DiffusionNLParser:
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
                        if key in ['ly_target']:
                            params[key] = float(val)
                        elif key in ['c_cu_target', 'c_ni_target']:
                            params[key] = float(val)
                        elif key in ['num_heads', 'd_head', 'seed']:
                            params[key] = int(float(val))
                        elif key == 'sigma':
                            params[key] = float(val)
                    except (ValueError, TypeError):
                        pass
                    break
        
        params['ly_target'] = np.clip(params['ly_target'], 30.0, 120.0)
        params['c_cu_target'] = np.clip(params['c_cu_target'], 0.0, 2.9e-3)
        params['c_ni_target'] = np.clip(params['c_ni_target'], 0.0, 1.8e-3)
        params['sigma'] = np.clip(params['sigma'], 0.05, 0.5)
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
                res['ly_target'] = np.clip(float(res.get('ly_target', 60)), 30, 120)
                res['c_cu_target'] = np.clip(float(res.get('c_cu_target', 1.5e-3)), 0, 2.9e-3)
                res['c_ni_target'] = np.clip(float(res.get('c_ni_target', 0.5e-3)), 0, 1.8e-3)
                res['sigma'] = np.clip(float(res.get('sigma', 0.2)), 0.05, 0.5)
                if regex_params:
                    for k in ['ly_target', 'c_cu_target', 'c_ni_target']:
                        if k in regex_params and abs(res[k] - regex_params[k]) > 1e-4:
                            res[k] = regex_params[k]
                return res
        except Exception as e:
            st.warning(f"LLM parsing failed: {e}. Falling back to regex.")
        return self.parse_regex(text)

    def hybrid_parse(self, text: str, tokenizer, model, use_llm: bool = True) -> Dict:
        regex_params = self.parse_regex(text)
        if use_llm and tokenizer and model:
            cache_key = hashlib.md5((text + st.session_state.get('llm_backend_loaded', '')).encode()).hexdigest()
            if 'llm_cache' not in st.session_state:
                st.session_state.llm_cache = OrderedDict()
            if cache_key in st.session_state.llm_cache:
                llm_res = st.session_state.llm_cache[cache_key]
            else:
                llm_res = self.parse_with_llm(text, tokenizer, model, regex_params)
                if len(st.session_state.llm_cache) > 20:
                    st.session_state.llm_cache.popitem(last=False)
                st.session_state.llm_cache[cache_key] = llm_res
            
            final = self.defaults.copy()
            for k in final:
                if llm_res[k] != self.defaults[k]:
                    final[k] = llm_res[k]
                elif regex_params[k] != self.defaults[k]:
                    final[k] = regex_params[k]
            return final
        return regex_params

    def get_explanation(self, params: dict, original_text: str) -> str:
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
# 5. ENHANCED SOLUTION LOADER
# =============================================
class EnhancedSolutionLoader:
    def __init__(self, solutions_dir: str = SOLUTION_DIR):
        self.solutions_dir = solutions_dir
        self._ensure_directory()
        self.cache = {}
    
    def _ensure_directory(self):
        os.makedirs(self.solutions_dir, exist_ok=True)
    
    def scan_solutions(self) -> List[Dict[str, Any]]:
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
        params = {}
        
        ly_match = re.search(r'Ly[_-]?([0-9.]+)', filename, re.IGNORECASE)
        if ly_match:
            params['Ly'] = float(ly_match.group(1))
        
        ccu_match = re.search(r'C_Cu[_-]?([0-9.eE+-]+)', filename, re.IGNORECASE)
        if ccu_match:
            params['C_Cu'] = float(ccu_match.group(1))
        
        cni_match = re.search(r'C_Ni[_-]?([0-9.eE+-]+)', filename, re.IGNORECASE)
        if cni_match:
            params['C_Ni'] = float(cni_match.group(1))
        
        lx_match = re.search(r'Lx[_-]?([0-9.]+)', filename, re.IGNORECASE)
        if lx_match:
            params['Lx'] = float(lx_match.group(1))
        
        tmax_match = re.search(r't[_-]?max[_-]?([0-9.]+)', filename, re.IGNORECASE)
        if tmax_match:
            params['t_max'] = float(tmax_match.group(1))
        
        return params
    
    def _ensure_2d(self, arr):
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
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
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
                if 'params' in data and isinstance(data['params'], dict):
                    standardized['params'].update(data['params'])
                if 'parameters' in data and isinstance(data['parameters'], dict):
                    standardized['params'].update(data['parameters'])
                
                # ✅ FIXED: Complete 'if' statements with 'data' operand
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
                
                if not standardized['params']:
                    parsed = self.parse_filename(os.path.basename(file_path))
                    standardized['params'].update(parsed)
                    st.sidebar.info(f"Parsed parameters from filename: {os.path.basename(file_path)}")
                
                params = standardized['params']
                params.setdefault('Ly', 60.0)
                params.setdefault('C_Cu', 1.5e-3)
                params.setdefault('C_Ni', 0.5e-3)
                params.setdefault('Lx', 60.0)
                params.setdefault('t_max', 200.0)
                
                if not standardized['c1_preds'] or not standardized['c2_preds']:
                    st.sidebar.warning(f"No concentration fields in {os.path.basename(file_path)}")
                    return None
                
                self._convert_tensors(standardized)
                
                return standardized
            else:
                st.sidebar.warning(f"Unexpected data format in {os.path.basename(file_path)}")
                return None
                
        except Exception as e:
            st.sidebar.error(f"Error loading {os.path.basename(file_path)}: {e}")
            return None
    
    def load_all_solutions(self, use_cache: bool = True, max_files: Optional[int] = None) -> Tuple[List[Dict], List[str]]:
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
            
            if use_cache and cache_key in self.cache:
                solutions.append(self.cache[cache_key])
                load_logs.append(f"✓ {item['filename']} (from cache)")
                continue
            
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
# 6. ATTENTION-BASED INTERPOLATOR
# =============================================
class MultiParamAttentionInterpolator(nn.Module):
    def __init__(self, sigma=0.2, num_heads=4, d_head=8):
        super().__init__()
        self.sigma = sigma
        self.num_heads = num_heads
        self.d_head = d_head
        self.W_q = nn.Linear(3, self.num_heads * self.d_head, bias=False)
        self.W_k = nn.Linear(3, self.num_heads * self.d_head, bias=False)

    def normalize_params(self, params: Union[Tuple, List, np.ndarray], is_target: bool = False) -> np.ndarray:
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
        if not params_list:
            raise ValueError("Empty parameter list provided for interpolation.")
            
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

    def forward(self, solutions, params_list, ly_target, c_cu_target, c_ni_target):
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
    ny, nx = c1.shape
    
    def laplacian(c):
        lap = np.zeros_like(c)
        lap[1:-1, 1:-1] = (
            (c[2:, 1:-1] - 2*c[1:-1, 1:-1] + c[:-2, 1:-1]) / dy**2 +
            (c[1:-1, 2:] - 2*c[1:-1, 1:-1] + c[1:-1, :-2]) / dx**2
        )
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
    
    residual1 = -(D11 * lap_c1 + D12 * lap_c2)
    residual2 = -(D21 * lap_c1 + D22 * lap_c2)
    
    return np.sqrt(residual1**2 + residual2**2)


def enforce_boundary_conditions(c1: np.ndarray, c2: np.ndarray,
                               x: np.ndarray, y: np.ndarray,
                               c_cu_top: float, c_cu_bottom: float,
                               c_ni_top: float, c_ni_bottom: float,
                               enforce_type: str = 'hard') -> Tuple[np.ndarray, np.ndarray]:
    c1_bc = c1.copy()
    c2_bc = c2.copy()
    ny, nx = c1.shape
    
    if enforce_type == 'hard':
        c1_bc[:, -1] = c_cu_top
        c2_bc[:, -1] = c_ni_top
        c1_bc[:, 0] = c_cu_bottom
        c2_bc[:, 0] = c_ni_bottom
    elif enforce_type == 'soft':
        boundary_width = 3
        y_weight = np.exp(-((np.arange(ny) - ny/2)**2) / (2 * (ny/4)**2))
        x_weight = np.exp(-((np.arange(nx) - nx/2)**2) / (2 * (nx/4)**2))
        X, Y = np.meshgrid(x_weight, y_weight)
        blend = 1 - X * Y
        
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
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    dy = y[1] - y[0] if len(y) > 1 else 1.0
    area = dx * dy
    
    mass_c1 = np.sum(c1) * area
    mass_c2 = np.sum(c2) * area
    
    nx, ny = c1.shape
    initial_mass_c1 = c_cu_initial * nx * ny * area
    initial_mass_c2 = c_ni_initial * nx * ny * area
    
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
    metrics = ValidationMetrics()
    
    flat_interp_c1 = interp_c1.flatten()
    flat_interp_c2 = interp_c2.flatten()
    flat_gt_c1 = gt_c1.flatten()
    flat_gt_c2 = gt_c2.flatten()
    
    metrics.mse_c1 = mean_squared_error(flat_gt_c1, flat_interp_c1)
    metrics.mse_c2 = mean_squared_error(flat_gt_c2, flat_interp_c2)
    metrics.mae_c1 = mean_absolute_error(flat_gt_c1, flat_interp_c1)
    metrics.mae_c2 = mean_absolute_error(flat_gt_c2, flat_interp_c2)
    metrics.max_error_c1 = np.max(np.abs(flat_gt_c1 - flat_interp_c1))
    metrics.max_error_c2 = np.max(np.abs(flat_gt_c2 - flat_interp_c2))
    
    if np.var(flat_gt_c1) > 1e-12:
        metrics.r2_c1 = r2_score(flat_gt_c1, flat_interp_c1)
    if np.var(flat_gt_c2) > 1e-12:
        metrics.r2_c2 = r2_score(flat_gt_c2, flat_interp_c2)
    
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
    
    ny, nx = interp_c1.shape
    metrics.bc_error_top_c1 = np.abs(np.mean(interp_c1[:, -1]) - params.get('c_cu_top', 0))
    metrics.bc_error_top_c2 = np.abs(np.mean(interp_c2[:, -1]) - params.get('c_ni_top', 0))
    metrics.bc_error_bottom_c1 = np.abs(np.mean(interp_c1[:, 0]) - params.get('c_cu_bottom', 0))
    metrics.bc_error_bottom_c2 = np.abs(np.mean(interp_c2[:, 0]) - params.get('c_ni_bottom', 0))
    
    mass_metrics = compute_mass_conservation(
        interp_c1, interp_c2, x, y,
        params.get('c_cu_initial', 1.5e-3),
        params.get('c_ni_initial', 4.0e-4)
    )
    metrics.mass_error = mass_metrics['mass_error_max']
    
    if weights is not None and len(weights) > 0:
        eps = 1e-10
        weights_arr = np.asarray(weights, dtype=np.float64).flatten()
        weights_arr = np.clip(weights_arr, eps, 1.0)
        weights_arr = weights_arr / (np.sum(weights_arr) + eps)
        metrics.weight_entropy = float(-np.sum(weights_arr * np.log(weights_arr + eps)))
    
    if 'target_params' in params and 'source_params' in params:
        target = params['target_params']
        sources = params['source_params']
        if sources and len(sources) > 0:
            distances = []
            for src in sources:
                d_ly = abs(target.get('Ly', 60) - src.get('Ly', 60)) / 90
                d_cu = abs(target.get('C_Cu', 1.5e-3) - src.get('C_Cu', 1.5e-3)) / 2.9e-3
                d_ni = abs(target.get('C_Ni', 4e-4) - src.get('C_Ni', 4e-4)) / 1.8e-3
                distances.append(np.sqrt(d_ly**2 + d_cu**2 + d_ni**2))
            metrics.param_distance = float(min(distances)) if distances else 1.0
    
    if ensemble_fields and len(ensemble_fields) > 1:
        c1_ensemble = np.stack([f['c1'] for f in ensemble_fields], axis=0)
        c2_ensemble = np.stack([f['c2'] for f in ensemble_fields], axis=0)
        metrics.ensemble_variance_c1 = np.mean(np.var(c1_ensemble, axis=0))
        metrics.ensemble_variance_c2 = np.mean(np.var(c2_ensemble, axis=0))
    
    scores = []
    scores.append(np.exp(-metrics.mse_c1 / 1e-6))
    scores.append(np.exp(-metrics.mse_c2 / 1e-6))
    scores.append(metrics.r2_c1 if metrics.r2_c1 > 0 else 0)
    scores.append(metrics.r2_c2 if metrics.r2_c2 > 0 else 0)
    scores.append(metrics.ssim_c1)
    scores.append(metrics.ssim_c2)
    scores.append(np.exp(-metrics.pde_residual_mean / 1e-4))
    scores.append(1 - metrics.mass_error)
    scores.append(np.exp(-metrics.param_distance / 0.3))
    
    metrics.overall_score = float(np.mean(scores))
    
    return metrics


# =============================================
# 9. RESULT PERSISTENCE UTILITIES - FIXED
# =============================================
@dataclass
class SavedValidationRun:
    run_id: str
    timestamp: str
    config: Dict
    case_results: Dict[str, Dict]
    summary_metrics: List[Dict]
    
    def to_dict(self) -> Dict:
        return {
            'run_id': self.run_id,
            'timestamp': self.timestamp,
            'config': self.config,
            'case_results': self.case_results,
            'summary_metrics': self.summary_metrics
        }
    
    # ✅ FIXED: Proper method signature (no type shadowing)
    @classmethod
    def from_dict(cls, data: dict) -> 'SavedValidationRun':
        return cls(
            run_id=data['run_id'],
            timestamp=data['timestamp'],
            config=data['config'],
            case_results=data['case_results'],
            summary_metrics=data['summary_metrics']
        )
    
    def save_to_disk(self, directory: str = SAVED_RESULTS_DIR):
        filepath = os.path.join(directory, f"run_{self.run_id}.json")
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return filepath
    
    @classmethod
    def load_from_disk(cls, run_id: str, directory: str = SAVED_RESULTS_DIR) -> 'SavedValidationRun':
        filepath = os.path.join(directory, f"run_{run_id}.json")
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @staticmethod
    def list_saved_runs(directory: str = SAVED_RESULTS_DIR) -> List[str]:
        pattern = os.path.join(directory, "run_*.json")
        files = glob_module.glob(pattern)
        run_ids = []
        for f in files:
            match = re.search(r'run_(.+)\.json', os.path.basename(f))
            if match:
                run_ids.append(match.group(1))
        return sorted(run_ids, reverse=True)
    
    def get_case_label(self, case_idx: str) -> str:
        # ✅ Defensive: ensure case_idx is string for dict lookup
        case_idx_str = str(case_idx)
        if case_idx_str in self.case_results:
            case_data = self.case_results[case_idx_str]
            target_params = case_data.get('target_params', {})
            ly = target_params.get('Ly', target_params.get('ly_target', 60.0))
            c_cu = target_params.get('C_Cu', target_params.get('c_cu_target', target_params.get('c_cu', 1.5e-3)))
            c_ni = target_params.get('C_Ni', target_params.get('c_ni_target', target_params.get('c_ni', 0.5e-3)))
            return f"Case #{case_idx_str} (Ly={ly:.1f}μm, Cu={c_cu:.1e}, Ni={c_ni:.1e})"
        return f"Case #{case_idx_str}"

def generate_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:6]
    return f"{timestamp}_{random_suffix}"


def format_case_label(case_idx: str, target_params: Dict) -> str:
    ly = target_params.get('Ly', target_params.get('ly_target', 60.0))
    c_cu = target_params.get('C_Cu', target_params.get('c_cu_target', target_params.get('c_cu', 1.5e-3)))
    c_ni = target_params.get('C_Ni', target_params.get('c_ni_target', target_params.get('c_ni', 0.5e-3)))
    return f"Case #{case_idx} (Ly={ly:.1f}μm, Cu={c_cu:.1e}, Ni={c_ni:.1e})"


# =============================================
# 10. ENHANCED VISUALIZATION FUNCTIONS WITH METRIC SELECTION
# =============================================

def apply_plot_customization(fig: go.Figure, cust: Dict, chart_type: str = 'bar') -> go.Figure:
    fig.update_layout(
        width=cust.get('figure_width', 800),
        height=cust.get('figure_height', 500),
        font=dict(size=cust.get('font_size', 12)),
        title_font=dict(size=cust.get('title_font_size', cust.get('font_size', 12) + 2)),
        legend=dict(
            font=dict(size=cust.get('legend_font_size', cust.get('font_size', 12))),
            orientation=cust.get('legend_orientation', 'v'),
            x=1.02,
            y=1.0,
            xanchor='left',
            yanchor='top'
        )
    )
    
    axis_style = dict(
        title_font=dict(size=cust.get('axis_title_font_size', cust.get('font_size', 12) + 1)),
        tickfont=dict(size=cust.get('tick_font_size', cust.get('font_size', 12) - 1)),
        linewidth=cust.get('axis_line_width', 1.5),
        gridwidth=cust.get('grid_line_width', 1.0),
        gridcolor=cust.get('grid_color', '#D3D3D3'),
        showgrid=cust.get('show_grid', True)
    )
    
    if chart_type == 'bar':
        fig.update_xaxes(**axis_style)
        fig.update_yaxes(**axis_style)
        if cust.get('x_tick_angle', 0) != 0:
            fig.update_xaxes(tickangle=cust.get('x_tick_angle', 0))
    elif chart_type == 'radar':
        fig.update_polars(
            radialaxis=dict(
                title_font=dict(size=cust.get('axis_title_font_size', cust.get('font_size', 12) + 1)),
                tickfont=dict(size=cust.get('tick_font_size', cust.get('font_size', 12) - 1)),
                linewidth=cust.get('axis_line_width', 1.5),
                gridwidth=cust.get('grid_line_width', 1.0),
                gridcolor=cust.get('grid_color', '#D3D3D3'),
                showgrid=cust.get('show_grid', True)
            ),
            angularaxis=dict(
                tickfont=dict(size=cust.get('tick_font_size', cust.get('font_size', 12) - 1)),
                linewidth=cust.get('axis_line_width', 1.5),
                gridwidth=cust.get('grid_line_width', 1.0),
                gridcolor=cust.get('grid_color', '#D3D3D3'),
                showgrid=cust.get('show_grid', True)
            )
        )
    
    if cust.get('x_title'):
        if chart_type == 'bar':
            fig.update_xaxes(title_text=cust['x_title'])
        elif chart_type == 'radar':
            fig.update_polars(radialaxis=dict(title_text=cust['x_title']))
    if cust.get('y_title') and chart_type == 'bar':
        fig.update_yaxes(title_text=cust['y_title'])
    
    return fig


def _normalize_metric_value(metric_name: str, value: float, normalization_scales: Dict) -> float:
    """Helper function to normalize metric values for visualization."""
    if metric_name in ['mse_c1', 'mse_c2']:
        scale = normalization_scales.get('mse', 1e-6)
        return max(0.0, min(1.0, np.exp(-value / scale)))
    elif metric_name in ['mae_c1', 'mae_c2']:
        scale = normalization_scales.get('mae', 1e-4)
        return max(0.0, min(1.0, np.exp(-value / scale)))
    elif metric_name in ['max_error_c1', 'max_error_c2']:
        scale = normalization_scales.get('max_error', 1e-4)
        return max(0.0, min(1.0, np.exp(-value / scale)))
    elif metric_name in ['pde_residual_mean', 'pde_residual_max']:
        scale = normalization_scales.get('pde_residual_mean', 1e-4)
        return max(0.0, min(1.0, np.exp(-value / scale)))
    elif metric_name.startswith('bc_error'):
        scale = normalization_scales.get('bc_error', 1e-4)
        return max(0.0, min(1.0, np.exp(-value / scale)))
    elif metric_name == 'mass_error':
        return max(0.0, min(1.0, 1.0 - value))
    elif metric_name == 'param_distance':
        scale = normalization_scales.get('param_distance', 0.3)
        return max(0.0, min(1.0, np.exp(-value / scale)))
    elif metric_name in ['r2_c1', 'r2_c2', 'ssim_c1', 'ssim_c2', 'overall_score']:
        return max(0.0, min(1.0, value))
    elif metric_name in ['weight_entropy', 'ensemble_variance_c1', 'ensemble_variance_c2']:
        return max(0.0, min(1.0, np.exp(-value)))
    else:
        return value


def plot_metrics_bar_chart(metrics_df: pd.DataFrame, 
                          title: str = "Validation Metrics",
                          color_map: Optional[Dict] = None,
                          customization: Optional[Dict] = None,
                          normalization_scales: Optional[Dict] = None,
                          run_label: str = "Current Run",
                          selected_metrics: Optional[List[str]] = None) -> go.Figure:
    if color_map is None:
        color_map = {
            'pointwise': '#2E86AB',
            'physics': '#A23B72',
            'uncertainty': '#F18F01',
            'composite': '#C73E1D'
        }
    if customization is None:
        customization = {}
    if normalization_scales is None:
        normalization_scales = {
            'mse': 1e-6,
            'mae': 1e-4,
            'max_error': 1e-4,
            'pde_residual_mean': 1e-4,
            'pde_residual_max': 1e-4,
            'bc_error': 1e-4,
            'mass_error': 1.0,
            'param_distance': 0.3
        }
    
    if selected_metrics is None:
        selected_metrics = list(ALL_VISUAL_METRICS.keys())
    
    df = metrics_df.copy()
    
    filtered_rows = []
    for idx, row in df.iterrows():
        metric_name = row['Metric']
        display_name = None
        for disp_name, attr_name in ALL_VISUAL_METRICS.items():
            if attr_name == metric_name and disp_name in selected_metrics:
                display_name = disp_name
                break
        if display_name is None:
            continue
        
        val = row['Value']
        normalized_val = _normalize_metric_value(metric_name, val, normalization_scales)
        filtered_rows.append({
            'Metric': display_name,
            'Value': normalized_val,
            'Category': row['Category'],
            'InternalName': metric_name
        })
    
    if not filtered_rows:
        filtered_rows = []
        for idx, row in df.iterrows():
            metric_name = row['Metric']
            display_name = None
            for disp_name, attr_name in ALL_VISUAL_METRICS.items():
                if attr_name == metric_name:
                    display_name = disp_name
                    break
            if display_name is None:
                display_name = metric_name
            val = row['Value']
            normalized_val = _normalize_metric_value(metric_name, val, normalization_scales)
            filtered_rows.append({
                'Metric': display_name,
                'Value': normalized_val,
                'Category': row['Category'],
                'InternalName': metric_name
            })
    
    df_filtered = pd.DataFrame(filtered_rows)
    
    fig = go.Figure()
    
    for category in df_filtered['Category'].unique():
        cat_df = df_filtered[df_filtered['Category'] == category]
        show_text = customization.get('show_bar_labels', True)
        text_vals = cat_df['Value'].apply(lambda x: f'{x:.3f}') if show_text else None
        fig.add_trace(go.Bar(
            x=cat_df['Metric'],
            y=cat_df['Value'],
            name=f"{category} ({run_label})",
            marker_color=color_map.get(category, '#666666'),
            text=text_vals,
            textposition=customization.get('bar_label_position', 'auto'),
            hovertemplate='<b>%{x}</b><br>Score: %{y:.3f}<br>Run: ' + run_label + '<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=customization.get('title_font_size', 16))),
        xaxis_title=customization.get('x_title', "Metric"),
        yaxis_title=customization.get('y_title', "Normalized Score (0-1, higher=better)"),
        barmode='group',
        legend_title="Category",
        hovermode='x unified',
        margin=dict(l=50, r=250, t=50, b=50)
    )
    
    fig = apply_plot_customization(fig, customization, chart_type='bar')
    
    scale_text = (
        f"<b>Error → Score Conversion:</b><br>"
        f"Score = exp(−error / scale)<br>"
        f"MSE scale: {normalization_scales['mse']:.0e}<br>"
        f"PDE/MAE scale: {normalization_scales['pde_residual_mean']:.0e}<br>"
        f"Mass: 1 − error (0-1)<br>"
        f"Param Dist scale: {normalization_scales['param_distance']}<br>"
        f"R²/SSIM: Untransformed (0–1)<br>"
        f"<i>Higher score = better agreement</i>"
    )
    
    fig.add_annotation(
        text=scale_text,
        xref="paper", yref="paper",
        x=1.01, y=0.5,
        showarrow=False,
        font=dict(size=10, family="Arial"),
        align="left",
        bgcolor="rgba(245,245,245,0.95)",
        bordercolor="#888888",
        borderwidth=1,
        borderpad=4
    )
    
    return fig


def plot_multi_case_bar_chart(case_metrics_dict: Dict[str, pd.DataFrame],
                             title: str = "Multi-Case Comparison",
                             customization: Optional[Dict] = None,
                             normalization_scales: Optional[Dict] = None,
                             selected_metrics: Optional[List[str]] = None) -> go.Figure:
    if customization is None:
        customization = {}
    if normalization_scales is None:
        normalization_scales = {
            'mse': 1e-6, 'mae': 1e-4, 'max_error': 1e-4,
            'pde_residual_mean': 1e-4, 'pde_residual_max': 1e-4,
            'bc_error': 1e-4, 'mass_error': 1.0, 'param_distance': 0.3
        }
    
    if selected_metrics is None:
        selected_metrics = list(ALL_VISUAL_METRICS.keys())
    
    color_cycle = px.colors.qualitative.Set1
    fig = go.Figure()
    
    for idx, (run_label, metrics_df) in enumerate(case_metrics_dict.items()):
        df = metrics_df.copy()
        
        filtered_rows = []
        for metric_idx, row in df.iterrows():
            metric_name = row['Metric']
            display_name = None
            for disp_name, attr_name in ALL_VISUAL_METRICS.items():
                if attr_name == metric_name and disp_name in selected_metrics:
                    display_name = disp_name
                    break
            if display_name is None:
                continue
            
            val = row['Value']
            normalized_val = _normalize_metric_value(metric_name, val, normalization_scales)
            filtered_rows.append({
                'Metric': display_name,
                'Value': normalized_val,
                'Category': row['Category'],
                'InternalName': metric_name
            })
        
        # ✅ FIXED: Handle empty filtered_rows gracefully
        if not filtered_rows:
            continue
            
        df_filtered = pd.DataFrame(filtered_rows)
        
        for category in df_filtered['Category'].unique():
            cat_df = df_filtered[df_filtered['Category'] == category]
            fig.add_trace(go.Bar(
                x=cat_df['Metric'],
                y=cat_df['Value'],
                name=f"{category} - {run_label}",
                marker_color=color_cycle[idx % len(color_cycle)],
                opacity=0.7,
                hovertemplate='<b>%{x}</b><br>Score: %{y:.3f}<br>Run: ' + run_label + '<extra></extra>'
            ))
    
    # ✅ FIXED: Handle case where no traces were added
    if not fig.data:
        fig.add_annotation(
            text="⚠️ No metrics selected or available for display",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray"),
            align="center",
            bgcolor="rgba(245,245,245,0.8)"
        )
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Metric",
        yaxis_title="Normalized Score (0-1)",
        barmode='group',
        legend_title="Category - Run",
        hovermode='x unified',
        margin=dict(l=50, r=250, t=50, b=50)
    )
    
    fig = apply_plot_customization(fig, customization, chart_type='bar')
    
    return fig


def plot_radar_chart(metrics: ValidationMetrics, 
                    title: str = "Validation Radar Plot",
                    customization: Optional[Dict] = None,
                    normalization_scales: Optional[Dict] = None,
                    run_label: str = "Current Run",
                    selected_metrics: Optional[List[str]] = None) -> go.Figure:
    if customization is None:
        customization = {}
    if normalization_scales is None:
        normalization_scales = {
            'mse': 1e-6,
            'pde_residual': 1e-4,
            'mass_error': 1.0,
            'param_distance': 0.3
        }
    
    if selected_metrics is None:
        selected_metrics = DEFAULT_RADAR_METRICS
    
    categories = []
    values = []

    for display_name, attr_name in ALL_VISUAL_METRICS.items():
        if display_name not in selected_metrics:
            continue
        categories.append(display_name)
        val = _normalize_metric_value(attr_name, getattr(metrics, attr_name, 0.0), normalization_scales)
        values.append(val)

    if categories:
        categories += [categories[0]]
        values += [values[0]]

    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=run_label,
        line=dict(width=customization.get('line_width', 2), color=customization.get('radar_line_color', '#2E86AB')),
        fillcolor=customization.get('radar_fill_color', 'rgba(46, 134, 171, 0.3)')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat='.1f'
            )
        ),
        title=dict(text=title, x=0.5),
        showlegend=True,
        height=customization.get('figure_height', 500),
        width=customization.get('figure_width', 600),
        margin=dict(l=50, r=250, t=50, b=50),
        legend=dict(x=1.02, y=1.0)
    )
    
    scale_text = (
        f"<b>Normalization:</b><br>"
        f"MSE → exp(−MSE / {normalization_scales['mse']:.0e})<br>"
        f"PDE → exp(−Res / {normalization_scales['pde_residual']:.0e})<br>"
        f"Mass → 1 − error<br>"
        f"Param → exp(−dist / {normalization_scales['param_distance']})"
    )
    
    fig.add_annotation(
        text=scale_text,
        xref="paper", yref="paper",
        x=1.02, y=0.5,
        showarrow=False,
        font=dict(size=10),
        align="left",
        bgcolor="rgba(245,245,245,0.95)",
        bordercolor="#888888",
        borderwidth=1,
        borderpad=5
    )
    
    return fig


def plot_multi_case_radar_chart(case_metrics_dict: Dict[str, ValidationMetrics],
                               title: str = "Multi-Case Radar Comparison",
                               customization: Optional[Dict] = None,
                               normalization_scales: Optional[Dict] = None,
                               selected_metrics: Optional[List[str]] = None) -> go.Figure:
    if customization is None:
        customization = {}
    if normalization_scales is None:
        normalization_scales = {
            'mse': 1e-6, 'pde_residual': 1e-4,
            'mass_error': 1.0, 'param_distance': 0.3
        }
    
    if selected_metrics is None:
        selected_metrics = DEFAULT_RADAR_METRICS
    
    color_cycle = px.colors.qualitative.Set1
    fig = go.Figure()
    
    for idx, (run_label, metrics) in enumerate(case_metrics_dict.items()):
        values = []
        categories = []
        
        for display_name, attr_name in ALL_VISUAL_METRICS.items():
            if display_name not in selected_metrics:
                continue
            categories.append(display_name)
            val = _normalize_metric_value(attr_name, getattr(metrics, attr_name, 0.0), normalization_scales)
            values.append(val)
        
        if not categories:
            continue
            
        values += [values[0]]
        cats = categories + [categories[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=cats,
            fill='toself',
            name=run_label,
            line=dict(color=color_cycle[idx % len(color_cycle)], width=customization.get('line_width', 2)),
            fillcolor=customization.get('radar_fill_color', 'rgba(46, 134, 171, 0.3)')
        ))
    
    # ✅ FIXED: Handle empty radar chart
    if not fig.data:
        fig.add_annotation(
            text="⚠️ No metrics selected or available for radar display",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray"),
            align="center",
            bgcolor="rgba(245,245,245,0.8)"
        )
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickformat='.1f')
        ),
        title=dict(text=title, x=0.5),
        showlegend=True,
        legend=dict(x=1.05, y=0.5),
        height=customization.get('figure_height', 600),
        width=customization.get('figure_width', 700),
        margin=dict(l=50, r=250, t=50, b=50)
    )
    
    return fig


def plot_residual_heatmap(residual: np.ndarray, x: np.ndarray, y: np.ndarray,
                         title: str = "PDE Residual Field") -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=residual.T,
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
                           title: str = "Uncertainty vs Parameter Distance",
                           customization: Optional[Dict] = None) -> go.Figure:
    if customization is None:
        customization = {}
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
        height=customization.get('figure_height', 500),
        width=customization.get('figure_width', 700),
        font=dict(size=customization.get('font_size', 12)),
        coloraxis_colorbar=dict(title="Score")
    )
    
    return fig


def plot_comparison_heatmaps(interp_c1: np.ndarray, gt_c1: np.ndarray,
                           interp_c2: np.ndarray, gt_c2: np.ndarray,
                           x: np.ndarray, y: np.ndarray,
                           title_prefix: str = "Field Comparison") -> Tuple[go.Figure, go.Figure]:
    fig_c1 = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Interpolated", "Ground Truth", "Absolute Error"),
        horizontal_spacing=0.05
    )
    
    fig_c1.add_trace(go.Heatmap(z=interp_c1.T, x=x, y=y, colorscale='viridis', showscale=True, name='Interp'), row=1, col=1)
    fig_c1.add_trace(go.Heatmap(z=gt_c1.T, x=x, y=y, colorscale='viridis', showscale=False, name='GT'), row=1, col=2)
    fig_c1.add_trace(go.Heatmap(z=np.abs(interp_c1 - gt_c1).T, x=x, y=y, colorscale='RdYlBu_r', showscale=True, name='Error'), row=1, col=3)
    
    fig_c1.update_layout(title=dict(text=f"{title_prefix} - Cu Concentration", x=0.5), height=400)
    
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
# 11. PHYSICS-INFORMED INTERPOLATION ENHANCEMENT
# =============================================
class PhysicsAwareInterpolator:
    def __init__(self, base_interpolator, physics_constants: Dict = None):
        self.base = base_interpolator
        self.constants = physics_constants or PHYSICS_CONSTANTS
        self.enforce_bc = True
        self.pde_weight = 0.1
        self.mass_weight = 0.05
        
    def interpolate_with_physics(self, solutions: List[Dict], params_list: List[Dict],
                               target_params: Dict, target_shape: Tuple[int, int] = (50, 50),
                               time_norm: float = 1.0, optimize: bool = True) -> Dict:
        result = self.base(solutions, params_list, 
                          target_params.get('Ly', 60),
                          target_params.get('c_cu', 1.5e-3),
                          target_params.get('c_ni', 4e-4))
        
        if not result or 'fields' not in result:
            return result
        
        fields = result['fields']
        c1 = fields.get('c1_preds', [np.zeros(target_shape)])[0]
        c2 = fields.get('c2_preds', [np.zeros(target_shape)])[0]
        
        if self.enforce_bc:
            Lx = target_params.get('Lx', 60)
            Ly = target_params.get('Ly', 60)
            x = np.linspace(0, Lx, target_shape[1])
            y = np.linspace(0, Ly, target_shape[0])
            
            # ✅ Defensive: provide defaults for boundary condition keys
            c1, c2 = enforce_boundary_conditions(
                c1, c2, x, y,
                target_params.get('c_cu_top', target_params.get('C_Cu', 1.59e-3)),
                target_params.get('c_cu_bottom', 0.0),
                target_params.get('c_ni_top', 0.0),
                target_params.get('c_ni_bottom', target_params.get('C_Ni', 4e-4)),
                enforce_type='hard'
            )
        
        if optimize and torch.cuda.is_available():
            c1, c2 = self._physics_refinement(
                c1, c2, target_params, target_shape, time_norm
            )
        
        result['fields']['c1_preds'] = [c1]
        result['fields']['c2_preds'] = [c2]
        result['physics_enforced'] = True
        
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
        c1_t = torch.tensor(c1, dtype=torch.float32, requires_grad=True)
        c2_t = torch.tensor(c2, dtype=torch.float32, requires_grad=True)
        
        Lx = target_params.get('Lx', 60)
        Ly = target_params.get('Ly', 60)
        t = time_norm * target_params.get('t_max', 200)
        
        x = torch.linspace(0, Lx, target_shape[1])
        y = torch.linspace(0, Ly, target_shape[0])
        # ✅ FIXED: Added indexing='ij' for consistency
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        optimizer = optim.Adam([c1_t, c2_t], lr=lr)
        
        for step in range(n_steps):
            optimizer.zero_grad()
            
            lap_c1 = self._laplacian_torch(c1_t, X, Y)
            lap_c2 = self._laplacian_torch(c2_t, X, Y)
            
            residual1 = -(self.constants['D11'] * lap_c1 + self.constants['D12'] * lap_c2)
            residual2 = -(self.constants['D21'] * lap_c1 + self.constants['D22'] * lap_c2)
            
            pde_loss = torch.mean(residual1**2 + residual2**2)
            
            bc_loss = (
                torch.mean((c1_t[:, -1] - target_params.get('c_cu_top', target_params.get('C_Cu', 1.59e-3)))**2) +
                torch.mean((c1_t[:, 0] - target_params.get('c_cu_bottom', 0.0))**2) +
                torch.mean((c2_t[:, -1] - target_params.get('c_ni_top', 0.0))**2) +
                torch.mean((c2_t[:, 0] - target_params.get('c_ni_bottom', target_params.get('C_Ni', 4e-4)))**2)
            )
            
            loss = self.pde_weight * pde_loss + self.mass_weight * bc_loss
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                c1_t.clamp_(0, self.constants['C_CU_RANGE'][1])
                c2_t.clamp_(0, self.constants['C_NI_RANGE'][1])
        
        return c1_t.detach().numpy(), c2_t.detach().numpy()
    
    def _laplacian_torch(self, c: torch.Tensor, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        dc_dx = torch.autograd.grad(
            c.sum(), X, create_graph=True, retain_graph=True
        )[0]
        dc_dy = torch.autograd.grad(
            c.sum(), Y, create_graph=True, retain_graph=True
        )[0]
        
        d2c_dx2 = torch.autograd.grad(
            dc_dx.sum(), X, create_graph=True, retain_graph=True
        )[0]
        d2c_dy2 = torch.autograd.grad(
            dc_dy.sum(), Y, create_graph=True, retain_graph=True
        )[0]
        
        return d2c_dx2 + d2c_dy2


# =============================================
# 12. SESSION STATE INITIALIZATION
# =============================================
def initialize_session_state():
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
        'params_list': [],
        'current_run_id': None,
        'saved_runs_loaded': {},
        'plot_customization': {
            'font_size': 12,
            'title_font_size': 16,
            'axis_title_font_size': 13,
            'tick_font_size': 11,
            'legend_font_size': 12,
            'figure_width': 800,
            'figure_height': 500,
            'line_width': 2,
            'axis_line_width': 1.5,
            'grid_line_width': 1.0,
            'grid_color': '#D3D3D3',
            'show_grid': True,
            'show_bar_labels': True,
            'bar_label_position': 'auto',
            'x_tick_angle': 0,
            'legend_orientation': 'v',
            'radar_line_color': '#2E86AB',
            'radar_fill_color': 'rgba(46, 134, 171, 0.3)',
            'x_title': 'Metric',
            'y_title': 'Normalized Score (0-1, higher=better)'
        },
        'selected_bar_metrics': DEFAULT_BAR_METRICS.copy(),
        'selected_radar_metrics': DEFAULT_RADAR_METRICS.copy()
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# =============================================
# 13. MAIN STREAMLIT APPLICATION
# =============================================
def main():
    st.set_page_config(page_title="Physics-Informed Validation Dashboard", layout="wide")
    st.title("🔬 Physics-Informed Validation & Uncertainty Quantification")
    
    initialize_session_state()
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.header("📁 Data Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Load Solutions", width='stretch'):
                with st.spinner("Loading solutions from pinn_solutions/..."):
                    if st.session_state.solution_loader is None:
                        st.session_state.solution_loader = EnhancedSolutionLoader(SOLUTION_DIR)
                    
                    solutions, logs = st.session_state.solution_loader.load_all_solutions()
                    st.session_state.solutions = solutions
                    
                    params_list = []
                    for sol in solutions:
                        p = sol['params']
                        params_list.append((p['Ly'], p['C_Cu'], p['C_Ni']))
                    st.session_state.params_list = params_list
                    
                    st.session_state.solutions_loaded = len(solutions) > 0
                    
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
        
        # ===== EXPANDED: Plot Customization Panel =====
        with st.expander("🎨 Plot Customization (Bar & Radar)", expanded=False):
            st.subheader("General")
            st.session_state.plot_customization['font_size'] = st.slider("Base font size", 8, 20, st.session_state.plot_customization.get('font_size', 12))
            st.session_state.plot_customization['title_font_size'] = st.slider("Title font size", 10, 30, st.session_state.plot_customization.get('title_font_size', 16))
            st.session_state.plot_customization['axis_title_font_size'] = st.slider("Axis title font size", 8, 20, st.session_state.plot_customization.get('axis_title_font_size', 13))
            st.session_state.plot_customization['tick_font_size'] = st.slider("Tick font size", 6, 18, st.session_state.plot_customization.get('tick_font_size', 11))
            col_w, col_h = st.columns(2)
            with col_w:
                st.session_state.plot_customization['figure_width'] = st.number_input("Figure width", 400, 1500, st.session_state.plot_customization.get('figure_width', 800), step=50)
            with col_h:
                st.session_state.plot_customization['figure_height'] = st.number_input("Figure height", 300, 1000, st.session_state.plot_customization.get('figure_height', 500), step=50)
            
            st.subheader("Line & Grid")
            st.session_state.plot_customization['axis_line_width'] = st.slider("Axis line width", 0.5, 5.0, st.session_state.plot_customization.get('axis_line_width', 1.5), 0.5)
            st.session_state.plot_customization['grid_line_width'] = st.slider("Grid line width", 0.5, 3.0, st.session_state.plot_customization.get('grid_line_width', 1.0), 0.1)
            
            st.session_state.plot_customization['grid_color'] = safe_color_picker(
                "Grid color", 
                key='grid_color_picker',
                default=st.session_state.plot_customization.get('grid_color', '#D3D3D3'),
                help="Select grid line color (hex format required)"
            )
            st.session_state.plot_customization['show_grid'] = st.checkbox("Show grid", st.session_state.plot_customization.get('show_grid', True))
            
            st.subheader("Bar Chart Specific")
            st.session_state.plot_customization['show_bar_labels'] = st.checkbox("Show bar value labels", st.session_state.plot_customization.get('show_bar_labels', True))
            st.session_state.plot_customization['bar_label_position'] = st.selectbox("Bar label position", ['auto', 'inside', 'outside'], index=0)
            st.session_state.plot_customization['x_tick_angle'] = st.slider("X-axis tick angle (deg)", -90, 90, st.session_state.plot_customization.get('x_tick_angle', 0))
            st.session_state.plot_customization['x_title'] = st.text_input("X-axis title", st.session_state.plot_customization.get('x_title', 'Metric'))
            st.session_state.plot_customization['y_title'] = st.text_input("Y-axis title", st.session_state.plot_customization.get('y_title', 'Normalized Score (0-1, higher=better)'))
            
            st.subheader("Radar Chart Specific")
            st.session_state.plot_customization['radar_line_color'] = safe_color_picker(
                "Radar line color",
                key='radar_line_color_picker',
                default=st.session_state.plot_customization.get('radar_line_color', '#2E86AB'),
                help="Select radar chart line color"
            )
            st.session_state.plot_customization['radar_fill_color'] = st.text_input(
                "Radar fill color (rgba)", 
                st.session_state.plot_customization.get('radar_fill_color', 'rgba(46, 134, 171, 0.3)'),
                help="Use rgba(r,g,b,a) format, e.g., rgba(46,134,171,0.3)"
            )
            st.session_state.plot_customization['line_width'] = st.slider("Radar line width", 1, 5, st.session_state.plot_customization.get('line_width', 2))
        
        # =============================================
        # METRIC SELECTION UI - USER CUSTOMIZABLE ASPECTS
        # =============================================
        with st.expander("📊 Choose Metrics to Display", expanded=True):
            st.markdown("**Bar Chart Metrics:**")
            st.session_state.selected_bar_metrics = st.multiselect(
                "Select metrics for bar charts",
                options=list(ALL_VISUAL_METRICS.keys()),
                default=st.session_state.get('selected_bar_metrics', DEFAULT_BAR_METRICS),
                key="bar_metrics_selector"
            )
            
            st.markdown("**Radar Chart Metrics:**")
            st.session_state.selected_radar_metrics = st.multiselect(
                "Select metrics for radar charts",
                options=list(ALL_VISUAL_METRICS.keys()),
                default=st.session_state.get('selected_radar_metrics', DEFAULT_RADAR_METRICS),
                key="radar_metrics_selector"
            )
            
            if st.button("🔄 Reset to Defaults", key="reset_metrics_btn"):
                st.session_state.selected_bar_metrics = DEFAULT_BAR_METRICS.copy()
                st.session_state.selected_radar_metrics = DEFAULT_RADAR_METRICS.copy()
                st.rerun()
        
        if st.button("🚀 Run Validation", type="primary", width='stretch'):
            if not st.session_state.solutions_loaded:
                st.error("❌ Please load solutions first!")
                st.stop()
            
            with st.spinner("Running validation pipeline..."):
                if st.session_state.interpolator is None:
                    st.session_state.interpolator = MultiParamAttentionInterpolator()
                if physics_aware and st.session_state.physics_interpolator is None:
                    st.session_state.physics_interpolator = PhysicsAwareInterpolator(st.session_state.interpolator)
                
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
                    results[idx] = {
                        'metrics': metrics, 
                        'interp_c1': interp_c1, 
                        'interp_c2': interp_c2, 
                        'gt_c1': gt_c1, 
                        'gt_c2': gt_c2, 
                        'x': x, 
                        'y': y, 
                        'params': target,
                        'target_params': target
                    }
                
                st.session_state.validation_results = results
                st.session_state.current_run_id = generate_run_id()
                st.success(f"✅ Validation completed on {len(results)} held-out cases. Run ID: {st.session_state.current_run_id}")
        
        st.header("💾 Save/Load Results")
        if st.session_state.validation_results and st.session_state.current_run_id:
            if st.button("💾 Save Current Run", width='stretch'):
                summary_data = []
                for idx, res in st.session_state.validation_results.items():
                    m = res['metrics']
                    target_params = res.get('target_params', {})
                    summary_data.append({
                        'Case': f"#{idx}",
                        'Case_Label': format_case_label(idx, target_params),
                        'Overall Score': m.overall_score,
                        'MSE (Cu)': m.mse_c1,
                        'MSE (Ni)': m.mse_c2,
                        'PDE Residual': m.pde_residual_mean,
                        'Mass Error': m.mass_error,
                        'Weight Entropy': m.weight_entropy,
                        'Param Distance': m.param_distance,
                        'Ly': target_params.get('Ly', 60.0),
                        'C_Cu': target_params.get('C_Cu', 1.5e-3),
                        'C_Ni': target_params.get('C_Ni', 0.5e-3)
                    })
                
                case_results = {}
                for idx, res in st.session_state.validation_results.items():
                    target_params = res.get('target_params', {})
                    case_results[str(idx)] = {
                        'metrics': res['metrics'].to_dict(),
                        'target_params': {
                            'Ly': target_params.get('Ly', 60.0),
                            'C_Cu': target_params.get('C_Cu', 1.5e-3),
                            'C_Ni': target_params.get('C_Ni', 0.5e-3)
                        }
                    }
                
                saved_run = SavedValidationRun(
                    run_id=st.session_state.current_run_id,
                    timestamp=datetime.now().isoformat(),
                    config={
                        'held_out_fraction': held_out_frac,
                        'physics_aware': physics_aware,
                        'optimize_fields': optimize_fields
                    },
                    case_results=case_results,
                    summary_metrics=summary_data
                )
                
                filepath = saved_run.save_to_disk()
                st.success(f"✅ Saved to: {filepath}")
        
        saved_runs = SavedValidationRun.list_saved_runs()
        if saved_runs:
            st.subheader("📂 Saved Runs")
            selected_runs = st.multiselect("Select runs to compare", saved_runs, default=[])
            if selected_runs:
                st.session_state.selected_comparison_runs = selected_runs
    
    if not st.session_state.solutions_loaded:
        st.info("👈 Use the sidebar to load solutions and run validation.")
        return
    
    if not st.session_state.validation_results:
        st.warning("⚠️ No validation results available. Click 'Run Validation' in the sidebar.")
        return
    
    results = st.session_state.validation_results
    
    st.subheader("📊 Validation Summary")
    summary_data = []
    for idx, res in results.items():
        m = res['metrics']
        target_params = res.get('target_params', {})
        summary_data.append({
            'Case': format_case_label(idx, target_params),
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
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Single-Case Charts", "🎯 Multi-Case Comparison", "🔍 Field Comparison", "📉 Uncertainty Analysis", "💾 Load & Compare Saved"])
    
    with tab1:
        st.subheader("Single-Case Validation Metrics")
        
        case_options = ["📊 Aggregate (All Cases)"] + [
            format_case_label(idx, results[idx].get('target_params', {})) 
            for idx in results.keys()
        ]
        selected_case_label = st.selectbox("Select Case to View", case_options, key="single_case_selector")
        
        if "Aggregate" in selected_case_label:
            all_metrics_df = pd.concat([res['metrics'].to_dataframe() for res in results.values()])
            avg_metrics = all_metrics_df.groupby('Metric')['Value'].mean().reset_index()
            avg_metrics['Category'] = all_metrics_df.groupby('Metric')['Category'].first().values
            chart_title = "Average Validation Metrics (All Cases)"
        else:
            case_idx_match = re.search(r'Case #(\d+)', selected_case_label)
            if case_idx_match:
                case_idx = int(case_idx_match.group(1))
                avg_metrics = results[case_idx]['metrics'].to_dataframe()
                chart_title = f"Validation Metrics - {selected_case_label}"
            else:
                st.error("Could not parse case index")
                return
        
        norm_scales = {
            'mse': 1e-6, 'mae': 1e-4, 'max_error': 1e-4,
            'pde_residual_mean': 1e-4, 'pde_residual_max': 1e-4,
            'bc_error': 1e-4, 'mass_error': 1.0, 'param_distance': 0.3
        }
        
        fig_bar = plot_metrics_bar_chart(
            avg_metrics, 
            title=chart_title, 
            customization=st.session_state.plot_customization,
            normalization_scales=norm_scales,
            run_label=st.session_state.current_run_id or "Current",
            selected_metrics=st.session_state.get('selected_bar_metrics')
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab2:
        st.subheader("Multi-Case Comparison (Current Run)")
        
        case_options = [format_case_label(idx, results[idx].get('target_params', {})) for idx in results.keys()]
        selected_case_labels = st.multiselect(
            "Select cases to compare",
            case_options,
            default=case_options[:3] if len(case_options) >= 3 else case_options
        )
        
        if selected_case_labels:
            case_metrics_dict = {}
            for label in selected_case_labels:
                case_idx_match = re.search(r'Case #(\d+)', label)
                if case_idx_match:
                    case_idx = int(case_idx_match.group(1))
                    if case_idx in results:
                        case_metrics_dict[label] = results[case_idx]['metrics'].to_dataframe()
            
            fig_multi_bar = plot_multi_case_bar_chart(
                case_metrics_dict,
                title="Multi-Case Bar Chart Comparison",
                customization=st.session_state.plot_customization,
                selected_metrics=st.session_state.get('selected_bar_metrics')
            )
            st.plotly_chart(fig_multi_bar, use_container_width=True)
            
            radar_metrics_dict = {}
            for label in selected_case_labels:
                case_idx_match = re.search(r'Case #(\d+)', label)
                if case_idx_match:
                    case_idx = int(case_idx_match.group(1))
                    if case_idx in results:
                        radar_metrics_dict[label] = results[case_idx]['metrics']
            
            fig_multi_radar = plot_multi_case_radar_chart(
                radar_metrics_dict,
                title="Multi-Case Radar Comparison",
                customization=st.session_state.plot_customization,
                selected_metrics=st.session_state.get('selected_radar_metrics')
            )
            st.plotly_chart(fig_multi_radar, use_container_width=True)
    
    with tab3:
        st.subheader("Field Comparison: Interpolated vs Ground Truth")
        
        case_options = [format_case_label(idx, results[idx].get('target_params', {})) for idx in results.keys()]
        selected_case_label = st.selectbox("Select Validation Case", case_options, format_func=lambda x: x, key="field_case_selector")
        
        case_idx_match = re.search(r'Case #(\d+)', selected_case_label)
        if case_idx_match:
            case_idx = int(case_idx_match.group(1))
            if case_idx in results:
                res = results[case_idx]
                
                fig_c1, fig_c2 = plot_comparison_heatmaps(
                    res['interp_c1'], res['gt_c1'],
                    res['interp_c2'], res['gt_c2'],
                    res['x'], res['y'],
                    title_prefix=selected_case_label
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_c1, use_container_width=True)
                with col2:
                    st.plotly_chart(fig_c2, use_container_width=True)
                
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
        
        metrics_list = [res['metrics'] for res in results.values()]
        fig_scatter = plot_uncertainty_scatter(metrics_list, ['Ly', 'c_cu', 'c_ni'], customization=st.session_state.plot_customization)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.markdown("### 🔑 Key Insights")
        avg_entropy = np.mean([m.weight_entropy for m in metrics_list])
        avg_distance = np.mean([m.param_distance for m in metrics_list])
        avg_score = np.mean([m.overall_score for m in metrics_list])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg. Weight Entropy", f"{avg_entropy:.3f}", help="Higher = more uncertain")
        with col2:
            st.metric("Avg. Parameter Distance", f"{avg_distance:.3f}", help="Lower = closer to training")
        with col3:
            st.metric("Avg. Validation Score", f"{avg_score:.3f} / 1.0", help="Higher = better")
    
    with tab5:
        st.subheader("💾 Load & Compare Saved Validation Runs")
        
        saved_runs = SavedValidationRun.list_saved_runs()
        if not saved_runs:
            st.info("📭 No saved validation runs found. Run validation and save results to compare.")
        else:
            st.markdown(f"**Available saved runs:** {len(saved_runs)}")
            
            runs_to_compare = st.multiselect(
                "Select saved runs to compare",
                saved_runs,
                default=saved_runs[:2] if len(saved_runs) >= 2 else saved_runs
            )
            
            if runs_to_compare:
                st.session_state.saved_runs_loaded = {}
                case_metrics_dict = {}
                radar_metrics_dict = {}
                
                for run_id in runs_to_compare:
                    try:
                        saved_run = SavedValidationRun.load_from_disk(run_id)
                        st.session_state.saved_runs_loaded[run_id] = saved_run
                        
                        if saved_run.summary_metrics and isinstance(saved_run.summary_metrics, list):
                            summary_df = pd.DataFrame(saved_run.summary_metrics)
                            
                            if not summary_df.empty and 'Overall Score' in summary_df.columns:
                                avg_metrics = pd.DataFrame({
                                    'Metric': ['overall_score', 'mse_c1', 'mse_c2', 'pde_residual_mean', 'mass_error'],
                                    'Value': [
                                        summary_df['Overall Score'].mean() if 'Overall Score' in summary_df else 0,
                                        summary_df['MSE (Cu)'].mean() if 'MSE (Cu)' in summary_df else 0,
                                        summary_df['MSE (Ni)'].mean() if 'MSE (Ni)' in summary_df else 0,
                                        summary_df['PDE Residual'].mean() if 'PDE Residual' in summary_df else 0,
                                        summary_df['Mass Error'].mean() if 'Mass Error' in summary_df else 0
                                    ],
                                    'Category': ['composite', 'pointwise', 'pointwise', 'physics', 'physics']
                                })
                                case_metrics_dict[f"Saved: {run_id[:16]}"] = avg_metrics
                        
                        if saved_run.case_results:
                            first_case = list(saved_run.case_results.values())[0]
                            if 'metrics' in first_case:
                                metrics_obj = ValidationMetrics(**first_case['metrics'])
                                radar_metrics_dict[f"Saved: {run_id[:16]}"] = metrics_obj
                        
                    except KeyError as e:
                        st.error(f"Error loading run {run_id}: Missing key {e}. The saved file may be from an older version.")
                    except Exception as e:
                        st.error(f"Error loading run {run_id}: {type(e).__name__}: {e}")
                
                if case_metrics_dict:
                    st.subheader("📊 Multi-Run Bar Chart Comparison")
                    fig_saved_bar = plot_multi_case_bar_chart(
                        case_metrics_dict,
                        title="Comparison Across Saved Validation Runs",
                        customization=st.session_state.plot_customization,
                        selected_metrics=st.session_state.get('selected_bar_metrics')
                    )
                    st.plotly_chart(fig_saved_bar, use_container_width=True)
                
                if radar_metrics_dict:
                    st.subheader("🎯 Multi-Run Radar Comparison")
                    fig_saved_radar = plot_multi_case_radar_chart(
                        radar_metrics_dict,
                        title="Radar Comparison of Saved Runs",
                        customization=st.session_state.plot_customization,
                        selected_metrics=st.session_state.get('selected_radar_metrics')
                    )
                    st.plotly_chart(fig_saved_radar, use_container_width=True)
                
                st.subheader("📋 Saved Run Details")
                for run_id in runs_to_compare:
                    if run_id in st.session_state.saved_runs_loaded:
                        saved_run = st.session_state.saved_runs_loaded[run_id]
                        with st.expander(f"📄 Run {run_id} ({saved_run.timestamp})", expanded=False):
                            st.write("**Configuration:**")
                            st.json(saved_run.config)
                            if saved_run.summary_metrics:
                                st.write("**Summary Metrics:**")
                                summary_display = pd.DataFrame(saved_run.summary_metrics)
                                if 'Case_Label' in summary_display.columns:
                                    st.dataframe(summary_display[['Case_Label', 'Overall Score', 'MSE (Cu)', 'MSE (Ni)', 'PDE Residual', 'Mass Error']])
                                else:
                                    st.dataframe(summary_display[['Case', 'Overall Score', 'MSE (Cu)', 'MSE (Ni)', 'PDE Residual', 'Mass Error']])


if __name__ == "__main__":
    main()
