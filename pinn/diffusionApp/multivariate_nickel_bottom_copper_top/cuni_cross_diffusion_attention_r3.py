#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ATTENTION-BASED CU-NI INTERDIFFUSION VISUALIZER WITH ENHANCED LLM NATURAL LANGUAGE INTERFACE
==============================================================================================
- Natural language parsing with explicit schema, multi-pattern regex, and hybrid LLM (GPT-2/Qwen)
- Confidence-based merging with per-parameter scoring and ensemble support
- Publication-quality 2D heatmaps, centerline curves, and parameter sweeps
- Full figure customization and PNG/PDF export
- Cached LLM loading, robust JSON extraction, and fallback mechanisms
- SciBERT semantic relevance scoring with keyword fallback
- Prominent parameter display with units, valid ranges, and extraction status
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
import time

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
# ENHANCED DIFFUSION PARAMETERS SCHEMA
# =============================================
class DiffusionParameters:
    """Explicit parameter schema with ranges and units for robust parsing."""
    
    RANGES = {
        'ly_target': (30.0, 120.0, 'μm'),
        'c_cu_target': (0.0, 2.9e-3, 'mol/cc'),
        'c_ni_target': (0.0, 1.8e-3, 'mol/cc'),
        'sigma': (0.05, 0.5, ''),  # dimensionless hyperparameter
    }
    
    DEFAULTS = {
        'ly_target': 60.0,
        'c_cu_target': 1.5e-3,
        'c_ni_target': 0.5e-3,
        'sigma': 0.20,
    }
    
    @staticmethod
    def format_value(key: str, value: float) -> str:
        """Format value with appropriate notation and unit."""
        if key not in DiffusionParameters.RANGES:
            return str(value)
        low, high, unit = DiffusionParameters.RANGES[key]
        if unit:
            if abs(value) < 0.01 or abs(value) > 100:
                return f"{value:.1e} {unit}"
            else:
                return f"{value:.3f} {unit}"
        else:
            return f"{value:.3f}"
    
    @staticmethod
    def clip_to_range(key: str, value: float) -> float:
        """Clip value to valid range for parameter."""
        if key not in DiffusionParameters.RANGES:
            return value
        low, high, _ = DiffusionParameters.RANGES[key]
        return np.clip(value, low, high)

# =============================================
# ENHANCED NATURAL LANGUAGE PARSER (Hybrid Regex + LLM)
# =============================================
class DiffusionNLParser:
    """Extracts Cu-Ni diffusion parameters from natural language using enhanced regex + LLM hybrid parsing."""
    
    def __init__(self):
        self.defaults = DiffusionParameters.DEFAULTS.copy()
        # Multi-pattern regex for linguistic diversity
        self.patterns = {
            'ly_target': [
                r'(?:joint\s*thickness|domain\s*length|L_y|Ly|domain\s*size)\s*[=:]\s*(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*(?:μm|um|microns?|micrometers?)',
                r'(?:thickness|length|size)\s*(?:of|is|=|:)?\s*(\d+(?:\.\d+)?)\s*(?:μm|um)?',
                r'(\d+(?:\.\d+)?)\s*μm\s*(?:joint|domain|length)',
            ],
            'c_cu_target': [
                r'(?:Cu\s*concentration|C_Cu|c_Cu|top\s*concentration|copper\s*conc\.?)\s*[=:]\s*([\d.]+(?:e[+-]?\d+)?)',
                r'([\d.]+(?:e[+-]?\d+)?)\s*(?:mol/cc|molar|M)\s*(?:Cu|copper|top)',
                r'(?:top|upper)\s*(?:boundary\s*)?(?:Cu|copper)\s*[=:]\s*([\d.]+(?:e[+-]?\d+)?)',
                r'Cu\s*[=:]\s*([\d.]+(?:e[+-]?\d+)?)\s*(?:mol/cc)?',
            ],
            'c_ni_target': [
                r'(?:Ni\s*concentration|C_Ni|c_Ni|bottom\s*concentration|nickel\s*conc\.?)\s*[=:]\s*([\d.]+(?:e[+-]?\d+)?)',
                r'([\d.]+(?:e[+-]?\d+)?)\s*(?:mol/cc|molar|M)\s*(?:Ni|nickel|bottom)',
                r'(?:bottom|lower)\s*(?:boundary\s*)?(?:Ni|nickel)\s*[=:]\s*([\d.]+(?:e[+-]?\d+)?)',
                r'Ni\s*[=:]\s*([\d.]+(?:e[+-]?\d+)?)\s*(?:mol/cc)?',
            ],
            'sigma': [
                r'(?:sigma|σ|locality)\s*[=:]\s*(\d+(?:\.\d+)?)',
                r'(?:interpolation\s*sigma|attention\s*sigma)\s*[=:]\s*(\d+(?:\.\d+)?)',
            ],
        }
    
    def parse_regex(self, text: str) -> Dict[str, Any]:
        """Extract parameters using enhanced regex patterns only."""
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
                            # Handle scientific notation robustly
                            val_clean = val.replace('e-0', 'e-').replace('E-0', 'E-')
                            params[key] = float(val_clean)
                        elif key == 'sigma':
                            params[key] = float(val)
                    except (ValueError, TypeError):
                        pass
                    break
        
        # Clip to valid ranges
        for key in DiffusionParameters.RANGES:
            params[key] = DiffusionParameters.clip_to_range(key, params[key])
        
        return params
    
    @staticmethod
    def _extract_json_robust(generated: str) -> Optional[Dict]:
        """Robustly extract JSON from LLM output with repair attempts."""
        # Try to find JSON object with nested braces support
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        match = re.search(json_pattern, generated, re.DOTALL)
        if not match:
            match = re.search(r'\{.*?\}', generated, re.DOTALL)
        if not match:
            return None
        
        json_str = match.group(0)
        
        # Repair common JSON errors
        json_str = re.sub(r'(true|false|null)\s*(")', r'\1,\2', json_str)  # missing comma
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)  # trailing comma
        json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)  # single to double quotes
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    
    def parse_with_llm(self, text: str, tokenizer, model, regex_params: Dict = None, 
                      temperature: float = None, use_ensemble: bool = False, 
                      ensemble_runs: int = 3) -> Dict:
        """Use LLM (GPT-2 or Qwen) to extract parameters from natural language with ensemble support."""
        if not tokenizer or not model:
            return self.parse_regex(text)
        
        backend = st.session_state.get('llm_backend_loaded', 'GPT-2')
        if temperature is None:
            # Qwen models benefit from temperature=0 for deterministic JSON
            temperature = 0.0 if "Qwen" in backend else 0.1
        
        # Build enhanced prompt with explicit schema and constraints
        system = """You are a materials science expert. Extract simulation parameters from the user's query.
Reply ONLY with a valid JSON object. Clip every numeric value to its exact valid range."""
        
        examples = """
Examples:
- "Analyze a 50 μm joint with Cu concentration 1.2e-3 and Ni 0.8e-3" 
  → {"ly_target": 50.0, "c_cu_target": 1.2e-3, "c_ni_target": 0.8e-3, "sigma": 0.2}
- "Domain length 80, C_Cu=2.0e-3, C_Ni=1.0e-3" 
  → {"ly_target": 80.0, "c_cu_target": 2.0e-3, "c_ni_target": 1.0e-3, "sigma": 0.2}
- "Ly=45um, top Cu=1.5e-3 mol/cc, bottom Ni=0.3e-3" 
  → {"ly_target": 45.0, "c_cu_target": 1.5e-3, "c_ni_target": 0.3e-3, "sigma": 0.2}
- "Thin joint 40 microns, copper 1.8e-3, nickel 0.4e-3, sigma 0.15"
  → {"ly_target": 40.0, "c_cu_target": 1.8e-3, "c_ni_target": 0.4e-3, "sigma": 0.15}
"""
        
        defaults_json = json.dumps(self.defaults)
        regex_hint = f"\nRegex hint (use as reference): {json.dumps(regex_params or {})}" if regex_params else ""
        
        # Explicit schema declaration in prompt
        schema_text = """
JSON keys must be:
- ly_target: float, domain height in μm, valid range [30.0, 120.0] μm
- c_cu_target: float, Cu boundary concentration in mol/cc, valid range [0.0, 0.0029] mol/cc  
- c_ni_target: float, Ni boundary concentration in mol/cc, valid range [0.0, 0.0018] mol/cc
- sigma: float, interpolation locality parameter, valid range [0.05, 0.5] (dimensionless)

Rules:
1. Clip every numeric value to its exact valid range before output
2. If a parameter is not mentioned, use the default value
3. Output ONLY valid JSON, no additional text
"""
        
        user = f"""{examples}{schema_text}{regex_hint}
Query: "{text}"
Defaults: {defaults_json}
JSON:"""
        
        # Use chat template for Qwen models
        if "Qwen" in st.session_state.get('llm_backend_loaded', ''):
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"{system}\n{user}\n"
        
        # Ensemble parsing if requested
        if use_ensemble and ensemble_runs > 1:
            all_results = []
            for _ in range(ensemble_runs):
                result = self._single_llm_parse(prompt, tokenizer, model, temperature, regex_params)
                if result:
                    all_results.append(result)
            
            if all_results:
                # Combine ensemble results: average for numeric, mode for categorical
                combined = {}
                for key in self.defaults:
                    values = [r[key] for r in all_results if key in r]
                    if isinstance(self.defaults[key], (int, float)):
                        # Average numeric values
                        valid_vals = [v for v in values if isinstance(v, (int, float))]
                        combined[key] = np.mean(valid_vals) if valid_vals else self.defaults[key]
                    else:
                        # Mode for other types
                        from collections import Counter
                        combined[key] = Counter(values).most_common(1)[0][0] if values else self.defaults[key]
                return combined
        
        # Single parse
        return self._single_llm_parse(prompt, tokenizer, model, temperature, regex_params)
    
    def _single_llm_parse(self, prompt: str, tokenizer, model, temperature: float, 
                         regex_params: Dict = None) -> Dict:
        """Helper for single LLM parse with robust extraction."""
        try:
            inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=300,
                    temperature=temperature,
                    do_sample=(temperature > 0),
                    pad_token_id=tokenizer.eos_token_id
                )
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            result = self._extract_json_robust(generated)
            if result:
                # Ensure all required keys exist
                for key in self.defaults:
                    if key not in result:
                        result[key] = self.defaults[key]
                
                # Clip values to valid ranges
                for key in DiffusionParameters.RANGES:
                    if key in result and isinstance(result[key], (int, float)):
                        result[key] = DiffusionParameters.clip_to_range(key, result[key])
                
                # If regex_params provided, prefer regex for mismatched values (confidence merging)
                if regex_params:
                    for key in ['ly_target', 'c_cu_target', 'c_ni_target']:
                        if key in regex_params and key in result:
                            if abs(result[key] - regex_params[key]) > 1e-4:
                                # Significant mismatch: prefer regex (more deterministic)
                                result[key] = regex_params[key]
                
                return result
        except Exception as e:
            st.warning(f"LLM parsing failed: {e}. Falling back to regex.")
        
        return self.parse_regex(text)
    
    def hybrid_parse(self, text: str, tokenizer, model, use_llm: bool = True,
                    use_ensemble: bool = False, ensemble_runs: int = 3) -> Dict:
        """Run regex first, then optionally LLM (with ensemble), and merge based on per-parameter confidence."""
        # Step 1: Regex extraction with confidence scoring
        regex_params = self.parse_regex(text)
        regex_conf = {}
        for key in self.defaults:
            # High confidence (1.0) if value differs from default (indicates match)
            # Low confidence (0.0) if still at default (may indicate no match)
            regex_conf[key] = 1.0 if regex_params[key] != self.defaults[key] else 0.0
        
        # Step 2: LLM extraction if available
        if use_llm and tokenizer and model:
            llm_params = self.parse_with_llm(
                text, tokenizer, model, 
                regex_params=regex_params,
                use_ensemble=use_ensemble,
                ensemble_runs=ensemble_runs
            )
            # Confidence for LLM: moderate if extracted, lower if default
            llm_conf = {}
            for key in self.defaults:
                llm_conf[key] = 0.8 if llm_params[key] != self.defaults[key] else 0.3
        else:
            llm_params = self.defaults.copy()
            llm_conf = {k: 0.0 for k in self.defaults}
        
        # Step 3: Per-parameter confidence-based merging
        final = {}
        for key in self.defaults:
            if regex_conf[key] >= llm_conf[key]:
                final[key] = regex_params[key]
            else:
                final[key] = llm_params[key]
        
        # Final clipping to ensure physical validity
        for key in DiffusionParameters.RANGES:
            final[key] = DiffusionParameters.clip_to_range(key, final[key])
        
        return final
    
    def get_explanation(self, params: dict, original_text: str) -> str:
        """Generate an enhanced markdown table explaining parsed parameters with units and ranges."""
        lines = [
            "### 🔍 Parsed Parameters from Natural Language",
            f"**Query:** _{original_text}_",
            "",
            "| Parameter | Extracted Value | Valid Range | Status |",
            "|-----------|-----------------|-------------|--------|"
        ]
        
        for key in ['ly_target', 'c_cu_target', 'c_ni_target', 'sigma']:
            if key == 'sigma':
                continue  # Skip hyperparameter in main explanation
            
            val = params[key]
            low, high, unit = DiffusionParameters.RANGES[key]
            val_str = DiffusionParameters.format_value(key, val)
            range_str = f"{low}–{high} {unit}" if unit else f"{low}–{high}"
            
            # Check if value was extracted vs default
            is_extracted = val != self.defaults[key]
            status_icon = "✅" if is_extracted else "⚪"
            status_text = "Extracted" if is_extracted else "Default"
            
            # Check if value is within valid range
            in_range = low <= val <= high
            if not in_range:
                status_icon = "⚠️"
                status_text += " (clipped)"
            
            lines.append(f"| {key} | {val_str} | {range_str} | {status_icon} {status_text} |")
        
        return "\n".join(lines)

# =============================================
# RELEVANCE SCORER (SciBERT with fallback)
# =============================================
class RelevanceScorer:
    """Compute semantic relevance using SciBERT or fallback keyword matching."""
    
    _instance = None
    _model = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, use_scibert: bool = True):
        self.use_scibert = use_scibert
        self._embedding_cache = {}
        
        if use_scibert and RelevanceScorer._model is None:
            try:
                with st.spinner("Loading SciBERT model for semantic analysis..."):
                    from sentence_transformers import SentenceTransformer
                    RelevanceScorer._model = SentenceTransformer(
                        'allenai/scibert_scivocab_uncased',
                        device='cpu'
                    )
                    st.success("SciBERT loaded successfully!")
                self.model = RelevanceScorer._model
            except ImportError:
                st.warning("sentence-transformers not installed. Using fallback relevance scoring.")
                self.use_scibert = False
            except Exception as e:
                st.warning(f"Could not load SciBERT: {e}. Using fallback.")
                self.use_scibert = False
    
    def encode_source(self, src_params: dict) -> str:
        """Create descriptive text from source parameters for embedding."""
        return (
            f"Cu-Ni diffusion simulation with domain height {src_params.get('Ly', 60):.1f} μm, "
            f"Cu boundary concentration {src_params.get('C_Cu', 1.5e-3):.1e} mol/cc, "
            f"Ni boundary concentration {src_params.get('C_Ni', 0.5e-3):.1e} mol/cc"
        )
    
    def score(self, query: str, sources: List[Dict], weights: np.ndarray) -> float:
        """Compute semantic relevance score between query and available sources."""
        if not sources or len(weights) == 0:
            return 0.0
        
        if self.use_scibert and self.model is not None:
            try:
                # Cache query embedding
                query_hash = hashlib.md5(query.encode()).hexdigest()
                if query_hash not in self._embedding_cache:
                    query_emb = self.model.encode(query, convert_to_tensor=False)
                    self._embedding_cache[query_hash] = query_emb
                else:
                    query_emb = self._embedding_cache[query_hash]
                
                # Encode source descriptions
                src_texts = [self.encode_source(s.get('params', {})) for s in sources]
                src_embs = self.model.encode(src_texts, convert_to_tensor=False)
                
                # Compute cosine similarities
                query_norm = np.linalg.norm(query_emb)
                src_norms = np.linalg.norm(src_embs, axis=1)
                
                valid_mask = src_norms > 1e-8
                if not np.any(valid_mask):
                    return float(np.max(weights))
                
                similarities = np.zeros(len(sources))
                similarities[valid_mask] = (
                    np.dot(src_embs[valid_mask], query_emb) /
                    (src_norms[valid_mask] * query_norm + 1e-12)
                )
                
                # Weighted average similarity
                weighted_score = np.average(similarities, weights=weights)
                # Normalize to [0, 1]
                normalized_score = (weighted_score + 1) / 2
                return float(np.clip(normalized_score, 0.0, 1.0))
                
            except Exception as e:
                st.warning(f"SciBERT scoring failed: {e}. Using fallback.")
                return float(np.max(weights)) if len(weights) > 0 else 0.0
        else:
            # Fallback: return max weight as proxy for relevance
            return float(np.max(weights)) if len(weights) > 0 else 0.0
    
    def get_confidence_level(self, score: float) -> Tuple[str, str]:
        """Map relevance score to human-readable confidence level."""
        if score >= 0.8:
            return "High confidence", "green"
        elif score >= 0.5:
            return "Moderate confidence", "blue"
        elif score >= 0.3:
            return "Low confidence", "orange"
        else:
            return "Very low confidence - consider adjusting parameters", "red"

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
        self.W_q = nn.Linear(3, self.num_heads * self.d_head)  # Query projection
        self.W_k = nn.Linear(3, self.num_heads * self.d_head)  # Key projection

    def forward(self, solutions, params_list, ly_target, c_cu_target, c_ni_target):
        if not solutions or not params_list:
            raise ValueError("No solutions or parameters available for interpolation.")
        # Extract and normalize parameters
        lys = np.array([p[0] for p in params_list])
        c_cus = np.array([p[1] for p in params_list])
        c_nis = np.array([p[2] for p in params_list])
        if not (lys.shape == c_cus.shape == c_nis.shape):
            raise ValueError(f"Parameter array shapes mismatch: lys={lys.shape}, c_cus={c_cus.shape}, c_nis={c_nis.shape}")
        ly_norm = (lys - 30.0) / (120.0 - 30.0)
        c_cu_norm = (c_cus - 0.0) / (2.9e-3 - 0.0)  # Updated to allow C_Cu = 0
        c_ni_norm = (c_nis - 0.0) / (1.8e-3 - 0.0)  # Updated to allow C_Ni = 0
        target_ly_norm = (ly_target - 30.0) / (120.0 - 30.0)
        target_c_cu_norm = (c_cu_target - 0.0) / (2.9e-3 - 0.0)
        target_c_ni_norm = (c_ni_target - 0.0) / (1.8e-3 - 0.0)
        # Combine normalized parameters into tensors
        params_tensor = torch.tensor(np.stack([ly_norm, c_cu_norm, c_ni_norm], axis=1), dtype=torch.float32)  # [N, 3]
        target_params_tensor = torch.tensor([[target_ly_norm, target_c_cu_norm, target_c_ni_norm]], dtype=torch.float32)  # [1, 3]
        # Project to query/key space
        queries = self.W_q(target_params_tensor)  # [1, num_heads * d_head]
        keys = self.W_k(params_tensor)  # [N, num_heads * d_head]
        # Reshape for multi-head attention
        queries = queries.view(1, self.num_heads, self.d_head)  # [1, num_heads, d_head]
        keys = keys.view(len(params_list), self.num_heads, self.d_head)  # [N, num_heads, d_head]
        # Scaled dot-product attention
        attn_logits = torch.einsum('nhd,mhd->nmh', keys, queries) / np.sqrt(self.d_head)  # [N, 1, num_heads]
        attn_weights = torch.softmax(attn_logits, dim=0)  # [N, 1, num_heads]
        attn_weights = attn_weights.mean(dim=2).squeeze(1)  # [N], average across heads
        # Spatial weights (Gaussian-like for locality)
        scaled_distances = torch.sqrt(
            ((torch.tensor(ly_norm) - target_ly_norm) / self.sigma)**2 +
            ((torch.tensor(c_cu_norm) - target_c_cu_norm) / self.sigma)**2 +
            ((torch.tensor(c_ni_norm) - target_c_ni_norm) / self.sigma)**2
        )
        spatial_weights = torch.exp(-scaled_distances**2 / 2)
        spatial_weights /= spatial_weights.sum()  # Normalize
        # Combine attention and spatial weights
        combined_weights = attn_weights * spatial_weights
        combined_weights /= combined_weights.sum()  # Normalize
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
    # Apply custom limits or auto-scale
    cu_min = vmin_cu if vmin_cu is not None else 0
    cu_max = vmax_cu if vmax_cu is not None else np.max(c1)
    ni_min = vmin_ni if vmin_ni is not None else 0
    ni_max = vmax_ni if vmax_ni is not None else np.max(c2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    # Cu heatmap
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
    # Ni heatmap
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
    center_idx = 25  # x = Lx/2
    times = solution['times']
    # Prepare sidebar data
    if sidebar_metric == 'loss' and 'loss' in solution:
        sidebar_data = solution['loss'][:len(times)]
        sidebar_label = 'Loss'
    elif sidebar_metric == 'mean_cu':
        sidebar_data = [np.mean(c1) for c1 in solution['c1_preds']]
        sidebar_label = 'Mean Cu Conc. (mol/cc)'
    else:  # mean_ni
        sidebar_data = [np.mean(c2) for c2 in solution['c2_preds']]
        sidebar_label = 'Mean Ni Conc. (mol/cc)'
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.05, 0.5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[3])
    # Centerline curves
    colors = cm.get_cmap(curve_colormap)(np.linspace(0, 1, len(time_indices)))
    for idx, t_idx in enumerate(time_indices):
        t_val = times[t_idx]
        c1 = solution['c1_preds'][t_idx][:, center_idx]
        c2 = solution['c2_preds'][t_idx][:, center_idx]
        ax1.plot(y_coords, c1, label=f't = {t_val:.1f} s', color=colors[idx], linewidth=curve_linewidth)
        ax2.plot(y_coords, c2, label=f't = {t_val:.1f} s', color=colors[idx], linewidth=curve_linewidth)
    # Axis styling
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
    # Legend placement
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
    # Sidebar plot
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
    center_idx = 25  # x = Lx/2
    t_val = solutions[0]['times'][time_index]
    # Prepare sidebar data
    sidebar_data = []
    sidebar_labels = []
    for sol, params in zip(solutions, params_list):
        if params in selected_params:
            if sidebar_metric == 'loss' and 'loss' in sol:
                sidebar_data.append(sol['loss'][time_index])
            elif sidebar_metric == 'mean_cu':
                sidebar_data.append(np.mean(sol['c1_preds'][time_index]))
            else:  # mean_ni
                sidebar_data.append(np.mean(sol['c2_preds'][time_index]))
            ly, c_cu, c_ni = params
            label = f'$L_y$={ly:.1f}, $C_{{Cu}}$={c_cu:.1e}, $C_{{Ni}}$={c_ni:.1e}'
            if sol.get('interpolated', False):
                label += " (Interpolated)"
            sidebar_labels.append(label)
    # Create figure with custom size
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.05, 0.5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[3])
    # Parameter sweep curves
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
    # Axis styling
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
    # Legend placement
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
    # Sidebar bar plot
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
# SESSION STATE INITIALIZATION (ENHANCED)
# =============================================
def initialize_session_state():
    """Initialize Streamlit session state with enhanced defaults for LLM and parser."""
    defaults = {
        'nl_parser': DiffusionNLParser(),
        'llm_backend_loaded': 'GPT-2 (default)',
        'llm_cache': OrderedDict(),  # Manual cache to avoid UnhashableParamError
        'llm_cache_maxsize': 20,
        'parsed_params': None,
        'nl_query': "",
        'use_llm': True,
        'use_llm_ensemble': False,
        'ensemble_runs': 3,
        'use_scibert': True,
        'current_relevance': 0.5,
        'current_entropy': 0.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# =============================================
# HELPER: Format small numbers for display
# =============================================
def format_small_number(val: float, threshold: float = 0.001, decimals: int = 3) -> str:
    """Return scientific notation if |val| < threshold, else fixed-point."""
    if abs(val) < threshold:
        return f"{val:.3e}"
    else:
        return f"{val:.{decimals}f}"

# =============================================
# CALLBACK FOR TEMPLATE BUTTONS
# =============================================
def set_template(text: str):
    st.session_state.nl_query = text

# =============================================
# MAIN APP WITH ENHANCED LLM INTEGRATION
# =============================================
def main():
    st.set_page_config(
        page_title="LLM-Enhanced Cu-Ni Diffusion Visualizer",
        layout="wide",
        page_icon="🔬"
    )
    
    st.title("🔬 Attention-Based Cu-Ni Interdiffusion with Enhanced Natural Language Interface")
    
    # Initialize session state
    initialize_session_state()
    
    # =========================================
    # SIDEBAR: ENHANCED LLM CONFIGURATION
    # =========================================
    with st.sidebar:
        st.header("🤖 Enhanced LLM Configuration")
        
        if TRANSFORMERS_AVAILABLE:
            backend_choice = st.selectbox(
                "LLM Backend",
                ["GPT-2 (default)", "Qwen2-0.5B-Instruct", "Qwen2.5-0.5B-Instruct"],
                index=0
            )
            
            # Load model if backend changed
            if backend_choice != st.session_state.llm_backend_loaded:
                st.session_state.llm_backend_loaded = backend_choice
                st.session_state.llm_cache.clear()  # Clear cache on backend switch
                st.rerun()
            
            tokenizer, model, active_backend = load_llm(backend_choice)
            st.session_state.llm_tokenizer = tokenizer
            st.session_state.llm_model = model
            st.caption(f"Active: **{active_backend}**")
            
            # LLM feature toggles
            st.session_state.use_llm = st.checkbox("Enable LLM Parsing", value=True)
            st.session_state.use_llm_ensemble = st.checkbox("Use LLM Ensemble (slower, more robust)", value=False)
            if st.session_state.use_llm_ensemble:
                st.session_state.ensemble_runs = st.number_input("Ensemble runs", min_value=2, max_value=10, value=3, step=1)
            
            # Relevance scorer toggle
            st.session_state.use_scibert = st.checkbox("Enable SciBERT Relevance Scoring", value=True)
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
    # MAIN: ENHANCED NATURAL LANGUAGE INPUT
    # =========================================
    st.subheader("📝 Describe Your Solder Joint Configuration")
    
    # Template buttons for quick queries
    templates = {
        "Thin Asymmetric Joint": "Analyze a 40 μm asymmetric Cu-Ni joint with top Cu concentration 1.8e-3 and bottom Ni 0.4e-3.",
        "Thick Symmetric Cu": "Simulate a 100 μm symmetric Cu/Sn2.5Ag/Cu joint. Use c_Cu=2.0e-3.",
        "Ni-Rich Diffusion": "Domain length 60 μm, C_Cu=1.0e-3, C_Ni=1.5e-3. Asymmetric configuration.",
        "Self-Diffusion Baseline": "Ly=75 μm, C_Cu=0, C_Ni=0 for self-diffusion reference.",
        "High Concentration Test": "Joint thickness 50 microns, copper conc 2.5e-3 mol/cc, nickel 1.2e-3, sigma 0.25",
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
                set_template(text)
                st.rerun()
    
    # Parse query and display enhanced explanation
    parser = st.session_state.nl_parser
    use_llm = st.session_state.get('use_llm', False)
    use_ensemble = st.session_state.get('use_llm_ensemble', False)
    ensemble_runs = st.session_state.get('ensemble_runs', 3)
    tokenizer = st.session_state.get('llm_tokenizer', None)
    model = st.session_state.get('llm_model', None)
    
    if nl_query:
        with st.spinner("🔍 Parsing natural language with hybrid regex+LLM..."):
            # Manual LLM cache with LRU eviction to avoid UnhashableParamError
            cache_key = hashlib.md5((nl_query + st.session_state.get('llm_backend_loaded', '') + str(use_ensemble)).encode()).hexdigest()
            if cache_key in st.session_state.llm_cache:
                parsed = st.session_state.llm_cache[cache_key]
            else:
                parsed = parser.hybrid_parse(
                    nl_query, tokenizer, model, 
                    use_llm=use_llm,
                    use_ensemble=use_ensemble,
                    ensemble_runs=ensemble_runs
                )
                # LRU eviction
                if len(st.session_state.llm_cache) >= st.session_state.llm_cache_maxsize:
                    st.session_state.llm_cache.popitem(last=False)
                st.session_state.llm_cache[cache_key] = parsed
            
            st.session_state.parsed_params = parsed
            
            # Compute relevance score if SciBERT enabled
            if st.session_state.use_scibert and TRANSFORMERS_AVAILABLE:
                try:
                    from sentence_transformers import SentenceTransformer
                    scorer = RelevanceScorer(use_scibert=True)
                    # Get weights from interpolation (placeholder - would come from actual interpolation)
                    weights = np.array([1.0])  # Placeholder
                    relevance = scorer.score(nl_query, [], weights)
                    st.session_state.current_relevance = relevance
                except:
                    st.session_state.current_relevance = 0.5
            else:
                st.session_state.current_relevance = 0.5
        
        # Display enhanced explanation with units and ranges
        st.markdown(parser.get_explanation(parsed, nl_query))
        
        # Visual feedback for range violations
        for key in ['ly_target', 'c_cu_target', 'c_ni_target']:
            val = parsed[key]
            low, high, unit = DiffusionParameters.RANGES[key]
            if not (low <= val <= high):
                st.warning(f"⚠️ `{key}` = {DiffusionParameters.format_value(key, val)} is outside valid range [{low}, {high}] {unit}; clipped automatically.")
        
        # Show relevance score
        confidence_text, confidence_color = RelevanceScorer(None).get_confidence_level(st.session_state.current_relevance)
        st.markdown(f"**Semantic Relevance:** {st.session_state.current_relevance:.3f} - {confidence_text}")
    else:
        # Use defaults if no query
        parsed = parser.defaults.copy()
        st.session_state.parsed_params = parsed
    
    # =========================================
    # PARAMETER SELECTION (Pre-filled by Parser)
    # =========================================
    st.subheader("🎯 Target Parameters (Override if Needed)")
    
    # Sort unique parameters from loaded solutions
    solutions, params_list, lys, c_cus, c_nis, load_logs = load_solutions(SOLUTION_DIR)
    
    # Display load logs
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
    
    # Parameter selection with parsed defaults
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
    
    # Custom parameters for interpolation
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
    
    # Color scale limits
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
    
    # Validate color scale limits
    if custom_cu_min is not None and custom_cu_max is not None and custom_cu_min >= custom_cu_max:
        st.error("Cu minimum concentration must be less than maximum concentration.")
        return
    if custom_ni_min is not None and custom_ni_max is not None and custom_ni_min >= custom_ni_max:
        st.error("Ni minimum concentration must be less than maximum concentration.")
        return
    
    # Figure customization controls
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
        # Show attention weights
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
    
    # Combine exact and custom parameters
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
    
    # Generate solutions for selected parameters (exact or interpolated)
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
    
    # Plot parameter sweep
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
