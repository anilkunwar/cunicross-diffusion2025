#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ATTENTION-BASED CU-NI INTERDIFFUSION VISUALIZER WITH LLM NATURAL LANGUAGE INTERFACE
====================================================================================
EXPANDED VERSION: Full parameter mapping, validation, and extraction pipeline

KEY FEATURES:
- Natural language parsing with regex + LLM hybrid extraction
- EXPLICIT PARAMETER MAPPING: extracted_values → validated_params → target_params
- Multi-head attention with spatial locality for physics-aware interpolation
- Publication-quality 2D heatmaps, centerline curves, and parameter sweeps
- Full figure customization and PNG/PDF export
- Cached LLM loading, robust JSON extraction, confidence-based fallbacks
- EXTRACTABLE CONCENTRATION FEATURES: profiles, gradients, integrated totals, CSV export
- Parameter validation with domain constraints and unit normalization
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
from typing import Dict, Any, Optional, List, Tuple, Union

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

# =============================================
# PARAMETER SCHEMA & CONSTRAINTS
# =============================================
class ParameterSchema:
    """Defines parameter names, units, constraints, and mapping rules."""
    
    # Canonical parameter names (internal use)
    CANONICAL_NAMES = {
        'ly_target': 'domain_height',
        'c_cu_target': 'cu_boundary_concentration', 
        'c_ni_target': 'ni_boundary_concentration',
        'sigma': 'attention_locality_sigma'
    }
    
    # Parameter constraints: (min, max, default, unit)
    CONSTRAINTS = {
        'ly_target': (30.0, 120.0, 60.0, 'μm'),
        'c_cu_target': (0.0, 2.9e-3, 1.5e-3, 'mol/cc'),
        'c_ni_target': (0.0, 1.8e-3, 0.5e-3, 'mol/cc'),
        'sigma': (0.05, 0.50, 0.20, 'dimensionless')
    }
    
    # Unit conversion factors (to internal units)
    UNIT_CONVERSIONS = {
        'ly_target': {'μm': 1.0, 'um': 1.0, 'micron': 1.0, 'microns': 1.0, 'mm': 1000.0},
        'c_cu_target': {'mol/cc': 1.0, 'mol/cm³': 1.0, 'M': 1.0},  # M = mol/L = 1e-3 mol/cc
        'c_ni_target': {'mol/cc': 1.0, 'mol/cm³': 1.0, 'M': 1.0}
    }
    
    @classmethod
    def validate(cls, param_name: str, value: float) -> Tuple[bool, str, float]:
        """Validate parameter value against constraints. Returns (is_valid, message, clipped_value)."""
        if param_name not in cls.CONSTRAINTS:
            return False, f"Unknown parameter: {param_name}", value
        
        min_val, max_val, default, unit = cls.CONSTRAINTS[param_name]
        
        if not isinstance(value, (int, float)):
            return False, f"Value must be numeric, got {type(value).__name__}", default
        
        if value < min_val or value > max_val:
            clipped = np.clip(value, min_val, max_val)
            return False, f"Value {value} {unit} out of range [{min_val}, {max_val}] {unit}, clipped to {clipped}", clipped
        
        return True, "Valid", value
    
    @classmethod
    def convert_units(cls, param_name: str, value: float, from_unit: str) -> float:
        """Convert value from specified unit to internal unit."""
        if param_name not in cls.UNIT_CONVERSIONS:
            return value
        conversions = cls.UNIT_CONVERSIONS[param_name]
        factor = conversions.get(from_unit.lower(), 1.0)
        return value * factor


# =============================================
# AVAILABLE COLORMAPS
# =============================================
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
# ENHANCED NATURAL LANGUAGE PARSER WITH EXPLICIT MAPPING
# =============================================
class DiffusionNLParser:
    """
    Extracts Cu-Ni diffusion parameters from natural language using regex + LLM hybrid parsing.
    
    EXPLICIT PARAMETER MAPPING FLOW:
    1. Raw text → regex patterns → extracted_values (with confidence scores)
    2. extracted_values → unit conversion → normalized_values
    3. normalized_values → constraint validation → validated_params
    4. validated_params → target_params (with fallback to defaults)
    """
    
    def __init__(self):
        # Default parameter values (used when extraction fails)
        self.defaults = {
            'ly_target': 60.0,
            'c_cu_target': 1.5e-3,
            'c_ni_target': 0.5e-3,
            'sigma': 0.20,
        }
        
        # Regex patterns for parameter extraction
        # Format: {canonical_name: [pattern1, pattern2, ...]}
        # Patterns capture: (value, optional_unit)
        self.patterns = {
            'ly_target': [
                # "domain length 80", "Ly=45um", "joint thickness: 100 μm"
                r'(?:joint\s*thickness|domain\s*length|domain\s*height|L_y|Ly)\s*[=:\s]*\s*(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(μm|um|microns?|mm)?',
                # Standalone value with unit: "80 μm joint"
                r'(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(μm|um|microns?|mm)\s*(?:joint|domain|length)?',
            ],
            'c_cu_target': [
                # "C_Cu=2.0e-3", "Cu concentration: 1.5e-3 mol/cc"
                r'(?:Cu\s*concentration|C_Cu|c_Cu|top\s*concentration|Cu\s*boundary)\s*[=:\s]*\s*([\d.]+(?:e[+-]?\d+)?)\s*(mol/cc|mol/cm³|M)?',
                # Simple: "Cu 1.2e-3"
                r'(?<!\w)Cu\s*[=:\s]*\s*([\d.]+(?:e[+-]?\d+)?)\s*(mol/cc|mol/cm³|M)?',
            ],
            'c_ni_target': [
                # "C_Ni=1.0e-3", "Ni concentration: 0.8e-3 mol/cc"
                r'(?:Ni\s*concentration|C_Ni|c_Ni|bottom\s*concentration|Ni\s*boundary)\s*[=:\s]*\s*([\d.]+(?:e[+-]?\d+)?)\s*(mol/cc|mol/cm³|M)?',
                # Simple: "Ni 0.5e-3"
                r'(?<!\w)Ni\s*[=:\s]*\s*([\d.]+(?:e[+-]?\d+)?)\s*(mol/cc|mol/cm³|M)?',
            ],
            'sigma': [
                r'(?:attention\s*sigma|locality\s*σ|sigma)\s*[=:\s]*\s*([\d.]+(?:e[+-]?\d+)?)',
            ]
        }
        
        # Aliases for parameter names in natural language
        self.aliases = {
            'ly_target': ['domain length', 'joint thickness', 'Ly', 'L_y', 'domain height', 'height'],
            'c_cu_target': ['Cu concentration', 'C_Cu', 'c_Cu', 'top concentration', 'Cu boundary'],
            'c_ni_target': ['Ni concentration', 'C_Ni', 'c_Ni', 'bottom concentration', 'Ni boundary'],
        }

    def parse_regex(self, text: str) -> Dict[str, Any]:
        """
        Extract parameters using flexible regex patterns with explicit mapping.
        
        Returns dict with: {param_name: {'value': float, 'unit': str, 'confidence': float, 'source': str}}
        """
        if not text:
            return {key: {'value': val, 'unit': ParameterSchema.CONSTRAINTS[key][3], 
                         'confidence': 0.0, 'source': 'default'} 
                   for key, val in self.defaults.items()}
        
        extracted = {}
        text_lower = text.lower()
        
        for param_name, patterns in self.patterns.items():
            for pattern_idx, pat in enumerate(patterns):
                match = re.search(pat, text_lower, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    value_str = groups[0]
                    unit = groups[1] if len(groups) > 1 and groups[1] else ParameterSchema.CONSTRAINTS[param_name][3]
                    
                    try:
                        # Convert to float (handles scientific notation)
                        raw_value = float(value_str)
                        
                        # Unit conversion to internal units
                        converted_value = ParameterSchema.convert_units(param_name, raw_value, unit)
                        
                        # Validation and clipping
                        is_valid, message, clipped_value = ParameterSchema.validate(param_name, converted_value)
                        
                        extracted[param_name] = {
                            'value': clipped_value,
                            'unit': ParameterSchema.CONSTRAINTS[param_name][3],  # Internal unit
                            'original_value': raw_value,
                            'original_unit': unit,
                            'confidence': 0.9 if pattern_idx == 0 else 0.7,  # Higher confidence for primary patterns
                            'source': f'regex_pattern_{pattern_idx + 1}',
                            'validation_message': message,
                            'is_valid': is_valid
                        }
                        break  # Use first successful match for this parameter
                    except (ValueError, TypeError) as e:
                        continue  # Try next pattern
        
        # Fill missing parameters with defaults
        for param_name in self.defaults:
            if param_name not in extracted:
                extracted[param_name] = {
                    'value': self.defaults[param_name],
                    'unit': ParameterSchema.CONSTRAINTS[param_name][3],
                    'confidence': 0.0,
                    'source': 'default',
                    'validation_message': 'Not extracted, using default',
                    'is_valid': True
                }
        
        return extracted

    @staticmethod
    def _extract_json_robust(generated: str) -> Optional[Dict]:
        """Robustly extract JSON from LLM output with repair attempts."""
        # Try nested JSON first
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        match = re.search(json_pattern, generated, re.DOTALL)
        if not match:
            # Fallback to simple JSON
            match = re.search(r'\{.*?\}', generated, re.DOTALL)
        if not match:
            return None
        
        json_str = match.group(0)
        
        # Repair common JSON issues
        json_str = re.sub(r'(true|false|null)\s*(")', r'\1,\2', json_str)  # Fix missing commas
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)  # Remove trailing commas
        json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)  # Convert single to double quotes
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Last resort: try to extract key-value pairs manually
            try:
                pairs = re.findall(r'"?(\w+)"?\s*[:=]\s*"?([\d.e+-]+)"?', json_str)
                return {k: float(v) if '.' in v or 'e' in v.lower() else int(v) for k, v in pairs}
            except:
                return None

    def parse_with_llm(self, text: str, tokenizer, model, 
                      regex_extracted: Dict = None, 
                      temperature: float = None) -> Dict[str, Any]:
        """
        Use LLM (GPT-2 or Qwen) to extract parameters from natural language.
        
        Returns dict with same structure as parse_regex for consistent mapping.
        """
        if not tokenizer or not model:
            return self.parse_regex(text)
        
        if temperature is None:
            backend = st.session_state.get('llm_backend_loaded', 'GPT-2')
            temperature = 0.0 if "Qwen" in backend else 0.1
        
        system_prompt = """You are a materials science expert extracting simulation parameters.
Reply ONLY with a valid JSON object using these EXACT keys:
- ly_target: float (domain height in μm, range 30-120)
- c_cu_target: float (Cu boundary concentration in mol/cc, range 0-2.9e-3)  
- c_ni_target: float (Ni boundary concentration in mol/cc, range 0-1.8e-3)
- sigma: float (attention locality parameter, range 0.05-0.5)

Use scientific notation for small numbers (e.g., 2.0e-3).
If a parameter is not mentioned, use these defaults:
{"ly_target": 60.0, "c_cu_target": 1.5e-3, "c_ni_target": 0.5e-3, "sigma": 0.2}
"""
        
        examples = """
Examples:
Input: "Analyze a 50 μm joint with Cu concentration 1.2e-3 and Ni 0.8e-3"
Output: {"ly_target": 50.0, "c_cu_target": 1.2e-3, "c_ni_target": 0.8e-3, "sigma": 0.2}

Input: "Domain length 80, C_Cu=2.0e-3, C_Ni=1.0e-3"
Output: {"ly_target": 80.0, "c_cu_target": 2.0e-3, "c_ni_target": 1.0e-3, "sigma": 0.2}

Input: "Ly=45um, top Cu=1.5e-3 mol/cc, bottom Ni=0.3e-3"
Output: {"ly_target": 45.0, "c_cu_target": 1.5e-3, "c_ni_target": 0.3e-3, "sigma": 0.2}
"""
        
        regex_hint = ""
        if regex_extracted:
            hint_values = {k: v['value'] for k, v in regex_extracted.items()}
            regex_hint = f"\nRegex extraction hint (use as reference if helpful): {json.dumps(hint_values)}"
        
        user_prompt = f"""{examples}{regex_hint}
User query: "{text}"
JSON output:"""
        
        # Format prompt for specific model
        if "Qwen" in st.session_state.get('llm_backend_loaded', ''):
            messages = [
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": user_prompt}
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"{system_prompt}\n{user_prompt}\n"
        
        try:
            # Tokenize and generate
            inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=200,
                    temperature=temperature,
                    do_sample=(temperature > 0),
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract and parse JSON
            llm_result = self._extract_json_robust(generated)
            
            if llm_result:
                # Convert LLM output to our structured format with validation
                structured = {}
                for param_name in self.defaults:
                    if param_name in llm_result:
                        raw_value = llm_result[param_name]
                        # Validate and clip
                        is_valid, message, clipped_value = ParameterSchema.validate(param_name, raw_value)
                        structured[param_name] = {
                            'value': clipped_value,
                            'unit': ParameterSchema.CONSTRAINTS[param_name][3],
                            'original_value': raw_value,
                            'confidence': 0.95 if is_valid else 0.5,
                            'source': 'llm',
                            'validation_message': message,
                            'is_valid': is_valid
                        }
                    else:
                        # Use default if not provided by LLM
                        structured[param_name] = {
                            'value': self.defaults[param_name],
                            'unit': ParameterSchema.CONSTRAINTS[param_name][3],
                            'confidence': 0.0,
                            'source': 'default',
                            'validation_message': 'Not provided by LLM, using default',
                            'is_valid': True
                        }
                
                # If regex extraction was provided, prefer high-confidence regex values
                if regex_extracted:
                    for param_name in structured:
                        regex_val = regex_extracted.get(param_name, {})
                        llm_val = structured[param_name]
                        
                        # Prefer regex if it has higher confidence and is valid
                        if (regex_val.get('confidence', 0) > llm_val.get('confidence', 0) and 
                            regex_val.get('is_valid', False)):
                            structured[param_name] = regex_val
                
                return structured
                
        except Exception as e:
            st.warning(f"LLM parsing failed: {e}. Falling back to regex extraction.")
        
        # Fallback to regex
        return self.parse_regex(text)

    def merge_extractions(self, regex_extracted: Dict, llm_extracted: Dict) -> Dict[str, Any]:
        """
        Merge regex and LLM extractions using confidence-based selection.
        
        Priority: 
        1. High-confidence valid extractions (>0.8)
        2. Medium-confidence valid extractions (0.5-0.8)  
        3. Defaults for missing/invalid parameters
        """
        merged = {}
        
        for param_name in self.defaults:
            regex_val = regex_extracted.get(param_name, {})
            llm_val = llm_extracted.get(param_name, {})
            
            # Score each extraction: confidence * validity_bonus
            regex_score = regex_val.get('confidence', 0) * (1.0 if regex_val.get('is_valid', False) else 0.5)
            llm_score = llm_val.get('confidence', 0) * (1.0 if llm_val.get('is_valid', False) else 0.5)
            
            # Select best extraction
            if regex_score >= llm_score and regex_score > 0:
                merged[param_name] = regex_val
            elif llm_score > 0:
                merged[param_name] = llm_val
            else:
                # Use default
                merged[param_name] = {
                    'value': self.defaults[param_name],
                    'unit': ParameterSchema.CONSTRAINTS[param_name][3],
                    'confidence': 0.0,
                    'source': 'default',
                    'validation_message': 'No valid extraction, using default',
                    'is_valid': True
                }
        
        return merged

    def get_target_params(self, extracted: Dict[str, Any]) -> Dict[str, float]:
        """
        Convert structured extraction to simple target parameter dict for simulation.
        
        This is the critical mapping step: extracted (with metadata) → target (clean values)
        """
        target = {}
        for param_name in self.defaults:
            param_data = extracted.get(param_name, {})
            # Always use the validated, clipped value
            target[param_name] = param_data.get('value', self.defaults[param_name])
        return target

    def hybrid_parse(self, text: str, tokenizer, model, use_llm: bool = True) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Run full extraction pipeline: regex → optional LLM → merge → target params.
        
        Returns: (structured_extraction, target_params_dict)
        """
        # Step 1: Regex extraction
        regex_extracted = self.parse_regex(text)
        
        # Step 2: Optional LLM extraction
        if use_llm and tokenizer and model:
            # Cache LLM results to avoid repeated calls
            cache_key = hashlib.md5((text + st.session_state.get('llm_backend_loaded', '')).encode()).hexdigest()
            if 'llm_cache' not in st.session_state:
                st.session_state.llm_cache = OrderedDict()
            
            if cache_key in st.session_state.llm_cache:
                llm_extracted = st.session_state.llm_cache[cache_key]
            else:
                llm_extracted = self.parse_with_llm(text, tokenizer, model, regex_extracted)
                # Cache management
                if len(st.session_state.llm_cache) > 20:
                    st.session_state.llm_cache.popitem(last=False)
                st.session_state.llm_cache[cache_key] = llm_extracted
            
            # Step 3: Merge extractions
            merged = self.merge_extractions(regex_extracted, llm_extracted)
        else:
            merged = regex_extracted
        
        # Step 4: Convert to target params
        target_params = self.get_target_params(merged)
        
        return merged, target_params

    def get_explanation(self, structured: Dict[str, Any], original_text: str) -> str:
        """Generate a markdown table explaining parsed parameters with mapping details."""
        lines = [
            "### 🔍 Parsed Parameters from Natural Language", 
            f"**Query:** _{original_text}_", 
            "",
            "| Parameter | Extracted Value | Unit | Status | Confidence | Source |",
            "|-----------|----------------|------|--------|------------|--------|"
        ]
        
        for param_name in ['ly_target', 'c_cu_target', 'c_ni_target']:  # Exclude sigma for brevity
            param_data = structured.get(param_name, {})
            value = param_data.get('value', self.defaults[param_name])
            unit = param_data.get('unit', ParameterSchema.CONSTRAINTS[param_name][3])
            confidence = param_data.get('confidence', 0.0)
            source = param_data.get('source', 'unknown')
            is_valid = param_data.get('is_valid', True)
            validation_msg = param_data.get('validation_message', '')
            
            # Format value for display
            if isinstance(value, float):
                if value < 0.01 or value > 100:
                    val_str = f"{value:.1e}"
                else:
                    val_str = f"{value:.3f}"
            else:
                val_str = str(value)
            
            # Status indicator
            if confidence > 0.8 and is_valid:
                status = "✅ Extracted"
            elif confidence > 0.5 and is_valid:
                status = "⚠️ Low confidence"
            elif not is_valid:
                status = f"❌ Invalid: {validation_msg[:30]}..."
            else:
                status = "⚪ Default"
            
            lines.append(f"| {param_name} | {val_str} | {unit} | {status} | {confidence:.2f} | {source} |")
        
        return "\n".join(lines)


# =============================================
# UNIFIED LLM LOADER WITH CACHING
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
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2-0.5B-Instruct", 
                torch_dtype="auto", 
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
        elif "Qwen2.5" in backend_name:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-0.5B-Instruct",
                torch_dtype="auto",
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
        else:
            return None, None, "Unknown Backend"
        
        model.eval()
        return tokenizer, model, backend_name
        
    except Exception as e:
        st.error(f"Failed to load LLM backend '{backend_name}': {e}")
        return None, None, f"Error: {backend_name}"


# =============================================
# SOLUTION LOADING WITH PARAMETER TRACKING
# =============================================
@st.cache_data
def load_solutions(solution_dir):
    """Load precomputed PINN solutions with parameter metadata."""
    solutions = []
    params_list = []
    load_logs = []
    lys = []
    c_cus = []
    c_nis = []
    
    for fname in sorted(os.listdir(solution_dir)):
        if fname.endswith(".pkl"):
            try:
                with open(os.path.join(solution_dir, fname), "rb") as f:
                    sol = pickle.load(f)
                
                # Validate required keys
                required_keys = ['params', 'X', 'Y', 'c1_preds', 'c2_preds', 'times']
                if not all(key in sol for key in required_keys):
                    missing = [k for k in required_keys if k not in sol]
                    load_logs.append(f"{fname}: ❌ Missing keys: {missing}")
                    continue
                
                # Validate data quality
                if (np.any(np.isnan(sol['c1_preds'])) or np.any(np.isnan(sol['c2_preds'])) or
                        np.all(sol['c1_preds'] == 0) or np.all(sol['c2_preds'] == 0)):
                    load_logs.append(f"{fname}: ❌ Invalid data (NaNs or all zeros)")
                    continue
                
                # Extract parameter tuple for indexing
                params = sol['params']
                ly = params.get('Ly', params.get('ly_target', 60.0))
                c_cu = params.get('C_Cu', params.get('c_cu_target', 1.5e-3))
                c_ni = params.get('C_Ni', params.get('c_ni_target', 0.5e-3))
                
                param_tuple = (float(ly), float(c_cu), float(c_ni))
                
                # Log concentration ranges for debugging
                c1_min, c1_max = np.min(sol['c1_preds'][0]), np.max(sol['c1_preds'][0])
                c2_min, c2_max = np.min(sol['c2_preds'][0]), np.max(sol['c2_preds'][0])
                
                solutions.append(sol)
                params_list.append(param_tuple)
                lys.append(param_tuple[0])
                c_cus.append(param_tuple[1])
                c_nis.append(param_tuple[2])
                
                load_logs.append(
                    f"{fname}: ✅ Loaded | Cu: [{c1_min:.2e}, {c1_max:.2e}], Ni: [{c2_min:.2e}, {c2_max:.2e}] | "
                    f"Ly={param_tuple[0]:.1f}μm, C_Cu={param_tuple[1]:.1e}, C_Ni={param_tuple[2]:.1e}"
                )
                
            except Exception as e:
                load_logs.append(f"{fname}: ❌ Load error: {str(e)}")
    
    # Summary
    if len(solutions) < 1:
        load_logs.append("🚨 ERROR: No valid solutions loaded. Interpolation will fail.")
    else:
        load_logs.append(f"📊 Loaded {len(solutions)}/32 expected solutions")
    
    return solutions, params_list, lys, c_cus, c_nis, load_logs


# =============================================
# MULTI-PARAMETER ATTENTION INTERPOLATOR
# =============================================
class MultiParamAttentionInterpolator(nn.Module):
    """
    Physics-aware interpolation using multi-head attention with spatial locality.
    
    Maps target parameters to weighted combination of precomputed solutions.
    """
    
    def __init__(self, sigma=0.2, num_heads=4, d_head=8):
        super().__init__()
        self.sigma = sigma
        self.num_heads = num_heads
        self.d_head = d_head
        
        # Learnable projection layers for attention
        self.W_q = nn.Linear(3, self.num_heads * self.d_head)  # Query: target params
        self.W_k = nn.Linear(3, self.num_heads * self.d_head)  # Key: solution params
        
    def forward(self, solutions, params_list, ly_target, c_cu_target, c_ni_target):
        """
        Interpolate solutions for target parameters using attention mechanism.
        
        Args:
            solutions: List of precomputed solution dicts
            params_list: List of (Ly, C_Cu, C_Ni) tuples for each solution
            ly_target, c_cu_target, c_ni_target: Target parameter values
            
        Returns:
            Interpolated solution dict with same structure as input solutions
        """
        if not solutions or not params_list:
            raise ValueError("No solutions or parameters available for interpolation.")
        
        # Extract parameter arrays
        lys = np.array([p[0] for p in params_list])
        c_cus = np.array([p[1] for p in params_list])
        c_nis = np.array([p[2] for p in params_list])
        
        # Normalize parameters to [0, 1] for attention computation
        def normalize(val, min_val, max_val):
            return (val - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        
        lys_norm = normalize(lys, 30.0, 120.0)
        c_cus_norm = normalize(c_cus, 0.0, 2.9e-3)
        c_nis_norm = normalize(c_nis, 0.0, 1.8e-3)
        
        target_ly_norm = normalize(ly_target, 30.0, 120.0)
        target_c_cu_norm = normalize(c_cu_target, 0.0, 2.9e-3)
        target_c_ni_norm = normalize(c_ni_target, 0.0, 1.8e-3)
        
        # Create tensors for attention
        params_tensor = torch.tensor(
            np.stack([lys_norm, c_cus_norm, c_nis_norm], axis=1), 
            dtype=torch.float32
        )
        target_tensor = torch.tensor(
            [[target_ly_norm, target_c_cu_norm, target_c_ni_norm]], 
            dtype=torch.float32
        )
        
        # Compute attention scores
        queries = self.W_q(target_tensor)  # [1, num_heads * d_head]
        keys = self.W_k(params_tensor)     # [n_solutions, num_heads * d_head]
        
        queries = queries.view(1, self.num_heads, self.d_head)
        keys = keys.view(len(params_list), self.num_heads, self.d_head)
        
        # Scaled dot-product attention
        attn_logits = torch.einsum('nhd,mhd->nmh', keys, queries) / np.sqrt(self.d_head)
        attn_weights = torch.softmax(attn_logits, dim=0)
        attn_weights = attn_weights.mean(dim=2).squeeze(1)  # Average over heads
        
        # Spatial locality weights (Gaussian kernel)
        scaled_distances = torch.sqrt(
            ((torch.tensor(lys_norm) - target_ly_norm) / self.sigma)**2 +
            ((torch.tensor(c_cus_norm) - target_c_cu_norm) / self.sigma)**2 +
            ((torch.tensor(c_nis_norm) - target_c_ni_norm) / self.sigma)**2
        )
        spatial_weights = torch.exp(-scaled_distances**2 / 2)
        spatial_weights /= spatial_weights.sum()
        
        # Combine attention and spatial weights
        combined_weights = attn_weights * spatial_weights
        combined_weights /= combined_weights.sum()
        
        return self._physics_aware_interpolation(
            solutions, combined_weights.detach().numpy(), 
            ly_target, c_cu_target, c_ni_target
        )
    
    def _physics_aware_interpolation(self, solutions, weights, ly_target, c_cu_target, c_ni_target):
        """
        Interpolate concentration fields with physics-aware boundary enforcement.
        """
        # Get grid dimensions from first solution
        Lx = solutions[0]['params']['Lx']
        t_max = solutions[0]['params']['t_max']
        
        # Create interpolation grid
        x_coords = np.linspace(0, Lx, 50)
        y_coords = np.linspace(0, ly_target, 50)
        times = np.linspace(0, t_max, 50)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Initialize interpolated fields
        c1_interp = np.zeros((len(times), 50, 50))  # Cu
        c2_interp = np.zeros((len(times), 50, 50))  # Ni
        
        # Weighted interpolation with coordinate scaling
        for t_idx in range(len(times)):
            for sol, weight in zip(solutions, weights):
                # Scale y-coordinates to match target domain height
                scale_factor = ly_target / sol['params']['Ly']
                Y_scaled = sol['Y'][0, :] * scale_factor
                
                # Create interpolators for this time step
                interp_c1 = RegularGridInterpolator(
                    (sol['X'][:, 0], Y_scaled), 
                    sol['c1_preds'][t_idx],
                    method='linear', 
                    bounds_error=False, 
                    fill_value=0
                )
                interp_c2 = RegularGridInterpolator(
                    (sol['X'][:, 0], Y_scaled), 
                    sol['c2_preds'][t_idx],
                    method='linear',
                    bounds_error=False,
                    fill_value=0
                )
                
                # Evaluate at target grid points
                points = np.stack([X.flatten(), Y.flatten()], axis=1)
                c1_interp[t_idx] += weight * interp_c1(points).reshape(50, 50)
                c2_interp[t_idx] += weight * interp_c2(points).reshape(50, 50)
        
        # Enforce boundary conditions (physics-aware)
        c1_interp[:, :, 0] = c_cu_target  # Top boundary: fixed Cu concentration
        c2_interp[:, :, -1] = c_ni_target  # Bottom boundary: fixed Ni concentration
        
        # Construct output solution dict
        param_set = solutions[0]['params'].copy()
        param_set.update({
            'Ly': ly_target,
            'C_Cu': c_cu_target,
            'C_Ni': c_ni_target
        })
        
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
# INTERPOLATION WRAPPER WITH EXACT MATCH CHECK
# =============================================
@st.cache_data
def load_and_interpolate_solution(solutions, params_list, ly_target, c_cu_target, c_ni_target, 
                                 tolerance_ly=0.1, tolerance_c=1e-5):
    """
    Load exact solution if available, otherwise interpolate.
    
    Critical for parameter mapping: ensures target params match extracted params.
    """
    # Check for exact match first (within tolerance)
    for sol, params in zip(solutions, params_list):
        ly, c_cu, c_ni = params
        if (abs(ly - ly_target) < tolerance_ly and
                abs(c_cu - c_cu_target) < tolerance_c and
                abs(c_ni - c_ni_target) < tolerance_c):
            sol_copy = sol.copy()
            sol_copy['interpolated'] = False
            return sol_copy
    
    # No exact match: interpolate
    if not solutions:
        raise ValueError("No solutions available for interpolation.")
    
    interpolator = MultiParamAttentionInterpolator(sigma=0.2)
    return interpolator(solutions, params_list, ly_target, c_cu_target, c_ni_target)


# =============================================
# CONCENTRATION FEATURE EXTRACTION
# =============================================
def compute_concentration_features(solution, time_index: int, y_positions: Optional[List[float]] = None):
    """
    Extract quantitative concentration features from solution at given time.
    
    Returns dict with DataFrames and scalars for analysis/export.
    """
    if time_index < 0 or time_index >= len(solution['times']):
        raise ValueError(f"time_index {time_index} out of range (0-{len(solution['times'])-1})")
    
    X = solution['X']
    Y = solution['Y']
    Lx = solution['params']['Lx']
    Ly = solution['params']['Ly']
    c1 = solution['c1_preds'][time_index]   # Cu concentration
    c2 = solution['c2_preds'][time_index]   # Ni concentration
    
    # Centerline (x = Lx/2) extraction
    center_idx = np.argmin(np.abs(X[:, 0] - Lx/2))
    y_coords = Y[0, :]
    
    # 1. Centerline profiles
    centerline_cu = c1[center_idx, :]
    centerline_ni = c2[center_idx, :]
    
    # 2. Concentrations at specified y positions
    if y_positions is None:
        y_positions = [0.0, Ly/4, Ly/2, 3*Ly/4, Ly]
    y_indices = [np.argmin(np.abs(y_coords - y)) for y in y_positions]
    conc_at_y = {
        'y (μm)': y_positions,
        'Cu (mol/cc)': [c1[center_idx, idx] for idx in y_indices],
        'Ni (mol/cc)': [c2[center_idx, idx] for idx in y_indices]
    }
    df_conc_at_y = pd.DataFrame(conc_at_y)
    
    # 3. Integrated concentration (trapezoidal rule)
    dy = y_coords[1] - y_coords[0]
    integrated_cu = np.trapz(centerline_cu, dx=dy)
    integrated_ni = np.trapz(centerline_ni, dx=dy)
    
    # 4. Concentration gradients
    grad_cu = np.gradient(centerline_cu, dy)
    grad_ni = np.gradient(centerline_ni, dy)
    max_grad_cu_idx = np.argmax(np.abs(grad_cu))
    max_grad_ni_idx = np.argmax(np.abs(grad_ni))
    
    features = {
        'time': solution['times'][time_index],
        'Ly': Ly,
        'centerline_y': y_coords,
        'centerline_cu': centerline_cu,
        'centerline_ni': centerline_ni,
        'conc_at_y_table': df_conc_at_y,
        'integrated_cu': integrated_cu,
        'integrated_ni': integrated_ni,
        'max_grad_cu': grad_cu[max_grad_cu_idx],
        'max_grad_ni': grad_ni[max_grad_ni_idx],
        'y_max_grad_cu': y_coords[max_grad_cu_idx],
        'y_max_grad_ni': y_coords[max_grad_ni_idx],
        'gradient_cu': grad_cu,
        'gradient_ni': grad_ni,
    }
    return features


def format_features_for_download(features: dict) -> pd.DataFrame:
    """Create DataFrame from scalar features for CSV export."""
    data = {
        'Feature': [
            'Time (s)', 'Ly (μm)', 
            'Integrated Cu (mol/μm per x-unit)', 
            'Integrated Ni (mol/μm per x-unit)',
            'Max |dCu/dy| (mol/cc/μm)', 'y of max |dCu/dy| (μm)', 
            'Max |dNi/dy| (mol/cc/μm)', 'y of max |dNi/dy| (μm)'
        ],
        'Value': [
            features['time'], features['Ly'], 
            features['integrated_cu'], features['integrated_ni'],
            features['max_grad_cu'], features['y_max_grad_cu'], 
            features['max_grad_ni'], features['y_max_grad_ni']
        ]
    }
    return pd.DataFrame(data)


# =============================================
# PLOTTING FUNCTIONS (Publication-Quality)
# =============================================
def plot_2d_concentration(solution, time_index, output_dir="figures", 
                         cmap_cu='viridis', cmap_ni='magma',
                         vmin_cu=None, vmax_cu=None, vmin_ni=None, vmax_ni=None):
    """Generate 2D concentration heatmaps for Cu and Ni."""
    x_coords = solution['X'][:, 0]
    y_coords = solution['Y'][0, :]
    t_val = solution['times'][time_index]
    Lx = solution['params']['Lx']
    Ly = solution['params']['Ly']
    c1 = solution['c1_preds'][time_index]
    c2 = solution['c2_preds'][time_index]
    
    # Determine color limits
    cu_min = vmin_cu if vmin_cu is not None else 0
    cu_max = vmax_cu if vmax_cu is not None else np.max(c1)
    ni_min = vmin_ni if vmin_ni is not None else 0
    ni_max = vmax_ni if vmax_ni is not None else np.max(c2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    
    # Cu heatmap
    im1 = ax1.imshow(c1, origin='lower', extent=[0, Lx, 0, Ly], 
                    cmap=cmap_cu, vmin=cu_min, vmax=cu_max)
    ax1.set_xlabel('x (μm)')
    ax1.set_ylabel('y (μm)')
    ax1.set_title(f'Cu Concentration, t = {t_val:.1f} s')
    ax1.grid(True)
    cb1 = fig.colorbar(im1, ax=ax1, label='Cu Conc. (mol/cc)', format='%.1e')
    cb1.ax.tick_params(labelsize=10)
    
    # Ni heatmap
    im2 = ax2.imshow(c2, origin='lower', extent=[0, Lx, 0, Ly],
                    cmap=cmap_ni, vmin=ni_min, vmax=ni_max)
    ax2.set_xlabel('x (μm)')
    ax2.set_ylabel('y (μm)')
    ax2.set_title(f'Ni Concentration, t = {t_val:.1f} s')
    ax2.grid(True)
    cb2 = fig.colorbar(im2, ax=ax2, label='Ni Conc. (mol/cc)', format='%.1e')
    cb2.ax.tick_params(labelsize=10)
    
    # Title with parameters
    param_text = f"$L_y$ = {Ly:.1f} μm, $C_{{Cu}}$ = {solution['params']['C_Cu']:.1e}, $C_{{Ni}}$ = {solution['params']['C_Ni']:.1e}"
    if solution.get('interpolated', False):
        param_text += " (Interpolated)"
    fig.suptitle(f'Concentration Profiles\n{param_text}', fontsize=14)
    
    # Save figures
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"conc_2d_t_{t_val:.1f}_ly_{Ly:.1f}_ccu_{solution['params']['C_Cu']:.1e}_cni_{solution['params']['C_Ni']:.1e}"
    plt.savefig(os.path.join(output_dir, f"{base_filename}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f"{base_filename}.pdf"), bbox_inches='tight')
    plt.close()
    
    return fig, base_filename


def plot_centerline_curves(solution, time_indices, sidebar_metric='mean_cu', output_dir="figures",
                          label_size=12, title_size=14, tick_label_size=10, legend_loc='upper right',
                          curve_colormap='viridis', axis_linewidth=1.5, tick_major_width=1.5,
                          tick_major_length=4.0, fig_width=8.0, fig_height=6.0, curve_linewidth=1.0,
                          grid_alpha=0.3, grid_linestyle='--', legend_frameon=True, legend_framealpha=0.8):
    """Generate centerline concentration curves with sidebar metric."""
    x_coords = solution['X'][:, 0]
    y_coords = solution['Y'][0, :]
    Lx = solution['params']['Lx']
    Ly = solution['params']['Ly']
    center_idx = 25  # x = Lx/2
    times = solution['times']
    
    # Sidebar metric data
    if sidebar_metric == 'loss' and 'loss' in solution:
        sidebar_data = solution['loss'][:len(times)]
        sidebar_label = 'Loss'
    elif sidebar_metric == 'mean_cu':
        sidebar_data = [np.mean(c1) for c1 in solution['c1_preds']]
        sidebar_label = 'Mean Cu Conc. (mol/cc)'
    else:
        sidebar_data = [np.mean(c2) for c2 in solution['c2_preds']]
        sidebar_label = 'Mean Ni Conc. (mol/cc)'
    
    # Create figure with grid layout
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.05, 0.5])
    ax1 = fig.add_subplot(gs[0])  # Cu curves
    ax2 = fig.add_subplot(gs[1])  # Ni curves
    ax3 = fig.add_subplot(gs[3])  # Sidebar metric
    
    # Plot curves for selected times
    colors = cm.get_cmap(curve_colormap)(np.linspace(0, 1, len(time_indices)))
    for idx, t_idx in enumerate(time_indices):
        t_val = times[t_idx]
        c1_center = solution['c1_preds'][t_idx][:, center_idx]
        c2_center = solution['c2_preds'][t_idx][:, center_idx]
        ax1.plot(y_coords, c1_center, label=f't = {t_val:.1f} s', color=colors[idx], linewidth=curve_linewidth)
        ax2.plot(y_coords, c2_center, label=f't = {t_val:.1f} s', color=colors[idx], linewidth=curve_linewidth)
    
    # Apply styling to all axes
    for ax in [ax1, ax2, ax3]:
        for spine in ax.spines.values():
            spine.set_linewidth(axis_linewidth)
        ax.tick_params(axis='both', which='major', width=tick_major_width, 
                      length=tick_major_length, labelsize=tick_label_size)
        ax.grid(True, linestyle=grid_linestyle, alpha=grid_alpha)
    
    # Legend positioning
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
    
    # Label axes and titles
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
    
    # Sidebar metric plot
    ax3.plot(sidebar_data, times, 'k-', linewidth=curve_linewidth)
    ax3.set_xlabel(sidebar_label, fontsize=label_size)
    ax3.set_ylabel('Time (s)', fontsize=label_size)
    ax3.set_title('Metric vs. Time', fontsize=title_size)
    ax3.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    
    # Figure title with parameters
    param_text = f"$L_y$ = {Ly:.1f} μm, $C_{{Cu}}$ = {solution['params']['C_Cu']:.1e}, $C_{{Ni}}$ = {solution['params']['C_Ni']:.1e}"
    if solution.get('interpolated', False):
        param_text += " (Interpolated)"
    fig.suptitle(f'Centerline Concentration Profiles\n{param_text}', fontsize=title_size)
    
    # Save figures
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"conc_centerline_ly_{Ly:.1f}_ccu_{solution['params']['C_Cu']:.1e}_cni_{solution['params']['C_Ni']:.1e}"
    plt.savefig(os.path.join(output_dir, f"{base_filename}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f"{base_filename}.pdf"), bbox_inches='tight')
    plt.close()
    
    return fig, base_filename


def plot_parameter_sweep(solutions, params_list, selected_params, time_index, sidebar_metric='mean_cu', 
                        output_dir="figures", label_size=12, title_size=14, tick_label_size=10, 
                        legend_loc='upper right', curve_colormap='tab10', axis_linewidth=1.5, 
                        tick_major_width=1.5, tick_major_length=4.0, fig_width=8.0, fig_height=6.0, 
                        curve_linewidth=1.0, grid_alpha=0.3, grid_linestyle='--', legend_frameon=True, 
                        legend_framealpha=0.8):
    """Generate parameter sweep comparison plots."""
    Lx = solutions[0]['params']['Lx']
    center_idx = 25
    t_val = solutions[0]['times'][time_index]
    
    # Collect sidebar data for selected parameters
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
    
    # Create figure
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.05, 0.5])
    ax1 = fig.add_subplot(gs[0])  # Cu
    ax2 = fig.add_subplot(gs[1])  # Ni
    ax3 = fig.add_subplot(gs[3])  # Sidebar
    
    # Plot curves for selected parameter sets
    colors = cm.get_cmap(curve_colormap)(np.linspace(0, 1, len(selected_params)))
    for idx, (sol, params) in enumerate(zip(solutions, params_list)):
        ly, c_cu, c_ni = params
        if params in selected_params:
            y_coords = sol['Y'][0, :]
            c1_center = sol['c1_preds'][time_index][:, center_idx]
            c2_center = sol['c2_preds'][time_index][:, center_idx]
            label = f'$L_y$={ly:.1f}, $C_{{Cu}}$={c_cu:.1e}, $C_{{Ni}}$={c_ni:.1e}'
            if sol.get('interpolated', False):
                label += " (Interpolated)"
            ax1.plot(y_coords, c1_center, label=label, color=colors[idx], linewidth=curve_linewidth)
            ax2.plot(y_coords, c2_center, label=label, color=colors[idx], linewidth=curve_linewidth)
    
    # Apply styling
    for ax in [ax1, ax2, ax3]:
        for spine in ax.spines.values():
            spine.set_linewidth(axis_linewidth)
        ax.tick_params(axis='both', which='major', width=tick_major_width,
                      length=tick_major_length, labelsize=tick_label_size)
        ax.grid(True, linestyle=grid_linestyle, alpha=grid_alpha)
    
    # Legend positioning
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
    
    # Labels and titles
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
    
    # Sidebar bar chart
    ax3.barh(range(len(sidebar_data)), sidebar_data, color='gray', edgecolor='black')
    ax3.set_yticks(range(len(sidebar_data)))
    ax3.set_yticklabels(sidebar_labels, fontsize=tick_label_size)
    ax3.set_xlabel(
        'Mean Cu Conc. (mol/cc)' if sidebar_metric == 'mean_cu' else 
        'Mean Ni Conc. (mol/cc)' if sidebar_metric == 'mean_ni' else 'Loss',
        fontsize=label_size
    )
    ax3.set_title('Metric per Parameter', fontsize=title_size)
    ax3.grid(True, axis='x', linestyle=grid_linestyle, alpha=grid_alpha)
    ax3.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    
    fig.suptitle('Concentration Profiles for Parameter Sweep', fontsize=title_size)
    
    # Save
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
    """Initialize Streamlit session state with defaults."""
    defaults = {
        'nl_parser': DiffusionNLParser(),
        'llm_backend_loaded': 'GPT-2 (default)',
        'llm_cache': OrderedDict(),
        'parsed_structured': None,  # Full extraction with metadata
        'parsed_target': None,      # Clean target params for simulation
        'nl_query': "",
        'use_llm': True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================
# MAIN APP WITH PARAMETER MAPPING PIPELINE
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
    # MAIN: NATURAL LANGUAGE INPUT & PARSING
    # =========================================
    st.subheader("📝 Describe Your Solder Joint Configuration")
    
    templates = {
        "Thin Asymmetric Joint": "Analyze a 40 μm asymmetric Cu-Ni joint with top Cu concentration 1.8e-3 and bottom Ni 0.4e-3.",
        "Thick Symmetric Cu": "Simulate a 100 μm symmetric Cu/Sn2.5Ag/Cu joint. Use c_Cu=2.0e-3.",
        "Ni-Rich Diffusion": "Domain length 60 μm, C_Cu=1.0e-3, C_Ni=1.5e-3. Asymmetric configuration.",
        "Self-Diffusion Baseline": "Ly=75 μm, C_Cu=0, C_Ni=0 for self-diffusion reference.",
        "User Example": "Domain length 80, C_Cu=2.0e-3, C_Ni=1.0e-3",  # From user query
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
    
    # =========================================
    # PARAMETER EXTRACTION & MAPPING PIPELINE
    # =========================================
    parser = st.session_state.nl_parser
    use_llm = st.session_state.get('use_llm', False)
    tokenizer = st.session_state.get('llm_tokenizer', None)
    model = st.session_state.get('llm_model', None)
    
    if nl_query:
        # Run full extraction pipeline
        structured_extraction, target_params = parser.hybrid_parse(
            nl_query, tokenizer, model, use_llm=use_llm
        )
        st.session_state.parsed_structured = structured_extraction
        st.session_state.parsed_target = target_params
        
        # Display extraction explanation
        st.markdown(parser.get_explanation(structured_extraction, nl_query))
        
        # Show mapping summary
        st.info(f"""
        **Parameter Mapping Summary:**
        - Extracted values → Validated → Target parameters
        - All values clipped to physical constraints
        - Confidence scores: >0.8 = high, 0.5-0.8 = medium, <0.5 = low
        """)
    else:
        # Use defaults
        structured_extraction = {key: {'value': val, 'unit': ParameterSchema.CONSTRAINTS[key][3],
                                      'confidence': 0.0, 'source': 'default', 'is_valid': True}
                              for key, val in parser.defaults.items()}
        target_params = parser.defaults.copy()
        st.session_state.parsed_structured = structured_extraction
        st.session_state.parsed_target = target_params
    
    # =========================================
    # PARAMETER SELECTION UI (Pre-filled from Parser)
    # =========================================
    st.subheader("🎯 Target Parameters (Override if Needed)")
    
    # Load solutions
    solutions, params_list, lys, c_cus, c_nis, load_logs = load_solutions(SOLUTION_DIR)
    
    if load_logs:
        with st.expander("📋 Load Log"):
            for log in load_logs:
                st.write(log)
    
    if not solutions:
        st.error("No valid solution files found in pinn_solutions directory.")
        return
    
    # Get unique parameter values for dropdowns
    lys = sorted(set(lys))
    c_cus = sorted(set(c_cus))
    c_nis = sorted(set(c_nis))
    
    # Get target values from parser (with fallback to defaults)
    target = st.session_state.parsed_target
    ly_default = target['ly_target']
    c_cu_default = target['c_cu_target']
    c_ni_default = target['c_ni_target']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        ly_choice = st.selectbox(
            "Domain Height (Ly, μm)",
            options=lys,
            format_func=lambda x: f"{x:.1f}",
            index=min(lys.index(ly_default), len(lys)-1) if ly_default in lys else 0
        )
    with col2:
        c_cu_choice = st.selectbox(
            "Cu Boundary Concentration (mol/cc)",
            options=c_cus,
            format_func=lambda x: f"{x:.1e}",
            index=min(c_cus.index(c_cu_default), len(c_cus)-1) if c_cu_default in c_cus else 0
        )
    with col3:
        c_ni_choice = st.selectbox(
            "Ni Boundary Concentration (mol/cc)",
            options=c_nis,
            format_func=lambda x: f"{x:.1e}",
            index=min(c_nis.index(c_ni_default), len(c_nis)-1) if c_ni_default in c_nis else 0
        )
    
    # Custom parameter override
    use_custom_params = st.checkbox("Use Custom Parameters for Interpolation", value=False)
    if use_custom_params:
        ly_target = st.number_input(
            "Custom Ly (μm)",
            min_value=30.0,
            max_value=120.0,
            value=ly_default,
            step=0.1,
            format="%.1f"
        )
        c_cu_target = st.number_input(
            "Custom C_Cu (mol/cc)",
            min_value=0.0,
            max_value=2.9e-3,
            value=c_cu_default,
            step=0.1e-3,
            format="%.1e"
        )
        c_ni_target = st.number_input(
            "Custom C_Ni (mol/cc)",
            min_value=0.0,
            max_value=1.8e-3,
            value=c_ni_default,
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
            cu_max_default = float(np.max([np.max(sol['c1_preds']) for sol in solutions]))
            custom_cu_max = st.number_input("Cu Max", value=cu_max_default, format="%.2e", key="cu_max")
        with col2:
            st.write("**Ni Concentration Limits**")
            custom_ni_min = st.number_input("Ni Min", value=0.0, format="%.2e", key="ni_min")
            ni_max_default = float(np.max([np.max(sol['c2_preds']) for sol in solutions]))
            custom_ni_max = st.number_input("Ni Max", value=ni_max_default, format="%.2e", key="ni_max")
    
    # Validate color limits
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
        solution = load_and_interpolate_solution(
            solutions, params_list, ly_target, c_cu_target, c_ni_target
        )
    except Exception as e:
        st.error(f"Failed to load or interpolate solution: {str(e)}")
        return
    
    # Display solution details with parameter mapping confirmation
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
                'Weight': weights
            })
            st.dataframe(
                weight_df.style.format({'Weight': '{:.4f}'}).bar(subset=['Weight'], color='#5fba7d'),
                use_container_width=True
            )
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
    feature_time_idx = st.number_input(
        "Time index for feature extraction",
        min_value=0, max_value=len(solution['times'])-1,
        value=time_index, step=1, key="feature_time_idx"
    )
    try:
        features = compute_concentration_features(solution, feature_time_idx)
        
        # Display key metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Integrated Cu (mol/μm per x-unit)", f"{features['integrated_cu']:.3e}")
            st.metric("Max |dCu/dy| (mol/cc/μm)", 
                     f"{features['max_grad_cu']:.3e} at y={features['y_max_grad_cu']:.1f} μm")
        with col2:
            st.metric("Integrated Ni (mol/μm per x-unit)", f"{features['integrated_ni']:.3e}")
            st.metric("Max |dNi/dy| (mol/cc/μm)", 
                     f"{features['max_grad_ni']:.3e} at y={features['y_max_grad_ni']:.1f} μm")
        
        # Show concentration table
        st.write("**Concentrations at Selected y‑positions (centerline):**")
        st.dataframe(features['conc_at_y_table'], use_container_width=True)
        
        # Download buttons
        df_scalar = format_features_for_download(features)
        csv_scalar = df_scalar.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Scalar Features as CSV",
            data=csv_scalar,
            file_name=f"concentration_features_scalar_t{features['time']:.1f}.csv",
            mime="text/csv"
        )
        
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
        default=[0, len(solution['times'])//4, len(solution['times'])//2, 
                3*len(solution['times'])//4, len(solution['times'])-1],
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
                min_value=30.0, max_value=120.0,
                value=ly_choice, step=0.1, format="%.1f", key=f"ly_custom_{i}"
            )
            c_cu_custom = st.number_input(
                f"Custom C_Cu (mol/cc) {i+1}",
                min_value=0.0, max_value=2.9e-3,
                value=max(c_cu_choice, 1.5e-3), step=0.1e-3, format="%.1e", key=f"c_cu_custom_{i}"
            )
            c_ni_custom = st.number_input(
                f"Custom C_Ni (mol/cc) {i+1}",
                min_value=0.0, max_value=1.8e-3,
                value=max(c_ni_choice, 1.0e-4), step=0.1e-4, format="%.1e", key=f"c_ni_custom_{i}"
            )
            custom_params.append((ly_custom, c_cu_custom, c_ni_custom))
    
    # Select from precomputed solutions
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
    
    # Load/interpolate solutions for sweep
    sweep_solutions = []
    sweep_params_list = []
    for params in selected_params:
        ly, c_cu, c_ni = params
        try:
            sol = load_and_interpolate_solution(solutions, params_list, ly, c_cu, c_ni)
            sweep_solutions.append(sol)
            sweep_params_list.append(params)
        except Exception as e:
            st.warning(f"Failed to load/interpolate for Ly={ly:.1f}, C_Cu={c_cu:.1e}, C_Ni={c_ni:.1e}: {str(e)}")
    
    sweep_time_index = st.slider("Select Time Index for Sweep", 0, len(solution['times'])-1, len(solution['times'])-1)
    if sweep_solutions and sweep_params_list:
        fig_sweep, filename_sweep = plot_parameter_sweep(
            sweep_solutions, sweep_params_list, sweep_params_list, sweep_time_index, 
            sidebar_metric=sidebar_metric,
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
