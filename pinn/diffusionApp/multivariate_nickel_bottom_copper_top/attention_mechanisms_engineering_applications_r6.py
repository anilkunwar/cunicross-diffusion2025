
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
import json

# === Page Config ===
st.set_page_config(page_title="AI-Attention Diffusion Inference", layout="wide")
st.title("🧠 AI-Attention Engineering Analysis for Cu-Ni Interdiffusion")

# === Initialize Session State ===
session_defaults = {
    'computation_complete': False,
    'results': None,
    'diffusion_data': None,
    'ai_insights_generated': False,
    'faq_questions': [],
    'selected_faq': None
}

for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# === Enhanced Database Handler ===
class FAQDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self.ensure_tables()
    
    def ensure_tables(self):
        """Create FAQ table if it doesn't exist"""
        if os.path.exists(self.db_path):
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS faq_questions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        topic TEXT NOT NULL,
                        question TEXT NOT NULL,
                        context_template TEXT,
                        weight_factor REAL DEFAULT 1.0,
                        category TEXT DEFAULT 'general'
                    )
                ''')
                
                # Insert default FAQ questions if table is empty
                cursor.execute("SELECT COUNT(*) FROM faq_questions")
                if cursor.fetchone()[0] == 0:
                    self.insert_default_faqs(cursor)
                
                conn.commit()
                conn.close()
            except Exception as e:
                st.error(f"Database setup error: {e}")
    
    def insert_default_faqs(self, cursor):
        """Insert default engineering FAQ questions"""
        default_faqs = [
            ("Domain Length Effects", 
             "How does joint thickness (L_y) affect diffusion kinetics and IMC growth?",
             "Analyze domain length effects for L_y={ly_target}μm with attention weights from sources {sources}.",
             1.2, "kinetics"),
            
            ("Boundary Concentration Dynamics",
             "What is the impact of boundary concentrations on flux distribution and IMC formation?",
             "Evaluate boundary effects: Cu={c_cu_target:.1e} mol/cc, Ni={c_ni_target:.1e} mol/cc with flux ratios.",
             1.1, "concentrations"),
            
            ("Uphill Diffusion Analysis",
             "Assess the risk and mechanisms of uphill diffusion in this configuration.",
             "Analyze uphill diffusion risk for Cu/Ni ratio {cu_ni_ratio:.2f} with cross-interaction potential.",
             1.3, "mechanisms"),
            
            ("Substrate Configuration Impact",
             "How does substrate symmetry/asymmetry influence IMC morphology and void formation?",
             "Evaluate {substrate_type} configuration effects on IMC growth and reliability.",
             1.0, "morphology"),
            
            ("Joining Path Dependencies",
             "What are the joining path effects on interfacial chemistry and IMC evolution?",
             "Analyze {joining_path} effects on Ni distribution and interface reactions.",
             1.1, "processing"),
            
            ("IMC Growth Kinetics",
             "Predict IMC growth kinetics and thermal cycling reliability.",
             "Model IMC growth: Cu6Sn5={imc_cu:.2f}μm, Ni3Sn4={imc_ni:.2f}μm after {time}s.",
             1.4, "kinetics"),
            
            ("Attention Mechanism Insights",
             "Interpret the attention weight distribution and parameter interpolation significance.",
             "Explain attention distribution: dominant source {dominant_source} ({max_weight:.1%}) implications.",
             1.0, "modeling")
        ]
        
        cursor.executemany('''
            INSERT INTO faq_questions (topic, question, context_template, weight_factor, category)
            VALUES (?, ?, ?, ?, ?)
        ''', default_faqs)
    
    def get_faq_questions(self):
        """Retrieve all FAQ questions"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id, topic, question, context_template, weight_factor FROM faq_questions")
            questions = cursor.fetchall()
            conn.close()
            return questions
        except Exception as e:
            st.error(f"Error reading FAQ questions: {e}")
            return []
    
    def get_experimental_context(self):
        """Get experimental context from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT description FROM experiments LIMIT 1")
            row = cursor.fetchone()
            conn.close()
            return row[0] if row else "Standard Cu-Ni interdiffusion in Sn2.5Ag solder joints."
        except:
            return "Standard Cu-Ni interdiffusion in Sn2.5Ag solder joints."

# === Initialize Database ===
DB_DIR = os.path.join(os.path.dirname(__file__), "nlp_information")
DB_FILENAME = "description_of_experiment.db"
DB_PATH = os.path.join(DB_DIR, DB_FILENAME)

faq_db = FAQDatabase(DB_PATH)
EXPERIMENTAL_DESCRIPTION = faq_db.get_experimental_context()

# === Enhanced Model Definition ===
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

# === Diffusion Analyzer ===
class DiffusionAnalyzer:
    def __init__(self):
        self.D_Cu = 1.2e-9
        self.D_Ni = 8.5e-10
        
    def concentration_profile(self, y, L, C_top, C_bottom, D, time):
        y_norm = y / L
        return C_bottom + (C_top - C_bottom) * (1 - y_norm)
    
    def calculate_flux(self, concentration_gradient, D):
        return -D * concentration_gradient
    
    def integrate_imc_growth(self, flux_Cu, flux_Ni, time_seconds):
        k_Cu6Sn5 = 2.3e-14
        k_Ni3Sn4 = 1.8e-14
        
        imc_Cu_thickness = k_Cu6Sn5 * np.cbrt(time_seconds) * 1e12
        imc_Ni_thickness = k_Ni3Sn4 * np.cbrt(time_seconds) * 1e12
        
        return imc_Cu_thickness, imc_Ni_thickness

# === Enhanced AI Model Handler ===
class AIModelHandler:
    def __init__(self):
        self.supported_models = {
            "Lightweight Models": {
                "gpt2": "GPT-2 (Fastest)",
                "facebook/opt-350m": "OPT-350M",
                "distilgpt2": "DistilGPT-2",
                "microsoft/DialoGPT-small": "DialoGPT-Small"
            },
            "Medium Models": {
                "facebook/opt-1.3b": "OPT-1.3B",
                "google/flan-t5-base": "FLAN-T5 Base",
                "tiiuae/falcon-7b-instruct": "Falcon-7B-Instruct"
            },
            "Advanced Models (if available)": {
                "llama-2-7b-chat": "LLaMA-2-7B-Chat",
                "mistral-7b-instruct": "Mistral-7B-Instruct",
                "openllm/llama-2": "OpenLLM LLaMA-2",
                "custom": "Custom Model"
            }
        }
    
    def get_model_list(self):
        """Return organized model list for dropdown"""
        model_list = []
        for category, models in self.supported_models.items():
            model_list.append(f"--- {category} ---")
            model_list.extend(models.values())
        return model_list
    
    def get_model_name(self, display_name):
        """Convert display name to model identifier"""
        for category, models in self.supported_models.items():
            for key, value in models.items():
                if value == display_name:
                    return key
        return display_name
    
    def load_model(self, model_name, custom_model_path=None):
        """Load the appropriate model"""
        if model_name == "custom" and custom_model_path:
            model_name = custom_model_path
        
        try:
            # For OpenLLM models
            if "openllm" in model_name.lower():
                return pipeline("text-generation", model=model_name, device=-1, trust_remote_code=True)
            
            # For LLaMA models (may require specific handling)
            elif "llama" in model_name.lower() or "mistral" in model_name.lower():
                return pipeline("text-generation", model=model_name, device=-1, torch_dtype=torch.float16)
            
            # Standard models
            else:
                return pipeline("text-generation", model=model_name, device=-1)
                
        except Exception as e:
            st.error(f"Failed to load model {model_name}: {e}")
            # Fallback to a lightweight model
            return pipeline("text-generation", model="distilgpt2", device=-1)

# === Sidebar Configuration ===
with st.sidebar:
    st.header("🧠 AI Model Configuration")
    
    # Initialize model handler
    model_handler = AIModelHandler()
    
    # Model selection
    model_display = st.selectbox(
        "Select AI Model",
        options=model_handler.get_model_list(),
        index=3  # Default to DistilGPT-2
    )
    
    # Custom model input
    custom_model_path = None
    if "Custom Model" in model_display:
        custom_model_path = st.text_input("Custom Model Path/Name", "username/model-name")
    
    # Get actual model name
    nlp_model = model_handler.get_model_name(model_display)
    
    st.header("🔧 Analysis Parameters")
    sigma = st.slider("Attention Locality σ", 0.05, 0.50, 0.20, 0.01)
    num_heads = st.slider("Attention Heads", 1, 8, 4)
    
    st.header("⚙️ Joint Configuration")
    substrate_type = st.selectbox("Substrate", 
        ["Cu(top)-Ni(bottom) Asymmetric", "Cu/Sn2.5Ag/Cu Symmetric", "Ni/Sn2.5Ag/Ni Symmetric"])
    joining_path = st.selectbox("Joining Path", ["Path I (Cu→Ni)", "Path II (Ni→Cu)", "N/A"])
    joint_length = st.slider("Joint Thickness L_y (μm)", 30.0, 120.0, 60.0, 1.0)
    reflow_time = st.slider("Reflow Time (s)", 1, 1000, 90)

# === FAQ System ===
st.sidebar.header("📚 Engineering FAQ Topics")

# Load FAQ questions
if not st.session_state.faq_questions:
    st.session_state.faq_questions = faq_db.get_faq_questions()

# Display FAQ selection
faq_options = ["Select a specific engineering topic..."] + [q[2] for q in st.session_state.faq_questions]
selected_faq_index = st.sidebar.selectbox("Focus Analysis On:", faq_options)

if selected_faq_index > 0:
    st.session_state.selected_faq = st.session_state.faq_questions[selected_faq_index - 1]

# === Source Solutions ===
st.subheader("📊 Precomputed Source Simulations")
num_sources = st.slider("Number of Attention Sources", 2, 6, 3, key="num_sources")
params_list = []
for i in range(num_sources):
    with st.expander(f"Source {i+1} Configuration"):
        col1, col2, col3 = st.columns(3)
        ly = col1.number_input(f"L_y", 30.0, 120.0, 30.0 + 30*i, 0.1, key=f"ly_{i}")
        c_cu = col2.number_input(f"C_Cu (top)", 0.0, 2.9e-3, 1.5e-3, 1e-4, format="%.1e", key=f"ccu_{i}")
        c_ni = col3.number_input(f"C_Ni (bottom)", 0.0, 1.8e-3, 0.1e-3 + 0.4e-3*i, 1e-5, format="%.1e", key=f"cni_{i}")
        params_list.append((ly, c_cu, c_ni))

# === Target Inference ===
st.subheader("🎯 Target Joint for AI Analysis")
col1, col2 = st.columns(2)
with col1:
    ly_target = st.number_input("Target L_y (μm)", 30.0, 120.0, joint_length, 0.1, key="ly_target")
with col2:
    c_cu_target = st.number_input("Top C_Cu (mol/cc)", 0.0, 2.9e-3, 2.0e-3, 1e-4, format="%.1e", key="c_cu_target")
    c_ni_target = st.number_input("Bottom C_Ni (mol/cc)", 0.0, 1.8e-3, 1.0e-3, 1e-4, format="%.1e", key="c_ni_target")

# === Main Analysis ===
if st.button("🚀 Run AI-Attention Analysis", type="primary"):
    with st.spinner("Computing attention weights and generating AI insights..."):
        # Perform computation
        interpolator = MultiParamAttentionInterpolator(sigma, num_heads, 8)
        results = interpolator.compute_weights(params_list, ly_target, c_cu_target, c_ni_target)
        
        analyzer = DiffusionAnalyzer()
        y_positions = np.linspace(0, ly_target, 100)
        
        Cu_profile = analyzer.concentration_profile(y_positions, ly_target, c_cu_target, 0, analyzer.D_Cu, reflow_time)
        Ni_profile = analyzer.concentration_profile(ly_target - y_positions, ly_target, c_ni_target, 0, analyzer.D_Ni, reflow_time)
        
        gradient_Cu = np.gradient(Cu_profile, y_positions)
        gradient_Ni = np.gradient(Ni_profile, y_positions)
        flux_Cu = analyzer.calculate_flux(gradient_Cu, analyzer.D_Cu)
        flux_Ni = analyzer.calculate_flux(gradient_Ni, analyzer.D_Ni)
        
        imc_Cu_thickness, imc_Ni_thickness = analyzer.integrate_imc_growth(
            np.mean(flux_Cu), np.mean(flux_Ni), reflow_time
        )

        # Store results
        st.session_state.results = results
        st.session_state.diffusion_data = {
            'y_positions': y_positions, 'Cu_profile': Cu_profile, 'Ni_profile': Ni_profile,
            'flux_Cu': flux_Cu, 'flux_Ni': flux_Ni, 'imc_Cu_thickness': imc_Cu_thickness,
            'imc_Ni_thickness': imc_Ni_thickness, 'params_list': params_list,
            'ly_target': ly_target, 'c_cu_target': c_cu_target, 'c_ni_target': c_ni_target,
            'substrate_type': substrate_type, 'joining_path': joining_path, 'reflow_time': reflow_time
        }
        st.session_state.computation_complete = True

        # Generate AI insights
        with st.spinner("🤖 Generating AI-powered engineering insights..."):
            try:
                w = results['combined_weights']
                dominant_source = np.argmax(w) + 1
                
                # Build context-aware prompt
                if st.session_state.selected_faq:
                    # FAQ-focused prompt
                    faq_id, topic, question, template, weight_factor = st.session_state.selected_faq
                    context = template.format(
                        ly_target=ly_target, sources=len(params_list),
                        c_cu_target=c_cu_target, c_ni_target=c_ni_target,
                        cu_ni_ratio=c_cu_target/c_ni_target, substrate_type=substrate_type,
                        joining_path=joining_path, imc_cu=imc_Cu_thickness,
                        imc_ni=imc_Ni_thickness, time=reflow_time,
                        dominant_source=dominant_source, max_weight=w.max()
                    )
                    
                    ai_prompt = f"""
                    As a materials engineering expert, answer this specific question:
                    QUESTION: {question}
                    
                    CONTEXT: {context}
                    
                    ATTENTION ANALYSIS: Dominant source S{dominant_source} ({w.max():.1%} weight)
                    CALCULATED RESULTS: IMC thickness Cu6Sn5={imc_Cu_thickness:.2f}μm, Ni3Sn4={imc_Ni_thickness:.2f}μm
                    
                    Provide a detailed, quantitative engineering answer focusing specifically on the question.
                    """
                else:
                    # General comprehensive analysis
                    ai_prompt = f"""
                    Generate comprehensive engineering insights for Cu-Ni interdiffusion:

                    ATTENTION ANALYSIS RESULTS:
                    - Dominant source: S{dominant_source} with {w.max():.1%} weight
                    - Parameter blending from {len(params_list)} simulations
                    - Target: {ly_target}μm, Cu={c_cu_target:.1e} mol/cc, Ni={c_ni_target:.1e} mol/cc
                    - Configuration: {substrate_type}, {joining_path}
                    - Predicted IMC: Cu6Sn5={imc_Cu_thickness:.2f}μm, Ni3Sn4={imc_Ni_thickness:.2f}μm

                    Focus on:
                    1. Domain length effects and diffusion kinetics
                    2. Boundary concentration impacts on flux
                    3. Uphill diffusion risks (Cu/Ni ratio: {c_cu_target/c_ni_target:.2f})
                    4. Substrate configuration effects
                    5. Joining path dependencies  
                    6. IMC growth kinetics and reliability
                    7. Attention mechanism interpretation

                    Provide quantitative, specific engineering predictions.
                    """

                # Load and run model
                generator = model_handler.load_model(nlp_model, custom_model_path)
                outputs = generator(ai_prompt, max_length=800, do_sample=True, temperature=0.7, 
                                  num_return_sequences=1)
                st.session_state.ai_insights = outputs[0]["generated_text"]
                st.session_state.ai_insights_generated = True
                
            except Exception as e:
                st.error(f"AI insight generation failed: {e}")
                st.session_state.ai_insights_generated = False

    st.success("✅ AI-Attention analysis complete!")

# === Display Results ===
if st.session_state.computation_complete:
    results = st.session_state.results
    data = st.session_state.diffusion_data
    
    # Attention Results
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("📈 Attention Weight Analysis")
        w = results['combined_weights']
        dominant_source = np.argmax(w) + 1
        
        df_weights = pd.DataFrame({
            'Source': [f"S{i+1}" for i in range(len(data['params_list']))],
            'Attention': np.round(results['attention_weights'], 4),
            'Spatial': np.round(results['spatial_weights'], 4),
            'Combined': np.round(results['combined_weights'], 4)
        })
        st.dataframe(df_weights.style.bar(subset=['Combined'], color='#5fba7d'))
        
        # Parameter space visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        src = results['norm_sources']
        tgt = results['norm_target']
        scatter = ax.scatter(src[:, 0], src[:, 1], c=src[:, 2], s=200, cmap='viridis', 
                           alpha=0.7, edgecolors='black')
        ax.scatter(tgt[0], tgt[1], c='red', s=400, marker='*', edgecolors='black', label='Target')
        ax.set_xlabel("Normalized L_y")
        ax.set_ylabel("Normalized C_Cu")
        ax.set_title("Attention Parameter Space")
        plt.colorbar(scatter, ax=ax, label="Normalized C_Ni")
        st.pyplot(fig)

    with col2:
        st.subheader("🔍 Attention Insights")
        st.metric("Dominant Influence", f"Source {dominant_source}")
        st.metric("Weight Concentration", f"{w.max():.1%}")
        st.metric("Analysis Focus", 
                 st.session_state.selected_faq[1] if st.session_state.selected_faq else "Comprehensive")
        
        # Quick metrics
        st.info(f"""
        **Quick Analysis:**
        - IMC Ratio (Cu/Ni): {data['imc_Cu_thickness']/data['imc_Ni_thickness']:.2f}
        - Flux Dominance: {'Cu' if np.mean(data['flux_Cu']) > np.mean(data['flux_Ni']) else 'Ni'}
        - Uphill Risk: {'High' if data['c_ni_target']/data['c_cu_target'] > 0.5 else 'Moderate'}
        """)

    # === AI-Generated Insights ===
    st.subheader("🤖 AI Engineering Analysis")
    
    if st.session_state.ai_insights_generated:
        st.markdown("#### 🎯 AI-Powered Insights")
        st.write(st.session_state.ai_insights)
        
        # Add calculated results emphasis
        st.markdown("#### 📊 Calculated Results Integration")
        st.success(f"""
        **Attention-Weighted Results:**
        - Dominant Source Influence: **S{dominant_source}** ({w.max():.1%})
        - Predicted IMC Growth: **Cu6Sn5 = {data['imc_Cu_thickness']:.2f}μm**, **Ni3Sn4 = {data['imc_Ni_thickness']:.2f}μm**
        - Flux Characteristics: **J_Cu = {np.mean(data['flux_Cu']):.2e} mol/m²s**, **J_Ni = {np.mean(data['flux_Ni']):.2e} mol/m²s**
        - Process Optimization: **{data['reflow_time']}s** reflow adequate for IMC formation
        """)
    else:
        st.warning("AI insights generation failed. Showing attention-based analysis...")

    # === FAQ Context Display ===
    if st.session_state.selected_faq:
        st.info(f"""
        **Focused Analysis Context:**
        *Topic: {st.session_state.selected_faq[1]}*
        - Question: {st.session_state.selected_faq[2]}
        - Database Weight Factor: {st.session_state.selected_faq[4]}x
        - Experimental Reference: {EXPERIMENTAL_DESCRIPTION[:200]}...
        """)

    # === Export Section ===
    st.subheader("📤 Export Analysis")
    
    if st.button("💾 Save Complete Analysis Report"):
        report_data = {
            'attention_analysis': {
                'dominant_source': int(dominant_source),
                'max_weight': float(w.max()),
                'weight_distribution': results['combined_weights'].tolist()
            },
            'engineering_results': {
                'imc_growth': {
                    'cu6sn5_thickness_um': float(data['imc_Cu_thickness']),
                    'ni3sn4_thickness_um': float(data['imc_Ni_thickness'])
                },
                'flux_analysis': {
                    'avg_cu_flux': float(np.mean(data['flux_Cu'])),
                    'avg_ni_flux': float(np.mean(data['flux_Ni']))
                }
            },
            'ai_insights': st.session_state.ai_insights if st.session_state.ai_insights_generated else "N/A",
            'faq_context': st.session_state.selected_faq[1] if st.session_state.selected_faq else "Comprehensive",
            'model_used': model_display
        }
        
        json_report = json.dumps(report_data, indent=2)
        st.download_button("Download Report (JSON)", json_report, "ai_engineering_analysis.json", "application/json")

# === Experimental Reference ===
with st.expander("📚 Experimental Reference & FAQ Database"):
    st.markdown(f"**Reference Context:** {EXPERIMENTAL_DESCRIPTION}")
    st.markdown("**Available FAQ Topics:**")
    for faq in st.session_state.faq_questions:
        st.write(f"• **{faq[1]}**: {faq[2]}")

# Clear results
if st.session_state.computation_complete and st.button("🔄 New Analysis"):
    for key in session_defaults:
        st.session_state[key] = session_defaults[key]
    st.rerun()
