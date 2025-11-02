import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Cloud Transformers LLM Demo")

st.title("Cloud LLM Text Generation for Engineering Insights")

# User prompt input
prompt = st.text_area(
    "Enter prompt for Engineering Insight generation:",
    "Generate engineering insights on Cu-Ni interdiffusion and IMC growth based on attention model outputs."
)

# Model selection (optional)
model_name = st.selectbox(
    "Select a Hugging Face model:",
    ["gpt2", "facebook/opt-1.3b", "tiiuae/falcon-7b-instruct", "google/flan-t5-large"]
)

if st.button("Generate Insights"):
    if prompt.strip():
        try:
            with st.spinner(f"Loading model {model_name}..."):
                # Load the Hugging Face text-generation pipeline
                generator = pipeline(
                    "text-generation",
                    model=model_name,
                    device= 0 #-1  # -1 for CPU, change to 0 for GPU
                )

                # Generate output
                outputs = generator(prompt, max_length=300, do_sample=True, temperature=0.7)
                text_output = outputs[0]["generated_text"]

            st.subheader("Generated Engineering Insights (Cloud Transformers):")
            st.write(text_output)

        except Exception as e:
            st.error(f"Error generating insights: {e}")
    else:
        st.error("Please enter a prompt to generate insights.")
