import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="Local LLM Integration Demo")

st.title("Lightweight Local LLM Text Generation for Engineering Insights")

# User prompt input
prompt = st.text_area(
    "Enter prompt for Engineering Insight generation:",
    "Generate engineering insights on Cu-Ni interdiffusion and IMC growth based on attention model outputs."
)

if st.button("Generate Insights"):
    if prompt.strip():
        try:
            # Initialize local Ollama model
            # Replace "llama3" with your local model name
            llm = Ollama(model="llama3", verbose=True)

            # Send prompt as a human message
            response = llm.invoke([HumanMessage(content=prompt)])

            st.subheader("Generated Engineering Insights (Local Ollama):")
            st.write(response.content)

        except Exception as e:
            st.error(f"Error generating insights: {e}")
    else:
        st.error("Please enter a prompt to generate insights.")
