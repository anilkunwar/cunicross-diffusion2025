import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

st.set_page_config(page_title="Lightweight LLM Integration Demo")

st.title("Lightweight LLM Text Generation for Engineering Insights")

# Sidebar to input OpenAI API key securely
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

if not openai_api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to activate the LLM.")
    st.stop()

# Input prompt for inference
prompt = st.text_area("Enter prompt for Engineering Insight generation:", 
                      "Generate engineering insights on Cu-Ni interdiffusion and IMC growth based on attention model outputs.")

if st.button("Generate Insights"):
    if prompt.strip():
        # Initialize the lightweight ChatOpenAI model with user's API key
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=openai_api_key)

        # Sending prompt as a human message and getting response
        response = llm([HumanMessage(content=prompt)])

        st.subheader("Generated Engineering Insights:")
        st.write(response.content)
    else:
        st.error("Please enter a prompt to generate insights.")
