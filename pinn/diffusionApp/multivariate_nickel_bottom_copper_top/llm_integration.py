import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="Lightweight LLM Integration Demo")

st.title("Lightweight LLM Text Generation for Engineering Insights")

# Sidebar: secure API key input
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

if not openai_api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to activate the LLM.")
    st.stop()

# User prompt input
prompt = st.text_area(
    "Enter prompt for Engineering Insight generation:",
    "Generate engineering insights on Cu-Ni interdiffusion and IMC growth based on attention model outputs."
)

# Run inference
if st.button("Generate Insights"):
    if prompt.strip():
        try:
            # Initialize the LLM
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",  # or "gpt-4o-mini" if available
                temperature=0.7,
                openai_api_key=openai_api_key
            )

            # Use the correct method for message-based input
            response = llm.invoke([HumanMessage(content=prompt)])

            # Display results
            st.subheader("Generated Engineering Insights:")
            st.write(response.content)

        except Exception as e:
            st.error(f"Error generating insights: {e}")
    else:
        st.error("Please enter a prompt to generate insights.")
