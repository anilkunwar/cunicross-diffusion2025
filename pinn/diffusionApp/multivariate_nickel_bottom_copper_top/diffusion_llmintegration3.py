import streamlit as st

# === Import necessary libraries with graceful handling ===
try:
    from langchain_community.llms import Ollama
except ImportError:
    Ollama = None

try:
    from openllm import AutoLLM
except ImportError:
    AutoLLM = None

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
except ImportError:
    ChatOpenAI = None
    HumanMessage = None

st.set_page_config(page_title="Hybrid LLM Engineering Insights")

st.title("Hybrid LLM Text Generation for Engineering Insights")

# === User prompt ===
prompt = st.text_area(
    "Enter prompt for Engineering Insight generation:",
    "Generate engineering insights on Cu-Ni interdiffusion and IMC growth based on attention model outputs."
)

# === Optional OpenAI API key ===
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key (optional)", type="password")

# === Optional Hugging Face API key ===
hf_api_key = st.sidebar.text_input("Enter your Hugging Face API Key (optional)", type="password")

if st.button("Generate Insights"):
    if not prompt.strip():
        st.error("Please enter a prompt to generate insights.")
        st.stop()

    response_text = None

    # === 1️⃣ Try local Ollama first ===
    if Ollama is not None:
        try:
            st.info("Trying local Ollama model...")
            llm = Ollama(model="llama3", verbose=True)
            response = llm.invoke([HumanMessage(content=prompt)])
            response_text = response.content
        except Exception as e:
            st.warning(f"Local Ollama failed: {e}")

    # === 2️⃣ Fallback to OpenLLM Hugging Face ===
    if response_text is None and AutoLLM is not None:
        try:
            st.info("Trying OpenLLM Hugging Face model...")
            llm = AutoLLM.from_pretrained(
                "tiiuae/falcon-7b-instruct",
                backend="huggingface",
                use_auth_token=hf_api_key if hf_api_key else None
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            response_text = response.content
        except Exception as e:
            st.warning(f"OpenLLM Hugging Face failed: {e}")

    # === 3️⃣ Fallback to OpenAI ===
    if response_text is None and ChatOpenAI is not None and openai_api_key:
        try:
            st.info("Trying OpenAI Chat API...")
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=openai_api_key)
            response = llm.invoke([HumanMessage(content=prompt)])
            response_text = response.content
        except Exception as e:
            st.warning(f"OpenAI failed: {e}")

    # === 4️⃣ No model worked ===
    if response_text is None:
        st.error(
            "All LLM backends failed. Please ensure:\n"
            "- Ollama is installed locally, or\n"
            "- OpenLLM with Hugging Face is available, or\n"
            "- OpenAI API key is provided and has quota."
        )
    else:
        st.subheader("Generated Engineering Insights:")
        st.write(response_text)
