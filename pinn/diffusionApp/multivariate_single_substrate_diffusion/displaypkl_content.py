import streamlit as st
import pickle
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="PKL File Parser", layout="wide")
st.title("📄 PKL File Content Parser")

# Security warning
st.warning("⚠️ Only load PKL files from trusted sources!")

uploaded_file = st.file_uploader("Choose a PKL file", type="pkl")

def display_content(data, depth=0):
    """Recursively display content from pickle file"""
    indent = "&nbsp;" * depth * 4
    
    if isinstance(data, (dict, pd.core.frame.DataFrame)) and not isinstance(data, pd.Series):
        if isinstance(data, pd.core.frame.DataFrame):
            st.subheader("📊 DataFrame Contents")
            st.dataframe(data)
            st.write("Shape:", data.shape)
            st.write("Columns:", list(data.columns))
        else:
            for key, value in data.items():
                st.markdown(f"{indent}**{key}:**", unsafe_allow_html=True)
                display_content(value, depth + 1)
    
    elif isinstance(data, (list, tuple)):
        st.markdown(f"{indent}List/Tuple ({len(data)} items):", unsafe_allow_html=True)
        for i, item in enumerate(data[:10]):  # Show first 10 items
            st.markdown(f"{indent}- Item {i}:", unsafe_allow_html=True)
            display_content(item, depth + 1)
        if len(data) > 10:
            st.markdown(f"{indent}... and {len(data) - 10} more items")
    
    elif isinstance(data, pd.Series):
        st.subheader("📈 Series Data")
        st.write(data)
    
    elif hasattr(data, '__dict__'):  # Custom objects
        st.markdown(f"{indent}Object of type: `{type(data).__name__}`", unsafe_allow_html=True)
        display_content(data.__dict__, depth + 1)
    
    else:
        st.markdown(f"{indent}`{str(data)}`", unsafe_allow_html=True)

if uploaded_file is not None:
    try:
        # Read the pickle file
        bytes_data = uploaded_file.read()
        data = pickle.loads(bytes_data)
        
        # Display file info
        st.success("✅ File loaded successfully!")
        col1, col2 = st.columns(2)
        with col1:
            st.write("File name:", uploaded_file.name)
        with col2:
            st.write("Data type:", type(data).__name__)
        
        # Display content based on type
        st.markdown("---")
        display_content(data)
        
        # Show raw data in expander
        with st.expander("🔧 Raw Data Structure"):
            st.write(data)
            
    except Exception as e:
        st.error(f"Error loading pickle file: {str(e)}")
        st.info("This might be due to: \n- Incompatible Python version \n- Missing dependencies \n- Corrupted file")

else:
    st.info("👆 Please upload a PKL file to get started")
    st.markdown("""
    ### Supported data types:
    - DataFrames and Series
    - Dictionaries and lists
    - Custom objects
    - Basic data types
    """)
