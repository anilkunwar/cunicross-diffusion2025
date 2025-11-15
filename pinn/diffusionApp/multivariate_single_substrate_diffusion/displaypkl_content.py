import streamlit as st
import pickle
import pandas as pd
import os
from io import BytesIO

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTION_DIR = os.path.join(SCRIPT_DIR, "pinn_solutions")

st.set_page_config(page_title="PKL File Parser", layout="wide")
st.title("📄 PKL File Content Parser")

# Security warning
st.warning("⚠️ Only load PKL files from trusted sources!")

def get_pkl_files():
    """Get list of PKL files in solutions directory"""
    try:
        if not os.path.exists(SOLUTION_DIR):
            st.error(f"Solutions directory not found: {SOLUTION_DIR}")
            return []
        
        pkl_files = [f for f in os.listdir(SOLUTION_DIR) if f.endswith('.pkl')]
        return sorted(pkl_files)
    except Exception as e:
        st.error(f"Error accessing solutions directory: {str(e)}")
        return []

def load_pkl_file(filename):
    """Load a specific PKL file from solutions directory"""
    try:
        file_path = os.path.join(SOLUTION_DIR, filename)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading {filename}: {str(e)}")
        return None

def display_content(data, depth=0):
    """Recursively display content from pickle file"""
    indent = "&nbsp;" * depth * 4
    
    if isinstance(data, (dict, pd.core.frame.DataFrame)) and not isinstance(data, pd.Series):
        if isinstance(data, pd.core.frame.DataFrame):
            st.subheader("📊 DataFrame Contents")
            st.dataframe(data)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**Shape:**", data.shape)
            with col2:
                st.write("**Columns:**", list(data.columns))
            with col3:
                st.write("**Data types:**")
                st.write(data.dtypes)
        else:
            st.write(f"**Dictionary with {len(data)} keys:**")
            for key, value in data.items():
                with st.expander(f"Key: {key}"):
                    display_content(value, depth + 1)
    
    elif isinstance(data, (list, tuple)):
        st.write(f"**List/Tuple ({len(data)} items)**")
        for i, item in enumerate(data[:5]):  # Show first 5 items
            with st.expander(f"Item {i}"):
                display_content(item, depth + 1)
        if len(data) > 5:
            st.info(f"... and {len(data) - 5} more items (truncated)")
    
    elif isinstance(data, pd.Series):
        st.subheader("📈 Series Data")
        st.write(data)
        st.write("**Series info:**")
        st.write(f"Length: {len(data)}, dtype: {data.dtype}")
    
    elif hasattr(data, '__dict__'):  # Custom objects
        st.write(f"**Object:** `{type(data).__name__}`")
        display_content(data.__dict__, depth + 1)
    
    else:
        st.write(f"{indent}`{str(data)}`")

# Get available PKL files
pkl_files = get_pkl_files()

if pkl_files:
    st.success(f"✅ Found {len(pkl_files)} PKL files in solutions directory")
    
    # File selection dropdown
    selected_file = st.selectbox(
        "Select a PKL file to parse:",
        pkl_files,
        index=0 if pkl_files else None
    )
    
    if selected_file:
        st.info(f"Selected: **{selected_file}**")
        
        # Load and display the selected file
        data = load_pkl_file(selected_file)
        
        if data is not None:
            # Display file info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**File:**", selected_file)
            with col2:
                st.write("**Data type:**", type(data).__name__)
            with col3:
                if hasattr(data, '__len__'):
                    st.write("**Size:**", len(data))
                else:
                    st.write("**Size:** Single object")
            
            # Display content based on type
            st.markdown("---")
            st.subheader("📋 File Contents")
            display_content(data)
            
            # Show raw data in expander
            with st.expander("🔧 Raw Data Structure (Advanced)"):
                st.write(data)
                
            # Additional info for large datasets
            if isinstance(data, pd.DataFrame):
                with st.expander("📈 Dataset Statistics"):
                    st.write("**Description:**")
                    st.write(data.describe())
                    
        else:
            st.error("Failed to load the selected file.")
    
else:
    st.error("No PKL files found in the solutions directory.")
    st.info(f"Looking in: {SOLUTION_DIR}")

# Sidebar with additional info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This app parses and displays the contents of PKL files from the solutions directory.")
    
    st.header("📁 Files Info")
    if pkl_files:
        st.write(f"**Total files:** {len(pkl_files)}")
        for file in pkl_files:
            st.write(f"- {file}")
    else:
        st.write("No PKL files available")
    
    st.header("🛠️ Features")
    st.write("""
    - Dropdown file selection
    - DataFrame visualization
    - Nested structure parsing
    - Object introspection
    - Raw data view
    """)
