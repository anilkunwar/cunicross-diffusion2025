import streamlit as st
import os
import pickle

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTION_DIR = os.path.join(SCRIPT_DIR, "pinn_solutions")
FIGURE_DIR = os.path.join(SCRIPT_DIR, "figures")
DB_PATH = os.path.join(SCRIPT_DIR, "sunburst_data.db")

# Create figure directory if it doesn't exist
os.makedirs(FIGURE_DIR, exist_ok=True)

# Define the path to your pickle file
PICKLE_FILE = os.path.join(SOLUTION_DIR, 'your_file.pkl')

@st.cache
def load_pickle(file_path):
    """Load a pickle file and return its contents"""
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return None
    
    with open(file_path, 'rb') as f:
        try:
            return pickle.load(f)
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None

def main():
    st.title("Pickle File Viewer")
    
    # Load the pickle file using the function defined above
    loaded_data = load_pickle(PICKLE_FILE)
    
    if loaded_data is not None:
        # Display the contents of the pickle file in a Streamlit app
        st.header("Contents of your_file.pkl:")
        if isinstance(loaded_data, dict):
            for key, value in loaded_data.items():
                st.write(f"**{key}:**")
                st.write(value)
        else:
            st.write(loaded_data)

if __name__ == "__main__":
    main()
