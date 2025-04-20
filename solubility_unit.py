import streamlit as st

# Constants
ATOMIC_MASS_CU = 63.546  # g/mol
ATOMIC_MASS_NI = 58.693  # g/mol

# Function to calculate density of liquid Sn
def calculate_density(temperature_k):
    """
    Calculate density of liquid Sn using the formula: rho = 7.2986 - 6.48E-04 * T
    
    Parameters:
    temperature_k (float): Temperature in Kelvin
    
    Returns:
    float: Density in g/cm^3
    """
    return 7.2986 - 6.48e-4 * temperature_k

# Function to convert wt.% to mol/cm^3
def wt_percent_to_molar_concentration(wt_percent, atomic_mass, density):
    """
    Convert weight percent solubility to molar concentration (mol/cm^3).
    
    Parameters:
    wt_percent (float): Weight percent of solute
    atomic_mass (float): Atomic mass of solute (g/mol)
    density (float): Density of solution (g/cm^3)
    
    Returns:
    float: Molar concentration in mol/cm^3
    """
    if wt_percent < 0:
        return None  # Invalid input
    # Molar concentration = (wt_percent * density) / (100 * atomic_mass)
    molar_conc = (wt_percent * density) / (100 * atomic_mass)
    return molar_conc

# Streamlit app
st.title("Solubility Converter: Cu and Ni in Liquid Sn")
st.write("Enter the temperature in Kelvin and the solubility of Copper (Cu) and Nickel (Ni) in weight percent (wt. %) to convert to molar concentration (mol/cm³).")

# Input fields
temperature_k = st.number_input("Temperature (K)", min_value=505.0, value=513.0, step=1.0, format="%.1f")
wt_percent_cu = st.number_input("Solubility of Cu (wt. %)", min_value=0.0, value=0.0, step=0.01, format="%.4f")
wt_percent_ni = st.number_input("Solubility of Ni (wt. %)", min_value=0.0, value=0.0, step=0.01, format="%.4f")

# Calculate density
density_sn = calculate_density(temperature_k)

# Calculate button
if st.button("Calculate"):
    # Calculate molar concentrations
    conc_cu = wt_percent_to_molar_concentration(wt_percent_cu, ATOMIC_MASS_CU, density_sn)
    conc_ni = wt_percent_to_molar_concentration(wt_percent_ni, ATOMIC_MASS_NI, density_sn)
    
    # Display results
    st.subheader("Results")
    st.write(f"Density of liquid Sn at {temperature_k} K: {density_sn:.6f} g/cm³")
    if conc_cu is not None:
        st.write(f"Molar concentration of Cu: {conc_cu:.6e} mol/cm³")
    else:
        st.error("Invalid input for Cu solubility. Please enter a non-negative value.")
        
    if conc_ni is not None:
        st.write(f"Molar concentration of Ni: {conc_ni:.6e} mol/cm³")
    else:
        st.error("Invalid input for Ni solubility. Please enter a non-negative value.")

st.write(f"Note: Calculations use the density of liquid Sn at {temperature_k} K, calculated as {density_sn:.6f} g/cm³ using the formula ρ = 7.2986 - 6.48E-04 * T. The solution is assumed to be dilute.")
