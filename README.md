# Cross-diffusion modeling for Cu and Ni species

Computational tool for modeling cross-diffusion of Cu and Ni in liquid Sn-2.5Ag alloy, featuring physics-informed neural network (PINN) and finite element method (FEM) approaches

Basic Model with the Concentration Field Visualization (Includes the variation in Ly):
[![Visualization via Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://crossdiffusion2d-basic-model.streamlit.app/)

# Size effect of the UBM on the diffusion mechanism

Ly of the domain is changed whereas Lx remains constant

Height-variation (Ly) Model with Visualization of Concentration and Flux:
[![Visualization via Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sizeeffectdiffusion2d.streamlit.app/)

# How to perform the proper rendering of interactive visualization?

a. Tracking the boundary conditions at the appropriate geometry location:


b. Preventing the GUI crash during the movement of time slider: 


# Attention mechanisms for interpolating the solutions variables from PINN

a. Gaussian attention weights

b. Physics aware interpolation function

