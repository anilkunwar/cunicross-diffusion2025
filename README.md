# Cross-diffusion modeling for Cu and Ni species

Computational tool for modeling cross-diffusion of Cu and Ni in liquid Sn-2.5Ag alloy, featuring physics-informed neural network (PINN) and finite element method (FEM) approaches

Basic Model with the Concentration Field Visualization (Includes the variation in Ly):
[![Visualization via Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://crossdiffusion2d-basic-model.streamlit.app/)

# Size effect of the UBM on the diffusion mechanism

Ly of the domain is changed whereas Lx remains constant

------------------------------------------------------------------

*Ni UBM Top and Cu Pillar Bottom (Cross Diffusion - Case II)* 

------------------------------------------------------------------

Height-variation (Ly) Model with Visualization of Concentration and Flux:
[![Visualization via Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sizeeffectdiffusion2d.streamlit.app/)

Visualization of the Concentration and Flux fields for variable solubilities of solutes (Ni and Cu) and height Ly: 
[![Visualization via Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://multivariatecrossdiffusion2d.streamlit.app/ )

Optimized visualization of concentration fields
[![meaningtowords](https://img.shields.io/badge/AttentivePinnConcentration-streamlit-red)](https://visualizeconcentrationprofiles.streamlit.app/)

Visualization of the training history 
[![Visualization via Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://diffusionpinntraining-history.streamlit.app/)

*Ni UBM Bottom and Cu Pillar Top ( Cross Diffusion -Case I)*

Training History:  
[![meaningtowords](https://img.shields.io/badge/metricsNiCu-streamlit-red)](https://cunidiffusionpinn-traininghistory.streamlit.app/)

Concentration Fields:
[![meaningtowords](https://img.shields.io/badge/optimizedConc-streamlit-red)](https://visualizeconcentrationprofiles-cuni.streamlit.app/)

with limits for colorscale: [![meaningtowords](https://img.shields.io/badge/optimizedConc-streamlit-red)](https://enhancedvisualizationconcentrationprofile-cuni.streamlit.app/)

with training datasets for zero values for BCs: [![meaningtowords](https://img.shields.io/badge/broadRangeConc-streamlit-red)](https://concentrationprofilecuni-broadrange-pinn.streamlit.app/)


Concentration Fields, Flux and Others:
[![meaningtowords](https://img.shields.io/badge/solutions-streamlit-red)](https://multivariatecrossdiffusion2d-cuni.streamlit.app/)


*Cu Pillar Bottom and Cu Pillar Top ( SelfDiffusion - Symmetry)*

Training History: 

[![meaningtowords](https://img.shields.io/badge/metricsNiCu-streamlit-red)](https://cudiffusionpinn-traininghistory.streamlit.app/)

Concentration Fields:
[![meaningtowords](https://img.shields.io/badge/optimizedConc-streamlit-red)](https://visualizeconcentrationprofiles-cu.streamlit.app/)

with limits for colorscale: [![meaningtowords](https://img.shields.io/badge/optimizedConc-streamlit-red)](https://enhancedvisualizationconcentrationprofile-cu.streamlit.app/)

Concentration Fields, Flux and Others:
[![meaningtowords](https://img.shields.io/badge/solutions-streamlit-red)](https://multivariatecrossdiffusion2d-cu.streamlit.app/)


*Ni UBM Bottom and Ni UBM Top (  SelfDiffusion - Symmetry)*

Training History: 
basic
[![meaningtowords](https://img.shields.io/badge/metricsNiCu-streamlit-red)](https://nidiffusionpinn-traininghistory.streamlit.app/)
enhanced features
[![meaningtowords](https://img.shields.io/badge/metricsNiCu-streamlit-red)](https://visualization-traininghistory-cuni.streamlit.app/)


Concentration Fields:
[![meaningtowords](https://img.shields.io/badge/optimizedConc-streamlit-red)](https://visualizeconcentrationprofiles-ni.streamlit.app/)

with limits for colorscale: [![meaningtowords](https://img.shields.io/badge/optimizedConc-streamlit-red)](https://enhancedvisualizationconcentrationprofiles-ni.streamlit.app/)

with training datasets for zero values for diriclet BCs: [![meaningtowords](https://img.shields.io/badge/broadRangeConc-streamlit-red)](https://concentrationprofilesni-broadrange-pinn.streamlit.app/)

Concentration Fields, Flux and Others:
[![meaningtowords](https://img.shields.io/badge/solutions-streamlit-red)](https://multivariatecrossdiffusion2d-ni.streamlit.app/)




# How to perform the proper rendering of interactive visualization?

a. Tracking the boundary conditions at the appropriate geometry location:


b. Preventing the GUI crash during the movement of time slider: 


# Attention mechanisms for interpolating the solutions variables from PINN

a. Gaussian attention weights

b. Physics aware interpolation function

