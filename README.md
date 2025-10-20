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


Cloud based training
[![meaningtowords](https://img.shields.io/badge/trainingCuNi-streamlit-red)](https://crossdiffusionpinn-ni-cu.streamlit.app/)



------------------------------------------------------------------

*Ni UBM Bottom and Cu Pillar Top ( Cross Diffusion -Case I)*

------------------------------------------------------------------

Training History:  
[![meaningtowords](https://img.shields.io/badge/metricsNiCu-streamlit-red)](https://cunidiffusionpinn-traininghistory.streamlit.app/)

Concentration Fields:
[![meaningtowords](https://img.shields.io/badge/optimizedConc-streamlit-red)](https://visualizeconcentrationprofiles-cuni.streamlit.app/)

with limits for colorscale: [![meaningtowords](https://img.shields.io/badge/optimizedConc-streamlit-red)](https://enhancedvisualizationconcentrationprofile-cuni.streamlit.app/)

with training datasets for zero values for BCs (Gaussian-based Attention): [![meaningtowords](https://img.shields.io/badge/broadRangeConc-streamlit-red)](https://concentrationprofilecuni-broadrange-pinn.streamlit.app/)

with training datasets for 0 BCs (Transformer-inspired Attention): [![meaningtowords](https://img.shields.io/badge/broadRangeConcAttn-streamlit-red)](https://concentrationprofile-cuni-mpattentioninterpolator.streamlit.app/)
 [![meaningtowords](https://img.shields.io/badge/broadRangeConcAttnSideBC-streamlit-red)](https://attentiveinterpolator-physicsawaresidebc.streamlit.app/) 

 [![meaningtowords](https://img.shields.io/badge/advBroadRangeConcAttnSideBC-streamlit-red)](https://advanced-attentive-concentration-interpolator.streamlit.app/) 
 

 debugging interpolation errors and making it inline with pinn solutions
 [![meaningtowords](https://img.shields.io/badge/debugerrorcloudComp-streamlit-red)]( https://pinninterpolation-error-debugging.streamlit.app/)   


  cloud computing with side bc corrected:
  [![meaningtowords](https://img.shields.io/badge/cloudComp-streamlit-red)](https://crossdiffusion2dpinn-sidebc-modeling.streamlit.app/)   

   [![meaningtowords](https://img.shields.io/badge/advcloudComp-streamlit-red)](https://advanced-crossdiffusion2dpinn-sidebc.streamlit.app/)  

   pinn solutions with experimental diffusion lengths Ly
 [![meaningtowords](https://img.shields.io/badge/experimentalLy-streamlit-red)](https://crossdiffusion-ly-60-90.streamlit.app/) 

50 micrometers

[![meaningtowords](https://img.shields.io/badge/experimentalLy50-streamlit-red)](https://cross-diffusion2dpinn-shorter-joint.streamlit.app/) 

90 micrometers

[![meaningtowords](https://img.shields.io/badge/experimentalLy90-streamlit-red)](https://cross-diffusion2dpinn-longer-joint.streamlit.app/) 

Concentration Fields, Flux and Others:
[![meaningtowords](https://img.shields.io/badge/solutions-streamlit-red)](https://multivariatecrossdiffusion2d-cuni.streamlit.app/)

------------------------------------------------------------------

*Cu Pillar Bottom and Cu Pillar Top ( SelfDiffusion - Symmetry)/ Cu Substrate on Top (C_Cu=S_sat) with Preset BC on bottom for Cu (C_Cu = 0)*

------------------------------------------------------------------

Training History: 

[![meaningtowords](https://img.shields.io/badge/metricsNiCu-streamlit-red)](https://cudiffusionpinn-traininghistory.streamlit.app/)

Concentration Fields:
[![meaningtowords](https://img.shields.io/badge/optimizedConc-streamlit-red)](https://visualizeconcentrationprofiles-cu.streamlit.app/)

with limits for colorscale: [![meaningtowords](https://img.shields.io/badge/optimizedConc-streamlit-red)](https://enhancedvisualizationconcentrationprofile-cu.streamlit.app/)

with training datasets for zero values for diriclet BCs: [![meaningtowords](https://img.shields.io/badge/broadRangeConc-streamlit-red)](https://concentrationprofilescu-broadrange-pinn.streamlit.app/)

Concentration Fields, Flux and Others:
[![meaningtowords](https://img.shields.io/badge/solutions-streamlit-red)](https://multivariatecrossdiffusion2d-cu.streamlit.app/)

50 micrometers

[![meaningtowords](https://img.shields.io/badge/cuselfdiffusionLy50-streamlit-red)](https://cu-selfdiffusion2dpinn-ly-50.streamlit.app/) 

90 micrometers

[![meaningtowords](https://img.shields.io/badge/cuselfdiffusionLy90-streamlit-red)](https://cu-selfdiffusion2dpinn-ly-90.streamlit.app/) 



------------------------------------------------------------------

*Ni UBM Bottom and Ni UBM Top (  SelfDiffusion - Symmetry)/ Ni UBM Bottom (C_Ni = C_sat) with Preset BC on top for Ni (C_Ni = 0)*

------------------------------------------------------------------

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

50 micrometers

[![meaningtowords](https://img.shields.io/badge/niselfdiffusionLy50-streamlit-red)](https://ni-selfdiffusion2dpinn-ly-50.streamlit.app/) 

90 micrometers

[![meaningtowords](https://img.shields.io/badge/niselfdiffusionLy90-streamlit-red)](https://ni-selfdiffusion2dpinn-ly-90.streamlit.app/) 


# Comparison of self and cross diffusion at Ly = 50/90 micrometers 
[![meaningtowords](https://img.shields.io/badge/selfcrossdiffpost-streamlit-red)](https://self-and-cross-diffusioncomparison.streamlit.app/) 

[![meaningtowords](https://img.shields.io/badge/selfcrossdiffpost2-streamlit-red)](https://self-and-cross-diffusioncomparison2.streamlit.app/) 

[![meaningtowords](https://img.shields.io/badge/selfcrossdiffpost3-streamlit-red)](https://self-and-cross-diffusioncomparison3.streamlit.app/) 

[![meaningtowords](https://img.shields.io/badge/selfcrossdiffpost4-streamlit-red)](https://self-and-cross-diffusioncomparison4.streamlit.app/) 

[![meaningtowords](https://img.shields.io/badge/selfcrossdiffpost5-streamlit-red)](https://self-and-cross-diffusioncomparison5.streamlit.app/) 

# For analysis of uphill diffusion
[![meaningtowords](https://img.shields.io/badge/selfcrossdiffmath-streamlit-red)](https://mathematicalvisualization-crossdiffusion2dpinn.streamlit.app/) 

[![meaningtowords](https://img.shields.io/badge/selfcrossdiffmath2-streamlit-red)](https://mathematicalvisualization-crossdiffusion2dpinn2.streamlit.app/) 

[![meaningtowords](https://img.shields.io/badge/selfcrossdiffmath3-streamlit-red)](https://mathematicalvisualization-crossdiffusion2dpinn3.streamlit.app/) 

[![meaningtowords](https://img.shields.io/badge/selfcrossdiffmath4-streamlit-red)](https://mathematicalvisualization-crossdiffusion2dpinn4.streamlit.app/) 





-----------------------------------------------------------------
Multivariate crossdiffusion 2D pinn
 ------------------------------------------------------------------
30 < Ly < 120 , varying Cs_Cu and Cs_Ni

Training
[![meaningtowords](https://img.shields.io/badge/multivarpinn-streamlit-red)](https://multivariate-crossdiffusion2dpinn-training.streamlit.app/) 



 ---------------------------------------------------------


# How to perform the proper rendering of interactive visualization?

a. Tracking the boundary conditions at the appropriate geometry location:


b. Preventing the GUI crash during the movement of time slider: 


# Attention mechanisms for interpolating the solutions variables from PINN

a. Gaussian attention weights

b. Physics aware interpolation function

