[Mesh]
  type = GeneratedMesh
  dim = 2
  nx = 25 #25 #20 #50 #40 #50 #20
  ny = 25 #25 #20 #50 50 #40 #50 #20
  nz = 0
  xmin = 0
  xmax = 60 #80
  ymin = 0
  ymax = 90 #50
  zmin = 0
  zmax = 0
  elem_type = QUAD4
[]

[Variables]
  [./c1] #ccu
  order = FIRST
  family = LAGRANGE
  initial_condition = 1.0e-08 # Initially there is no Cu in liquid Sn-2.5Ag
  #scaling = 1.0E-04
  [../]
  
  [./c2] #ccu
  order = FIRST
  family = LAGRANGE
  initial_condition = 1.0e-08 # Initially there is no Ni in liquid Sn-2.5Ag
  #scaling = 1.0E-04
  [../]
[]


[Kernels]
########################################################################################
####Laplacian of c1 and c2 variables
########################################################################################
# Kernel for c1 variable
######################################################################################## 
  [./dc1dt]
    type = TimeDerivative
    variable = c1
  [../]
########################### 
  [./Laplacianc1-T1]
    type = MatDiffusion
    variable = c1
    diffusivity = D11 # self-diffusion coefficient 
  [../]
###########################
 [./Laplacianc1-T2]
    type = MatDiffusion
    variable = c1
    v = c2
    diffusivity = D12 # cross-diffusion effect (presence of gradient of Ni concentration on c1)
  [../]
########################################################################################
######################################################################################## 
  [./dc2dt]
    type = TimeDerivative
    variable = c2
  [../]
########################### 
  [./Laplacianc2-T1]
    type = MatDiffusion
    variable = c2
    diffusivity = D22 # self-diffusion coefficient 
  [../]
###########################
 [./Laplacianc2-T2]
    type = MatDiffusion
    variable = c2
    v = c1
    diffusivity = D21 # cross-diffusion effect (presence of gradient of Ni concentration on c1)
  [../]
########################################################################################
[]


[BCs]
[./neumannc1-left]
        type = NeumannBC
        boundary = 'left'
        variable = 'c1'
        value = 0
[../]
[./neumannc2-left]
       type = NeumannBC
       boundary = 'left'
       variable = 'c2'
       value = 0
[../]
[./neumannc1-right]
        type = NeumannBC
        boundary = 'right'
        variable = 'c1'
        value = 0
[../]
[./neumannc2-right]
       type = NeumannBC
       boundary = 'right'
       variable = 'c2'
       value = 0
[../]
[./c1-bottom] 
       type = DirichletBC
        boundary = 'bottom'
        variable = 'c1'
        value = 1.59e-3 #1.59e-3 #unit is mol/cc
   [../]
   [./c2-bottom] 
       type = DirichletBC
        boundary = 'bottom'
        variable = 'c2'
        value = 0.0 #1.3E-04 #0.0 #unit is mol/cc
   [../]
   [./c1-top] 
       type = DirichletBC
        boundary = 'top'
        variable = 'c1'
        value = 0.0 #unit is mol/cc
   [../]
   [./c2-top] 
       type = DirichletBC
        boundary = 'top'
        variable = 'c2'
        value = 4.0e-4 #1.3E-03 #1.25e-3 #unit is mol/cc
   [../]
[]

[Materials]
#########################################
  # Coefficients of Diffusion Matrix at 523 K
  [./diffusion_coefficients] 
      type = GenericConstantMaterial
      prop_names = 'D11 D12 D21 D22'
      prop_values = '6000.0 4270.0 3697.0 5400.0' # unit is um^2/s
      #prop_values = '100.0 -70.0 -90.0 80.0' # unit is um^2/s
      #prop_values = '0.006 -0.00427 -0.003697 0.0054' # unit is um^2/s
  [../]
[]

[Executioner]
  type = Transient
  solve_type = 'PJFNK'
  petsc_options_iname = '-pc_type -sub_pc_type   -sub_pc_factor_shift_type'
  petsc_options_value = 'asm       ilu            nonzero'
  l_max_its = 100 #100 #30
  nl_max_its = 50 #10
  l_tol = 1.0e-4
  nl_rel_tol = 1.0e-10
  nl_abs_tol = 1.0e-11

  #num_steps = 100 #2
  dt = 0.5
  end_time=200.0 #2.0 s
  #end_time=7.2E+012   #2 hr
[]

########################################################################################

[Outputs]

############################################
  exodus = true
  csv  = true
  interval = 1 #50 #50 #50 #1 #20 #1 #50 # 10 #20 #5 #50 #1 #50 #20 #10
  checkpoint = true
############################################
 #[./my_checkpoint]
  #  type = Checkpoint
  #  num_files = 4
  #  interval = 10 #5
 # [../]

[]

########################################################################################


[Debug]
  show_var_residual_norms = true
[]
