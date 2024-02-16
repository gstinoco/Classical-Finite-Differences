'''
Example 1 for Classical Finite Difference Schemes to solve the 1D Diffusion Equation.

All the codes were developed by:
    Dr. Gerardo Tinoco Guerrero
    Universidad Michoacana de San Nicolás de Hidalgo
    gerardo.tinoco@umich.mx
 
With the funding of:
    National Council of Humanities, Sciences and Technologies, CONAHCyT (Consejo Nacional de Humanidades, Ciencias y Tecnologías, CONAHCyT). México.
    Coordination of Scientific Research, CIC-UMSNH (Coordinación de la Investigación Científica de la Universidad Michoacana de San Nicolás de Hidalgo, CIC-UMSNH). México
    Aula CIMNE-Morelia. México
 
Date:
    October, 2022.

Last Modification:
    January, 2024.
'''

# Path Importation
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
root_dir    = os.path.dirname(parent_dir)
sys.path.insert(1, root_dir)

# Library Importation
import numpy as np
from Diffusion_Equation_Matrix import Diffusion1D as DEM
from Diffusion_Equation_Iterative import Diffusion1D as DEI
from Diffusion_Equation_Matrix import Diffusion1D_CN as DEM_CN
from Diffusion_Equation_Iterative import Diffusion1D_CN as DEI_CN
from Scripts.Graphs import Graph_1D
from Scripts.Error_norms import Error_norms_1D

# Problem Parameters
m    = 11
t    = 200
u    = lambda x, t, nu: np.exp(-nu*t)*np.sin(x)
nu   = 0.2

# Variable initialization.
u_ex = np.zeros([m, t])

# Mesh generation
x    = np.linspace(0, 1, m)
T    = np.linspace(0, 1, t)

# Theoretical Solution
for k in range(t):
    for i in range(m):
        u_ex[i, k] = u(x[i], T[k], nu)

# Problem-solving
u_ap = DEM(x, T, u, nu)
# Plot the solution
Graph_1D('1D Diffusion Equation. Matrix', x, u_ap, u_ex, save = True)
# Error computation
print('\n1D Poisson Equation. Matrix.')
Error_norms_1D(u_ap, u_ex)

# Problem-solving
u_ap = DEI(x, T, u, nu)
# Plot the solution
Graph_1D('1D Diffusion Equation. Iterative', x, u_ap, u_ex, save = True)
# Error computation
print('\n1D Poisson Equation. Iterative.')
Error_norms_1D(u_ap, u_ex)

# Problem-solving
u_ap = DEM_CN(x, T, u, nu)
# Plot the solution
Graph_1D('1D Diffusion Equation. Crank-Nicolson. Matrix', x, u_ap, u_ex, save = True)
# Error computation
print('\n1D Poisson Equation. Crank-Nicolson. Matrix.')
Error_norms_1D(u_ap, u_ex)

# Problem-solving
u_ap = DEI_CN(x, T, u, nu)
# Plot the solution
Graph_1D('1D Diffusion Equation. Crank-Nicolson. Iterative', x, u_ap, u_ex, save = True)
# Error computation
print('\n1D Poisson Equation. Crank-Nicolson. Iterative.')
Error_norms_1D(u_ap, u_ex)