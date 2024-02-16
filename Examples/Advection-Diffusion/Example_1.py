'''
Example 1 for Classical Finite Difference Schemes to solve the 1D Advection-Diffusion Equation.

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
from Advection_Diffusion_Equation_Matrix import AdvectionDiffusion1D as AD1M
from Advection_Diffusion_Equation_Matrix import AdvectionDiffusion1D_CN as AD1CNM
from Advection_Diffusion_Equation_Iterative import AdvectionDiffusion1D as AD1I
from Advection_Diffusion_Equation_Iterative import AdvectionDiffusion1D_CN as AD1CNI
from Scripts.Graphs import Graph_1D
from Scripts.Error_norms import Error_norms_1D

# Problem Parameters
m    = 21
t    = 200
u    = lambda x, t, nu, a: (1/np.sqrt(4*t+1))*np.exp((-(x-0.5-a*t)**2)/(nu*(4*t+1)))
nu   = 0.1
a    = 0.1

# Variable initialization.
u_ex = np.zeros([m, t])

# Mesh generation
x    = np.linspace(0, 1, m)
T    = np.linspace(0, 1, t)

# Theoretical Solution
for k in range(t):
    for i in range(m):
        u_ex[i, k] = u(x[i], T[k], nu, a)

# Problem-solving
u_ap = AD1M(x, T, u, nu, a)
# Plot the solution
Graph_1D('1D Advection-Diffusion Equation. Matrix', x, u_ap, u_ex, save = True)
# Error computation
print('\n1D Poisson Equation. Matrix.')
Error_norms_1D(u_ap, u_ex)

# Problem-solving
u_ap = AD1I(x, T, u, nu, a)
# Plot the solution
Graph_1D('1D Advection-Diffusion Equation. Iterative', x, u_ap, u_ex, save = True)
# Error computation
print('\n1D Poisson Equation. Iterative.')
Error_norms_1D(u_ap, u_ex)

# Problem-solving
u_ap = AD1CNM(x, T, u, nu, a)
# Plot the solution
Graph_1D('1D Advection-Diffusion Equation. Crank-Nicolson. Matrix', x, u_ap, u_ex, save = True)
# Error computation
print('\n1D Poisson Equation. Crank-Nicolson. Matrix.')
Error_norms_1D(u_ap, u_ex)

# Problem-solving
u_ap = AD1CNI(x, T, u, nu, a)
# Plot the solution
Graph_1D('1D Advection-Diffusion Equation. Crank-Nicolson. Iterative', x, u_ap, u_ex, save = True)
# Error computation
print('\n1D Poisson Equation. Crank-Nicolson. Iterative.')
Error_norms_1D(u_ap, u_ex)