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
from Diffusion_Equation import Diffusion_1D_Matrix
from Diffusion_Equation import Diffusion_1D_Iter
from Diffusion_Equation import Diffusion_1D_CN_Matrix
from Diffusion_Equation import Diffusion_1D_CN_Iter
from Diffusion_Equation import Diffusion_1D_MOL
from Scripts.Graphs import Graph_1D

# Problem Parameters
m    = 11
t    = 200
u    = lambda x,t,nu: np.exp(-nu*t)*np.sin(x)
nu   = 0.2

# Variable initialization.
u_ex = np.zeros([m,t])

# Mesh generation
x    = np.linspace(0, 1, m)
T    = np.linspace(0, 1, t)

# Exact Solution
for k in range(t):
    for i in range(m):
        u_ex[i,k] = u(x[i], T[k], nu)

# Problem solving
u_ap = Diffusion_1D_Matrix(x, T, u, nu)
# Plot the solution
Graph_1D('1D Diffusion Equation. Matrix.', x, u_ap, u_ex)

# Problem solving
u_ap = Diffusion_1D_Iter(x, T, u, nu)
# Plot the solution
Graph_1D('1D Diffusion Equation. Iterative.', x, u_ap, u_ex)

# Problem solving
u_ap = Diffusion_1D_CN_Matrix(x, T, u, nu)
# Plot the solution
Graph_1D('1D Diffusion Equation. Crank-Nicolson. Matrix.', x, u_ap, u_ex)

# Problem solving
u_ap = Diffusion_1D_CN_Iter(x, T, u, nu)
# Plot the solution
Graph_1D('1D Diffusion Equation. Crank-Nicolson. Iterative.', x, u_ap, u_ex)

# Problem solving
u_ap = Diffusion_1D_MOL(x, T, u, nu, 2)
# Plot the solution
Graph_1D('1D Diffusion Equation. MOL. RK2.', x, u_ap, u_ex)

# Problem solving
u_ap = Diffusion_1D_MOL(x, T, u, nu, 3)
# Plot the solution
Graph_1D('1D Diffusion Equation. MOL. RK3.', x, u_ap, u_ex)