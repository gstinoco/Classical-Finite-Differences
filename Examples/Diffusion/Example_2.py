""""
Example 2 for Classical Finite Difference Schemes to solve Diffusion Equation.

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
    September, 2023.
"""

# Path Importation
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.insert(1, root_dir)

# Library Importation
import numpy as np
from Diffusion_Equation import Diffusion_2D
from Diffusion_Equation import Diffusion_2D_CN
from Diffusion_Equation import Diffusion_2D_MOL
from Scripts.Graphs import Graph_2D
from Scripts.Error_norms import l2_err_t
 
# Problem Parameters
m       = 11
n       = 11
t       = 200
u       = lambda x,y,t,nu: np.exp(-2*np.pi**2*nu*t)*np.cos(np.pi*x)*np.cos(np.pi*y)
nu      = 0.2

# Variable initialization.
u_ex = np.zeros([m,n,t])

# Mesh generation
x    = np.linspace(0, 1, m)
y    = np.linspace(0, 1, n)
x, y = np.meshgrid(x,y)
T    = np.linspace(0, 1, t)

# Exact Solution
for k in range(t):
    for i in range(m):
        for j in range(n):
            u_ex[i,j,k] = u(x[i,j], y[i,j], T[k], nu)

# Problem solving
u_ap = Diffusion_2D(x,y,T, u, nu)
# Plot the solution
Graph_2D('2D Diffusion Equation. Iterative.', x, y, u_ap, u_ex)

# Problem solving
u_ap = Diffusion_2D_CN(x,y,T, u, nu)
# Plot the solution
Graph_2D('2D Diffusion Equation. Crank-Nicolson.', x, y, u_ap, u_ex)

# Problem solving
u_ap = Diffusion_2D_MOL(x,y,T, u, nu, 2)
# Plot the solution
Graph_2D('2D Diffusion Equation. MOL. Runge-Kutta 2.', x, y, u_ap, u_ex)

# Problem solving
u_ap = Diffusion_2D_MOL(x,y,T, u, nu, 3)
# Plot the solution
Graph_2D('2D Diffusion Equation. MOL. Runge-Kutta 3.', x, y, u_ap, u_ex)