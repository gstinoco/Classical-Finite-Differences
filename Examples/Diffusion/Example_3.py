'''
Example 3 for Classical Finite Difference Schemes to solve the Diffusion Equation with a MOL approach.

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
from Diffusion_Equation_Iterative import Diffusion1D_MOL
from Diffusion_Equation_Iterative import Diffusion2D_MOL
from Scripts.Graphs import Graph_1D
from Scripts.Graphs import Graph_2D
from Scripts.Error_norms import Error_norms_1D
from Scripts.Error_norms import Error_norms_2D

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
u_ap = Diffusion1D_MOL(x, T, u, nu, 2)
# Plot the solution
Graph_1D('1D Diffusion Equation. MOL. Runge-Kutta 2', x, u_ap, u_ex, save = True)
# Error computation
print('\n1D Diffusion Equation. MOL. Runge-Kutta 2.')
Error_norms_1D(u_ap, u_ex)

# Problem-solving
u_ap = Diffusion1D_MOL(x, T, u, nu, 3)
# Plot the solution
Graph_1D('1D Diffusion Equation. MOL. Runge-Kutta 3', x, u_ap, u_ex, save = True)
# Error computation
print('\n1D Diffusion Equation. MOL. Runge-Kutta 3.')
Error_norms_1D(u_ap, u_ex)

# Problem-solving
u_ap = Diffusion1D_MOL(x, T, u, nu, 4)
# Plot the solution
Graph_1D('1D Diffusion Equation. MOL. Runge-Kutta 4', x, u_ap, u_ex, save = True)
# Error computation
print('\n1D Diffusion Equation. MOL. Runge-Kutta 4.')
Error_norms_1D(u_ap, u_ex)


# Problem Parameters
m       = 11
n       = 11
t       = 200
u       = lambda x, y, t, nu: np.exp(-2*np.pi**2*nu*t)*np.cos(np.pi*x)*np.cos(np.pi*y)
nu      = 0.2

# Variable initialization.
u_ex = np.zeros([m, n, t])

# Mesh generation
x    = np.linspace(0, 1, m)
y    = np.linspace(0, 1, n)
x, y = np.meshgrid(x, y)
T    = np.linspace(0, 1, t)

# Theoretical Solution
for k in range(t):
    for i in range(m):
        for j in range(n):
            u_ex[i, j, k] = u(x[i, j], y[i, j], T[k], nu)

# Problem-solving
u_ap = Diffusion2D_MOL(x, y, T, u, nu, 2)
# Plot the solution
Graph_2D('2D Diffusion Equation. MOL. Runge-Kutta 2', x, y, u_ap, u_ex, save = True)
# Error computation
print('\n2D Diffusion Equation. MOL. Runge-Kutta 2.')
Error_norms_2D(u_ap, u_ex)

# Problem-solving
u_ap = Diffusion2D_MOL(x, y, T, u, nu, 3)
# Plot the solution
Graph_2D('2D Diffusion Equation. MOL. Runge-Kutta 3', x, y, u_ap, u_ex, save = True)
# Error computation
print('\n2D Diffusion Equation. MOL. Runge-Kutta 3.')
Error_norms_2D(u_ap, u_ex)

# Problem-solving
u_ap = Diffusion2D_MOL(x, y, T, u, nu, 4)
# Plot the solution
Graph_2D('2D Diffusion Equation. MOL. Runge-Kutta 4', x, y, u_ap, u_ex, save = True)
# Error computation
print('\n2D Diffusion Equation. MOL. Runge-Kutta 4.')
Error_norms_2D(u_ap, u_ex)