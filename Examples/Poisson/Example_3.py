'''
Example 3 for Classical Finite Difference Schemes to solve the 2D Poisson Equation.

The problem to solve is:
    u(x,y)_{xx} + u(x,y)_yy = -f(x,y)

Subject to conditions:
    u(x,y)_\Omega = g(x,y)

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
from Poisson_Equation_Matrix import Poisson2D as PM
from Poisson_Equation_Iterative import Poisson2D as PI
from Scripts.Graphs import Graph_2D
from Scripts.Error_norms import Error_norms_2D

# Problem Parameters
m       = 21
f       = lambda x,y: 10*np.exp(2*x+y)
u       = lambda x,y: 2*np.exp(2*x+y)

# Mesh generation
x      = np.linspace(0, 1, m)                                           # x Discretization.
y      = np.linspace(0, 1, m)                                           # y Discretization.
x, y   = np.meshgrid(x, y)                                              # Mesh generation.

# Theoretical Solution
u_ex = np.zeros([m,m])
for i in range(m):
    for j in range(m):
        u_ex[i,j] = u(x[i,j], y[i,j])

# Problem solving
u_ap = PM(x, y, f, u)
# Plot the solutions
Graph_2D('2D Poisson Equation. Matrix.', x, y, u_ap, u_ex)
# Error computation
print('\n2D Poisson Equation. Matrix.')
Error_norms_2D(u_ap, u_ex)

# Problem solving
u_ap = PI(x, y, f, u)
# Plot the solutions
Graph_2D('2D Poisson Equation. Iterative.', x, y, u_ap, u_ex)
# Error computation
print('\n2D Poisson Equation. Iterative.')
Error_norms_2D(u_ap, u_ex)