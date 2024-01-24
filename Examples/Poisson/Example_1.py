'''
Example 1 for Classical Finite Difference Schemes to solve the 1D Poisson Equation.

The problem to solve is:
    u(x)_{xx} = -(2Sin(x) + xCos(x))

Subject to conditions:
    u(x)_\Omega = xCos(x)

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
from Poisson_Equation_Matrix import Poisson1D as PM
from Poisson_Equation_Iterative import Poisson1D as PI
from Scripts.Graphs import Graph_1D
from Scripts.Error_norms import Error_norms_1D

# Problem Parameters
a       = 0
b       = 2*np.pi
m       = 10
f       = lambda x: -2*np.sin(x) - x*np.cos(x)
u       = lambda x: x*np.cos(x)

# Mesh generation
x       = np.linspace(a,b,m)

# Theoretical Solution
u_ex    = u(x)

# Problem solving
u_ap = PM(x, f, u)
# Plot the solutions
Graph_1D('1D Poisson Equation. Matrix.', x, u_ap, u_ex)
# Error computation
print('\n1D Poisson Equation. Matrix.')
Error_norms_1D(u_ap, u_ex)

# Problem solving
u_ap = PI(x, f, u)
# Plot the solutions
Graph_1D('1D Poisson Equation. Iterative.', x, u_ap, u_ex)
# Error computation
print('\n1D Poisson Equation. Iterative.')
Error_norms_1D(u_ap, u_ex)