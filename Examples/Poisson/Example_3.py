""""
Example 3 for Classical Finite Difference Schemes to solve Poisson Equation.

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
    August, 2023.
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
from Poisson_Equation import Poisson2D_Matrix_2
from Poisson_Equation import Poisson2D_Matrix
from Poisson_Equation import Poisson2D_Iter
from Scripts.Graphs import Graph_2D_Static

# Problem Parameters
a       = 0
b       = 1
m       = 20
f       = lambda x,y: 10*np.exp(2*x+y)
u       = lambda x,y: 2*np.exp(2*x+y)

# Problem solving
x, y, u_ap = Poisson2D_Matrix_2(m, f, u)

# Exact Solution
u_ex = np.zeros([m,m])
for i in range(m):
    for j in range(m):
        u_ex[i,j] = u(x[i,j], y[i,j])

# Plot the solutions
Graph_2D_Static(x, y, u_ap, u_ex)