'''
Example 2 for Classical Finite Difference Schemes to solve the 1D Poisson Equation.

The problem to solve is:
    u(x)_{xx} = exp(x)

Subject to conditions:
    u'(x_0) = a
    u(x_m)  = b

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
from Poisson_Equation_Matrix import Poisson1D_Neumann_1 as NM_1
from Poisson_Equation_Matrix import Poisson1D_Neumann_2 as NM_2
from Poisson_Equation_Matrix import Poisson1D_Neumann_3 as NM_3
from Poisson_Equation_Iterative import Poisson1D_Neumann_1 as NI_1
from Poisson_Equation_Iterative import Poisson1D_Neumann_2 as NI_2
from Poisson_Equation_Iterative import Poisson1D_Neumann_3 as NI_3
from Scripts.Graphs import Graph_1D

# Problem Parameters
a = 0
b = 10
m = 20
f = lambda x: np.exp(x)

# Mesh generation
x = np.linspace(0, 1, m)

# Problem-solving
u_ap = NM_1(x, f, a, b)
# Plot the solution
Graph_1D('1D Poisson Equation. Neumann 1. Matrix', x, u_ap, save = True)

# Problem-solving
u_ap = NI_1(x, f, a, b)
# Plot the solution
Graph_1D('1D Poisson Equation. Neumann 1. Iterative', x, u_ap, save = True)

# Problem-solving
u_ap = NM_2(x, f, a, b)
# Plot the solution
Graph_1D('1D Poisson Equation. Neumann 2. Matrix', x, u_ap, save = True)

# Problem-solving
u_ap = NI_2(x, f, a, b)
# Plot the solution
Graph_1D('1D Poisson Equation. Neumann 2. Iterative', x, u_ap, save = True)

# Problem-solving
u_ap = NM_3(x, f, a, b)
# Plot the solution
Graph_1D('1D Poisson Equation. Neumann 3. Matrix', x, u_ap, save = True)

# Problem-solving
u_ap = NI_3(x, f, a, b)
# Plot the solution
Graph_1D('1D Poisson Equation. Neumann 3. Iterative', x, u_ap, save = True)