""""
Example 1 for Classical Finite Difference Schemes to solve Poisson Equation.

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
    August, 2023.
"""

# Library Importation
import numpy as np
from Poisson_Equation import Poisson1D_Matrix_Neumann_2
from Scripts.Graphs import Graph_1D_Stationary_1

# Problem Parameters
a       = 0
b       = 1
m       = 11
f       = lambda x: 0*x
u       = lambda x: 0*x

x, u_ap = Poisson1D_Matrix_Neumann_2(a, b, m, f, 2, 0)

Graph_1D_Stationary_1(a, b, m, u_ap)