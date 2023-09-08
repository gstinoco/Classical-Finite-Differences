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

# Library Importation
import numpy as np
from Poisson_Equation import Poisson2D_Matrix

# Problem Parameters
a       = 0
b       = 1
m       = 20
f       = lambda x,y: np.exp(x,y)
u       = lambda x,y: 0*x + 0*y

x, u_ap = Poisson2D_Matrix(21, f, u)