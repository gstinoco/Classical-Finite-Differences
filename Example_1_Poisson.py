'''
Example 1 for Classical Finite Difference Schemes to solve Poisson Equation.

The problem to solve is:
    \phi(x)_{xx} = -(2Sin(x) + xCos(x))

Subject to conditions:
    \phi(x)_\Omega = xCos(x)

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
'''

# Library Importation
import numpy as np
from Poisson_Equation import Poisson1D_Matrix
from Scripts.Graphs import Graph_1D_Stationary

# Problem Parameters
a = 0
b = 2*np.pi
m = 21
f = lambda x: 2*np.sin(x) + x*np.cos(x)
g = lambda x: x*np.cos(x)

x, phi_ap = Poisson1D_Matrix(a, b, m, f, g)
x         = np.linspace(a,b,m)
phi_ex    = g(x)

Graph_1D_Stationary(a, b, m, phi_ap, phi_ex)