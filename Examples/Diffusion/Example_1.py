""""
Example 1 for Classical Finite Difference Schemes to solve Diffusion Equation.

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
from Diffusion_Equation import Diffusion_1D_0
from Scripts.Graphs import Graph_1D_Transient

# Problem Parameters
m       = 21
t       = 800
u       = lambda x,t,nu: np.exp(-nu*t)*np.sin(x)
nu      = 0.2

x, T, u_ap = Diffusion_1D_0(m, t, u, nu)

u_ex = np.zeros([m,t])

for k in range(t):
    for i in range(m):
        u_ex[i,k] = u(x[i], T[k], nu)

Graph_1D_Transient(u_ap, u_ex, x, t)