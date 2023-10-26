""""
Example 1 for Classical Finite Difference Schemes to solve Advection Equation.

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
    October, 2023.
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
from Advection_Equation import Advection_1D_FTCS
from Scripts.Graphs import Graph_1D_Transient
from Scripts.Error_norms import l2_err_t


# Problem Parameters
m    = 10
t    = 200
u    = lambda x,t,a: np.sin(x-a*t)
a    = 0.5
x    = np.linspace(0, 2*np.pi, m)
T    = np.linspace(0, 1, t)
u_ex = np.zeros([m,t])

for k in range(t):
    for i in range(m):
        u_ex[i,k] = u(x[i], T[k], a)

u_ap = Advection_1D_FTCS(m, t, u, a)
Graph_1D_Transient(x, t, u_ap, u_ex)
er = l2_err_t(u_ap, u_ex)
print(er)