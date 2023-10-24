""""
Example 2 for Classical Finite Difference Schemes to solve Diffusion Equation.

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
from Diffusion_Equation import Diffusion_2D_1
from Diffusion_Equation import Diffusion_2D_MOL_RK
from Diffusion_Equation import Diffusion_2D_CN_1
from Scripts.Graphs import Graph_2D_Transient
from Scripts.Error_norms import l2_err_t

# Problem Parameters
m       = 11
n       = 11
t       = 200
u       = lambda x,y,t,nu: np.exp(-2*np.pi**2*nu*t)*np.cos(np.pi*x)*np.cos(np.pi*y)
nu      = 0.2
x    = np.linspace(0, 1, m)
y    = np.linspace(0, 1, n)
x, y = np.meshgrid(x,y)
T    = np.linspace(0, 1, t)
u_ex = np.zeros([m,n,t])

for k in range(t):
    for i in range(m):
        for j in range(n):
            u_ex[i,j,k] = u(x[i,j], y[i,j], T[k], nu)

u_ap = Diffusion_2D_MOL_RK(m, n, t, u, nu)
Graph_2D_Transient(x, y, u_ap, u_ex)