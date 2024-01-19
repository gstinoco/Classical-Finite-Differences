'''
Example 2 for Classical Finite Difference Schemes to solve the 2D Advection Equation.

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
from Advection_Equation import Advection_2D_FTBS
from Advection_Equation import Advection_2D_FTFS
from Advection_Equation import Advection_2D_FTCS
from Advection_Equation import Advection_2D_Lax_Friedrichs
from Advection_Equation import Advection_2D_Lax_Wendroff
from Advection_Equation import Advection_2D_Beam_Warming
from Scripts.Graphs import Graph_2D

 
# Problem Parameters
m    = 41
n    = 41
t    = 800
u    = lambda x,y,t,a,b: 0.2*np.exp((-(x-.5-a*t)**2-(y-.5-b*t)**2)/.01)
a    = 0.4
b    = 0.4

# Variable initialization.
u_ex = np.zeros([m, n, t])

# Mesh generation
x    = np.linspace(0, 1, m)
y    = np.linspace(0, 1, n)
x, y = np.meshgrid(x, y)
T    = np.linspace(0, 1, t)

# Exact Solution
for k in range(t):
    for i in range(m):
        for j in range(n):
            u_ex[i,j,k] = u(x[i,j], y[i,j], T[k], a, b)

# Problem solving
u_ap = Advection_2D_FTBS(x, y, T, u, a, b)
# Plot the solution
Graph_2D('2D Advection Equation. FTBS.', x, y, u_ap, u_ex)

# Problem solving
u_ap = Advection_2D_FTCS(x, y, T, u, a, b)
# Plot the solution
Graph_2D('2D Advection Equation. FTCS.', x, y, u_ap, u_ex)

# Problem solving
u_ap = Advection_2D_Lax_Friedrichs(x, y, T, u, a, b)
# Plot the solution
Graph_2D('2D Advection Equation. Lax-Friedrichs.', x, y, u_ap, u_ex)

# Problem solving
u_ap = Advection_2D_Lax_Wendroff(x, y, T, u, a, b)
# Plot the solution
Graph_2D('2D Advection Equation. Lax-Wendroff.', x, y, u_ap, u_ex)

# Problem solving
u_ap = Advection_2D_Beam_Warming(x, y, T, u, a, b)
# Plot the solution
Graph_2D('2D Advection Equation. Beam-Warming.', x, y, u_ap, u_ex)

# Exact Solution
for k in range(t):
    for i in range(m):
        for j in range(n):
            u_ex[i,j,k] = u(x[i,j], y[i,j], T[k], -a, -b)

# Problem solving
u_ap = Advection_2D_FTFS(x, y, T, u, -a, -b)
# Plot the solution
Graph_2D('2D Advection Equation. FTFS.', x, y, u_ap, u_ex)

# Problem solving
u_ap = Advection_2D_FTCS(x, y, T, u, -a, -b)
# Plot the solution
Graph_2D('2D Advection Equation. FTCS.', x, y, u_ap, u_ex)

# Problem solving
u_ap = Advection_2D_Lax_Friedrichs(x, y, T, u, -a, -b)
# Plot the solution
Graph_2D('2D Advection Equation. Lax-Friedrichs.', x, y, u_ap, u_ex)

# Problem solving
u_ap = Advection_2D_Lax_Wendroff(x, y, T, u, -a, -b)
# Plot the solution
Graph_2D('2D Advection Equation. Lax-Wendroff.', x, y, u_ap, u_ex)

# Problem solving
u_ap = Advection_2D_Beam_Warming(x, y, T, u, -a, -b)
# Plot the solution
Graph_2D('2D Advection Equation. Beam-Warming.', x, y, u_ap, u_ex)