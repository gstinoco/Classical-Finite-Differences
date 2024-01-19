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
from Advection_Equation import Advection_1D_FTBS
from Advection_Equation import Advection_1D_FTCS
from Advection_Equation import Advection_1D_FTFS
from Advection_Equation import Advection_1D_Lax_Friedrichs_v1
from Advection_Equation import Advection_1D_Leapfrog
from Advection_Equation import Advection_1D_Lax_Friedrichs_v2
from Advection_Equation import Advection_1D_Lax_Wendroff
from Advection_Equation import Advection_1D_Bean_Warming
from Scripts.Graphs import Graph_1D

 
# Problem Parameters
m    = 161
t    = 1600
u    = lambda x,t,a: np.sin(x-a*t)
a    = 0.5

# Variable initialization.
u_ex = np.zeros([m,t])

# Mesh generation
x    = np.linspace(0, 2*np.pi, m)
T    = np.linspace(0, 1, t)

# Exact Solution
for k in range(t):
    for i in range(m):
        u_ex[i,k] = u(x[i], T[k], a)

# Problem solving
#u_ap = Advection_1D_FTBS(x, T, u, a)
# Plot the solution
#Graph_1D('1D Advection Equation. FTBS.', x, u_ap, u_ex)

# Problem solving
#u_ap = Advection_1D_FTCS(x, T, u, a)
# Plot the solution
#Graph_1D('1D Advection Equation. FTCS.', x, u_ap, u_ex)

# Problem solving
#u_ap = Advection_1D_Lax_Friedrichs_v1(x, T, u, a)
# Plot the solution
#Graph_1D('1D Advection Equation. Lax-Friedrichs v1.', x, u_ap, u_ex)
        
# Problem solving
#u_ap = Advection_1D_Leapfrog(x, T, u, a)
# Plot the solution
#Graph_1D('1D Advection Equation. Leapfrog.', x, u_ap, u_ex)

# Problem solving
#u_ap = Advection_1D_Lax_Friedrichs_v2(x, T, u, a)
# Plot the solution
#Graph_1D('1D Advection Equation. Lax-Friedrichs v2.', x, u_ap, u_ex)

# Problem solving
#u_ap = Advection_1D_Lax_Wendroff(x, T, u, a)
# Plot the solution
#Graph_1D('1D Advection Equation. Lax-Wendroff.', x, u_ap, u_ex)

# Problem solving
u_ap = Advection_1D_Bean_Warming(x, T, u, a)
# Plot the solution
Graph_1D('1D Advection Equation. Beam-Warming.', x, u_ap, u_ex)

# Exact Solution
for k in range(t):
    for i in range(m):
        u_ex[i,k] = u(x[i], T[k], -a)

# Problem solving
#u_ap = Advection_1D_FTFS(x, T, u, -a)
# Plot the solution
#Graph_1D('1D Advection Equation. FTFS.', x, u_ap, u_ex)

# Problem solving
#u_ap = Advection_1D_FTCS(x, T, u, -a)
# Plot the solution
#Graph_1D('1D Advection Equation. FTCS.', x, u_ap, u_ex)

# Problem solving
#u_ap = Advection_1D_Lax_Friedrichs_v1(x, T, u, -a)
# Plot the solution
#Graph_1D('1D Advection Equation. Lax-Friedrichs v1.', x, u_ap, u_ex)
        
# Problem solving
#u_ap = Advection_1D_Leapfrog(x, T, u, -a)
# Plot the solution
#Graph_1D('1D Advection Equation. Leapfrog', x, u_ap, u_ex)

# Problem solving
#u_ap = Advection_1D_Lax_Friedrichs_v2(x, T, u, -a)
# Plot the solution
#Graph_1D('1D Advection Equation. Lax-Friedrichs v2.', x, u_ap, u_ex)

# Problem solving
#u_ap = Advection_1D_Lax_Wendroff(x, T, u, -a)
# Plot the solution
#Graph_1D('1D Advection Equation. Lax-Wendroff.', x, u_ap, u_ex)

# Problem solving
u_ap = Advection_1D_Bean_Warming(x, T, u, -a)
# Plot the solution
Graph_1D('1D Advection Equation. Beam-Warming.', x, u_ap, u_ex)