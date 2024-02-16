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
from Advection_Equation_Matrix import Advection_2D_FTCS as A2FTCSM
from Advection_Equation_Matrix import Advection_2D_FTBS as A2FTBSM
from Advection_Equation_Matrix import Advection_2D_FTFS as A2FTFSM
from Advection_Equation_Matrix import Advection_2D_Lax_Wendroff as A2LWM
from Advection_Equation_Matrix import Advection_2D_Beam_Warming as A2BWM
from Advection_Equation_Iterative import Advection_2D_FTCS as A2FTCSI
from Advection_Equation_Iterative import Advection_2D_FTBS as A2FTBSI
from Advection_Equation_Iterative import Advection_2D_FTFS as A2FTFSI
from Advection_Equation_Iterative import Advection_2D_Lax_Wendroff as A2LWI
from Advection_Equation_Iterative import Advection_2D_Beam_Warming as A2BWI
from Scripts.Graphs import Graph_2D
from Scripts.Error_norms import Error_norms_2D

# Problem Parameters
m    = 21
n    = 21
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

# Theoretical Solution
for k in range(t):
    for i in range(m):
        for j in range(n):
            u_ex[i,j,k] = u(x[i,j], y[i,j], T[k], a, b)

# Problem-solving
u_ap = A2FTCSM(x, y, T, u, a, b)
# Plot the solution
Graph_2D('2D Advection Equation. FTCS. Matrix', x, y, u_ap, u_ex, save = True)
# Error computation
print('\n2D Advection Equation. FTCS. Matrix.')
Error_norms_2D(u_ap, u_ex)

# Problem-solving
u_ap = A2FTCSI(x, y, T, u, a, b)
# Plot the solution
Graph_2D('2D Advection Equation. FTCS. Iterative', x, y, u_ap, u_ex, save = True)
# Error computation
print('\n2D Advection Equation. FTCS. Iterative.')
Error_norms_2D(u_ap, u_ex)

# Problem-solving
u_ap = A2FTBSM(x, y, T, u, a, b)
# Plot the solution
Graph_2D('2D Advection Equation. FTBS. Matrix', x, y, u_ap, u_ex, save = True)
# Error computation
print('\n2D Advection Equation. FTBS. Matrix.')
Error_norms_2D(u_ap, u_ex)

# Problem-solving
u_ap = A2FTBSI(x, y, T, u, a, b)
# Plot the solution
Graph_2D('2D Advection Equation. FTBS. Iterative', x, y, u_ap, u_ex, save = True)
# Error computation
print('\n2D Advection Equation. FTBS. Iterative.')
Error_norms_2D(u_ap, u_ex)
            
# Problem-solving
u_ap = A2LWM(x, y, T, u, a, b)
# Plot the solution
Graph_2D('2D Advection Equation. Lax-Wendroff. Matrix', x, y, u_ap, u_ex, save = True)
# Error computation
print('\n2D Advection Equation. Lax-Wendroff. Matrix.')
Error_norms_2D(u_ap, u_ex)

# Problem-solving
u_ap = A2LWI(x, y, T, u, a, b)
# Plot the solution
Graph_2D('2D Advection Equation. Lax-Wendroff. Iterative', x, y, u_ap, u_ex, save = True)
# Error computation
print('\n2D Advection Equation. Lax-Wendroff. Iterative.')
Error_norms_2D(u_ap, u_ex)

# Problem-solving
u_ap = A2BWM(x, y, T, u, a, b)
# Plot the solution
Graph_2D('2D Advection Equation. Beam Warming. Matrix', x, y, u_ap, u_ex, save = True)
# Error computation
print('\n2D Advection Equation. Beam Warming. Matrix.')
Error_norms_2D(u_ap, u_ex)

# Problem-solving
u_ap = A2BWI(x, y, T, u, a, b)
# Plot the solution
Graph_2D('2D Advection Equation. Beam Warming. Iterative', x, y, u_ap, u_ex, save = True)
# Error computation
print('\n2D Advection Equation. Beam Warming. Iterative.')
Error_norms_2D(u_ap, u_ex)

# Theoretical Solution
for k in range(t):
    for i in range(m):
        for j in range(n):
            u_ex[i,j,k] = u(x[i,j], y[i,j], T[k], -a, -b)

# Problem-solving
u_ap = A2FTCSM(x, y, T, u, -a, -b)
# Plot the solution
Graph_2D('2D Advection Equation. FTCS. Matrix', x, y, u_ap, u_ex, save = True)
# Error computation
print('\n2D Advection Equation. FTCS. Matrix.')
Error_norms_2D(u_ap, u_ex)

# Problem-solving
u_ap = A2FTCSI(x, y, T, u, -a, -b)
# Plot the solution
Graph_2D('2D Advection Equation. FTCS. Iterative', x, y, u_ap, u_ex, save = True)
# Error computation
print('\n2D Advection Equation. FTCS. Iterative.')
Error_norms_2D(u_ap, u_ex)
            
# Problem-solving
u_ap = A2FTFSM(x, y, T, u, -a, -b)
# Plot the solution
Graph_2D('2D Advection Equation. FTFS. Matrix', x, y, u_ap, u_ex, save = True)
# Error computation
print('\n2D Advection Equation. FTFS. Matrix.')
Error_norms_2D(u_ap, u_ex)

# Problem-solving
u_ap = A2FTFSI(x, y, T, u, -a, -b)
# Plot the solution
Graph_2D('2D Advection Equation. FTFS. Iterative', x, y, u_ap, u_ex, save = True)
# Error computation
print('\n2D Advection Equation. FTFS. Iterative.')
Error_norms_2D(u_ap, u_ex)
            
# Problem-solving
u_ap = A2LWM(x, y, T, u, -a, -b)
# Plot the solution
Graph_2D('2D Advection Equation. Lax-Wendroff. Matrix', x, y, u_ap, u_ex, save = True)
# Error computation
print('\n2D Advection Equation. Lax-Wendroff. Matrix.')
Error_norms_2D(u_ap, u_ex)

# Problem-solving
u_ap = A2LWI(x, y, T, u, -a, -b)
# Plot the solution
Graph_2D('2D Advection Equation. Lax-Wendroff. Iterative', x, y, u_ap, u_ex, save = True)
# Error computation
print('\n2D Advection Equation. Lax-Wendroff. Iterative.')
Error_norms_2D(u_ap, u_ex)
            
# Problem-solving
u_ap = A2BWM(x, y, T, u, -a, -b)
# Plot the solution
Graph_2D('2D Advection Equation. Beam Warming. Matrix', x, y, u_ap, u_ex, save = True)
# Error computation
print('\n2D Advection Equation. Beam Warming. Matrix.')
Error_norms_2D(u_ap, u_ex)

# Problem-solving
u_ap = A2BWI(x, y, T, u, -a, -b)
# Plot the solution
Graph_2D('2D Advection Equation. Beam Warming. Iterative', x, y, u_ap, u_ex, save = True)
# Error computation
print('\n2D Advection Equation. Beam Warming. Iterative.')
Error_norms_2D(u_ap, u_ex)