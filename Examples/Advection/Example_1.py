'''
Example 1 for Classical Finite Difference Schemes to solve the 1D Advection Equation.

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
from Advection_Equation_Matrix import Advection1D_FTCS as A1FTCSM
from Advection_Equation_Matrix import Advection1D_FTBS as A1FTBSM
from Advection_Equation_Matrix import Advection1D_FTFS as A1FTFSM
from Advection_Equation_Matrix import Advection1D_Leapfrog as A1LM
from Advection_Equation_Matrix import Advection1D_Lax_Friedrichs as A1LFM
from Advection_Equation_Matrix import Advection1D_Lax_Friedrichs_v2 as A1LFM2
from Advection_Equation_Matrix import Advection1D_Lax_Wendroff as A1LWM
from Advection_Equation_Matrix import Advection1D_Bean_Warming as A1BWM
from Advection_Equation_Iterative import Advection1D_FTCS as A1FTCSI
from Advection_Equation_Iterative import Advection1D_FTBS as A1FTBSI
from Advection_Equation_Iterative import Advection1D_FTFS as A1FTFSI
from Advection_Equation_Iterative import Advection1D_Leapfrog as A1LI
from Advection_Equation_Iterative import Advection1D_Lax_Friedrichs as A1LFI
from Advection_Equation_Iterative import Advection1D_Lax_Friedrichs_v2 as A1LFI2
from Advection_Equation_Iterative import Advection1D_Lax_Wendroff as A1LWI
from Advection_Equation_Iterative import Advection1D_Bean_Warming as A1BWI
from Scripts.Graphs import Graph_1D
from Scripts.Error_norms import Error_norms_1D

 
# Problem Parameters
m    = 21
t    = 400
u    = lambda x, t, a: np.sin(x - a*t)
a    = 0.5

# Variable initialization.
u_ex = np.zeros([m, t])

# Mesh generation
x    = np.linspace(0, 2*np.pi, m)
T    = np.linspace(0, 1, t)

# Theoretical Solution
for k in range(t):
    for i in range(m):
        u_ex[i, k] = u(x[i], T[k], a)

# Problem-solving
u_ap = A1FTCSM(x, T, u, a)
# Plot the solution
Graph_1D('1D Advection Equation. FTCS. Matrix.', x, u_ap, u_ex)
# Error computation
print('\n1D Advection Equation. FTCS. Matrix.')
Error_norms_1D(u_ap, u_ex)

# Problem-solving
u_ap = A1FTCSI(x, T, u, a)
# Plot the solution
Graph_1D('1D Advection Equation. FTCS. Iterative.', x, u_ap, u_ex)
# Error computation
print('\n1D Advection Equation. FTCS. Iterative.')
Error_norms_1D(u_ap, u_ex)

# Problem-solving
u_ap = A1FTBSM(x, T, u, a)
# Plot the solution
Graph_1D('1D Advection Equation. FTBS. Matrix.', x, u_ap, u_ex)
# Error computation
print('\n1D Advection Equation. FTBS. Matrix.')
Error_norms_1D(u_ap, u_ex)

# Problem-solving
u_ap = A1FTBSI(x, T, u, a)
# Plot the solution
Graph_1D('1D Advection Equation. FTBS. Iterative.', x, u_ap, u_ex)
# Error computation
print('\n1D Advection Equation. FTBS. Iterative.')
Error_norms_1D(u_ap, u_ex)
        
# Problem-solving
u_ap = A1LM(x, T, u, a)
# Plot the solution
Graph_1D('1D Advection Equation. Leapfrog. Matrix.', x, u_ap, u_ex)
# Error computation
print('\n1D Advection Equation. Leapfrog. Matrix.')
Error_norms_1D(u_ap, u_ex)

# Problem-solving
u_ap = A1LI(x, T, u, a)
# Plot the solution
Graph_1D('1D Advection Equation. Leapfrog. Iterative.', x, u_ap, u_ex)
# Error computation
print('\n1D Advection Equation. Leapfrog. Iterative.')
Error_norms_1D(u_ap, u_ex)

# Problem-solving
u_ap = A1LFM(x, T, u, a)
# Plot the solution
Graph_1D('1D Advection Equation. Lax-Friedrichs. Matrix.', x, u_ap, u_ex)
# Error computation
print('\n1D Advection Equation. Lax-Friedrichs. Matrix.')
Error_norms_1D(u_ap, u_ex)

# Problem-solving
u_ap = A1LFI(x, T, u, a)
# Plot the solution
Graph_1D('1D Advection Equation. Lax-Friedrichs. Iterative.', x, u_ap, u_ex)
# Error computation
print('\n1D Advection Equation. Lax-Friedrichs. Iterative.')
Error_norms_1D(u_ap, u_ex)

# Problem-solving
u_ap = A1LFM2(x, T, u, a)
# Plot the solution
Graph_1D('1D Advection Equation. Lax-Friedrichs v2. Matrix.', x, u_ap, u_ex)
# Error computation
print('\n1D Advection Equation. Lax-Friedrichs v2. Matrix.')
Error_norms_1D(u_ap, u_ex)

# Problem-solving
u_ap = A1LFI2(x, T, u, a)
# Plot the solution
Graph_1D('1D Advection Equation. Lax-Friedrichs v2. Iterative.', x, u_ap, u_ex)
# Error computation
print('\n1D Advection Equation. Lax-Friedrichs v2. Iterative.')
Error_norms_1D(u_ap, u_ex)

# Problem-solving
u_ap = A1LWM(x, T, u, a)
# Plot the solution
Graph_1D('1D Advection Equation. Lax-Wendroff. Matrix.', x, u_ap, u_ex)
# Error computation
print('\n1D Advection Equation. Lax-Wendroff. Matrix.')
Error_norms_1D(u_ap, u_ex)

# Problem-solving
u_ap = A1LWI(x, T, u, a)
# Plot the solution
Graph_1D('1D Advection Equation. Lax-Wendroff. Iterative.', x, u_ap, u_ex)
# Error computation
print('\n1D Advection Equation. Lax-Wendroff. Iterative.')
Error_norms_1D(u_ap, u_ex)
        
# Problem-solving
u_ap = A1BWM(x, T, u, a)
# Plot the solution
Graph_1D('1D Advection Equation. Beam Warming. Matrix.', x, u_ap, u_ex)
# Error computation
print('\n1D Advection Equation. Beam Warming. Matrix.')
Error_norms_1D(u_ap, u_ex)

# Problem-solving
u_ap = A1BWI(x, T, u, a)
# Plot the solution
Graph_1D('1D Advection Equation. Beam Warming. Iterative.', x, u_ap, u_ex)
# Error computation
print('\n1D Advection Equation. Beam Warming. Iterative.')
Error_norms_1D(u_ap, u_ex)

# Theoretical Solution
for k in range(t):
    for i in range(m):
        u_ex[i, k] = u(x[i], T[k], -a)

# Problem-solving
u_ap = A1FTCSM(x, T, u, -a)
# Plot the solution
Graph_1D('1D Advection Equation. FTCS. Matrix.', x, u_ap, u_ex)
# Error computation
print('\n1D Advection Equation. FTCS. Matrix.')
Error_norms_1D(u_ap, u_ex)

# Problem-solving
u_ap = A1FTCSI(x, T, u, -a)
# Plot the solution
Graph_1D('1D Advection Equation. FTCS. Iterative.', x, u_ap, u_ex)
# Error computation
print('\n1D Advection Equation. FTCS. Iterative.')
Error_norms_1D(u_ap, u_ex)
        
# Problem-solving
u_ap = A1FTFSM(x, T, u, -a)
# Plot the solution
Graph_1D('1D Advection Equation. FTFS. Matrix.', x, u_ap, u_ex)
# Error computation
print('\n1D Advection Equation. FTFS. Matrix.')
Error_norms_1D(u_ap, u_ex)

# Problem-solving
u_ap = A1FTFSI(x, T, u, -a)
# Plot the solution
Graph_1D('1D Advection Equation. FTFS. Iterative.', x, u_ap, u_ex)
# Error computation
print('\n1D Advection Equation. FTFS. Iterative.')
Error_norms_1D(u_ap, u_ex)

# Problem-solving
u_ap = A1LM(x, T, u, -a)
# Plot the solution
Graph_1D('1D Advection Equation. Leapfrog. Matrix.', x, u_ap, u_ex)
# Error computation
print('\n1D Advection Equation. Leapfrog. Matrix.')
Error_norms_1D(u_ap, u_ex)

# Problem-solving
u_ap = A1LI(x, T, u, -a)
# Plot the solution
Graph_1D('1D Advection Equation. Leapfrog. Iterative.', x, u_ap, u_ex)
# Error computation
print('\n1D Advection Equation. Leapfrog. Iterative.')
Error_norms_1D(u_ap, u_ex)
        
# Problem-solving
u_ap = A1LFM(x, T, u, -a)
# Plot the solution
Graph_1D('1D Advection Equation. Lax-Friedrichs. Matrix.', x, u_ap, u_ex)
# Error computation
print('\n1D Advection Equation. Lax-Friedrichs. Matrix.')
Error_norms_1D(u_ap, u_ex)

# Problem-solving
u_ap = A1LFI(x, T, u, -a)
# Plot the solution
Graph_1D('1D Advection Equation. Lax-Friedrichs. Iterative.', x, u_ap, u_ex)
# Error computation
print('\n1D Advection Equation. Lax-Friedrichs. Iterative.')
Error_norms_1D(u_ap, u_ex)

# Problem-solving
u_ap = A1LFM2(x, T, u, -a)
# Plot the solution
Graph_1D('1D Advection Equation. Lax-Friedrichs v2. Matrix.', x, u_ap, u_ex)
# Error computation
print('\n1D Advection Equation. Lax-Friedrichs v2. Matrix.')
Error_norms_1D(u_ap, u_ex)

# Problem-solving
u_ap = A1LFI2(x, T, u, -a)
# Plot the solution
Graph_1D('1D Advection Equation. Lax-Friedrichs v2. Iterative.', x, u_ap, u_ex)
# Error computation
print('\n1D Advection Equation. Lax-Friedrichs v2. Iterative.')
Error_norms_1D(u_ap, u_ex)
        
# Problem-solving
u_ap = A1LWM(x, T, u, -a)
# Plot the solution
Graph_1D('1D Advection Equation. Lax-Wendroff. Matrix.', x, u_ap, u_ex)
# Error computation
print('\n1D Advection Equation. Lax-Wendroff. Matrix.')
Error_norms_1D(u_ap, u_ex)

# Problem-solving
u_ap = A1LWI(x, T, u, -a)
# Plot the solution
Graph_1D('1D Advection Equation. Lax-Wendroff. Iterative.', x, u_ap, u_ex)
# Error computation
print('\n1D Advection Equation. Lax-Wendroff. Iterative.')
Error_norms_1D(u_ap, u_ex)
        
# Problem-solving
u_ap = A1BWM(x, T, u, -a)
# Plot the solution
Graph_1D('1D Advection Equation. Beam Warming. Matrix.', x, u_ap, u_ex)
# Error computation
print('\n1D Advection Equation. Beam Warming. Matrix.')
Error_norms_1D(u_ap, u_ex)

# Problem-solving
u_ap = A1BWI(x, T, u, -a)
# Plot the solution
Graph_1D('1D Advection Equation. Beam Warming. Iterative.', x, u_ap, u_ex)
# Error computation
print('\n1D Advection Equation. Beam Warming. Iterative.')
Error_norms_1D(u_ap, u_ex)