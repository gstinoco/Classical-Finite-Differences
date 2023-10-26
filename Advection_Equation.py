"""
Classical Finite Difference Schemes to solve Advection Equation.

The problem to solve is:
    u(x,y,t)_{t} + au(x,y,t)_{x} + bu(x,y,t)_{y} = 0

Subject to conditions:
    u(x,y,0) = g(x,y)
    u(x,y,t)_{Boundary} = h(t)
    
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

# Library Importation
import numpy as np

def Advection_1D_FTCS(m, t, u, a):
    x    = np.linspace(0,2*np.pi,m)                                               # Mesh generation.
    T    = np.linspace(0, 1, t)                                             # Mesh generation in time.
    dx   = x[1] - x[0]                                                      # dx is defined as the space step-length.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    u_ap = np.zeros([m,t])                                                  # u_ap is initialized with zeros.
    r    = a*dt/(2*dx)
    
    # Initial Conditions
    for i in range(m):
        u_ap[i,0] = u(x[i],T[0], a)
    
    # Boundary Conditions
    for k in range(t):
        u_ap[0,k]  = u(x[0],T[k], a)
        u_ap[-1,k] = u(x[-1],T[k], a)
    
    for k in range(t-1):
        for i in range(1,m-1):
            u_ap[i,k+1] = u_ap[i,k] - r*(u_ap[i+1,k] - u_ap[i-1,k])

    return u_ap                                                             # Return the approximated solution.