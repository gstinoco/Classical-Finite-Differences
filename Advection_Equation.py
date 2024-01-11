"""
Classical Finite Difference Schemes to solve Advection Equation.

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
"""

# Library Importation
import numpy as np

def Advection_1D_FTCS(m, t, u, a):
    '''
        Advection_1D_FTCS

        This function solves a 1D Advection problem on a regular grid with Dirichlet boundary conditions
        it performs the approximation by using a Forward Time Centered Space stencil.

        Arguments:
            m                       Integer         Number of nodes in the grid.
            t                       Integer         Number od time steps.
            u                       Function        Function for the boundary conditions.
            a                      Float            Advective velocity in x direction.
        
        Returns:
            x           m x 1       Array           Array with the grid generated for the problem.
            u_ap        m x 1       Array           Array with the computed solution of the method.
    '''

    x    = np.linspace(0,2*np.pi,m)                                         # Mesh generation.
    T    = np.linspace(0, 1, t)                                             # Mesh generation in time.
    dx   = x[1] - x[0]                                                      # dx is defined as the space step-length.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    u_ap = np.zeros([m,t])                                                  # u_ap is initialized with zeros.
    r    = a*dt/(2*dx)                                                      # r is defined as the CFL coefficient.
    
    # Initial Conditions
    for i in range(m):                                                      # For all the grid nodes.
        u_ap[i,0] = u(x[i],T[0], a)                                         # The initial condition is assigned.
    
    # Boundary Conditions
    for k in range(t):                                                      # For all the time steps.
        u_ap[0,k]  = u(x[0],T[k], a)                                        # Boundary condition at x = 0.
        u_ap[-1,k] = u(x[-1],T[k], a)                                       # Boundary condition at x = 2pi.
    
    for k in range(t-1):                                                    # For all the time-steps.
        for i in range(1,m-1):                                              # For all the inner nodes.
            u_ap[i,k+1] = u_ap[i,k] - r*(u_ap[i+1,k] - u_ap[i-1,k])         # The new approximation is performed.

    return x, u_ap                                                          # Return the mesh and the computed solution.