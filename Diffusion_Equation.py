"""
Classical Finite Difference Schemes to solve Diffusion Equation.

The problem to solve is:
    u(x,y,t)_{t} = nu(u(x,y,t)_{xx} + u(x,y,t)_{yy}

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
from Scripts.Runge_Kutta import RungeKutta3

def Diffusion_1D_0(m, t, u, nu):
    x    = np.linspace(0,1,m)                                               # Mesh generation.
    T    = np.linspace(0, 1, t)                                             # Mesh generation in time.
    dx   = x[1] - x[0]                                                      # dx is defined as the space step-length.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    u_ap = np.zeros([m,t])                                                  # u_ap is initialized with zeros.
    A    = np.zeros([m-2,m])                                                # A is the Finite Differences matrix.

    # Initial condition
    for i in range(m):                                                      # For all the grid nodes.
        u_ap[i,0] = u(x[i],T[0],nu)                                         # The initial condition is assigned.
    
    # Boundary conditions
    for k in range(t):                                                      # For all the time steps.
        u_ap[0,k]  = u(x[0],T[k],nu)                                        # Boundary condition at x = 0.
        u_ap[-1,k] = u(x[-1],T[k],nu)                                       # Boundary condition at x = 1.
    
    # Finite Differences Matrix
    for i in range(m-2):                                                    # For the nodes in the range.
        A[i,i]   = 1                                                        # First diagonal of the matrix.
        A[i,i+1] = -2                                                       # Second diagonal of the matrix.
        A[i,i+2] = 1                                                        # Third diagonal of the matrix.
    
    A *= nu*(dt/(dx**2))                                                    # A updated.

    for k in range(t-1):                                                    # For all the time-steps.
        u_ap[1:-1,k+1] = u_ap[1:-1,k] + \
                         (A@u_ap[:,k])                                      # The new approximation is performed.
    
    return x, T, u_ap                                                       # Return the meshes and the approximated solution.

def Diffusion_1D_1(m, t, u, nu):
    x    = np.linspace(0,1,m)                                               # Mesh generation.
    T    = np.linspace(0, 1, t)                                             # Mesh generation in time.
    dx   = x[1] - x[0]                                                      # dx is defined as the space step-length.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    u_ap = np.zeros([m,t])                                                  # u_ap is initialized with zeros.

    # Initial condition
    for i in range(m):                                                      # For all the grid nodes.
        u_ap[i,0] = u(x[i],T[0],nu)                                         # The initial condition is assigned.
    
    # Boundary conditions
    for k in range(t):                                                      # For all the time steps.
        u_ap[0,k]  = u(x[0],T[k],nu)                                        # Boundary condition at x = 0.
        u_ap[-1,k] = u(x[-1],T[k],nu)                                       # Boundary condition at x = 1.
    
    for k in range(t-1):                                                    # For all the time-steps.
        for i in range(1,m-1):                                              # For all the inner nodes.
            u_ap[i,k+1] = u_ap[i,k] + nu*(dt/(dx**2))* \
                            (u_ap[i-1,k] - 2*u_ap[i,k] + u_ap[i+1,k])       # The new approximation is performed.
    
    return x, T, u_ap                                                       # Return the meshes and the approximated solution.

def Diffusion_1D_CN_0(m, t, u, nu):
    x    = np.linspace(0,1,m)                                               # Mesh generation.
    T    = np.linspace(0, 1, t)                                             # Mesh generation in time.
    dx   = x[1] - x[0]                                                      # dx is defined as the space step-length.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    u_ap = np.zeros([m,t])                                                  # u_ap is initialized with zeros.
    r    = nu*dt/(2*dx**2)
    iter = np.zeros([t])
    tol     = np.sqrt(np.finfo(float).eps)                                  # Tolerance of the method.

    # Initial condition
    for i in range(m):                                                      # For all the grid nodes.
        u_ap[i,0] = u(x[i],T[0],nu)                                         # The initial condition is assigned.
    
    # Boundary conditions
    for k in range(t):                                                      # For all the time steps.
        u_ap[0,k]  = u(x[0],T[k],nu)                                        # Boundary condition at x = 0.
        u_ap[-1,k] = u(x[-1],T[k],nu)                                       # Boundary condition at x = 1.

    for k in range(t-1):                                                    # For all the time-steps.
        err = 1
        while err >= tol:                                                   # While the error is greater than the tolerance.
            iter[k] += 1                                                    # A new iteration is performed.
            err      = 0                                                    # The error of this iteration is 0.
            for i in range(1,m-1):                                          # For all the grid nodes.
                t   = (u_ap[i,k] + \
                    r*(u_ap[i-1,k] - 2*u_ap[i,k] + u_ap[i+1,k]) + \
                    r*(u_ap[i-1,k+1] + u_ap[i+1,k+1]))/(1+2*r)              # Finite Differences Approximation.
                err = max(err, abs(t - u_ap[i,k+1]))                        # New error is computed.
                u_ap[i,k+1] = t                                             # The approximation is saved.
        #print('Se hicieron ', iter[k], 'iteraciones para el paso ', k)
    
    return x, T, u_ap                                                       # Return the meshes and the approximated solution.

def Diffusion_1D_MOL_RK(m, t, u, nu):
    x    = np.linspace(0,1,m)                                               # Mesh generation.
    T    = np.linspace(0, 1, t)                                             # Mesh generation in time.
    dx   = x[1] - x[0]                                                      # dx is defined as the space step-length.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    u_ap = np.zeros([m,t])                                                  # u_ap is initialized with zeros.

    # Initial condition
    for i in range(m):                                                      # For all the grid nodes.
        u_ap[i,0] = u(x[i],T[0],nu)                                         # The initial condition is assigned.
    
    # Boundary conditions
    for k in range(t):                                                      # For all the time steps.
        u_ap[0,k]  = u(x[0],T[k],nu)                                        # Boundary condition at x = 0.
        u_ap[-1,k] = u(x[-1],T[k],nu)                                       # Boundary condition at x = 1.
    
    # Runge-Kutta
    u_ap[1:-1,:] = RungeKutta3(x, T, nu, u, u_ap)

    return x, T, u_ap

def Diffusion_2D_1(m, t, u, nu):
    x    = np.linspace(0,1,m)                                               # Mesh generation for x.
    y    = np.linspace(0,1,m)                                               # Mesh generation for y.
    dx   = x[1] - x[0]                                                      # dx is defined as the space step-length in x.
    dy   = y[1] - y[0]                                                      # dy is defined as the space step-length in y.
    x, y = np.meshgrid(x,y)                                                 # Mesh generation.
    T    = np.linspace(0, 1, t)                                             # Mesh generation in time.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    u_ap = np.zeros([m,m,t])                                                # u_ap is initialized with zeros.
    
    # Initial condition
    for i in range(m):                                                      # For all the grid nodes in x.
        for j in range(m):                                                  # For all the grid nodes in y.
            u_ap[i,0] = u(x[i,j],y[i,j],T[0],nu)                            # The initial condition is assigned.
    
    # Boundary conditions
    for k in range(t):                                                      # For all the time steps.
        for i in range(m):
            u_ap[i,0,k]  = u(x[i,0],  y[i,0], T[k],nu)                      # Boundary condition at x = 0.
            u_ap[i,-1,k] = u(x[i,-1], y[i,-1],T[k],nu)                      # Boundary condition at x = 1.
            u_ap[0,i,k]  = u(x[0,i],  y[0,i], T[k],nu)                      # Boundary condition at x = 0.
            u_ap[-1,i,k] = u(x[-1,i], y[-1,i],T[k],nu)                      # Boundary condition at x = 1.
    
    # Finite Differences Matrix
    for i in range(m-2):                                                    # For the nodes in the range.
        A[i,i]   = 1                                                        # First diagonal of the matrix.
        A[i,i+1] = -4                                                       # Second diagonal of the matrix.
        A[i,i+2] = 1                                                        # Third diagonal of the matrix.
    
    A *= nu*(dt/(dx**2))                                                    # A updated.

    for k in range(t-1):                                                    # For all the time-steps.
        u_ap[1:-1,k+1] = u_ap[1:-1,k] + \
                         (A@u_ap[:,k])                                      # The new approximation is performed.
    
    return x, T, u_ap                                                       # Return the meshes and the approximated solution.