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
from Scripts.Runge_Kutta import RungeKutta2_1D
from Scripts.Runge_Kutta import RungeKutta3_1D
from Scripts.Runge_Kutta import RungeKutta2_2D
from Scripts.Runge_Kutta import RungeKutta3_2D

def Diffusion_1D_0(m, t, u, nu):
    x    = np.linspace(0,1,m)                                               # Mesh generation.
    T    = np.linspace(0, 1, t)                                             # Mesh generation in time.
    dx   = x[1] - x[0]                                                      # dx is defined as the space step-length.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    u_ap = np.zeros([m,t])                                                  # u_ap is initialized with zeros.
    r    = nu*dt/(dx**2)                                                    # r has all the coefficients of the method.
    
    # Initial condition
    for i in range(m):                                                      # For all the grid nodes.
        u_ap[i,0] = u(x[i],T[0],nu)                                         # The initial condition is assigned.
    
    # Boundary conditions
    for k in range(t):                                                      # For all the time steps.
        u_ap[0,k]  = u(x[0],T[k],nu)                                        # Boundary condition at x = 0.
        u_ap[-1,k] = u(x[-1],T[k],nu)                                       # Boundary condition at x = 1.

    # Finite Differences Matrix
    dA     = np.diag(-2*np.ones(m))                                         # Main diagonal of the Matrix.
    dAp1   = np.diag(np.ones((m)-1), k = 1)                                 # Lower diagonal of the Matrix.
    dAm1   = np.diag(np.ones((m)-1), k = -1)                                # Upper diagonal of the Matrix.
    A      = dA + dAp1 + dAm1                                               # Matrix assembly.
    
    # Boundary Conditions in the Matrix
    A[0,:] = A[-1,:] = 0                                                    # Make 0 the first and last rows of the matrix.
    A[0,0] = A[-1,-1] = 1                                                   # Ones at the first and last elements to consider boundary conditions.

    A     *= r                                                              # A updated.

    # Finite Differences Approximation
    for k in range(t-1):                                                    # For all the time-steps.
        u_new = u_ap[:, k] + A@u_ap[:, k]                                   # The new approximation is computed.
        u_ap[1:-1, k+1] = u_new[1:-1]                                       # Tha approximation is saved.

    return u_ap                                                             # Return the approximated solution.

def Diffusion_1D_1(m, t, u, nu):
    x    = np.linspace(0,1,m)                                               # Mesh generation.
    T    = np.linspace(0, 1, t)                                             # Mesh generation in time.
    dx   = x[1] - x[0]                                                      # dx is defined as the space step-length.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    u_ap = np.zeros([m,t])                                                  # u_ap is initialized with zeros.
    r    = nu*dt/(dx**2)                                                    # r has all the coefficients of the method.

    # Initial condition
    for i in range(m):                                                      # For all the grid nodes.
        u_ap[i,0] = u(x[i],T[0],nu)                                         # The initial condition is assigned.
    
    # Boundary conditions
    for k in range(t):                                                      # For all the time steps.
        u_ap[0,k]  = u(x[0],T[k],nu)                                        # Boundary condition at x = 0.
        u_ap[-1,k] = u(x[-1],T[k],nu)                                       # Boundary condition at x = 1.
    
    for k in range(t-1):                                                    # For all the time-steps.
        for i in range(1,m-1):                                              # For all the inner nodes.
            u_ap[i,k+1] = u_ap[i,k] + r* \
                            (u_ap[i-1,k] - 2*u_ap[i,k] + u_ap[i+1,k])       # The new approximation is performed.
            
    return u_ap                                                             # Return the approximated solution.

def Diffusion_1D_CN_0(m, t, u, nu):
    x    = np.linspace(0,1,m)                                               # Mesh generation.
    T    = np.linspace(0, 1, t)                                             # Mesh generation in time.
    dx   = x[1] - x[0]                                                      # dx is defined as the space step-length.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    u_ap = np.zeros([m,t])                                                  # u_ap is initialized with zeros.
    r    = nu*dt/(2*dx**2)                                                  # r has all the coefficients of the method.

    # Initial condition
    for i in range(m):                                                      # For all the grid nodes.
        u_ap[i,0] = u(x[i],T[0],nu)                                         # The initial condition is assigned.
    
    # Boundary conditions
    for k in range(t):                                                      # For all the time steps.
        u_ap[0, k] = u(x[0], T[k],nu)                                       # Boundary condition at x = 0.
        u_ap[-1,k] = u(x[-1],T[k],nu)                                       # Boundary condition at x = 1.

    # Finite Differences Matrix
    dA     = np.diag(-2*np.ones(m))                                         # Main diagonal of the Matrix.
    dAp1   = np.diag(np.ones((m)-1), k = 1)                                 # Lower diagonal of the Matrix.
    dAm1   = np.diag(np.ones((m)-1), k = -1)                                # Upper diagonal of the Matrix.
    A      = dA + dAp1 + dAm1                                               # Matrix assembly.
    
    # Boundary Conditions in the Matrix
    A[0,:] = A[-1,:] = 0                                                    # Make 0 the first and last rows of the matrix.
    A[0,0] = A[-1,-1] = 1                                                   # Ones at the first and last elements to consider boundary conditions.

    A      = A*r                                                            # A updated.
    I      = np.identity(m)                                                 # I is an identity.
    B      = np.linalg.inv(I-A)                                             # The inverse matrix.

    # Finite Differences Approximation
    for k in range(t-1):                                                    # For all the time-steps.
        u_new = B@(u_ap[:,k] + A@u_ap[:,k])                                 # The new approximation is computed.
        u_ap[1:-1, k+1] = u_new[1:-1]                                       # Tha approximation is saved.

    return u_ap                                                             # Return the approximated solution.

def Diffusion_1D_CN_1(m, t, u, nu):
    x    = np.linspace(0,1,m)                                               # Mesh generation.
    T    = np.linspace(0, 1, t)                                             # Mesh generation in time.
    dx   = x[1] - x[0]                                                      # dx is defined as the space step-length.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    u_ap = np.zeros([m,t])                                                  # u_ap is initialized with zeros.
    r    = nu*dt/(2*dx**2)                                                  # r has all the coefficients of the method.
    iter = np.zeros([t])                                                    # To save the number of iterations for each time.
    tol     = np.sqrt(np.finfo(float).eps)                                  # Tolerance of the method.

    # Initial condition
    for i in range(m):                                                      # For all the grid nodes.
        u_ap[i,0] = u(x[i],T[0],nu)                                         # The initial condition is assigned.
    
    # Boundary conditions
    for k in range(t):                                                      # For all the time steps.
        u_ap[0,k]  = u(x[0],T[k],nu)                                        # Boundary condition at x = 0.
        u_ap[-1,k] = u(x[-1],T[k],nu)                                       # Boundary condition at x = 1.

    # Predictor-Corrector Method for Finite Differences
    for k in range(t-1):                                                    # For all the time-steps.
        err = 1                                                             # The difference is one to do at least one iteration.
        while err >= tol:                                                   # While the error is greater than the tolerance.
            iter[k] += 1                                                    # A new iteration is performed.
            err      = 0                                                    # The error of this iteration is 0.
            for i in range(1,m-1):                                          # For all the grid nodes.
                u_new   = (u_ap[i,k] + \
                    r*(u_ap[i-1,k] - 2*u_ap[i,k] + u_ap[i+1,k]) + \
                    r*(u_ap[i-1,k+1] + u_ap[i+1,k+1]))/(1+2*r)              # Finite Differences Approximation.
                err = max(err, abs(u_new - u_ap[i,k+1]))                    # New difference is computed.
                u_ap[i,k+1] = u_new                                         # The approximation is saved.

    return u_ap                                                             # Return the approximated solution.

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
    u_ap[1:-1,:] = RungeKutta3_1D(x, T, nu, u, u_ap)                        # Runge-Kutta method to obtain the new approximation.

    return u_ap                                                             # Return the approximated solution.

def Diffusion_2D_1(m, n, t, u, nu):
    x    = np.linspace(0,1,m)                                               # Mesh generation for x.
    y    = np.linspace(0,1,n)                                               # Mesh generation for y.
    dx   = x[1] - x[0]                                                      # dx is defined as the space step-length in x.
    dy   = y[1] - y[0]                                                      # dy is defined as the space step-length in y.
    x, y = np.meshgrid(x,y)                                                 # Mesh generation.
    T    = np.linspace(0, 1, t)                                             # Mesh generation in time.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    u_ap = np.zeros([m,n,t])                                                # u_ap is initialized with zeros.
    r_x  = nu*dt/(dx**2)                                                    # r has all the coefficients of the method.
    r_y  = nu*dt/(dy**2)                                                    # r has all the coefficients of the method.

    # Initial condition
    for i in range(m):                                                      # For all the grid nodes in x.
        for j in range(n):                                                  # For all the grid nodes in y.
            u_ap[i,j,0] = u(x[i,j],y[i,j],T[0],nu)                          # The initial condition is assigned.
    
    # Boundary conditions
    for k in range(t):                                                      # For all the time steps.
        for i in range(m):
            u_ap[i,0,k]  = u(x[i,0],  y[i,0], T[k],nu)                      # Boundary condition at y = 0.
            u_ap[i,-1,k] = u(x[i,-1], y[i,-1],T[k],nu)                      # Boundary condition at y = 1.
        for j in range(n):
            u_ap[0,j,k]  = u(x[0,j],  y[0,j], T[k],nu)                      # Boundary condition at x = 0.
            u_ap[-1,j,k] = u(x[-1,j], y[-1,j],T[k],nu)                      # Boundary condition at x = 1.
    
    for k in range(t-1):                                                    # For all the time-steps.
        for i in range(1,m-1):                                              # For all the inner nodes in y.
            for j in range(1,m-1):                                          # For all the inner nodes in x.
                u_ap[i,j,k+1] = u_ap[i,j,k] + \
                    r_x*(u_ap[i-1,j,k] - 2*u_ap[i,j,k] + u_ap[i+1,j,k]) + \
                    r_y*(u_ap[i,j-1,k] - 2*u_ap[i,j,k] + u_ap[i,j+1,k])       # The new approximation is performed.
            
    return u_ap                                                             # Return the approximated solution.

def Diffusion_2D_CN_1(m, n, t, u, nu):
    x    = np.linspace(0,1,m)                                               # Mesh generation for x.
    y    = np.linspace(0,1,n)                                               # Mesh generation for y.
    dx   = x[1] - x[0]                                                      # dx is defined as the space step-length in x.
    dy   = y[1] - y[0]                                                      # dy is defined as the space step-length in y.
    x, y = np.meshgrid(x,y)                                                 # Mesh generation.
    T    = np.linspace(0, 1, t)                                             # Mesh generation in time.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    u_ap = np.zeros([m,n,t])                                                # u_ap is initialized with zeros.
    r_x  = nu*dt/(2*dx**2)                                                  # r has all the coefficients of the method.
    r_y  = nu*dt/(2*dy**2)                                                  # r has all the coefficients of the method.
    iter = np.zeros([t])                                                    # To save the number of iterations for each time.
    tol  = np.sqrt(np.finfo(float).eps)                                     # Tolerance of the method.

    # Initial condition
    for i in range(m):                                                      # For all the grid nodes in x.
        for j in range(n):                                                  # For all the grid nodes in y.
            u_ap[i,j,0] = u(x[i,j],y[i,j],T[0],nu)                          # The initial condition is assigned.
    
    # Boundary conditions
    for k in range(t):                                                      # For all the time steps.
        for i in range(m):
            u_ap[i,0,k]  = u(x[i,0],  y[i,0], T[k],nu)                      # Boundary condition at y = 0.
            u_ap[i,-1,k] = u(x[i,-1], y[i,-1],T[k],nu)                      # Boundary condition at y = 1.
        for j in range(n):
            u_ap[0,j,k]  = u(x[0,j],  y[0,j], T[k],nu)                      # Boundary condition at x = 0.
            u_ap[-1,j,k] = u(x[-1,j], y[-1,j],T[k],nu)                      # Boundary condition at x = 1.

    # Predictor-Corrector Method for Finite Differences
    for k in range(t-1):                                                    # For all the time-steps.
        err = 1                                                             # The difference is one to do at least one iteration.
        while err >= tol:                                                   # While the error is greater than the tolerance.
            iter[k] += 1                                                    # A new iteration is performed.
            err      = 0                                                    # The error of this iteration is 0.
            for i in range(1,m-1):                                          # For all the grid nodes.
                for j in range(1,n-1):
                    u_new = ((u_ap[i,j,k] + 
                        r_x*(u_ap[i-1,j,k] - 2*u_ap[i,j,k] + u_ap[i+1,j,k] 
                        + u_ap[i-1,j,k+1] + u_ap[i+1,j,k+1]) + 
                        r_y*(u_ap[i,j-1,k] - 2*u_ap[i,j,k] + u_ap[i,j+1,k] 
                        + u_ap[i,j-1,k+1] + u_ap[i,j+1,k+1]))/
                        (1+2*r_x+2*r_y))                                    # New approximation
                    err = max(err, abs(u_new - u_ap[i,j,k+1]))              # New difference is computed.
                    u_ap[i,j,k+1] = u_new                                   # The approximation is saved.

    return u_ap                                                             # Return the approximated solution.

def Diffusion_2D_MOL_RK(m, n, t, u, nu):
    x    = np.linspace(0,1,m)                                               # Mesh generation for x.
    y    = np.linspace(0,1,n)                                               # Mesh generation for y.
    dx   = x[1] - x[0]                                                      # dx is defined as the space step-length in x.
    dy   = y[1] - y[0]                                                      # dy is defined as the space step-length in y.
    x, y = np.meshgrid(x,y)                                                 # Mesh generation.
    T    = np.linspace(0, 1, t)                                             # Mesh generation in time.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    u_ap = np.zeros([m,n,t])                                                # u_ap is initialized with zeros.

    # Initial condition
    for i in range(m):                                                      # For all the grid nodes in x.
        for j in range(n):                                                  # For all the grid nodes in y.
            u_ap[i,j,0] = u(x[i,j],y[i,j],T[0],nu)                          # The initial condition is assigned.
    
    # Boundary conditions
    for k in range(t):                                                      # For all the time steps.
        for i in range(m):
            u_ap[i,0,k]  = u(x[i,0],  y[i,0], T[k],nu)                      # Boundary condition at y = 0.
            u_ap[i,-1,k] = u(x[i,-1], y[i,-1],T[k],nu)                      # Boundary condition at y = 1.
        for j in range(n):
            u_ap[0,j,k]  = u(x[0,j],  y[0,j], T[k],nu)                      # Boundary condition at x = 0.
            u_ap[-1,j,k] = u(x[-1,j], y[-1,j],T[k],nu)                      # Boundary condition at x = 1.
    
    # Runge-Kutta
    u_ap[1:-1,1:-1,:] = RungeKutta2_2D(x, y, T, nu, u, u_ap)                # Runge-Kutta method to obtain the new approximation.

    return u_ap                                                             # Return the approximated solution.