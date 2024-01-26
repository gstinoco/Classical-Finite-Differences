'''
Classical Finite Difference Schemes to solve Advection Equation.
The codes presented in this file correspond to an Iterative Formulation of the problem.

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

# Library Importation
import numpy as np
from Scripts.Runge_Kutta_Advection import RungeKutta2_1D
from Scripts.Runge_Kutta_Advection import RungeKutta3_1D
from Scripts.Runge_Kutta_Advection import RungeKutta4_1D
from Scripts.Runge_Kutta_Advection import RungeKutta2_2D
from Scripts.Runge_Kutta_Advection import RungeKutta3_2D
from Scripts.Runge_Kutta_Advection import RungeKutta4_2D

def Advection1D_FTCS(x, T, u, a):
    '''
        Advection1D_FTCS

        This function solves a 1D Advection problem on a regular grid with Dirichlet boundary conditions
        it performs the approximation by using a Forward Time Centered Space stencil.

        Arguments:
            x           m x 1       Array           Array with the grid generated for the problem.
            T           t x 1       Array           Array with the time grid with t partitions.
            u                       Function        Function for the boundary conditions.
            a                       Float           Advective velocity in x direction.
        
        Returns:
            u_ap        m x t       Array           Array with the computed solution of the method.
    '''

    # Variable initialization
    m    = len(x)                                                           # Size of the mesh in space.
    t    = len(T)                                                           # Size of the mesh in time.
    dx   = x[1] - x[0]                                                      # dx is defined as the space step-length.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    r    = a*dt/(2*dx)                                                      # r is defined as the CFL coefficient.

    # Solution initialization.
    u_ap = np.zeros([m, t])                                                 # u_ap is initialized with zeros.
    
    # Initial Conditions
    for i in range(m):                                                      # For all the grid nodes.
        u_ap[i, 0] = u(x[i], T[0], a)                                       # The initial condition is assigned.
    
    # Boundary Conditions
    for k in range(t):                                                      # For all the time steps.
        u_ap[0,  k] = u(x[0],  T[k], a)                                     # Boundary condition at x = 0.
        u_ap[-1, k] = u(x[-1], T[k], a)                                     # Boundary condition at x = m*dx.
    
    # Finite Differences Method
    for k in range(t-1):                                                    # For all the time-steps.
        for i in range(1, m-1):                                             # For all the inner nodes.
            u_ap[i, k+1] = u_ap[i, k] - r*(u_ap[i+1, k] - u_ap[i-1, k])     # The new approximation is performed.

    return u_ap                                                             # Return the computed solution.

def Advection1D_FTBS(x, T, u, a):
    '''
        Advection1D_FTBS

        This function solves a 1D Advection problem on a regular grid with Dirichlet boundary conditions
        it performs the approximation by using a Forward Time Backward Space stencil.

        Arguments:
            x           m x 1       Array           Array with the grid generated for the problem.
            T           t x 1       Array           Array with the time grid with t partitions.
            u                       Function        Function for the boundary conditions.
            a                       Float           Advective velocity in x direction.
        
        Returns:
            u_ap        m x t       Array           Array with the computed solution of the method.
    '''

    # Variable initialization
    m    = len(x)                                                           # Size of the mesh in space.
    t    = len(T)                                                           # Size of the mesh in time.
    dx   = x[1] - x[0]                                                      # dx is defined as the space step-length.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    r    = a*dt/dx                                                          # r is defined as the CFL coefficient.

    # Solution initialization.
    u_ap = np.zeros([m, t])                                                 # u_ap is initialized with zeros.
    
    # Initial Conditions
    for i in range(m):                                                      # For all the grid nodes.
        u_ap[i, 0] = u(x[i], T[0], a)                                       # The initial condition is assigned.
    
    # Boundary Conditions
    for k in range(t):                                                      # For all the time steps.
        u_ap[0, k] = u(x[0],  T[k], a)                                      # Boundary condition at x = 0.
    
    # Finite Differences Method
    for k in range(t-1):                                                    # For all the time-steps.
        for i in range(1, m):                                               # For all the inner nodes.
            u_ap[i, k+1] = u_ap[i, k] - r*(u_ap[i, k] - u_ap[i-1, k])       # The new approximation is performed.

    return u_ap                                                             # Return the computed solution.

def Advection1D_FTFS(x, T, u, a):
    '''
        Advection1D_FTFS

        This function solves a 1D Advection problem on a regular grid with Dirichlet boundary conditions
        it performs the approximation by using a Forward Time Forward Space stencil.

        Arguments:
            x           m x 1       Array           Array with the grid generated for the problem.
            T           t x 1       Array           Array with the time grid with t partitions.
            u                       Function        Function for the boundary conditions.
            a                       Float           Advective velocity in x direction.
        
        Returns:
            u_ap        m x t       Array           Array with the computed solution of the method.
    '''

    # Variable initialization
    m    = len(x)                                                           # Size of the mesh in space.
    t    = len(T)                                                           # Size of the mesh in time.
    dx   = x[1] - x[0]                                                      # dx is defined as the space step-length.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    r    = a*dt/dx                                                          # r is defined as the CFL coefficient.

    # Solution initialization.
    u_ap = np.zeros([m, t])                                                 # u_ap is initialized with zeros.
    
    # Initial Conditions
    for i in range(m):                                                      # For all the grid nodes.
        u_ap[i, 0] = u(x[i], T[0], a)                                       # The initial condition is assigned.
    
    # Boundary Conditions
    for k in range(t):                                                      # For all the time steps.
        u_ap[-1, k] = u(x[-1], T[k], a)                                     # Boundary condition at x = m*dx.
    
    # Finite Differences Method
    for k in range(t-1):                                                    # For all the time-steps.
        for i in range(0, m-1):                                             # For all the inner nodes.
            u_ap[i, k+1] = u_ap[i, k] - r*(u_ap[i+1, k] - u_ap[i, k])       # The new approximation is performed.

    return u_ap                                                             # Return the computed solution.

def Advection1D_Leapfrog(x, T, u, a):
    '''
        Advection1D_Leapfrog

        This function solves a 1D Advection problem on a regular grid with Dirichlet boundary conditions
        it performs the approximation by using a Leapfrog approximation and a FTCS for the second time
        step.

        Arguments:
            x           m x 1       Array           Array with the grid generated for the problem.
            T           t x 1       Array           Array with the time grid with t partitions.
            u                       Function        Function for the boundary conditions.
            a                       Float           Advective velocity in x direction.
        
        Returns:
            u_ap        m x t       Array           Array with the computed solution of the method.
    '''

    # Variable initialization
    m    = len(x)                                                           # Size of the mesh in space.
    t    = len(T)                                                           # Size of the mesh in time.
    dx   = x[1] - x[0]                                                      # dx is defined as the space step-length.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    r    = a*dt/dx                                                          # r is defined as the CFL coefficient.

    # Solution initialization.
    u_ap = np.zeros([m, t])                                                 # u_ap is initialized with zeros.
    
    # Initial Conditions
    for i in range(m):                                                      # For all the grid nodes.
        u_ap[i, 0] = u(x[i], T[0], a)                                       # The initial condition is assigned.
    
    # Boundary Conditions
    for k in range(t):                                                      # For all the time steps.
        u_ap[0,  k] = u(x[0],  T[k], a)                                     # Boundary condition at x = 0.
        u_ap[-1, k] = u(x[-1], T[k], a)                                     # Boundary condition at x = m*dx.
    
    # Finite Differences Method (First Time Step)
    for i in range(1, m-1):
        u_ap[i, 1] = u_ap[i, 0] - (r/2)*(u_ap[i+1, 0] - u_ap[i-1, 0])       # Approximation for the second time step with FTCS.

    # Finite Differences Method (All the other Time Steps)
    for k in range(1, t-1):                                                 # For all the time-steps.
        for i in range(1, m-1):                                             # For all the inner nodes.
            u_ap[i, k+1] = u_ap[i, k-1] - r*(u_ap[i+1, k] - u_ap[i-1, k])   # The new approximation is performed.

    return u_ap                                                             # Return the computed solution.

def Advection1D_Lax_Friedrichs(x, T, u, a):
    '''
        Advection1D_Lax_Friedrichs

        This function solves a 1D Advection problem on a regular grid with Dirichlet boundary conditions
        it performs the approximation by using a Lax-Friedrichs approximation with an averaged value of
        u_i^k.

        Arguments:
            x           m x 1       Array           Array with the grid generated for the problem.
            T           t x 1       Array           Array with the time grid with t partitions.
            u                       Function        Function for the boundary conditions.
            a                       Float           Advective velocity in x direction.
        
        Returns:
            u_ap        m x t       Array           Array with the computed solution of the method.
    '''

    # Variable initialization
    m    = len(x)                                                           # Size of the mesh in space.
    t    = len(T)                                                           # Size of the mesh in time.
    dx   = x[1] - x[0]                                                      # dx is defined as the space step-length.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    r    = a*dt/(2*dx)                                                      # r is defined as the CFL coefficient.

    # Solution initialization.
    u_ap = np.zeros([m, t])                                                 # u_ap is initialized with zeros.
    
    # Initial Conditions
    for i in range(m):                                                      # For all the grid nodes.
        u_ap[i, 0] = u(x[i], T[0], a)                                       # The initial condition is assigned.
    
    # Boundary Conditions
    for k in range(t):                                                      # For all the time steps.
        u_ap[0,  k] = u(x[0],  T[k], a)                                     # Boundary condition at x = 0.
        u_ap[-1, k] = u(x[-1], T[k], a)                                     # Boundary condition at x = m*dx.
    
    # Finite Differences Method
    for k in range(t-1):                                                    # For all the time-steps.
        for i in range(1, m-1):                                             # For all the inner nodes.
            u_ap[i, k+1] = (1/2)*(u_ap[i+1, k] + u_ap[i-1, k]) - \
                               r*(u_ap[i+1, k] - u_ap[i-1, k])              # The new approximation is performed.
            
    # Finite Differences Method
    for k in range(t-1):                                                    # For all the time-steps.
        for i in range(1, m-1):                                             # For all the inner nodes.
            u_ap[i, k+1] = u_ap[i, k] - r*(u_ap[i+1, k] - u_ap[i-1, k]) + \
                           (1/2)*(u_ap[i-1, k] - 2*u_ap[i, k] + u_ap[i+1, k])
                                                                            # The new approximation is performed.

    return u_ap                                                             # Return the computed solution.

def Advection1D_Lax_Friedrichs_v2(x, T, u, a):
    '''
        Advection1D_Lax_Friedrichs_v2

        This function solves a 1D Advection problem on a regular grid with Dirichlet boundary conditions
        it performs the approximation by using a Lax-Friedrichs approximation with numerical diffusion.

        Arguments:
            x           m x 1       Array           Array with the grid generated for the problem.
            T           t x 1       Array           Array with the time grid with t partitions.
            u                       Function        Function for the boundary conditions.
            a                       Float           Advective velocity in x direction.
        
        Returns:
            u_ap        m x t       Array           Array with the computed solution of the method.
    '''

    # Variable initialization
    m    = len(x)                                                           # Size of the mesh in space.
    t    = len(T)                                                           # Size of the mesh in time.
    dx   = x[1] - x[0]                                                      # dx is defined as the space step-length.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    r    = a*dt/(2*dx)                                                      # r is defined as the CFL coefficient.

    # Solution initialization.
    u_ap = np.zeros([m, t])                                                 # u_ap is initialized with zeros.
    
    # Initial Conditions
    for i in range(m):                                                      # For all the grid nodes.
        u_ap[i, 0] = u(x[i], T[0], a)                                       # The initial condition is assigned.
    
    # Boundary Conditions
    for k in range(t):                                                      # For all the time steps.
        u_ap[0,  k] = u(x[0],  T[k], a)                                     # Boundary condition at x = 0.
        u_ap[-1, k] = u(x[-1], T[k], a)                                     # Boundary condition at x = m*dx.
            
    # Finite Differences Method
    for k in range(t-1):                                                    # For all the time-steps.
        for i in range(1, m-1):                                             # For all the inner nodes.
            u_ap[i, k+1] = u_ap[i, k] - r*(u_ap[i+1, k] - u_ap[i-1, k]) + \
                           (1/2)*(u_ap[i-1, k] - 2*u_ap[i, k] + u_ap[i+1, k])
                                                                            # The new approximation is performed.

    return u_ap                                                             # Return the computed solution.

def Advection1D_Lax_Wendroff(x, T, u, a):
    '''
        Advection1D_Lax_Wendroff

        This function solves a 1D Advection problem on a regular grid with Dirichlet boundary conditions
        it performs the approximation by using a Lax-Wendroff approximation.

        Arguments:
            x           m x 1       Array           Array with the grid generated for the problem.
            T           t x 1       Array           Array with the time grid with t partitions.
            u                       Function        Function for the boundary conditions.
            a                       Float           Advective velocity in x direction.
        
        Returns:
            u_ap        m x t       Array           Array with the computed solution of the method.
    '''

    # Variable initialization
    m    = len(x)                                                           # Size of the mesh in space.
    t    = len(T)                                                           # Size of the mesh in time.
    dx   = x[1] - x[0]                                                      # dx is defined as the space step-length.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    r    = a*dt/dx                                                          # r is defined as the CFL coefficient.

    # Solution initialization.
    u_ap = np.zeros([m, t])                                                 # u_ap is initialized with zeros.
    
    # Initial Conditions
    for i in range(m):                                                      # For all the grid nodes.
        u_ap[i, 0] = u(x[i], T[0], a)                                       # The initial condition is assigned.
    
    # Boundary Conditions
    for k in range(t):                                                      # For all the time steps.
        u_ap[0,  k] = u(x[0],  T[k], a)                                     # Boundary condition at x = 0.
        u_ap[-1, k] = u(x[-1], T[k], a)                                     # Boundary condition at x = m*dx.
    
    # Finite Differences Method
    for k in range(t-1):                                                    # For all the time-steps.
        for i in range(1, m-1):                                             # For all the inner nodes.
            u_ap[i, k+1] = u_ap[i, k] - (r/2)*(u_ap[i+1, k] - u_ap[i-1, k]) + \
                        (r**2/2)*(u_ap[i-1, k] - 2*u_ap[i, k] + u_ap[i+1, k])
                                                                            # The new approximation is performed.

    return u_ap                                                             # Return the computed solution.

def Advection1D_Bean_Warming(x, T, u, a):
    '''
        Advection1D_Beam_Warming

        This function solves a 1D Advection problem on a regular grid with Dirichlet boundary conditions
        it performs the approximation by using a Bean-Warming second order approximation.

        Arguments:
            x           m x 1       Array           Array with the grid generated for the problem.
            T           t x 1       Array           Array with the time grid with t partitions.
            u                       Function        Function for the boundary conditions.
            a                       Float           Advective velocity in x direction.
        
        Returns:
            u_ap        m x t       Array           Array with the computed solution of the method.
    '''

    # Variable initialization
    m    = len(x)                                                           # Size of the mesh in space.
    t    = len(T)                                                           # Size of the mesh in time.
    dx   = x[1] - x[0]                                                      # dx is defined as the space step-length.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    r    = a*dt/(2*dx)                                                      # r is defined as the CFL coefficient.
    s    = (r**2)*2                                                         # s is (r^2)*2 to make the code easier to read.

    # Solution initialization.
    u_ap = np.zeros([m, t])                                                 # u_ap is initialized with zeros.
    
    # Initial Conditions
    for i in range(m):                                                      # For all the grid nodes.
        u_ap[i, 0] = u(x[i], T[0], a)                                       # The initial condition is assigned.
    
    # Boundary Conditions
    for k in range(t):                                                      # For all the time steps.
        if a > 0:                                                           # If the advective velocity is positive.
            u_ap[0, k] = u(x[0],  T[k], a)                                  # Boundary condition at x = 0.
            u_ap[1, k] = u(x[1], T[k], a)                                   # Boundary condition at x = dx.
        else:                                                               # If the advective velocity is negative.
            u_ap[-2, k] = u(x[-2], T[k], a)                                 # Boundary condition at x = m*dx - dx.
            u_ap[-1, k] = u(x[-1], T[k], a)                                 # Boundary condition at x = m*dx.
    
    # Finite Differences Method
    for k in range(t-1):                                                    # For all the time-steps.
        if a > 0:                                                           # If the advective velocity is positive.
            for i in range(2, m):                                           # For all the inner nodes.
                u_ap[i, k+1] = u_ap[i, k] - \
                        r*(3*u_ap[i, k] - 4*u_ap[i-1, k] + u_ap[i-2, k]) + \
                        s*(1*u_ap[i, k] - 2*u_ap[i-1, k] + u_ap[i-2, k])
                                                                            # The new approximation is performed.
        else:                                                               # If the advective velocity is negative.
            for i in range(m-2):                                            # For all the inner nodes.
                u_ap[i, k+1] = u_ap[i, k] - \
                        r*(-3*u_ap[i, k] + 4*u_ap[i+1, k] - u_ap[i+2, k]) + \
                        s*( 1*u_ap[i, k] - 2*u_ap[i+1, k] + u_ap[i+2, k])
                                                                            # The new approximation is performed.


    return u_ap                                                             # Return the computed solution.

def Advection_2D_FTCS(x, y, T, u, a, b):
    '''
        Advection_2D_FTCS

        This function solves a 2D Advection problem on a regular grid with Dirichlet boundary conditions
        using an Iterative formulation of the Classical Finite Difference Centered Scheme.

        Arguments:
            x           m x n       Array           Array with the x values of the nodes of the generated grid.
            y           m x n       Array           Array with the y values of the nodes of the generated grid.
            T           t x 1       Array           Array with the time grid with t partitions.
            u                       Function        Function for the boundary conditions.
            a                       Float           Advective velocity on x-axis for the problem.
            b                       Float           Advective velocity on y-axis for the problem.
        
        Returns:
            u_ap        m x n x t   Array           Array with the computed solution of the method.
    '''

    # Variable initialization
    m, n = x.shape                                                          # Size of the mesh.
    t    = len(T)                                                           # Size of the mesh in time.
    dx   = x[0, 1] - x[0, 0]                                                # dx is defined as the space step-length in x.
    dy   = y[1, 0] - y[0, 0]                                                # dy is defined as the space step-length in y.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    r_x  = a*dt/(2*dx)                                                      # r_x is defined as the CFL coefficient.
    r_y  = b*dt/(2*dy)                                                      # r_y is defined as the CFL coefficient.

    # Solution initialization.
    u_ap = np.zeros([m, n, t])                                              # u_ap is initialized with zeros.

    # Initial condition
    for i in range(m):                                                      # For all the grid nodes in x.
        for j in range(n):                                                  # For all the grid nodes in y.
            u_ap[i, j, 0] = u(x[i, j], y[i, j], T[0], a, b)                 # The initial condition is assigned.
    
    # Boundary conditions
    for k in range(t):                                                      # For all the time steps.
        for i in range(m):
            u_ap[i, 0, k]  = u(x[i, 0],  y[i, 0],  T[k], a, b)              # Boundary condition at y = 0.
            u_ap[i, -1, k] = u(x[i, -1], y[i, -1], T[k], a, b)              # Boundary condition at y = 1.
        for j in range(n):
            u_ap[0, j, k]  = u(x[0,  j], y[0,  j], T[k], a, b)              # Boundary condition at x = 0.
            u_ap[-1, j, k] = u(x[-1, j], y[-1, j], T[k], a, b)              # Boundary condition at x = 1.
    
    # Finite Differences Method
    for k in range(t-1):                                                    # For all the time-steps.
        for i in range(1, m-1):                                             # For all the inner nodes in y.
            for j in range(1, n-1):                                         # For all the inner nodes in x.
                u_ap[i, j, k+1] = u_ap[i, j, k] - \
                    r_x*(u_ap[i+1, j,   k] - u_ap[i-1, j,   k]) - \
                    r_y*(u_ap[i,   j+1, k] - u_ap[i,   j-1, k])             # The new approximation is performed.
            
    return u_ap                                                             # Return the computed solution.

def Advection_2D_FTBS(x, y, T, u, a, b):
    '''
        Advection_2D_FTBS

        This function solves a 2D Advection problem on a regular grid with Dirichlet boundary conditions
        using an Iterative formulation of the Classical Finite Difference Upwind Scheme.

        Arguments:
            x           m x n       Array           Array with the x values of the nodes of the generated grid.
            y           m x n       Array           Array with the y values of the nodes of the generated grid.
            T           t x 1       Array           Array with the time grid with t partitions.
            u                       Function        Function for the boundary conditions.
            a                       Float           Advective velocity on x-axis for the problem.
            b                       Float           Advective velocity on y-axis for the problem.
        
        Returns:
            u_ap        m x n x t   Array           Array with the computed solution of the method.
    '''

    # Variable initialization
    m, n = x.shape                                                          # Size of the mesh.
    t    = len(T)                                                           # Size of the mesh in time.
    dx   = x[0, 1] - x[0, 0]                                                # dx is defined as the space step-length in x.
    dy   = y[1, 0] - y[0, 0]                                                # dy is defined as the space step-length in y.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    r_x  = a*dt/dx                                                          # r_x is defined as the CFL coefficient.
    r_y  = b*dt/dy                                                          # r_y is defined as the CFL coefficient.

    # Solution initialization.
    u_ap = np.zeros([m, n, t])                                              # u_ap is initialized with zeros.

    # Initial condition
    for i in range(m):                                                      # For all the grid nodes in x.
        for j in range(n):                                                  # For all the grid nodes in y.
            u_ap[i, j, 0] = u(x[i, j], y[i, j], T[0], a, b)                 # The initial condition is assigned.
    
    # Boundary conditions
    for k in range(t):                                                      # For all the time steps.
        for i in range(m):
            u_ap[i, 0, k]  = u(x[i, 0],  y[i, 0],  T[k], a, b)              # Boundary condition at y = 0.
        for j in range(n):
            u_ap[0, j, k]  = u(x[0, j],  y[0, j],  T[k], a, b)              # Boundary condition at x = 0.
    
    # Finite Differences Method
    for k in range(t-1):                                                    # For all the time-steps.
        for i in range(1, m):                                               # For all the inner nodes in y.
            for j in range(1, n):                                           # For all the inner nodes in x.
                u_ap[i, j, k+1] = u_ap[i, j, k] - \
                    r_x*(u_ap[i, j, k] - u_ap[i-1, j,   k]) - \
                    r_y*(u_ap[i, j, k] - u_ap[i,   j-1, k])                 # The new approximation is performed.
            
    return u_ap                                                             # Return the computed solution.

def Advection_2D_FTFS(x, y, T, u, a, b):
    '''
        Advection_2D_FTFS

        This function solves a 2D Advection problem on a regular grid with Dirichlet boundary conditions
        using an Iterative formulation of the Classical Finite Difference Forward Time Forward Space Scheme.

        Arguments:
            x           m x n       Array           Array with the x values of the nodes of the generated grid.
            y           m x n       Array           Array with the y values of the nodes of the generated grid.
            T           t x 1       Array           Array with the time grid with t partitions.
            u                       Function        Function for the boundary conditions.
            a                       Float           Advective velocity on x-axis for the problem.
            b                       Float           Advective velocity on y-axis for the problem.
        
        Returns:
            u_ap        m x n x t   Array           Array with the computed solution of the method.
    '''

    # Variable initialization
    m, n = x.shape                                                          # Size of the mesh.
    t    = len(T)                                                           # Size of the mesh in time.
    dx   = x[0, 1] - x[0, 0]                                                # dx is defined as the space step-length in x.
    dy   = y[1, 0] - y[0, 0]                                                # dy is defined as the space step-length in y.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    r_x  = a*dt/dx                                                          # r_x is defined as the CFL coefficient.
    r_y  = b*dt/dy                                                          # r_y is defined as the CFL coefficient.

    # Solution initialization.
    u_ap = np.zeros([m, n, t])                                              # u_ap is initialized with zeros.

    # Initial condition
    for i in range(m):                                                      # For all the grid nodes in x.
        for j in range(n):                                                  # For all the grid nodes in y.
            u_ap[i, j, 0] = u(x[i, j], y[i, j], T[0], a, b)                 # The initial condition is assigned.
    
    # Boundary conditions
    for k in range(t):                                                      # For all the time steps.
        for i in range(m):
            u_ap[i, -1, k] = u(x[i, -1], y[i, -1], T[k], a, b)              # Boundary condition at y = 1.
        for j in range(n):
            u_ap[-1, j, k] = u(x[-1, j], y[-1, j], T[k], a, b)              # Boundary condition at x = 1.
    
    # Finite Differences Method
    for k in range(t-1):                                                    # For all the time-steps.
        for i in range(m-1):                                                # For all the inner nodes in y.
            for j in range(n-1):                                            # For all the inner nodes in x.
                u_ap[i, j, k+1] = u_ap[i, j, k] - \
                    r_x*(u_ap[i+1, j,   k] - u_ap[i, j, k]) - \
                    r_y*(u_ap[i,   j+1, k] - u_ap[i, j, k])                 # The new approximation is performed.
            
    return u_ap                                                             # Return the computed solution.

def Advection_2D_Lax_Wendroff(x, y, T, u, a, b):
    '''
        Advection_2D_Lax_Wendroff

        This function solves a 2D Advection problem on a regular grid with Dirichlet boundary conditions
        using an Iterative formulation of a Lax-Wendroff scheme.

        Arguments:
            x           m x n       Array           Array with the x values of the nodes of the generated grid.
            y           m x n       Array           Array with the y values of the nodes of the generated grid.
            T           t x 1       Array           Array with the time grid with t partitions.
            u                       Function        Function for the boundary conditions.
            a                       Float           Advective velocity on x-axis for the problem.
            b                       Float           Advective velocity on y-axis for the problem.
        
        Returns:
            u_ap        m x n x t   Array           Array with the computed solution of the method.
    '''

    # Variable initialization
    m, n = x.shape                                                          # Size of the mesh.
    t    = len(T)                                                           # Size of the mesh in time.
    dx   = x[0, 1] - x[0, 0]                                                # dx is defined as the space step-length in x.
    dy   = y[1, 0] - y[0, 0]                                                # dy is defined as the space step-length in y.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    r_x  = a*dt/dx                                                          # r_x is defined as the CFL coefficient.
    r_y  = b*dt/dy                                                          # r_y is defined as the CFL coefficient.

    # Solution initialization.
    u_ap = np.zeros([m, n, t])                                              # u_ap is initialized with zeros.

    # Initial condition
    for i in range(m):                                                      # For all the grid nodes in x.
        for j in range(n):                                                  # For all the grid nodes in y.
            u_ap[i, j, 0] = u(x[i, j], y[i, j], T[0], a, b)                 # The initial condition is assigned.
    
    # Boundary conditions
    for k in range(t):                                                      # For all the time steps.
        for i in range(m):
            u_ap[i, 0, k]  = u(x[i, 0],  y[i, 0],  T[k], a, b)              # Boundary condition at y = 0.
            u_ap[i, -1, k] = u(x[i, -1], y[i, -1], T[k], a, b)              # Boundary condition at y = 1.
        for j in range(n):
            u_ap[0,  j, k] = u(x[0,  j], y[0,  j], T[k], a, b)              # Boundary condition at x = 0.
            u_ap[-1, j, k] = u(x[-1, j], y[-1, j], T[k], a, b)              # Boundary condition at x = 1.
    
    # Finite Differences Method
    for k in range(t-1):                                                    # For all the time-steps.
        for i in range(1, m-1):                                             # For all the inner nodes in y.
            for j in range(1, n-1):                                         # For all the inner nodes in x.
                u_ap[i, j, k+1] = u_ap[i, j, k] - \
                (r_x/2)*(u_ap[i+1, j,   k] - u_ap[i-1, j,   k]) - \
                (r_y/2)*(u_ap[i,   j+1, k] - u_ap[i,   j-1, k]) + \
             (r_x**2/2)*(u_ap[i-1, j,   k] - 2*u_ap[i, j, k] + u_ap[i+1, j,   k]) + \
             (r_y**2/2)*(u_ap[i,   j-1, k] - 2*u_ap[i, j, k] + u_ap[i,   j+1, k])
                                                                            # The new approximation is performed.

    return u_ap                                                             # Return the computed solution.

def Advection_2D_Beam_Warming(x, y, T, u, a, b):
    '''
        Advection_2D_Beam-Warming

        This function solves a 2D Advection problem on a regular grid with Dirichlet boundary conditions
        using an Iterative formulation of a Bean-Warming scheme.

        Arguments:
            x           m x n       Array           Array with the x values of the nodes of the generated grid.
            y           m x n       Array           Array with the y values of the nodes of the generated grid.
            T           t x 1       Array           Array with the time grid with t partitions.
            u                       Function        Function for the boundary conditions.
            a                       Float           Advective velocity on x-axis for the problem.
            b                       Float           Advective velocity on y-axis for the problem.
        
        Returns:
            u_ap        m x n x t   Array           Array with the computed solution of the method.
    '''

    # Variable initialization
    m, n = x.shape                                                          # Size of the mesh.
    t    = len(T)                                                           # Size of the mesh in time.
    dx   = x[0, 1] - x[0, 0]                                                # dx is defined as the space step-length in x.
    dy   = y[1, 0] - y[0, 0]                                                # dy is defined as the space step-length in y.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    r_x  = a*dt/(2*dx)                                                      # r_x is defined as the CFL coefficient.
    r_y  = b*dt/(2*dy)                                                      # r_y is defined as the CFL coefficient.
    s_x  = (r_x**2)*2                                                       # s_x is (r_x^2)*2 to make the code easier to read.
    s_y  = (r_y**2)*2                                                       # s_y is (r_y^2)*2 to make the code easier to read.

    # Solution initialization.
    u_ap = np.zeros([m, n, t])                                              # u_ap is initialized with zeros.

    # Initial condition
    for i in range(m):                                                      # For all the grid nodes in x.
        for j in range(n):                                                  # For all the grid nodes in y.
            u_ap[i, j, 0] = u(x[i, j], y[i, j], T[0], a, b)                 # The initial condition is assigned.
    
    # Boundary conditions
    for k in range(t):                                                      # For all the time steps.
        for i in range(m):
            if b > 0:
                u_ap[i, 0, k] = u(x[i, 0], y[i, 0], T[k], a, b)             # Boundary condition at y = 0.
                u_ap[i, 1, k] = u(x[i, 1], y[i, 1], T[k], a, b)             # Boundary condition at y = dy.
            else:
                u_ap[i, -1, k] = u(x[i, -1], y[i, -1], T[k], a, b)          # Boundary condition at y = n*dy.
                u_ap[i, -2, k] = u(x[i, -2], y[i, -2], T[k], a, b)          # Boundary condition at y = n*dy-dy.
        for j in range(n):
            if a > 0:
                u_ap[0, j, k] = u(x[0, j], y[0, j], T[k], a, b)             # Boundary condition at x = 0.
                u_ap[1, j, k] = u(x[1, j], y[1, j], T[k], a, b)             # Boundary condition at x = dx.
            else:
                u_ap[-1, j, k] = u(x[-1, j], y[-1, j], T[k], a, b)          # Boundary condition at x = m*dx.
                u_ap[-2, j, k] = u(x[-2, j], y[-2, j], T[k], a, b)          # Boundary condition at x = m*dx-dx.
    
    # Finite Differences Method
    for k in range(t-1):                                                    # For all the time-steps.
        if a > 0:
            for i in range(2, m):                                           # For all the inner nodes in y.
                if b > 0:
                    for j in range(2, n):                                   # For all the inner nodes in x.
                        u_ap[i, j, k+1] = u_ap[i, j, k] - \
                        (r_x)*(3*u_ap[i, j, k] - 4*u_ap[i-1, j,   k] + u_ap[i-2, j,   k]) - \
                        (r_y)*(3*u_ap[i, j, k] - 4*u_ap[i,   j-1, k] + u_ap[i,   j-2, k]) + \
                        (s_x)*(  u_ap[i, j, k] - 2*u_ap[i-1, j,   k] + u_ap[i-2, j,   k]) + \
                        (s_y)*(  u_ap[i, j, k] - 2*u_ap[i,   j-1, k] + u_ap[i,   j-2, k])
                                                                            # The new approximation is performed.
                else:
                    for j in range(n-2):                                    # For all the inner nodes in x.
                        u_ap[i, j, k+1] = u_ap[i, j, k] - \
                        (r_x)*( 3*u_ap[i, j, k] - 4*u_ap[i-1, j,   k] + u_ap[i-2, j,   k]) - \
                        (r_y)*(-3*u_ap[i, j, k] + 4*u_ap[i,   j+1, k] - u_ap[i,   j+2, k]) + \
                        (s_x)*(   u_ap[i, j, k] - 2*u_ap[i-1, j,   k] + u_ap[i-2, j,   k]) + \
                        (s_y)*(   u_ap[i, j, k] - 2*u_ap[i,   j+1, k] + u_ap[i,   j+2, k])
                                                                            # The new approximation is performed.
        else:
            for i in range(m-2):                                            # For all the inner nodes in y.
                if b > 0:
                    for j in range(2, n):                                   # For all the inner nodes in x.
                        u_ap[i, j, k+1] = u_ap[i, j, k] - \
                        (r_x)*(-3*u_ap[i, j, k] + 4*u_ap[i+1, j,   k] - u_ap[i+2, j,   k]) - \
                        (r_y)*( 3*u_ap[i, j, k] - 4*u_ap[i,   j-1, k] + u_ap[i,   j-2, k]) + \
                        (s_x)*(   u_ap[i, j, k] - 2*u_ap[i+1, j,   k] + u_ap[i+2, j,   k]) + \
                        (s_y)*(   u_ap[i, j, k] - 2*u_ap[i,   j-1, k] + u_ap[i,   j-2, k])
                                                                            # The new approximation is performed.
                else:
                    for j in range(n-2):                                    # For all the inner nodes in x.
                        u_ap[i, j, k+1] = u_ap[i, j, k] - \
                        (r_x)*(-3*u_ap[i, j, k] + 4*u_ap[i+1, j,   k] - u_ap[i+2, j,   k]) - \
                        (r_y)*(-3*u_ap[i, j, k] + 4*u_ap[i,   j+1, k] - u_ap[i,   j+2, k]) + \
                        (s_x)*(   u_ap[i, j, k] - 2*u_ap[i+1, j,   k] + u_ap[i+2, j,   k]) + \
                        (s_y)*(   u_ap[i, j, k] - 2*u_ap[i,   j+1, k] + u_ap[i,   j+2, k])
                                                                            # The new approximation is performed.

    return u_ap                                                             # Return the computed solution.

def Advection1D_MOL(x, T, u, a, RK):
    '''
        Advection1D_MOL

        This function solves a 1D Advection problem on a regular grid with Dirichlet boundary conditions
        using a Method of Lines for Classical Finite Differences.

        Arguments:
            x           m x 1       Array           Array with the grid generated for the problem.
            T           t x 1       Array           Array with the time grid with t partitions.
            u                       Function        Function for the boundary conditions.
            a                       Float           Advective velocity in x direction.
            RK                      Integer         Runge-Kutta Order (2, 3, 4)
        
        Returns:
            u_ap        m x t       Array           Array with the computed solution of the method.
    '''

    # Variable initialization
    m    = len(x)                                                           # Size of the mesh in space.
    t    = len(T)                                                           # Size of the mesh in time.
    
    # Solution initialization.
    u_ap = np.zeros([m, t])                                                 # u_ap is initialized with zeros.
    
    # Initial Conditions
    for i in range(m):                                                      # For all the grid nodes.
        u_ap[i, 0] = u(x[i], T[0], a)                                       # The initial condition is assigned.
    
    # Boundary Conditions
    for k in range(t):                                                      # For all the time steps.
        u_ap[0,  k] = u(x[0],  T[k], a)                                     # Boundary condition at x = 0.
        u_ap[-1, k] = u(x[-1], T[k], a)                                     # Boundary condition at x = m*dx.
    
    # Runge-Kutta
    if RK == 2:
        u_ap[1:-1, :] = RungeKutta2_1D(x, T, a, u, u_ap)                   # Runge-Kutta method to obtain the new approximation.
    elif RK == 3:
        u_ap[1:-1, :] = RungeKutta3_1D(x, T, a, u, u_ap)                   # Runge-Kutta method to obtain the new approximation.
    elif RK == 4:
        u_ap[1:-1, :] = RungeKutta4_1D(x, T, a, u, u_ap)                   # Runge-Kutta method to obtain the new approximation.

    return u_ap                                                             # Return the computed solution.

def Advection2D_MOL(x, y, T, u, a, b, RK):
    '''
        Advection2D_MOL

        This function solves a 2D Advection problem on a regular grid with Dirichlet boundary conditions
        using a Method of Lines formulation of the Classical Finite Difference centered scheme.

        Arguments:
            x           m x n       Array           Array with the x values of the nodes of the generated grid.
            y           m x n       Array           Array with the y values of the nodes of the generated grid.
            T           t x 1       Array           Array with the time grid with t partitions.
            u                       Function        Function for the boundary conditions.
            a                       Float           Advective velocity on x-axis for the problem.
            b                       Float           Advective velocity on y-axis for the problem.
            RK                      Integer         Runge-Kutta Order (2, 3)
        
        Returns:
            u_ap        m x m       Array           Array with the computed solution of the method.
    '''

    # Variable initialization
    m, n = x.shape                                                          # Size of the mesh.
    t    = len(T)                                                           # Size of the mesh in time.
    
    # Solution initialization.
    u_ap = np.zeros([m, n, t])                                              # u_ap is initialized with zeros.

    # Initial condition
    for i in range(m):                                                      # For all the grid nodes in x.
        for j in range(n):                                                  # For all the grid nodes in y.
            u_ap[i, j, 0] = u(x[i, j], y[i, j], T[0], a, b)                 # The initial condition is assigned.
    
    # Boundary conditions
    for k in range(t):                                                      # For all the time steps.
        for i in range(m):
            u_ap[i, 0, k]  = u(x[i, 0],  y[i, 0],  T[k], a, b)              # Boundary condition at y = 0.
            u_ap[i, -1, k] = u(x[i, -1], y[i, -1], T[k], a, b)              # Boundary condition at y = 1.
        for j in range(n):
            u_ap[0, j, k]  = u(x[0,  j], y[0,  j], T[k], a, b)              # Boundary condition at x = 0.
            u_ap[-1, j, k] = u(x[-1, j], y[-1, j], T[k], a, b)              # Boundary condition at x = 1.

    # Runge-Kutta
    if RK == 2:
        u_ap[1:-1, 1:-1, :] = RungeKutta2_2D(x, y, T, a, b, u, u_ap)          # Runge-Kutta method to obtain the new approximation.
    elif RK == 3:
        u_ap[1:-1, 1:-1, :] = RungeKutta3_2D(x, y, T, a, b, u, u_ap)          # Runge-Kutta method to obtain the new approximation.
    elif RK == 4:
        u_ap[1:-1, 1:-1, :] = RungeKutta4_2D(x, y, T, a, b, u, u_ap)          # Runge-Kutta method to obtain the new approximation.

    return u_ap                                                             # Return the computed solution.