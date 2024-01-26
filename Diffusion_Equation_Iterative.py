'''
Classical Finite Difference Schemes to solve Diffusion Equation.
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
from Scripts.Runge_Kutta_Diffusion import RungeKutta2_1D
from Scripts.Runge_Kutta_Diffusion import RungeKutta3_1D
from Scripts.Runge_Kutta_Diffusion import RungeKutta4_1D
from Scripts.Runge_Kutta_Diffusion import RungeKutta2_2D
from Scripts.Runge_Kutta_Diffusion import RungeKutta3_2D
from Scripts.Runge_Kutta_Diffusion import RungeKutta4_2D

def Diffusion1D(x, T, u, nu):
    '''
        Diffusion1D

        This function solves a 1D Diffusion problem on a regular grid with Dirichlet boundary conditions
        using an Iterative formulation of the Classical Finite Difference Centered Scheme.

        Arguments:
            x           m x 1       Array           Array with the grid generated for the problem.
            T           t x 1       Array           Array with the time grid with t partitions.
            u                       Function        Function for the boundary conditions.
            nu                      Float           Diffusion coefficient for the problem.
        
        Returns:
            u_ap        m x t       Array           Array with the computed solution of the method.
    '''

    # Variable initialization
    m    = len(x)                                                           # Size of the mesh in space.
    t    = len(T)                                                           # Size of the mesh in time.
    dx   = x[1] - x[0]                                                      # dx is defined as the space step-length.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    r    = nu*dt/(dx**2)                                                    # r has all the coefficients of the method.

    # Solution initialization.
    u_ap = np.zeros([m, t])                                                 # u_ap is initialized with zeros.

    # Initial condition
    for i in range(m):                                                      # For all the grid nodes.
        u_ap[i, 0] = u(x[i], T[0], nu)                                      # The initial condition is assigned.
    
    # Boundary conditions
    for k in range(t):                                                      # For all the time steps.
        u_ap[0,  k] = u(x[0],  T[k], nu)                                    # Boundary condition at x = 0.
        u_ap[-1, k] = u(x[-1], T[k], nu)                                    # Boundary condition at x = 1.
    
    # Finite Differences Method
    for k in range(t-1):                                                    # For all the time-steps.
        for i in range(1, m-1):                                             # For all the inner nodes.
            u_ap[i, k+1] = u_ap[i, k] + r* \
                          (u_ap[i-1, k] - 2*u_ap[i, k] + u_ap[i+1, k])      # The new approximation is performed.
            
    return u_ap                                                             # Return the computed solution.

def Diffusion1D_CN(x, T, u, nu):
    '''
        Diffusion1D_CN

        This function solves a 1D Diffusion problem on a regular grid with Dirichlet boundary conditions
        using an Iterative formulation of the Classical Finite Difference Crank-Nicolson Scheme.

        Arguments:
            x           m x 1       Array           Array with the grid generated for the problem.
            T           t x 1       Array           Array with the time grid with t partitions.
            u                       Function        Function for the boundary conditions.
            nu                      Float           Diffusion coefficient for the problem.
        
        Returns:
            u_ap        m x t       Array           Array with the computed solution of the method.
    '''

    # Variable initialization
    m    = len(x)                                                           # Size of the mesh in space.
    t    = len(T)                                                           # Size of the mesh in time.
    dx   = x[1] - x[0]                                                      # dx is defined as the space step-length.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    r    = nu*dt/(2*dx**2)                                                  # r has all the coefficients of the method.
    iter = np.zeros([t])                                                    # To save the number of iterations for each time.
    tol  = np.sqrt(np.finfo(float).eps)                                     # Tolerance of the method.

    # Solution initialization.
    u_ap = np.zeros([m, t])                                                 # u_ap is initialized with zeros.

    # Initial condition
    for i in range(m):                                                      # For all the grid nodes.
        u_ap[i, 0] = u(x[i], T[0], nu)                                      # The initial condition is assigned.
    
    # Boundary conditions
    for k in range(t):                                                      # For all the time steps.
        u_ap[0,  k] = u(x[0],  T[k], nu)                                    # Boundary condition at x = 0.
        u_ap[-1, k] = u(x[-1], T[k], nu)                                    # Boundary condition at x = 1.

    # Finite Differences Method
    for k in range(t-1):                                                    # For all the time-steps.
        err = 1                                                             # The difference is one to do at least one iteration.
        while err >= tol:                                                   # While the error is greater than the tolerance.
            iter[k] += 1                                                    # A new iteration is performed.
            err      = 0                                                    # The error of this iteration is 0.
            for i in range(1, m-1):                                         # For all the grid nodes.
                u_new   = (u_ap[i, k] + \
                    r*(u_ap[i-1, k] - 2*u_ap[i,   k] + u_ap[i+1, k]) + \
                    r*(u_ap[i-1, k+1] + u_ap[i+1, k+1]))/(1+2*r)            # Finite Differences Approximation.
                err = max(err, abs(u_new - u_ap[i, k+1]))                   # New difference is computed.
                u_ap[i, k+1] = u_new                                        # The approximation is saved.

    return u_ap                                                             # Return the computed solution.

def Diffusion2D(x, y, T, u, nu):
    '''
        Diffusion2D

        This function solves a 2D Diffusion problem on a regular grid with Dirichlet boundary conditions
        using an Iterative formulation of the Classical Finite Difference centered scheme.

        Arguments:
            x           m x n       Array           Array with the x values of the nodes of the generated grid.
            y           m x n       Array           Array with the y values of the nodes of the generated grid.
            T           t x 1       Array           Array with the time grid with t partitions.
            u                       Function        Function for the boundary conditions.
            nu                      Float           Diffusion coefficient for the problem.
        
        Returns:
            u_ap        m x n x t   Array           Array with the computed solution of the method.
    '''

    # Variable initialization
    m, n = x.shape                                                          # Size of the mesh.
    t    = len(T)                                                           # Size of the mesh in time.
    dx   = x[0, 1] - x[0, 0]                                                # dx is defined as the space step-length in x.
    dy   = y[1, 0] - y[0, 0]                                                # dy is defined as the space step-length in y.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    r_x  = nu*dt/(dx**2)                                                    # r has all the coefficients of the method.
    r_y  = nu*dt/(dy**2)                                                    # r has all the coefficients of the method.

    # Solution initialization.
    u_ap = np.zeros([m, n, t])                                              # u_ap is initialized with zeros.
    
    # Initial condition
    for i in range(m):                                                      # For all the grid nodes in x.
        for j in range(n):                                                  # For all the grid nodes in y.
            u_ap[i, j, 0] = u(x[i, j], y[i, j], T[0], nu)                   # The initial condition is assigned.
    
    # Boundary conditions
    for k in range(t):                                                      # For all the time steps.
        for i in range(m):
            u_ap[i, 0,  k] = u(x[i, 0],  y[i, 0],  T[k], nu)                # Boundary condition at y = 0.
            u_ap[i, -1, k] = u(x[i, -1], y[i, -1], T[k], nu)                # Boundary condition at y = 1.
        for j in range(n):
            u_ap[0, j,  k] = u(x[0, j],  y[0,  j], T[k], nu)                # Boundary condition at x = 0.
            u_ap[-1, j, k] = u(x[-1, j], y[-1, j], T[k], nu)                # Boundary condition at x = 1.
    
    # Finite Differences Method
    for k in range(t-1):                                                    # For all the time-steps.
        for i in range(1, m-1):                                             # For all the inner nodes in y.
            for j in range(1, m-1):                                         # For all the inner nodes in x.
                u_ap[i, j, k+1] = u_ap[i, j, k] + \
                    r_x*(u_ap[i-1, j,   k] - 2*u_ap[i, j, k] + u_ap[i+1, j,   k]) + \
                    r_y*(u_ap[i,   j-1, k] - 2*u_ap[i, j, k] + u_ap[i,   j+1, k])
                                                                            # The new approximation is performed.
            
    return u_ap                                                             # Return the computed solution.

def Diffusion2D_CN(x, y, T, u, nu):
    '''
        Diffusion2D_CN

        This function solves a 2D Diffusion problem on a regular grid with Dirichlet boundary conditions
        using an Iterative formulation of the Classical Finite Difference Crank-Nicolson scheme.

        Arguments:
            x           m x n       Array           Array with the x values of the nodes of the generated grid.
            y           m x n       Array           Array with the y values of the nodes of the generated grid.
            T           t x 1       Array           Array with the time grid with t partitions.
            u                       Function        Function for the boundary conditions.
            nu                      Float           Diffusion coefficient for the problem.
        
        Returns:
            u_ap        m x n x t   Array           Array with the computed solution of the method.
    '''

    # Variable initialization
    m, n = x.shape                                                          # Size of the mesh.
    t    = len(T)                                                           # Size of the mesh in time.
    dx   = x[0, 1] - x[0, 0]                                                # dx is defined as the space step-length in x.
    dy   = y[1, 0] - y[0, 0]                                                # dy is defined as the space step-length in y.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    r_x  = nu*dt/(2*dx**2)                                                  # r has all the coefficients of the method.
    r_y  = nu*dt/(2*dy**2)                                                  # r has all the coefficients of the method.
    iter = np.zeros([t])                                                    # To save the number of iterations for each time.
    tol  = np.sqrt(np.finfo(float).eps)                                     # Tolerance of the method.

    # Solution initialization.
    u_ap = np.zeros([m, n, t])                                              # u_ap is initialized with zeros.

    # Initial condition
    for i in range(m):                                                      # For all the grid nodes in x.
        for j in range(n):                                                  # For all the grid nodes in y.
            u_ap[i, j, 0] = u(x[i, j], y[i, j], T[0], nu)                   # The initial condition is assigned.
    
    # Boundary conditions
    for k in range(t):                                                      # For all the time steps.
        for i in range(m):
            u_ap[i, 0,  k] = u(x[i, 0],  y[i, 0],  T[k], nu)                # Boundary condition at y = 0.
            u_ap[i, -1, k] = u(x[i, -1], y[i, -1], T[k], nu)                # Boundary condition at y = 1.
        for j in range(n):
            u_ap[0, j,  k] = u(x[0, j],  y[0, j],  T[k], nu)                # Boundary condition at x = 0.
            u_ap[-1, j, k] = u(x[-1, j], y[-1, j], T[k], nu)                # Boundary condition at x = 1.

    # Finite Differences Method
    for k in range(t-1):                                                    # For all the time-steps.
        err = 1                                                             # The difference is one to do at least one iteration.
        while err >= tol:                                                   # While the error is greater than the tolerance.
            iter[k] += 1                                                    # A new iteration is performed.
            err      = 0                                                    # The error of this iteration is 0.
            for i in range(1, m-1):                                         # For all the grid nodes.
                for j in range(1, n-1):
                    u_new = ((u_ap[i, j, k] + 
                        r_x*(u_ap[i-1, j,   k] - 2*u_ap[i,   j,   k] + u_ap[i+1, j,   k] 
                           + u_ap[i-1, j,   k+1] + u_ap[i+1, j,   k+1]) + 
                        r_y*(u_ap[i,   j-1, k] - 2*u_ap[i,   j,   k] + u_ap[i,   j+1, k] 
                           + u_ap[i,   j-1, k+1] + u_ap[i,   j+1, k+1]))/
                        (1+2*r_x+2*r_y))                                    # New approximation
                    err = max(err, abs(u_new - u_ap[i, j, k+1]))            # New difference is computed.
                    u_ap[i, j, k+1] = u_new                                 # The approximation is saved.

    return u_ap                                                             # Return the computed solution.

def Diffusion1D_MOL(x, T, u, nu, RK):
    '''
        Diffusion1D_MOL

        This function solves a 1D Diffusion problem on a regular grid with Dirichlet boundary conditions
        using a Method of Lines for Classical Finite Differences.

        Arguments:
            x           m x 1       Array           Array with the grid generated for the problem.
            T           t x 1       Array           Array with the time grid with t partitions.
            u                       Function        Function for the boundary conditions.
            nu                      Float           Diffusion coefficient for the problem.
            RK                      Integer         Runge-Kutta Order (2, 3)
        
        Returns:
            u_ap        m x t       Array           Array with the computed solution of the method.
    '''

    # Variable initialization
    m    = len(x)                                                           # Size of the mesh in space.
    t    = len(T)                                                           # Size of the mesh in time.
    
    # Solution initialization.
    u_ap = np.zeros([m, t])                                                 # u_ap is initialized with zeros.

    # Initial condition
    for i in range(m):                                                      # For all the grid nodes.
        u_ap[i, 0] = u(x[i], T[0], nu)                                      # The initial condition is assigned.
    
    # Boundary conditions
    for k in range(t):                                                      # For all the time steps.
        u_ap[0,  k] = u(x[0],  T[k], nu)                                    # Boundary condition at x = 0.
        u_ap[-1, k] = u(x[-1], T[k], nu)                                    # Boundary condition at x = 1.
    
    # Runge-Kutta
    if RK == 2:
        u_ap[1:-1, :] = RungeKutta2_1D(x, T, nu, u, u_ap)                   # Runge-Kutta method to obtain the new approximation.
    elif RK == 3:
        u_ap[1:-1, :] = RungeKutta3_1D(x, T, nu, u, u_ap)                   # Runge-Kutta method to obtain the new approximation.
    elif RK == 4:
        u_ap[1:-1, :] = RungeKutta4_1D(x, T, nu, u, u_ap)                   # Runge-Kutta method to obtain the new approximation.

    return u_ap                                                             # Return the computed solution.

def Diffusion2D_MOL(x, y, T, u, nu, RK):
    '''
        Diffusion_2D_MOL

        This function solves a 2D Diffusion problem on a regular grid with Dirichlet boundary conditions
        using a Method of Lines formulation of the Classical Finite Difference centered scheme.

        Arguments:
            x           m x n       Array           Array with the x values of the nodes of the generated grid.
            y           m x n       Array           Array with the y values of the nodes of the generated grid.
            T           t x 1       Array           Array with the time grid with t partitions.
            u                       Function        Function for the boundary conditions.
            nu                      Float           Diffusion coefficient for the problem.
            RK                      Integer         Runge-Kutta Order (2, 3)
        
        Returns:
            u_ap        m x n x t   Array           Array with the computed solution of the method.
    '''

    # Variable initialization
    m, n = x.shape                                                          # Size of the mesh.
    t    = len(T)                                                           # Size of the mesh in time.
    
    # Solution initialization.
    u_ap = np.zeros([m, n, t])                                              # u_ap is initialized with zeros.

    # Initial condition
    for i in range(m):                                                      # For all the grid nodes in x.
        for j in range(n):                                                  # For all the grid nodes in y.
            u_ap[i, j, 0] = u(x[i, j], y[i, j], T[0], nu)                   # The initial condition is assigned.
    
    # Boundary conditions
    for k in range(t):                                                      # For all the time steps.
        for i in range(m):
            u_ap[i, 0,  k] = u(x[i, 0],  y[i, 0],  T[k], nu)                # Boundary condition at y = 0.
            u_ap[i, -1, k] = u(x[i, -1], y[i, -1], T[k], nu)                # Boundary condition at y = 1.
        for j in range(n):
            u_ap[0, j,  k] = u(x[0, j],  y[0, j],  T[k], nu)                # Boundary condition at x = 0.
            u_ap[-1, j, k] = u(x[-1, j], y[-1, j], T[k], nu)                # Boundary condition at x = 1.
    
    # Runge-Kutta
    if RK == 2:
        u_ap[1:-1, 1:-1, :] = RungeKutta2_2D(x, y, T, nu, u, u_ap)          # Runge-Kutta method to obtain the new approximation.
    elif RK == 3:
        u_ap[1:-1, 1:-1, :] = RungeKutta3_2D(x, y, T, nu, u, u_ap)          # Runge-Kutta method to obtain the new approximation.
    elif RK == 4:
        u_ap[1:-1, 1:-1, :] = RungeKutta4_2D(x, y, T, nu, u, u_ap)          # Runge-Kutta method to obtain the new approximation.

    return u_ap                                                             # Return the computed solution.