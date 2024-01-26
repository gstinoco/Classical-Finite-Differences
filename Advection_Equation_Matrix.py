'''
Classical Finite Difference Schemes to solve Advection Equation.
The codes presented in this file correspond to a Matrix Formulation of the problem.

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
    A    = np.zeros([m, m])                                                 # Initialization of Matrix A.
    r    = a*dt/(2*dx)                                                      # r is defined as the CFL coefficient.

    # Solution initialization.
    u_ap = np.zeros([m, t])                                                 # u_ap is initialized with zeros.
    
    # Initial Conditions
    for i in range(m):                                                      # For all the grid nodes.
        u_ap[i, 0]  = u(x[i], T[0], a)                                      # The initial condition is assigned.
    
    # Boundary Conditions
    for k in range(t):                                                      # For all the time steps.
        u_ap[0,  k] = u(x[0],  T[k], a)                                     # Boundary condition at x = 0.
        u_ap[-1, k] = u(x[-1], T[k], a)                                     # Boundary condition at x = m*dx.
    
    # Finite Differences Matrix
    for i in range(1, m-1):                                                 # Loop through the Matrix.
        A[i, i+1] = -r                                                      # Superior diagonal.
        A[i, i-1] = r                                                       # Inferior diagonal.
    A[0, 0] = A[-1, -1] = 1                                                 # Boundary Conditions.

    # Finite Differences Approximation
    for k in range(t-1):                                                    # For all the time-steps.
        u_new = u_ap[:, k] + A@u_ap[:, k]                                   # The new approximation is computed.
        u_ap[1:-1, k+1] = u_new[1:-1]                                       # Tha approximation is saved.

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
    A    = np.zeros([m, m])                                                 # Initialization of Matrix A.
    r    = a*dt/dx                                                          # r is defined as the CFL coefficient.

    # Solution initialization.
    u_ap = np.zeros([m, t])                                                 # u_ap is initialized with zeros.
    
    # Initial Conditions
    for i in range(m):                                                      # For all the grid nodes.
        u_ap[i, 0]  = u(x[i], T[0], a)                                      # The initial condition is assigned.
    
    # Boundary Conditions
    for k in range(t):                                                      # For all the time steps.
        u_ap[0,  k] = u(x[0],  T[k], a)                                     # Boundary condition at x = 0.
    
    # Finite Differences Matrix
    for i in range(1, m):                                                   # Loop through the Matrix.
        A[i, i]   = -r                                                      # Main diagonal.
        A[i, i-1] = r                                                       # Inferior diagonal.
    A[0, 0] = 1                                                             # Boundary Conditions.

    # Finite Differences Approximation
    for k in range(t-1):                                                    # For all the time-steps.
        u_new = u_ap[:, k] + A@u_ap[:, k]                                   # The new approximation is computed.
        u_ap[1:, k+1] = u_new[1:]                                           # Tha approximation is saved.

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
    A    = np.zeros([m, m])                                                 # Initialization of Matrix A.
    r    = a*dt/dx                                                          # r is defined as the CFL coefficient.

    # Solution initialization.
    u_ap = np.zeros([m, t])                                                 # u_ap is initialized with zeros.
    
    # Initial Conditions
    for i in range(m):                                                      # For all the grid nodes.
        u_ap[i, 0]  = u(x[i], T[0], a)                                      # The initial condition is assigned.
    
    # Boundary Conditions
    for k in range(t):                                                      # For all the time steps.
        u_ap[-1, k] = u(x[-1], T[k], a)                                     # Boundary condition at x = m*dx.
    
    # Finite Differences Matrix
    for i in range(m-1):                                                    # Loop through the Matrix.
        A[i, i+1] = -r                                                      # Superior diagonal.
        A[i, i]   = r                                                       # Main diagonal.
    A[-1, -1] = 1                                                           # Boundary Conditions.

    # Finite Differences Approximation
    for k in range(t-1):                                                    # For all the time-steps.
        u_new = u_ap[:, k] + A@u_ap[:, k]                                   # The new approximation is computed.
        u_ap[:-1, k+1] = u_new[:-1]                                         # Tha approximation is saved.

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
    A    = np.zeros([m, m])                                                 # Initialization of Matrix A.
    r    = a*dt/dx                                                          # r is defined as the CFL coefficient.

    # Solution initialization.
    u_ap = np.zeros([m, t])                                                 # u_ap is initialized with zeros.
    
    # Initial Conditions
    for i in range(m):                                                      # For all the grid nodes.
        u_ap[i, 0]  = u(x[i], T[0], a)                                      # The initial condition is assigned.
    
    # Boundary Conditions
    for k in range(t):                                                      # For all the time steps.
        u_ap[0,  k] = u(x[0],  T[k], a)                                     # Boundary condition at x = 0.
        u_ap[-1, k] = u(x[-1], T[k], a)                                     # Boundary condition at x = m*dx.
    
    # Finite Differences Matrix (First Time Step)
    for i in range(1, m-1):                                                 # Loop through the Matrix.
        A[i, i+1] = -r/2                                                    # Superior diagonal.
        A[i, i-1] = r/2                                                     # Inferior diagonal.
    A[0, 0] = A[-1, -1] = 1                                                 # Boundary Conditions.

    # Finite Differences Approximation (First Time Step)
    u_new = u_ap[:, 0] + A@u_ap[:, 0]                                       # The new approximation is computed.
    u_ap[1:-1, 1] = u_new[1:-1]                                             # Tha approximation is saved.

    # Finite Differences Matrix (All the other Time Steps)
    for i in range(1, m-1):                                                 # Loop through the Matrix.
        A[i, i+1] = -r                                                      # Superior diagonal.
        A[i, i-1] = r                                                       # Inferior diagonal.
    A[0, 0] = A[-1, -1] = 1                                                 # Boundary Conditions.

    # Finite Differences Approximation (All the other Time Steps)
    for k in range(1, t-1):                                                 # For all the time-steps.
        u_new = u_ap[:, k-1] + A@u_ap[:, k]                                 # The new approximation is computed.
        u_ap[1:-1, k+1] = u_new[1:-1]                                       # Tha approximation is saved.

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
    A    = np.zeros([m, m])                                                 # Initialization of Matrix A.
    r    = a*dt/(2*dx)                                                      # r is defined as the CFL coefficient.

    # Solution initialization.
    u_ap = np.zeros([m, t])                                                 # u_ap is initialized with zeros.
    
    # Initial Conditions
    for i in range(m):                                                      # For all the grid nodes.
        u_ap[i, 0]  = u(x[i], T[0], a)                                      # The initial condition is assigned.
    
    # Boundary Conditions
    for k in range(t):                                                      # For all the time steps.
        u_ap[0,  k] = u(x[0],  T[k], a)                                     # Boundary condition at x = 0.
        u_ap[-1, k] = u(x[-1], T[k], a)                                     # Boundary condition at x = m*dx.
    
    # Finite Differences Matrix
    for i in range(1, m-1):                                                 # Loop through the Matrix.
        A[i, i+1] = (1/2) - r                                               # Superior diagonal.
        A[i, i-1] = (1/2) + r                                               # Inferior diagonal.
    A[0, 0] = A[-1, -1] = 1                                                 # Boundary Conditions.

    # Finite Differences Approximation
    for k in range(t-1):                                                    # For all the time-steps.
        u_new = A@u_ap[:, k]                                                # The new approximation is computed.
        u_ap[1:-1, k+1] = u_new[1:-1]                                       # Tha approximation is saved.

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
    A    = np.zeros([m, m])                                                 # Initialization of Matrix A.
    r    = a*dt/(2*dx)                                                      # r is defined as the CFL coefficient.

    # Solution initialization.
    u_ap = np.zeros([m, t])                                                 # u_ap is initialized with zeros.
    
    # Initial Conditions
    for i in range(m):                                                      # For all the grid nodes.
        u_ap[i, 0]  = u(x[i], T[0], a)                                      # The initial condition is assigned.
    
    # Boundary Conditions
    for k in range(t):                                                      # For all the time steps.
        u_ap[0,  k] = u(x[0],  T[k], a)                                     # Boundary condition at x = 0.
        u_ap[-1, k] = u(x[-1], T[k], a)                                     # Boundary condition at x = m*dx.
    
    # Finite Differences Matrix
    for i in range(1, m-1):                                                 # Loop through the Matrix.
        A[i, i+1] = (1/2) - r                                               # Superior diagonal.
        A[i, i]   = -1                                                      # Main diagonal.
        A[i, i-1] = (1/2) + r                                               # Inferior diagonal.
    A[0, 0] = A[-1, -1] = 1                                                 # Boundary Conditions.

    # Finite Differences Approximation
    for k in range(t-1):                                                    # For all the time-steps.
        u_new = u_ap[:, k] + A@u_ap[:, k]                                   # The new approximation is computed.
        u_ap[1:-1, k+1] = u_new[1:-1]                                       # Tha approximation is saved.

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
    A    = np.zeros([m, m])                                                 # Initialization of Matrix A.
    r    = a*dt/dx                                                          # r is defined as the CFL coefficient.

    # Solution initialization.
    u_ap = np.zeros([m, t])                                                 # u_ap is initialized with zeros.
    
    # Initial Conditions
    for i in range(m):                                                      # For all the grid nodes.
        u_ap[i, 0]  = u(x[i], T[0], a)                                      # The initial condition is assigned.
    
    # Boundary Conditions
    for k in range(t):                                                      # For all the time steps.
        u_ap[0,  k] = u(x[0],  T[k], a)                                     # Boundary condition at x = 0.
        u_ap[-1, k] = u(x[-1], T[k], a)                                     # Boundary condition at x = m*dx.
    
    # Finite Differences Matrix
    for i in range(1, m-1):                                                 # Loop through the Matrix.
        A[i, i+1] = -(r/2) + (r**2)/2                                       # Superior diagonal.
        A[i, i]   = -r**2                                                   # Main diagonal.
        A[i, i-1] = (r/2) + (r**2)/2                                        # Inferior diagonal.
    A[0, 0] = A[-1, -1] = 1                                                 # Boundary Conditions.

    # Finite Differences Approximation
    for k in range(t-1):                                                    # For all the time-steps.
        u_new = u_ap[:, k] + A@u_ap[:, k]                                   # The new approximation is computed.
        u_ap[1:-1, k+1] = u_new[1:-1]                                       # Tha approximation is saved.

    return u_ap                                                             # Return the computed solution.

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
    A    = np.zeros([m, m])                                                 # Initialization of Matrix A.
    r    = a*dt/(2*dx)                                                      # r is defined as the CFL coefficient.
    s    = (r**2)*2                                                         # s is (r^2)*2 to make the code easier to read.

    # Solution initialization.
    u_ap = np.zeros([m, t])                                                 # u_ap is initialized with zeros.
    
    # Initial Conditions
    for i in range(m):                                                      # For all the grid nodes.
        u_ap[i, 0]  = u(x[i], T[0], a)                                      # The initial condition is assigned.
    
    # Boundary Conditions
    for k in range(t):                                                      # For all the time steps.
        if a > 0:                                                           # If the advective velocity is positive.
            u_ap[0, k] = u(x[0], T[k], a)                                   # Boundary condition at x = 0.
            u_ap[1, k] = u(x[1], T[k], a)                                   # Boundary condition at x = dx.
        else:                                                               # If the advective velocity is negative.
            u_ap[-2, k] = u(x[-2], T[k], a)                                 # Boundary condition at x = m*dx - dx.
            u_ap[-1, k] = u(x[-1], T[k], a)                                 # Boundary condition at x = m*dx.
    
    # Finite Differences Matrix
    if a > 0:                                                               # If the advective velocity is positive.
        for i in range(2, m):                                               # Loop through the Matrix.
            A[i, i]   = -3*r + s                                            # Main diagonal
            A[i, i-1] = 4*r - 2*s                                           # Inferior diagonal.
            A[i, i-2] = -r + s                                              # Second Inferior diagonal.
            A[0, 0] = A[1, 1] = 1                                           # Boundary Conditions.
        # Finite Differences Approximation
        for k in range(t-1):                                                # For all the time-steps.
            u_new = u_ap[:, k] + A@u_ap[:, k]                               # The new approximation is computed.
            u_ap[2:, k+1] = u_new[2:]                                       # Tha approximation is saved.

    else:                                                                   # If the advective velocity is negative.
        for i in range(m-2):                                                # Loop through the Matrix.
            A[i, i+2] = r + s                                               # Second Upper diagonal
            A[i, i+1] = -4*r - 2*s                                          # Upper diagonal
            A[i, i]   = 3*r + s                                             # Main diagonal
            A[-2, -2] = A[-1, -1] = 1                                       # Boundary Conditions.
        # Finite Differences Approximation
        for k in range(t-1):                                                # For all the time-steps.
            u_new = u_ap[:, k] + A@u_ap[:, k]                               # The new approximation is computed.
            u_ap[:-2, k+1] = u_new[:-2]                                     # Tha approximation is saved.

    return u_ap                                                             # Return the computed solution.

def Advection_2D_FTCS(x, y, T, u, a, b):
    '''
        Advection_2D_FTCS

        This function solves a 2D Advection problem on a regular grid with Dirichlet boundary conditions
        using a Classical Finite Difference Centered Scheme.

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
    A    = np.zeros([m*n, m*n])                                             # A is initialized as a (m*n)x(m*n) square matrix.
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
    
    # Finite Differences Matrix
    for i in range(m):                                                      # For all the nodes in one direction.
        for j in range(n):                                                  # For all the nodes in the other direction.
            k = i * m + j                                                   # Linearized Index.
            if i == 0 or i == m - 1 or j == 0 or j == m - 1:                # If the node is in the boundary.
                A[k, k] = 1                                                 # A with ones to keep the boundary condition.
            else:                                                           # If the node is an inner node.
                A[k, k-1] = r_y                                             # A matrix value for downer node.
                A[k, k+1] = -r_y                                            # A matrix value for upper node.
                A[k, k-m] = r_x                                             # A matrix value for left node.
                A[k, k+m] = -r_x                                            # A matrix value for right node.
    
    # Finite Differences Approximation
    for k in range(t-1):                                                    # For all the time-steps.
        u_flat = u_ap[:, :, k].reshape(m*m)                                 # Transform the solution into a vector.
        u_new  = u_ap[:, :, k] + (A@u_flat).reshape(m, m)                   # The new approximation is computed and reshaped.
        u_ap[1:-1, 1:-1, k+1] = u_new[1:-1, 1:-1]                           # Tha approximation is saved.
            
    return u_ap                                                             # Return the computed solution.

def Advection_2D_FTBS(x, y, T, u, a, b):
    '''
        Advection_2D_FTBS

        This function solves a 2D Advection problem on a regular grid with Dirichlet boundary conditions
        using a Classical Finite Difference Upwind Scheme.

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
    A    = np.zeros([m*n, m*n])                                             # A is initialized as a (m*n)x(m*n) square matrix.
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
            u_ap[i, 0, k]  = u(x[i, 0], y[i, 0], T[k], a, b)                # Boundary condition at y = 0.
        for j in range(n):
            u_ap[0, j, k]  = u(x[0, j], y[0, j], T[k], a, b)                # Boundary condition at x = 0.
    
    # Finite Differences Matrix
    for i in range(1, m):                                                   # For all the nodes in one direction.
        for j in range(1, n):                                               # For all the nodes in the other direction.
            k = i * m + j                                                   # Linearized Index.
            if i == 0 or j == 0:                                            # If the node is in the boundary.
                A[k, k] = 1                                                 # A with ones to keep the boundary condition.
            else:                                                           # If the node is an inner node.
                A[k, k]   = -r_x - r_y                                      # A matrix value for central node.
                A[k, k-1] = r_y                                             # A matrix value for downer node.
                A[k, k-m] = r_x                                             # A matrix value for left node.
    
    # Finite Differences Approximation
    for k in range(t-1):                                                    # For all the time-steps.
        u_flat = u_ap[:, :, k].reshape(m*m)                                 # Transform the solution into a vector.
        u_new  = u_ap[:, :, k] + (A@u_flat).reshape(m, m)                   # The new approximation is computed and reshaped.
        u_ap[1:, 1:, k+1] = u_new[1:, 1:]                                   # Tha approximation is saved.
            
    return u_ap                                                             # Return the computed solution.

def Advection_2D_FTFS(x, y, T, u, a, b):
    '''
        Advection_2D_FTFS

        This function solves a 2D Advection problem on a regular grid with Dirichlet boundary conditions
        using a Classical Finite Difference Forward Time Forward Space Scheme.

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
    A    = np.zeros([m*n, m*n])                                             # A is initialized as a (m*n)x(m*n) square matrix.
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
            u_ap[0, j, k]  = u(x[0,  j], y[0,  j], T[k], a, b)              # Boundary condition at x = 0.
            u_ap[-1, j, k] = u(x[-1, j], y[-1, j], T[k], a, b)              # Boundary condition at x = 1.
    
    # Finite Differences Matrix
    for i in range(m-1):                                                    # For all the nodes in one direction.
        for j in range(n-1):                                                # For all the nodes in the other direction.
            k = i * m + j                                                   # Linearized Index.
            if i == 0 or i == m - 1 or j == 0 or j == m - 1:                # If the node is in the boundary.
                A[k, k] = 1                                                 # A with ones to keep the boundary condition.
            else:                                                           # If the node is an inner node.
                A[k, k]   = r_x + r_y                                       # A matrix value for central node.
                A[k, k+1] = -r_y                                            # A matrix value for upper node.
                A[k, k+m] = -r_x                                            # A matrix value for right node.
    
    # Finite Differences Approximation
    for k in range(t-1):                                                    # For all the time-steps.
        u_flat = u_ap[:, :, k].reshape(m*m)                                 # Transform the solution into a vector.
        u_new  = u_ap[:, :, k] + (A@u_flat).reshape(m, m)                   # The new approximation is computed and reshaped.
        u_ap[:-1, :-1, k+1] = u_new[:-1, :-1]                               # Tha approximation is saved.
            
    return u_ap                                                             # Return the computed solution.

def Advection_2D_Lax_Wendroff(x, y, T, u, a, b):
    '''
        Advection_2D_Lax_Wendroff

        This function solves a 2D Advection problem on a regular grid with Dirichlet boundary conditions
        using a Lax-Wendroff scheme.

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
    A    = np.zeros([m*n, m*n])                                             # A is initialized as a (m*n)x(m*n) square matrix.
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
            u_ap[0, j, k]  = u(x[0,  j], y[0,  j], T[k], a, b)              # Boundary condition at x = 0.
            u_ap[-1, j, k] = u(x[-1, j], y[-1, j], T[k], a, b)              # Boundary condition at x = 1.
    
    # Finite Differences Matrix
    for i in range(m):                                                      # For all the nodes in one direction.
        for j in range(n):                                                  # For all the nodes in the other direction.
            k = i * m + j                                                   # Linearized Index.
            if i == 0 or i == m - 1 or j == 0 or j == m - 1:                # If the node is in the boundary.
                A[k, k] = 1                                                 # A with ones to keep the boundary condition.
            else:                                                           # If the node is an inner node.
                A[k, k]   = -r_x**2 - r_y**2                                # A matrix value for central node.
                A[k, k-1] = (r_y/2) + (r_y**2)/2                            # A matrix value for downer node.
                A[k, k+1] = -(r_y/2) + (r_y**2)/2                           # A matrix value for upper node.
                A[k, k-m] = (r_x/2) + (r_x**2)/2                            # A matrix value for left node.
                A[k, k+m] = -(r_x/2) + (r_x**2)/2                           # A matrix value for right node.
    
    # Finite Differences Approximation
    for k in range(t-1):                                                    # For all the time-steps.
        u_flat = u_ap[:, :, k].reshape(m*m)                                 # Transform the solution into a vector.
        u_new  = u_ap[:, :, k] + (A@u_flat).reshape(m, m)                   # The new approximation is computed and reshaped.
        u_ap[1:-1, 1:-1, k+1] = u_new[1:-1, 1:-1]                           # Tha approximation is saved.
            
    return u_ap                                                             # Return the computed solution.

def Advection_2D_Beam_Warming(x, y, T, u, a, b):
    '''
        Advection_2D_Beam-Warming

        This function solves a 2D Advection problem on a regular grid with Dirichlet boundary conditions
        using a Bean-Warming scheme.

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
    A    = np.zeros([m*n, m*n])                                             # A is initialized as a (m*n)x(m*n) square matrix.
    r_x  = a*dt/(2*dx)                                                      # r_x is defined as the CFL coefficient.
    r_y  = b*dt/(2*dy)                                                      # r_y is defined as the CFL coefficient.
    s_x  = (r_x**2)/2                                                       # s_x is (r_x^2)*2 to make the code easier to read.
    s_y  = (r_y**2)/2                                                       # s_y is (r_y^2)*2 to make the code easier to read.

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
    
    # Finite Differences Matrix
    if a > 0:
        for i in range(2, m):                                               # For all the nodes in one direction.
            if b > 0:
                for j in range(2, n):                                       # For all the nodes in the other direction.
                    k = i * m + j                                           # Linearized Index.
                    if i == 0 or j == 0:                                    # If the node is in the boundary.
                        A[k, k] = 1                                         # A with ones to keep the boundary condition.
                    else:                                                   # If the node is an inner node.
                        A[k, k]     = -3*r_x - 3*r_y + s_x + s_y            # A matrix value for central node.
                        A[k, k-1]   = 4*r_y - 2*s_y                         # A matrix value for downer node.
                        A[k, k-2]   = -r_y + s_y                            # A matrix value for the second downer node.
                        A[k, k-m]   = 4*r_x - 2*s_x                         # A matrix value for left node.
                        A[k, k-m-m] = -r_x + s_y                            # A matrix value for right node.
            else:
                for j in range(n-2):                                        # For all the nodes in the other direction.
                    k = i * m + j                                           # Linearized Index.
                    if i == 0 or j == m - 1:                                # If the node is in the boundary.
                        A[k, k] = 1                                         # A with ones to keep the boundary condition.
                    else:                                                   # If the node is an inner node.
                        A[k, k]     = -3*r_x + 3*r_y + s_x + s_y            # A matrix value for central node.
                        A[k, k+1]   = -4*r_y - 2*s_y                        # A matrix value for upper node.
                        A[k, k+2]   = r_y + s_y                             # A matrix value for the second upper node.
                        A[k, k-m]   = 4*r_x - 2*s_x                         # A matrix value for left node.
                        A[k, k-m-m] = -r_x + s_y                            # A matrix value for right node.
    else:
        for i in range(m-2):                                                # For all the nodes in one direction.
            if b > 0:
                for j in range(2, n):                                       # For all the nodes in the other direction.
                    k = i * m + j                                           # Linearized Index.
                    if i == n - 1 or j == 0:                                # If the node is in the boundary.
                        A[k, k] = 1                                         # A with ones to keep the boundary condition.
                    else:                                                   # If the node is an inner node.
                        A[k, k]     = 3*r_x - 3*r_y + s_x + s_y             # A matrix value for central node.
                        A[k, k-1]   = 4*r_y - 2*s_y                         # A matrix value for downer node.
                        A[k, k-2]   = -r_y + s_y                            # A matrix value for the second downer node.
                        A[k, k+m]   = -4*r_x - 2*s_x                        # A matrix value for left node.
                        A[k, k+m+m] = r_x + s_y                             # A matrix value for right node.
            else:
                for j in range(n-2):                                        # For all the nodes in the other direction.
                    k = i * m + j                                           # Linearized Index.
                    if i == n - 1 or j == m - 1:                            # If the node is in the boundary.
                        A[k, k] = 1                                         # A with ones to keep the boundary condition.
                    else:                                                   # If the node is an inner node.
                        A[k, k]     = 3*r_x + 3*r_y + s_x + s_y             # A matrix value for central node.
                        A[k, k+1]   = -4*r_y - 2*s_y                        # A matrix value for upper node.
                        A[k, k+2]   = r_y + s_y                             # A matrix value for the second upper node.
                        A[k, k+m]   = -4*r_x - 2*s_x                        # A matrix value for left node.
                        A[k, k+m+m] = r_x + s_y                             # A matrix value for right node.
    
    # Finite Differences Approximation
    for k in range(t-1):                                                    # For all the time-steps.
        if a > 0:
            if b > 0:
                u_flat = u_ap[:, :, k].reshape(m*m)                         # Transform the solution into a vector.
                u_new  = u_ap[:, :, k] + (A@u_flat).reshape(m, m)           # The new approximation is computed and reshaped.
                u_ap[2:, 2:, k+1] = u_new[2:, 2:]                           # Tha approximation is saved.
            else:
                u_flat = u_ap[:, :, k].reshape(m*m)                         # Transform the solution into a vector.
                u_new  = u_ap[:, :, k] + (A@u_flat).reshape(m, m)           # The new approximation is computed and reshaped.
                u_ap[2:, :-2, k+1] = u_new[2:, :-2]                         # Tha approximation is saved.
        else:
            if b > 0:
                u_flat = u_ap[:, :, k].reshape(m*m)                         # Transform the solution into a vector.
                u_new  = u_ap[:, :, k] + (A@u_flat).reshape(m, m)           # The new approximation is computed and reshaped.
                u_ap[:-2, 2:, k+1] = u_new[:-2, 2:]                           # Tha approximation is saved.
            else:
                u_flat = u_ap[:, :, k].reshape(m*m)                         # Transform the solution into a vector.
                u_new  = u_ap[:, :, k] + (A@u_flat).reshape(m, m)           # The new approximation is computed and reshaped.
                u_ap[:-2, :-2, k+1] = u_new[:-2, :-2]                       # Tha approximation is saved.
    return u_ap                                                             # Return the computed solution.