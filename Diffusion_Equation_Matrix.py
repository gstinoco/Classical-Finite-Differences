'''
Classical Finite Difference Schemes to solve Diffusion Equation.
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

def Diffusion1D(x, T, u, nu):
    '''
        Diffusion1D

        This function solves a 1D Diffusion problem on a regular grid with Dirichlet boundary conditions
        using a Matrix formulation of the Classical Finite Difference Centered Scheme.

        Arguments:
            x           m x 1       Array           Array with the grid generated for the problem.
            T           t x 1       Array           Array with the time grid with t partitions.
            u                       Function        Function for the boundary conditions.
            nu                      Float           Diffusion coefficient for the problem.
        
        Returns:
            u_ap        m x 1       Array           Array with the computed solution of the method.
    '''
    
    m    = len(x)                                                           # Size of the mesh in space.
    t    = len(T)                                                           # Size of the mesh in time.
    dx   = x[1] - x[0]                                                      # dx is defined as the space step-length.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    A    = np.zeros([m, m])                                                 # Initialization of Matrix A.
    r    = nu*dt/(dx**2)                                                    # r has all the coefficients of the method.
    
    # Solution initialization.
    u_ap = np.zeros([m, t])                                                 # u_ap is initialized with zeros.

    # Initial condition
    for i in range(m):                                                      # For all the grid nodes.
        u_ap[i, 0] = u(x[i], T[0], nu)                                      # The initial condition is assigned.
    
    # Boundary conditions
    for k in range(t):                                                      # For all the time steps.
        u_ap[0,  k]  = u(x[0], T[k], nu)                                    # Boundary condition at x = 0.
        u_ap[-1, k] = u(x[-1], T[k], nu)                                    # Boundary condition at x = 1.

    # Finite Differences Matrix
    for i in range(1, m-1):                                                 # Loop through the Matrix.
        A[i, i-1] = r                                                       # Superior diagonal.
        A[i, i]   = -2*r                                                    # Main diagonal.
        A[i, i+1] = r                                                       # Inferior diagonal.
    A[0, 0] = A[-1, -1] = 1                                                 # Boundary Conditions.

    # Finite Differences Approximation
    for k in range(t-1):                                                    # For all the time-steps.
        u_new = u_ap[:, k] + A@u_ap[:, k]                                   # The new approximation is computed.
        u_ap[1:-1, k+1] = u_new[1:-1]                                       # Tha approximation is saved.

    return u_ap                                                             # Return the computed solution.

def Diffusion1D_CN(x, T, u, nu):
    '''
        Diffusion1D_CN

        This function solves a 1D Diffusion problem on a regular grid with Dirichlet boundary conditions
        using a Matrix formulation of the Classical Finite Difference Crank-Nicolson Scheme.

        Arguments:
            x           m x 1       Array           Array with the grid generated for the problem.
            T           t x 1       Array           Array with the time grid with t partitions.
            u                       Function        Function for the boundary conditions.
            nu                      Float           Diffusion coefficient for the problem.
        
        Returns:
            u_ap        m x 1       Array           Array with the computed solution of the method.
    '''
    
    m    = len(x)                                                           # Size of the mesh in space.
    t    = len(T)                                                           # Size of the mesh in time.
    dx   = x[1] - x[0]                                                      # dx is defined as the space step-length.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    u_ap = np.zeros([m, t])                                                 # u_ap is initialized with zeros.
    A    = np.zeros([m, m])                                                 # Initialization of Matrix A.
    B    = np.zeros([m, m])                                                 # Initialization of Matrix B.
    r    = nu*dt/(2*dx**2)                                                  # r has all the coefficients of the method.

    # Initial condition
    for i in range(m):                                                      # For all the grid nodes.
        u_ap[i, 0] = u(x[i], T[0], nu)                                      # The initial condition is assigned.
    
    # Boundary conditions
    for k in range(t):                                                      # For all the time steps.
        u_ap[0,  k] = u(x[0],  T[k], nu)                                    # Boundary condition at x = 0.
        u_ap[-1, k] = u(x[-1], T[k], nu)                                    # Boundary condition at x = 1.

    # Matrices for Crank-Nicolson
    for i in range(1, m-1):                                                 # Loop through the Matrix.
        A[i, i-1] = -r                                                      # Superior diagonal.
        A[i, i]   = 1 + 2*r                                                 # Main diagonal.
        A[i, i+1] = -r                                                      # Inferior diagonal.
    A[0, 0] = A[-1, -1] = 1                                                 # Boundary Conditions.
    
    for i in range(1, m-1):                                                 # Loop through the Matrix.
        B[i, i-1] = r                                                       # Superior diagonal.
        B[i, i]   = 1 - 2*r                                                 # Main diagonal.
        B[i, i+1] = r                                                       # Inferior diagonal.
    B[0, 0] = B[-1, -1] = 1                                                 # Boundary Conditions.
    
    # Time-stepping with Crank-Nicolson
    for k in range(t-1):                                                    # For all the time-steps.
        u_new = np.linalg.solve(A, B@u_ap[:, k])                            # The new approximation is computed.
        u_ap[1:-1, k+1] = u_new[1:-1]                                       # Tha approximation is saved.

    return u_ap                                                             # Return the computed solution.

def Diffusion2D(x, y, T, u, nu):
    '''
        Diffusion2D

        This function solves a 2D Diffusion problem on a regular grid with Dirichlet boundary conditions
        using a Matrix formulation of the Classical Finite Difference centered scheme.

        Arguments:
            x           m x n       Array           Array with the x values of the nodes of the generated grid.
            y           m x n       Array           Array with the y values of the nodes of the generated grid.
            T           t x 1       Array           Array with the time grid with t partitions.
            u                       Function        Function for the boundary conditions.
            nu                      Float           Diffusion coefficient for the problem.
        
        Returns:
            u_ap        m x n x t   Array           Array with the computed solution of the method.
    '''

    m, n = x.shape                                                          # Size of the mesh.
    t    = len(T)                                                           # Size of the mesh in time.
    dx   = x[0, 1] - x[0, 0]                                                # dx is defined as the space step-length in x.
    dy   = y[1, 0] - y[0, 0]                                                # dy is defined as the space step-length in y.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    A    = np.zeros([m*n, m*n])                                             # A is initialized as a (m*n)x(m*n) square matrix.
    r_x  = nu*dt/(dx**2)                                                    # r_x has all the coefficients of the method.
    r_y  = nu*dt/(dy**2)                                                    # r_y has all the coefficients of the method.

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
            u_ap[0,  j, k] = u(x[0, j],  y[0, j],  T[k], nu)                # Boundary condition at x = 0.
            u_ap[-1, j, k] = u(x[-1, j], y[-1, j], T[k], nu)                # Boundary condition at x = 1.
    
    # Finite Differences Matrix
    for i in range(m):                                                      # For all the nodes in one direction.
        for j in range(n):                                                  # For all the nodes in the other direction.
            k = i * m + j                                                   # Linearized Index.
            if i == 0 or i == m-1 or j == 0 or j == m-1:                    # If the node is in the boundary.
                A[k, k] = 1                                                 # A with ones to keep the boundary condition.
            else:                                                           # If the node is an inner node.
                A[k, k]   = -2*r_x - 2*r_y                                  # A matrix value for central node.
                A[k, k-1] = 1*r_y                                           # A matrix value for left node.
                A[k, k+1] = 1*r_y                                           # A matrix value for right node.
                A[k, k-m] = 1*r_x                                           # A matrix value for downer node.
                A[k, k+m] = 1*r_x                                           # A matrix value for upper node.
    
    # Finite Differences Approximation
    for k in range(t-1):                                                    # For all the time-steps.
        u_flat = u_ap[:, :, k].reshape(m*n)                                 # Transform the solution into a vector.
        u_new  = u_ap[:, :, k] + (A@u_flat).reshape(m, n)                   # The new approximation is computed and reshaped.
        u_ap[1:-1, 1:-1, k+1] = u_new[1:-1, 1:-1]                           # Tha approximation is saved.
            
    return u_ap                                                             # Return the computed solution.

def Diffusion2D_CN(x, y, T, u, nu):
    '''
        Diffusion2D_CN

        This function solves a 2D Diffusion problem on a regular grid with Dirichlet boundary conditions
        using a Matrix formulation of the Classical Finite Difference Crank-Nicolson scheme.

        Arguments:
            x           m x n       Array           Array with the x values of the nodes of the generated grid.
            y           m x n       Array           Array with the y values of the nodes of the generated grid.
            T           t x 1       Array           Array with the time grid with t partitions.
            u                       Function        Function for the boundary conditions.
            nu                      Float           Diffusion coefficient for the problem.
        
        Returns:
            u_ap        m x n x t   Array           Array with the computed solution of the method.
    '''

    m, n = x.shape                                                          # Size of the mesh.
    t    = len(T)                                                           # Size of the mesh in time.
    dx   = x[0, 1] - x[0, 0]                                                # dx is defined as the space step-length in x.
    dy   = y[1, 0] - y[0, 0]                                                # dy is defined as the space step-length in y.
    dt   = T[1] - T[0]                                                      # dt is defined as the time step-length.
    u_ap = np.zeros([m, n , t])                                             # u_ap is initialized with zeros.
    A    = np.zeros([m*n, m*n])                                             # Matrix A initialized.
    B    = np.zeros([m*n, m*n])                                             # Matrix B initialized.
    r_x  = nu*dt/(dx**2)                                                    # r has all the coefficients of the method.
    r_y  = nu*dt/(dy**2)                                                    # r has all the coefficients of the method.
    
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
            u_ap[0,  j, k] = u(x[0,  j], y[0,  j], T[k], nu)                # Boundary condition at x = 0.
            u_ap[-1, j, k] = u(x[-1, j], y[-1, j], T[k], nu)                # Boundary condition at x = 1.

    # Matrices for Crank-Nicolson
    for i in range(m):                                                      # For all the nodes in one direction.
        for j in range(n):                                                  # For all the nodes in the other direction.
            k = i * m + j                                                   # Linearized Index.
            if i == 0 or i == m-1 or j == 0 or j == n-1:                    # If the node is in the boundary.
                A[k, k] = 1                                                 # A with ones to keep the boundary condition.
                B[k, k] = 1                                                 # B with ones to keep the boundary condition.
            else:                                                           # If the node is an inner node.
                A[k, k]   = 1 + r_x + r_y                                   # A matrix value for central node.
                A[k, k-1] = A[k, k+1] = -r_x/2                              # A matrix value for left and right nodes.
                A[k, k-m] = A[k, k+m] = -r_y/2                              # A matrix value for downer and upper nodes.
                
                B[k, k]   = 1 - r_x - r_y                                   # B matrix value for central node.
                B[k, k-1] = B[k, k+1] = r_x/2                               # B matrix value for left and right nodes.
                B[k, k-m] = B[k, k+m] = r_y/2                               # B matrix value for downer and upper nodes.

    # Finite Differences Approximation
    for k in range(t-1):                                                    # For all the time-steps.
        u_flat = u_ap[:, :, k].reshape(m*n)                                 # Transform the solution into a vector.
        u_new  = np.linalg.solve(A, B@u_flat)                               # The new approximation is computed and reshaped.
        u_ap[1:-1, 1:-1, k+1] = u_new.reshape(m, n)[1:-1, 1:-1]             # Tha approximation is saved.

    return u_ap                                                             # Return the computed solution.