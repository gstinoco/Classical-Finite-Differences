'''
Classical Finite Difference Schemes to solve Poisson Equation.
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

def Poisson1D(x, f, u):
    '''
        Poisson1D

        This code solves the 1D Poisson problem on a regular grid with Dirichlet boundary conditions
        using a Matrix formulation of the Classical Finite Difference centered scheme.

        Arguments:
            x                       Array           Mesh of the region.
            f                       Function        Function with the sources and sinks.
            u                       Function        Function for the boundary conditions.
        
        Returns:
            u_ap        m x 1       Array           Array with the computed solution of the method.
    '''

    # Variable Initialization
    m = len(x)                                                              # Size of the mesh.
    h = x[1] - x[0]                                                         # h definition as dx.
    A = np.zeros([m, m])                                                    # Initialization of Matrix A.
    F = np.zeros(m)                                                         # Initialization of Vector F.

    # Finite Differences Matrix
    for i in range(1, m-1):                                                 # Loop through the Matrix.
        A[i, i-1] = 1                                                       # Superior diagonal.
        A[i, i]   = -2                                                      # Main diagonal.
        A[i, i+1] = 1                                                       # Inferior diagonal.
        F[i]      = f(x[i])                                                 # Components of the RHS vector.
    
    # Dirichlet conditions
    A[0,   0]     = h**2                                                    # Complete the Matrix.
    A[-1, -1]     = h**2                                                    # Complete the Matrix.
    A             = A/h**2                                                  # Complete the Matrix.

    # Right Hand Size (RHS)
    F[0]          = u(x[0])                                                 # Boundary condition on th RHS.
    F[-1]         = u(x[-1])                                                # Boundary condition on the RHS.

    # Problem Solving
    u_ap          = np.linalg.solve(A, F)                                   # Solve the algebraic problem.

    return u_ap                                                             # Return the computed solution.

def Poisson1D_Neumann_1(x, f, sig, beta):
    '''
        Poisson1D_Neumann_1

        This code solves the 1D Poisson problem on a regular grid with Neumann and Dirichlet boundary conditions
        using a Matrix formulation of the Finite Difference centered scheme.

        The Neumann boundary condition is applied with a two-point-backward finite difference scheme.

        Arguments:
            x                       Array           Mesh of the region.
            f                       Function        Function with the sources and sinks.
            sig                     Real            Value of the derivative on the Neumann boundary condition.
            beta                    Real            Value of the function on the Dirichlet boundary condition.
        
        Returns:
            u           m x 1       Array           Array with the computed solution of the method.
    '''

    # Variable Initialization
    m = len(x)                                                              # Size of the mesh.
    h = x[1] - x[0]                                                         # h definition as dx.
    A = np.zeros([m, m])                                                    # Initialization of Matrix A.
    F = np.zeros(m)                                                         # Initialization of Vector F.

    # Finite Differences Matrix
    for i in range(1, m-1):                                                 # Loop through the Matrix.
        A[i, i-1] = 1                                                       # Superior diagonal.
        A[i, i]   = -2                                                      # Main diagonal.
        A[i, i+1] = 1                                                       # Inferior diagonal.
        F[i]      = f(x[i])                                                 # Components of the RHS vector.

    # Neumann conditions
    A[0,   0]     = -h                                                      # Complete the Matrix.
    A[0,   1]     = h                                                       # Complete the Matrix.
    A[-1, -1]     = h**2                                                    # Complete the Matrix.
    A             = A/h**2                                                  # Complete the Matrix.

    # Right Hand Size (RHS)
    F[0]          = sig                                                     # Boundary condition on the RHS.
    F[-1]         = beta                                                    # Boundary condition on the RHS.
    
    # Problem Solving
    u_ap          = np.linalg.solve(A, F)                                   # Solve the algebraic problem.

    return u_ap                                                             # Return the computed solution.

def Poisson1D_Neumann_2(x, f, sig, beta):
    '''
        Poisson1D_Neumann_2

        This code solves the 1D Poisson problem on a regular grid with Neumann and Dirichlet boundary conditions
        using a Matrix formulation of the Finite Difference centered scheme.

        The Neumann boundary condition is applied with a two-point-centered finite difference scheme.

        Arguments:
            x                       Array           Mesh of the region.
            f                       Function        Function with the sources and sinks.
            sig                     Real            Value of the derivative on the Neumann boundary condition.
            beta                    Real            Value of the function on the Dirichlet boundary condition.
        
        Returns:
            u           m x 1       Array           Array with the computed solution of the method.
    '''

    # Variable Initialization
    m = len(x)                                                              # Size of the mesh.
    h = x[1] - x[0]                                                         # h definition as dx.
    A = np.zeros([m, m])                                                    # Initialization of Matrix A.
    F = np.zeros(m)                                                         # Initialization of Vector F.

    # Finite Differences Matrix
    for i in range(1, m-1):                                                 # Loop through the Matrix.
        A[i, i-1] = 1                                                       # Superior diagonal.
        A[i, i]   = -2                                                      # Main diagonal.
        A[i, i+1] = 1                                                       # Inferior diagonal.
        F[i]      = f(x[i])                                                 # Components of the RHS vector.

    # Neumann conditions
    A[0,   0]    = -h                                                       # Complete the Matrix.
    A[0,   1]    = h                                                        # Complete the Matrix.
    A[-1, -2]    = 0                                                        # Complete the Matrix.
    A[-1, -1]    = h**2                                                     # Complete the Matrix.
    A            = A/h**2                                                   # Complete the Matrix.
    
    # Right Hand Size (RHS)
    F[0]         = sig+((h/2)*f(x[0]))                                      # Boundary condition on the RHS.
    F[-1]        = beta                                                     # Boundary condition on the RHS.
    
    # Problem Solving
    u_ap         = np.linalg.solve(A, F)                                    # Solve the algebraic problem.

    return u_ap                                                             # Return the computed solution.

def Poisson1D_Neumann_3(x, f, sig, beta):
    '''
        Poisson1D_Neumann_3

        This code solves the 1D Poisson problem on a regular grid with Neumann and Dirichlet boundary conditions
        using a Matrix formulation of the Finite Difference centered scheme.

        The Neumann boundary condition is applied with a three-point-forward finite difference scheme.

        Arguments:
            x                       Array           Mesh of the region.
            f                       Function        Function with the sources and sinks.
            sig                     Real            Value of the derivative on the Neumann boundary condition.
            beta                    Real            Value of the function on the Dirichlet boundary condition.
        
        Returns:
            u           m x 1       Array           Array with the computed solution of the method.
    '''

    # Variable Initialization
    m = len(x)                                                              # Size of the mesh.
    h = x[1] - x[0]                                                         # h definition as dx.
    A = np.zeros([m, m])                                                    # Initialization of Matrix A.
    F = np.zeros(m)                                                         # Initialization of Vector F.

    # Finite Differences Matrix
    for i in range(1, m-1):                                                 # Loop through the Matrix.
        A[i, i-1] = 1                                                       # Superior diagonal.
        A[i, i]   = -2                                                      # Main diagonal.
        A[i, i+1] = 1                                                       # Inferior diagonal.
        F[i]      = f(x[i])                                                 # Components of the RHS vector.

    # Neumann conditions
    A[0,   0]     = (3/2)*h                                                 # Complete the Matrix.
    A[0,   1]     = -2*h                                                    # Complete the Matrix.
    A[0,   2]     = (1/2)*h                                                 # Complete the Matrix.
    A[-1, -2]     = 0                                                       # Complete the Matrix.
    A[-1, -1]     = h**2                                                    # Complete the Matrix.
    A             = A/h**2                                                  # Complete the Matrix.
    
    # Right Hand Size (RHS)
    F[0]          = sig                                                     # Boundary condition on the RHS.
    F[-1]         = beta                                                    # Boundary condition on the RHS.
    
    u_ap          = np.linalg.solve(A, F)                                   # Solve the algebraic problem.

    return u_ap                                                             # Return the computed solution.

def Poisson2D(x, y, f, u):
    '''
        Poisson2D

        This code solves the 2D Poisson problem on a regular grid with Dirichlet boundary conditions
        using a Matrix formulation of the Finite Difference centered scheme.

        Arguments:
            x           m x m       Array           Array with the x values of the nodes of the grid.
            y           m x m       Array           Array with the y values of the nodes of the grid.
            f                       Function        Function with the sources and sinks.
            u                       Function        Function for the boundary conditions.
        
        Returns:
            u_ap        m x m       Array           Array with the computed solution of the method.
    '''

    # Variable Initialization
    m   = x.shape[0]                                                        # Size of the mesh.
    h   = x[0, 1] - x[0, 0]                                                 # h is defined as dx = dy.
    A   = np.zeros([m*m, m*m])                                              # A is initialized as a (m*m)x(m*m) square matrix.
    F   = np.zeros(m*m)                                                     # F is initialized as a (m*m) vector.
    u_b = u(x, y)                                                           # u_b to take the boundary conditions.

    # Finite Differences Matrix
    for i in range(m):                                                      # For all the nodes in one direction.
        for j in range(m):                                                  # For all the nodes in the other direction.
            k = i * m + j                                                   # Linearized Index.
            if i == 0 or i == m-1 or j == 0 or j == m-1:                    # If the node is in the boundary.
                A[k, k] = 1                                                 # A with one to keep the boundary condition.
                F[k] = u_b[i, j]                                            # The RHS has the boundary condition.
            else:                                                           # If the node is an inner node.
                A[k, k]   = -4/h**2                                         # A matrix value for central node.
                A[k, k-1] = 1/h**2                                          # A matrix value for left node.
                A[k, k+1] = 1/h**2                                          # A matrix value for right node.
                A[k, k-m] = 1/h**2                                          # A matrix value for downer node.
                A[k, k+m] = 1/h**2                                          # A matrix value for upper node.
                F[k] = f(x[i, j], y[i, j])                                  # RHS with the values of f.

    u_ap = np.linalg.solve(A, F).reshape((m, m))                            # Solve the linear problem.

    return u_ap                                                             # Return the computed solution.