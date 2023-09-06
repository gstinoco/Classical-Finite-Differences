"""
Classical Finite Difference Schemes to solve Poisson Equation.

The problem to solve is:
    u(x)_{xx} = -f(x)

Subject to conditions:
    u(x)_\Omega = g(x)

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
    August, 2023.
"""

# Library Importation
import numpy as np

def Poisson1D_Matrix(a, b, m, f, u):
    '''
        Poisson1D_Matrix

        This code solves the 1D Poisson problem on a regular grid with Dirichlet boundary conditions
        using a Matrix formulation of the Finite Difference centered scheme.

        Input:
            a                       Real            Initial value of the test region.
            b                       Real            Final value of the test region.
            m                       Integer         Number of nodes in the grid.
            f                       Function        Function with the sources and sinks.
            u                       Function        Function for the boundary conditions.
        
        Output:
            x           m x 1       Array           Array with the grid generated for the problem.
            u_ap        m x 1       Array           Array with the computed solution of the method.
    '''
    # Variable Initialization
    x           = np.linspace(a,b,m)                        # Creation of the mesh.
    h           = x[1] - x[0]                               # h definition as dx.
    u_ap        = np.zeros([m])                             # u_ap initialization.

    # Boundary Conditions
    alpha       = u(x[0])                                   # Boundary condition at x = a.
    beta        = u(x[-1])                                  # Boundary condition at x = b.

    # Finite Differences Matrix
    dA          = np.diag(-2*np.ones(m-2))                  # Main diagonal of the Matrix.
    dAp1        = np.diag(np.ones((m-2)-1), k = 1)          # Lower diagonal of the Matrix.
    dAm1        = np.diag(np.ones((m-2)-1), k = -1)         # Upper diagonal of the Matrix.
    A           = dA + dAp1 + dAm1                          # Matrix assembly.
    A           = A/h**2                                    # Divide the Matrix by h^2.

    # Right Hand Size (RHS)
    F           = -f(x[1:m-1])                              # Components of the RHS vector.
    F[0]       -= alpha/h**2                                # Boundary condition on th RHS.
    F[-1]      -= beta/h**2                                 # Boundary condition on the RHS.

    # Problem Solving
    A           = np.linalg.inv(A)                          # Solving the algebraic problem.
    u           = A@F                                       # Problem solution.

    # Approximation saving
    u_ap[1:m-1] = u                                         # Save the computed solution.
    u_ap[0]     = alpha                                     # Add the boundary condition at x = a.
    u_ap[-1]    = beta                                      # Add the boundary condition at x = b.

    return x, u_ap                                          # Return the mesh and the computed solution.

def Poisson1D_Matrix_Neumann(a, b, m, f, sig, beta):
    '''
        Poisson1D_Matrix

        This code solves the 1D Poisson problem on a regular grid with Neumann and Dirichlet boundary conditions
        using a Matrix formulation of the Finite Difference centered scheme.

        The Neumann boundary condition is applied with a backwards finite difference scheme.

        Input:
            a                       Real            Initial value of the test region.
            b                       Real            Final value of the test region.
            m                       Integer         Number of nodes in the grid.
            f                       Function        Function with the sources and sinks.
            sig                     Real            Value of the derivative on the Neumann boundary condition.
            beta                    Real            Value of the function on the Dirichlet boundary condition.
        
        Output:
            x           m x 1       Array           Array with the grid generated for the problem.
            u           m x 1       Array           Array with the computed solution of the method.
    '''
    # Variable Initialization
    x           = np.linspace(a,b,m)                        # Creation of the mesh.
    h           = x[1] - x[0]                               # h definition as dx.

    # Finite Differences Matrix
    dA          = np.diag(-2*np.ones(m))                    # Main diagonal of the Matrix.
    dAp1        = np.diag(np.ones(m-1), k = 1)              # Lower diagonal of the Matrix.
    dAm1        = np.diag(np.ones(m-1), k = -1)             # Upper diagonal of the Matrix.
    A           = dA + dAp1 + dAm1                          # Matrix assembly.

    # Handcrafted Neumann conditions
    A[0,0]      = -h                                        # Complete the Matrix.
    A[0,1]      = h                                         # Complete the Matrix.
    A[m-1,m-2]  = 0                                         # Complete the Matrix.
    A[m-1,m-1]  = h**2                                      # Complete the Matrix.
    A            = A/h**2                                   # Complete the Matrix.

    # Right Hand Size (RHS)
    F           = -f(x[0:m])                                # Components of the RHS vector.
    F[0]       -= sig                                       # Boundary condition on th RHS.
    F[-1]      -= beta                                      # Boundary condition on the RHS.

    # Problem Solving
    A           = np.linalg.inv(A)                          # Solving the algebraic problem.
    u           = A@F                                       # Problem solution.

    return x, u                                             # Return the mesh and the computed solution.

def Poisson1D_Matrix_Neumann_2(a, b, m, f, sig, beta):
    '''
        Poisson1D_Matrix

        This code solves the 1D Poisson problem on a regular grid with Neumann and Dirichlet boundary conditions
        using a Matrix formulation of the Finite Difference centered scheme.

        The Neumann boundary condition is applied with a centered finite difference scheme.

        Input:
            a                       Real            Initial value of the test region.
            b                       Real            Final value of the test region.
            m                       Integer         Number of nodes in the grid.
            f                       Function        Function with the sources and sinks.
            sig                     Real            Value of the derivative on the Neumann boundary condition.
            beta                    Real            Value of the function on the Dirichlet boundary condition.
        
        Output:
            x           m x 1       Array           Array with the grid generated for the problem.
            u           m x 1       Array           Array with the computed solution of the method.
    '''
    # Variable Initialization
    x           = np.linspace(a,b,m)                        # Creation of the mesh.
    h           = x[1] - x[0]                               # h definition as dx.

    # Finite Differences Matrix
    dA          = np.diag(-2*np.ones(m))                    # Main diagonal of the Matrix.
    dAp1        = np.diag(np.ones(m-1), k = 1)              # Lower diagonal of the Matrix.
    dAm1        = np.diag(np.ones(m-1), k = -1)             # Upper diagonal of the Matrix.
    A           = dA + dAp1 + dAm1                          # Matrix assembly.

    # Handcrafted Neumann conditions
    A[0,0]      = -h                                        # Complete the Matrix.
    A[0,1]      = h                                         # Complete the Matrix.
    A[m-1,m-2]  = 0                                         # Complete the Matrix.
    A[m-1,m-1]  = h**2                                      # Complete the Matrix.
    A            = A/h**2                                   # Complete the Matrix.
    
    # Right Hand Size (RHS)
    F           = -f(x[0:m])                                # Components of the RHS vector.
    F[0]       -= sig+((h/2)*f(x[0]))                       # Boundary condition on th RHS.
    F[-1]      -= beta                                      # Boundary condition on the RHS.
    
    # Problem Solving
    A           = np.linalg.inv(A)                          # Solving the algebraic problem.
    u           = A@F                                       # Problem solution.

    return x, u                                             # Return the mesh and the computed solution.