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
    # Variable Initialization
    x           = np.linspace(a,b,m)                    # Creation of the mesh.
    h           = x[1] - x[0]                           # h definition as dx.
    u_ap        = np.zeros([m])                         # u_ap initialization.

    # Boundary Conditions
    alpha       = u(x[0])                               # Boundary condition at x = a.
    beta        = u(x[-1])                              # Boundary condition at x = b.

    # Finite Differences Matrix
    dA          = np.diag(-2*np.ones(m-2))              # Main diagonal of the Matrix.
    dAp1        = np.diag(np.ones((m-2)-1), k = 1)      # Lower diagonal of the Matrix.
    dAm1        = np.diag(np.ones((m-2)-1), k = -1)     # Upper diagonal of the Matrix.
    A           = dA + dAp1 + dAm1                      # Matrix assembly.
    A           = A/h**2                                # Divide the Matrix by h^2.

    # Right Hand Size (RHS)
    F           = -f(x[1:m-1])                          # Components of the RHS vector.
    F[0]       -= alpha/h**2                            # Boundary condition on th RHS.
    F[-1]      -= beta/h**2                             # Boundary condition on the RHS.

    # Problem Solving
    A           = np.linalg.inv(A)                      # Solving the algebraic problem.
    u           = A@F                                   # Problem solution.

    # Approximation saving
    u_ap[1:m-1] = u                                     # Save the computed solution.
    u_ap[0]     = alpha                                 # Add the boundary condition at x = a.
    u_ap[-1]    = beta                                  # Add the boundary condition at x = b.

    return x, u_ap                                      # Return the mesh and the computed solution.

def Poisson1D_Matrix_Neumann(a, b, m, f, sig, beta):
    # Variable Initialization
    x           = np.linspace(a,b,m)
    h           = x[1] - x[0]
    u_ap        = np.zeros([m])

    # Finite Differences Matrix
    dA          = np.diag(-2*np.ones(m))
    dAp1        = np.diag(np.ones((m)-1), k = 1)
    dAm1        = np.diag(np.ones((m)-1), k = -1)
    A           = dA + dAp1 + dAm1
    A[0,0]      = -h
    A[0,1]      = h
    A[m-1,m-2]  = 0
    A[m-1,m-1]  = h**2

    A            = A/h**2
    F            = -f(x[0:m])
    F[0]        -= sig
    F[-1]       -= beta

    # Problem Solving
    A           = np.linalg.inv(A)
    u           = A@F

    # Approximation saving
    u_ap = u

    return x, u_ap

def Poisson1D_Matrix_Neumann_2(a, b, m, f, sig, beta):
    # Variable Initialization
    x           = np.linspace(a,b,m)
    h           = x[1] - x[0]
    u_ap        = np.zeros([m])

    # Finite Differences Matrix
    dA          = np.diag(-2*np.ones(m))
    dAp1        = np.diag(np.ones((m)-1), k = 1)
    dAm1        = np.diag(np.ones((m)-1), k = -1)
    A           = dA + dAp1 + dAm1
    A[0,0]      = -h
    A[0,1]      = h
    A[m-1,m-2]  = 0
    A[m-1,m-1]  = h**2

    A            = A/h**2
    F            = -f(x[0:m])
    F[0]        -= sig+((h/2)*f(x[0]))
    F[-1]       -= beta

    # Problem Solving
    A           = np.linalg.inv(A)
    u           = A@F

    # Approximation saving
    u_ap = u

    return x, u_ap