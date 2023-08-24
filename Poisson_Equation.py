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
    x           = np.linspace(a,b,m)
    dx          = x[1] - x[0]
    u_ap        = np.zeros([m])

    # Boundary Conditions
    alpha       = u(x[0])
    beta        = u(x[-1])

    # Finite Differences Matrix
    dA          = np.diag(-2*np.ones(m-2))
    dAp1        = np.diag(np.ones((m-2)-1), k = 1)
    dAm1        = np.diag(np.ones((m-2)-1), k = -1)
    A           = dA + dAp1 + dAm1
    A           = A/dx**2
    F           = -f(x[1:m-1])
    F[0]       -= alpha/dx**2
    F[-1]      -= beta/dx**2

    # Problem Solving
    A           = np.linalg.inv(A)
    u           = A@F

    # Approximation saving
    u_ap[1:m-1] = u
    u_ap[0]     = alpha
    u_ap[-1]    = beta

    return x, u_ap