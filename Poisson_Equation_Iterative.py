'''
Classical Finite Difference Schemes to solve Poisson Equation.
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

def Poisson1D(x, f, u):
    '''
        Poisson1D_Iter

        This code solves the 1D Poisson problem on a regular grid with Dirichlet boundary conditions
        using an Iterative formulation of the Finite Difference centered scheme.

        Arguments:
            x                       Array           Mesh of the region.
            f                       Function        Function with the sources and sinks.
            u                       Function        Function for the boundary conditions.
        
        Returns:
            u_ap        m x 1       Array           Array with the computed solution of the method.
    '''

    # Variable Initialization
    m        = len(x)                                                       # Size of the mesh.
    h        = x[2] - x[1]                                                  # h definition as dx.
    u_ap     = np.zeros(m)                                                  # u_ap initialization with zeros.
    err      = 1                                                            # err initialization with 1 to guarantee at least one iteration.
    tol      = np.sqrt(np.finfo(float).eps)                                 # Tolerance of the method.
    
    # Boundary Conditions
    u_ap[0]  = u(x[0])                                                      # Boundary condition at x = a
    u_ap[-1] = u(x[-1])                                                     # Boundary condition at x = b.

    # Finite Difference Solution
    while err >= tol:                                                       # While the error is greater than the tolerance.
        err = 0                                                             # The error of this iteration is 0.
        for i in range(1, m-1):                                             # For all the grid nodes.
            t   = (1/2)*(u_ap[i-1] + u_ap[i+1] - h**2*f(x[i]))              # Finite Differences Approximation.
            err = max(err, abs(t - u_ap[i]))                                # New error is computed.
            u_ap[i] = t                                                     # The approximation is saved.
    
    return u_ap                                                             # Return the mesh and the computed solution.

def Poisson1D_Neumann_1(x, f, sig, beta):
    '''
        Poisson1D_Neumann_1

        This code solves the 1D Poisson problem on a regular grid with Neumann and Dirichlet boundary conditions
        using an Iterative formulation of the Finite Difference centered scheme.

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
    m        = len(x)                                                       # Size of the mesh.
    h        = x[2] - x[1]                                                  # h definition as dx.
    u_ap     = np.zeros(m)                                                  # u_ap initialization with zeros.
    err      = 1                                                            # err initialization with 1 to guarantee at least one iteration.
    tol      = np.sqrt(np.finfo(float).eps)                                 # Tolerance of the method.
    
    # Boundary Conditions
    u_ap[-1] = beta                                                         # Boundary condition at x = b.

    # Finite Difference Solution
    while err >= tol:                                                       # While the error is greater than the tolerance.
        err     = 0                                                         # The error of this iteration is 0.
        for i in range(1, m-1):                                             # For all the grid nodes.
            t       = (1/2)*(u_ap[i-1] + u_ap[i+1] - h**2*f(x[i]))          # Finite Differences Approximation.
            err     = max(err, abs(t - u_ap[i]))                            # New error is computed.
            u_ap[i] = t                                                     # The approximation is saved.

        u_ap[0] = u_ap[1] - h*sig
    
    return u_ap                                                             # Return the mesh and the computed solution.

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
    m        = len(x)                                                       # Size of the mesh.
    h        = x[2] - x[1]                                                  # h definition as dx.
    u_ap     = np.zeros(m)                                                  # u_ap initialization with zeros.
    err      = 1                                                            # err initialization with 1 to guarantee at least one iteration.
    tol      = np.sqrt(np.finfo(float).eps)                                 # Tolerance of the method.
    
    # Boundary Conditions
    u_ap[-1] = beta                                                         # Boundary condition at x = b.

    # Finite Difference Solution
    while err >= tol:                                                       # While the error is greater than the tolerance.
        err     = 0                                                         # The error of this iteration is 0.
        for i in range(1, m-1):                                             # For all the grid nodes.
            t       = (1/2)*(u_ap[i-1] + u_ap[i+1] - h**2*f(x[i]))          # Finite Differences Approximation.
            err     = max(err, abs(t - u_ap[i]))                            # New error is computed.
            u_ap[i] = t                                                     # The approximation is saved.

        u_ap[0] = u_ap[1] - (h*sig + (h**2/2)*f(x[0]))

    return u_ap                                                             # Return the mesh and the computed solution.

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
    m        = len(x)                                                       # Size of the mesh.
    h        = x[2] - x[1]                                                  # h definition as dx.
    u_ap     = np.zeros(m)                                                  # u_ap initialization with zeros.
    err      = 1                                                            # err initialization with 1 to guarantee at least one iteration.
    tol      = np.sqrt(np.finfo(float).eps)                                 # Tolerance of the method.
    
    # Boundary Conditions
    u_ap[-1] = beta                                                         # Boundary condition at x = b.

    # Finite Difference Solution
    while err >= tol:                                                       # While the error is greater than the tolerance.
        err     = 0                                                         # The error of this iteration is 0.
        for i in range(1, m-1):                                             # For all the grid nodes.
            t       = (1/2)*(u_ap[i-1] + u_ap[i+1] - h**2*f(x[i]))          # Finite Differences Approximation.
            err     = max(err, abs(t - u_ap[i]))                            # New error is computed.
            u_ap[i] = t                                                     # The approximation is saved.

        u_ap[0] = (2/3)*(h*sig + 2*u_ap[1] - (1/2)*u_ap[2])
    
    return u_ap                                                             # Return the mesh and the computed solution.

def Poisson2D(x, y, f, u):
    '''
        Poisson2D

        This code solves the 2D Poisson problem on a regular grid with Dirichlet boundary conditions
        using an Iterative formulation of the Finite Difference centered scheme.

        Arguments:
            x           m x m       Array           Array with the x values of the nodes of the grid.
            y           m x m       Array           Array with the y values of the nodes of the grid.
            f                       Function        Function with the sources and sinks.
            u                       Function        Function for the boundary conditions.
        
        Returns:
            u_ap        m x m       Array           Array with the computed solution of the method.
    '''
    
    # Variable Initialization
    m    = x.shape[0]                                                       # Size of the mesh.
    h    = x[0, 1] - x[0, 0]                                                # h is defined as dx = dy.
    u_ap = np.zeros([m, m])                                                 # u_ap initialization with zeros.
    err  = 1                                                                # err initialization with 1 to guarantee at least one iteration.
    tol  = np.sqrt(np.finfo(float).eps)                                     # Tolerance of the method.
    
    # Boundary Condition
    for i in range(m):                                                      # For all the boundary nodes.
        u_ap[i,  0] = u(x[i,  0], y[i,  0])                                 # The boundary on the bottom is added.
        u_ap[i, -1] = u(x[i, -1], y[i, -1])                                 # The boundary on the top is added.
        u_ap[0,  i] = u(x[0,  i], y[0,  i])                                 # The boundary on the right is added.
        u_ap[-1, i] = u(x[-1, i], y[-1, i])                                 # The boundary on the left is added.
    
    while err >= tol:                                                       # While the error is greater than the tolerance.
        err = 0                                                             # The error of this iteration is 0.
        for i in range(1, m-1):                                             # For all the grid nodes in x.
            for j in range(1, m-1):                                         # For all the grin nodes in y.
                t = (1/4)*(u_ap[i-1, j]   + u_ap[i+1, j] + \
                           u_ap[i,   j-1] + u_ap[i,   j+1] - \
                    (h**2)*f(x[i, j], y[i, j]))                             # The new approximated solution is computed.
                err = max(err, abs(t - u_ap[i, j]))                         # The new error is computed.
                u_ap[i, j] = t                                              # The approximated solution is stored.
    
    return u_ap                                                             # Return the mesh and the computed solution.