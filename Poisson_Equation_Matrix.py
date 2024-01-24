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
    m           = len(x)                                                    # Size of the mesh.
    h           = x[1] - x[0]                                               # h definition as dx.
    A           = np.zeros([m,m])                                           # Initialization of Matrix A.
    F           = np.zeros(m)                                               # Initialization of Vector F.

    # Matrix assembly
    for i in range(1, m-1):                                                 # Loop through the Matrix.
        A[i, i-1] = 1                                                       # Superior diagonal.
        A[i, i]   = -2                                                      # Main diagonal.
        A[i, i+1] = 1                                                       # Inferior diagonal.
        F[i]      = f(x[i])                                                 # Components of the RHS vector.
    
    # Dirichlet conditions
    A[0,0]      = h**2                                                      # Complete the Matrix.
    A[-1,-1]    = h**2                                                      # Complete the Matrix.
    A           = A/h**2                                                    # Complete the Matrix.

    # Right Hand Size (RHS)
    F[0]        = u(x[0])                                                   # Boundary condition on th RHS.
    F[-1]       = u(x[-1])                                                  # Boundary condition on the RHS.

    # Problem Solving
    u_ap        = np.linalg.solve(A,F)                                      # Solve the algebraic problem.

    return u_ap                                                             # Return the mesh and the computed solution.

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
    m           = len(x)                                                    # Size of the mesh.
    h           = x[1] - x[0]                                               # h definition as dx.
    A           = np.zeros([m,m])                                           # Initialization of Matrix A.
    F           = np.zeros(m)                                               # Initialization of Vector F.

    # Matrix assembly
    for i in range(1, m-1):                                                 # Loop through the Matrix.
        A[i, i-1] = 1                                                       # Superior diagonal.
        A[i, i]   = -2                                                      # Main diagonal.
        A[i, i+1] = 1                                                       # Inferior diagonal.
        F[i]      = f(x[i])                                                 # Components of the RHS vector.

    # Neumann conditions
    A[0,0]      = -h                                                        # Complete the Matrix.
    A[0,1]      = h                                                         # Complete the Matrix.
    A[-1,-1]    = h**2                                                      # Complete the Matrix.
    A           = A/h**2                                                    # Complete the Matrix.

    # Right Hand Size (RHS)
    F[0]        = sig                                                       # Boundary condition on the RHS.
    F[-1]       = beta                                                      # Boundary condition on the RHS.
    
    # Problem Solving
    u_ap        = np.linalg.solve(A,F)                                      # Solve the algebraic problem.

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
    m           = len(x)                                                    # Size of the mesh.
    h           = x[1] - x[0]                                               # h definition as dx.
    A           = np.zeros([m,m])                                           # Initialization of Matrix A.
    F           = np.zeros(m)                                               # Initialization of Vector F.

    # Matrix assembly
    for i in range(1, m-1):                                                 # Loop through the Matrix.
        A[i, i-1] = 1                                                       # Superior diagonal.
        A[i, i]   = -2                                                      # Main diagonal.
        A[i, i+1] = 1                                                       # Inferior diagonal.
        F[i]      = f(x[i])                                                 # Components of the RHS vector.

    # Neumann conditions
    A[0,0]      = -h                                                        # Complete the Matrix.
    A[0,1]      = h                                                         # Complete the Matrix.
    A[-1,-2]    = 0                                                         # Complete the Matrix.
    A[-1,-1]    = h**2                                                      # Complete the Matrix.
    A           = A/h**2                                                    # Complete the Matrix.
    
    # Right Hand Size (RHS)
    F[0]        = sig+((h/2)*f(x[0]))                                       # Boundary condition on the RHS.
    F[-1]       = beta                                                      # Boundary condition on the RHS.
    
    # Problem Solving
    u_ap        = np.linalg.solve(A,F)                                      # Solve the algebraic problem.

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
    m           = len(x)                                                    # Size of the mesh.
    h           = x[1] - x[0]                                               # h definition as dx.
    A           = np.zeros([m,m])                                           # Initialization of Matrix A.
    F           = np.zeros(m)                                               # Initialization of Vector F.

    # Matrix assembly
    for i in range(1, m-1):                                                 # Loop through the Matrix.
        A[i, i-1] = 1                                                       # Superior diagonal.
        A[i, i]   = -2                                                      # Main diagonal.
        A[i, i+1] = 1                                                       # Inferior diagonal.
        F[i]      = f(x[i])                                                 # Components of the RHS vector.

    # Neumann conditions
    A[0,0]      = (3/2)*h                                                   # Complete the Matrix.
    A[0,1]      = -2*h                                                      # Complete the Matrix.
    A[0,2]      = (1/2)*h                                                   # Complete the Matrix.
    A[-1,-2]    = 0                                                         # Complete the Matrix.
    A[-1,-1]    = h**2                                                      # Complete the Matrix.
    A           = A/h**2                                                    # Complete the Matrix.
    
    # Right Hand Size (RHS)
    F[0]        = sig                                                       # Boundary condition on the RHS.
    F[-1]       = beta                                                      # Boundary condition on the RHS.
    
    u_ap        = np.linalg.solve(A,F)                                      # Solve the algebraic problem.

    return u_ap                                                             # Return the mesh and the computed solution.

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
    m      = x.shape[0]                                                     # Size of the mesh.
    h      = x[0,1] - x[0,0]                                                # h is defined as dx = dy.
    A      = np.zeros([(m-2)*(m-2),(m-2)*(m-2)])                            # A is initialized as a (m-2)*(m-2)x(m-2)*(m-2) square matrix.
    F      = np.zeros([(m-2)*(m-2),1])                                      # F is initialized as a (m-1)*(m-2)x1 vector.
    u_ap   = np.zeros([m,m])                                                # u_ap is initialized with zeros.
    
    # Finite Differences Matrix
    dB   = np.diag(4*np.ones(m-2))                                          # Main diagonal of first block of the matrix.
    dBp1 = np.diag(1*np.ones((m-2)-1), k=1)                                 # Upper diagonal of the first block of the matrix.
    dBm1 = np.diag(1*np.ones((m-2)-1), k=-1)                                # Lower diagonal of the first block of the matrix.
    B    = (dB - dBp1 - dBm1)                                               # First block of the matrix.
    I    = -np.identity(m-2)                                                # Identity matrix for the other two diagonals of the matrix.
    temp = 1                                                                # temp as a counter.

    for i in range(0,(m-2)*(m-2),(m-2)):                                    # For all the nodes in the range.
        A[i:temp*(m-2), i:temp*(m-2)] = B                                   # Save the main block of the matrix.
        if temp*(m-2) < (m-2)*(m-2):                                        # If we are in the neighbor nodes.
            A[temp*(m-2):temp*(m-2)+(m-2), i:temp*(m-2)] = I                # Save the identity in the matrix.
            A[i:temp*(m-2), temp*(m-2):temp*(m-2)+(m-2)] = I                # Save the identity in the matrix.
        temp += 1

    # Right Hand Size (RHS)
    for i in range(1,m-1):                                                  # For all the inner nodes.
        temp       = i-1                                                    # Value of a temporal counter.
        F[temp] += u(x[i,0], y[i,0])                                        # Add the Right Hand Size to F.
        temp       = (i-1) + (m-2)*((m-1)-2)                                # Value of a temporal counter.
        F[temp] += u(x[i,m-1], y[i,m-1])                                    # Add the Right Hand Size to F.
        temp       = (m-2)*(i-1)                                            # Value of a temporal counter.
        F[temp] += u(x[0,i], y[0,i])                                        # Add the Right Hand Size to F.
        temp       = ((m-1)-2) + (m-2)*(i-1)                                # Value of a temporal counter.
        F[temp] += u(x[m-1,i], y[m-1,i])                                    # Add the Right Hand Size to F.

    for i in range(1,m-1):                                                  # For all the inner nodes.
        for j in range(1,m-2):                                              # For all the inner nodes.
            temp       = (i-1) + (m-2)*(j-1)                                # Value of a temporal counter.    
            F[temp] += -(h**2)*f(x[i,j], y[i,j])                            # Add -f to the RHS.

    # Problem Solving
    A  = np.linalg.pinv(A)                                                  # Inverse matrix.
    u2 = A@F                                                                # Problem solving.
    u2 = np.reshape(u2, (m-2,m-2)).transpose()                              # Reshape the solution into a square matrix.

    # Approximation saving
    u_ap[1:(m-1), 1:(m-1)] = u2                                             # Add the solution to u_ap.
    for i in range(m):                                                      # For all the nodes in the boundary.
        u_ap[i,0]   = u(x[i,0],y[i,0])                                      # Add the boundary condition to the solution.
        u_ap[i,m-1] = u(x[i,m-1],y[i,m-1])                                  # Add the boundary condition to the solution.
        u_ap[0,i]   = u(x[0,i],y[0,i])                                      # Add the boundary condition to the solution.
        u_ap[m-1,i] = u(x[m-1,i],y[m-1,i])                                  # Add the boundary condition to the solution.

    return u_ap                                                             # Return the mesh and the computed solution.

def Poisson2D_2(x, y, f, u):
    '''
        Poisson2D_2

        This code solves the 2D Poisson problem on a regular grid with Dirichlet boundary conditions
        using a Matrix formulation of the Finite Difference centered scheme. In this code, the Right Hand
        Size is formulated as a matrix and the flatten to be a vector.

        Arguments:
            x           m x m       Array           Array with the x values of the nodes of the grid.
            y           m x m       Array           Array with the y values of the nodes of the grid.
            f                       Function        Function with the sources and sinks.
            u                       Function        Function for the boundary conditions.
        
        Returns:
            u_ap        m x m       Array           Array with the computed solution of the method.
    '''

    # Variable Initialization
    m      = x.shape[0]                                                     # Size of the mesh.
    h      = x[0,1] - x[0,0]                                                # h is defined as dx = dy.
    A      = np.zeros([(m-2)*(m-2),(m-2)*(m-2)])                            # A is initialized as a (m-2)*(m-2)x(m-2)*(m-2) square matrix.
    F      = np.zeros([(m-2),(m-2)])                                        # F is initialized as a (m-1)*(m-2)x1 vector.
    u_ap   = np.zeros([m,m])                                                # u_ap is initialized with zeros.

    # Finite Differences Matrix
    dB   = np.diag(4*np.ones(m-2))                                          # Main diagonal of first block of the matrix.
    dBp1 = np.diag(1*np.ones((m-2)-1), k=1)                                 # Upper diagonal of the first block of the matrix.
    dBm1 = np.diag(1*np.ones((m-2)-1), k=-1)                                # Lower diagonal of the first block of the matrix.
    B    = (dB - dBp1 - dBm1)                                               # First block of the matrix.
    I    = -np.identity(m-2)                                                # Identity matrix for the other two diagonals of the matrix.
    temp = 1                                                                # temp as a counter.

    for i in range(0,(m-2)*(m-2),(m-2)):                                    # For all the nodes in the range.
        A[i:temp*(m-2), i:temp*(m-2)] = B                                   # Save the main block of the matrix.
        if temp*(m-2) < (m-2)*(m-2):                                        # If we are in the neighbor nodes.
            A[temp*(m-2):temp*(m-2)+(m-2), i:temp*(m-2)] = I                # Save the identity in the matrix.
            A[i:temp*(m-2), temp*(m-2):temp*(m-2)+(m-2)] = I                # Save the identity in the matrix.
        temp += 1

    # Right Hand Size (RHS)
    for i in range(m-2):                                                    # For all the inner nodes in x direction.
        for j in range(m-2):                                                # For all the inner nodes in y direction.
            F[i,j] -= (h**2)*f(x[i+1,j+1], y[i+1,j+1])                      # Value of the RHS.
        F[i,0]  += u(x[i+1,0],  y[i+1,0])                                   # Boundary conditions to the RHS.
        F[i,-1] += u(x[i+1,-1], y[i+1,-1])                                  # Boundary conditions to the RHS.
        F[0,i]  += u(x[0,i+1],  y[0,i+1])                                   # Boundary conditions to the RHS.
        F[-1,i] += u(x[-1,i+1], y[-1,i+1])                                  # Boundary conditions to the RHS.
    
    F = F.flatten(order='F')                                                # Make F a column vector.

    # Problem Solving
    A  = np.linalg.pinv(A)                                                  # Inverse matrix.
    u2 = A@F                                                                # Problem solving.
    u2 = np.reshape(u2, (m-2,m-2)).transpose()                              # Reshape the solution into a square matrix.

    # Approximation saving
    u_ap[1:(m-1), 1:(m-1)] = u2                                             # Add the solution to u_ap.
    for i in range(m):                                                      # For all the nodes in the boundary.
        u_ap[i,0]   = u(x[i,0],y[i,0])                                      # Add the boundary condition to the solution.
        u_ap[i,m-1] = u(x[i,m-1],y[i,m-1])                                  # Add the boundary condition to the solution.
        u_ap[0,i]   = u(x[0,i],y[0,i])                                      # Add the boundary condition to the solution.
        u_ap[m-1,i] = u(x[m-1,i],y[m-1,i])                                  # Add the boundary condition to the solution.

    return u_ap                                                             # Return the mesh and the computed solution.