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

def Poisson1D_Matrix_Neumann_1(a, b, m, f, sig, beta):
    '''
        Poisson1D_Matrix_Neumann_1

        This code solves the 1D Poisson problem on a regular grid with Neumann and Dirichlet boundary conditions
        using a Matrix formulation of the Finite Difference centered scheme.

        The Neumann boundary condition is applied with a two-point-backward finite difference scheme.

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
    A[-1,-2]    = 0                                         # Complete the Matrix.
    A[-1,-1]    = h**2                                      # Complete the Matrix.
    A           = A/h**2                                    # Complete the Matrix.

    # Right Hand Size (RHS)
    F           = f(x[0:m])                                 # Components of the RHS vector.
    F[0]        = sig                                       # Boundary condition on th RHS.
    F[-1]       = beta                                      # Boundary condition on the RHS.
    
    # Problem Solving
    A           = np.linalg.inv(A)                          # Solving the algebraic problem.
    u           = A@F                                       # Problem solution.

    return x, u                                             # Return the mesh and the computed solution.

def Poisson1D_Matrix_Neumann_2(a, b, m, f, sig, beta):
    '''
        Poisson1D_Matrix_Neumann_2

        This code solves the 1D Poisson problem on a regular grid with Neumann and Dirichlet boundary conditions
        using a Matrix formulation of the Finite Difference centered scheme.

        The Neumann boundary condition is applied with a two-point-centered finite difference scheme.

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
    A[-1,-2]    = 0                                         # Complete the Matrix.
    A[-1,-1]    = h**2                                      # Complete the Matrix.
    A           = A/h**2                                    # Complete the Matrix.
    
    # Right Hand Size (RHS)
    F           = f(x[0:m])                                 # Components of the RHS vector.
    F[0]        = sig+((h/2)*f(x[0]))                       # Boundary condition on th RHS.
    F[-1]       = beta                                      # Boundary condition on the RHS.
    
    # Problem Solving
    A           = np.linalg.inv(A)                          # Solving the algebraic problem.
    u           = A@F                                       # Problem solution.

    return x, u                                             # Return the mesh and the computed solution.

def Poisson1D_Matrix_Neumann_3(a, b, m, f, sig, beta):
    '''
        Poisson1D_Matrix_Neumann_3

        This code solves the 1D Poisson problem on a regular grid with Neumann and Dirichlet boundary conditions
        using a Matrix formulation of the Finite Difference centered scheme.

        The Neumann boundary condition is applied with a three-point-forward finite difference scheme.

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
    A[0,0]      = (3/2)*h                                   # Complete the Matrix.
    A[0,1]      = -2*h                                      # Complete the Matrix.
    A[0,2]      = (1/2)*h                                   # Complete the Matrix.
    A[-1,-2]    = 0                                         # Complete the Matrix.
    A[-1,-1]    = h**2                                      # Complete the Matrix.
    A           = A/h**2                                    # Complete the Matrix.
    
    # Right Hand Size (RHS)
    F           = f(x[0:m])                                 # Components of the RHS vector.
    F[0]        = sig                                       # Boundary condition on th RHS.
    F[-1]       = beta                                      # Boundary condition on the RHS.
    
    # Problem Solving
    A           = np.linalg.inv(A)                          # Solving the algebraic problem.
    u           = A@F                                       # Problem solution.

    return x, u                                             # Return the mesh and the computed solution.

def Poisson2D_Matrix(m, f, u):
    '''
        Poisson2D_Matrix

        This code solves the 2D Poisson problem on a regular grid with Dirichlet boundary conditions
        using a Matrix formulation of the Finite Difference centered scheme.

        Input:
            m                       Integer         Number of nodes in each direction of the grid.
            f                       Function        Function with the sources and sinks.
            u                       Function        Function for the boundary conditions.
        
        Output:
            x           m x m       Array           Array with the x values of the nodes of the generated grid.
            y           m x m       Array           Array with the y values of the nodes of the generated grid.
            u_ap        m x m       Array           Array with the computed solution of the method.
    '''
    # Variable Initialization
    x      = np.linspace(0,1,m)                             # Se hace la discretización en x del intervalo [0, 1].
    y      = np.linspace(0,1,m)                             # Se hace la discretización en y del intervalo [0, 1].
    h      = x[2] - x[1]                                    # Se calcula h para el método de Diferencias Finitas.
    x, y   = np.meshgrid(x,y)                               # Se hace la malla de la región para trabajar [0,1]x[0,1].
    A      = np.zeros([(m-2)*(m-2),(m-2)*(m-2)])            # Se inicializa la matriz A con ceros.
    F      = np.zeros([(m-2)*(m-2),1])                      # Se inicializa el vector F con ceros.
    u_ap   = np.zeros([m,m])                                # Se inicializa la variable para la solución aproximada.
    
    # Finite Differences Matrix
    dB   = np.diag(4*np.ones(m-2))                          # Se hace una matriz diagonal con 4s.
    dBp1 = np.diag(1*np.ones((m-2)-1), k=1)                 # Se hace una matriz identidad negativa inferior.
    dBm1 = np.diag(1*np.ones((m-2)-1), k=-1)                # Se hace una matriz identidad negativa superior.
    B    = (dB - dBp1 - dBm1)                               # Se ensamblan los bloques de la diagonal.
    I    = -np.identity(m-2)                                # Se hace una matriz identidad negativa.
    temp = 1                                                # Se inicializa un contador para guardar lo anterior en la matriz A.

    for i in range(0,(m-2)*(m-2),(m-2)):                    # Para cada uno de los bloques.
        A[i:temp*(m-2), i:temp*(m-2)] = B                   # Se hace la diagonal interior de bloques B.
        if temp*(m-2) < (m-2)*(m-2):                        # Si estamos arriba o abajo de la diagonal interior.
            A[temp*(m-2):temp*(m-2)+(m-2), i:temp*(m-2)] = I# Se pone una identidad negativa en la diagonal superior.
            A[i:temp*(m-2), temp*(m-2):temp*(m-2)+(m-2)] = I# Se pone una identidad negativa en la diagonal inferior.
        temp += 1                                           # Se aumenta el contador.

    # Right Hand Size (RHS)
    for i in range(1,m-1):                                  # Se recorren los nodos de la frontera.
        temp       = i-1                                    # Se asigna el valor de temp como los nodos de la primera frontera en x.
        F[temp] += u(x[i,0], y[i,0])                        # Se guarda la condición de frontera en el lado derecho.
        temp       = (i-1) + (m-2)*((m-1)-2)                # Se asigna el valor de temp como los nodos de la última frontera en x.
        F[temp] += u(x[i,m-1], y[i,m-1])                    # Se guarda la condición de frontera en el lado derecho.
        temp       = (m-2)*(i-1)                            # Se asigna el valor de temp para los nodos de la primera frontera en y.
        F[temp] += u(x[0,i], y[0,i])                        # Se guarda la condición de frontera en el lado derecho.
        temp       = ((m-1)-2) + (m-2)*(i-1)                # Se asgina el valor de temp para los nodos de la última frontera en y.
        F[temp] += u(x[m-1,i], y[m-1,i])                    # Se guarda la condición de frontera en el lado derecho.

    for i in range(1,m-1):                                  # Para todos los nodos en x.
        for j in range(1,m-2):                              # Para todos los nodos en y.
            temp       = (i-1) + (m-2)*(j-1)                # Se buscan los nodos que son frontera.
            F[temp] += -(h**2)*f(x[i,j], y[i,j])            # Se agrega f al lado derecho.

    # Problem Solving
    A  = np.linalg.pinv(A)                                  # La inversa nos servirá para multiplicar por el lado derecho.
    u2 = A@F                                                # Se multiplica la inversa por el lado derecho.
    u2 = np.reshape(u2, (m-2,m-2)).transpose()              # Se convierte el vector columna en matriz.

    # Approximation saving
    u_ap[1:(m-1), 1:(m-1)] = u2                             # Se guarda la aproximación calculada dentro de phi_ap.
    for i in range(m):                                      # Para todos los nodos.
        u_ap[i,0]   = u(x[i,0],y[i,0])                      # Se guarda la condición de frontera x = 0 en la aproximación.
        u_ap[i,m-1] = u(x[i,m-1],y[i,m-1])                  # Se guarda la condición de frontera x = m en la aproximación.
        u_ap[0,i]   = u(x[0,i],y[0,i])                      # Se guarda la condición de frontera y = 0 en la aproximación.
        u_ap[m-1,i] = u(x[m-1,i],y[m-1,i])                  # Se guarda la condición de frontera y = m en la aproximación.

    return x, y, u_ap                                       # Return the mesh and the computed solution.