import numpy as np

def Poisson_GFD_iterative(x, y, u, f, L):
    # Variable initialization
    [m, n] = x.shape                                                                # The size of the mesh is found.
    err    = 1                                                                      # err initialization in 1.
    tol    = 1e-6                                                                   # The tolerance is defined.
    u_ap   = np.zeros([m,n])                                                        # u_ap initialization with zeros.
    u_ex   = np.zeros([m,n])                                                        # u_ex initialization with zeros.

    # Boundary conditions
    for i in range(m):                                                              # For each of the nodes on the x boundaries.
        u_ap[i, 0]   = u(x[i, 0],   y[i, 0])                                        # The boundary condition is assigned at the first y.
        u_ap[i, n-1] = u(x[i, n-1], y[i, n-1])                                      # The boundary condition is assigned at the last y.
    for j in range(n):                                                              # For each of the nodes on the y boundaries.
        u_ap[0,   j] = u(x[0,   j], y[0,   j])                                      # The boundary condition is assigned at the first x.
        u_ap[m-1, j] = u(x[m-1, j], y[m-1, j])                                      # The boundary condition is assigned at the last x.

    # Computation of Gamma values
    Gamma = Gammas(x, y, L)                                                         # Gamma computation.

    # A Generalized Finite Differences Method
    while err >= tol:                                                               # As long as the error is greater than the tolerance.
        err = 0                                                                     # Error becomes zero to be able to update.
        for i in range(1,m-1):                                                      # For each of the nodes on the x axis.
            for j in range(1,n-1):                                                  # For each of the nodes on the y axis.
                t = (f(x[i, j], y[i, j]) - (              \
                    Gamma[i, j, 1]*u_ap[i + 1, j    ] + \
                    Gamma[i, j, 2]*u_ap[i + 1, j + 1] + \
                    Gamma[i, j, 3]*u_ap[i    , j + 1] + \
                    Gamma[i, j, 4]*u_ap[i - 1, j + 1] + \
                    Gamma[i, j, 5]*u_ap[i - 1, j    ] + \
                    Gamma[i, j, 6]*u_ap[i - 1, j - 1] + \
                    Gamma[i, j, 7]*u_ap[i    , j - 1] + \
                    Gamma[i, j, 8]*u_ap[i + 1, j - 1]))/Gamma[i, j, 0]              # u_ap is calculated at the central node.
                err = max(err, abs(t - u_ap[i, j]));                                # Error computation.
                u_ap[i,j] = t;                                                      # The previously computed value is assigned.
    
    # Theoretical Solution
    for i in range(m):                                                              # For all the nodes on x.
        for j in range(n):                                                          # For all the nodes on y.
            u_ex[i,j] = u(x[i,j], y[i,j])                                           # The theoretical solution is computed.
    
    return u_ap, u_ex

def Poisson_GFD_Matrix(x, y, u, f, L):
    # Variable initialization
    [m, n] = x.shape                                                                # The size of the mesh is found.
    u_ap   = np.zeros([m,n])                                                        # u_ap initialization with zeros.
    u_ex   = np.zeros([m,n])                                                        # u_ex initialization with zeros.

    # Boundary conditions
    for i in range(m):                                                              # For each of the nodes on the x boundaries.
        u_ap[i, 0]   = u(x[i, 0],   y[i, 0])                                        # The boundary condition is assigned at the first y.
        u_ap[i, n-1] = u(x[i, n-1], y[i, n-1])                                      # The boundary condition is assigned at the last y.
    for j in range(n):                                                              # For each of the nodes on the y boundaries.
        u_ap[0,   j] = u(x[0,   j], y[0,   j])                                      # The boundary condition is assigned at the first x.
        u_ap[m-1, j] = u(x[m-1, j], y[m-1, j])                                      # The boundary condition is assigned at the last x.

    # Computation of Gamma values
    K, R = Gammas_K(x, y, L, u, f)                                                # Gamma computation.

    # A Generalized Finite Differences Method
    un = np.linalg.pinv(K)@R
    
    for i in np.arange(1,m-1):                                                      # For each of the interior nodes on x.
        for j in np.arange(1,n-1):                                                  # For each of the interior nodes on y.
            u_ap[i, j] = un[i + (j)*m]                                              # u_ap values are assigned.
    
    # Theoretical Solution
    for i in range(m):                                                              # For all the nodes on x.
        for j in range(n):                                                          # For all the nodes on y.
            u_ex[i,j] = u(x[i,j], y[i,j])                                           # The theoretical solution is computed.
    
    return u_ap, u_ex

def Gammas(x, y, L):
    [m, n] = x.shape                                                              # The size of the mesh is found.
    Gamma  = np.zeros([m,n,9])                                                    # Gamma initialization with zeros.

    for i in range(1,m-1):                                                          # For each of the nodes in x.
        for j in range(1,n-1):                                                      # For each of the nodes in y.
            dx = np.array([x[i + 1, j]   - x[i, j], x[i + 1, j + 1] - x[i, j], \
                           x[i, j + 1]   - x[i, j], x[i - 1, j + 1] - x[i, j], \
                           x[i - 1, j]   - x[i, j], x[i - 1, j - 1] - x[i, j], \
                           x[i, j - 1]   - x[i, j], x[i + 1, j - 1] - x[i, j]])     # dx is computed.
            
            dy = np.array([y[i + 1, j]   - y[i, j], y[i + 1, j + 1] - y[i, j], \
                           y[i, j + 1]   - y[i, j], y[i - 1, j + 1] - y[i, j], \
                           y[i - 1, j]   - y[i, j], y[i - 1, j - 1] - y[i, j], \
                           y[i, j - 1]   - y[i, j], y[i + 1, j - 1] - y[i, j]])     # dy is computed.
            
            M = np.vstack([[dx], [dy], [dx**2], [dx*dy], [dy**2]])                  # M matrix is assembled.
            M = np.linalg.pinv(M)                                                   # The pseudoinverse of matrix M.
            YY = M@L                                                                # M*L computation.
            Gem = np.vstack([-sum(YY), YY])                                         # Gamma values are found.
            for k in np.arange(9):                                                  # For each of the Gamma values.
                Gamma[i,j,k] = Gem[k]                                               # The Gamma value is stored.

    return Gamma

def Gammas_K(x, y, L, u, f):
    [m, n] = x.shape
    K      = np.zeros([m*n, m*n])
    R      = np.zeros([m*n, 1])

    # Gammas computation and Matrix assembly
    for i in np.arange(1,m-1):                                                      # For each of the inner nodes on x.
        for j in np.arange(1,n-1):                                                  # For each of the inner nodes on y.
            X  = np.array(x[i-1:i+2, j-1:j+2])                                      # X is formed with the x-coordinates of the stencil.
            Y  = np.array(y[i-1:i+2, j-1:j+2])                                      # Y is formed with the y-coordinates of the stencil.
            dx = np.hstack([X[0,0] - X[1,1], X[1,0] - X[1,1], \
                            X[2,0] - X[1,1], X[0,1] - X[1,1], \
                            X[2,1] - X[1,1], X[0,2] - X[1,1], \
                            X[1,2] - X[1,1], X[2,2] - X[1,1]])                      # dx computation.
            dy = np.hstack([Y[0,0] - Y[1,1], Y[1,0] - Y[1,1], \
                            Y[2,0] - Y[1,1], Y[0,1] - Y[1,1], \
                            Y[2,1] - Y[1,1], Y[0,2] - Y[1,1], \
                            Y[1,2] - Y[1,1], Y[2,2] - Y[1,1]])                      # dy computation
            M  = np.vstack([[dx], [dy], [dx**2], [dx*dy], [dy**2]])                 # M matrix is assembled.
            M  = np.linalg.pinv(M)                                                  # The pseudoinverse of matrix M.
            YY = M@L                                                                # M*L computation.
            Gamma = np.vstack([-sum(YY), YY])                                       # Gamma values are found.
            p           = m*(j) + i                                                 # Variable to find the correct position in the Matrix.
            K[p, p]     = Gamma[0]                                                  # Gamma 0 assignation
            K[p, p-1-m] = Gamma[1]                                                  # Gamma 1 assignation
            K[p, p-m]   = Gamma[2]                                                  # Gamma 2 assignation
            K[p, p+1-m] = Gamma[3]                                                  # Gamma 3 assignation
            K[p, p-1]   = Gamma[4]                                                  # Gamma 4 assignation
            K[p, p+1]   = Gamma[5]                                                  # Gamma 5 assignation
            K[p, p-1+m] = Gamma[6]                                                  # Gamma 6 assignation
            K[p, p+m]   = Gamma[7]                                                  # Gamma 7 assignation
            K[p, p+1+m] = Gamma[8]                                                  # Gamma 8 assignation

            temp     = (i-1) + (m-2)*(j-1)                                          # Boundary nodes.
            R[temp] -= Gamma[0]*f(x[i,j], y[i,j])                                   # f to RHS.
    
    for j in np.arange(n):                                                          # For all the nodes in y.
        K[m*j, m*j] = 1                                                             # Zeros for the boundary nodes.
        K[m*j, :]   = 0
        if j%2 == 0:
            R[m*j] = u(x[0,j],y[0,j])
        else:
            R[m*j] = u(x[-1,j],y[-1,j])
    
    for i in range(m):
        if i%2 == 0:
            R[n*i] = u(x[i,0],y[i,0])
        else:
            R[n*1] = u(x[i,-1],y[i,-1])


    return K, R