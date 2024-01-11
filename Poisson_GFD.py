import numpy as np
import Scripts.Gammas as Gammas
import Scripts.Neighbors as Neighbors

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
    Gamma = Gammas.Mesh(x, y, L)                                                    # Gamma computation.

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

def Cloud(p, tt, u, f):
    # Variable initialization
    m    = len(p[:,0])                                                              # The total number of nodes is calculated.
    nvec = 8                                                                        # The maximum number of nodes.
    u_ap = np.zeros([m])                                                            # u_ap initialization with zeros.
    u_ex = np.zeros([m])                                                            # u_ex initialization with zeros.

    # Boundary conditions
    for i in np.arange(m):                                                          # For all the nodes.
        if p[i,2] == 1 or p[i,2] == 2:                                              # If the node is in the boundary.
            u_ap[i]   = u(p[i, 0], p[i, 1])                                         # The boundary condition is assigned.
    
    vec = Neighbors.Cloud(p, nvec)                                                  # Neighbor search with the proper routine.

    # Computation of Gamma values
    L = np.vstack([[0], [0], [2], [0], [2]])                                        # The values of the differential operator are assigned.
    K, R = Gammas.Cloud(p, vec, L, u, f)                                            # Gamma computation.
    
    # A Generalized Finite Differences Method
    K = np.linalg.pinv(K)
    un  = K@R
    for i in np.arange(m):                                                          # For all the nodes.
        if p[i,2] == 0:                                                             # If the node is an inner node.
            u_ap[i] = un[i]                                                         # Save the computed solution.
    
    # Theoretical Solution
    u_ex = u(p[:,0], p[:,1])                                                        # The theoretical solution is computed.

    return u_ap, u_ex, vec



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
    K, R = Gammas.Mesh_K(x, y, L, u, f)                                             # Gamma computation.

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