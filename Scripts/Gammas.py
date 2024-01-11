import numpy as np

def Mesh(x, y, L):
    [m, n] = x.shape                                                                # The size of the mesh is found.
    Gamma  = np.zeros([m,n,9])                                                      # Gamma initialization with zeros.

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

def Mesh_K(x, y, L, phi, f):
    [m, n] = x.shape
    K = np.zeros([(m-2)*(n-2), (m-2)*(n-2)])
    R  = np.zeros([(m-2)*(n-2), 1])

    for i in np.arange(1,m-1):
        for j in np.arange(1,n-1):
            u = np.array(x[i-1:i+2, j-1:j+2])
            v = np.array(y[i-1:i+2, j-1:j+2])
            dx = np.hstack([u[0,0] - u[1,1], u[1,0] - u[1,1], \
                            u[2,0] - u[1,1], u[0,1] - u[1,1], \
                            u[2,1] - u[1,1], u[0,2] - u[1,1], \
                            u[1,2] - u[1,1], u[2,2] - u[1,1]])
            dy = np.hstack([v[0,0] - v[1,1], v[1,0] - v[1,1], \
                            v[2,0] - v[1,1], v[0,1] - v[1,1], \
                            v[2,1] - v[1,1], v[0,2] - v[1,1], \
                            v[1,2] - v[1,1], v[2,2] - v[1,1]])
            M = np.vstack([[dx], [dy], [dx**2], [dx*dy], [dy**2]])
            M = np.linalg.pinv(M)
            YY = M@L
            Gamma = np.vstack([-sum(YY), YY])
            p = m*(j) + i
            K[p, p-1]   = Gamma[4]
            K[p, p]     = Gamma[0]
            K[p, p+1]   = Gamma[5]
            K[p, p-1-m] = Gamma[1]
            K[p, p-m]   = Gamma[2]
            K[p, p+1-m] = Gamma[3]
            K[p, p-1+m] = Gamma[6]
            K[p, p+m]   = Gamma[7]
            K[p, p+1+m] = Gamma[8]

            temp     = (i-1) + (m-2)*(j-1)
            R[temp] += Gamma[0]*f(x[i,j], y[i,j])
    
    for j in np.arange(n):
        K[m*j, m*j] = 0
    
    for i in np.arange(1,m-1):
        p       = i+(n-1)*m
        K[i, i] = 0
        K[p, p] = 0

        temp       = i-1
        R[temp] += phi(x[i,0], y[i,0])
        temp       = (i-1) + (m-2)*((m-1)-2)
        R[temp] += phi(x[i,m-1], y[i,m-1])
        temp       = (m-2)*(i-1)
        R[temp] += phi(x[0,i], y[0,i])
        temp       = ((m-1)-2) + (m-2)*(i-1)
        R[temp] += phi(x[m-1,i], y[m-1,i])
    
    return K, R

def Cloud(p, vec, L, u, f):
    # Variable initialization
    nvec  = len(vec[0,:])                                                           # The maximum number of neighbors.
    m     = len(p[:,0])                                                             # The total number of nodes.
    K     = np.zeros([m,m])                                                         # K initialization with zeros.
    R     = np.zeros([m])                                                           # K initialization with zeros.
    
    # Gammas computation and Matrix assembly
    for i in np.arange(m):                                                          # For each of the nodes.
        if p[i,2] == 0:                                                             # If the node is an inner node.
            nvec = sum(vec[i,:] != -1)                                              # The total number of neighbors of the node.
            dx   = np.zeros([nvec])                                                 # dx initialization with zeros.
            dy   = np.zeros([nvec])                                                 # dy initialization with zeros.
            for j in np.arange(nvec):                                               # For each of the neighbor nodes.
                vec1  = int(vec[i, j])                                              # The neighbor index is found.
                dx[j] = p[vec1, 0] - p[i,0]                                         # dx is computed.
                dy[j] = p[vec1, 1] - p[i,1]                                         # dy is computed.
            M     = np.vstack([[dx], [dy], [dx**2], [dx*dy], [dy**2]])              # M matrix is assembled.
            M     = np.linalg.pinv(M)                                               # The pseudoinverse of matrix M.
            YY    = M@L                                                             # M*L computation.
            Gamma = np.vstack([-sum(YY), YY]).transpose()                           # Gamma values are found.
            K[i,i] = Gamma[0,0]                                                     # The corresponding Gamma for the central node.
            for j in np.arange(nvec):                                               # For each of the neighbor nodes.
                K[i, vec[i,j]] = Gamma[0,j+1]                                       # The corresponding Gamma for the neighbor node.
            
        if p[i,2] == 1 or p[i,2] == 2:                                              # If the node is in the boundary.
            for j in np.arange(nvec):                                               # For each of the neighbor nodes.
                K[i, :] = 0                                                         # Neighbor node weight is equal to 0.
                K[i, i] = 1                                                         # Central node weight is equal to 1.
    
    # R computation
    for i in np.arange(m):
        if p[i,2] == 0:
            R[i] = f(p[i, 0], p[i, 1])
        if p[i,2] == 1 or p[i,2] == 2:
            R[i] = u(p[i,0], p[i,1])

    return K, R

def Cloud_t(p, vec, L):
    # Variable initialization
    nvec  = len(vec[0,:])                                                           # The maximum number of neighbors.
    m     = len(p[:,0])                                                             # The total number of nodes.
    K     = np.zeros([m,m])                                                         # K initialization with zeros.
    
    # Gammas computation and Matrix assembly
    for i in np.arange(m):                                                          # For each of the nodes.
        if p[i,2] == 0:                                                             # If the node is an inner node.
            nvec = sum(vec[i,:] != -1)                                              # The total number of neighbors of the node.
            dx   = np.zeros([nvec])                                                 # dx initialization with zeros.
            dy   = np.zeros([nvec])                                                 # dy initialization with zeros.
            for j in np.arange(nvec):                                               # For each of the neighbor nodes.
                vec1  = int(vec[i, j])                                              # The neighbor index is found.
                dx[j] = p[vec1, 0] - p[i,0]                                         # dx is computed.
                dy[j] = p[vec1, 1] - p[i,1]                                         # dy is computed.
            M     = np.vstack([[dx], [dy], [dx**2], [dx*dy], [dy**2]])              # M matrix is assembled.
            M     = np.linalg.pinv(M)                                               # The pseudoinverse of matrix M.
            YY    = M@L                                                             # M*L computation.
            Gamma = np.vstack([-sum(YY), YY]).transpose()                           # Gamma values are found.
            K[i,i] = Gamma[0,0]                                                     # The corresponding Gamma for the central node.
            for j in np.arange(nvec):                                               # For each of the neighbor nodes.
                K[i, vec[i,j]] = Gamma[0,j+1]                                       # The corresponding Gamma for the neighbor node.
            
        if p[i,2] == 1 or p[i,2] == 2:                                              # If the node is in the boundary.
            for j in np.arange(nvec):                                               # For each of the neighbor nodes.
                K[i, :] = 0                                                         # Neighbor node weight is equal to 0.
                K[i, i] = 1                                                         # Central node weight is equal to 1.

    return K

def Cloud_2(p, vec, L, phi, f):
    # Variable initialization
    nvec  = len(vec[0,:])                                                           # The maximum number of neighbors.
    m     = len(p[:,0])                                                             # The total number of nodes.
    K     = np.zeros([m,m])                                                         # K initialization with zeros.
    R     = np.zeros([m])                                                           # K initialization with zeros.
    
    # Gammas computation and Matrix assembly
    for i in np.arange(m):                                                          # For each of the nodes.
        if p[i,2] == 0:                                                             # If the node is an inner node.
            nvec = sum(vec[i,:] != -1)                                              # The total number of neighbors of the node.
            dx   = np.zeros([nvec])                                                 # dx initialization with zeros.
            dy   = np.zeros([nvec])                                                 # dy initialization with zeros.
            for j in np.arange(nvec):                                               # For each of the neighbor nodes.
                vec1  = int(vec[i, j])                                              # The neighbor index is found.
                dx[j] = p[vec1, 0] - p[i,0]                                         # dx is computed.
                dy[j] = p[vec1, 1] - p[i,1]                                         # dy is computed.
            M     = np.vstack([[dx], [dy], [dx**2], [dx*dy], [dy**2]])              # M matrix is assembled.
            M     = np.linalg.pinv(M)                                               # The pseudoinverse of matrix M.
            YY    = M@L                                                             # M*L computation.
            Gamma = np.vstack([-sum(YY), YY]).transpose()                           # Gamma values are found.
            K[i,i] = Gamma[0,0]                                                     # The corresponding Gamma for the central node.
            for j in np.arange(nvec):                                               # For each of the neighbor nodes.
                K[i, vec[i,j]] = Gamma[0,j+1]                                       # The corresponding Gamma for the neighbor node.
            
        if p[i,2] == 1 or p[i,2] == 2:                                              # If the node is in the boundary.
            for j in np.arange(nvec):                                               # For each of the neighbor nodes.
                K[i, :] = 0                                                         # Neighbor node weight is equal to 0.
                K[i, i] = 1                                                         # Central node weight is equal to 1.
    
    # R computation
    for i in np.arange(m):                                                          # For each of the nodes.
        if p[i,2] == 0:                                                             # If the node is an inner node.
            R[i] += f(p[i, 0], p[i, 1])                                             # The value is the value of the function.
        if p[i,2] == 1 or p[i,2] == 2:                                              # If the node is in the boundary.
            R   -= K[:,i]*phi(p[i,0], p[i,1])                                       # R has the contributions of all the neighbors.
            R[i] += phi(p[i,0], p[i,1])                                             # The value of the function is substracted from the corresponding R.

    return K, R