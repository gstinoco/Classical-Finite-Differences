"""
=========================================================================================
Classical Finite Difference Schemes to Solve the Advection Equation
=========================================================================================

This module provides numerical solvers for the 1D and 2D Advection Equation
using Classical Finite Difference methods.

Author
------
Dr. Gerardo Tinoco Guerrero
- Universidad Michoacana de San Nicolás de Hidalgo
- gerardo.tinoco@umich.mx

Funding & Support
-----------------
This project is made possible through the generous support of:
- Secretariat of Science, Humanities, Technology and Innovation, SeCiHTI 
  (Secretaría de Ciencia, Humanidades, Tecnología e Innovación, SeCiHTI). México.
- Coordination of Scientific Research of the Universidad Michoacana de San Nicolás 
  de Hidalgo, CIC-UMSNH (Coordinación de la Investigación Científica de la 
  Universidad Michoacana de San Nicolás de Hidalgo, CIC-UMSNH). México.
- SIIIA MATH: Soluciones en Ingeniería.
- Aula CIMNE-Morelia. México.

Revision History
----------------
- Initial Release: October, 2022.
- Last Update: July, 2026.
=========================================================================================
"""
import numpy as np                                                                              # Numerical computing and array structures.

def Advection1D(x, t, u, a, method='FTCS'):
    """
    Transient 1D linear advection using explicit finite-difference schemes.
    
    PDE
        u_t + a * u_x = 0 on x in [a, b], t in [0, 1].
    
    Parameters
        x : numpy.ndarray (m,)
            Uniform 1D spatial mesh including boundary nodes.
        t : int
        Number of time steps.
        u : Callable
        Exact/initial function with signature u(x, t, a), vectorizable in x and t.
        a : float
        Advection speed (can be positive or negative).
        method : {'FTCS','FTBS','FTFS','LaxWendroff'}
            Time-stepping scheme for advection.
    
    Returns
        u_ap : numpy.ndarray (m, t)
            Approximate solution over time computed by the chosen scheme.
        u_ex : numpy.ndarray (m, t)
            Exact solution evaluated for comparison.
    
    Notes
        - Boundary values are reimposed from the exact function at all times (Dirichlet).
        - FTCS is unstable for pure advection and is included for didactic comparison.
        - Upwind choice: FTBS is appropriate for a > 0, FTFS for a < 0.
        - Stability recommendations:
            * FTBS/FTFS: |c| = |a*dt/dx| ≤ 1
            * Lax–Wendroff: second-order, requires |a*dt/dx| ≤ 1
    """
    # Variables initialization
    T    = np.linspace(0, 1, t)                                                                 # Time grid over [0, 1].
    m    = len(x)                                                                               # Number of spatial nodes.
    dx   = x[1] - x[0]                                                                          # Spatial step size (uniform grid).
    dt   = T[1] - T[0]                                                                          # Time step size.
    u_ap = np.zeros([m, t])                                                                     # Approximate solution container.
    u_ex = np.zeros([m, t])                                                                     # Exact solution container.
    
    # Initial condition
    u_ap[:, 0] = u(x, T[0], a)                                                                  # Apply initial state at t = 0.
    
    # Boundary conditions
    u_ap[0,  :] = u(x[0],  T, a)                                                                # Boundary at x = 0 over all k.
    u_ap[-1, :] = u(x[-1], T, a)                                                                # Boundary at x = 1 over all k.
    
    # Operator matrix
    A      = A_1D_calc(m, a, dt, dx, method)                                                    # 1D advection stencil operator.

    # Explicit time integration
    for k in range(t-1):                                                                        # Loop over time steps.
        u_new = u_ap[:, k] + np.einsum('ij,j->i', A, u_ap[:, k])                                # Compute new approximation from current state.
        if method == 'FTCS':
            u_ap[1:-1, k+1] = u_new[1:-1]                                                       # Store only interior FTCS values, preserving boundaries.
        elif method == 'FTBS':
            u_ap[1:-1, k+1] = u_new[1:-1]                                                       # Store only interior FTBS values, preserving boundaries.
        elif method == 'FTFS':
            u_ap[1:-1, k+1] = u_new[1:-1]                                                       # Store only interior FTFS values, preserving boundaries.
        elif method == 'LaxWendroff':
            u_ap[1:-1, k+1] = u_new[1:-1]                                                       # Store only interior Lax-Wendroff values, preserving boundaries.
        else:
            raise ValueError("Method not implemented.")

    # Exact solution for comparison
    X, TT = np.meshgrid(x, T, indexing='ij')                                                    # Create 2D grid for evaluation.
    u_ex  = u(X, TT, a)                                                                         # Evaluate exact solution on the grid.

    return u_ap, u_ex                                                                           # Return approximate and exact.

def Advection1D_iter(x, t, u, a, method='FTCS'):
    """
    Transient 1D linear advection via explicit iterative updates (node-wise).
    
    Parameters
        x : numpy.ndarray (m,)
            Uniform 1D spatial mesh including boundary nodes.
        t : int
        Number of time steps.
        u : Callable
        Exact/initial function with signature u(x, t, a).
        a : float
        Advection speed (can be positive or negative).
        method : {'FTCS','FTBS','FTFS','LaxWendroff'}
            Explicit stencil used for the node-wise update.
    
    Returns
        u_ap : numpy.ndarray (m, t)
            Approximate solution computed by iterative stencil updates.
        u_ex : numpy.ndarray (m, t)
            Exact solution evaluated for comparison.
    
    Notes
        - Dirichlet boundary values are enforced for all times at both ends.
        - FTCS is unstable for advection; FTBS/FTFS are upwind first-order schemes.
        - Lax–Wendroff achieves second-order accuracy when |a*dt/dx| ≤ 1.
    """
    # Variables initialization
    T    = np.linspace(0, 1, t)                                                                 # Time grid over [0, 1].
    m    = len(x)                                                                               # Number of spatial nodes.
    dx   = x[1] - x[0]                                                                          # Spatial step size (uniform grid).
    dt   = T[1] - T[0]                                                                          # Time step size.
    u_ap = np.zeros([m, t])                                                                     # Approximate solution container.
    u_ex = np.zeros([m, t])                                                                     # Exact solution container.

    # Initial condition
    u_ap[:, 0] = u(x, T[0], a)                                                                  # Apply initial state at t = 0.

    # Dirichlet boundaries (enforced for all times)
    u_ap[0,  :] = u(x[0],  T, a)                                                                # Boundary at x = 0 over all k.
    u_ap[-1, :] = u(x[-1], T, a)                                                                # Boundary at x = 1 over all k.

    # Iterative explicit stencil update
    for k in range(t-1):                                                                        # Time steps from 1 to t-1.
        if method == 'FTCS':
            r    = a*dt/(2*dx)                                                                  # r is defined as the CFL coefficient.
            for i in range(1, m-1):                                                             # For all the inner nodes.
                u_ap[i, k+1] = u_ap[i, k] - r*(u_ap[i+1, k] - u_ap[i-1, k])                     # The new approximation is performed.
        elif method == 'FTBS':
            r    = a*dt/dx                                                                      # r is defined as the CFL coefficient.
            for i in range(1, m-1):                                                             # For all the inner nodes.
                u_ap[i, k+1] = u_ap[i, k] - r*(u_ap[i, k] - u_ap[i-1, k])                       # The new approximation is performed.
        elif method == 'FTFS':
            r    = a*dt/dx                                                                      # r is defined as the CFL coefficient.
            for i in range(1, m-1):                                                             # For all the inner nodes.
                u_ap[i, k+1] = u_ap[i, k] - r*(u_ap[i+1, k] - u_ap[i, k])                       # The new approximation is performed.
        elif method == 'LaxWendroff':
            r    = a*dt/dx                                                                      # r is defined as the CFL coefficient.
            for i in range(1, m-1):                                                             # For all the inner nodes.
                u_ap[i, k+1] = (                                                                # Lax-Wendroff update at the current interior node.
                    u_ap[i, k]                                                                  # Previous value at node i.
                    - (r/2)*(u_ap[i+1, k] - u_ap[i-1, k])                                       # Centered first-derivative contribution.
                    + (r**2/2)*(u_ap[i-1, k] - 2*u_ap[i, k] + u_ap[i+1, k])                     # Second-derivative correction.
                )
        else:
            raise ValueError('Method not implemented.')
        
    # Exact solution for comparison
    X, TT = np.meshgrid(x, T, indexing='ij')                                                    # Create 2D grid for evaluation.
    u_ex  = u(X, TT, a)                                                                         # Evaluate exact solution on the grid.

    return u_ap, u_ex                                                                           # Return approximate and exact.

def Advection2D(x, y, t, u, a, b, method='FTCS'):
    """
    Transient 2D linear advection using explicit finite-difference schemes.
    
    PDE
        u_t + a * u_x + b * u_y = 0 on (x, y) in [a, b]^2, t in [0, 1].
    
    Parameters
        x, y : numpy.ndarray (m, m)
            2D meshes produced by meshgrid (include boundary nodes).
        t : int
        Number of time steps.
        u : Callable
        Exact/initial function with signature u(x, y, t, a, b), vectorizable.
        a, b : float
        Advection speeds along x and y (can be positive or negative).
        method : {'FTCS','FTBS','FTFS','LaxWendroff'}
            Time-stepping scheme for 2D advection.
    
    Returns
        u_ap : numpy.ndarray (m, m, t)
            Approximate solution over time computed by the chosen scheme.
        u_ex : numpy.ndarray (m, m, t)
            Exact solution evaluated for comparison.
    
    Notes
        - Dirichlet boundary values are reimposed at each time step on all four edges.
        - FTCS is unstable for pure advection; provided for comparison.
        - Stability recommendations (typical):
            * Upwind first-order (separable): |c_x| ≤ 1, |c_y| ≤ 1
            * Combined CFL (conservative): |c_x| + |c_y| ≤ 1
            * Lax–Wendroff: second-order; requires |c_x| ≤ 1 and |c_y| ≤ 1
        where c_x = a*dt/dx and c_y = b*dt/dy.
    """
    # Variable initialization
    T    = np.linspace(0, 1, t)                                                                 # Time grid over [0, 1].
    m, n = x.shape                                                                              # Size of the mesh.
    dx   = x[0, 1] - x[0, 0]                                                                    # dx is defined as the space step-length in x.
    dy   = y[1, 0] - y[0, 0]                                                                    # dy is defined as the space step-length in y.
    dt   = T[1] - T[0]                                                                          # Time step size.

    # Solution initialization.
    u_ap = np.zeros([m, n, t])                                                                  # u_ap is initialized with zeros.
    u_ex = np.zeros([m, n, t])                                                                  # u_ex is initialized with zeros.

    # Initial condition
    u_ap[:, :, 0] = u(x, y, T[0], a, b)                                                         # Apply initial state at t = 0.
    
    # Dirichlet boundaries
    for k in range(t):                                                                          # Loop over time steps.
        u_ap[:, 0,  k] = u(x[:, 0],  y[:, 0],  T[k], a, b)                                      # Left boundary at the first x-column.
        u_ap[:, -1, k] = u(x[:, -1], y[:, -1], T[k], a, b)                                      # Right boundary at the last x-column.
        u_ap[0, :,  k] = u(x[0, :],  y[0, :],  T[k], a, b)                                      # Bottom boundary at the first y-row.
        u_ap[-1, :, k] = u(x[-1, :], y[-1, :], T[k], a, b)                                      # Top boundary at the last y-row.
    
    # Discrete operator
    A = A_2D_calc(m, n, a, b, dt, dx, dy, method)                                               # 2D advection operator.
    
    # Finite Differences Approximation
    for k in range(t-1):                                                                        # For all the time-steps.
        u_flat = u_ap[:, :, k].reshape(m*n)                                                     # Flatten current mesh values in row-major order.
        u_new  = u_ap[:, :, k] + np.einsum('ij,j->i', A, u_flat).reshape(m, n)                  # Apply matrix update and reshape to the mesh.
        if method == 'FTCS':
            u_ap[1:-1, 1:-1, k+1] = u_new[1:-1, 1:-1]                                           # Store interior FTCS values, preserving all edges.
        elif method == 'FTBS':
            u_ap[1:-1, 1:-1, k+1] = u_new[1:-1, 1:-1]                                           # Store interior FTBS values, preserving all edges.
        elif method == 'FTFS':
            u_ap[1:-1, 1:-1, k+1] = u_new[1:-1, 1:-1]                                           # Store interior FTFS values, preserving all edges.
        elif method == 'LaxWendroff':
            u_ap[1:-1, 1:-1, k+1] = u_new[1:-1, 1:-1]                                           # Store interior Lax-Wendroff values, preserving all edges.
        else:
            raise ValueError('Method not implemented.') 
    
    # Exact solution for comparison
    for k in range(t):                                                                          # Evaluate at all time steps.
        u_ex[:, :, k] = u(x, y, T[k], a, b)                                                     # Exact solution.
            
    return u_ap, u_ex                                                                           # Return the computed solution.

def Advection_2D_iter(x, y, t, u, a, b, method='FTCS'):
    """
    Transient 2D linear advection via explicit iterative node-wise updates.
    
    Parameters
        x, y : numpy.ndarray (m, m)
            2D meshes produced by meshgrid (include boundary nodes).
        t : int
        Number of time steps.
        u : Callable
        Exact/initial function with signature u(x, y, t, a, b).
        a, b : float
        Advection speeds along x and y (can be positive or negative).
        method : {'FTCS','FTBS','FTFS','LaxWendroff'}
            Explicit stencil used for the node-wise update.
    
    Returns
        u_ap : numpy.ndarray (m, m, t)
            Approximate solution computed by iterative stencil updates.
        u_ex : numpy.ndarray (m, m, t)
            Exact solution evaluated for comparison.
    
    Notes
        - Dirichlet boundary values are enforced along all four edges for all times.
        - Upwind variants (FTBS/FTFS) update interior nodes only.
        - Suggested CFL conditions:
            * FTBS/FTFS: |c_x| ≤ 1, |c_y| ≤ 1 (often |c_x| + |c_y| ≤ 1)
            * Lax–Wendroff: second-order; |c_x| ≤ 1 and |c_y| ≤ 1
    """
    T    = np.linspace(0, 1, t)                                                                 # Time grid over [0, 1].
    m, n = x.shape                                                                              # Size of the mesh.
    dx   = x[0, 1] - x[0, 0]                                                                    # dx is defined as the space step-length in x.
    dy   = y[1, 0] - y[0, 0]                                                                    # dy is defined as the space step-length in y.
    dt   = T[1] - T[0]                                                                          # Time step size.

    # Solution initialization.
    u_ap = np.zeros([m, n, t])                                                                  # u_ap is initialized with zeros.
    u_ex = np.zeros([m, n, t])                                                                  # u_ex is initialized with zeros.

    # Initial condition
    u_ap[:, :, 0] = u(x, y, T[0], a, b)                                                         # Apply initial state at t = 0.
    
    # Dirichlet boundaries
    for k in range(t):                                                                          # Loop over time steps.
        u_ap[:, 0,  k] = u(x[:, 0],  y[:, 0],  T[k], a, b)                                      # Left boundary at the first x-column.
        u_ap[:, -1, k] = u(x[:, -1], y[:, -1], T[k], a, b)                                      # Right boundary at the last x-column.
        u_ap[0, :,  k] = u(x[0, :],  y[0, :],  T[k], a, b)                                      # Bottom boundary at the first y-row.
        u_ap[-1, :, k] = u(x[-1, :], y[-1, :], T[k], a, b)                                      # Top boundary at the last y-row.
    
    # Finite Differences Method
    for k in range(t-1):                                                                        # For all the time-steps.
        if method == 'FTCS':
            r_x  = a*dt/(2*dx)                                                                  # r_x is defined as the CFL coefficient.
            r_y  = b*dt/(2*dy)                                                                  # r_y is defined as the CFL coefficient.
            for i in range(1, m-1):                                                             # For all the inner nodes in y.
                for j in range(1, n-1):                                                         # For all the inner nodes in x.
                    u_ap[i, j, k+1] = (                                                         # FTCS update at interior node (i, j).
                        u_ap[i, j, k]                                                           # Previous value at the node.
                        - r_x*(u_ap[i, j+1, k] - u_ap[i, j-1, k])                               # Centered x-advection contribution.
                        - r_y*(u_ap[i+1, j, k] - u_ap[i-1, j, k])                               # Centered y-advection contribution.
                    )
        elif method == 'FTBS':
            r_x  = a*dt/dx                                                                      # r_x is defined as the CFL coefficient.
            r_y  = b*dt/dy                                                                      # r_y is defined as the CFL coefficient.
            for i in range(1, m-1):                                                             # Interior nodes in y.
                for j in range(1, n-1):                                                         # Interior nodes in x.
                    u_ap[i, j, k+1] = (                                                         # FTBS upwind update at interior node (i, j).
                        u_ap[i, j, k]                                                           # Previous value at the node.
                        - r_x*(u_ap[i, j, k] - u_ap[i, j-1, k])                                 # Backward x-difference contribution.
                        - r_y*(u_ap[i, j, k] - u_ap[i-1, j, k])                                 # Backward y-difference contribution.
                    )
        elif method == 'FTFS':
            r_x  = a*dt/dx                                                                      # r_x is defined as the CFL coefficient.
            r_y  = b*dt/dy                                                                      # r_y is defined as the CFL coefficient.
            for i in range(1, m-1):                                                             # Interior nodes in y.
                for j in range(1, n-1):                                                         # Interior nodes in x.
                    u_ap[i, j, k+1] = (                                                         # FTFS upwind update at interior node (i, j).
                        u_ap[i, j, k]                                                           # Previous value at the node.
                        - r_x*(u_ap[i, j+1, k] - u_ap[i, j, k])                                 # Forward x-difference contribution.
                        - r_y*(u_ap[i+1, j, k] - u_ap[i, j, k])                                 # Forward y-difference contribution.
                    )
        elif method == 'LaxWendroff':
            r_x  = a*dt/dx                                                                      # r_x is defined as the CFL coefficient.
            r_y  = b*dt/dy                                                                      # r_y is defined as the CFL coefficient.
            for i in range(1, m-1):                                                             # For all the inner nodes in y.
                for j in range(1, n-1):                                                         # For all the inner nodes in x.
                    u_ap[i, j, k+1] = (                                                         # Lax-Wendroff update at interior node (i, j).
                        u_ap[i, j, k]                                                           # Previous value at the node.
                        - (r_x/2)*(u_ap[i, j+1, k] - u_ap[i, j-1, k])                           # Centered x-advection contribution.
                        - (r_y/2)*(u_ap[i+1, j, k] - u_ap[i-1, j, k])                           # Centered y-advection contribution.
                        + (r_x**2/2)*(u_ap[i, j-1, k] - 2*u_ap[i, j, k] + u_ap[i, j+1, k])      # Lax-Wendroff x correction.
                        + (r_y**2/2)*(u_ap[i-1, j, k] - 2*u_ap[i, j, k] + u_ap[i+1, j, k])      # Lax-Wendroff y correction.
                    )
        else:
            raise ValueError('Method not implemented.')
        
    # Exact solution for comparison
    for k in range(t):                                                                          # Evaluate at all time steps.
        u_ex[:, :, k] = u(x, y, T[k], a, b)                                                     # Exact solution.

    return u_ap, u_ex                                                                           # Return the computed solution.

def A_1D_calc(m, a, dt, dx, method="FTCS"):
    """
    Construct the 1D explicit advection update operator for a chosen scheme.
    
    Parameters
        m : int
        Total number of nodes in the 1D mesh (includes boundaries).
        a : float
        Advection speed.
        dt : float
        Time step size.
        dx : float
        Spatial step size.
        method : {'FTCS','FTBS','FTFS','LaxWendroff'}
            Advection scheme used to assemble the stencil.
    
    Returns
        A : numpy.ndarray (m, m)
            Matrix with:
              - Boundary rows: identity to keep Dirichlet values fixed.
              - Interior rows: scheme-specific advection stencil entries.
    
    Notes
        - FTBS/FTFS implement first-order upwind stencils depending on flow direction.
        - Lax–Wendroff includes second-order correction terms.
    """
    # Matrix initialization
    A = np.zeros([m, m])                                                                        # Initialize A with zeros.

    if method == 'FTCS':
        r = a*dt/(2*dx)                                                                         # r is defined as the CFL coefficient.
                    # Finite Differences Matrix
        for i in range(1, m-1):                                                                 # Loop through the Matrix.
            A[i, i+1] = -r                                                                      # Superior diagonal.
            A[i, i-1] = r                                                                       # Inferior diagonal.
        A[0, 0] = A[-1, -1] = 0                                                                 # Boundary Conditions.
    elif method == 'FTBS':
        r    = a*dt/dx                                                                          # r is defined as the CFL coefficient.
                    # Finite Differences Matrix
        for i in range(1, m):                                                                   # Loop through the Matrix.
            A[i, i]   = -r                                                                      # Main diagonal.
            A[i, i-1] = r                                                                       # Inferior diagonal.
        A[0, 0] = 0                                                                             # Boundary Conditions.
    elif method == 'FTFS':
        r    = a*dt/dx                                                                          # r is defined as the CFL coefficient.
                    # Finite Differences Matrix
        for i in range(m-1):                                                                    # Loop through the Matrix.
            A[i, i+1] = -r                                                                      # Superior diagonal.
            A[i, i]   = r                                                                       # Main diagonal.
        A[-1, -1] = 0                                                                           # Boundary Conditions.
    elif method == 'LaxWendroff':
        r    = a*dt/dx                                                                          # r is defined as the CFL coefficient.
                    # Finite Differences Matrix
        for i in range(1, m-1):                                                                 # Loop through the Matrix.
            A[i, i+1] = -(r/2) + (r**2)/2                                                       # Superior diagonal.
            A[i, i]   = -r**2                                                                   # Main diagonal.
            A[i, i-1] = (r/2) + (r**2)/2                                                        # Inferior diagonal.
        A[0, 0] = A[-1, -1] = 0                                                                 # Boundary Conditions.
    else:
        raise ValueError('Method not implemented.')
    return A


def A_2D_calc(m, n, a, b, dt, dx, dy, method="FTCS"):
    """
    Construct the 2D explicit advection update operator for a chosen scheme.
    
    Parameters
        m, n : int
        Nodes per spatial dimension (mesh is m × n).
        a, b : float
        Advection speeds along x and y.
        dt : float
        Time step size.
        dx, dy : float
        Spatial step sizes along x and y.
        method : {'FTCS','FTBS','FTFS','LaxWendroff'}
            Advection scheme used to assemble the stencil.
    
    Returns
        A : numpy.ndarray (m*n, m*n)
            Matrix with:
              - Boundary rows: identity to keep Dirichlet values fixed.
              - Interior rows: advection stencil entries based on flow direction.
    
    Notes
        - Linearization uses k = i*m + j for mapping (i, j) to a single index.
        - Stencil coefficients reflect upwind/central differences per method.
    """
    # Matrix initialization
    A    = np.zeros([m*n, m*n])                                                                 # A is initialized as a (m*n)x(m*n) square matrix.

    if method == 'FTCS':
        r_x = a*dt/(2*dx)                                                                       # r_x is defined as the CFL coefficient.
        r_y = b*dt/(2*dy)                                                                       # r_y is defined as the CFL coefficient.
                    # Finite Differences Matrix
        for i in range(m):                                                                      # For all the nodes in one direction.
            for j in range(n):                                                                  # For all the nodes in the other direction.
                k = i * n + j                                                                   # Linearized Index.
                if i == 0 or i == m - 1 or j == 0 or j == n - 1:                                # If the node is in the boundary.
                    A[k, k] = 0                                                                 # A with zeros to keep the boundary condition in explicit.
                else:                                                                           # If the node is an inner node.
                    A[k, k-1] = r_x                                                             # A matrix value for left node.
                    A[k, k+1] = -r_x                                                            # A matrix value for right node.
                    A[k, k-n] = r_y                                                             # A matrix value for downer node.
                    A[k, k+n] = -r_y                                                            # A matrix value for upper node.
    elif method == 'FTBS':
        r_x = a*dt/dx                                                                           # r_x is defined as the CFL coefficient.
        r_y = b*dt/dy                                                                           # r_y is defined as the CFL coefficient.
                    # Finite Differences Matrix
        for i in range(m):                                                                      # For all the nodes in one direction.
            for j in range(n):                                                                  # For all the nodes in the other direction.
                k = i * n + j                                                                   # Linearized Index.
                if i == 0 or i == m - 1 or j == 0 or j == n - 1:                                # If the node is in the boundary.
                    A[k, k] = 0                                                                 # A with zeros to keep the boundary condition in explicit.
                else:                                                                           # If the node is an inner node.
                    A[k, k]   = -r_x - r_y                                                      # A matrix value for central node.
                    A[k, k-1] = r_x                                                             # A matrix value for left node.
                    A[k, k-n] = r_y                                                             # A matrix value for downer node.
    elif method == 'FTFS':
        r_x = a*dt/dx                                                                           # r_x is defined as the CFL coefficient.
        r_y = b*dt/dy                                                                           # r_y is defined as the CFL coefficient.
                    # Finite Differences Matrix
        for i in range(m):                                                                      # For all the nodes in one direction.
            for j in range(n):                                                                  # For all the nodes in the other direction.
                k = i * n + j                                                                   # Linearized Index.
                if i == 0 or i == m - 1 or j == 0 or j == n - 1:                                # If the node is in the boundary.
                    A[k, k] = 0                                                                 # A with zeros to keep the boundary condition in explicit.
                else:                                                                           # If the node is an inner node.
                    A[k, k]   = r_x + r_y                                                       # A matrix value for central node.
                    A[k, k+1] = -r_x                                                            # A matrix value for right node.
                    A[k, k+n] = -r_y                                                            # A matrix value for upper node.
    elif method == 'LaxWendroff':
        r_x = a*dt/dx                                                                           # r_x is defined as the CFL coefficient.
        r_y = b*dt/dy                                                                           # r_y is defined as the CFL coefficient.
                    # Finite Differences Matrix
        for i in range(m):                                                                      # For all the nodes in one direction.
            for j in range(n):                                                                  # For all the nodes in the other direction.
                k = i * n + j                                                                   # Linearized Index.
                if i == 0 or i == m - 1 or j == 0 or j == n - 1:                                # If the node is in the boundary.
                    A[k, k] = 0                                                                 # A with zeros to keep the boundary condition in explicit.
                else:                                                                           # If the node is an inner node. 
                    A[k, k]   = -r_x**2 - r_y**2                                                # A matrix value for central node.
                    A[k, k-1] = r_x/2 + r_x**2/2                                                # A matrix value for left node.
                    A[k, k+1] = -r_x/2 + r_x**2/2                                               # A matrix value for right node.
                    A[k, k-n] = r_y/2 + r_y**2/2                                                # A matrix value for downer node.
                    A[k, k+n] = -r_y/2 + r_y**2/2                                               # A matrix value for upper node.
    else:
        raise ValueError('Method not implemented.')
    return A


if __name__ == '__main__':
    print("This module defines transient advection solvers. Run Examples/CFDM_Advection_examples.py for examples.")       # Inform users that examples live outside this module.
