"""
=========================================================================================
Classical Finite Difference Schemes to Solve the Diffusion Equation
=========================================================================================

This module provides numerical solvers for the 1D and 2D Diffusion Equation
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

def Diffusion1D(x, t, u, v):
    """
    Transient 1D diffusion using explicit Euler and a discrete Laplacian.

    Parameters
    ----------
    x       : numpy.ndarray of shape (m,)
                1D uniform spatial mesh including boundary nodes.
    t : int
        Number of time steps.
    u : Callable
        Exact solution / boundary condition function (signature: u(x, t, v)).
                Must accept NumPy arrays and be vectorizable over x and t.
    v : float
        Positive diffusion coefficient.

    Returns
    -------
    u_ap    : numpy.ndarray of shape (m, t)
                Approximate solution computed with the explicit method.
    u_ex    : numpy.ndarray of shape (m, t)
                Exact solution evaluated for comparison.

    Details
    -------
    - Integration: u^{k+1} = u^{k} + v*dt*(A @ u^{k}), where A is the
      tridiagonal 1D Laplacian and dt = T[1] - T[0].
    - Dirichlet boundaries are reapplied at every time k+1.
    - Explicit stability recommendation: v*dt/dx^2 <= 1/2.
    """
    # Variables initialization
    m    = len(x)                                                                               # Number of spatial nodes.
    dx   = x[1] - x[0]                                                                          # Spatial step size (uniform grid).
    T    = np.linspace(0, 1, t)                                                                 # Time grid over [0, 1].
    dt   = T[1] - T[0]                                                                          # Time step size.
    r    = v*dt/(dx**2)                                                                         # r has all the coefficients of the method.
    u_ap = np.empty([m, t])                                                                     # Allocate approximate solution values assigned below.

    # Initial condition
    u_ap[:, 0]  = u(x, T[0], v)                                                                 # Apply initial state at t = 0.

    # Boundary conditions
    u_ap[0,  :] = u(x[0],  T, v)                                                                # Boundary at x = 0 over all k.
    u_ap[-1, :] = u(x[-1], T, v)                                                                # Boundary at x = 1 over all k.
    
    # Operator matrix
    A = A_1D_calc(m, r)                                                                         # 1D Laplacian tridiagonal operator.

    # Explicit time integration with Dirichlet boundaries
    for k in range(t-1):                                                                        # Loop over time steps.
        u_new           = u_ap[:, k] + A@u_ap[:, k]                                             # Compute new approximation from current state.
        u_ap[1:-1, k+1] = u_new[1:-1]                                                           # The approximation is saved for interior nodes.

    # Exact solution for comparison
    X, TT = np.meshgrid(x, T, indexing='ij')                                                    # Create 2D grid for evaluation.
    u_ex  = u(X, TT, v)                                                                         # Evaluate exact solution on the grid.

    return u_ap, u_ex                                                                           # Return approximate and exact.

def Diffusion1D_iter(x, t, u, v):
    """
    Transient 1D diffusion using a Gauss–Seidel-style explicit update.

    Parameters
    ----------
    x       : numpy.ndarray of shape (m,)
                1D uniform spatial mesh including boundary nodes.
    t : int
        Number of time steps.
    u : Callable
        Exact solution / boundary condition function (signature: u(x, t, v)).
    v : float
        Positive diffusion coefficient.

    Returns
    -------
    u_ap    : numpy.ndarray of shape (m, t)
                Approximate solution computed by the iterative stencil update.
    u_ex    : numpy.ndarray of shape (m, t)
                Exact solution evaluated for comparison.

    Details
    -------
    - Updates each interior node with the 1D stencil [1, -2, 1].
    - Dirichlet boundaries are enforced across the entire time horizon.
    - Serves as a baseline to compare against the explicit A @ u method.
    """
    # Variables initialization
    m    = len(x)                                                                               # Number of spatial nodes.
    dx   = x[1] - x[0]                                                                          # Spatial step size (uniform grid).
    T    = np.linspace(0, 1, t)                                                                 # Time grid over [0, 1].
    dt   = T[1] - T[0]                                                                          # Time step size.
    r    = v*dt/(dx**2)                                                                         # r has all the coefficients of the method.
    u_ap = np.empty([m, t])                                                                     # Allocate approximate solution values assigned below.

    # Initial condition
    u_ap[:, 0] = u(x, T[0], v)                                                                  # Apply initial state at t = 0.

    # Dirichlet boundaries (enforced for all times)
    u_ap[0,  :] = u(x[0],  T, v)                                                                # Boundary at x = 0 over all k.
    u_ap[-1, :] = u(x[-1], T, v)                                                                # Boundary at x = 1 over all k.

    # Iterative explicit stencil update
    for k in range(1, t):                                                                       # Time steps from 1 to t-1.
        for i in range(1, m-1):                                                                 # Interior spatial indices.
            u_ap[i, k] = u_ap[i, k-1] + r*(                                                     # Explicit Euler + stencil.
                u_ap[i+1, k-1] - 2*u_ap[i, k-1] + u_ap[i-1, k-1]                                # 1D second derivative approximation.
            )
    
    # Exact solution for comparison
    X, TT = np.meshgrid(x, T, indexing='ij')                                                    # Create 2D grid for evaluation.
    u_ex  = u(X, TT, v)                                                                         # Evaluate exact solution on the grid.

    return u_ap, u_ex                                                                           # Return approximate and exact.

def Diffusion2D(x, y, t, u, v, implicit=False, lam=0.5):
    """
    Transient 2D diffusion using explicit Euler and a 2D Laplacian via
    Kronecker products. Optional implicit update controlled by 'implicit' and 'lam'.

    Parameters
    ----------
    x, y        : numpy.ndarray of shape (m, n)
                    2D spatial meshes produced by meshgrid (include boundaries).
    t : int
        Number of time steps.
    u : Callable
        Function with signature u(x, y, t, v), vectorizable over x, y, and t.
    v : float
        Positive diffusion coefficient.
    implicit : bool
        If True, use an implicit/Crank–Nicolson-like update. Default False.
    lam : float
        Parameter in [0, 1] controlling implicitness when implicit=True. Default 0.5.

    Returns
    -------
    u_ap    : numpy.ndarray of shape (m, n, t)
                Approximate solution computed by the chosen time scheme.
    u_ex    : numpy.ndarray of shape (m, n, t)
                Exact solution evaluated for comparison.

    Details
    -------
    - Operator: L = kron(I, T1D) + kron(T1D, I), with tridiagonal T1D [-2, 1, 1];
      scaled by 1/h^2.
    - Explicit integration: u^{k+1} = u^{k} + v*dt*L(u^{k}).
    - Dirichlet boundaries are reimposed at every time k+1 on all four edges.
    - Explicit stability recommendation: v*dt/h^2 <= 1/4 for isotropic meshes.
    """
    # Variables initialization
    m, n = x.shape                                                                              # Number of mesh rows and columns.
    dx   = x[0, 1] - x[0, 0]                                                                    # Uniform column spacing in the x direction.
    dy   = y[1, 0] - y[0, 0]                                                                    # Uniform row spacing in the y direction.
    T    = np.linspace(0, 1, t)                                                                 # Time grid over [0, 1].
    dt   = T[1] - T[0]                                                                          # Time step size.
    r_x  = v*dt/(dx**2)                                                                         # Diffusive CFL coefficient in the x direction.
    r_y  = v*dt/(dy**2)                                                                         # Diffusive CFL coefficient in the y direction.
    u_ap = np.empty([m, n, t])                                                                  # Allocate approximate solution values assigned below.

    # Initial condition
    u_ap[:, :, 0] = u(x, y, T[0], v)                                                            # Apply initial state at t = 0.

    # Dirichlet boundaries
    for k in np.arange(t):                                                                      # Loop over time steps.
        u_ap[0,  :, k] = u(x[0, :],  y[0, :],  T[k], v)                                         # Bottom boundary at the first y-row.
        u_ap[-1, :, k] = u(x[-1, :], y[-1, :], T[k], v)                                         # Top boundary at the last y-row.
        u_ap[:,  0, k] = u(x[:, 0],  y[:, 0],  T[k], v)                                         # Left boundary at the first x-column.
        u_ap[:, -1, k] = u(x[:, -1], y[:, -1], T[k], v)                                         # Right boundary at the last x-column.

    # Discrete operator scaled by v*dt
    A = A_2D_calc(m, n, r_x, r_y)                                                               # 2D Laplacian operator.

    # Time-stepping matrix (explicit or implicit)
    if implicit is False:                                                                       # Explicit scheme.
        K2 = np.identity(m*n) + A                                                               # Forward Euler step: (I + A).
    else:                                                                                       # Implicit/Crank–Nicolson-like scheme.
        K2 = np.linalg.solve(                                                                   # Two-parameter implicit update.
            np.identity(m*n) - (1 - lam)*A,                                                     # Left-hand operator for the implicit weight.
            np.identity(m*n) + lam*A                                                            # Right-hand operator for the explicit weight.
        )

    # Time integration on interior nodes
    for k in range(t-1):                                                                        # Loop over time steps.
        u_old                 = u_ap[:, :, k].reshape(m*n)                                      # Flatten current state (row-major).
        u_new                 = np.einsum('ij,j->i', K2, u_old)                                 # Apply time-stepping matrix.
        urr                   = u_new.reshape(m, n)                                             # Reshape back to 2D.
        u_ap[1:-1, 1:-1, k+1] = urr[1:-1, 1:-1]                                                 # Update interior (keep boundaries).
        
    # Exact solution for comparison
    u_ex = u(x[:, :, None], y[:, :, None], T[None, None, :], v)                                 # Evaluate exact solution over the full space-time mesh.

    return u_ap, u_ex                                                                           # Return approximate and exact.

def Diffusion2D_iter(x, y, t, u, v, implicit=False, lam=0.5):
    """
    Transient 2D diffusion via a Gauss–Seidel-style iterative scheme
    with the 5-point stencil on interior nodes.

    Parameters
    ----------
    x, y : numpy.ndarray of shape (m, n)
        2D spatial meshes produced by meshgrid (include boundaries).
    t : int
        Number of time steps.
    u : Callable
        Function with signature u(x, y, t, v), vectorizable over x, y, and t.
    v : float
        Positive diffusion coefficient.

    Returns
    -------
    u_ap : numpy.ndarray of shape (m, n, t)
        Approximate solution computed by iterative stencil updates.
    u_ex : numpy.ndarray of shape (m, n, t)
        Exact solution evaluated for comparison.

    Details
    -------
    - Interior nodes are updated by summing contributions in x and y (5-point stencil)
      over the previous state.
    - Dirichlet boundaries are enforced at all four edges across time.
    - Explicit stability recommendation for isotropic mesh (dx=dy=h): v*dt/h^2 <= 1/4.
    """
    # Variables initialization
    m, n = x.shape                                                                              # Number of mesh rows and columns.
    dx   = x[0, 1] - x[0, 0]                                                                    # Uniform column spacing in the x direction.
    dy   = y[1, 0] - y[0, 0]                                                                    # Uniform row spacing in the y direction.
    T    = np.linspace(0, 1, t)                                                                 # Time grid over [0, 1].
    dt   = T[1] - T[0]                                                                          # Time step size.
    r_x  = v*dt/(dx**2)                                                                         # r_x has all the coefficients of the method.
    r_y  = v*dt/(dy**2)                                                                         # r_y has all the coefficients of the method.
    u_ap = np.empty([m, n, t])                                                                  # Allocate approximate solution values assigned below.
    
    # Initial condition
    u_ap[:, :, 0] = u(x, y, T[0], v)                                                            # Apply initial state at t = 0.

    # Dirichlet boundaries
    for k in range(t):                                                                          # Loop over time steps.
        u_ap[:, 0,  k] = u(x[:, 0],  y[:, 0],  T[k], v)                                         # Left boundary at the first x-column.
        u_ap[:, -1, k] = u(x[:, -1], y[:, -1], T[k], v)                                         # Right boundary at the last x-column.
        u_ap[0, :,  k] = u(x[0, :],  y[0, :],  T[k], v)                                         # Bottom boundary at the first y-row.
        u_ap[-1, :, k] = u(x[-1, :], y[-1, :], T[k], v)                                         # Top boundary at the last y-row.

    # Explicit stencil update on interior nodes
    for k in range(t-1):                                                                        # Time steps from 0 to t-2.
        for i in range(1, m-1):                                                                 # Interior y indices.
            for j in range(1, n-1):                                                             # Interior x indices.
                u_ap[i, j, k+1] = (                                                             # Start from previous state and add contributions.
                    u_ap[i, j, k]                                                               # Previous value at node (i, j).
                    + r_x*(                                                                     # Contribution along x.
                        u_ap[i, j+1, k] - 2*u_ap[i, j, k] + u_ap[i, j-1, k]                     # Centered second difference in x.
                    )
                    + r_y*(                                                                     # Contribution along y.
                        u_ap[i+1, j, k] - 2*u_ap[i, j, k] + u_ap[i-1, j, k]                     # Centered second difference in y.
                    )
                )                                                                               # 5-point stencil sum for 2D diffusion.
    
    # Exact solution for comparison
    u_ex = u(x[:, :, None], y[:, :, None], T[None, None, :], v)                                 # Evaluate exact solution over the full space-time mesh.

    return u_ap, u_ex                                                                           # Return approximate and exact.

def A_1D_calc(m, r):
    """
    Construct the 1D explicit update operator for diffusion using the
    centered Finite Difference stencil multiplied by r = v*dt/dx².

    Parameters
    ----------
    m : int
        Total number of nodes in the 1D mesh (includes boundaries).
    r : float
        Explicit update coefficient r = v*dt/dx².

    Returns
    -------
    A : np.ndarray
        Matrix with:
        - Boundary rows: identity to keep Dirichlet values fixed.
        - Interior rows: r * [1, -2, 1] tridiagonal stencil.
    """
    # Matrix initialization
    A = np.zeros([m, m])                                                                        # Initialize A with zeros.

    # Interior stencil
    for i in range(1, m-1):                                                                     # Traverse interior nodes.
        A[i, i-1] = r                                                                           # Left neighbor contribution.
        A[i, i]   = -2*r                                                                        # Central node contribution.
        A[i, i+1] = r                                                                           # Right neighbor contribution.
    
    return A

def A_2D_calc(m, n, r_x, r_y):
    """
    Construct the 2D Diffusion operator matrix over an m×n mesh.

    Parameters
    ----------
    m, n : int
        Total number of nodes per direction in the 2D mesh (includes boundaries).
    r_x, r_y : float
        Diffusive coefficients in x and y directions respectively.

    Returns
    -------
    A   : numpy.ndarray of shape (m*n, m*n)
        Matrix representing the discrete operator with explicitly applied
        Dirichlet boundaries (zeros) for interior stability calculations.
    """
    # Matrix initialization
    A    = np.zeros([m*n, m*n])                                                                 # Initialize A with zeros.
    
    # Fill entries
    for i in range(m):                                                                          # Traverse mesh rows.
        for j in range(n):                                                                      # Traverse mesh columns.
            k = i * n + j                                                                       # Map node (i, j) to row-major vector index.
            if not (i == 0 or i == m-1 or j == 0 or j == n-1):                                  # Interior node.
                A[k, k]   = -2*r_x - 2*r_y                                                      # Central node contribution.
                A[k, k-1] = r_x                                                                 # Left neighbor contribution.
                A[k, k+1] = r_x                                                                 # Right neighbor contribution.
                A[k, k-n] = r_y                                                                 # Bottom neighbor contribution.
                A[k, k+n] = r_y                                                                 # Top neighbor contribution.
    
    return A

if __name__ == '__main__':
    print("This module defines transient diffusion solvers. Run Examples/CFDM_Diffusion_examples.py for examples.")
                                                                                                # Inform users that examples live outside this module.
