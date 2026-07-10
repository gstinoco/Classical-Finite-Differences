"""
=========================================================================================
Classical Finite Difference Schemes to Solve the Advection-Diffusion Equation
=========================================================================================

This module provides numerical solvers for the 1D and 2D Advection-Diffusion Equation
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

def AdvectionDiffusion1D(x, t, u, v, a, implicit=False, lam=0.5):
    """
    Solve the Transient 1D Advection-Diffusion Equation with Dirichlet boundary conditions
    using a Matrix Formulation. Supports Explicit Euler or Implicit (Crank-Nicolson) schemes.

    Parameters
    ----------
    x       : numpy.ndarray of shape (m,)
                1D uniform spatial mesh including boundary nodes.
    t : int
        Number of time steps.
    u : Callable
        Exact solution / boundary condition function (signature: u(x, t, v, a)).
    v : float
        Positive diffusion coefficient.
    a : float
        Advective velocity in the x direction.
    implicit : bool
        If True, use an implicit/Crank–Nicolson-like update. Default False.
    lam : float
        Parameter in [0, 1] controlling implicitness when implicit=True. Default 0.5.

    Returns
    -------
    u_ap    : numpy.ndarray of shape (m, t)
                Approximate solution computed by the chosen time scheme.
    u_ex    : numpy.ndarray of shape (m, t)
                Exact solution evaluated for comparison.
    """
    # Variables initialization
    m    = len(x)                                                                               # Number of spatial nodes.
    dx   = x[1] - x[0]                                                                          # Spatial step size (uniform grid).
    T    = np.linspace(0, 1, t)                                                                 # Time grid over [0, 1].
    dt   = T[1] - T[0]                                                                          # Time step size.
    r_d  = v*dt/(dx**2)                                                                         # r_d has all the diffusive coefficients of the method.
    r_a  = a*dt/(2*dx)                                                                          # r_a has all the advective coefficients of the method.
    u_ap = np.empty([m, t])                                                                     # Allocate approximate solution values assigned below.

    # Initial condition
    u_ap[:, 0]  = u(x, T[0], v, a)                                                              # Apply initial state at t = 0.

    # Dirichlet boundaries
    u_ap[0,  :] = u(x[0],  T, v, a)                                                             # Boundary at x = 0 over all k.
    u_ap[-1, :] = u(x[-1], T, v, a)                                                             # Boundary at x = 1 over all k.
    
    # Operator matrix
    A = A_1D_calc(m, r_d, r_a)                                                                  # 1D Laplacian + Advection tridiagonal operator.

    # Time-stepping matrix (explicit or implicit)
    if implicit is False:                                                                       # Explicit scheme.
        K = np.identity(m) + A                                                                  # Forward Euler step: (I + A).
    else:                                                                                       # Implicit/Crank–Nicolson-like scheme.
        K = np.linalg.solve(                                                                    # Two-parameter implicit update.
            np.identity(m) - (1 - lam)*A,                                                       # Left-hand operator for the implicit weight.
            np.identity(m) + lam*A                                                              # Right-hand operator for the explicit weight.
        )

    # Time integration with Dirichlet boundaries
    for k in range(t-1):                                                                        # Loop over time steps.
        u_new           = np.einsum('ij,j->i', K, u_ap[:, k])                                   # Compute new approximation from current state.
        u_ap[1:-1, k+1] = u_new[1:-1]                                                           # The approximation is saved for interior nodes.

    # Exact solution for comparison
    X, TT = np.meshgrid(x, T, indexing='ij')                                                    # Create 2D grid for evaluation.
    u_ex  = u(X, TT, v, a)                                                                      # Evaluate exact solution on the grid.

    return u_ap, u_ex                                                                           # Return approximate and exact.

def AdvectionDiffusion1D_iter(x, t, u, v, a, implicit=False, lam=0.5):
    """
    Solve the Transient 1D Advection-Diffusion Equation with Dirichlet boundary conditions
    using a node-wise stencil formulation. Supports Explicit or Implicit schemes.

    Parameters
    ----------
    x       : numpy.ndarray of shape (m,)
                1D uniform spatial mesh including boundary nodes.
    t : int
        Number of time steps.
    u : Callable
        Exact solution / boundary condition function (signature: u(x, t, v, a)).
    v : float
        Positive diffusion coefficient.
    a : float
        Advective velocity in the x direction.
    implicit : bool
        If True, use an implicit/Crank–Nicolson-like update. Default False.
    lam : float
        Parameter in [0, 1] controlling implicitness when implicit=True. Default 0.5.

    Returns
    -------
    u_ap    : numpy.ndarray of shape (m, t)
                Approximate solution computed by the iterative stencil update.
    u_ex    : numpy.ndarray of shape (m, t)
                Exact solution evaluated for comparison.
    """
    # Variables initialization
    m    = len(x)                                                                               # Number of spatial nodes.
    dx   = x[1] - x[0]                                                                          # Spatial step size (uniform grid).
    T    = np.linspace(0, 1, t)                                                                 # Time grid over [0, 1].
    dt   = T[1] - T[0]                                                                          # Time step size.
    r_d  = v*dt/(dx**2)                                                                         # Diffusive coefficient.
    r_a  = a*dt/(2*dx)                                                                          # Advective coefficient.
    tol  = np.sqrt(np.finfo(float).eps)                                                         # Tolerance for implicit scheme.
    u_ap = np.zeros([m, t])                                                                     # Store solution values and provide the implicit initial guess.

    # Initial condition
    u_ap[:, 0]  = u(x, T[0], v, a)                                                              # Apply initial state at t = 0.

    # Dirichlet boundaries
    u_ap[0,  :] = u(x[0],  T, v, a)                                                             # Boundary at x = 0 over all k.
    u_ap[-1, :] = u(x[-1], T, v, a)                                                             # Boundary at x = 1 over all k.

    # Iterative stencil update
    for k in range(t-1):                                                                        # Time steps from 0 to t-2.
        if implicit is False:
            for i in range(1, m-1):                                                             # Interior spatial indices.
                u_ap[i, k+1] = u_ap[i, k] + r_d*(                                               # Explicit Euler + stencil.
                    u_ap[i+1, k] - 2*u_ap[i, k] + u_ap[i-1, k]                                  # 1D second derivative approximation.
                ) - r_a*(u_ap[i+1, k] - u_ap[i-1, k])                                           # Centered first derivative for advection.
        else:                                                                                   # Implicit scheme.
            err = 1                                                                             # Initial value to enter the loop.
            while err >= tol:                                                                   # Stop by tolerance.
                err = 0                                                                         # Reset maximum difference per iteration.
                for i in range(1, m-1):                                                         # Interior spatial indices.
                    t_val = (                                                                   # Compute implicit iterative update.
                        u_ap[i, k] +                                                            # Current state contribution.
                        (1-lam) * (                                                             # Explicit-weighted contribution from time level k.
                            r_d * (u_ap[i+1, k] - 2*u_ap[i, k] + u_ap[i-1, k])                  # Diffusion stencil at time level k.
                            - r_a * (u_ap[i+1, k] - u_ap[i-1, k])                               # Advection stencil at time level k.
                        ) +                                                                     # Add the explicit-weighted part to the current state.
                        lam * (                                                                 # Implicit-weighted neighbor contribution at k+1.
                            r_d * (u_ap[i+1, k+1] + u_ap[i-1, k+1])                             # Diffusion neighbors at time level k+1.
                            - r_a * (u_ap[i+1, k+1] - u_ap[i-1, k+1])                           # Advection neighbors at time level k+1.
                        )
                    ) / (1 + 2*lam*r_d)                                                         # Scaling factor for the diagonal term.
                    err = max(err, abs(t_val - u_ap[i, k+1]))                                   # Infinity-norm update.
                    u_ap[i, k+1] = t_val                                                        # Assign new value at interior node.

    # Exact solution for comparison
    X, TT = np.meshgrid(x, T, indexing='ij')                                                    # Create 2D grid for evaluation.
    u_ex  = u(X, TT, v, a)                                                                      # Evaluate exact solution on the grid.

    return u_ap, u_ex                                                                           # Return approximate and exact.

def AdvectionDiffusion2D(x, y, t, u, v, a, b, implicit=False, lam=0.5):
    """
    Solve the Transient 2D Advection-Diffusion Equation with Dirichlet boundary conditions
    using a Matrix Formulation. Supports Explicit Euler or Implicit (Crank-Nicolson) schemes.

    Parameters
    ----------
    x, y        : numpy.ndarray of shape (m, n)
                    2D spatial meshes produced by meshgrid (include boundaries).
    t : int
        Number of time steps.
    u : Callable
        Function with signature u(x, y, t, v, a, b).
    v : float
        Positive diffusion coefficient.
    a : float
        Advective velocity in the x direction.
    b : float
        Advective velocity in the y direction.
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
    """
    # Variables initialization
    m, n = x.shape                                                                              # Number of mesh rows and columns.
    dx   = x[0, 1] - x[0, 0]                                                                    # Uniform column spacing in the x direction.
    dy   = y[1, 0] - y[0, 0]                                                                    # Uniform row spacing in the y direction.
    T    = np.linspace(0, 1, t)                                                                 # Time grid over [0, 1].
    dt   = T[1] - T[0]                                                                          # Time step size.
    r_dx = v*dt/(dx**2)                                                                         # Diffusive coefficient x.
    r_dy = v*dt/(dy**2)                                                                         # Diffusive coefficient y.
    r_a  = a*dt/(2*dx)                                                                          # Advective coefficient x.
    r_b  = b*dt/(2*dy)                                                                          # Advective coefficient y.
    u_ap = np.empty([m, n, t])                                                                  # Allocate approximate solution values assigned below.

    # Initial condition
    u_ap[:, :, 0] = u(x, y, T[0], v, a, b)                                                      # Apply initial state at t = 0.

    # Dirichlet boundaries
    for k in np.arange(t):                                                                      # Loop over time steps.
        u_ap[0,  :, k] = u(x[0, :],  y[0, :],  T[k], v, a, b)                                   # Bottom boundary at the first y-row.
        u_ap[-1, :, k] = u(x[-1, :], y[-1, :], T[k], v, a, b)                                   # Top boundary at the last y-row.
        u_ap[:,  0, k] = u(x[:, 0],  y[:, 0],  T[k], v, a, b)                                   # Left boundary at the first x-column.
        u_ap[:, -1, k] = u(x[:, -1], y[:, -1], T[k], v, a, b)                                   # Right boundary at the last x-column.

    # Discrete operator scaled by dt
    A = A_2D_calc(m, n, r_dx, r_dy, r_a, r_b)                                                   # 2D operator.

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
    u_ex = u(x[:, :, None], y[:, :, None], T[None, None, :], v, a, b)                           # Evaluate exact solution over the full space-time mesh.

    return u_ap, u_ex                                                                           # Return approximate and exact.

def AdvectionDiffusion2D_iter(x, y, t, u, v, a, b, implicit=False, lam=0.5):
    """
    Solve the Transient 2D Advection-Diffusion Equation with Dirichlet boundary conditions
    using a node-wise stencil formulation. Supports Explicit or Implicit schemes.

    Parameters
    ----------
    x, y        : numpy.ndarray of shape (m, n)
                    2D spatial meshes produced by meshgrid (include boundaries).
    t : int
        Number of time steps.
    u : Callable
        Function with signature u(x, y, t, v, a, b).
    v : float
        Positive diffusion coefficient.
    a : float
        Advective velocity in the x direction.
    b : float
        Advective velocity in the y direction.
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
    """
    # Variables initialization
    m, n = x.shape                                                                              # Number of mesh rows and columns.
    dx   = x[0, 1] - x[0, 0]                                                                    # Uniform column spacing in the x direction.
    dy   = y[1, 0] - y[0, 0]                                                                    # Uniform row spacing in the y direction.
    T    = np.linspace(0, 1, t)                                                                 # Time grid over [0, 1].
    dt   = T[1] - T[0]                                                                          # Time step size.
    r_dx = v*dt/(dx**2)                                                                         # Diffusive coefficient x.
    r_dy = v*dt/(dy**2)                                                                         # Diffusive coefficient y.
    r_a  = a*dt/(2*dx)                                                                          # Advective coefficient x.
    r_b  = b*dt/(2*dy)                                                                          # Advective coefficient y.
    tol  = np.sqrt(np.finfo(float).eps)                                                         # Tolerance for iterative.
    u_ap = np.zeros([m, n, t])                                                                  # Store solution values and provide the implicit initial guess.

    # Initial condition
    u_ap[:, :, 0] = u(x, y, T[0], v, a, b)                                                      # Apply initial state at t = 0.

    # Dirichlet boundaries
    for k in np.arange(t):                                                                      # Loop over time steps.
        u_ap[0,  :, k] = u(x[0, :],  y[0, :],  T[k], v, a, b)                                   # Bottom boundary at the first y-row.
        u_ap[-1, :, k] = u(x[-1, :], y[-1, :], T[k], v, a, b)                                   # Top boundary at the last y-row.
        u_ap[:,  0, k] = u(x[:, 0],  y[:, 0],  T[k], v, a, b)                                   # Left boundary at the first x-column.
        u_ap[:, -1, k] = u(x[:, -1], y[:, -1], T[k], v, a, b)                                   # Right boundary at the last x-column.

    # Iterative update
    for k in range(t-1):                                                                        # Time steps from 0 to t-2.
        if implicit is False:
            for i in range(1, m-1):                                                             # Interior spatial indices y.
                for j in range(1, n-1):                                                         # Interior spatial indices x.
                    u_ap[i, j, k+1] = (                                                         # Explicit advection-diffusion update at node (i, j).
                        u_ap[i, j, k]                                                           # Previous value at the node.
                        + r_dx*(u_ap[i, j+1, k] - 2*u_ap[i, j, k] + u_ap[i, j-1, k])            # Diffusion contribution in x.
                        + r_dy*(u_ap[i+1, j, k] - 2*u_ap[i, j, k] + u_ap[i-1, j, k])            # Diffusion contribution in y.
                        - r_a*(u_ap[i, j+1, k] - u_ap[i, j-1, k])                               # Centered advection contribution in x.
                        - r_b*(u_ap[i+1, j, k] - u_ap[i-1, j, k])                               # Centered advection contribution in y.
                    )
        else:                                                                                   # Implicit scheme.
            err = 1                                                                             # Initial value to enter the loop.
            while err >= tol:                                                                   # Stop by tolerance.
                err = 0                                                                         # Reset maximum difference per iteration.
                for i in range(1, m-1):                                                         # Interior spatial indices y.
                    for j in range(1, n-1):                                                     # Interior spatial indices x.
                        t_val = (                                                               # Compute implicit iterative update.
                            u_ap[i, j, k] +                                                     # Current state contribution.
                            (1-lam) * (                                                         # Explicit-weighted part (1-lambda).
                                r_dx*(u_ap[i, j+1, k] - 2*u_ap[i, j, k] + u_ap[i, j-1, k]) +    # Explicit diffusion contribution in x.
                                r_dy*(u_ap[i+1, j, k] - 2*u_ap[i, j, k] + u_ap[i-1, j, k]) -    # Explicit diffusion contribution in y.
                                r_a*(u_ap[i, j+1, k] - u_ap[i, j-1, k]) -                       # Explicit advection contribution in x.
                                r_b*(u_ap[i+1, j, k] - u_ap[i-1, j, k])                         # Explicit advection contribution in y.
                            ) +                                                                 # Add explicit-weighted contributions before implicit neighbors.
                            lam * (                                                             # Implicit-weighted part (lambda).
                                r_dx*(u_ap[i, j+1, k+1] + u_ap[i, j-1, k+1]) +                  # Implicit diffusion neighbors in x.
                                r_dy*(u_ap[i+1, j, k+1] + u_ap[i-1, j, k+1]) -                  # Implicit diffusion neighbors in y.
                                r_a*(u_ap[i, j+1, k+1] - u_ap[i, j-1, k+1]) -                   # Implicit advection neighbors in x.
                                r_b*(u_ap[i+1, j, k+1] - u_ap[i-1, j, k+1])                     # Implicit advection neighbors in y.
                            )
                        ) / (1 + 2*lam*r_dx + 2*lam*r_dy)                                       # Scaling factor.
                        err = max(err, abs(t_val - u_ap[i, j, k+1]))                            # Infinity-norm update.
                        u_ap[i, j, k+1] = t_val                                                 # Assign new value at interior node.
        
    # Exact solution for comparison
    u_ex = u(x[:, :, None], y[:, :, None], T[None, None, :], v, a, b)                            # Evaluate exact solution over the full space-time mesh.

    return u_ap, u_ex                                                                           # Return approximate and exact.

def A_1D_calc(m, r_d, r_a):
    """
    Construct the 1D Advection-Diffusion operator matrix using centered Finite Differences.

    Parameters
    ----------
    m : int
        Total number of nodes in the 1D mesh (includes boundaries).
    r_d : float
        Diffusive coefficient v*dt/dx^2.
    r_a : float
        Advective coefficient a*dt/(2*dx).

    Returns
    -------
    A : np.ndarray
        Matrix with boundary rows as 0 (for I+A identity) and interior 
            rows having the tridiagonal stencil.
    """
    # Matrix initialization
    A = np.zeros([m, m])                                                                        # Initialize A with zeros.

    # Interior stencil
    for i in range(1, m-1):                                                                     # Traverse interior nodes.
        A[i, i-1] = r_d + r_a                                                                   # Left neighbor.
        A[i, i]   = -2*r_d                                                                      # Central node.
        A[i, i+1] = r_d - r_a                                                                   # Right neighbor.
    
    return A

def A_2D_calc(m, n, r_dx, r_dy, r_a, r_b):
    """
    Construct the 2D Advection-Diffusion operator matrix over an m×n mesh.

    Parameters
    ----------
    m, n : int
        Total number of nodes per direction in the 2D mesh (includes boundaries).
    r_dx : float
        Diffusive coefficient in x.
    r_dy : float
        Diffusive coefficient in y.
    r_a : float
        Advective coefficient in x.
    r_b : float
        Advective coefficient in y.

    Returns
    -------
    A : np.ndarray
        Matrix representing the discrete operator.
    """
    # Matrix initialization
    A    = np.zeros([m*n, m*n])                                                                 # Initialize A with zeros.
    
    # Fill entries
    for i in range(m):                                                                          # For all rows (y direction).
        for j in range(n):                                                                      # For all columns (x direction).
            k = i * n + j                                                                       # Linearized index.
            if not (i == 0 or i == m-1 or j == 0 or j == n-1):                                  # If the node is an inner node.
                A[k, k]   = -2*r_dx - 2*r_dy                                                    # Central node.
                A[k, k-1] = r_dx + r_a                                                          # Left neighbor (x direction).
                A[k, k+1] = r_dx - r_a                                                          # Right neighbor (x direction).
                A[k, k-n] = r_dy + r_b                                                          # Bottom neighbor (y direction).
                A[k, k+n] = r_dy - r_b                                                          # Top neighbor (y direction).
    return A

if __name__ == '__main__':
    print("This module defines transient advection-diffusion solvers. Run Examples/CFDM_Advection_Diffusion_examples.py for examples.")
                                                                                                # Inform users that examples live outside this module.
