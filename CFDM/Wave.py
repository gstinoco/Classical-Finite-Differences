"""
=========================================================================================
Classical Finite Difference Schemes to Solve the Wave Equation
=========================================================================================

This module provides numerical solvers for the 1D and 2D Wave Equation using
Classical Finite Difference methods.

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

def Wave1D(x, t, u, c):
    """
    Solve the transient 1D wave equation, u_tt = c² u_xx, with Dirichlet
    boundary conditions using the matrix finite-difference formulation.

    Parameters
    ----------
    x : np.ndarray
        Uniform 1D spatial mesh including boundary nodes.
    t : int
        Number of temporal nodes over the interval [0, 1].
    u : Callable
        Exact solution and boundary function with signature u(x, t, c).
    c : float
        Wave propagation speed.

    Returns
    -------
    u_ap : np.ndarray
        Approximate solution with shape (m, t).
    u_ex : np.ndarray
        Exact solution evaluated with shape (m, t).
    """
    # Variables initialization
    m    = len(x)                                                                               # Number of spatial nodes.
    T    = np.linspace(0, 1, t)                                                                 # Time grid over [0, 1].
    dx   = x[1] - x[0]                                                                          # Uniform spatial step size.
    dt   = T[1] - T[0]                                                                          # Uniform temporal step size.
    r    = (c*dt/dx)**2                                                                         # Squared Courant number for the wave stencil.
    u_ap = np.empty([m, t])                                                                     # Allocate approximate solution values assigned below.

    # Initial time levels
    u_ap[:, 0] = u(x, T[0], c)                                                                  # Set initial displacement.
    u_ap[:, 1] = u(x, T[1], c)                                                                  # Set second time level from exact data.

    # Dirichlet boundaries
    u_ap[0,  :] = u(x[0],  T, c)                                                                # Left boundary over all time levels.
    u_ap[-1, :] = u(x[-1], T, c)                                                                # Right boundary over all time levels.

    # Operator matrix
    A           = A_1D_calc(m, r)                                                               # Assemble the 1D second-derivative wave operator.

    # Time integration
    for k in range(1, t-1):                                                                     # Advance from the second known time level.
        u_new           = 2*u_ap[:, k] - u_ap[:, k-1] + np.einsum('ij,j->i', A, u_ap[:, k])     # Apply the centered wave update.
        u_ap[1:-1, k+1] = u_new[1:-1]                                                           # Store only interior nodes, preserving boundaries.

    # Exact solution for comparison
    X, TT = np.meshgrid(x, T, indexing='ij')                                                    # Create a space-time grid for vectorized evaluation.
    u_ex  = u(X, TT, c)                                                                         # Evaluate exact solution on the full grid.

    return u_ap, u_ex                                                                           # Return approximate and exact solutions.


def Wave1D_iter(x, t, u, c):
    """
    Solve the transient 1D wave equation, u_tt = c² u_xx, with Dirichlet
    boundary conditions using an explicit node-wise stencil.

    Parameters
    ----------
    x : np.ndarray
        Uniform 1D spatial mesh including boundary nodes.
    t : int
        Number of temporal nodes over the interval [0, 1].
    u : Callable
        Exact solution and boundary function with signature u(x, t, c).
    c : float
        Wave propagation speed.

    Returns
    -------
    u_ap : np.ndarray
        Approximate solution with shape (m, t).
    u_ex : np.ndarray
        Exact solution evaluated with shape (m, t).
    """
    # Variables initialization
    m    = len(x)                                                                               # Number of spatial nodes.
    T    = np.linspace(0, 1, t)                                                                 # Time grid over [0, 1].
    dx   = x[1] - x[0]                                                                          # Uniform spatial step size.
    dt   = T[1] - T[0]                                                                          # Uniform temporal step size.
    r    = (c*dt/dx)**2                                                                         # Squared Courant number for the wave stencil.
    u_ap = np.empty([m, t])                                                                     # Allocate approximate solution values assigned below.

    # Initial time levels
    u_ap[:, 0] = u(x, T[0], c)                                                                  # Set initial displacement.
    u_ap[:, 1] = u(x, T[1], c)                                                                  # Set second time level from exact data.

    # Dirichlet boundaries
    u_ap[0,  :] = u(x[0],  T, c)                                                                # Left boundary over all time levels.
    u_ap[-1, :] = u(x[-1], T, c)                                                                # Right boundary over all time levels.

    # Explicit centered stencil
    for k in range(1, t-1):                                                                     # Advance from the second known time level.
        for i in range(1, m-1):                                                                 # Traverse interior spatial nodes.
            u_ap[i, k+1] = (                                                                    # Centered second-order wave update.
                2*u_ap[i, k] - u_ap[i, k-1]                                                     # Time-centered contribution.
                + r*(u_ap[i+1, k] - 2*u_ap[i, k] + u_ap[i-1, k])                                # Centered spatial second derivative.
            )

    # Exact solution for comparison
    X, TT = np.meshgrid(x, T, indexing='ij')                                                    # Create a space-time grid for vectorized evaluation.
    u_ex  = u(X, TT, c)                                                                         # Evaluate exact solution on the full grid.

    return u_ap, u_ex                                                                           # Return approximate and exact solutions.


def Wave2D(x, y, t, u, c):
    """
    Solve the transient 2D wave equation, u_tt = c² Δu, with Dirichlet
    boundary conditions using the matrix finite-difference formulation.

    Parameters
    ----------
    x, y : np.ndarray
        2D spatial meshes generated with meshgrid and including boundary nodes.
    t : int
        Number of temporal nodes over the interval [0, 1].
    u : Callable
        Exact solution and boundary function with signature u(x, y, t, c).
    c : float
        Wave propagation speed.

    Returns
    -------
    u_ap : np.ndarray
        Approximate solution with shape (m, n, t).
    u_ex : np.ndarray
        Exact solution evaluated with shape (m, n, t).
    """
    # Variables initialization
    m, n = x.shape                                                                              # Number of mesh rows and columns.
    T    = np.linspace(0, 1, t)                                                                 # Time grid over [0, 1].
    dx   = x[0, 1] - x[0, 0]                                                                    # Uniform spacing in the x direction.
    dy   = y[1, 0] - y[0, 0]                                                                    # Uniform spacing in the y direction.
    dt   = T[1] - T[0]                                                                          # Uniform temporal step size.
    r_x  = (c*dt/dx)**2                                                                         # Squared Courant number in the x direction.
    r_y  = (c*dt/dy)**2                                                                         # Squared Courant number in the y direction.
    u_ap = np.empty([m, n, t])                                                                  # Allocate approximate solution values assigned below.

    # Initial time levels
    u_ap[:, :, 0] = u(x, y, T[0], c)                                                            # Set initial displacement.
    u_ap[:, :, 1] = u(x, y, T[1], c)                                                            # Set second time level from exact data.

    # Dirichlet boundaries
    for k in range(t):                                                                          # Traverse all temporal nodes.
        u_ap[0,  :, k] = u(x[0, :],  y[0, :],  T[k], c)                                         # Bottom boundary.
        u_ap[-1, :, k] = u(x[-1, :], y[-1, :], T[k], c)                                         # Top boundary.
        u_ap[:,  0, k] = u(x[:, 0],  y[:, 0],  T[k], c)                                         # Left boundary.
        u_ap[:, -1, k] = u(x[:, -1], y[:, -1], T[k], c)                                         # Right boundary.

    # Operator matrix
    A = A_2D_calc(m, n, r_x, r_y)                                                               # Assemble the 2D wave operator over the flattened mesh.

    # Time integration
    for k in range(1, t-1):                                                                     # Advance from the second known time level.
        u_old                 = u_ap[:, :, k].reshape(m*n)                                      # Flatten current mesh values.
        u_prev                = u_ap[:, :, k-1].reshape(m*n)                                    # Flatten previous time level.
        u_new                 = 2*u_old - u_prev + np.einsum('ij,j->i', A, u_old)               # Apply the centered wave update.
        urr                   = u_new.reshape(m, n)                                             # Restore the 2D mesh shape.
        u_ap[1:-1, 1:-1, k+1] = urr[1:-1, 1:-1]                                                 # Store only interior nodes, preserving boundaries.

    # Exact solution for comparison
    u_ex = u(x[:, :, None], y[:, :, None], T[None, None, :], c)                                 # Evaluate exact solution over the full space-time mesh.

    return u_ap, u_ex                                                                           # Return approximate and exact solutions.


def Wave2D_iter(x, y, t, u, c):
    """
    Solve the transient 2D wave equation, u_tt = c² Δu, with Dirichlet
    boundary conditions using an explicit node-wise stencil.

    Parameters
    ----------
    x, y : np.ndarray
        2D spatial meshes generated with meshgrid and including boundary nodes.
    t : int
        Number of temporal nodes over the interval [0, 1].
    u : Callable
        Exact solution and boundary function with signature u(x, y, t, c).
    c : float
        Wave propagation speed.

    Returns
    -------
    u_ap : np.ndarray
        Approximate solution with shape (m, n, t).
    u_ex : np.ndarray
        Exact solution evaluated with shape (m, n, t).
    """
    # Variables initialization
    m, n = x.shape                                                                              # Number of mesh rows and columns.
    T    = np.linspace(0, 1, t)                                                                 # Time grid over [0, 1].
    dx   = x[0, 1] - x[0, 0]                                                                    # Uniform spacing in the x direction.
    dy   = y[1, 0] - y[0, 0]                                                                    # Uniform spacing in the y direction.
    dt   = T[1] - T[0]                                                                          # Uniform temporal step size.
    r_x  = (c*dt/dx)**2                                                                         # Squared Courant number in the x direction.
    r_y  = (c*dt/dy)**2                                                                         # Squared Courant number in the y direction.
    u_ap = np.empty([m, n, t])                                                                  # Allocate approximate solution values assigned below.

    # Initial time levels
    u_ap[:, :, 0] = u(x, y, T[0], c)                                                            # Set initial displacement.
    u_ap[:, :, 1] = u(x, y, T[1], c)                                                            # Set second time level from exact data.

    # Dirichlet boundaries
    for k in range(t):                                                                          # Traverse all temporal nodes.
        u_ap[0,  :, k] = u(x[0, :],  y[0, :],  T[k], c)                                         # Bottom boundary.
        u_ap[-1, :, k] = u(x[-1, :], y[-1, :], T[k], c)                                         # Top boundary.
        u_ap[:,  0, k] = u(x[:, 0],  y[:, 0],  T[k], c)                                         # Left boundary.
        u_ap[:, -1, k] = u(x[:, -1], y[:, -1], T[k], c)                                         # Right boundary.

    # Explicit centered stencil
    for k in range(1, t-1):                                                                     # Advance from the second known time level.
        for i in range(1, m-1):                                                                 # Traverse interior rows.
            for j in range(1, n-1):                                                             # Traverse interior columns.
                u_ap[i, j, k+1] = (                                                             # Centered second-order wave update.
                    2*u_ap[i, j, k] - u_ap[i, j, k-1]                                           # Time-centered contribution.
                    + r_x*(u_ap[i, j+1, k] - 2*u_ap[i, j, k] + u_ap[i, j-1, k])                 # Spatial second derivative in x.
                    + r_y*(u_ap[i+1, j, k] - 2*u_ap[i, j, k] + u_ap[i-1, j, k])                 # Spatial second derivative in y.
                )

    # Exact solution for comparison
    u_ex = u(x[:, :, None], y[:, :, None], T[None, None, :], c)                                 # Evaluate exact solution over the full space-time mesh.

    return u_ap, u_ex                                                                           # Return approximate and exact solutions.


def A_1D_calc(m, r):
    """
    Construct the 1D wave-equation spatial operator for the centered stencil.

    Parameters
    ----------
    m : int
        Total number of spatial nodes including boundaries.
    r : float
        Squared Courant number (c*dt/dx)^2.

    Returns
    -------
    A : np.ndarray
        Dense matrix containing the interior stencil r*[1, -2, 1].
    """
    # Matrix initialization
    A = np.zeros([m, m])                                                                        # Allocate dense operator with zero boundary rows.

    # Interior stencil
    for i in range(1, m-1):                                                                     # Traverse interior nodes.
        A[i, i-1] = r                                                                           # Left neighbor coefficient.
        A[i, i]   = -2*r                                                                        # Central node coefficient.
        A[i, i+1] = r                                                                           # Right neighbor coefficient.

    return A                                                                                    # Return the assembled 1D operator.


def A_2D_calc(m, n, r_x, r_y):
    """
    Construct the 2D wave-equation spatial operator for the centered 5-point stencil.

    Parameters
    ----------
    m, n : int
        Number of mesh rows and columns including boundaries.
    r_x, r_y : float
        Squared Courant numbers along x and y.

    Returns
    -------
    A : np.ndarray
        Dense matrix containing the interior 5-point wave stencil.
    """
    # Matrix initialization
    A = np.zeros([m*n, m*n])                                                                    # Allocate dense operator with zero boundary rows.

    # Fill entries
    for i in range(m):                                                                          # Traverse mesh rows.
        for j in range(n):                                                                      # Traverse mesh columns.
            k = i * n + j                                                                       # Map node (i, j) to row-major vector index.
            if not (i == 0 or i == m-1 or j == 0 or j == n-1):                                  # Interior node.
                A[k, k]   = -2*r_x - 2*r_y                                                      # Central node coefficient.
                A[k, k-1] = r_x                                                                 # Left neighbor coefficient.
                A[k, k+1] = r_x                                                                 # Right neighbor coefficient.
                A[k, k-n] = r_y                                                                 # Bottom neighbor coefficient.
                A[k, k+n] = r_y                                                                 # Top neighbor coefficient.

    return A                                                                                    # Return the assembled 2D operator.


if __name__ == "__main__":
    print("This module defines transient wave solvers. Run Examples/CFDM_Wave_examples.py for examples.")
                                                                                                # Inform users that examples live outside this module.
