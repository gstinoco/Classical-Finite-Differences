"""
=========================================================================================
Classical Finite Difference Schemes to Solve the Poisson Equation
=========================================================================================

This module provides numerical solvers for the 1D and 2D Poisson Equation
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

def Poisson1D(x, phi, f):
    """
    Solve the 1D Poisson equation, d²φ/dx² = -f, with Dirichlet boundary conditions
    using the direct method (assemble matrix and solve the linear system).

    Parameters
    ----------
    x : np.ndarray
        Uniform 1D grid including boundary nodes.
    phi : Callable
        Exact solution used to set boundary conditions and for comparison.
    f : Callable
        Source term f(x) in d²φ/dx² = -f.

    Returns
    -------
    phi_ap : np.ndarray
        Approximate solution at grid nodes (includes boundaries).
    phi_ex : np.ndarray
        Exact solution evaluated at grid nodes.
    """
    # Variable initialization
    m      = len(x)                                                                             # Number of nodes in the 1D mesh.
    dx     = x[1] - x[0]                                                                        # Uniform distance between adjacent nodes.
    
    # Operator matrix
    A      = A_1D_calc(m, dx)                                                                   # Assemble the 1D Laplacian matrix with Dirichlet rows.

    # Right-hand side
    rhs    = RHS_1D_calc(x, dx, f, phi)                                                         # Assemble the RHS vector with boundary and source values.

    # Linear solve
    phi_ap = np.linalg.solve(A, rhs)                                                            # Solve the Dirichlet linear system.

    # Exact solution
    phi_ex = phi(x)                                                                             # Exact solution for comparison.
    return phi_ap, phi_ex                                                                       # Return approximate and exact solutions.

def Poisson1D_iter(x, phi, f, max_iter=1000, tol=np.sqrt(np.finfo(float).eps)):
    """
    Solve the 1D Poisson equation, d²φ/dx² = -f, with Dirichlet boundary conditions
    using a Gauss–Seidel iterative scheme.

    Parameters
    ----------
    x : np.ndarray
        Uniform 1D grid including boundary nodes.
    phi : Callable
        Exact solution used to set boundary conditions and for comparison.
    f : Callable
        Source term f(x) in d²φ/dx² = -f.
    max_iter : int
        Maximum number of iterations (default 1000).
    tol : float
        Infinity-norm tolerance for stopping criterion (default sqrt(eps)). 

    Returns
    -------
    phi_ap : np.ndarray
        Approximate solution at grid nodes (includes boundaries).
    phi_ex : np.ndarray
        Exact solution evaluated at grid nodes.
    """
    # Variable initialization
    m           = len(x)                                                                        # Number of nodes in the 1D mesh.
    dx          = x[1] - x[0]                                                                   # Uniform distance between adjacent nodes.
    phi_ap      = np.zeros([m])                                                                 # Store the iterated solution values.
    err         = 1.0                                                                           # Start above tolerance to enter the iteration loop.

    # Boundary conditions
    phi_ap[0]   = phi(x[0])                                                                     # Left Dirichlet value at the first grid node.
    phi_ap[-1]  = phi(x[-1])                                                                    # Right Dirichlet value at the last grid node.

    # Iterative loop
    while err >= tol and max_iter > 0:                                                          # Stop by tolerance or max iterations.
        max_iter -= 1                                                                           # Iteration counter decrement.
        err       = 0.0                                                                         # Reset maximum update difference.
        for i in range(1, m-1):                                                                 # Update all interior nodes.
            t         = 0.5*(phi_ap[i-1] + phi_ap[i+1] + (dx**2)*f(x[i]))                       # Candidate value from the 1D Poisson stencil.
            err       = max(err, abs(t - phi_ap[i]))                                            # Track the largest nodal update this iteration.
            phi_ap[i] = t                                                                       # Store the new value immediately for GS ordering.
        
    # Exact solution
    phi_ex      = phi(x)                                                                        # Exact solution for comparison.
    return phi_ap, phi_ex                                                                       # Return approximate and exact solutions.

def Poisson2D(x, y, phi, f):
    """
    Solve the 2D Poisson equation, Δφ = -f, with Dirichlet boundary conditions
    using the direct method (5-point stencil and linear solve).

    Parameters
    ----------
    x, y    : ndarrays of shape (m, n)
                2D mesh generated with meshgrid (nodes including boundaries).
    phi : Callable
        Exact solution used to set boundary conditions and for comparison.
    f : Callable
        Source term f(x, y) in Δφ = -f.

    Returns
    -------
    phi_ap : np.ndarray
        Approximate solution on the 2D mesh (includes boundaries).
    phi_ex : np.ndarray
        Exact solution evaluated on the 2D mesh.
    """
    # Variable initialization
    m, n   = x.shape                                                                            # Number of mesh rows and columns.
    dx     = x[0, 1] - x[0, 0]                                                                  # Uniform column spacing in the x direction.
    dy     = y[1, 0] - y[0, 0]                                                                  # Uniform row spacing in the y direction.
    
    # Operator matrix
    A = A_2D_calc(m, n, dx, dy)                                                                 # Assemble the 2D Laplacian matrix with Dirichlet rows.
    
    # Right-hand side
    rhs = RHS_2D_calc(x, y, f, phi, m, n)                                                       # Assemble the RHS vector with boundary and source terms.
    
    # Linear solve
    phi_ap = np.linalg.solve(A, rhs).reshape((m, n))                                            # Solve the flattened system and restore the mesh shape.
    
    # Exact solution
    phi_ex = phi(x, y)                                                                          # Evaluate exact solution over the full mesh.
    
    return phi_ap, phi_ex                                                                       # Return approximate and exact solutions on the mesh.

def Poisson2D_iter(x, y, phi, f, max_iter=1000, tol=np.sqrt(np.finfo(float).eps)):
    """
    Solve the 2D Poisson equation, Δφ = -f, with Dirichlet boundary conditions
    using a Gauss–Seidel iterative scheme on interior nodes.

    Parameters
    ----------
    x, y        : ndarrays of shape (m, n)
                    2D mesh generated with meshgrid (nodes including boundaries).
    phi : Callable
        Exact solution used to set boundary conditions and for comparison.
    f : Callable
        Source term f(x, y) in Δφ = -f.
    max_iter : int
        Maximum number of iterations (default 1000).
    tol : float
        Infinity-norm tolerance for stopping criterion (default sqrt(eps)).

    Returns
    -------
    phi_ap : np.ndarray
        Approximate solution on the 2D mesh (includes boundaries).
    phi_ex : np.ndarray
        Exact solution evaluated on the 2D mesh.
    """
    # Variable initialization
    m, n   = x.shape                                                                            # Number of mesh rows and columns.
    dx     = x[0, 1] - x[0, 0]                                                                  # Uniform column spacing in the x direction.
    dy     = y[1, 0] - y[0, 0]                                                                  # Uniform row spacing in the y direction.

    # Initial solution and error
    phi_ap = np.zeros((m, n))                                                                   # Store the iterated solution, initialized with zero interior values.
    err    = 1                                                                                  # Start above tolerance to enter the iteration loop.

    # Boundary conditions
    for i in range(m):                                                                          # Traverse rows to impose left and right edges.
        phi_ap[i, 0]  = phi(x[i, 0],   y[i, 0])                                                 # Left boundary at the first x-column.
        phi_ap[i, -1] = phi(x[i, -1],  y[i, -1])                                                # Right boundary at the last x-column.
    for j in range(n):                                                                          # Traverse columns to impose bottom and top edges.
        phi_ap[0,  j] = phi(x[0,   j], y[0,  j])                                                # Bottom boundary at the first y-row.
        phi_ap[-1, j] = phi(x[-1, j],  y[-1, j])                                                # Top boundary at the last y-row.

    # Gauss–Seidel iteration on interior nodes
    while err >= tol and max_iter > 0:                                                          # Stop by tolerance or max iterations.
        max_iter -= 1                                                                           # Decrement iteration counter.
        err       = 0                                                                           # Reset maximum difference per iteration.
        for i in range(1, m-1):                                                                 # Interior mesh rows.
            for j in range(1, n-1):                                                             # Interior mesh columns.
                t = (                                                                           # Gauss-Seidel candidate value at node (i, j).
                    (phi_ap[i, j-1] + phi_ap[i, j+1]) / dx**2 +                                 # Left and right neighbors in the x direction.
                    (phi_ap[i-1, j] + phi_ap[i+1, j]) / dy**2 +                                 # Bottom and top neighbors in the y direction.
                    f(x[i, j], y[i, j])                                                         # Source term for Delta phi = -f.
                ) / (2/dx**2 + 2/dy**2)                                                         # Divide by the diagonal coefficient of the 5-point operator.
                err    = max(err, abs(t - phi_ap[i, j]))                                        # Track the largest nodal update this iteration.
                phi_ap[i, j] = t                                                                # Store the new interior value immediately for GS ordering.

    # Exact solution for reference
    phi_ex = phi(x, y)                                                                          # Evaluate the exact solution over the full 2D mesh.

    return phi_ap, phi_ex                                                                       # Return approximate and exact solutions on the mesh.

def A_1D_calc(m, dx):
    """
    Construct the 1D Laplacian operator matrix using centered Finite Differences,
    including identity-like boundary rows to impose Dirichlet conditions.

    Parameters
    ----------
    m : int
        Total number of nodes in the 1D mesh (includes boundaries).
    dx : float
        Uniform step size between nodes.

    Returns
    -------
    A : np.ndarray
        Matrix with:
            - Boundary rows: identity to fix φ(a) and φ(b).
            - Interior rows: tridiagonal stencil scaled by 1/dx².
    """
    # Matrix initialization
    A = np.zeros([m, m])                                                                        # Allocate the dense 1D operator matrix.

    # Interior stencil
    for i in range(1, m-1):                                                                     # Fill the centered stencil on interior rows.
        A[i, i-1] = 1                                                                           # Coefficient multiplying phi_{i-1}.
        A[i, i]   = -2                                                                          # Coefficient multiplying phi_i.
        A[i, i+1] = 1                                                                           # Coefficient multiplying phi_{i+1}.
    
    # Dirichlet boundary rows
    A[0,   0]     = dx**2                                                                       # Left Dirichlet row becomes identity after scaling.
    A[-1, -1]     = dx**2                                                                       # Right Dirichlet row becomes identity after scaling.
    # Scaling
    A             = A/dx**2                                                                     # Convert stencil coefficients to second-derivative units.
    return A

def RHS_1D_calc(x, dx, f, phi):
    """
    Assemble the 1D right-hand side vector including Dirichlet boundaries
    and the interior source term.

    Parameters
    ----------
    x : np.ndarray
        Uniform 1D grid including boundaries.
    dx : float
        Uniform step size between nodes.
    f : Callable
        Source term f(x) in d²φ/dx² = -f, evaluated at nodes.
    phi : Callable
        Boundary condition and exact solution function φ(x).

    Returns
    -------
    rhs : np.ndarray
        RHS vector with:
            - Boundary entries: φ(a) and φ(b).
            - Interior entries: -f(x_i).
    """
    # Initialize RHS
    rhs           = -f(x)                                                                       # Initialize entries with source values; boundaries are overwritten below.
    
    # Apply boundary values
    rhs[0]        = phi(x[0])                                                                   # Left Dirichlet value at the first grid node.
    rhs[-1]       = phi(x[-1])                                                                  # Right Dirichlet value at the last grid node.
    return rhs

def A_2D_calc(m, n, dx, dy):
    """
    Construct the 2D Laplacian operator matrix with the 5-point stencil
    over an m×n mesh, including identity-like boundary rows to impose Dirichlet.

    Parameters
    ----------
    m, n : int
        Total number of nodes per direction in the 2D mesh (includes boundaries).
    dx, dy : float
        Uniform step sizes along x and y.

    Returns
    -------
    A : np.ndarray
        Matrix with:
                - Boundary rows: identity to fix φ on ∂Ω.
                - Interior rows: 5-point stencil scaled by 1/dx² and 1/dy².
    """
    # Matrix initialization
    A    = np.zeros([m*n, m*n])                                                                 # Allocate the dense matrix for the flattened mesh.
    
    # Fill entries
    for i in range(m):                                                                          # Traverse mesh rows.
        for j in range(n):                                                                      # Traverse mesh columns.
            k = i * n + j                                                                       # Map node (i, j) to row-major vector index.
            if i == 0 or i == m-1 or j == 0 or j == n-1:                                        # Boundary node.
                A[k, k] = 1                                                                     # Identity row enforces the Dirichlet boundary value.
            else:                                                                               # Interior node.
                A[k, k]   = -2/dx**2 - 2/dy**2                                                  # Coefficient multiplying phi_{i,j}.
                A[k, k-1] = 1/dx**2                                                             # Coefficient for the left neighbor phi_{i,j-1}.
                A[k, k+1] = 1/dx**2                                                             # Coefficient for the right neighbor phi_{i,j+1}.
                A[k, k-n] = 1/dy**2                                                             # Coefficient for the bottom neighbor phi_{i-1,j}.
                A[k, k+n] = 1/dy**2                                                             # Coefficient for the top neighbor phi_{i+1,j}.
    return A

def RHS_2D_calc(x, y, f, phi, m, n):
    """
    Assemble the 2D right-hand side vector including boundary contributions
    and the interior source term.

    Parameters
    ----------
    x, y    : ndarrays of shape (m, n)
                2D mesh generated with meshgrid.
    f : Callable
        Source term f(x, y) in Δφ = -f, evaluated at interior nodes.
    phi : Callable
        Boundary condition and exact solution function φ(x, y).
    m, n : int
        Number of rows and columns in the 2D mesh.

    Returns
    -------
    rhs : np.ndarray
        RHS vector with:
                - Boundary entries: φ on ∂Ω.
                - Interior entries: -f(x_i, y_j).
    """
    # Initialize RHS
    rhs = np.empty(m*n)                                                                         # Allocate RHS entries; every node is assigned below.

    # Loop nodes and assign values
    for i in range(m):                                                                          # Traverse mesh rows.
        for j in range(n):                                                                      # Traverse mesh columns.
            k = i * n + j                                                                       # Map node (i, j) to row-major vector index.
            if i == 0 or i == m-1 or j == 0 or j == n-1:                                        # Boundary node.
                rhs[k] = phi(x[i, j], y[i, j])                                                  # Dirichlet value imposed by the identity row.
            else:                                                                               # Interior node.
                rhs[k] = -f(x[i, j], y[i, j])                                                   # Source term for Delta phi = -f.
    return rhs


def Poisson1D_Neumann_1(x, phi, f, sig, beta):
    """
    Solve the 1D Poisson equation with Neumann BC 1 (Two-Point Backward) at left, Dirichlet at right.
    Uses the Matrix Formulation.

    Parameters
    ----------
    x : numpy.ndarray
        1D spatial mesh (includes boundaries).
    phi : Callable
        Exact solution phi(x), used to evaluate exact solution.
    f : Callable
        Source term f(x).
    sig : float
        Derivative at the left boundary (x=0).
    beta : float
        Dirichlet boundary value at the right boundary (x=L).

    Returns
    -------
    phi_ap : numpy.ndarray
        Approximate solution.
    phi_ex : numpy.ndarray
        Exact solution evaluated on x.
    """
    m       = len(x)                                                                            # Number of nodes in the 1D mesh.
    dx      = x[1] - x[0]                                                                       # Uniform distance between adjacent nodes.
    
    A       = A_1D_calc(m, dx)                                                                  # Start from the Dirichlet 1D Poisson matrix.
    A[0, 0] = -dx / dx**2                                                                       # Coefficient -1/dx in (phi_1 - phi_0)/dx = sig.
    A[0, 1] = dx / dx**2                                                                        # Coefficient  1/dx in (phi_1 - phi_0)/dx = sig.
    
    rhs     = RHS_1D_calc(x, dx, f, phi)                                                        # Start from the Dirichlet/source RHS vector.
    rhs[0]  = sig                                                                               # Set the left derivative value for the Neumann row.
    rhs[-1] = beta                                                                              # Set the right Dirichlet boundary value.
    
    phi_ap  = np.linalg.solve(A, rhs)                                                           # Solve the modified Neumann-Dirichlet linear system.
    phi_ex  = phi(x)                                                                            # Exact solution for comparison.
    return phi_ap, phi_ex                                                                       # Return approximate and exact solutions.

def Poisson1D_Neumann_1_iter(x, phi, f, sig, beta, max_iter=20000, tol=1e-12):
    """
    Solve the 1D Poisson equation with Neumann BC 1 (Iterative).
    Uses a Gauss-Seidel iterative approach with Two-Point Backward stencil at the left boundary.

    Parameters
    ----------
    x : numpy.ndarray
        1D spatial mesh (includes boundaries).
    phi : Callable
        Exact solution phi(x), used to evaluate exact solution.
    f : Callable
        Source term f(x).
    sig : float
        Derivative at the left boundary (x=0).
    beta : float
        Dirichlet boundary value at the right boundary (x=L).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for the stopping criterion.

    Returns
    -------
    phi_ap : numpy.ndarray
        Approximate solution.
    phi_ex : numpy.ndarray
        Exact solution evaluated on x.
    """
    m      = len(x)                                                                             # Number of nodes in the 1D mesh.
    dx     = x[1] - x[0]                                                                        # Uniform distance between adjacent nodes.
    phi_ap = np.zeros([m])                                                                      # Store the iterated solution values.
    err    = 1.0                                                                                # Start above tolerance to enter the iteration loop.

    phi_ap[-1] = beta                                                                           # Fix the right Dirichlet boundary throughout iteration.

    while err >= tol and max_iter > 0:                                                          # Stop by tolerance or max iterations.
        max_iter -= 1                                                                           # Iteration counter decrement.
        err       = 0.0                                                                         # Reset maximum update difference.
        for i in range(1, m-1):                                                                 # Update all interior nodes.
            t         = 0.5*(phi_ap[i-1] + phi_ap[i+1] + (dx**2)*f(x[i]))                       # Gauss-Seidel update for interior.
            err       = max(err, abs(t - phi_ap[i]))                                            # Track the largest interior update.
            phi_ap[i] = t                                                                       # Store the new interior value immediately for GS ordering.
        
        old_boundary = phi_ap[0]                                                                # Store previous Neumann boundary value.
        phi_ap[0]    = phi_ap[1] - dx * sig                                                     # Enforce (phi_1 - phi_0)/dx = sig at the left boundary.
        err          = max(err, abs(phi_ap[0] - old_boundary))                                  # Include boundary update in stopping criterion.

    phi_ex = phi(x)                                                                             # Exact solution for comparison.
    return phi_ap, phi_ex                                                                       # Return approximate and exact solutions.

def Poisson1D_Neumann_2(x, phi, f, sig, beta):
    """
    Solve the 1D Poisson equation with Neumann BC 2 (Two-Point Centered) at left, Dirichlet at right.
    Uses the Matrix Formulation.

    Parameters
    ----------
    x : numpy.ndarray
        1D spatial mesh (includes boundaries).
    phi : Callable
        Exact solution phi(x), used to evaluate exact solution.
    f : Callable
        Source term f(x).
    sig : float
        Derivative at the left boundary (x=0).
    beta : float
        Dirichlet boundary value at the right boundary (x=L).

    Returns
    -------
    phi_ap : numpy.ndarray
        Approximate solution.
    phi_ex : numpy.ndarray
        Exact solution evaluated on x.
    """
    m      = len(x)                                                                             # Number of nodes in the 1D mesh.
    dx     = x[1] - x[0]                                                                        # Uniform distance between adjacent nodes.
    
    A       = A_1D_calc(m, dx)                                                                  # Start from the Dirichlet 1D Poisson matrix.
    A[0, 0] = -dx / dx**2                                                                       # Coefficient -1/dx in the reduced centered Neumann row.
    A[0, 1] = dx / dx**2                                                                        # Coefficient  1/dx in the reduced centered Neumann row.
    
    rhs     = RHS_1D_calc(x, dx, f, phi)                                                        # Start from the Dirichlet/source RHS vector.
    rhs[0]  = sig - ((dx / 2) * f(x[0]))                                                        # Add the source correction from the centered Neumann formula.
    rhs[-1] = beta                                                                              # Set the right Dirichlet boundary value.
    
    phi_ap  = np.linalg.solve(A, rhs)                                                           # Solve the modified Neumann-Dirichlet linear system.
    phi_ex  = phi(x)                                                                            # Exact solution for comparison.
    return phi_ap, phi_ex                                                                       # Return approximate and exact solutions.

def Poisson1D_Neumann_2_iter(x, phi, f, sig, beta, max_iter=20000, tol=1e-12):
    """
    Solve the 1D Poisson equation with Neumann BC 2 (Iterative).
    Uses a Gauss-Seidel iterative approach with Two-Point Centered stencil at the left boundary.

    Parameters
    ----------
    x : numpy.ndarray
        1D spatial mesh (includes boundaries).
    phi : Callable
        Exact solution phi(x), used to evaluate exact solution.
    f : Callable
        Source term f(x).
    sig : float
        Derivative at the left boundary (x=0).
    beta : float
        Dirichlet boundary value at the right boundary (x=L).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for the stopping criterion.

    Returns
    -------
    phi_ap : numpy.ndarray
        Approximate solution.
    phi_ex : numpy.ndarray
        Exact solution evaluated on x.
    """
    m      = len(x)                                                                             # Number of nodes in the 1D mesh.
    dx     = x[1] - x[0]                                                                        # Uniform distance between adjacent nodes.
    phi_ap = np.zeros([m])                                                                      # Store the iterated solution values.
    err    = 1.0                                                                                # Start above tolerance to enter the iteration loop.

    phi_ap[-1] = beta                                                                           # Fix the right Dirichlet boundary throughout iteration.

    while err >= tol and max_iter > 0:                                                          # Stop by tolerance or max iterations.
        max_iter -= 1                                                                           # Iteration counter decrement.
        err       = 0.0                                                                         # Reset maximum update difference.
        for i in range(1, m-1):                                                                 # Update all interior nodes.
            t         = 0.5*(phi_ap[i-1] + phi_ap[i+1] + (dx**2)*f(x[i]))                       # Gauss-Seidel update for interior.
            err       = max(err, abs(t - phi_ap[i]))                                            # Track the largest interior update.
            phi_ap[i] = t                                                                       # Store the new interior value immediately for GS ordering.
        
        old_boundary = phi_ap[0]                                                                # Store previous Neumann boundary value.
        phi_ap[0]    = phi_ap[1] - dx * sig + (dx**2 / 2) * f(x[0])                             # Enforce the centered Neumann condition after eliminating the ghost node.
        err          = max(err, abs(phi_ap[0] - old_boundary))                                  # Include boundary update in stopping criterion.

    phi_ex = phi(x)                                                                             # Exact solution for comparison.
    return phi_ap, phi_ex                                                                       # Return approximate and exact solutions.

def Poisson1D_Neumann_3(x, phi, f, sig, beta):
    """
    Solve the 1D Poisson equation with Neumann BC 3 (Three-Point Forward) at left, Dirichlet at right.
    Uses the Matrix Formulation.

    Parameters
    ----------
    x : numpy.ndarray
        1D spatial mesh (includes boundaries).
    phi : Callable
        Exact solution phi(x), used to evaluate exact solution.
    f : Callable
        Source term f(x).
    sig : float
        Derivative at the left boundary (x=0).
    beta : float
        Dirichlet boundary value at the right boundary (x=L).

    Returns
    -------
    phi_ap : numpy.ndarray
        Approximate solution.
    phi_ex : numpy.ndarray
        Exact solution evaluated on x.
    """
    m      = len(x)                                                                             # Number of nodes in the 1D mesh.
    dx     = x[1] - x[0]                                                                        # Uniform distance between adjacent nodes.
    
    A       = A_1D_calc(m, dx)                                                                  # Start from the Dirichlet 1D Poisson matrix.
    A[0, 0] = -(3 / 2) * dx / dx**2                                                             # Coefficient -3/(2dx) in the forward Neumann row.
    A[0, 1] = 2 * dx / dx**2                                                                    # Coefficient  2/dx in the forward Neumann row.
    A[0, 2] = -(1 / 2) * dx / dx**2                                                             # Coefficient -1/(2dx) in the forward Neumann row.
    
    rhs     = RHS_1D_calc(x, dx, f, phi)                                                        # Start from the Dirichlet/source RHS vector.
    rhs[0]  = sig                                                                               # Set the left derivative value for the Neumann row.
    rhs[-1] = beta                                                                              # Set the right Dirichlet boundary value.
    
    phi_ap  = np.linalg.solve(A, rhs)                                                           # Solve the modified Neumann-Dirichlet linear system.
    phi_ex  = phi(x)                                                                            # Exact solution for comparison.
    return phi_ap, phi_ex                                                                       # Return approximate and exact solutions.

def Poisson1D_Neumann_3_iter(x, phi, f, sig, beta, max_iter=20000, tol=1e-12):
    """
    Solve the 1D Poisson equation with Neumann BC 3 (Iterative).
    Uses a Gauss-Seidel iterative approach with Three-Point Forward stencil at the left boundary.

    Parameters
    ----------
    x : numpy.ndarray
        1D spatial mesh (includes boundaries).
    phi : Callable
        Exact solution phi(x), used to evaluate exact solution.
    f : Callable
        Source term f(x).
    sig : float
        Derivative at the left boundary (x=0).
    beta : float
        Dirichlet boundary value at the right boundary (x=L).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for the stopping criterion.

    Returns
    -------
    phi_ap : numpy.ndarray
        Approximate solution.
    phi_ex : numpy.ndarray
        Exact solution evaluated on x.
    """
    m      = len(x)                                                                             # Number of nodes in the 1D mesh.
    dx     = x[1] - x[0]                                                                        # Uniform distance between adjacent nodes.
    phi_ap = np.zeros([m])                                                                      # Store the iterated solution values.
    err    = 1.0                                                                                # Start above tolerance to enter the iteration loop.

    phi_ap[-1] = beta                                                                           # Fix the right Dirichlet boundary throughout iteration.

    while err >= tol and max_iter > 0:                                                          # Stop by tolerance or max iterations.
        max_iter -= 1                                                                           # Iteration counter decrement.
        err       = 0.0                                                                         # Reset maximum update difference.
        for i in range(1, m-1):                                                                 # Update all interior nodes.
            t         = 0.5*(phi_ap[i-1] + phi_ap[i+1] + (dx**2)*f(x[i]))                       # Gauss-Seidel update for interior.
            err       = max(err, abs(t - phi_ap[i]))                                            # Track the largest interior update.
            phi_ap[i] = t                                                                       # Store the new interior value immediately for GS ordering.
        
        old_boundary = phi_ap[0]                                                                # Store previous Neumann boundary value.
        phi_ap[0]    = (2 / 3) * (2 * phi_ap[1] - (1 / 2) * phi_ap[2] - dx * sig)               # Enforce (-3phi_0 + 4phi_1 - phi_2)/(2dx) = sig.
        err          = max(err, abs(phi_ap[0] - old_boundary))                                  # Include boundary update in stopping criterion.

    phi_ex = phi(x)                                                                             # Exact solution for comparison.
    return phi_ap, phi_ex                                                                       # Return approximate and exact solutions.

if __name__ == '__main__':
    print("This module defines Poisson solvers and utilities. Run Examples/CFDM_Poisson_examples.py for examples.")
                                                                                                # Inform users that examples live outside this module.
