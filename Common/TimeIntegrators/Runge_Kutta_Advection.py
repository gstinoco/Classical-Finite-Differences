"""
=========================================================================================
Runge-Kutta Methods for Advection
=========================================================================================

This module provides Runge-Kutta time integrators for Advection equations.

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
# Library Importation
import numpy as np                                                                              # Numerical arrays and finite-difference arithmetic.


def RungeKutta2_1D(x, T, a, u, u_ap):
    """
    Second-Order Runge-Kutta (RK2) Method for 1D Advection.

    Advances the numerical solution of a 1D Advection problem in time 
    using a second-order Runge-Kutta scheme (Method of Lines).

    Parameters
    ----------
    x : np.ndarray
        1D array representing the spatial mesh.
    T : np.ndarray
        1D array representing the time grid.
    a : float
        Advective velocity in the x direction.
    u : Callable
        Function defining the boundary and initial conditions.
    u_ap : np.ndarray
        2D array containing the initialized solution values.

    Returns
    -------
    np.ndarray
        2D array of the computed interior approximations in time.
    """
    t  = len(T)                                                                                 # Number of temporal nodes.
    dt = T[1] - T[0]                                                                            # Uniform time-step size.

    for k in range(t - 1):
        k1 = rhs_1D(u_ap[:, k], T, k, x, a, u)                                                  # First RK2 slope from the current state.
        k2 = rhs_1D(u_ap[:, k] + (dt / 2) * k1, T + (dt / 2), k, x, a, u)                       # Second RK2 slope from the midpoint state.

        u_ap[1:-1, k + 1] = u_ap[1:-1, k] + (dt / 2) * (k1[1:-1] + k2[1:-1])                    # Advance interior nodes with the RK2 average slope.
    return u_ap[1:-1, :]


def RungeKutta3_1D(x, T, a, u, u_ap):
    """
    Third-Order Runge-Kutta (RK3) Method for 1D Advection.

    Advances the numerical solution of a 1D Advection problem in time 
    using a third-order Runge-Kutta scheme (Method of Lines).

    Parameters
    ----------
    x : np.ndarray
        1D array representing the spatial mesh.
    T : np.ndarray
        1D array representing the time grid.
    a : float
        Advective velocity in the x direction.
    u : Callable
        Function defining the boundary and initial conditions.
    u_ap : np.ndarray
        2D array containing the initialized solution values.

    Returns
    -------
    np.ndarray
        2D array of the computed interior approximations in time.
    """
    t  = len(T)                                                                                 # Number of temporal nodes.
    dt = T[1] - T[0]                                                                            # Uniform time-step size.

    for k in range(t - 1):
        k1 = rhs_1D(u_ap[:, k], T, k, x, a, u)                                                  # First RK3 slope from the current state.
        k2 = rhs_1D(u_ap[:, k] + (dt / 2) * k1, T + (dt / 2), k, x, a, u)                       # Second RK3 slope from the midpoint state.
        k3 = rhs_1D(u_ap[:, k] + (2 * dt) * (k2 - k1), T + dt, k, x, a, u)                      # Third RK3 slope from the endpoint predictor.

        u_ap[1:-1, k + 1] = u_ap[1:-1, k] + (dt / 6) * (k1[1:-1] + 4 * k2[1:-1] + k3[1:-1])     # Advance interior nodes with RK3 weights.
    return u_ap[1:-1, :]


def RungeKutta4_1D(x, T, a, u, u_ap):
    """
    Fourth-Order Runge-Kutta (RK4) Method for 1D Advection.

    Advances the numerical solution of a 1D Advection problem in time 
    using a fourth-order Runge-Kutta scheme (Method of Lines).

    Parameters
    ----------
    x : np.ndarray
        1D array representing the spatial mesh.
    T : np.ndarray
        1D array representing the time grid.
    a : float
        Advective velocity in the x direction.
    u : Callable
        Function defining the boundary and initial conditions.
    u_ap : np.ndarray
        2D array containing the initialized solution values.

    Returns
    -------
    np.ndarray
        2D array of the computed interior approximations in time.
    """
    t  = len(T)                                                                                 # Number of temporal nodes.
    dt = T[1] - T[0]                                                                            # Uniform time-step size.

    for k in range(t - 1):
        k1 = rhs_1D(u_ap[:, k], T, k, x, a, u)                                                  # First RK4 slope from the current state.
        k2 = rhs_1D(u_ap[:, k] + (dt / 2) * k1, T + (dt / 2), k, x, a, u)                       # Second RK4 slope from the first midpoint state.
        k3 = rhs_1D(u_ap[:, k] + (dt / 2) * k2, T + (dt / 2), k, x, a, u)                       # Third RK4 slope from the second midpoint state.
        k4 = rhs_1D(u_ap[:, k] + dt * k3, T + dt, k, x, a, u)                                   # Fourth RK4 slope from the endpoint state.

        u_ap[1:-1, k + 1] = u_ap[1:-1, k] + (dt / 6) * (                                        # Advance 1D interior nodes with RK4 weights.
            k1[1:-1] + 2 * k2[1:-1] + 2 * k3[1:-1] + k4[1:-1]                                   # Classical RK4 weighted slope sum.
        )
    return u_ap[1:-1, :]


def rhs_1D(u_ap, T, k, x, a, u):
    """
    Right-Hand Side (RHS) of the 1D Advection ODE system.

    Evaluates the spatial derivatives using central finite differences 
    to form the RHS of the ODE system for the Method of Lines.

    Parameters
    ----------
    u_ap : np.ndarray
        1D array of the solution values at the current time step.
    T : np.ndarray or float
        Time grid array or current time scalar.
    k : int
        Current time index.
    x : np.ndarray
        1D array representing the spatial mesh.
    a : float
        Advective velocity in the x direction.
    u : Callable
        Function defining the boundary conditions.

    Returns
    -------
    np.ndarray
        1D array of the evaluated spatial derivatives.
    """
    m     = len(x)                                                                              # Number of spatial nodes.
    dx    = x[1] - x[0]                                                                         # Uniform spatial step size.
    s     = np.zeros(m)                                                                         # Allocate RHS vector.

    s[0]  = u(x[0], T[k], a)                                                                    # Left boundary value at time T[k].
    s[-1] = u(x[-1], T[k], a)                                                                   # Right boundary value at time T[k].

    for i in range(1, m - 1):
        s[i] = -(a / (2 * dx)) * (u_ap[i + 1] - u_ap[i - 1])                                    # Centered first derivative scaled by velocity.
    return s

def RungeKutta2_2D(x, y, T, a, b, u, u_ap):
    """
    Second-Order Runge-Kutta (RK2) Method for 2D Advection.

    Advances the numerical solution of a 2D Advection problem in time 
    using a second-order Runge-Kutta scheme (Method of Lines).

    Parameters
    ----------
    x : np.ndarray
        2D array representing the x-coordinates of the spatial mesh.
    y : np.ndarray
        2D array representing the y-coordinates of the spatial mesh.
    T : np.ndarray
        1D array representing the time grid.
    a : float
        Advective velocity in the x direction.
    b : float
        Advective velocity in the y direction.
    u : Callable
        Function defining the boundary and initial conditions.
    u_ap : np.ndarray
        3D array containing the initialized solution values.

    Returns
    -------
    np.ndarray
        3D array of the computed interior approximations in time.
    """
    t  = len(T)                                                                                 # Number of temporal nodes.
    dt = T[1] - T[0]                                                                            # Uniform time-step size.

    for k in range(t - 1):
        k1 = rhs_2D(u_ap[:, :, k], T, k, x, y, a, b, u)                                         # First RK2 slope from the current 2D state.
        k2 = rhs_2D(u_ap[:, :, k] + (dt / 2) * k1, T + (dt / 2), k, x, y, a, b, u)              # Second RK2 slope from the midpoint 2D state.

        u_ap[1:-1, 1:-1, k + 1] = u_ap[1:-1, 1:-1, k] + (dt / 2) * (k1[1:-1, 1:-1] + k2[1:-1, 1:-1])
                                                                                                # Advance interior mesh nodes with RK2 weights.
    return u_ap[1:-1, 1:-1, :]


def RungeKutta3_2D(x, y, T, a, b, u, u_ap):
    """
    Third-Order Runge-Kutta (RK3) Method for 2D Advection.

    Advances the numerical solution of a 2D Advection problem in time 
    using a third-order Runge-Kutta scheme (Method of Lines).

    Parameters
    ----------
    x : np.ndarray
        2D array representing the x-coordinates of the spatial mesh.
    y : np.ndarray
        2D array representing the y-coordinates of the spatial mesh.
    T : np.ndarray
        1D array representing the time grid.
    a : float
        Advective velocity in the x direction.
    b : float
        Advective velocity in the y direction.
    u : Callable
        Function defining the boundary and initial conditions.
    u_ap : np.ndarray
        3D array containing the initialized solution values.

    Returns
    -------
    np.ndarray
        3D array of the computed interior approximations in time.
    """
    t  = len(T)                                                                                 # Number of temporal nodes.
    dt = T[1] - T[0]                                                                            # Uniform time-step size.

    for k in range(t - 1):
        k1 = rhs_2D(u_ap[:, :, k], T, k, x, y, a, b, u)                                         # First RK3 slope from the current 2D state.
        k2 = rhs_2D(u_ap[:, :, k] + (dt / 2) * k1, T + (dt / 2), k, x, y, a, b, u)              # Second RK3 slope from the midpoint 2D state.
        k3 = rhs_2D(u_ap[:, :, k] + dt * (2 * k2 - k1), T + dt, k, x, y, a, b, u)               # Third RK3 slope from the endpoint predictor.
        u_ap[1:-1, 1:-1, k + 1] = u_ap[1:-1, 1:-1, k] + (dt / 6) * (                            # Advance 2D interior nodes with RK3 weights.
            k1[1:-1, 1:-1] + 4 * k2[1:-1, 1:-1] + k3[1:-1, 1:-1]                                # RK3 weighted slope sum on interior nodes.
        )
    return u_ap[1:-1, 1:-1, :]


def RungeKutta4_2D(x, y, T, a, b, u, u_ap):
    """
    Fourth-Order Runge-Kutta (RK4) Method for 2D Advection.

    Advances the numerical solution of a 2D Advection problem in time 
    using a fourth-order Runge-Kutta scheme (Method of Lines).

    Parameters
    ----------
    x : np.ndarray
        2D array representing the x-coordinates of the spatial mesh.
    y : np.ndarray
        2D array representing the y-coordinates of the spatial mesh.
    T : np.ndarray
        1D array representing the time grid.
    a : float
        Advective velocity in the x direction.
    b : float
        Advective velocity in the y direction.
    u : Callable
        Function defining the boundary and initial conditions.
    u_ap : np.ndarray
        3D array containing the initialized solution values.

    Returns
    -------
    np.ndarray
        3D array of the computed interior approximations in time.
    """
    t  = len(T)                                                                                 # Number of temporal nodes.
    dt = T[1] - T[0]                                                                            # Uniform time-step size.

    for k in range(t - 1):
        k1 = rhs_2D(u_ap[:, :, k], T, k, x, y, a, b, u)                                         # First RK4 slope from the current 2D state.
        k2 = rhs_2D(u_ap[:, :, k] + (dt / 2) * k1, T + (dt / 2), k, x, y, a, b, u)              # Second RK4 slope from the first midpoint state.
        k3 = rhs_2D(u_ap[:, :, k] + (dt / 2) * k2, T + (dt / 2), k, x, y, a, b, u)              # Third RK4 slope from the second midpoint state.
        k4 = rhs_2D(u_ap[:, :, k] + dt * k3, T + dt, k, x, y, a, b, u)                          # Fourth RK4 slope from the endpoint state.
        u_ap[1:-1, 1:-1, k + 1] = u_ap[1:-1, 1:-1, k] + (dt / 6) * (                            # Advance 2D interior nodes with RK4 weights.
            k1[1:-1, 1:-1] + 2 * k2[1:-1, 1:-1] + 2 * k3[1:-1, 1:-1] + k4[1:-1, 1:-1]           # Classical RK4 weighted slope sum.
        )
    return u_ap[1:-1, 1:-1, :]


def rhs_2D(u_ap, T, k, x, y, a, b, u):
    """
    Right-Hand Side (RHS) of the 2D Advection ODE system.

    Evaluates the spatial derivatives using central finite differences 
    to form the RHS of the ODE system for the Method of Lines.

    Parameters
    ----------
    u_ap : np.ndarray
        2D array of the solution values at the current time step.
    T : np.ndarray or float
        Time grid array or current time scalar.
    k : int
        Current time index.
    x : np.ndarray
        2D array representing the x-coordinates of the spatial mesh.
    y : np.ndarray
        2D array representing the y-coordinates of the spatial mesh.
    a : float
        Advective velocity in the x direction.
    b : float
        Advective velocity in the y direction.
    u : Callable
        Function defining the boundary conditions.

    Returns
    -------
    np.ndarray
        2D array of the evaluated spatial derivatives.
    """
    m, n = x.shape                                                                              # Number of mesh rows and columns.
    dx   = x[0, 1] - x[0, 0]                                                                    # Uniform column spacing in the x direction.
    dy   = y[1, 0] - y[0, 0]                                                                    # Uniform row spacing in the y direction.
    s    = np.zeros([m, n])                                                                     # Allocate the 2D RHS array.

    for i in range(m):
        s[i, 0]  = u(x[i, 0], y[i, 0], T[k], a, b)                                              # Left boundary at the first x-column.
        s[i, -1] = u(x[i, -1], y[i, -1], T[k], a, b)                                            # Right boundary at the last x-column.
    for j in range(n):
        s[0, j]  = u(x[0, j], y[0, j], T[k], a, b)                                              # Bottom boundary at the first y-row.
        s[-1, j] = u(x[-1, j], y[-1, j], T[k], a, b)                                            # Top boundary at the last y-row.

    for i in range(1, m - 1):
        for j in range(1, n - 1):
            s[i, j] = -(a / (2 * dx)) * (u_ap[i, j + 1] - u_ap[i, j - 1]) + -(b / (2 * dy)) * ( # Add centered x and y advection terms.
                u_ap[i + 1, j] - u_ap[i - 1, j]                                                 # Centered first derivative in y.
            )

    return s


if __name__ == "__main__":
    print("This module defines Runge-Kutta helpers for advection. Import it from solver or test modules.")  # Inform users that this file provides reusable integrators.
