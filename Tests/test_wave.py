"""
=========================================================================================
Regression Tests for Wave Solvers
=========================================================================================

This module provides regression tests for 1D and 2D wave-equation solvers,
including accuracy against exact solutions and matrix-vs-iterative consistency.

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
import sys                                                                                      # Provides access to the Python module search path.
from pathlib import Path                                                                        # Builds portable paths for local imports.
import numpy as np                                                                              # Provides vectorized arrays and numerical functions.

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)                                      # Stores the repository root path.
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)                                                            # Makes local project packages importable.

import CFDM.Wave as Wave                                                                        # Imports the wave finite-difference solvers.


def _mae_rmse(exact, approx):
    """
    Compute mean absolute error and root mean square error.

    Parameters
    ----------
    exact : np.ndarray
        Reference solution values.
    approx : np.ndarray
        Approximate solution values.

    Returns
    -------
    tuple[float, float]
        Mean absolute error and root mean square error.
    """
    diff = exact - approx                                                                       # Compute pointwise error.
    mae  = np.mean(np.abs(diff))                                                                # Compute mean absolute error.
    rmse = np.sqrt(np.mean(diff**2))                                                            # Compute root mean square error.
    return mae, rmse                                                                            # Return both metrics.


def test_wave_1d_matches_exact_solution_with_small_mae_rmse():
    """
    Verify 1D wave accuracy against an exact standing-wave solution.

    Parameters
    ----------
    None
        This test builds a stable 1D wave benchmark.

    Returns
    -------
    None
        Assertions validate CFL, MAE, and RMSE against the exact solution.
    """
    nodes      = 81                                                                             # Number of spatial nodes.
    time_steps = 240                                                                            # Number of temporal nodes.
    c          = 0.5                                                                            # Wave propagation speed.
    x          = np.linspace(0, 1, nodes)                                                       # Build the unit interval mesh.
    dt         = 1 / (time_steps - 1)                                                           # Compute time step over [0, 1].
    dx         = x[1] - x[0]                                                                    # Compute spatial step.
    u          = lambda x, t, c: np.sin(np.pi*x) * np.cos(np.pi*c*t)                            # Exact 1D standing-wave solution.

    u_ap, u_ex = Wave.Wave1D(x, time_steps, u, c)                                               # Solve the 1D wave problem.
    mae, rmse  = _mae_rmse(u_ex, u_ap)                                                          # Compute global transient errors.

    assert c*dt/dx <= 1                                                                         # Confirm the 1D CFL condition.
    assert mae < 5e-5                                                                           # Validate mean absolute error.
    assert rmse < 6e-5                                                                          # Validate root mean square error.


def test_wave_2d_matches_exact_solution_with_small_mae_rmse():
    """
    Verify 2D wave accuracy against an exact rectangular standing-wave solution.

    Parameters
    ----------
    None
        This test builds a stable 2D wave benchmark.

    Returns
    -------
    None
        Assertions validate CFL, MAE, and RMSE against the exact solution.
    """
    x_1d       = np.linspace(0, 1, 35)                                                          # Build x coordinates.
    y_1d       = np.linspace(0, 1, 29)                                                          # Build y coordinates.
    x, y       = np.meshgrid(x_1d, y_1d)                                                        # Build a rectangular 2D mesh.
    time_steps = 420                                                                            # Number of temporal nodes.
    c          = 0.45                                                                           # Wave propagation speed.
    dt         = 1 / (time_steps - 1)                                                           # Compute time step over [0, 1].
    dx         = x[0, 1] - x[0, 0]                                                              # Compute x spacing.
    dy         = y[1, 0] - y[0, 0]                                                              # Compute y spacing.
    omega      = np.pi*c*np.sqrt(2)                                                             # Standing-wave angular frequency.
    u          = lambda x, y, t, c: np.sin(np.pi*x) * np.sin(np.pi*y) * np.cos(omega*t)         # Exact 2D standing-wave solution.

    u_ap, u_ex = Wave.Wave2D(x, y, time_steps, u, c)                                            # Solve the 2D wave problem.
    mae, rmse  = _mae_rmse(u_ex, u_ap)                                                          # Compute global transient errors.

    assert c*dt*np.sqrt(1/dx**2 + 1/dy**2) <= 1                                                 # Confirm the 2D CFL condition.
    assert mae < 2e-4                                                                           # Validate mean absolute error.
    assert rmse < 3e-4                                                                          # Validate root mean square error.


def test_wave_matrix_and_iterative_solvers_are_consistent():
    """
    Verify that matrix and node-wise wave solvers produce the same results.

    Parameters
    ----------
    None
        This test builds 1D and 2D stable wave benchmarks.

    Returns
    -------
    None
        Assertions validate agreement between matrix and stencil implementations.
    """
    c    = 0.4                                                                                  # Wave propagation speed.
    x_1d = np.linspace(0, 1, 31)                                                                # Build 1D mesh.
    u_1d = lambda x, t, c: np.sin(np.pi*x) * np.cos(np.pi*c*t)                                  # Exact 1D solution.

    u_fd_1d, _ = Wave.Wave1D(x_1d, 160, u_1d, c)                                                # Solve with matrix formulation.
    u_gs_1d, _ = Wave.Wave1D_iter(x_1d, 160, u_1d, c)                                           # Solve with node-wise formulation.

    xs         = np.linspace(0, 1, 17)                                                          # Build x coordinates.
    ys         = np.linspace(0, 1, 13)                                                          # Build y coordinates.
    x_2d, y_2d = np.meshgrid(xs, ys)                                                            # Build rectangular mesh.
    omega      = np.pi*c*np.sqrt(2)                                                             # Standing-wave angular frequency.
    u_2d       = lambda x, y, t, c: np.sin(np.pi*x) * np.sin(np.pi*y) * np.cos(omega*t)         # Exact 2D solution.

    u_fd_2d, _ = Wave.Wave2D(x_2d, y_2d, 120, u_2d, c)                                          # Solve with matrix formulation.
    u_gs_2d, _ = Wave.Wave2D_iter(x_2d, y_2d, 120, u_2d, c)                                     # Solve with node-wise formulation.

    assert np.allclose(u_fd_1d, u_gs_1d, atol=1e-14)                                            # Validate 1D solver consistency.
    assert np.allclose(u_fd_2d, u_gs_2d, atol=1e-14)                                            # Validate 2D solver consistency.
