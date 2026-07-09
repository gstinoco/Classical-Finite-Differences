"""
=========================================================================================
Tests for CFDM Advection-Diffusion Solvers
=========================================================================================

This module provides regression tests for advection-diffusion solvers, including
finite-value stability checks and matrix-vs-iterative consistency on rectangular meshes.

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
import numpy as np
import pytest
import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CFDM import Advection_Diffusion


def _mae_rmse(y_true, y_pred):
    """
    Compute MAE and RMSE for compact numerical accuracy assertions.

    Parameters
    ----------
    y_true : numpy.ndarray
        Reference solution values.
    y_pred : numpy.ndarray
        Approximate solution values.

    Returns
    -------
    tuple[float, float]
        Mean absolute error and root mean square error.
    """
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))

    return mae, rmse


def test_advection_diffusion_1d_matches_exact_solution_with_small_mae_rmse():
    """
    Verify 1D advection-diffusion accuracy against an exact transient solution.

    Parameters
    ----------
    None
        This test builds a stable 1D advection-diffusion benchmark.

    Returns
    -------
    None
        Assertions validate MAE and RMSE against the exact solution.
    """
    nodes = 31
    time_steps = 1000
    diffusivity = 0.005
    speed = 0.05
    x = np.linspace(0, 1, nodes)
    u = lambda x, t, v, a: np.exp(-np.pi**2*v*t) * np.cos(np.pi*(x - a*t))

    u_ap, u_ex = Advection_Diffusion.AdvectionDiffusion1D(x, time_steps, u, diffusivity, speed, implicit=False)
    mae, rmse = _mae_rmse(u_ex, u_ap)

    assert mae < 1.2e-4
    assert rmse < 1.5e-4


def test_advection_diffusion_2d_matches_exact_solution_with_small_mae_rmse():
    """
    Verify 2D advection-diffusion accuracy against an exact transient solution.

    Parameters
    ----------
    None
        This test builds a stable 2D advection-diffusion benchmark.

    Returns
    -------
    None
        Assertions validate MAE and RMSE against the exact solution.
    """
    x_1d = np.linspace(0, 1, 13)
    y_1d = np.linspace(0, 1, 10)
    x, y = np.meshgrid(x_1d, y_1d)
    time_steps = 301
    diffusivity = 0.001
    speed_x = 0.02
    speed_y = 0.015
    u = lambda x, y, t, v, a, b: np.exp(-2*np.pi**2*v*t) * np.cos(np.pi*(x - a*t)) * np.cos(np.pi*(y - b*t))

    u_ap, u_ex = Advection_Diffusion.AdvectionDiffusion2D(x, y, time_steps, u, diffusivity, speed_x, speed_y, implicit=False)
    mae, rmse = _mae_rmse(u_ex, u_ap)

    assert mae < 2.5e-4
    assert rmse < 4e-4


def test_advection_diffusion_2d_explicit_respects_documented_cfl_and_remains_finite():
    """
    Verify that explicit iterative 2D advection-diffusion remains finite under CFL bounds.

    Parameters
    ----------
    None
        This test builds its own stable 2D advection-diffusion benchmark.

    Returns
    -------
    None
        Assertions validate that the approximate solution contains no NaN or Inf.
    """
    m = 21
    x = np.linspace(0, 1, m)
    y = np.linspace(0, 1, m)
    x, y = np.meshgrid(x, y)
    t = 100
    nu = 0.01
    a = 0.1
    b = 0.1
    dx = x[0, 1] - x[0, 0]
    dy = y[1, 0] - y[0, 0]
    dt = 1 / (t - 1)
    diffusive_cfl = nu*dt/dx**2 + nu*dt/dy**2
    advective_cfl = abs(a)*dt/dx + abs(b)*dt/dy
    
    u = lambda x, y, t, nu, a, b: (1 / np.sqrt(4 * t + 1)) * np.exp(-((x - 0.5 - a * t)**2 + (y - 0.5 - b * t)**2) / (nu * (4 * t + 1)))
    
    u_ap, u_ex = Advection_Diffusion.AdvectionDiffusion2D_iter(x, y, t, u, nu, a, b, implicit=False)
    
    assert diffusive_cfl <= 0.5
    assert advective_cfl <= 1
    assert not np.isnan(u_ap).any(), "Advection_Diffusion 2D iter produced NaNs"
    assert not np.isinf(u_ap).any(), "Advection_Diffusion 2D iter produced Infs"


def test_advection_diffusion_2d_matrix_matches_iterative_on_rectangular_grid():
    """
    Verify that matrix and iterative 2D advection-diffusion updates agree.

    Parameters
    ----------
    None
        This test builds a rectangular mesh and transient benchmark solution.

    Returns
    -------
    None
        Assertions validate output shape and matrix-vs-iterative consistency.
    """
    xs = np.linspace(0, 1, 6)
    ys = np.linspace(0, 1, 4)
    x, y = np.meshgrid(xs, ys)
    t = 5
    nu = 0.01
    a = 0.1
    b = 0.2

    u = lambda x, y, t, nu, a, b: np.exp(-t) * (x + 2*y + x*y)

    u_matrix, _ = Advection_Diffusion.AdvectionDiffusion2D(x, y, t, u, nu, a, b, implicit=False)
    u_iter, _ = Advection_Diffusion.AdvectionDiffusion2D_iter(x, y, t, u, nu, a, b, implicit=False)

    assert u_matrix.shape == x.shape + (t,)
    assert np.allclose(u_matrix, u_iter, atol=1e-12)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
