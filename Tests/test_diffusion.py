"""
=========================================================================================
Tests for CFDM Diffusion Solvers
=========================================================================================

This module provides regression tests for diffusion solvers, including explicit
stability checks, implicit matrix checks, and rectangular-grid consistency.

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

from CFDM import Diffusion


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


def test_diffusion_1d_matches_exact_solution_with_small_mae_rmse():
    """
    Verify 1D diffusion accuracy against an exact transient solution.

    Parameters
    ----------
    None
        This test builds a stable 1D diffusion benchmark.

    Returns
    -------
    None
        Assertions validate MAE and RMSE against the exact solution.
    """
    nodes = 31
    time_steps = 1600
    diffusivity = 0.02
    x = np.linspace(0, 1, nodes)
    u = lambda x, t, v: np.exp(-np.pi**2*v*t) * np.cos(np.pi*x)

    u_ap, u_ex = Diffusion.Diffusion1D(x, time_steps, u, diffusivity)
    mae, rmse = _mae_rmse(u_ex, u_ap)

    assert mae < 5e-5
    assert rmse < 7e-5


def test_diffusion_2d_matches_exact_solution_with_small_mae_rmse():
    """
    Verify 2D diffusion accuracy against an exact transient solution.

    Parameters
    ----------
    None
        This test builds a stable 2D diffusion benchmark.

    Returns
    -------
    None
        Assertions validate MAE and RMSE against the exact solution.
    """
    nodes = 17
    time_steps = 2500
    diffusivity = 0.005
    x_1d = np.linspace(0, 1, nodes)
    y_1d = np.linspace(0, 1, nodes)
    x, y = np.meshgrid(x_1d, y_1d)
    u = lambda x, y, t, v: np.exp(-2*np.pi**2*v*t) * np.cos(np.pi*x) * np.cos(np.pi*y)

    u_ap, u_ex = Diffusion.Diffusion2D(x, y, time_steps, u, diffusivity)
    mae, rmse = _mae_rmse(u_ex, u_ap)

    assert mae < 5e-5
    assert rmse < 8e-5


def test_diffusion_2d_explicit_respects_documented_cfl_and_remains_finite():
    """
    Verify that explicit 2D diffusion remains finite under the CFL condition.

    Parameters
    ----------
    None
        This test builds its own stable 2D diffusion benchmark.

    Returns
    -------
    None
        Assertions validate that the approximate solution contains no NaN or Inf.
    """
    m = 17
    x = np.linspace(0, 1, m)
    y = np.linspace(0, 1, m)
    x, y = np.meshgrid(x, y)
    t = 201
    v = 0.01
    dx = x[0, 1] - x[0, 0]
    dy = y[1, 0] - y[0, 0]
    dt = 1 / (t - 1)
    cfl = v*dt/dx**2 + v*dt/dy**2
    
    u = lambda x, y, t, v: np.exp(-2*np.pi**2*v*t) * np.cos(np.pi*x) * np.cos(np.pi*y)
    
    u_ap, u_ex = Diffusion.Diffusion2D(x, y, t, u, v, implicit=False)
    
    assert cfl <= 0.5
    assert not np.isnan(u_ap).any(), "Diffusion 2D explicit produced NaNs"
    assert not np.isinf(u_ap).any(), "Diffusion 2D explicit produced Infs"

def test_diffusion_2d_implicit():
    """
    Verify that the implicit 2D diffusion matrix method remains finite.

    Parameters
    ----------
    None
        This test builds its own 2D diffusion benchmark.

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
    v = 0.2
    
    u = lambda x, y, t, v: np.exp(-2*np.pi**2*v*t) * np.cos(np.pi*x) * np.cos(np.pi*y)
    
    u_ap, u_ex = Diffusion.Diffusion2D(x, y, t, u, v, implicit=True)
    
    assert not np.isnan(u_ap).any(), "Diffusion 2D implicit produced NaNs"
    assert not np.isinf(u_ap).any(), "Diffusion 2D implicit produced Infs"


def test_diffusion_2d_matrix_matches_iterative_on_rectangular_grid():
    """
    Verify that matrix and iterative 2D diffusion updates agree on a rectangular mesh.

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
    v = 0.01

    u = lambda x, y, t, v: np.exp(-t) * (x + 2*y + x*y)

    u_matrix, _ = Diffusion.Diffusion2D(x, y, t, u, v, implicit=False)
    u_iter, _ = Diffusion.Diffusion2D_iter(x, y, t, u, v, implicit=False)

    assert u_matrix.shape == x.shape + (t,)
    assert np.allclose(u_matrix, u_iter, atol=1e-12)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
