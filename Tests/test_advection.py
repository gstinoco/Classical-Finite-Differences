"""
=========================================================================================
Tests for CFDM Advection Solvers
=========================================================================================

This module provides regression tests for advection solvers, including stable
upwind regimes, boundary preservation, and finite-value 2D solutions.

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

from CFDM import Advection


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


def test_advection_1d_stability():
    """
    Verify that the stable 1D FTBS advection method remains finite.

    Parameters
    ----------
    None
        This test builds its own stable 1D advection benchmark.

    Returns
    -------
    None
        Assertions validate that the approximate solution contains no NaN or Inf.
    """
    m = 21
    x = np.linspace(0, 1, m)
    t = 100
    a = 0.5
    dx = x[1] - x[0]
    dt = 1 / (t - 1)
    cfl = abs(a)*dt/dx
    u = lambda x, t, a: np.exp(-(x - 0.5 - a*t)**2 / 0.01)
    
    u_ap, u_ex = Advection.Advection1D(x, t, u, a, method='FTBS')

    assert cfl <= 1
    assert not np.isnan(u_ap).any(), "FTBS produced NaNs"
    assert not np.isinf(u_ap).any(), "FTBS produced Infs"


def test_advection_1d_upwind_matches_exact_solution_with_documented_cfl():
    """
    Verify 1D upwind advection accuracy against an exact translated pulse.

    Parameters
    ----------
    None
        This test builds stable FTBS and FTFS benchmarks.

    Returns
    -------
    None
        Assertions validate CFL, MAE, and RMSE against the exact solution.
    """
    nodes = 51
    time_steps = 401
    speed = 0.1
    x = np.linspace(0, 1, nodes)
    dx = x[1] - x[0]
    dt = 1 / (time_steps - 1)
    u = lambda x, t, a: np.exp(-((x - 0.35 - a*t)**2) / 0.02)

    cases = [
        ('FTBS', speed),
        ('FTFS', -speed),
    ]

    for method, a in cases:
        u_ap, u_ex = Advection.Advection1D_iter(x, time_steps, u, a, method=method)
        mae, rmse = _mae_rmse(u_ex, u_ap)

        assert abs(a)*dt/dx <= 1
        assert mae < 1.5e-2
        assert rmse < 2.5e-2


def test_advection_1d_upwind_methods_preserve_dirichlet_boundaries():
    """
    Verify that 1D upwind schemes preserve imposed Dirichlet boundary values.

    Parameters
    ----------
    None
        This test builds FTBS and FTFS matrix/iterative cases.

    Returns
    -------
    None
        Assertions validate both boundary traces against the exact solution.
    """
    m = 21
    x = np.linspace(0, 1, m)
    t = 20
    a = 0.5
    u = lambda x, t, a: x + t + a

    cases = [
        (Advection.Advection1D, 'FTBS', a),
        (Advection.Advection1D, 'FTFS', -a),
        (Advection.Advection1D_iter, 'FTBS', a),
        (Advection.Advection1D_iter, 'FTFS', -a),
    ]

    for solver, method, speed in cases:
        u_ap, u_ex = solver(x, t, u, speed, method=method)
        assert np.allclose(u_ap[0, :], u_ex[0, :])
        assert np.allclose(u_ap[-1, :], u_ex[-1, :])

def test_advection_2d_stability():
    """
    Verify that the stable 2D FTBS advection method remains finite.

    Parameters
    ----------
    None
        This test builds its own stable 2D advection benchmark.

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
    a = 0.2
    b = 0.2
    dx = x[0, 1] - x[0, 0]
    dy = y[1, 0] - y[0, 0]
    dt = 1 / (t - 1)
    cfl = abs(a)*dt/dx + abs(b)*dt/dy
    u = lambda x, y, t, a, b: np.exp(-((x - 0.5 - a*t)**2 + (y - 0.5 - b*t)**2) / 0.01)
    
    u_ap, u_ex = Advection.Advection2D(x, y, t, u, a, b, method='FTBS')

    assert cfl <= 1
    assert not np.isnan(u_ap).any(), "FTBS 2D produced NaNs"
    assert not np.isinf(u_ap).any(), "FTBS 2D produced Infs"


def test_advection_2d_matrix_matches_iterative_on_non_symmetric_rectangular_grid():
    """
    Verify 2D advection matrix and iterative updates on a non-symmetric grid.

    Parameters
    ----------
    None
        This test builds a rectangular mesh with unequal advection speeds.

    Returns
    -------
    None
        Assertions validate consistency and help detect crossed 2D axes.
    """
    xs = np.linspace(0, 1, 9)
    ys = np.linspace(0, 1, 7)
    x, y = np.meshgrid(xs, ys)
    time_steps = 30
    a = 0.1
    b = 0.07
    u = lambda x, y, t, a, b: 1 + 2*x + 3*y + 4*x*y + a*t - b*t

    u_matrix, _ = Advection.Advection2D(x, y, time_steps, u, a, b, method='FTBS')
    u_iter, _ = Advection.Advection_2D_iter(x, y, time_steps, u, a, b, method='FTBS')

    assert u_matrix.shape == x.shape + (time_steps,)
    assert np.allclose(u_matrix, u_iter, atol=1e-12)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
