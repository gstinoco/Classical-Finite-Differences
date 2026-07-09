"""
=========================================================================================
Tests for CFDM Poisson Solvers
=========================================================================================

This module provides regression tests for Poisson solvers, including rectangular
meshes, sign convention checks, and Neumann boundary condition consistency.

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
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CFDM import Poisson


def test_poisson_2d_rectangular_grid_matches_exact_quadratic():
    """
    Verify that Poisson2D solves an exact quadratic solution on a rectangular mesh.

    Parameters
    ----------
    None
        This test builds its own mesh, exact solution, and source term.

    Returns
    -------
    None
        Assertions validate the shape and numerical solution.
    """
    xs = np.linspace(0, 1, 5)
    ys = np.linspace(0, 1, 4)
    x, y = np.meshgrid(xs, ys)

    phi = lambda x, y: x**2 + y**2
    f = lambda x, y: -4 + 0*x

    phi_ap, phi_ex = Poisson.Poisson2D(x, y, phi, f)

    assert phi_ap.shape == x.shape
    assert np.allclose(phi_ap, phi_ex, atol=1e-12)


def test_poisson_2d_non_symmetric_rectangular_polynomial_is_exact():
    """
    Verify Poisson2D on a non-symmetric rectangular polynomial benchmark.

    Parameters
    ----------
    None
        This test builds a rectangular mesh with dx != dy and mixed x-y terms.

    Returns
    -------
    None
        Assertions validate exact recovery and help detect crossed 2D axes.
    """
    xs = np.linspace(0, 1.2, 7)
    ys = np.linspace(0, 0.7, 5)
    x, y = np.meshgrid(xs, ys)

    phi = lambda x, y: 2*x**2 + 3*y**2 + x*y + 0.5*x - 0.25*y + 1
    f = lambda x, y: -10 + 0*x

    phi_ap, phi_ex = Poisson.Poisson2D(x, y, phi, f)

    assert phi_ap.shape == x.shape
    assert np.allclose(phi_ap, phi_ex, atol=1e-11)


def test_poisson_1d_and_2d_use_delta_phi_equals_minus_f_convention():
    """
    Verify that 1D and 2D Poisson solvers use the convention Delta phi = -f.

    Parameters
    ----------
    None
        This test builds exact polynomial solutions and matching source terms.

    Returns
    -------
    None
        Assertions validate agreement with the exact solutions.
    """
    x_1d = np.linspace(0, 1, 7)
    phi_1d = lambda x: x**2
    f_1d = lambda x: -2*np.ones_like(x)

    phi_ap_1d, phi_ex_1d = Poisson.Poisson1D(x_1d, phi_1d, f_1d)

    xs = np.linspace(0, 1, 6)
    ys = np.linspace(0, 1, 5)
    x_2d, y_2d = np.meshgrid(xs, ys)
    phi_2d = lambda x, y: x**2 + y**2
    f_2d = lambda x, y: -4 + 0*x

    phi_ap_2d, phi_ex_2d = Poisson.Poisson2D(x_2d, y_2d, phi_2d, f_2d)

    assert np.allclose(phi_ap_1d, phi_ex_1d, atol=1e-12)
    assert np.allclose(phi_ap_2d, phi_ex_2d, atol=1e-12)


def test_poisson_1d_neumann_iterative_matches_matrix_or_exact_solution():
    """
    Verify Neumann iterative solvers against matrix or exact reference solutions.

    Parameters
    ----------
    None
        This test builds a 1D Neumann/Dirichlet benchmark problem.

    Returns
    -------
    None
        Assertions validate iterative convergence for the Neumann variants.
    """
    x = np.linspace(0, 1, 41)
    phi = lambda x: x**2 - x + 1
    f = lambda x: -2*np.ones_like(x)
    sig = -1.0
    beta = 1.0

    n1_matrix, _ = Poisson.Poisson1D_Neumann_1(x, phi, f, sig, beta)
    n1_iter, _ = Poisson.Poisson1D_Neumann_1_iter(x, phi, f, sig, beta)
    n2_iter, phi_ex = Poisson.Poisson1D_Neumann_2_iter(x, phi, f, sig, beta)
    n3_iter, _ = Poisson.Poisson1D_Neumann_3_iter(x, phi, f, sig, beta)

    assert np.allclose(n1_iter, n1_matrix, atol=1e-9)
    assert np.allclose(n2_iter, phi_ex, atol=1e-9)
    assert np.allclose(n3_iter, phi_ex, atol=1e-9)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
