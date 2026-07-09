"""
=========================================================================================
Tests for Runge-Kutta Time Integrator RHS Functions
=========================================================================================

This module provides regression tests for 2D right-hand-side helpers used by
Runge-Kutta time integrators on rectangular spatial grids.

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

from Common.TimeIntegrators import Runge_Kutta_AdvDif
from Common.TimeIntegrators import Runge_Kutta_Advection
from Common.TimeIntegrators import Runge_Kutta_Diffusion


def test_runge_kutta_rhs_2d_supports_rectangular_grids():
    """
    Verify that 2D Runge-Kutta RHS helpers preserve rectangular grid shapes.

    Parameters
    ----------
    None
        This test builds its own rectangular mesh and benchmark functions.

    Returns
    -------
    None
        Assertions validate the returned RHS shapes.
    """
    xs = np.linspace(0, 1, 6)
    ys = np.linspace(0, 1, 4)
    x, y = np.meshgrid(xs, ys)
    T = np.linspace(0, 1, 3)
    u_ap = np.zeros(x.shape)

    u_diff = lambda x, y, t, nu: x + y + t + nu
    u_adv = lambda x, y, t, a, b: x + y + t + a + b
    u_advdif = lambda x, y, t, nu, a, b: x + y + t + nu + a + b

    rhs_diff = Runge_Kutta_Diffusion.rhs_2D(u_ap, T, 0, x, y, 0.1, u_diff)
    rhs_adv = Runge_Kutta_Advection.rhs_2D(u_ap, T, 0, x, y, 0.1, 0.2, u_adv)
    rhs_advdif = Runge_Kutta_AdvDif.rhs_2D(u_ap, T, 0, x, y, 0.1, 0.2, 0.3, u_advdif)

    assert rhs_diff.shape == x.shape
    assert rhs_adv.shape == x.shape
    assert rhs_advdif.shape == x.shape


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
