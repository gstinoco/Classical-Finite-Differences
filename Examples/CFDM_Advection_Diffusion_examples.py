"""
=========================================================================================
Examples for CFDM Advection-Diffusion solvers
=========================================================================================

This script demonstrates transient 1D/2D advection-diffusion solvers.

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
from pathlib import Path                                                                        # Builds portable paths for project and output folders.
import numpy as np                                                                              # Provides vectorized arrays and elementary functions.

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)                                      # Stores the repository root that contains CFDM and Common.
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)                                                            # Makes local project packages importable when running this file directly.

import CFDM.Advection_Diffusion as CAdvDiff                                                     # Imports the advection-diffusion finite-difference solvers.
import Common.ExampleTools as ExampleTools                                                      # Imports shared helpers for metrics, headers, and output paths.
import Common.Graphs as Graphs                                                                  # Imports shared plotting routines for transient examples.

RESULTS_DIR = Path("Results/Advection_Diffusion")                                               # Defines the default folder for advection-diffusion figures.


def run_1d_example(show=False, save_path=RESULTS_DIR, nodes=21, time_steps=200, nu=0.1, a=0.1):
    """
    Run the 1D transient advection-diffusion example.

    Parameters
    ----------
    show : bool
        If True, display the generated figures.
    save_path : str, pathlib.Path, or None
        Directory where figures are saved. If None, figures are not saved.
    nodes : int
        Number of nodes in the 1D spatial mesh.
    time_steps : int
        Number of temporal nodes used over the interval [0, 1].
    nu : float
        Diffusion coefficient used in the exact solution and numerical solver.
    a : float
        Advection velocity used in the exact solution and numerical solver.

    Returns
    -------
    None
        Results are printed and figures are optionally displayed or saved.
    """
    def u_exact(x, t, nu_val, a_val):
        """
        Evaluate the exact 1D advection-diffusion benchmark solution.

        Parameters
        ----------
        x : np.ndarray or float
            Spatial coordinate values.
        t : np.ndarray or float
            Time coordinate values.
        nu_val : float
            Diffusion coefficient.
        a_val : float
            Advection velocity.

        Returns
        -------
        np.ndarray or float
            Exact solution evaluated at the requested coordinates.
        """
        return (1 / np.sqrt(4*t + 1)) * np.exp(-((x - 0.5 - a_val*t)**2) / (nu_val * (4*t + 1)))

    x = np.linspace(0, 1, nodes)                                                                # Builds the 1D spatial mesh over the unit interval.

    u_explicit_fd, u_ex = CAdvDiff.AdvectionDiffusion1D(x, time_steps, u_exact, nu, a, implicit=False)
                                                                                                # Solves the 1D explicit problem with the FD matrix formulation.
    u_explicit_gs, _ = CAdvDiff.AdvectionDiffusion1D_iter(x, time_steps, u_exact, nu, a, implicit=False)
                                                                                                # Solves the 1D explicit problem with the GS iterative formulation.
    u_cn_fd, _ = CAdvDiff.AdvectionDiffusion1D(x, time_steps, u_exact, nu, a, implicit=True, lam=0.5)
                                                                                                # Solves the 1D Crank-Nicolson problem with the FD matrix formulation.
    u_cn_gs, _ = CAdvDiff.AdvectionDiffusion1D_iter(x, time_steps, u_exact, nu, a, implicit=True, lam=0.5)
                                                                                                # Solves the 1D Crank-Nicolson problem with the GS iterative formulation.

    ExampleTools.print_metrics("1D Results (Transient)", [                                      # Prints a common metric table over the full 1D time history.
        ExampleTools.transient_row("Explicit FD", u_ex, u_explicit_fd),                         # Adds explicit FD transient error metrics.
        ExampleTools.transient_row("Explicit GS", u_ex, u_explicit_gs),                         # Adds explicit GS transient error metrics.
        ExampleTools.transient_row("Crank-Nicolson FD", u_ex, u_cn_fd),                         # Adds Crank-Nicolson FD transient error metrics.
        ExampleTools.transient_row("Crank-Nicolson GS", u_ex, u_cn_gs),                         # Adds Crank-Nicolson GS transient error metrics.
    ])                                                                                          # Closes the 1D metric table definition.

    Graphs.Transient_1D(x, u_explicit_fd, u_ex, title="Advection-Diffusion 1D - Explicit FD", show=show, save_path=ExampleTools.output_path(save_path, "1D_FD_Explicit.gif"))
                                                                                                # Animates the explicit FD approximation against the exact solution.
    Graphs.Transient_1D(x, u_explicit_gs, u_ex, title="Advection-Diffusion 1D - Explicit GS", show=show, save_path=ExampleTools.output_path(save_path, "1D_GS_Explicit.gif"))
                                                                                                # Animates the explicit GS approximation against the exact solution.
    Graphs.Transient_1D(x, u_cn_fd, u_ex, title="Advection-Diffusion 1D - Crank-Nicolson FD", show=show, save_path=ExampleTools.output_path(save_path, "1D_FD_CN.gif"))
                                                                                                # Animates the Crank-Nicolson FD approximation against the exact solution.
    Graphs.Transient_1D(x, u_cn_gs, u_ex, title="Advection-Diffusion 1D - Crank-Nicolson GS", show=show, save_path=ExampleTools.output_path(save_path, "1D_GS_CN.gif"))
                                                                                                # Animates the Crank-Nicolson GS approximation against the exact solution.


def run_2d_example(show=False, save_path=RESULTS_DIR, nodes=21, time_steps=200, nu=0.1, a=0.1, b=0.1):
    """
    Run the 2D transient advection-diffusion example.

    Parameters
    ----------
    show : bool
        If True, display the generated figures.
    save_path : str, pathlib.Path, or None
        Directory where figures are saved. If None, figures are not saved.
    nodes : int
        Number of nodes per direction in the 2D spatial mesh.
    time_steps : int
        Number of temporal nodes used over the interval [0, 1].
    nu : float
        Diffusion coefficient used in the exact solution and numerical solver.
    a : float
        Advection velocity in the x direction.
    b : float
        Advection velocity in the y direction.

    Returns
    -------
    None
        Results are printed and figures are optionally displayed or saved.
    """
    def u_exact(x, y, t, nu_val, a_val, b_val):
        """
        Evaluate the exact 2D advection-diffusion benchmark solution.

        Parameters
        ----------
        x, y : np.ndarray or float
            Spatial coordinate values.
        t : np.ndarray or float
            Time coordinate values.
        nu_val : float
            Diffusion coefficient.
        a_val : float
            Advection velocity in the x direction.
        b_val : float
            Advection velocity in the y direction.

        Returns
        -------
        np.ndarray or float
            Exact solution evaluated at the requested coordinates.
        """
        return (1 / (4*t + 1)) * np.exp(-((x - 0.5 - a_val*t)**2 + (y - 0.5 - b_val*t)**2) / (nu_val * (4*t + 1)))

    x_1d = np.linspace(0, 1, nodes)                                                             # Builds the x-coordinate mesh nodes.
    y_1d = np.linspace(0, 1, nodes)                                                             # Builds the y-coordinate mesh nodes.
    x, y = np.meshgrid(x_1d, y_1d)                                                              # Expands 1D coordinate vectors into a rectangular 2D mesh.

    u_explicit_fd, u_ex = CAdvDiff.AdvectionDiffusion2D(x, y, time_steps, u_exact, nu, a, b, implicit=False)
                                                                                                # Solves the 2D explicit problem with the FD matrix formulation.
    u_explicit_gs, _ = CAdvDiff.AdvectionDiffusion2D_iter(x, y, time_steps, u_exact, nu, a, b, implicit=False)
                                                                                                # Solves the 2D explicit problem with the GS iterative formulation.
    u_cn_fd, _ = CAdvDiff.AdvectionDiffusion2D(x, y, time_steps, u_exact, nu, a, b, implicit=True, lam=0.5)
                                                                                                # Solves the 2D Crank-Nicolson problem with the FD matrix formulation.
    u_cn_gs, _ = CAdvDiff.AdvectionDiffusion2D_iter(x, y, time_steps, u_exact, nu, a, b, implicit=True, lam=0.5)
                                                                                                # Solves the 2D Crank-Nicolson problem with the GS iterative formulation.

    ExampleTools.print_metrics("2D Results (Transient)", [                                      # Prints a common metric table over the full 2D time history.
        ExampleTools.transient_row("Explicit FD", u_ex, u_explicit_fd),                         # Adds explicit FD transient error metrics.
        ExampleTools.transient_row("Explicit GS", u_ex, u_explicit_gs),                         # Adds explicit GS transient error metrics.
        ExampleTools.transient_row("Crank-Nicolson FD", u_ex, u_cn_fd),                         # Adds Crank-Nicolson FD transient error metrics.
        ExampleTools.transient_row("Crank-Nicolson GS", u_ex, u_cn_gs),                         # Adds Crank-Nicolson GS transient error metrics.
    ])                                                                                          # Closes the 2D metric table definition.

    Graphs.Transient_2D(x, y, u_explicit_fd, u_ex, title="Advection-Diffusion 2D - Explicit FD", show=show, save_path=ExampleTools.output_path(save_path, "2D_FD_Explicit.gif"))
                                                                                                # Animates the explicit FD approximation against the exact surface.
    Graphs.Transient_2D(x, y, u_explicit_gs, u_ex, title="Advection-Diffusion 2D - Explicit GS", show=show, save_path=ExampleTools.output_path(save_path, "2D_GS_Explicit.gif"))
                                                                                                # Animates the explicit GS approximation against the exact surface.
    Graphs.Transient_2D(x, y, u_cn_fd, u_ex, title="Advection-Diffusion 2D - Crank-Nicolson FD", show=show, save_path=ExampleTools.output_path(save_path, "2D_FD_CN.gif"))
                                                                                                # Animates the Crank-Nicolson FD approximation against the exact surface.
    Graphs.Transient_2D(x, y, u_cn_gs, u_ex, title="Advection-Diffusion 2D - Crank-Nicolson GS", show=show, save_path=ExampleTools.output_path(save_path, "2D_GS_CN.gif"))
                                                                                                # Animates the Crank-Nicolson GS approximation against the exact surface.


def main(show=False, save_path=RESULTS_DIR, nodes_1d=21, nodes_2d=21, time_steps=200):
    """
    Run all advection-diffusion examples.

    Parameters
    ----------
    show : bool
        If True, display the generated figures.
    save_path : str, pathlib.Path, or None
        Directory where figures are saved. If None, figures are not saved.
    nodes_1d : int
        Number of nodes used in 1D examples.
    nodes_2d : int
        Number of nodes per direction used in 2D examples.
    time_steps : int
        Number of temporal nodes used over the interval [0, 1].

    Returns
    -------
    None
        All advection-diffusion example blocks are executed.
    """
    ExampleTools.print_example_header("Advection-Diffusion Equation CFDM Solvers - Examples")   # Prints the standard section header for this script.
    run_1d_example(show=show, save_path=save_path, nodes=nodes_1d, time_steps=time_steps)       # Runs the 1D advection-diffusion benchmark.
    run_2d_example(show=show, save_path=save_path, nodes=nodes_2d, time_steps=time_steps)       # Runs the 2D advection-diffusion benchmark.


if __name__ == "__main__":
    main()                                                                                      # Executes all advection-diffusion examples when launched as a script.
