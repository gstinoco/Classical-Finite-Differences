"""
=========================================================================================
Examples for CFDM Diffusion solvers
=========================================================================================

This script demonstrates transient 1D/2D diffusion solvers.

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

import CFDM.Diffusion as CDiffusion                                                             # Imports the diffusion finite-difference solvers.
import Common.ExampleTools as ExampleTools                                                      # Imports shared helpers for metrics, headers, and output paths.
import Common.Graphs as Graphs                                                                  # Imports shared plotting routines for transient examples.

RESULTS_DIR = Path("Results/Diffusion")                                                         # Defines the default folder for diffusion figures.


def run_1d_example(show=False, save_path=RESULTS_DIR, nodes=21, time_steps=400, diffusivity=0.2):
    """
    Run the 1D transient diffusion example.

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
    diffusivity : float
        Diffusion coefficient used in the exact solution and numerical solver.

    Returns
    -------
    None
        Results are printed and figures are optionally displayed or saved.
    """
    u = lambda x, t, v: np.exp(-np.pi**2*v*t) * np.cos(np.pi*x)                                 # Defines the exact 1D heat-equation benchmark solution.
    x = np.linspace(0, 1, nodes)                                                                # Builds the 1D spatial mesh over the unit interval.

    u_fd, u_ex = CDiffusion.Diffusion1D(x, time_steps, u, diffusivity)                          # Solves the 1D transient problem with the matrix scheme.
    u_gs, _ = CDiffusion.Diffusion1D_iter(x, time_steps, u, diffusivity)                        # Solves the same 1D problem with Gauss-Seidel iterations.

    ExampleTools.print_metrics("1D Results (Transient)", [                                      # Prints a common metric table over the full 1D time history.
        ExampleTools.transient_row("FD", u_ex, u_fd),                                           # Adds matrix-scheme transient error metrics.
        ExampleTools.transient_row("GS", u_ex, u_gs),                                           # Adds iterative-scheme transient error metrics.
    ])                                                                                          # Closes the 1D metric table definition.

    Graphs.Transient_1D(x, u_fd, u_ex, title="Diffusion 1D - FD", show=show, save_path=ExampleTools.output_path(save_path, "1D_FD.gif"))
                                                                                                # Animates the matrix approximation against the exact solution.
    Graphs.Transient_1D(x, u_gs, u_ex, title="Diffusion 1D - GS", show=show, save_path=ExampleTools.output_path(save_path, "1D_GS.gif"))
                                                                                                # Animates the iterative approximation against the exact solution.


def run_2d_example(show=False, save_path=RESULTS_DIR, nodes=21, time_steps=400, diffusivity=0.2):
    """
    Run the 2D transient diffusion example.

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
    diffusivity : float
        Diffusion coefficient used in the exact solution and numerical solver.

    Returns
    -------
    None
        Results are printed and figures are optionally displayed or saved.
    """
    u = lambda x, y, t, v: np.exp(-2*np.pi**2*v*t) * np.cos(np.pi*x) * np.cos(np.pi*y)          # Defines the exact separable 2D diffusion solution.
    x_1d = np.linspace(0, 1, nodes)                                                             # Builds the x-coordinate mesh nodes.
    y_1d = np.linspace(0, 1, nodes)                                                             # Builds the y-coordinate mesh nodes.
    x, y = np.meshgrid(x_1d, y_1d)                                                              # Expands 1D coordinate vectors into a rectangular 2D mesh.

    u_fd, u_ex = CDiffusion.Diffusion2D(x, y, time_steps, u, diffusivity)                       # Solves the 2D transient problem with the matrix scheme.
    u_gs, _ = CDiffusion.Diffusion2D_iter(x, y, time_steps, u, diffusivity)                     # Solves the same 2D problem with Gauss-Seidel iterations.

    ExampleTools.print_metrics("2D Results (Transient)", [                                      # Prints a common metric table over the full 2D time history.
        ExampleTools.transient_row("FD", u_ex, u_fd),                                           # Adds matrix-scheme transient error metrics.
        ExampleTools.transient_row("GS", u_ex, u_gs),                                           # Adds iterative-scheme transient error metrics.
    ])                                                                                          # Closes the 2D metric table definition.

    Graphs.Transient_2D(x, y, u_fd, u_ex, title="Diffusion 2D - FD", show=show, save_path=ExampleTools.output_path(save_path, "2D_FD.gif"))
                                                                                                # Animates the matrix approximation against the exact surface.
    Graphs.Transient_2D(x, y, u_gs, u_ex, title="Diffusion 2D - GS", show=show, save_path=ExampleTools.output_path(save_path, "2D_GS.gif"))
                                                                                                # Animates the iterative approximation against the exact surface.


def main(show=False, save_path=RESULTS_DIR, nodes_1d=21, nodes_2d=21, time_steps=400):
    """
    Run all diffusion examples.

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
        All diffusion example blocks are executed.
    """
    ExampleTools.print_example_header("Diffusion Equation CFDM Solvers - Examples")             # Prints the standard section header for this script.
    run_1d_example(show=show, save_path=save_path, nodes=nodes_1d, time_steps=time_steps)       # Runs the 1D diffusion benchmark.
    run_2d_example(show=show, save_path=save_path, nodes=nodes_2d, time_steps=time_steps)       # Runs the 2D diffusion benchmark.


if __name__ == "__main__":
    main()                                                                                      # Executes all diffusion examples when launched as a script.
