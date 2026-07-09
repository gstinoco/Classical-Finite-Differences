"""
=========================================================================================
Examples for CFDM Poisson solvers
=========================================================================================

This script demonstrates stationary 1D/2D Poisson solvers and 1D Neumann variants.

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

import CFDM.Poisson as CPoisson                                                                 # Imports the Poisson finite-difference solvers.
import Common.ExampleTools as ExampleTools                                                      # Imports shared helpers for metrics, headers, and output paths.
import Common.Graphs as Graphs                                                                  # Imports shared plotting routines for stationary examples.

RESULTS_DIR = Path("Results/Poisson")                                                           # Defines the default folder for Poisson figures.


def run_1d_example(show=False, save_path=RESULTS_DIR, nodes=41):
    """
    Run the 1D Poisson Dirichlet example.

    Parameters
    ----------
    show : bool
        If True, display the generated figures.
    save_path : str, pathlib.Path, or None
        Directory where figures are saved. If None, figures are not saved.
    nodes : int
        Number of nodes in the 1D spatial mesh.

    Returns
    -------
    None
        Results are printed and figures are optionally displayed or saved.
    """
    phi = lambda x: x*np.cos(x)                                                                 # Defines the exact solution used to impose Dirichlet boundaries.
    f   = lambda x: 2*np.sin(x) + x*np.cos(x)                                                   # Defines the source term consistent with Delta phi = -f.
    x   = np.linspace(0, 2*np.pi, nodes)                                                        # Builds the 1D spatial mesh over the benchmark interval.

    phi_fd, phi_ex = CPoisson.Poisson1D(x, phi, f)                                              # Solves the 1D problem with the direct finite-difference system.
    phi_gs, _      = CPoisson.Poisson1D_iter(x, phi, f, max_iter=10000, tol=1e-12)              # Solves the same 1D problem with Gauss-Seidel iterations.

    ExampleTools.print_metrics("1D Results", [                                                  # Prints a common metric table for the 1D Dirichlet case.
        ExampleTools.stationary_row("FD", phi_ex, phi_fd),                                      # Adds the direct solver error metrics against the exact solution.
        ExampleTools.stationary_row("GS", phi_ex, phi_gs),                                      # Adds the iterative solver error metrics against the exact solution.
    ])                                                                                          # Closes the 1D metric table definition.

    Graphs.Stationary_1D(x, phi_fd, phi_ex, title="Poisson 1D - FD", show=show, save_path=ExampleTools.output_path(save_path, "1D_FD.png"))
                                                                                                # Plots the direct 1D approximation and exact curve.
    Graphs.Stationary_1D(x, phi_gs, phi_ex, title="Poisson 1D - GS", show=show, save_path=ExampleTools.output_path(save_path, "1D_GS.png"))
                                                                                                # Plots the iterative 1D approximation and exact curve.


def run_2d_example(show=False, save_path=RESULTS_DIR, nodes=41):
    """
    Run the 2D Poisson Dirichlet example.

    Parameters
    ----------
    show : bool
        If True, display the generated figures.
    save_path : str, pathlib.Path, or None
        Directory where figures are saved. If None, figures are not saved.
    nodes : int
        Number of nodes per direction in the 2D spatial mesh.

    Returns
    -------
    None
        Results are printed and figures are optionally displayed or saved.
    """
    phi  = lambda x, y: 2*np.exp(2*x + y)                                                       # Defines the exact 2D solution used on the Dirichlet boundary.
    f    = lambda x, y: -10*np.exp(2*x + y)                                                     # Defines the 2D source term consistent with Delta phi = -f.
    x_1d = np.linspace(0, 1, nodes)                                                             # Builds the x-coordinate mesh nodes.
    y_1d = np.linspace(0, 1, nodes)                                                             # Builds the y-coordinate mesh nodes.
    x, y = np.meshgrid(x_1d, y_1d)                                                              # Expands 1D coordinate vectors into a rectangular 2D mesh.

    phi_fd, phi_ex = CPoisson.Poisson2D(x, y, phi, f)                                           # Solves the 2D problem with the direct finite-difference system.
    phi_gs, _      = CPoisson.Poisson2D_iter(x, y, phi, f, max_iter=10000, tol=1e-10)           # Solves the same 2D problem with Gauss-Seidel iterations.

    ExampleTools.print_metrics("2D Results", [                                                  # Prints a common metric table for the 2D Dirichlet case.
        ExampleTools.stationary_row("FD", phi_ex, phi_fd),                                      # Adds the direct solver error metrics against the exact surface.
        ExampleTools.stationary_row("GS", phi_ex, phi_gs),                                      # Adds the iterative solver error metrics against the exact surface.
    ])                                                                                          # Closes the 2D metric table definition.

    Graphs.Stationary_2D(x, y, phi_fd, phi_ex, title="Poisson 2D - FD", show=show, save_path=ExampleTools.output_path(save_path, "2D_FD.png"))
                                                                                                # Plots the direct 2D approximation and exact surface.
    Graphs.Stationary_2D(x, y, phi_gs, phi_ex, title="Poisson 2D - GS", show=show, save_path=ExampleTools.output_path(save_path, "2D_GS.png"))
                                                                                                # Plots the iterative 2D approximation and exact surface.


def run_neumann_example(show=False, save_path=RESULTS_DIR, nodes=41):
    """
    Run 1D Poisson examples with one Neumann and one Dirichlet boundary.

    Parameters
    ----------
    show : bool
        If True, display the generated figures.
    save_path : str, pathlib.Path, or None
        Directory where figures are saved. If None, figures are not saved.
    nodes : int
        Number of nodes in the 1D spatial mesh.

    Returns
    -------
    None
        Results are printed and figures are optionally displayed or saved.
    """
    phi  = lambda x: x**2 - x + 1                                                               # Defines the quadratic exact solution for the Neumann benchmark.
    f    = lambda x: -2*np.ones_like(x)                                                         # Defines the source term consistent with the quadratic solution.
    sig  = -1.0                                                                                 # Sets the prescribed derivative at the Neumann boundary.
    beta = 1.0                                                                                  # Sets the prescribed value at the Dirichlet boundary.
    x    = np.linspace(0, 1, nodes)                                                             # Builds the 1D mesh for the mixed-boundary benchmark.

    phi_n1_fd, phi_ex = CPoisson.Poisson1D_Neumann_1(x, phi, f, sig, beta)                      # Solves with the first-order two-point Neumann formula.
    phi_n2_fd, _      = CPoisson.Poisson1D_Neumann_2(x, phi, f, sig, beta)                      # Solves with the centered two-point Neumann formula.
    phi_n3_fd, _      = CPoisson.Poisson1D_Neumann_3(x, phi, f, sig, beta)                      # Solves with the second-order three-point Neumann formula.

    phi_n1_gs, _      = CPoisson.Poisson1D_Neumann_1_iter(x, phi, f, sig, beta, max_iter=20000, tol=1e-12)
                                                                                                # Iterates the first-order Neumann system.
    phi_n2_gs, _      = CPoisson.Poisson1D_Neumann_2_iter(x, phi, f, sig, beta, max_iter=20000, tol=1e-12) 
                                                                                                # Iterates the centered Neumann system.
    phi_n3_gs, _      = CPoisson.Poisson1D_Neumann_3_iter(x, phi, f, sig, beta, max_iter=20000, tol=1e-12)
                                                                                                # Iterates the three-point Neumann system.

    ExampleTools.print_metrics("1D Neumann Results", [                                          # Prints one comparable table for all Neumann variants.
        ExampleTools.stationary_row("N1 FD (Two-Point)", phi_ex, phi_n1_fd),                    # Adds direct metrics for the first Neumann stencil.
        ExampleTools.stationary_row("N1 GS (Two-Point)", phi_ex, phi_n1_gs),                    # Adds iterative metrics for the first Neumann stencil.
        ExampleTools.stationary_row("N2 FD (Centered)", phi_ex, phi_n2_fd),                     # Adds direct metrics for the centered Neumann stencil.
        ExampleTools.stationary_row("N2 GS (Centered)", phi_ex, phi_n2_gs),                     # Adds iterative metrics for the centered Neumann stencil.
        ExampleTools.stationary_row("N3 FD (Three-Point)", phi_ex, phi_n3_fd),                  # Adds direct metrics for the three-point Neumann stencil.
        ExampleTools.stationary_row("N3 GS (Three-Point)", phi_ex, phi_n3_gs),                  # Adds iterative metrics for the three-point Neumann stencil.
    ])                                                                                          # Closes the Neumann metric table definition.

    neumann_results = [                                                                         # Groups Neumann plot metadata to keep plotting uniform.
        ("1D_FD_Neumann_1.png", "Poisson 1D Neumann 1 - FD", phi_n1_fd),                        # Stores output data for the direct first-order plot.
        ("1D_GS_Neumann_1.png", "Poisson 1D Neumann 1 - GS", phi_n1_gs),                        # Stores output data for the iterative first-order plot.
        ("1D_FD_Neumann_2.png", "Poisson 1D Neumann 2 - FD", phi_n2_fd),                        # Stores output data for the direct centered-stencil plot.
        ("1D_GS_Neumann_2.png", "Poisson 1D Neumann 2 - GS", phi_n2_gs),                        # Stores output data for the iterative centered-stencil plot.
        ("1D_FD_Neumann_3.png", "Poisson 1D Neumann 3 - FD", phi_n3_fd),                        # Stores output data for the direct three-point-stencil plot.
        ("1D_GS_Neumann_3.png", "Poisson 1D Neumann 3 - GS", phi_n3_gs),                        # Stores output data for the iterative three-point-stencil plot.
    ]
    for filename, title, phi_ap in neumann_results:
        Graphs.Stationary_1D(x, phi_ap, phi_ex, title=title, show=show, save_path=ExampleTools.output_path(save_path, filename))
                                                                                                # Plots each Neumann approximation against the exact solution.


def main(show=False, save_path=RESULTS_DIR, nodes_1d=41, nodes_2d=41):
    """
    Run all Poisson examples.

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

    Returns
    -------
    None
        All Poisson example blocks are executed.
    """
    ExampleTools.print_example_header("Poisson Equation CFDM Solvers - Examples")               # Prints the standard section header for this script.
    run_1d_example(show=show, save_path=save_path, nodes=nodes_1d)                              # Runs the 1D Dirichlet Poisson benchmark.
    run_2d_example(show=show, save_path=save_path, nodes=nodes_2d)                              # Runs the 2D Dirichlet Poisson benchmark.
    run_neumann_example(show=show, save_path=save_path, nodes=nodes_1d)                         # Runs the 1D mixed Neumann-Dirichlet benchmarks.


if __name__ == "__main__":
    main()                                                                                      # Executes all Poisson examples when launched as a script.
