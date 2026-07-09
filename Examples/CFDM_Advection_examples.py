"""
=========================================================================================
Examples for CFDM Advection solvers
=========================================================================================

This script demonstrates transient 1D/2D advection solvers.

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

import CFDM.Advection as CAdvection                                                             # Imports the advection finite-difference solvers.
import Common.ExampleTools as ExampleTools                                                      # Imports shared helpers for metrics, headers, and output paths.
import Common.Graphs as Graphs                                                                  # Imports shared plotting routines for transient examples.

RESULTS_DIR = Path("Results/Advection")                                                         # Defines the default folder for advection figures.

ADVECTION_METHODS_1D = [                                                                        # Lists the 1D methods and velocity signs used in the examples.
    ("FTCS", "FTCS", 1),                                                                        # Uses the FTCS solver with positive advection speed.
    ("FTBS", "FTBS", 1),                                                                        # Uses the backward spatial stencil with positive speed.
    ("FTFS", "FTFS", -1),                                                                       # Uses the forward spatial stencil with negative speed.
    ("Lax-Wendroff", "LaxWendroff", 1),                                                         # Uses the Lax-Wendroff solver with positive speed.
]

ADVECTION_METHODS_2D = [                                                                        # Lists the 2D methods and velocity signs used in the examples.
    ("FTCS", "FTCS", 1, 1),                                                                     # Uses the FTCS solver with positive x and y speeds.
    ("FTBS", "FTBS", 1, 1),                                                                     # Uses backward spatial stencils with positive x and y speeds.
    ("FTFS", "FTFS", -1, -1),                                                                   # Uses forward spatial stencils with negative x and y speeds.
    ("Lax-Wendroff", "LaxWendroff", 1, 1),                                                      # Uses the Lax-Wendroff solver with positive x and y speeds.
]


def run_1d_example(show=False, save_path=RESULTS_DIR, nodes=41, time_steps=800, speed=0.5):
    """
    Run transient 1D advection examples.

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
    speed : float
        Magnitude of the advection velocity used in the 1D method cases.

    Returns
    -------
    None
        Results are printed and figures are optionally displayed or saved.
    """
    x = np.linspace(0, 2*np.pi, nodes)                                                          # Builds the periodic-looking 1D mesh for the traveling wave.
    u = lambda x, t, a: np.sin(x - a*t)                                                         # Defines the exact translated sine wave.

    rows = []                                                                                   # Accumulates metric rows for all 1D methods.
    plot_cases = []                                                                             # Accumulates plot metadata for all 1D method outputs.
    for display_name, method_arg, sign in ADVECTION_METHODS_1D:
        a_case          = sign * speed                                                          # Selects the signed velocity compatible with the stencil direction.

        u_matrix, u_ex = CAdvection.Advection1D(x, time_steps, u, a_case, method=method_arg)    # Solves the 1D case with the matrix implementation.
        u_iter, _      = CAdvection.Advection1D_iter(x, time_steps, u, a_case, method=method_arg)
                                                                                                # Solves the 1D case with the iterative implementation.

        rows.append(ExampleTools.transient_row(f"{display_name} FD", u_ex, u_matrix))           # Adds finite-difference formulation transient metrics for this stencil.
        rows.append(ExampleTools.transient_row(f"{display_name} GS", u_ex, u_iter))             # Adds Gauss-Seidel-style transient metrics for this stencil.
        plot_cases.extend([(f"1D_FD_{method_arg}.gif", f"Advection 1D - {display_name} FD", u_matrix, u_ex), (f"1D_GS_{method_arg}.gif", f"Advection 1D - {display_name} GS", u_iter, u_ex)])
                                                                                                # Registers FD and GS animations for this stencil.

    ExampleTools.print_metrics("1D Results (Transient)", rows)                                  # Prints one comparable table for all 1D advection methods.

    for filename, title, u_ap, u_ex in plot_cases:
        Graphs.Transient_1D(x, u_ap, u_ex, title=title, show=show, save_path=ExampleTools.output_path(save_path, filename))
                                                                                                # Animates each 1D approximation against the exact wave.


def run_2d_example(show=False, save_path=RESULTS_DIR, nodes=41, time_steps=800, speed_x=0.4, speed_y=0.4):
    """
    Run transient 2D advection examples.

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
    speed_x : float
        Magnitude of the x-direction advection velocity.
    speed_y : float
        Magnitude of the y-direction advection velocity.

    Returns
    -------
    None
        Results are printed and figures are optionally displayed or saved.
    """
    x_1d = np.linspace(0, 1, nodes)                                                             # Builds the x-coordinate mesh nodes.
    y_1d = np.linspace(0, 1, nodes)                                                             # Builds the y-coordinate mesh nodes.
    x, y = np.meshgrid(x_1d, y_1d)                                                              # Expands 1D coordinate vectors into a rectangular 2D mesh.
    u    = lambda x, y, t, a, b: 0.2*np.exp(-((x - 0.5 - a*t)**2 + (y - 0.5 - b*t)**2) / 0.01) # Defines a transported Gaussian pulse.

    rows = []                                                                                   # Accumulates metric rows for all 2D methods.
    plot_cases = []                                                                             # Accumulates plot metadata for all 2D method outputs.
    for display_name, method_arg, sign_x, sign_y in ADVECTION_METHODS_2D:
        a_case         = sign_x * speed_x                                                       # Selects the signed x-velocity compatible with the stencil direction.
        b_case         = sign_y * speed_y                                                       # Selects the signed y-velocity compatible with the stencil direction.

        u_matrix, u_ex = CAdvection.Advection2D(x, y, time_steps, u, a_case, b_case, method=method_arg)
                                                                                                # Solves the 2D case with the matrix implementation.
        u_iter, _      = CAdvection.Advection_2D_iter(x, y, time_steps, u, a_case, b_case, method=method_arg)
                                                                                                # Solves the 2D case with the iterative implementation.

        rows.append(ExampleTools.transient_row(f"{display_name} FD", u_ex, u_matrix))           # Adds finite-difference formulation transient metrics for this stencil.
        rows.append(ExampleTools.transient_row(f"{display_name} GS", u_ex, u_iter))             # Adds Gauss-Seidel-style transient metrics for this stencil.
        plot_cases.extend([(f"2D_FD_{method_arg}.gif", f"Advection 2D - {display_name} FD", u_matrix, u_ex), (f"2D_GS_{method_arg}.gif", f"Advection 2D - {display_name} GS", u_iter, u_ex)])
                                                                                                # Registers FD and GS animations for this stencil.

    ExampleTools.print_metrics("2D Results (Transient)", rows)                                  # Prints one comparable table for all 2D advection methods.

    for filename, title, u_ap, u_ex in plot_cases:
        Graphs.Transient_2D(x, y, u_ap, u_ex, title=title, show=show, save_path=ExampleTools.output_path(save_path, filename))
                                                                                                # Animates each 2D approximation against the exact surface.


def main(show=False, save_path=RESULTS_DIR, nodes_1d=41, nodes_2d=41, time_steps=800):
    """
    Run all advection examples.

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
        All advection example blocks are executed.
    """
    ExampleTools.print_example_header("Advection Equation CFDM Solvers - Examples")             # Prints the standard section header for this script.
    run_1d_example(show=show, save_path=save_path, nodes=nodes_1d, time_steps=time_steps)       # Runs the 1D advection benchmark suite.
    run_2d_example(show=show, save_path=save_path, nodes=nodes_2d, time_steps=time_steps)       # Runs the 2D advection benchmark suite.


if __name__ == "__main__":
    main()                                                                                      # Executes all advection examples when launched as a script.
