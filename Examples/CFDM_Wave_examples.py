"""
=========================================================================================
Examples for CFDM Wave solvers
=========================================================================================

This script demonstrates transient 1D/2D wave-equation solvers.

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

import CFDM.Wave as CWave                                                                       # Imports the wave finite-difference solvers.
import Common.ExampleTools as ExampleTools                                                      # Imports shared helpers for metrics, headers, and output paths.
import Common.Graphs as Graphs                                                                  # Imports shared plotting routines for transient examples.

RESULTS_DIR = Path("Results/Wave")                                                              # Defines the default folder for wave-equation figures.


def run_1d_example(show=False, save_path=RESULTS_DIR, nodes=81, time_steps=240, speed=1):
    """
    Run the 1D transient wave-equation example.

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
        Wave propagation speed.

    Returns
    -------
    None
        Results are printed and figures are optionally displayed or saved.
    """
    u = lambda x, t, c: 1 + np.sin(np.pi*x) * np.cos(np.pi*c*t)                                 # Defines the exact 1D standing-wave benchmark solution.
    x = np.linspace(0, 1, nodes)                                                                # Builds the 1D spatial mesh over the unit interval.

    u_matrix, u_ex = CWave.Wave1D(x, time_steps, u, speed)                                      # Solves the 1D transient problem with the matrix scheme.
    u_stencil, _   = CWave.Wave1D_iter(x, time_steps, u, speed)                                 # Solves the same 1D problem with the node-wise stencil.

    ExampleTools.print_metrics("1D Results (Transient)", [                                      # Prints a common metric table over the full 1D time history.
        ExampleTools.transient_row("Matrix", u_ex, u_matrix),                                   # Adds matrix-scheme transient error metrics.
        ExampleTools.transient_row("Stencil", u_ex, u_stencil),                                 # Adds node-wise stencil transient error metrics.
    ])                                                                                          # Closes the 1D metric table definition.

    Graphs.Transient_1D(x, u_matrix, u_ex, title="Wave 1D - Matrix", show=show, save_path=ExampleTools.output_path(save_path, "1D_Matrix.gif"))
                                                                                                # Animates the matrix approximation against the exact solution.
    Graphs.Transient_1D(x, u_stencil, u_ex, title="Wave 1D - Stencil", show=show, save_path=ExampleTools.output_path(save_path, "1D_Stencil.gif"))
                                                                                                # Animates the node-wise stencil approximation against the exact solution.


def run_2d_example(show=False, save_path=RESULTS_DIR, nodes=35, time_steps=420, speed=1):
    """
    Run the 2D transient wave-equation example.

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
    speed : float
        Wave propagation speed.

    Returns
    -------
    None
        Results are printed and figures are optionally displayed or saved.
    """
    omega = np.pi*speed*np.sqrt(2)                                                              # Angular frequency for the separable 2D standing wave.
    u     = lambda x, y, t, c: 1 + np.sin(np.pi*x) * np.sin(np.pi*y) * np.cos(omega*t)          # Defines the exact 2D standing-wave benchmark solution.
    x_1d  = np.linspace(0, 1, nodes)                                                            # Builds the x-coordinate mesh nodes.
    y_1d  = np.linspace(0, 1, nodes)                                                            # Builds the y-coordinate mesh nodes.
    x, y  = np.meshgrid(x_1d, y_1d)                                                             # Expands 1D coordinate vectors into a rectangular 2D mesh.

    u_matrix, u_ex = CWave.Wave2D(x, y, time_steps, u, speed)                                   # Solves the 2D transient problem with the matrix scheme.
    u_stencil, _   = CWave.Wave2D_iter(x, y, time_steps, u, speed)                              # Solves the same 2D problem with the node-wise stencil.

    ExampleTools.print_metrics("2D Results (Transient)", [                                      # Prints a common metric table over the full 2D time history.
        ExampleTools.transient_row("Matrix", u_ex, u_matrix),                                   # Adds matrix-scheme transient error metrics.
        ExampleTools.transient_row("Stencil", u_ex, u_stencil),                                 # Adds node-wise stencil transient error metrics.
    ])                                                                                          # Closes the 2D metric table definition.

    Graphs.Transient_2D(x, y, u_matrix, u_ex, title="Wave 2D - Matrix", show=show, save_path=ExampleTools.output_path(save_path, "2D_Matrix.gif"))
                                                                                                # Animates the matrix approximation against the exact surface.
    Graphs.Transient_2D(x, y, u_stencil, u_ex, title="Wave 2D - Stencil", show=show, save_path=ExampleTools.output_path(save_path, "2D_Stencil.gif"))
                                                                                                # Animates the node-wise stencil approximation against the exact surface.


def main(show=False, save_path=RESULTS_DIR, nodes_1d=81, nodes_2d=35, time_steps_1d=240, time_steps_2d=420):
    """
    Run all wave-equation examples.

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
    time_steps_1d : int
        Number of temporal nodes used in 1D examples.
    time_steps_2d : int
        Number of temporal nodes used in 2D examples.

    Returns
    -------
    None
        All wave-equation example blocks are executed.
    """
    ExampleTools.print_example_header("Wave Equation CFDM Solvers - Examples")                  # Prints the standard section header for this script.
    run_1d_example(show=show, save_path=save_path, nodes=nodes_1d, time_steps=time_steps_1d)    # Runs the 1D wave benchmark.
    run_2d_example(show=show, save_path=save_path, nodes=nodes_2d, time_steps=time_steps_2d)    # Runs the 2D wave benchmark.


if __name__ == "__main__":
    main()                                                                                      # Executes all wave examples when launched as a script.
