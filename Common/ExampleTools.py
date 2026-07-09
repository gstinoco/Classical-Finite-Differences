"""
=========================================================================================
Shared Utilities for CFDM Example Scripts
=========================================================================================

This module provides reusable helpers for example scripts, including standardized
console headers, output paths, and formatted metric rows for stationary and transient
numerical solutions.

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

from pathlib import Path                                                                        # Portable path handling for results directories.
import Common.Metrics as Metrics                                                                # Shared error metrics and table printer.

METRIC_HEADERS = ["Method", "MAE", "MSE", "RMSE", "MAPE(%)", "R^2"]                             # Standard metric columns for all examples.


def print_example_header(title):
    """
    Print a compact, consistent header for an example script.

    Parameters
    ----------
    title : str
        Text shown as the example title.

    Returns
    -------
    None
        The formatted header is printed to the console.
    """
    line = "=" * len(title)                                                                     # Match underline length to the title text.
    print(line)                                                                                 # Print top separator.
    print(title)                                                                                # Print the example title.
    print(line)                                                                                 # Print bottom separator.
    print("")                                                                                   # Add one blank line before the first table.


def output_path(results_dir, filename):
    """
    Return a file path under results_dir; use results_dir=None to skip saving.

    Parameters
    ----------
    results_dir : str, pathlib.Path, or None
        Directory where output files should be written. If None, saving is disabled.
    filename : str
        Name of the output file.

    Returns
    -------
    str or None
        Full output path as a string, or None when results_dir is None.
    """
    if results_dir is None:
        return None
    return str(Path(results_dir) / filename)


def stationary_row(method, exact, approx):
    """
    Build one formatted metrics row for stationary examples.

    Parameters
    ----------
    method : str
        Name of the numerical method shown in the table.
    exact : np.ndarray
        Exact/reference solution values.
    approx : np.ndarray
        Approximate numerical solution values.

    Returns
    -------
    list[str]
        Formatted table row containing method name and error metrics.
    """
    return [
        method,                                                                                 # Method name shown in the first column.
        f"{Metrics.mae(exact, approx):.6f}",                                                    # Mean absolute error.
        f"{Metrics.mse(exact, approx):.6f}",                                                    # Mean squared error.
        f"{Metrics.rmse(exact, approx):.6f}",                                                   # Root mean squared error.
        f"{Metrics.mape(exact, approx, porcentaje=True):.3f}",                                  # Mean absolute percentage error.
        f"{Metrics.r2(exact, approx):.6f}",                                                     # Coefficient of determination.
    ]


def transient_row(method, exact, approx):
    """
    Build one formatted metrics row for transient examples.

    Parameters
    ----------
    method : str
        Name of the numerical method shown in the table.
    exact : np.ndarray
        Exact/reference solution values over time.
    approx : np.ndarray
        Approximate numerical solution values over time.

    Returns
    -------
    list[str]
        Formatted table row containing method name and time-averaged error metrics.
    """
    return [
        method,                                                                                 # Method name shown in the first column.
        f"{Metrics.mae_t_mean(exact, approx):.6f}",                                             # Time-averaged mean absolute error.
        f"{Metrics.mse_t_mean(exact, approx):.6f}",                                             # Time-averaged mean squared error.
        f"{Metrics.rmse_t_mean(exact, approx):.6f}",                                            # Time-averaged root mean squared error.
        f"{Metrics.mape_t_mean(exact, approx, porcentaje=True):.3f}",                           # Time-averaged percentage error.
        f"{Metrics.r2_t_mean(exact, approx):.6f}",                                              # Time-averaged coefficient of determination.
    ]


def print_metrics(title, rows):
    """
    Print a metrics table with the standard example headers.

    Parameters
    ----------
    title : str
        Title printed above the metrics table.
    rows : list[list[str]]
        Table rows with already formatted metric values.

    Returns
    -------
    None
        The formatted metrics table is printed to the console.
    """
    Metrics.print_metrics_table(title, METRIC_HEADERS, rows)                                     # Delegate aligned console formatting to Metrics.


if __name__ == "__main__":
    print("This module defines shared example helpers. Import it from Examples/*.py.")            # Inform users that this helper module is not a standalone example.
