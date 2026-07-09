"""
=========================================================================================
Error Metrics for Numerical Solutions
=========================================================================================

This module provides various error metrics (MAE, MSE, RMSE, MAPE, R^2) to evaluate numerical solutions.

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
import sys                                                                                      # Python import path management.
import numpy as np                                                                              # Numerical computing and array structures.
from pathlib import Path                                                                        # Portable filesystem path handling.

# Ensure the project root directory is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)                                      # Resolve project root by going one level up.
if PROJECT_ROOT not in sys.path:                                                                # Add only if not already present.
    sys.path.insert(0, PROJECT_ROOT)                                                            # Prepend project root to import path.

def _to_numpy_1d(x):
    """
    Convert the input to a 1D NumPy float array.

    Parameters
        x: array-like
            List, tuple, or NumPy array convertible to float.

    Returns
        x_np: numpy.ndarray
            1D float array.

    Errors
        TypeError
            If conversion to a float array is not possible.
        ValueError
            If the array contains non-finite values (NaN or Inf).
    """
    # Type conversion to float array
    try:
        x_np = np.asarray(x, dtype=float)                                                       # Convert to NumPy float array.
    except Exception as exc:
        raise TypeError("Input is not convertible to a float array") from exc                   # Type error if not convertible.

    # Shape normalization
    x_np = np.ravel(x_np)                                                                       # Ensure 1D.

    # Finiteness validation
    if not np.all(np.isfinite(x_np)):
        raise ValueError("Input contains non-finite values (NaN/Inf)")                          # Validate finiteness.
    return x_np                                                                                 # Return validated array.


def _validate_pair(y_true, y_pred):
    """
    Validate and prepare the pair y_true/y_pred as 1D NumPy arrays.

    Parameters
        y_true: array-like
            Ground-truth values.
        y_pred: array-like
            Predicted/approximated values.

    Returns
        y_t, y_p: numpy.ndarray, numpy.ndarray
            Validated 1D arrays of equal length.

    Errors
        ValueError
            If lengths differ or there are no elements.
    """
    # Convert inputs to 1D arrays
    y_t = _to_numpy_1d(y_true)                                                                  # Convert and validate y_true.
    y_p = _to_numpy_1d(y_pred)                                                                  # Convert and validate y_pred.

    # Length and emptiness checks
    if y_t.size != y_p.size:
        raise ValueError("y_true and y_pred must have the same length")                         # Validate equal length.
    if y_t.size == 0:
        raise ValueError("Inputs cannot be empty")                                              # Validate non-empty inputs.
    return y_t, y_p                                                                             # Return validated pair.


def _to_numpy_nd(x):
    """
    Convert the input to an n-dimensional NumPy float array and validate finiteness.

    Parameters
        x: array-like
            List, tuple, or NumPy array convertible to float.

    Returns
        x_np: numpy.ndarray
            n-dimensional float array.

    Errors
        TypeError
            If conversion to a float array is not possible.
        ValueError
            If the array contains non-finite values (NaN or Inf).
    """
    # Type conversion to nD float array
    try:
        x_np = np.asarray(x, dtype=float)                                                       # Convert input to NumPy float array.
    except Exception as exc:
        raise TypeError("Input is not convertible to a float array") from exc                   # Raise type error if conversion fails.

    # Finiteness validation
    if not np.all(np.isfinite(x_np)):                                                           # Check for NaN/Inf values.
        raise ValueError("Input contains non-finite values (NaN/Inf)")                          # Raise value error if non-finite.
    return x_np                                                                                 # Return validated nD array.


def _validate_pair_nd(y_true, y_pred):
    """
    Validate and prepare y_true/y_pred as n-dimensional NumPy arrays.

    Parameters
        y_true: array-like
            Ground-truth values (n-dimensional).
        y_pred: array-like
            Predicted/approximated values (n-dimensional).

    Returns
        y_t, y_p: numpy.ndarray, numpy.ndarray
            Validated arrays with equal shape and size.

    Errors
        ValueError
            If shapes differ or there are no elements.
    """
    # Convert inputs to validated nD arrays
    y_t = _to_numpy_nd(y_true)                                                                  # Convert and validate ground-truth array.
    y_p = _to_numpy_nd(y_pred)                                                                  # Convert and validate prediction array.

    # Shape and emptiness checks
    if y_t.shape != y_p.shape:                                                                  # Ensure arrays have identical shapes.
        raise ValueError("y_true and y_pred must have the same shape")                          # Raise error on shape mismatch.
    if y_t.size == 0:                                                                           # Disallow empty arrays.
        raise ValueError("Inputs cannot be empty")                                              # Raise error on empty inputs.
    return y_t, y_p                                                                             # Return validated pair.


def _prepare_time_arrays(y_true, y_pred, axis_time=-1):
    """
    Reorganize arrays to treat the time axis uniformly and flatten the space.

    Returns pairs (Y_true, Y_pred) with shape (S, T) where S=flattened space and T=time.
    """
    # Normalize axis and validate inputs
    y_t, y_p  = _validate_pair_nd(y_true, y_pred)                                               # Validate and obtain aligned arrays.
    ndim      = y_t.ndim                                                                        # Number of dimensions in inputs.
    axis_time = int(axis_time)                                                                  # Normalize axis index to int.
    if axis_time < 0:                                                                           # Support negative indices for time axis.
        axis_time += ndim                                                                       # Convert negative axis to positive.
    if axis_time < 0 or axis_time >= ndim:                                                      # Bounds check for axis index.
        raise ValueError("axis_time out of range for inputs")
    
    # Move time axis to the end and flatten space
    y_t = np.moveaxis(y_t, axis_time, -1)                                                       # Place time axis at last position.
    y_p = np.moveaxis(y_p, axis_time, -1)                                                       # Same operation for predictions.
    S   = int(np.prod(y_t.shape[:-1]))                                                          # Flattened spatial size.
    T   = int(y_t.shape[-1])                                                                    # Number of time steps.

    # Reshape to (S, T)
    Yt = y_t.reshape(S, T)                                                                      # Reshape ground truth to (S, T).
    Yp = y_p.reshape(S, T)                                                                      # Reshape predictions to (S, T).
    return Yt, Yp                                                                               # Return prepared (S, T) arrays.


def mae(y_true, y_pred):
    """
    Compute the Mean Absolute Error (MAE).

    Formula
        MAE = (1/n) * sum |y_true - y_pred|

    Parameters
        y_true: array-like
            Ground-truth values.
        y_pred: array-like
            Predicted/approximated values.

    Returns
        float
            Scalar MAE value.
    """
    # Input validation
    y_t, y_p = _validate_pair(y_true, y_pred)                                                   # Prepare validated inputs.

    # Compute average absolute error
    return float(np.mean(np.abs(y_t - y_p)))                                                    # Average absolute error.


def mse(y_true, y_pred):
    """
    Compute the Mean Squared Error (MSE).

    Formula
        MSE = (1/n) * sum (y_true - y_pred)^2

    Parameters
        y_true: array-like
            Ground-truth values.
        y_pred: array-like
            Predicted/approximated values.

    Returns
        float
            Scalar MSE value.
    """
    # Input validation
    y_t, y_p = _validate_pair(y_true, y_pred)                                                   # Prepare validated inputs.

    # Compute squared differences
    diff     = y_t - y_p                                                                        # Pointwise differences.
    return float(np.mean(diff*diff))                                                            # Average squared error.


def rmse(y_true, y_pred):
    """
    Compute the Root Mean Squared Error (RMSE).

    Formula
        RMSE = sqrt(MSE)

    Parameters
        y_true: array-like
            Ground-truth values.
        y_pred: array-like
            Predicted/approximated values.

    Returns
        float
            Scalar RMSE value.
    """
    # Compute RMSE from MSE
    return float(np.sqrt(mse(y_true, y_pred)))                                                  # Square root of MSE.


def mape(y_true, y_pred, porcentaje=True, epsilon=1e-8, ignore_zeros=True):
    """
    Compute the Mean Absolute Percentage Error (MAPE).

    Formula
        MAPE = (1/n) * sum |(y_true - y_pred) / y_true| * 100

    Notes
        - MAPE can be unrepresentative when y_true is zero or very small.
        - If `ignore_zeros=True`, exclude positions where |y_true| <= epsilon
          from the average (avoids extreme percentages due to near-zero references).
        - If `ignore_zeros=False`, use a small “epsilon” in the denominator to stabilize the division.
        - If `porcentaje=True`, return percentage.

    Parameters
        y_true: array-like
            Ground-truth values.
        y_pred: array-like
            Predicted/approximated values.
        porcentaje : bool
        If True, return percentage (0–100). If False, return fraction.
        epsilon : float
        Small offset to stabilize division.
        ignore_zeros : bool
        If True, ignore entries with |y_true| <= epsilon in the average.

    Returns
        float
            Scalar MAPE value (percentage or fraction, per flag).
    """
    # Input validation
    y_t, y_p = _validate_pair(y_true, y_pred)                                                   # Prepare validated inputs.

    # Branch for handling zeros in y_true
    if ignore_zeros:
        mask = np.abs(y_t) > float(epsilon)                                                     # Mask: discard near-zero references.
        if not np.any(mask):
            raise ValueError("MAPE undefined: y_true is zero (~0) in all entries")

        # Compute relative errors over valid subset
        frac = np.abs(y_t[mask] - y_p[mask]) / np.abs(y_t[mask])                                # Absolute relative error where y_true is significant.
        val  = float(np.mean(frac))                                                             # Average absolute relative error over subset.
    else:
        # Stabilized denominator for all entries
        denom = np.abs(y_t) + float(epsilon)                                                    # Stabilized denominator to avoid division by zero.
        frac  = np.abs(y_t - y_p) / denom                                                       # Absolute relative error per sample.
        val   = float(np.mean(frac))                                                            # Average absolute relative error.
    return (100.0*val) if porcentaje else val                                                   # Scale to percentage if requested.


def r2(y_true, y_pred):
    """
    Compute the coefficient of determination (R^2).

    Formula
        R^2 = 1 - SS_res / SS_tot
        where SS_res = sum (y_true - y_pred)^2 and SS_tot = sum (y_true - mean(y_true))^2.

    Notes
        - If `SS_tot == 0` (constant y_true), return:
            * 1.0 if `SS_res == 0` (perfect prediction).
            * 0.0 otherwise (practical definition with no variation).

    Parameters
        y_true: array-like
            Ground-truth values.
        y_pred: array-like
            Predicted/approximated values.

    Returns
        float
            Scalar R^2 value.
    """
    # Input validation and residuals
    y_t, y_p = _validate_pair(y_true, y_pred)                                                   # Prepare validated inputs.
    ss_res   = float(np.sum((y_t - y_p)**2))                                                    # Sum of squared residuals.
    y_mean   = float(np.mean(y_t))                                                              # Mean of y_true.
    ss_tot   = float(np.sum((y_t - y_mean)**2))                                                 # Total sum of squares.

    # Degenerate case: constant y_true
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0                                                    # Degenerate case (constant y_true).

    # Compute R^2
    return float(1.0 - (ss_res / ss_tot))                                                       # Compute and return R^2.


# =============================
# Metrics for transient cases
# =============================

def mae_t(y_true, y_pred, axis_time=-1):
    """
    MAE per time instant. Returns a vector of length T.

    Parameters
        y_true, y_pred: array-like (n-dimensional with time axis)
        axis_time : int
        Index of the time axis (default -1).
    """
    # Prepare arrays (S, T)
    Yt, Yp = _prepare_time_arrays(y_true, y_pred, axis_time=axis_time)                          # Prepare arrays with shape (S, T).

    # Compute per-time MAE
    return np.mean(np.abs(Yt - Yp), axis=0).astype(float)                                       # Vector of MAE for each time step.


def mse_t(y_true, y_pred, axis_time=-1):
    """
    MSE per time instant. Returns a vector of length T.
    """
    # Prepare arrays (S, T)
    Yt, Yp = _prepare_time_arrays(y_true, y_pred, axis_time=axis_time)                          # Prepare arrays with shape (S, T).

    # Residuals across space for each time step
    diff   = Yt - Yp                                                                            # Residuals across space for each time step.

    # Compute per-time MSE
    return np.mean(diff * diff, axis=0).astype(float)                                           # Vector of MSE for each time step.


def rmse_t(y_true, y_pred, axis_time=-1):
    """
    RMSE per time instant. Returns a vector of length T.
    """
    # Compute per-time RMSE from MSE_t
    return np.sqrt(mse_t(y_true, y_pred, axis_time=axis_time)).astype(float)                    # Vector of RMSE for each time step.


def mape_t(y_true, y_pred, axis_time=-1, porcentaje=True, epsilon=1e-8, ignore_zeros=True):
    """
    MAPE per time instant. Returns a vector of length T.

    Notes
        - If `ignore_zeros=True`, exclude spatial positions with |y_true| <= epsilon.
        - If `porcentaje=True`, return percentage.
    """
    # Prepare arrays (S, T)
    Yt, Yp = _prepare_time_arrays(y_true, y_pred, axis_time=axis_time)                          # Prepare arrays with shape (S, T).

    # Initialize output vector
    S, T = Yt.shape                                                                             # S=flattened space, T=time steps.
    out  = np.empty(T, dtype=float)                                                             # Allocate output vector over time.
    if ignore_zeros:
        # Mask positions with significant y_true
        mask = np.abs(Yt) > float(epsilon)                                                      # Mask positions with significant y_true.
        for k in range(T):
            # Mask and guard for time k
            mk = mask[:, k]                                                                     # Mask for time k.
            if not np.any(mk):                                                                  # Guard against empty valid set.
                raise ValueError("MAPE_t undefined: y_true is zero (~0) across space for some time")
            
            # Compute relative error and average at time k
            frac = np.abs(Yt[mk, k] - Yp[mk, k]) / np.abs(Yt[mk, k])                            # Absolute relative error at time k.
            out[k] = float(np.mean(frac))                                                       # Average relative error for time k.
    else:
        for k in range(T):
                        # Stabilized denominator and relative error
            denom = np.abs(Yt[:, k]) + float(epsilon)                                           # Stabilized denominator for time k.
            frac = np.abs(Yt[:, k] - Yp[:, k]) / denom                                          # Absolute relative error at time k.
            out[k] = float(np.mean(frac))                                                       # Average relative error for time k.
    return (100.0 * out) if porcentaje else out                                                 # Return percentage or fraction per flag.


def r2_t(y_true, y_pred, axis_time=-1):
    """
    R^2 per time instant. Returns a vector of length T.
    """
    # Prepare arrays (S, T)
    Yt, Yp = _prepare_time_arrays(y_true, y_pred, axis_time=axis_time)                          # Prepare arrays with shape (S, T).

    # Initialize output vector
    S, T = Yt.shape                                                                             # S=flattened space, T=time steps.
    out  = np.empty(T, dtype=float)                                                             # Allocate output vector over time.
    for k in range(T):
        # Residuals and sums at time k
        diff   = Yt[:, k] - Yp[:, k]                                                            # Residuals at time k across space.
        ss_res = float(np.sum(diff * diff))                                                     # Sum of squared residuals at time k.
        y_mean = float(np.mean(Yt[:, k]))                                                       # Mean ground truth at time k.
        ss_tot = float(np.sum((Yt[:, k] - y_mean) ** 2))                                        # Total sum of squares at time k.
        if ss_tot == 0.0:
            # Degenerate case: constant signal
            out[k] = 1.0 if ss_res == 0.0 else 0.0                                              # Degenerate case: constant signal at time k.
        else:
            # Compute R^2 for time k
            out[k] = float(1.0 - (ss_res / ss_tot))                                             # R^2 for time k.
    return out                                                                                  # Return vector of R^2 over time.


# Temporal averages (aggregated over time)
def mae_t_mean(y_true, y_pred, axis_time=-1):
    """Temporal average of MAE over time (scalar)."""
    return float(np.mean(mae_t(y_true, y_pred, axis_time=axis_time)))                           # Average MAE across time (scalar).


def mse_t_mean(y_true, y_pred, axis_time=-1):
    """Temporal average of MSE over time (scalar)."""
    return float(np.mean(mse_t(y_true, y_pred, axis_time=axis_time)))                           # Average MSE across time (scalar).


def rmse_t_mean(y_true, y_pred, axis_time=-1):
    """Temporal average of RMSE over time (scalar)."""
    return float(np.mean(rmse_t(y_true, y_pred, axis_time=axis_time)))                          # Average RMSE across time (scalar).


def mape_t_mean(y_true, y_pred, axis_time=-1, porcentaje=True, epsilon=1e-8, ignore_zeros=True):
    """Temporal average of MAPE over time (scalar)."""
    vals = mape_t(y_true, y_pred, axis_time=axis_time, porcentaje=porcentaje, epsilon=epsilon, ignore_zeros=ignore_zeros)
                                                                                                # Compute MAPE at each time step.
    return float(np.mean(vals))                                                                 # Average MAPE across time (scalar).


def r2_t_mean(y_true, y_pred, axis_time=-1):
    """Temporal average of R^2 over time (scalar)."""
    return float(np.mean(r2_t(y_true, y_pred, axis_time=axis_time)))                            # Average R^2 across time (scalar).

def print_metrics_table(title, headers, rows):
    """
    Print a table to the console with title, headers, and rows.

    Parameters
    ----------
    title : str
        Table title.
    headers : list[str]
        Column headers.
    rows : list[list[str]]
        Rows with cells in string format.
    """
    # Compute column widths
    widths = [max(len(headers[i]), max(len(row[i]) for row in rows)) for i in range(len(headers))]
                                                                                                # Width needed for each table column.

    # Format header and separator
    header = " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))               # Format header row.
    sep    = "-+-".join("-"*widths[i] for i in range(len(headers)))                             # Separator line.

    # Print table
    print(title)                                                                                # Print title.
    print(header)                                                                               # Print header.
    print(sep)                                                                                  # Print separator.
    for row in rows:                                                                            # Iterate over rows to print each line.
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))                 # Print formatted row.
    print(f" ")                                                                                 # Print a trailing blank line after the table.


if __name__ == "__main__":
    print("This module defines error metrics and table formatting helpers. Import it from other modules.")  # Inform users that metrics are library utilities.
