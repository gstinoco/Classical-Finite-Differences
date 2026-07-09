"""
=========================================================================================
Plotting Utilities for Numerical Solutions
=========================================================================================

This module provides comprehensive plotting tools for 1D and 2D numerical solutions.

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
import matplotlib.pyplot as plt                                                                 # Plotting interface.
from matplotlib.lines import Line2D                                                             # Legend proxy for 3D surfaces.
import matplotlib.animation as animation                                                        # Animation helpers and writers for transient plots.

# Ensure the project root directory is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)                                      # Resolve project root by going one level up.
if PROJECT_ROOT not in sys.path:                                                                # Add only if not already present.
    sys.path.insert(0, PROJECT_ROOT)                                                            # Prepend project root to import path.

def Stationary_1D(x, phi_ap, phi_ex, title='Solution Comparison (1D)', show=True, save_path=None):
    """
    Generate a 1D figure overlaying approximate and exact solutions on the same axis.

    Parameters
        x: numpy.ndarray
            1D mesh coordinates over the interval.
        phi_ap: numpy.ndarray
            Approximate solution at nodes.
        phi_ex: numpy.ndarray
            Exact solution at nodes.
        title: str, optional
            Main figure title.
        show : bool
        If True, display the interactive window; otherwise close the figure.
        save_path: str | os.PathLike, optional
            Path where to save the image; saves if provided.

    Returns
        None

    Notes
        - Axis limits use the combined min/max of both solutions.
        - If `show=False` and `save_path` is not None, the image is saved
          without opening a visualization window.
    """
    # Figure setup
    plt.rcParams["figure.figsize"] = (16, 9)                                                    # Set default figure size.
    fig, ax = plt.subplots(1, 1)                                                                # Create single-axis canvas.
    plt.suptitle(title)                                                                         # Set figure title.

    # Axis limits
    minu = min(phi_ap.min(), phi_ex.min())                                                      # Lower y-limit from both solutions.
    maxu = max(phi_ap.max(), phi_ex.max())                                                      # Upper y-limit from both solutions.

    # Plot data series
    ax.plot(x, phi_ap, label='Approximate Solution', color='C0', linestyle='-')                 # Plot approximate solution.
    ax.plot(x, phi_ex, label='Exact Solution',       color='C1', linestyle='--')                # Plot exact solution.

    # Limits and legend
    ax.set_ylim([minu, maxu])                                                                   # Apply common y-limits.
    ax.legend(loc='best')                                                                       # Show legend for distinction.

    # Save image
    if save_path is not None:                                                                   # Save when a path is provided.
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)                               # Ensure the output directory exists.
        plt.savefig(save_path, bbox_inches='tight')                                             # Save image if a path is provided.

    # Display control
    if show:
        try:
            if save_path is None:
                plt.show(block=True)                                                            # Block execution when no file is being saved.
            else:
                plt.show(block=False)                                                           # Show briefly without blocking when saving.
                plt.pause(0.01)                                                                 # Let the GUI event loop refresh.
        except Exception:
            pass                                                                                # Ignore display failures in headless environments.
    
    if save_path is not None or not show:
        plt.close(fig)                                                                          # Release the figure when no interactive window remains.


if __name__ == "__main__":
    print("This module defines shared plotting utilities. Import it from Examples/*.py.")        # Inform users that plotting demos live in the example scripts.

def Stationary_2D(x, y, phi_ap, phi_ex, title='Solution Comparison (2D)', show=True, save_path=None, dpi=150):
    """
    Plot 3D approximate and exact solutions on the same axis using different styles.
    
    Parameters
        x, y: numpy.ndarray (m x m)
            2D meshes.
        phi_ap: numpy.ndarray (m x m)
            Approximate solution over the grid.
        phi_ex: numpy.ndarray (m x m)
            Exact solution over the grid.
        title: str
            Figure title.
        show : bool
        If True, display the figure.
        save_path: str | None
            Path to save the figure if specified.
        dpi : int
        Resolution for saving the figure.
    """
    # Figure setup
    plt.rcParams["figure.figsize"] = (16, 9)                                                    # Set figure size.
    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})                               # Create a single 3D axis.
    plt.suptitle(title)                                                                         # Set figure title.

    # Axis limits
    minu  = min(phi_ap.min(), phi_ex.min())                                                     # Lower z-limit from both surfaces.
    maxu  = max(phi_ap.max(), phi_ex.max())                                                     # Upper z-limit from both surfaces.

    # Plot surfaces
    ax.plot_surface(x, y, phi_ap, cmap='viridis', alpha=0.7)                                    # Plot approximate surface.
    ax.plot_surface(x, y, phi_ex, cmap='plasma',  alpha=0.7)                                    # Plot exact surface.
    ax.set_zlim([minu, maxu])                                                                   # Apply common z-limits.
    # Legend proxies
    ax.legend(handles=[                                                                         # Legend proxies for surfaces.
        Line2D([0], [0], color='C0', lw=2, label='Approximate Solution'),                       # Proxy for approximate.
        Line2D([0], [0], color='C1', lw=2, label='Exact Solution'),                             # Proxy for exact.
    ], loc='best')                                                                              # Place legend where it least obstructs the surfaces.

    # Save image
    if save_path is not None:                                                                   # Save when a path is provided.
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)                               # Ensure the output directory exists.
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')                                    # Save figure if path provided.

    # Display control
    if show:
        try:
            if save_path is None:
                plt.show(block=True)                                                            # Block execution when no file is being saved.
            else:
                plt.show(block=False)                                                           # Show briefly without blocking when saving.
                plt.pause(0.01)                                                                 # Let the GUI event loop refresh.
        except Exception:
            pass                                                                                # Ignore display failures in headless environments.
            
    if save_path is not None or not show:
        plt.close(fig)                                                                          # Release the figure when no interactive window remains.

def Transient_1D(x, u_ap, u_ex, title='Time Evolution (1D)', show=True, save_path=None, dpi=150, max_frames=50):
    """
    Plot 1D temporal evolution overlaying approximate and exact solutions on the same axis.
    
    Parameters
        x: numpy.ndarray (m,)
            1D mesh.
        u_ap: numpy.ndarray (m x t)
            Approximate solution over time.
        u_ex: numpy.ndarray (m x t)
            Exact solution over time.
        title: str, optional
            Figure title.
        show : bool
        If True, display the figure (non-blocking).
        save_path: str | None, optional
            Path to save an animation; supports GIF via PillowWriter and video formats via FFMpegWriter.
            Falls back to saving the last frame if writers are unavailable.
        dpi : int
        Resolution when saving.
        max_frames: int | None, optional
            Maximum number of frames used for animation; defaults to 50 when None.
    """
    # Time and animation parameters
    t          = u_ex.shape[-1]                                                                 # Number of time steps.
    max_frames = 50 if max_frames is None else int(max_frames)                                  # Normalize max frames.
    step       = max(1, int(np.ceil(t / max_frames)))                                           # Frame step for animation.
    minu       = float(u_ex.min())                                                              # Lower bound across exact solution.
    maxu       = float(u_ex.max())                                                              # Upper bound across exact solution.
    T          = np.linspace(0, 1, t)                                                           # Time vector for labels.

    # Figure setup
    plt.rcParams["figure.figsize"] = (16, 9)                                                    # Set figure size.
    fig, ax = plt.subplots(1, 1)                                                                # Create single axis.

    # Animation only when show=True; if show=False, no interactive windows
    # Animation loop
    if show:
        for k in np.arange(0, t, step):
            tin = float(T[k])                                                                   # Current physical time for the frame title.
            plt.suptitle(f'{title} - t = {tin:1.3f} s')                                         # Update frame title with current time.
            ax.plot(x, u_ap[:, k], label='Approximate Solution', color='C0', linestyle='-')     # Draw approximate profile at frame k.
            ax.plot(x, u_ex[:, k], label='Exact Solution',       color='C1', linestyle='--')    # Draw exact profile at frame k.
            ax.set_ylim([minu, maxu])                                                           # Keep a fixed vertical range across frames.
            ax.legend(loc='best')                                                               # Add legend for the current frame.
            plt.pause(0.01)                                                                     # Allow the interactive frame to render.
            ax.clear()                                                                          # Clear axes before drawing the next frame.

    # Final frame
    tin = float(T[-1])                                                                          # Final time.
    plt.suptitle(f'{title} - t = {tin:1.3f} s')                                                 # Final title using provided title.
    ax.plot(x, u_ap[:, -1], label='Approximate Solution', color='C0', linestyle='-')            # Plot final approximate profile.
                                                                                                # Plot final approximate.
    ax.plot(x, u_ex[:, -1], label='Exact Solution',       color='C1', linestyle='--')           # Plot final exact profile.
                                                                                                # Plot final exact.
    ax.set_ylim([minu, maxu])                                                                   # Apply common y-limits.
    ax.legend(loc='best')                                                                       # Add legend.

    # Save image
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)                               # Ensure the output directory exists.
        frames = list(np.arange(0, t, step))                                                    # Select the saved animation frames.
        def _update_1d(k):
            """
            Update the 1D animation frame.

            Parameters
            ----------
            k : int
                Time index used to redraw approximate and exact solutions.

            Returns
            -------
            list
                Empty artist list required by Matplotlib when blitting is disabled.
            """
            ax.clear()                                                                          # Clear previous animation artists.
            tin = float(T[k])                                                                   # Current physical time for the frame title.
            plt.suptitle(f'{title} - t = {tin:1.3f} s')                                         # Update saved frame title.
            ax.plot(x, u_ap[:, k], label='Approximate Solution', color='C0', linestyle='-')     # Draw approximate profile for saved frame.
            ax.plot(x, u_ex[:, k], label='Exact Solution',       color='C1', linestyle='--')    # Draw exact profile for saved frame.
            ax.set_ylim([minu, maxu])                                                           # Keep fixed y-limits across saved frames.
            ax.legend(loc='best')                                                               # Add legend to saved frame.
            return []                                                                           # Return no artists because blitting is disabled.
        anim = animation.FuncAnimation(fig, _update_1d, frames=frames, blit=False)              # Build a Matplotlib animation object.
        ext = str(Path(save_path).suffix).lower()                                               # Determine writer from file extension.
        try:
            if ext == '.gif':
                from matplotlib.animation import PillowWriter                                   # GIF writer backend.
                writer = PillowWriter(fps=10)                                                   # Save GIF at ten frames per second.
            else:
                from matplotlib.animation import FFMpegWriter                                   # Video writer backend.
                writer = FFMpegWriter(fps=10)                                                   # Save video at ten frames per second.
            anim.save(save_path, writer=writer, dpi=dpi)                                        # Write animation to disk.
        except Exception:
            try:
                fig.savefig(save_path, dpi=dpi, bbox_inches='tight')                            # Fall back to a static final-frame image.
            except Exception:
                pass                                                                            # Ignore save failures after writer fallback fails.

    # Display control
    if show:
        try:
            if save_path is None:
                plt.show(block=True)                                                            # Block execution when no file is being saved.
            else:
                plt.show(block=False)                                                           # Show briefly without blocking when saving.
                plt.pause(0.01)                                                                 # Let the GUI event loop refresh.
        except Exception:
            pass                                                                                # Ignore display failures in headless environments.
            
    if save_path is not None or not show:
        plt.close(fig)                                                                          # Release the figure when no interactive window remains.

def Transient_2D(x, y, u_ap, u_ex, title='Time Evolution (2D)', show=True, save_path=None, dpi=150, max_frames=50):
    """
    Plot 3D temporal evolution overlaying approximate and exact solutions on the same axis.
    
    Parameters
        x, y: numpy.ndarray (m x m)
            2D meshes.
        u_ap: numpy.ndarray (m x m x t)
            Approximate solution over time.
        u_ex: numpy.ndarray (m x m x t)
            Exact solution over time.
        title: str, optional
            Figure title.
        show : bool
        If True, display the figure (non-blocking).
        save_path: str | None, optional
            Path to save an animation; supports GIF via PillowWriter and video formats via FFMpegWriter.
            Falls back to saving the last frame if writers are unavailable.
        dpi : int
        Resolution when saving.
        max_frames: int | None, optional
            Maximum number of frames used for animation; defaults to 50 when None.
    """
    # Time and animation parameters
    t          = u_ex.shape[-1]                                                                 # Number of time steps.
    max_frames = 50 if max_frames is None else int(max_frames)                                  # Normalize max frames.
    step       = max(1, int(np.ceil(t / max_frames)))                                           # Frame step for animation.
    minu       = float(u_ex.min())                                                              # Lower bound across exact solution.
    maxu       = float(u_ex.max())                                                              # Upper bound across exact solution.
    T          = np.linspace(0, 1, t)                                                           # Time vector for labels.

    # Figure setup
    plt.rcParams["figure.figsize"] = (16, 9)                                                    # Figure size.
    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})                               # Single 3D axis.

    # Animation loop
    if show:
        for k in np.arange(0, t, step):
            tin = float(T[k])                                                                   # Current physical time for the frame title.
            plt.suptitle(f'{title} - t = {tin:1.3f} s')                                         # Update frame title with current time.
            ax.plot_surface(x, y, u_ap[:, :, k], cmap='viridis', alpha=0.7)                     # Draw approximate surface at frame k.
            ax.plot_surface(x, y, u_ex[:, :, k], cmap='plasma',  alpha=0.7)                     # Draw exact surface at frame k.
            ax.set_zlim([minu, maxu])                                                           # Keep a fixed vertical range across frames.
            ax.legend(handles=[                                                                 # Build surface legend from proxy line handles.
                Line2D([0], [0], color='C0', lw=2, label='Approximate Solution'),               # Legend proxy for approximate surface.
                Line2D([0], [0], color='C1', lw=2, label='Exact Solution'),                     # Legend proxy for exact surface.
            ], loc='best')                                                                      # Place legend where it least obstructs the surface.
            plt.pause(0.01)                                                                     # Allow the interactive frame to render.
            ax.clear()                                                                          # Clear axes before drawing the next frame.

    # Final frame
    tin = float(T[-1])                                                                          # Final time.
    plt.suptitle(f'{title} - t = {tin:1.3f} s')                                                 # Final title using provided title.
    ax.plot_surface(x, y, u_ap[:, :, -1], cmap='viridis', alpha=0.7)                            # Plot final approximate surface.
    ax.plot_surface(x, y, u_ex[:, :, -1], cmap='plasma',  alpha=0.7)                            # Plot final exact surface.
    ax.set_zlim([minu, maxu])                                                                   # Apply common z-limits.
    ax.legend(handles=[                                                                         # Legend proxies for final frame.
        Line2D([0], [0], color='C0', lw=2, label='Approximate Solution'),                       # Legend proxy for approximate surface.
        Line2D([0], [0], color='C1', lw=2, label='Exact Solution'),                             # Legend proxy for exact surface.
    ], loc='best')                                                                              # Place legend where it least obstructs the final surface.

    # Save image
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)                               # Ensure the output directory exists.
        frames = list(np.arange(0, t, step))                                                    # Select the saved animation frames.
        def _update_2d(k):
            """
            Update the 2D animation frame.

            Parameters
            ----------
            k : int
                Time index used to redraw approximate and exact surfaces.

            Returns
            -------
            list
                Empty artist list required by Matplotlib when blitting is disabled.
            """
            ax.clear()                                                                          # Clear previous animation artists.
            tin = float(T[k])                                                                   # Current physical time for the frame title.
            plt.suptitle(f'{title} - t = {tin:1.3f} s')                                         # Update saved frame title.
            ax.plot_surface(x, y, u_ap[:, :, k], cmap='viridis', alpha=0.7)                     # Draw approximate surface for saved frame.
            ax.plot_surface(x, y, u_ex[:, :, k], cmap='plasma',  alpha=0.7)                     # Draw exact surface for saved frame.
            ax.set_zlim([minu, maxu])                                                           # Keep fixed z-limits across saved frames.
            ax.legend(handles=[                                                                 # Build surface legend from proxy line handles.
                Line2D([0], [0], color='C0', lw=2, label='Approximate Solution'),               # Legend proxy for approximate surface.
                Line2D([0], [0], color='C1', lw=2, label='Exact Solution'),                     # Legend proxy for exact surface.
            ], loc='best')                                                                      # Place legend where it least obstructs the surface.
            return []                                                                           # Return no artists because blitting is disabled.
        anim = animation.FuncAnimation(fig, _update_2d, frames=frames, blit=False)              # Build a Matplotlib animation object.
        ext = str(Path(save_path).suffix).lower()                                               # Determine writer from file extension.
        try:
            if ext == '.gif':
                from matplotlib.animation import PillowWriter                                   # GIF writer backend.
                writer = PillowWriter(fps=10)                                                   # Save GIF at ten frames per second.
            else:
                from matplotlib.animation import FFMpegWriter                                   # Video writer backend.
                writer = FFMpegWriter(fps=10)                                                   # Save video at ten frames per second.
            anim.save(save_path, writer=writer, dpi=dpi)                                        # Write animation to disk.
        except Exception:
            try:
                fig.savefig(save_path, dpi=dpi, bbox_inches='tight')                            # Fall back to a static final-frame image.
            except Exception:
                pass                                                                            # Ignore save failures after writer fallback fails.

    # Display control
    if show:
        try:
            if save_path is None:
                plt.show(block=True)                                                            # Block execution when no file is being saved.
            else:
                plt.show(block=False)                                                           # Show briefly without blocking when saving.
                plt.pause(0.01)                                                                 # Let the GUI event loop refresh.
        except Exception:
            pass                                                                                # Ignore display failures in headless environments.
            
    if save_path is not None or not show:
        plt.close(fig)                                                                          # Release the figure when no interactive window remains.
