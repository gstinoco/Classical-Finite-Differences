'''
Codes to create graphics to visually analyze the results of the methods.

All the codes were developed by:
    Dr. Gerardo Tinoco Guerrero
    Universidad Michoacana de San Nicolás de Hidalgo
    gerardo.tinoco@umich.mx
 
With the funding of:
    National Council of Humanities, Sciences and Technologies, CONAHCyT (Consejo Nacional de Humanidades, Ciencias y Tecnologías, CONAHCyT). México.
    Coordination of Scientific Research, CIC-UMSNH (Coordinación de la Investigación Científica de la Universidad Michoacana de San Nicolás de Hidalgo, CIC-UMSNH). México
    Aula CIMNE-Morelia. México
 
Date:
    October, 2022.

Last Modification:
    January, 2024.
'''

# Library Importation
import numpy as np
import matplotlib.pyplot as plt
import cv2
import moviepy.editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage
from Scripts.Error_norms import Error_norms_1D
from Scripts.Error_norms import Error_norms_2D

def Graph_1D(tit, x, u_ap, u_ex = np.zeros(2), save = False):
    '''
        Graph_1D

        General code for all the graphics for 1D spatial problems.

        Arguments:
            tit                         String          Title for the graphic.
            x           m x 1           Array           Array with the x values of the nodes of the grid.
            u_ap        m x t           Array           Array with the computed solution of the method.
            u_ex        m x t           Array           Array with the exact solution.
                                                        (Default: 0)
        
        Returns:
            Nothing.
    '''

    if u_ex.max() == 0 and u_ex.min() == 0:                                 # If there isn't theoretical solution.
        if np.size(u_ap.shape) == 1:                                        # Check for Transient or Stationary problem.
            Graph_1D_Stationary_1(x, u_ap, tit, save)                       # Stationary case.
        else:
            Graph_1D_Transient_1(x, u_ap, tit, save)                        # Transient case.
    else:                                                                   # If there is a theoretical solution.
        if np.size(u_ap.shape) == 1:                                        # Check for Transient or Stationary problem.
            Graph_1D_Stationary(x, u_ap, u_ex, tit, save)                   # Stationary case.
        else:
            Graph_1D_Transient(x, u_ap, u_ex, tit, save)                    # Transient case.


def Graph_2D(tit, x, y, u_ap, u_ex = 0, save = False):
    '''
    Graph_2D

    General code for all the graphics for 2D spatial problems.
    
    Arguments:
        tit                         String          Title for the graphic.
        x           m x n           Array           Array with the x-coordinates of the nodes.
        y           m x n           Array           Array with the y-coordinates of the nodes.
        u_ap        m x n x t       Array           Array with the computed solution.
        u_ex        m x n x t       Array           Array with the theoretical solution.
                                                    (Default: 0)
    
    Returns:
        Nothing.
    '''

    if np.size(u_ap.shape) == 2:                                            # Check for Transient or Stationary problem.
        Graph_2D_Stationary(x, y, u_ap, u_ex, tit, save)                    # Stationary case.
    else:
        Graph_2D_Transient(x, y, u_ap, u_ex, tit, save)                     # Transient case.


def Graph_1D_Stationary(x, u_ap, u_ex, tit, save):
    '''
    Graph_1D_Stationary

    This function graphs the approximated and theoretical solutions of the problem being solved.
    Both solutions are presented side by side to help perform graphical comparisons between both solutions.
    
    Input:
        x           m x 1           Array           Array with the x-coordinates of the nodes.
        u_ap        m x 1           Array           Array with the computed solution.
        u_ex        m x 1           Array           Array with the theoretical solution.
        tit                         String          Title for the graphic.
    
    Output:
        None
    '''

    min = u_ex.min()                                                        # Look for the minimum of the exact solution for the axis.
    max = u_ex.max()                                                        # Look for the maximum of the exact solution for the axis.
    _, _, _, LQ = Error_norms_1D(u_ap, u_ex, verbose = False)               # Compute the norms of the error.

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 8))                 # Create a figure with 2 subplots.
    plt.suptitle(tit + '\n' + 'Quadratic Mean Error = ' + str(LQ))          # Place the title for the figure.

    ax1.set_title('Computed Solution')                                      # Title for the first subplot.
    ax1.plot(x, u_ap)                                                       # Plot the computed solution.
    ax1.set_ylim([min,max])                                                 # Set the axis limits for subplot 1.
    ax1.grid(True)                                                          # Plot a grid on subplot 1.

    ax2.set_title('Theoretical Solution')                                   # Title for the second subplot.
    ax2.plot(x, u_ex)                                                       # Plot the theoretical solution.
    ax2.set_ylim([min,max])                                                 # Set the axis limits for subplot 2.
    ax2.grid(True)                                                          # Plot a grid on subplot 2.
    
    if save:                                                                # If saving results is requested.
        nom = 'Results/' + tit + '.png'                                     # The name for the file under the 'Results' folder.
        plt.savefig(nom)                                                    # Save the figure.
    plt.show()                                                              # Show the plot.


def Graph_1D_Stationary_1(x, u_ap, tit, save):
    '''
    Graph_1D_Stationary_1

    This function graphs the approximated solution of the problem.
    
    Input:
        x           m x 1           Array           Array with the x-coordinates of the nodes.
        u_ap        m x t           Array           Array with the computed solution.
        tit                         String          Title for the graphic.
    
    Output:
        None
    '''

    min = u_ap.min()                                                        # Look for the minimum of the computed solution for the axis.
    max = u_ap.max()                                                        # Look for the maximum of the computed solution for the axis.

    fig = plt.figure(figsize = (10, 8))                                       # Create a new figure.
    plt.title('Computed Solution for ' + tit)                               # Place the title for the figure.

    plt.plot(x,u_ap)                                                        # Plot the computed solution in a new figure.
    plt.ylim([min,max])                                                     # Set the axis limits.
    plt.grid(True)                                                          # Plot a grid on the figure.
    
    if save:                                                                # If saving results is requested.
        nom = 'Results/' + tit + '.png'                                     # The name for the file under the 'Results' folder.
        plt.savefig(nom)                                                    # Save the figure.
    plt.show()                                                              # Show the plot.


def Graph_1D_Transient(x, u_ap, u_ex, tit, save):
    '''
    Graph_1D_Transient

    This function graphs the approximated and theoretical solutions of the problem being solved at several time levels.
    Both solutions are presented side by side to help perform graphical comparisons between both solutions.
    
    Input:
        x           m x 1           Array           Array with the x-coordinates of the nodes.
        u_ap        m x t           Array           Array with the computed solution.
        u_ex        m x t           Array           Array with the theoretical solution.
        tit                         String          Title for the graphic.
    
    Output:
        None
    '''
    
    min  = u_ex.min()                                                       # Look for the minimum of the exact solution for the axis.
    max  = u_ex.max()                                                       # Look for the maximum of the exact solution for the axis.
    t    = len(u_ex[0,:])                                                   # Check the number of time steps.
    step = int(np.ceil(t/50))                                               # Fix a step for the plots so only 50 time-steps are plotted.
    T    = np.linspace(0,1,t)                                               # Create a time mesh.
    _, _, _, LQ = Error_norms_1D(u_ap, u_ex, verbose = False)               # Compute the norms of the error.

    if save:                                                                # If saving results is requested.
        nom    = 'Results/' + tit + '.mp4'                                  # The name for the file under the 'Results' folder.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')                            # Choose the codec for the file.
        out    = cv2.VideoWriter(nom, fourcc, 5, (1000, 800))               # Create a the file to save the video.
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 8))                 # Create a figure with 2 subplots.

    for k in range(0,t,step):                                               # Iterate over the time.
        plt.suptitle(tit + '\n' + 'Solution at t = ' + str(T[k]) + 's.' +
                     '\n' + 'Quadratic Mean Error =' + str(LQ[k]))          # Place the title for the figure.

        ax1.set_title('Computed Solution')                                  # Title for the first subplot.
        ax1.plot(x, u_ap[:,k])                                              # Plot the computed solution.
        ax1.set_ylim([min, max])                                            # Set the axis limits for subplot 1.
        ax1.grid(True)                                                      # Plot a grid on subplot 1.
    
        ax2.set_title('Theoretical Solution')                               # Title for the second subplot.
        ax2.plot(x, u_ex[:,k])                                              # Plot the theoretical solution.
        ax2.set_ylim([min, max])                                            # Set the axis limits for subplot 2.
        ax2.grid(True)                                                      # Plot a grid on subplot 2.
    
        plt.pause(0.01)                                                     # Pause for 0.01 seconds.
        if save:                                                            # If saving results is requested.
            fig.canvas.draw()                                               # Draw the canvas of the figure.
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]          # Convert the canvas into a np.array.
            out.write(frame)                                                # Save the current frame.
            
        ax1.clear()                                                         # Clear the first subplot.
        ax2.clear()                                                         # Clear the second subplot.
        plt.cla()                                                           # Clear the graphic.
    
    plt.suptitle(tit + '\n' + 'Solution at t = ' + str(T[-1]) + 's.' +
                     '\n' + 'Quadratic Mean Error =' + str(LQ[-1]))         # Place the title for the figure.

    ax1.set_title('Computed Solution')                                      # Title for the first subplot.
    ax1.plot(x, u_ap[:,-1])                                                 # Plot the computed solution.
    ax1.set_ylim([min,max])                                                 # Set the axis limits for subplot 1.
    ax1.grid(True)                                                          # Plot a grid on subplot 1.
    
    ax2.set_title('Theoretical Solution')                                   # Title for the first subplot.
    ax2.plot(x, u_ex[:,-1])                                                 # Plot the theoretical solution.
    ax2.set_ylim([min,max])                                                 # Set the axis limits for subplot 2.
    ax2.grid(True)                                                          # Plot a grid on subplot 2.

    plt.pause(0.1)                                                          # Pause for 0.1 seconds.
    if save:                                                                # If saving results is requested.
        fig.canvas.draw()                                                   # Draw the canvas of the figure.
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]              # Convert the canvas into a np.array.
        out.write(frame)                                                    # Save the current frame.
        out.release()                                                       # Save the frames into the video.
    plt.close()


def Graph_1D_Transient_1(x, u_ap, tit, save):
    '''
    Graph_1D_Transient_1

    This function graphs the approximated solution of the problem being solved at several time levels.
    
    Input:
        x           m x n           Array           Array with the x-coordinates of the nodes.
        u_ap        m x 1           Array           Array with the computed solution.
        tit                         String          Title for the graphic.
    
    Output:
        None
    '''
    
    min  = u_ap.min()                                                       # Look for the minimum of the computed solution for the axis.
    max  = u_ap.max()                                                       # Look for the maximum of the computed solution for the axis.
    t    = len(u_ap[0,:])                                                   # Check the number of time steps.
    step = int(np.ceil(t/50))                                               # Fix a step for the plots so only 50 time-steps are plotted.
    T    = np.linspace(0,1,t)                                               # Create a time mesh.

    if save:                                                                # If saving results is requested.
        nom    = 'Results/' + tit + '.mp4'                                  # The name for the file under the 'Results' folder.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')                            # Choose the codec for the file.
        out    = cv2.VideoWriter(nom, fourcc, 5, (1000, 800))               # Create a the file to save the video.
    
    fig = plt.figure(figsize = (10, 8))                                     # Create a new figure.

    for k in range(0,t,step):                                               # Iterate over the time.
        plt.title(tit + '\n' + 'Solution at t = ' + str(T[k]) + 's.')       # Place the title for the figure.

        plt.plot(x,u_ap[:,k])                                               # Plot the computed solution in a new figure.
        plt.ylim([min,max])                                                 # Set the axis limits.
        plt.grid(True)                                                      # Plot a grid on the figure.

        plt.pause(0.01)                                                     # Pause for 0.01 seconds.
        if save:                                                            # If saving results is requested.
            fig.canvas.draw()                                               # Draw the canvas of the figure.
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]          # Convert the canvas into a np.array.
            out.write(frame)                                                # Save the current frame.

        plt.cla()                                                           # Clear the graphic.

    plt.title(tit + '\n' + 'Solution at t = ' + str(T[-1]) + 's.')          # Place the title for the figure.
        
    plt.plot(x,u_ap[:,-1])                                                  # Plot the computed solution in a new figure.
    plt.grid(True)                                                          # Plot a grid on the figure.
    plt.ylim([min,max])                                                     # Set the axis limits.

    plt.pause(0.1)                                                          # Pause for 0.1 seconds.
    if save:                                                                # If saving results is requested.
        fig.canvas.draw()                                                   # Draw the canvas of the figure.
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]              # Convert the canvas into a np.array.
        out.write(frame)                                                    # Save the current frame.
        out.release()                                                       # Save the frames into the video.
    plt.close()


def Graph_2D_Stationary(x, y, u_ap, u_ex, tit, save):
    '''
    Graph_2D_Stationary

    This function graphs the approximated and theoretical solutions of the problem.
    Both solutions are presented side by side to help perform graphical comparisons between both solutions.
    
    Input:
        x           m x n           Array           Array with the x-coordinates of the nodes.
        y           m x n           Array           Array with the y-coordinates of the nodes.
        u_ap        m x n           Array           Array with the computed solution.
        u_ex        m x n           Array           Array with the theoretical solution.
        tit                         String          Title for the graphic.
    
    Output:
        None
    '''

    min = u_ex.min()                                                        # Look for the minimum of the exact solution for the axis.
    max = u_ex.max()                                                        # Look for the maximum of the exact solution for the axis.

    _, _, _, LQ = Error_norms_2D(u_ap, u_ex, verbose = False)               # Compute the norms of the error.
    
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, \
                                   figsize = (10, 8))                       # Create a figure with 2 subplots.
    plt.suptitle(tit + '\n' + 'Quadratic Mean Error = ' + str(LQ))          # Place the title for the figure.
    
    ax1.set_title('Computed solution')                                      # Title for the first subplot.
    ax1.plot_surface(x, y, u_ap)                                            # Plot the computed solution.
    ax1.set_zlim([min, max])                                                # Set the axis limits for subplot 1.
    ax1.grid(True)                                                          # Plot a grid on subplot 1.
    
    ax2.set_title('Theoretical Solution')                                   # Title for the second subplot.
    ax2.plot_surface(x, y, u_ex)                                            # Plot the theoretical solution.
    ax2.set_zlim([min, max])                                                # Set the axis limits for subplot 2.
    ax2.grid(True)                                                          # Plot a grid on subplot 2.

    if save:                                                                # If saving results is requested.
        nom = 'Results/' + tit + '.png'                                     # The name for the file under the 'Results' folder.
        plt.savefig(nom)                                                    # Save the figure.
    plt.show()                                                              # Show the plot.


def Graph_2D_Transient(x, y, u_ap, u_ex, tit, save):
    '''
    Graph_2D_Transient

    This function graphs the approximated and theoretical solutions of the problem being solved at several time levels.
    Both solutions are presented side by side to help perform graphical comparisons between both solutions.
    
    Input:
        x           m x n           Array           Array with the x-coordinates of the nodes.
        y           m x n           Array           Array with the y-coordinates of the nodes.
        u_ap        m x n x t       Array           Array with the computed solution.
        u_ex        m x n x t       Array           Array with the theoretical solution.
        tit                         String          Title for the graphic.
    
    Output:
        None
    '''
    
    min  = u_ex.min()                                                       # Look for the minimum of the exact solution for the axis.
    max  = u_ex.max()                                                       # Look for the maximum of the exact solution for the axis.
    t    = len(u_ex[0,0,:])                                                 # Check the number of time steps.
    step = int(np.ceil(t/50))                                               # Fix a step for the plots so only 50 time-steps are plotted.
    T    = np.linspace(0,1,t)                                               # Create a time mesh.

    _, _, _, LQ = Error_norms_2D(u_ap, u_ex, verbose = False)               # Compute the norms of the error.

    if save:                                                                # If saving results is requested.
        nom    = 'Results/' + tit + '.mp4'                                  # The name for the file under the 'Results' folder.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')                            # Choose the codec for the file.
        out    = cv2.VideoWriter(nom, fourcc, 5, (1000, 800))               # Create a the file to save the video.
    
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, \
                                   figsize = (10, 8))                       # Create a figure with 2 subplots.

    for k in np.arange(0,t,step):
        plt.suptitle(tit + '\n' + 'Solution at t = ' + str(T[k]) + 's.' +
                     '\n' + 'Quadratic Mean Error =' + str(LQ[k]))          # Place the title for the figure.
        
        ax1.set_title('Computed Solution')                                  # Title for the first subplot.
        ax1.plot_surface(x, y, u_ap[:,:,k])                                 # Plot the computed solution.
        ax1.set_zlim([min, max])                                            # Set the axis limits for subplot 1.
        ax1.grid(True)                                                      # Plot a grid on subplot 1.

        ax2.set_title('Theoretical Solution')                               # Title for the second subplot.
        ax2.plot_surface(x, y, u_ex[:,:,k])                                 # Plot the theoretical solution.
        ax2.set_zlim([min, max])                                            # Set the axis limits for subplot 2.
        ax2.grid(True)                                                      # Plot a grid on subplot 2.

        plt.pause(0.01)                                                     # Pause for 0.01 seconds.
        if save:                                                            # If saving results is requested.
            fig.canvas.draw()                                               # Draw the canvas of the figure.
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]          # Convert the canvas into a np.array.
            out.write(frame)                                                # Save the current frame.
        ax1.clear()                                                         # Clear the first subplot.
        ax2.clear()                                                         # Clear the second subplot.
        plt.cla()                                                           # Clear the graphic.
    
    plt.suptitle(tit + '\n' + 'Solution at t = ' + str(T[-1]) + 's.' +
                     '\n' + 'Quadratic Mean Error =' + str(LQ[-1]))         # Place the title for the figure.
    
    ax1.set_title('Computed Solution')                                      # Title for the first subplot.
    ax1.plot_surface(x, y, u_ap[:,:,-1])                                    # Plot the computed solution.
    ax1.set_zlim([min, max])                                                # Set the axis limits for subplot 1.
    ax1.grid(True)                                                          # Plot a grid on subplot 1.

    ax2.set_title('Theoretical Solution')                                   # Title for the second subplot.
    ax2.plot_surface(x, y, u_ex[:,:,-1])                                    # Plot the theoretical solution.
    ax2.set_zlim([min, max])                                                # Set the axis limits for subplot 2.
    ax2.grid(True)                                                          # Plot a grid on subplot 2.

    plt.pause(0.1)                                                          # Pause for 0.1 seconds.
    if save:                                                                # If saving results is requested.
        fig.canvas.draw()                                                   # Draw the canvas of the figure.
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]              # Convert the canvas into a np.array.
        out.write(frame)                                                    # Save the current frame.
        out.release()                                                       # Save the frames into the video.
    plt.close()