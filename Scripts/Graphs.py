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
from mpl_toolkits import mplot3d
from matplotlib import cm

def Graph_1D(tit, x, u_ap, u_ex = np.zeros(2)):
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
            Graph_1D_Stationary_1(x, u_ap, tit)                             # Stationary case.
        else:
            Graph_1D_Transient_1(x, u_ap, tit)                              # Transient case.
    else:                                                                   # If there is a theoretical solution.
        if np.size(u_ap.shape) == 1:                                        # Check for Transient or Stationary problem.
            Graph_1D_Stationary(x, u_ap, u_ex, tit)                         # Stationary case.
        else:
            Graph_1D_Transient(x, u_ap, u_ex, tit)                          # Transient case.


def Graph_2D(tit, x, y, u_ap, u_ex = 0):
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
        Graph_2D_Stationary(x, y, u_ap, u_ex, tit)                          # Stationary case.
    else:
        Graph_2D_Transient(x, y, u_ap, u_ex, tit)                           # Transient case.


def Graph_1D_Stationary(x, u_ap, u_ex, tit):
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))                   # Create a figure with 2 subplots.
    plt.suptitle(tit)                                                       # Place the title for the figure.

    ax1.set_title('Computed Solution')                                      # Title for the first subplot.
    ax1.plot(x, u_ap)                                                       # Plot the computed solution.
    ax1.set_ylim([min,max])                                                 # Set the axis limits for subplot 1.
    ax1.grid(True)                                                          # Plot a grid on subplot 1.

    ax2.set_title('Theoretical Solution')                                   # Title for the second subplot.
    ax2.plot(x, u_ex)                                                       # Plot the theoretical solution.
    ax2.set_ylim([min,max])                                                 # Set the axis limits for subplot 2.
    ax2.grid(True)                                                          # Plot a grid on subplot 2.

    plt.show()                                                              # Show the plot.


def Graph_1D_Stationary_1(x, u_ap, tit):
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

    fig = plt.figure(figsize=(10, 6))                                       # Create a new figure.
    plt.title('Computed Solution for ' + tit)                               # Place the title for the figure.

    plt.plot(x,u_ap)                                                        # Plot the computed solution in a new figure.
    plt.ylim([min,max])                                                     # Set the axis limits.
    plt.grid(True)                                                          # Plot a grid on the figure.
    plt.show()                                                              # Show the figure.


def Graph_1D_Transient(x, u_ap, u_ex, tit):
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
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))                   # Create a figure with 2 subplots.

    for k in range(0,t,step):                                               # Iterate over the time.
        plt.suptitle(tit + '\n' + 'Solution at t = ' + str(T[k]) + 's.')    # Place the title for the figure.

        ax1.set_title('Computed Solution')                                  # Title for the first subplot.
        ax1.plot(x, u_ap[:,k])                                              # Plot the computed solution.
        ax1.set_ylim([min, max])                                            # Set the axis limits for subplot 1.
        ax1.grid(True)                                                      # Plot a grid on subplot 1.
    
        ax2.set_title('Theoretical Solution')                               # Title for the second subplot.
        ax2.plot(x, u_ex[:,k])                                              # Plot the theoretical solution.
        ax2.set_ylim([min, max])                                            # Set the axis limits for subplot 2.
        ax2.grid(True)                                                      # Plot a grid on subplot 2.
    
        plt.pause(0.01)                                                     # Pause for 0.01 seconds.
        ax1.clear()                                                         # Clear the first subplot.
        ax2.clear()                                                         # Clear the second subplot.
        plt.cla()                                                           # Clear the graphic.
    
    plt.suptitle(tit + '\n' + 'Solution at t = ' + str(T[-1]) + 's.')       # Place the title for the figure.

    ax1.set_title('Computed Solution')                                      # Title for the first subplot.
    ax1.plot(x, u_ap[:,-1])                                                 # Plot the computed solution.
    ax1.set_ylim([min,max])                                                 # Set the axis limits for subplot 1.
    ax1.grid(True)                                                          # Plot a grid on subplot 1.
    
    ax2.set_title('Theoretical Solution')                                   # Title for the first subplot.
    ax2.plot(x, u_ex[:,-1])                                                 # Plot the theoretical solution.
    ax2.set_ylim([min,max])                                                 # Set the axis limits for subplot 2.
    ax2.grid(True)                                                          # Plot a grid on subplot 2.
    
    plt.pause(1)                                                            # Pause for 1 second.


def Graph_1D_Transient_1(x, u_ap, tit):
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
    
    plt.figure(figsize=(10, 6))                                             # Create a new figure.

    for k in range(0,t,step):                                               # Iterate over the time.
        plt.title(tit + '\n' + 'Solution at t = ' + str(T[k]) + 's.')       # Place the title for the figure.

        plt.plot(x,u_ap[:,k])                                               # Plot the computed solution in a new figure.
        plt.ylim([min,max])                                                 # Set the axis limits.
        plt.grid(True)                                                      # Plot a grid on the figure.

        plt.pause(0.01)                                                     # Pause for 0.01 seconds.
        plt.cla()                                                           # Clear the graphic.

    plt.title(tit + '\n' + 'Solution at t = ' + str(T[-1]) + 's.')          # Place the title for the figure.
        
    plt.plot(x,u_ap[:,-1])                                                  # Plot the computed solution in a new figure.
    plt.grid(True)                                                          # Plot a grid on the figure.
    plt.ylim([min,max])                                                     # Set the axis limits.

    plt.pause(1)                                                            # Pause for 1 second.

def Graph_2D_Stationary(x, y, u_ap, u_ex, tit):
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
    
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, \
                                   figsize=(10, 6))                         # Create a figure with 2 subplots.
    fig.suptitle(tit)                                                       # Place the title for the figure.
    
    ax1.set_title('Computed solution')                                      # Title for the first subplot.
    ax1.plot_surface(x, y, u_ap)                                            # Plot the computed solution.
    ax1.set_zlim([min, max])                                                # Set the axis limits for subplot 1.
    ax1.grid(True)                                                          # Plot a grid on subplot 1.
    
    ax2.set_title('Theoretical Solution')                                   # Title for the second subplot.
    ax2.plot_surface(x, y, u_ex)                                            # Plot the theoretical solution.
    ax2.set_zlim([min, max])                                                # Set the axis limits for subplot 2.
    ax2.grid(True)                                                          # Plot a grid on subplot 2.

    plt.show()                                                              # Show the plot.


def Graph_2D_Transient(x, y, u_ap, u_ex,tit):
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
    
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, \
                                   figsize=(10, 6))                         # Create a figure with 2 subplots.

    for k in np.arange(0,t,step):
        plt.suptitle(tit + '\n' + 'Solution at t = ' + str(T[k]) + 's.')    # Place the title for the figure.
        
        ax1.set_title('Computed Solution')                                  # Title for the first subplot.
        ax1.plot_surface(x, y, u_ap[:,:,k])                                 # Plot the computed solution.
        ax1.set_zlim([min, max])                                            # Set the axis limits for subplot 1.
        ax1.grid(True)                                                      # Plot a grid on subplot 1.

        ax2.set_title('Theoretical Solution')                               # Title for the second subplot.
        ax2.plot_surface(x, y, u_ex[:,:,k])                                 # Plot the theoretical solution.
        ax2.set_zlim([min, max])                                            # Set the axis limits for subplot 2.
        ax2.grid(True)                                                      # Plot a grid on subplot 2.

        plt.pause(0.01)                                                     # Pause for 0.01 seconds.
        ax1.clear()                                                         # Clear the first subplot.
        ax2.clear()                                                         # Clear the second subplot.
        plt.cla()                                                           # Clear the graphic.
    
    plt.suptitle(tit + '\n' + 'Solution at t = ' + str(T[-1]) + 's.')       # Place the title for the figure.
    
    ax1.set_title('Computed Solution')                                      # Title for the first subplot.
    ax1.plot_surface(x, y, u_ap[:,:,-1])                                    # Plot the computed solution.
    ax1.set_zlim([min, max])                                                # Set the axis limits for subplot 1.
    ax1.grid(True)                                                          # Plot a grid on subplot 1.

    ax2.set_title('Theoretical Solution')                                   # Title for the second subplot.
    ax2.plot_surface(x, y, u_ex[:,:,-1])                                    # Plot the theoretical solution.
    ax2.set_zlim([min, max])                                                # Set the axis limits for subplot 2.
    ax2.grid(True)                                                          # Plot a grid on subplot 2.

    plt.pause(1)                                                            # Pause for 1 second.

def Cloud_Static(p, tt, u_ap, u_ex):
    if tt.min() == 1:
        tt -= 1
    
    min  = u_ex.min()
    max  = u_ex.max()

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw = {"projection": "3d"}, figsize=(10, 6))
    
    ax1.plot_trisurf(p[:,0], p[:,1], u_ap[:], triangles=tt, cmap=cm.coolwarm)
    ax1.set_zlim([min, max])
    ax1.set_title('Approximation')
    
    ax2.plot_trisurf(p[:,0], p[:,1], u_ex[:], triangles=tt, cmap=cm.coolwarm)
    ax2.set_zlim([min, max])
    ax2.set_title('Theoretical Solution')

    plt.show()

def Cloud_Transient(p, tt, u_ap, u_ex):
    if tt.min() == 1:
        tt -= 1
    t    = len(u_ex[0,:])
    step = int(np.ceil(t/50))
    min  = u_ex.min()
    max  = u_ex.max()
    T    = np.linspace(0,1,t)

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw = {"projection": "3d"}, figsize=(10, 6))

    for k in np.arange(0,t,step):
        
        tin = float(T[k])
        plt.suptitle('Solution at t = %1.3f s.' %tin)

        ax1.plot_trisurf(p[:,0], p[:,1], u_ap[:,k], triangles=tt, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax1.set_zlim([min, max])
        ax1.set_title('Approximation')
        
        ax2.plot_trisurf(p[:,0], p[:,1], u_ex[:,k], triangles=tt, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax2.set_zlim([min, max])
        ax2.set_title('Theoretical Solution')

        plt.pause(0.01)
        ax1.clear()
        ax2.clear()

    tin = float(T[-1])
    plt.suptitle('Solution at t = %1.3f s.' %tin)

    ax1.plot_trisurf(p[:,0], p[:,1], u_ap[:,-1], triangles=tt, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax1.set_zlim([min, max])
    ax1.set_title('Approximation')
    
    ax2.plot_trisurf(p[:,0], p[:,1], u_ex[:,-1], triangles=tt, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax2.set_zlim([min, max])
    ax2.set_title('Theoretical Solution')

    plt.pause(0.1)