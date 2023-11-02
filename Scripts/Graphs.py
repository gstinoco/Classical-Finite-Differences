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
    September, 2023.
'''
# Library Importation
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def Graph_1D_Stationary(x, u_ap, u_ex):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.rcParams['figure.figsize'] = (10,5)
    plt.suptitle('Solution Comparison')
    min  = u_ex.min()
    max  = u_ex.max()

    ax1.set_title('Computed Solution')
    ax1.plot(x, u_ap)
    ax1.set_ylim([min,max])

    ax2.set_title('Theoretical Solution')
    ax2.plot(x, u_ex)
    ax2.set_ylim([min,max])
    plt.show()

def Graph_1D_Stationary_1(x, u_ap):
    fig = plt.plot(x,u_ap)
    plt.title('Computed Solution')
    plt.rcParams['figure.figsize'] = (10,5)
    
    plt.show()

def Graph_1D_Transient(x, t, u_ap, u_ex):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.rcParams["figure.figsize"] = (10,5)
    plt.suptitle('Solution Comparison')
    min  = u_ex.min()
    max  = u_ex.max()
    p = int(np.ceil(t/100))

    for k in range(0,t,p):
        ax1.plot(x, u_ap[:,k])
        ax1.set_ylim([min,max])
        ax1.set_title('Computed Solution')
        ax1.grid(True)
    
        ax2.plot(x, u_ex[:,k])
        ax2.set_ylim([min,max])
        ax2.set_title('Theoretical Solution')
        ax2.grid(True)
    
        plt.pause(0.01)
        ax1.clear()
        ax2.clear()
    plt.pause(0.1)

def Graph_2D_Static(x, y, u_ap, u_ex):
    min  = u_ex.min()
    max  = u_ex.max()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(8, 4))
    
    ax1.set_title('Computed solution')
    ax1.set_zlim([min, max])
    ax1.plot_surface(x, y, u_ap)
    
    ax2.set_title('Theoretical Solution')
    ax2.set_zlim([min, max])
    ax2.plot_surface(x, y, u_ex)

    plt.show()

def Graph_2D_Transient(x, y, u_ap, u_ex):
    """
    Graph_2D_Transient

    This function graphs the approximated and theoretical solutions of the problem being solved at several time levels.
    Both solutions are presented side by side to help perform graphical comparisons between both solutions.
    
    Input:
        x           m x n           Array           Array with the x-coordinates of the nodes.
        y           m x n           Array           Array with the y-coordinates of the nodes.
        u_ap        m x n x t       Array           Array with the computed solution.
        u_ex        m x n x t       Array           Array with the theoretical solution.
    
    Output:
        None
    """
    t    = len(u_ex[0,0,:])
    step = int(np.ceil(t/50))
    min  = u_ex.min()
    max  = u_ex.max()
    T    = np.linspace(0,1,t)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw = {"projection": "3d"}, figsize=(8, 4))

    for k in np.arange(0,t,step):
        tin = float(T[k])
        plt.suptitle('Solution at t = %1.3f s.' %tin)
        
        ax1.plot_surface(x, y, u_ap[:,:,k])
        ax1.set_zlim([min, max])
        ax1.set_title('Approximation')

        ax2.plot_surface(x, y, u_ex[:,:,k])
        ax2.set_zlim([min, max])
        ax2.set_title('Theoretical Solution')

        plt.pause(0.01)
        ax1.clear()
        ax2.clear()
        plt.cla()
    
    tin = float(T[t-1])
    plt.suptitle('Solution at t = %1.3f s.' %tin)
    
    ax1.plot_surface(x, y, u_ap[:,:,t-1])
    ax1.set_zlim([min, max])
    ax1.set_title('Approximation')

    ax2.plot_surface(x, y, u_ex[:,:,t-1])
    ax2.set_zlim([min, max])
    ax2.set_title('Theoretical Solution')

    plt.pause(0.1)

def Graph_1D_Transient_1(x, t, u_ap, u_ex):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.rcParams["figure.figsize"] = (10,5)
    plt.suptitle('Solution Comparison')
    min  = u_ex.min()
    max  = u_ex.max()
    p = int(np.ceil(t/100))

    k = 0
    ax1.plot(x, u_ap[:,k])
    ax1.set_ylim([min,max])
    ax1.set_title('Computed Solution')
    ax1.grid(True)
    ax2.plot(x, u_ex[:,k])
    ax2.set_ylim([min,max])
    ax2.set_title('Theoretical Solution')
    ax2.grid(True)
    plt.show()
    ax1.clear()
    ax2.clear()

    k = np.ceil(t/2)
    ax1.plot(x, u_ap[:,k])
    ax1.set_ylim([min,max])
    ax1.set_title('Computed Solution')
    ax1.grid(True)
    ax2.plot(x, u_ex[:,k])
    ax2.set_ylim([min,max])
    ax2.set_title('Theoretical Solution')
    ax2.grid(True)
    plt.show()
    ax1.clear()
    ax2.clear()

    ax1.plot(x, u_ap[:,-1])
    ax1.set_ylim([min,max])
    ax1.set_title('Computed Solution')
    ax1.grid(True)
    ax2.plot(x, u_ex[:,-1])
    ax2.set_ylim([min,max])
    ax2.set_title('Theoretical Solution')
    ax2.grid(True)
    plt.show()

def Graph_2D_Transient_1(x, y, u_ap, u_ex):
    """
    Graph_2D_Transient_1

    This function graphs the approximated and theoretical solutions of the problem being solved at several time levels.
    Both solutions are presented side by side to help perform graphical comparisons between both solutions.
    
    Input:
        x           m x n           Array           Array with the x-coordinates of the nodes.
        y           m x n           Array           Array with the y-coordinates of the nodes.
        u_ap        m x n x t       Array           Array with the computed solution.
        u_ex        m x n x t       Array           Array with the theoretical solution.
    
    Output:
        None
    """
    t    = len(u_ex[0,0,:])
    min  = u_ex.min()
    max  = u_ex.max()
    T    = np.linspace(0,1,t)
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw = {"projection": "3d"}, figsize=(8, 4))

    k = np.ceil(0)
    tin = float(T[k])
    plt.suptitle('Solution at t = %1.3f s.' %tin)
    ax1.plot_surface(x, y, u_ap[:,:,k])
    ax1.set_zlim([min, max])
    ax1.set_title('Approximation')
    ax2.plot_surface(x, y, u_ex[:,:,k])
    ax2.set_zlim([min, max])
    ax2.set_title('Theoretical Solution')
    plt.show()

    ax1.clear()
    ax2.clear()
    plt.cla()

    k = np.ceil(t/2)
    tin = float(T[k])
    plt.suptitle('Solution at t = %1.3f s.' %tin)
    ax1.plot_surface(x, y, u_ap[:,:,k])
    ax1.set_zlim([min, max])
    ax1.set_title('Approximation')
    ax2.plot_surface(x, y, u_ex[:,:,k])
    ax2.set_zlim([min, max])
    ax2.set_title('Theoretical Solution')
    plt.show()

    ax1.clear()
    ax2.clear()
    plt.cla()
    
    tin = float(T[-1])
    plt.suptitle('Solution at t = %1.3f s.' %tin)
    ax1.plot_surface(x, y, u_ap[:,:,-1])
    ax1.set_zlim([min, max])
    ax1.set_title('Approximation')
    ax2.plot_surface(x, y, u_ex[:,:,-1])
    ax2.set_zlim([min, max])
    ax2.set_title('Theoretical Solution')
    plt.show()