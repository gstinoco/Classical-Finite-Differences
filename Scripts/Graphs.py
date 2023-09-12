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
    August, 2023.
'''
# Library Importation
import numpy as np
import matplotlib.pyplot as plt

def Graph_1D_Stationary(a, b, m, u_ap, u_ex):
    x = np.linspace(a, b, m)
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

def Graph_1D_Stationary_1(a, b, m, u_ap):
    x = np.linspace(a, b, m)
    fig = plt.plot(x,u_ap)
    plt.title('Computed Solution')
    plt.rcParams['figure.figsize'] = (10,5)
    
    plt.show()

def Mesh_Static(x, y, u_ap, u_ex):
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