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
from mpl_toolkits import mplot3d

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

def Transient_1D(u_ap, u_ex, x, t):
    fig, (ax1, ax2) = plt.subplots(1, 2)                                            # Se hace una figura con dos figuras incrustadas.
    plt.rcParams["figure.figsize"] = (10,5)                                         # Se define el tamaño de la figura principal.
    plt.suptitle('Diffusion Equation')                                  # Se pone un título a la figura principal.
    min  = u_ex.min()                                                               # Se encuentra el valor mínimo de la solución.
    max  = u_ex.max()                                                               # Se encuentra el valor máximo de la solución.
    p = int(np.ceil(t/100))                                                         # Se decide cuantos pasos de tiempo mostrar.

    for i in range(0,t,p):                                                          # Para el tiempo desde 0 hasta 1.
        ax1.plot(x, u_ap[:,i])                                                      # Se grafica la solución aproximada en la primera figura incrustada.
        ax1.set_ylim([min,max])                                                     # Se fijan los ejes en y.
        ax1.set_title('Solución Aproximada')                                        # Se pone el título de la primera figura incrustada.
    
        ax2.plot(x, u_ex[:,i])                                                     # Se grafica la solución exacta en la segunda figura incrustada.
        ax2.set_ylim([min,max])                                                     # Se fijan los ejes en y.
        ax2.set_title('Solución Exacta')                                            # Se pone el título de la segunda figura incrustada.
    
        plt.pause(0.01)                                                             # Se muestra la figura.
        ax1.clear()                                                                 # Se limpia la gráfica de la primera figura.
        ax2.clear()                                                                 # Se limpia la gráfica de la segunda figura.

    plt.show()                                                                      # Se muestra el último paso de tiempo.