'''
Runge-Kutta Methods for MOL

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

def RungeKutta2(x, T, nu, u, u_ap):
    m = len(x)
    t = len(T)
    dx = x[1] - x[0]
    dt = T[1] - T[0]

    for k in range(t-1):
        k1 = rhs(u_ap[:,k],           T,        k, x, nu, u)
        k2 = rhs(u_ap[:,k]+(dt/2)*k1, T+(dt/2), k, x, nu, u)
        u_ap[1:-1,k+1] = u_ap[1:-1,k] + (dt/2)*(k1[1:-1] + k2[1:-1])
    
    return u_ap[1:-1,:]

def RungeKutta3(x, T, nu, u, u_ap):
    m = len(x)
    t = len(T)
    dx = x[1] - x[0]
    dt = T[1] - T[0]

    for k in range(t-1):
        k1 = rhs(u_ap[:,k],           T,        k, x, nu, u)
        k2 = rhs(u_ap[:,k]+(dt/2)*k1, T+(dt/2), k, x, nu, u)
        k3 = rhs(u_ap[:,k]+dt*(2*k2-k1), T+dt, k, x, nu, u)
        u_ap[1:-1,k+1] = u_ap[1:-1,k] + (dt/6)*(k1[1:-1] + 4*k2[1:-1] + k3[1:-1])
    
    return u_ap[1:-1,:]
    
def rhs(u_ap, T, k, x, nu, u):
    m = len(x)
    t = len(T)
    dx = x[1] - x[0]
    dt = T[1] - T[0]
    s = np.zeros([m])

    s[0]  = u(x[0],T[k],nu)
    s[-1] = u(x[-1],T[k],nu)

    for i in range(1,m-1):
        s[i] = (nu/dx**2)*(u_ap[i-1] - 2*u_ap[i] + u_ap[i+1])
    
    return s