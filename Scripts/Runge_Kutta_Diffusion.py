'''
Runge-Kutta Methods for MOL in Diffusion problems.

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

def RungeKutta2_1D(x, T, nu, u, u_ap):
    t  = len(T)
    dt = T[1] - T[0]

    for k in range(t-1):
        k1 = rhs_1D(u_ap[:, k],             T,          k, x, nu, u)
        k2 = rhs_1D(u_ap[:, k] + (dt/2)*k1, T + (dt/2), k, x, nu, u)
        
        u_ap[1:-1, k+1] = u_ap[1:-1, k] + (dt/2)*(k1[1:-1] + k2[1:-1])
    return u_ap[1:-1, :]

def RungeKutta3_1D(x, T, nu, u, u_ap):
    t  = len(T)
    dt = T[1] - T[0]

    for k in range(t-1):
        k1 = rhs_1D(u_ap[:, k],                    T,          k, x, nu, u)
        k2 = rhs_1D(u_ap[:, k] + (dt/2)*k1,        T + (dt/2), k, x, nu, u)
        k3 = rhs_1D(u_ap[:, k] + (2*dt)*(k2 - k1), T + dt,     k, x, nu, u)

        u_ap[1:-1, k+1] = u_ap[1:-1, k] + (dt/6)*(k1[1:-1] + 4*k2[1:-1] + k3[1:-1])
    return u_ap[1:-1, :]

def RungeKutta4_1D(x, T, nu, u, u_ap):
    t  = len(T)
    dt = T[1] - T[0]

    for k in range(t-1):
        k1 = rhs_1D(u_ap[:, k],                T,          k, x, nu, u)
        k2 = rhs_1D(u_ap[:, k] + (dt/2)*k1,    T + (dt/2), k, x, nu, u)
        k3 = rhs_1D(u_ap[:, k] + (dt/2)*k2,    T + (dt/2), k, x, nu, u)
        k4 = rhs_1D(u_ap[:, k] + dt*k3,        T + dt,     k, x, nu, u)

        u_ap[1:-1, k+1] = u_ap[1:-1, k] + (dt/6)*(k1[1:-1] + 2*k2[1:-1] + 2*k3[1:-1] + k4[1:-1])
    return u_ap[1:-1, :]
    
def rhs_1D(u_ap, T, k, x, nu, u):
    m  = len(x)
    dx = x[1] - x[0]
    s  = np.zeros(m)

    s[0]  = u(x[0],  T[k], nu)
    s[-1] = u(x[-1], T[k], nu)

    for i in range(1, m-1):
        s[i] = (nu/dx**2)*(u_ap[i-1] - 2*u_ap[i] + u_ap[i+1])
    return s

def RungeKutta2_2D(x, y, T, nu, u, u_ap):
    t  = len(T)
    dt = T[1] - T[0]

    for k in range(t-1):
        k1 = rhs_2D(u_ap[:, :, k],             T,          k, x, y, nu, u)
        k2 = rhs_2D(u_ap[:, :, k] + (dt/2)*k1, T + (dt/2), k, x, y, nu, u)

        u_ap[1:-1, 1:-1, k+1] = u_ap[1:-1, 1:-1, k] + (dt/2)*(k1[1:-1, 1:-1] + k2[1:-1, 1:-1])
    return u_ap[1:-1, 1:-1, :]

def RungeKutta3_2D(x, y, T, nu, u, u_ap):
    t  = len(T)
    dt = T[1] - T[0]

    for k in range(t-1):
        k1 = rhs_2D(u_ap[:, :, k],                T,          k, x, y, nu, u)
        k2 = rhs_2D(u_ap[:, :, k] + (dt/2)*k1,    T + (dt/2), k, x, y, nu, u)
        k3 = rhs_2D(u_ap[:, :, k] + dt*(2*k2-k1), T + dt,     k, x, y, nu, u)
        u_ap[1:-1, 1:-1, k+1] = u_ap[1:-1, 1:-1, k] + (dt/6)*(k1[1:-1, 1:-1] + 4*k2[1:-1, 1:-1] + k3[1:-1, 1:-1])
    return u_ap[1:-1, 1:-1, :]

def RungeKutta4_2D(x, y, T, nu, u, u_ap):
    t  = len(T)
    dt = T[1] - T[0]

    for k in range(t-1):
        k1 = rhs_2D(u_ap[:, :, k],                T,          k, x, y, nu, u)
        k2 = rhs_2D(u_ap[:, :, k] + (dt/2)*k1,    T + (dt/2), k, x, y, nu, u)
        k3 = rhs_2D(u_ap[:, :, k] + (dt/2)*k2,    T + (dt/2), k, x, y, nu, u)
        k4 = rhs_2D(u_ap[:, :, k] + dt*k3,        T + dt,     k, x, y, nu, u)
        u_ap[1:-1, 1:-1, k+1] = u_ap[1:-1, 1:-1, k] + (dt/6)*(k1[1:-1, 1:-1] + 2*k2[1:-1, 1:-1] + 2*k3[1:-1, 1:-1] + k4[1:-1, 1:-1])
    return u_ap[1:-1, 1:-1, :]

def rhs_2D(u_ap, T, k, x, y, nu, u):
    m  = len(x[0, :])
    n  = len(y[:, 0])
    dx = x[0, 1] - x[0, 0]
    dy = y[1, 0] - y[0, 0]
    s  = np.zeros([m, n])

    for i in range(m):
        s[i, 0]  = u(x[i, 0],  y[i, 0],  T[k], nu)
        s[i, -1] = u(x[i, -1], y[i, -1], T[k], nu)
    for j in range(n):
        s[0, j]  = u(x[0, j],  y[0, j],  T[k], nu)
        s[-1, j] = u(x[-1, j], y[-1, j], T[k], nu)


    for i in range(1, m-1):
        for j in range(1, n-1):
            s[i, j] = (nu/dx**2)*(u_ap[i-1, j] - 2*u_ap[i, j] + u_ap[i+1, j]) + \
                      (nu/dy**2)*(u_ap[i, j-1] - 2*u_ap[i, j] + u_ap[i, j+1])
    
    return s