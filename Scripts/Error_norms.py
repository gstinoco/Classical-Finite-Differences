'''
Codes to compute different norms for the error for all the Finite Difference Schemes.

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

import numpy as np

def Error_norms_1D(u_ap, u_ex, verbose = True):
    if np.size(u_ap.shape) == 1:                                            # Check for Transient or Stationary problem.
        L1 = L1_norm(u_ap, u_ex)
        L2 = L2_norm(u_ap, u_ex)
        LI = Infinity_norm(u_ap, u_ex)
        LQ = QME_1D_static(u_ap, u_ex)
        if verbose == True:
            print('L1 norm: \t\t', L1)
            print('L2 norm: \t\t', L2)
            print('Infinity norm: \t\t', LI)
            print('Quadratic Mean Error: \t', LQ)
    else:
        L1 = L1_norm(u_ap, u_ex)
        L2 = L2_norm(u_ap, u_ex)
        LI = Infinity_norm(u_ap, u_ex)
        LQ = QME_1D_transient(u_ap, u_ex)
        if verbose == True:
            print('L1 norm: \t\t\t', np.mean(L1))
            print('L2 norm: \t\t\t', np.mean(L2))
            print('Infinity norm: \t\t', np.mean(LI))
            print('Mean of the Quadratic Mean Error: \t', np.mean(LQ))
    return L1, L2, LI, LQ

def Error_norms_2D(u_ap, u_ex, verbose = True):
    if np.size(u_ap.shape) == 2:
        L1 = L1_norm(u_ap, u_ex)
        L2 = L2_norm(u_ap, u_ex)
        LI = Infinity_norm(u_ap, u_ex)
        LQ = QME_2D_static(u_ap, u_ex)
        if verbose == True:
            print('L1 norm: \t\t', L1)
            print('L2 norm: \t\t', L2)
            print('Infinity norm: \t\t', LI)
            print('Quadratic Mean Error: \t', LQ)
    else:
        L1 = L1_norm(u_ap, u_ex)
        L2 = L2_norm(u_ap, u_ex)
        LI = Infinity_norm(u_ap, u_ex)
        LQ = QME_2D_transient(u_ap, u_ex)
        if verbose == True:
            print('L1 norm: \t\t\t', np.mean(L1))
            print('L2 norm: \t\t\t', np.mean(L2))
            print('Infinity norm: \t\t', np.mean(LI))
            print('Mean of the Quadratic Mean Error: \t', np.mean(LQ))
    return L1, L2, LI, LQ

def L1_norm(u_ap, u_ex):
    E = np.abs(u_ap - u_ex)
    E = np.sum(E)
    return E

def L2_norm(u_ap, u_ex):
    E = np.sqrt(np.sum((u_ap - u_ex)**2))
    return E

def Infinity_norm(u_ap, u_ex):
    E = np.max(np.abs(u_ap - u_ex))
    return E

def QME_1D_static(u_ap, u_ex):
    m = u_ap.shape
    E = np.sqrt(((np.sum(u_ap[:] - u_ex[:]))**2)/m)
    return E[0]

def QME_2D_static(u_ap, u_ex):
    m, n = u_ap.shape
    E    = np.sqrt(((np.sum(u_ap[:, :] - u_ex[:, :]))**2)/(m*n))
    return E

def QME_1D_transient(u_ap, u_ex):
    m, t = u_ap.shape
    E      = np.zeros(t)
    for k in np.arange(t):
        E[k] = np.sqrt(((np.sum(u_ap[:, k] - u_ex[:, k]))**2)/m)
    return E

def QME_2D_transient(u_ap, u_ex):
    m, n, t = u_ap.shape
    E      = np.zeros(t)
    for k in np.arange(t):
        E[k] = np.sqrt(((np.sum(u_ap[:, :, k] - u_ex[:, :, k]))**2)/(m*n))
    return E