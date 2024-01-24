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

def Error_norms_1D(u_ap, u_ex):
    if np.size(u_ap.shape) == 1:                                            # Check for Transient or Stationary problem.
        print('L1 norm: \t\t', L1_norm(u_ap, u_ex))
        print('L2 norm: \t\t', L2_norm(u_ap, u_ex))
        print('Infinity norm: \t\t', Infinity_norm(u_ap, u_ex))
        print('Quadratic Mean Error: \t', QME(u_ap, u_ex))
    else:
        print('Mean of the L1 norm: \t\t\t', np.mean(L1_norm(u_ap, u_ex)))
        print('Mean of the L2 norm: \t\t\t', np.mean(L2_norm(u_ap, u_ex)))
        print('Mean of the Infinity norm: \t\t', np.mean(Infinity_norm(u_ap, u_ex)))
        print('Mean of the Quadratic Mean Error: \t', np.mean(QME(u_ap, u_ex)))

def Error_norms_2D(u_ap, u_ex):
    if np.size(u_ap.shape) == 2:                                            # Check for Transient or Stationary problem.
        print('L1 norm: \t\t', L1_norm(u_ap, u_ex))
        print('L2 norm: \t\t', L2_norm(u_ap, u_ex))
        print('Infinity norm: \t\t', Infinity_norm(u_ap, u_ex))
        print('Quadratic Mean Error: \t', QME(u_ap, u_ex))
    else:
        print('Mean of the L1 norm: \t\t\t', np.mean(L1_norm(u_ap, u_ex)))
        print('Mean of the L2 norm: \t\t\t', np.mean(L2_norm(u_ap, u_ex)))
        print('Mean of the Infinity norm: \t\t', np.mean(Infinity_norm(u_ap, u_ex)))
        print('Mean of the Quadratic Mean Error: \t', np.mean(QME(u_ap, u_ex)))

def L1_norm(u_ap, u_ex):
    E = np.abs(u_ap - u_ex)
    E = np.sum(E)
    return E

def L2_norm(u_ap, u_ex):
    E = (u_ap - u_ex)**2
    E = np.sum(E)
    E = np.sqrt(E)
    return E

def Infinity_norm(u_ap, u_ex):
    E = np.abs(u_ap - u_ex)
    E = np.max(E)
    return E

def QME(u_ap, u_ex):
    E = (u_ap - u_ex)**2
    E = np.mean(E)
    return E