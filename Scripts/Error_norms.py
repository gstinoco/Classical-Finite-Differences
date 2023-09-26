import numpy as np

def E_inf(u_ap, u_ex):
    E     = abs(u_ap - u_ex)
    E_inf = E.max()
    return E_inf

def E_uno(u_ap, u_ex):
    m     = len(u_ap)
    E     = abs(u_ap - u_ex)
    E_uno = (1/m)*np.sum(abs(E))
    return E_uno

def E_dos(u_ap, u_ex):
    m      = len(u_ap)
    E      = abs(u_ap-u_ex)
    E      = np.sqrt((1/m)*np.sum((E**2)))

    return E