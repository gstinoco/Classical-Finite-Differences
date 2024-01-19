# Path Importation
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
root_dir    = os.path.dirname(parent_dir)
sys.path.insert(1, root_dir)

import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import Scripts.Graphs as Graphs
import Diffusion_Equation

mat  = loadmat('Data/CUA_1_hole.mat')
p   = mat['p']
tt  = mat['tt']
if tt.min() == 1:
    tt -= 1

# Boundary and problem conditions
u       = lambda x,y,t,nu: np.exp(-2*np.pi**2*nu*t)*np.cos(np.pi*x)*np.cos(np.pi*y)
nu      = 0.2
t       = 1000

u_ap, u_ex, vec = Diffusion_Equation.Cloud(p, u, nu, t)

Graphs.Cloud_Transient(p, tt, u_ap, u_ex)