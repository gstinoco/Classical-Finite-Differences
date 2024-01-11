# Path Importation
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.insert(1, root_dir)

import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import Scripts.Graphs as Graphs
import Poisson_GFD

mat  = loadmat('Data/CUA_1_hole.mat')
p   = mat['p']
tt  = mat['tt']
if tt.min() == 1:
    tt -= 1

# Boundary and problem conditions
#   u = 2e^{2x+y}
#
#   f = 10e^{2x+y}

f    = lambda x,y: 10*np.exp(2*x+y)
u    = lambda x,y: 2*np.exp(2*x+y)

u_ap, u_ex, vec = Poisson_GFD.Cloud(p, tt, u, f)

Graphs.Cloud_Static(p, tt, u_ap, u_ex)