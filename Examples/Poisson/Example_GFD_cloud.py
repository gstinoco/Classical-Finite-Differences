# Path Importation
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.insert(1, root_dir)

# Library Importation
import numpy as np
import matplotlib.pyplot as plt
from Poisson_GFD import Poisson_GFD_iterative
from Poisson_GFD import Poisson_GFD_Matrix
from Scripts.Graphs import Graph_2D_Static

m = 21
n = 21

f    = lambda x,y: 10*np.exp(2*x+y)
u    = lambda x,y: 2*np.exp(2*x+y)

x    = np.linspace(0,1,m)
y    = np.logspace(0,1,n)
y   /= max(y)

plt.figure()
plt.scatter(x, y)
plt.show()


A = C = 2
B = D = E = 0
L = np.vstack([[D], [E], [A], [B], [C]])

#u_ap, u_ex = Poisson_GFD_iterative(x, y, u, f, L)
#u_ap, u_ex = Poisson_GFD_Matrix(x, y, u, f, L)

#Graph_2D_Static(x, y, u_ap, u_ex)