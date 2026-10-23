import numpy as np

from poly_funcs import *

from scipy.spatial import ConvexHull, Delaunay

import pickle

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
import matplotlib.ticker as mtick
from matplotlib.ticker import MaxNLocator


arm_name = '3R'

fig = plt.figure(figsize=(10,10))
axs = fig.add_subplot(1,1,1, projection='3d')

with open(f'./regions/{arm_name}_c_free_polys.pkl', 'rb') as f:
    c_free = pickle.load(f)

for j, region in enumerate(c_free):

    colors = ['blue']#, 'blue', 'orange', 'magenta', 'pink', 'purple']
    r_vert = poly_to_vert(region).T
    r_hull = ConvexHull(r_vert.T)
    for s in r_hull.simplices:
        tri = Poly3DCollection([r_vert.T[s]])
        tri.set_color(colors[j%len(colors)])
        tri.set_edgecolor('none')
        tri.set_alpha(0.35)
        axs.add_collection3d(tri)


axs.set_xlim([-np.pi,np.pi])
axs.set_ylim([-np.pi,np.pi])
axs.set_zlim([-np.pi,np.pi])

axs.set_xlabel(r'$\theta_{1}$',fontsize=28)
axs.set_ylabel(r'$\theta_{2}$',fontsize=28)
axs.set_zlabel(r'$\theta_{3}$',fontsize=28,rotation=0)

plt.show()

del axs
del fig
