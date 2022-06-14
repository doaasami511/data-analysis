from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt



def f(x, y):
    return 2*x + y

x1 = np.arange(0, 2.1, 0.05)
x2 = np.arange(0, 4, 0.05)

X1,X2 = np.meshgrid(x1,x2)
Z=f(X1,X2)

Z[X1 + X2 > 3]=np.nan
Z[X1 + 2*X2 > 5]=np.nan

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_wireframe(X1, X2, Z,cmap=plt.cm.hot)
plt.show()

# Optimize By Scipy

from scipy.optimize import linprog
C  = [2,1]
A  = [[1,1],[1,2]]
B  = [3,5]
x1_bnds = (0,2)
x2_bnds = (0,None)
res = linprog(C, A, B, bounds=(x1_bnds, x2_bnds))
print(res)