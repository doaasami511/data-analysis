# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 08:04:14 2019

@author: hp
"""
import numpy as np
from scipy.optimize import minimize 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def fun(u):
    x=u[0]
    y=u[1]
    z=3*(1-x)**2*np.exp(-x**2-(y+1)**2)-10*(x/5-x**3-y**5)*np.exp(-x**2-y**2)-(np.exp(-(x-1)**2-y**2)/3)
    return z
   
    
    

x0=[3,-3]
res = minimize(fun, x0, method='powell',
               options={'xtol': 1e-8, 'disp': True})#global minimum
print(res)

x = np.linspace(-3, 2, 20)
y = np.linspace(-3, 2, 20)

a,b = np.meshgrid(x, y)
u=[a,b]
Z = fun(u)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(a, b, Z, 50, cmap='binary')

x0=[5,7]
res = minimize(fun, x0, method='powell',options={'xtol': 1e-8, 'disp': True})
a=[(1,2),(1,3),(4,2),(3,-3)] #local and global minimum
for i in a:
    res = minimize(fun, i, method='powell',options={'xtol': 1e-8, 'disp': True})
    print(res)