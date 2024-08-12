#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 22:56:40 2021

@author: Sarat Moka

Python code for generating 3d-plots of 
a convex function with a (unique) global minimum 
and a non-convex function with several local extrema.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm

textsize = 20

#mycmap = plt.get_cmap('gist_earth')
#mycmap = plt.get_cmap('magma')

# =============================================================================
# Covex and non-convex functions
# =============================================================================
f_non_convex = lambda t: 3*((1 - t[0])**2)*np.exp(-t[0]**2 - (t[1] + 1)**2) - 10*(t[0]/5 - t[0]**3 - t[1]**5)*np.exp(-t[0]**2 - t[1]**2) - (1/3)*np.exp(-(t[0]+1)**2 - t[1]**2)
f_convex = lambda t: t[0]**2 + t[1]**2


#%%
# =============================================================================
# This cell for 3d plot of a convex function
# =============================================================================

t1_range = np.arange(-2.5, 2.5, 0.01)
t2_range = np.arange(-2.5, 2.5, 0.01)

A = np.meshgrid(t1_range, t2_range)
Z = f_convex(A)
X, Y = np.meshgrid(t1_range, t2_range)

fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=5,  antialiased=False, edgecolor='none', vmin =np.min(Z), vmax =1*np.max(Z))
#ax.plot_surface(X, Y, Z, linewidth=2, rcount=50, ccount=50, antialiased=False, edgecolor='none', vmin =np.min(Z), vmax =1.5*np.max(Z))

ax.view_init(40, -56)
ax.set_xlabel(r'$\theta_1$', fontsize=textsize)
ax.set_ylabel(r'$\theta_2$', fontsize=textsize)
#plt.xticks(size=textsize)
#plt.yticks(size=textsize)
plt.show()

#%%
# =============================================================================
# This cell for contour plot of a convex function
# =============================================================================

t1_range = np.arange(-2.5, 2.5, 0.01)
t2_range = np.arange(-2.5, 2.5, 0.01)

A = np.meshgrid(t1_range, t2_range)
Z = f_convex(A)
X, Y = np.meshgrid(t1_range, t2_range)

# Create figure and axes
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)

# Create contour plot
ax.contour(X, Y, Z, levels=np.exp(np.arange(np.min(Z)-8, 0.4*np.max(Z), 0.1)), locator=ticker.LogLocator(), cmap=cm.PuBu_r, linewidths=2, zorder=0)
#ax.contour(X, Y, Z, levels=np.exp(np.arange(np.min(Z)-8, 0.2*np.max(Z), 0.1)),  locator=ticker.LogLocator(), cmap='viridis', linewidths=1, zorder=0)

# Set labels and view angle
ax.set_xlabel(r'$\theta_1$', fontsize=20)
ax.set_ylabel(r'$\theta_2$', fontsize=20)



plt.show()

#%%
# =============================================================================
# This cell plots a non-convex function
# =============================================================================

t1_range = np.arange(-2.5, 2.5, 0.005)
t2_range = np.arange(-2.5, 2.5, 0.005)

A = np.meshgrid(t1_range, t2_range)
Z = f_non_convex(A)
X, Y = np.meshgrid(t1_range, t2_range)

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False, edgecolor='none', vmin =np.min(Z), vmax =2*np.max(Z), alpha=1)
ax.view_init(22, -56)
ax.set_xlabel(r'$\theta_1$', fontsize=textsize)
ax.set_ylabel(r'$\theta_2$', fontsize=textsize)
ax.set_zlabel(r'$C(\theta_1, \theta_2)$', fontsize=textsize)
#plt.savefig('non-convex-3d.png', edgecolor='none', bbox_inches='tight')
plt.show()


#%%
# =============================================================================
# This cell for contour plot of a non-convex function
# =============================================================================

t1_range = np.arange(-2.5, 2.5, 0.01)
t2_range = np.arange(-2.5, 2.5, 0.01)

A = np.meshgrid(t1_range, t2_range)
Z = f_non_convex(A)
X, Y = np.meshgrid(t1_range, t2_range)

# Create figure and axes
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)

##### Create contour plot
# Number of segments
num_segments = 100
Z_min = np.min(Z)-3
Z_max = 1.3*np.max(Z)

exp_values = np.exp(np.linspace(0, 1, num_segments))
norm_vals = (exp_values - exp_values.min()) / (exp_values.max() - exp_values.min())


norm_vals_r = -norm_vals.copy() + 1
norm_vals_r = norm_vals_r[::-1][1:]

Z_mid = (Z_max + Z_min)/2

left_arr = norm_vals*(Z_mid - Z_min) + Z_min
right_arr = norm_vals_r*(Z_max - Z_mid) + Z_mid

arr = np.concatenate((left_arr, right_arr))


ax.contour(X, Y, Z, levels=arr,  cmap=cm.PuBu_r, linewidths=1)
#ax.contour(X, Y, Z, levels=np.exp(np.arange(np.min(Z)-8, 0.2*np.max(Z), 0.1)),  locator=ticker.LogLocator(), cmap='viridis', linewidths=1, zorder=0)

# Set labels and view angle
ax.set_xlabel(r'$\theta_1$', fontsize=20)
ax.set_ylabel(r'$\theta_2$', fontsize=20)

plt.show()


#%%
# =============================================================================
# This cell for surface plot of a CE loss function
# =============================================================================

X = np.array([-3,-2,-1,1,2,3])
y = np.array([1, 0, 0, 1, 1, 0])

def sigmoid(x, w, b):    
    return 1/(1 + np.exp(b + w*x))

def celoss(w, b):
    n = y.shape[0]
    s = 0.0
    
    for i in range(n):
        y_hat = sigmoid(X[i], w, b)
        s += y[i]*np.log(y_hat) + (1 - y[i])*np.log(1 - y_hat)
    
    return -s/n

def loss2(w, b):
    n = y.shape[0]
    s = 0.0
    for i in range(n):
        s += (y[i] - sigmoid(X[i], w, b))**2
    
    return s

w_min = -5
w_max = 5
b_min = -10
b_max = 10

w_range = np.arange(w_min, w_max, 0.01)
b_range = np.arange(b_min, b_max, 0.01)

w_grid, b_grid = np.meshgrid(w_range, b_range)
loss_grid = loss2(w_grid, b_grid)
#loss_grid = celoss(w_grid, b_grid)

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')


ax.plot_surface(w_grid, b_grid, loss_grid, cmap='Blues_r', rstride=1, cstride=1, linewidth=0, antialiased=False, edgecolor='none', vmin =np.min(loss_grid), vmax =2*np.max(loss_grid))
ax.set_ylabel(r'$\theta_1$', fontsize=20)
ax.set_xlabel(r'$\theta_2$', fontsize=20)
ax.set_zlabel(r'$C(\theta_1, \theta_2)$', fontsize=20)  
ax.view_init(25, -120)

plt.show()

#%%
# =============================================================================
# This cell for contour plot of a CE loss function
# =============================================================================

X = np.array([-3,-2,-1,1,2,3])
y = np.array([1, 0, 0, 1, 1, 0])

def sigmoid(x, w, b):    
    return 1/(1 + np.exp(b + w*x))

def celoss(w, b):
    n = y.shape[0]
    s = 0.0
    
    for i in range(n):
        y_hat = sigmoid(X[i], w, b)
        s += y[i]*np.log(y_hat) + (1 - y[i])*np.log(1 - y_hat)
    
    return -s/n

def loss2(w, b):
    n = y.shape[0]
    s = 0.0
    for i in range(n):
        s += (y[i] - sigmoid(X[i], w, b))**2
    
    return s

w_min = -5
w_max = 5
b_min = -10
b_max = 10

w_range = np.arange(w_min, w_max, 0.01)
b_range = np.arange(b_min, b_max, 0.01)

w_grid, b_grid = np.meshgrid(w_range, b_range)
loss_grid = loss2(w_grid, b_grid)
#loss_grid = celoss(w_grid, b_grid)

num_segments = 150
Z_min = np.min(loss_grid)-1
Z_max = 1*np.max(loss_grid)
exp_values = np.exp(np.linspace(0, 1, num_segments))
norm_vals = (exp_values - exp_values.min()) / (exp_values.max() - exp_values.min())
arr = norm_vals*(Z_max - Z_min) + Z_min

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)


#ax.contour(w_grid, b_grid, loss_grid, levels=arr,   cmap='Blues_r', linewidths=1, zorder=0)
ax.contour(w_grid, b_grid, loss_grid, levels=np.arange(Z_min-1, Z_max+1, 0.02),   cmap=cm.PuBu_r, linewidths=1, zorder=0)
ax.set_ylabel(r'$\theta_1$', fontsize=20)
ax.set_xlabel(r'$\theta_2$', fontsize=20)

plt.show()