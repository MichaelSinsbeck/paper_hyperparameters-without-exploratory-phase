#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 09:00:31 2018

@author: sinsbeck
"""

import numpy as np
from scipy.fftpack import fft2, ifft2

def heat_equation(x_source, output_resolution = 3):
    
    n_division = 20
    resolution = 512
    
    grid_1d = np.linspace(-1, 1, resolution + 1)[0:-1]
    x_grid, y_grid = np.meshgrid(grid_1d, grid_1d)
    
    x = x_source[0]
    y = x_source[1]
    
    x_mirror, y_mirror = np.meshgrid(np.array([x-2,-x,x,2-x]),np.array([y-2,-y,y,2-y]))
    x_mirror = x_mirror.flatten()
    y_mirror = y_mirror.flatten()
        
    h = 0.05
    s_0 = 2/ (2*np.pi * h**2)
    s = np.zeros((resolution, resolution))
    
    x_diff = x_mirror[:,np.newaxis,np.newaxis]-x_grid
    y_diff = y_mirror[:,np.newaxis,np.newaxis]-y_grid
    
    s = s_0 * np.exp(-(x_diff**2 + y_diff**2)/(2*h**2)).sum(0)
    
    # differentiation in fourier space
    ramp = np.pi * np.concatenate((np.arange(0, resolution/2+1), np.arange(resolution/2-1,0,-1)))
    lambda2 = ramp**2 + ramp[:,np.newaxis]**2
    
    # first mini-step:
    dt = 0.1*0.5**n_division
    u1 = 0.5*s*dt
    u1 = np.real(ifft2( fft2(u1) * np.exp(-lambda2*dt)))
    u1 = u1 + 0.5*s*dt
    
    # time until t=0.1
    for i in np.arange(n_division-1,-1,-1):
        dt = 0.1 * 0.5**(i+1)
        u1 = np.real(ifft2(fft2(u1)*np.exp(-lambda2*dt)) + u1)
        
    # get intermediate aoutput
    
    # Second episode until t=0.2
    dt = 0.1
    u2 = np.real(ifft2(fft2(u1)*np.exp(-lambda2*dt)) + u1)
    
    idx = np.linspace(0, resolution/2, num=output_resolution, dtype=int)
    
    measurement1 = u1[idx,idx[:,np.newaxis]].flatten()
    measurement2 = u2[idx,idx[:,np.newaxis]].flatten()
    
    measurement = np.concatenate((measurement1, measurement2))
    return measurement