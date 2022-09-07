# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 15:23:17 2022

@author: Kristin
"""

'''
This function calculates the trace of a test particle ( = fieldline) in the magnetosphere of Mercury using the runge-kutta-algorithm of 4th order.

Input:  x,y,z coordinates in MSO coordinate system in km, startpoint of the fieldline
        delta_t, timesteps in runge kutta algorithm, default is delta_t = 0.3, also negative delta_t are possible, to get the other direction
        (towards or away from the planet)

        r_hel, heliocentric distance in AU, must be between 0.307 and 0.466 AU
        di, disturbance index, must be between 0 and 100. If you don't know the di, use 50 (average)

Output: x,y,z array, points on the field line (trace of test particle) in MSO coordinate system.
        Bx, By, Bz, magnetic field for each given point on the field line

The Program calculates a maximum of i = 5000 iterations. If you want to change this, change the value in the while-loop.

'''

import numpy as np
from kth14_model_for_mercury_v7 import kth14_model_for_mercury_v7

def trace_fieldline(x_start, y_start, z_start, r_hel, di, delta_t=0.3):

    R_M = 2440  #Radius of Mercury in km
    r = np.sqrt((x_start / R_M) ** 2 + (y_start / R_M) ** 2 + (z_start / R_M) ** 2)

    if r < 1:
        print('Radius of start point is smaller than 1. You start inside the planet! Radius = ', r)
        exit()
    #else:
        #print('Radius = ', r)

    def f(x, y, z):
        return kth14_model_for_mercury_v7(x, y, z, r_hel, di, True, True, False, True, True)

    x_trace = [x_start]
    y_trace = [y_start]
    z_trace = [z_start]

    i = 0

    while r > 1 and i < 4000:
        

        #print('i = ', i)
        r = np.sqrt((x_trace[i] / R_M) ** 2 + (y_trace[i] / R_M) ** 2 + (z_trace[i] / R_M) ** 2)
        if r < 1.001:
            #print('r < 1RM')
            break
        '''
        if i > 1 and (z_trace[1] > z_start):    #fieldline starts northward
            if z_trace[i] < 479:                # stop if fieldline crosses mag. equator
                break
        
        if i > 1 and (z_trace[1] < z_start):    #fieldline starts southward
            if z_trace[i] > 479:                 # stop if fieldline crosses mag. equator
                break
        '''    
        B = f(x_trace[i], y_trace[i], z_trace[i])
        #p = np.array([x_trace[i], y_trace[i], z_trace[i]])

        k1 = delta_t * B

        k2 = delta_t * f(x_trace[i] + 0.5 * k1[0], y_trace[i] + 0.5 * k1[1], z_trace[i] + 0.5 + k1[2])

        k3 = delta_t * f(x_trace[i] + 0.5 * k2[0], y_trace[i] + 0.5 * k2[1], z_trace[i] + 0.5 * k2[2])

        k4 = delta_t * f(x_trace[i] + k3[0], y_trace[i] + k3[1], z_trace[i] + k3[2])

        x_trace.append(x_trace[i] + (1 / 6) * (k1[0] + 2 * k2[0] + 3 * k3[0] + k4[0]))
        y_trace.append(y_trace[i] + (1 / 6) * (k1[1] + 2 * k2[1] + 3 * k3[1] + k4[1]))
        z_trace.append(z_trace[i] + (1 / 6) * (k1[2] + 2 * k2[2] + 3 * k3[2] + k4[2]))

        i = i + 1

        x_array = np.asarray(x_trace)
        y_array = np.asarray(y_trace)
        z_array = np.asarray(z_trace)
        
        
        usable_indices = np.loadtxt('C:\\Users\\Kristin\\Documents\\PhD\\KTH_Model_V7\\usable_indices.txt')
        if len(np.atleast_1d(usable_indices)) == 0: 
            break
        
        if x_array[-1] <= -4 * R_M: 
            break
        

    return (np.array([x_array, y_array, z_array]))

