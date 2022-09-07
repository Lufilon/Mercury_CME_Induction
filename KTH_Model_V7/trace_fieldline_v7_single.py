# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 16:01:46 2022

@author: Kristin
"""

import numpy as np
import matplotlib.pyplot as plt
from trace_fieldline_v7 import trace_fieldline

R_M = 2440
#plt.close()
mercury1 = plt.Circle((0, (479/2440)), 1, color = '0.75')
directory = 'C:\\Users\Kristin\Documents\PhD\OCB\Fieldlinetraces_v7_onlyDipole\\'


x_start= -1.2*R_M
y_start = 0
z_start = 479
r_hel = 0.37
di = 50
    

fieldlinetrace = trace_fieldline(x_start, y_start, z_start, r_hel, di, delta_t=0.3)
fieldlinetrace = trace_fieldline(x_start, y_start, z_start, r_hel, di, delta_t=-0.3)
#print(fieldlinetrace)
x = fieldlinetrace[0]   #x
y = fieldlinetrace[1]   #y
z = fieldlinetrace[2]   #z
    


fig, ax1 = plt.subplots()
ax1.add_artist(mercury1) 
plt.plot(x/R_M, z/R_M)
ax1.axis('square')
plt.xlim((-5, 3))
plt.ylim((-3, 3)) 
ax1.grid()
ax1.invert_xaxis()
    