# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 15:20:45 2022

@author: Kristin
"""
import numpy as np
import matplotlib.pyplot as plt
from trace_fieldline_v7 import trace_fieldline

R_M = 2450

y_counter_array = [ 11, 12]

mercury1 = plt.Circle((0, (479/2440)), 1, color = '0.75')



directory = 'C:\\Users\Kristin\Documents\\PhD\\OCB\\Fieldlinetraces_v7_withNS_OCB_far\\'
alpha = np.arange(0.0, np.pi/2 , 0.15) # angle between x and z
beta = np.arange(0.0, np.pi, 0.15) # angle between x and y
np.savetxt(directory + 'alpha.txt', alpha)
r_hel = 0.46
di = 50


for c in range(len(y_counter_array)): 
    
    y_counter = y_counter_array[c]
    print('y_counter at: ', y_counter)


    fieldline_z_starts = np.sin(alpha) * R_M
    hilfslinie = np.cos(alpha) * R_M
    fieldline_x_starts = np.cos(beta[y_counter]) * hilfslinie
    fieldline_y_starts =  np.sin(beta[y_counter]) * hilfslinie



    check = np.zeros(len(hilfslinie))
    for a in range(len(alpha)):
        check[a] = np.sqrt(fieldline_x_starts[a]**2 + fieldline_y_starts[a]**2 + fieldline_z_starts[a]**2)


    print(fieldline_x_starts)
    print(fieldline_y_starts)
    print(fieldline_z_starts)

    fieldlinetrace_matrix = []





    for k in range(len(fieldline_x_starts)):

    
        print('k dayside= ', k)

    

        x_start= fieldline_x_starts[k]
        y_start =  -fieldline_y_starts[k]
        z_start = fieldline_z_starts[k]
    
        #print(x_start, y_start, z_start)
    

        
    

        fieldlinetrace1 = trace_fieldline(x_start, y_start, z_start, r_hel, di, delta_t=0.4)
        fieldlinetrace2 = trace_fieldline(x_start, y_start, z_start, r_hel, di, delta_t=-0.4)
        #print(fieldlinetrace)
        x = np.concatenate((fieldlinetrace1[0], fieldlinetrace2[0]))  #x
        y = np.concatenate((fieldlinetrace1[1], fieldlinetrace2[1]))   #y
        z = np.concatenate((fieldlinetrace1[2], fieldlinetrace2[2]))    #z

        np.savetxt(directory + 'fieldline_y' + str(y_counter) + '_'+str(k)+'.txt', np.array((x,y,z)))


    for k in range(len(fieldline_x_starts)):
        print('k nightside = ', k)
        
        

        x_start = -fieldline_x_starts[k]
        y_start = -fieldline_y_starts[k]
        z_start = fieldline_z_starts[k]

        #print(x_start, y_start, z_start)
    
    

        fieldlinetrace1 = trace_fieldline(x_start, y_start, z_start, r_hel, di, delta_t=0.4)
        fieldlinetrace2 = trace_fieldline(x_start, y_start, z_start, r_hel, di, delta_t=-0.4)
        #print(fieldlinetrace)
        x = np.concatenate((fieldlinetrace1[0], fieldlinetrace2[0]))  #x
        y = np.concatenate((fieldlinetrace1[1], fieldlinetrace2[1]))   #y
        z = np.concatenate((fieldlinetrace1[2], fieldlinetrace2[2]))    #z

        np.savetxt(directory + 'fieldline_y' + str(y_counter) + '_'+str(k+len(alpha))+'.txt', np.array((x,y,z)))
    
    
