# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 18:13:19 2022

@author: Kristin
"""

import numpy as np #
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

R_M = 2440
mu_0 = 4 * np.pi * 10**(-7)

direction = 'southward'
directory = 'C:\\Users\\Kristin\\Documents\PhD\FAC\Residuen_imf_sorted\\'+ direction + '\\'
diff_x_matrix = np.loadtxt(directory + 'diff_x_matrix.txt')
diff_y_matrix = np.loadtxt(directory + 'diff_y_matrix.txt')
diff_z_matrix = np.loadtxt(directory + 'diff_z_matrix.txt')

diff_r_matrix = np.loadtxt(directory + 'diff_r_matrix.txt')
diff_theta_matrix = np.loadtxt(directory + 'diff_theta_matrix.txt')
diff_phi_matrix = np.loadtxt(directory + 'diff_phi_matrix.txt')

theta = np.loadtxt(directory + 'theta_steps.txt')
phi = np.loadtxt(directory + 'phi_steps.txt')

r_mean = np.loadtxt(directory + 'r_mean.txt')
theta_mean = np.loadtxt(directory + 'theta_mean.txt')
phi_mean = np.loadtxt(directory + 'phi_mean.txt')
print(r_mean)
print(theta_mean)
print(phi_mean)
#theta_steps = 10*2*np.pi/360
#phi_steps = 10*2*np.pi/360

theta_steps = 10
phi_steps = 10


rot_B_matrix = np.zeros([len(phi)-1, len(theta)-1])
rot_B_matrix_2 = np.zeros([len(phi)-1, len(theta)-1])
rot_B_matrix_3 = np.zeros([len(phi)-1, len(theta)-1])
#j_4 = np.zeros([len(phi)-1, len(theta)-4]) #gleiche matrix wie j_3, ohne die äußersten 3 ringe (theta nur von 0 bis 60°)


for i in range(len(theta)-1):
    print('i= ', i, '/8')
    if i == 0: 
        continue
    if i == 8: 
        continue
    for j in range(len(phi)-1): 
        if j == 0: 
            rot_B_matrix_3[j,i] = 1/(R_M*1000) *(((diff_phi_matrix[j, i+1]-diff_phi_matrix[j, i-1])/(2*theta_steps))-(diff_phi_matrix[j,i]*(np.cos(theta_mean[j, i])/np.sin(theta_mean[j, i])))-((diff_theta_matrix[j+1,i] -diff_theta_matrix[35,i])/(2*phi_steps))/np.sin(theta_mean[j,i]))
        elif j == 35: 
            rot_B_matrix_3[j,i] = 1/(R_M*1000) *(((diff_phi_matrix[j, i+1]-diff_phi_matrix[j, i-1])/(2*theta_steps))-(diff_phi_matrix[j,i]*(np.cos(theta_mean[j, i])/np.sin(theta_mean[j, i])))-((diff_theta_matrix[0,i] -diff_theta_matrix[j-1,i])/(2*phi_steps))/np.sin(theta_mean[j,i]))
        
        else: 
            print('j: ', j)
            ending = 'theta'+ str(int(theta[i]))+'_'+str(int(theta[i+1]))+'phi'+str(int(phi[j]))+'_'+str(int(phi[j+1]))+'.txt'
        
            print('ending: ', ending)
        #rot_B_matrix[j,i] = 1/(r_mean[j,i] * np.sin(theta_mean[j,i])) * ((((diff_phi_matrix[j, i+1]-diff_phi_matrix[j, i-1])*np.sin(theta_mean[j, i])/2*theta_steps) + np.cos(theta_mean[j,i])*diff_phi_matrix[j,i])-(diff_theta_matrix[j+1,i] -diff_theta_matrix[j-1,i])/(2*phi_steps))
        #rot_B_matrix[j,i] = 1/(r_mean[j,i]*1000 * np.sin(theta_mean[j,i])) * ((((diff_phi_matrix[j, i+1]-diff_phi_matrix[j, i-1])*np.sin(theta_mean[j, i])/(2*theta_steps)) + np.cos(theta_mean[j,i])*diff_phi_matrix[j,i])-(diff_theta_matrix[j+1,i] -diff_theta_matrix[j-1,i])/(2*phi_steps))
        #rot_B_matrix_2[j,i] = 1/(R_M*1000 * np.sin(theta_mean[j,i])) * ((((diff_phi_matrix[j, i+1]-diff_phi_matrix[j, i-1])*np.sin(theta_mean[j, i])/(2*theta_steps)) + np.cos(theta_mean[j,i])*diff_phi_matrix[j,i])-(diff_theta_matrix[j+1,i] -diff_theta_matrix[j-1,i])/(2*phi_steps))
            rot_B_matrix_3[j,i] = 1/(R_M*1000) *(((diff_phi_matrix[j, i+1]-diff_phi_matrix[j, i-1])/(2*theta_steps))-(diff_phi_matrix[j,i]*(np.cos(theta_mean[j, i])/np.sin(theta_mean[j, i])))-((diff_theta_matrix[j+1,i] -diff_theta_matrix[j-1,i])/(2*phi_steps))/np.sin(theta_mean[j,i]))
        
        
        
j = 1/mu_0 * rot_B_matrix
j_2 = 1/mu_0 * rot_B_matrix_2
j_3 = 1/mu_0 * rot_B_matrix_3
j_4 = j_3[0:36, 0:6]


    
rad = np.linspace(0, 90, 10)
azm = np.linspace(0, 2 * np.pi, 37)

diff_B = rot_B_matrix- rot_B_matrix_2
    
lim = 50.0
'''   
fig = plt.figure()
ax = Axes3D(fig)
#rad = np.linspace(0, 90, 10)
#azm = np.linspace(0, 2 * np.pi, 37)
r, th = np.meshgrid(rad, azm)
#z = (r ** 2.0) / 4.0
z = j
#z = np.random.uniform(-1, 1, (n,m))
plt.subplot(projection="polar")
plt.pcolormesh(th, r, z, cmap = 'seismic')
#plt.pcolormesh(th, z, r)
plt.clim(-lim, lim)
plt.plot(azm, r,   ls='none') 
plt.grid()
cbar = plt.colorbar()
cbar.set_label('   nA/$m^2$', labelpad=-40, y=1.05, rotation=0)
plt.title('rot B (radial)')
plt.show()
fig = plt.figure()


fig = plt.figure()
ax = Axes3D(fig)
#rad = np.linspace(0, 90, 10)
#azm = np.linspace(0, 2 * np.pi, 37)
r, th = np.meshgrid(rad, azm)
#z = (r ** 2.0) / 4.0
z = j_2
#z = np.random.uniform(-1, 1, (n,m))
plt.subplot(projection="polar")
plt.pcolormesh(th, r, z, cmap = 'seismic')
#plt.pcolormesh(th, z, r)
plt.clim(-lim, lim)
plt.plot(azm, r,   ls='none') 
plt.grid()
cbar = plt.colorbar()
cbar.set_label('   nA/m$^2$', labelpad=-40, y=1.05, rotation=0)
plt.title('rot B (radial) mit Faktor R$_M$')
plt.show()
'''

fig = plt.figure()
ax = Axes3D(fig)
#rad = np.linspace(0, 90, 10)
#azm = np.linspace(0, 2 * np.pi, 37)
r, th = np.meshgrid(rad, azm)
#z = (r ** 2.0) / 4.0
z = j_3
#z = np.random.uniform(-1, 1, (n,m))
plt.subplot(projection="polar")
plt.pcolormesh(th, r, z, cmap = 'seismic')
#plt.pcolormesh(th, z, r)
plt.clim(-lim, lim)
plt.plot(azm, r,   ls='none') 
plt.grid()
cbar = plt.colorbar()
cbar.set_label('   nA/m$^2$', labelpad=-40, y=1.05, rotation=0)
plt.title('j (radial) mit Faktor R$_M$, ' + direction)
plt.show()


rad = np.linspace(0, 60, 7)
print('rad: ', rad)
azm = np.linspace(0, 2 * np.pi, 37)

fig = plt.figure()
ax = Axes3D(fig)
#rad = np.linspace(0, 90, 10)
#azm = np.linspace(0, 2 * np.pi, 37)
r, th = np.meshgrid(rad, azm)
#z = (r ** 2.0) / 4.0
z = j_4
#z = np.random.uniform(-1, 1, (n,m))
plt.subplot(projection="polar")
plt.pcolormesh(th, r, z, cmap = 'seismic')
#plt.pcolormesh(th, z, r)
plt.clim(-lim, lim)
plt.plot(azm, r,   ls='none') 
plt.grid()
cbar = plt.colorbar()
cbar.set_label('   nA/m$^2$', labelpad=-40, y=1.05, rotation=0)
plt.title('j (radial) mit Faktor R$_M$, zoomed, ' + direction)
plt.savefig(directory + 'j_radial_zoomed_5deg.png')
plt.show()



