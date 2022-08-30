# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 13:57:58 2022

@author: Kristin
"""
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




directory = 'C:\\Users\\Kristin\\Documents\PhD\FAC\\Residuen_plots_6panels\\'

#ending_panel = 'bz_neg.txt'
#ending_panel = 'DI_high.txt'
ending_panel = 'tao_5_low.txt'
diff_x_matrix = np.loadtxt(directory + 'diff_x_matrix_leveled_' + ending_panel)
diff_y_matrix = np.loadtxt(directory + 'diff_y_matrix_leveled_' + ending_panel)
diff_z_matrix = np.loadtxt(directory + 'diff_z_matrix_leveled_' + ending_panel)

diff_r_matrix = np.loadtxt(directory + 'diff_r_matrix_leveled_' + ending_panel)
diff_theta_matrix = np.loadtxt(directory + 'diff_theta_matrix_leveled_' + ending_panel)
diff_phi_matrix = np.loadtxt(directory + 'diff_phi_matrix_leveled_' + ending_panel)

theta = np.loadtxt(directory + 'theta_steps_leveled.txt')
phi = np.loadtxt(directory + 'phi_steps_leveled.txt')

r_mean = np.loadtxt(directory + 'r_mean_rm_' + ending_panel)
theta_mean = np.loadtxt(directory + 'theta_mean_leveled_1.txt')
phi_mean = np.loadtxt(directory + 'phi_mean_leveled_1.txt')
print(r_mean)
print(theta_mean)
print(phi_mean)
theta_steps = 10*2*np.pi/360
phi_steps = 10*2*np.pi/360

#theta_steps = 10
#phi_steps = 10


rot_B_matrix = np.zeros([len(phi)-1, len(theta)-1])
#j_4 = np.zeros([len(phi)-1, len(theta)-4]) #gleiche matrix wie j_3, ohne die äußersten 3 ringe (theta nur von 0 bis 60°)


for i in range(len(theta)-1):
    print('i= ', i, '/8')
    if i == 0: 
        continue
    if i == 8: 
        continue
    for j in range(len(phi)-1): 
        if j == 0: 
            rot_B_matrix[j,i] = 1/(R_M*1000) *(((diff_phi_matrix[j, i+1]-diff_phi_matrix[j, i-1])/(2*theta_steps))-(diff_phi_matrix[j,i]*(np.cos(theta_mean[j, i])/np.sin(theta_mean[j, i])))-((diff_theta_matrix[j+1,i] -diff_theta_matrix[35,i])/(2*phi_steps))/np.sin(theta_mean[j,i]))
        elif j == 35: 
            rot_B_matrix[j,i] = 1/(R_M*1000) *(((diff_phi_matrix[j, i+1]-diff_phi_matrix[j, i-1])/(2*theta_steps))-(diff_phi_matrix[j,i]*(np.cos(theta_mean[j, i])/np.sin(theta_mean[j, i])))-((diff_theta_matrix[0,i] -diff_theta_matrix[j-1,i])/(2*phi_steps))/np.sin(theta_mean[j,i]))
        
        else: 
            print('j: ', j)
            ending = 'theta'+ str(int(theta[i]))+'_'+str(int(theta[i+1]))+'phi'+str(int(phi[j]))+'_'+str(int(phi[j+1]))+'.txt'
        
            print('ending: ', ending)
            rot_B_matrix[j,i] = 1/(R_M*1000) *(((diff_phi_matrix[j, i+1]-diff_phi_matrix[j, i-1])/(2*theta_steps))-(diff_phi_matrix[j,i]*(np.cos(theta_mean[j, i])/np.sin(theta_mean[j, i])))-((diff_theta_matrix[j+1,i] -diff_theta_matrix[j-1,i])/(2*phi_steps))/np.sin(theta_mean[j,i]))
        
        
        
j_y = 1/mu_0 * rot_B_matrix
j_y_zoom = j_y[0:36, 0:6]
print(ending_panel)
np.savetxt(directory + 'j_y_' + ending_panel, j_y)
np.savetxt(directory + 'j_y_zoomed_' + ending_panel, j_y_zoom)
    
rad = np.linspace(0, 90, 10)
azm = np.linspace(0, 2 * np.pi, 37)


    
lim = 500.0
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
plt.title('rot B (radial), scaled with R$_M$')
plt.show()
'''

fig = plt.figure(figsize = (8,6))
ax = Axes3D(fig)
#rad = np.linspace(0, 90, 10)
#azm = np.linspace(0, 2 * np.pi, 37)
r, th = np.meshgrid(rad, azm)
#z = (r ** 2.0) / 4.0
z = j_y
#z = np.random.uniform(-1, 1, (n,m))
plt.subplot(projection="polar")
plt.pcolormesh(th, r, z, cmap = 'seismic')
#plt.pcolormesh(th, z, r)
plt.clim(-lim, lim)
plt.plot(azm, r,   ls='none') 
cbar = plt.colorbar()
cbar.set_label('   nA/m$^2$', labelpad=-40, y=1.05, rotation=0)
plt.title('j (radial), scaled with R$_M$, with pedersen correction')
#plt.savefig(directory + 'j_radial_' + plasma_beta + '_IMF_Bz_' + bz_direction + '.png')
plt.show()


rad = np.linspace(0, 60, 7)
print('rad: ', rad)
azm = np.linspace(0, 2 * np.pi, 37)

fig = plt.figure(figsize = (8,6))
ax = Axes3D(fig)
#rad = np.linspace(0, 90, 10)
#azm = np.linspace(0, 2 * np.pi, 37)
r, th = np.meshgrid(rad, azm)
#z = (r ** 2.0) / 4.0
z = j_y_zoom
#z = np.random.uniform(-1, 1, (n,m))
plt.subplot(projection="polar")
plt.pcolormesh(th, r, z, cmap = 'seismic')
#plt.pcolormesh(th, z, r)
plt.clim(-lim, lim)
plt.plot(azm, r,   ls='none') 
cbar = plt.colorbar()
cbar.set_label('   nA/m$^2$', labelpad=-40, y=1.05, rotation=0)
plt.title('j (radial), scaled with R$_M$, zoomed, with pedersen correction' )
#plt.savefig(directory + 'j_radial_zoomed_' + plasma_beta + '_IMF_Bz_' + bz_direction + '.png')
plt.show()



