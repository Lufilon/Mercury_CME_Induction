# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 16:16:46 2022

@author: Kristin
"""




import numpy as np #
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

R_M = 2440
mu_0 = 4 * np.pi * 10**(-7)


directory = 'C:\\Users\\Kristin\\Documents\PhD\FAC\Residuen_only_dipole\\'
diff_x_matrix = np.loadtxt(directory + 'diff_x_matrix_leveled_1.txt')
diff_y_matrix = np.loadtxt(directory + 'diff_y_matrix_leveled_1.txt')
diff_z_matrix = np.loadtxt(directory + 'diff_z_matrix_leveled_1.txt')

diff_r_matrix = np.loadtxt(directory + 'diff_r_matrix_leveled_1.txt')
diff_theta_matrix = np.loadtxt(directory + 'diff_theta_matrix_leveled_1.txt')
diff_phi_matrix = np.loadtxt(directory + 'diff_phi_matrix_leveled_1.txt')

theta = np.loadtxt(directory + 'theta_steps_leveled.txt')
phi = np.loadtxt(directory + 'phi_steps_leveled.txt')

r_mean = np.loadtxt(directory + 'r_mean_leveled_1.txt')
theta_mean = np.loadtxt(directory + 'theta_mean_leveled_1.txt')
phi_mean = np.loadtxt(directory + 'phi_mean_leveled_1.txt')
print(r_mean)
print(theta_mean)
print(phi_mean)
#theta_steps = 10*2*np.pi/360
#phi_steps = 10*2*np.pi/360

theta_steps = 5*2*np.pi/360
phi_steps = 5*2*np.pi/360

i = 0
j = 0

rot_B_matrix_3 = np.zeros([len(phi)-1, len(theta)-1])
#j_4 = np.zeros([len(phi)-1, len(theta)-4]) #gleiche matrix wie j_3, ohne die äußersten 3 ringe (theta nur von 0 bis 60°)


for i in range(len(theta)-2):
    print('i= ', i)

    ending = 'theta'+ str(int(theta[i]))+'_'+str(int(theta[i+1]))+'phi'+str(int(phi[j]))+'_'+str(int(phi[j+1]))+'.txt'
    if i == 0: 
        continue
    #if i == 8: 
        #continue
    for j in range(len(phi)-1): 
        print('j= ', j)
        if j == 0: 
            rot_B_matrix_3[j,i] = 1/(R_M*1000) *(((diff_phi_matrix[j, i+1]-diff_phi_matrix[j, i-1])/(2*theta_steps))-(diff_phi_matrix[j,i]*(np.cos(theta_mean[j, i])/np.sin(theta_mean[j, i])))-((diff_theta_matrix[j+1,i] -diff_theta_matrix[71,i])/(2*phi_steps))/np.sin(theta_mean[j,i]))
        elif j == 72: 
            rot_B_matrix_3[j,i] = 1/(R_M*1000) *(((diff_phi_matrix[j, i+1]-diff_phi_matrix[j, i-1])/(2*theta_steps))-(diff_phi_matrix[j,i]*(np.cos(theta_mean[j, i])/np.sin(theta_mean[j, i])))-((diff_theta_matrix[0,i] -diff_theta_matrix[j-1,i])/(2*phi_steps))/np.sin(theta_mean[j,i]))
        
        else:         
            print('ending: ', ending)
            rot_B_matrix_3[j,i] = 1/(R_M*1000) *(((diff_phi_matrix[j, i+1]-diff_phi_matrix[j, i-1])/(2*theta_steps))-(diff_phi_matrix[j,i]*(np.cos(theta_mean[j, i])/np.sin(theta_mean[j, i])))-((diff_theta_matrix[j+1,i] -diff_theta_matrix[j-1,i])/(2*phi_steps))/np.sin(theta_mean[j,i]))
        
        
        

j_3 = 1/mu_0 * rot_B_matrix_3
j_4 = j_3[0:73, 0:12]


    
rad = np.linspace(0, 90, 19)
azm = np.linspace(0, 2 * np.pi, 73)


    
lim = 400
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
plt.title('j (radial), scaled with R$_M$, with B$_x$ correction')
plt.show()


rad = np.linspace(0, 60, 13)
print('rad: ', rad)
azm = np.linspace(0, 2 * np.pi, 73)

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
plt.title(r'j (radial), scaled with R$_M$, with pedersen-correction, $\alpha$ = 1')
plt.savefig(directory + 'j_radial_zoomed_pedersen_correction_1.png')
plt.show()



