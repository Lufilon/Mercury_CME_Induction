# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 09:56:45 2020

@author: Kristin
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 13:22:56 2020

@author: Kristin
"""

import numpy as np 
import matplotlib.pyplot as plt
from kth14_model_for_mercury_v7 import kth14_model_for_mercury_v7
from mpl_toolkits.mplot3d import Axes3D
import json

directory2 = 'C:\\Users\\Kristin\Documents\PhD\FAC\Positions_Theta_Phi_5deg\\'

directory = 'C:\\Users\\Kristin\\Documents\PhD\FAC\Residuen_only_dipole\\'

#theta_index = ['00_10', '10_20', '20_30', '30_40', '40_50', '50_60', '60_70', '70_80', '80_90']



theta =  np.arange(0.0, 100.0, 5)
print('theta = ', theta)
phi = np.arange(-180.0, 190.0, 5)
print('phi = ', phi)

np.savetxt(directory + 'theta_steps_leveled.txt', theta)
np.savetxt(directory + 'phi_steps_leveled.txt', phi)

# find all positions on the dayside
#for i in range(len(theta)-1): 
    #for j in range(len(phi)-1):        
       
R_M = 2440
alpha = 1
        
        
diff_x_matrix = np.zeros([len(phi)-1, len(theta)-1])
diff_y_matrix = np.zeros([len(phi)-1, len(theta)-1])
diff_z_matrix = np.zeros([len(phi)-1, len(theta)-1])

diff_r_matrix = np.zeros([len(phi)-1, len(theta)-1])
diff_theta_matrix = np.zeros([len(phi)-1, len(theta)-1])
diff_phi_matrix = np.zeros([len(phi)-1, len(theta)-1])

r_mean = np.zeros([len(phi)-1, len(theta)-1])
theta_mean = np.zeros([len(phi)-1, len(theta)-1])
phi_mean = np.zeros([len(phi)-1, len(theta)-1])

empty_boxes = []
r_coord_list = []
theta_coord_list= []
r_mean_coord_list = []
theta_mean_coord_list= []

for i in range(len(theta)-1):
    print('i= ', i, '/8')
    for j in range(len(phi)-1):  
        print('j: ', j)
        ending = 'theta'+ str(int(theta[i]))+'_'+str(int(theta[i+1]))+'phi'+str(int(phi[j]))+'_'+str(int(phi[j+1]))+'.txt'
        
        print('ending: ', ending)
        
        #orbit_pos_x = np.loadtxt(directory2 + 'orbit_pos_x_theta_' + theta_index[i] + '_phi_' + phi_index[j])
        #orbit_pos_y = np.loadtxt(directory2 + 'orbit_pos_y_theta_' + theta_index[i] + '_phi_' + phi_index[j])
        #orbit_pos_z = np.loadtxt(directory2 + 'orbit_pos_z_theta_' + theta_index[i] + '_phi_' + phi_index[j])
        #orbit_b_x = np.loadtxt(directory2 + 'orbit_b_x_theta_' + theta_index[i] + '_phi_' + phi_index[j])
        #orbit_b_y = np.loadtxt(directory2 + 'orbit_b_y_theta_' + theta_index[i] + '_phi_' + phi_index[j])
        #orbit_b_z = np.loadtxt(directory2 + 'orbit_b_z_theta_' + theta_index[i] + '_phi_' + phi_index[j])
        
        
        #load positions and magnetic field data in mso coordinates
        orbit_pos_x = np.loadtxt(directory2 + 'orbit_pos_x_mso_' + ending)
        if len(orbit_pos_x) == 0: 
            empty_boxes.append(ending)
            continue
        orbit_pos_y = np.loadtxt(directory2 + 'orbit_pos_y_mso_' + ending)
        orbit_pos_z = np.loadtxt(directory2 + 'orbit_pos_z_mso_' + ending)
        orbit_b_x = np.loadtxt(directory2 + 'orbit_b_x_'  + ending)
        orbit_b_y = np.loadtxt(directory2 + 'orbit_b_y_'  + ending)
        orbit_b_z = np.loadtxt(directory2 + 'orbit_b_z_'  + ending)
        orbit_number = np.loadtxt(directory2 + 'orbit_number_'  + ending)
        
        
        with open("C:\\Users\Kristin\Documents\PhD\KTH14_Python_in_progress\\dictionary_orbit_numbers_r_hel.txt") as file:
            dict_orbit_number_r_hel = json.load(file)
     
        with open("C:\\Users\Kristin\Documents\PhD\Fit_DI\\dictionary_orbit_number_DI.json") as file2:
            dict_orbit_number_DI = json.load(file2)
            
            
        r_hel = np.zeros(len(np.atleast_1d(orbit_number)))
        for a in range(len(orbit_number)): 
            #r_hel = dict_orbit_number_r_hel['12']
            r_hel[a] = dict_orbit_number_r_hel[str(int(orbit_number[a]))]
    
        DI = np.zeros(len(np.atleast_1d(orbit_number)))
        for b in range(len(np.atleast_1d(orbit_number))): 
            #r_hel = dict_orbit_number_r_hel['12']
            if (str(orbit_number[b])) in dict_orbit_number_DI.keys():
                DI[b] = dict_orbit_number_DI[str(orbit_number[b])]
            else: 
                DI[b] = 50
        #r_hel = np.ones(len(orbit_pos_x))*0.37
        #di = np.ones(len(orbit_pos_x))*50
        
        
        

        B_xyz_KTH = kth14_model_for_mercury_v7(orbit_pos_x, orbit_pos_y, orbit_pos_z, r_hel, DI, True, False, False, True, True)
        #np.savetxt('C:\\Users\Kristin\Documents\Backup_Masterarbeit_20200608\Test_VGL_Dipole_v4_v5\Bz_v4_dip_int\\', B_xyz_KTH[2]))
        usable_indices = np.loadtxt('C:\\Users\\Kristin\\Documents\PhD\\KTH_Model_V7\\usable_indices.txt').astype(int)
        #print('usable_indices: ', usable_indices)
        
        bx_KTH = B_xyz_KTH[0]
        by_KTH = B_xyz_KTH[1]
        bz_KTH = B_xyz_KTH[2]
            


        
        #rotate data to spherical coordinates
        
        r_coord = np.sqrt(orbit_pos_x[usable_indices]**2 + orbit_pos_y[usable_indices]**2 + orbit_pos_z[usable_indices]**2)
        r_surface = (r_coord)
        r_mean[j,i] = np.mean(r_coord)
        print('r_mean: ', r_mean[j,i])
        theta_coord = np.arccos(orbit_pos_z[usable_indices]/r_coord)
        theta_mean[j,i] = np.mean(theta_coord)
        print('theta_mean: ', theta_mean[j,i]*360/2/np.pi)
        phi_coord = np.arctan2(orbit_pos_y[usable_indices], orbit_pos_x[usable_indices])
        phi_mean[j,i] = np.mean(phi_coord)
        print('phi_mean: ', phi_mean[j,i]*360/2/np.pi)
        
        
        r_coord_list.extend(r_coord)
        theta_coord_list.extend(theta_coord)
        
        r_mean_coord_list.extend(r_mean)
        theta_mean_coord_list.extend(theta_mean)
        
        #calculate differences in mso (carthesian)
        diff_x = orbit_b_x[usable_indices]- bx_KTH
        diff_x_leveled = (orbit_b_x[usable_indices]- bx_KTH)*((r_surface/R_M)**alpha)
        correction_x = diff_x - diff_x_leveled
        diff_y = orbit_b_y[usable_indices] - by_KTH
        diff_y_leveled = (orbit_b_y[usable_indices]- by_KTH)*((r_surface/R_M)**alpha)
        correction_y = diff_y - diff_y_leveled
        diff_z = orbit_b_z[usable_indices] - bz_KTH
        diff_z_leveled = (orbit_b_z[usable_indices]- bz_KTH)*((r_surface/R_M)**alpha)
        correction_z = diff_z - diff_z_leveled
        
        
        np.savetxt(directory + 'r_mean_leveled_1.txt', r_mean)
        np.savetxt(directory + 'theta_mean_leveled_1.txt', theta_mean)
        np.savetxt(directory + 'phi_mean_leveled_1.txt', phi_mean)
        
        
        #Einheitsvektoren für Transformation
        
        
        #MESSENGER data (magnetic field) from carthesian to shperical
        b_r_mess = (orbit_b_x[usable_indices] - correction_x )* np.sin(theta_coord) * np.cos(phi_coord) + (orbit_b_y[usable_indices] - correction_y) * np.sin(theta_coord)*np.sin(phi_coord) + (orbit_b_z[usable_indices] - correction_z)  * np.cos(theta_coord)
        b_theta_mess = (orbit_b_x[usable_indices] - correction_x) * np.cos(theta_coord) * np.cos(phi_coord)  + (orbit_b_y[usable_indices] - correction_y)  * np.cos(theta_coord) * np.cos(phi_coord) - (orbit_b_z[usable_indices] - correction_z)  * np.sin(theta_coord)
        b_phi_mess = - (orbit_b_x[usable_indices] - correction_x) * np.sin(phi_coord)  + (orbit_b_y[usable_indices] - correction_y)  * np.cos(phi_coord) + orbit_b_z[usable_indices] * 0
        
        #KTH-results from carthesian to shperical
        b_r_KTH = bx_KTH * np.sin(theta_coord) * np.cos(phi_coord) + by_KTH * np.sin(theta_coord)*np.sin(phi_coord) + bz_KTH * np.cos(theta_coord)
        b_theta_KTH = bx_KTH * np.cos(theta_coord) * np.cos(phi_coord)  + by_KTH * np.cos(theta_coord) * np.cos(phi_coord) - bz_KTH * np.sin(theta_coord)
        b_phi_KTH = - bx_KTH * np.sin(phi_coord)  + by_KTH * np.cos(phi_coord) + bz_KTH * 0
        
        #calculate differences in mso (carthesian)
        diff_r = b_r_mess - b_r_KTH
        diff_theta = b_theta_mess - b_theta_KTH
        diff_phi = b_phi_mess - b_phi_KTH
        
        
        #old calculation: 
        #diff_r = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
        #diff_theta = np.arccos(diff_z/diff_r)
        #diff_phi = np.arctan2(diff_y, diff_x)
        
        #diff_x = np.mean(B_xyz_KTH[0])
        #diff_y = np.mean(B_xyz_KTH[1])
        #diff_z = np.mean(B_xyz_KTH[2])
            
        diff_x_matrix[j, i] = np.mean(diff_x_leveled)
        diff_y_matrix[j, i] = np.mean(diff_y)
        diff_z_matrix[j, i] = np.mean(diff_z)
        
        diff_r_matrix[j, i] = np.mean(diff_r)
        diff_theta_matrix[j, i] = np.mean(diff_theta)
        diff_phi_matrix[j, i] = np.mean(diff_phi)
        
        print('difference r matrix: ', diff_r_matrix[j,i])
        print('difference theta matrix: ', diff_theta_matrix[j,i])
        print('difference phi matrix: ', diff_phi_matrix[j,i])
        #np.savetxt('C:\\Users\Kristin\Documents\Backup_Masterarbeit_20200608\Test_VGL_Dipole_v4_v5\Bz_v4_dip_int', diff_z_matrix)


r_mean_rm = r_mean/R_M

diff_x_matrix_mean = np.mean(np.abs(diff_x_matrix))
diff_y_matrix_mean = np.mean(np.abs(diff_y_matrix))
diff_z_matrix_mean = np.mean(np.abs(diff_z_matrix))

diff_r_matrix_mean = np.mean(np.abs(diff_r_matrix))
diff_theta_matrix_mean = np.mean(np.abs(diff_theta_matrix))
diff_phi_matrix_mean = np.mean(np.abs(diff_phi_matrix))



np.savetxt(directory + 'diff_x_matrix_leveled_1.txt', diff_x_matrix)
np.savetxt(directory + 'diff_y_matrix_leveled_1.txt', diff_y_matrix)
np.savetxt(directory + 'diff_z_matrix_leveled_1.txt', diff_z_matrix)

np.savetxt(directory + 'diff_r_matrix_leveled_1.txt', diff_r_matrix)
np.savetxt(directory + 'diff_theta_matrix_leveled_1.txt', diff_theta_matrix)
np.savetxt(directory + 'diff_phi_matrix_leveled_1.txt', diff_phi_matrix)
np.savetxt(directory + 'r_mean_rm.txt', r_mean_rm)
np.savetxt(directory + 'r_coord_list.txt', r_coord_list)



#i = 8
#j=26

#print(phi_mean[j,i]*360/2/np.pi)
#print(theta_mean[j,i]*360/2/np.pi)
#diff_theta_matrix[j,i]= 50
#diff_phi_matrix[j,i]= 50

lim = 40.0
     
rad = np.linspace(0, 90, 19)
azm = np.linspace(0, 2 * np.pi, 73)
       
fig = plt.figure(figsize = (7,7))
ax = Axes3D(fig)
#rad = np.linspace(0, 90, 10)
#azm = np.linspace(0, 2 * np.pi, 37)
r, th = np.meshgrid(rad, azm)
#z = (r ** 2.0) / 4.0
z = diff_x_matrix
#z = np.random.uniform(-1, 1, (n,m))
plt.subplot(projection="polar")
plt.pcolormesh(th, r, z, cmap = 'seismic')
#plt.pcolormesh(th, z, r)
plt.clim(-lim, lim)
plt.plot(azm, r,   ls='none') 
plt.grid()
cbar = plt.colorbar()
cbar.set_label('   nT', labelpad=-40, y=1.05, rotation=0)
plt.suptitle('B$_x$(Messenger)(leveled) - B$_x$(KTH Version 7, only dipole)')
plt.title('        mean difference = ' + str(np.round(diff_x_matrix_mean, 3)))
plt.show()

fig = plt.figure(figsize = (7,7))
ax = Axes3D(fig)
r, th = np.meshgrid(rad, azm)
#z = (r ** 2.0) / 4.0
z = diff_y_matrix
#z = np.random.uniform(-1, 1, (n,m))
plt.subplot(projection="polar")
plt.pcolormesh(th, r, z, cmap = 'seismic')
#plt.pcolormesh(th, z, r)
plt.plot(azm, r,   ls='none') 
plt.grid()
plt.clim(-lim, lim)
cbar = plt.colorbar()
cbar.set_label('   nT', labelpad=-40, y=1.05, rotation=0)
plt.suptitle('B$_y$(Messenger) - B$_y$(KTH Version 7, only dipole)')
plt.title('        mean difference = ' + str(np.round(diff_y_matrix_mean, 3)))
plt.show()

fig = plt.figure(figsize = (7,7))
ax = Axes3D(fig)
r, th = np.meshgrid(rad, azm)
#z = (r ** 2.0) / 4.0
z = diff_z_matrix
#z = np.random.uniform(-1, 1, (n,m))
plt.subplot(projection="polar")
plt.pcolormesh(th, r, z, cmap = 'seismic')
#plt.pcolormesh(th, z, r)
plt.plot(azm, r,   ls='none') 
plt.grid()
plt.clim(-lim, lim)
cbar = plt.colorbar()
cbar.set_label('   nT', labelpad=-40, y=1.05, rotation=0)
plt.suptitle('B$_z$(Messenger) - B$_z$(KTH Version 7, only dipole)')
plt.title('        mean difference = ' + str(np.round(diff_z_matrix_mean, 3)))
plt.show()


##########################################################
fig = plt.figure(figsize = (7,7))
ax = Axes3D(fig)
#rad = np.linspace(0, 90, 10)
#azm = np.linspace(0, 2 * np.pi, 37)
r, th = np.meshgrid(rad, azm)
#z = (r ** 2.0) / 4.0
z = diff_r_matrix
#z = np.random.uniform(-1, 1, (n,m))
plt.subplot(projection="polar")
plt.pcolormesh(th, r, z, cmap = 'seismic')
#plt.pcolormesh(th, z, r)
plt.clim(-lim, lim)
plt.plot(azm, r,   ls='none') 
plt.grid()
cbar = plt.colorbar()
cbar.set_label('   nT', labelpad=-40, y=1.05, rotation=0)
plt.suptitle('B$_r$(Messenger) - B$_r$(KTH Version 7, only dipole)')
plt.title('        mean difference = ' + str(np.round(diff_r_matrix_mean, 3)))
plt.show()

fig = plt.figure(figsize = (7,7))
ax = Axes3D(fig)
r, th = np.meshgrid(rad, azm)
#z = (r ** 2.0) / 4.0
z = diff_theta_matrix
#z = np.random.uniform(-1, 1, (n,m))
plt.subplot(projection="polar")
plt.pcolormesh(th, r, z, cmap = 'seismic')
#plt.pcolormesh(th, z, r)
plt.plot(azm, r,   ls='none') 
plt.grid()
plt.clim(-lim, lim)
cbar = plt.colorbar()
cbar.set_label('   nT', labelpad=-40, y=1.05, rotation=0)
plt.suptitle('B$_{theta}$(Messenger) - B$_{theta}$(KTH Version 7, only dipole)')
plt.title('        mean difference = ' + str(np.round(diff_theta_matrix_mean, 3)))
plt.show()

fig = plt.figure(figsize = (7,7))
ax = Axes3D(fig)
r, th = np.meshgrid(rad, azm)
#z = (r ** 2.0) / 4.0
z = diff_phi_matrix
#z = np.random.uniform(-1, 1, (n,m))
plt.subplot(projection="polar")
plt.pcolormesh(th, r, z, cmap = 'seismic')
#plt.pcolormesh(th, z, r)
plt.plot(azm, r,   ls='none') 
plt.grid()
plt.clim(-lim, lim)
cbar = plt.colorbar()
cbar.set_label('   nT', labelpad=-40, y=1.05, rotation=0)
plt.suptitle('B$_{phi}$(Messenger) - B$_{phi}$(KTH Version 7, only dipole)')
plt.title('        mean difference = ' + str(np.round(diff_phi_matrix_mean, 3)))
#plt.title('        mean difference = ' + str(np.round(diff_phi_matrix_mean, 3)))
plt.show()



rad = np.linspace(0, 90, 19)
azm = np.linspace(0, 2 * np.pi, 73)

fig = plt.figure(figsize = (7,7))
ax = Axes3D(fig)
r, th = np.meshgrid(rad, azm)
#z = (r ** 2.0) / 4.0
z = r_mean
#z = np.random.uniform(-1, 1, (n,m))
plt.subplot(projection="polar")
plt.pcolormesh(th, r, z, cmap = 'Greys')
#plt.pcolormesh(th, z, r)
plt.plot(azm, r,   ls='none') 
plt.grid()
plt.clim(2440, 4000)
cbar = plt.colorbar()
cbar.set_label('   km', labelpad=-40, y=1.05, rotation=0)
plt.title('Mean Radius')
#plt.title('        mean difference = ' + str(np.round(diff_phi_matrix_mean, 3)))
#plt.title('        mean difference = ' + str(np.round(diff_phi_matrix_mean, 3)))
plt.show()


plt.figure()
plt.plot(theta_coord_list, r_coord_list, '.')
plt.title('Radius depending on theta angle')
plt.xlabel('Theta in rad')
plt.ylabel('Radius in km in MSO')
plt.grid()
plt.show()

plt.figure()
plt.plot(theta_mean_coord_list, r_mean_coord_list, '.')
plt.title('Mean Radius depending on theta angle')
plt.xlabel('Theta in rad')
plt.ylabel('Radius in km in MSO')
plt.grid()
plt.show()

