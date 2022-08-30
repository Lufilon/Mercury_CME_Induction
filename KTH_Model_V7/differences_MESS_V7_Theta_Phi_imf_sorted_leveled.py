# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 13:11:43 2022

@author: Kristin
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 15:58:59 2022

@author: Kristin
"""
import numpy as np 
import matplotlib.pyplot as plt
from kth14_model_for_mercury_v7 import kth14_model_for_mercury_v7
from mpl_toolkits.mplot3d import Axes3D
import json


directory_imf = 'C:\\Users\\Kristin\\Documents\\PhD\\IMF_conditions\\'
orbit_numbers_northward = np.loadtxt(directory_imf + 'orbit_numbers_imf_northward')
orbit_numbers_southward = np.loadtxt(directory_imf + 'orbit_numbers_imf_southward')
orbit_numbers_sunward = np.loadtxt(directory_imf + 'orbit_numbers_imf_sundward')
orbit_numbers_antisunward = np.loadtxt(directory_imf + 'orbit_numbers_imf_antosunward')
orbit_numbers_duskward = np.loadtxt(directory_imf + 'orbit_numbers_imf_duskward')
orbit_numbers_dawnward = np.loadtxt(directory_imf + 'orbit_numbers_imf_dawnward')





directory2 = 'C:\\Users\\Kristin\Documents\PhD\FAC\Positions_Theta_Phi_5deg\\'

directory = 'C:\\Users\\Kristin\\Documents\PhD\FAC\Residuen_leveled_imf_sorted\\'

#theta_index = ['00_10', '10_20', '20_30', '30_40', '40_50', '50_60', '60_70', '70_80', '80_90']



theta =  np.arange(0.0, 95.0, 5)
print('theta = ', theta)
phi = np.arange(-180.0, 185.0, 5)
print('phi = ', phi)

R_M = 2440
alpha = 1

direction = 'southward\\'
np.savetxt(directory + direction + 'theta_steps.txt', theta)
np.savetxt(directory + direction + 'phi_steps.txt', phi)

# find all positions on the dayside
#for i in range(len(theta)-1): 
    #for j in range(len(phi)-1):        
        
        
        
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


number_of_datapoints = np.zeros([len(phi)-1, len(theta)-1])

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
        
        
        
        
        
        indices_imf_condition = np.where(np.isin(orbit_number, orbit_numbers_southward))
        
        if len(indices_imf_condition[0]) == 0: 
            print('Empty at ' + ending)
            continue
        
        
        orbit_pos_x_imf_sorted = orbit_pos_x[indices_imf_condition]
        orbit_pos_y_imf_sorted = orbit_pos_y[indices_imf_condition]
        orbit_pos_z_imf_sorted = orbit_pos_z[indices_imf_condition]
        r_hel_imf_sorted = r_hel[indices_imf_condition]
        DI_imf_sorted = DI[indices_imf_condition]
        
        

        B_xyz_KTH = kth14_model_for_mercury_v7(orbit_pos_x_imf_sorted, orbit_pos_y_imf_sorted, orbit_pos_z_imf_sorted, r_hel_imf_sorted, DI_imf_sorted, True, False, False, True, True)
        #np.savetxt('C:\\Users\Kristin\Documents\Backup_Masterarbeit_20200608\Test_VGL_Dipole_v4_v5\Bz_v4_dip_int\\', B_xyz_KTH[2]))
        usable_indices = np.loadtxt('C:\\Users\\Kristin\\Documents\PhD\\KTH_Model_V7\\usable_indices.txt').astype(int)
        #print('usable_indices: ', usable_indices)
        
        number_of_datapoints[j,i] = len(np.atleast_1d(usable_indices))
        
     

            
        bx_KTH = B_xyz_KTH[0]
        by_KTH = B_xyz_KTH[1]
        bz_KTH = B_xyz_KTH[2]      

        
        r_coord = np.sqrt(orbit_pos_x[usable_indices]**2 + orbit_pos_y[usable_indices]**2 + orbit_pos_z[usable_indices]**2)
        r_surface = (r_coord)
        r_mean[j,i] = np.mean(r_coord)
        r_mean[j,i] = np.mean(r_coord)
        #print('r_mean: ', r_mean[j,i])
        theta_coord = np.arccos(orbit_pos_z[usable_indices]/r_coord)
        theta_mean[j,i] = np.mean(theta_coord)
        #print('theta_mean: ', theta_mean[j,i]*360/2/np.pi)
        phi_coord = np.arctan2(orbit_pos_y[usable_indices], orbit_pos_x[usable_indices])
        phi_mean[j,i] = np.mean(phi_coord)
        #print('phi_mean: ', phi_mean[j,i]*360/2/np.pi)
        
        np.savetxt(directory + direction + 'r_mean.txt', r_mean)
        np.savetxt(directory + direction + 'theta_mean.txt', theta_mean)
        np.savetxt(directory + direction + 'phi_mean.txt', phi_mean)
        
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
            
        diff_x_matrix[j, i] = np.mean(diff_x)
        diff_y_matrix[j, i] = np.mean(diff_y)
        diff_z_matrix[j, i] = np.mean(diff_z)
        
        diff_r_matrix[j, i] = np.mean(diff_r)
        diff_theta_matrix[j, i] = np.mean(diff_theta)
        diff_phi_matrix[j, i] = np.mean(diff_phi)
        
        #print('difference r matrix: ', diff_r_matrix[j,i])
        #print('difference theta matrix: ', diff_theta_matrix[j,i])
        #print('difference phi matrix: ', diff_phi_matrix[j,i])
        #np.savetxt('C:\\Users\Kristin\Documents\Backup_Masterarbeit_20200608\Test_VGL_Dipole_v4_v5\Bz_v4_dip_int', diff_z_matrix)

for i in range(len(theta)-1):
    for j in range(len(phi)-1): 
        if (number_of_datapoints[j,i] < 2): 
            number_of_datapoints[j,i] = np.nan
            
            
diff_x_matrix_mean = np.mean(np.abs(diff_x_matrix))
diff_y_matrix_mean = np.mean(np.abs(diff_y_matrix))
diff_z_matrix_mean = np.mean(np.abs(diff_z_matrix))

diff_r_matrix_mean = np.mean(np.abs(diff_r_matrix))
diff_theta_matrix_mean = np.mean(np.abs(diff_theta_matrix))
diff_phi_matrix_mean = np.mean(np.abs(diff_phi_matrix))



np.savetxt(directory + direction  + 'diff_x_matrix.txt', diff_x_matrix)
np.savetxt(directory + direction  + 'diff_y_matrix.txt', diff_y_matrix)
np.savetxt(directory + direction  + 'diff_z_matrix.txt', diff_z_matrix)

np.savetxt(directory + direction  + 'diff_r_matrix.txt', diff_r_matrix)
np.savetxt(directory + direction  + 'diff_theta_matrix.txt', diff_theta_matrix)
np.savetxt(directory + direction  + 'diff_phi_matrix.txt', diff_phi_matrix)
np.savetxt(directory + direction  + 'number_of_datapoints.txt', number_of_datapoints)


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
plt.suptitle('B$_x$(Messenger) - B$_x$(KTH Version 7, only dipole)')
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
plt.savefig(directory + direction + 'b_r.png')
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
plt.savefig(directory + direction + 'b_theta.png')
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
plt.savefig(directory + direction + 'b_phi.png')
plt.show()



fig = plt.figure(figsize = (7,7))
ax = Axes3D(fig)
r, th = np.meshgrid(rad, azm)
#z = (r ** 2.0) / 4.0
z = number_of_datapoints
#z = np.random.uniform(-1, 1, (n,m))
plt.subplot(projection="polar")
plt.pcolormesh(th, r, z, cmap = 'Greys')
#plt.pcolormesh(th, z, r)
plt.plot(azm, r,   ls='none') 
plt.grid()
#plt.clim(0, 30)
cbar = plt.colorbar()
cbar.set_label('   nT', labelpad=-40, y=1.05, rotation=0)
plt.title('Number of Datapoints per Box, IMF Direction: southward')
#plt.title('        mean difference = ' + str(np.round(diff_phi_matrix_mean, 3)))
plt.savefig(directory + direction + 'plot_number_of_datapoints.png')
plt.show()

    



