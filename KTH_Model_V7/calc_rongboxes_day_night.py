# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 15:20:59 2022

@author: Kristin
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 09:40:35 2021

@author: Kristin
"""

#Rong Boxen magnetfeld differenzen und strom berechnen auf der Tagseite und nachtseite 


#kopiert aus calc_jy_rong_paper
import numpy as np
from kth14_model_for_mercury_v7 import kth14_model_for_mercury_v7
import matplotlib.pyplot as plt
import json
import warnings
import math

length_x = 42     # Anzahl der x intrvalle
length_z = 25     # Anzahl der y intervalle

g10_int = -195.0

t_a = 1.0

mu_0 = 4*np.pi*10**(-7)

directory = 'C:\\Users\\Kristin\\Documents\\PhD\\Neutralschichtstromanpassung_neu\\Positions_Rongboxes_Day+Night\\'

directory2 = 'C:\\Users\Kristin\Documents\PhD\Fit_hel_Distance\\'
directory3 = 'C:\\Users\\Kristin\\Documents\\PhD\\Neutralschichtstromanpassung_neu\\data_mes_kth_dipole\\'


#ending = '_all_small.txt'

with open("C:\\Users\Kristin\Documents\PhD\KTH14_Python_in_progress\\dictionary_orbit_numbers_r_hel.txt") as file:
     dict_orbit_number_r_hel = json.load(file)
     
with open("C:\\Users\Kristin\Documents\PhD\Fit_DI\\dictionary_orbit_number_DI.json") as file2:
     dict_orbit_number_DI = json.load(file2)





average_orbit_b_x = np.zeros((length_z, length_x))
average_orbit_b_y = np.zeros((length_z, length_x))
average_orbit_b_z = np.zeros((length_z, length_x))

average_kth_b_x = np.zeros((length_z, length_x))
average_kth_b_y = np.zeros((length_z, length_x))
average_kth_b_z = np.zeros((length_z, length_x))

delta_b_x_average = np.zeros((length_z, length_x))
delta_b_y_average = np.zeros((length_z, length_x))
delta_b_z_average = np.zeros((length_z, length_x))

average_orbit_b_x_dipole = np.zeros((length_z, length_x))
average_orbit_b_y_dipole = np.zeros((length_z, length_x))
average_orbit_b_z_dipole = np.zeros((length_z, length_x))
delta_b_ges_average = np.zeros((length_z, length_x))

average_kth_b_x_dipole = np.zeros((length_z, length_x))
average_kth_b_y_dipole = np.zeros((length_z, length_x))
average_kth_b_z_dipole = np.zeros((length_z, length_x))


b_x_neutralcurrent = np.zeros((length_z, length_x))
b_y_neutralcurrent = np.zeros((length_z, length_x))
b_z_neutralcurrent = np.zeros((length_z, length_x))
b_ges_neutralcurrent = np.zeros((length_z, length_x))


for i in range(length_x): 
    for j in range(length_z):
        end = str(i) + '_' + str(j)
       
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, append=1)

            print(end)
        
            orbit_pos_x = np.loadtxt(directory + 'orbit_pos_x_mso_' + end)
            
            print('Länge: ', len(np.atleast_1d(orbit_pos_x)))
        
            if len(np.atleast_1d(orbit_pos_x)) == 0: 
                average_orbit_b_x[j, i] = np.nan
                average_orbit_b_y[j, i] = np.nan 
                average_orbit_b_z[j, i] = np.nan        
                average_kth_b_x[j, i] = np.nan
                average_kth_b_y[j, i] = np.nan
                average_kth_b_z[j, i] = np.nan
        
                delta_b_x_average[j, i]= np.nan
                delta_b_z_average[j, i]= np.nan
                
                average_orbit_b_x_dipole[j, i] = np.nan
                average_orbit_b_z_dipole[j, i] = np.nan
                continue 
            
            
            orbit_pos_y = np.loadtxt(directory + 'orbit_pos_y_mso_' + end)
            orbit_pos_z = np.loadtxt(directory + 'orbit_pos_z_mso_' + end)
            orbit_b_x = np.loadtxt(directory + 'orbit_b_x_' + end)
            orbit_b_y = np.loadtxt(directory + 'orbit_b_y_' + end)
            orbit_b_z = np.loadtxt(directory + 'orbit_b_z_' + end)
            orbit_number = np.loadtxt(directory + 'orbit_number_' + end)
            #calculate and subtract dipole field  (Bx -> B'x)
    
            
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
        #print('Orbit without DI: ', orbit_number[i] )
            
            #indices_DI = np.where(DI < 66.6)
            #orbit_pos_x = orbit_pos_x[indices_DI] 
            #orbit_pos_y = orbit_pos_y[indices_DI]
            #orbit_pos_z = orbit_pos_z[indices_DI]
            #orbit_b_x = orbit_b_x[indices_DI] 
            #orbit_b_y = orbit_b_y[indices_DI]
            #orbit_b_z = orbit_b_z[indices_DI]
            #r_hel = r_hel[indices_DI]
            #DI = DI[indices_DI]
            
            #if len(np.atleast_1d(orbit_pos_x)) == 0: 
                #average_orbit_b_x[j, i] = np.nan
                #average_orbit_b_z[j, i] = np.nan        
                #average_kth_b_x[j, i] = np.nan
                #average_kth_b_z[j, i] = np.nan
        
                #delta_b_x_average[j, i]= np.nan
                #delta_b_z_average[j, i]= np.nan
            
                #continue 
            
            with open("C:\\Users\Kristin\\Documents\PhD\KTH_Model_V7\\control_params_v7.json", "r") as file:
                control_params = json.load(file)
            #control_params['t_a'] = 3.075
            #control_params['d_0'] = 0.35 #before: 0.35
            control_params['g10_int'] = g10_int
            control_params['t_a'] = t_a
            #control_params['a'] = 250.0
            #control_params['c'] = 0.2
            
            #control_params['t_b'] = t_b
            #print('control params: ', control_params)
            with open("control_params_v7_tmp.json", 'w') as f:
                json.dump(control_params, f)
            
    
            KTH_B = kth14_model_for_mercury_v7(orbit_pos_x, orbit_pos_y, orbit_pos_z, r_hel, DI, True, True, False,  True, True)
            kth_dipole = kth14_model_for_mercury_v7(orbit_pos_x, orbit_pos_y, orbit_pos_z, r_hel, DI, True, False, False, True, False)            
            
            kth_neutralcurrent = kth14_model_for_mercury_v7(orbit_pos_x, orbit_pos_y, orbit_pos_z, r_hel, DI, False, True, False, True, False)
            
            

            if len(orbit_b_x) != len(KTH_B[0]): 
                print('Arrays do not have the same lenght!')
            #exit()            
            
            if len(orbit_b_x) != len(KTH_B[0]): 
                average_orbit_b_x[j, i] = np.nan
                average_orbit_b_y[j, i] = np.nan 
                average_orbit_b_z[j, i] = np.nan        
                average_kth_b_x[j, i] = np.nan
                average_kth_b_y[j, i] = np.nan
                average_kth_b_z[j, i] = np.nan
        
                delta_b_x_average[j, i]= np.nan
                delta_b_z_average[j, i]= np.nan
                
                average_orbit_b_x_dipole[j, i] = np.nan
                average_orbit_b_z_dipole[j, i] = np.nan
                
                b_x_neutralcurrent[j, i] = np.nan
                b_y_neutralcurrent[j, i] = np.nan
                b_z_neutralcurrent[j, i] = np.nan
                
                continue
                
            
            b_x = orbit_b_x - KTH_B[0]
            b_y = orbit_b_y - KTH_B[1]
            b_z = orbit_b_z - KTH_B[2]
        
        #data_array = np.array([orbit_pos_x, b_x, orbit_pos_z, b_z])
        #sortedData = data_array[:,data_array[0].argsort()]
        #orbit_pos_x = sortedData[0]
        #b_x = sortedData[1]
        #orbit_pos_z = sortedData[2]
        #b_z = sortedData[3]
        
        #b_x = dipole_field[0]
        #b_z = dipole_field[2]
        
        #b_x = orbit_b_x
        #b_z = orbit_b_z
        
            average_orbit_b_x[j, i] = np.mean(orbit_b_x)
            average_orbit_b_y[j, i] = np.mean(orbit_b_y)
            average_orbit_b_z[j, i] = np.mean(orbit_b_z)
            
            average_orbit_b_x_dipole[j, i] = np.mean((orbit_b_x - kth_dipole[0]))
            average_orbit_b_y_dipole[j, i] = np.mean(orbit_b_y - kth_dipole[1])
            average_orbit_b_z_dipole[j, i] = np.mean(orbit_b_z - kth_dipole[2])

        
            average_kth_b_x[j, i] = np.mean(KTH_B[0])
            average_kth_b_y[j, i] = np.mean(KTH_B[1])
            average_kth_b_z[j, i] = np.mean(KTH_B[2])
            
            average_kth_b_x_dipole[j, i] = np.mean(kth_dipole[0])
            average_kth_b_y_dipole[j, i] = np.mean(kth_dipole[1])
            average_kth_b_z_dipole[j, i] = np.mean(kth_dipole[2])
        
            #delta_b_x_average[j, i]= np.mean(b_x)
            #delta_b_y_average[j, i]= np.mean(b_y)
            #delta_b_z_average[j, i]= np.mean(b_z)
            
            delta_b_x_average[j, i]= np.mean(orbit_b_x - KTH_B[0])
            delta_b_y_average[j, i]= np.mean(orbit_b_y - KTH_B[1])
            delta_b_z_average[j, i]= np.mean(orbit_b_z - KTH_B[2])
            
            delta_b_ges_average[j,i] = np.sqrt(delta_b_x_average[j, i]**2 + delta_b_y_average[j, i]**2 + delta_b_z_average[j, i]**2)
            
            b_x_neutralcurrent[j, i]= np.mean(kth_neutralcurrent[0])
            b_y_neutralcurrent[j, i]= np.mean(kth_neutralcurrent[1])
            b_z_neutralcurrent[j, i]= np.mean(kth_neutralcurrent[2])
            
            b_ges_neutralcurrent[j,i] = np.sqrt(b_x_neutralcurrent[j, i]**2 + b_y_neutralcurrent[j, i]**2 + b_z_neutralcurrent[j, i]**2)
'''           
np.savetxt(directory3 + 'average_b_x_MES.txt', average_orbit_b_x)          
np.savetxt(directory3 + 'average_b_y_MES.txt', average_orbit_b_y)  
np.savetxt(directory3 + 'average_b_z_MES.txt', average_orbit_b_z)    

np.savetxt(directory3 + 'delta_b_x_MES-KTH_dipole195.txt', average_orbit_b_x_dipole) 
np.savetxt(directory3 + 'delta_b_y_MES-KTH_dipole195.txt', average_orbit_b_y_dipole) 
np.savetxt(directory3 + 'delta_b_z_MES-KTH_dipole195.txt', average_orbit_b_z_dipole) 

np.savetxt(directory3 + 'average_b_x_KTH195.txt', average_kth_b_x)          
np.savetxt(directory3 + 'average_b_y_KTH195.txt', average_kth_b_y)  
np.savetxt(directory3 + 'average_b_z_KTH195.txt', average_kth_b_z)  

np.savetxt(directory3 + 'average_b_x_KTH195_dipole.txt', average_kth_b_x_dipole)          
np.savetxt(directory3 + 'average_b_y_KTH195_dipole.txt', average_kth_b_y_dipole)  
np.savetxt(directory3 + 'average_b_z_KTH195_dipole.txt', average_kth_b_z_dipole) 
'''           
            
####################################################################    
            #calc j_y
            

mu_0 = 4*np.pi*10**(-7)
R_M = 2440
#directory = 'C:\\Users\\kriss\\Documents\\Masterarbeit\\Backup_Masterarbeit_20200504\\Positions_Sheet_Rong_Paper\\'


j_y_2_average = np.zeros((length_z, length_x))
j_y_2 = np.zeros((length_z, length_x))


for i in range(1, length_x-1): 
    for j in range(1,length_z-1):
        end = str(i) + '_' + str(j)
        #if i > 0 & j > 0 & i < length_x & j <length_z:
        #print(i)
        #print(j)
        j_y_2[j, i] = ((delta_b_x_average[j+1, i] - delta_b_x_average[j-1, i])/(0.2 *R_M )) - ((delta_b_z_average[j, i+1] - delta_b_z_average[j, i-1])/(0.2*R_M))
        
        
        j_y_2_average[j,i] = j_y_2[j,i] *(10**-3)/mu_0
        
#np.savetxt('C:\\Users\\Kristin\\Documents\\PhD\\Neutralschichtstromanpassung_neu\\jy_Day_Nightside\\j_y_190nT')
np.savetxt('C:\\Users\\Kristin\\Documents\\PhD\\Neutralschichtstromanpassung_neu\\jy_Day_Nightside\\j_y_195nT.txt', j_y_2_average)


j_y_1_average = np.zeros((length_z, length_x))
j_y_1 = np.zeros((length_z, length_x))


for k in range(1, length_x-1): 
    for l in range(1,length_z-1):
        end = str(k) + '_' + str(l)
        #if i > 0 & j > 0 & i < length_x & j <length_z:
        #print(k)
        #print(l)
        j_y_1[l, k] = ((average_orbit_b_x_dipole[l+1, k] - average_orbit_b_x_dipole[l-1, k])/(0.2  *R_M)) - ((average_orbit_b_z_dipole[l, k+1] - average_orbit_b_z_dipole[l, k-1])/(0.2 *R_M))
        
        
        j_y_1_average[l,k] = j_y_1[l,k] *(10**-3)/mu_0
        
  
j_y_3_average = np.zeros((length_z, length_x))   #Neutralschichtstrom, der nach der Hankeltrafo übrig bleibt, berechnet nur aus KTH Neutralcurrentsheet Modul 
j_y_3 = np.zeros((length_z, length_x))

for m in range(1, length_x-1): 
    for n in range(1,length_z-1):
        end = str(m) + '_' + str(n)
        #if i > 0 & j > 0 & i < length_x & j <length_z:
        #print(m)
        #print(n)
        j_y_3[n, m] = ((b_x_neutralcurrent[n+1, m] - b_x_neutralcurrent[n-1, m])/(0.2  *R_M)) - ((b_z_neutralcurrent[n, m+1] - b_z_neutralcurrent[n, m-1])/(0.2 *R_M))
        
        
        j_y_3_average[n,m] = j_y_3[n,m] *(10**-3)/mu_0
        if math.isnan(j_y_2_average[n,m]): 
            print('set np nan') 
            print(n)
            j_y_3_average[n,m] = np.nan
            
np.savetxt(directory3 + 'j_y_1_average_MES-KTH.txt', j_y_2_average) 
np.savetxt(directory3 + 'j_y_2_average_KTH.txt', j_y_3_average) 

diff_j = j_y_1_average - j_y_2_average

x_condition_list = [1.3, 0.8, 0.3, -0.2, -0.7, -1.2,  -1.7,  -2.2,  -2.7]
z_condition_list = [ -1.0,  -0.5,   0.0,  0.5,  1.0, 1.5]

#plt.close()
#plt.close()
#plt.close()
#plt.close()
#plt.close()
#plt.close()

plt.style.use('classic')

lim_a = 30
lim_b = 50
lim_c = 60

add = 'g$^1_{0,int}$ = '+ str(g10_int) + ', t$_a$ = ' + str(t_a) + ', sheet thickness = 0.1 R$_M$'

###################################################################

fig, axs = plt.subplots(2, 3)
fig.suptitle('Model Parameters: ' + add, fontsize=20)

array = delta_b_x_average
m = np.ma.masked_where(np.isnan(array),array)
plot1 = axs[0,0].pcolor(m, cmap='RdBu')
#plt.colorbar(plot1) 
axs[0,0].grid()      
plot1.set_clim(-lim_b,lim_b)
axs[0,0].set_xlabel('x in R$_M$')
axs[0,0].set_ylabel('z in R$_M$')
axs[0,0].set_title('$\Delta$Bx (MES - KTH)')
column_labels = x_condition_list 
row_labels = z_condition_list
axs[0,0].set_xticklabels(column_labels)
axs[0,0].set_yticklabels(row_labels)

#view_colormap('RdBu')


###################################################################
#fig, ax = plt.subplots()
array = delta_b_z_average
m = np.ma.masked_where(np.isnan(array),array)
plot2 = axs[1,0].pcolor(m, cmap='RdBu')
#axs[1,0].colorbar(plot2) 
plt.grid()      
plot2.set_clim(-lim_b,lim_b)
axs[1,0].set_xlabel('x in R$_M$ in MSO')
axs[1,0].set_ylabel('z in R$_M$ in MSO')
axs[1,0].set_title('$\Delta$Bz (MES - KTH)')
column_labels = x_condition_list 
row_labels = z_condition_list
axs[1,0].set_xticklabels(column_labels)
axs[1,0].set_yticklabels(row_labels)

#view_colormap('RdBu')

###################################################################

#fig, ax = plt.subplots()
#array = delta_b_y_average
#m = np.ma.masked_where(np.isnan(array),array)
#plt.pcolor(m, cmap='RdBu')
#plt.colorbar() 
#plt.grid()      
#plt.clim(-lim_a,lim_a)
#plt.show()
#plt.xlabel('x in R$_M$ in MSO')
#plt.ylabel('z in R$_M$ in MSO')
#plt.title('$\Delta$By (MES - KTH)' + add)
#column_labels = x_condition_list 
#row_labels = z_condition_list
#ax.set_xticklabels(column_labels)
#ax.set_yticklabels(row_labels)
#view_colormap('RdBu')

###################################################################

#fig, ax = plt.subplots()
#array = average_kth_b_x
#m = np.ma.masked_where(np.isnan(array),array)
#plt.pcolor(m)
#plt.colorbar()       
#plt.show()
#plt.xlabel('x in R$_M$ in MSO')
#plt.ylabel('z in R$_M$ in MSO')
#plt.grid()
#plt.clim(-lim_b,lim_b)
#plt.title('Bx (KTH)' + add)
#column_labels = x_condition_list 
#row_labels = z_condition_list
#ax.set_xticklabels(column_labels)
#ax.set_yticklabels(row_labels)

####################################################################

#fig, ax = plt.subplots()
#array = average_kth_b_z
#m = np.ma.masked_where(np.isnan(array),array)
#plt.pcolor(m)
#plt.colorbar()       
#plt.show()
#plt.xlabel('x in R$_M$ in MSO')
#plt.ylabel('z in R$_M$ in MSO')
#plt.grid()
#plt.clim(-lim_b,lim_b)
#plt.title('Bz (KTH)' + add)
#column_labels = x_condition_list 
#row_labels = z_condition_list
#ax.set_xticklabels(column_labels)
#ax.set_yticklabels(row_labels)

####################################################################


array = average_orbit_b_x_dipole
m = np.ma.masked_where(np.isnan(array),array)
plot3 = axs[0,1].pcolor(m, cmap='RdBu')
#axs[0,1].colorbar(plot3)       
plt.show()
axs[0,1].set_xlabel('x in R$_M$ in MSO')
axs[0,1].set_ylabel('z in R$_M$ in MSO')
plt.grid()
plot3.set_clim(-lim_b,lim_b)
axs[0,1].set_title('$\Delta$Bx (MES - KTH DIPOLE)')
column_labels = x_condition_list 
row_labels = z_condition_list
axs[0,1].set_xticklabels(column_labels)
axs[0,1].set_yticklabels(row_labels)


####################################################################


array = average_orbit_b_z_dipole
m = np.ma.masked_where(np.isnan(array),array)
plot4 = axs[1,1].pcolor(m, cmap='RdBu')
axs[1,1].set_xlabel('x in R$_M$ in MSO')
axs[1,1].set_ylabel('z in R$_M$ in MSO')
axs[1,1].grid()
plot4.set_clim(-lim_b,lim_b)
axs[1,1].set_title('$\Delta$Bz (MES - KTH DIPOLE)' )
column_labels = x_condition_list 
row_labels = z_condition_list
axs[1,1].set_xticklabels(column_labels)
axs[1,1].set_yticklabels(row_labels)


####################################################################



array = j_y_1_average
m = np.ma.masked_where(np.isnan(array),array)
plot5= axs[0,2].pcolor(m, cmap='RdBu')
axs[0,2].set_xlabel('x in R_M')
axs[0,2].set_ylabel('z in R_M')
axs[0,2].grid()
plot5.set_clim(-lim_b,lim_b)
axs[0,2].set_title('j_y (calculated with $\Delta$B (MES - KTH DIPOLE))')
column_labels = x_condition_list 
row_labels = z_condition_list
axs[0,2].set_xticklabels(column_labels)
axs[0,2].set_yticklabels(row_labels)

###################################################################


array = j_y_2_average
#array[array<0.5]=np.nan
m = np.ma.masked_where(np.isnan(array),array)
plot6 = axs[1,2].pcolor(m, cmap='RdBu')
axs[1,2].set_xlabel('x in R$_M$')
axs[1,2].set_ylabel('z in R$_M$')
axs[1,2].grid()
plot6.set_clim(-lim_b,lim_b)
axs[1,2].set_title('remaining j$_y$ (calculated with x and z component from total $\Delta$B)' )
column_labels = x_condition_list 
row_labels = z_condition_list
axs[1,2].set_xticklabels(column_labels)
axs[1,2].set_yticklabels(row_labels)

'''

'''
fig, ax = plt.subplots()
array = diff_j
#array[array<0.5]=np.nan
m = np.ma.masked_where(np.isnan(array),array)
plt.pcolor(m, cmap='RdBu')
plt.colorbar()       
plt.show()
plt.xlabel('x in R$_M$')
plt.ylabel('z in R$_M$')
plt.grid()
#plt.clim(-80,80)
plt.title('Difference in j between Model with/without neutralcurrent')
column_labels = x_condition_list 
row_labels = z_condition_list
ax.set_xticklabels(column_labels)
ax.set_yticklabels(row_labels)

#########################################################

fig, ax = plt.subplots()
array = j_y_3_average
#array[array<0.5]=np.nan
m = np.ma.masked_where(np.isnan(array),array)
plt.pcolor(m, cmap='RdBu')
plt.colorbar()       
plt.show()
plt.xlabel('x in R$_M$')
plt.ylabel('z in R$_M$')
plt.grid()
#plt.clim(-80,80)
plt.title('j$_y$ Neutralsheetcurrent calculated only from B(KTH)')
column_labels = x_condition_list 
row_labels = z_condition_list
ax.set_xticklabels(column_labels)
ax.set_yticklabels(row_labels)


'''


fig.subplots_adjust(right=0.8)
cbar = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar.set_title('B in nT or j in nA/m$^2$')
fig.colorbar(plot1, cax=cbar)


plt.show()

'''
#np.savetxt(directory + 'jy2mitDipolfeld',j_y_2_average)

#jy2minusDipolfeld = np.loadtxt(directory + 'jy2mitDipolfeld')
#jy2mitDipolfeld = np.loadtxt(directory + 'jy2minusDipolfeld')


#difference = jy2mitDipolfeld - jy2minusDipolfeld

#fig, ax = plt.subplots()
#array = difference
#array[array<0.5]=np.nan
#m = np.ma.masked_where(np.isnan(array),array)
#plt.pcolor(m)
#plt.colorbar()       
#plt.show()
#plt.xlabel('x in R_M')
#plt.ylabel('z in R_M')
#plt.grid()
#plt.clim(-2,2)
#plt.title('differenz')
#column_labels = x_condition_list 
#row_labels = z_condition_list
#ax.set_xticklabels(column_labels)
#ax.set_yticklabels(row_labels)
'''

fig, axs = plt.subplots()
#fig.suptitle('Model Parameters: ' + add, fontsize=20)

array = j_y_1_average
#array[array<0.5]=np.nan
m = np.ma.masked_where(np.isnan(array),array)
plot7 = axs.pcolor(m, cmap='seismic')
axs.set_xlabel('x in R$_M$')
axs.set_ylabel('z in R$_M$')
cbar = plt.colorbar(plot7)
cbar.set_label('   nA/m$^2$', labelpad=-40, y=1.05, rotation=0)
axs.grid()
plot7.set_clim(-lim_b,lim_b)
axs.set_title('j$_y$ (calculated with $\Delta$B (MES - KTH DIPOLE))')
column_labels = x_condition_list 
row_labels = z_condition_list
axs.set_xticklabels(column_labels)
axs.set_yticklabels(row_labels)


'''

z_condition_list = [ -1.5, -1.0,  -0.5,   0.0,  0.5,  1.0, 1.5]

fig,ax = plt.subplots()
array = j_y_1_average
m = np.ma.masked_where(np.isnan(array),array)
plt.pcolor(m, cmap='seismic')
plt.xlabel('x in R$_M$')
plt.ylabel('z in R$_M$')
plt.grid()
cbar = plt.colorbar()
cbar.set_label('   nA/m$^2$', labelpad=-40, y=1.05, rotation=0)
plt.clim(-lim_b,lim_b)
plt.title('j_y (calculated with $\Delta$B (MES - KTH DIPOLE))')
ax.axis('equal')
column_labels = x_condition_list 
row_labels = z_condition_list
ax.set_xticklabels(column_labels)
ax.set_yticklabels(row_labels)


'''
plt.figure()
array = j_y_1_average
m = np.ma.masked_where(np.isnan(array),array)
plt.pcolor(m, cmap='RdBu')
plt.xlabel('x in R$_M$')
plt.ylabel('z in R$_M$')
plt.grid()
plt.clim(-lim_b,lim_b)
plt.title('j_y (calculated with $\Delta$B (MES - KTH DIPOLE))')
column_labels = x_condition_list 
row_labels = z_condition_list
ax.set_xticklabels(column_labels)
ax.set_yticklabels(row_labels)
'''
