# -*- coding: utf-8 -*-
"""
Created on Wed May 18 10:12:51 2022

@author: Kristin
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 10:52:26 2021

@author: Kristin
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 09:40:35 2021

@author: Kristin
"""

# in Rong Boxen Magnetfeld  + Differenzen und Strom berechnen auf der Tagseite und Nachtseite 
# nur aus KTH Daten 


#kopiert aus calc_jy_rong_paper
import numpy as np
from kth14_model_for_mercury_v7 import kth14_model_for_mercury_v7
import matplotlib.pyplot as plt
import json
import warnings
import math


R_M = 2440

steps = 0.1
pos_x_array = np.arange(1.001, -5.001, -steps)*R_M
pos_z_array = np.arange(-2.001, 2.001, steps)*R_M

length_x = len(pos_x_array)
length_z = len(pos_z_array)

print(pos_x_array)
print(pos_z_array)




average_kth_b_x = np.zeros((length_z, length_x))
average_kth_b_y = np.zeros((length_z, length_x))
average_kth_b_z = np.zeros((length_z, length_x))

average_kth_b_x_dipole = np.zeros((length_z, length_x))
average_kth_b_y_dipole = np.zeros((length_z, length_x))
average_kth_b_z_dipole = np.zeros((length_z, length_x))
delta_b_ges_average = np.zeros((length_z, length_x))

b_x_neutralcurrent = np.zeros((length_z, length_x))
b_y_neutralcurrent = np.zeros((length_z, length_x))
b_z_neutralcurrent = np.zeros((length_z, length_x))
b_ges_neutralcurrent = np.zeros((length_z, length_x))

b_x_ringcurrent = np.zeros((length_z, length_x))
b_y_ringcurrent = np.zeros((length_z, length_x))
b_z_ringcurrent = np.zeros((length_z, length_x))
b_ges_ringcurrent = np.zeros((length_z, length_x))


for i in range(length_x): 
    for j in range(length_z):
        end = str(i) + '_' + str(j)
       
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, append=1)
            #print(i)
            #print(j)
            print(end)
            #print(pos_x)
            #print(pos_x_array[i])
            #print(pos_z_array[j])
            
            if (np.sqrt(pos_x_array[i]**2 + pos_z_array[j]**2)) < 2440: 
                average_kth_b_x[j, i] = np.nan
                average_kth_b_z[j, i] = np.nan        
                average_kth_b_x_dipole[j, i] = np.nan
                average_kth_b_z_dipole[j, i] = np.nan
        
                b_x_neutralcurrent[j, i]= np.nan
                b_z_neutralcurrent[j, i]= np.nan
                b_x_ringcurrent[j, i]= np.nan
                b_z_ringcurrent[j, i]= np.nan
                print('nan at: ', i)
                print('nan at: ', j)
                continue
                
        
            pos_x = pos_x_array[i]
            pos_z = pos_z_array[j]
            
    
            
            r_hel = 0.4   
    
            DI = 50
            
            pos_y = 0.0
 
            
            with open("C:\\Users\Kristin\\Documents\PhD\KTH_Model_V7\\control_params_v7.json", "r") as file:
                control_params = json.load(file)
                
            control_params['t_a'] = 1.0
            control_params['t_b'] = 0.0    
            #control_params['a'] = 89.0
            control_params['a'] = 95.0
            control_params['c'] = 0.4
    
            control_params['d_0'] = 0.16
            control_params['delta_x'] = -0.05
            control_params['delta_y'] = 0.5
            control_params['scale_x_d'] = 1.8
            
            control_params['d'] = 1.0
            control_params['e'] = 1.1
            control_params['f'] = 0.1
            #print('control params: ', control_params)
            with open("control_params_v7_tmp.json", 'w') as f:
                json.dump(control_params, f)
            
    
            KTH_B = kth14_model_for_mercury_v7(np.array([pos_x, pos_x]), np.array([pos_y, pos_y]), np.array([pos_z, pos_z]), np.array([r_hel, r_hel]), np.array([DI, DI]), True, True, False, True, True)
            kth_dipole = kth14_model_for_mercury_v7(np.array([pos_x, pos_x]), np.array([pos_y, pos_y]), np.array([pos_z, pos_z]), np.array([r_hel, r_hel]), np.array([DI, DI]), True, False, False, True, True)            
            
            kth_neutralcurrent = kth14_model_for_mercury_v7(np.array([pos_x, pos_x]), np.array([pos_y, pos_y]), np.array([pos_z, pos_z]), np.array([r_hel, r_hel]), np.array([DI, DI]), False, True, False, True, True)           
            #print(kth_neutralcurrent)
            kth_ringcurrent = kth14_model_for_mercury_v7(np.array([pos_x, pos_x]), np.array([pos_y, pos_y]), np.array([pos_z, pos_z]), np.array([r_hel, r_hel]), np.array([DI, DI]), False, False, True, True, False)
        

            
            average_kth_b_x_dipole[j, i] = np.mean(kth_dipole[0])
            average_kth_b_y_dipole[j, i] = np.mean(kth_dipole[1])
            average_kth_b_z_dipole[j, i] = np.mean(kth_dipole[2])

        
            average_kth_b_x[j, i] = np.mean(KTH_B[0])
            average_kth_b_y[j, i] = np.mean(KTH_B[1])
            average_kth_b_z[j, i] = np.mean(KTH_B[2])
        
            #delta_b_x_average[j, i]= np.mean(b_x)
            #delta_b_y_average[j, i]= np.mean(b_y)
            #delta_b_z_average[j, i]= np.mean(b_z)
            
            #delta_b_x_average[j, i]= np.mean(orbit_b_x - KTH_B[0])
            #delta_b_y_average[j, i]= np.mean(orbit_b_y - KTH_B[1])
            #delta_b_z_average[j, i]= np.mean(orbit_b_z - KTH_B[2])
            
            #delta_b_ges_average[j,i] = np.sqrt(delta_b_x_average[j, i]**2 + delta_b_y_average[j, i]**2 + delta_b_z_average[j, i]**2)
            #print('pos x: ', pos_x)
            #print('kth neutralcurrent: ', kth_neutralcurrent)
            #print(kth_neutralcurrent[1])
            b_x_neutralcurrent[j, i]= np.mean(kth_neutralcurrent[0])
            b_y_neutralcurrent[j, i]= np.mean(kth_neutralcurrent[1])
            b_z_neutralcurrent[j, i]= np.mean(kth_neutralcurrent[2])
            
            b_x_ringcurrent[j, i]= np.mean(kth_ringcurrent[0])
            b_y_ringcurrent[j, i]= np.mean(kth_ringcurrent[1])
            b_z_ringcurrent[j, i]= np.mean(kth_ringcurrent[2])
            
            #b_ges_neutralcurrent[j,i] = np.sqrt(b_x_neutralcurrent[j, i]**2 + b_y_neutralcurrent[j, i]**2 + b_z_neutralcurrent[j, i]**2)
           
            
####################################################################    
            #calc j_y
            

mu_0 = 4*np.pi*10**(-7)
R_M = 2440
#directory = 'C:\\Users\\kriss\\Documents\\Masterarbeit\\Backup_Masterarbeit_20200504\\Positions_Sheet_Rong_Paper\\'

       
 
j_y_3_average = np.zeros((length_z, length_x))   #Neutralschichtstrom, der nach der Hankeltrafo übrig bleibt, berechnet nur aus KTH Neutralcurrentsheet Modul 
j_y_3 = np.zeros((length_z, length_x))

for m in range(1, length_x-1): 
    for n in range(1,length_z-1):
        end = str(m) + '_' + str(n)
        #if i > 0 & j > 0 & i < length_x & j <length_z:
        #print(m)
        #print(n)
        j_y_3[n, m] = ((b_x_neutralcurrent[n+1, m] - b_x_neutralcurrent[n-1, m])/(2*steps  *R_M)) - ((b_z_neutralcurrent[n, m-1] - b_z_neutralcurrent[n, m+1])/(2*steps *R_M))
        
        
        j_y_3_average[n,m] = j_y_3[n,m] *(10**-3)/mu_0
        if math.isnan(b_x_neutralcurrent[n,m]): 
            #print('set np nan') 
            #print(n)
            j_y_3_average[n,m] = np.nan
            

j_y_2_average = np.zeros((length_z, length_x))   #Strom aus gesamter berechnung 
j_y_2 = np.zeros((length_z, length_x))

for m in range(1, length_x-1): 
    for n in range(1,length_z-1):
        end = str(m) + '_' + str(n)
        #if i > 0 & j > 0 & i < length_x & j <length_z:
        #print(m)
        #print(n)
        j_y_2[n, m] = ((average_kth_b_x[n+1, m] - average_kth_b_x[n-1, m])/(2*steps  *R_M)) - ((average_kth_b_z[n, m-1] - average_kth_b_z[n, m+1])/(2*steps *R_M))
        
        
        j_y_2_average[n,m] = j_y_2[n,m] *(10**-3)/mu_0
        if math.isnan(b_x_neutralcurrent[n,m]): 
            #print('set np nan') 
            #print(n)
            j_y_2_average[n,m] = np.nan
            
j_y_4_average = np.zeros((length_z, length_x))   #Strom aus partial Ring current
j_y_4 = np.zeros((length_z, length_x))

for m in range(1, length_x-1): 
    for n in range(1,length_z-1):
        end = str(m) + '_' + str(n)
        #if i > 0 & j > 0 & i < length_x & j <length_z:
        #print(m)
        #print(n)
        j_y_4[n, m] = ((b_x_ringcurrent[n+1, m] - b_x_ringcurrent[n-1, m])/(2*steps  *R_M)) - ((b_z_ringcurrent[n, m-1] - b_z_ringcurrent[n, m+1])/(2*steps *R_M))
        
        
        j_y_4_average[n,m] = j_y_4[n,m] *(10**-3)/mu_0
        if math.isnan(b_x_neutralcurrent[n,m]): 
            #print('set np nan') 
            #print(n)
            j_y_4_average[n,m] = np.nan
            


#diff_j = j_y_1_average - j_y_2_average

x_condition_list = [1.0, 0.0, -1.0,  -2.0,  -3.0, -4.0]
z_condition_list = [ -2.0, -1.5,  -1.0, -0.5,   0.0, 0.5,   1.0, 1.5,  2.0]

#plt.close()
#plt.close()
#plt.close()
#plt.close()
#plt.close()
#plt.close()

plt.style.use('classic')

lim_b = 150
lim_a = 150

lim_j = 100
#add = ', g10_int = -208.0 nT'


'''
###################################################################
fig, ax = plt.subplots()
array = average_kth_b_x
m = np.ma.masked_where(np.isnan(array),array)
plt.pcolor(m, cmap='RdBu')
plt.colorbar() 
plt.grid()      
plt.clim(-lim_b,lim_b)
plt.show()
plt.xlabel('x in R$_M$')
plt.ylabel('z in R$_M$')
plt.title('Bx (KTH Ges)' + add)
column_labels = x_condition_list 
row_labels = z_condition_list
ax.set_xticklabels(column_labels)
ax.set_yticklabels(row_labels)

#view_colormap('RdBu')


###################################################################
fig, ax = plt.subplots()
array = average_kth_b_z
m = np.ma.masked_where(np.isnan(array),array)
plt.pcolor(m, cmap='RdBu')
plt.colorbar() 
plt.grid()      
plt.clim(-lim_b,lim_b)
plt.show()
plt.xlabel('x in R$_M$ in MSO')
plt.ylabel('z in R$_M$ in MSO')
plt.title('Bz (KTH Ges)' + add)
column_labels = x_condition_list 
row_labels = z_condition_list
ax.set_xticklabels(column_labels)
ax.set_yticklabels(row_labels)

#view_colormap('RdBu')

###################################################################
fig, ax = plt.subplots()
array = b_x_neutralcurrent
m = np.ma.masked_where(np.isnan(array),array)
plt.pcolor(m, cmap='RdBu')
plt.colorbar() 
plt.grid()      
plt.clim(-lim_a,lim_a)
plt.show()
plt.xlabel('x in R$_M$')
plt.ylabel('z in R$_M$')
plt.title('Bx (KTH Neutralcurrent, int + ext)' + add)
column_labels = x_condition_list 
row_labels = z_condition_list
ax.set_xticklabels(column_labels)
ax.set_yticklabels(row_labels)

#view_colormap('RdBu')


###################################################################
fig, ax = plt.subplots()
array = b_z_neutralcurrent
m = np.ma.masked_where(np.isnan(array),array)
plt.pcolor(m, cmap='RdBu')
plt.colorbar() 
plt.grid()      
plt.clim(-lim_a,lim_a)
plt.show()
plt.xlabel('x in R$_M$ in MSO')
plt.ylabel('z in R$_M$ in MSO')
plt.title('Bz (KTH Neutralcurrent, int + ext)' + add)
column_labels = x_condition_list 
row_labels = z_condition_list
ax.set_xticklabels(column_labels)
ax.set_yticklabels(row_labels)

#view_colormap('RdBu')

###################################################################

fig, ax = plt.subplots()
array = average_kth_b_x_dipole
m = np.ma.masked_where(np.isnan(array),array)
plt.pcolor(m, cmap='RdBu')
plt.colorbar() 
plt.grid()      
plt.clim(-lim_a,lim_a)
plt.show()
plt.xlabel('x in R$_M$ in MSO')
plt.ylabel('z in R$_M$ in MSO')
plt.title('Bx (KTH) only Dipole int + ext' + add)
column_labels = x_condition_list 
row_labels = z_condition_list
ax.set_xticklabels(column_labels)
ax.set_yticklabels(row_labels)
#view_colormap('RdBu')

fig, ax = plt.subplots()
array = average_kth_b_z_dipole
m = np.ma.masked_where(np.isnan(array),array)
plt.pcolor(m, cmap='RdBu')
plt.colorbar() 
plt.grid()      
plt.clim(-lim_a,lim_a)
plt.show()
plt.xlabel('x in R$_M$ in MSO')
plt.ylabel('z in R$_M$ in MSO')
plt.title('Bz (KTH) only Dipole int + ext' + add)
column_labels = x_condition_list 
row_labels = z_condition_list
ax.set_xticklabels(column_labels)
ax.set_yticklabels(row_labels)
'''
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
plt.clim(-lim_j, lim_j)
plt.title('j$_y$ Neutralsheetcurrent calculated only from B(KTH)')
column_labels = x_condition_list 
row_labels = z_condition_list
ax.set_xticklabels(column_labels)
ax.set_yticklabels(row_labels)

fig, ax = plt.subplots()
array = j_y_4_average
#array[array<0.5]=np.nan
m = np.ma.masked_where(np.isnan(array),array)
plt.pcolor(m, cmap='RdBu')
plt.colorbar()       
plt.show()
plt.xlabel('x in R$_M$')
plt.ylabel('z in R$_M$')
plt.grid()
plt.clim(-lim_j, lim_j)
plt.title('j$_y$ partial Ringcurrent calculated only from B(KTH)')
column_labels = x_condition_list 
row_labels = z_condition_list
ax.set_xticklabels(column_labels)
ax.set_yticklabels(row_labels)



fig, ax = plt.subplots()
array = j_y_2_average
#array[array<0.5]=np.nan
m = np.ma.masked_where(np.isnan(array),array)
plt.pcolor(m, cmap='RdBu')
plt.colorbar()       
plt.show()
plt.xlabel('x in R$_M$')
plt.ylabel('z in R$_M$')
plt.grid()
plt.clim(-lim_j, lim_j)
plt.title('j$_y$ Neutralsheetcurrent calculated only from B(KTH) (all), int + ext')
column_labels = x_condition_list 
row_labels = z_condition_list
ax.set_xticklabels(column_labels)
ax.set_yticklabels(row_labels)


#add = 'updated Neutralcurrentsheet Parameters'

###################################################################

fig, axs = plt.subplots(2, 3)
fig.suptitle('Updated Parameters for Neutralcurrentsheet', fontsize=20)

array = average_kth_b_x_dipole #average_kth_b_x
m = np.ma.masked_where(np.isnan(array),array)
plot1 = axs[0,0].pcolor(m, cmap='RdBu')
#plt.colorbar(plot1) 
axs[0,0].grid()      
plot1.set_clim(-lim_b,lim_b)
axs[0,0].set_xlabel('x in R$_M$')
axs[0,0].set_ylabel('z in R$_M$')
axs[0,0].set_title('Bx KTH Dipole')
column_labels = x_condition_list 
row_labels = z_condition_list
axs[0,0].set_xticklabels(column_labels)
axs[0,0].set_yticklabels(row_labels)

#view_colormap('RdBu')


###################################################################
#fig, ax = plt.subplots()
array = average_kth_b_z_dipole
m = np.ma.masked_where(np.isnan(array),array)
plot2 = axs[1,0].pcolor(m, cmap='RdBu')
#axs[1,0].colorbar(plot2) 
plt.grid()      
plot2.set_clim(-lim_b,lim_b)
axs[1,0].set_xlabel('x in R$_M$ in MSO')
axs[1,0].set_ylabel('z in R$_M$ in MSO')
axs[1,0].set_title('Bz KTH Dipole')
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


array = b_x_neutralcurrent
m = np.ma.masked_where(np.isnan(array),array)
plot3 = axs[0,1].pcolor(m, cmap='RdBu')
#axs[0,1].colorbar(plot3)       
plt.show()
axs[0,1].set_xlabel('x in R$_M$ in MSO')
axs[0,1].set_ylabel('z in R$_M$ in MSO')
plt.grid()
plot3.set_clim(-lim_b,lim_b)
axs[0,1].set_title('Bx Neutralcurrentsheet')
column_labels = x_condition_list 
row_labels = z_condition_list
axs[0,1].set_xticklabels(column_labels)
axs[0,1].set_yticklabels(row_labels)


####################################################################


array = b_z_neutralcurrent
m = np.ma.masked_where(np.isnan(array),array)
plot4 = axs[1,1].pcolor(m, cmap='RdBu')
axs[1,1].set_xlabel('x in R$_M$ in MSO')
axs[1,1].set_ylabel('z in R$_M$ in MSO')
axs[1,1].grid()
plot4.set_clim(-lim_b,lim_b)
axs[1,1].set_title('Bz Neutralcurrentsheet' )
column_labels = x_condition_list 
row_labels = z_condition_list
axs[1,1].set_xticklabels(column_labels)
axs[1,1].set_yticklabels(row_labels)


####################################################################



array = average_kth_b_x
m = np.ma.masked_where(np.isnan(array),array)
plot5= axs[0,2].pcolor(m, cmap='RdBu')
axs[0,2].set_xlabel('x in R_M')
axs[0,2].set_ylabel('z in R_M')
axs[0,2].grid()
plot5.set_clim(-lim_b,lim_b)
axs[0,2].set_title('Bx Dipole + Neutralcurrentsheet')
column_labels = x_condition_list 
row_labels = z_condition_list
axs[0,2].set_xticklabels(column_labels)
axs[0,2].set_yticklabels(row_labels)

###################################################################


array = average_kth_b_z
#array[array<0.5]=np.nan
m = np.ma.masked_where(np.isnan(array),array)
plot6 = axs[1,2].pcolor(m, cmap='RdBu')
axs[1,2].set_xlabel('x in R$_M$')
axs[1,2].set_ylabel('z in R$_M$')
axs[1,2].grid()
plot6.set_clim(-lim_b,lim_b)
axs[1,2].set_title('Bz Dipole + Neutralcurrentsheet' )
column_labels = x_condition_list 
row_labels = z_condition_list
axs[1,2].set_xticklabels(column_labels)
axs[1,2].set_yticklabels(row_labels)




fig.subplots_adjust(right=0.8)
cbar = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar.set_title('B in nT or j in nA/m$^2$')
fig.colorbar(plot1, cax=cbar)


plt.show()










