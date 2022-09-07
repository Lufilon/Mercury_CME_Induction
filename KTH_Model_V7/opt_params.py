# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 11:29:01 2022

@author: Kristin
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 16:53:57 2021

@author: Kristin
"""


'''

import numpy as np
import matplotlib.pyplot as plt
from kth14_model_for_mercury_v7 import kth14_model_for_mercury_v7
from scipy.optimize import minimize
from scipy.optimize import basinhopping
import json
import pandas as pd



#plt.close()
print('Load Messenger Data')
directory = 'C:\\Users\\Kristin\Documents\\PhD\\opt_params\\Positions\\'

orbit_pos_x = np.loadtxt(directory + 'orbit_pos_x_mso.txt')
orbit_pos_y = np.loadtxt(directory + 'orbit_pos_y_mso.txt')
orbit_pos_z = np.loadtxt(directory + 'orbit_pos_z_mso.txt')
orbit_b_x = np.loadtxt(directory + 'orbit_b_x.txt')
orbit_b_y = np.loadtxt(directory + 'orbit_b_y.txt')
orbit_b_z = np.loadtxt(directory + 'orbit_b_z.txt')
orbit_number = np.loadtxt(directory + 'orbit_number.txt')

with open("C:\\Users\\Kristin\\Documents\\PhD\\KTH14_Python_in_progress\\dictionary_orbit_numbers_r_hel.txt") as file:
     dict_orbit_number_r_hel = json.load(file)
     
with open("C:\\Users\\Kristin\\Documents\\PhD\\Fit_DI\\dictionary_orbit_number_DI.json") as file2:
     dict_orbit_number_DI = json.load(file2)
dict_orbit_number_DI['36.0'] = 50.0
dict_orbit_number_DI['37.0'] = 50.0
dict_orbit_number_DI['135.0'] = 50.0
dict_orbit_number_DI['136.0'] = 50.0
dict_orbit_number_DI['137.0'] = 50.0
dict_orbit_number_DI['138.0'] = 50.0
dict_orbit_number_DI['139.0'] = 50.0
dict_orbit_number_DI['140.0'] = 50.0
dict_orbit_number_DI['141.0'] = 50.0
dict_orbit_number_DI['142.0'] = 50.0
dict_orbit_number_DI['143.0'] = 50.0
dict_orbit_number_DI['144.0'] = 50.0
dict_orbit_number_DI['145.0'] = 50.0
dict_orbit_number_DI['146.0'] = 50.0
dict_orbit_number_DI['147.0'] = 50.0
dict_orbit_number_DI['148.0'] = 50.0
dict_orbit_number_DI['149.0'] = 50.0
dict_orbit_number_DI['150.0'] = 50.0
dict_orbit_number_DI['151.0'] = 50.0
dict_orbit_number_DI['152.0'] = 50.0
dict_orbit_number_DI['153.0'] = 50.0
dict_orbit_number_DI['960.0'] = 50.0
dict_orbit_number_DI['1599.0'] = 50.0
dict_orbit_number_DI['1606.0'] = 50.0
dict_orbit_number_DI['1751.0'] = 50.0
dict_orbit_number_DI['1752.0'] = 50.0
dict_orbit_number_DI['2698.0'] = 50.0
dict_orbit_number_DI['3742.0'] = 50.
     
r_hel = np.zeros(len(orbit_number))
for i in range(len(orbit_number)): 
    r_hel[i] = dict_orbit_number_r_hel[str(int(orbit_number[i]))]
    
di = np.zeros(len(orbit_number))
for i in range(len(orbit_number)): 
    di[i] = dict_orbit_number_DI[str(int(orbit_number[i]))+'.0']
    
    
#take only points with disturbance index lower than 80
    
low_di_indices = np.where(di < 66.6)
orbit_pos_x = orbit_pos_x[low_di_indices]
orbit_pos_y = orbit_pos_y[low_di_indices]
orbit_pos_z = orbit_pos_z[low_di_indices]
orbit_b_x = orbit_b_x[low_di_indices]
orbit_b_y = orbit_b_y[low_di_indices]
orbit_b_z = orbit_b_z[low_di_indices]
orbit_number = orbit_number[low_di_indices]
r_hel = r_hel[low_di_indices]
di = di[low_di_indices]


#komponentenweise differenz minimieren 


#kth14_model_for_mercury_v7(orbit_pos_x, orbit_pos_y, orbit_pos_z, r_hel, di , True, False, False, True, False)

#betragweise differenz minimieren 

start = -220.0
steps = 0.5
stop = -180.0
variable = np.arange(start, stop, steps)

var_str = 'g10_int'




#t_b = np.arange(start_b, stop_b, steps_b)
#KTH_sheet = kth14_model_for_mercury_v7(orbit_pos_x, orbit_pos_y, orbit_pos_z, r_hel, di, False, True, False, True, True)
#print('Calculate usable indices')
#usable_indices = np.loadtxt('usable_indices.txt')
#usable_indices = usable_indices.astype(int)


summe = np.zeros(len(variable))
summe_x = np.zeros(len(variable))
summe_y = np.zeros(len(variable))
summe_z = np.zeros(len(variable))
print('start for loop')
for i in range(len(variable)):
    print(var_str + ' = '+ str(variable[i]))
    #t = control_params['t_a'] + control_params['t_b'] * di
    with open("C:\\Users\Kristin\\Documents\PhD\KTH_Model_V7\\control_params_v7_May22.json", "r") as file:
        control_params = json.load(file)
    

    
    
    control_params[var_str] = variable[i]
    #print(control_params)
    #print('control params: ', control_params)
    with open("control_params_v7_tmp.json", 'w') as f:
        json.dump(control_params, f)
    
        
    

    B_KTH_Dipole = kth14_model_for_mercury_v7(orbit_pos_x, orbit_pos_y, orbit_pos_z, r_hel, di, True, False, False, True, True)
    usable_indices = np.loadtxt('usable_indices.txt')
    usable_indices = usable_indices.astype(int)
    B_KTH_x = B_KTH_Dipole[0]# + B_KTH_sheet[0]
    B_KTH_y = B_KTH_Dipole[1]# + B_KTH_sheet[1]
    B_KTH_z = B_KTH_Dipole[2]# + B_KTH_sheet[2]
    #print('B KTH x', np.mean(B_KTH_x))
    #print('B KTH y', np.mean(B_KTH_y))
    #print('B KTH z', np.mean(B_KTH_z))
    dif_x = B_KTH_x - orbit_b_x[usable_indices]
    dif_y = B_KTH_y - orbit_b_y[usable_indices]
    dif_z = B_KTH_z - orbit_b_z[usable_indices]
    summe_x[i] = np.mean(np.sqrt(dif_x**2))
    summe_y[i] = np.mean(np.sqrt(dif_y**2))
    summe_z[i] = np.mean(np.sqrt(dif_z**2))
    summe[i] = np.mean(np.sqrt(dif_x**2 + dif_y**2 + dif_z**2))
    print('Total Residuum: ', summe[i])


plt.figure()
plt.plot(variable, summe, 'o-', label = '$\Delta$B')
#plt.plot(variable, summe_x, label = 'Bx')
#plt.plot(variable, summe_y, label = 'By')
#plt.plot(variable, summe_z, label = 'Bz')
plt.title('Residuals between MESSENGER data and KTH22 dipole module')
plt.ylabel('Difference B(KTH)-B(MES) in nT')
plt.xlabel('g$_1^0$ in nT')
plt.legend()
plt.grid()

#directory = 'C:\\Users\Kristin\Documents\PhD\Neutralschichtstromanpassung_neu\\Anpassung_Komponentenabhängig\\'

df_t_fit = pd.DataFrame(variable, columns=[var_str])
df_t_fit.insert(1, " Summe Bx", summe_x, True)
df_t_fit.insert(2, " Summe By", summe_y, True)
df_t_fit.insert(3, " Summe Bz", summe_z, True)
df_t_fit.insert(4, " Summe B", summe, True)
#df_t_fit.to_pickle('C:\\Users\\Kristin\\Documents\\PhD\\Neutralschichtstromanpassung_neu\\Anpassung_Komponentenabhängig\\df_g10_int.pkl')
'''

'''



def func_min_parameters(x0): 
    
    with open("C:\\Users\\Kristin\\Documents\\PhD\\KTH_Model_V7\\control_params_v7.json", "r") as file:
        control_params = json.load(file)
    control_params['t_a'] = 1.0
    control_params['t_b'] = 0.0    
    #control_params['a'] = 89.0
   # control_params['a'] = 95.0
    control_params['c'] = 0.4
    
    control_params['d_0'] = 0.16
    control_params['delta_x'] = -0.05
    control_params['delta_y'] = 0.5
    control_params['scale_x_d'] = 2.25
    
    control_params['a'] = x0[0]
    #print(x0)
    print(control_params['a'])

    with open("control_params_v7_tmp.json", 'w') as f:
        json.dump(control_params, f)        
    
    B_KTH_v7 = kth14_model_for_mercury_v7(orbit_pos_x, orbit_pos_y, orbit_pos_z, r_hel, di, True, True, False, True, True)   
    B_KTH_x = B_KTH_v7[0]
    B_KTH_y = B_KTH_v7[1]
    B_KTH_z = B_KTH_v7[2]
    dif_x = B_KTH_x - orbit_b_x[usable_indices]
    dif_y = B_KTH_y - orbit_b_y[usable_indices]
    dif_z = B_KTH_z - orbit_b_z[usable_indices]
    summe = np.mean(np.sqrt(dif_x**2 + dif_y**2 + dif_z**2))
    #summe_dif_x = np.mean(np.sqrt(dif_x**2))
    print('Zwischensumme:      ', summe)
    return summe

x0 = 95.0
#res = basinhopping(f_dipol_parameter, x0)
print('start optimizing')
res = minimize(func_min_parameters, x0, method = 'BFGS')
#res = opt.root(f_1, x0)
print('minimum: ', res.x)
  ''' 
    
    
    