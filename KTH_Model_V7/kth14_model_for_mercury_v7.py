# KTH14 Model for Mercury


###################################################################################################################

# Description:
#      Calculates the magnetospheric field for Mercury. Based on Korth et al., (2015) with some improvements.
#      Model is intended for planning purposes of the BepiColombo mission. Keep the model within the team. 
#      If you plan to make a publication with the aid of this model, the opportunity to participate as co-author
#      would be appreciated. 
#      If you have suggestions for improvements, do not hesitate to write me an email.
#      
#      Takes into account:
#        - internal dipole field (offset dipole)
#        - field from neutral sheet current
#        - respective shielding fields from magnetopause currents
#        - aberration effect due to orbital motion of Mercury
#        - scaling with heliocentric distance
#        - scaling with Disturbance Indec (DI)
#      If no keywords are set, the total field from all modules will be calculated.
#
#       Required python packages: numpy, scipy
#
# Parameters:
#      x_mso: in, required, X-positions (array) in MSO base given in km
#      y_mso: in, required, Y-positions (array) in MSO base given in km
#      z_mso: in, required, z-positions (array) in MSO base given in km
#      r_hel: in, required, heliocentric distance in AU 
#      DI: in, required, disturbance index (0 < DI < 100), if not known: 50 
#      modules: dipole (internal and external), neutralsheet (internal and external)
#      "external = True" calculates the cf-fields (shielding fields) for each module which is set true
# 
# Return: 
#     Bx, By, Bz in nT for each coordinate given (x_mso, y_mso, z_mso)
#      
#    :Author:
#      Daniel Heyner, Institute for Geophysics and extraterrestrial Physics, Braunschweig, Germany, d.heyner@tu-bs.de
#      Kristin Pump, Institute for Geophysics and extraterrestrial Physics, Braunschweig, Germany, k.pump@tu-bs.de
#

#
#
###################################################################################################################

import numpy as np
import json
from model_field_v7 import model_field_v7


def kth14_model_for_mercury_v7(x_mso, y_mso, z_mso, r_hel, di, dipole=True, neutralsheet=True, pRC = True, internal=True,
                                  external=True):


    if len(np.atleast_1d(x_mso)) != len(np.atleast_1d(y_mso)) != len(np.atleast_1d(z_mso)):
        print('Number of positions (x,y,z) do not match')
        exit()

    if internal == False and external == False:
        print('Internal and external field are both set \"False\". Set at least one True.')
        quit()

    if (dipole == False and neutralsheet == False and pRC == False):
        print('Dipole, neutralsheet and pRC are set \"False\". Set at least one True.')
        quit()

    ############################################################################################
    #        Reading control parameters and shielding coefficients from file                   #
    ############################################################################################

    with open("control_params_v7_tmp.json", "r") as file:
        control_params = json.load(file)

    shielding_input = 'kth_own_cf_fit_parameters_v7.dat'
    shielding_par_file = open(shielding_input, "r")
    shielding_params = np.loadtxt(shielding_par_file)
    shielding_par_file.close()

    # defining the lengths of the following arrays
    n_lin_int = shielding_params[0].astype(int)
    n_non_lin_int = shielding_params[1].astype(int)
    n_lin_neutralcurrent = shielding_params[2].astype(int)
    n_non_lin_neutralcurrent = shielding_params[3].astype(int)

    # length check
    length_check = 4 + n_lin_int + 3 * n_non_lin_int + n_lin_neutralcurrent + 3 * n_non_lin_neutralcurrent

    if len(shielding_params) != length_check:
        print('Wrong shielding coefficients file length.')
        exit()

    # define coefficient arrays

    factor_dipole = 1.02895       #TO DO: dynamisch ändern                                                    ###################################################

    low = 4
    high = low + n_lin_int
    lin_coeff_int = shielding_params[low:high]
    control_params['lin_coeff_int'] = lin_coeff_int * factor_dipole

    low = high
    high = low + n_lin_neutralcurrent
    lin_coeff_disk = shielding_params[low:high]
    control_params['lin_coeff_disk'] = lin_coeff_disk

    low = high
    high = low + n_non_lin_int
    p_i_int = shielding_params[low:high]
    control_params['p_i_int'] = p_i_int * factor_dipole

    low = high
    high = low + n_non_lin_neutralcurrent
    p_i_disk = shielding_params[low:high]
    control_params['p_i_disk'] = p_i_disk

    low = high
    high = low + n_non_lin_int
    q_i_int = shielding_params[low:high]
    control_params['q_i_int'] = q_i_int * factor_dipole

    low = high
    high = low + n_non_lin_neutralcurrent
    q_i_disk = shielding_params[low:high]
    control_params['q_i_disk'] = q_i_disk

    low = high
    high = low + n_non_lin_int
    x_shift_int = shielding_params[low:high]
    control_params['x_shift_int'] = x_shift_int * factor_dipole

    low = high
    high = low + n_non_lin_neutralcurrent
    x_shift_disk = shielding_params[low:high]
    control_params['x_shift_disk'] = x_shift_disk


    #######################################################################################
    # DI-Scaling
    #######################################################################################
    if len(np.atleast_1d(di)) > 1:
        if any(t > 100 for t in di):
            print('At least one element in DI is greater than 100. DI must be between 0 and 100. If you don\'t know the '
                'exact value, use 50.')
            exit()

        if any(t < 0 for t in di):
            print(
             'At least one element in DI is negative. DI must be between 0 and 100. If you don\'t know the exact value, use 50.')
            exit()
    elif len(np.atleast_1d(di)) == 1:
        if di < 0:
            print('Disturbance index di must be between 0 and 100. If you don\'t know the exact value, use 50.')
            exit()
        if di > 100:
            print('Disturbance index di must be between 0 and 100. If you don\'t know the exact value, use 50.')
            exit()

    f = 2.0695 - (0.00355 * di)  # f is a factor for the scaling for R_SS (subsolar standoff distance)

    #######################################################################################
    # RMP-Scaling
    #######################################################################################

    if len(np.atleast_1d(r_hel)) > 1:
        if any(r_hel) > 1:
            print('Please use r_hel (heliocentric distance) in AU, not in km.')
            exit()
    if len(np.atleast_1d(r_hel)) == 1:
        if r_hel > 1:
            print('Please use r_hel (heliocentric distance) in AU, not in km.')
            exit()

    R_SS = f * r_hel ** (1 / 3) * control_params['RPL']

    control_params['kappa'] = control_params['RMP'] / R_SS
    control_params['kappa3'] = (control_params['kappa']) ** 3
    #R_SS = 10
    #control_params['kappa'] = 1.0
    #control_params['kappa3'] = 1.0
    
    #print('Achtung, Rss auf 10, kappa auf 1')
    ################################################################################################################
    # Application of the offset: MSO->MSM coordinate system
    # Scaling to planetary radius
    # Scaling with heliocentric distance
    ################################################################################################################

    dipole_offset = 479 / control_params['RPL']  # value of offset by Anderson et al. (2012)
    x_msm = x_mso / control_params['RPL']
    y_msm = y_mso / control_params['RPL']
    z_msm = z_mso / control_params['RPL'] - dipole_offset

    ################################################################################################################
    # Check for points lying outside the magnetopause. The magnetic field for these will be set to zero.
    ################################################################################################################

    r_mp_data = np.sqrt((x_mso) ** 2 + (y_mso) ** 2 + (z_mso) ** 2) / control_params['RPL']


    if len(np.atleast_1d(r_mp_data)) > 1:
        #for i in range(len(r_mp_data)):
            #if r_mp_data[i] < 0.5:
                #print('You chose points inside the planet. Aborting calculation.')
                #exit()

        r_mp_data = np.sqrt((x_msm) ** 2 + (y_msm) ** 2 + (z_msm) ** 2)
        rho_x = np.sqrt(y_msm * y_msm + z_msm * z_msm)
        epsilon = np.arctan2(rho_x, x_msm)

        r_mp_check = R_SS * (2 / (1 + np.cos(epsilon))) ** control_params['alpha']

        r_mp_check = r_mp_check / control_params['RPL']
        n_points = len(np.atleast_1d(x_msm))

        usable_indices = np.where(r_mp_data <= (r_mp_check))
        
        #np.savetxt('usable_indices.txt', usable_indices)


        if len(usable_indices[0]) == 0:
            print('No points within the magnetopause! Aborting calculation...')
            usable_indices = []
            np.savetxt('usable_indices.txt', usable_indices)
            return np.array([np.nan, np.nan, np.nan])
            #exit()
            #print('Achtung, hier # weg nehmen')

        elif len([usable_indices]) < n_points:
            x_msm = x_msm[usable_indices]
            y_msm = y_msm[usable_indices]
            z_msm = z_msm[usable_indices]
            di = di[usable_indices]
            #print("Achtung, usable indices ausgestellt") 

        points_outside = len(np.atleast_1d(x_mso)) - len(usable_indices[0])
        
        #print(usable_indices)
        #print(len([usable_indices]))
        #print(len(usable_indices[0]))
        
        np.savetxt('usable_indices.txt', usable_indices)
         
            
        if points_outside > 0:
            print(str(points_outside) + ' of ' + str(len(np.atleast_1d(x_mso))) + ' points were taken out because they are outside the calculated magnetopause.')

        control_params['kappa'] = control_params['kappa'][usable_indices]
        control_params['kappa3'] = control_params['kappa3'][usable_indices]        
        #print('Achtung, keine skalierung mit kappa')
        
    elif len(np.atleast_1d(x_mso)) == 1:
        r_mp_data = np.sqrt((x_mso) ** 2 + (y_mso) ** 2 + (z_mso) ** 2) / control_params['RPL']
        #if r_mp_data < 1:
            #print('You chose the point inside the planet. Aborting calculation.')
            #exit()
        
        r_mp_data = np.sqrt((x_msm) ** 2 + (y_msm) ** 2 + (z_msm) ** 2)
        rho_x = np.sqrt(y_msm * y_msm + z_msm * z_msm)
        epsilon = np.arctan2(rho_x, x_msm)

        r_mp_check = R_SS * (2 / (1 + np.cos(epsilon))) ** control_params['alpha']

        r_mp_check = r_mp_check / control_params['RPL']
        

        if r_mp_data <= (r_mp_check):
            usable_indices = np.where(r_mp_data <= (r_mp_check))
            pass

        elif r_mp_data > (r_mp_check):
            print('No points within the magnetopause! Aborting calculation...')
            usable_indices = []
            print('Empty List of usable indices.')
            #exit()
        
        np.savetxt('usable_indices.txt', usable_indices)




    ##############################################################
    # Calculation of the model field
    #############################################################
    #np.savetxt('usable_indices_flyby', usable_indices)
    result = model_field_v7(x_msm, y_msm, z_msm, di, dipole, neutralsheet, pRC, internal, external, control_params)
    #print('result bx, by, bz: \n' , result)

    return result

