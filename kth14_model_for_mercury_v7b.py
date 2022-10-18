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
import scipy.special as special
from scipy.integrate import simps
import mpmath
import sys 
import matplotlib.pyplot as plt



def kth14_model_for_mercury_v7b(x_mso, y_mso, z_mso, r_hel, di, control_param_path, fit_param_path, dipole=True, neutralsheet=True, pRC = True, internal=True,
                                  external=True):

    #Daniel: 
    # reshape, so that the coordinates are all 1D
    x_mso = np.array(x_mso).flatten()
    y_mso = np.array(y_mso).flatten()
    z_mso = np.array(z_mso).flatten()
    
    
    if type(x_mso) != float :
        shape = np.array(x_mso.shape) 
        if shape.size >= 2 :
            print('Positions must be float or 1D arrays! Aborting... ')
            sys.exit()
            

    if x_mso.size != y_mso.size != z_mso.size:
        print('Number of positions (x,y,z) do not match')
        sys.exit()

    if internal == False and external == False:
        print('Internal and external field are both set \"False\". Set at least one True.')
        sys.exit()

    if (dipole == False and neutralsheet == False and pRC == False):
        print('Dipole, neutralsheet and pRC are set \"False\". Set at least one True.')
        sys.exit()

    ############################################################################################
    #        Reading control parameters and shielding coefficients from file                   #
    ############################################################################################

    #Daniel: with open("control_params_v7_tmp.json", "r") as file:
    with open(control_param_path, "r") as file:
        control_params = json.load(file)


    #Daniel: shielding_input = 'kth_own_cf_fit_parameters_v7b.dat'
    shielding_input = fit_param_path
    shielding_par_file = open(shielding_input, "r")
    shielding_params = np.loadtxt(shielding_par_file)
    shielding_par_file.close()

    # defining the lengths of the following arrays
    n_lin_int = shielding_params[0].astype(int)   #16
    n_non_lin_int = shielding_params[1].astype(int)   #4
    n_lin_neutralcurrent = shielding_params[2].astype(int)   #0
    n_non_lin_neutralcurrent = shielding_params[3].astype(int)   #0
    #print(shielding_params)

    # length check
    length_check = 4 + n_lin_int + 3 * n_non_lin_int + n_lin_neutralcurrent + 3 * n_non_lin_neutralcurrent

    if len(shielding_params) != length_check:
        print('Wrong shielding coefficients file length.')
        sys.exit()

    # define coefficient arrays

    # Daniel: Shielding parameters were determined for a dipole coefficient of -190 nT. Thus, for the new value of the dipole field 
    # the shield field must be adapted as well
    # Daniel: factor_dipole = -190.0/ control_params['g10_int']           
    #factor_dipole =  control_params['g10_int'] / -190.0 # if the chosen internal dipole coefficient is increased, the shielding field must also be increased
    
    low = 4
    high = low + n_lin_int
    lin_coeff_int = shielding_params[low:high]
    control_params['lin_coeff_int'] = lin_coeff_int
    #print('lin_coeff_in in kth main: ', lin_coeff_int)


    low = high
    high = low + n_lin_neutralcurrent
    lin_coeff_disk = shielding_params[low:high]
    control_params['lin_coeff_disk'] = lin_coeff_disk


    low = high
    high = low + n_non_lin_int
    p_i_int = shielding_params[low:high]
    control_params['p_i_int'] = p_i_int 


    low = high
    high = low + n_non_lin_neutralcurrent
    p_i_disk = shielding_params[low:high]
    control_params['p_i_disk'] = p_i_disk

    low = high
    high = low + n_non_lin_int
    q_i_int = shielding_params[low:high]
    control_params['q_i_int'] = q_i_int 

    low = high
    high = low + n_non_lin_neutralcurrent
    q_i_disk = shielding_params[low:high]
    control_params['q_i_disk'] = q_i_disk

    low = high
    high = low + n_non_lin_int
    x_shift_int = shielding_params[low:high]
    control_params['x_shift_int'] = x_shift_int 

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
            sys.exit()

        if any(t < 0 for t in di):
            print(
             'At least one element in DI is negative. DI must be between 0 and 100. If you don\'t know the exact value, use 50.')
            sys.exit()
    elif len(np.atleast_1d(di)) == 1:
        if di < 0:
            print('Disturbance index di must be between 0 and 100. If you don\'t know the exact value, use 50.')
            sys.exit()
        if di > 100:
            print('Disturbance index di must be between 0 and 100. If you don\'t know the exact value, use 50.')
            sys.exit()

    #f = 2.0695 - (0.00355 * di)  # f is a factor for the scaling for R_SS (subsolar standoff distance)
    f = control_params['f_a'] + (control_params['f_b'] * di)  # f is a factor for the scaling for R_SS (subsolar standoff distance)

    #######################################################################################
    # RMP-Scaling
    #######################################################################################

    if len(np.atleast_1d(r_hel)) > 1:
        if any(r_hel) > 5:
            print('Please use r_hel (heliocentric distance) in AU, not in km.')
            sys.exit()
    if len(np.atleast_1d(r_hel)) == 1:
        if r_hel > 5:
            print('Please use r_hel (heliocentric distance) in AU, not in km.')
            sys.exit()

    R_SS = f * r_hel ** (1 / 3) * control_params['RPL']
    

    control_params['kappa'] = control_params['RMP'] / R_SS
    control_params['kappa3'] = (control_params['kappa']) ** 3
    

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
    # Also give a warning for calculation within the core
    ################################################################################################################

    r_mso = np.sqrt((x_mso) ** 2 + (y_mso) ** 2 + (z_mso) ** 2) 
    r_msm = np.sqrt((x_msm) ** 2 + (y_msm) ** 2 + (z_msm) ** 2) 
    
    
    if min(r_mso) < 1: print('Warning: You chose points inside the core.')
      
    r_mp_check = shue_mp_calc_r_mp(x_msm, y_msm, z_msm, R_SS, control_params['alpha']) / control_params['RPL']
    #print('r_mp_check: ', r_mp_check[0:10])
    #print('r_msm: ', r_msm[0:10])

    usable_indices = np.where(r_msm <= r_mp_check)
    #usable_indices = np.array(usable_indices)
    #print(usable_indices)
    np.savetxt('usable_indices.txt', usable_indices)

    n_points = x_msm.size
    
    #print('length usable indices in kth: ', len(usable_indices[0]))
    #print('n_points in kth: ', n_points)
    
    #print(n_points)
    if usable_indices[0].size == 0:
        print('No points within the magnetopause! Setting result to zero...')
        return np.zeros(x_msm.size)

    
    elif len([usable_indices[0]]) < n_points:
        #restrict to points within magnetopause
        #print('xmsm vorher: ', len(x_msm))
        #print(x_msm)
        x_msm = x_msm[usable_indices]
        #print('x_msm nachher: ', len(x_msm))
        #print(x_msm)
        y_msm = y_msm[usable_indices]
        z_msm = z_msm[usable_indices]
        control_params['kappa'] = control_params['kappa'][usable_indices]
        control_params['kappa3'] = control_params['kappa3'][usable_indices]
        di = di[usable_indices]
        
    points_outside = len(np.atleast_1d(x_mso)) - len(usable_indices[0])
    # print(points_outside)
       
    x_msm=x_msm.flatten()
    y_msm=y_msm.flatten()
    z_msm=z_msm.flatten()
    
    ##############################################################
    # Calculation of the model field
    #############################################################

    result = model_field_v7b(x_msm, y_msm, z_msm, di, dipole, neutralsheet, pRC, internal, external, control_params, R_SS)

    
    return result

def model_field_v7b(x_msm_in, y_msm_in, z_msm_in, di, dipole, neutralsheet, pRC, internal, external, control_params, R_SS):

    g10_int_ind = control_params['g10_int_ind']
    kappa3      = control_params['kappa3']
    g10_int     = control_params['g10_int']
    aberration  = control_params['aberration']
    
    

    t = control_params['t_a'] + control_params['t_b'] * di
    #print('t = ', t)
    # t = 3.075 + 0.0065 * DI

    n_points = x_msm_in.size
    # application of the aberration to the coordinates
    x_msm = x_msm_in * np.cos(aberration) + y_msm_in * np.sin(aberration)
    y_msm = - x_msm_in * np.sin(aberration) + y_msm_in * np.cos(aberration)
    z_msm = z_msm_in

    # multiply with kappa
    x_msm_k = x_msm * control_params['kappa']
    y_msm_k = y_msm * control_params['kappa']
    z_msm_k = z_msm * control_params['kappa']


    #Daniel: B_total = np.zeros([3, len(np.atleast_1d(x_msm_in))])
    if n_points == 1:
        B_total = np.zeros(3)
    else : 
        B_total = np.zeros([3, n_points])
        
    ##################################################
    # calculate fields
    #################################################


    if dipole:
        if internal:

            B_int = kappa3 * internal_field_v7b(x_msm_k, y_msm_k, z_msm_k, control_params)
            #Daniel: B_total = B_int
            B_total = B_total + B_int
            
        if external:
            #print('calc dipole external')
            
            #Daniel: 
            # This was intended to include any induced internal dipole fields. 
            # The coefficient was based on empirical work. 
            # TODO: This needs to be checked again!
            induction_scale_fac = -0.0052631579 * (g10_int + g10_int_ind)
            B_cf_int = kappa3 * induction_scale_fac * cf_field_v7b(x_msm_k, y_msm_k, z_msm_k,
                                                                  control_params['lin_coeff_int'],
                                                                  control_params['p_i_int'], control_params['q_i_int'],
                                                                  control_params['x_shift_int'])
            
            #Daniel: this takes into account that the shielding fields were calculated for a g10_int = -190 nT 
            # It is not clear whether this scaling is really ok. Was this truely checked?
            B_total = B_total + B_cf_int


    if neutralsheet:
        if internal:
            z_offset = 0.
            '''
            try:
                x_msm_load = np.loadtxt('x_msm_input_in_NSC_BS.txt')
                if np.array_equal(x_msm_load, x_msm): 
                    B_tail_ns = np.loadtxt('B_tail_ns.txt')
                    print('B_tail_ns found.')
            
            except: 
                print('No Input found. Calculate new B_tail_ns.')
                B_tail_ns = tail_field_ns_bs_v7b(x_msm, y_msm, z_msm, z_offset)           
                np.savetxt('x_msm_input_in_NSC_BS.txt', x_msm)           
                np.savetxt('B_tail_ns.txt', B_tail_ns)
            else: 
                B_tail_ns = tail_field_ns_bs_v7b(x_msm, y_msm, z_msm, z_offset)           
                np.savetxt('x_msm_input_in_NSC_BS.txt', x_msm)           
                np.savetxt('B_tail_ns.txt', B_tail_ns)
            '''    
                
            B_tail_ns = tail_field_ns_bs_v7b(x_msm, y_msm, z_msm, z_offset,  control_params)      
            B_total = B_total + t * B_tail_ns
                

            

        if external:
            #add image sheets at twice the asymptotic tail radius with the opposite current direction
            '''
            R_tail_asymptotic = R_SS * 2. / control_params['RPL']
            
            z_offset = 2. * R_tail_asymptotic
            B_image1_ns = -tail_field_ns_bs_v7b(x_msm, y_msm, z_msm, z_offset)

            z_offset = -2. * R_tail_asymptotic
            B_image2_ns = -tail_field_ns_bs_v7b(x_msm, y_msm, z_msm, z_offset)
            '''
            
            #calculate the Chapman-Ferraro part
            B_cf_ns = 0.05 * cf_field_v7b(x_msm_k, y_msm_k, z_msm_k, control_params['lin_coeff_disk'],
                                    control_params['p_i_disk'], control_params['q_i_disk'],
                                    control_params['x_shift_disk'])
            
            #add all together
            B_total = B_total + (t * B_cf_ns)# + B_image1_ns + B_image2_ns 
            
    if pRC:
        if internal:
            #print('calc neutralsheet internal')
            scale_fac = -4.0
            B_tail_ring = scale_fac * tail_field_pRC_v7b(x_msm_k, y_msm_k, z_msm_k, di, control_params)
            B_total = B_total + B_tail_ring

        if external:
            #Daniel: the ringe current field is so small, so a shielding is not necessarily required. This 
            # is why the cf_disk part is deactivated. 
            #print('calc neutralsheet external')

            B_total = B_total 


    ##################################################
    # rotate magnetic field back to MSO base
    #################################################
    #if len(np.atleast_1d(x_msm_in)) == 1:
    if x_msm_in.size == 1:
        bx = B_total[0]
        by = B_total[1]
        bz = B_total[2] 
    else:
        bx = B_total[0, :]
        by = B_total[1, :]
        bz = B_total[2, :]

    b_x = bx * np.cos(aberration) - by * np.sin(aberration)
    b_y = bx * np.sin(aberration) + by * np.cos(aberration)
    b_z = bz

    return np.array([b_x, b_y, b_z])


def cf_field_v7b(x_msm: np.ndarray, y_msm: np.ndarray, z_msm: np.ndarray, lin_coeff: list, p_i, q_i, x_shift):
    # this function calculates the chapman-ferraro-field (schielding field) for the KTH Model

    x_msm = np.array(x_msm)
    y_msm = np.array(y_msm)
    z_msm = np.array(z_msm)
    
    n_vec = x_msm.size
 
    N = len(p_i)
    

    b_x_cf = np.zeros(n_vec)
    b_y_cf = np.zeros(n_vec)
    b_z_cf = np.zeros(n_vec)

    for i_vec in range(n_vec):
        for i in range(N):
            
            for k in range(N):
                #print('i = ', i)
                #print('k= ', k)
                #print('N = ', N)
                pq = np.sqrt(p_i[i] * p_i[i] + q_i[k] * q_i[k])
                
                lin_index = i * N + k
                #print('lin index: ', lin_index)
                #print('lin coeff: ', lin_coeff[lin_index])

                b_x_cf[i_vec] = b_x_cf[i_vec] - pq * lin_coeff[lin_index] * np.exp(
                    pq * (x_msm[i_vec] - x_shift[i])) * np.cos(p_i[i] * y_msm[i_vec]) * np.sin(
                    q_i[k] * z_msm[i_vec])
                        
                b_y_cf[i_vec] = b_y_cf[i_vec] + p_i[i] * lin_coeff[lin_index] * np.exp(
                    pq * (x_msm[i_vec] - x_shift[i])) * np.sin(p_i[i] * y_msm[i_vec]) * np.sin(
                    q_i[k] * z_msm[i_vec])
                        
                b_z_cf[i_vec] = b_z_cf[i_vec] - q_i[k] * lin_coeff[lin_index] * np.exp(
                    pq * (x_msm[i_vec] - x_shift[i])) * np.cos(p_i[i] * y_msm[i_vec]) * np.cos(
                    q_i[k] * z_msm[i_vec])


    return np.array([b_x_cf, b_y_cf, b_z_cf])


def current_profile_pRC_v7b(rho, control_params):
    #Daniel: this function calculates the current profile of the partial ring current
    
    d = control_params['d']
    e = control_params['e']
    f = control_params['f']
    
    
    current = -(d/f*np.sqrt(2*np.pi)) * np.exp(-(rho-e)**2/(2*f**2))
    
    #plt.figure()
    #plt.plot(-rho, current)
    #plt.title('Current from current_profile_pRC.py')
    
    return current

def internal_field_v7b(x_msm, y_msm, z_msm, control_params):
    """Text
    Params:
        x_msm: Arary mit den Daten xyz
    """
    # this calculates the magnetic field of an internal axisymmetric
    # dipole in a standard spherical harmonic expansion. The field
    # is then rotated back to the cartesian coordinate system base.

    # INPUT COORDINATES ARE IN PLANETARY RADII



    g10_int_ind = control_params['g10_int_ind']
    g10 = control_params['g10_int']

    # transform to MSO coordinates

    x_mso = np.array(x_msm)
    y_mso = np.array(y_msm)
    z_mso = np.array(z_msm) + 0.196
    #z_mso = np.array(z_msm)

    r_mso       = np.sqrt(x_mso ** 2 + y_mso ** 2 + z_mso ** 2)
    phi_mso     = np.arctan2(y_mso, x_mso)
    theta_mso   = np.arccos(z_mso / r_mso)

    # spherical harmonic synthesis of axisymmetric components
    # Daniel: higher degree coefficients from Anderson et al. 2012

    g20 = -74.6
    g30 = -22.0
    g40 = -5.7

    # l=1
    b_r_dip = 2. * (1. / r_mso) ** 3. * (g10 + g10_int_ind) * np.cos(theta_mso)
    b_t_dip = (1. / r_mso) ** 3. * (g10 + g10_int_ind) * np.sin(theta_mso)


    # l=2
    b_r_quad = 3. * (1. / r_mso) ** 4. * g20 * 0.5 * (3. * np.cos(theta_mso) ** 2. - 1.)
    b_t_quad = (1. / r_mso) ** 4. * g20 * 3. * (np.cos(theta_mso) * np.sin(theta_mso))


    # l=3
    b_r_oct = 4. * (1. / r_mso) ** 5. * g30 * 0.5 * (5. * np.cos(theta_mso) ** 3. - 3. * np.cos(theta_mso))
    b_t_oct = (1. / r_mso) ** 5. * g30 * 0.375 * (np.sin(theta_mso) + 5. * np.sin(3. * theta_mso))


    # l=4
    b_r_hex = 5. * (1. / r_mso) ** 6. * g40 * (0.125 * (35. * np.cos(theta_mso) ** 4. - 30. * np.cos(theta_mso) ** 2. + 3.))
    b_t_hex = (1. / r_mso) ** 6. * g40 * (0.3125 * (2. * np.sin(2. * theta_mso) + 7. * np.sin(4. * theta_mso)))


    # add multipoles together
    b_r = b_r_dip + b_r_quad + b_r_oct + b_r_hex
    b_t = b_t_dip + b_t_quad + b_t_oct + b_t_hex


    # rotate to mso coordinate base
    b_x_mso_int = b_r * np.sin(theta_mso) * np.cos(phi_mso) + b_t * np.cos(theta_mso) * np.cos(phi_mso)
    b_y_mso_int = b_r * np.sin(theta_mso) * np.sin(phi_mso) + b_t * np.cos(theta_mso) * np.sin(phi_mso)
    b_z_mso_int = b_r * np.cos(theta_mso) - b_t * np.sin(theta_mso)



    return np.array([b_x_mso_int, b_y_mso_int, b_z_mso_int])



        
def a_phi_hankel_v7b(H_current, rho_z_in, phi, z, lambda_arr, d_0, delta_x, scale_x_d, delta_y):
    # This function calculates the vector potential a_phi with the results from the Hankel transformation of
    # the neutral sheet current.

    x = rho_z_in * np.cos(phi)
    y = rho_z_in * np.sin(phi)

    sheet_thickness = (d_0 + delta_x * np.exp((x) / scale_x_d) + delta_y * (y) ** 2)
    
    
    #Daniel: test thinnning
    #rho_0 = 1. * 2440.
    #sheet_thickness = (d_0 + delta_x * np.exp((x) / scale_x_d) + delta_y * (y) ** 2) * (1.-rho_0**2 / (rho_0**2 + rho_z_in**2))

    
    integrand = H_current * special.j1(lambda_arr * rho_z_in) * np.exp( -lambda_arr * np.sqrt(z ** 2 + sheet_thickness ** 2))
  
    result_a_phi_hankel = simps(integrand, x=lambda_arr)

    return result_a_phi_hankel


    
def fx(xj, control_params):
    # xj is considered in units of planetary radii
        # result is in nA/m^2
    x_1D = np.reshape(xj, xj.size ) 
    result_1D = np.zeros(x_1D.size)
    
    a = control_params['a'] 
    c = -0.39
    
    good_indices = np.array(np.where(x_1D < -1.))
    if good_indices.size >= 1: 
        result_1D[good_indices] = a * (x_1D[good_indices] + 1.)**2 * np.exp(c*(x_1D[good_indices] + 1.)**2)
    
    return np.reshape(result_1D, xj.shape)
    
def fy(yj): 
    # yj is expected to be in units of planetary radii
    # result is in nA/m^2
    return np.exp(-0.5*(yj)**2)

def fz(zj, z_offset):
    # zj and z_offset are expected to be in units of planetary radii
    # result is in nA/m^2
    return np.exp(-0.1 * ((zj-z_offset) / 0.12)**2)


def tail_field_ns_bs_v7b(x_target,y_target,z_target, z_offset, control_params):
    
    #TODO: y- and z-bounds should be adjusted to the asymptotic tail radius in the future
    mu = 4e-7 * np.pi 
    #mu = 1.
    RPL = 2440e3
    
    #bounds for the integration over j_y
    x_bounds = [-5.*RPL, -1.*RPL]
    y_bounds = [-1.*RPL, 1.*RPL]
    z_bounds = [-1.*RPL + z_offset*RPL, 1.*RPL+ z_offset*RPL]

    # number of grid points in each dimension
    x_steps = 50
    y_steps = 20
    z_steps = 20
    
    #differential volumen
    dV = (x_bounds[1] - x_bounds[0]) / float(x_steps-1.) * (y_bounds[1] - y_bounds[0]) / float(y_steps-1.) * (z_bounds[1] - z_bounds[0]) / float(z_steps-1.)
    

    #initialize relative coordinates
    x_rel = np.zeros([x_steps, y_steps, z_steps])
    y_rel = np.zeros([x_steps, y_steps, z_steps])
    z_rel = np.zeros([x_steps, y_steps, z_steps])
                

    #create integration mesh with meshgrid 
    x_coords_1d = np.arange(x_steps) / float(x_steps-1) * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
    y_coords_1d = np.arange(y_steps) / float(y_steps-1) * (y_bounds[1] - y_bounds[0]) + y_bounds[0]
    z_coords_1d = np.arange(z_steps) / float(z_steps-1) * (z_bounds[1] - z_bounds[0]) + z_bounds[0]
    
    #prepare coordinates in multidimensional arrays - this information goes into the f_xyz functions that define the current density
    x_coords, y_coords, z_coords = np.meshgrid(x_coords_1d, y_coords_1d, z_coords_1d)
    
    x_coords = np.reshape(x_coords, x_coords.size)
    y_coords = np.reshape(y_coords, y_coords.size)
    z_coords = np.reshape(z_coords, z_coords.size)
    
    # loop over input vectors
    n_vec = len(x_target)
    
    Bx_arr = np.zeros(n_vec)
    By_arr = np.zeros(n_vec)
    Bz_arr = np.zeros(n_vec)
            
    for i_vec in range(n_vec):
        
        x_rel = x_target[i_vec]*RPL - x_coords 
        y_rel = y_target[i_vec]*RPL - y_coords
        z_rel = z_target[i_vec]*RPL - z_coords
        
        r = (x_rel**2 + y_rel**2 + z_rel**2)
        #check for singularity
        bad_indices = np.array(np.where(r <1e-6))
        if bad_indices.size >=1 : 
            r[bad_indices] = 1e-6
                            
        
        r_inv = r**(-3./2.)
        
    
        #perform integration by summing the integrand
        # result is in Nanotesla
        Bx = mu / (4. * np.pi) * np.sum( fx(x_coords/RPL, control_params) * fy(y_coords/RPL) * fz(z_coords/RPL,z_offset) * z_rel * r_inv) * dV 
        Bz = mu / (4. * np.pi) * np.sum(-fx(x_coords/RPL, control_params) * fy(y_coords/RPL) * fz(z_coords/RPL,z_offset) * x_rel * r_inv) * dV 
    
        Bx_arr[i_vec] = Bx
        Bz_arr[i_vec] = Bz
        
    return np.array([Bx_arr, By_arr, Bz_arr])

def tail_field_neutralsheet_v7b(x_msm, y_msm, z_msm, di, control_params):

    rho_z = np.sqrt(x_msm ** 2 + y_msm ** 2)
    phi = np.arctan2(y_msm, x_msm)

    d_0         = control_params['d_0']
    delta_x     = control_params['delta_x']
    scale_x_d   = control_params['scale_x_d']
    delta_y     = control_params['delta_y']


    t = control_params['t_a'] + control_params['t_b'] * di
    
    
    mu_0 = 1.0

 

    # range for any significant current density --> j should be zero for rho_max 
    rho_min = 0.0
    rho_max = 10.0
    steps = 100
    rho_hankel = np.arange(steps) / float(steps-1.) * (rho_max - rho_min) + rho_min
    current = current_profile_v7b(rho_hankel, control_params)
    
    
    lambda_min = 10 ** (-2)  # std value   
    lambda_max = 10  # std value
    lambda_steps = 100  # This value is from experience. When you change the current profile this should be checked again for sufficient convergence.

    lambda_arr = 10 ** (np.arange(lambda_steps) / float(lambda_steps-1) * (np.log10(lambda_max) - np.log10(lambda_min)) + np.log10(lambda_min))


    #perform Hankel-Transform. This is independent of the 
    #location where the magnetic field is calculated and does only depend on the 
    #current profile 
    #--> This is calculated outside the loop over magnetic field positions
    result_hankel_trafo = np.zeros(lambda_steps)
    for i in range(lambda_steps):
        result_hankel_trafo[i] = simps(special.j1(lambda_arr[i] * rho_hankel) * current * rho_hankel, x=rho_hankel)  
        # special.j1 = Bessel function of the first kind of order 1



    H_current = mu_0 / 2.0 * result_hankel_trafo

    ###############################################################

    n_vec = x_msm.size
    b_tail_x    = np.zeros(n_vec)
    b_tail_y    = np.zeros(n_vec)
    b_tail_z    = np.zeros(n_vec)
    b_tail_rho   = np.zeros(n_vec)


    for i in range(n_vec):
        a_phi = a_phi_hankel_v7b(H_current, rho_z[i], phi[i], z_msm[i], lambda_arr, d_0, delta_x, scale_x_d, delta_y)

        # numerically approximate the derivatives
        delta_z = 10 ** (-5)

        d_a_phi_d_z = (a_phi_hankel_v7b(H_current, rho_z[i], phi[i], z_msm[i] + delta_z, lambda_arr, d_0, delta_x, scale_x_d, delta_y) - 
                       a_phi_hankel_v7b(H_current, rho_z[i], phi[i], z_msm[i] - delta_z, lambda_arr, d_0, delta_x, scale_x_d, delta_y)) / ( 2 * delta_z)

        delta_rho = 10 ** (-5)
        d_a_phi_d_rho = (a_phi_hankel_v7b(H_current, rho_z[i] + delta_rho, phi[i], z_msm[i], lambda_arr, d_0, delta_x, scale_x_d, delta_y) - 
                         a_phi_hankel_v7b(H_current, rho_z[i] - delta_rho, phi[i], z_msm[i], lambda_arr, d_0, delta_x, scale_x_d, delta_y)) / (2 * delta_rho)

        b_tail_rho[i] = t * (- d_a_phi_d_z)

        if rho_z[i] <= 10 ** (-4):
            b_tail_z[i] = t * (1.0 + d_a_phi_d_rho)

        else:
            b_tail_z[i] = t * (a_phi / rho_z[i] + d_a_phi_d_rho)

        # rotate back to cartesian
        b_tail_x[i] = b_tail_rho[i] * np.cos(phi[i])
        b_tail_y[i] = b_tail_rho[i] * np.sin(phi[i])


    return np.array([b_tail_x, b_tail_y, b_tail_z])

    
def current_profile_v7b(rho, control_params):
    #this function calculates the current profile of the neutral current sheet
    
    a = control_params['a']
    # Daniel: standard parameter a= 95.
    b = 1.0
    c = control_params['c']
     # Daniel: standard parameter c=0.39
    
    
   
    current = a * (-rho + b) **2 * np.exp(-c * (-rho+b)**2)
  
    #Daniel: 
    #set current inside the planet to zero

    bad_indices = np.array(np.where(rho <1.))
    if bad_indices.size != 0 : 
        current[bad_indices] = 0.

    
    return current

def tail_field_pRC_v7b(x_msm, y_msm, z_msm, di, control_params):



    rho = np.sqrt(x_msm ** 2 + y_msm ** 2)
    phi = np.arctan2(y_msm, x_msm)

    d_0 = control_params['d_0']
    delta_x = control_params['delta_x']
    scale_x_d = control_params['scale_x_d']
    delta_y = control_params['delta_y']

    t = control_params['t_a'] + control_params['t_b'] * di

    mu_0 = 1.0
    steps = 100
    rho_min = 0.5 #these values are adapted for the specific current profile for the ring current
    rho_max = 2
    h_steps = 100  # This value is from experience. When you change the current profile this should be checked again for sufficient convergence.

    rho_hankel = np.arange(steps) / float(steps-1.) * (rho_max - rho_min) + rho_min

    current = current_profile_pRC_v7b(rho_hankel, control_params)

    #lambda_max = 10  # std value
    lambda_max = 20  # std value
    #lambda_min = 10 ** (-2)  # std value
    lambda_min = 10 ** (-1)  # std value

    lambda_result = 10 ** (np.divide(range(h_steps), (float(h_steps) - 1)) * (
            np.log10(lambda_max) - np.log10(lambda_min)) + np.log10(lambda_min))
    lambda_out = lambda_result

    integrand = current

    result_hankel_trafo = np.zeros(h_steps)
    for i in range(h_steps):
        result_hankel_trafo[i] = simps(special.j1(lambda_result[i] * rho_hankel) * integrand * rho_hankel,
                                       x=rho_hankel)  # special.j1 = Bessel function of the first kind of order 1



    H_current = mu_0 / 2.0 * result_hankel_trafo

    ###############################################################

    n_vec = len(np.atleast_1d(x_msm))
    b_tail_disk_x = np.zeros(n_vec)
    b_tail_disk_y = np.zeros(n_vec)
    b_tail_disk_z = np.zeros(n_vec)
    b_tail_disk_rho = np.zeros(n_vec)

  
    for i in range(n_vec):
        a_phi = a_phi_hankel_v7b(H_current, rho[i], phi[i], z_msm[i], lambda_out, d_0, delta_x,
                               scale_x_d, delta_y)

        # numerically approximate the derivatives
        delta_z = 10 ** (-5)

        d_a_phi_d_z = (a_phi_hankel_v7b(H_current, rho[i], phi[i], z_msm[i] + delta_z, lambda_out, d_0,
                                      delta_x, scale_x_d,
                                      delta_y) - a_phi_hankel_v7b(H_current, rho[i],
                                                                    phi[i],
                                                                    z_msm[i] - delta_z,
                                                                    lambda_out,
                                                                    d_0, delta_x,
                                                                    scale_x_d,
                                                                    delta_y)) / (
                          2 * delta_z)

        delta_rho = 10 ** (-5)
        d_a_phi_d_rho = (a_phi_hankel_v7b(H_current, rho[i] + delta_rho, phi[i], z_msm[i], lambda_out,
                                        d_0, delta_x,
                                        scale_x_d, delta_y) - a_phi_hankel_v7b(
            H_current, rho[i] - delta_rho, phi[i],
            z_msm[i], lambda_out, d_0, delta_x, scale_x_d,
            delta_y)) / (2 * delta_rho)


        b_tail_disk_rho[i] = t[i] * (- d_a_phi_d_z)

        if rho[i] <= 10 ** (-4):
            b_tail_disk_z[i] = t[i] * (1.0 + d_a_phi_d_rho)

        else:
            b_tail_disk_z[i] = t[i] * (a_phi / rho[i] + d_a_phi_d_rho)

        # rotate back to cartesian
        b_tail_disk_x[i] = b_tail_disk_rho[i] * np.cos(phi[i])
        b_tail_disk_y[i] = b_tail_disk_rho[i] * np.sin(phi[i])

    


    return np.array([b_tail_disk_x, b_tail_disk_y, b_tail_disk_z])

def current_profile_pRC_v7b(rho, control_params):
    #this function calculates the current profile of the neutralcurrentsheet
    
    d = control_params['d']
    e = control_params['e']
    f = control_params['f']
    
    
    current = -(d/f*np.sqrt(2*np.pi)) * np.exp(-(rho-e)**2/(2*f**2))
    
    #plt.figure()
    #plt.plot(-rho, current)
    #plt.title('Current from current_profile_pRC.py')
    
    return current

def shue_mp_calc_r_mp(x, y, z, RMP, alpha):
	"""
	This calculates the magnetopause distance after the Shue et al. magnetopause model
	for the radial extension of an arbitrary point.
	
	x,y,z : coordinates - arbitrary units in MSM coordinate system
	RMP : subsolar standoff distance - arbitrary units --> result will have the same units
	alpha : mp flaring parameter
	
	return : magnetopause distance w.r.t. planetary center 
	"""
	#distance to x-axis
	rho_x = np.sqrt(y**2 + z**2)
	#angle with x-axis
	epsilon = np.arctan2(rho_x,x)
	
	
	#Shue's formula
	mp_distance = RMP * np.power((2. / (1. + np.cos(epsilon))),alpha)
	
	return mp_distance
	

#test at Mercury
#should return 1.41
#print(shue_mp_calc_r_mp(1,0,0, 1.41, 0.5))

def mp_normal_v7b(x_msm, y_msm, z_msm, RMP, alpha):
    
    r       = np.sqrt(x_msm**2 + y_msm**2 + z_msm**2)
    phi     = np.arctan2(y_msm, x_msm)
    theta   = np.arccos(z_msm / r)
    
    r_mp = shue_mp_calc_r_mp(x_msm, y_msm, z_msm, RMP, alpha)
    mp_loc_x = r_mp * np.sin(theta) * np.cos(phi)
    mp_loc_y = r_mp * np.sin(theta) * np.sin(phi)
    mp_loc_z = r_mp * np.cos(theta)
    
    # first tangential vector: along rotation of gamma (rotation axis: x-axis)
    gamma     = np.arctan2(mp_loc_y, mp_loc_z)
    e_gamma_x = 0.
    e_gamma_y = np.cos(gamma)
    e_gamma_z = - np.sin(gamma)
    
    # second tangential vector: along the change of epsilon. This does NOT change gamma
    epsilon     = np.cos(2. * ((r_mp / RMP)**(- 1. / alpha)) - 1.)
    d_epsilon   = 1e-3
    new_epsilon = epsilon + d_epsilon
    
    #with the new epsilon angle, calculate the corresponding magnetpause position
    new_r_mp      = RMP * (2. / (1. + np.cos(new_epsilon)))**alpha
    new_mp_loc_x  = new_r_mp * np.cos(new_epsilon)
    new_mp_loc_y  = new_r_mp * np.sin(new_epsilon) * np.sin(gamma)
    new_mp_loc_z  = new_r_mp * np.sin(new_epsilon) * np.cos(gamma)
    
    #the difference vector is the connection vector
    connect_x = new_mp_loc_x - mp_loc_x
    connect_y = new_mp_loc_y - mp_loc_y
    connect_z = new_mp_loc_z - mp_loc_z
    
    #normalize and take the opposite direction
    magnitude = np.sqrt(connect_x**2 + connect_y**2 + connect_z**2)
  
    connect_x = -connect_x / magnitude
    connect_y = -connect_y / magnitude
    connect_z = -connect_z / magnitude
    
    #get normal direction by cross-product of tangentials
    #since both vectors are nomalized to 1 the cross-product has also the length 1
    mp_normal_x = connect_y * e_gamma_z - e_gamma_y * connect_z
    mp_normal_y = connect_z * e_gamma_x - e_gamma_z * connect_x
    mp_normal_z = connect_x * e_gamma_y - e_gamma_x * connect_y

    return mp_normal_x, mp_normal_y, mp_normal_z

def trace_field_line_single_v7b():
    mercury1 = plt.Circle((0, (479/2440)), 1, color = '0.75')

    R_M = 2440.
    x_start= -1.2*R_M
    y_start = 0
    z_start = 479
    r_hel = 0.37
    di = 50
    

    fieldlinetrace = trace_fieldline_v7b(x_start, y_start, z_start, r_hel, di, delta_t=0.3)
    fieldlinetrace = trace_fieldline_v7b(x_start, y_start, z_start, r_hel, di, delta_t=-0.3)
    #print(fieldlinetrace)
    x = fieldlinetrace[0]   #x
    y = fieldlinetrace[1]   #y
    z = fieldlinetrace[2]   #z
    


    fig, ax1 = plt.subplots()
    ax1.add_artist(mercury1) 
    plt.plot(x/R_M, z/R_M)
    ax1.axis('square')
    plt.xlim((-5, 3))
    plt.ylim((-3, 3)) 
    ax1.grid()
    ax1.invert_xaxis()
    
def trace_fieldline_v7b(x_start, y_start, z_start, r_hel, di, delta_t=0.3):

    R_M = 2440  #Radius of Mercury in km
    r = np.sqrt((x_start / R_M) ** 2 + (y_start / R_M) ** 2 + (z_start / R_M) ** 2)

    if r < 1:
        print('Radius of start point is smaller than 1. You start inside the planet! Radius = ', r)
        sys.exit()
    #else:
        #print('Radius = ', r)

    def f(x, y, z):
        return kth14_model_for_mercury_v7b(x, y, z, r_hel, di, True, True, False, True, True)

    x_trace = [x_start]
    y_trace = [y_start]
    z_trace = [z_start]

    i = 0

    while r > 1 and i < 4000:
        

        #print('i = ', i)
        r = np.sqrt((x_trace[i] / R_M) ** 2 + (y_trace[i] / R_M) ** 2 + (z_trace[i] / R_M) ** 2)
        if r < 1.001:
            #print('r < 1RM')
            break
        '''
        if i > 1 and (z_trace[1] > z_start):    #fieldline starts northward
            if z_trace[i] < 479:                # stop if fieldline crosses mag. equator
                break
        
        if i > 1 and (z_trace[1] < z_start):    #fieldline starts southward
            if z_trace[i] > 479:                 # stop if fieldline crosses mag. equator
                break
        '''    
        B = f(x_trace[i], y_trace[i], z_trace[i])
        #p = np.array([x_trace[i], y_trace[i], z_trace[i]])

        k1 = delta_t * B

        k2 = delta_t * f(x_trace[i] + 0.5 * k1[0], y_trace[i] + 0.5 * k1[1], z_trace[i] + 0.5 + k1[2])

        k3 = delta_t * f(x_trace[i] + 0.5 * k2[0], y_trace[i] + 0.5 * k2[1], z_trace[i] + 0.5 * k2[2])

        k4 = delta_t * f(x_trace[i] + k3[0], y_trace[i] + k3[1], z_trace[i] + k3[2])

        x_trace.append(x_trace[i] + (1 / 6) * (k1[0] + 2 * k2[0] + 3 * k3[0] + k4[0]))
        y_trace.append(y_trace[i] + (1 / 6) * (k1[1] + 2 * k2[1] + 3 * k3[1] + k4[1]))
        z_trace.append(z_trace[i] + (1 / 6) * (k1[2] + 2 * k2[2] + 3 * k3[2] + k4[2]))

        i = i + 1

        x_array = np.asarray(x_trace)
        y_array = np.asarray(y_trace)
        z_array = np.asarray(z_trace)
        
        
        usable_indices = np.loadtxt('C:\\Users\\Kristin\\Documents\\PhD\\KTH_Model_V7\\usable_indices.txt')
        if len(np.atleast_1d(usable_indices)) == 0: 
            break
        
        if x_array[-1] <= -4 * R_M: 
            break
        

    return (np.array([x_array, y_array, z_array]))