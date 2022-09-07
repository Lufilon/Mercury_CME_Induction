import numpy as np
from internal_field_v7 import internal_field_v7
from tail_field_neutralsheet_v7 import tail_field_neutralsheet_v7
from cf_field_v7 import cf_field_v7
from tail_field_pRC_v7 import tail_field_pRC_v7


def model_field_v7(x_msm_in, y_msm_in, z_msm_in, di, dipole, neutralsheet, pRC, internal, external, control_params):

    g10_int_ind = control_params['g10_int_ind']
    kappa3 = control_params['kappa3']
    g10_int = control_params['g10_int']
    aberration = control_params['aberration']

    t = control_params['t_a'] + control_params['t_b'] * di
    # t = 3.075 + 0.0065 * DI

    # application of the aberration to the coordinates
    x_msm = x_msm_in * np.cos(aberration) + y_msm_in * np.sin(aberration)
    y_msm = - x_msm_in * np.sin(aberration) + y_msm_in * np.cos(aberration)
    z_msm = z_msm_in

    # multiply with kappa
    x_msm_k = x_msm * control_params['kappa']
    y_msm_k = y_msm * control_params['kappa']
    z_msm_k = z_msm * control_params['kappa']

    B_total = np.zeros([3, len(np.atleast_1d(x_msm_in))])

    ##################################################
    # calculate fields
    #################################################

    #print('x_msm in model field: ', x_msm)

    if dipole:
        if internal:
            #print('calc dipole internal')
            B_int = kappa3 * internal_field_v7(x_msm_k, y_msm_k, z_msm_k, control_params)
            B_total = B_int


        if external:
            #print('calc dipole external')
            induction_scale_fac = -0.0052631579 * (g10_int + g10_int_ind)
            B_cf_int = kappa3 * induction_scale_fac * cf_field_v7(x_msm_k, y_msm_k, z_msm_k,
                                                                  control_params['lin_coeff_int'],
                                                                  control_params['p_i_int'], control_params['q_i_int'],
                                                                  control_params['x_shift_int'])
            B_total = B_total + (control_params['g10_int'] / (-190.0)) * B_cf_int


    if neutralsheet:
        if internal:
            #print('calc neutralsheet internal')
            scale_fac = -1.0
            B_tail_disk = scale_fac * tail_field_neutralsheet_v7(x_msm_k, y_msm_k, z_msm_k, di, control_params)
            B_total = B_total + B_tail_disk

        if external:
            #print('calc neutralsheet external')
            B_cf_disk = cf_field_v7(x_msm_k, y_msm_k, z_msm_k, control_params['lin_coeff_disk'],
                                    control_params['p_i_disk'], control_params['q_i_disk'],
                                    control_params['x_shift_disk'])
            B_total = B_total + ((t * B_cf_disk))
            
    if pRC:
        if internal:
            #print('calc neutralsheet internal')
            scale_fac = -4.0
            B_tail_ring = scale_fac * tail_field_pRC_v7(x_msm_k, y_msm_k, z_msm_k, di, control_params)
            B_total = B_total + B_tail_ring

        if external:
            #print('calc neutralsheet external')
            #B_cf_disk = cf_field_v7(x_msm_k, y_msm_k, z_msm_k, control_params['lin_coeff_disk'],
                                    #control_params['p_i_disk'], control_params['q_i_disk'],
                                    #control_params['x_shift_disk'])
            B_total = B_total 


    ##################################################
    # rotate magnetic field back to MSO base
    #################################################
    if len(np.atleast_1d(x_msm_in)) == 1:
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
