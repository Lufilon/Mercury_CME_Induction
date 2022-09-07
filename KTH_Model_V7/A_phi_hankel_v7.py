# this function is called by tail_field_neutralsheet

import numpy as np
import scipy.special as special
from scipy.integrate import simps


def a_phi_hankel_v7(H_current, rho, phi, z, lambda_result, d_0, delta_x, scale_x_d, delta_y):
    # This function calculates the vector potential a_phi with the results from the Hankel transformation of
    # the neutral sheet current.

    x = rho * np.cos(phi)
    y = rho * np.sin(phi)

    sheet_thickness = d_0 + delta_x * np.exp(x / scale_x_d) + delta_y * (y) ** 2
    #print('sheet_thickness: ', sheet_thickness)
    #sheet_thickness = 0.0
    #sheet_thickness = 0.3
    
    
    y_axis = H_current * special.j1(lambda_result * rho) * np.exp(
        -lambda_result * np.sqrt(z ** 2 + sheet_thickness ** 2))

    result_a_phi_hankel = simps(y_axis, x=lambda_result)

    return result_a_phi_hankel
