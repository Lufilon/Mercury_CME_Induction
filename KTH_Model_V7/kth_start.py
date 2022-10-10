# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 10:34:04 2022

@author: Luis-
"""

from numpy import pi, sin, cos, hypot, float64, array, full
from kth14_model_for_mercury_v7 import kth14_model_for_mercury_v7
import json
import matplotlib.pyplot as plt


def kth_start(r_hel, settings):
    """
    An auxilliary function to calculate the Schmidt quasi-normalized
    associated Legendre-polynomials of
    degree l and order m for the variable x
    Parameters
    ----------
    r_hel : float_array
        
    settings : string_array
        Settings used for calculating the magnetic field with the kth-model
        dipole:
            
        neutralsheet:
            
        prc:
            
        internal:
            
        external:
            
    Returns
    -------
    Magnetic field components B_r, B_theta, B_phi
    """
    # =========================================================================
    # set basic parameters and import angle_data
    # =========================================================================
    R_M = 2440
    R_M = R_M * (1 + 1E-6)  # factor needed cause of rounding in model_field_v7
    
    with open('data/angle_data.json') as f:
        angle = json.load(f)

    N_THETA, N_PHI = angle["N_THETA"], angle["N_PHI"]
    num_pts = angle["num_pts"]
    theta, phi = array(angle["theta"]), array(angle["phi"])
    theta_arr, phi_arr = array(angle["theta_arr"]), array(angle["phi_arr"])

    di_val = float64(50)
    di = full(num_pts, di_val)

    # =========================================================================
    # create the data points in msm coordinates
    # =========================================================================
    x = R_M * sin(theta) * cos(phi)
    y = R_M * sin(theta) * sin(phi)
    z = R_M * cos(theta)

    # =========================================================================
    # calculating the magnetic field components
    # =========================================================================
    B_x, B_y, B_z = kth14_model_for_mercury_v7(
        x, y, z, r_hel, di,
        settings[0], settings[1], settings[2], settings[3], settings[4]
        )

    # =====================================================================
    # transform into spherical coordinate base
    # =====================================================================
    B_r = sin(theta) * cos(phi) * B_x
    B_r += sin(theta) * sin(phi) * B_y
    B_r += cos(theta) * B_z
    B_theta = cos(theta) * cos(phi) * B_x
    B_theta += cos(theta) * sin(phi) * B_y
    B_theta += - sin(theta) * B_z
    B_phi = - sin(phi) * B_x
    B_phi += cos(phi) * B_y

    # =============================================================================
    # return the magnetic field components
    # =============================================================================
    return B_r, B_theta, B_phi