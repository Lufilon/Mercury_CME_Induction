# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 18:09:59 2023

@author: Luis-
"""

# own packages
from legendre_polynomials import P_dP

# third party packages
import matplotlib.pyplot as plt


def orbit(amp_riki_h, amp_riki_l, phase_riki_h, phase_riki_l, induced_h_phase,
          induced_l_phase, gauss_list_ext, theta_arr, R_M=2440):

    Br_400_h = zeros((len(gauss_list_ext), len(theta_arr)))
    Bt_400_h = zeros((len(gauss_list_ext), len(theta_arr)))
    B_400_h = zeros((len(gauss_list_ext), len(theta_arr)))
    B_r_400_l = zeros((len(gauss_list_ext), len(theta_arr)))
    B_theta_400_l = zeros((len(gauss_list_ext), len(theta_arr)))
    B_400_l = zeros((len(gauss_list_ext), len(theta_arr)))
    
    phase_riki_h_temp = phase_riki_h.copy()
    phase_riki_l_temp = phase_riki_l.copy()

    amp_riki_h_temp = amp_riki_h.copy()
    amp_riki_l_temp = amp_riki_l.copy()

    induced_h_phase0 = induced_h.copy()
    induced_l_phase0 = induced_l.copy()
    
    P_lm, dP_lm = P_dP(l, m, cos(theta_arr))
    dP_lm[0] = nan  # fragment caused by legendre polynomial
    
    return 0