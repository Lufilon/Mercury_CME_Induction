# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 18:09:59 2023

@author: Luis-
"""

# own packages
from signal_processing import gaussian_f_to_t
from legendre_polynomials import P_dP
from plotting import plot_savefig

# third party packages
from numpy import zeros, hypot, sin, cos, nan, exp
import matplotlib.pyplot as plt


def orbit(t, freq, gauss_list_ext, theta_arr, coeff_ext_sec_f_h,
          coeff_ext_sec_f_l, coeff_ext_f_phase, phase_riki_h,
          phase_riki_l, induced_h, induced_l, height, R_M=2440, case="orbit"):
    orbit_height = R_M + height
    
    # create empy arrays for the magnetic field components and total field
    Br_h, Bt_h, B_h = zeros((3, len(gauss_list_ext), len(theta_arr)))
    Br_l, Bt_l, B_l = zeros((3, len(gauss_list_ext), len(theta_arr)))

    if case=="orbit":
        induced_h_temp = induced_h
        induced_l_temp = induced_l

    if case=="orbit_diff":
        # create new arrays to modify
        coeff_ext_sec_f_h_copy = coeff_ext_sec_f_h.copy()
        coeff_ext_sec_f_l_copy = coeff_ext_sec_f_l.copy()

        induced_h_copy = induced_h.copy()
        induced_l_copy = induced_l.copy()

        # get rid of the phase of the rikitakefactor for the lowest frequency
        coeff_ext_sec_f_h_copy[:, 1] = coeff_ext_sec_f_h[:, 1]/exp(0+1j * phase_riki_h[:, 1])
        coeff_ext_sec_f_l_copy[:, 1] = coeff_ext_sec_f_l[:, 1]/exp(0+1j * phase_riki_l[:, 1])

        # transform gaussians of induced field from freq to time domaine
        for l, m in gauss_list_ext:
            index = gauss_list_ext.index((l, m))

            induced_h_copy[index] = gaussian_f_to_t(
                t, freq[index], coeff_ext_sec_f_h_copy[index],
                coeff_ext_f_phase[index])
            induced_l_copy[index] = gaussian_f_to_t(
                t, freq[index], coeff_ext_sec_f_l_copy[index],
                coeff_ext_f_phase[index])

        induced_h_temp = induced_h - induced_h_copy
        # print(max(abs(induced_h_temp[0])))
        # print(max(abs(induced_h_temp[1])))
        induced_l_temp = induced_l - induced_l_copy

    elif case!="orbit":
        print("Not a valid case for the orbit plot. \n" + 
              "Valid cases at the moment are 'orbit' and 'orbit_diff'.")
        exit(0)
    
    Br_h, Bt_h, Br_l, Bt_l = orbit_calc(theta_arr, induced_h_temp, induced_l_temp,
                                        gauss_list_ext, R_M, orbit_height)

    fig, ax = orbit_plot(gauss_list_ext, theta_arr, Br_h, Bt_h, Br_l, Bt_l,
                          "Magnetic field for " + str(height) + " km orbit " +
                          "with $\\varphi = 0$", '$\\vartheta$ [$rad$]',
                          '$B_r$ [$nT$]', '$B_\\vartheta$ [$nT$]',
                          '$|B|$ [$nT$]', f"{height}km_orbit_{case}")

    return fig, ax, Br_h, Bt_h, Br_l, Bt_l


def orbit_calc(theta_arr, induced_h, induced_l,
                gauss_list_ext=[(1, 0), (2, 1)], R_M=2440, orbit_height=2840):
    Br_h, Bt_h, Br_l, Bt_l = zeros((4, len(gauss_list_ext), len(theta_arr)))
    
    for l, m in gauss_list_ext:
        index = gauss_list_ext.index((l, m))
        
        P_lm, dP_lm = P_dP(l, m, cos(theta_arr))
        dP_lm[0] = nan  # fragment caused by legendre polynomial
        dP_lm = dP_lm * (-sin(theta_arr))  # account for the inner derivative

        Br_h[index] = (l+1) * (R_M/(orbit_height))**(l+2) * \
            max(abs(induced_h[index])) * P_lm
        Bt_h[index] = - (R_M/(orbit_height))**(l+2) * \
            max(abs(induced_h[index])) * dP_lm
        Br_l[index] = (l+1) * (R_M/(orbit_height))**(l+2) * \
            max(abs(induced_l[index])) * P_lm
        Bt_l[index] = - (R_M/(orbit_height))**(l+2) * \
            max(abs(induced_l[index])) * dP_lm

    return Br_h, Bt_h, Br_l, Bt_l


def orbit_plot(gauss_list_ext, theta_arr, Br_h, Bt_h, Br_l, Bt_l, title,
               xlabel, ylabelr, ylabel_t, ylabel_tot, name):
    fig, ax = plt.subplots(3, sharex=True)
    plt.subplots_adjust(hspace=0)
    ax[0].set_title(title)

    B_h = hypot(Br_h, Bt_h)
    B_l = hypot(Br_l, Bt_l)

    for l, m in gauss_list_ext:
        index = gauss_list_ext.index((l, m))

        label = "$g_{" + str(l) + str(m) + "}$, $\\sigma_h$"

        ax[0].plot(theta_arr, Br_h[index], label=label)
        ax[0].plot(theta_arr, Br_l[index], label=label)
        ax[1].plot(theta_arr, Bt_l[index], label=label)
        ax[1].plot(theta_arr, Bt_h[index], label=label)
        ax[2].plot(theta_arr, B_h[index], label=label)
        ax[2].plot(theta_arr, B_l[index], label=label)
        
        # ax[0].plot(theta_arr, Br_h[index])
        # ax[0].plot(theta_arr, Br_l[index])
        # ax[1].plot(theta_arr, Bt_l[index])
        # ax[1].plot(theta_arr, Bt_h[index])
        # ax[2].plot(theta_arr, B_h[index])
        # ax[2].plot(theta_arr, B_l[index])

    ax[2].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabelr)
    ax[1].set_ylabel(ylabel_t)
    ax[2].set_ylabel(ylabel_tot)
    ax[0].legend(fontsize='small')
    ax[1].legend(fontsize='small')
    ax[2].legend(fontsize='small')

    plot_savefig(fig, 'plots/', name, dpi=600)
    
    return fig, ax
