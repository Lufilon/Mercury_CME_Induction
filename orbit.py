# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 18:09:59 2023

@author: Luis-
"""

# own packages
from legendre_polynomials import P_dP
from plotting import plot_savefig

# third party packages
from numpy import hypot
import matplotlib.pyplot as plt


def orbit(amp_riki_h, amp_riki_l, phase_riki_h, phase_riki_l, induced_h_phase,
          induced_l_phase, gauss_list_ext, theta_arr, R_M=2440):

    Br = zeros((len(gauss_list_ext), len(theta_arr)))
    Bt = zeros((len(gauss_list_ext), len(theta_arr)))
    B_h = zeros((len(gauss_list_ext), len(theta_arr)))
    Br_l = zeros((len(gauss_list_ext), len(theta_arr)))
    Bt_l = zeros((len(gauss_list_ext), len(theta_arr)))
    B_l = zeros((len(gauss_list_ext), len(theta_arr)))
    
    phase_riki_h_temp = phase_riki_h.copy()
    phase_riki_l_temp = phase_riki_l.copy()

    amp_riki_h_temp = amp_riki_h.copy()
    amp_riki_l_temp = amp_riki_l.copy()

    induced_h_phase0 = induced_h.copy()
    induced_l_phase0 = induced_l.copy()
    
    P_lm, dP_lm = P_dP(l, m, cos(theta_arr))
    dP_lm[0] = nan  # fragment caused by legendre polynomial
    
    return 0


def orbit_plot(gauss_list_ext, theta_arr, Br_h, Br_l, Bt_h, Bt_l, title,
               xlabel, ylabel_r, ylabel_t, ylabel_tot, name):
    fig, ax = plt.subplots(3, sharex=True)
    plt.subplots_adjust(hspace=0)
    ax[0].set_title(title)

    B_h = hypot(Br_h, Bt_h)
    B_l = hypot(Br_l, Bt_l)

    for l, m in gauss_list_ext:
        index = gauss_list_ext.index((l, m))

        label = "$g_{" + str(l) + str(m) + "}$, $\\sigma_h$"

        ax[0].plot(theta_arr, Br_h[index], label)
        ax[0].plot(theta_arr, Br_l[index], label)
        ax[1].plot(theta_arr, Bt_l[index], label)
        ax[1].plot(theta_arr, Bt_h[index], label)
        ax[2].plot(theta_arr, B_h[index], label)
        ax[2].plot(theta_arr, B_l[index], label)

    ax[2].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel_r)
    ax[1].set_ylabel(ylabel_t)
    ax[2].set_ylabel(ylabel_tot)
    ax[0].legend(fontsize='small')
    ax[1].legend(fontsize='small')
    ax[2].legend(fontsize='small')

    plot_savefig(fig, 'plots/', name, dpi=600)
    
    return fig, ax
