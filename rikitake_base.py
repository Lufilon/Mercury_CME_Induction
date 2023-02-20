# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 15:31:16 2022

@author: Luis-
"""

import matplotlib.pyplot as plt
from rikitake import rikitake
from numpy import pi, sqrt, real, imag, log10, arange, array


"""
Documentation
returns:
    low and high rikitake factor. (could be interesting to view g_ind(t) in both)
"""


def rikitake_calc(l, f, r_arr, sigma_arr_h, sigma_arr_l):
    omega = 2 * pi * f

    k_arr_h = sqrt((0-1j * omega * 4E-7 * pi * sigma_arr_h))
    k_arr_l = sqrt((0-1j * omega * 4E-7 * pi * sigma_arr_l))

    rikitake_h = rikitake(l, k_arr_h, r_arr)
    rikitake_l = rikitake(l, k_arr_l, r_arr)

    return real(rikitake_h), imag(rikitake_h), real(rikitake_l), imag(rikitake_l)


def rikitake_plot(l, f, riki_h, riki_l, amp, color1, color2):
    # creation of the frequencies
    fmin, fmax = 1E-15, 1E5
    omega_min, omega_max = 2. * pi * fmin, 2. * pi * fmax
    omega_steps = 100
    d_omega = (log10(omega_max) - log10(omega_min))/omega_steps
    omega = pow(10, arange(log10(omega_min), log10(omega_max), d_omega))

    # specifiy mercuries layers - low and high conductivity cases
    r_arr = array([0, 1740E3, 1940E3, 2040E3, 2300E3, 2440E3])
    sigma_arr_l = array([0, 1E5, 1E2, 10**-0.5, 10**-3, 1E-7])
    sigma_arr_h = array([0, 1E7, 1E3, 10**0.5, 10**0.7, 1E-2])

    # calculation of the rikitkae factor
    rikitake_h = array([0+0j] * omega_steps)  # high conductivity
    rikitake_l = array([0+0j] * omega_steps)  # low conductivity

    for i in range(0, len(omega)):

        k_arr_h = sqrt((0-1j * omega[i] * 4E-7 * pi * sigma_arr_h))
        k_arr_l = sqrt((0-1j * omega[i] * 4E-7 * pi * sigma_arr_l))

        rikitake_h[i] = rikitake(l, k_arr_h, r_arr)
        rikitake_l[i] = rikitake(l, k_arr_l, r_arr)

    # creation of the transfer-function
    plt.figure("Transfer function")
    plt.title("Transfer function with alpha plot of simulated data")

    # plt.plot(omega/(2*pi), abs(rikitake_h), c=color1,
    #           label="$\\sigma_{high}$, $l=" + str(l) + "$", linewidth='2') #1f77b4
    # plt.plot(omega/(2*pi), abs(rikitake_l), c=color2,
    #          label="$\\sigma_{low}$,  $l=" + str(l) + "$", linewidth='2') #ff7f0e
    plt.plot(omega/(2*pi), abs(rikitake_h),
              label="$\\sigma_{high}$, $l=" + str(l) + "$", linewidth='2')
    plt.plot(omega/(2*pi), abs(rikitake_l),
             label="$\\sigma_{low}$,  $l=" + str(l) + "$", linewidth='2')

    # # add alpha plot of calculated data
    plt.scatter(f, riki_h, alpha=amp/max(amp), color='red', s=20)
    plt.scatter(f, riki_l, alpha=amp/max(amp), color='green', s=20)

    plt.grid(which='major', axis='both', linestyle='-', color='lavender')
    plt.xlabel('$f$ [$Hz$]')
    plt.ylabel('$|\\mathcal{R}|$')
    plt.xscale('log')
    plt.xlim(fmin, fmax)
    # plt.xlim(pow(10, -6), pow(10, -2))
    plt.ylim(-0.035, l/(l+1) + 0.05)
    # plt.ylim(0.2, 0.6)
    plt.legend(loc='upper left')

    # known excitements
    # Liljeblad and Karlsson (2017)
    # KH-Oscillations
    f30mHz = 30E-3
    plt.vlines(f30mHz, 0, l/(l+1), colors='forestgreen', linestyle='dotted')
    plt.annotate('30mHz', (f30mHz*0.8, -0.03), color='forestgreen')

    # Dungey-cycle
    f2min = 1/(2*60)
    plt.vlines(f2min, 0, l/(l+1), colors='firebrick', linestyle='dotted')
    plt.annotate('2min', (f2min*0.1, -0.03), color='firebrick')

    # solar rotation
    f642h = 1/(642*3600)
    plt.vlines(f642h, 0, l/(l+1), colors='darkorchid', linestyle='dotted')
    plt.annotate('642h', (f642h*2.2, -0.03), color='darkorchid')

    # planetary rotation
    f44 = 1/(44*24*3600)
    plt.vlines(f44, 0, l/(l+1), colors='#1f77b4', linestyle='dotted')
    plt.annotate('44d', (f44*0.3, -0.03), color='#1f77b4')

    f88 = 1/(88*24*3600)
    plt.vlines(f88, 0, l/(l+1), colors='black', linestyle='dotted')
    plt.annotate('88d', (f88*0.05, -0.03), color='black')

    # solar cicle
    f22y = 1/(22*365*24*3600)
    plt.vlines(f22y, 0, l/(l+1), colors='goldenrod', linestyle='dotted')
    plt.annotate('22y', (f22y*0.4, -0.03), color='goldenrod')

    plt.savefig('plots/alpha_plot_l=' + str(l) + '.jpg', dpi=600)


def transferfunction(l):
    # creation of the frequencies
    fmin, fmax = 1E-15, 1E5
    omega_min, omega_max = 2. * pi * fmin, 2. * pi * fmax
    omega_steps = 100
    d_omega = (log10(omega_max) - log10(omega_min))/omega_steps
    omega = pow(10, arange(log10(omega_min), log10(omega_max), d_omega))

    # specifiy mercuries layers - low and high conductivity cases
    r_arr = array([0, 1740E3, 1940E3, 2040E3, 2300E3, 2440E3])
    sigma_arr_l = array([0, 1E5, 1E2, 10**-0.5, 10**-3, 1E-7])
    sigma_arr_h = array([0, 1E7, 1E3, 10**0.5, 10**0.7, 1E-2])

    # calculation of the rikitkae factor
    rikitake_h = array([0+0j] * omega_steps)  # high conductivity
    rikitake_l = array([0+0j] * omega_steps)  # low conductivity

    for i in range(0, len(omega)):

        k_arr_h = sqrt((0-1j * omega[i] * 4E-7 * pi * sigma_arr_h))
        k_arr_l = sqrt((0-1j * omega[i] * 4E-7 * pi * sigma_arr_l))

        rikitake_h[i] = rikitake(l, k_arr_h, r_arr)
        rikitake_l[i] = rikitake(l, k_arr_l, r_arr)

    # creation of the transfer-function
    plt.figure("Transfer function")
    plt.title("Transfer function")

    plt.plot(omega/(2*pi), abs(rikitake_h),
             label="$\\sigma_{high}$, $l=" + str(l) + "$", linewidth='2')
    plt.plot(omega/(2*pi), abs(rikitake_l),
             label="$\\sigma_{low}$,  $l=" + str(l) + "$", linewidth='2')

    plt.grid(which='major', axis='both', linestyle='-', color='lavender')
    plt.xlabel('$f$ [$Hz$]')
    plt.ylabel('$|\\mathcal{R}|$')
    plt.xscale('log')
    plt.xlim(fmin, fmax)
    plt.ylim(-0.035, l/(l+1) + 0.05)
    plt.legend(loc='upper left')

# =============================================================================
#     # # known excitements
#     # Liljeblad and Karlsson (2017)
#     # KH-Oscillations
#     f30mHz = 30E-3
#     # plt.vlines(f30mHz, 0, l/(l+1), colors='forestgreen', linestyle='dotted')
#     plt.annotate('30mHz', (f30mHz*0.8, -0.03), color='forestgreen')
# 
#     # Dungey-cycle
#     f2min = 1/(2*60)
#     plt.vlines(f2min, 0, l/(l+1), colors='firebrick', linestyle='dotted')
#     plt.annotate('2min', (f2min*0.1, -0.03), color='firebrick')
# 
#     # solar rotation
#     f642h = 1/(642*3600)
#     plt.vlines(f642h, 0, l/(l+1), colors='darkorchid', linestyle='dotted')
#     plt.annotate('642h', (f642h*2.2, -0.03), color='darkorchid')
# 
#     # planetary rotation
#     f44 = 1/(44*24*3600)
#     plt.vlines(f44, 0, l/(l+1), colors='sky#1f77b4', linestyle='dotted')
#     plt.annotate('44d', (f44*0.3, -0.03), color='sky#1f77b4')
# 
#     f88 = 1/(88*24*3600)
#     plt.vlines(f88, 0, l/(l+1), colors='black', linestyle='dotted')
#     plt.annotate('88d', (f88*0.05, -0.03), color='black')
# 
#     # solar cicle
#     f22y = 1/(22*365*24*3600)
#     plt.vlines(f22y, 0, l/(l+1), colors='goldenrod', linestyle='dotted')
#     plt.annotate('22y', (f22y*0.4, -0.03), color='goldenrod')
# =============================================================================

    # # specific excitations for the different layers of the conductivity model

    # 4 for high profile in blue
    plt.vlines(1.5 * pow(10, -15), -0.1, l/(l+1), colors='#1f77b4', linestyle='dotted')
    plt.annotate('$f_A$', (1.5 * pow(10, -15), -0.03), color='#1f77b4', size=12)
    plt.vlines(pow(10, -9), -0.1, l/(l+1), colors='#1f77b4', linestyle='dotted')
    plt.annotate('$f_B$', (pow(10, -9), -0.03), color='#1f77b4', size=12)
    plt.vlines(2 * pow(10, -8), -0.1, l/(l+1), colors='#1f77b4', linestyle='dotted')
    plt.annotate('$f_{C/D}$', (2 * pow(10, -8), -0.03), color='#1f77b4', size=12)
    plt.vlines(pow(10, -4), -0.1, l/(l+1), colors='#1f77b4', linestyle='dotted')
    plt.annotate('$f_E$', (pow(10, -4), -0.03), color='#1f77b4', size=12)
    # 5 for low profile in orange
    plt.vlines(5 * pow(10, -14), -0.1, l/(l+1), colors='#ff7f0e', linestyle='dotted')
    plt.annotate('$f_A$', (5 * pow(10, -14), -0.03), color='#ff7f0e', size=12)
    plt.vlines(5 * pow(10, -9), -0.1, l/(l+1), colors='#ff7f0e', linestyle='dotted')
    plt.annotate('$f_B$', (5 * pow(10, -9), -0.03), color='#ff7f0e', size=12)
    plt.vlines(pow(10, -5), -0.1, l/(l+1), colors='#ff7f0e', linestyle='dotted')
    plt.annotate('$f_C$', (pow(10, -5), -0.03), color='#ff7f0e', size=12)
    plt.vlines(5 * pow(10, -4), -0.1, l/(l+1), colors='#ff7f0e', linestyle='dotted')
    plt.annotate('$f_D$', (5 * pow(10, -4), -0.03), color='#ff7f0e', size=12)
    plt.vlines(pow(10, 1), -0.1, l/(l+1), colors='#ff7f0e', linestyle='dotted')
    plt.annotate('$f_E$', (pow(10, 1), -0.03), color='#ff7f0e', size=12)

    plt.tight_layout()

    plt.savefig('plots/transfer_function_l=' + str(l) + '.jpg', dpi=600)
