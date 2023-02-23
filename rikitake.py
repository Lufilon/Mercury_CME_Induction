# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 15:31:16 2022

@author: Luis-
"""

from signal_processing import rebuild
import matplotlib.pyplot as plt
from numpy import pi, exp, sqrt, real, imag, log10
from numpy import savetxt, loadtxt, zeros, arange, array, arctan2, hypot
import mpmath
mpmath.mp.dps = 6

"""
TODO: Hier weiter aufräumen
"""
def rikitake_get(t, freq, coeff_ext_f_amp, coeff_ext_f_phase, rel_indices,
                 t_steps, freqnr, gauss_list_ext, r_arr, sigma_h, sigma_l,
                 path='data/helios_1/ns=True/rikitake/res=100_freqnr=3601'):
    # calculation of rikitake-factor for each selected frequency for both
    # high and low condutivity model.
    try:
        # load precalculated data from files if available
        rikitake_h_re = loadtxt(path + '_h_re.gz')
        rikitake_h_im = loadtxt(path + '_h_im.gz')
        rikitake_l_re = loadtxt(path + '_l_re.gz')
        rikitake_l_im = loadtxt(path + '_l_im.gz')

        rikitake_h_re = rikitake_h_re.reshape((len(gauss_list_ext), freqnr))
        rikitake_h_im = rikitake_h_im.reshape((len(gauss_list_ext), freqnr))
        rikitake_l_re = rikitake_l_re.reshape((len(gauss_list_ext), freqnr))
        rikitake_l_im = rikitake_l_im.reshape((len(gauss_list_ext), freqnr))

        print("Finished importing the real and imag parts of the rikitake " +
              "factor for the each conductivity profil.")

    except OSError:
        print("No rikitake calculation for this combination of resolution" +
              "and freqnr was done yet - Starting the calculation.")
        # calculate the rikitake factor
        rikitake_h_re = zeros((len(gauss_list_ext), t_steps//2 + 1))
        rikitake_h_im = zeros((len(gauss_list_ext), t_steps//2 + 1))
        rikitake_l_re = zeros((len(gauss_list_ext), t_steps//2 + 1))
        rikitake_l_im = zeros((len(gauss_list_ext), t_steps//2 + 1))

        for l, m in gauss_list_ext:
            index = gauss_list_ext.index((l, m))

            for i in range(t_steps//2 + 1):
                if i in rel_indices and i > 0:
                    rikitake_h_re[index][i],
                    rikitake_h_im[index][i],
                    rikitake_l_re[index][i],
                    rikitake_l_im[index][i] = rikitake_calc(
                        l, freq[index][i], r_arr, sigma_h, sigma_l)

        print("Finished calculating the rikitake factor parts.")

        # save for runtime purposes on mulitple runs
        rikitake_save(
            rikitake_h_re, rikitake_h_im, rikitake_l_re, rikitake_l_im, path
            )

    amp_rikitake_h = zeros((len(gauss_list_ext), t_steps//2 + 1),
                           dtype=complex)
    amp_rikitake_l = zeros((len(gauss_list_ext), t_steps//2 + 1),
                           dtype=complex)
    phase_rikitake_h = zeros((len(gauss_list_ext), t_steps//2 + 1),
                             dtype=complex)
    phase_rikitake_l = zeros((len(gauss_list_ext), t_steps//2 + 1),
                             dtype=complex)
    induced_h = zeros((len(gauss_list_ext), t_steps))
    induced_l = zeros((len(gauss_list_ext), t_steps))

    for l, m in gauss_list_ext:
        index = gauss_list_ext.index((l, m))

        phase_rikitake_h[index] = arctan2(
            rikitake_h_im[index], rikitake_h_re[index])
        phase_rikitake_l[index] = arctan2(
            rikitake_l_im[index], rikitake_l_re[index])

        amp_rikitake_h[index] = coeff_ext_f_amp[index] * hypot(
            rikitake_h_re[index], rikitake_h_im[index]) * exp(
                0+1j * phase_rikitake_h[index])
        amp_rikitake_l[index] = coeff_ext_f_amp[index] * hypot(
            rikitake_l_re[index], rikitake_l_im[index]) * exp(
                0+1j * phase_rikitake_l[index])

        induced_h[index] = rebuild(
            t, freq[index], amp_rikitake_h[index], coeff_ext_f_phase[index])
        induced_l[index] = rebuild(
            t, freq[index], amp_rikitake_l[index], coeff_ext_f_phase[index])

    print("Finished performing the inverse FFT of the rikitake modified " +
          "freq-dependant Gauss coefficients.")

    return induced_h, induced_l


def rikitake_calc(l, freq, r_arr, sigma_h, sigma_l):
    omega = 2 * pi * freq

    k_arr_h = sqrt((0-1j * omega * 4E-7 * pi * sigma_h))
    k_arr_l = sqrt((0-1j * omega * 4E-7 * pi * sigma_l))

    rikitake_h = rikitake(l, k_arr_h, r_arr)
    rikitake_l = rikitake(l, k_arr_l, r_arr)

    return real(rikitake_h), imag(rikitake_h), real(rikitake_l), imag(rikitake_l)


def rikitake(l, k, r):
    """
    Function to calculate the Rikitake factor, the ratio of the gaussian-
    coefficients of the induced and inducing magnetic field.
    
    Parameters
    ----------
    l : int
        Degree of the magnetic field
    k : complex128
        Induction parameter
    r : float
        Input to evaluate the function at
    Returns
    -------
    Rikitake factor for the given parameters.
    
    """
    def dp__p(l, z):
        return dp_l__dz(l, z) / p_l(l, z)

    def dq__q(l, z):
        return dq_l__dz(l, z) / q_l(l, z)

    def q__p(l, z):
        return q_l(l, z) / p_l(l, z)

    def d__c(l, j, k, r):
        if j > 2:
            z1 = k[j] * r[j-1]
            z2 = k[j-1] * r[j-1]
            dp__p_save1 = k[j] * dp__p(l, z1)
            dp__p_save2 = k[j-1] * dp__p(l, z2)
            dq__q_save1 = k[j] * dq__q(l, z1)
            dq__q_save2 = k[j-1] * dq__q(l, z2)
            q__p_save1 = q__p(l, z1)
            q__p_save2 = q__p(l, z2)
            d__c_save = d__c(l, j-1, k, r)

            zaehler1 = dp__p_save1 - dp__p_save2
            zaehler2 = d__c_save * q__p_save2 * (dp__p_save1 - dq__q_save2)
            nenn1 = dp__p_save2 - dq__q_save1
            nenn2 = d__c_save * q__p_save2 * (dq__q_save2 - dq__q_save1)

            return 1 / q__p_save1 * (zaehler1 + zaehler2) / (nenn1 + nenn2)

        # end of recursion
        dp__p_res = k[1] * dp__p(l, k[1] * r[1])

        zaehler = k[2] * dp__p(l, k[2] * r[1]) - dp__p_res
        nenner = dp__p_res - k[2] * dq__q(l, k[2] * r[1])

        return 1 / (q__p(l, k[2] * r[1])) * zaehler / nenner

    def p_l(l, z):
        return sqrt(pi / (2*z)) * mpmath.besseli(l+1/2, z)

    def q_l(l, z):
        return sqrt(pi / (2*z)) * mpmath.besselk(l+1/2, z)

    def dp_l__dz(l, z):
        return p_l(l-1, z) - (l+1) / z * p_l(l, z)

    def dq_l__dz(l, z):
        return - q_l(l-1, z) - (l+1)/z * q_l(l, z)
        # return pow(-1, l) * q_l(l-1, z) - pow(-1, l+1) * (l+1)/z * q_l(l, z)

    jmax = len(k)-1

    dp__p_res = dp__p(l, k[jmax] * r[jmax])
    dq__q_res = dq__q(l, k[jmax] * r[jmax])
    q__p_res = q__p(l, k[jmax] * r[jmax])
    d__c_res = d__c(l, jmax, k, r)

    zaehler1 = -r[jmax] * k[jmax] * dp__p_res
    zaehler2 = d__c_res * q__p_res * (l - r[jmax] * k[jmax] * dq__q_res)
    nenner1 = r[jmax] * k[jmax] * dp__p_res
    nenner2 = d__c_res * q__p_res * (l+1 + r[jmax] * k[jmax] * dq__q_res)

    return -l / (l+1) * (zaehler1 + zaehler2 + l) / (nenner1 + nenner2 + l+1)


def rikitake_save(high_real, high_imag, low_real, low_imag,
                  path='data/helios_1/ns=True/rikitake/res=100_freqnr=3601'):
    """
    Save the rikitakefactor for a high and low conductivity profile for a given
    resolution and number of frequencies from the fft.

    Parameters
    ----------
    high_real : numpy.ndarray.float64
        Real part of the rikitake factor for the high conductivity model.
    high_imag : numpy.ndarray.float64
        Imaginary part of the rikitake factor for the high conductivity model.
    low_real : numpy.ndarray.float64
        Real part of the rikitake factor for the low conductivity model.
    low_imag : numpy.ndarray.float64
        Imaginary part of the rikitake factor for the low conductivity model.
    path : string, optional
        Path to the directory where the values are stored.
        The default is 'data/helios_1/ns=True/rikitake/res=100_freqnr=3601'.

    Returns
    -------
    None.

    """
    savetxt(path + '_h_re.gz', high_real.ravel())
    savetxt(path + '_h_im.gz', high_imag.ravel())
    savetxt(path + '_l_re.gz', low_real.ravel())
    savetxt(path + '_l_im.gz', low_imag.ravel())

    print("Finished saving the rikitake factor parts to file.")


def rikitake_plot(l, f, riki_h, riki_l, amp, color1, color2):
    # creation of the frequencies
    fmin, fmax = 1E-15, 1E5
    omega_min, omega_max = 2. * pi * fmin, 2. * pi * fmax
    omega_steps = 100
    d_omega = (log10(omega_max) - log10(omega_min))/omega_steps
    omega = pow(10, arange(log10(omega_min), log10(omega_max), d_omega))

    # specifiy mercuries layers - low and high conductivity cases
    r_arr = array([0, 1740E3, 1940E3, 2040E3, 2300E3, 2440E3])
    sigma_l = array([0, 1E5, 1E2, 10**-0.5, 10**-3, 1E-7])
    sigma_h = array([0, 1E7, 1E3, 10**0.5, 10**0.7, 1E-2])

    # calculation of the rikitkae factor
    rikitake_h = array([0+0j] * omega_steps)  # high conductivity
    rikitake_l = array([0+0j] * omega_steps)  # low conductivity

    for i in range(0, len(omega)):

        k_arr_h = sqrt((0-1j * omega[i] * 4E-7 * pi * sigma_h))
        k_arr_l = sqrt((0-1j * omega[i] * 4E-7 * pi * sigma_l))

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
    sigma_l = array([0, 1E5, 1E2, 10**-0.5, 10**-3, 1E-7])
    sigma_h = array([0, 1E7, 1E3, 10**0.5, 10**0.7, 1E-2])

    # calculation of the rikitkae factor
    rikitake_h = array([0+0j] * omega_steps)  # high conductivity
    rikitake_l = array([0+0j] * omega_steps)  # low conductivity

    for i in range(0, len(omega)):

        k_arr_h = sqrt((0-1j * omega[i] * 4E-7 * pi * sigma_h))
        k_arr_l = sqrt((0-1j * omega[i] * 4E-7 * pi * sigma_l))

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
