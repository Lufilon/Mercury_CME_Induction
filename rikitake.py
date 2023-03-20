# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 15:31:16 2022

@author: Luis-
"""

from signal_processing import gaussian_f_to_t
from plotting import plot_savefig
import matplotlib.pyplot as plt
from numpy import pi, exp, sqrt, real, imag, log10
from numpy import savetxt, loadtxt, zeros, arange, array, arctan2, hypot
import mpmath
mpmath.mp.dps = 6


def rikitake_get(t, freq, coeff_ext_f_amp, coeff_ext_f_phase, rel_indices,
                 r_arr, sigma_h, sigma_l, t_steps, freqnr, resolution,
                 gauss_list_ext, path):
    # Calculate the frequency dependant gauss coefficients of the secondary field
    # using the rikitake factor and based on a model of the planets interior.

    try:
        # load precalculated data from files if available
        riki_h_re = loadtxt(path + str(resolution) + '_freqnr=' \
                                + str(freqnr) + '_h_re.gz')
        riki_h_im = loadtxt(path + str(resolution) + '_freqnr=' \
                                + str(freqnr) + '_h_im.gz')
        riki_l_re = loadtxt(path + str(resolution) + '_freqnr=' \
                                + str(freqnr) + '_l_re.gz')
        riki_l_im = loadtxt(path + str(resolution) + '_freqnr=' \
                                + str(freqnr) + '_l_im.gz')

        riki_h_re = riki_h_re.reshape((len(gauss_list_ext), freqnr))
        riki_h_im = riki_h_im.reshape((len(gauss_list_ext), freqnr))
        riki_l_re = riki_l_re.reshape((len(gauss_list_ext), freqnr))
        riki_l_im = riki_l_im.reshape((len(gauss_list_ext), freqnr))

        print("Finished importing the real and imag parts of the rikitake " +
              "factor for both conductivity profiles.")

    except OSError:
        print("No rikitake calculation for this combination of resolution " +
              "and freqnr was done yet - Starting the calculation.")
        # calculate the rikitake factor
        riki_h_re = zeros((len(gauss_list_ext), t_steps//2 + 1))
        riki_h_im = zeros((len(gauss_list_ext), t_steps//2 + 1))
        riki_l_re = zeros((len(gauss_list_ext), t_steps//2 + 1))
        riki_l_im = zeros((len(gauss_list_ext), t_steps//2 + 1))

        for l, m in gauss_list_ext:
            index = gauss_list_ext.index((l, m))

            for i in range(t_steps//2 + 1):
                if i in rel_indices and i > 0:
                    result = rikitake_calc(l, freq[index][i], r_arr, sigma_h,
                                           sigma_l)

                    riki_h_re[index][i] = result[0]
                    riki_h_im[index][i] = result[1]
                    riki_l_re[index][i] = result[2]
                    riki_l_im[index][i] = result[3]

        print("Finished calculating the rikitake factor parts.")

        # save for runtime purposes on mulitple runs
        rikitake_save(resolution, freqnr, riki_h_re, riki_h_im,
                      riki_l_re, riki_l_im, path)

    amp_riki_h = hypot(riki_h_re, riki_h_im)
    amp_riki_l = hypot(riki_l_re, riki_l_im)

    phase_riki_h = arctan2(riki_h_im, riki_h_re)
    phase_riki_l = arctan2(riki_l_im, riki_l_re)

    coeff_ext_sec_f_h = coeff_ext_f_amp * amp_riki_h \
        * exp(0+1j * phase_riki_h)
    coeff_ext_sec_f_l = coeff_ext_f_amp * amp_riki_l \
        * exp(0+1j * phase_riki_l)

    induced_h = zeros((len(gauss_list_ext), t_steps))
    induced_l = zeros((len(gauss_list_ext), t_steps))

    for l, m in gauss_list_ext:
        index = gauss_list_ext.index((l, m))

        induced_h[index] = gaussian_f_to_t(
            t, freq[index], coeff_ext_sec_f_h[index],
            coeff_ext_f_phase[index])
        induced_l[index] = gaussian_f_to_t(
            t, freq[index], coeff_ext_sec_f_l[index],
            coeff_ext_f_phase[index])

    print("Finished performing the inverse FFT of the rikitake modified " +
          "freq-dependant Gauss coefficients.")

    return coeff_ext_sec_f_h, coeff_ext_sec_f_l, amp_riki_h, amp_riki_l, phase_riki_h, phase_riki_l, induced_h, induced_l


def rikitake_calc(l, freq, r_arr, sigma_h, sigma_l):
    """
    Calculated the complexe rikitake factor for a given range of frequencies, 
    degree of magnetic field and conductivity model.

    Parameters
    ----------
    l : int
        Magnetic field degree.
    freq : numpy.ndarray.float64
        Frequencies that the rikitake factor is evaluated at.
    r_arr : numpy.ndarray.float64
        The radii of the planetary layers.
    sigma_h : numpy.ndarray.float64
        The low conductivity profile of the planetary layers.
    sigma_l : numpy.ndarray.float64
        The high conductivity profile of the planetary layers.

    Returns
    -------
    numpy.ndarray.float64
        Real and imaginary part of the rikitake factor for both cond. profiles.

    """
    omega = 2 * pi * freq

    k_arr_h = sqrt((0-1j * omega * 4E-7 * pi * sigma_h))
    k_arr_l = sqrt((0-1j * omega * 4E-7 * pi * sigma_l))

    riki_h = rikitake(l, k_arr_h, r_arr)
    riki_l = rikitake(l, k_arr_l, r_arr)

    return real(riki_h), imag(riki_h), real(riki_l), imag(riki_l)


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


def rikitake_save(resolution, freqnr, high_real, high_imag, low_real, low_imag,
                  path):
    """
    Save the rikitakefactor for a high and low conductivity profile for a given
    resolution and number of frequencies from the fft.

    Parameters
    ----------
    resolution : int
        Number of distances for which the magnetic field is calculated for.
    freqnr : int
        Number of frequencies returned.
        Can be used to determine influence of sampled number of frequencies.
    high_real : numpy.ndarray.float64
        Real part of the rikitake factor for the high conductivity model.
    high_imag : numpy.ndarray.float64
        Imaginary part of the rikitake factor for the high conductivity model.
    low_real : numpy.ndarray.float64
        Real part of the rikitake factor for the low conductivity model.
    low_imag : numpy.ndarray.float64
        Imaginary part of the rikitake factor for the low conductivity model.
    path : string
        Path to the directory where the values are stored.

    Returns
    -------
    None.

    """
    savetxt(path + str(resolution) + '_freqnr=' + str(freqnr) \
            + '_h_re.gz', high_real.ravel())
    savetxt(path + str(resolution) + '_freqnr=' + str(freqnr) \
            + '_h_im.gz', high_imag.ravel())
    savetxt(path + str(resolution) + '_freqnr=' + str(freqnr) \
            + '_l_re.gz', low_real.ravel())
    savetxt(path + str(resolution) + '_freqnr=' + str(freqnr) \
            + '_l_im.gz', low_imag.ravel())

    print("Finished saving the rikitake factor parts to file.")


def rikitake_transferfunction(l, known_excitements=False, spec_freq=False):
    """
    Plots the amplitude of the rikitakefactor for a wide frequency range.

    Parameters
    ----------
    l : int
        Order of the magnetic field.
    known_excitements : boolean, optional
        Decides if some known excitements are added as vlines to the plot.
        The default is False.
    spec_freq : boolean, optional
        Decides if the specific frequencies are added as vlnies to the plot.
        The default is False.

    Returns
    -------
    None.

    """
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
    riki_h = array([0+0j] * omega_steps)  # high conductivity
    riki_l = array([0+0j] * omega_steps)  # low conductivity

    for i in range(0, len(omega)):
        k_arr_h = sqrt((0-1j * omega[i] * 4E-7 * pi * sigma_h))
        k_arr_l = sqrt((0-1j * omega[i] * 4E-7 * pi * sigma_l))

        riki_h[i] = rikitake(l, k_arr_h, r_arr)
        riki_l[i] = rikitake(l, k_arr_l, r_arr)

    # creation of the transfer-function
    fig, ax = plt.subplots()
    ax.set_title("Transfer function for the amplitude of the rikitake factor")

    ax.plot(omega/(2*pi), abs(riki_h),
             label="$\\sigma_{high}$, $l=" + str(l) + "$", linewidth='2')
    ax.plot(omega/(2*pi), abs(riki_l),
             label="$\\sigma_{low}$,  $l=" + str(l) + "$", linewidth='2')

    ax.grid(which='major', axis='both', linestyle='-', color='lavender')
    ax.set_xlabel('$f$ [$Hz$]')
    ax.set_ylabel('$|\\mathcal{R}|$')
    ax.set_xscale('log')
    ax.set_xlim(fmin, fmax)
    ax.set_ylim(-0.035, l/(l+1) + 0.05)
    ax.legend(loc='upper left')

    if known_excitements:
        # Liljeblad and Karlsson (2017)
        # KH-Oscillations
        f30mHz = 30E-3
        ax.vlines(f30mHz, 0, l/(l+1), colors='forestgreen',linestyle='dotted')
        ax.annotate('30mHz', (f30mHz*0.8, -0.03), color='forestgreen')

        # Dungey-cycle
        f2min = 1/(2*60)
        ax.vlines(f2min, 0, l/(l+1), colors='firebrick', linestyle='dotted')
        ax.annotate('2min', (f2min*0.1, -0.03), color='firebrick')

        # solar rotation
        f642h = 1/(642*3600)
        ax.vlines(f642h, 0, l/(l+1), colors='darkorchid', linestyle='dotted')
        ax.annotate('642h', (f642h*2.2, -0.03), color='darkorchid')

        # planetary rotation
        f44 = 1/(44*24*3600)
        ax.vlines(f44, 0, l/(l+1), colors='sky#1f77b4', linestyle='dotted')
        ax.annotate('44d', (f44*0.3, -0.03), color='sky#1f77b4')

        f88 = 1/(88*24*3600)
        ax.vlines(f88, 0, l/(l+1), colors='black', linestyle='dotted')
        ax.annotate('88d', (f88*0.05, -0.03), color='black')

        # solar cicle
        f22y = 1/(22*365*24*3600)
        ax.vlines(f22y, 0, l/(l+1), colors='goldenrod', linestyle='dotted')
        ax.annotate('22y', (f22y*0.4, -0.03), color='goldenrod')

    if spec_freq:
        # 4 for high profile in blue
        ax.vlines(1.5 * pow(10, -15), -0.1, l/(l+1), colors='#1f77b4',
                   linestyle='dotted')
        ax.annotate('$f_A$', (1.5 * pow(10, -15), -0.03), color='#1f77b4',
                     size=12)
        ax.vlines(pow(10, -9), -0.1, l/(l+1), colors='#1f77b4',
                   linestyle='dotted')
        ax.annotate('$f_B$', (pow(10, -9), -0.03), color='#1f77b4',
                     size=12)
        ax.vlines(2 * pow(10, -8), -0.1, l/(l+1), colors='#1f77b4',
                   linestyle='dotted')
        ax.annotate('$f_{C/D}$', (2 * pow(10, -8), -0.03), color='#1f77b4',
                     size=12)
        ax.vlines(pow(10, -4), -0.1, l/(l+1), colors='#1f77b4',
                   linestyle='dotted')
        ax.annotate('$f_E$', (pow(10, -4), -0.03), color='#1f77b4',
                     size=12)
        # 5 for low profile in orange
        ax.vlines(5 * pow(10, -14), -0.1, l/(l+1), colors='#ff7f0e',
                   linestyle='dotted')
        ax.annotate('$f_A$', (5 * pow(10, -14), -0.03), color='#ff7f0e',
                     size=12)
        ax.vlines(5 * pow(10, -9), -0.1, l/(l+1), colors='#ff7f0e',
                   linestyle='dotted')
        ax.annotate('$f_B$', (5 * pow(10, -9), -0.03), color='#ff7f0e',
                     size=12)
        ax.vlines(pow(10, -5), -0.1, l/(l+1), colors='#ff7f0e',
                   linestyle='dotted')
        ax.annotate('$f_C$', (pow(10, -5), -0.03), color='#ff7f0e',
                     size=12)
        ax.vlines(5 * pow(10, -4), -0.1, l/(l+1), colors='#ff7f0e',
                   linestyle='dotted')
        ax.annotate('$f_D$', (5 * pow(10, -4), -0.03), color='#ff7f0e',
                     size=12)
        ax.vlines(pow(10, 1), -0.1, l/(l+1), colors='#ff7f0e',
                   linestyle='dotted')
        ax.annotate('$f_E$', (pow(10, 1), -0.03), color='#ff7f0e',
                     size=12)

    plot_savefig(fig, 'plots/',
                 'transfer_function_l=' + str(l) + '.jpeg', dpi=600)

    return fig, ax


def rikitake_plot(fig, ax, l, freq, amp_riki_h, amp_riki_l, amp):
    ax.set_title("Transfer function with alpha plot of simulated data")

    # add alpha plot of calculated data
    ax.scatter(freq, amp_riki_h, alpha=amp/max(amp), color='red', s=20)
    ax.scatter(freq, amp_riki_l, alpha=amp/max(amp), color='green', s=20)

    ax.legend(loc='upper left')# muss die noch?

    plot_savefig(fig, 'plots/',
                 'transfer_function_l=' + str(l) + '_alpha.jpeg', dpi=600)

    return ax
