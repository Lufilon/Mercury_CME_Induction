# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 16:57:50 2022

@author: Luis-
"""

from numpy.fft import rfft, rfftfreq
from numpy import pi, exp, real, angle, zeros, asarray, flip, isin
import matplotlib.pyplot as plt


def gaussian_t_to_f(coeff_ext_t, t, t_steps, gauss_list_ext=[(1, 0), (2, 1)],
                    freqnr=3601):
    """
    Fourier-transform the coefficients to the frequency domain

    Parameters
    ----------
    coeff_ext_t : numpy.ndarray.float64
        Time dependant external Gauss coefficients of the primary field.
    t : numpy.ndarray.int
        Time since start in seconds.
    t_steps : int
        Number of measurements.
    gauss_list_ext : list.tupel, optional
        List containing gauss coefficients to be analyzed.
        The default is [(1, 0), (2, 1)].
    freqnr : int, optional
        Number of frequencies returned.
        Can be used to determine influence of sampled number of frequencies.
        The default is 3601.

    Returns
    -------
    freq : numpy.ndarray.float64
        Frequency of the primary gauss coefficients.
    coeff_ext_f_amp : numpy.ndarray.float64
        Amplitude of the primary gauss coefficients.
    coeff_ext_f_phase : numpy.ndarray.float64
        Phase of the primary gauss coefficients.

    """
    freq = zeros((len(gauss_list_ext), t_steps//2 + 1))
    coeff_ext_f_amp = zeros((len(gauss_list_ext), t_steps//2 + 1))
    coeff_ext_f_phase = zeros((len(gauss_list_ext), t_steps//2 + 1))

    relIndices = zeros((len(gauss_list_ext), freqnr), dtype='int')

    for l, m in gauss_list_ext:
        index = gauss_list_ext.index((l, m))
        freq[index], coeff_ext_f_amp[index], coeff_ext_f_phase[index] = fft_own(
            t, t_steps, asarray([coeff_ext_t[i][m][l] for i in range(t_steps)]))

        # get relevant frequencies, filter f_0 = 0 Hz beforehand
        relIndices[index] = flip(coeff_ext_f_amp[index].argsort()[-freqnr:])
        mask = isin(coeff_ext_f_amp[index], coeff_ext_f_amp[index][relIndices[index]],
                    invert=True)
        coeff_ext_f_amp[index][mask] = 0

    print("Finished fourier-transforming the external Gauss coefficients.")
    
    return freq, coeff_ext_f_amp, coeff_ext_f_phase


def gaussian_f_to_t():
    """TODO"""
    return 0


def fft_own(t, data, N=3601):
    """
    Performs the fast fourier transform on a given set of data.

    Parameters
    ----------
    t : numpy.ndarray.int
        Time since start in seconds.
    data : numpy.ndarray.float64
        Time series of some data.
    N : int, optional
        Size of the input data. The default is 3601.

    Returns
    -------
    freq : numpy.ndarray.float64
        Frequency of the fourier transformed signal.
    amp : numpy.ndarray.float64
        Amplitude of the fourier transformed signal.
    phase : numpy.ndarray.float64
        Phase of the fourier transformed signal.

    """
    dt = t[1] - t[0]

    freq = rfftfreq(N, dt)
    data_fft = rfft(data, N)
    amp = abs(data_fft) / N
    for i in range(len(freq)):
        if i > 0:
            amp[i] = 2 * amp[i]
    phase = angle(data_fft)

    return freq, amp, phase


def rebuild(t, freq, amp, phase):
    """
    Transforms a given signal in the frequency domaine to the time domaine.
    Is equal to the inverse fourier transform but frequencies can be sampled.

    Parameters
    ----------
    t : numpy.ndarray.int
        Time since start in seconds.
    freq : numpy.ndarray.float64
        Frequency of the fourier transformed signal.
    amp : numpy.ndarray.float64
        Amplitude of the fourier transformed signal.
    phase : numpy.ndarray.float64
        Phase of the fourier transformed signal.

    Returns
    -------
    data_rebuild : numpy.ndarray.float64
        Fourier transform of the input signal in the time domaine.

    """
    # Rebuild the initial signal with a reduced amount of frequencies.
    data_rebuild = amp * exp(0+1j * (2 * pi * freq * t + phase))
    data_rebuild = sum(real(data_rebuild))

    return data_rebuild


def gaussian_f_plot(freq, coeff_ext_f_amp, gauss_list_ext):
    """
    

    Parameters
    ----------
    freq : TYPE
        DESCRIPTION.
    coeff_ext_f_amp : TYPE
        DESCRIPTION.
    gauss_list_ext : TYPE
        DESCRIPTION.

    Returns
    -------
    fig_gauss_f : TYPE
        DESCRIPTION.
    ax_gauss_f_pri : TYPE
        DESCRIPTION.

    """
    fig_gauss_f, ax_gauss_f_pri = plt.subplots(
        len(gauss_list_ext), sharex=True)
    plt.subplots_adjust(hspace=0)
    ax_gauss_f_pri[0].set_title(
        "Freq-dependant primary and secondary Gauss coefficients")

    for l, m in gauss_list_ext:
        index = gauss_list_ext.index((l, m))

        ax_gauss_f_pri[index].plot(
            freq[index][1:], coeff_ext_f_amp[index][1:],
            label="$g_{" + str(l) + str(m) + ", \\mathrm{pri}}$")

        ax_gauss_f_pri[index].set_xscale('log')
        ax_gauss_f_pri[index].set_ylabel("$A_\\mathrm{pri}$ $[nT]$")
        ax_gauss_f_pri[index].legend(loc='upper center')
        ax_gauss_f_pri[index].axvline(
            x=freq[index][1], color='goldenrod', linestyle='dotted')

    ax_gauss_f_pri[1].set_xlabel("$f$ $[Hz]$")

    plt.savefig('plots/gaussian_f_pri.jpeg', dpi=600)
    
    return fig_gauss_f, ax_gauss_f_pri
