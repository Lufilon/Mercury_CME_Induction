# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 16:57:50 2022

@author: Luis-
"""

from numpy.fft import rfft, rfftfreq
from numpy import pi, exp, real, angle
from numpy import zeros, asarray, flip, isin, dot, newaxis, array, linspace
import matplotlib.pyplot as plt


def gaussian_t_to_f(coeff_ext_t, t, t_steps, gauss_list_ext, freqnr):
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
    gauss_list_ext : list.tupel
        List containing gauss coefficients to be analyzed.
    freqnr : int
        Number of frequencies returned.
        Can be used to determine influence of sampled number of frequencies.

    Returns
    -------
    freq : numpy.ndarray.float64
        Frequency of the primary gauss coefficients.
    coeff_ext_f_amp : numpy.ndarray.float64
        Amplitude of the primary gauss coefficients.
    coeff_ext_f_phase : numpy.ndarray.float64
        Phase of the primary gauss coefficients.
    rel_indices : numpy.ndarray.int
        The freqnr indices used for the analysis.

    """
    freq = zeros((len(gauss_list_ext), t_steps//2 + 1))
    coeff_ext_f_amp = zeros((len(gauss_list_ext), t_steps//2 + 1))
    coeff_ext_f_phase = zeros((len(gauss_list_ext), t_steps//2 + 1))

    rel_indices = zeros((len(gauss_list_ext), freqnr), dtype='int')

    for l, m in gauss_list_ext:
        index = gauss_list_ext.index((l, m))
        freq[index], coeff_ext_f_amp[index], coeff_ext_f_phase[index] = fft_own(
            t, asarray([coeff_ext_t[i][m][l] for i in range(t_steps)]), t_steps)

        # get relevant frequencies, filter f_0 = 0 Hz beforehand
        rel_indices[index] = flip(coeff_ext_f_amp[index].argsort()[-freqnr:])
        mask = isin(coeff_ext_f_amp[index], coeff_ext_f_amp[index][rel_indices[index]],
                    invert=True)
        coeff_ext_f_amp[index][mask] = 0

    print("Finished fourier-transforming the external Gauss coefficients.")
    
    return freq, coeff_ext_f_amp, coeff_ext_f_phase, rel_indices


def gaussian_f_to_t(t, freq, amp, phase):
    """
    Transforms a given signal in the frequency domaine to the time domaine.
    Is equal to the inverse fourier transform but frequencies can be sampled.

    Parameters
    ----------
    t : numpy.ndarray.float64
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
    # rebuild the initial signal with a reduced amount of frequencies.
    data_rebuild = amp[:, newaxis] * exp(0+1j * (2 * pi * dot(
        freq[:, newaxis], t[newaxis, :]) + phase[:, newaxis]))
    data_rebuild = real(data_rebuild).sum(axis=0)

    return data_rebuild


def fft_own(t, data, N):
    """
    Performs the fast fourier transform on a given set of data.

    Parameters
    ----------
    t : numpy.ndarray.int
        Time since start in seconds.
    data : numpy.ndarray.float64
        Time series of some data.
    N : int
        Size of the input data.

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
