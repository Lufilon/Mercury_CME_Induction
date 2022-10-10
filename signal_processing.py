# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 16:57:50 2022

@author: Luis-
"""

from numpy.fft import rfft, rfftfreq
from numpy import pi, exp, real, angle


def fft_own(t, N, data):
    # =========================================================================
    # Perform the fast fourier transformation on the given data.
    # =========================================================================
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
    # =========================================================================
    # Rebuild the initial signal with a reduced amount of frequencies.
    # =========================================================================
    data_rebuild = [amp[i] * exp(0+1j * (2 * pi * freq[i] * t + phase[i]))
                    for i in range(len(freq))]

    return sum(real(data_rebuild))
