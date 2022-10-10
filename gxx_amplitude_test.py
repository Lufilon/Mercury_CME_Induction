# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 20:49:36 2022

@author: Luis-
"""

from numpy.fft import fft, fftfreq, ifft
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm


def data():
    # asymmetric gaussian
    height = 1
    data = height * skewnorm.pdf(t, a=2, loc=t[int(N/2)])

    return data


def fft_own(data):
    freq = fftfreq(N, dt)
    data_fft = fft(data)
    amp = np.abs(data_fft) / N
    phase = np.angle(data_fft)
    peaks, = np.where(amp >= threshold)  # use whole spectrum for rebuild

    # plt.plot(freq, amp)
    # plt.xlim(0, 10)
    # plt.ylim(0, 1)

    return freq, amp, phase, peaks, data_fft


def rebuild(fft_own):
    freq, amp, phase, peaks, data_fft = fft_own

    data_fft[np.abs(data_fft) < threshold*N] = 0

    counter = 0  # counts the number of used peaks
    for i in data_fft:
        if i != 0:
            counter += 1
    print(counter)

    rebuild = ifft(data_fft).real

    f, ax = plt.subplots(1, 1)
    ax.plot(t, data_init, label="original")
    ax.plot(t, rebuild, label="rebuild")
    ax.legend()

# =============================================================================
# parameters -> soon to be value imported from file
# =============================================================================
t0 = 0
t1 = 10
N = 7200
t = np.linspace(t0, t1, int(N))
dt = t[1] - t[0]

threshold = 0  # determines the number of frequencies used for the rebuild

data_init = data()
fft_init = fft_own(data_init)
rebuild_init = rebuild(fft_init)