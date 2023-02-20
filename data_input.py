# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 10:14:44 2022

@author: Luis-
"""


from datetime import datetime
import numpy as np
import pandas as pd
from scipy.constants import m_p
import matplotlib.pyplot as plt


def data_input(path, empty_rows=0, plot=False):
    """
    Read the (cme)-data and calculate the heliocentric distance and subsolar
    standoff distance, if requested plot them.

    Parameters
    ----------
    path : String
        Path to .txt file containing the data.
        Time needs to be in ISO 8601 format, given in t_regex.
    empty_rows : int, optional
        Rows to skip in data file. The default is 0.
    plot : TYPE, boolean
        Plot the data and subsolar standoff distance if required.
        The default is False.

    Returns
    -------
    t : integer array
        Time since start in seconds.
    t_plotting : datetime array
        Time in UTC, used for plotting.
    pseudo_distance : float array
        Heliocentric peuso-distance in au, calculated via pognan et al (2018).
    R_ss : float array
        Subsolar standoff distance calculated for each pseudo_distance with the
        KTH22-model.

    """

    colnames = [
        't [s]', 'N_p [1/cm^3]', 'v_r [km/s]', 'v_t [km/s]', 'v_n [km/s]'
        ]
    df = pd.read_csv(path, header=None, delim_whitespace=True, names=colnames,
                     skiprows=empty_rows)

    t = df['t [s]']

    t_start = np.array(t[0], dtype='datetime64')
    t_plotting = np.empty(t.size, dtype='datetime64[ms]')

    for i in range(0, t.size):
        t_plotting[i] = t_start + np.timedelta64(i, 'm')

    t_REGEX = '%Y-%m-%dT%H:%M:%S.%f'
    t_0 = datetime.strptime(t[0], t_REGEX).timestamp()

    for i in t:
        t = t.replace(i, datetime.strptime(i, t_REGEX).timestamp() - t_0)

    t = t.to_numpy()

    print("Imported the cme data.")

    N_p = df['N_p [1/cm^3]'].to_numpy()
    N_p = N_p * 1E6  # 1/cm^3 -> 1/m^3
    v_r = df['v_r [km/s]'].to_numpy()
    v_t = df['v_t [km/s]'].to_numpy()
    v_n = df['v_n [km/s]'].to_numpy()
    v = np.sqrt(v_r**2 + v_t**2 + v_n**2)
    v = v * 1E3  # km/s -> m/s

    # calculate the heliocentric distance using the solarwind velocity
    rho = m_p * N_p
    p_dyn = 1/2 * rho * v**2
    T_SOLAR = 4.6E9  # alter der sonne in jahren
    pseudo_distance = np.sqrt(6.1E-7/p_dyn * (T_SOLAR/1E6)**(-0.67))

    print("Calculated the heliocentric pseudo_distances.")

    di = 50
    R_M = 1
    f = 2.0695 - (0.00355 * di)
    R_ss = f * pseudo_distance ** (1 / 3) * R_M

    # the KTH-Modell can't process R_ss < R_M - remove those data points
    for i in range(R_ss.size):
        if R_ss[i] < 1 + 1E-1:  # 1E-1 caused by rounding in kth-model
            R_ss[i] = np.nan
            pseudo_distance[i] = np.nan

    # interpolate data and cut sides, required for fft
    R_ss = np.interp(
        np.arange(R_ss.size),
        np.arange(R_ss.size)[np.isnan(R_ss) == False],
        R_ss[np.isnan(R_ss) == False])
    pseudo_distance = np.interp(
        np.arange(pseudo_distance.size),
        np.arange(pseudo_distance.size)[np.isnan(pseudo_distance) == False],
        pseudo_distance[np.isnan(pseudo_distance) == False])

    print("Calculated the subsolar standoff-distance")

    if plot:
        data_plot(t_plotting, N_p, v, R_ss)

    return t, t_plotting, pseudo_distance, R_ss


def data_plot(t_plotting, N_p, v, R_ss):
    # cme_signal
    fig_data, (ax_N_p, ax_v) = plt.subplots(2, sharex=True)
    plt.subplots_adjust(hspace=0)
    ax_N_p.set_title("Number of particles and particle velocity measured" +
                     " by Helios 1")
    ax_N_p.plot(t_plotting, N_p*1E-6, linewidth=2)
    ax_N_p.set_yscale('log')
    ax_N_p.set_yticks([pow(10, 1), pow(10, 2), pow(10, 3)])
    ax_v.plot(t_plotting, v*1E-3, linewidth=2)
    ax_N_p.set_ylabel("$N_p$ [$1/cm^3$]")
    ax_v.set_ylabel("$v$ [$km/s$]")
    fig_data.savefig('plots/data.jpg', dpi=600)

    print("Plotted the given data")

    # subsolar standoff-distance
    fig_cme, ax_cme = plt.subplots()
    ax_cme.set_title("Data from Helios 1 at $r_{hel} = 0.31$ AU")
    ax_cme.plot(t_plotting, R_ss)
    ax_cme.set_ylabel("$R_{\\mathrm{SS}}}}}$ [$R_\\mathrm{M}$]")
    fig_cme.savefig('plots/CME_profile.jpg', dpi=600)

    print("Plotted the subsolar standoff-distance")