# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 10:14:44 2022

@author: Luis-
"""


from numpy import linspace, nanmin, nanmax, array, empty, timedelta64, sqrt, nan
from pandas import DataFrame, read_csv
from scipy.constants import m_p
from datetime import datetime
import matplotlib.pyplot as plt


def data_get(path, empty_rows, resolution, plot=False):
    """
    Read the (cme)-data and calculate the heliocentric distance and subsolar
    standoff distance, if requested plot them.

    Parameters
    ----------
    path : String
        Path to .txt file containing the data.
        Time needs to be in ISO 8601 format, given in regex.
    empty_rows : int
        Rows to skip in data file.
    resolution : int
        Number of distances for which the magnetic field is calculated for.
    plot : TYPE, boolean
        Plot the data and subsolar standoff distance if required.
        The default is False.

    Returns
    -------
    t : numpy.ndarray.int
        Time since start in seconds.
    t_plotting : anumpy.ndarray.datetime
        Time in UTC, used for plotting.
    t_plotting : anumpy.ndarray.datetime
        Time in UTC, used for plotting.
    t_steps : int
        Number of measurements.
    r_hel : numpy.ndarray.float64
        Heliocentric peuso-distance in au, calculated via pognan et al (2018).
    R_ss : numpy.ndarray.float64
        Subsolar standoff distance calculated for each r_hel with the
        KTH22-model.
    possible_distances : numpy.ndarray.float64
        Heliocentric (pseudo) distance.

    """

    colnames = [
        't [s]', 'N_p [1/cm^3]', 'v_r [km/s]', 'v_t [km/s]', 'v_n [km/s]'
        ]
    df = read_csv(path, header=None, delim_whitespace=True, names=colnames,
                  skiprows=empty_rows)

    t = df['t [s]']

    t_start = array(t[0], dtype='datetime64')
    t_plotting = empty(t.size, dtype='datetime64[ms]')

    for i in range(0, t.size):
        t_plotting[i] = t_start + timedelta64(i, 'm')

    regex = '%Y-%m-%dT%H:%M:%S.%f'
    t_0 = datetime.strptime(t[0], regex).timestamp()

    for i in t:
        t = t.replace(i, datetime.strptime(i, regex).timestamp() - t_0)

    t = t.to_numpy()

    print("Imported the cme data.")

    N_p = df['N_p [1/cm^3]'].to_numpy()
    N_p = N_p * 1E6  # 1/cm^3 -> 1/m^3
    v_r = df['v_r [km/s]'].to_numpy()
    v_t = df['v_t [km/s]'].to_numpy()
    v_n = df['v_n [km/s]'].to_numpy()
    v = sqrt(v_r**2 + v_t**2 + v_n**2)
    v = v * 1E3  # km/s -> m/s

    # calculate the heliocentric distance using the solarwind velocity
    rho = m_p * N_p
    p_dyn = 1/2 * rho * v**2
    T_SOLAR = 4.6E9  # alter der sonne in jahren
    r_hel = sqrt(6.1E-7/p_dyn * (T_SOLAR/1E6)**(-0.67))

    print("Calculated the heliocentric pseudo distance.")

    di = 50
    R_M = 1
    f = 2.0695 - (0.00355 * di)
    R_ss = f * r_hel ** (1 / 3) * R_M

    # the KTH-Modell can't process R_ss < R_M - remove those data points
    for i in range(R_ss.size):
        if R_ss[i] < 1 + 1E-1:  # 1E-1 caused by rounding in kth-model
            R_ss[i] = nan
            r_hel[i] = nan

    # interpolate data and cut sides, required for fft
    R_ss = DataFrame(R_ss).interpolate(
        method='linear', limit_direction='both').to_numpy()
    r_hel = DataFrame(r_hel).interpolate(
        method='linear', limit_direction='both').to_numpy()

    print("Calculated the subsolar standoff distance")

    t_steps = t.size

    # distances for which the magnetic field is calculated
    possible_distances = linspace(nanmin(r_hel), nanmax(r_hel), resolution)
    print(nanmin(r_hel))
    print(nanmax(r_hel))
    print(possible_distances)

    if plot:
        data_plot(t_plotting, N_p, v, R_ss)

    return t, t_plotting, t_steps, r_hel, R_ss, possible_distances


def data_plot(t_plotting, N_p, v, R_ss):
    """
    Plot the given Data and the resulting subsolar standoff distance over time.

    Parameters
    ----------
    t_plotting : numpy.ndarray.datetime
        Time in UTC, used for plotting.
    N_p : numpy.ndarray.float64
        Measured number of particles.
    v : numpy.ndarray.float64
        Measured particle velocity.
    R_ss : numpy.ndarray.float64
        Subsolar standoff distance.

    Returns
    -------
    None.

    """
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
