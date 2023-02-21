# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 15:31:28 2022

@author: Luis-
"""

from math import factorial
from numpy import sin, cos, sqrt, pi, zeros, savetxt, loadtxt, nan, isnan
from scipy.special import lpmv
from scipy.integrate import simpson
import matplotlib.pyplot as plt


def SHA_by_integration_get(
        theta_arr, phi_arr, r_hel, possible_distances, t_steps, Br_possible,
        Bt_possible, Bp_possible, num_theta=200, num_phi=400, degree_max=2,
        resolution=100, ref_radius=2440, ana_radius=2440,
        path='data/helios_1/ns=True/gaussian_t/resolution='):
    """
    Calculate and save the time dependant Gauss coefficients via the SHA.

    Parameters
    ----------
    theta_arr : numpy.ndarray.float64
        Lattitude of the data.
    phi_arr : numpy.ndarray.float64
        Longitude of the data.
    r_hel : numpy.ndarray.float64
        Heliocentric peuso-distance in au.
    possible_distances : numpy.ndarray.float64
        Sampled heliocentric peuso-distance in au for.
    t_steps : int
        Number of measurements.
    Br_possible : numpy.ndarray.float64
        Radial magnetic field component at given r_hel.
    Bt_possible : numpy.ndarray.float64
        Latitudinal magnetic field component at possible_distances.
    Bp_possible : numpy.ndarray.float64
        Longitudinal magnetic field component at possible_distances.
    num_theta : int, optional
        Number of points in latteral direction. The default is 200.
    num_phi : int, optional
        Number of points in longitudinal  direction. The default is 400.
    degree_max : int, optional
        Maximum degree for the SHA. The default is 2.
    resolution : int, optional
        Number of distances for which the magnetic field is calculated for.
        The default is 100.
    ref_radius : int, optional
        Radius of the reference shell, by default R_M. The default is 2440.
    ana_radius : int, optional
        Radius where the magnetic field was calculated at. The default is 2440.
    path : string, optional
        Path to the directory where the Gauss coefficients are stored.
        The default is 'data/helios_1/ns=True/gaussian_t/resolution='.

    Returns
    -------
    coeff_ext_t : TYPE
        Time dependant external Gauss coefficients of the primary field.

    """
    try:
        coeff_ext_t_possible = loadtxt(
            path + str(resolution) + '_degree_max=' +
            str(degree_max) + '_external.gz')

        coeff_ext_t_possible = coeff_ext_t_possible.reshape(
            resolution, pow(degree_max, 2) // degree_max+1,
            degree_max+1)

        print("Finished importing the time dependant Gauss coefficients"
              + "with resolution=" + str(resolution) + ".")

    except OSError:
        print("No file for this combination of pseudo_distance resolution " +
              "and max degree of spherical analyzed Gauss coefficients was " +
              "stored for the given CME and given kth-modell parameters. " +
              "- Started the calculation.")

        coeff_ext_t_possible = zeros((resolution, degree_max+1, degree_max+1))

        for i in range(resolution):
            for m in range(degree_max + 1):
                coeff_ext_t_possible[i][m] = SHA_by_integration_calc(
                    theta_arr, phi_arr, num_theta, num_phi,
                    Br_possible[i], Bt_possible[i], Bp_possible[i],
                    ref_radius, ana_radius, degree_max, m)[1]

        print("Finished calculating the time dependant Gauss coefficients " +
              "for the given resolution using the SHA by integration.")

        savetxt(path + str(resolution) + '_degree_max=' +
                str(degree_max) + '_external.gz',
                coeff_ext_t_possible.reshape(
                    coeff_ext_t_possible.shape[0], -1))

        print("Finished saving the time dependant Gauss coefficients.")

    # assign the values to the data points with the smallest deviation
    coeff_ext_t = zeros((t_steps, degree_max+1, degree_max+1))

    for i in range(t_steps):
        if not isnan(r_hel[i]):
            nearest_distance_index = (
                abs(possible_distances - r_hel[i])).argmin()
            coeff_ext_t[i] = coeff_ext_t_possible[nearest_distance_index]
        else:
            coeff_ext_t[i] = nan

    print("Finished upscaling the lower resolution Gauss coefficients.")

    return coeff_ext_t


def SHA_by_integration_calc(theta_arr, phi_arr, num_theta, num_phi, Br, Bt, Bp,
                            ref_radius, ana_radius, degree_max, m):
    """
    Calculate time dependant Gauss coefficients via the SHA for a given field.

    Parameters
    ----------
    theta_arr : numpy.ndarray.float64
        Lattitude of the data.
    phi_arr : numpy.ndarray.float64
        Longitude of the data.
    num_theta : int, optional
        Number of points in latteral direction. The default is 200.
    num_phi : int, optional
        Number of points in longitudinal  direction. The default is 400.
    Br : TYPE
        Radial magnetic field component.
    Bt : TYPE
        Lattitudinal magnetic field component.
    Bp : TYPE
        Longitudinal magnetic field component.
    ref_radius : int, optional
        Radius of the reference shell, by default R_M. The default is 2440.
    ana_radius : int, optional
        Radius where the magnetic field was calculated at. The default is 2440.
    degree_max : int, optional
        Maximum degree for the SHA. The default is 2.
    m : int
        Order of the magnetic field.

    Returns
    -------
    coeff_ext_t : TYPE
        Time dependant external Gauss coefficients of the primary field..

    """
    def a_lm(l, m):
        if l <= 0:
            return 0
        return l * (m - l - 1) / (2 * l + 1)

    def b_lm(l, m):
        return (l + 1) * (l + m) / (2 * l + 1)

    def P(l, m, x):
        """
        An auxilliary function to calculate the Schmidt quasi-normalized
        associated Legendre-polynomials of
        degree l and order m for the variable x
        Parameters
        ----------
        l : int
            Degree of the Legendre polynomial
        m : int
            Order of the Legendre polynomial
        x : float
            Input to evaluate the Legendre polynomial at
        Returns
        -------
        None.
        """

        # Schmid quasi-normalization
        if m == 0:
            norm = sqrt(factorial(l-m) / factorial(l+m))
        else:
            norm = sqrt(2 * factorial(l-m) / factorial(l+m))

        # compensate also for the condon phase factor
        return pow(-1, m) * norm * lpmv(m, l, x)

    Br = Br.reshape(num_theta, num_phi)
    Bt = Bt.reshape(num_theta, num_phi)
    Bp = Bp.reshape(num_theta, num_phi)

    # calculation of the the coefficients
    c_theta = cos(theta_arr)
    s_theta = sin(theta_arr)

    coeff_int = zeros(degree_max + 1)
    coeff_ext = zeros(degree_max + 1)

    # analysis of the Br component - independent of the order
    GR_lm_arr = zeros(degree_max + 1)
    integrand_r = zeros(num_phi)
    for l in range(m, degree_max + 1):

        # integration over theta for every phi
        for i_phi in range(0, num_phi):
            # print(simpson(Br[:, i_phi], theta_arr))
            integrand_r[i_phi] = simpson(Br[:, i_phi] * P(l, m, c_theta)
                                         * s_theta, theta_arr)
            # print(integrand_r[i_phi])

        # integration over phi
        int_result = simpson(integrand_r * cos(m * phi_arr), phi_arr)
        GR_lm_arr[l] = (2 * l + 1) / (4 * pi) * int_result

    if m == 0:
        # analysis of the Bt component  # (2.40) bis (2.44)
        V_lm_arr = zeros(degree_max + 2)

        for l in range(0, degree_max + 2):
            integrand_t = zeros(num_phi)

            # integration over theta for every phi
            for i_phi in range(0, num_phi):
                integrand_t[i_phi] = simpson(Bt[:, i_phi] * P(l, m, c_theta)
                                             * s_theta**2, theta_arr)

            # integration over phi
            int_result = simpson(integrand_t * cos(m * phi_arr), phi_arr)
            V_lm_arr[l] = (2 * l + 1) / (4 * pi) * int_result

        # successive calculation of general Gauss-coefficients
        # l is the actual degree; the order is always the same: m
        GT_lm = zeros(degree_max + 1)

        for l in range(1, degree_max + 1):
            if l-2 <= 0:
                GT_lm[l] = -V_lm_arr[l-1] / b_lm(l, m)
            else:
                GT_lm[l] = - (V_lm_arr[l-1] - a_lm(l-2, m) / b_lm(l-2, m)
                                * V_lm_arr[l-3]) / b_lm(l, m)

        # calculate the actual gauss coefficients
        for l in range(m, degree_max + 1):
            coeff_int[l] = 1/(2*l + 1) * (ana_radius/ref_radius)**(l+2) * (GR_lm_arr[l] - l * GT_lm[l])
            coeff_ext[l] = -1/(2*l + 1) * (ref_radius/ana_radius)**(l-1) * (GR_lm_arr[l] + (l+1) * GT_lm[l])

    else:  # m > 0
        # analysis of the Bp component  # (2.45)
        GP_lm_arr = zeros(degree_max + 1)
        integrand_p = zeros(num_phi)
        for l in range(m, degree_max + 1):
            # integration over theta for every phi
            for i_phi in range(0, num_phi):
                integrand_p[i_phi] = simpson(Bp[:, i_phi] * P(l, m, c_theta)
                                             * s_theta**2, theta_arr)
            int_result = simpson(integrand_p * sin(m * phi_arr), phi_arr)
            GP_lm_arr[l] = (2*l + 1)/(4*pi) * int_result
    
        # calculate the gauss coefficients  # (2.46)
        for l in range(m, degree_max + 1):
            coeff_int[l] = (l*GP_lm_arr[l] + m*GR_lm_arr[l])/((2*l + 1)*m) * (ana_radius/ref_radius)**(l+2)
            coeff_ext[l] = ((l+1)*GP_lm_arr[l] - m*GR_lm_arr[l])/((2*l + 1)*m) * (ana_radius/ref_radius)**(-l+1)

    return coeff_int, coeff_ext


def SHA_by_integration_plot(t_plotting, t_steps, gauss_list_ext, coeff_ext_t):
    """
    Plot external time-dependant inducing Gauss coefficients.

    Parameters
    ----------
    t_plotting : numpy.ndarray.datetime
        Time in UTC, used for plotting.
    t_steps : int
        Number of measurements.
    gauss_list_ext : list.tupel.int
        List containing all to be analyzed Gauss coefficients as pairs of
        (degree, order).
    coeff_ext_t : numpy.ndarray.float64
        Time dependant external Gauss coefficients of the primary field..

    Returns
    -------
    None.

    """
    fig_gauss_t_inducing, ax_gauss_t_inducing = plt.subplots(
        len(gauss_list_ext),  sharex=True)
    plt.subplots_adjust(hspace=0)
    ax_gauss_t_inducing[0].set_title("Time-dependant primary Gauss " +
                                     "coefficients")

    for l, m in gauss_list_ext:
        index = gauss_list_ext.index((l, m))

        ax_gauss_t_inducing[index].plot(
            t_plotting, [coeff_ext_t[i][m][l] for i in range(t_steps)],
            label="$g_{" + str(l) + str(m) + ", \\mathrm{pri}}}$")

        ax_gauss_t_inducing[index].set_ylabel("$A_\\mathrm{pri}$ $[nT]$")
        ax_gauss_t_inducing[index].legend(loc='lower left')

    plt.savefig('plots/gaussian_t_solo.jpeg', dpi=600)
