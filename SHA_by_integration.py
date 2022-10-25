# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 15:31:28 2022

@author: Luis-
"""

from math import factorial
from numpy import sin, cos, sqrt, pi, zeros, array
import json
from scipy.special import lpmv
from scipy.integrate import simpson


def SHA_by_integration(B_r, B_theta, B_phi, ana_radius, ref_radius, DEGREE_MAX, m):
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
    
    
    # =========================================================================
    # reshape magnetic field component arrays and import angle_data
    # =========================================================================
    with open('data/angle_data.json') as f:
        angle = json.load(f)
    
    N_THETA, N_PHI = angle["N_THETA"], angle["N_PHI"]
    theta_arr, phi_arr = array(angle["theta_arr"]), array(angle["phi_arr"])
    
    B_r = B_r.reshape(N_THETA, N_PHI)
    B_theta = B_theta.reshape(N_THETA, N_PHI)
    B_phi = B_phi.reshape(N_THETA, N_PHI)
    
    # =========================================================================
    # calculation of the the coefficients
    # =========================================================================
    
    c_theta = cos(theta_arr)
    s_theta = sin(theta_arr)
    
    coeff_int = zeros(DEGREE_MAX + 1)
    coeff_ext = zeros(DEGREE_MAX + 1)
    
    # analysis of the B_r component - independent of the order
    GR_lm_arr = zeros(DEGREE_MAX + 1)
    integrand_r = zeros(N_PHI)
    for l in range(m, DEGREE_MAX + 1):
    
        # integration over theta for every phi
        for i_phi in range(0, N_PHI):
            # print(simpson(B_r[:, i_phi], theta_arr))
            integrand_r[i_phi] = simpson(B_r[:, i_phi] * P(l, m, c_theta)
                                         * s_theta, theta_arr)
            # print(integrand_r[i_phi])
    
        # integration over phi
        int_result = simpson(integrand_r * cos(m * phi_arr), phi_arr)
        GR_lm_arr[l] = (2 * l + 1) / (4 * pi) * int_result
    
    if m == 0:
        # analysis of the B_theta component  # (2.40) bis (2.44)
        V_lm_arr = zeros(DEGREE_MAX + 2)
    
        for l in range(0, DEGREE_MAX + 2):
            integrand_t = zeros(N_PHI)
    
            # integration over theta for every phi
            for i_phi in range(0, N_PHI):
                integrand_t[i_phi] = simpson(B_theta[:, i_phi] * P(l, m, c_theta)
                                             * s_theta**2, theta_arr)
    
            # integration over phi
            int_result = simpson(integrand_t * cos(m * phi_arr), phi_arr)
            V_lm_arr[l] = (2 * l + 1) / (4 * pi) * int_result
    
        # successive calculation of general Gauss-coefficients
        # l is the actual degree; the order is always the same: m
        GT_lm = zeros(DEGREE_MAX + 1)
    
        for l in range(1, DEGREE_MAX + 1):
            if l-2 <= 0:
                GT_lm[l] = -V_lm_arr[l-1] / b_lm(l, m)
            else:
                GT_lm[l] = - (V_lm_arr[l-1] - a_lm(l-2, m) / b_lm(l-2, m)
                                * V_lm_arr[l-3]) / b_lm(l, m)
    
        # calculate the actual gauss coefficients
        for l in range(m, DEGREE_MAX + 1):
            coeff_int[l] = 1/(2*l + 1) * (ana_radius/ref_radius)**(l+2) * (GR_lm_arr[l] - l * GT_lm[l])
            coeff_ext[l] = -1/(2*l + 1) * (ref_radius/ana_radius)**(l-1) * (GR_lm_arr[l] + (l+1) * GT_lm[l])
    
    else:  # m > 0
        # analysis of the B_phi component  # (2.45)
        GP_lm_arr = zeros(DEGREE_MAX + 1)
        integrand_p = zeros(N_PHI)
        for l in range(m, DEGREE_MAX + 1):
            # integration over theta for every phi
            for i_phi in range(0, N_PHI):
                integrand_p[i_phi] = simpson(B_phi[:, i_phi] * P(l, m, c_theta)
                                             * s_theta**2, theta_arr)
            int_result = simpson(integrand_p * sin(m * phi_arr), phi_arr)
            GP_lm_arr[l] = (2*l + 1)/(4*pi) * int_result
    
        # calculate the gauss coefficients  # (2.46)
        for l in range(m, DEGREE_MAX + 1):
            coeff_int[l] = (l*GP_lm_arr[l] + m*GR_lm_arr[l])/((2*l + 1)*m) * (ana_radius/ref_radius)**(l+2)
            coeff_ext[l] = ((l+1)*GP_lm_arr[l] - m*GR_lm_arr[l])/((2*l + 1)*m) * (ana_radius/ref_radius)**(-l+1)
    
    return coeff_int, coeff_ext
