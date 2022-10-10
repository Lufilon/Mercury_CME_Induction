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


    # # =======================================================================
    # # test with Gauss-coefficients
    # # =======================================================================
    # import pyshtools.legendre as leg
    # from pyshtools.legendre import PlmIndex as legind
    
    
    # def calcLegPol(lmax, theta_arr, csph=False):
    #     """
    #     This function computes the associated Legendre polynomials and their
    #     derivatives with respect to the colatitude for a given maximal spherical
    #     harmonic degree and a given colatitude. Here the shtools library
    #     (https://shtools.oca.eu) is used for the calculations. shtools calculates
    #     the Legendre polynomials with respect to a variable z, which is chosen to
    #     be z=cos(theta) in this case. Therefore also a correction is needed for
    #     the derivative.
    #     Parameters
    #     ----------
    #     lmax : int
    #         maximal spherical harmonic degree for the calculation.
    #     theta : float
    #         colatidude in radian.
    #     csph: bool, optional
    #         Usage of a Condon-Shortley phase. If True then the CS phase is
    #         included. Default is False.
    #     Returns
    #     --------
    #     Plm : float
    #         Array with all Legendre polynimals of degree and order (l,m) = lmax
    #         for the given colatitude
    #     dPlm : float
    #         Array with the corresponding derivatives of Plm with respect to the
    #         colatitude
    #     """
    #     if csph:
    #         cs = -1
    #     else:
    #         cs = 1
    
    #     Plm, dPlm = leg.PlmSchmidt_d1(lmax, cos(theta_arr), csphase=cs)
    #     dPlm *= -sin(theta_arr)
    
    #     return Plm, dPlm
    
    
    # N_THETA = int(200)
    # N_PHI = int(2 * N_THETA)
    
    # phi_arr = linspace(1E-4, 2*pi, N_PHI, endpoint=False)
    # theta_arr = linspace(1E-4, pi, N_THETA, endpoint=False)
    # phi_arr_2D, theta_arr_2D = meshgrid(phi_arr, theta_arr)
    
    # # =======================================================================
    # # generating the internal test magnetic field
    # # =======================================================================
    # def dP(l, m, theta_arr_2D):
    #     theta_temp = ravel(theta_arr_2D)
    #     return asarray([calcLegPol(3, theta)[1][legind(l, m)] for theta in theta_temp]).reshape(N_THETA, N_PHI)
    # # in order: g_1_0, g_1_1, h_1_1, g_2_0, g_2_1, h_2_1, g_2_2, h_2_2
    # g_int = [-192, 2.6, 0.1, -78, -2, 0, 0, -1]
    
    # br, bp, bt = 0, 0, 0
    # # case l=1, m=0
    # br += 2 * g_int[0] * P(1, 0, cos(theta_arr_2D))
    # bp += 0
    # bt += - g_int[0] * dP(1, 0, theta_arr_2D)
    # # case l=1, m=1
    # br += 2 * (g_int[1]*cos(1*phi_arr_2D) + g_int[2]*sin(1*phi_arr_2D)) * P(1, 1, cos(theta_arr_2D))
    # bp += (g_int[1]*sin(1*phi_arr_2D) - g_int[2]*cos(1*phi_arr_2D)) * P(1, 1, cos(theta_arr_2D))/sin(theta_arr_2D)
    # bt += - (g_int[1]*cos(1*phi_arr_2D) + g_int[2]*sin(1*phi_arr_2D)) * dP(1, 1, theta_arr_2D)
    # # case l=2, m=0
    # br += 3 * g_int[3] * P(2, 0, cos(theta_arr_2D))
    # bp += 0
    # bt += - g_int[3] * dP(2, 0, theta_arr_2D)
    # # case l=2, m=1
    # br += 3 * (g_int[4]*cos(1*phi_arr_2D) + g_int[5]*sin(1*phi_arr_2D)) * P(2, 1, cos(theta_arr_2D))
    # bp += (g_int[4]*sin(1*phi_arr_2D) - g_int[5]*cos(1*phi_arr_2D)) * P(2, 1, cos(theta_arr_2D))/sin(theta_arr_2D)
    # bt += - (g_int[4]*cos(1*phi_arr_2D) + g_int[5]*sin(1*phi_arr_2D)) * dP(2, 1, theta_arr_2D)
    # # case l=2, m=2
    # br += 3 * (g_int[6]*cos(2*phi_arr_2D) + g_int[7]*sin(2*phi_arr_2D)) * P(2, 2, cos(theta_arr_2D))
    # bp += 2 * (g_int[6]*sin(2*phi_arr_2D) - g_int[7]*cos(2*phi_arr_2D)) * P(2, 2, cos(theta_arr_2D))/sin(theta_arr_2D)
    # bt += - (g_int[6]*cos(2*phi_arr_2D) + g_int[7]*sin(2*phi_arr_2D)) * dP(2, 2, theta_arr_2D)
    # # =======================================================================
    # # generating the external test magnetic field
    # # =======================================================================
    # # in order: g_1_0, g_1_1, h_1_1, g_2_0, g_2_1, h_2_1, g_2_2, h_2_2
    # g_ext = [40, 2, 10, 0, 20, 0, 0, 0]
    
    # # case l=1, m=0
    # br += - g_ext[0] * P(1, 0, cos(theta_arr_2D))
    # bp += 0
    # bt += - g_ext[0] * dP(1, 0, theta_arr_2D)
    # # case l=1, m=1
    # br += - (g_ext[1]*cos(phi_arr_2D) + g_ext[2]*sin(phi_arr_2D)) * P(1, 1, cos(theta_arr_2D))
    # bp += (g_ext[1]*sin(phi_arr_2D) - g_ext[2]*cos(phi_arr_2D)) *P(1, 1, cos(theta_arr_2D))/sin(theta_arr_2D)
    # bt += - (g_ext[1]*cos(phi_arr_2D) + g_ext[2]*sin(phi_arr_2D)) * dP(1, 1, theta_arr_2D)
    # # case l=2, m=0
    # br += - 2 * g_ext[3] * P(2, 0, cos(theta_arr_2D))
    # bp += 0
    # bt += - g_ext[3] * dP(2, 0, theta_arr_2D)
    # # case l=2, m=1
    # br += - 2* (g_ext[4]*cos(phi_arr_2D) + g_ext[5]*sin(phi_arr_2D)) * P(2, 1, cos(theta_arr_2D))
    # bp += (g_ext[4]*sin(phi_arr_2D) - g_ext[5]*cos(phi_arr_2D)) * P(2, 1, cos(theta_arr_2D))/sin(theta_arr_2D)
    # bt += - (g_ext[4]*cos(phi_arr_2D) + g_ext[5]*sin(phi_arr_2D)) * dP(2, 1, theta_arr_2D)
    # # case l=2, m=2
    # br  += - 2 * (g_ext[6]*cos(2*phi_arr_2D) + g_ext[7]*sin(2*phi_arr_2D)) * P(2, 2, cos(theta_arr_2D))
    # bp += 2 * (g_ext[6]*sin(2*phi_arr_2D) - g_ext[7]*cos(2*phi_arr_2D)) * P(2, 2, cos(theta_arr_2D))/sin(theta_arr_2D)
    # bt += - (g_ext[6]*cos(2*phi_arr_2D) + g_ext[7]*sin(2*phi_arr_2D)) * dP(2, 2, theta_arr_2D)
    
    # for target_order in range(DEGREE_MAX):
    #     coeff_int, coeff_ext = SHA_by_integration(br, bp, bt, ana_radius,
    #                                               ref_radius, theta_arr, phi_arr,
    #                                               DEGREE_MAX, target_order)
    #     print("m = ", target_order)
    #     print("internal: ", coeff_int[1:])
    #     print("external: ", coeff_ext[1:], "\n")