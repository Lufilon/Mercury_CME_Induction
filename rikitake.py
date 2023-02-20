# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 15:31:42 2022

@author: Luis-
"""

from numpy import sqrt, pi
import mpmath
mpmath.mp.dps = 6


"""
Programm to calculate the Rikitake factor for a given degree of magnetic field,
induction parameter and radial distance.
"""

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

        # Rekursionsende
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
