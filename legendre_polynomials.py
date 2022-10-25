# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 10:15:43 2022

@author: Luis-
"""
from numpy import sqrt, zeros
from scipy.special import factorial, lpmn


def P_dP(l, m, x):
    """
    An auxilliary function to calculate the Schmidt quasi-normalized
    associated Legendre-polynomials and it's derivative of degree l and order m 
    for the variable x
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
    The Schmidt quasi-normalized associated Legendre-polynomials of degree l
    and order m and it's derivative for the variable x
    """

    # Schmid quasi-normalization
    if m == 0:
        norm = sqrt(factorial(l-m) / factorial(l+m))
    else:
        norm = sqrt(2 * factorial(l-m) / factorial(l+m))

    # calculate assosiated legendre polynomials for given l, m
    P_lm = zeros(len(x))
    dP_lm = zeros(len(x))
    
    for i in range(len(x)):
        P_lm[i] = lpmn(m, l, x[i])[0][m][l]
        dP_lm[i] = lpmn(m, l, x[i])[1][m][l]
        P_lm[i] = norm * P_lm[i]
        dP_lm[i] = norm * dP_lm[i]

    # compensate also for the condon phase factor
    P_lm, dP_lm = pow(-1, m) * P_lm, pow(-1, m) * dP_lm

    return P_lm, dP_lm
