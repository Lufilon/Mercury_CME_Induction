#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 11:22:06 2021

@author: patkolhe
"""
import numpy as np
from pyshtools.legendre import PlmIndex as legind
# from lsq_inv import least_square
# from tikhonov_svd_inv import tikhonov
# from capon_inv import capon
from libinvbs import build_dsgn_mtrx, calcLegPol
import matplotlib.pyplot as plt

"""
This small program computes the magnetic field from the foward problem A*g = b.
As a comparison the magnetic field is also computed strictly from the summation
of the Gauss coefficients and the realtions contained in the design matrix A.
Furthermore in the inversion problem g = A^(-1) * b is tested with
analytical magnetic field data + some gaussian noise. Choice of coordinates
very arbitrary, so bad inversion results possible.

"""


def calc_B_ana(lmax, R_p, g, r, theta, phi, ndat, intern=True, ext=False,
               mie=False):

    B_r = np.zeros(ndat, dtype=np.float64)
    B_t = np.zeros(ndat, dtype=np.float64)
    B_p = np.zeros(ndat, dtype=np.float64)

    if mie:
        b = (np.max(r) + np.min(r))/2.
        rho = (r - b) / R_p
        ncoeff_mie = len(g)

    for i in range(0, ndat):

        Plm, dPlm = calcLegPol(lmax, theta[i])

        if intern:

            k = 0

            for ell in range(1, lmax+1):

                rfac = (R_p/r[i])**(ell+2)

                for m in range(0, ell+1):

                    B_r[i] += (np.float64(ell+1) * rfac
                               * (g[k]*np.cos(np.float64(m)*phi[i])
                                  + g[k+1]*np.sin(np.float64(m)*phi[i]))
                               * Plm[legind(ell, m)])

                    B_t[i] += (-rfac * (g[k]*np.cos(np.float64(m)*phi[i])
                                        + g[k+1]*np.sin(np.float64(m)*phi[i]))
                               * dPlm[legind(ell, m)])

                    B_p[i] += (np.float64(m) * rfac
                               * (g[k] * np.sin(np.float64(m)*phi[i])
                                  - g[k+1] * np.cos(np.float64(m)*phi[i]))
                               * Plm[legind(ell, m)]/np.sin(theta[i]))

                    k += 1
                    if (ell - m >= 1 and m != 0):
                        k += 1
                k += 1

        if ext:

            k = 0

            for ell in range(1, lmax+1):

                rfac = (r[i]/R_p)**(ell-1)

                for m in range(0, ell+1):

                    B_r[i] += (- np.float64(ell) * rfac
                               * (g[k]*np.cos(np.float64(m)*phi[i])
                                  + g[k+1]*np.sin(np.float64(m)*phi[i]))
                               * Plm[legind(ell, m)])

                    B_t[i] += (- rfac * (g[k]*np.cos(np.float64(m)*phi[i])
                                         + g[k+1]*np.sin(np.float64(m)*phi[i]))
                               * dPlm[legind(ell, m)])

                    B_p[i] += (np.float64(m) * rfac
                               * (g[k]*np.sin(np.float64(m)*phi[i])
                                  - g[k+1]*np.cos(np.float64(m)*phi[i]))
                               * Plm[legind(ell, m)]/np.sin(theta[i]))

                    k += 1
                    if(ell - m >= 1 and m != 0):
                        k += 1
                k += 1

        if mie:

            k = 0
            j = k + ncoeff_mie//2

            for ell in range(1, lmax+1):

                rfac = R_p / r[i]

                for m in range(0, ell+1):

                    B_r[i] = 0.0

                    B_t[i] += (rfac * np.float64(m)
                               * ((-g[k] * np.sin(np.float64(m)*phi[i])
                                   + g[k+1] * np.cos(np.float64(m)*phi[i]))
                                  + rho[i]*(-g[j]
                                            * np.sin(np.float64(m)*phi[i])
                                            + g[j+1]
                                            * np.cos(np.float64(m)*phi[i])))
                               * Plm[legind(ell, m)]/np.sin(theta[i]))

                    B_p[i] += (-rfac * ((g[k]*np.cos(np.float64(m)*phi[i])
                                         + g[k+1]*np.sin(np.float64(m)*phi[i]))
                                        + rho[i] * (
                                            g[j]*np.cos(np.float64(m)*phi[i])
                                            + g[j+1]
                                            * np.sin(np.float64(m)*phi[i])))
                               * dPlm[legind(ell, m)])

                    k += 1
                    j = k + ncoeff_mie//2

                    if(ell - m >= 1 and m != 0):
                        k += 1
                        j = k + ncoeff_mie//2

                k += 1
                j = k + ncoeff_mie//2

    return B_r, B_t, B_p


lmax_int = 2
lmax_ext = 2
lmax_mie = 2

ncoeff_in = int((lmax_int+2) * lmax_int)
ncoeff_ext = int((lmax_ext+2) * lmax_ext)
ncoeff_mie = int((lmax_mie+2) * lmax_mie)

# odering g_1_0, g_1_1, h_1_1, g_2_0, g_2_1, h_2_1, g_2_2, h_2_2
g_int = np.zeros(ncoeff_in, dtype=np.float64)
g_ext = np.zeros(ncoeff_ext, dtype=np.float64)
g_mie = np.zeros(2*ncoeff_mie, dtype=np.float64)
# siehe Reihenfolge oben
g_int[0] = -192.0
g_int[1] = 2.6
g_int[2] = 0.1
g_int[3] = -78.0
g_int[4] = -2.0
g_int[7] = -1.0

g_ext[0] = 40.0
g_ext[1] = 2.0
g_ext[2] = 10.
g_ext[4] = 20.

g_mie[0] = 40.  # die sollten dann rausgelassen werden
g_mie[8] = 30.
g_mie[9] = 20.

g_dat = np.concatenate((g_int, g_ext, g_mie))

R_M = 2440.*1.0e3

r = np.ones(300, dtype=np.float64) * R_M

# r = np.linspace(1.1, 2.5, num=300) * R_M

# theta = np.deg2rad(np.linspace(1.0e-4, 179, num=300, dtype=np.float64))

# phi = np.deg2rad(np.linspace(1.0, 345., num=300, dtype=np.float64))

# =============================================================================
# Temp
# =============================================================================
theta = np.linspace(1, np.pi-1, 300, endpoint=False)
phi = np.linspace(1, 2*np.pi-1, 300, endpoint=False)
# =============================================================================
# Temp
# =============================================================================

ndat = np.size(r)

# build desgin matrix
A = build_dsgn_mtrx(lmax_int, lmax_ext, lmax_mie, ndat, r, theta, phi, R_M,
                    internal=True, ext=True, mie=True)

# solve forward problem
B = np.dot(A, g_dat)

# calculate analytical solution
Br_ana_int = np.zeros(ndat, dtype=np.float64)
Bt_ana_int = np.zeros(ndat, dtype=np.float64)
Bp_ana_int = np.zeros(ndat, dtype=np.float64)

Br_ana_ext = np.zeros(ndat, dtype=np.float64)
Bt_ana_ext = np.zeros(ndat, dtype=np.float64)
Bp_ana_ext = np.zeros(ndat, dtype=np.float64)

Br_ana_mie = np.zeros(ndat, dtype=np.float64)
Bt_ana_mie = np.zeros(ndat, dtype=np.float64)
Bp_ana_mie = np.zeros(ndat, dtype=np.float64)

# hier dann nur die int und ext anteile anschauen und mit SHA vergleichen
Br_ana_int, Bt_ana_int, Bp_ana_int = calc_B_ana(lmax_int, R_M, g_int, r, theta,
                                                phi, ndat, intern=True,
                                                ext=False, mie=False)

Br_ana_ext, Bt_ana_ext, Bp_ana_ext = calc_B_ana(lmax_ext, R_M, g_ext, r, theta,
                                                phi, ndat, intern=False,
                                                ext=True, mie=False)

Br_ana_mie, Bt_ana_mie, Bp_ana_mie = calc_B_ana(lmax_mie, R_M, g_mie, r, theta,
                                                phi, ndat, intern=False,
                                                ext=False, mie=True)
# =============================================================================
# temp
# =============================================================================
# plt.plot(Br_ana_int, label="r_int")
plt.plot(Bt_ana_int, label="theta_int")
# plt.plot(Bp_ana_int, label="B_phi,int")
plt.legend()
plt.show()

# plt.plot(Br_ana_ext, label="r_ext")
plt.plot(Bt_ana_ext, label="theta_ext")
# plt.plot(Bp_ana_ext, label="B_phi,ext")
plt.legend()
plt.show()
# =============================================================================
# temp
# =============================================================================

Br_ana = Br_ana_int + Br_ana_ext + Br_ana_mie
Bt_ana = Bt_ana_int + Bt_ana_ext + Bt_ana_mie
Bp_ana = Bp_ana_int + Bp_ana_ext + Bp_ana_mie

# compare to foward problem (design matrix check)
diff_r = B[::3] - Br_ana
diff_t = B[1::3] - Bt_ana
diff_p = B[2::3] - Bp_ana

B_r = Br_ana + np.random.randn(ndat)*np.mean(Br_ana)*1.0e-4
B_t = Bt_ana + np.random.randn(ndat)*np.mean(Bt_ana)*1.0e-4
B_p = Bp_ana + np.random.randn(ndat)*np.mean(Bp_ana)*1.0e-4


# # solve inverse problem

# lsq = least_square(Br=B_r, Btheta=B_t, Bphi=B_p, r=r, theta=theta, phi=phi,
#                    R_p=R_M, lmax_int=lmax_int, lmax_ext=lmax_ext,
#                    lmax_mie=lmax_mie, internal=True, ext=True, mie=True)
# lsq.lsq_classic()
# print('Solution vector for classic least square inversion')
# print(lsq.g_lsq[::3])

# lsq.lsq_svd(lcurve=False, eps=1.0e-3, plt_lcrv=False, pick_index=False)
# print('Solution vector for truncated SVD inversion')
# print(lsq.g_tsvd)

# tikh = tikhonov(Br=B_r, Btheta=B_t, Bphi=B_p, r=r, theta=theta, phi=phi,
#                 R_p=R_M, lmax_int=lmax_int, lmax_ext=lmax_ext,
#                 lmax_mie=lmax_mie, internal=True, ext=True, mie=True)

# tikh.tikhonov_svd(alph_max=1.0e1, alph_min=1.0e-6, alph_step=0.5,
#                   linstep=False, logstep=True, plt_lcrv=False,
#                   pick_index=False)

# print('Solution vector for Tikhonov regularization inversion')
# print(tikh.g_tikh)

# cap = capon(Br=B_r, Btheta=B_t, Bphi=B_p, r=r, theta=theta, phi=phi,
#             R_p=R_M, lmax_int=lmax_int, lmax_ext=lmax_ext,
#             lmax_mie=lmax_mie, internal=True, ext=True, mie=True)

# cap.capon_inv(dlp_search=False, sigma=1000.)
# print('Solution vector for inversion with Capons Method')
# print(cap.g_c)
