#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 13:48:32 2021

@author: patkolhe
"""
import numpy as np
from lsq_inv import least_square
from tikhonov_svd_inv import tikhonov
from capon_inv import capon


"""
This small test routine reads the data set from Toefer et al.
(2020b) to test the inversion routines. Data is given in the
MASO coordinate system and has to be converted into a spherical coordinate
system. Magnetic fiel data of the orbits where calculated by A.I.K.E.F.
(Müller et al., 2011), where an external field is included.
Internal Gauss coefficients in the simulation where chosen as follows:

g_1_0 = -190 nT ,
g_2_0 = -78 nT ,
g_3_0 = -20 nT ,

The solar wind magnetic field is direct along the coordinates
(x, y, z)^T = (0.0, 0.43, 0.9)^T
"""

d = np.genfromtxt('../test_data/BepiGESAMTmehr.dat_B_TL160000.txt',
                  dtype=np.float64)

R_m = 2440.*1.0e3

(x, y, z) = (d[:, 1]*R_m, d[:, 2]*R_m,  d[:, 3]*R_m)

(Bx, By, Bz) = (d[:, 4]*20., d[:, 5]*20., d[:, 6]*20.)

l_int = 3
l_ext = 3
l_mie = 2

lsq = least_square(x=x, y=y, z=z, Bx=Bx, By=By, Bz=Bz, R_p=R_m, lmax_int=l_int,
                   lmax_ext=l_ext, lmax_mie=l_mie, internal=True, ext=True,
                   mie=True)

lsq.lsq_classic()
print(lsq.g_lsq)
print('Solution vector with least square matrix inversion')


lsq.lsq_svd(lcurve=False, eps=1.0e0, plt_lcrv=False, pick_index=False)

print('Solution vector for truncated SVD')
print(lsq.g_tsvd)

tikh = tikhonov(x=x, y=y, z=z, Bx=Bx, By=By, Bz=Bz, R_p=R_m, lmax_int=l_int,
                lmax_ext=l_ext, lmax_mie=l_mie, internal=True, ext=True,
                mie=True)

tikh.tikhonov_svd(alph_min=1.0e-6, alph_max=1.0e1, alph_step=0.5, logstep=True,
                  linstep=False, plt_lcrv=False)

print('Solution vector with Tikhonov regularization')
print(tikh.g_tikh)

cap = capon(x=x, y=y, z=z, Bx=Bx, By=By, Bz=Bz, R_p=R_m, lmax_int=l_int,
            lmax_ext=l_ext, lmax_mie=l_mie, internal=True, ext=True,
            mie=True)

cap.capon_inv(dlp_search=True, sig_min=1.0e1, sig_max=1.0e4, sig_step=1000,
              plt_lcurve=False)

print('Solution vector for Capon Method')
print(cap.g_c)
