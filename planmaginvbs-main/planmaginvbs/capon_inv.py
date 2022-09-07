#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 08:34:52 2022

@author: patrick
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv
from libinvbs import build_dsgn_mtrx, rot_cart_to_spher


class capon():
    """
    class to perfom inversion of magentic field date to determine Gauss (and
    toroidal coefficient in Gauss-Mie representation) with Capon's Method.
    """

    def __init__(self, Bx=None, By=None, Bz=None, Br=None,
                 Btheta=None, Bphi=None, x=None, y=None, z=None, r=None,
                 theta=None, phi=None, R_p=1., lmax_int=2, lmax_ext=0,
                 lmax_mie=0, internal=True, ext=False, mie=False,
                 plot_solution=False):
        """
        Parameters
        ----------
        Bx : array_like, optional
            Magentic field component in x-direction. The default is None.
        By : array_like, optional
            Magentic field component in y-direction. The default is None.
        Bz : array_like, optional
            Magentic field component in z-direction. The default is None.
        Br : array_like, optional
            Radial magnetic field component. The default is None.
        Btheta : array_like, optional
            Latitudinal magnetic field component. The default is None.
        Bphi : array_like, optional
            Azimuthal magnetic field component. The default is None.
        x : array_like, optional
            x coordinate vector. The default is None.
        y : array_like, optional
            y coordinate vector. The default is None.
        z : array_like, optional
            z coordinate vector. The default is None.
        r : array_like, optional
            Radius vector. The default is None.
        theta : array_like, optional
            Colatitude vector. The default is None.
        phi : array_like, optional
            Azimuthal vector. The default is None.
        R_p : float, optional
            Planetray radius. The default is 1..
        lmax_int : int, optional
            Maximum degree and order of internal Gauss coefficients.
            The default is 2.
        lmax_ext : int, optional
            Maximum degree and order of external Gauss coefficients.
            The default is 0.
        lmax_mie : int, optional
            Maximum degree and order of toroidal Gauss coefficients in the
            Gauss-Mie representation of the inversion problem.
            The default is 0.
        internal : bool, optional
            Solve for internal Gauss coefficients? The default is True.
        ext : bool, optional
            Solve for external Gauss coefficients? The default is False.
        mie : bool, optional
            Solve for toroidal coefficients in the Gauss-Mie representation.
            The default is False.
        plot_solution : bool, optional
            Solve foward problem and compare data with model in a plot.
            The default is False.

        Raises
        ------
        ValueError
            Checkup for complete data set.

        Returns
        -------
        None.

        """

        if x is not None and y is not None and z is not None:

            self.x = x
            self.y = y
            self.z = z

        elif r is not None and theta is not None and phi is not None:

            self.r = r
            self.theta = theta
            self.phi = phi

        else:

            raise ValueError('No coordinate data given or incomplete')

        if Bx is not None and By is not None and Bz is not None:

            self.Bx = Bx
            self.By = By
            self.Bz = Bz

        elif Br is not None and Btheta is not None and Bphi is not None:

            self.Br = Br
            self.Bt = Btheta
            self.Bp = Bphi

        else:

            raise ValueError('No magnetic field data given or incomplete')

        self.R_p = R_p
        self.lmax_int = lmax_int
        self.lmax_ext = lmax_ext
        self.lmax_mie = lmax_mie
        self.internal = internal
        self.ext = ext
        self.mie = mie
        self.plot_sol = plot_solution

        if (hasattr(self, 'Bx') and hasattr(self, 'By') and
            hasattr(self, 'Bz') and hasattr(self, 'x') and hasattr(self, 'y')
                and hasattr(self, 'z')):

            (self.r, self.theta, self.phi,
             self.Br, self.Bt, self.Bp) = rot_cart_to_spher(self.x, self.y,
                                                            self.z, self.Bx,
                                                            self.By, self.Bz)

    def capon_inv(self, dlp_search=True, sig_min=1.0e2, sig_max=1.0e4,
                  sig_step=500, plt_lcurve=False, pick_index=False,
                  sigma=500.):
        """
        Perform inversion with Capon's method including the diagonal loading
        parameter.

        Parameters
        ----------
        dlp_search : bool, optional
            search for optimal diagonal loading paramter with L-curve approach.
            The default is True.
        sig_min : float, optional
            Starting guess for diagonal loading parameter.
            The default is 1.0e1.
        sig_max : flaot, optional
            Last value for diagonal loading parameter. The default is 1.0e4.
        sig_step : float, optional
            Step size for diagonal loading parameter search.
            The default is 500.
        plt_lcurve : bool, optional
            Plot the resulting L-curve from the diagonal loading parameter
            search. The default is False.
        pick_index : bool, optional
            Pick index by yourself to correct automatic pick from diagonal
            loading paramter search. The default is False.
        sigma : float, optional
            When dlp_search is False a fixed value for the diagonal loading
            parameter can be set. The default is 500.

        Returns
        -------
        None.

        """

        ndat = np.size(self.Br)

        B_all = np.zeros(3*ndat, dtype=np.float64)

        B_all[::3] = self.Br
        B_all[1::3] = self.Bt
        B_all[2::3] = self.Bp

        # design/shape matrix
        if not hasattr(self, 'A'):
            self.A = build_dsgn_mtrx(self.lmax_int, self.lmax_ext,
                                     self.lmax_mie, ndat, self.r, self.theta,
                                     self.phi, self.R_p,
                                     internal=self.internal, ext=self.ext,
                                     mie=self.mie)

        # covariance matrix
        self.M = np.outer(B_all, B_all)

        if dlp_search:

            nstep = nstep = int((sig_max - sig_min)/sig_step)
            sigma = np.zeros(nstep, dtype=np.float64)
            trwtw = np.zeros(nstep, dtype=np.float64)
            trhth = np.zeros(nstep, dtype=np.float64)

            for i in range(0, nstep):

                sigma[i] = sig_min + np.float64(i) * sig_step

                M_sig = self.M + sigma[i]**2 * np.identity(3*ndat)

                Msig_inv = pinv(M_sig)

                # filter matrix
                w = np.dot(np.dot(Msig_inv, self.A),
                           pinv(np.dot(np.dot(self.A.T, Msig_inv), self.A)))

                trwtw[i] = np.trace(np.dot(w.T, w))

                trhth[i] = np.trace(np.dot(np.dot(w.T, M_sig), w))

            scal_hth = (trhth - np.min(trhth)) / \
                (np.max(trhth) - np.min(trhth))
            scal_wth = (trwtw - np.min(trwtw)) / \
                (np.max(trwtw) - np.min(trwtw))
            rtr = np.sqrt(scal_wth**2 + scal_hth**2)
            ind_min = int(np.where(rtr == np.min(rtr))[0])

            if plt_lcurve:

                fig = plt.figure(figsize=(9, 9))
                ax = fig.add_subplot(111)
                ax.set_xlabel(r'$\mathrm{tr}[\mathbf{w}^T (\mathbf{M} '
                              r'+ \sigma_d^2 \mathbf{I}) \mathbf{w}]$',
                              fontsize=14)
                ax.set_ylabel(r'$\mathrm{tr}[\mathbf{w}^T \mathbf{w}]$',
                              fontsize=14)
                ax.tick_params(axis='both', labelsize=12, width=1, length=3)
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.plot(trhth, trwtw, color='k', linewidth=2)
                ax.plot(trhth[ind_min], trwtw[ind_min], color='red',
                        markersize=6, linestyle='None', marker='o')
                print('The automatic pick chose '
                      'sigma={:.2f}'.format(sigma[ind_min]))

                plt.show()
                plt.pause(4)

                if pick_index:
                    print(
                        'current index is for sigma is: {:d}'.format(ind_min))
                    ind_min = int(input('choose your index from {:d} '
                                        'possible ones\n'.format(nstep)))
                    print('value for sigma is now: {:.2f}'.format(
                        sigma[ind_min]))

                    ax.plot(trhth[ind_min], trwtw[ind_min], color='green',
                            markersize=6, linestyle='None', marker='o')

                    plt.pause(4)
                    plt.close()

                plt.close()

            self.M += sigma[ind_min]**2 * np.identity(3*ndat)

            del(w, M_sig)

        else:

            self.M += sigma**2 * np.identity(3*ndat)

        M_inv = pinv(self.M)
        AMA_inv = pinv(np.dot(np.dot(self.A.T, M_inv), self.A))

        self.g_c = np.dot(np.dot(np.dot(AMA_inv, self.A.T), M_inv), B_all)
