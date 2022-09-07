#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 09:44:54 2021

@author: patkolhe
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, svd
from libinvbs import build_dsgn_mtrx, rot_cart_to_spher


class tikhonov():
    """
    class to perfom inversion of magentic field data to determine Gauss
    coefficients (and toroidal coefficient in Gauss-Mie representation) with a
    Tikhonov regularization scheme.

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

    def tikhonov_svd(self, eps=1.0e-6, alph_min=1.0e-6, alph_max=3.0,
                     alph_step=1.0e-2, plt_lcrv=False, linstep=True,
                     logstep=False, pick_index=False):
        """
        Perform the SVD inversion with a Tikhonov regularization.

        Parameters
        ----------
        eps : float, optional
            Lower bound for the singular values. Singular values below this
            value are regarded. The default is 1.0e-6.
        alph_min : float, optional
            Starting value for determining the optimal damping parameter alpha.
            The default is 1.0e-6.
        alph_max : float, optional
            Last value for determining the optimal damping parameter alpha.
            The default is 3.0.
        alph_step : float, optional
            Step size for determining the optimal damping parameter alpha.
            The default is 1.0e-2.
        plt_lcrv : bool, optional
            Plot the resulting L-curve to determine alpha.
            The default is False.
        linstep : bool, optional
            Search the area for alpha in linear steps. The default is True.
        logstep : bool, optional
            Search the area for alpha in logarithmic steps. alpha_step is then
            the exponent of the power with basis 10.
            The default is False.
        pick_index : bool, optional
            Pick the damping parameter alpha by hand if automatic pick is
            not satisfying. The default is False.

        Returns
        -------
        None.

        """

        ndat = np.size(self.Br)

        B_all = np.zeros(3*ndat)

        B_all[::3] = self.Br
        B_all[1::3] = self.Bt
        B_all[2::3] = self.Bp

        A = build_dsgn_mtrx(self.lmax_int, self.lmax_ext, self.lmax_mie, ndat,
                            self.r, self.theta, self.phi, self.R_p,
                            internal=self.internal, ext=self.ext, mie=self.mie)

        U, self.sig, Vt = svd(A, full_matrices=True)

        V = Vt.T

        print('Maximal Eigenvalue: {:.3e}, Minimal Eigenvalue: {:.3e}'
              .format(np.max(self.sig), np.min(self.sig)))

        Sig = np.zeros_like(A)

        vsize = np.min(np.shape(A))

        Sig[:vsize, :vsize] = np.diag(self.sig)

        Sig_inv = np.zeros_like(A.T)

        cutoff_ind = np.where(self.sig < eps)

        if np.size(cutoff_ind) > 0:
            ncut = cutoff_ind[0][0]
            vsize = ncut

        if linstep:
            nstep = int((alph_max - alph_min)/alph_step)
        elif logstep:
            nstep = int((np.log10(alph_max) - np.log10(alph_min))/alph_step)

        mod_norm = np.zeros(nstep)

        misfit_norm = np.zeros(nstep)

        alph = np.zeros(nstep, dtype=np.float64)

        for i in range(0, nstep):

            if linstep:
                alph[i] = alph_min + np.float64(i)*alph_step
            elif logstep:
                alph_log = np.log10(alph_min) + np.float64(i)*alph_step
                alph[i] = np.power(10, alph_log)

            Sig_inv[:vsize, :vsize] = np.diag(self.sig[:vsize]
                                              / (self.sig[:vsize]**2 +
                                                 alph[i]**2))

            m = np.dot(np.dot(np.dot(V, Sig_inv), U.T), B_all)

            fwd_mod = np.dot(A, m)

            mod_norm[i] = norm(m, ord=2)

            misfit_norm[i] = norm(fwd_mod - B_all, ord=2)

        misfit_scal = ((misfit_norm-np.min(misfit_norm))
                       / (np.max(misfit_norm) - np.min(misfit_norm)))

        mod_scal = ((mod_norm - np.min(mod_norm))
                    / (np.max(mod_norm) - np.min(mod_norm)))

        r_scal = np.sqrt(misfit_scal**2 + mod_scal**2)

        ind_min = int(np.where(r_scal == np.min(r_scal))[0])

        cond_number_full = self.sig[0]/self.sig[-1]
        self.cond_number = (np.max(self.sig + alph[ind_min]**2/self.sig)
                            / np.min(self.sig + alph[ind_min]**2/self.sig))

        print('The automatic picked dammping parameter is: '
              '{:.3e}'.format(alph[ind_min]))

        print('The condition number without damping is: {:.2f}'.format(
            cond_number_full))
        print('The condititon with damping after automatic pick '
              'is: {:.2f}'.format(self.cond_number))

        if plt_lcrv:

            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(121)
            ax.set_xlabel(r'$\left\Vert Gm-d \right\Vert_2$', fontsize=14)
            ax.set_ylabel(r'$\left\Vert m \right\Vert_2$', fontsize=14)
            ax.tick_params(axis='both', labelsize=12, width=1, length=3)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.plot(misfit_norm, mod_norm, color='k', linewidth=2)
            ax.plot(misfit_norm[ind_min], mod_norm[ind_min], linestyle='None',
                    color='red', marker='o', markersize=5)

            ax2 = fig.add_subplot(122)
            ax2.set_xlabel('Number of singular value', fontsize=14)
            ax2.set_ylabel('Singular value', fontsize=14)
            ax2.tick_params(axis='both', labelsize=12, width=1, length=3)
            ax2.plot(self.sig, color='black', linewidth=2)

            fig.tight_layout()

            plt.show()
            plt.pause(4)

            if pick_index:
                print('pick index by hand, automatic pick is '
                      '{:d}'.format(ind_min))
                ind_min = int(input('\ninsert index\n'))
                print(r'alpha={:.3e}'.format(alph[ind_min]))
                ax.plot(misfit_norm[ind_min], mod_norm[ind_min],
                        linestyle='None', color='green', marker='o',
                        markersize=5)
                plt.pause(4)
                plt.close()
                self.cond_number = (np.max(self.sig
                                           + alph[ind_min]**2/self.sig)
                                    / np.min(self.sig
                                             + alph[ind_min]**2/self.sig))
                print('The condition number is now: {:.2f}'.format(
                    self.cond_number))

            plt.close()

        Sig[:vsize, :vsize] = np.diag((self.sig**2 + alph[ind_min]**2)
                                      / self.sig)

        Sig_inv[:vsize, :vsize] = np.diag(self.sig / (self.sig**2
                                                      + alph[ind_min]**2))
        self.A = np.dot(np.dot(U, Sig), Vt)
        self.g_tikh = np.dot(np.dot(np.dot(V, Sig_inv), U.T), B_all)
