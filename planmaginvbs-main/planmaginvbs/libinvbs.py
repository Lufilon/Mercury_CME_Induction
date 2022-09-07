#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 14:13:30 2022

@author: patrick
"""

import numpy as np
from numpy.linalg import eig
import pyshtools.legendre as leg
from pyshtools.legendre import PlmIndex as legind


def calcLegPol(lmax, theta, csph=False):
    """
    This function computes the associated Legendre polynomials and their
    derivatives with respect to the colatitude for a given maximal spherical
    harmonic degree and a given colatitude. Here the shtools library
    (https://shtools.oca.eu) is used for the calculations. shtools calculates
    the Legendre polynomials with respect to a variable z, which is chosen to
    be z=cos(theta) in this case. Therefore also a correction is needed for
    the derivative.

    Parameters
    ----------
    lmax : int
        maximal spherical harmonic degree for the calculation.
    theta : float
        colatidude in radian.
    csph: bool, optional
        Usage of a Condon-Shortley phase. If True then the CS phase is
        included. Default is False.

    Returns
    --------
    Plm : float
        Array with all Legendre polynimals of degree and order (l,m) = lmax
        for the given colatitude
    dPlm : float
        Array with the corresponding derivatives of Plm with respect to the
        colatitude

    """
    if csph:
        cs = -1
    else:
        cs = 1

    Plm, dPlm = leg.PlmSchmidt_d1(lmax, np.cos(theta), csphase=cs)
    dPlm *= -np.sin(theta)

    return Plm, dPlm


def build_dsgn_mtrx(lmax_in, lmax_ext, lmax_mie, ndat, r, theta, phi, R_p,
                    internal=True, ext=False, mie=False):
    """
    Purpose of this function is to build the design matrix of the forward
    problem A*g=b, where A is the design matrix, g the coefficient vector
    containing the Gauss coeffiecients and b the resulting magnetic field
    vector.

    Parameters:
    ----------
    lmax_in : int
        maximum degree and order of the internal Gauss coefficients.
    lmax_ext : int
        maximum degree and order of the external Gauss coefficients.
    lmax_mie : int
        maximum degree and order of the toroidal coefficients
        (Mie representation, thin shell approximation).
    ndat : int
        length of one component from the magnetic field vector. For the 3D
        magntic field the data vector b will be of size 3*ndat and ordered as
        follows: b = (Br_1, Btheta_1, Bphi_1, Br_2, Btheta_2, Bphi2, ... ),
        where the subscript denotes the number of the time stamp.
    r : array_like
        radial postion vector of size ndat.
    theta : array_like
        colatitdue vector of size ndat in radian.
    phi : array_like
        azimtuth vector of size ndat in radian.
    R_p : float
        planetary radius.
    internal : bool, optional
        Should the design matrix include the components for the internal Gauss
        coefficients?
        The default is True.
    ext : bool, optional
        Should the design matrix include the components for the external Gauss
        coefficients? The defaul is False.
    mie : bool, optional
        Should the design matrix include the components for the toroidal
        coefficients of the Mie representation in thin shell approx.?
        The default is False.

    Returns:
    -------
    A : array_like
        Design matrix of the foward/inversion problem of size (3*ndat, ncoeff).
        ncoeff is determined by the maximum degree and order lmax of the Gauss
        coefficients. Is an external field included besides an internal
        magnetic field the size of matrix is
        (3*ndat, (ncoeff_int, ncoeff_ext)). With the toroidal coefficients
        the size is (3*ndat, (ncoeff_int, ncoeff_ext, ncoeff_tor)).
    """

    ncoeff_in = int((lmax_in + 2) * lmax_in)
    ncoeff_ext = int((lmax_ext + 2) * lmax_ext)
    ncoeff_mie = int((lmax_mie + 2) * lmax_mie)

    if internal:
        A_int = np.zeros((3*ndat, ncoeff_in), dtype=np.float64)

    if ext:
        A_ext = np.zeros((3*ndat, ncoeff_ext), dtype=np.float64)

    if mie:
        A_mie = np.zeros((3*ndat, ncoeff_mie), dtype=np.float64)
        A_mie_str = np.zeros((3*ndat, ncoeff_mie), dtype=np.float64)
        b = (np.max(r) + np.min(r))/2.
        rho = (r-b)/R_p

    for i in range(0, ndat):

        k = 3*i

        Plm, dPlm = calcLegPol(np.max((lmax_in, lmax_ext)), theta[i],
                               csph=False)

        if internal:

            ell = 1
            m = 0

            j = 0

            while j < ncoeff_in:

                # g coefficients

                rfac = (R_p/r[i])**(ell+2)

                # r
                A_int[k, j] = (np.float64(ell+1) * rfac
                               * np.cos(np.float64(m)*phi[i])
                               * Plm[legind(ell, m)])
                # theta
                A_int[k+1, j] = (- rfac * np.cos(np.float64(m)*phi[i])
                                 * dPlm[legind(ell, m)])
                # phi
                A_int[k+2, j] = (np.float64(m) * rfac
                                 * np.sin(np.float64(m)*phi[i])
                                 * Plm[legind(ell, m)] / np.sin(theta[i]))

                if m > 0:

                    # h coefficients

                    j += 1

                    # r
                    A_int[k, j] = (np.float64(ell+1) * rfac
                                   * np.sin(np.float64(m)*phi[i])
                                   * Plm[legind(ell, m)])
                    # theta
                    A_int[k+1, j] = (-rfac * np.sin(np.float64(m)*phi[i])
                                     * dPlm[legind(ell, m)])
                    # phi
                    A_int[k+2, j] = (-np.float64(m) * rfac
                                     * np.cos(np.float64(m)*phi[i])
                                     * Plm[legind(ell, m)] / np.sin(theta[i]))
                if (m == ell):

                    m = 0
                    ell += 1

                else:

                    m += 1

                j += 1

        if ext:

            j = 0

            ell = 1
            m = 0

            while j < ncoeff_ext:

                # g coefficents

                rfac = (r[i]/R_p)**(ell-1)

                # r
                A_ext[k, j] = (-np.float64(ell) * rfac
                               * np.cos(np.float64(m)*phi[i])
                               * Plm[legind(ell, m)])

                # theta
                A_ext[k+1, j] = (-rfac * np.cos(np.float64(m)*phi[i])
                                 * dPlm[legind(ell, m)])

                # phi
                A_ext[k+2, j] = (np.float64(m) * rfac
                                 * np.sin(np.float64(m)*phi[i])
                                 * Plm[legind(ell, m)] / np.sin(theta[i]))

                if m > 0:

                    # h coefficients

                    j += 1

                    # r
                    A_ext[k, j] = (-np.float64(ell) * rfac
                                   * np.sin(np.float64(m)*phi[i])
                                   * Plm[legind(ell, m)])

                    # theta
                    A_ext[k+1, j] = (-rfac * np.sin(np.float64(m)*phi[i])
                                     * dPlm[legind(ell, m)])

                    # phi
                    A_ext[k+2, j] = (-np.float64(m) * rfac
                                     * np.cos(np.float64(m)*phi[i])
                                     * Plm[legind(ell, m)] / np.sin(theta[i]))

                if (m == ell):

                    m = 0
                    ell += 1

                else:

                    m += 1

                j += 1

        if mie:

            j = 0

            ell = 1
            m = 0

            while j < ncoeff_mie:

                # a and a' coefficients

                rfac = R_p/r[i]

                # r
                A_mie[k, j] = np.float64(0.0)

                A_mie_str[k, j] = np.float64(0.0)

                # theta

                A_mie[k+1, j] = (-rfac * np.float64(m)
                                 * np.sin(np.float64(m)*phi[i])
                                 * Plm[legind(ell, m)] / np.sin(theta[i]))

                A_mie_str[k+1, j] = (- rfac * rho[i] * np.float64(m)
                                     * np.sin(np.float64(m)*phi[i])
                                     * Plm[legind(ell, m)] / np.sin(theta[i]))

                # phi

                A_mie[k+2, j] = (-rfac * np.cos(np.float64(m)*phi[i])
                                 * dPlm[legind(ell, m)])

                A_mie_str[k+2, j] = (-rfac * rho[i]
                                     * np.cos(np.float64(m)*phi[i])
                                     * dPlm[legind(ell, m)])

                if m > 0:

                    # b and b' coefficients

                    j += 1

                    # r
                    A_mie[k, j] = np.float64(0.)

                    A_mie_str[k, j] = np.float64(0.)

                    # theta
                    A_mie[k+1, j] = (rfac * np.float64(m)
                                     * np.cos(np.float64(m)*phi[i])
                                     * Plm[legind(ell, m)] / np.sin(theta[i]))

                    A_mie_str[k+1, j] = (rfac * rho[i] * np.float64(m)
                                         * np.cos(np.float64(m)*phi[i])
                                         * Plm[legind(ell, m)]
                                         / np.sin(theta[i]))
                    # phi
                    A_mie[k+2, j] = (-rfac * np.sin(np.float64(m)*phi[i])
                                     * dPlm[legind(ell, m)])

                    A_mie_str[k+2, j] = (-rfac * rho[i]
                                         * np.sin(np.float64(m)*phi[i])
                                         * dPlm[legind(ell, m)])

                if (m == ell):
                    ell += 1
                    m = 0
                else:
                    m += 1

                j += 1

    if internal and not ext:

        A = A_int

    elif ext and not internal:

        A = A_ext

    if internal and ext:

        A = np.concatenate((A_int, A_ext), axis=1)

    if internal and ext and mie:

        A = np.concatenate((A_int, A_ext, A_mie, A_mie_str), axis=1)

    if not internal and not ext:

        KeyError('internal and ext are both set to False. No desgin matrix'
                 'can be returned')

    return A


def init_rot_mat(ndat, phi, theta):
    """
    Initialize the rotation matrix.

    Parameters
    ----------
    ndat : int
        size of data vector (one component).
    phi : array_like
        azimuth vector.
    theta : array_like
        colatitude vector.

    Returns
    -------
    rotmat : array_like
        rotation matrix to transform from spherical to Cartesion coordinates.
        Transpose the matrix to rotate from Cartesian to spherical coordinates.

    """

    rotmat = np.zeros((ndat, 3, 3))

    for i in range(0, ndat):

        rotmat[i, ...] = np.array(((np.sin(theta[i]) * np.cos(phi[i]),
                                    np.cos(theta[i]) * np.cos(phi[i]),
                                    -np.sin(phi[i])),
                                   (np.sin(theta[i]) * np.sin(phi[i]),
                                    np.cos(theta[i]) * np.sin(phi[i]),
                                    np.cos(phi[i])),
                                   (np.cos(theta[i]), -np.sin(theta[i]), 0)))

    return rotmat


def rot_cart_to_spher(x, y, z, Bx, By, Bz):
    """
    Rotate coordinates and magnetic field data from Cartesian to spherical
    coordinate system.

    Parameters
    ----------
    x : array_like
        x coordinate.
    y : array_like
        y coordinate.
    z : array_like
        z coordinate.
    Bx : array_like
        Magnetic field component in x direction.
    By : array_like
        Magnetic field component in y direction.
    Bz : array_like
        Magnetic field component in z direction.

    Returns
    -------
    r : array_like
        radius vector.
    theta : array_like
        colatitude vector.
    phi : array_like
        azimuth vector.
    Br : array_like
        radial magnetic field component.
    Btheta : array_like
        latitudenal magnetic field component.
    Bphi : array_like
        azimuthal magnetic field component.

    """

    ndat = int(np.size(Bx))

    r = np.sqrt(x**2 + y**2 + z**2)

    theta = np.arccos(z/r)  # in radians

    phi = np.arctan2(y, x)  # in radians

    B_cart = np.array([Bx, By, Bz]).T

    B_spher = np.zeros_like(B_cart)

    rotmat = init_rot_mat(ndat, phi, theta)

    for i in range(0, ndat):

        B_spher[i, :] = np.dot(rotmat[i, ...].T, B_cart[i, :])

    Br = B_spher[:, 0]
    Btheta = B_spher[:, 1]
    Bphi = B_spher[:, 2]

    return r, theta, phi, Br, Btheta, Bphi


def rot_spher_to_cart(r, theta, phi, Br, Btheta, Bphi):
    """
    Rotate coordinates and magnetic field data from spherical to Cartesian
    coordinate system.

    Parameters
    ----------
    r : array_like
        radius vector.
    theta : array_like
        colatitude vector.
    phi : array_like
        azimuth vector.
    Br : array_like
        radial magnetic field component.
    Btheta : array_like
        latitudenal magnetic field component.
    Bphi : array_like
        azimuthal magnetic field component.

    Returns
    -------
    x : array_like
        x coordinate.
    y : array_like
        y coordinate.
    z : array_like
        z coordinate.
    Bx : array_like
        Magnetic field component in x direction.
    By : array_like
        Magnetic field component in y direction.
    Bz : array_like
        Magnetic field component in z direction.

    """

    ndat = int(np.size(Br))

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    B_spher = np.array([Br, Btheta, Bphi]).T

    B_cart = np.zeros_like(B_spher)

    rotmat = init_rot_mat(ndat, phi, theta)

    for i in range(0, ndat):

        B_cart[i, :] = np.dot(rotmat[i, ...], B_spher[i, :])

    Bx = B_cart[:, 0]
    By = B_cart[:, 1]
    Bz = B_cart[:, 2]

    return x, y, z, Bx, By, Bz


def perform_MVA(Bx, By, Bz):
    """
    This function computes the minimum variance direction of an input data set
    with three components in x, y, z direction of length N. The function
    computes the covariance of the resulting input array of size Nx3 and
    determines the corresponding eigenvalues and eigenvectors. With the help
    of the eigenvectors the input array can be rotated in the minimum variance
    system.

    Parameters
    ----------
    Bx : float
        input x-component of data vector with size N.
    By : float
        input y-component of data vector with size N.
    Bz : float
        input z-component of data vector with size N.

    Raises
    ------
    ValueError
        If vector components are not of the same size a value error is raised.

    Returns
    -------
    B_out : float
        Nx3 array of the input array rotated in minimum variance system.
        Components are ordered from the maximum variance direction to the
        mimimum variance direction.
    eigB : float
        Three eigenvalues of the covariance matrix sorted from the largest
        eigenvalue to lowest eigenvalue.
    evecB : float
        Corresponding eigenvectors in a 3 x 3 array sorted by like the
        eigenvalues above.
    """

    if np.size(Bx) != np.size(By) != np.size(Bz):
        raise ValueError('magnetic field vector must be of the same size')

    # put components togehter in one array
    B = np.array((Bx, By, Bz), dtype=np.float64)

    # compute covariance matrix
    covB = np.cov(B)

    # compute eigenvalues and corresponding eigenvectors
    eigB, evecB = eig(covB)

    # deterimine largest and lowest eigenvalues and sort them
    ind_max = int(np.where(eigB == np.max(eigB))[0])
    ind_min = int(np.where(eigB == np.min(eigB))[0])
    ind_mid = int(np.where(np.logical_and(eigB != eigB[ind_max],
                                          eigB != eigB[ind_min]))[0])

    eigB = np.array((eigB[ind_max], eigB[ind_mid], eigB[ind_min]))

    evecB = np.array((evecB[:, ind_max], evecB[:, ind_mid],
                      evecB[:, ind_min])).T

    # rotate data into the maximum/minimum variance system
    B_out = np.dot(B.T, evecB)

    return B_out, eigB, evecB
