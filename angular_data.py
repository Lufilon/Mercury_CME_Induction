# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 18:46:23 2023

@author: Luis-
"""

from numpy import pi, linspace, meshgrid, ravel


def angular_data(num_theta, num_phi):
    """
    Creates evenly spaced data points on (num_theta x num_phi)-grid.

    Parameters
    ----------
    num_theta : int
        Number of points in latteral direction.
    num_phi : int
        Number of points in longitudinal direction.

    Returns
    -------
    num_pts : int
        Number of points.
    theta_arr : numpy.ndarray.float64
        num_theta evenly spaced values in [0, pi).
    phi_arr : numpy.ndarray.float64
        num_phi evently spaced values in [0, 2*pi).
    theta : numpy.ndarray.float64
        Lattitude of data points.
    phi : numpy.ndarray.float64
        Longitude of data points.

    """
    num_pts = int(num_theta * num_phi)
    theta_arr = linspace(0, pi, num_theta, endpoint=False)
    phi_arr = linspace(0, 2*pi, num_phi, endpoint=False)
    phi_arr_2D, theta_arr_2D = meshgrid(phi_arr, theta_arr)
    theta, phi = ravel(theta_arr_2D), ravel(phi_arr_2D)

    print("Finished creating the angular data.")

    return num_pts, theta_arr, phi_arr, theta, phi
