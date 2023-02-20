# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 10:34:04 2022

@author: Luis-
"""

from numpy import sin, cos, full, loadtxt, savetxt, ravel, zeros
from numpy import linspace, nanmin, nanmax, hypot, pi
from kth14_model_for_mercury_v7b import kth14_model_for_mercury_v7b
import matplotlib.pyplot as plt


def magneticfield_sum(r_hel, R_ss, phi, theta, n_theta, n_phi, resolution=100,
                      settings=[True, True, False, True, True], plot=False,
                      runtime_dir='/data/runtime',
                      path='data/helios_1/ns=True/magnetic/resolution='):
    try:
        num_pts = n_theta * n_phi

        Br_possible = loadtxt(path + str(resolution) + '_Br.gz')
        Bt_possible = loadtxt(path + str(resolution) + '_Bt.gz')
        Bp_possible = loadtxt(path + str(resolution) + '_Bp.gz')

        Br_possible = Br_possible.reshape((resolution, num_pts))
        Bt_possible = Bt_possible.reshape((resolution, num_pts))
        Bp_possible = Bp_possible.reshape((resolution, num_pts))

        print("Finished importing the magnetic field components with " +
              "resolution=" + str(resolution) + ".")

    except OSError:
        print("No magnetic field for this r_hel resolution was calculated yet"
              + " - Starting the calculation.")

        # Calculate the internal dipole and neutralsheet magnetic field, as
        # they are the same for every R_ss distance.
        try:
            Br_di_int = loadtxt(runtime_dir + 'Br_di_int.gz')
            Bt_di_int = loadtxt(runtime_dir + 'Bt_di_int.gz')
            Bp_di_int = loadtxt(runtime_dir + 'Bp_di_int.gz')
            Br_ns_int = loadtxt(runtime_dir + 'Br_ns_int.gz')
            Bt_ns_int = loadtxt(runtime_dir + 'Bt_ns_int.gz')
            Bp_ns_int = loadtxt(runtime_dir + 'Bp_ns_int.gz')

            print("Finished loading the precalculated parts from file.")

        except OSError:
            Br_di_int, Bt_di_int, Bp_di_int = magneticfield_calc(
                1, [True, False, False, True, False])
            Br_ns_int, Bt_ns_int, Bp_ns_int = magneticfield_calc(
                1, [False, True, False, True, False])

            savetxt(runtime_dir + 'Br_di_int.gz', Br_di_int)
            savetxt(runtime_dir + 'Bt_di_int.gz', Bt_di_int)
            savetxt(runtime_dir + 'Bp_di_int.gz', Bp_di_int)
            savetxt(runtime_dir + 'Br_ns_int.gz', Br_ns_int)
            savetxt(runtime_dir + 'Bt_ns_int.gz', Bt_ns_int)
            savetxt(runtime_dir + 'Bp_ns_int.gz', Bp_ns_int)

            print("Finished calculating dipole_int and neutralsheet_int.")

        # Use the calculated pseudo_distances to calculate the magnetic field.
        Br_possible = zeros((resolution, num_pts))
        Bt_possible = zeros((resolution, num_pts))
        Bp_possible = zeros((resolution, num_pts))

        possible_distance = linspace(nanmin(r_hel), nanmax(r_hel), resolution)

        if settings[3]:
            if settings[0]:
                for i in range(len(possible_distance)):
                    Br_possible[i] += Br_di_int
                    Bt_possible[i] += Bt_di_int
                    Bp_possible[i] += Bp_di_int

            if settings[1]:
                for i in range(len(possible_distance)):
                    Br_possible[i] += Br_ns_int
                    Bt_possible[i] += Bt_ns_int
                    Bp_possible[i] += Bp_ns_int

        if settings[4] and settings[0] or settings[1]:
            for i in range(len(possible_distance)):
                result = magneticfield_calc(
                    possible_distance[i], phi, theta, num_pts,
                    [settings[0], settings[1], False, False, True])
                Br_possible[i] += result[0]
                Bt_possible[i] += result[1]
                Bp_possible[i] += result[2]

        print("Finished calculating field components for given resolution.")

        magneticfield_save(Br_possible, Bt_possible, Bp_possible,
                           path + str(resolution))

    magneticfield_plot(Br_possible, Bt_possible, Bp_possible, R_ss, phi, theta,
                       resolution, n_theta, n_phi, settings)

    return Br_possible, Bt_possible, Bp_possible


def magneticfield_calc(r_hel, phi, theta, num_pts=80000,
                       settings=[True, True, False, True, True],
                       di_val=50.):
    # set basic parameters and import angle_data
    CONTROL_PARAM_PATH = 'control_params_v7bOct2.json'
    FIT_PARAM_PATH = 'kth_own_cf_fit_parameters_opt_total.dat'

    R_M = 2440

    di = full(num_pts, di_val)

    # create the data points in mso coordinates
    x = R_M * sin(theta) * cos(phi)
    y = R_M * sin(theta) * sin(phi)
    z = R_M * cos(theta)

    # calculating the magnetic field components
    B_x, B_y, B_z = kth14_model_for_mercury_v7b(
        x, y, z, r_hel, di, CONTROL_PARAM_PATH, FIT_PARAM_PATH,
        settings[0], settings[1], settings[2], settings[3], settings[4]
        )

    # transform into spherical coordinate base
    Br = sin(theta) * cos(phi) * B_x
    Br += sin(theta) * sin(phi) * B_y
    Br += cos(theta) * B_z
    Bt = cos(theta) * cos(phi) * B_x
    Bt += cos(theta) * sin(phi) * B_y
    Bt += - sin(theta) * B_z
    Bp = - sin(phi) * B_x
    Bp += cos(phi) * B_y

    return Br, Bt, Bp


def magneticfield_plot(Br, Bt, Bp, R_ss, phi_arr, theta_arr, resolution=100,
                       n_theta=200, n_phi=400,
                       settings=[True, True, False, True, True]):
    # Plot the data to a mollweide projection if requested
    possible_R_ss = linspace(nanmin(R_ss), nanmax(R_ss), resolution)

    for i, val in enumerate(possible_R_ss[:1]):
        B = hypot(Bp[i], hypot(Br[i], Bt[i]))

        plt.figure("Mollweide projection of magnetic field")
        plt.subplot(projection='mollweide')
        # plt.title("Magnetic field strength on Mercury's surface, " +
        #           "calculated with KTH22 for $R_{SS} = $" +
        #           str(round(val, 2)) + " $R_\\mathrm{M}$\n\n" +
        #           "settings: dipole, neutralsheet, prc, internal, " +
        #           "external = " + str(settings), fontsize=7)
        # plt.suptitle("$R_\\mathrm{SS} = " + str(round(val, 2)) + "$\n " +
        #               "Neutralsheet = " + str(settings[1]))
        plt.grid()
        lon, lat = phi_arr - pi, pi/2 - theta_arr
        im = plt.contourf(lon, lat, B.reshape(n_theta, n_phi),
                          levels=linspace(0, 700, 8, endpoint=True),
                          extend='both')
        cbar = plt.colorbar(im)
        cbar.set_label('$B$ [$nT$]')

        plt.tight_layout()
        plt.savefig('plots/mollweide/' + str(i) + '.jpg', dpi=600)


def magneticfield_save(Br, Bt, Bp,
                       path='data/helios_1/ns=True/magnetic/resolution=100'):
    """
    Save the magnetic field components for plotting of already calculated data.

    Parameters
    ----------
    Br : numpy.ndarray.float64
        Array containing data for the radial component of the magnetic field.
    Bt : numpy.ndarray.float64
        DESCRIPTION.
    Bp : numpy.ndarray.float64
        DESCRIPTION.
    path : string, optional
        Relative path from base directory 'code' where the data is stored.
        Contains information about parameters used for the calculation.
        The default is '/data/helios_1/ns=True/magnetic/resolution=100'.

    Returns
    -------
    None.

    """
    savetxt(path + '_Br.gz', ravel(Br))
    savetxt(path + '_Bt.gz', ravel(Bt))
    savetxt(path + '_Bp.gz', ravel(Bp))

    print("Finished saving field components for given resolution.")
