# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 10:34:04 2022

@author: Luis-
"""

from numpy import sin, cos, full, loadtxt, savetxt, ravel, zeros
from numpy import linspace, nanmin, nanmax, hypot, pi
from kth14_model_for_mercury_v7b import kth14_model_for_mercury_v7b
import matplotlib.pyplot as plt


def magnetic_field_get(possible_distances, R_ss, theta, phi, num_theta, num_phi,
                      resolution=100, settings=[True, True, False, True, True],
                      plot=False, runtime_dir='/data/runtime',
                      path='data/helios_1/ns=True/magnetic/resolution='):
    """
    Calculate the magnetic field on a grid with the kth22-model.

    Parameters
    ----------
    possible_distances : numpy.ndarray.float64
        Heliocentric (pseudo) distance.
    R_ss : numpy.ndarray.float64
        Subsolar standoff distance.
    theta : numpy.ndarray.float64
        Lattitude of the data.
    phi : numpy.ndarray.float64
        Longitude of the data.
    num_theta : int, optional
        Number of points in latteral direction. The default is 200.
    num_phi : int, optional
        Number of points in longitudinal  direction. The default is 400.
    resolution : int, optional
        Number of distances for which the magnetic field is calculated for.
        The default is 100.
    settings : list.boolean, optional
        Parameters for the kth22-modell.
        [dipole, neutralsheet, pcr, internal, external].
        The default is [True, True, False, True, True].
    plot : boolean, optional
        Controls whether or not the magnetic field ist plotted.
        The default is False.
    runtime_dir : string, optional
        Path to the directoy where the di_int and ns_int are saved for runtime.
        The default is '/data/runtime'.
    path : string, optional
        Path to the directory where the magnetic field data for the given com-
        bination of resolution and settings ist stored.
        The default is 'data/helios_1/ns=True/magnetic/resolution='.

    Returns
    -------
    Br_possible : numpy.ndarray.float64
        r-component of the magnetic field for (num_theta*num_phi)-points.
    Bt_possible : numpy.ndarray.float64
        theta-component of the magnetic field for (num_theta*num_phi)-points.
    Bp_possible : numpy.ndarray.float64
        phi-component of the magnetic field for (num_theta*num_phi)-points.

    """
    try:
        print("Importing precalculated magnetic field components with " +
              "resolution=" + str(resolution) + " if available.")
        num_pts = num_theta * num_phi

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

        # di_int and ns_int are the same for every R_ss distance
        try:
            Br_di_int = loadtxt(runtime_dir + 'Br_di_int.gz')
            Bt_di_int = loadtxt(runtime_dir + 'Bt_di_int.gz')
            Bp_di_int = loadtxt(runtime_dir + 'Bp_di_int.gz')
            Br_ns_int = loadtxt(runtime_dir + 'Br_ns_int.gz')
            Bt_ns_int = loadtxt(runtime_dir + 'Bt_ns_int.gz')
            Bp_ns_int = loadtxt(runtime_dir + 'Bp_ns_int.gz')

            print("Finished loading the precalculated parts from file.")

        except OSError:
            Br_di_int, Bt_di_int, Bp_di_int = magnetic_field_calc(
                possible_distances[0], theta, phi, num_pts,
                [True, False, False, True, False], 50.)
            Br_ns_int, Bt_ns_int, Bp_ns_int = magnetic_field_calc(
                possible_distances[0], theta, phi, num_pts,
                [False, True, False, True, False], 50.)

            savetxt(runtime_dir + 'Br_di_int.gz', Br_di_int)
            savetxt(runtime_dir + 'Bt_di_int.gz', Bt_di_int)
            savetxt(runtime_dir + 'Bp_di_int.gz', Bp_di_int)
            savetxt(runtime_dir + 'Br_ns_int.gz', Br_ns_int)
            savetxt(runtime_dir + 'Bt_ns_int.gz', Bt_ns_int)
            savetxt(runtime_dir + 'Bp_ns_int.gz', Bp_ns_int)

            print("Finished calculating dipole_int and neutralsheet_int.")

        # use the calculated pseudo_distances to calculate the magnetic field.
        Br_possible = zeros((resolution, num_pts))
        Bt_possible = zeros((resolution, num_pts))
        Bp_possible = zeros((resolution, num_pts))

        if settings[0] and settings[3]:
            Br_possible += Br_di_int
            Bt_possible += Bt_di_int
            Bp_possible += Bp_di_int

        if settings[1] and settings[3]:
            Br_possible += Br_ns_int
            Bt_possible += Bt_ns_int
            Bp_possible += Bp_ns_int

        if settings[4] and settings[0] or settings[1]:
            for i, val in enumerate(possible_distances):
                result = magnetic_field_calc(
                    val, theta, phi, num_pts,
                    [settings[0], settings[1], False, False, True])
                Br_possible[i] += result[0]
                Bt_possible[i] += result[1]
                Bp_possible[i] += result[2]

        print("Finished calculating field components for given resolution.")

        magnetic_field_save(Br_possible, Bt_possible, Bp_possible,
                           path + str(resolution))

    magnetic_field_plot(Br_possible, Bt_possible, Bp_possible, R_ss, theta, phi,
                       num_theta, num_phi, resolution, settings)

    return Br_possible, Bt_possible, Bp_possible


def magnetic_field_calc(possible_distance, theta, phi, num_pts=80000,
                       settings=[True, True, False, True, True],
                       di_val=50.):
    """
    Helper method that tranforms the data to msm coordinateed, forwards it to
    the kth22-model and re-transforms the magnetic fiel data.

    Parameters
    ----------
    possible_distance : float64
        Heliocentric (pseudo) distance for which the field is calculated.
    theta : numpy.ndarray.float64
        Lattitude of the data.
    phi : numpy.ndarray.float64
        Longitude of the data.
    num_pts : int, optional
        Number of data points for which the field is calculated.
        The default is 80000.
    settings : list.boolean, optional
        Parameters for the kth22-modell.
        The default is [True, True, False, True, True].
    di_val : float, optional
        Disturbance index.
        The default is 50..

    Returns
    -------
    Br : TYPE
        Radial magnetic field component for data points at given r_hel.
    Bt : TYPE
        Latitudinal magnetic field component for data points at given r_hel.
    Bp : TYPE
        Longitudinal magnetic field component for data points at given r_hel.

    """
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
        x, y, z, possible_distance, di, CONTROL_PARAM_PATH, FIT_PARAM_PATH,
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


def magnetic_field_plot(Br, Bt, Bp, R_ss, theta, phi,
                       num_theta=200, num_phi=400, resolution=100,
                       settings=[True, True, False, True, True]):
    """
    Plot the data to a mollweide projection for a given R_ss for a given grid.

    Parameters
    ----------
    Br : numpy.ndarray.float64^
        r-component of the magnetic field for (num_theta*num_phi)-points.
    Bt : numpy.ndarray.float64
        theta-component of the magnetic field for (num_theta*num_phi)-points.
    Bp : numpy.ndarray.float64
        phi-component of the magnetic field for (num_theta*num_phi)-points.
    R_ss : numpy.ndarray.float64
        Subsolar standoff distance.
    theta : numpy.ndarray.float64
        Lattitude of the data.
    phi : numpy.ndarray.float64
        Longitude of the data.
    num_theta : int, optional
        Number of points in latteral direction. The default is 200.
    num_phi : int, optional
        Number of points in longitudinal  direction. The default is 400.
    resolution : int, optional
        Number of distances for which the magnetic field is calculated for.
        The default is 100.
    settings : list.boolean, optional
        Parameters for the kth22-modell.
        [dipole, neutralsheet, pcr, internal, external].
        The default is [True, True, False, True, True].

    Returns
    -------
    None.

    """
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
        lon, lat = phi[:400] - pi, pi/2 - theta[::400]
        im = plt.contourf(lon, lat, B.reshape(num_theta, num_phi),
                          levels=linspace(0, 700, 8, endpoint=True),
                          extend='both')
        cbar = plt.colorbar(im)
        cbar.set_label('$B$ [$nT$]')

        plt.tight_layout()
        plt.savefig('plots/mollweide/' + str(i) + '.jpg', dpi=600)


def magnetic_field_save(Br, Bt, Bp,
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
