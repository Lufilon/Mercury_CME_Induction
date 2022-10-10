# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 10:24:28 2022

@author: Luis-
"""

import sys
sys.path.append('C:/Users/Luis-/OneDrive/Dokumente/Tu Braunschweig/6. Semester'
                + '/Bachelorarbeit/code/KTH_Model_V7')
import json
import matplotlib.pyplot as plt
from numpy import array, linspace, meshgrid, ravel, zeros, asarray
from numpy import nanmax, nanmin, savetxt, loadtxt, pi, nan, isnan, hypot
from kth_start import kth_start
from cme_data_input import cme_data_input
from SHA_by_integration import SHA_by_integration
from signal_processing import signal_processing
# from signal_processing import fft_own, rebuild  # das so versuchen einzubauen
from rikitake_base import rikitake_calc, rikitake_plot
from time import time
t0 = time()
"""
Routine to calculate the external magnetic field components induced by a CME.
"""

# basic parameters for the routine -> runtime saving
RESOLUTION = 100
DEGREE_MAX = 2
REQ_RESOLUTION = 100
REQ_DEGREE_MAX = 2

# parameters for the kth-modell
DIPOLE, NEUTRALSHEET = True, False
INTERNAL, EXTERNAL = True, True
settings = [DIPOLE, NEUTRALSHEET, False, INTERNAL, EXTERNAL]

# which parts of the routine are performed
IMPORTANDANGLE, PLOTCME = False, False
CALCMAGNETIC, LOADMAGNETIC, PLOTMAGNETIC = False, False, False
CALCGAUSSIAN_t, LOADGAUSSIAN_t, PLOTGAUSSIAN_t = False, False, False
UPSCALING, CALCGAUSSIAN_f, PLOTGAUSSIAN_f = False, True, True
RIKITAKE, PLOTRIKITAKE = False, False

# gaussian to be fourier transformed. Tupel is (l, m), l=degree, m=order
GAUSS_LIST_INT = [(1, 0), (1, 1), (2, 0)]
GAUSS_LIST_EXT = [(1, 0), (1, 1), (2, 0)]

# number of frequencies used for the rikitake calculation
FREQNR = 20

"""
TODO
"""
# specifiy mercuries layers - high and low conductivity cases
r_arr = array([0, 1740E3, 1940E3, 2040E3, 2300E3, 2440E3])
sigma_arr_h = array([0, 1E7, 1E3, 10**0.5, 10**0.7, 1E-2])
sigma_arr_l = array([0, 1E5, 1E2, 10**-0.5, 10**-3, 1E-7])

# magnetic field degree for the rikitake calculation
"""
TODO
"""

# magnetic field degree for the rikitake plot
RIKITAKEDEGREE = 1

# paths to directories
CME_FILE_NAME = 'output-helios1_e1_np_helios1_e1_vp_0_1_2_1980148000000000'
CME_BASE_DIRECTORY = 'data/'
RUNTIME_DIRECTORY = CME_BASE_DIRECTORY + 'runtime/'
CME_BASE_DIRECTORY += CME_FILE_NAME + '/'
CME_FILE_NAME += '.txt'
EMPTY_ROWS = 89
CME_SETTINGS_DIRECTORY = CME_BASE_DIRECTORY + 'settings=' + str(settings) + '/'
RUNTIME_MAGNETIC_DIRECTORY = CME_SETTINGS_DIRECTORY + 'runtime/magnetic/'
RUNTIME_GAUSSIAN_t_DIRECTORY = CME_SETTINGS_DIRECTORY + 'runtime/gaussian_t/'
RUNTIME_GAUSSIAN_f_DIRECTORY = CME_SETTINGS_DIRECTORY + 'runtime/gaussian_f/'

if IMPORTANDANGLE:
    # =========================================================================
    # Import cme_data and calculate the pseudo_distance later used as r_hel.
    # =========================================================================
    t, R_SS, r_hel = cme_data_input(
        CME_BASE_DIRECTORY+CME_FILE_NAME, EMPTY_ROWS)
    t_steps = len(r_hel)
    possible_distance = linspace(nanmin(r_hel), nanmax(r_hel), RESOLUTION)

    print("Finished importing the cme data.")

    # =========================================================================
    # Create angular data for 200x400 points on a sphere.
    # =========================================================================
    N_THETA = int(200)
    N_PHI = int(2*N_THETA)

    num_pts = int(N_THETA * N_PHI)
    theta_arr = linspace(0, pi, N_THETA, endpoint=False)
    phi_arr = linspace(0, 2*pi, N_PHI, endpoint=False)
    phi_arr_2D, theta_arr_2D = meshgrid(phi_arr, theta_arr)
    theta, phi = ravel(theta_arr_2D), ravel(phi_arr_2D)
    with open('data/angle_data.json', 'w') as f:
        json.dump({"N_THETA": int(N_THETA), "N_PHI": int(N_PHI),
                   "num_pts": int(num_pts), "theta_arr": list(theta_arr),
                   "phi_arr": list(phi_arr), "theta": list(theta),
                   "phi": list(phi)}, f)

    print("Finished creating the angular data.")

if PLOTCME:
    # =========================================================================
    # Plot the time-dependant distance of the subsolar point.
    # =========================================================================
    try:
        PLT_SUPTITLE = "Data from Helios 1 at $r_{hel} = 0.31$ AU"
        PLT_TITLE = "1980-05-28T00:00:00.000 - 1980-06-01T23:59:00.000"
        plt.scatter(t, R_SS, s=4)
        plt.suptitle(PLT_SUPTITLE)
        plt.title(PLT_TITLE)
        plt.xlabel("$t$ [$s$]")
        plt.ylabel("$R_SS}$ [$R_{Mercury}$]")

    except NameError:
        print("Import CME and create angular data first.")

if CALCMAGNETIC:
    # =========================================================================
    # Calculate the internal dipole and neutralsheet magnetic field, as they
    # are the same for every R_SS distance.
    # =========================================================================
    try:
        B_r_di_int = loadtxt(RUNTIME_DIRECTORY + 'B_r_di_int.gz')
        B_theta_di_int = loadtxt(RUNTIME_DIRECTORY + 'B_theta_di_int.gz')
        B_phi_di_int = loadtxt(RUNTIME_DIRECTORY + 'B_phi_di_int.gz')
        B_r_ns_int = loadtxt(RUNTIME_DIRECTORY + 'B_r_ns_int.gz')
        B_theta_ns_int = loadtxt(RUNTIME_DIRECTORY + 'B_theta_ns_int.gz')
        B_phi_ns_int = loadtxt(RUNTIME_DIRECTORY + 'B_phi_ns_int.gz')
        B_r_ns_ext = loadtxt(RUNTIME_DIRECTORY + 'B_r_ns_int.gz')
        B_theta_ns_ext = loadtxt(RUNTIME_DIRECTORY + 'B_theta_ns_int.gz')
        B_phi_ns_ext = loadtxt(RUNTIME_DIRECTORY + 'B_phi_ns_int.gz')

        print("Finished loading the precalculated parts from file.")

    except FileNotFoundError:
        B_r_di_int, B_theta_di_int, B_phi_di_int = kth_start(
            1, [True, False, False, True, False])
        B_r_ns_int, B_theta_ns_int, B_phi_ns_int = kth_start(
            1, [False, True, False, True, False])
        B_r_ns_ext, B_theta_ns_ext, B_phi_ns_ext = kth_start(
            1, [False, True, False, False, True])

        savetxt(RUNTIME_DIRECTORY + 'B_r_di_int.gz', B_r_di_int)
        savetxt(RUNTIME_DIRECTORY + 'B_theta_di_int.gz', B_theta_di_int)
        savetxt(RUNTIME_DIRECTORY + 'B_phi_di_int.gz', B_phi_di_int)
        savetxt(RUNTIME_DIRECTORY + 'B_r_ns_int.gz', B_r_ns_int)
        savetxt(RUNTIME_DIRECTORY + 'B_theta_ns_int.gz', B_theta_ns_int)
        savetxt(RUNTIME_DIRECTORY + 'B_phi_ns_int.gz', B_phi_ns_int)
        savetxt(RUNTIME_DIRECTORY + 'B_r_ns_ext.gz', B_r_ns_ext)
        savetxt(RUNTIME_DIRECTORY + 'B_theta_ns_ext.gz', B_theta_ns_ext)
        savetxt(RUNTIME_DIRECTORY + 'B_phi_ns_ext.gz', B_phi_ns_ext)

        print("Finished calculating internal dipole and neutralsheet fields.")

    # =========================================================================
    # Use the calculated pseudo_distances to calculate the magnetic field.
    # =========================================================================
    B_r_possible = zeros((RESOLUTION, N_THETA*N_PHI))
    B_theta_possible = zeros((RESOLUTION, N_THETA*N_PHI))
    B_phi_possible = zeros((RESOLUTION, N_THETA*N_PHI))

    possible_distance = linspace(nanmin(r_hel), nanmax(r_hel), RESOLUTION)

    if DIPOLE:
        if INTERNAL:
            for i in range(len(possible_distance)):
                B_r_possible[i] += B_r_di_int
                B_theta_possible[i] += B_theta_di_int
                B_phi_possible[i] += B_phi_di_int
        if EXTERNAL:
            for i in range(len(possible_distance)):
                result = kth_start(possible_distance[i],
                                   [DIPOLE, False, False, False, EXTERNAL])
                B_r_possible[i] += result[0]
                B_theta_possible[i] += result[1]
                B_phi_possible[i] += result[2]

        print("Finished calculating field components for given resolution.")

    if NEUTRALSHEET:
        if INTERNAL:
            for i in range(len(possible_distance)):
                B_r_possible[i] += B_r_ns_int
                B_theta_possible[i] += B_theta_ns_int
                B_phi_possible[i] += B_phi_ns_int
        if EXTERNAL:
            for i in range(len(possible_distance)):
                B_r_possible[i] += B_r_ns_ext
                B_theta_possible[i] += B_theta_ns_ext
                B_phi_possible[i] += B_phi_ns_ext

    print("Finished adding precalculated field components to calculated field.")

    savetxt(RUNTIME_MAGNETIC_DIRECTORY + 'RESOLUTION=' + str(RESOLUTION) +
            '_B_r.gz', ravel(B_r_possible))
    savetxt(RUNTIME_MAGNETIC_DIRECTORY + 'RESOLUTION=' + str(RESOLUTION) +
            '_B_theta.gz', ravel(B_theta_possible))
    savetxt(RUNTIME_MAGNETIC_DIRECTORY + 'RESOLUTION=' + str(RESOLUTION) +
            '_B_phi.gz', ravel(B_phi_possible))

    print("Finished saving field components for given resolution.")

elif LOADMAGNETIC:
    # =========================================================================
    # Load the magnetic field components for the given requested resolution.
    # =========================================================================
    try:
        B_r_possible = loadtxt(
            RUNTIME_MAGNETIC_DIRECTORY + 'RESOLUTION=' + str(REQ_RESOLUTION)
            + '_B_r.gz')
        B_theta_possible = loadtxt(
            RUNTIME_MAGNETIC_DIRECTORY + 'RESOLUTION=' + str(REQ_RESOLUTION)
            + '_B_theta.gz')
        B_phi_possible = loadtxt(
            RUNTIME_MAGNETIC_DIRECTORY + 'RESOLUTION=' + str(REQ_RESOLUTION)
            + '_B_phi.gz')

        B_r_possible = B_r_possible.reshape((REQ_RESOLUTION, num_pts))
        B_theta_possible = B_theta_possible.reshape((REQ_RESOLUTION, num_pts))
        B_phi_possible = B_phi_possible.reshape((REQ_RESOLUTION, num_pts))

        print("Finished importing the magnetic field components with " +
              "Resolution=" + str(REQ_RESOLUTION) + ".")

    except FileNotFoundError:
        print("No magnetic field for this r_hel resolution was calculated yet")

if PLOTMAGNETIC:
    # =========================================================================
    # Plot the data to a mollweide projection if requested
    # =========================================================================
    try:
        possible_distance = linspace(nanmin(r_hel), nanmax(r_hel), RESOLUTION)

        for i in [0]:
            B = hypot(hypot(B_r_possible[i], B_theta_possible[i]), B_phi_possible[i])

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='mollweide')
            ax.set_title("r_hel = " + str(possible_distance[i]))
            plt.suptitle("dipole, neutralsheet, prc, internal, external = "
                          + str(settings))
            ax.grid()
            lon, lat = phi_arr - pi, pi/2 - theta_arr
            im = ax.contourf(lon, lat, B.reshape(N_THETA, N_PHI), levels=10)
            cbar = plt.colorbar(im)
            cbar.set_label('$B$ [$nT$]')
            plt.savefig('plots/mollweide.png', dpi=600)

    except NameError:
        print("Import or calculate magnetic field data first.")

if CALCGAUSSIAN_t:
    # =========================================================================
    # Calculate the gaussian-coefficients via a spherical harmonic analysis.
    # =========================================================================
    R_M = 2440
    ref_radius = R_M
    ana_radius = ref_radius

    coeff_int_t_possible = zeros((RESOLUTION, DEGREE_MAX+1, DEGREE_MAX+1))
    coeff_ext_t_possible = zeros((RESOLUTION, DEGREE_MAX+1, DEGREE_MAX+1))

    try:
        for i in range(RESOLUTION):
            for m in range(DEGREE_MAX + 1):
                coeff_int_t_possible[i][m], coeff_ext_t_possible[i][m] = SHA_by_integration(
                    B_r_possible[i], B_theta_possible[i], B_phi_possible[i],
                    ana_radius, ref_radius, DEGREE_MAX, m
                    )

        print("Finished calculating the time dependant gaussian coefficients" +
              "for the given resolution using the SHA by integration.")

        savetxt(RUNTIME_GAUSSIAN_t_DIRECTORY + 'RESOLUTION=' + str(RESOLUTION)
                + '_DEGREE_MAX=' + str(DEGREE_MAX) + '_internal.gz',
                coeff_int_t_possible.reshape(coeff_int_t_possible.shape[0], -1))
        savetxt(RUNTIME_GAUSSIAN_t_DIRECTORY + 'RESOLUTION=' + str(RESOLUTION)
                + '_DEGREE_MAX=' + str(DEGREE_MAX) + '_external.gz',
                coeff_ext_t_possible.reshape(coeff_ext_t_possible.shape[0], -1))

        print("Finished saving the time dependant gaussian coefficients.")

    except NameError:
        print("Calculate or import the magnetic field data first.")
        del coeff_int_t_possible, coeff_ext_t_possible

elif LOADGAUSSIAN_t:
    # =========================================================================
    # Import time dependant gaussians for given resolution and maximum degree.
    # =========================================================================
    try:
        coeff_int_t_possible = loadtxt(
            RUNTIME_GAUSSIAN_t_DIRECTORY + 'RESOLUTION=' + str(REQ_RESOLUTION)
            + '_DEGREE_MAX=' + str(REQ_DEGREE_MAX) + '_internal.gz')
        coeff_ext_t_possible = loadtxt(
            RUNTIME_GAUSSIAN_t_DIRECTORY + 'RESOLUTION=' + str(REQ_RESOLUTION)
            + '_DEGREE_MAX=' + str(REQ_DEGREE_MAX) + '_external.gz')

        coeff_int_t_possible = coeff_int_t_possible.reshape(
            REQ_RESOLUTION, pow(REQ_DEGREE_MAX, 2) // REQ_DEGREE_MAX+1,
            REQ_DEGREE_MAX+1)
        coeff_ext_t_possible = coeff_ext_t_possible.reshape(
            REQ_RESOLUTION, pow(REQ_DEGREE_MAX, 2) // REQ_DEGREE_MAX+1,
            REQ_DEGREE_MAX+1)

        print("Finished importing the time dependant gaussian coefficiens "
              + "with Resolution=" + str(REQ_RESOLUTION) + ".")

    except FileNotFoundError:
        print("No file for this combination of pseudo_distance resolution" +
              "and max degree of spherical analyzed gaussians was stored" +
              "for the given CME and given kth-modell parameters.")

if UPSCALING:
    # =========================================================================
    # Assign the values to the data points with the smallest deviation
    # =========================================================================
    possible_distance = linspace(nanmin(r_hel), nanmax(r_hel), RESOLUTION)
    
    coeff_int_t = zeros((t_steps, DEGREE_MAX+1, DEGREE_MAX+1))
    coeff_ext_t = zeros((t_steps, DEGREE_MAX+1, DEGREE_MAX+1))

    for i in range(t_steps):
        if not isnan(r_hel[i]):
            nearest_distance_index = (abs(possible_distance-r_hel[i])).argmin()
            coeff_int_t[i] = coeff_int_t_possible[nearest_distance_index]
            coeff_ext_t[i] = coeff_ext_t_possible[nearest_distance_index]
        else:
            coeff_int_t[i] = nan
            coeff_ext_t[i] = nan

    print("Finished upscaling of the lower resolution solution time data.")

if PLOTGAUSSIAN_t:
    # =========================================================================
    # Plot internal and external time-dependant inducing gaussians.
    # =========================================================================
    try:
        f, ax = plt.subplots(2, 1)
        f.suptitle("Internal and external time-dependant inducing gaussians.")
        f.supxlabel("$t$ [$s$]")
        f.supylabel("$A$ [$nT$]")

        for l, m in GAUSS_LIST_INT:
            ax[0].plot([coeff_int_t[i][m][l] for i in range(t_steps)],
                       label="l = " + str(l) + ", m = " + str(m) + "_int")

        for l, m in GAUSS_LIST_EXT:
            ax[1].plot([coeff_ext_t[i][m][l] for i in range(t_steps)],
                       label="l = " + str(l) + ", m = " + str(m) + "_ext")

        ax[0].legend()
        ax[1].legend()

    except NameError:
        print("Import or calculate time-dependant gaussian data first.")

if CALCGAUSSIAN_f:
    # =========================================================================
    # Fourier transform the coefficients to the frequency domain
    # =========================================================================
    f = zeros((len(GAUSS_LIST_EXT), FREQNR))
    coeff_ext_f = zeros((len(GAUSS_LIST_EXT), FREQNR))
    phase = zeros((len(GAUSS_LIST_EXT), FREQNR))
    # f = zeros((len(GAUSS_LIST_EXT), t_steps//2 +1 ))
    # coeff_ext_f = zeros((len(GAUSS_LIST_EXT), t_steps//2 +1 ))
    # phase = zeros((len(GAUSS_LIST_EXT), t_steps//2 +1 ))

    for l, m in GAUSS_LIST_EXT:
        index = GAUSS_LIST_EXT.index((l, m))
        f[index], coeff_ext_f[index], phase[index] = signal_processing(
            t, asarray([coeff_ext_t[i][m][l] for i in range(t_steps)]),
            FREQNR, t_steps, (l, m), PLOTGAUSSIAN_f)
        # """
        # TODO
        # """
        # import numpy as np
        # posIndices = np.flip(coeff_ext_f.argsort()[-FREQNR:])
        # mask = np.isin(coeff_ext_f[index], coeff_ext_f[index][posIndices], invert=True)
        # coeff_ext_f[index][mask] = 0
        # plt.plot(coeff_ext_f[index])
        # only for showing which frequencies where choosed
        # f, coeff_ext_f, phase = f[posIndices], coeff_ext_f[posIndices], phase[posIndices]

    print("Finished fourier transforming the external gaussian coefficients.")

# if PLOTGAUSSIAN_f:
    # 

if RIKITAKE:
    # =========================================================================
    # rikitake zeug
    # =========================================================================
    """
    TODO
    """
    # for i in GAUSS_LIST_EXT:  # so ungefähr. Dann muss f und amp array aber 2D
    #     for j in freq_list:
            # TODO
    import numpy as np
    a, b = rikitake_calc(1, 1, r_arr, sigma_arr_h, sigma_arr_l)
    print(a, b)

if PLOTRIKITAKE:
    """
    TODO
    """
    rikitake_plot(RIKITAKEDEGREE)


t1 = time()
print("Time for the Process: " + str(t1 - t0) + " seconds.")
