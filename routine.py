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
from numpy import array, linspace, meshgrid, ravel, zeros, asarray, flip, isin
from numpy import nanmax, nanmin, savetxt, loadtxt, pi, nan, isnan, hypot, exp, arctan2
from kth_start import kth_start
from cme_data_input import cme_data_input
from SHA_by_integration import SHA_by_integration
from signal_processing import fft_own, rebuild
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
MAGNETICDATA, PLOTMAGNETICDATA = False, False
GAUSSIAN_t, PLOTGAUSSIAN_t = False, False
GAUSSIAN_f, PLOTGAUSSIAN_f = False, False
RIKITAKE, PLOTRIKITAKE = False, False

# gaussian to be fourier transformed. Tupel is (l, m), l=degree, m=order
GAUSS_LIST_INT = [(1, 0), (1, 1), (2, 0)]
GAUSS_LIST_EXT = [(1, 0), (1, 1), (2, 0)]

# number of frequencies used for the rikitake calculation - max: t_steps//2 + 1
FREQNR = 20

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

if MAGNETICDATA:
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
        print("No magnetic field for this r_hel resolution was calculated yet"
              + " - Started the calculation.")
        # =====================================================================
        # Calculate the internal dipole and neutralsheet magnetic field, as
        # they are the same for every R_SS distance.
        # =====================================================================
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

            print("Finished calculating dipole_int and neutralsheet fields.")

        # =====================================================================
        # Use the calculated pseudo_distances to calculate the magnetic field.
        # =====================================================================
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

        print("Finished adding precalculated field components to field.")

        savetxt(RUNTIME_MAGNETIC_DIRECTORY + 'RESOLUTION=' + str(RESOLUTION) +
                '_B_r.gz', ravel(B_r_possible))
        savetxt(RUNTIME_MAGNETIC_DIRECTORY + 'RESOLUTION=' + str(RESOLUTION) +
                '_B_theta.gz', ravel(B_theta_possible))
        savetxt(RUNTIME_MAGNETIC_DIRECTORY + 'RESOLUTION=' + str(RESOLUTION) +
                '_B_phi.gz', ravel(B_phi_possible))

        print("Finished saving field components for given resolution.")

if PLOTMAGNETICDATA:
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

if GAUSSIAN_t:
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
        
        # =====================================================================
        # Calculate gaussian-coefficients via a spherical harmonic analysis.
        # =====================================================================
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

    print("Finished upscaling the lower resolution time dependant gaussians.")

if PLOTGAUSSIAN_t:
    # =========================================================================
    # Plot internal and external time-dependant inducing gaussians.
    # =========================================================================
    try:
        fig, ax = plt.subplots(1, 1)
        fig.suptitle("External time-dependant inducing gaussians.")
        fig.supxlabel("$t$ [$s$]")
        fig.supylabel("$A$ [$nT$]")

        # for l, m in GAUSS_LIST_EXT:
        for l, m in [(1, 0)]:
            ax.plot(t, [coeff_ext_t[i][m][l] for i in range(t_steps)],
                       label="l = " + str(l) + ", m = " + str(m) + "_ext")

        ax.legend()

    except NameError:
        print("Import or calculate time-dependant gaussian data first.")

if GAUSSIAN_f:
    # =========================================================================
    # Fourier transform the coefficients to the frequency domain
    # =========================================================================    
    f = zeros((len(GAUSS_LIST_EXT), t_steps//2 + 1))
    coeff_ext_f = zeros((len(GAUSS_LIST_EXT), t_steps//2 + 1))
    phase = zeros((len(GAUSS_LIST_EXT), t_steps//2 + 1))
    
    relIndices = zeros((len(GAUSS_LIST_EXT), FREQNR), dtype='int')

    for l, m in GAUSS_LIST_EXT:
        index = GAUSS_LIST_EXT.index((l, m))
        f[index], coeff_ext_f[index], phase[index] = fft_own(
            t, t_steps, asarray([coeff_ext_t[i][m][l] for i in range(t_steps)]))

        # get relevant frequencies, filter f_0 = 0 Hz beforehand
        relIndices[index] = flip(coeff_ext_f[index].argsort()[-FREQNR:])
        mask = isin(coeff_ext_f[index], coeff_ext_f[index][relIndices[index]],
                    invert=True)
        coeff_ext_f[index][mask] = 0

    print("Finished fourier transforming the external gaussian coefficients.")

if PLOTGAUSSIAN_f:
    fig, ax = plt.subplots(2, 1)
    fig.suptitle("FFT and rebuild via chosen frequencies of external gaussians.")
    
    for l, m in GAUSS_LIST_EXT:
        index = GAUSS_LIST_EXT.index((l, m))
        ax[0].plot(f[index], coeff_ext_f[index], label="fft_l=" + str(l) + "_m=" + str(m))

        coeff_ext_t_rebuild = rebuild(
            t, f[index], coeff_ext_f[index], phase[index])

        ax[1].plot(t, [coeff_ext_t[i][m][l] for i in range(t_steps)], label="original")
        ax[1].plot(t, coeff_ext_t_rebuild, label="rebuild")

    ax[0].set_xlabel("$f$ [$Hz$]")
    ax[0].set_ylabel("$g$_(" + str(l) + ", " + str(m) + ") [$nT$]")
    ax[1].set_xlabel("$t$ [$s$]")
    ax[1].set_ylabel("$g$_(" + str(l) + ", " + str(m) + ") [$nT$]")
    ax[0].legend()
    ax[1].legend()

if RIKITAKE:
    # =========================================================================
    # calculation of the rikitake-factor for each selected frequency for both
    # the high and low condutivity model.
    # =========================================================================
    # for l, m in GAUSS_LIST_EXT:
    for l, m in [(1, 0)]:
        index = GAUSS_LIST_EXT.index((l, m))

        rikitake_h_real = zeros(t_steps//2 + 1)
        rikitake_h_imag = zeros(t_steps//2 + 1)
        rikitake_l_real = zeros(t_steps//2 + 1)
        rikitake_l_imag = zeros(t_steps//2 + 1)

        for i in range(t_steps//2 + 1):
            if i in relIndices and i > 0:
                rikitake_h_real[i], rikitake_h_imag[i], rikitake_l_real[i], rikitake_l_imag[i] = rikitake_calc(
                    l, f[index][i], r_arr, sigma_arr_h, sigma_arr_l)
      
        amp_h = coeff_ext_f[index] * hypot(rikitake_h_real, rikitake_h_imag)
        amp_h *= exp(arctan2(rikitake_h_imag, rikitake_h_real))
        amp_l = coeff_ext_f[index] * hypot(rikitake_h_real, rikitake_h_imag)
        amp_l *= exp(arctan2(rikitake_l_imag, rikitake_l_real))

        induced_h = rebuild(t, f[index], amp_h, phase[index])
        induced_l = rebuild(t, f[index], amp_l, phase[index])

        fig, ax = plt.subplots(1, 1)
        ax.plot(t, induced_h, label="high_l=" + str(l) + ", m =" + str(m))
        ax.plot(t, induced_l, label="low_l=" + str(l) + ", m =" + str(m))
        fig.suptitle("Induced time-dependant gaussians.")
        fig.supxlabel("$t$ [$s$]")
        fig.supylabel("$A$ [$nT$]")
        ax.legend()

if PLOTRIKITAKE:
    """
    TODO
    """
    rikitake_plot(RIKITAKEDEGREE)


print("Time for the Process: " + str(time()- t0) + " seconds.")
