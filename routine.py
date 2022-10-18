# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 10:24:28 2022

@author: Luis-
"""

import json
import matplotlib.pyplot as plt
from numpy import array, linspace, meshgrid, ravel, zeros, asarray, flip, isin, real
from numpy import nanmax, nanmin, savetxt, loadtxt, pi, nan, isnan, hypot, exp, arctan2
from kth_start import kth_start
from cme_data_input import cme_data_input
from SHA_by_integration import SHA_by_integration
from signal_processing import fft_own, rebuild
from rikitake_base import rikitake_calc, rikitake_plot
from time import time
t0 = time()

# for the creation of the mollweide-video
# import matplotlib
# matplotlib.use('Agg')

"""
Routine for the calculation of the induced gaussians based on solarwind
condition over time.

Basic parameters:

    RESOLUTION:
        Sampling of the possible pseudo-distances.

    DEGREE_MAX:
        Maxmimal degree to which spherical harmonic analysis is performed.

Magnetic field parameters:

    REQ_RESOLUTION/REQ_DEGREE_MAX:
        Search for already calculated data for the given combination.

    DIPOLE/NEUTRALSHEET/INTERNAL/EXTERNAL:
        Parameters for the magnetosphere-model.

Which part of the routine is performed:

    IMPORTANDANGLE/PLOTCME:
        Import cme-data and create data points, defined by two angles.
        Plot the cme-data.

    MAGNETICDATA/PLOTMAGNETICDATA:
        Use the kth-model to calculate the magnetic field components for the
        given magnetic field parameters.
        Plot the magnitude of the magnetic field for the defined data points
        using a mollweide projection.

    GAUSSIAN_t/PLOTGAUSSIAN_t:
        Calculate the tim-dependant gaussian coefficients up to DEGREE_Max
        via spherical harmonic analysis.
        Plot the gaussians in respect to the time for each gaussian listed in
        GAUSS_LIST_EXT.

    GAUSSIAN_f/PLOTGAUSSIAN_f:
        Fourier transform the time-dependant gaussians via FFT to 
"""

# basic parameters for the routine -> runtime saving
RESOLUTION = 100
DEGREE_MAX = 2
REQ_RESOLUTION = 100
REQ_DEGREE_MAX = 2

# parameters for the kth-modell
DIPOLE, NEUTRALSHEET = True, True
INTERNAL, EXTERNAL = True, True
settings = [DIPOLE, NEUTRALSHEET, False, INTERNAL, EXTERNAL]

# which parts of the routine are performed
IMPORTANDANGLE, PLOTCME = False, False
MAGNETICDATA, PLOTMAGNETICDATA = False, False
GAUSSIAN_t, PLOTGAUSSIAN_t = False, False
GAUSSIAN_f, PLOTGAUSSIAN_f = False, False
RIKITAKE, PLOTRIKITAKE = False, False

# gaussian to be fourier transformed. Tupel is (l, m), l=degree, m=order
# GAUSS_LIST_EXT = [(1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]
GAUSS_LIST_EXT = [(1, 0), (2, 1)]

# number of frequencies used for the rikitake calculation - max: t_steps//2 + 1
FREQNR = 3601

# specifiy mercuries layers - high and low conductivity cases
r_arr = array([0, 1740E3, 1940E3, 2040E3, 2300E3, 2440E3])
sigma_arr_h = array([0, 1E7, 1E3, 10**0.5, 10**0.7, 1E-2])
sigma_arr_l = array([0, 1E5, 1E2, 10**-0.5, 10**-3, 1E-7])

# magnetic field degree for the frequency range rikitake plot
RIKITAKEDEGREE = [1, 2]

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

        fig_cme, ax_cme = plt.subplots()
        fig_cme.suptitle(PLT_SUPTITLE)
        ax_cme.set_title(PLT_TITLE)
        ax_cme.scatter(t/3600, R_SS, s=4)
        ax_cme.set_xlabel("$t$ [$h$]")
        ax_cme.set_ylabel("$R_SS}$ [$R_{Mercury}$]")

    except NameError:
        print("Set IMPORTANDANGLE=True.")

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

    except OSError:
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

            print("Finished loading the precalculated parts from file.")

        except OSError:
            B_r_di_int, B_theta_di_int, B_phi_di_int = kth_start(
                1, [True, False, False, True, False])
            B_r_ns_int, B_theta_ns_int, B_phi_ns_int = kth_start(
                1, [False, True, False, True, False])

            savetxt(RUNTIME_DIRECTORY + 'B_r_di_int.gz', B_r_di_int)
            savetxt(RUNTIME_DIRECTORY + 'B_theta_di_int.gz', B_theta_di_int)
            savetxt(RUNTIME_DIRECTORY + 'B_phi_di_int.gz', B_phi_di_int)
            savetxt(RUNTIME_DIRECTORY + 'B_r_ns_int.gz', B_r_ns_int)
            savetxt(RUNTIME_DIRECTORY + 'B_theta_ns_int.gz', B_theta_ns_int)
            savetxt(RUNTIME_DIRECTORY + 'B_phi_ns_int.gz', B_phi_ns_int)

            print("Finished calculating dipole_int and neutralsheet_int.")

        # =====================================================================
        # Use the calculated pseudo_distances to calculate the magnetic field.
        # =====================================================================
        B_r_possible = zeros((RESOLUTION, N_THETA*N_PHI))
        B_theta_possible = zeros((RESOLUTION, N_THETA*N_PHI))
        B_phi_possible = zeros((RESOLUTION, N_THETA*N_PHI))

        possible_distance = linspace(nanmin(r_hel), nanmax(r_hel), RESOLUTION)

        if INTERNAL:
            if DIPOLE:
                for i in range(len(possible_distance)):
                    B_r_possible[i] += B_r_di_int
                    B_theta_possible[i] += B_theta_di_int
                    B_phi_possible[i] += B_phi_di_int

            if NEUTRALSHEET:
                for i in range(len(possible_distance)):
                    B_r_possible[i] += B_r_ns_int
                    B_theta_possible[i] += B_theta_ns_int
                    B_phi_possible[i] += B_phi_ns_int

        if EXTERNAL:
            if DIPOLE and NEUTRALSHEET:
                for i in range(len(possible_distance)):
                    result = kth_start(possible_distance[i],
                                        [True, True, False, False, True])
                    B_r_possible[i] += result[0]
                    B_theta_possible[i] += result[1]
                    B_phi_possible[i] += result[2]

            elif DIPOLE:
                for i in range(len(possible_distance)):
                    result = kth_start(possible_distance[i],
                                        [True, False, False, False, True])
                    B_r_possible[i] += result[0]
                    B_theta_possible[i] += result[1]
                    B_phi_possible[i] += result[2]

            elif NEUTRALSHEET:
                for i in range(len(possible_distance)):
                    result = kth_start(possible_distance[i],
                                        [False, True, False, False, True])
                    B_r_possible[i] += result[0]
                    B_theta_possible[i] += result[1]
                    B_phi_possible[i] += result[2]

        print("Finished calculating field components for given resolution.")

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

        # for i in range(RESOLUTION):
        for i in [5]:
            B = hypot(B_phi_possible[i],
                hypot(B_r_possible[i], B_theta_possible[i]))

            plt.figure("Mollweide projection of magnetic field")
            plt.subplot(projection='mollweide')
            plt.title("r_hel = " + str(possible_distance[i]))
            plt.suptitle("dipole, neutralsheet, prc, internal, external = "
                          + str(settings))
            plt.grid()
            lon, lat = phi_arr - pi, pi/2 - theta_arr
            im = plt.contourf(lon, lat, B.reshape(N_THETA, N_PHI), levels=10)
            cbar = plt.colorbar(im)
            cbar.set_label('$B$ [$nT$]')
            plt.savefig('plots/mollweide/' + str(i) + '.png', dpi=600)

    except NameError:
        print("Set MAGNETICDATA=True.")

if GAUSSIAN_t:
    # =========================================================================
    # Import time dependant gaussians for given resolution and maximum degree.
    # =========================================================================
    try:
        coeff_ext_t_possible = loadtxt(
            RUNTIME_GAUSSIAN_t_DIRECTORY + 'RESOLUTION=' + str(REQ_RESOLUTION)
            + '_DEGREE_MAX=' + str(REQ_DEGREE_MAX) + '_external.gz')

        coeff_ext_t_possible = coeff_ext_t_possible.reshape(
            REQ_RESOLUTION, pow(REQ_DEGREE_MAX, 2) // REQ_DEGREE_MAX+1,
            REQ_DEGREE_MAX+1)

        print("Finished importing the time dependant gaussian coefficiens "
              + "with Resolution=" + str(REQ_RESOLUTION) + ".")

    except OSError:
        print("No file for this combination of pseudo_distance resolution " +
              "and max degree of spherical analyzed gaussians was stored " +
              "for the given CME and given kth-modell parameters. " +
              "- Started the calculation.")
        
        # =====================================================================
        # Calculate gaussian-coefficients via a spherical harmonic analysis.
        # =====================================================================
        R_M = 2440
        ref_radius = R_M
        ana_radius = ref_radius

        coeff_ext_t_possible = zeros((RESOLUTION, DEGREE_MAX+1, DEGREE_MAX+1))

        try:
            for i in range(RESOLUTION):
                for m in range(DEGREE_MAX + 1):
                    coeff_ext_t_possible[i][m] = SHA_by_integration(
                        B_r_possible[i], B_theta_possible[i], B_phi_possible[i],
                        ana_radius, ref_radius, DEGREE_MAX, m
                        )[1]

            print("Finished calculating the time dependant gaussian coefficients" +
                  "for the given resolution using the SHA by integration.")

            savetxt(RUNTIME_GAUSSIAN_t_DIRECTORY + 'RESOLUTION=' + str(RESOLUTION)
                    + '_DEGREE_MAX=' + str(DEGREE_MAX) + '_external.gz',
                    coeff_ext_t_possible.reshape(coeff_ext_t_possible.shape[0], -1))

            print("Finished saving the time dependant gaussian coefficients.")

        except NameError:
            print("Calculate or import the magnetic field data first.")

    # =========================================================================
    # Assign the values to the data points with the smallest deviation
    # =========================================================================
    possible_distance = linspace(nanmin(r_hel), nanmax(r_hel), RESOLUTION)

    coeff_ext_t = zeros((t_steps, DEGREE_MAX+1, DEGREE_MAX+1))

    for i in range(t_steps):
        if not isnan(r_hel[i]):
            nearest_distance_index = (abs(possible_distance-r_hel[i])).argmin()
            coeff_ext_t[i] = coeff_ext_t_possible[nearest_distance_index]
        else:
            coeff_ext_t[i] = nan

    print("Finished upscaling the lower resolution time dependant gaussians.")

if PLOTGAUSSIAN_t:
    # =========================================================================
    # Plot external time-dependant inducing gaussians.
    # =========================================================================
    try:
        fig_gauss_t_inducing, ax_gauss_t_inducing = plt.subplots(len(GAUSS_LIST_EXT))
        fig_gauss_t_inducing.suptitle("Time-dependant gaussian.")
        fig_gauss_t_inducing.supxlabel("$t$ [$h$]")
        fig_gauss_t_inducing.supylabel("$A$ [$nT$]")

        for l, m in GAUSS_LIST_EXT:
            index = GAUSS_LIST_EXT.index((l, m))

            ax_gauss_t_inducing[index].plot(
                t/3600, [coeff_ext_t[i][m][l] for i in range(t_steps)],
                label="l = " + str(l) + ", m = " + str(m) + "_ext")

            ax_gauss_t_inducing[index].set_ylabel("inducing")
            ax_gauss_t_inducing[index].legend(loc='upper left')

    except NameError:
        print("Set GAUSSIAN_t=True.")

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
    try:
        fig_gauss_f_inducing, ax_gauss_f_inducing = plt.subplots(2)
        fig_gauss_f_inducing.suptitle("Freq-dependant gaussians.")
        fig_gauss_f_inducing.supxlabel("$f$ [$Hz$]")
        fig_gauss_f_inducing.supylabel("$A$ [$nT^2 / Hz$]")

        for l, m in GAUSS_LIST_EXT:
            index = GAUSS_LIST_EXT.index((l, m))

            # for non log plot
            # ax_gauss_f_inducing.plot(f[index], coeff_ext_f[index],
                                      # label="l=" + str(l) + "_m=" + str(m) + "_inducing")
            # for log plot
            ax_gauss_f_inducing[index].plot(
                f[index][1:], coeff_ext_f[index][1:],
                label="l=" + str(l) + "_m=" + str(m) + "_inducing")

            ax_gauss_f_inducing[index].set_xscale('log')
            ax_gauss_f_inducing[index].set_ylabel("inducing")
            ax_gauss_f_inducing[index].legend()

    except NameError:
        print("Set GAUSSIAN_f=True.")

if RIKITAKE:
    # =========================================================================
    # calculation of rikitake-factor for each selected frequency for both
    # high and low condutivity model.
    # =========================================================================
    rikitake_h_real = zeros((len(GAUSS_LIST_EXT), t_steps//2 + 1))
    rikitake_h_imag = zeros((len(GAUSS_LIST_EXT), t_steps//2 + 1))
    rikitake_l_real = zeros((len(GAUSS_LIST_EXT), t_steps//2 + 1))
    rikitake_l_imag = zeros((len(GAUSS_LIST_EXT), t_steps//2 + 1))

    phase_rikitake_h = zeros((len(GAUSS_LIST_EXT), t_steps//2 + 1), dtype=complex)
    phase_rikitake_l = zeros((len(GAUSS_LIST_EXT), t_steps//2 + 1), dtype=complex)

    amp_rikitake_h = zeros((len(GAUSS_LIST_EXT), t_steps//2 + 1), dtype=complex)
    amp_rikitake_l = zeros((len(GAUSS_LIST_EXT), t_steps//2 + 1), dtype=complex)

    induced_h = zeros((len(GAUSS_LIST_EXT), t_steps))
    induced_l = zeros((len(GAUSS_LIST_EXT), t_steps))
    
    for l, m in GAUSS_LIST_EXT:
        index = GAUSS_LIST_EXT.index((l, m))

        for i in range(t_steps//2 + 1):
            if i in relIndices and i > 0:
                rikitake_h_real[index][i], rikitake_h_imag[index][i], rikitake_l_real[index][i], rikitake_l_imag[index][i] = rikitake_calc(
                    l, f[index][i], r_arr, sigma_arr_h, sigma_arr_l)
      
        phase_rikitake_h[index] = arctan2(rikitake_h_imag[index], rikitake_h_real[index])
        phase_rikitake_l[index] = arctan2(rikitake_l_imag[index], rikitake_l_real[index])
        
        amp_rikitake_h[index] = coeff_ext_f[index] * hypot(rikitake_h_real[index], rikitake_h_imag[index])
        amp_rikitake_l[index] = coeff_ext_f[index] * hypot(rikitake_h_real[index], rikitake_h_imag[index])
        amp_rikitake_h[index] = amp_rikitake_h[index] * exp(0+1j * phase_rikitake_h[index])
        amp_rikitake_l[index] = amp_rikitake_l[index] * exp(0+1j * phase_rikitake_l[index])

        induced_h[index] = rebuild(t, f[index], amp_rikitake_h[index], phase[index])
        induced_l[index] = rebuild(t, f[index], amp_rikitake_l[index], phase[index])

if PLOTRIKITAKE:
    try:
        # add to time-dependant plot
        ax_gauss_t_induced = array(
            [a.twinx() for a in ax_gauss_t_inducing.ravel()]).reshape(
                ax_gauss_t_inducing.shape)
        
        # add to frequency-dependant plot
        ax_gauss_f_induced = array(
            [a.twinx() for a in ax_gauss_f_inducing.ravel()]).reshape(
                ax_gauss_f_inducing.shape)
        
        # phase of rikitake
        fig_rikitake_f, ax_rikitake_f = plt.subplots()
        fig_rikitake_f.suptitle("Phase of Rikitake.")
        fig_rikitake_f.supxlabel("$f$ [$Hz$]")

        for l, m in GAUSS_LIST_EXT:
            index = GAUSS_LIST_EXT.index((l, m))
            
            ax_gauss_t_induced[index].plot(
                t/3600, induced_h[index],
                label="high_l=" + str(l) + ", m=" + str(m), color='red')
            ax_gauss_t_induced[index].plot(
                t/3600, induced_l[index],
                label="low_l=" + str(l) + ", m=" + str(m), color='green')

            ax_gauss_f_induced[index].plot(
                f[index][1:], real(amp_rikitake_h[index][1:]),
                label="high_l=" + str(l) + ", m=" + str(m), color='red')
            ax_gauss_f_induced[index].plot(
                f[index][1:], real(amp_rikitake_l[index][1:]),
                label="low_l=" + str(l) + ", m=" + str(m), color='green')

            ax_rikitake_f.plot(
                f[index][1:], real(phase_rikitake_h[index][1:]),
                label="high_l=" + str(l) + ", m=" + str(m))
            ax_rikitake_f.plot(
                f[index][1:], real(phase_rikitake_l[index][1:]),
                label="high_l=" + str(l) + ", m=" + str(m))

            ax_gauss_f_induced[index].set_xscale('log')
            ax_rikitake_f.set_xscale('log')
            ax_gauss_t_induced[index].set_ylabel("induced")
            ax_gauss_f_induced[index].set_ylabel("induced")
            ax_gauss_t_induced[index].legend(loc='upper right')
            ax_gauss_f_induced[index].legend(loc='upper left')
            ax_rikitake_f.legend()

        # plot transfer function for each degree up to RIKITAKEDEGREE
        # including alpha plot for frequencies in given data
        for i in RIKITAKEDEGREE:
            for l, m in GAUSS_LIST_EXT:
                if i==l:
                    index = GAUSS_LIST_EXT.index((l, m))
                    rikitake_plot(
                        i, f[index], rikitake_h_real[index],
                        rikitake_l_real[index], coeff_ext_f[index]
                        )

    except NameError:
        print("Set RIKITAKE=True or PLOTGAUSSIAN_t and PLOTGAUSSIAN_t=True.")

print("Time for the Process: " + str(time() - t0) + " seconds.")



T_h = [real(phase_rikitake_h[index][1:])/f[index][1:] for index in [0, 1]]
T_l = [real(phase_rikitake_l[index][1:])/f[index][1:] for index in [0, 1]]

plt.plot(f[0][1:], T_h[0], label="high_g10")
plt.plot(f[0][1:], T_l[0], label="low_g10")
plt.plot(f[1][1:], T_h[1], label="high_g21")
plt.plot(f[1][1:], T_l[1], label="low_g21")
plt.legend()
