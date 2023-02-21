# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 10:24:28 2022

@author: Luis-
"""

# from numba import njit
from time import time
from legendre_polynomials import P_dP
from rikitake_base import rikitake_calc, rikitake_plot
from signal_processing import fft_own, rebuild
from SHA_by_integration import SHA_by_integration

from numpy import nanmax, nanmin, savetxt, loadtxt, pi, nan, isnan, hypot, exp
from numpy import array, linspace, meshgrid, ravel, zeros, asarray, flip
from numpy import arctan2, cos, sin, isin, real
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})

# own packages
from data_input import data_input
from magneticfield import magneticfield_sum

t0 = time()

# =============================================================================
# Define the parameters of the routine
# =============================================================================
# reduce sample size -> decreases runtime significantly
resolution = 100

# define grid for magnetic field calculation with the kth22-model
num_theta = 200
num_phi = 2 * num_theta

# parameters for the kth22-modell
dipole, neutralsheet = True, True
internal, external = True, True
settings = [dipole, neutralsheet, False, internal, external]
di_val = 50.

# Maximum degree for the SHA
degree_max = 2

# which parts of the routine are performed - muss noch weg!!!
GAUSSIAN_t, PLOTGAUSSIAN_t = False, False
GAUSSIAN_f, PLOTGAUSSIAN_f = False, False
RIKITAKE, PLOTRIKITAKE = False, False

# array containing gaussians to be analyzed. Tupel is (l, m), l=degree, m=order
gauss_list_ext = [(1, 0), (2, 1)]

# number of frequencies used for the rikitake calculation - max: t_steps//2 + 1
freqnr = 3601

# radius of planet Mercury
R_M = 2440

# specifiy mercuries layers - high and low conductivity cases from grimmich2019
r_arr = array([0, 1740E3, 1940E3, 2040E3, 2300E3, 2440E3])
sigma_arr_h = array([0, 1E7, 1E3, 10**0.5, 10**0.7, 1E-2])
sigma_arr_l = array([0, 1E5, 1E2, 10**-0.5, 10**-3, 1E-7])

# magnetic field degree for the frequency range rikitake plot
rikitakedegree = [1, 2]

# relative paths to the directories
mission_name = 'helios_1'
file_name = 'output-helios1_e1_np_helios1_e1_vp_0_1_2_1980148000000000.txt'
empty_rows = 89  # defines number of rows until relevant input

data_dir = 'data/'
runtime_dir = data_dir + 'runtime/'
mission_dir = data_dir + mission_name + '/'

case_dir = mission_dir + 'ns=' + str(settings[1])
magn_dir = case_dir + '/magnetic/resolution=' + str(resolution)
gauss_t_dir = case_dir + '/gaussian_t/resolution=' + str(resolution)
gauss_f_dir = case_dir + '/gaussian_f/resolution=' + str(resolution)
riki_dir = case_dir + '/rikitake/resolution=' + str(resolution)

# =============================================================================
# START OF THE ROUTINE
# =============================================================================
# load, format and plot data
t, t_plotting, r_hel, R_ss = data_input(mission_dir+file_name, empty_rows,
                                        plot=True)
t_steps = t.size

# distances for which the magnetic field is calculated
possible_distances = linspace(nanmin(r_hel), nanmax(r_hel), resolution)

# Create angular data for 200x400 points on a sphere.
num_pts = int(num_theta * num_phi)
theta_arr = linspace(0, pi, num_theta, endpoint=False)
phi_arr = linspace(0, 2*pi, num_phi, endpoint=False)
phi_arr_2D, theta_arr_2D = meshgrid(phi_arr, theta_arr)
theta, phi = ravel(theta_arr_2D), ravel(phi_arr_2D)

print("Finished creating the angular data.")

# calulate the magnetic field via the kth22-modell and plot it
Br_possible, Bt_possible, Bp_possible = magneticfield_sum(
    possible_distances, R_ss, theta, phi, num_theta, num_phi, resolution,
    settings, True, runtime_dir,
    path='data/helios_1/ns=True/magnetic/resolution=')

"""
TODO: Ab hier weiter
"""

if GAUSSIAN_t:
    # Import time dependant Gauss coefficients for given resolution and maximum degree.
    try:
        coeff_ext_t_possible = loadtxt(
            gauss_t_dir + 'resolution=' + str(resolution)
            + '_degree_max=' + str(degree_max) + '_external.gz')

        coeff_ext_t_possible = coeff_ext_t_possible.reshape(
            resolution, pow(degree_max, 2) // degree_max+1,
            degree_max+1)

        print("Finished importing the time dependant Gauss coefficients"
              + "with resolution=" + str(resolution) + ".")

    except OSError:
        print("No file for this combination of pseudo_distance resolution " +
              "and max degree of spherical analyzed Gauss coefficients was stored " +
              "for the given CME and given kth-modell parameters. " +
              "- Started the calculation.")

        # Calculate gaussian-coefficients via a spherical harmonic analysis.
        ref_radius = R_M
        ana_radius = ref_radius

        coeff_ext_t_possible = zeros((resolution, degree_max+1, degree_max+1))

        try:
            for i in range(resolution):
                for m in range(degree_max + 1):
                    coeff_ext_t_possible[i][m] = SHA_by_integration(
                        Br_possible[i], Bt_possible[i], Bp_possible[i],
                        ana_radius, ref_radius, degree_max, m
                    )[1]

            print("Finished calculating the time dependant Gauss coefficients " +
                  "for the given resolution using the SHA by integration.")

            savetxt(gauss_t_dir + 'resolution=' + str(resolution)
                    + '_degree_max=' + str(degree_max) + '_external.gz',
                    coeff_ext_t_possible.reshape(coeff_ext_t_possible.shape[0], -1))

            print("Finished saving the time dependant Gauss coefficients.")

        except NameError:
            print("Calculate or import the magnetic field data first.")

    # Assign the values to the data points with the smallest deviation
    coeff_ext_t = zeros((t_steps, degree_max+1, degree_max+1))

    for i in range(t_steps):
        if not isnan(r_hel[i]):
            nearest_distance_index = (abs(possible_distances - r_hel[i])).argmin()
            coeff_ext_t[i] = coeff_ext_t_possible[nearest_distance_index]
        else:
            coeff_ext_t[i] = nan

    print("Finished upscaling the lower resolution time dependant Gauss coefficients.")

if PLOTGAUSSIAN_t:
    # Plot external time-dependant inducing Gauss coefficients.
    try:
        fig_gauss_t_inducing, ax_gauss_t_inducing = plt.subplots(
            len(gauss_list_ext),  sharex=True)
        plt.subplots_adjust(hspace=0)
        ax_gauss_t_inducing[0].set_title("Time-dependant primary " +
                                         "Gauss coefficients")

        for l, m in gauss_list_ext:
            index = gauss_list_ext.index((l, m))

            ax_gauss_t_inducing[index].plot(
                t_plotting, [coeff_ext_t[i][m][l] for i in range(t_steps)],
                label="$g_{" + str(l) + str(m) + ", \\mathrm{pri}}}$")

            ax_gauss_t_inducing[index].set_ylabel("$A_\\mathrm{pri}$ $[nT]$")
            ax_gauss_t_inducing[index].legend(loc='lower left')

        plt.savefig('plots/gaussian_t_solo.jpeg', dpi=600)

    except NameError:
        print("Set GAUSSIAN_t=True.")

if GAUSSIAN_f:
    # Fourier transform the coefficients to the frequency domain
    f = zeros((len(gauss_list_ext), t_steps//2 + 1))
    coeff_ext_f = zeros((len(gauss_list_ext), t_steps//2 + 1))
    phase = zeros((len(gauss_list_ext), t_steps//2 + 1))

    relIndices = zeros((len(gauss_list_ext), freqnr), dtype='int')

    for l, m in gauss_list_ext:
        index = gauss_list_ext.index((l, m))
        f[index], coeff_ext_f[index], phase[index] = fft_own(
            t, t_steps, asarray([coeff_ext_t[i][m][l] for i in range(t_steps)]))

        # get relevant frequencies, filter f_0 = 0 Hz beforehand
        relIndices[index] = flip(coeff_ext_f[index].argsort()[-freqnr:])
        mask = isin(coeff_ext_f[index], coeff_ext_f[index][relIndices[index]],
                    invert=True)
        coeff_ext_f[index][mask] = 0

    print("Finished fourier transforming the external Gauss coefficients.")

if PLOTGAUSSIAN_f:
    try:
        fig_gauss_f_inducing, ax_gauss_f_inducing = plt.subplots(
            len(gauss_list_ext), sharex=True)
        plt.subplots_adjust(hspace=0)
        ax_gauss_f_inducing[0].set_title("Freq-dependant primary and " +
                                         "secondary Gauss coefficients")

        for l, m in gauss_list_ext:
            index = gauss_list_ext.index((l, m))

            ax_gauss_f_inducing[index].plot(
                f[index][1:], coeff_ext_f[index][1:],
                label="$g_{" + str(l) + str(m) + ", \\mathrm{pri}}$")

            ax_gauss_f_inducing[index].set_xscale('log')
            ax_gauss_f_inducing[index].set_ylabel("$A_\\mathrm{pri}$ $[nT]$")
            ax_gauss_f_inducing[index].legend(loc='upper center')
            ax_gauss_f_inducing[index].axvline(
                x=f[index][1], color='goldenrod', linestyle='dotted')

        ax_gauss_f_inducing[1].set_xlabel("$f$ $[Hz]$")

        # plt.savefig('plots/test.jpeg', dpi=600)

    except NameError:
        print("Set GAUSSIAN_f=True.")

if RIKITAKE:
    # calculation of rikitake-factor for each selected frequency for both
    # high and low condutivity model.
    try:
        rikitake_h_real = loadtxt(
            riki_dir + '_freqnr=' + str(freqnr) + '_h_real.gz')
        rikitake_h_imag = loadtxt(
            riki_dir + '_freqnr=' + str(freqnr) + '_h_imag.gz')
        rikitake_l_real = loadtxt(
            riki_dir + '_freqnr=' + str(freqnr) + '_l_real.gz')
        rikitake_l_imag = loadtxt(
            riki_dir + '_freqnr=' + str(freqnr) + '_l_imag.gz')

        rikitake_h_real = rikitake_h_real.reshape((len(gauss_list_ext), freqnr))
        rikitake_h_imag = rikitake_h_imag.reshape((len(gauss_list_ext), freqnr))
        rikitake_l_real = rikitake_l_real.reshape((len(gauss_list_ext), freqnr))
        rikitake_l_imag = rikitake_l_imag.reshape((len(gauss_list_ext), freqnr))

        print("Finished importing the real and imag parts of the rikitake " +
              "factor for the each conductivity profil.")

    except OSError:
        print("No rikitake calculation for this freqnr was done yet " +
              "- Started the calculation.")
        # Calculate the rikitake factor for the given conductivity profiles for
        # the given freqnr.
        rikitake_h_real = zeros((len(gauss_list_ext), t_steps//2 + 1))
        rikitake_h_imag = zeros((len(gauss_list_ext), t_steps//2 + 1))
        rikitake_l_real = zeros((len(gauss_list_ext), t_steps//2 + 1))
        rikitake_l_imag = zeros((len(gauss_list_ext), t_steps//2 + 1))

        for l, m in gauss_list_ext:
            index = gauss_list_ext.index((l, m))

            for i in range(t_steps//2 + 1):
                if i in relIndices and i > 0:
                    rikitake_h_real[index][i], rikitake_h_imag[index][i], rikitake_l_real[index][i], rikitake_l_imag[index][i] = rikitake_calc(
                        l, f[index][i], r_arr, sigma_arr_h, sigma_arr_l)

        print("Finished calculating the rikitake factor parts.")

        savetxt(riki_dir + '_freqnr=' + str(freqnr)
                + '_h_real.gz', rikitake_h_real.ravel())
        savetxt(riki_dir + '_freqnr=' + str(freqnr)
                + '_h_imag.gz', rikitake_h_imag.ravel())
        savetxt(riki_dir + '_freqnr=' + str(freqnr)
                + '_l_real.gz', rikitake_l_real.ravel())
        savetxt(riki_dir + '_freqnr=' + str(freqnr)
                + '_l_imag.gz', rikitake_l_imag.ravel())

        print("Finished saving the rikitake factor parts to file.")

    phase_rikitake_h = zeros(
        (len(gauss_list_ext), t_steps//2 + 1), dtype=complex)
    phase_rikitake_l = zeros(
        (len(gauss_list_ext), t_steps//2 + 1), dtype=complex)

    amp_rikitake_h = zeros(
        (len(gauss_list_ext), t_steps//2 + 1), dtype=complex)
    amp_rikitake_l = zeros(
        (len(gauss_list_ext), t_steps//2 + 1), dtype=complex)

    induced_h = zeros((len(gauss_list_ext), t_steps))
    induced_l = zeros((len(gauss_list_ext), t_steps))

    for l, m in gauss_list_ext:
        index = gauss_list_ext.index((l, m))

        phase_rikitake_h[index] = arctan2(
            rikitake_h_imag[index], rikitake_h_real[index])
        phase_rikitake_l[index] = arctan2(
            rikitake_l_imag[index], rikitake_l_real[index])

        amp_rikitake_h[index] = coeff_ext_f[index] * hypot(
            rikitake_h_real[index], rikitake_h_imag[index])
        amp_rikitake_h[index] = amp_rikitake_h[index] * exp(
            0+1j * phase_rikitake_h[index])

        amp_rikitake_l[index] = coeff_ext_f[index] * hypot(
            rikitake_l_real[index], rikitake_l_imag[index])
        amp_rikitake_l[index] = amp_rikitake_l[index] * exp(
            0+1j * phase_rikitake_l[index])

        induced_h[index] = rebuild(
            t, f[index], amp_rikitake_h[index], phase[index])
        induced_l[index] = rebuild(
            t, f[index], amp_rikitake_l[index], phase[index])

    print("Finished performing the inverse FFT of the rikitake modified " +
          "freq-dependant Gauss coefficients.")

if PLOTRIKITAKE:
    try:        
        color = iter(plt.cm.prism(linspace(0, 0.5, 4)))
        # add to time-dependant plot
        ax_gauss_t_induced = array(
            [a.twinx() for a in ax_gauss_t_inducing.ravel()]).reshape(
                ax_gauss_t_inducing.shape)

        # add to frequency-dependant plot
        ax_gauss_f_induced = array(
            [a.twinx() for a in ax_gauss_f_inducing.ravel()]).reshape(
                ax_gauss_f_inducing.shape)

        # phase of rikitake
        fig_rikitake_phi, ax_rikitake_phi = plt.subplots()
        ax_rikitake_phi.set_title("Argument of the Rikitake factor")

        for l, m in gauss_list_ext:
            index = gauss_list_ext.index((l, m))
            
            color1 = next(color)
            color2 = next(color)

            ax_gauss_t_induced[index].plot(
                t_plotting, induced_h[index],
                label="$\\sigma_{high}$",
                color=color1)
            ax_gauss_t_induced[index].plot(
                t_plotting, induced_l[index],
                label="$\\sigma_{low}$",
                color=color2)

            ax_gauss_f_induced[index].plot(
                f[index][1:], real(amp_rikitake_h[index][1:]),
                label="$\\sigma_{high}$",
                color='red')
            ax_gauss_f_induced[index].plot(
                f[index][1:], real(amp_rikitake_l[index][1:]),
                label="$\\sigma_{low}$",
                color='green')

            ax_rikitake_phi.plot(
                f[index][1:], real(phase_rikitake_h[index][1:]),
                label="$\\sigma_{high}$, $l=" + str(l) + ", m=" + str(m) + "$")
            ax_rikitake_phi.plot(
                f[index][1:], real(phase_rikitake_l[index][1:]),
                label="$\\sigma_{low}$,  $l=" + str(l) + ", m=" + str(m) + "$")

            ax_gauss_f_induced[index].set_xscale('log')
            ax_rikitake_phi.set_xscale('log')
            ax_gauss_t_induced[index].set_ylabel("$A_\\mathrm{sec}$ $[nT]$")
            ax_gauss_f_induced[index].set_ylabel("$A_\\mathrm{sec}$ $[nT]$")
            ax_gauss_t_induced[index].legend(loc='lower right')
            ax_gauss_f_induced[index].legend(loc='upper right')
            ax_rikitake_phi.legend()

        ax_rikitake_phi.set_xlabel("$f$ [$Hz$]")
        ax_rikitake_phi.set_ylabel("$\\varphi$ [$rad$]")

        ax_gauss_f_induced[1].annotate(
            '$f_1$', (f[1][1]+5E-7, ax_gauss_f_induced[1].get_ylim()[0]+0.2))

        fig_gauss_t_inducing.savefig(
            'plots/gaussian_t_' + str(resolution) + '.jpg', dpi=600)
        fig_gauss_f_inducing.savefig(
            'plots/gaussian_f_' + str(resolution) + '.jpg', dpi=600)
        fig_rikitake_phi.savefig(
            'plots/rikitake_phase_' + str(resolution) + '.jpg', dpi=600)

        color = iter(plt.cm.prism(linspace(0, 0.5, 4)))
        # plot transfer function for each degree up to rikitakedegree
        # including alpha plot for frequencies in given data
        for i in rikitakedegree:
            c = next(color)
            d = next(color)
            for l, m in gauss_list_ext:
                if i == l:
                    index = gauss_list_ext.index((l, m))
                    rikitake_plot(
                        i, f[index],
                        hypot(rikitake_h_real[index], rikitake_h_imag[index]),
                        hypot(rikitake_l_real[index], rikitake_l_imag[index]),
                        coeff_ext_f[index], c, d
                    )

        """
        TODO
            noch sehr quick and dirty
        """
        # plot the transit time of the induced signal
        T_h = [real(phase_rikitake_h[index][1:])/(2*pi*f[index][1:])
               for index in range(len(gauss_list_ext))]
        T_l = [real(phase_rikitake_l[index][1:])/(2*pi*f[index][1:])
               for index in range(len(gauss_list_ext))]

        plt.figure("Transit time of the primary signal")
        plt.title("Transit time of the secondary signal")
        plt.plot(f[0][1:], -T_h[0]/60, label="$\\sigma_{high}$, $g_{10}$")
        plt.plot(f[0][1:], -T_l[0]/60, label="$\\sigma_{low}$, $g_{10}$")
        plt.plot(f[1][1:], -T_h[1]/60, label="$\\sigma_{high}$, $g_{21}$")
        plt.plot(f[1][1:], -T_l[1]/60, label="$\\sigma_{low}$, $g_{21}$")

        plt.xscale('log')
        # plt.yscale('symlog')
        plt.yscale('log')
        # plt.ylim(-2E2, 0.5)
        plt.xlabel("f [$Hz$]")
        plt.ylabel("T [$min$]")
        plt.legend()

        plt.savefig('plots/transit_time_' + str(resolution) + '.jpg', dpi=600)

        # transform to magnetic field for polar orbit at 400 km over surface for phi=0

        B_r_400_h = zeros((len(gauss_list_ext), len(theta_arr)))
        B_theta_400_h = zeros((len(gauss_list_ext), len(theta_arr)))
        B_400_h = zeros((len(gauss_list_ext), len(theta_arr)))
        B_r_400_l = zeros((len(gauss_list_ext), len(theta_arr)))
        B_theta_400_l = zeros((len(gauss_list_ext), len(theta_arr)))
        B_400_l = zeros((len(gauss_list_ext), len(theta_arr)))

        phase_rikitake_h_temp = phase_rikitake_h.copy()
        phase_rikitake_l_temp = phase_rikitake_l.copy()

        amp_rikitake_h_temp = amp_rikitake_h.copy()
        amp_rikitake_l_temp = amp_rikitake_l.copy()

        induced_h_phase0 = induced_h.copy()
        induced_l_phase0 = induced_l.copy()

        for l, m in gauss_list_ext:
            index = gauss_list_ext.index((l, m))

        # for the magnetic field conductivity difference plot
            phase_rikitake_h_temp[index] = phase_rikitake_h[index].copy()
            phase_rikitake_h_temp[index][1] = 0
            phase_rikitake_l_temp[index] = phase_rikitake_l[index].copy()
            phase_rikitake_l_temp[index][1] = 0

            amp_rikitake_h_temp[index] = coeff_ext_f[index] * hypot(
                rikitake_h_real[index], rikitake_h_imag[index])
            amp_rikitake_h_temp[index] = amp_rikitake_h[index] * exp(
                0+1j * phase_rikitake_h_temp[index])

            amp_rikitake_l_temp[index] = coeff_ext_f[index] * hypot(
                rikitake_l_real[index], rikitake_l_imag[index])
            amp_rikitake_l_temp[index] = amp_rikitake_l[index] * exp(
                0+1j * phase_rikitake_l_temp[index])

            induced_h_phase0[index] = rebuild(
                t, f[index], amp_rikitake_h_temp[index], phase[index])
            induced_l_phase0[index] = rebuild(
                t, f[index], amp_rikitake_l_temp[index], phase[index])

            P_lm, dP_lm = P_dP(l, m, cos(theta_arr))
            dP_lm[0] = nan  # fragment caused by legendre polynomial

            # account for the inner derivative
            dP_lm = dP_lm * (-sin(theta_arr))

            B_r_400_h[index] = (l+1) * (R_M/(R_M+400))**(l+2) * \
                max(abs(induced_h[index])) * P_lm
            B_theta_400_h[index] = - (R_M/(R_M+400))**(l+2) * \
                max(abs(induced_h[index])) * dP_lm
            B_r_400_l[index] = (l+1) * (R_M/(R_M+400))**(l+2) * \
                max(abs(induced_l[index])) * P_lm
            B_theta_400_l[index] = - (R_M/(R_M+400))**(l+2) * \
                max(abs(induced_l[index])) * dP_lm

            # for difference plot
            B_r_400_h[index] = (l+1) * (R_M/(R_M+400))**(l+2) * \
                abs(max(abs(induced_h[index])) - max(abs(induced_h_phase0[index]))) * P_lm
            B_theta_400_h[index] = - (R_M/(R_M+400))**(l+2) * \
                abs(max(abs(induced_h[index])) - max(abs(induced_h_phase0[index]))) * dP_lm
            B_r_400_l[index] = (l+1) * (R_M/(R_M+400))**(l+2) * \
                abs(max(abs(induced_l[index])) - max(abs(induced_l_phase0[index]))) * P_lm
            B_theta_400_l[index] = - (R_M/(R_M+400))**(l+2) * \
                abs(max(abs(induced_l[index])) - max(abs(induced_l_phase0[index]))) * dP_lm

            B_400_h[index] = hypot(B_r_400_h[index], B_theta_400_h[index])
            B_400_l[index] = hypot(B_r_400_l[index], B_theta_400_l[index])

        fig_400, (ax_400_r, ax_400_theta, ax_400) = plt.subplots(
            3, sharex=True)
        plt.subplots_adjust(hspace=0)
        ax_400_r.set_title("Magnetic field components for polar orbit with " +
                            "$\\varphi = 0$ in $R_\\mathrm{M}+400 km$")
        # ax_400_r.set_title("Difference of magnetic field components for " +
        #                    "$\\sigma_h$ and $\\sigma_l$\n for polar orbit with" +
        #                    " $\\varphi = 0$ in $R_\\mathrm{M} + 400 km$")
        # ax_400_r.set_title("Difference of magnetic field components for " +
        #                     "$\\sigma_h$ and $\\sigma_l$\n caused by in- and" +
        #                     " excluding the phase information for $f_1$")

        for l, m in gauss_list_ext:
            index = gauss_list_ext.index((l, m))

            ax_400_r.plot(theta_arr, B_r_400_h[index],
                          label="$g_{" + str(l) + str(m) + "}$, $\\sigma_h$")
            ax_400_theta.plot(theta_arr, B_theta_400_h[index],
                              label="$g_{" + str(l) + str(m) + "}$, $\\sigma_h$")
            ax_400.plot(theta_arr, B_400_h[index],
                        label="$g_{" + str(l) + str(m) + "}$, $\\sigma_h$")
            ax_400_r.plot(theta_arr, B_r_400_l[index],
                          label="$g_{" + str(l) + str(m) + "}$, $\\sigma_l$")
            ax_400_theta.plot(theta_arr, B_theta_400_l[index],
                              label="$g_{" + str(l) + str(m) + "}$, $\\sigma_l$")
            ax_400.plot(theta_arr, B_400_l[index],
                        label="$g_{" + str(l) + str(m) + "}$, $\\sigma_l$")


        ax_400_r.set_ylabel("$B_r$ [$nT$]")
        ax_400_theta.set_ylabel("$B_\\vartheta$ [$nT$]")
        ax_400.set_ylabel("$|B|$ [$nT$]")
        ax_400_r.legend(fontsize='small')
        ax_400_theta.legend(fontsize='small')
        ax_400.legend(fontsize='small')
        ax_400.set_xlabel("$\\vartheta$ [$rad$]")

        fig_400.savefig(
            'plots/400km_orbit_' + str(resolution) + '.jpg', dpi=600)

        # is it possible to get the phase information from the data?
        """
        TODO
            Works only for len(gauss_list_ext) = 2
        """
        fig_phase, (ax_phase_10, ax_phase_21) = plt.subplots(2, sharex=True)
        plt.subplots_adjust(hspace=0)
        ax_phase_10.set_title("Difference of the time dependant secondary " +
                              "Gauss coefficient using\n solely $f_1$ for " +
                              "the rebuild using $\\sigma_h$, in- and " + 
                              "excluding phase information")
        # ax_phase_10.set_title("Difference of the time dependant secondary " +
        #                       "Gauss coefficient using\n solely $f_1$ for " +
        #                       "the rebuild for the high and low conductivity" +
        #                       " profiles")
        # ax_phase_10.set_title("Time dependant secondary Gauss coefficient " + 
        #                       "using \n solely $f_1$ in- and excluding the " +
        #                       "phase information")

        # difference for same \sigma
        # ax_phase_10.plot(t_plotting,
        #                   abs(rebuild(
        #                       t, [f[0][1]], [induced_h[0][1]],
        #                       [phase[0][1]]) - rebuild(
        #                           t, [f[0][1]], [induced_h_phase0[0][1]],
        #                           [phase[0][1]])),
        #                   label="$g_{10}$, $\\sigma_h$")
        # ax_phase_21.plot(t_plotting,
        #                   abs(rebuild(
        #                       t, [f[1][1]], [induced_h[1][1]],
        #                       [phase[1][1]]) - rebuild(
        #                           t, [f[1][1]], [induced_h_phase0[1][1]],
        #                           [phase[1][1]])),
        #                   label="$g_{21}$, $\\sigma_h$")

        # ax_phase_10.plot(t_plotting,
        #                   abs(rebuild(
        #                       t, [f[0][1]], [induced_l[0][1]],
        #                       [phase[0][1]]) - rebuild(
        #                           t, [f[0][1]], [induced_l_phase0[0][1]],
        #                           [phase[0][1]])),
        #                   label="$g_{10}$, $\\sigma_l$")
        # ax_phase_21.plot(t_plotting,
        #                   abs(rebuild(
        #                       t, [f[1][1]], [induced_l[1][1]],
        #                       [phase[1][1]]) - rebuild(
        #                           t, [f[1][1]], [induced_l_phase0[1][1]],
        #                           [phase[1][1]])),
        #                   label="$g_{21}$, $\\sigma_l$")

        # ax_phase_10.plot(t_plotting,
        #                   rebuild(t, [f[0][1]], [induced_h[0][1]],
        #                           [phase[0][1]]),
        #                   label="$g_{10}$, $\\sigma_{high}$")
        # ax_phase_10.plot(t_plotting,
        #                   rebuild(t, [f[0][1]], [induced_h_phase0[0][1]],
        #                           [phase[0][1]]),
        #                   label="$g_{10}$, $\\varphi=0$, $\\sigma_{high}$")
        # ax_phase_21.plot(t_plotting,
        #                   rebuild(t, [f[1][1]], [induced_h[1][1]],
        #                           [phase[1][1]]),
        #                   label="$g_{21}$, $\\sigma_{high}$")
        # ax_phase_21.plot(t_plotting,
        #                   rebuild(t, [f[1][1]], [induced_h_phase0[1][1]],
        #                           [phase[1][1]]),
        #                   label="$g_{21}$, $\\varphi=0$, $\\sigma_{high}$")

        # ax_phase_10.plot(t_plotting,
        #                   rebuild(t, [f[0][1]], [induced_l[0][1]],
        #                           [phase[0][1]]),
        #                   label="$g_{10}$, $\\sigma_{low}$")
        # ax_phase_10.plot(t_plotting,
        #                   rebuild(t, [f[0][1]], [induced_l_phase0[0][1]],
        #                           [phase[0][1]]),
        #                   label="$g_{10}$, $\\varphi=0$, $\\sigma_{low}$")
        # ax_phase_21.plot(t_plotting,
        #                   rebuild(t, [f[1][1]], [induced_l[1][1]],
        #                           [phase[1][1]]),
        #                   label="$g_{21}$, $\\sigma_{low}$")
        # ax_phase_21.plot(t_plotting,
        #                   rebuild(t, [f[1][1]], [induced_l_phase0[1][1]],
        #                           [phase[1][1]]),
        #                   label="$g_{21}$, $\\varphi=0$, $\\sigma_{low}$")

        ax_phase_10.legend()
        ax_phase_21.legend()

        ax_phase_10.set_ylabel("$A$ [$nT$]")
        ax_phase_21.set_ylabel("$A$ [$nT$]")

        fig_phase.savefig(
            'plots/single_freq_rebuild_' + str(resolution) + '.jpg', dpi=600)

        for l, m in gauss_list_ext:
            index = gauss_list_ext.index((l, m))

            ax_solo = plt.subplot(len(gauss_list_ext), 1, index + 1)

            ax_solo.plot(t_plotting,
                      rebuild(t, [f[index][1]], [coeff_ext_f[index][1]], [phase[index][1]]),
                      label="g" + str(l) + str(m))
            ax_solo.plot(t_plotting,
                      rebuild(t, [f[index][1]], [coeff_ext_f[index][1]], [0]),
                      label="g" + str(l) + str(m) + " $\\varphi=0$")

            ax_solo.legend()

    except NameError as err:
        # print("Set RIKITAKE=True or PLOTGAUSSIAN_t and PLOTGAUSSIAN_t=True.")
        print(NameError, err)


print("Time for the Process: " + str(time() - t0) + " seconds.")

plt.close('all')  # closes all figures


# fig_400, (ax_400_r, ax_400_theta, ax_400) = plt.subplots(
#     3, sharex=True)
# plt.subplots_adjust(hspace=0)
# ax_400_r.set_title("Difference of the time dependant secondary " +
#                       "Gauss coefficient using\n solely $f_1$ for " +
#                       "the rebuild using $\\sigma_h$, in- and " + 
#                       "excluding phase information\n for resolution 100 and 200")

# for l, m in gauss_list_ext:
#     index = gauss_list_ext.index((l, m))

#     ax_400_r.plot(theta_arr, A1[index] - B1[index],
#                   label="$g_{" + str(l) + str(m) + "}$, $\\sigma_h$")
#     ax_400_theta.plot(theta_arr, abs(A2[index] - B2[index]),
#                       label="$g_{" + str(l) + str(m) + "}$, $\\sigma_h$")
#     ax_400.plot(theta_arr, A3[index] - B3[index],
#                 label="$g_{" + str(l) + str(m) + "}$, $\\sigma_h$")
#     ax_400_r.plot(theta_arr, A4[index] - B4[index],
#                   label="$g_{" + str(l) + str(m) + "}$, $\\sigma_l$")
#     ax_400_theta.plot(theta_arr, A5[index] - B5[index],
#                       label="$g_{" + str(l) + str(m) + "}$, $\\sigma_l$")
#     ax_400.plot(theta_arr, A6[index] - B6[index],
#                 label="$g_{" + str(l) + str(m) + "}$, $\\sigma_l$")

# fig_400.savefig(
#     'plots/400km_resolution.jpg', dpi=600)