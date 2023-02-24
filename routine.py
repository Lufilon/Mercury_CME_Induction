# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 10:24:28 2022

@author: Luis-
"""

# own packages
from data_input import data_get
from angular_data import angular_data
from magnetic_field import magnetic_field_get
from SHA_by_integration import SHA_by_integration_get
from signal_processing import gaussian_t_to_f, gaussian_f_to_t
from rikitake import rikitake_get, rikitake_plot, rikitake_transferfunction
from legendre_polynomials import P_dP
from plotting import plot_simple, plot_gauss_solo, plot_gauss_twinx

# third party packages
from time import time
from numpy import pi, nan, hypot, exp, sin, cos
from numpy import array, linspace, zeros, real
import matplotlib.pyplot as plt

# get time to obtaine the runtime of the routine
t0 = time()

# =============================================================================
# Define the parameters of the routine
# =============================================================================
# radius of planet Mercury
R_M = 2440

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

# maximum degree for the SHA
degree_max = 2

# radius for the reference shell that determines the inner and outer region
ref_radius = R_M
# radius where to evaluate via the SHA
ana_radius = R_M

# array containing analyzed Gauss coefficients.
# tuple is (l, m), l=degree, m=order
gauss_list_ext = [(1, 0), (2, 1)]

# number of frequencies used for the rikitake calculation - max: t_steps//2 + 1
freqnr = 3601

# specifiy mercuries layers - high and low conductivity cases from grimmich2019
r_arr = array([0, 1740E3, 1940E3, 2040E3, 2300E3, 2440E3])
sigma_h = array([0, 1E7, 1E3, 10**0.5, 10**0.7, 1E-2])
sigma_l = array([0, 1E5, 1E2, 10**-0.5, 10**-3, 1E-7])

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
t, t_plotting, t_steps, r_hel, R_ss, possible_distances = data_get(
    mission_dir+file_name, empty_rows, resolution, True)

# create angular data for 200x400 points on a sphere.
num_pts, theta_arr, phi_arr, theta, phi = angular_data(num_theta, num_phi)

# calculate the magnetic field via the kth22-modell and plot it
Br_possible, Bt_possible, Bp_possible = magnetic_field_get(
    possible_distances, R_ss, theta, phi, num_theta, num_phi, resolution,
    settings, True, runtime_dir, magn_dir)

# calculate the time dependant Gauss coefficients via the SHA
coeff_ext_t = SHA_by_integration_get(
    theta_arr, phi_arr, r_hel, possible_distances, t_steps, Br_possible,
    Bt_possible, Bp_possible, num_theta, num_phi, degree_max,
    resolution, ref_radius, ana_radius, gauss_t_dir)

# plot the time dependant primary gauss coefficients
coeff_ext_t_plotting = zeros((len(gauss_list_ext), t_steps))
for l, m in gauss_list_ext:
    index = gauss_list_ext.index((l, m))
    coeff_ext_t_plotting[index] = coeff_ext_t[:, m, l]

fig_gauss_t, ax_gauss_t_pri = plot_gauss_solo(
    t_plotting, coeff_ext_t_plotting, gauss_list_ext, xscale=None,
    yscale='linear', xlabel=None, ylabel="$A_\\mathrm{pri}$ $[nT]$",
    loc='lower left', title="Primary Gauss coefficients over time",
    name="gaussian_t_pri.jpeg", sharex=True)

# fourier-transform the time dependant gauss coefficients
freq, coeff_ext_f_amp, coeff_ext_f_phase, rel_indices = gaussian_t_to_f(
    coeff_ext_t, t, t_steps, gauss_list_ext, freqnr)

# plot the freq dependant primary gauss coefficients
fig_gauss_f, ax_gauss_f_pri = plot_gauss_solo(
    freq[0, 1:], coeff_ext_f_amp[:, 1:], gauss_list_ext, xscale='log',
    yscale='linear', xlabel="$f$ $[Hz]$", ylabel="$A_\\mathrm{pri}$ $[nT]$",
    loc='upper center', title="Primary Gauss coefficients over freq.",
    name='gaussian_f_pri.jpeg', sharex=True)

# use the rikitake factor to calculate the secondary gauss coefficients
amp_riki_h, amp_riki_l, phase_riki_h, phase_riki_l, induced_h, induced_l = rikitake_get(
    t, freq, coeff_ext_f_amp, coeff_ext_f_phase, rel_indices, r_arr, sigma_h,
    sigma_l, t_steps, freqnr, resolution, gauss_list_ext, riki_dir)

# plot the freq dependant secondary gauss coefficients
ax_gauss_f_sec = plot_gauss_twinx(
    fig_gauss_f, ax_gauss_f_pri, freq[0, 1:], real(amp_riki_h[:, 1:]),
    real(amp_riki_l[:, 1:]), gauss_list_ext, ylabel="$A_\\mathrm{sec}$ $[nT]$",
    loc='upper right', name='gaussian_f_sec.jpeg',
    title="Primary and secondary Gauss coefficients over freq.", axvline=True)

# plot the time dependant secondary gauss coefficients
ax_gauss_t_sec = plot_gauss_twinx(
    fig_gauss_t, ax_gauss_t_pri, t_plotting, induced_h, induced_l,
    gauss_list_ext, ylabel="$A_\\mathrm{sec}$ $[nT]$", loc='lower right',
    name='gaussian_t_sec.jpeg',
    title="Primary and secondary Gauss coefficients over time.", axvline=False)

# plot the phase of the rikitake factor
fig_riki_phase, ax_riki_phase = plot_simple(
    freq[0, 1:], real(phase_riki_h[:, 1:]), real(phase_riki_l[:, 1:]),
    gauss_list_ext, xscale='log', yscale='linear', xlabel="$f$ [$Hz$]",
    ylabel="$\\varphi$ [$rad$]", loc='best',
    title="Argument of the rikitake factor", name="rikitake_phase.jpeg")

"""
TODO: Hier weiter
"""
if False:
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
                    coeff_ext_f_amp[index], c, d
                )

    """
    TODO
        noch sehr quick and dirty
    """
    # plot the transit time of the induced signal
    T_h = [real(phase_riki_h[index][1:])/(2*pi*f[index][1:])
           for index in range(len(gauss_list_ext))]
    T_l = [real(phase_riki_l[index][1:])/(2*pi*f[index][1:])
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

    phase_riki_h_temp = phase_riki_h.copy()
    phase_riki_l_temp = phase_riki_l.copy()

    amp_riki_h_temp = amp_riki_h.copy()
    amp_riki_l_temp = amp_riki_l.copy()

    induced_h_phase0 = induced_h.copy()
    induced_l_phase0 = induced_l.copy()

    for l, m in gauss_list_ext:
        index = gauss_list_ext.index((l, m))

    # for the magnetic field conductivity difference plot
        phase_riki_h_temp[index] = phase_riki_h[index].copy()
        phase_riki_h_temp[index][1] = 0
        phase_riki_l_temp[index] = phase_riki_l[index].copy()
        phase_riki_l_temp[index][1] = 0

        amp_riki_h_temp[index] = coeff_ext_f_amp[index] * hypot(
            rikitake_h_real[index], rikitake_h_imag[index])
        amp_riki_h_temp[index] = amp_riki_h[index] * exp(
            0+1j * phase_riki_h_temp[index])

        amp_riki_l_temp[index] = coeff_ext_f_amp[index] * hypot(
            rikitake_l_real[index], rikitake_l_imag[index])
        amp_riki_l_temp[index] = amp_riki_l[index] * exp(
            0+1j * phase_riki_l_temp[index])

        induced_h_phase0[index] = rebuild(
            t, f[index], amp_riki_h_temp[index], phase[index])
        induced_l_phase0[index] = rebuild(
            t, f[index], amp_riki_l_temp[index], phase[index])

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
                  rebuild(t, [f[index][1]], [coeff_ext_f_amp[index][1]], [phase[index][1]]),
                  label="g" + str(l) + str(m))
        ax_solo.plot(t_plotting,
                  rebuild(t, [f[index][1]], [coeff_ext_f_amp[index][1]], [0]),
                  label="g" + str(l) + str(m) + " $\\varphi=0$")

        ax_solo.legend()


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

# transferfunction(2)