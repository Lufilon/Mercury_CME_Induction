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
from signal_processing import gaussian_t_to_f
from rikitake import rikitake_get, rikitake_plot, rikitake_transferfunction
from plotting import plot_simple, plot_gauss_solo, plot_gauss_twinx
from orbit import orbit

# third party packages
from time import time
from numpy import pi, array, zeros, real

# for measuring the runtime of the routine
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
coeff_ext_sec_f_h, coeff_ext_sec_f_l, amp_riki_h, amp_riki_l, phase_riki_h, phase_riki_l, induced_h, induced_l = rikitake_get(
    t, freq, coeff_ext_f_amp, coeff_ext_f_phase, rel_indices, r_arr, sigma_h,
    sigma_l, t_steps, freqnr, resolution, gauss_list_ext, riki_dir)

# plot the freq dependant secondary gauss coefficients
ax_gauss_f_sec = plot_gauss_twinx(
    fig_gauss_f, ax_gauss_f_pri, freq[0, 1:], real(coeff_ext_sec_f_h[:, 1:]),
    real(coeff_ext_sec_f_l[:, 1:]), gauss_list_ext,
    ylabel="$A_\\mathrm{sec}$ $[nT]$", loc='upper right',
    name='gaussian_f_sec.jpeg',
    title="Primary and secondary Gauss coefficients over freq.", axvline=True)

# plot the time dependant secondary gauss coefficients
ax_gauss_t_sec = plot_gauss_twinx(
    fig_gauss_t, ax_gauss_t_pri, t_plotting, induced_h, induced_l,
    gauss_list_ext, ylabel="$A_\\mathrm{sec}$ $[nT]$", loc='lower right',
    name='gaussian_t_sec.jpeg',
    title="Primary and secondary Gauss coefficients over time.", axvline=False)

# plot the transferfunction for the amplitude of the rikitake factor
fig_transfer_1, ax_transfer_1 = rikitake_transferfunction(
    l=1, known_excitements=False, spec_freq=False)
fig_transfer_2, ax_transfer_2 = rikitake_transferfunction(
    l=2, known_excitements=False, spec_freq=False)

# plot the transferfunction with an alpha plot of the data
ax_alpha_1 = rikitake_plot(
    fig_transfer_1, ax_transfer_1, 1, freq[0, 1:], amp_riki_h[0, 1:],
    amp_riki_l[0, 1:], coeff_ext_f_amp[0, 1:])
ax_alpha_2 = rikitake_plot(
    fig_transfer_2, ax_transfer_2, 2, freq[1, 1:], amp_riki_h[1, 1:],
    amp_riki_l[1, 1:], coeff_ext_f_amp[1, 1:])

# plot the phase of the rikitake factor
fig_riki_phase, ax_riki_phase = plot_simple(
    freq[0, 1:], real(phase_riki_h[:, 1:]), real(phase_riki_l[:, 1:]),
    gauss_list_ext, xscale='log', yscale='linear', xlabel="$f$ [$Hz$]",
    ylabel="$\\varphi$ [$rad$]", loc='best',
    title="Argument of the rikitake factor", name="rikitake_phase.jpeg")

# calculate and plot the timedelta between the primary and secondary field
T_h = -phase_riki_h[:, 1:]/(2*pi*freq[:, 1:])/60
T_l = -phase_riki_l[:, 1:]/(2*pi*freq[:, 1:])/60

fig_riki_timedelta, ax_riki_timedelta = plot_simple(
    freq[0, 1:], T_h, T_l, gauss_list_ext, xscale='log', yscale='log',
    xlabel="$f$ [$Hz$]", ylabel="$T$ [$min$]", loc='best',
    title="Timedelta between primary and secondary field",
    name="timedelta.jpeg")

# calculate and plot the induced magnetic field for a 400km orbit
fig_400km, ax_400km, Br_h, Bt_h, Br_l, Bt_l = orbit(
    t, freq, gauss_list_ext, theta_arr, coeff_ext_sec_f_h, coeff_ext_sec_f_l,
    coeff_ext_f_phase, phase_riki_h, phase_riki_l, induced_h, induced_l,
    height=400, R_M=2440, case="orbit")

# calculate and plot the difference for the induced magnetic field for a 400km
# orbit in- and excluding the phase of the rikitakefactor for the lowest freq
fig_400km_diff, ax_400km_diff, Br_h_diff, Bt_h_diff, Br_l_diff, Bt_l_diff = orbit(
    t, freq, gauss_list_ext, theta_arr, coeff_ext_sec_f_h, coeff_ext_sec_f_l,
    coeff_ext_f_phase, phase_riki_h, phase_riki_l, induced_h, induced_l,
    height=400, R_M=2440, case="orbit_diff")

print("Time for the Process: " + str(time() - t0) + " seconds.")
