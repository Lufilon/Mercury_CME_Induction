import numpy as np
import scipy.special as special
from scipy.integrate import simps
from A_phi_hankel_v7 import a_phi_hankel_v7
from current_profile_pRC_v7 import current_profile_pRC_v7


def tail_field_pRC_v7(x_msm, y_msm, z_msm, di, control_params):



    rho = np.sqrt(x_msm ** 2 + y_msm ** 2)
    phi = np.arctan2(y_msm, x_msm)

    d_0 = control_params['d_0']
    delta_x = control_params['delta_x']
    scale_x_d = control_params['scale_x_d']
    delta_y = control_params['delta_y']

    t = control_params['t_a'] + control_params['t_b'] * di

    mu_0 = 1.0
    steps = 100
    rho_min = 0.0
    rho_max = 10.0
    h_steps = 100  # This value is from experience. When you change the current profile this should be checked again for sufficient convergence.

    rho_hankel = np.divide(range(steps), (float(steps - 1))) * (rho_max - rho_min) + rho_min

    current = current_profile_pRC_v7(rho_hankel, control_params)

    lambda_max = 10  # std value
    lambda_min = 10 ** (-2)  # std value

    lambda_result = 10 ** (np.divide(range(h_steps), (float(h_steps) - 1)) * (
            np.log10(lambda_max) - np.log10(lambda_min)) + np.log10(lambda_min))
    lambda_out = lambda_result

    integrand = current

    result_hankel_trafo = np.zeros(h_steps)
    for i in range(h_steps):
        result_hankel_trafo[i] = simps(special.j1(lambda_result[i] * rho_hankel) * integrand * rho_hankel,
                                       x=rho_hankel)  # special.j1 = Bessel function of the first kind of order 1



    H_current = mu_0 / 2.0 * result_hankel_trafo

    ###############################################################

    n_vec = len(np.atleast_1d(x_msm))
    b_tail_disk_x = np.zeros(n_vec)
    b_tail_disk_y = np.zeros(n_vec)
    b_tail_disk_z = np.zeros(n_vec)
    b_tail_disk_rho = np.zeros(n_vec)

    if n_vec > 1:
        for i in range(n_vec):
            a_phi = a_phi_hankel_v7(H_current, rho[i], phi[i], z_msm[i], lambda_out, d_0, delta_x,
                                   scale_x_d, delta_y)

            # numerically approximate the derivatives
            delta_z = 10 ** (-5)

            d_a_phi_d_z = (a_phi_hankel_v7(H_current, rho[i], phi[i], z_msm[i] + delta_z, lambda_out, d_0,
                                          delta_x, scale_x_d,
                                          delta_y) - a_phi_hankel_v7(H_current, rho[i],
                                                                        phi[i],
                                                                        z_msm[i] - delta_z,
                                                                        lambda_out,
                                                                        d_0, delta_x,
                                                                        scale_x_d,
                                                                        delta_y)) / (
                              2 * delta_z)

            delta_rho = 10 ** (-5)
            d_a_phi_d_rho = (a_phi_hankel_v7(H_current, rho[i] + delta_rho, phi[i], z_msm[i], lambda_out,
                                            d_0, delta_x,
                                            scale_x_d, delta_y) - a_phi_hankel_v7(
                H_current, rho[i] - delta_rho, phi[i],
                z_msm[i], lambda_out, d_0, delta_x, scale_x_d,
                delta_y)) / (2 * delta_rho)

            b_tail_disk_rho[i] = t[i] * (- d_a_phi_d_z)

            if rho[i] <= 10 ** (-4):
                b_tail_disk_z[i] = t[i] * (1.0 + d_a_phi_d_rho)

            else:
                b_tail_disk_z[i] = t[i] * (a_phi / rho[i] + d_a_phi_d_rho)

            # rotate back to cartesian
            b_tail_disk_x[i] = b_tail_disk_rho[i] * np.cos(phi[i])
            b_tail_disk_y[i] = b_tail_disk_rho[i] * np.sin(phi[i])

    elif n_vec ==1:

        a_phi = a_phi_hankel_v7(H_current, rho, phi, z_msm, lambda_out, d_0, delta_x,
                                scale_x_d, delta_y)

        # numerically approximate the derivatives
        delta_z = 10 ** (-5)

        d_a_phi_d_z = (a_phi_hankel_v7(H_current, rho, phi, z_msm + delta_z, lambda_out, d_0, delta_x, scale_x_d,
                                    delta_y) - a_phi_hankel_v7(H_current, rho, phi, z_msm - delta_z, lambda_out, d_0,
                                                               delta_x, scale_x_d, delta_y)) / (2 * delta_z)

        delta_rho = 10 ** (-5)
        d_a_phi_d_rho = (a_phi_hankel_v7(H_current, rho + delta_rho, phi, z_msm, lambda_out, d_0, delta_x,
                                        scale_x_d, delta_y) -
                         a_phi_hankel_v7(H_current, rho - delta_rho, phi, z_msm, lambda_out, d_0, delta_x, scale_x_d,
                                        delta_y)) / (2 * delta_rho)

        b_tail_disk_rho = t * (- d_a_phi_d_z)

        if rho <= 10 ** (-4):
            b_tail_disk_z = t * (1.0 + d_a_phi_d_rho)

        else:
            b_tail_disk_z = t * (a_phi / rho + d_a_phi_d_rho)

        # rotate back to cartesian
        b_tail_disk_x = np.asarray(b_tail_disk_rho * np.cos(phi)).astype(float)
        b_tail_disk_y = np.asarray(b_tail_disk_rho * np.sin(phi)).astype(float)


    return np.array([b_tail_disk_x, b_tail_disk_y, b_tail_disk_z])
