import numpy as np


def internal_field_v7(x_msm, y_msm, z_msm, control_params):
    # this calculates the magnetic field of an internal axisymmetric
    # dipole in a standard spherical harmonic expansion. The field
    # is then rotated back to the cartesian coordinate system base.

    # INPUT COORDINATES ARE IN PLANETARY RADII



    g10_int_ind = control_params['g10_int_ind']
    g10 = control_params['g10_int']

    # transform to MSO coordinates

    x_mso = np.array(x_msm)
    y_mso = np.array(y_msm)
    z_mso = np.array(z_msm) + 0.196

    r_mso = np.sqrt(x_mso ** 2 + y_mso ** 2 + z_mso ** 2)
    phi_mso = np.arctan2(y_mso, x_mso)
    theta_mso = np.arccos(z_mso / r_mso)

    # spherical harmonic synthesis of axisymmetric components
    # internal dipole coefficient g_1**0 = -190 nT from Anderson et al. 2012

    g20 = -74.6
    g30 = -22.0
    g40 = -5.7

    # l=1
    b_r_dip = 2 * (1 / r_mso) ** 3 * (g10 + g10_int_ind) * np.cos(theta_mso)
    b_t_dip = (1 / r_mso) ** 3 * (g10 + g10_int_ind) * np.sin(theta_mso)


    # l=2
    b_r_quad = 3 * (1 / r_mso) ** 4 * g20 * 0.5 * (3 * np.cos(theta_mso) ** 2 - 1)
    b_t_quad = (1 / r_mso) ** 4 * g20 * 3 * (np.cos(theta_mso) * np.sin(theta_mso))


    # l=3
    b_r_oct = 4 * (1 / r_mso) ** 5 * g30 * 0.5 * (5 * np.cos(theta_mso) ** 3 - 3 * np.cos(theta_mso))
    b_t_oct = (1 / r_mso) ** 5 * g30 * 0.375 * (np.sin(theta_mso) + 5 * np.sin(3 * theta_mso))


    # l=4
    b_r_hex = 5 * (1 / r_mso) ** 6 * g40 * (0.125 * (35 * np.cos(theta_mso) ** 4 - 30 * np.cos(theta_mso) ** 2 + 3))
    b_t_hex = (1 / r_mso) ** 6 * g40 * (0.3125 * (2 * np.sin(2 * theta_mso) + 7 * np.sin(4 * theta_mso)))


    # add multipoles together
    b_r = b_r_dip + b_r_quad + b_r_oct + b_r_hex
    b_t = b_t_dip + b_t_quad + b_t_oct + b_t_hex


    # rotate to mso coordinate base
    b_x_mso_int = b_r * np.sin(theta_mso) * np.cos(phi_mso) + b_t * np.cos(theta_mso) * np.cos(phi_mso)
    b_y_mso_int = b_r * np.sin(theta_mso) * np.sin(phi_mso) + b_t * np.cos(theta_mso) * np.sin(phi_mso)
    b_z_mso_int = b_r * np.cos(theta_mso) - b_t * np.sin(theta_mso)



    return np.array([b_x_mso_int, b_y_mso_int, b_z_mso_int])
