"""
Dokumentation einfuegen
"""
from math import factorial
from numpy import sin, cos, tan, sqrt, pi
from numpy import zeros, linspace, meshgrid, asarray, ravel
from scipy.special import lpmv
from scipy.integrate import simpson
import matplotlib.pyplot as plt
import pyshtools.legendre as leg
from pyshtools.legendre import PlmIndex as legind


def a_lm(l, m):
    if l <= 0:
        return 0
    return l * (m - l - 1) / (2 * l + 1)


def b_lm(l, m):
    return (l + 1) * (l + m) / (2 * l + 1)


def calcLegPol(lmax, theta_arr, csph=False):
    """
    This function computes the associated Legendre polynomials and their
    derivatives with respect to the colatitude for a given maximal spherical
    harmonic degree and a given colatitude. Here the shtools library
    (https://shtools.oca.eu) is used for the calculations. shtools calculates
    the Legendre polynomials with respect to a variable z, which is chosen to
    be z=cos(theta) in this case. Therefore also a correction is needed for
    the derivative.

    Parameters
    ----------
    lmax : int
        maximal spherical harmonic degree for the calculation.
    theta : float
        colatidude in radian.
    csph: bool, optional
        Usage of a Condon-Shortley phase. If True then the CS phase is
        included. Default is False.

    Returns
    --------
    Plm : float
        Array with all Legendre polynimals of degree and order (l,m) = lmax
        for the given colatitude
    dPlm : float
        Array with the corresponding derivatives of Plm with respect to the
        colatitude

    """
    if csph:
        cs = -1
    else:
        cs = 1

    Plm, dPlm = leg.PlmSchmidt_d1(lmax, cos(theta_arr), csphase=cs)
    dPlm *= -sin(theta_arr)

    return Plm, dPlm


def P(l, m, x):
    """
    An auxilliary function to calculate the Schmidt quasi-normalized
    associated Legendre-polynomials of
    degree l and order m for the variable x
    Parameters
    ----------
    l : int
        Degree of the Legendre polynomial
    m : int
        Order of the Legendre polynomial
    x : float
        Input to evaluate the Legendre polynomial at
    Returns
    -------
    None.
    """

    # Schmid quasi-normalization
    if m == 0:
        norm = sqrt(factorial(l-m) / factorial(l+m))
    else:
        norm = sqrt(2 * factorial(l-m) / factorial(l+m))

    # compensate also for the condon phase factor
    return pow(-1, m) * norm * lpmv(m, l, x)


def SHA_by_integration(br, bp, bt, ana_radius, ref_radius, theta_arr, phi_arr,
                       degree_max, m):
    n_phi = len(phi_arr)

    c_theta = cos(theta_arr)
    s_theta = sin(theta_arr)

    coeff_int = zeros(degree_max + 1)
    coeff_ext = zeros(degree_max + 1)

    # analysis of the B_r component
    # this is independent of the order
    GR_lm_arr = zeros(degree_max + 1)  # first entry not used
    integrand_r = zeros(n_phi)
    for l in range(m, degree_max + 1):

        # integration over theta for every phi
        for i_phi in range(0, n_phi):
            integrand_r[i_phi] = simpson(br[:, i_phi] * P(l, m, c_theta)
                                         * s_theta, theta_arr)

        # integration over phi
        int_result = simpson(integrand_r * cos(m * phi_arr), phi_arr)
        GR_lm_arr[l] = (2 * l + 1) / (4 * pi) * int_result

    if m == 0:
        # analysis of the B_theta component  # (2.40) bis (2.44)
        V_lm_arr = zeros(degree_max + 2)

        for l in range(0, degree_max + 2):
            integrand_t = zeros(n_phi)

            # integration over theta for every phi
            for i_phi in range(0, n_phi):
                integrand_t[i_phi] = simpson(bt[:, i_phi] * P(l, m, c_theta)
                                             * s_theta**2, theta_arr)

            # integration over phi
            int_result = simpson(integrand_t * cos(m * phi_arr), phi_arr)
            # if l == 1:
                # print("GT_int-result", int_result/1.2)

            V_lm_arr[l] = (2 * l + 1) / (4 * pi) * int_result

        # successive calculation of general Gauss-coefficients
        # l is the actual degree; the order is always the same: m
        GT_lm = zeros(degree_max + 1)

        for l in range(1, degree_max + 1):
            if l-2 <= 0:
                GT_lm[l] = -V_lm_arr[l-1] / b_lm(l, m)
            else:
                GT_lm[l] = - (V_lm_arr[l-1] - a_lm(l-2, m) / b_lm(l-2, m)
                                * V_lm_arr[l-3]) / b_lm(l, m)

        # finally calculate the actual gauss coefficients
        for l in range(m, degree_max + 1):
            coeff_int[l] = 1/(2*l + 1) * (ana_radius/ref_radius)**(l+2) * (GR_lm_arr[l] - l * GT_lm[l])
            coeff_ext[l] = -1/(2*l + 1) * (ref_radius/ana_radius)**(l-1) * (GR_lm_arr[l] + (l+1) * GT_lm[l])

    else:  # m > 0
        # analysis of the B_phi component  # (2.45)
        GP_lm_arr = zeros(degree_max + 1)
        integrand_p = zeros(n_phi)
        for l in range(m, degree_max + 1):
            # integration over theta for every phi
            for i_phi in range(0, n_phi):
                integrand_p[i_phi] = simpson(bp[:, i_phi] * P(l, m, c_theta)
                                             * s_theta**2, theta_arr)
            # integration over phi
            int_result = simpson(integrand_p * sin(m * phi_arr), phi_arr)
            # if l == 1:
                # print("GP_int-result", int_result)

            GP_lm_arr[l] = (2*l + 1)/(4*pi) * int_result

        # calculate the gauss coefficients  # (2.46)
        for l in range(m, degree_max + 1):
            coeff_int[l] = (l*GP_lm_arr[l] + m*GR_lm_arr[l])/((2*l + 1)*m) * (ana_radius/ref_radius)**(l+2)
            coeff_ext[l] = ((l+1)*GP_lm_arr[l] - m*GR_lm_arr[l])/((2*l + 1)*m) * (ana_radius/ref_radius)**(-l+1)

        # print("G_phi[1]", GP_lm_arr[1])

    return coeff_int, coeff_ext


# =============================================================================
# # test with Gauss-coefficients
# =============================================================================
n_theta = int(200)
n_phi = int(2 * n_theta)

# theta_arr = linspace(1E-4, pi, n_theta, endpoint=False)
# phi_arr = linspace(1E-4, 2*pi, n_phi, endpoint=False)
# =============================================================================
# temp
# =============================================================================
phi_arr = linspace(1E-4, 2*pi, n_phi, endpoint=False)
theta_arr = linspace(1E-4, pi, n_theta, endpoint=False)
# =============================================================================
# temp
# =============================================================================
phi_arr_2D, theta_arr_2D = meshgrid(phi_arr, theta_arr)

# =============================================================================
# generating the internal test magnetic field
# =============================================================================
def dP(l, m, theta_arr_2D):
    theta_temp = ravel(theta_arr_2D)
    return asarray([calcLegPol(3, theta)[1][legind(l, m)] for theta in theta_temp]).reshape(n_theta, n_phi)
g_1_0_int, g_1_1_int = -192, 2.6
h_1_1_int = 0.1
g_2_0_int, g_2_1_int, g_2_2_int = -78, -2, 0
h_2_1_int, h_2_2_int = 0, -1

br, bp, bt = 0, 0, 0
# case l=1, m=0
br += 2 * g_1_0_int * P(1, 0, cos(theta_arr_2D))
bp += 0
bt += - g_1_0_int * dP(1, 0, theta_arr_2D)
# case l=1, m=1
br += 2 * (g_1_1_int*cos(1*phi_arr_2D) + h_1_1_int*sin(1*phi_arr_2D)) * P(1, 1, cos(theta_arr_2D))
bp += (g_1_1_int*sin(1*phi_arr_2D) - h_1_1_int*cos(1*phi_arr_2D)) * P(1, 1, cos(theta_arr_2D))/sin(theta_arr_2D)
bt += - (g_1_1_int*cos(1*phi_arr_2D) + h_1_1_int*sin(1*phi_arr_2D)) * dP(1, 1, theta_arr_2D)
# case l=2, m=0
br += 3 * g_2_0_int * P(2, 0, cos(theta_arr_2D))
bp += 0
bt += - g_2_0_int * dP(2, 0, theta_arr_2D)
# case l=2, m=1
br += 3 * (g_2_1_int*cos(1*phi_arr_2D) + h_2_1_int*sin(1*phi_arr_2D)) * P(2, 1, cos(theta_arr_2D))
bp += (g_2_1_int*sin(1*phi_arr_2D) - h_2_1_int*cos(1*phi_arr_2D)) * P(2, 1, cos(theta_arr_2D))/sin(theta_arr_2D)
bt += - (g_2_1_int*cos(1*phi_arr_2D) + h_2_1_int*sin(1*phi_arr_2D)) * dP(2, 1, theta_arr_2D)
# case l=2, m=2
br += 3 * (g_2_2_int*cos(2*phi_arr_2D) + h_2_2_int*sin(2*phi_arr_2D)) * P(2, 2, cos(theta_arr_2D))
bp += 2 * (g_2_2_int*sin(2*phi_arr_2D) - h_2_2_int*cos(2*phi_arr_2D)) * P(2, 2, cos(theta_arr_2D))/sin(theta_arr_2D)
bt += - (g_2_2_int*cos(2*phi_arr_2D) + h_2_2_int*sin(2*phi_arr_2D)) * dP(2, 2, theta_arr_2D)
# =============================================================================
# generating the external test magnetic field
# =============================================================================
g_1_0_ext, g_1_1_ext = 40, 2
h_1_1_ext = 10
g_2_0_ext, g_2_1_ext, g_2_2_ext = 0, 20, 0
h_2_1_ext, h_2_2_ext = 0, 0

# case l=1, m=0
br += - g_1_0_ext * P(1, 0, cos(theta_arr_2D))
bp += 0
bt += - g_1_0_ext * dP(1, 0, theta_arr_2D)
# case l=1, m=1
br += - (g_1_1_ext*cos(phi_arr_2D) + h_1_1_ext*sin(phi_arr_2D)) * P(1, 1, cos(theta_arr_2D))
bp += (g_1_1_ext*sin(phi_arr_2D) - h_1_1_ext*cos(phi_arr_2D)) *P(1, 1, cos(theta_arr_2D))/sin(theta_arr_2D)
bt += - (g_1_1_ext*cos(phi_arr_2D) + h_1_1_ext*sin(phi_arr_2D)) * dP(1, 1, theta_arr_2D)
# case l=2, m=0
br += - 2 * g_2_0_ext * P(2, 0, cos(theta_arr_2D))
bp += 0
bt += - g_2_0_ext * dP(2, 0, theta_arr_2D)
# case l=2, m=1
br += - 2* (g_2_1_ext*cos(phi_arr_2D) + h_2_1_ext*sin(phi_arr_2D)) * P(2, 1, cos(theta_arr_2D))
bp += (g_2_1_ext*sin(phi_arr_2D) - h_2_1_ext*cos(phi_arr_2D)) * P(2, 1, cos(theta_arr_2D))/sin(theta_arr_2D)
bt += - (g_2_1_ext*cos(phi_arr_2D) + h_2_1_ext*sin(phi_arr_2D)) * dP(2, 1, theta_arr_2D)
# case l=2, m=2
br  += - 2 * (g_2_2_ext*cos(2*phi_arr_2D) + h_2_2_ext*sin(2*phi_arr_2D)) * P(2, 2, cos(theta_arr_2D))
bp += 2 * (g_2_2_ext*sin(2*phi_arr_2D) - h_2_2_ext*cos(2*phi_arr_2D)) * P(2, 2, cos(theta_arr_2D))/sin(theta_arr_2D)
bt += - (g_2_2_ext*cos(2*phi_arr_2D) + h_2_2_ext*sin(2*phi_arr_2D)) * dP(2, 2, theta_arr_2D)

# if Bx,By,Bz is imported from file, B_{x,y,z}.reshape(n_theta, n_phi) required

# =============================================================================
# Parameter
# =============================================================================
ref_radius = 1
ana_radius = ref_radius
degree_max = 3
target_order = 1
if target_order > degree_max:
    raise SystemExit("target Order can't be larger than the maximum degree")

coeff_int, coeff_ext = SHA_by_integration(br, bp, bt, ana_radius, ref_radius,
                                          theta_arr, phi_arr, degree_max,
                                          target_order)

print("g_1" + str(target_order) + "_int", coeff_int[1])
print("g_2" + str(target_order) + "_int", coeff_int[2])
# print("g_3" + str(target_order) + "_int", coeff_int[3])
print("g_1" + str(target_order) + "_ext", coeff_ext[1])
print("g_2" + str(target_order) + "_ext", coeff_ext[2])
# print("g_3" + str(target_order) + "_ext", coeff_ext[3])


# # =============================================================================
# #  test
# # =============================================================================
# def dP(l, m, theta):
#     return asarray([calcLegPol(degree_max, theta)[1][legind(l, m)] for theta in theta_arr])

# BR_int, BP_int, BT_int = 0, 0, 0
# # case l=1, m=0
# BR_int += 2 * g_1_0_int * P(1, 0, cos(theta_arr))
# BP_int += 0
# BT_int += - g_1_0_int * dP(1, 0, theta_arr)
# # case l=1, m=1
# BR_int += 2 * (g_1_1_int*cos(1*phi_arr) + h_1_1_int*sin(1*phi_arr)) * P(1, 1, cos(theta_arr))
# BP_int += (g_1_1_int*sin(1*phi_arr) - h_1_1_int*cos(1*phi_arr)) * P(1, 1, cos(theta_arr))/sin(theta_arr)
# BT_int += - (g_1_1_int*cos(1*phi_arr) + h_1_1_int*sin(1*phi_arr)) * dP(1, 1, theta_arr)
# # case l=2, m=0
# BR_int += 3 * g_2_0_int * P(2, 0, cos(theta_arr))
# BP_int += 0
# BT_int += - g_2_0_int * dP(2, 0, theta_arr)
# # case l=2, m=1
# BR_int += 3 * (g_2_1_int*cos(1*phi_arr) + h_2_1_int*sin(1*phi_arr)) * P(2, 1, cos(theta_arr))
# BP_int += (g_2_1_int*sin(1*phi_arr) - h_2_1_int*cos(1*phi_arr)) * P(2, 1, cos(theta_arr))/sin(theta_arr)
# BT_int += - (g_2_1_int*cos(1*phi_arr) + h_2_1_int*sin(1*phi_arr)) * dP(2, 1, theta_arr)
# # case l=2, m=2
# BR_int += 3 * (g_2_2_int*cos(2*phi_arr) + h_2_2_int*sin(2*phi_arr)) * P(2, 2, cos(theta_arr))
# BP_int += 2 * (g_2_2_int*sin(2*phi_arr) - h_2_2_int*cos(2*phi_arr)) * P(2, 2, cos(theta_arr))/sin(theta_arr)
# BT_int += - (g_2_2_int*cos(2*phi_arr) + h_2_2_int*sin(2*phi_arr)) * dP(2, 2, theta_arr)

# BR_ext, BP_ext, BT_ext = 0, 0, 0
# # case l=1, m=0
# BR_ext += - g_1_0_ext * P(1, 0, cos(theta_arr))
# BP_ext += 0
# BT_ext += - g_1_0_ext * dP(1, 0, theta_arr)
# # case l=1, m=1
# BR_ext += - (g_1_1_ext*cos(phi_arr) + h_1_1_ext*sin(phi_arr)) * P(1, 1, cos(theta_arr))
# BP_ext += (g_1_1_ext*sin(phi_arr) - h_1_1_ext*cos(phi_arr)) *P(1, 1, cos(theta_arr))/sin(theta_arr)
# BT_ext += - (g_1_1_ext*cos(phi_arr) + h_1_1_ext*sin(phi_arr)) * dP(1, 1, theta_arr)
# # case l=2, m=0
# BR_ext += - 2 * g_2_0_ext * P(2, 0, cos(theta_arr))
# BP_ext += 0
# BT_ext += - g_2_0_ext * dP(2, 0, theta_arr)
# # case l=2, m=1
# BR_ext += - 2* (g_2_1_ext*cos(phi_arr) + h_2_1_ext*sin(phi_arr)) * P(2, 1, cos(theta_arr))
# BP_ext += (g_2_1_ext*sin(phi_arr) - h_2_1_ext*cos(phi_arr)) * P(2, 1, cos(theta_arr))/sin(theta_arr)
# BT_ext += - (g_2_1_ext*cos(phi_arr) + h_2_1_ext*sin(phi_arr)) * dP(2, 1, theta_arr)
# # case l=2, m=2
# BR_ext  += - 2 * (g_2_2_ext*cos(2*phi_arr) + h_2_2_ext*sin(2*phi_arr)) * P(2, 2, cos(theta_arr))
# BP_ext += 2 * (g_2_2_ext*sin(2*phi_arr) - h_2_2_ext*cos(2*phi_arr)) * P(2, 2, cos(theta_arr))/sin(theta_arr)
# BT_ext += - (g_2_2_ext*cos(2*phi_arr) + h_2_2_ext*sin(2*phi_arr)) * dP(2, 2, theta_arr)

# # plt.plot(BR_int, label="B_r,int, SHA")
# # plt.plot(BP_int, label="B_phi,int, SHA")
# # plt.plot(BT_int, label="B_theta,int, SHA")
# # plt.plot(BR_ext, label="B_r,ext, SHA")
# # plt.plot(BP_ext, label="B_phi,ext, SHA")
# # plt.plot(BT_ext, label="B_theta,ext, SHA")
# # plt.legend()
