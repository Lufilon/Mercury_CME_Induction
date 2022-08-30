from numpy import sin, cos, tan, sqrt, pi
from numpy import zeros, arange, meshgrid
from math import factorial
from scipy.special import lpmv
from scipy.integrate import simpson


def a_lm(l, m):
    if l <= 0:
        return 0
    else:
        return l * (m - l - 1) / (2 * l + 1)


def b_lm(l, m):
    return (l + 1) * (l + m) / (2 * l + 1)


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

        # print("V_lm", V_lm_arr[0])
        # print("GT_lm[1] ", GT_lm[1])
        # print("GR_lm_arr[1] ", GR_lm_arr[1])

        # finally calculate the actual gauss coefficients
        for l in range(m, degree_max + 1):
            coeff_int[l] = 1/(2*l + 1) * (ana_radius/ref_radius)**(l+2) * (GR_lm_arr[l] - l * GT_lm[l])
            coeff_ext[l] = -1/(2*l + 1) * (ref_radius/ana_radius)**(l-1) * (GR_lm_arr[l] + (l+1) * GT_lm[l])

        # print("G_theta[1]", GT_lm[1])

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

theta_arr = arange(1E-4, pi, pi/n_theta)  # wegen 1/sin(x) -> nicht bei 0 beginn
phi_arr = arange(1E-4, 2*pi, 2*pi/n_phi)  # kann später vermutlich geändert werden
phi_arr_2D, theta_arr_2D = meshgrid(phi_arr, theta_arr)

# =============================================================================
# generating the internal test magnetic field
# =============================================================================
g_1_0_int, g_1_1_int = 10, 20
h_1_1_int = 30
g_2_0_int, g_2_1_int, g_2_2_int = 40, 50, 60
h_2_1_int, h_2_2_int = 70, 80

br, bp, bt = 0, 0, 0
# case l=1, m=0
br += 2 * g_1_0_int * cos(theta_arr_2D)
bp += 0
bt += g_1_0_int * sin(theta_arr_2D)
# case l=1, m=1
br += 2 * (g_1_1_int*cos(1*phi_arr_2D) + h_1_1_int*sin(1*phi_arr_2D)) * cos(theta_arr_2D)
bp += (g_1_1_int*sin(1*phi_arr_2D) - h_1_1_int*cos(1*phi_arr_2D)) * 1/tan(theta_arr_2D)
bt += (g_1_1_int*cos(1*phi_arr_2D) + h_1_1_int*sin(1*phi_arr_2D)) * sin(theta_arr_2D)
# case l=2, m=0
br += 3 * g_2_0_int * (3 * cos(theta_arr_2D)**2 - 1)/2
bp += 0
bt += g_2_0_int * (3 * cos(theta_arr_2D) * sin(theta_arr_2D))   
# case l=2, m=1
br += 3 * (g_2_1_int*cos(1*phi_arr_2D) + h_2_1_int*sin(1*phi_arr_2D)) * (3 * cos(theta_arr_2D)**2 - 1)/2
bp += (g_2_1_int*sin(1*phi_arr_2D) - h_2_1_int*cos(1*phi_arr_2D)) * (3 * cos(theta_arr_2D)**2 - 1)/(2*sin(theta_arr_2D))
bt += (g_2_1_int*cos(1*phi_arr_2D) + h_2_1_int*sin(1*phi_arr_2D)) * (3 * cos(theta_arr_2D) * sin(theta_arr_2D))
# case l=2, m=2
br += 3 * (g_2_2_int*cos(2*phi_arr_2D) + h_2_2_int*sin(2*phi_arr_2D)) * (3 * cos(theta_arr_2D)**2 - 1)/2
bp += 2 * (g_2_2_int*sin(2*phi_arr_2D) - h_2_2_int*cos(2*phi_arr_2D)) * (3 * cos(theta_arr_2D)**2 - 1)/(2*sin(theta_arr_2D))
bt += (g_2_2_int*cos(2*phi_arr_2D) + h_2_2_int*sin(2*phi_arr_2D)) * (3 * cos(theta_arr_2D) * sin(theta_arr_2D))

# =============================================================================
# generating the external test magnetic field
# =============================================================================
g_1_0_ext, g_1_1_ext = 10, 20
h_1_1_ext = 30
g_2_0_ext, g_2_1_ext, g_2_2_ext = 40, 50, 60
h_2_1_ext, h_2_2_ext = 70, 80

# case l=1, m=0
br += - g_1_0_ext * cos(theta_arr_2D)
bp += 0
bt += g_1_0_ext * sin(theta_arr_2D)
# case l=1, m=1
br += - (g_1_1_ext*cos(phi_arr_2D) + h_1_1_ext*sin(phi_arr_2D)) * cos(theta_arr_2D)
bp += (g_1_1_ext*sin(phi_arr_2D) - h_1_1_ext*cos(phi_arr_2D)) * 1/tan(theta_arr_2D)
bt +=  (g_1_1_ext*cos(phi_arr_2D) + h_1_1_ext*sin(phi_arr_2D))* sin(theta_arr_2D)
# case l=2, m=0
br += - 2 * g_2_0_ext * (3 * cos(theta_arr_2D)**2 - 1)/2
bp += 0
bt += g_2_0_ext * (3 * cos(theta_arr_2D) * sin(theta_arr_2D))   
# case l=2, m=1
br += - 2* (g_2_1_ext*cos(phi_arr_2D) + h_2_1_ext*sin(phi_arr_2D)) * (3 * cos(theta_arr_2D)**2 - 1)/2
bp += (g_2_1_ext*sin(phi_arr_2D) - h_2_1_ext*cos(phi_arr_2D)) * (3 * cos(theta_arr_2D)**2 - 1)/(2*sin(theta_arr_2D))
bt += (g_2_1_ext*cos(phi_arr_2D) + h_2_1_ext*sin(phi_arr_2D)) * (3 * cos(theta_arr_2D) * sin(theta_arr_2D))
# case l=2, m=2
br += - 2 * (g_2_2_ext*cos(2*phi_arr_2D) + h_2_2_ext*sin(2*phi_arr_2D)) * (3 * cos(theta_arr_2D)**2 - 1)/2
bp += 2 * (g_2_2_ext*sin(2*phi_arr_2D) - h_2_2_ext*cos(2*phi_arr_2D)) * (3 * cos(theta_arr_2D)**2 - 1)/(2*sin(theta_arr_2D))
bt += (g_2_2_ext*cos(2*phi_arr_2D) + h_2_2_ext*sin(2*phi_arr_2D)) * (3 * cos(theta_arr_2D) * sin(theta_arr_2D))

# wenn Bx,By,Bz (etc) aus Datei importiert wird muss B_{x,y,z}.reshape(n_theta, n_phi) passieren

# =============================================================================
# Parameter
# =============================================================================
ref_radius = 1
ana_radius = ref_radius
degree_max = 3
m = 0
if m > degree_max:
    raise SystemExit("target Order can't be larger than the maximum degree")

coeff_int, coeff_ext = SHA_by_integration(br, bp, bt, ana_radius, ref_radius,
                                          theta_arr, phi_arr, degree_max, m)

print("g_1" + str(m) + "_int", coeff_int[1])
print("g_2" + str(m) + "_int", coeff_int[2])
# print("g_3" + str(m) + "_int", coeff_int[3])  # es sollte |g3m_int|<<1 sein
print("g_1" + str(m) + "_ext", coeff_ext[1])
print("g_2" + str(m) + "_ext", coeff_ext[2])
# print("g_3" + str(m) + "_ext", coeff_ext[3])  # es sollte |g3m_ext|<<1 sein
