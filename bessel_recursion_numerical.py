"""
Programm to verify numerically that the recursion formula in "Abramowitz and
Stegun: Handbook of Mathematical Functions (1972)" is faulty.
"""
import matplotlib.pyplot as plt
import numpy as np
from mpmath import mp, besselk
mp.dps = 6
mp.pretty = True
plt.rcParams.update({'font.size': 10})


def q_l(l, z):
    """
    Description
    ----------
        Function to calculate the modified Bessel function of the second kind
        with a scaling factor with degree l and order m for the variable z.
    Parameters
    ----------
        l : int
            Degree of the Bessel function
        m : int
            Order of the Bessel function
        z : float
            Input to evaluate the Bessel function at
    Returns
    -------
        <class 'mpmath.ctx_mp_python.mpf'>
    """
    return np.sqrt(np.pi / (2 * z)) * besselk(l+1 / 2, z)


def dq_l__dz(l, z):
    """
    Description
    ----------
        Function to calculate the first derivative with respect to z of the
        modified Bessel function of the second kind with degree l and order m
        for the variable z using a recursion formula.
        formulat
    Parameters
    ----------
        l : int
            Degree of the Bessel function
        m : int
            Order of the Bessel function
        z : float
            Input to evaluate the Bessel function at
    Returns
    -------
        <class 'mpmath.ctx_mp_python.mpf'>
    """
    if abra == True:
        return pow(-1, l) * q_l(l-1, z) - (l+1) / z * pow(-1, l+1) * q_l(l, z)
    else: 
        return - q_l(l-1, z) - (l+1) / z * q_l(l, z)


# =============================================================================
# where to evaluate the recursion at and which one to use
# =============================================================================
dx = 1E-2  # can't be 0 because of divergence of bessel function
x = np.arange(dx, 10 + dx, dx)
"""
abra = True means the (faulty) recursion from "Abramowitz and Stegun: Handbook
of Mathematical Functions (1972)" is used.
"""
abra = True  # controls if the wrong or correct equation is used

# =============================================================================
# plot the first derivative of the bessel function for l = start, .., stop
# =============================================================================
start, stop = 2, 4
for l in range(start, stop + 1, 2):
    y = [dq_l__dz(l, z) for z in x]

    plt.plot(x, y, label="l = " + str(l) + " recursion")
    plt.plot(x, np.gradient([q_l(l, z) for z in x], dx),
                label="l = " + str(l) + " difference quotient")
plt.ylim(-1E4, 1E4)
plt.yscale('symlog')
plt.xlabel("$z$")
plt.ylabel("$d_z q_l(z)$")
plt.legend()
if abra == True:
    plt.title("Recursion according to Abramowitz and Stegun (1972)")
else:
    plt.title("Correct recursion equation")
plt.tight_layout()
# plt.savefig('plots/recursion_correct_2.jpg', dpi=600)
