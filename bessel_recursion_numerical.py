import matplotlib.pyplot as plt
import numpy as np
from mpmath import mp, besselk
mp.dps = 6
mp.pretty = True


dx = 1E-2
x = np.arange(dx, 10 + dx, dx)


def q_l(l, z):
    return np.sqrt(np.pi / (2*z)) * besselk(l+1/2, z)


def dq_l__dz(l, z):  # auskommentiert ist die Rekursion laut Abra. und Stegun
    # return pow(-1, l) * q_l(l-1, z) - (l+1) / z * pow(-1, l+1) * q_l(l, z)
    return - q_l(l-1, z) - (l+1) / z * q_l(l, z)


for l in range(1, 3):
    y = [dq_l__dz(l, z) for z in x]

    plt.plot(x, y, label="l = " + str(l) + " Rekursion")
    plt.plot(x, np.gradient([q_l(l, z) for z in x], dx),
                label="l = " + str(l) + " Differenzenquotient")

plt.ylim(-1E4, 1E4)
plt.xlabel("$z$")
plt.ylabel("$\\frac{d}{dz}q_l(z)$")
plt.legend()
plt.yscale('symlog')
plt.title("Rekursion ohne alternierendes Vorzeichen")
# plt.savefig('plots/bessel_falsch_2.jpg', dpi=600)
