"""
Dokumentation einfuegen
"""
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.constants import m_p
import matplotlib.pyplot as plt


# =============================================================================
# get the data from the .txt file
# =============================================================================
PATH = 'KTH_Model_V7/data/output-helios1_e1_np_helios1_e1_vp_0_1_2_1980148000000000.txt'
colnames = ['t [s]', 'N_p [1/cm^3]', 'v_r [km/s]', 'v_t [km/s]', 'v_n [km/s]']
df = pd.read_csv(
    PATH, header=None, delim_whitespace=True, names=colnames, skiprows=89
    )

t = df['t [s]']
t_regex = '%Y-%m-%dT%H:%M:%S.%f'
t0 = datetime.strptime(t[0], t_regex).timestamp()
for i in t:
    t  = t.replace(i, datetime.strptime(i, t_regex).timestamp() - t0)
N_p = df['N_p [1/cm^3]']
N_p = N_p * 1E6  # 1/cm^3 -> 1/m^3
v_r, v_t, v_n = df['v_r [km/s]'], df['v_t [km/s]'], df['v_n [km/s]']
v = np.sqrt(v_r**2 + v_t**2 + v_n**2)
v = v * 1E3  # km/s -> m/s

# =============================================================================
# calculate the heliocentric distance using the solarwind velocity
# ============================================================================="
"""
source: Pognan, Q., C. Garraffo, O. Cohen, and J. J. Drake (2018),
        The Solar Wind Environment in Time,
        The Astrophysical Journal, 856 (1), 53, doi:10.3847/1538-4357/aaaebb.
"""
rho = m_p * N_p
p_dyn = 1/2 * rho * v**2
T_SOLAR = 4.6E9  # alter der sonne in jahren
pseudo_distance = np.sqrt(6.1E-7/p_dyn * (T_SOLAR/1E6)**(-0.67))

# =============================================================================
# the KTH-Modell can't process R_SS < R_M - remove those data points
# =============================================================================
di = 50
R_M = 1
f = 2.0695 - (0.00355 * di)
R_SS = f * pseudo_distance ** (1 / 3) * R_M

for i in range(len(R_SS)):
    if R_SS[i] < 1:
        R_SS[i] = np.nan
        pseudo_distance[i] = np.nan

# =============================================================================
# interpolate data, required for fft
# =============================================================================
pseudo_distance = pseudo_distance.interpolate(method='linear').tolis
t()
"""
pseudo_distance is the parameter that is required for the KTH-Modell
"""
# =============================================================================
# save the data
# =============================================================================


# =============================================================================
# plot data
# =============================================================================
# plt.scatter(t, R_SS, s=4)
# plt.suptitle("Data from Helios 1 at $r_{hel} = 0.31$ AU")
# plt.title("1980-05-28T00:00:00.000 - 1980-06-01T23:59:00.000")
# plt.xlabel("$t$ [$s$]")
# plt.ylabel("$R_SS}$ [$R_{Mercury}$]")
