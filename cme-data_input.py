import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


path = 'KTH_Model_V7/data/output-helios1_e1_np_helios1_e1_vp_0_1_2_1980148000000000.txt'
colnames=['t [s]', 'N_p [1/cm^3]', 'v_r [km/s]', 'v_t [km/s]', 'v_n [km/s]'] 
df = pd.read_csv(
    path, header=None, delim_whitespace=True, names=colnames, skiprows=89
    )

t = df['t [s]']
t0 = datetime.strptime(t[0], '%Y-%m-%dT%H:%M:%S.%f').timestamp()
for i in t:
    t  = t.replace(i, datetime.strptime(i, '%Y-%m-%dT%H:%M:%S.%f').timestamp()-t0)

v_r, v_t, v_n = df['v_r [km/s]'], df['v_t [km/s]'], df['v_n [km/s]']
v = np.sqrt(v_r**2 + v_t**2 + v_n**2)

plt.plot(t, v)
plt.suptitle("Helios 1 at $r_{hel} = 0.31$ AU")
plt.title("1980-05-28T00:00:00.000 - 1980-06-01T23:59:00.000")
plt.xlabel("$t$ [$s$]")
plt.ylabel("$v_{sw}$ [$km/s$]")

# rho = 5E-6
# p_dyn = 1/2 * rho * v**2
