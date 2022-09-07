"""
Dokumentation einfügen
"""
import matplotlib.pyplot as plt
from rikitake import rikitake
from numpy import pi, sqrt, real, log10, arange, array


# Erzeugen der Frequenzen
fmin, fmax = 1E-15, 1E5
omega_min, omega_max = 2. * pi * fmin, 2. * pi * fmax
omega_steps = 100
d_omega = (log10(omega_max) - log10(omega_min))/omega_steps
omega = pow(10, arange(log10(omega_min), log10(omega_max), d_omega))

# Schichten spezifizieren
r_arr = array([0, 1740E3, 1940E3, 2040E3, 2300E3, 2440E3])
sigma_arr_h = array([0, 1E7, 1E3, 10**0.5, 10**0.7, 1E-2])
sigma_arr_l = array([0, 1E5, 1E2, 10**-0.5, 10**-3, 1E-7])

# Magnetfeld Ordnung
l = 1

# Berechnung der Rikitake Faktoren
rikitake_h = array([0+0j] * omega_steps)  # hohe Leitfähigkeiten
rikitake_l = array([0+0j] * omega_steps)  # niedrige Leitfähigkeiten

for i in range(0, len(omega)):

    k_arr_h = sqrt((0-1j * omega[i] * 4E-7 * pi * sigma_arr_h))
    k_arr_l = sqrt((0-1j * omega[i] * 4E-7 * pi * sigma_arr_l))

    rikitake_h[i] = rikitake(l, k_arr_h, r_arr)
    rikitake_l[i] = rikitake(l, k_arr_l, r_arr)

# Erzeugen der Diagramme
plt.figure(2)
plt.plot(omega/(2*pi), real(rikitake_h), label='high $\\sigma$', linewidth='2')
plt.plot(omega/(2*pi), real(rikitake_l), label='low $\\sigma$', linewidth='2')
plt.xscale('log')
plt.legend(loc='upper left')
plt.ylabel('Real($\\mathcal{R}$)')
plt.xlabel('$f\\hspace{0.3}(Hz)$')
plt.grid(which='major', axis='both', linestyle='-', color='lavender')
plt.xlim(fmin, fmax)
plt.title("l = " + str(l))

# Bekannte Anregungsfrequenzen

# Liljeblad and Karlsson (2017)
# KH-Oscillations
f30mHz = 30E-3
plt.vlines(f30mHz, 0, l/(l+1), colors='forestgreen', linestyle='dotted')
plt.annotate('30mHz', (f30mHz*1.5, -0.03), color='forestgreen')

# Dungey-cycle
f2min = 1/(2*60)
plt.vlines(f2min, 0, l/(l+1), colors='firebrick', linestyle='dotted')
plt.annotate('2min', (f2min*0.10, -0.03), color='firebrick')

# solar rotation
f642h = 1/(642*3600)
plt.vlines(f642h, 0, l/(l+1), colors='darkorchid', linestyle='dotted')
plt.annotate('642h', (f642h*2.10, -0.03), color='darkorchid')

# planetary rotation
f88 = 1/(88*24*3600)
plt.vlines(f88, 0, l/(l+1), colors='black', linestyle='dotted')
plt.annotate('88d', (f88*0.10, -0.03), color='black')

f44 = 1/(44*24*3600)
plt.vlines(f44, 0, l/(l+1), colors='skyblue', linestyle='dotted')
plt.annotate('44d', (f44*0.50, -0.06), color='skyblue')

f22y = 1/(22*365*24*3600)
plt.vlines(f22y, 0, l/(l+1), colors='goldenrod', linestyle='dotted')
plt.annotate('22y', (f22y*0.10, -0.03), color='goldenrod')

# plt.savefig('plots/rikitake_l=2.jpg', dpi=600)
