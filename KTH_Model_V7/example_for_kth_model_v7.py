from numpy import pi, sin, cos, sqrt
from numpy import arange, full, meshgrid, ravel
from kth14_model_for_mercury_v7 import kth14_model_for_mercury_v7
import json
import os
import matplotlib.pyplot as plt
from PIL import Image


def data(r_hel_val, di_val, settings):
    x = R_M * sin(theta) * cos(phi)
    y = R_M * sin(theta) * sin(phi)
    z = R_M * cos(theta)
    r_hel = full(num_pts, float(r_hel_val))  # [0.4, 0.47]
    di = full(num_pts, float(di_val))  # 0.5 is mean disturbance
    kappa = 3464.8/(2.0695 - (0.00355 * di_val) * r_hel_val**(1/3) * R_M)

    new = {'settings': settings, 'kappa': kappa}

    return x, y, z, r_hel, di, new


def save_and_plot():
    def cart2sph():
        B_r = sin(theta)*cos(phi) * B_x + sin(theta)*sin(phi) * B_y
        + cos(theta) * B_z
        B_theta = cos(theta)*cos(phi) * B_x + cos(theta)*sin(phi) * B_y
        - sin(theta) * B_z
        B_phi = - sin(phi) * B_x + cos(phi) * B_y

        return B_r, B_theta, B_phi

    def save():
        with open('data/sph_coords.json', 'w') as f:
            json.dump({"r": R_M, "phi": list(phi),
                       "theta": list(theta)}, f)
        with open('data/sph_magnetic.json', 'w') as f:
            json.dump({"B_r": list(B_r), "B_phi": list(B_phi),
                       "B_theta": list(B_theta)}, f)

    def mollweide_plot():
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='mollweide')
        ax.set_title("r_hel = " + str(r_hel_val) + ", di = " + str(di_val))
        plt.suptitle("dipole, neutralsheet, prc, internal, external = "
                     + str(new['settings']))
        ax.grid()
        lon, lat = phi_arr - pi, theta_arr - pi/2
        im = ax.contourf(lon, lat, B.reshape(n_theta, n_phi), levels=10)
        cbar = plt.colorbar(im)
        cbar.set_label('$B$ [$nT$]')
        plt.savefig('plots/mollweide.png', dpi=600)

    if copyExists:
        print("No parameter was changed.")
        img = Image.open('plots/mollweide.png')
        img.show()
    else:
        B_r, B_theta, B_phi = cart2sph()
        B = sqrt(B_r**2 + B_theta**2 + B_phi**2)
        save()
        mollweide_plot()


# =============================================================================
# parameters
# =============================================================================
R_M = 2440
dipole, neutralsheet, prc, internal, external = True, False, False, False, True
r_hel_val, di_val = 0.5, 50

R_M = R_M * (1 + 1E-6)  # factor needed cause of rounding in model_field_v7
settings = [dipole, neutralsheet, prc, internal, external]

n_theta = int(200)
n_phi = int(2*n_theta)
num_pts = int(n_theta * n_phi)
theta_arr = arange(0, pi, pi/n_theta)
phi_arr = arange(0, 2*pi, 2*pi/n_phi)
phi_arr_2D, theta_arr_2D = meshgrid(phi_arr, theta_arr)
phi, theta = ravel(phi_arr_2D), ravel(theta_arr_2D)


# =============================================================================
# creating the data points
# =============================================================================
x, y, z, r_hel, di, new = data(r_hel_val, di_val, settings)


# =============================================================================
# calculating the magnetic field components
# =============================================================================
copyExists = False
if os.path.exists('data/parameters.json'):
    with open('data/parameters.json') as f:
        old = json.load(f)
    if old == new:
        copyExists = True
    else:
        B_x, B_y, B_z = kth14_model_for_mercury_v7(
            x, y, z, r_hel, di, dipole, neutralsheet, prc, internal, external
            )
        with open('data/parameters.json', 'w') as f:
            json.dump(new, f)
else:
    B_x, B_y, B_z = kth14_model_for_mercury_v7(
        x, y, z, r_hel, di, dipole, neutralsheet, prc, internal, external
        )
    with open('data/parameters.json', 'w') as f:
        json.dump(new, f)

# =============================================================================
# saving the data and plotting it via a mollweide-projection
# =============================================================================
save_and_plot()
