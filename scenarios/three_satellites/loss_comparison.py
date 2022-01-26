import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
import matplotlib.pyplot as plt
from consts import AVERAGE_EARTH_RADIUS as R_E
from libs.aux_functions import distance
import scenarios.three_satellites.common_functions as cf
from scenarios.three_satellites.common_params import base_params
import rsmf


def db(x):
    return 10 * np.log10(x)


formatter = rsmf.setup(r"\documentclass[twocolumn]{revtex4-2}")

colors_rgb = [(86, 180, 233), (230, 159, 0), (204, 121, 167), (240, 228, 66),
              (0, 158, 115), (213, 94, 0), (0, 114, 178), (0, 0, 0)]
colors = [tuple((i / 255.0 for i in seq)) for seq in colors_rgb]


divergence_half_angle = base_params["DIVERGENCE_THETA"]
sender_aperture_radius = base_params["SENDER_APERTURE_RADIUS"]
receiver_aperture_radius = base_params["RECEIVER_APERTURE_RADIUS"]
eta_atm = cf.eta_atm
eta_dif = cf.eta_dif
elevation_curved = cf.elevation_curved
sat_dist_curved = cf.sat_dist_curved


def loss_one_satellite(total_ground_distance, orbital_height):
    eta = eta_atm(elevation_curved(total_ground_distance / 2, orbital_height)) \
        * eta_dif(sat_dist_curved(total_ground_distance / 2, orbital_height), divergence_half_angle, sender_aperture_radius, receiver_aperture_radius)
    try:
        return 1 / eta
    except ZeroDivisionError:
        return np.nan


def loss_three_satellites(total_ground_distance, orbital_height, first_satellite_multiplier):
    eta_ground = eta_atm(elevation_curved(total_ground_distance * first_satellite_multiplier, orbital_height)) \
               * eta_dif(sat_dist_curved(total_ground_distance * first_satellite_multiplier, orbital_height), divergence_half_angle, sender_aperture_radius, receiver_aperture_radius)

    def position_from_angle(radius, angle):
        return radius * np.array([np.sin(angle), np.cos(angle)])

    sat_A_pos = position_from_angle(radius=R_E + orbital_height, angle=total_ground_distance * first_satellite_multiplier / R_E)
    sat_central_pos = position_from_angle(radius=R_E + orbital_height, angle=total_ground_distance * 0.5 / R_E)
    eta_space = eta_dif(distance(sat_A_pos, sat_central_pos), divergence_half_angle, sender_aperture_radius, receiver_aperture_radius)
    try:
        return 1 / (eta_ground * eta_space)
    except ZeroDivisionError:
        return np.nan


fig = formatter.figure(aspect_ratio=0.8, width_ratio=1.0, wide=False)
ground_distances = np.linspace(100e3, 8800e3, num=500)
orbital_height = base_params["ORBITAL_HEIGHT"]
y1 = [db(loss_one_satellite(l, orbital_height)) for l in ground_distances]
first_sat_multipliers = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
ys = {}
for first_sat_multiplier in first_sat_multipliers:
    ys[first_sat_multiplier] = [db(loss_three_satellites(total_ground_distance=l, orbital_height=orbital_height, first_satellite_multiplier=first_sat_multiplier)) for l in ground_distances]
# plt.plot(ground_distances / 1000, y1, label="one_sat")
for first_sat_multiplier, color in zip(first_sat_multipliers, colors):
    plt.plot(ground_distances / 1000, ys[first_sat_multiplier], label=f"$S_A$ @ {int(first_sat_multiplier*100):d}%", c=color)
plt.grid()
plt.legend(loc="lower right", prop={"size": 6})
plt.xlabel("Ground distance $d$ [km]")
plt.ylabel("Channel loss [dB]")
plt.ylim(0, 60)
plt.xlim(0, 8800)
plt.tight_layout()
plt.savefig(os.path.join("results", "three_satellites", "loss_comparison.pdf"))
# plt.show()

# now look at pointing errors
distances = np.linspace(0, 4400e3, num=500)
pointing_error_sigmas = [0, 1e-6, 2e-6, 3e-6]
ys = [[eta_dif(distance=distance, divergence_half_angle=divergence_half_angle,
               sender_aperture_radius=sender_aperture_radius,
               receiver_aperture_radius=receiver_aperture_radius,
               pointing_error_sigma=pointing_error_sigma) for distance in distances]
      for pointing_error_sigma in pointing_error_sigmas]


fig = formatter.figure(aspect_ratio=1.0)
for pointing_error_sigma, y in zip(pointing_error_sigmas, ys):
    loss = db(1 / np.array(y))
    plt.plot(distances / 1000, loss, label=f"pointing error={int(pointing_error_sigma*1e6)}µrad")
plt.grid()
plt.legend(prop={"size": 6})
plt.xlabel("Distance [km]")
plt.ylabel("Loss [dB]")
plt.xlim(0, 4400)
plt.tight_layout()
plt.savefig(os.path.join("results", "three_satellites", "pointing_error_loss.pdf"))
# plt.show()


fig = formatter.figure(aspect_ratio=1.0)
base_loss = db(1 / np.array(ys[0]))
plt.plot([], [])  # advance color cycle
for pointing_error_sigma, y in zip(pointing_error_sigmas[1:], ys[1:]):
    loss = db(1 / np.array(y)) - base_loss
    plt.plot(distances / 1000, loss, label=f"pointing error={int(pointing_error_sigma*1e6)}µrad")
plt.grid()
plt.legend(prop={"size": 6})
plt.xlabel("Distance [km]")
plt.ylabel("Excess loss [dB]")
plt.xlim(0, 4400)
plt.tight_layout()
plt.savefig(os.path.join("results", "three_satellites", "pointing_error_excess_loss.pdf"))
# plt.show()
