import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
import matplotlib.pyplot as plt
from consts import AVERAGE_EARTH_RADIUS as R_E
from consts import ETA_ATM_PI_HALF_780_NM
from libs.aux_functions import distance


colors_rgb = [(86, 180, 233), (230, 159, 0), (204, 121, 167), (240, 228, 66),
              (0, 158, 115), (213, 94, 0), (0, 114, 178), (0, 0, 0)]
colors = [tuple((i / 255.0 for i in seq)) for seq in colors_rgb]


def eta_dif(distance, divergence_half_angle, sender_aperture_radius, receiver_aperture_radius):
    # calculated by simple geometry, because gaussian effects do not matter much
    x = sender_aperture_radius + distance * np.tan(divergence_half_angle)
    arriving_fraction = receiver_aperture_radius**2 / x**2
    if arriving_fraction > 1:
        arriving_fraction = 1
    return arriving_fraction


def eta_atm(elevation):
    # eta of pi/2 to the power of csc(theta), equation (A4) in https://arxiv.org/abs/2006.10636
    # eta of pi/2 (i.e. straight up) is ~0.8 for 780nm wavelength.
    if elevation < 0:
        return 0
    return ETA_ATM_PI_HALF_780_NM**(1 / np.sin(elevation))


def sat_dist_curved(ground_dist, h):
    # ground dist refers to distance between station and the "shadow" of the satellite
    alpha = ground_dist / R_E
    L = np.sqrt(R_E**2 + (R_E + h)**2 - 2 * R_E * (R_E + h) * np.cos(alpha))
    return L


def elevation_curved(ground_dist, h):
    # ground dist refers to distance between station and the "shadow" of the satellite
    alpha = ground_dist / R_E
    L = np.sqrt(R_E**2 + (R_E + h)**2 - 2 * R_E * (R_E + h) * np.cos(alpha))
    beta = np.arcsin(R_E / L * np.sin(alpha))
    gamma = np.pi - alpha - beta
    return gamma - np.pi / 2


def db(x):
    return 10 * np.log10(x)


divergence_half_angle = 2e-6
sender_aperture_radius = 0.15
receiver_aperture_radius = 0.50


def loss_one_satellite(total_ground_distance, orbital_height):
    eta = eta_atm(elevation_curved(total_ground_distance / 2, orbital_height)) \
        * eta_dif(sat_dist_curved(total_ground_distance / 2, orbital_height), divergence_half_angle, sender_aperture_radius, receiver_aperture_radius)
    return 1 / eta


def loss_three_satellites(total_ground_distance, orbital_height, first_satellite_multiplier):
    eta_ground = eta_atm(elevation_curved(total_ground_distance * first_satellite_multiplier, orbital_height)) \
               * eta_dif(sat_dist_curved(total_ground_distance * first_satellite_multiplier, orbital_height), divergence_half_angle, sender_aperture_radius, receiver_aperture_radius)

    def position_from_angle(radius, angle):
        return radius * np.array([np.sin(angle), np.cos(angle)])

    sat_A_pos = position_from_angle(radius=R_E + orbital_height, angle=total_ground_distance * first_satellite_multiplier / R_E)
    sat_central_pos = position_from_angle(radius=R_E + orbital_height, angle=total_ground_distance * 0.5 / R_E)
    eta_space = eta_dif(distance(sat_A_pos, sat_central_pos), divergence_half_angle, sender_aperture_radius, receiver_aperture_radius)
    return 1 / (eta_ground * eta_space)


ground_distances = np.linspace(0, 8800e3, num=5000)
orbital_height = 400e3
y1 = [db(loss_one_satellite(l, orbital_height)) for l in ground_distances]
first_sat_multipliers = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
ys = {}
for first_sat_multiplier in first_sat_multipliers:
    ys[first_sat_multiplier] = [db(loss_three_satellites(total_ground_distance=l, orbital_height=orbital_height, first_satellite_multiplier=first_sat_multiplier)) for l in ground_distances]
# plt.plot(ground_distances / 1000, y1, label="one_sat")
for first_sat_multiplier, color in zip(first_sat_multipliers, colors):
    plt.plot(ground_distances / 1000, ys[first_sat_multiplier], label=f"multiplier={first_sat_multiplier}", c=color)
plt.grid()
# plt.legend()
plt.xlabel("ground_distance [km]")
plt.ylabel("channel loss [dB]")
plt.ylim(0, 60)
plt.savefig(os.path.join("results", "three_satellites", "loss_comparison.pdf"))
plt.show()
