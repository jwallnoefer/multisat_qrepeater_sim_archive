import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
import matplotlib.pyplot as plt
from consts import AVERAGE_EARTH_RADIUS as R_E
from consts import EARTH_MASS as M
from consts import GRAVITATIONAL_CONSTANT as G
from libs.aux_functions import distance, assert_dir
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


def position_from_angle(radius, angle):
    return radius * np.array([np.sin(angle), np.cos(angle)])


def loss_one_satellite(total_ground_distance, orbital_height):
    eta = eta_atm(elevation_curved(total_ground_distance / 2, orbital_height)) \
        * eta_dif(sat_dist_curved(total_ground_distance / 2, orbital_height), divergence_half_angle, sender_aperture_radius, receiver_aperture_radius, pointing_error_sigma=base_params["POINTING_ERROR_SIGMA"])
    try:
        return 1 / eta
    except ZeroDivisionError:
        return np.nan


def loss_three_satellites(total_ground_distance, orbital_height, first_satellite_multiplier):
    eta_ground = eta_atm(elevation_curved(total_ground_distance * first_satellite_multiplier, orbital_height)) \
               * eta_dif(sat_dist_curved(total_ground_distance * first_satellite_multiplier, orbital_height), divergence_half_angle, sender_aperture_radius, receiver_aperture_radius, pointing_error_sigma=base_params["POINTING_ERROR_SIGMA"])

    sat_A_pos = position_from_angle(radius=R_E + orbital_height, angle=total_ground_distance * first_satellite_multiplier / R_E)
    sat_central_pos = position_from_angle(radius=R_E + orbital_height, angle=total_ground_distance * 0.5 / R_E)
    eta_space = eta_dif(distance(sat_A_pos, sat_central_pos), divergence_half_angle, sender_aperture_radius, receiver_aperture_radius, pointing_error_sigma=base_params["POINTING_ERROR_SIGMA"])
    try:
        return 1 / (eta_ground * eta_space)
    except ZeroDivisionError:
        return np.nan


def loss_one_satellite_pos(total_ground_distance, orbital_height, satellite_position):
    station_a_pos = position_from_angle(radius=R_E, angle=0)
    station_b_pos = position_from_angle(radius=R_E, angle=total_ground_distance / R_E)
    station_sat_pos = position_from_angle(radius=R_E + orbital_height,
                                          angle=satellite_position * total_ground_distance / R_E)
    elevation_left = elevation_curved(ground_dist=satellite_position * total_ground_distance,
                                      h=orbital_height)
    eta_left = (eta_dif(distance=distance(station_a_pos, station_sat_pos),
                        divergence_half_angle=divergence_half_angle,
                        sender_aperture_radius=sender_aperture_radius,
                        receiver_aperture_radius=receiver_aperture_radius,
                        pointing_error_sigma=base_params["POINTING_ERROR_SIGMA"])
                * eta_atm(elevation_left)
                )
    return eta_left


def loss_three_satellites_pos(total_ground_distance, orbital_height, configuration, offset):
    satellite_multipliers = np.array(configuration) + offset
    station_a_pos = position_from_angle(radius=R_E, angle=0)
    sat_A_pos, sat_C_pos, sat_B_pos = [position_from_angle(radius=R_E + orbital_height, angle=multiplier * total_ground_distance / R_E) for multiplier in satellite_multipliers]
    station_b_pos = position_from_angle(radius=R_E, angle=total_ground_distance / R_E)
    elevation_left = elevation_curved(ground_dist=np.abs(satellite_multipliers[0] * total_ground_distance),
                                      h=orbital_height)
    eta_ground = (eta_atm(elevation_left)
                  * eta_dif(distance(station_a_pos, sat_A_pos),
                            divergence_half_angle=divergence_half_angle,
                            sender_aperture_radius=sender_aperture_radius,
                            receiver_aperture_radius=receiver_aperture_radius,
                            pointing_error_sigma=base_params["POINTING_ERROR_SIGMA"]
                            )
                  )
    eta_space = eta_dif(distance(sat_A_pos, sat_C_pos),
                        divergence_half_angle=divergence_half_angle,
                        sender_aperture_radius=sender_aperture_radius,
                        receiver_aperture_radius=receiver_aperture_radius,
                        pointing_error_sigma=base_params["POINTING_ERROR_SIGMA"]
                        )
    try:
        return 1 / (eta_ground * eta_space)
    except ZeroDivisionError:
        return np.nan


result_path = os.path.join("results", "three_satellites")
partial_result_path = os.path.join(result_path, "loss_comparison")
assert_dir(partial_result_path)


case_path = os.path.join(partial_result_path, "loss_comparison")
assert_dir(case_path)
fig = formatter.figure(aspect_ratio=0.8, width_ratio=1.0, wide=False)
ground_distances = np.linspace(100e3, 8800e3, num=500)
orbital_height = base_params["ORBITAL_HEIGHT"]
try:
    y1 = np.load(os.path.join(case_path, "loss_one_satellite.npy"))
except FileNotFoundError:
    y1 = [db(loss_one_satellite(l, orbital_height)) for l in ground_distances]
    np.save(os.path.join(case_path, "loss_one_satellite.npy"), y1)
first_sat_multipliers = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
ys = {}
for first_sat_multiplier in first_sat_multipliers:
    try:
        ys[first_sat_multiplier] = np.load(os.path.join(case_path, f"first_sat_{int(10*first_sat_multiplier)}.npy"))
    except FileNotFoundError:
        ys[first_sat_multiplier] = [db(loss_three_satellites(total_ground_distance=l, orbital_height=orbital_height, first_satellite_multiplier=first_sat_multiplier)) for l in ground_distances]
        np.save(os.path.join(case_path, f"first_sat_{int(10*first_sat_multiplier)}.npy"), ys[first_sat_multiplier])
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
save_path = os.path.join(result_path, "loss_comparison.pdf")
plt.savefig(save_path)
print(f"Saved figure {save_path}")
# plt.show()

# now look at pointing errors
case_path = os.path.join(partial_result_path, "pointing_errors")
assert_dir(case_path)
distances = np.linspace(0, 4400e3, num=500)
pointing_error_sigmas = [0, 1e-6, 2e-6, 3e-6]
ys = []
for pointing_error_sigma in pointing_error_sigmas:
    try:
        res = np.load(os.path.join(case_path, f"sigma_{int(1e6 * pointing_error_sigma)}.npy"))
    except FileNotFoundError:
        res = [eta_dif(distance=distance, divergence_half_angle=divergence_half_angle,
                       sender_aperture_radius=sender_aperture_radius,
                       receiver_aperture_radius=receiver_aperture_radius,
                       pointing_error_sigma=pointing_error_sigma) for distance in distances]
        np.save(os.path.join(case_path, f"sigma_{int(1e6 * pointing_error_sigma)}.npy"), res)
    ys += [res]


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
save_path = os.path.join(result_path, "pointing_error_loss.pdf")
plt.savefig(save_path)
print(f"Saved figure {save_path}")
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
save_path = os.path.join(result_path, "pointing_error_excess_loss.pdf")
plt.savefig(save_path)
print(f"Saved figure {save_path}")
# plt.show()


# now plots for loss along path
case_path = os.path.join(partial_result_path, "satellite_path")
assert_dir(case_path)
fig = formatter.figure(aspect_ratio=0.8, width_ratio=1.0, wide=False)
configurations = [np.array([0, 0.5, 1]), np.array([0.1, 0.5, 0.9]),
                  np.array([0.2, 0.5, 0.8]), np.array([-0.1, 0.5, 1.1])]
labels = [0, 1, 2, -1]
offsets = np.linspace(-0.6, 0.6, num=501)
ground_distance = 4400e3
orbital_height = base_params["ORBITAL_HEIGHT"]
angles = offsets * ground_distance / R_E
orbital_period = 2 * np.pi * np.sqrt((R_E + orbital_height)**3 / (M * G))
print(orbital_period)
times = angles / (2 * np.pi) * orbital_period

for configuration, label, color in zip(configurations, labels, colors):
    try:
        loss = np.load(os.path.join(case_path, f"configuration_{label}.npy"))
    except FileNotFoundError:
        loss = [loss_three_satellites_pos(total_ground_distance=ground_distance,
                                          orbital_height=orbital_height,
                                          configuration=configuration,
                                          offset=offset) for offset in offsets]
        np.save(os.path.join(case_path, f"configuration_{label}.npy"), loss)
    if label == -1:
        color = "gray"
    else:
        color = color
    plt.plot(times, db(loss), ls="dashed", color=color)
    # total loss for symmetric cases
    total_loss = np.array(loss) * np.array(loss)[::-1]
    plt.plot(times, db(total_loss), ls="solid", color=color, label=f"configuration={label}")
plt.grid()
# plt.legend(prop={"size": 6})
plt.xlabel("Time [s]")
plt.ylabel("Loss [dB]")
plt.ylim(20, 140)
plt.tight_layout()
save_path = os.path.join(result_path, "satellite_path_loss.pdf")
plt.savefig(save_path)
print(f"Saved figure {save_path}")
# plt.show()
