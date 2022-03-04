import os, sys; sys.path.insert(0, os.path.abspath("."))
import libs.matrix as mat
from libs.aux_functions import binary_entropy, distance, assert_dir
from scenarios.three_satellites.common_functions import elevation_curved, eta_dif, eta_atm
from scenarios.three_satellites.common_params import base_params
from scipy.integrate import quad
from consts import AVERAGE_EARTH_RADIUS as R_E
from consts import EARTH_MASS as M
from consts import GRAVITATIONAL_CONSTANT as G
import numpy as np
import pandas as pd


def position_from_angle(radius, angle):
    return radius * np.array([np.sin(angle), np.cos(angle)])


def e91_eta_pos(length, divergence_half_angle=base_params["DIVERGENCE_THETA"], orbital_height=base_params["ORBITAL_HEIGHT"], satellite_position=0.5):
    eta_det = 0.7
    station_a_pos = position_from_angle(radius=R_E, angle=0)
    station_b_pos = position_from_angle(radius=R_E, angle=length / R_E)
    station_sat_pos = position_from_angle(radius=R_E + orbital_height,
                                          angle=satellite_position * length / R_E)
    elevation_left = elevation_curved(ground_dist=satellite_position * length,
                                      h=orbital_height)
    elevation_right = elevation_curved(ground_dist=(1 - satellite_position) * length,
                                       h=orbital_height)
    eta_left = (eta_det
                * eta_dif(distance=distance(station_a_pos, station_sat_pos),
                          divergence_half_angle=divergence_half_angle,
                          sender_aperture_radius=0.15,
                          receiver_aperture_radius=0.5,
                          pointing_error_sigma=base_params["POINTING_ERROR_SIGMA"])
                * eta_atm(elevation_left)
                )
    eta_right = (eta_det
                 * eta_dif(distance=distance(station_b_pos, station_sat_pos),
                           divergence_half_angle=divergence_half_angle,
                           sender_aperture_radius=0.15,
                           receiver_aperture_radius=0.5,
                           pointing_error_sigma=base_params["POINTING_ERROR_SIGMA"])
                 * eta_atm(elevation_right)
                 )
    return eta_left * eta_right


orbital_heights = [600e3, 1000e3, 1500e3, 2000e3]
divergence_half_angle = base_params["DIVERGENCE_THETA"]
length = 4400e3
R_S = 20e6  # repetition rate
res = []
for orbital_height in orbital_heights:
    orbital_period = 2 * np.pi * np.sqrt((R_E + orbital_height)**3 / (M * G))
    ground_dist_angle = length / R_E

    def integrand_rate(time):
        angle = time / orbital_period * 2 * np.pi
        satellite_position = 0.5 + angle / ground_dist_angle
        return R_S * e91_eta_pos(length=length,
                                 divergence_half_angle=divergence_half_angle,
                                 orbital_height=orbital_height,
                                 satellite_position=satellite_position)

    bits_per_pass = quad(integrand_rate, -orbital_period / 2, orbital_period / 2)[0]
    bits_per_time = bits_per_pass / orbital_period
    res += [[bits_per_pass, bits_per_time]]
    print(orbital_height, bits_per_pass, bits_per_time)

data = pd.DataFrame(res, index=np.array(orbital_heights), columns=["bits_per_pass", "bits_per_time"])
output_path = os.path.join("results", "one_satellite", "case_9_comparison")
assert_dir(output_path)
data.to_csv(os.path.join(output_path, "effective_rate.csv"))
