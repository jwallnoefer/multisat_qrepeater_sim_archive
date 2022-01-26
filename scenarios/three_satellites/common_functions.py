import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
from scipy.integrate import quad, nquad
from consts import ETA_ATM_PI_HALF_780_NM
from consts import AVERAGE_EARTH_RADIUS as R_E
from libs.aux_functions import y_noise_channel, z_noise_channel, w_noise_channel


def construct_dephasing_noise_channel(dephasing_time):
    def lambda_dp(t):
        return (1 - np.exp(-t / dephasing_time)) / 2

    def dephasing_noise_channel(rho, t):
        return z_noise_channel(rho=rho, epsilon=lambda_dp(t))

    return dephasing_noise_channel


def construct_y_noise_channel(epsilon):
    return lambda rho: y_noise_channel(rho=rho, epsilon=epsilon)


def construct_w_noise_channel(epsilon):
    return lambda rho: w_noise_channel(rho=rho, alpha=(1 - epsilon))


def alpha_of_eta(eta, p_d):
    return eta * (1 - p_d) / (1 - (1 - eta) * (1 - p_d)**2)


def eta_dif(distance, divergence_half_angle, sender_aperture_radius, receiver_aperture_radius, wavelength=780e-9, pointing_error_sigma=0):
    w_0 = wavelength / (divergence_half_angle * np.pi)
    w_z = w_0 * np.sqrt(1 + (divergence_half_angle / w_0 * distance)**2)
    P_0 = 1 / 2 * np.pi * w_0**2

    def I_1(r):
        return (w_0 / w_z)**2 * np.exp(-2 * r**2 / w_z**2)

    if pointing_error_sigma != 0 and distance != 0:
        assert pointing_error_sigma > 0
        assert distance > 0
        convolution_sigma = distance * pointing_error_sigma

        def g(r):
            return 1 / (2 * np.pi * convolution_sigma**2) * np.exp(-r**2 / (2 * convolution_sigma**2))

        def I_2(r):
            # 2d convolution with pointing error distribution
            def integrand_convolution(r_prime, theta_prime):
                r_dif = np.sqrt(r**2 + r_prime**2 - 2 * r * r_prime * np.cos(theta_prime))  # law of cosines
                return r_prime * I_1(r_prime) * g(r_dif)
            return nquad(integrand_convolution, [(0, np.inf), (0, 2 * np.pi)])[0]

        intensity = I_2
    else:
        intensity = I_1

    def integrand_receiver(r):
        return r * intensity(r)

    P = 2 * np.pi * quad(integrand_receiver, 0, receiver_aperture_radius)[0]
    return P / P_0


def _eta_dif_cone(distance, divergence_half_angle, sender_aperture_radius, receiver_aperture_radius):
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
