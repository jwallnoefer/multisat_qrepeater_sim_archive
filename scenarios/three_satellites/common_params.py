import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
from consts import SPEED_OF_LIGHT_IN_VACCUM as C

F_CLOCK = 20e6  # 20 MHz is very high
T_P = 0  # preparation time
E_M_A = 0  # misalignment error
P_D = 10**-6  # dark count probability per detector
P_BSM = 1  # BSM success probability  ## WARNING: Currently not implemented
LAMBDA_BSM = 1  # BSM ideality parameter
F = 1  # error correction inefficiency

T_DP = 100e-3  # dephasing time
ETA_MEM = 0.8  # memory efficiency
ETA_DET = 0.7  # detector efficiency

ORBITAL_HEIGHT = 400e3
SENDER_APERTURE_RADIUS = 0.15
RECEIVER_APERTURE_RADIUS = 0.50
DIVERGENCE_THETA = 3e-6  # beam divergence_half_angle

POINTING_ERROR_SIGMA = 1e-6


# background light calculation translated from Mustafa's Matlab file
fov = 3.14 * 1e-10  # field-of-view in steradian
h = 6.62e-34  # Planck constant
wavelength = 780e-9
nu = C / wavelength  # frequency of light
t = 1e-6  # time window of detection
RELATIVE_BRIGHTNESS = 1e-5
H_B = 150e6  # W / (m^2 Sr m)
A_REC = np.pi * RECEIVER_APERTURE_RADIUS**2  # telescope area
B_FILT = 0.02e-9  # filter bandwidth (is a length) corresponding to 10 GHz


P = RELATIVE_BRIGHTNESS * H_B * fov * A_REC * B_FILT
BACKGROUND_NOISE = t * P / (h * nu)  # noise per detection window


P_LINK = ETA_MEM * ETA_DET
base_params = {"P_LINK": P_LINK,
               "ETA_MEM": ETA_MEM,
               "ETA_DET": ETA_DET,
               "T_P": T_P,
               "T_DP": T_DP,
               "F_CLOCK": F_CLOCK,
               "E_MA": E_M_A,
               "P_D": P_D + BACKGROUND_NOISE,
               "LAMBDA_BSM": LAMBDA_BSM,
               "ORBITAL_HEIGHT": ORBITAL_HEIGHT,
               "SENDER_APERTURE_RADIUS": SENDER_APERTURE_RADIUS,
               "RECEIVER_APERTURE_RADIUS": RECEIVER_APERTURE_RADIUS,
               "DIVERGENCE_THETA": DIVERGENCE_THETA,
               "POINTING_ERROR_SIGMA": POINTING_ERROR_SIGMA}
