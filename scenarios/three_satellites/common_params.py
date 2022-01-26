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


P_LINK = ETA_MEM * ETA_DET
base_params = {"P_LINK": P_LINK,
               "ETA_MEM": ETA_MEM,
               "ETA_DET": ETA_DET,
               "T_P": T_P,
               "T_DP": T_DP,
               "F_CLOCK": F_CLOCK,
               "E_MA": E_M_A,
               "P_D": P_D,
               "LAMBDA_BSM": LAMBDA_BSM,
               "ORBITAL_HEIGHT": ORBITAL_HEIGHT,
               "SENDER_APERTURE_RADIUS": SENDER_APERTURE_RADIUS,
               "RECEIVER_APERTURE_RADIUS": RECEIVER_APERTURE_RADIUS,
               "DIVERGENCE_THETA": DIVERGENCE_THETA,
               "POINTING_ERROR_SIGMA": POINTING_ERROR_SIGMA}
