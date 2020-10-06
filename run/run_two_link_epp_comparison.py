import os, sys; sys.path.insert(0, os.path.abspath("."))
from scenarios.NSP_QR_cell import run as run_nsp
from scenarios.two_link_epp import run as run_epp
from libs.aux_functions import assert_dir, calculate_keyrate_time, calculate_keyrate_channel_use
import numpy as np

C = 2 * 10**8  # speed of light in optical fiber

result_path = os.path.join("results", "two_link_epp")

params = {"P_LINK": 30 * 10**-2,
          "f_clock": 5 * 10**6,
          "T_DP": 10 * 10**-3}

# ETA_P = 0.66  # preparation efficiency
# T_P = 2 * 10**-6  # preparation time
# ETA_C = 0.04 * 0.3  # phton-fiber coupling efficiency * wavelength conversion
# T_2 = 1  # dephasing time
# E_M_A = 0.01  # misalignment error
# P_D = 10**-8  # dark count probability per detector
# ETA_D = 0.3  # detector efficiency
# P_BSM = 1  # BSM success probability  ## WARNING: Currently not implemented
# LAMBDA_BSM = 0.97  # BSM ideality parameter
# # F = 1.16  # error correction inefficiency
# ETA_TOT = ETA_P * ETA_C * ETA_D
# luetkenhaus_params = {"P_LINK": ETA_TOT,
#                       "T_P": T_P,
#                       "T_DP": T_2,
#                       "E_MA": E_M_A,
#                       "P_D": P_D,
#                       "LAMBDA_BSM": LAMBDA_BSM}

length_list = np.concatenate([np.array([25000, 37500]), np.arange(50000, 125000, 10000), np.arange(125000, 425000, 25000)])
# length_list = np.concatenate([np.arange(1000, 61000, 2500), np.arange(61000, 69000, 1000), np.arange(69000, 84000, 2500)])
# params = luetkenhaus_params

# run without epp first:
print("Run without epp")
key_per_time_list = []
key_per_resource_list = []
for length in length_list:
    print(length)
    trial_time_manual = length / C
    p = run_nsp(length=length, max_iter=10000, params=params, cutoff_time=None, mode="sim")
    key_per_time = calculate_keyrate_time(p.correlations_z_list, p.correlations_x_list, 1, p.world.event_queue.current_time + 2 * length / C)
    key_per_resource = calculate_keyrate_channel_use(p.correlations_z_list, p.correlations_x_list, 1, p.resource_cost_max_list)
    key_per_time_list += [key_per_time]
    key_per_resource_list += [key_per_resource]
    if (10 * np.log10(key_per_resource / 2)) < (-60):
        break
path = os.path.join(result_path, "without_epp")
assert_dir(path)
np.savetxt(os.path.join(path, "length_list.txt"), length_list[:len(key_per_resource_list)])
np.savetxt(os.path.join(path, "key_per_time_list.txt"), key_per_time_list)
np.savetxt(os.path.join(path, "key_per_resource_list.txt"), key_per_resource_list)

print("Run with epp")
key_per_time_list = []
key_per_resource_list = []
for length in length_list:
    print(length)
    trial_time_manual = length / C
    p = run_epp(length=length, max_iter=10000, params=params, cutoff_time=None, mode="sim")
    key_per_time = calculate_keyrate_time(p.correlations_z_list, p.correlations_x_list, 1, p.world.event_queue.current_time + 2 * length / C)
    key_per_resource = calculate_keyrate_channel_use(p.correlations_z_list, p.correlations_x_list, 1, p.resource_cost_max_list)
    key_per_time_list += [key_per_time]
    key_per_resource_list += [key_per_resource]
    if (10 * np.log10(key_per_resource / 2)) < (-60):
        break
path = os.path.join(result_path, "with_epp")
assert_dir(path)
np.savetxt(os.path.join(path, "length_list.txt"), length_list[:len(key_per_resource_list)])
np.savetxt(os.path.join(path, "key_per_time_list.txt"), key_per_time_list)
np.savetxt(os.path.join(path, "key_per_resource_list.txt"), key_per_resource_list)