import os, sys; sys.path.insert(0, os.path.abspath("."))
from scenarios.NSP_QR_cell import run
from libs.aux_functions import assert_dir, binary_entropy, calculate_keyrate_time, calculate_keyrate_channel_use
import numpy as np
import matplotlib.pyplot as plt

ETA_P = 0.66  # preparation efficiency
T_P = 2 * 10**-6  # preparation time
ETA_C = 0.04 * 0.3  # phton-fiber coupling efficiency * wavelength conversion
T_2 = 1  # dephasing time
E_M_A = 0.01  # misalignment error
P_D = 10**-8  # dark count probability per detector
ETA_D = 0.3  # detector efficiency
P_BSM = 1  # BSM success probability  ## WARNING: Currently not implemented
LAMBDA_BSM = 0.97  # BSM ideality parameter
F = 1.16  # error correction inefficiency
ETA_TOT = ETA_P * ETA_C * ETA_D

C = 2 * 10**8 # speed of light in optical fiber
result_path = os.path.join("results", "luetkenhaus_as_nsp")

luetkenhaus_params = {"P_LINK": ETA_TOT,
                      "T_P": T_P,
                      "T_DP": T_2,
                      "E_MA": E_M_A,
                      "P_D": P_D,
                      "LAMBDA_BSM": LAMBDA_BSM}


if __name__ == "__main__":
    length_list = np.concatenate([np.arange(1000, 61000, 2500), np.arange(61000, 69000, 1000)])
    mode="seq"
    key_per_time_list = []
    key_per_resource_list = []
    for l in length_list:
        print(l)
        p = run(length=l, max_iter=10000, params=luetkenhaus_params, mode=mode)
        key_per_time = calculate_keyrate_time(p.correlations_z_list, p.correlations_x_list, F, p.world.event_queue.current_time + 2 * l / C)
        key_per_resource = calculate_keyrate_channel_use(p.correlations_z_list, p.correlations_x_list, F, p.resource_cost_max_list)

        key_per_time_list += [key_per_time]
        key_per_resource_list += [key_per_resource]

    assert_dir(result_path)
    np.savetxt(os.path.join(result_path, "length_list_%s.txt" % mode), length_list)
    np.savetxt(os.path.join(result_path, "key_per_time_list_%s.txt" % mode), key_per_time_list)
    np.savetxt(os.path.join(result_path, "key_per_resource_list_%s.txt" % mode), key_per_resource_list)

    plt.plot(length_list, key_per_time_list)
    plt.yscale("log")
    plt.xlabel("total length")
    plt.ylabel("key_rate_per_time")
    plt.show()

    plt.plot(length_list, key_per_resource_list)
    plt.yscale("log")
    plt.xlabel("total length")
    plt.ylabel("key rate per channel use")
    plt.show()
