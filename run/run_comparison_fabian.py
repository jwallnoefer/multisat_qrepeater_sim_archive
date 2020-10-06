import os, sys; sys.path.insert(0, os.path.abspath("."))
from scenarios.NSP_QR_cell import run
from libs.aux_functions import assert_dir, binary_entropy, calculate_keyrate_time, calculate_keyrate_channel_use
import numpy as np
import matplotlib.pyplot as plt

C = 2 * 10**8 # speed of light in optical fiber

result_path = os.path.join("results", "fabian_compare")

params = {"P_LINK": 0.8,
          "T_P": 0,
          "T_DP": 1,
          "E_MA": 0.01,
          "P_D": 0,
          "LAMBDA_BSM": 0.98}

cutoff_time = 0.08

length_list = np.arange(500, 20500, 500)
mode="sim"
key_per_time_list = []
key_per_resource_list = []
for l in length_list:
    print(l)
    p = run(length=l, max_iter=10000, params=params, cutoff_time=cutoff_time, mode=mode)
    key_per_time = calculate_keyrate_time(p.correlations_z_list, p.correlations_x_list, 1, p.world.event_queue.current_time + 2 * l / C)
    key_per_resource = calculate_keyrate_channel_use(p.correlations_z_list, p.correlations_x_list, 1, p.resource_cost_max_list)

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
