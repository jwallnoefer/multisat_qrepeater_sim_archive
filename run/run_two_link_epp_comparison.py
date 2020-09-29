import os, sys; sys.path.insert(0, os.path.abspath("."))
from scenarios.NSP_QR_cell import run_nsp
from scenarios.two_link_epp import run_epp
from libs.aux_functions import assert_dir, binary_entropy, calculate_keyrate_time, calculate_keyrate_channel_use
import numpy as np
import matplotlib.pyplot as plt

C = 2 * 10**8  # speed of light in optical fiber

result_path = os.path.join("results", "two_link_epp")


params_available_Rb = {"P_LINK": 50 * 10**-2,
                       "f_clock": 5 * 10**6,
                       "T_DP": 100 * 10**-3}


length_list = np.arange(25000, 425000, 25000)

# run without epp first:
print("Run without epp")
for l in length_list:
    print(l)
    trial_time_manual = l / C
    p = run_nsp(length=l, max_iter=10000, params=params, cutoff_time=None, mode="sim")
    key_per_time = calculate_keyrate_time(p.correlations_z_list, p.correlations_x_list, 1, p.world.event_queue.current_time + 2 * l / C)
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
for l in length_list:
    print(l)
    trial_time_manual = l / C
    p = run_epp(length=l, max_iter=10000, params=params, cutoff_time=None, mode="sim")
    key_per_time = calculate_keyrate_time(p.correlations_z_list, p.correlations_x_list, 1, p.world.event_queue.current_time + 2 * l / C)
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
