import os, sys; sys.path.insert(0, os.path.abspath("."))
from scenarios.multi_memory_variant import run
from libs.aux_functions import assert_dir, binary_entropy, calculate_keyrate_time, calculate_keyrate_channel_use
import numpy as np
import matplotlib.pyplot as plt
from time import time
from multiprocessing import Pool

C = 2 * 10**8  # speed of light in optical fiber
L_ATT = 22 * 10**3  # attenuation length
# result_path = os.path.join("results", "multimemory_variant_cutoff")

# # # values taken from Róbert Trényi, Norbert Lütkenhaus https://arxiv.org/abs/1910.10962
T_P = 2 * 10**-6  # preparation time
E_M_A = 0.01  # misalignment error
P_D = 1.8 * 10**-11  # dark count probability per detector
P_BSM = 1  # BSM success probability  ## WARNING: Currently not implemented
LAMBDA_BSM = 0.98  # BSM ideality parameter
F = 1.16  # error correction inefficiency

T_2 = 2  # dephasing time
ETA_P = 0.66  # preparation efficiency
ETA_C = 0.05 * 0.5  # phton-fiber coupling efficiency * wavelength conversion
ETA_D = 0.7  # detector efficiency

ETA_TOT = ETA_P * ETA_C * ETA_D  # = 0.0115
params = {"P_LINK": ETA_TOT,
          "T_P": T_P,
          "T_DP": T_2,
          "E_MA": E_M_A,
          "P_D": P_D,
          "LAMBDA_BSM": LAMBDA_BSM}


def do_the_thing(length, max_iter, params, cutoff_time, num_memories):
    p = run(length=length, max_iter=max_iter, params=params, cutoff_time=cutoff_time, num_memories=num_memories)
    key_per_time = calculate_keyrate_time(p.correlations_z_list, p.correlations_x_list, F, p.world.event_queue.current_time + 2 * length / C)
    key_per_resource = calculate_keyrate_channel_use(p.correlations_z_list, p.correlations_x_list, F, p.resource_cost_max_list)
    return key_per_time, key_per_resource

if __name__ == "__main__":
    # # length_list = np.arange(20000, 400000, 20000)
    # # length_list = np.arange(5000, 200000, 5000)
    # length_list = np.arange(10000, 400000, 10000)
    # max_iter = 10000
    # for num_memories in [1, 5, 10, 50, 100, 400]:
    #     key_per_time_list = []
    #     key_per_resource_list = []
    #     from time import time
    #     start_time = time()
    #     for l in length_list:
    #         print(l)
    #         # BEGIN cutoff estimation
    #         trial_time_manual = T_P + 2 * (l / 2) / C
    #         expected_time = trial_time_manual / (ETA_TOT * np.exp(-(l / 2) / L_ATT))  # expected time ONE memory would take to have a successful pair
    #         cutoff_time = 3 * expected_time
    #         # END cutoff estimation
    #         p = run(length=l, max_iter=max_iter, params=params, cutoff_time=cutoff_time, num_memories=num_memories)
    #         key_per_time = calculate_keyrate_time(p.correlations_z_list, p.correlations_x_list, F, p.world.event_queue.current_time + 2 * l / C)
    #         key_per_resource = calculate_keyrate_channel_use(p.correlations_z_list, p.correlations_x_list, F, p.resource_cost_max_list)
    #
    #         key_per_time_list += [key_per_time]
    #         key_per_resource_list += [key_per_resource]
    #         if key_per_resource < 0:
    #             break
    #         # print("l=%d took %s seconds" % (l, str(time() - start_time)))
    #
    #     output_path = os.path.join(result_path, "%d_memories" % num_memories)
    #     assert_dir(output_path)
    #
    #     np.savetxt(os.path.join(output_path, "length_list.txt"), length_list[:len(key_per_resource_list)])
    #     np.savetxt(os.path.join(output_path, "key_per_time_list.txt"), key_per_time_list)
    #     np.savetxt(os.path.join(output_path, "key_per_resource_list.txt"), key_per_resource_list)
    #
    #     print("num_memories=%d took %s seconds" % (num_memories, str(time() - start_time)))
    #
    #     # plt.plot(length_list[:len(key_per_resource_list)], key_per_time_list)
    #     # plt.yscale("log")
    #     # plt.xlabel("total length")
    #     # plt.ylabel("key_rate_per_time")
    #     # plt.show()
    #     #
    #     # plt.plot(length_list[:len(key_per_resource_list)], key_per_resource_list)
    #     # plt.yscale("log")
    #     # plt.xlabel("total length")
    #     # plt.ylabel("key rate per channel use")
    #     # plt.show()

    # # cluster variant
    # result_path = os.path.join("results", "multimemory_variant_cutoff")
    # num_processes = 32
    # length_list = np.arange(10000, 400000, 2500)
    # memories_list = [1, 5, 10, 50, 100, 400]
    # max_iter = 1e6
    # res = {}
    # start_time = time()
    # with Pool(num_processes) as pool:
    #     for num_memories in memories_list:
    #         # BEGIN cutoff estimation
    #         trial_time_manual = T_P + 2 * (length_list / 2) / C
    #         expected_time = trial_time_manual / (ETA_TOT * np.exp(-(length_list / 2) / L_ATT))  # expected time ONE memory would take to have a successful pair
    #         cutoff_time = 3 * expected_time
    #         # END cutoff estimation
    #         num_calls = len(length_list)
    #         aux_list = zip(length_list, [max_iter] * num_calls, [params] * num_calls, cutoff_time, [num_memories] * num_calls)
    #         res[num_memories] = pool.starmap_async(do_the_thing, aux_list)
    #     pool.close()
    #     # pool.join()
    #
    #     for num_memories in memories_list:
    #         key_per_time_list, key_per_resource_list = zip(*list(res[num_memories].get()))
    #         print("memories=%s finished after %.2f minutes." % (str(num_memories), (time() - start_time) / 60.0))
    #
    #         output_path = os.path.join(result_path, "%d_memories" % num_memories)
    #         assert_dir(output_path)
    #
    #         np.savetxt(os.path.join(output_path, "length_list.txt"), length_list[:len(key_per_resource_list)])
    #         np.savetxt(os.path.join(output_path, "key_per_time_list.txt"), key_per_time_list)
    #         np.savetxt(os.path.join(output_path, "key_per_resource_list.txt"), key_per_resource_list)
    #
    # print("The whole run took %s seconds." % str(time() - start_time))

    # fixed length, different memories
    result_path = os.path.join("results", "multimemory_variant_memories")
    num_processes = 32
    memories_list = np.floor(np.logspace(1, 400, num=50, endpoint=True), dtype=int)
    length_list = [50e3, 100e3, 150e3, 200e3]
    max_iter = 1e5
    res = {}
    start_time = time()
    with Pool(num_processes) as pool:
        for length in length_list:
            # BEGIN cutoff estimation
            trial_time_manual = T_P + 2 * (length / 2) / C
            expected_time = trial_time_manual / (ETA_TOT * np.exp(-(length / 2) / L_ATT))  # expected time ONE memory would take to have a successful pair
            cutoff_time = 3 * expected_time
            # END cutoff estimation
            num_calls = len(memories_list)
            aux_list = zip([length] * num_calls, [max_iter] * num_calls, [params] * num_calls, [cutoff_time] * num_calls, memories_list)
            res[length] = pool.starmap_async(do_the_thing, aux_list)
        pool.close()
        # pool.join()

        for length in length_list:
            key_per_time_list, key_per_resource_list = zip(*list(res[length].get()))
            print("length=%s finished after %.2f minutes." % (str(length), (time() - start_time) / 60.0))

            output_path = os.path.join(result_path, "%d_km" % (length / 1000))
            assert_dir(output_path)

            np.savetxt(os.path.join(output_path, "memories_list.txt"), memories_list)
            np.savetxt(os.path.join(output_path, "key_per_time_list.txt"), key_per_time_list)
            np.savetxt(os.path.join(output_path, "key_per_resource_list.txt"), key_per_resource_list)

    print("The whole run took %s seconds." % str(time() - start_time))
