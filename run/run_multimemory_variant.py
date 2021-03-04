import os, sys; sys.path.insert(0, os.path.abspath("."))
from scenarios.multi_memory_variant import run
from libs.aux_functions import assert_dir, standard_bipartite_evaluation
import numpy as np
import matplotlib.pyplot as plt
from time import time
from multiprocessing import Pool
import pandas as pd
from consts import C, L_ATT


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
    return p.data


if __name__ == "__main__":
    if int(sys.argv[1]) == 0:
        # cluster variant
        result_path = os.path.join("results", "multimemory_variant_cutoff")
        num_processes = 32
        length_list = np.arange(10000, 400000, 2500)
        memories_list = [1, 5, 10, 50, 100, 400]
        max_iter = 1e5
        res = {}
        start_time = time()
        with Pool(num_processes) as pool:
            for num_memories in memories_list:
                # BEGIN cutoff estimation
                trial_time_manual = T_P + 2 * (length_list / 2) / C
                expected_time = trial_time_manual / (ETA_TOT * np.exp(-(length_list / 2) / L_ATT))  # expected time ONE memory would take to have a successful pair
                cutoff_time = 3 * expected_time
                # END cutoff estimation
                num_calls = len(length_list)
                aux_list = zip(length_list, [max_iter] * num_calls, [params] * num_calls, cutoff_time, [num_memories] * num_calls)
                res[num_memories] = pool.starmap_async(do_the_thing, aux_list)
            pool.close()
            # pool.join()

            for num_memories in memories_list:
                data_series = pd.Series(data=res[num_memories].get(), index=length_list)
                print("memories=%s finished after %.2f minutes." % (str(num_memories), (time() - start_time) / 60.0))
                output_path = os.path.join(result_path, "%d_memories" % num_memories)
                assert_dir(output_path)
                data_series.to_pickle(os.path.join(output_path, "raw_data.bz2"))
                result_list = [standard_bipartite_evaluation(data_frame=df) for df in data_series]
                output_data = pd.DataFrame(data=result_list, index=length_list, columns=["fidelity", "fidelity_std", "key_per_time", "key_per_time_std", "key_per_resource", "key_per_resource_std"])
                output_data.to_csv(os.path.join(output_path, "result.csv"))

        print("The whole run took %s seconds." % str(time() - start_time))

    elif int(sys.argv[1]) == 1:
        # fixed length, different memories
        result_path = os.path.join("results", "multimemory_variant_memories")
        num_processes = 32
        memories_list = np.unique(np.logspace(np.log10(1), np.log10(400), num=50, endpoint=True, base=10, dtype=int))
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
                data_series = pd.Series(data=res[length].get(), index=memories_list)
                print("length=%s finished after %.2f minutes." % (str(length), (time() - start_time) / 60.0))
                output_path = os.path.join(result_path, "%d_km" % (length / 1000))
                assert_dir(output_path)
                data_series.to_pickle(os.path.join(output_path, "raw_data.bz2"))
                result_list = [standard_bipartite_evaluation(data_frame=df) for df in data_series]
                output_data = pd.DataFrame(data=result_list, index=memories_list, columns=["fidelity", "fidelity_std", "key_per_time", "key_per_time_std", "key_per_resource", "key_per_resource_std"])
                output_data.to_csv(os.path.join(output_path, "result.csv"))

        print("The whole run took %s seconds." % str(time() - start_time))

    elif int(sys.argv[1]) == 2:
        # fixed length, different cutoff times
        result_path = os.path.join("results", "multimemory_variant_by_cutoff")
        num_processes = 32
        memories_list = [1, 5, 10, 50, 100, 400]
        length = 150e3
        # BEGIN cutoff estimation
        trial_time_manual = T_P + 2 * (length / 2) / C
        expected_time = trial_time_manual / (ETA_TOT * np.exp(-(length / 2) / L_ATT))  # expected time ONE memory would take to have a successful pair
        cutoff_times = np.arange(0.25, 5.25, 0.25) * expected_time
        # END cutoff estimation
        max_iter = 1e5
        res = {}
        start_time = time()
        with Pool(num_processes) as pool:
            for num_memories in memories_list:
                num_calls = len(cutoff_times)
                aux_list = zip([length] * num_calls, [max_iter] * num_calls, [params] * num_calls, cutoff_times, [num_memories] * num_calls)
                res[num_memories] = pool.starmap_async(do_the_thing, aux_list)
            pool.close()
            # pool.join()

            for num_memories in memories_list:
                data_series = pd.Series(data=res[num_memories].get(), index=cutoff_times)
                print("memories=%s finished after %.2f minutes." % (str(num_memories), (time() - start_time) / 60.0))
                output_path = os.path.join(result_path, "%d_memories" % num_memories)
                assert_dir(output_path)
                data_series.to_pickle(os.path.join(output_path, "raw_data.bz2"))
                result_list = [standard_bipartite_evaluation(data_frame=df) for df in data_series]
                output_data = pd.DataFrame(data=result_list, index=cutoff_times, columns=["fidelity", "fidelity_std", "key_per_time", "key_per_time_std", "key_per_resource", "key_per_resource_std"])
                output_data.to_csv(os.path.join(output_path, "result.csv"))

        print("The whole run took %s seconds." % str(time() - start_time))

    elif int(sys.argv[1]) == 3:
        # fixed number of memories, variable cutoff_time
        result_path = os.path.join("results", "multimemory_variant_fixed_mem")
        num_processes = 32
        num_memories = 400
        length_list = np.arange(10000, 400000, 2500)
        # BEGIN cutoff estimation
        trial_time_manual = T_P + 2 * (length_list / 2) / C
        expected_time = trial_time_manual / (ETA_TOT * np.exp(-(length_list / 2) / L_ATT))  # expected time ONE memory would take to have a successful pair
        cutoff_multipliers = [0.001, 0.005, 0.010, 0.020, 0.030, 0.050, 0.100, 0.250, 0.500]

        # END cutoff estimation
        max_iter = 1e5
        res = {}
        start_time = time()
        with Pool(num_processes) as pool:
            for cutoff_multiplier in cutoff_multipliers:
                num_calls = len(length_list)
                aux_list = zip(length_list, [max_iter] * num_calls, [params] * num_calls, cutoff_multiplier * expected_time, [num_memories] * num_calls)
                res[cutoff_multiplier] = pool.starmap_async(do_the_thing, aux_list)
            pool.close()
            # pool.join()

            for cutoff_multiplier in cutoff_multipliers:
                data_series = pd.Series(data=res[cutoff_multiplier].get(), index=length_list)
                print("cutoff_multiplier=%s finished after %.2f minutes." % (str(cutoff_multiplier), (time() - start_time) / 60.0))
                output_path = os.path.join(result_path, "%.3f_cutoff" % cutoff_multiplier)
                assert_dir(output_path)
                data_series.to_pickle(os.path.join(output_path, "raw_data.bz2"))
                result_list = [standard_bipartite_evaluation(data_frame=df) for df in data_series]
                output_data = pd.DataFrame(data=result_list, index=length_list, columns=["fidelity", "fidelity_std", "key_per_time", "key_per_time_std", "key_per_resource", "key_per_resource_std"])
                output_data.to_csv(os.path.join(output_path, "result.csv"))

        print("The whole run took %s seconds." % str(time() - start_time))
