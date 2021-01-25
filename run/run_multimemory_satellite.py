import os, sys; sys.path.insert(0, os.path.abspath("."))
from scenarios.multi_memory_satellite import run, sat_dist_curved, elevation_curved
from libs.aux_functions import assert_dir, standard_bipartite_evaluation
import numpy as np
import matplotlib.pyplot as plt
from time import time
from multiprocessing import Pool
import pandas as pd
from consts import SPEED_OF_LIGHT_IN_VACCUM as C


# # # values taken from https://arxiv.org/abs/2006.10636
T_P = 0  # preparation time
E_M_A = 0  # misalignment error
P_D = 10**-6  # dark count probability per detector
P_BSM = 1  # BSM success probability  ## WARNING: Currently not implemented
LAMBDA_BSM = 1  # BSM ideality parameter
F = 1  # error correction inefficiency

# T_2 = 2  # dephasing time
ETA_MEM = 0.8  # memory efficiency
ETA_DET = 0.7  # detector efficiency

ORBITAL_HEIGHT = 400e3
SENDER_APERTURE_RADIUS = 0.15
RECEIVER_APERTURE_RADIUS = 0.50
DIVERGENCE_THETA = 10e-6

P_LINK = ETA_MEM * ETA_DET
base_params = {"P_LINK": P_LINK,
               "T_P": T_P,
               "E_MA": E_M_A,
               "P_D": P_D,
               "LAMBDA_BSM": LAMBDA_BSM,
               "ORBITAL_HEIGHT": ORBITAL_HEIGHT,
               "SENDER_APERTURE_RADIUS": SENDER_APERTURE_RADIUS,
               "RECEIVER_APERTURE_RADIUS": RECEIVER_APERTURE_RADIUS,
               "DIVERGENCE_THETA": DIVERGENCE_THETA}


def do_the_thing(length, max_iter, params, cutoff_time, num_memories):
    p = run(length=length, max_iter=max_iter, params=params, cutoff_time=cutoff_time, num_memories=num_memories)
    return p.data


if __name__ == "__main__":
    if int(sys.argv[1]) == 0 or int(sys.argv[1]) == 1:
        # fixed memory time, variable cutoff time
        params = dict(base_params)
        params["T_DP"] = 7.5
        result_path = os.path.join("results", "multimemory_satellite_cutoff")
        num_processes = 32
        length_list = np.linspace(10e3, 3200e3, num=120)
        num_memories = 1000
        max_iter = 1e5
        if int(sys.argv[1]) == 0:
            cutoff_multipliers = [0.001, 0.005, 0.010, 0.020]
        elif int(sys.argv[1]) == 1:
            cutoff_multipliers = [0.030, 0.050, 0.100, 0.250, 0.500]
        res = {}
        start_time = time()
        with Pool(num_processes) as pool:
            for cutoff_multiplier in cutoff_multipliers:
                cutoff_time = cutoff_multiplier * params["T_DP"]
                num_calls = len(length_list)
                aux_list = zip(length_list, [max_iter] * num_calls, [params] * num_calls, [cutoff_time] * num_calls, [num_memories] * num_calls)
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

    elif int(sys.argv[1]) == 2 or int(sys.argv[1]) == 3:
        # variable memory quality
        result_path = os.path.join("results", "multimemory_satellite_dephasing")
        num_processes = 32
        length_list = np.linspace(10e3, 3200e3, num=120)
        num_memories = 1000
        max_iter = 1e5
        cutoff_multiplier = 0.050
        dephasing_list = np.logspace(np.log10(100e-3), np.log10(7.5), num=10)
        if int(sys.argv[1]) == 2:
            dephasing_list = dephasing_list[:5]
        elif int(sys.argv[1]) == 3:
            dephasing_list = dephasing_list[5:]
        res = {}
        start_time = time()
        with Pool(num_processes) as pool:
            for t_dp in dephasing_list:
                params = dict(base_params)
                params["T_DP"] = t_dp
                cutoff_time = cutoff_multiplier * params["T_DP"]
                num_calls = len(length_list)
                aux_list = zip(length_list, [max_iter] * num_calls, [params] * num_calls, [cutoff_time] * num_calls, [num_memories] * num_calls)
                res[t_dp] = pool.starmap_async(do_the_thing, aux_list)
            pool.close()
            # pool.join()

            for t_dp in dephasing_list:
                data_series = pd.Series(data=res[t_dp].get(), index=length_list)
                print("dephasing=%s finished after %.2f minutes." % (str(t_dp), (time() - start_time) / 60.0))
                output_path = os.path.join(result_path, "%.2f_dephasing" % t_dp)
                assert_dir(output_path)
                data_series.to_pickle(os.path.join(output_path, "raw_data.bz2"))
                result_list = [standard_bipartite_evaluation(data_frame=df) for df in data_series]
                output_data = pd.DataFrame(data=result_list, index=length_list, columns=["fidelity", "fidelity_std", "key_per_time", "key_per_time_std", "key_per_resource", "key_per_resource_std"])
                output_data.to_csv(os.path.join(output_path, "result.csv"))
