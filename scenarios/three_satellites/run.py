import os, sys; sys.path.insert(0, os.path.abspath("."))
from scenarios.three_satellites.twolink_downlink import run, sat_dist_curved, elevation_curved
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

T_2 = 1  # dephasing time
ETA_MEM = 0.8  # memory efficiency
ETA_DET = 0.7  # detector efficiency

ORBITAL_HEIGHT = 400e3
SENDER_APERTURE_RADIUS = 0.15
RECEIVER_APERTURE_RADIUS = 0.50
DIVERGENCE_THETA = 10e-6

P_LINK = ETA_MEM * ETA_DET
base_params = {"P_LINK": P_LINK,
               "T_P": T_P,
               "T_DP": T_2,
               "E_MA": E_M_A,
               "P_D": P_D,
               "LAMBDA_BSM": LAMBDA_BSM,
               "ORBITAL_HEIGHT": ORBITAL_HEIGHT,
               "SENDER_APERTURE_RADIUS": SENDER_APERTURE_RADIUS,
               "RECEIVER_APERTURE_RADIUS": RECEIVER_APERTURE_RADIUS,
               "DIVERGENCE_THETA": DIVERGENCE_THETA}

def do_the_thing(length, max_iter, params, cutoff_time, num_memories, first_satellite_ground_dist_multiplier):
    p = run(length=length, max_iter=max_iter, params=params, cutoff_time=cutoff_time, num_memories=num_memories, first_satellite_ground_dist_multiplier=first_satellite_ground_dist_multiplier)
    return p.data


if __name__ == "__main__":
    # fixed memory time, variable cutoff time
    params = dict(base_params)
    result_path = os.path.join("results", "three_satellites", "twolink_downlink")
    num_processes = 32
    length_list = np.linspace(0, 3600e3, num=96)
    num_memories = 1000
    max_iter = 1e5
    cutoff_multiplier = 0.1
    first_satellite_multipliers = np.linspace(0, 0.5, num=9)
    if int(sys.argv[1]) == 0:
        first_satellite_multipliers = first_satellite_multipliers[0:2]
    elif int(sys.argv[1]) == 1:
        first_satellite_multipliers = first_satellite_multipliers[2:4]
    elif int(sys.argv[1]) == 2:
        first_satellite_multipliers = first_satellite_multipliers[4:6]
    elif int(sys.argv[1]) == 3:
        first_satellite_multipliers = first_satellite_multipliers[6:8]
    elif int(sys.argv[1]) == 4:
        first_satellite_multipliers = first_satellite_multipliers[-1]
    res = {}
    start_time = time()
    with Pool(num_processes) as pool:
        for first_satellite_multiplier in first_satellite_multipliers:
            cutoff_time = cutoff_multiplier * params["T_DP"]
            num_calls = len(length_list)
            aux_list = zip(length_list, [max_iter] * num_calls, [params] * num_calls, [cutoff_time] * num_calls, [num_memories] * num_calls, [first_satellite_multiplier] * num_calls)
            res[first_satellite_multiplier] = pool.starmap_async(do_the_thing, aux_list)
        pool.close()
        # pool.join()

        for first_satellite_multiplier in first_satellite_multipliers:
            data_series = pd.Series(data=res[first_satellite_multiplier].get(), index=length_list)
            print("first_satellite_multiplier=%s finished after %.2f minutes." % (str(first_satellite_multiplier), (time() - start_time) / 60.0))
            output_path = os.path.join(result_path, "%.3f_first_sat" % cutoff_multiplier)
            assert_dir(output_path)
            try:
                existing_series = pd.read_pickle(os.path.join(output_path, "raw_data.bz2"))
                combined_series = existing_series.append(data_series)
                combined_series.to_pickle(os.path.join(output_path, "raw_data.bz2"))
            except FileNotFoundError:
                data_series.to_pickle(os.path.join(output_path, "raw_data.bz2"))
            result_list = [standard_bipartite_evaluation(data_frame=df) for df in data_series]
            output_data = pd.DataFrame(data=result_list, index=length_list, columns=["fidelity", "fidelity_std", "key_per_time", "key_per_time_std", "key_per_resource", "key_per_resource_std"])
            try:
                existing_data = pd.read_csv(os.path.join(output_path, "result.csv"), index_col=0)
                combined_data = pd.concat([existing_data, output_data])
                combined_data.to_csv(os.path.join(output_path, "result.csv"))
            except FileNotFoundError:
                output_data.to_csv(os.path.join(output_path, "result.csv"))

    print("The whole run took %s seconds." % str(time() - start_time))
