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
    length_list = np.linspace(0, 3600e3, num=96)
    num_memories = 1000
    max_iter = 1e5
    cutoff_multiplier = 0.1
    num_processes = 32
    first_satellite_ground_dist_multiplier = 0
    num_calls = len(length_list)
    aux_list = zip(length_list, [max_iter] * num_calls, [base_params] * num_calls, [cutoff_multiplier * base_params["T_DP"]] * num_calls, [num_memories] * num_calls, [first_satellite_ground_dist_multiplier] * num_calls)
    with Pool(num_processes) as pool:
        result = pool.starmap_async(do_the_thing, aux_list)
        pool.close()
        actual_result = pd.Series(result.get(), index=length_list)
        evaluated_result = [standard_bipartite_evaluation(df) for df in actual_result]
        output = pd.DataFrame(data=evaluated_result, index=length_list, columns=["fidelity", "fidelity_std", "key_per_time", "key_per_time_std", "key_per_resource", "key_per_resource_std"])
    plt.scatter(output.index, output["key_per_time"])
    plt.yscale("log")
    plt.grid()
    plt.show()
