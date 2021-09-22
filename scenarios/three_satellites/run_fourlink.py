import os, sys; sys.path.insert(0, os.path.abspath("."))
from scenarios.three_satellites.fourlink import run, sat_dist_curved, elevation_curved
from libs.aux_functions import assert_dir, standard_bipartite_evaluation, save_result
import numpy as np
import matplotlib.pyplot as plt
from time import time
from multiprocessing import Pool
import pandas as pd
from consts import SPEED_OF_LIGHT_IN_VACCUM as C
import pickle

def max_length_horizon(first_satellite_multiplier):
    if first_satellite_multiplier <= 1 / 6:
        return 2 * 4400e3 / (1 - 2 * first_satellite_multiplier)
    else:
        return 2200e3 / first_satellite_multiplier

def split_list(my_list, chunksize):
    return [my_list[i:i+chunksize] for i in range(0, len(my_list), chunksize)]

def labeled_split_list(label, my_list, chunksize):
    return [(label, da_list) for da_list in split_list(my_list, chunksize)]

def reorder_runs(run_list):
    my_list = zip(*run_list)
    # strip extra nesting layer
    new_run_list = [run for nested_list in my_list for run in nested_list]
    return new_run_list

# # # values taken from https://arxiv.org/abs/2006.10636
F_CLOCK = 20e6  # 20 MHz is very high
T_P = 0  # preparation time
E_M_A = 0  # misalignment error
P_D = 10**-6  # dark count probability per detector
P_BSM = 1  # BSM success probability  ## WARNING: Currently not implemented
LAMBDA_BSM = 1  # BSM ideality parameter
F = 1  # error correction inefficiency

# T_2 = 1  # dephasing time
ETA_MEM = 0.8  # memory efficiency
ETA_DET = 0.7  # detector efficiency

ORBITAL_HEIGHT = 400e3
SENDER_APERTURE_RADIUS = 0.15
RECEIVER_APERTURE_RADIUS = 0.50
# DIVERGENCE_THETA = 10e-6

P_LINK = ETA_MEM * ETA_DET
base_params = {"P_LINK": P_LINK,
               "ETA_MEM": ETA_MEM,
               "ETA_DET": ETA_DET,
               "T_P": T_P,
               "F_CLOCK": F_CLOCK,
               "E_MA": E_M_A,
               "P_D": P_D,
               "LAMBDA_BSM": LAMBDA_BSM,
               "ORBITAL_HEIGHT": ORBITAL_HEIGHT,
               "SENDER_APERTURE_RADIUS": SENDER_APERTURE_RADIUS,
               "RECEIVER_APERTURE_RADIUS": RECEIVER_APERTURE_RADIUS}


def do_the_thing(length, max_iter, params, cutoff_time, num_memories, first_satellite_ground_dist_multiplier):
    np.random.seed()
    p = run(length=length, max_iter=max_iter, params=params, cutoff_time=cutoff_time, num_memories=num_memories, first_satellite_ground_dist_multiplier=first_satellite_ground_dist_multiplier)
    return p.data


if __name__ == "__main__":
    result_path = os.path.join("results", "three_satellites", "fourlink")
    path_to_custom_lengths = os.path.join(result_path, "explore")
    case_number = int(sys.argv[1])
    num_processes = int(sys.argv[2])
    if case_number in [0, 8]:
        out_path = os.path.join(result_path, "sat_positions")
        params = dict(base_params)
        params["DIVERGENCE_THETA"] = 2e-6
        params["T_DP"] = 100e-3
        num_memories = 1000
        # length_list = np.linspace(0, 8800e3, num=96)
        max_iter = 1e4
        cutoff_multiplier = 0.1
        min_cutoff_time = cutoff_multiplier * params["T_DP"]
        first_satellite_multipliers = np.linspace(0, 0.5, num=6)
        if case_number == 0:
            first_satellite_multipliers = first_satellite_multipliers[::2]
        elif case_number == 8:
            first_satellite_multipliers = first_satellite_multipliers[1::2]
        # first_satellite_multipliers = first_satellite_multipliers[4:]
        # length_cutoffs = [max_length_horizon(fsm) for fsm in first_satellite_multipliers]
        # length_cutoffs = [7000e3, 6000e3, 5500e3, 5000e3, 3800e3]
        # length_starts = [4400e3] * 4 + [3620e3]
        # custom_length_lists = [length_list[np.logical_and(length_list <= len_cutoff, length_list > length_start)] for len_cutoff, length_start in zip(length_cutoffs, length_starts)]
        with open(os.path.join(path_to_custom_lengths, f"custom_lengths_{case_number}.pickle"), "rb") as f:
            custom_length_lists = pickle.load(f)
        custom_length_lists = [custom_length_lists[key][:-1] for key in first_satellite_multipliers]
        # remove last two for particularly tricky case
        if case_number == 8:
            custom_length_lists[-1] = custom_length_lists[-1][:-2]
        result = {}
        start_time = time()
        with Pool(num_processes) as pool:
            for multiplier, lens in zip(first_satellite_multipliers, custom_length_lists):
                num_calls = len(lens)
                cutoff_times = [max(min_cutoff_time, 4 * length / C) for length in lens]
                aux_list = zip(lens, [max_iter] * num_calls, [params] * num_calls, cutoff_times, [num_memories] * num_calls, [multiplier] * num_calls)
                result[multiplier] = pool.starmap_async(do_the_thing, aux_list, chunksize=1)
            pool.close()
            for multiplier, lens in zip(first_satellite_multipliers, custom_length_lists):
                data_series = pd.Series(result[multiplier].get(), index=lens)
                output_path = os.path.join(out_path, "%.3f_first_sat" % multiplier)
                save_result(data_series=data_series, output_path=output_path)#, mode="append")
        print("The whole run took %.2f minutes." % ((time() - start_time) / 60))
    elif case_number in [1, 2, 3, 4]:
        out_path = os.path.join(result_path, "divergence_theta", str(sys.argv[1]))
        thetas = {1: 2e-6, 2: 4e-6, 3: 6e-6, 4: 8e-6}
        params = dict(base_params)
        params["DIVERGENCE_THETA"] = thetas[case_number]
        params["T_DP"] = 100e-3
        num_memories = 1000
        # length_list = np.linspace(0, 8800e3, num=96)
        max_iter = 1e4
        cutoff_multiplier = 0.1
        min_cutoff_time = cutoff_multiplier * params["T_DP"]
        first_satellite_multipliers = [0.0, 0.2]
        # first_satellite_multipliers = [0.000, 0.200, 0.400, 0.500]
        # first_satellite_multipliers = first_satellite_multipliers[2:]
        # length_cutoffs = [max_length_horizon(fsm) for fsm in first_satellite_multipliers]
        # cutoff_dict = {1: [4400e3, 4400e3],
        #                2: [4400e3, 4400e3],
        #                3: [3000e3] * 2,
        #                4: [2200e3] * 2}
        # start_dict = {#1: [2200e3] * 4,
        #               #2: [2200e3] * 4,
        #               3: [2200e3] * 4,
        #               # 4: [2200e3, 2200e3, 2200e3, 1390e3]
        #               }
        # length_cutoffs = cutoff_dict[case_number]
        # length_starts = start_dict[case_number]
        # custom_length_lists = [length_list[np.logical_and(length_list <= len_cutoff, length_list > length_start)] for len_cutoff, length_start in zip(length_cutoffs, length_starts)]
        with open(os.path.join(path_to_custom_lengths, f"custom_lengths_{case_number}.pickle"), "rb") as f:
            custom_length_lists = pickle.load(f)
        custom_length_lists = [custom_length_lists[key][:-1] for key in first_satellite_multipliers]
        result = {}
        start_time = time()
        with Pool(num_processes) as pool:
            for multiplier, lens in zip(first_satellite_multipliers, custom_length_lists):
                num_calls = len(lens)
                cutoff_times = [max(min_cutoff_time, 4 * length / C) for length in lens]
                aux_list = zip(lens, [max_iter] * num_calls, [params] * num_calls, cutoff_times, [num_memories] * num_calls, [multiplier] * num_calls)
                result[multiplier] = pool.starmap_async(do_the_thing, aux_list, chunksize=1)
            pool.close()
            for multiplier, lens in zip(first_satellite_multipliers, custom_length_lists):
                data_series = pd.Series(result[multiplier].get(), index=lens)
                output_path = os.path.join(out_path, "%.3f_first_sat" % multiplier)
                save_result(data_series=data_series, output_path=output_path)#, mode="append")
        print("The whole run took %.2f minutes." % ((time() - start_time) / 60))
    elif case_number in [5, 6]:
        out_path = os.path.join(result_path, "memories", str(sys.argv[1]))
        memories = {5: 100, 6: 1000}
        params = dict(base_params)
        params["DIVERGENCE_THETA"] = 2e-6
        first_satellite_multiplier = 0.2
        num_memories = memories[case_number]
        dephasing_times = [10e-3, 50e-3, 100e-3, 1.0]
        # length_list = np.linspace(0, 8800e3, num=96)
        with open(os.path.join(path_to_custom_lengths, f"custom_lengths_{case_number}.pickle"), "rb") as f:
            custom_length_lists = pickle.load(f)
        custom_length_lists = [custom_length_lists[key][:-1] for key in dephasing_times]
        max_iter = 1e4
        cutoff_multiplier = 0.1
        result = {}
        start_time = time()
        with Pool(num_processes) as pool:
            for t_dp, lens in zip(dephasing_times, custom_length_lists):
                t_params = dict(params)
                t_params["T_DP"] = t_dp
                # lens = custom_length_lists[t_dp]
                # lens = lens[:-1]
                min_cutoff_time = cutoff_multiplier * t_params["T_DP"]
                cutoff_times = [max(min_cutoff_time, 4 * length / C) for length in lens]
                num_calls = len(lens)
                aux_list = zip(lens, [max_iter] * num_calls, [t_params] * num_calls, cutoff_times, [num_memories] * num_calls, [first_satellite_multiplier] * num_calls)
                result[t_dp] = pool.starmap_async(do_the_thing, aux_list, chunksize=1)
            pool.close()
            for t_dp, lens in zip(dephasing_times, custom_length_lists):
                # lens = custom_length_lists[t_dp]
                # lens = lens[:-1]
                data_series = pd.Series(result[t_dp].get(), index=lens)
                output_path = os.path.join(out_path, "%d_t_dp" % int(t_dp * 1000))
                save_result(data_series=data_series, output_path=output_path)#, mode="append")
        print("The whole run took %.2f minutes." % ((time() - start_time) / 60))
    elif case_number == 7:
        #case 7: varying orbital heights
        out_path = os.path.join(result_path, "orbital_heights")
        params = dict(base_params)
        params["DIVERGENCE_THETA"] = 2e-6
        params["T_DP"] = 100e-3
        first_satellite_multiplier = 0.2
        num_memories = 1000
        orbital_heights = [400e3, 600e3, 1000e3, 1500e3, 2000e3]
        # length_list = np.linspace(0, 8800e3, num=96)
        with open(os.path.join(path_to_custom_lengths, f"custom_lengths_{case_number}.pickle"), "rb") as f:
            custom_length_lists = pickle.load(f)
        custom_length_lists = [custom_length_lists[key][:-1] for key in orbital_heights]
        max_iter = 1e4
        cutoff_multiplier = 0.1
        min_cutoff_time = cutoff_multiplier * params["T_DP"]
        result = {}
        start_time = time()
        with Pool(num_processes) as pool:
            for h, lens in zip(orbital_heights, custom_length_lists):
                h_params = dict(params)
                h_params["ORBITAL_HEIGHT"] = h
                # lens = custom_length_lists[h]
                # lens = lens[:-1]
                cutoff_times = [max(min_cutoff_time, 4 * length / C) for length in lens]
                num_calls = len(lens)
                aux_list = zip(lens, [max_iter] * num_calls, [h_params] * num_calls, cutoff_times, [num_memories] * num_calls, [first_satellite_multiplier] * num_calls)
                result[h] = pool.starmap_async(do_the_thing, aux_list, chunksize=1)
            pool.close()
            for h, lens in zip(orbital_heights, custom_length_lists):
                # lens = custom_length_lists[h]
                # lens = lens[:-1]
                data_series = pd.Series(result[h].get(), index=lens)
                output_path = os.path.join(out_path, "%d_orbital_height" % int(h / 1000))
                save_result(data_series=data_series, output_path=output_path)#, mode="append")
        print("The whole run took %.2f minutes." % ((time() - start_time) / 60))