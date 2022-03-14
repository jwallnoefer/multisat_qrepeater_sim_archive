import os, sys; sys.path.insert(0, os.path.abspath("."))
from scenarios.three_satellites.twolink_downlink import run
from scenarios.three_satellites.common_params import base_params
from libs.aux_functions import save_result
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
    return [my_list[i:i + chunksize] for i in range(0, len(my_list), chunksize)]


def labeled_split_list(label, my_list, chunksize):
    return [(label, da_list) for da_list in split_list(my_list, chunksize)]


def reorder_runs(run_list):
    my_list = zip(*run_list)
    # strip extra nesting layer
    new_run_list = [run for nested_list in my_list for run in nested_list]
    return new_run_list


def do_the_thing(length, max_iter, params, cutoff_time, num_memories, first_satellite_ground_dist_multiplier):
    np.random.seed()
    p = run(length=length, max_iter=max_iter, params=params, cutoff_time=cutoff_time, num_memories=num_memories, first_satellite_ground_dist_multiplier=first_satellite_ground_dist_multiplier)
    return p.data


def do_the_thing_alternate(length, max_iter, params, cutoff_time, num_memories, satellite_multipliers):
    np.random.seed()
    p = run(length=length, max_iter=max_iter, params=params, cutoff_time=cutoff_time, num_memories=num_memories, satellite_multipliers=satellite_multipliers)
    return p.data


if __name__ == "__main__":
    result_path = os.path.join("results", "three_satellites", "twolink_downlink")
    path_to_custom_lengths = os.path.join(result_path, "explore")
    case_number = int(sys.argv[1])
    num_processes = int(sys.argv[2])
    if case_number == 0:
        out_path = os.path.join(result_path, "sat_positions")
        params = dict(base_params)
        num_memories = 1000
        # length_list = np.linspace(0, 8800e3, num=96)
        max_iter = 1e5
        cutoff_multiplier = 0.1
        min_cutoff_time = cutoff_multiplier * params["T_DP"]
        first_satellite_multipliers = np.linspace(0, 0.5, num=6)
        with open(os.path.join(path_to_custom_lengths, f"custom_lengths_{case_number}.pickle"), "rb") as f:
            custom_length_lists = pickle.load(f)
        custom_length_lists = [custom_length_lists[key] for key in first_satellite_multipliers]
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
    elif case_number in [2, 3, 4]:
        out_path = os.path.join(result_path, "divergence_theta", str(sys.argv[1]))
        thetas = {1: 2e-6, 2: 4e-6, 3: 6e-6, 4: 8e-6}
        params = dict(base_params)
        params["DIVERGENCE_THETA"] = thetas[case_number]
        num_memories = 1000
        # length_list = np.linspace(0, 8800e3, num=96)
        max_iter = 1e5
        cutoff_multiplier = 0.1
        min_cutoff_time = cutoff_multiplier * params["T_DP"]
        first_satellite_multipliers = [0.0, 0.2]
        with open(os.path.join(path_to_custom_lengths, f"custom_lengths_{case_number}.pickle"), "rb") as f:
            custom_length_lists = pickle.load(f)
        custom_length_lists = [custom_length_lists[key] for key in first_satellite_multipliers]
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
    elif case_number in [6]:
        out_path = os.path.join(result_path, "memories", str(sys.argv[1]))
        memories = {5: 100, 6: 1000}
        params = dict(base_params)
        first_satellite_multiplier = 0.0
        num_memories = memories[case_number]
        dephasing_times = [2e-3, 3e-3, 4e-3, 5e-3, 10e-3, 50e-3, 100e-3, 1.0]
        # length_list = np.linspace(0, 8800e3, num=96)
        with open(os.path.join(path_to_custom_lengths, f"custom_lengths_{case_number}.pickle"), "rb") as f:
            custom_length_lists = pickle.load(f)
        max_iter = 1e5
        cutoff_multiplier = 0.1
        result = {}
        start_time = time()
        with Pool(num_processes) as pool:
            for t_dp in dephasing_times:
                t_params = dict(params)
                t_params["T_DP"] = t_dp
                lens = custom_length_lists[t_dp]
                min_cutoff_time = cutoff_multiplier * t_params["T_DP"]
                cutoff_times = [max(min_cutoff_time, 4 * length / C) for length in lens]
                num_calls = len(lens)
                aux_list = zip(lens, [max_iter] * num_calls, [t_params] * num_calls, cutoff_times, [num_memories] * num_calls, [first_satellite_multiplier] * num_calls)
                result[t_dp] = pool.starmap_async(do_the_thing, aux_list, chunksize=1)
            pool.close()
            for t_dp in dephasing_times:
                lens = custom_length_lists[t_dp]
                data_series = pd.Series(result[t_dp].get(), index=lens)
                output_path = os.path.join(out_path, "%d_t_dp" % int(t_dp * 1000))
                save_result(data_series=data_series, output_path=output_path)#, mode="append")
        print("The whole run took %.2f minutes." % ((time() - start_time) / 60))
    elif case_number == 7:
        # case 7: varying orbital heights
        out_path = os.path.join(result_path, "orbital_heights")
        params = dict(base_params)
        first_satellite_multiplier = 0.0
        num_memories = 1000
        orbital_heights = [400e3, 600e3, 1000e3, 1500e3, 2000e3]
        # length_list = np.linspace(0, 8800e3, num=96)
        with open(os.path.join(path_to_custom_lengths, f"custom_lengths_{case_number}.pickle"), "rb") as f:
            custom_length_lists = pickle.load(f)
        max_iter = 1e5
        cutoff_multiplier = 0.1
        min_cutoff_time = cutoff_multiplier * params["T_DP"]
        result = {}
        start_time = time()
        with Pool(num_processes) as pool:
            for h in orbital_heights:
                h_params = dict(params)
                h_params["ORBITAL_HEIGHT"] = h
                lens = custom_length_lists[h]
                cutoff_times = [max(min_cutoff_time, 4 * length / C) for length in lens]
                num_calls = len(lens)
                aux_list = zip(lens, [max_iter] * num_calls, [h_params] * num_calls, cutoff_times, [num_memories] * num_calls, [first_satellite_multiplier] * num_calls)
                result[h] = pool.starmap_async(do_the_thing, aux_list, chunksize=1)
            pool.close()
            for h in orbital_heights:
                lens = custom_length_lists[h]
                data_series = pd.Series(result[h].get(), index=lens)
                output_path = os.path.join(out_path, "%d_orbital_height" % int(h / 1000))
                save_result(data_series=data_series, output_path=output_path)#, mode="append")
        print("The whole run took %.2f minutes." % ((time() - start_time) / 60))
    elif case_number == 8:
        # case 8: varying cutoff times to show optimizing this is important
        out_path = os.path.join(result_path, "cutoff_times")
        params = dict(base_params)
        params["DIVERGENCE_THETA"] = 6e-6
        first_satellite_multiplier = 0.0
        num_memories = 1000
        cutoff_multipliers = [None, 0.5, 0.2, 0.1, 0.05, 0.02]
        with open(os.path.join(path_to_custom_lengths, f"custom_lengths_{case_number}.pickle"), "rb") as f:
            custom_length_lists = pickle.load(f)
        # custom_length_lists = [custom_length_lists[key][:-1] for key in cutoff_multipliers]
        custom_length_lists = [custom_length_lists[key] for key in cutoff_multipliers]
        max_iter = 1e5
        result = {}
        start_time = time()
        with Pool(num_processes) as pool:
            for cutoff_multiplier, lens in zip(cutoff_multipliers, custom_length_lists):
                try:
                    cutoff_time = cutoff_multiplier * params["T_DP"]
                except TypeError as e:
                    if cutoff_multiplier is None:
                        cutoff_time = None
                    else:
                        raise e
                num_calls = len(lens)
                aux_list = zip(lens, [max_iter] * num_calls, [params] * num_calls, [cutoff_time] * num_calls, [num_memories] * num_calls, [first_satellite_multiplier] * num_calls)
                result[cutoff_multiplier] = pool.starmap_async(do_the_thing, aux_list, chunksize=1)
            pool.close()
            for cutoff_multiplier, lens in zip(cutoff_multipliers, custom_length_lists):
                data_series = pd.Series(result[cutoff_multiplier].get(), index=lens)
                try:
                    dir_prefix = "%d" % int(cutoff_multiplier * 100)
                except TypeError as e:
                    if cutoff_multiplier is None:
                        dir_prefix = "None"
                    else:
                        raise e
                output_path = os.path.join(out_path, dir_prefix + "_cutoff_multiplier")
                save_result(data_series=data_series, output_path=output_path)#, mode="append")
        print("The whole run took %.2f minutes." % ((time() - start_time) / 60))
    elif case_number == 9:
        # Different positions along orbit for satellites (because apparently satellites move)
        out_path = os.path.join(result_path, "satellite_path")
        params = dict(base_params)
        length = 4400e3
        cutoff_multiplier = 0.1
        min_cutoff_time = cutoff_multiplier * params["T_DP"]
        cutoff_time = max(min_cutoff_time, 4 * length / C)
        num_memories = 1000
        configurations = [np.array([-0.1, 0.5, 1.1]), np.array([0, 0.5, 1]),
                          np.array([0.1, 0.5, 0.9]), np.array([0.2, 0.5, 0.8])]
        labels = [str(int(base_multipliers[0] * 10)) for base_multipliers in configurations]
        path_to_custom_variations = path_to_custom_lengths
        # num_calls = 97
        # variations = np.linspace(-0.2, 0.2, num=num_calls)
        with open(os.path.join(path_to_custom_variations, f"custom_variations_{case_number}.pickle"), "rb") as f:
            custom_variations = pickle.load(f)
        custom_variations = [custom_variations[label] for label in labels]
        max_iter = 1e5
        start_time = time()
        result = {}
        with Pool(num_processes) as pool:
            for base_multipliers, variations, label in zip(configurations, custom_variations, labels):
                multipliers = [base_multipliers + x for x in variations]
                num_calls = len(multipliers)
                aux_list = zip([length] * num_calls, [max_iter] * num_calls, [params] * num_calls, [cutoff_time] * num_calls, [num_memories] * num_calls, multipliers)
                result[label] = pool.starmap_async(do_the_thing_alternate, aux_list, chunksize=1)
            pool.close()
            for base_multipliers, variations, label in zip(configurations, custom_variations, labels):
                data_series = pd.Series(result[label].get(), index=variations)
                output_path = os.path.join(out_path, f"{label}_configuration")
                save_result(data_series=data_series, output_path=output_path)#, mode="append")
        print("The whole run took %.2f minutes." % ((time() - start_time) / 60))
