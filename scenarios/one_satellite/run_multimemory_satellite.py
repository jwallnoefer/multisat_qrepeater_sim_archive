import os, sys; sys.path.insert(0, os.path.abspath("."))
from scenarios.one_satellite.multi_memory_satellite import run
from scenarios.three_satellites.common_params import base_params
from libs.aux_functions import save_result
import numpy as np
import matplotlib.pyplot as plt
from time import time
from multiprocessing import Pool
import pandas as pd
from consts import SPEED_OF_LIGHT_IN_VACCUM as C
import pickle


def do_the_thing(length, max_iter, params, cutoff_time, num_memories):
    p = run(length=length, max_iter=max_iter, params=params,
            cutoff_time=cutoff_time, num_memories=num_memories)
    return p.data


def do_the_thing_alternate(length, max_iter, params, cutoff_time, num_memories, position_multiplier):
    p = run(length=length, max_iter=max_iter, params=params,
            cutoff_time=cutoff_time, num_memories=num_memories,
            position_multiplier=position_multiplier)
    return p.data


if __name__ == "__main__":
    # run with our standardized parameter set
    case_number = int(sys.argv[1])
    num_processes = int(sys.argv[2])
    result_path = os.path.join("results", "one_satellite")
    path_to_custom_lengths = os.path.join(result_path, "explore")
    if case_number in [1, 2, 3, 4]:
        out_path = os.path.join(result_path, "divergence_theta", str(sys.argv[1]))
        thetas = {1: 3e-6, 2: 4e-6, 3: 6e-6, 4: 8e-6}
        params = dict(base_params)
        params["DIVERGENCE_THETA"] = thetas[case_number]
        num_memories = 1000
        length_list = np.linspace(0, 4400e3, num=96)
        with open(os.path.join(path_to_custom_lengths, f"custom_lengths_{case_number}.pickle"), "rb") as f:
            custom_length_list = pickle.load(f)
        max_iter = 1e5
        cutoff_multiplier = 0.1
        min_cutoff_time = cutoff_multiplier * params["T_DP"]
        cutoff_times = [max(min_cutoff_time, 4 * length / C) for length in custom_length_list]
        # result = {}
        start_time = time()
        with Pool(num_processes) as pool:
            num_calls = len(custom_length_list)
            aux_list = zip(custom_length_list, [max_iter] * num_calls, [params] * num_calls, cutoff_times, [num_memories] * num_calls)
            result = pool.starmap_async(do_the_thing, aux_list)
            pool.close()
            data_series = pd.Series(result.get(), index=custom_length_list)
            output_path = out_path
            save_result(data_series=data_series, output_path=output_path)#, mode="append")
        print("The whole run took %.2f minutes." % ((time() - start_time) / 60))
    elif case_number in [6]:
        out_path = os.path.join(result_path, "memories", str(sys.argv[1]))
        memories = {5: 100, 6: 1000}
        params = dict(base_params)
        num_memories = memories[case_number]
        dephasing_times = [10e-3, 50e-3, 100e-3, 1.0]
        length_list = np.linspace(0, 4400e3, num=96)
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
                aux_list = zip(lens, [max_iter] * num_calls, [t_params] * num_calls, cutoff_times, [num_memories] * num_calls)
                result[t_dp] = pool.starmap_async(do_the_thing, aux_list)
            pool.close()
            for t_dp in dephasing_times:
                lens = custom_length_lists[t_dp]
                data_series = pd.Series(result[t_dp].get(), index=lens)
                output_path = os.path.join(out_path, "%d_t_dp" % int(t_dp * 1000))
                save_result(data_series=data_series, output_path=output_path)#, mode="append")
        print("The whole run took %.2f minutes." % ((time() - start_time) / 60))
    elif case_number == 7:
        #case 7: varying orbital heights
        out_path = os.path.join(result_path, "orbital_heights")
        params = dict(base_params)
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
                aux_list = zip(lens, [max_iter] * num_calls, [h_params] * num_calls, cutoff_times, [num_memories] * num_calls)
                result[h] = pool.starmap_async(do_the_thing, aux_list, chunksize=1)
            pool.close()
            for h in orbital_heights:
                lens = custom_length_lists[h]
                data_series = pd.Series(result[h].get(), index=lens)
                output_path = os.path.join(out_path, "%d_orbital_height" % int(h / 1000))
                save_result(data_series=data_series, output_path=output_path)#, mode="append")
        print("The whole run took %.2f minutes." % ((time() - start_time) / 60))
    elif case_number == 9:
        # Different positions along the orbit of the satellite.
        out_path = os.path.join(result_path, "satellite_path")
        length = 4400e3
        cutoff_multiplier = 0.1
        num_memories = 1000
        orbital_heights = [600e3, 1000e3, 1500e3, 2000e3]
        labels = [int(x / 1000) for x in orbital_heights]
        path_to_custom_variations = path_to_custom_lengths
        with open(os.path.join(path_to_custom_variations, f"custom_variations_{case_number}.pickle"), "rb") as f:
            custom_variations = pickle.load(f)
        custom_variations = [custom_variations[orbital_height] for orbital_height in orbital_heights]
        max_iter = 1e5
        start_time = time()
        result = {}
        with Pool(num_processes) as pool:
            for orbital_height, variations in zip(orbital_heights, custom_variations):
                multipliers = [0.5 + x for x in variations]
                params = dict(base_params)
                params["ORBITAL_HEIGHT"] = orbital_height
                min_cutoff_time = cutoff_multiplier * params["T_DP"]
                cutoff_time = max(min_cutoff_time, 4 * length / C)
                num_calls = len(multipliers)
                aux_list = zip([length] * num_calls,
                               [max_iter] * num_calls,
                               [params] * num_calls,
                               [cutoff_time] * num_calls,
                               [num_memories] * num_calls, multipliers)
                result[orbital_height] = pool.starmap_async(do_the_thing_alternate, aux_list, chunksize=1)
            pool.close()
            for orbital_height, variations in zip(orbital_heights, custom_variations):
                data_series = pd.Series(result[orbital_height].get(), index=variations)
                output_path = os.path.join(out_path, "%d_orbital_height" % int(orbital_height / 1000))
                save_result(data_series=data_series, output_path=output_path)  # , mode="append")
        print("The whole run took %.2f minutes." % ((time() - start_time) / 60))
