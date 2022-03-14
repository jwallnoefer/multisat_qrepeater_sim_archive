import os, sys; sys.path.insert(0, os.path.abspath("."))
from scenarios.three_satellites.twolink_downlink import run
from scenarios.three_satellites.common_params import base_params
from libs.aux_functions import save_result
import numpy as np
from time import time
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


def run_finish(case_number, subcase, length):
    result_path = os.path.join("results", "three_satellites", "twolink_downlink")
    if case_number == 0:
        out_path = os.path.join(result_path, "sat_positions")
        params = dict(base_params)
        num_memories = 1000
        # length_list = np.linspace(0, 8800e3, num=96)
        max_iter = 1e5
        cutoff_multiplier = 0.1
        min_cutoff_time = cutoff_multiplier * params["T_DP"]
        cutoff_time = max(min_cutoff_time, 4 * length / C)
        multiplier = subcase
        res = do_the_thing(length, max_iter, params, cutoff_time, num_memories, multiplier)
        data_series = pd.Series([res], index=[length])
        output_path = os.path.join(out_path, "%.3f_first_sat" % multiplier)
        save_result(data_series=data_series, output_path=output_path, mode="append")
    elif case_number in [2, 3, 4]:
        out_path = os.path.join(result_path, "divergence_theta", str(case_number))
        thetas = {1: 2e-6, 2: 4e-6, 3: 6e-6, 4: 8e-6}
        params = dict(base_params)
        params["DIVERGENCE_THETA"] = thetas[case_number]
        num_memories = 1000
        # length_list = np.linspace(0, 8800e3, num=96)
        max_iter = 1e5
        cutoff_multiplier = 0.1
        min_cutoff_time = cutoff_multiplier * params["T_DP"]
        cutoff_time = max(min_cutoff_time, 4 * length / C)
        multiplier = subcase
        res = do_the_thing(length, max_iter, params, cutoff_time, num_memories, multiplier)
        data_series = pd.Series([res], index=[length])
        output_path = os.path.join(out_path, "%.3f_first_sat" % multiplier)
        save_result(data_series=data_series, output_path=output_path, mode="append")
    elif case_number in [6]:
        out_path = os.path.join(result_path, "memories", str(case_number))
        memories = {5: 100, 6: 1000}
        params = dict(base_params)
        first_satellite_multiplier = 0.0
        num_memories = memories[case_number]
        t_dp = subcase
        max_iter = 1e5
        cutoff_multiplier = 0.1
        t_params = dict(params)
        t_params["T_DP"] = t_dp
        min_cutoff_time = cutoff_multiplier * t_params["T_DP"]
        cutoff_time = max(min_cutoff_time, 4 * length / C)
        res = do_the_thing(length, max_iter, t_params, cutoff_time, num_memories, first_satellite_multiplier)
        data_series = pd.Series([res], index=[length])
        output_path = os.path.join(out_path, "%d_t_dp" % int(t_dp * 1000))
        save_result(data_series=data_series, output_path=output_path, mode="append")
    elif case_number == 7:
        # case 7: varying orbital heights
        out_path = os.path.join(result_path, "orbital_heights")
        params = dict(base_params)
        first_satellite_multiplier = 0.0
        num_memories = 1000
        h = subcase
        max_iter = 1e5
        cutoff_multiplier = 0.1
        min_cutoff_time = cutoff_multiplier * params["T_DP"]
        cutoff_time = max(min_cutoff_time, 4 * length / C)
        h_params = dict(params)
        h_params["ORBITAL_HEIGHT"] = h
        res = do_the_thing(length, max_iter, h_params, cutoff_time, num_memories, first_satellite_multiplier)
        data_series = pd.Series([res], index=[length])
        output_path = os.path.join(out_path, "%d_orbital_height" % int(h / 1000))
        save_result(data_series=data_series, output_path=output_path, mode="append")
    elif case_number == 8:
        # case 8: varying cutoff times to show optimizing this is important
        out_path = os.path.join(result_path, "cutoff_times")
        params = dict(base_params)
        params["DIVERGENCE_THETA"] = 5e-6
        first_satellite_multiplier = 0.0
        num_memories = 1000
        cutoff_multiplier = subcase
        max_iter = 1e5
        try:
            cutoff_time = cutoff_multiplier * params["T_DP"]
        except TypeError as e:
            if cutoff_multiplier is None:
                cutoff_time = None
            else:
                raise e
        res = do_the_thing(length, max_iter, params, cutoff_time, num_memories, first_satellite_multiplier)
        data_series = pd.Series([res], index=[length])
        try:
            dir_prefix = "%d" % int(cutoff_multiplier * 100)
        except TypeError as e:
            if cutoff_multiplier is None:
                dir_prefix = "None"
            else:
                raise e
        output_path = os.path.join(out_path, dir_prefix + "_cutoff_multiplier")
        save_result(data_series=data_series, output_path=output_path, mode="append")
    elif case_number == 9:
        out_path = os.path.join(result_path, "satellite_path")
        params = dict(base_params)
        assert length == 4400e3
        min_cutoff_time = cutoff_multiplier * params["T_DP"]
        cutoff_time = max(min_cutoff_time, 4 * length / C)
        num_memories = 1000
        max_iter = 1e5
        base_multipliers, variation, label = subcase
        multipliers = base_multipliers + variation
        res = do_the_thing_alternate(length, max_iter, params, cutoff_time, num_memories, satellite_multipliers=multipliers)
        data_series = pd.Series([res], index=[variation])
        output_path = os.path.join(out_path, f"{label}_configuration")
        save_result(data_series=data_series, output_path=output_path, mode="append")
        multipliers = base_multipliers - variation
        res = do_the_thing_alternate(length, max_iter, params, cutoff_time, num_memories, satellite_multipliers=multipliers)
        data_series = pd.Series([res], index=[variation])
        output_path = os.path.join(out_path, f"{label}_configuration")
        save_result(data_series=data_series, output_path=output_path, mode="append")
        # reorder data
        existing_series = pd.read_pickle(os.path.join(output_path, "raw_data.bz2"))
        data_series = existing_series.sort_index()
        existing_data = pd.read_csv(os.path.join(output_path, "result.csv"), index_col=0)
        output_data = existing_data.sort_index()
        data_series.to_pickle(os.path.join(output_path, "raw_data.bz2"))
        output_data.to_csv(os.path.join(output_path, "result.csv"))


if __name__ == "__main__":
    index = int(sys.argv[1])
    with open(os.path.join("scenarios", "three_satellites", "twolink_downlink_tuples.pickle"), "rb") as f:
        case_tuples = pickle.load(f)
    case_tuple = case_tuples[index]
    run_finish(*case_tuple)
