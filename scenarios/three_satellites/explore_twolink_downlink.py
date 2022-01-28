import os, sys; sys.path.insert(0, os.path.abspath("."))
from scenarios.three_satellites.twolink_downlink import run
from scenarios.three_satellites.common_params import base_params
from libs.aux_functions import assert_dir, standard_bipartite_evaluation, save_result
import numpy as np
import matplotlib.pyplot as plt
from time import time
import pickle
from consts import SPEED_OF_LIGHT_IN_VACCUM as C


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

if __name__ == "__main__":
    case_number = int(sys.argv[1])
    output_path = os.path.join("results", "three_satellites", "twolink_downlink", "explore")
    assert_dir(output_path)
    if case_number == 0:
        # Case 0: The big plot - x-axis: length, thing we want to vary: satellite position
        total_begin_time = time()
        params = dict(base_params)
        num_memories = 1000
        # cutoff_time = 0.01
        length_list = np.linspace(0, 8800e3, num=96)
        first_satellite_multipliers = np.linspace(0, 0.5, num=6)
        cutoff_multiplier = 0.1
        min_cutoff_time = cutoff_multiplier * params["T_DP"]
        plot_info = {}
        custom_length_lists = {}
        for satellite_multiplier in first_satellite_multipliers:
            print("%=====================%")
            print(f"{satellite_multiplier=}")
            keys = []
            run_times = []
            for i, length in enumerate(length_list):
                start_time = time()
                print("----------")
                cutoff_time = max(min_cutoff_time, 4 * length / C)
                p, w = run(length=length, max_iter=1000, params=params, cutoff_time=cutoff_time, num_memories=num_memories, first_satellite_ground_dist_multiplier=satellite_multiplier, return_world=True)
                key_per_time = standard_bipartite_evaluation(p.data)[2]
                run_time = (time() - start_time)
                print(f"{length=} finished in {run_time:.2f} seconds.")
                print(f"{key_per_time=}")
                print("Event stats:")
                w.event_queue.print_stats()
                keys += [key_per_time]
                run_times += [run_time]
                if key_per_time < 1e-1:
                    break
            plot_info[satellite_multiplier] = {}
            plot_info[satellite_multiplier]["lengths"] = length_list[:i + 1]
            custom_length_lists[satellite_multiplier] = length_list[:i + 1]
            plot_info[satellite_multiplier]["keys"] = keys
            plot_info[satellite_multiplier]["run_times"] = run_times
        # save custom length_lists
        with open(os.path.join(output_path, f"custom_lengths_{case_number}.pickle"), "wb") as f:
            pickle.dump(custom_length_lists, f)
        print("%=======================================%")
        print(f"The whole case {case_number} finished exploring in {(time() - total_begin_time) / 60:.2f} minutes.")
        # now plot keys
        for satellite_multiplier in first_satellite_multipliers:
            x = plot_info[satellite_multiplier]["lengths"]
            y = plot_info[satellite_multiplier]["keys"]
            plt.scatter(x, y, s=10, label=f"{satellite_multiplier=}")
        plt.yscale("log")
        plt.xlabel("Ground distance")
        plt.ylabel("Key rate")
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(output_path, f"keys_{case_number}.png"))
        plt.close()
        # now plot run_times
        for satellite_multiplier in first_satellite_multipliers:
            x = plot_info[satellite_multiplier]["lengths"]
            y = plot_info[satellite_multiplier]["run_times"]
            plt.scatter(x, y, s=10, label=f"{satellite_multiplier=}")
        plt.xlabel("Ground distance")
        plt.ylabel("run time [s]")
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(output_path, f"run_times_{case_number}.png"))
        plt.close()
    elif case_number in [2, 3, 4]:
        total_begin_time = time()
        thetas = {1: 2e-6, 2: 4e-6, 3: 6e-6, 4: 8e-6}
        params = dict(base_params)
        params["DIVERGENCE_THETA"] = thetas[case_number]
        num_memories = 1000
        length_list = np.linspace(0, 8800e3, num=96)
        cutoff_multiplier = 0.1
        min_cutoff_time = cutoff_multiplier * params["T_DP"]
        first_satellite_multipliers = [0.0, 0.2]
        plot_info = {}
        custom_length_lists = {}
        for satellite_multiplier in first_satellite_multipliers:
            print("%=====================%")
            print(f"{satellite_multiplier=}")
            keys = []
            run_times = []
            for i, length in enumerate(length_list):
                start_time = time()
                print("----------")
                cutoff_time = max(min_cutoff_time, 4 * length / C)
                p, w = run(length=length, max_iter=1000, params=params, cutoff_time=cutoff_time, num_memories=num_memories, first_satellite_ground_dist_multiplier=satellite_multiplier, return_world=True)
                key_per_time = standard_bipartite_evaluation(p.data)[2]
                run_time = (time() - start_time)
                print(f"{length=} finished in {run_time:.2f} seconds.")
                print(f"{key_per_time=}")
                print("Event stats:")
                w.event_queue.print_stats()
                keys += [key_per_time]
                run_times += [run_time]
                if key_per_time < 1e-1:
                    break
            plot_info[satellite_multiplier] = {}
            plot_info[satellite_multiplier]["lengths"] = length_list[:i + 1]
            custom_length_lists[satellite_multiplier] = length_list[:i + 1]
            plot_info[satellite_multiplier]["keys"] = keys
            plot_info[satellite_multiplier]["run_times"] = run_times
        # save custom length_lists
        with open(os.path.join(output_path, f"custom_lengths_{case_number}.pickle"), "wb") as f:
            pickle.dump(custom_length_lists, f)
        print("%=======================================%")
        print(f"The whole case {case_number} finished exploring in {(time() - total_begin_time) / 60:.2f} minutes.")
        # now plot keys
        for satellite_multiplier in first_satellite_multipliers:
            x = plot_info[satellite_multiplier]["lengths"]
            y = plot_info[satellite_multiplier]["keys"]
            plt.scatter(x, y, s=10, label=f"{satellite_multiplier=}")
        plt.yscale("log")
        plt.xlabel("Ground distance")
        plt.ylabel("Key rate")
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(output_path, f"keys_{case_number}.png"))
        plt.close()
        # now plot run_times
        for satellite_multiplier in first_satellite_multipliers:
            x = plot_info[satellite_multiplier]["lengths"]
            y = plot_info[satellite_multiplier]["run_times"]
            plt.scatter(x, y, s=10, label=f"{satellite_multiplier=}")
        plt.xlabel("Ground distance")
        plt.ylabel("run time [s]")
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(output_path, f"run_times_{case_number}.png"))
        plt.close()
    elif case_number in [6]:
        # Case 2: The memory quality plot - x-axis: length, thing we want to vary: t_dp
        total_begin_time = time()
        memories = {5: 100, 6: 1000}
        num_memories = memories[case_number]
        # cutoff_time = 0.01
        length_list = np.linspace(0, 8800e3, num=96)
        satellite_multiplier = 0.0
        dephasing_times = [2e-3, 3e-3, 4e-3, 5e-3, 10e-3, 50e-3, 100e-3, 1.0]
        cutoff_multiplier = 0.1
        plot_info = {}
        custom_length_lists = {}
        for dephasing_time in dephasing_times:
            print("%=====================%")
            print(f"{dephasing_time=}")
            params = dict(base_params)
            params["T_DP"] = dephasing_time
            min_cutoff_time = cutoff_multiplier * params["T_DP"]
            keys = []
            run_times = []
            for i, length in enumerate(length_list):
                start_time = time()
                print("----------")
                cutoff_time = max(min_cutoff_time, 4 * length / C)
                p, w = run(length=length, max_iter=1000, params=params, cutoff_time=cutoff_time, num_memories=num_memories, first_satellite_ground_dist_multiplier=satellite_multiplier, return_world=True)
                key_per_time = standard_bipartite_evaluation(p.data)[2]
                run_time = (time() - start_time)
                print(f"{length=} finished in {run_time:.2f} seconds.")
                print(f"{key_per_time=}")
                print("Event stats:")
                w.event_queue.print_stats()
                keys += [key_per_time]
                run_times += [run_time]
                if key_per_time < 1e-1:
                    break
            plot_info[dephasing_time] = {}
            plot_info[dephasing_time]["lengths"] = length_list[:i + 1]
            custom_length_lists[dephasing_time] = length_list[:i + 1]
            plot_info[dephasing_time]["keys"] = keys
            plot_info[dephasing_time]["run_times"] = run_times
        # save custom length_lists
        with open(os.path.join(output_path, f"custom_lengths_{case_number}.pickle"), "wb") as f:
            pickle.dump(custom_length_lists, f)
        print("%=======================================%")
        print(f"The whole case {case_number} finished exploring in {(time() - total_begin_time) / 60:.2f} minutes.")
        # now plot keys
        for dephasing_time in dephasing_times:
            x = plot_info[dephasing_time]["lengths"]
            y = plot_info[dephasing_time]["keys"]
            plt.scatter(x, y, s=10, label=f"{dephasing_time=}")
        plt.yscale("log")
        plt.xlabel("Ground distance")
        plt.ylabel("Key rate")
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(output_path, f"keys_{case_number}.png"))
        plt.close()
        # now plot run_times
        for dephasing_time in dephasing_times:
            x = plot_info[dephasing_time]["lengths"]
            y = plot_info[dephasing_time]["run_times"]
            plt.scatter(x, y, s=10, label=f"{dephasing_time=}")
        plt.xlabel("Ground distance")
        plt.ylabel("run time [s]")
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(output_path, f"run_times_{case_number}.png"))
        plt.close()
    elif case_number == 7:
        # Case 2: The memory quality plot - x-axis: length, thing we want to vary: t_dp
        total_begin_time = time()
        num_memories = 1000
        # cutoff_time = 0.01
        length_list = np.linspace(0, 8800e3, num=96)
        satellite_multiplier = 0.0
        orbital_heights = [400e3, 600e3, 1000e3, 1500e3, 2000e3]
        cutoff_multiplier = 0.1
        plot_info = {}
        custom_length_lists = {}
        for orbital_height in orbital_heights:
            print("%=====================%")
            print(f"{orbital_height=}")
            params = dict(base_params)
            params["ORBITAL_HEIGHT"] = orbital_height
            min_cutoff_time = cutoff_multiplier * params["T_DP"]
            keys = []
            run_times = []
            for i, length in enumerate(length_list):
                start_time = time()
                print("----------")
                cutoff_time = max(min_cutoff_time, 4 * length / C)
                p, w = run(length=length, max_iter=1000, params=params, cutoff_time=cutoff_time, num_memories=num_memories, first_satellite_ground_dist_multiplier=satellite_multiplier, return_world=True)
                key_per_time = standard_bipartite_evaluation(p.data)[2]
                run_time = (time() - start_time)
                print(f"{length=} finished in {run_time:.2f} seconds.")
                print(f"{key_per_time=}")
                print("Event stats:")
                w.event_queue.print_stats()
                keys += [key_per_time]
                run_times += [run_time]
                if key_per_time < 1e-1:
                    break
            plot_info[orbital_height] = {}
            plot_info[orbital_height]["lengths"] = length_list[:i + 1]
            custom_length_lists[orbital_height] = length_list[:i + 1]
            plot_info[orbital_height]["keys"] = keys
            plot_info[orbital_height]["run_times"] = run_times
        # save custom length_lists
        with open(os.path.join(output_path, f"custom_lengths_{case_number}.pickle"), "wb") as f:
            pickle.dump(custom_length_lists, f)
        print("%=======================================%")
        print(f"The whole case {case_number} finished exploring in {(time() - total_begin_time) / 60:.2f} minutes.")
        # now plot keys
        for orbital_height in orbital_heights:
            x = plot_info[orbital_height]["lengths"]
            y = plot_info[orbital_height]["keys"]
            plt.scatter(x, y, s=10, label=f"{orbital_height=}")
        plt.yscale("log")
        plt.xlabel("Ground distance")
        plt.ylabel("Key rate")
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(output_path, f"keys_{case_number}.png"))
        plt.close()
        # now plot run_times
        for orbital_height in orbital_heights:
            x = plot_info[orbital_height]["lengths"]
            y = plot_info[orbital_height]["run_times"]
            plt.scatter(x, y, s=10, label=f"{orbital_height=}")
        plt.xlabel("Ground distance")
        plt.ylabel("run time [s]")
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(output_path, f"run_times_{case_number}.png"))
        plt.close()
    elif case_number == 8:
        # varying cutoff_times to prove they are
        total_begin_time = time()
        num_memories = 1000
        # cutoff_time = 0.01
        length_list = np.linspace(0, 8800e3, num=96)
        satellite_multiplier = 0.0
        cutoff_multipliers = [None, 0.5, 0.2, 0.1, 0.05, 0.02]
        plot_info = {}
        custom_length_lists = {}
        for cutoff_multiplier in cutoff_multipliers:
            print("%=====================%")
            print(f"{cutoff_multiplier=}")
            params = dict(base_params)
            params["DIVERGENCE_THETA"] = 5e-6
            try:
                cutoff_base_time = cutoff_multiplier * params["T_DP"]
            except TypeError as e:
                if cutoff_multiplier is None:
                    cutoff_base_time = None
                else:
                    raise e
            keys = []
            run_times = []
            for i, length in enumerate(length_list):
                start_time = time()
                print("----------")
                # cutoff_time = max(cutoff_base_time, 4 * length / C)
                cutoff_time = cutoff_base_time
                p, w = run(length=length, max_iter=1000, params=params, cutoff_time=cutoff_time, num_memories=num_memories, first_satellite_ground_dist_multiplier=satellite_multiplier, return_world=True)
                key_per_time = standard_bipartite_evaluation(p.data)[2]
                run_time = (time() - start_time)
                print(f"{length=} finished in {run_time:.2f} seconds.")
                print(f"{key_per_time=}")
                print("Event stats:")
                w.event_queue.print_stats()
                keys += [key_per_time]
                run_times += [run_time]
                if key_per_time < 1e-1:
                    break
            plot_info[cutoff_multiplier] = {}
            plot_info[cutoff_multiplier]["lengths"] = length_list[:i + 1]
            custom_length_lists[cutoff_multiplier] = length_list[:i + 1]
            plot_info[cutoff_multiplier]["keys"] = keys
            plot_info[cutoff_multiplier]["run_times"] = run_times
        # save custom length_lists
        with open(os.path.join(output_path, f"custom_lengths_{case_number}.pickle"), "wb") as f:
            pickle.dump(custom_length_lists, f)
        print("%=======================================%")
        print(f"The whole case {case_number} finished exploring in {(time() - total_begin_time) / 60:.2f} minutes.")
        # now plot keys
        for cutoff_multiplier in cutoff_multipliers:
            x = plot_info[cutoff_multiplier]["lengths"]
            y = plot_info[cutoff_multiplier]["keys"]
            plt.scatter(x, y, s=10, label=f"{cutoff_multiplier=}")
        plt.yscale("log")
        plt.xlabel("Ground distance")
        plt.ylabel("Key rate")
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(output_path, f"keys_{case_number}.png"))
        plt.close()
        # now plot run_times
        for cutoff_multiplier in cutoff_multipliers:
            x = plot_info[cutoff_multiplier]["lengths"]
            y = plot_info[cutoff_multiplier]["run_times"]
            plt.scatter(x, y, s=10, label=f"{cutoff_multiplier=}")
        plt.xlabel("Ground distance")
        plt.ylabel("run time [s]")
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(output_path, f"run_times_{case_number}.png"))
        plt.close()
    elif case_number == 9:
        total_begin_time = time()
        params = dict(base_params)
        length = 4400e3
        cutoff_multiplier = 0.1
        min_cutoff_time = cutoff_multiplier * params["T_DP"]
        cutoff_time = max(min_cutoff_time, 4 * length / C)
        num_memories = 1000
        configurations = [np.array([-0.1, 0.5, 1.1]), np.array([0, 0.5, 1]),
                          np.array([0.1, 0.5, 0.9]), np.array([0.2, 0.5, 0.8])]
        labels = [str(int(base_multipliers[0] * 10)) for base_multipliers in configurations]
        plot_info = {}
        custom_variations = {}
        variations = np.linspace(0, 0.4, num=96 // 2 + 1)
        for base_multipliers, label in zip(configurations, labels):
            print("%=====================%")
            print(f"configuration={label}")
            keys = []
            run_times = []
            multipliers = [base_multipliers + x for x in variations]
            for satellite_multipliers in multipliers:
                start_time = time()
                print("----------")
                p, w = run(length=length, max_iter=1000, params=params,
                           cutoff_time=cutoff_time, num_memories=num_memories,
                           satellite_multipliers=satellite_multipliers,
                           return_world=True)
                key_per_time = standard_bipartite_evaluation(p.data)[2]
                run_time = (time() - start_time)
                print(f"{satellite_multipliers=} finished in {run_time:.2f} seconds.")
                print(f"{key_per_time=}")
                print("Event stats:")
                w.event_queue.print_stats()
                keys += [key_per_time]
                run_times += [run_time]
                if key_per_time < 1e-1:
                    break
            plot_info[label] = {}
            plot_info[label]["variations"] = variations[:i + 1]
            custom_variations[label] = variations[:i + 1]
            plot_info[label]["keys"] = keys
            plot_info[label]["run_times"] = run_times
        # save custom length_lists
        with open(os.path.join(output_path, f"custom_variations_{case_number}.pickle"), "wb") as f:
            pickle.dump(custom_variations, f)
        print("%=======================================%")
        print(f"The whole case {case_number} finished exploring in {(time() - total_begin_time) / 60:.2f} minutes.")
        # now plot keys
        for base_multipliers, label in zip(configurations, labels):
            x = plot_info[label]["variations"]
            y = plot_info[label]["keys"]
            plt.scatter(x, y, s=10, label=f"configuration={label}")
        plt.yscale("log")
        plt.xlabel("offset")
        plt.ylabel("Key rate")
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(output_path, f"keys_{case_number}.png"))
        plt.close()
        # now plot run_times
        for base_multipliers, label in zip(configurations, labels):
            x = plot_info[label]["variations"]
            y = plot_info[label]["run_times"]
            plt.scatter(x, y, s=10, label=f"configuration={label}")
        plt.xlabel("offset")
        plt.ylabel("run time [s]")
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(output_path, f"run_times_{case_number}.png"))
        plt.close()
