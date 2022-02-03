import os, sys; sys.path.insert(0, os.path.abspath("."))
from scenarios.one_satellite.multi_memory_satellite import run
from scenarios.three_satellites.common_params import base_params
from libs.aux_functions import assert_dir, standard_bipartite_evaluation
import numpy as np
import matplotlib.pyplot as plt
from time import time
import pickle
from consts import SPEED_OF_LIGHT_IN_VACCUM as C


if __name__ == "__main__":
    case_number = int(sys.argv[1])
    output_path = os.path.join("results", "one_satellite", "explore")
    assert_dir(output_path)
    if case_number in [1, 2, 3, 4]:
        # Case 1: The divergence plot - x-axis: length, thing we want to vary: thetas
        total_begin_time = time()
        thetas = {1: 3e-6, 2: 4e-6, 3: 6e-6, 4: 8e-6}
        params = dict(base_params)
        params["DIVERGENCE_THETA"] = thetas[case_number]
        num_memories = 1000
        cutoff_multiplier = 0.1
        min_cutoff_time = cutoff_multiplier * params["T_DP"]
        length_list = np.linspace(0, 4400e3, num=96)
        plot_info = {}
        custom_length_lists = {}
        print("%=====================%")
        keys = []
        run_times = []
        for i, length in enumerate(length_list):
            start_time = time()
            print("----------")
            cutoff_time = max(min_cutoff_time, 4 * length / C)
            p, w = run(length=length, max_iter=1000, params=params, cutoff_time=cutoff_time, num_memories=num_memories, return_world=True)
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
        plot_info = {}
        plot_info["lengths"] = length_list[:i + 1]
        custom_length_list = length_list[:i + 1]
        plot_info["keys"] = keys
        plot_info["run_times"] = run_times
        # save custom length_lists
        with open(os.path.join(output_path, f"custom_lengths_{case_number}.pickle"), "wb") as f:
            pickle.dump(custom_length_list, f)
        print("%=======================================%")
        print(f"The whole case {case_number} finished exploring in {(time() - total_begin_time) / 60:.2f} minutes.")
        # now plot keys
        x = plot_info["lengths"]
        y = plot_info["keys"]
        plt.scatter(x, y, s=10)
        plt.yscale("log")
        plt.xlabel("Ground distance")
        plt.ylabel("Key rate")
        plt.grid()
        # plt.legend()
        plt.savefig(os.path.join(output_path, f"keys_{case_number}.png"))
        plt.close()
        # now plot run_times
        x = plot_info["lengths"]
        y = plot_info["run_times"]
        plt.scatter(x, y, s=10)
        plt.xlabel("Ground distance")
        plt.ylabel("run time [s]")
        plt.grid()
        # plt.legend()
        plt.savefig(os.path.join(output_path, f"run_times_{case_number}.png"))
        plt.close()
    elif case_number in [6]:
        # Case 2: The memory quality plot - x-axis: length, thing we want to vary: t_dp
        total_begin_time = time()
        memories = {5: 100, 6: 1000}
        num_memories = memories[case_number]
        # cutoff_time = 0.01
        length_list = np.linspace(0, 4400e3, num=96)
        dephasing_times = [10e-3, 50e-3, 100e-3, 1.0]
        plot_info = {}
        custom_length_lists = {}
        for dephasing_time in dephasing_times:
            print("%=====================%")
            print(f"{dephasing_time=}")
            params = dict(base_params)
            params["T_DP"] = dephasing_time
            cutoff_multiplier = 0.1
            min_cutoff_time = cutoff_multiplier * params["T_DP"]
            keys = []
            run_times = []
            for i, length in enumerate(length_list):
                start_time = time()
                print("----------")
                cutoff_time = max(min_cutoff_time, 4 * length / C)
                p, w = run(length=length, max_iter=1000, params=params, cutoff_time=cutoff_time, num_memories=num_memories, return_world=True)
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
        cutoff_multiplier = 0.1
        length_list = np.linspace(0, 8800e3, num=96)
        orbital_heights = [400e3, 600e3, 1000e3, 1500e3, 2000e3]
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
                p, w = run(length=length, max_iter=1000, params=params, cutoff_time=cutoff_time, num_memories=num_memories, return_world=True)
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
    elif case_number == 9:
        total_begin_time = time()
        length = 4400e3
        cutoff_multiplier = 0.1
        num_memories = 1000
        orbital_heights = [400e3, 600e3, 1000e3, 1500e3, 2000e3]
        labels = [int(x / 1000) for x in orbital_heights]
        plot_info = {}
        custom_variations = {}
        variations = np.linspace(0, 0.4, num=96 // 2 + 1)
        multipliers = [0.5 + x for x in variations]
        for orbital_height, label in zip(orbital_heights, labels):
            print("%=====================%")
            print(f"orbital height={label}km")
            keys = []
            run_times = []
            params = dict(base_params)
            params["ORBITAL_HEIGHT"] = orbital_height
            min_cutoff_time = cutoff_multiplier * params["T_DP"]
            cutoff_time = max(min_cutoff_time, 4 * length / C)
            for i, position_multiplier in enumerate(multipliers):
                start_time = time()
                print("----------")
                p, w = run(length=length, max_iter=100, params=params,
                           cutoff_time=cutoff_time, num_memories=num_memories,
                           position_multiplier=position_multiplier,
                           return_world=True)
                key_per_time = standard_bipartite_evaluation(p.data)[2]
                run_time = (time() - start_time)
                print(f"{position_multiplier=} finished in {run_time:.2f} seconds.")
                print(f"{key_per_time=}")
                print("Event stats:")
                w.event_queue.print_stats()
                keys += [key_per_time]
                run_times += [run_time]
                if key_per_time < 1e-1:
                    break
            plot_info[orbital_height] = {}
            plot_info[orbital_height]["variations"] = variations[:i + 1]
            used_variations = variations[:i + 1]
            custom_variations[orbital_height] = np.concatenate([-np.flip(used_variations[1:]), used_variations])
            plot_info[orbital_height]["keys"] = keys
            plot_info[orbital_height]["run_times"] = run_times
        # save custom length_lists
        with open(os.path.join(output_path, f"custom_variations_{case_number}.pickle"), "wb") as f:
            pickle.dump(custom_variations, f)
        print("%=======================================%")
        print(f"The whole case {case_number} finished exploring in {(time() - total_begin_time) / 60:.2f} minutes.")
        # now plot keys
        for orbital_heights, label in zip(orbital_height, labels):
            x = plot_info[orbital_height]["variations"]
            y = plot_info[orbital_height]["keys"]
            plt.scatter(x, y, s=10, label=f"orbital_height={label}km")
        plt.yscale("log")
        plt.xlabel("offset")
        plt.ylabel("Key rate")
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(output_path, f"keys_{case_number}.png"))
        plt.close()
        # now plot run_times
        for orbital_height, label in zip(orbital_height, labels):
            x = plot_info[orbital_height]["variations"]
            y = plot_info[orbital_height]["run_times"]
            plt.scatter(x, y, s=10, label=f"orbital_height={label}km")
        plt.xlabel("offset")
        plt.ylabel("run time [s]")
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(output_path, f"run_times_{case_number}.png"))
        plt.close()
