import os, sys; sys.path.insert(0, os.path.abspath("."))
from scenarios.one_satellite.multi_memory_satellite import run
from libs.aux_functions import assert_dir, standard_bipartite_evaluation, save_result
import numpy as np
import matplotlib.pyplot as plt
from time import time
from multiprocessing import Pool
import pandas as pd
import pickle
from consts import SPEED_OF_LIGHT_IN_VACCUM as C

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


if __name__ == "__main__":
    case_number = int(sys.argv[1])
    output_path = os.path.join("results", "one_satellite", "explore")
    assert_dir(output_path)
    if case_number in [1, 2, 3, 4]:
        # Case 1: The divergence plot - x-axis: length, thing we want to vary: thetas
        total_begin_time = time()
        thetas = {1: 2e-6, 2: 4e-6, 3: 6e-6, 4: 8e-6}
        params = dict(base_params)
        params["DIVERGENCE_THETA"] = thetas[case_number]
        params["T_DP"] = 100e-3
        num_memories = 1000
        # cutoff_time = 0.01
        length_list = np.linspace(0, 4400e3, num=96)
        plot_info = {}
        custom_length_lists = {}
        print("%=====================%")
        keys = []
        run_times = []
        for i, length in enumerate(length_list):
            start_time = time()
            print("----------")
            cutoff_time = max(0.01, 4 * length / C)
            p, w = run(length=length, max_iter=1000, params=params, cutoff_time=cutoff_time, num_memories=num_memories, return_world=True)
            key_per_time = standard_bipartite_evaluation(p.data)[2]
            run_time = (time()-start_time)
            print(f"{length=} finished in {run_time:.2f} seconds.")
            print(f"{key_per_time=}")
            print("Event stats:")
            w.event_queue.print_stats()
            keys += [key_per_time]
            run_times +=[run_time]
            if key_per_time < 1e-1:
                break
        plot_info = {}
        plot_info["lengths"] = length_list[:i+1]
        custom_length_list = length_list[:i+1]
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
        plt.savefig(os.path.join(output_path,f"keys_{case_number}.png"))
        plt.close()
        # now plot run_times
        x = plot_info["lengths"]
        y = plot_info["run_times"]
        plt.scatter(x, y, s=10)
        plt.xlabel("Ground distance")
        plt.ylabel("run time [s]")
        plt.grid()
        # plt.legend()
        plt.savefig(os.path.join(output_path,f"run_times_{case_number}.png"))
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
            params["DIVERGENCE_THETA"] = 5e-6
            params["T_DP"] = dephasing_time
            keys = []
            run_times = []
            for i, length in enumerate(length_list):
                start_time = time()
                print("----------")
                cutoff_time = max(0.1 * dephasing_time, 4 * length / C)
                p, w = run(length=length, max_iter=1000, params=params, cutoff_time=cutoff_time, num_memories=num_memories, return_world=True)
                key_per_time = standard_bipartite_evaluation(p.data)[2]
                run_time = (time()-start_time)
                print(f"{length=} finished in {run_time:.2f} seconds.")
                print(f"{key_per_time=}")
                print("Event stats:")
                w.event_queue.print_stats()
                keys += [key_per_time]
                run_times +=[run_time]
                if key_per_time < 1e-1:
                    break
            plot_info[dephasing_time] = {}
            plot_info[dephasing_time]["lengths"] = length_list[:i+1]
            custom_length_lists[dephasing_time] = length_list[:i+1]
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
        plt.savefig(os.path.join(output_path,f"keys_{case_number}.png"))
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
        plt.savefig(os.path.join(output_path,f"run_times_{case_number}.png"))
        plt.close()
    elif case_number == 7:
        # Case 2: The memory quality plot - x-axis: length, thing we want to vary: t_dp
        total_begin_time = time()
        num_memories = 1000
        # cutoff_time = 0.01
        length_list = np.linspace(0, 4400e3, num=96)
        orbital_heights = [400e3, 600e3, 1000e3, 1500e3, 2000e3]
        plot_info = {}
        custom_length_lists = {}
        for orbital_height in orbital_heights:
            print("%=====================%")
            print(f"{orbital_height=}")
            params = dict(base_params)
            params["DIVERGENCE_THETA"] = 2e-6
            params["T_DP"] = 100e-3
            params["ORBITAL_HEIGHT"] = orbital_height
            keys = []
            run_times = []
            for i, length in enumerate(length_list):
                start_time = time()
                print("----------")
                cutoff_time = max(0.01, 4 * length / C)
                p, w = run(length=length, max_iter=1000, params=params, cutoff_time=cutoff_time, num_memories=num_memories, return_world=True)
                key_per_time = standard_bipartite_evaluation(p.data)[2]
                run_time = (time()-start_time)
                print(f"{length=} finished in {run_time:.2f} seconds.")
                print(f"{key_per_time=}")
                print("Event stats:")
                w.event_queue.print_stats()
                keys += [key_per_time]
                run_times +=[run_time]
                if key_per_time < 1e-1:
                    break
            plot_info[orbital_height] = {}
            plot_info[orbital_height]["lengths"] = length_list[:i+1]
            custom_length_lists[orbital_height] = length_list[:i+1]
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
        plt.savefig(os.path.join(output_path,f"keys_{case_number}.png"))
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
        plt.savefig(os.path.join(output_path,f"run_times_{case_number}.png"))
        plt.close()