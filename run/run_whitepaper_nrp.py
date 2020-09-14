import os, sys; sys.path.insert(0, os.path.abspath("."))
import multiprocessing as mp
from scenarios.NRP_QR_cell import run
from libs.aux_functions import assert_dir, binary_entropy, calculate_keyrate_time, calculate_keyrate_channel_use
import numpy as np
import matplotlib.pyplot as plt

C = 2 * 10**8  # speed of light in optical fiber

result_path = os.path.join("results", "whitepaper_nrp")

params_available_NV = {"P_LINK": 5 * 10**-2,
                       "f_clock": 50 * 10**6,
                       "T_DP": 10 * 10**-3}
params_available_SiV = {"P_LINK": 5 * 10**-2,
                        "f_clock": 30 * 10**6,
                        "T_DP": 1 * 10**-3}
params_available_Qdot = {"P_LINK": 10 * 10**-2,
                         "f_clock": 1000 * 10**6,
                         "T_DP": 0.003 * 10**-3}
params_available_Ca = {"P_LINK": 25 * 10**-2,
                       "f_clock": 0.47 * 10**6,
                       "T_DP": 20 * 10**-3}
params_available_Rb = {"P_LINK": 50 * 10**-2,
                       "f_clock": 5 * 10**6,
                       "T_DP": 100 * 10**-3}
params_future_NV = {"P_LINK": 50 * 10**-2,
                    "f_clock": 250 * 10**6,
                    "T_DP": 10000 * 10**-3}
params_future_SiV = {"P_LINK": 50 * 10**-2,
                     "f_clock": 500 * 10**6,
                     "T_DP": 100 * 10**-3}
params_future_Qdot = {"P_LINK": 60 * 10**-2,
                      "f_clock": 1000 * 10**6,
                      "T_DP": 0.3 * 10**-3}
params_future_Ca = {"P_LINK": 50 * 10**-2,
                    "f_clock": 10 * 10**6,
                    "T_DP": 300 * 10**-3}
params_future_Rb = {"P_LINK": 70 * 10**-2,
                    "f_clock": 10 * 10**6,
                    "T_DP": 1000 * 10**-3}

available_params = [params_available_NV, params_available_SiV, params_available_Qdot, params_available_Ca, params_available_Rb]
future_params = [params_future_NV, params_future_SiV, params_future_Qdot, params_future_Ca, params_future_Rb]
name_list = ["NV", "SiV", "Qdot", "Ca", "Rb"]

def parallel_run(itera, params, length, cutoff_time):
    p = run(length=length, max_iter=itera, params=params, cutoff_time=cutoff_time, mode="sim")
    key_per_time = calculate_keyrate_time(p.correlations_z_list, p.correlations_x_list, 1, p.world.event_queue.current_time + 2 * length / C)
    key_per_resource = calculate_keyrate_channel_use(p.correlations_z_list, p.correlations_x_list, 1, p.resource_cost_max_list)
    return np.array([key_per_time, key_per_resource])

if __name__ == "__main__":
     length_list = np.arange(25000, 425000, 25000)
     iters = [1000]*16
     for name, params in zip(name_list, available_params):
         print(name)
         key_per_time_list = []
         key_per_resource_list = []
         for l in length_list:
            print(l)
            def wrapper(itera):
                return parallel_run(itera, params, l, None)
            with mp.Pool(mp.cpu_count()) as pool:
                rates = pool.map(wrapper, iters)
            av_rates = np.sum(np.array(rates), axis = 0) / 16
            key_per_time_list += [av_rates[0]]
            key_per_resource_list += [av_rates[1]]
         path = os.path.join(result_path, "available", name)
         assert_dir(path)
         np.savetxt(os.path.join(path, "length_list.txt"), length_list)
         np.savetxt(os.path.join(path, "key_per_time_list.txt"), key_per_time_list)
         np.savetxt(os.path.join(path, "key_per_resource_list.txt"), key_per_resource_list)
     for name, params in zip(name_list, future_params):
         print(name)
         key_per_time_list = []
         key_per_resource_list = []
         for l in length_list:
            print(l)
            def wrapper(itera):
                return parallel_run(itera, params, l, None)
            with mp.Pool(mp.cpu_count()) as pool:
                rates = pool.map(wrapper, iters)
            av_rates = np.sum(np.array(rates), axis = 0) / 16
            key_per_time_list += [av_rates[0]]
            key_per_resource_list += [av_rates[1]]
         path = os.path.join(result_path, "future", name)
         assert_dir(path)
         np.savetxt(os.path.join(path, "length_list.txt"), length_list)
         np.savetxt(os.path.join(path, "key_per_time_list.txt"), key_per_time_list)
         np.savetxt(os.path.join(path, "key_per_resource_list.txt"), key_per_resource_list)

   # ### here we plot the Rb lines for different cut-off times
     name = "Rb"
     params = params_available_Rb
     length_list = np.arange(25000, 425000, 25000)
     for m in [200, 500, 1000, 2000, 5000, 10000, 20000]:
         print("m=%d" % m)
         key_per_time_list = []
         key_per_resource_list = []
         for l in length_list:
             print(l)
            def wrapper(itera):
                return parallel_run(itera, params, l, m*trial_time_manual)
            with mp.Pool(mp.cpu_count()) as pool:
                rates = pool.map(wrapper, iters)
            av_rates = np.sum(np.array(rates), axis = 0) / 16
            key_per_time_list += [av_rates[0]]
            key_per_resource_list += [av_rates[1]]
         path = os.path.join(result_path, "available", "m_test", name)
         assert_dir(path)
         np.savetxt(os.path.join(path, "length_list_%d.txt" % m), length_list)
         np.savetxt(os.path.join(path, "key_per_time_list_%d.txt" % m), key_per_time_list)
         np.savetxt(os.path.join(path, "key_per_resource_list_%d.txt" % m), key_per_resource_list)

   # ### further investigate cutoff times - especially the claim that you can set it too low
     ### effect should be very visible if memory quality is very high
    #test_params = {"P_LINK": 10 * 10**-2,
                   #"T_DP": 1,
                   #"T_P": 1 / (50 * 10**6)}
    #length = 22 * 10**3
    #trial_time_manual = 1 / (50 * 10**6)
    #m_list = [m for m in range(1, 41, 2)]
    #cutoff_list = [m * trial_time_manual for m in m_list]
    #key_per_time_list = []
    #key_per_resource_list = []
    #iters = [1000]*16
    #for m, cutoff_time in zip(m_list, cutoff_list):
        #print(m)
        #def wrapper(itera):
            #return parallel_run(itera, test_params, length, cutoff_time)
        #with mp.Pool(mp.cpu_count()) as pool:
            #rates = pool.map(wrapper, iters)
        #av_rates = np.sum(np.array(rates), axis = 0) / 16
        #key_per_time_list += [av_rates[0]]
        #key_per_resource_list += [av_rates[1]]
    #path = os.path.join(result_path, "cutoff_test")
    #assert_dir(path)
    #np.savetxt(os.path.join(path, "m_list.txt"), m_list)
    #np.savetxt(os.path.join(path, "key_per_time_list.txt"), key_per_time_list)
    #np.savetxt(os.path.join(path, "key_per_resource_list.txt"), key_per_resource_list)
    #plt.plot(m_list, key_per_resource_list)
    #plt.grid()
    #plt.show()
