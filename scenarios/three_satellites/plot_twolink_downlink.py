import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scenarios.one_satellite.multi_memory_satellite import sat_dist_curved, elevation_curved, eta_atm, eta_dif

def e91_rate(length):
    R_S = 20e6  # 20 MHz repetition rate
    eta_tot = e91_eta(length)
    return R_S * eta_tot


def e91_eta(length):
    eta_det = 0.7
    sat_dist = sat_dist_curved(ground_dist=length / 2, h=400e3)
    elevation = elevation_curved(ground_dist=length / 2, h=400e3)
    eta_tot = (eta_det**2
               * eta_dif(distance=sat_dist, divergence_half_angle=10e-6, sender_aperture_radius=0.15, receiver_aperture_radius=0.5)**2
               * eta_atm(elevation=elevation)**2)
    return eta_tot


xx = np.linspace(0, 44e5, num=500)
yy = [e91_rate(i) for i in xx]


result_path = os.path.join("results", "three_satellites", "twolink_downlink")
scenario_str = "3 Satellites, twolink_downlink"
# first satellite positions
out_path = os.path.join(result_path, "sat_positions")
first_satellite_multipliers = np.linspace(0, 0.5, num=9)
for multiplier in first_satellite_multipliers:
    output_path = os.path.join(out_path, "%.3f_first_sat" % multiplier)
    try:
        df = pd.read_csv(os.path.join(output_path, "result.csv"), index_col=0)
    except FileNotFoundError:
        continue
    x = df.index / 1000
    y = np.real_if_close(np.array(df["key_per_time"], dtype=np.complex)) / 2
    plt.scatter(x, y, marker="o", s=10, label=f"first_sat_multiplier={multiplier}")
# compare to one satellite
path = os.path.join("results", "one_satellite", "divergence_theta", "1")
try:
    df = pd.read_csv(os.path.join(path, "result.csv"), index_col=0)
    x = df.index / 1000
    y = np.real_if_close(np.array(df["key_per_time"], dtype=np.complex)) / 2
    plt.scatter(x, y, marker="o", s=10, label="1 Satellite")
except FileNotFoundError:
    pass
plt.plot(xx / 1000, yy, linestyle="dashed", color="gray", label="E91 20MHz")
plt.yscale("log")
plt.ylim(1e-1, 1e5)
plt.legend()
plt.grid()
plt.xlabel("ground distance [km]")
plt.ylabel("key per time [Hz]")
plt.title(f"{scenario_str}: T_DP=0.1s, num_memories=1000, theta=2µrad")
manager = plt.get_current_fig_manager()
manager.window.maximize()
plt.show()

# now different thetas
thetas = {1: 2e-6, 2: 4e-6, 3: 6e-6, 4: 8e-6}
first_satellite_multipliers = [0.000, 0.200, 0.400, 0.500]
for i, theta in thetas.items():
    out_path = os.path.join(result_path, "divergence_theta", str(i))
    for multiplier in first_satellite_multipliers:
        output_path = os.path.join(out_path, "%.3f_first_sat" % multiplier)
        try:
            df = pd.read_csv(os.path.join(output_path, "result.csv"), index_col=0)
        except FileNotFoundError:
            continue
        x = df.index / 1000
        y = np.real_if_close(np.array(df["key_per_time"], dtype=np.complex)) / 2
        plt.scatter(x, y, marker="o", s=10, label=f"first_sat_multiplier={multiplier}")
    # compare to one satellite
    path = os.path.join("results", "one_satellite", "divergence_theta", str(i))
    try:
        df = pd.read_csv(os.path.join(path, "result.csv"), index_col=0)
        x = df.index / 1000
        y = np.real_if_close(np.array(df["key_per_time"], dtype=np.complex)) / 2
        plt.scatter(x, y, marker="o", s=10, label="1 Satellite")
    except FileNotFoundError:
        pass
    plt.plot(xx / 1000, yy, linestyle="dashed", color="gray", label="E91 20MHz")
    plt.yscale("log")
    plt.ylim(1e-1, 1e5)
    plt.legend()
    plt.grid()
    plt.xlabel("ground distance [km]")
    plt.ylabel("key per time [Hz]")
    plt.title(f"{scenario_str}: T_DP=0.1s, num_memories=1000, theta={theta * 1e6}µrad")
    manager = plt.get_current_fig_manager()
    manager.window.maximize()
    plt.show()


memories = {5: 100, 6: 1000}
dephasing_times = [10e-3, 50e-3, 100e-3]
for i, num_memories in memories.items():
    out_path = os.path.join(result_path, "memories", str(i))
    for t_dp in dephasing_times:
        output_path = os.path.join(out_path, "%d_t_dp" % int(t_dp * 1000))
        try:
            df = pd.read_csv(os.path.join(output_path, "result.csv"), index_col=0)
        except FileNotFoundError:
            continue
        x = df.index / 1000
        y = np.real_if_close(np.array(df["key_per_time"], dtype=np.complex)) / 2
        plt.scatter(x, y, marker="o", s=10, label=f"t_dp={t_dp * 1e3}ms")
    # compare to one satellite
    path = os.path.join("results", "one_satellite", "memories", str(i), "100_t_dp")
    try:
        df = pd.read_csv(os.path.join(path, "result.csv"), index_col=0)
        x = df.index / 1000
        y = np.real_if_close(np.array(df["key_per_time"], dtype=np.complex)) / 2
        plt.scatter(x, y, marker="o", s=10, label="1 Satellite, t_dp=100ms")
    except FileNotFoundError:
        pass
    xx = np.linspace(0, 44e5, num=500)
    yy = [e91_rate(i) for i in xx]
    plt.plot(xx / 1000, yy, linestyle="dashed", color="gray", label="E91 20MHz")
    plt.yscale("log")
    plt.ylim(1e-4, 1e5)
    plt.legend()
    plt.grid()
    plt.xlabel("ground distance [km]")
    plt.ylabel("key per time [Hz]")
    plt.title(f"{scenario_str}: theta=2µrad, first_sat_multiplier=0, {num_memories=}")
    manager = plt.get_current_fig_manager()
    manager.window.maximize()
    plt.show()


# result_path = os.path.join("results", "three_satellites", "twolink_downlink")
# for first_satellite_multiplier in np.linspace(0, 0.5, num=9):
#     try:
#         df = pd.read_csv(os.path.join(result_path, "%.3f_first_sat" % first_satellite_multiplier, "result.csv"), index_col=0)
#     except FileNotFoundError:
#         continue
#     x = df.index / 1000
#     y = df["key_per_resource"] / 2
#     yerr = df["key_per_resource_std"] / 2
#     plt.scatter(x, y, marker="o", s=10, label="first_sat_pos=%.3f" % first_satellite_multiplier)
#
# plt.yscale("log")
# plt.ylim(1e-9, 1e-2)
# plt.xlabel("Total ground distance [km]")
# plt.ylabel("Key per resource")
# plt.grid()
# plt.legend()
# plt.show()


# for first_satellite_multiplier in np.linspace(0, 0.5, num=9):
#     try:
#         df = pd.read_csv(os.path.join(result_path, "%.3f_first_sat" % first_satellite_multiplier, "result.csv"), index_col=0)
#     except FileNotFoundError:
#         continue
#     x = df.index / 1000
#     y = df["key_per_time"] / 2
#     yerr = df["key_per_time_std"] / 2
#     plt.scatter(x, y, marker="o", s=10, label="first_sat_pos=%.3f" % first_satellite_multiplier)
#
#
# plt.yscale("log")
# plt.ylim(1e-3, 1e4)
# plt.xlabel("Total ground distance [km]")
# plt.ylabel("Key per time [Hz]")
# plt.grid()
# plt.legend()
# plt.show()
