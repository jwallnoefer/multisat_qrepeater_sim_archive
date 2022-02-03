import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scenarios.three_satellites.common_functions import sat_dist_curved, elevation_curved, eta_atm, eta_dif


def e91_rate(length, divergence_half_angle=2e-6, orbital_height=400e3):
    R_S = 20e6  # 20 MHz repetition rate
    eta_tot = e91_eta(length, divergence_half_angle=divergence_half_angle, orbital_height=orbital_height)
    return R_S * eta_tot


def e91_eta(length, divergence_half_angle=2e-6, orbital_height=400e3):
    eta_det = 0.7
    sat_dist = sat_dist_curved(ground_dist=length / 2, h=orbital_height)
    elevation = elevation_curved(ground_dist=length / 2, h=orbital_height)
    eta_tot = (eta_det**2
               * eta_dif(distance=sat_dist, divergence_half_angle=divergence_half_angle, sender_aperture_radius=0.15, receiver_aperture_radius=0.5)**2
               * eta_atm(elevation=elevation)**2)
    return eta_tot


xx = np.linspace(0, 44e5, num=500)
yy = [e91_rate(i) for i in xx]


result_path = os.path.join("results", "three_satellites", "twolink_downlink")
scenario_str = "3 Satellites, twolink_downlink"
# first satellite positions
out_path = os.path.join(result_path, "sat_positions")
first_satellite_multipliers = np.linspace(0, 0.5, num=6)
for multiplier in first_satellite_multipliers:
    output_path = os.path.join(out_path, "%.3f_first_sat" % multiplier)
    try:
        df = pd.read_csv(os.path.join(output_path, "result.csv"), index_col=0)
    except FileNotFoundError:
        continue
    x = df.index / 1000
    y = np.real_if_close(np.array(df["key_per_time"], dtype=complex)) / 2
    plt.scatter(x, y, marker="o", s=10, label=f"first_sat_multiplier={multiplier:.1f}")
# compare to one satellite
path = os.path.join("results", "one_satellite", "divergence_theta", "1")
try:
    df = pd.read_csv(os.path.join(path, "result.csv"), index_col=0)
    x = df.index / 1000
    y = np.real_if_close(np.array(df["key_per_time"], dtype=complex)) / 2
    plt.scatter(x, y, marker="o", s=10, label="1 Satellite")
except FileNotFoundError:
    pass
plt.plot(xx / 1000, yy, linestyle="dashed", color="gray", label="E91 20MHz")
plt.yscale("log")
plt.ylim(1e0, 1e5)
plt.legend(loc="lower left")
plt.grid()
plt.xlabel("ground distance [km]")
plt.ylabel("key per time [Hz]")
plt.title(f"{scenario_str}: T_DP=0.1s, num_memories=1000, theta=2µrad")
plt.savefig(os.path.join(result_path, "sat_positions.png"))
manager = plt.get_current_fig_manager()
manager.window.maximize()
plt.show()

# now different thetas
# thetas = {1: 2e-6, 2: 4e-6, 3: 6e-6, 4: 8e-6}
thetas = {2: 4e-6, 3: 6e-6, 4: 8e-6}
# first_satellite_multipliers = [0.000, 0.200, 0.400, 0.500]
first_satellite_multipliers = [0.0, 0.2]
for i, theta in thetas.items():
    out_path = os.path.join(result_path, "divergence_theta", str(i))
    for multiplier in first_satellite_multipliers:
        output_path = os.path.join(out_path, "%.3f_first_sat" % multiplier)
        try:
            df = pd.read_csv(os.path.join(output_path, "result.csv"), index_col=0)
        except FileNotFoundError:
            continue
        x = df.index / 1000
        y = np.real_if_close(np.array(df["key_per_time"], dtype=complex)) / 2
        plt.scatter(x, y, marker="o", s=10, label=f"first_sat_multiplier={multiplier}")
    # compare to one satellite
    path = os.path.join("results", "one_satellite", "divergence_theta", str(i))
    try:
        df = pd.read_csv(os.path.join(path, "result.csv"), index_col=0)
        x = df.index / 1000
        y = np.real_if_close(np.array(df["key_per_time"], dtype=complex)) / 2
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
    plt.savefig(os.path.join(result_path, f"divergence_theta{int(theta * 1e6)}.png"))
    manager = plt.get_current_fig_manager()
    manager.window.maximize()
    plt.show()


# memories = {5: 100, 6: 1000}
memories = {6: 1000}
# dephasing_times = [10e-3, 50e-3, 100e-3, 1.0]
dephasing_times = [2e-3, 3e-3, 4e-3, 5e-3, 10e-3, 50e-3, 100e-3, 1.0]
for i, num_memories in memories.items():
    out_path = os.path.join(result_path, "memories", str(i))
    for t_dp in dephasing_times:
        output_path = os.path.join(out_path, "%d_t_dp" % int(t_dp * 1000))
        try:
            df = pd.read_csv(os.path.join(output_path, "result.csv"), index_col=0)
        except FileNotFoundError:
            continue
        x = df.index / 1000
        y = np.real_if_close(np.array(df["key_per_time"], dtype=complex)) / 2
        plt.scatter(x, y, marker="o", s=10, label=f"t_dp={t_dp * 1e3}ms")
    # compare to one satellite
    path = os.path.join("results", "one_satellite", "memories", str(i), "100_t_dp")
    try:
        df = pd.read_csv(os.path.join(path, "result.csv"), index_col=0)
        x = df.index / 1000
        y = np.real_if_close(np.array(df["key_per_time"], dtype=complex)) / 2
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
    plt.title(f"{scenario_str}: theta=5µrad, first_sat_multiplier=0, {num_memories=}")
    plt.savefig(os.path.join(result_path, f"memories_{num_memories}.png"))
    manager = plt.get_current_fig_manager()
    manager.window.maximize()
    plt.show()

# now do the orbital heights plot
orbital_heights = [400e3, 600e3, 1000e3, 1500e3, 2000e3]
out_path = os.path.join(result_path, "orbital_heights")
for orbital_height in orbital_heights:
    output_path = os.path.join(out_path, "%d_orbital_height" % int(orbital_height / 1000))
    try:
        df = pd.read_csv(os.path.join(output_path, "result.csv"), index_col=0)
    except FileNotFoundError:
        continue
    x = df.index / 1000
    y = np.real_if_close(np.array(df["key_per_time"], dtype=complex)) / 2
    # yerr = np.real_if_close(np.array(df["key_per_time_std"], dtype=complex)) / 2
    plt.scatter(x, y, marker="o", s=10, label=f"orbital_height={int(orbital_height / 1000)}km")
    # plt.errorbar(x, y, yerr, marker="o", label=f"orbital_height={int(orbital_height / 1000)}km")
    # # compare to one satellite
    # path = os.path.join("results", "one_satellite", "memories", str(i), "100_t_dp")
    # try:
    #     df = pd.read_csv(os.path.join(path, "result.csv"), index_col=0)
    #     x = df.index / 1000
    #     y = np.real_if_close(np.array(df["key_per_time"], dtype=complex)) / 2
    #     plt.scatter(x, y, marker="o", s=10, label="1 Satellite, t_dp=100ms")
    # except FileNotFoundError:
    #     pass
xx = np.linspace(0, 44e5, num=500)
yy = [e91_rate(i) for i in xx]
plt.plot(xx / 1000, yy, linestyle="dashed", color="gray", label="E91 20MHz")
plt.yscale("log")
plt.ylim(1e-4, 1e5)
plt.legend()
plt.grid()
plt.xlabel("ground distance [km]")
plt.ylabel("key per time [Hz]")
plt.title(f"{scenario_str}: theta=2µrad, first_sat_multiplier=0, num_memories=1000")
plt.savefig(os.path.join(result_path, "orbital_heights.png"))
manager = plt.get_current_fig_manager()
manager.window.maximize()
plt.show()

# unified theta plot
thetas = {2: 4e-6, 3: 6e-6, 4: 8e-6}
first_satellite_multipliers = [0.0, 0.2]
markers = {2: "o", 3: "x", 4: "s"}
colors = ["C0", "C1"]
for i, theta in thetas.items():
    out_path = os.path.join(result_path, "divergence_theta", str(i))
    for multiplier, color in zip(first_satellite_multipliers, colors):
        output_path = os.path.join(out_path, "%.3f_first_sat" % multiplier)
        try:
            df = pd.read_csv(os.path.join(output_path, "result.csv"), index_col=0)
        except FileNotFoundError:
            continue
        x = df.index / 1000
        y = np.real_if_close(np.array(df["key_per_time"], dtype=complex)) / 2
        plt.scatter(x, y, marker=markers[i], s=10, label=f"theta={int(theta*1e6)}µrad, first_sat_multiplier={multiplier}", c=color)
    # compare to one satellite
    path = os.path.join("results", "one_satellite", "divergence_theta", str(i))
    try:
        df = pd.read_csv(os.path.join(path, "result.csv"), index_col=0)
        x = df.index / 1000
        y = np.real_if_close(np.array(df["key_per_time"], dtype=complex)) / 2
        plt.scatter(x, y, marker=markers[i], s=10, label=f"1 Satellite, {int(theta*1e6)}µrad", c="C2")
    except FileNotFoundError:
        pass
    plt.plot(xx / 1000, [e91_rate(i, divergence_half_angle=theta) for i in xx], linestyle="dashed", color="gray", label="E91 20MHz")
plt.yscale("log")
plt.ylim(1e-1, 1e5)
plt.legend()
plt.grid()
plt.xlabel("ground distance [km]")
plt.ylabel("key per time [Hz]")
plt.title(f"{scenario_str}: T_DP=0.1s, num_memories=1000")
plt.savefig(os.path.join(result_path, "divergence_thetas.png"))
manager = plt.get_current_fig_manager()
manager.window.maximize()
plt.show()

# cutoff_time plot
cutoff_multipliers = [None, 1.0, 0.75, 0.5, 0.1, 0.05, 0.02]
out_path = os.path.join(result_path, "cutoff_times")
for cutoff_multiplier in cutoff_multipliers:
    try:
        dir_prefix = "%d" % int(cutoff_multiplier * 100)
    except TypeError as e:
        if cutoff_multiplier is None:
            dir_prefix = "None"
        else:
            raise e
    output_path = os.path.join(out_path, dir_prefix + "_cutoff_multiplier")
    try:
        df = pd.read_csv(os.path.join(output_path, "result.csv"), index_col=0)
    except FileNotFoundError:
        continue
    x = df.index / 1000
    y = np.real_if_close(np.array(df["key_per_time"], dtype=complex)) / 2
    plt.scatter(x, y, marker="o", s=10, label=f"{cutoff_multiplier=}")
xx = np.linspace(0, 44e5, num=500)
yy = [e91_rate(i) for i in xx]
plt.plot(xx / 1000, yy, linestyle="dashed", color="gray", label="E91 20MHz")
plt.yscale("log")
plt.ylim(1e-2, 1e5)
plt.legend()
plt.grid()
plt.xlabel("ground distance [km]")
plt.ylabel("key per time [Hz]")
plt.title(f"{scenario_str}: theta=5µrad, first_sat_multiplier=0, num_memories=1000")
plt.savefig(os.path.join(result_path, "cutoff_multipliers.png"))
plt.show()
