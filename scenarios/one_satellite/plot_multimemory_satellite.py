import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scenarios.three_satellites.common_functions import sat_dist_curved, elevation_curved, eta_atm, eta_dif

# Set color scheme
color_list = [
'#56B4E9',
'#E69F00',
'#CC79A7',
'#F0E442',
'#009E73',
'#D55E00',
'#0072B2',
'#000000'
]

font_color = "#000000"


def db(x):
    return 10 * np.log10(x)


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



result_path = os.path.join("results", "one_satellite")
scenario_str = "One Satellite"
thetas = {1: 2e-6, 2: 4e-6, 3: 6e-6, 4: 8e-6}
for i, theta in thetas.items():
    out_path = os.path.join(result_path, "divergence_theta", str(i))
    try:
        df = pd.read_csv(os.path.join(out_path, "result.csv"), index_col=0)
    except FileNotFoundError:
        continue
    x = df.index / 1000
    y = np.real_if_close(np.array(df["key_per_time"], dtype=complex)) / 2
    plt.scatter(x, y, marker="o", s=10, label=f"theta={theta * 1e6}µrad")

xx = np.linspace(0, 44e5, num=500)
yy = [e91_rate(i) for i in xx]
plt.plot(xx / 1000, yy, linestyle="dashed", color="gray", label="E91 20MHz")
plt.yscale("log")
# plt.ylim(1e-4, 1e5)
plt.legend()
plt.grid()
plt.xlabel("ground distance [km]")
plt.ylabel("key per time [Hz]")
plt.title(f"{scenario_str}: T_DP=0.1s, num_memories=1000")
manager = plt.get_current_fig_manager()
manager.window.maximize()
plt.show()

# memories = {5: 100, 6: 1000}
memories = {6: 1000}
dephasing_times = [10e-3, 50e-3, 100e-3, 1.0]
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
    xx = np.linspace(0, 44e5, num=500)
    yy = [e91_rate(i) for i in xx]
    plt.plot(xx / 1000, yy, linestyle="dashed", color="gray", label="E91 20MHz")
    plt.yscale("log")
    plt.ylim(1e-4, 1e5)
    plt.legend()
    plt.grid()
    plt.xlabel("ground distance [km]")
    plt.ylabel("key per time [Hz]")
    plt.title(f"{scenario_str}: theta=2µrad, {num_memories=}")
    manager = plt.get_current_fig_manager()
    manager.window.maximize()
    plt.show()


# now orbital_heights
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
    plt.scatter(x, y, marker="o", s=10, label=f"h={orbital_height / 1e3}km")
xx = np.linspace(0, 44e5, num=500)
yy = [e91_rate(i) for i in xx]
plt.plot(xx / 1000, yy, linestyle="dashed", color="gray", label="E91 20MHz")
plt.yscale("log")
plt.ylim(1e-4, 1e5)
plt.legend()
plt.grid()
plt.xlabel("ground distance [km]")
plt.ylabel("key per time [Hz]")
plt.title(f"{scenario_str}: theta=2µrad, {num_memories=}")
manager = plt.get_current_fig_manager()
manager.window.maximize()
plt.show()

# now case 9
# case 9 with changing satellite postitions
# fig = formatter.figure(width_ratio=1.0, wide=False)
out_path = os.path.join(result_path, "satellite_path")
orbital_heights = [600e3, 1000e3, 1500e3, 2000e3]
labels = [int(x / 1000) for x in orbital_heights]
for idx, (orbital_height, label) in enumerate(zip(orbital_heights, labels)):
    output_path = os.path.join(out_path, f"{label}_orbital_height")
    df = pd.read_csv(os.path.join(output_path, "result.csv"), index_col=0)
    x = df.index
    y = np.real_if_close(np.array(df["key_per_time"], dtype=complex)) / 2
    plt.scatter(x, y, c=color_list[idx], marker="o", s=1, label=f"orbital_height={label}km")
plt.yscale("log")
# plt.ylim(1e-1, 1e4)
# plt.legend()
plt.xlabel("Offset [ground distance]", color=font_color)
plt.ylabel("Key / time [Hz]", color=font_color)
plt.tight_layout()
plt.grid()
plt.legend()
figure_name = "configurations"
figure_name = figure_name + ".pdf"
plt.savefig(os.path.join(result_path, figure_name))
print(f"Plot saved as {figure_name}")

# result_path = os.path.join("results", "multimemory_satellite_cutoff")
# for cutoff_multiplier in [0.001, 0.005, 0.010, 0.020, 0.030, 0.050, 0.100, 0.250, 0.500]:
#     df = pd.read_csv(os.path.join(result_path, "%.3f_cutoff" % cutoff_multiplier, "result.csv"), index_col=0)
#     x = df.index / 1000
    # y = df["key_per_resource"] / 2
#     yerr = df["key_per_resource_std"] / 2
#     # plt.errorbar(x, y, yerr=yerr, fmt="o", label="cutoff_multiplier=%.3f" % cutoff_multiplier)
#     plt.scatter(x, y, marker="o", s=10, label="cutoff_multiplier=%.3f" % cutoff_multiplier)
#
# xx = np.linspace(0, 4e6, num=500)
# yy = [e91_eta(i) for i in xx]
# plt.plot(xx / 1000, yy, linestyle="dashed", color="gray", label="E91")
#
# plt.yscale("log")
# plt.ylim(1e-9, 1e-2)
# # plt.xlim(0, 300)
# plt.legend()
# plt.xlabel("ground distance [km]")
# plt.ylabel("key per resource")
# plt.title("dephasing time = 7.5s, memory modes = 1000")
# plt.grid()
# plt.savefig(os.path.join(result_path, "per_resource.png"))
# plt.show()
#
# result_path = os.path.join("results", "multimemory_satellite_cutoff")
# for cutoff_multiplier in [0.001, 0.005, 0.010, 0.020, 0.030, 0.050, 0.100, 0.250, 0.500]:
#     df = pd.read_csv(os.path.join(result_path, "%.3f_cutoff" % cutoff_multiplier, "result.csv"), index_col=0)
#     x = sat_dist_curved(df.index / 2, 400e3) / 1000
#     y = df["key_per_resource"] / 2
#     yerr = df["key_per_resource_std"] / 2
#     # plt.errorbar(x, y, yerr=yerr, fmt="o", label="cutoff_multiplier=%.3f" % cutoff_multiplier)
#     plt.scatter(x, y, marker="o", s=10, label="cutoff_multiplier=%.3f" % cutoff_multiplier)
#
# xx = np.linspace(0, 4e6, num=500)
# yy = [e91_eta(i) for i in xx]
# plt.plot(sat_dist_curved(xx / 2, 400e3) / 1000, yy, linestyle="dashed", color="gray", label="E91")
#
# plt.yscale("log")
# # plt.ylim(1e-7, 1e-2)
# # plt.xlim(0, 300)
# plt.legend()
# plt.xlabel("ground-satellite distance [km]")
# plt.ylabel("key per resource")
# plt.title("dephasing time = 7.5s, memory modes = 1000")
# plt.grid()
# plt.show()
#
#
# for cutoff_multiplier in [0.001, 0.005, 0.010, 0.020, 0.030, 0.050, 0.100, 0.250, 0.500]:
#     df = pd.read_csv(os.path.join(result_path, "%.3f_cutoff" % cutoff_multiplier, "result.csv"), index_col=0)
#     x = df.index / 1000
#     y = df["key_per_time"] / 2
#     yerr = df["key_per_time_std"] / 2
#     # plt.errorbar(x, y, yerr=yerr, fmt="o", label="cutoff_multiplier=%.3f" % cutoff_multiplier)
#     plt.scatter(x, y, marker="o", s=10, label="cutoff_multiplier=%.3f" % cutoff_multiplier)
#
# xx = np.linspace(0, 4e6, num=500)
# yy = [e91_rate(i) for i in xx]
# plt.plot(xx / 1000, yy, linestyle="dashed", color="gray", label="E91")
#
# plt.yscale("log")
# # plt.ylim(1e-7, 1e-2)
# # plt.xlim(0, 300)
# plt.legend()
# plt.xlabel("ground distance [km]")
# plt.ylabel("key per time")
# plt.title("dephasing time = 7.5s, memory modes = 1000")
# plt.grid()
# plt.savefig(os.path.join(result_path, "per_time.png"))
# plt.show()
#
# for cutoff_multiplier in [0.001, 0.005, 0.010, 0.020, 0.030, 0.050, 0.100, 0.250, 0.500]:
#     df = pd.read_csv(os.path.join(result_path, "%.3f_cutoff" % cutoff_multiplier, "result.csv"), index_col=0)
#     x = sat_dist_curved(df.index / 2, 400e3) / 1000
#     y = df["key_per_time"] / 2
#     yerr = df["key_per_time_std"] / 2
#     # plt.errorbar(x, y, yerr=yerr, fmt="o", label="cutoff_multiplier=%.3f" % cutoff_multiplier)
#     plt.scatter(x, y, marker="o", s=10, label="cutoff_multiplier=%.3f" % cutoff_multiplier)
#
# plt.plot(sat_dist_curved(xx / 2, 400e3) / 1000, yy, linestyle="dashed", color="gray", label="E91")
#
# plt.yscale("log")
# plt.ylim(1e-4, 1e4)
# # plt.xlim(0, 300)
# plt.legend()
# plt.xlabel("ground-satellite distance [km]")
# plt.ylabel("key per time")
# plt.title("dephasing time = 7.5s, memory modes = 1000")
# plt.grid()
# plt.show()
#
# # ================ DEPHASING PLOT ============
# result_path = os.path.join("results", "multimemory_satellite_dephasing")
# for t_dp in np.logspace(np.log10(100e-3), np.log10(7.5), num=10):
#     df = pd.read_csv(os.path.join(result_path, "%.2f_dephasing" % t_dp, "result.csv"), index_col=0)
#     x = df.index / 1000
#     y = df["key_per_resource"] / 2
#     yerr = df["key_per_resource_std"] / 2
#     # plt.errorbar(x, y, yerr=yerr, fmt="o", label="dephasing_time=%.2f" % t_dp)
#     plt.scatter(x, y, marker="o", s=10, label="dephasing_time=%.2f" % t_dp)
#
# xx = np.linspace(0, 4e6, num=500)
# yy = [e91_eta(i) for i in xx]
# plt.plot(xx / 1000, yy, linestyle="dashed", color="gray", label="E91")
#
# plt.yscale("log")
# plt.ylim(1e-9, 1e-2)
# # plt.xlim(0, 300)
# plt.legend()
# plt.xlabel("ground distance [km]")
# plt.ylabel("key per resource")
# plt.title("memory modes = 1000, cutoff_time = 0.05 * dephasing_time")
# plt.grid()
# plt.savefig(os.path.join(result_path, "per_resource.png"))
# plt.show()
#
# for t_dp in np.logspace(np.log10(100e-3), np.log10(7.5), num=10):
#     df = pd.read_csv(os.path.join(result_path, "%.2f_dephasing" % t_dp, "result.csv"), index_col=0)
#     x = df.index / 1000
#     y = df["key_per_time"] / 2
#     yerr = df["key_per_time_std"] / 2
#     # plt.errorbar(x, y, yerr=yerr, fmt="o", label="dephasing_time=%.2f" % t_dp)
#     plt.scatter(x, y, marker="o", s=10, label="dephasing_time=%.2f" % t_dp)
#
# xx = np.linspace(0, 4e6, num=500)
# yy = [e91_rate(i) for i in xx]
# plt.plot(xx / 1000, yy, linestyle="dashed", color="gray", label="E91")
#
# plt.yscale("log")
# plt.ylim(1e-5, 1e4)
# # plt.xlim(0, 300)
# plt.legend()
# plt.xlabel("ground distance [km]")
# plt.ylabel("key per time")
# plt.title("memory modes = 1000, cutoff_time = 0.05 * dephasing_time")
# plt.grid()
# plt.savefig(os.path.join(result_path, "per_time.png"))
# plt.show()
