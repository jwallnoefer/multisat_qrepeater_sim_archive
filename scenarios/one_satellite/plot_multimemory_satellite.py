import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scenarios.three_satellites.common_functions import sat_dist_curved, elevation_curved, eta_atm, eta_dif
from scenarios.three_satellites.common_params import base_params
import rsmf

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

formatter = rsmf.setup(r"\documentclass[twocolumn]{revtex4-2}")


def db(x):
    return 10 * np.log10(x)


def e91_rate(length, divergence_half_angle=3e-6, orbital_height=400e3):
    R_S = 20e6  # 20 MHz repetition rate
    eta_tot = e91_eta(length, divergence_half_angle=divergence_half_angle, orbital_height=orbital_height)
    return R_S * eta_tot


def e91_eta(length, divergence_half_angle=3e-6, orbital_height=400e3):
    eta_det = 0.7
    sat_dist = sat_dist_curved(ground_dist=length / 2, h=orbital_height)
    elevation = elevation_curved(ground_dist=length / 2, h=orbital_height)
    eta_tot = (eta_det**2
               * eta_dif(distance=sat_dist, divergence_half_angle=divergence_half_angle, sender_aperture_radius=0.15, receiver_aperture_radius=0.5, pointing_error_sigma=base_params["POINTING_ERROR_SIGMA"])**2
               * eta_atm(elevation=elevation)**2)
    return eta_tot


result_path = os.path.join("results", "one_satellite_bright_night")
scenario_str = "One Satellite"
fig = formatter.figure(width_ratio=1.0, wide=False)
thetas = {1: 3e-6, 2: 4e-6, 3: 6e-6, 4: 8e-6}
for idx, (i, theta) in enumerate(thetas.items()):
    out_path = os.path.join(result_path, "divergence_theta", str(i))
    try:
        df = pd.read_csv(os.path.join(out_path, "result.csv"), index_col=0)
    except FileNotFoundError:
        continue
    x = df.index / 1000
    y = np.real_if_close(np.array(df["key_per_time"], dtype=complex)) / 2
    plt.scatter(x, y, c=color_list[idx], marker="o", s=1, label=f"theta={theta * 1e6}µrad")

xx = np.linspace(0, 44e5, num=500)
yy = [e91_rate(i) for i in xx]
plt.plot(xx / 1000, yy, linestyle="dashed", color="gray", label="E91 20MHz")
plt.yscale("log")
plt.ylim(1e-2, 1e7)
# plt.legend()
plt.grid()
plt.xlabel("ground distance [km]")
plt.ylabel("key per time [Hz]")
plt.tight_layout()
# plt.title(f"{scenario_str}: T_DP=0.1s, num_memories=1000")
# manager = plt.get_current_fig_manager()
# manager.window.maximize()
# plt.show()
figure_name = "divergence_thetas"
figure_name = figure_name + ".pdf"
plt.savefig(os.path.join(result_path, figure_name))
print(f"Plot saved as {figure_name}")

# memories = {5: 100, 6: 1000}
fig = formatter.figure(width_ratio=1.0, wide=False)
memories = {6: 1000}
dephasing_times = [10e-3, 50e-3, 100e-3, 1.0]
for i, num_memories in memories.items():
    out_path = os.path.join(result_path, "memories", str(i))
    for idx, t_dp in enumerate(dephasing_times):
        output_path = os.path.join(out_path, "%d_t_dp" % int(t_dp * 1000))
        try:
            df = pd.read_csv(os.path.join(output_path, "result.csv"), index_col=0)
        except FileNotFoundError:
            continue
        x = df.index / 1000
        y = np.real_if_close(np.array(df["key_per_time"], dtype=complex)) / 2
        plt.scatter(x, y, marker="o", c=color_list[idx], s=1, label=f"t_dp={t_dp * 1e3}ms")
    xx = np.linspace(0, 44e5, num=500)
    yy = [e91_rate(i) for i in xx]
    plt.plot(xx / 1000, yy, linestyle="dashed", color="gray", label="E91 20MHz")
    plt.yscale("log")
    plt.ylim(1e-4, 1e5)
    # plt.legend()
    plt.grid()
    plt.xlabel("ground distance [km]")
    plt.ylabel("key per time [Hz]")
    plt.tight_layout()
    # plt.title(f"{scenario_str}: theta=3µrad, {num_memories=}")
    # manager = plt.get_current_fig_manager()
    # manager.window.maximize()
    # plt.show()
    figure_name = "dephasing_times"
    figure_name = figure_name + ".pdf"
    plt.savefig(os.path.join(result_path, figure_name))
    print(f"Plot saved as {figure_name}")


# now orbital_heights
fig = formatter.figure(width_ratio=1.0, wide=False)
orbital_heights = [400e3, 600e3, 1000e3, 1500e3, 2000e3]
out_path = os.path.join(result_path, "orbital_heights")
for idx, orbital_height in enumerate(orbital_heights):
    output_path = os.path.join(out_path, "%d_orbital_height" % int(orbital_height / 1000))
    try:
        df = pd.read_csv(os.path.join(output_path, "result.csv"), index_col=0)
    except FileNotFoundError:
        continue
    x = df.index / 1000
    y = np.real_if_close(np.array(df["key_per_time"], dtype=complex)) / 2
    plt.scatter(x, y, marker="o", c=color_list[idx], s=1, label=f"h={orbital_height / 1e3}km")
xx = np.linspace(0, 44e5, num=500)
yy = [e91_rate(i) for i in xx]
plt.plot(xx / 1000, yy, linestyle="dashed", color="gray", label="E91 20MHz")
plt.yscale("log")
plt.ylim(1e-4, 1e5)
# plt.legend()
plt.grid()
plt.xlabel("ground distance [km]")
plt.ylabel("key per time [Hz]")
plt.tight_layout()
# plt.title(f"{scenario_str}: theta=3µrad, {num_memories=}")
# manager = plt.get_current_fig_manager()
# manager.window.maximize()
# plt.show()
figure_name = "orbital_heights"
figure_name = figure_name + ".pdf"
plt.savefig(os.path.join(result_path, figure_name))
print(f"Plot saved as {figure_name}")


# now case 9
# case 9 with changing satellite postitions
fig = formatter.figure(width_ratio=1.0, wide=False)
out_path = os.path.join(result_path, "satellite_path")
orbital_heights = [600e3, 1000e3, 1500e3, 2000e3]
labels = [int(x / 1000) for x in orbital_heights]
for idx, (orbital_height, label) in enumerate(zip(orbital_heights, labels)):
    output_path = os.path.join(out_path, f"{label}_orbital_height")
    df = pd.read_csv(os.path.join(output_path, "result.csv"), index_col=0)
    x = df.index
    y = np.real_if_close(np.array(df["key_per_time"], dtype=complex)) / 2
    plt.scatter(x, y, c=color_list[idx + 1], marker="o", s=1, label=f"orbital_height={label}km")
plt.yscale("log")
# plt.ylim(1e-1, 1e4)
# plt.legend()
plt.xlabel("Offset [ground distance]", color=font_color)
plt.ylabel("Key / time [Hz]", color=font_color)
plt.tight_layout()
plt.grid()
# plt.legend()
figure_name = "configurations"
figure_name = figure_name + ".pdf"
plt.savefig(os.path.join(result_path, figure_name))
print(f"Plot saved as {figure_name}")
