import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt
import rsmf
import pandas as pd
from scenarios.three_satellites.common_functions import sat_dist_curved, elevation_curved, eta_atm, eta_dif
from scenarios.three_satellites.common_params import base_params
from libs.aux_functions import distance
from consts import AVERAGE_EARTH_RADIUS as R_E
from consts import EARTH_MASS as M
from consts import GRAVITATIONAL_CONSTANT as G

project_title = "satellite_repeater"
project_title = project_title + ".tex"
tex_source_path = os.path.join("manuscript")
# formatter = rsmf.setup(os.path.join(tex_source_path, project_title))#
formatter = rsmf.setup(r"\documentclass[twocolumn]{revtex4-2}")

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

font_color = '#000000'


def e91_rate(length, divergence_half_angle=base_params["DIVERGENCE_THETA"], orbital_height=base_params["ORBITAL_HEIGHT"]):
    R_S = 20e6  # 20 MHz repetition rate
    eta_tot = e91_eta(length, divergence_half_angle=divergence_half_angle, orbital_height=orbital_height)
    return R_S * eta_tot


def e91_eta(length, divergence_half_angle=base_params["DIVERGENCE_THETA"], orbital_height=base_params["ORBITAL_HEIGHT"]):
    eta_det = 0.7
    sat_dist = sat_dist_curved(ground_dist=length / 2, h=orbital_height)
    elevation = elevation_curved(ground_dist=length / 2, h=orbital_height)
    eta_tot = (eta_det**2
               * eta_dif(distance=sat_dist, divergence_half_angle=divergence_half_angle, sender_aperture_radius=0.15, receiver_aperture_radius=0.5, pointing_error_sigma=base_params["POINTING_ERROR_SIGMA"])**2
               * eta_atm(elevation=elevation)**2)
    return eta_tot


def position_from_angle(radius, angle):
    return radius * np.array([np.sin(angle), np.cos(angle)])


def e91_eta_pos(length, divergence_half_angle=base_params["DIVERGENCE_THETA"], orbital_height=base_params["ORBITAL_HEIGHT"], satellite_position=0.5):
    eta_det = 0.7
    station_a_pos = position_from_angle(radius=R_E, angle=0)
    station_b_pos = position_from_angle(radius=R_E, angle=length / R_E)
    station_sat_pos = position_from_angle(radius=R_E + orbital_height,
                                          angle=satellite_position * length / R_E)
    elevation_left = elevation_curved(ground_dist=satellite_position * length,
                                      h=orbital_height)
    elevation_right = elevation_curved(ground_dist=(1 - satellite_position) * length,
                                       h=orbital_height)
    eta_left = (eta_det
                * eta_dif(distance=distance(station_a_pos, station_sat_pos),
                          divergence_half_angle=divergence_half_angle,
                          sender_aperture_radius=0.15,
                          receiver_aperture_radius=0.5,
                          pointing_error_sigma=base_params["POINTING_ERROR_SIGMA"])
                * eta_atm(elevation_left)
                )
    eta_right = (eta_det
                 * eta_dif(distance=distance(station_b_pos, station_sat_pos),
                           divergence_half_angle=divergence_half_angle,
                           sender_aperture_radius=0.15,
                           receiver_aperture_radius=0.5,
                           pointing_error_sigma=base_params["POINTING_ERROR_SIGMA"])
                 * eta_atm(elevation_right)
                 )
    return eta_left * eta_right


def e91_rate_pos(length, divergence_half_angle=base_params["DIVERGENCE_THETA"], orbital_height=base_params["ORBITAL_HEIGHT"], satellite_position=0.5):
    R_S = 20e6  # 20 MHz repetition rate
    eta_tot = e91_eta_pos(length, divergence_half_angle=divergence_half_angle,
                          orbital_height=orbital_height,
                          satellite_position=satellite_position
                          )
    return R_S * eta_tot


xx = np.linspace(0, 44e5, num=10)
yy = [e91_rate(i) for i in xx]

result_path = os.path.join("results", "three_satellites", "twolink_downlink")
scenario_str = "3 Satellites, twolink_downlink"

# # first satellite positions
# # rsmf format figure
# fig = formatter.figure(width_ratio=1.0, wide=False)
# # begin... satellite postitions
# out_path = os.path.join(result_path, "sat_positions")
# first_satellite_multipliers = np.linspace(0, 0.5, num=6)
# plt.plot(xx / 1000, yy, linestyle="dashed", color="gray", zorder=0)
# for index, multiplier in enumerate(first_satellite_multipliers):
#     output_path = os.path.join(out_path, "%.3f_first_sat" % multiplier)
#     try:
#         df = pd.read_csv(os.path.join(output_path, "result.csv"), index_col=0)
#     except FileNotFoundError:
#         continue
#     x = df.index / 1000
#     y = np.real_if_close(np.array(df["key_per_time"], dtype=complex)) / 2
#     plt.scatter(x, y, marker="o", s=0.5, c=color_list[index], label=f"$S_A$ @ {int(multiplier * 100):d}%")
# # # compare to one satellite
# # path = os.path.join("results", "one_satellite", "divergence_theta", "1")
# # try:
# #     df = pd.read_csv(os.path.join(path, "result.csv"), index_col=0)
# #     x = df.index / 1000
# #     y = np.real_if_close(np.array(df["key_per_time"], dtype=complex)) / 2
# #     plt.scatter(x, y, marker="o", s=0.5, label="1 Satellite")
# # except FileNotFoundError:
# #     pass
# plt.yscale("log")
# plt.ylim(1e-2, 0.5e5)
# plt.legend(loc="lower left", fontsize=6)
# plt.grid()
# plt.xlabel("Ground distance $d$ [km]", color=font_color)
# plt.ylabel("Key / time [Hz]", color=font_color)
# # plt.title(f"{scenario_str}: T_DP=0.1s, num_memories=1000, theta=2µrad")
# plt.tight_layout()
# # end... satellite postitions
# # save the plot
# figure_name = "sat_positions"
# figure_name = figure_name + ".pdf"
# plt.savefig(os.path.join(result_path, figure_name))
# print(f"Plot saved as {figure_name}")
#
# # second unified theta plot
# fig = formatter.figure(width_ratio=1.0, wide=False)
# out_path = os.path.join(result_path, "sat_positions")
# thetas = {2: 4e-6, 3: 6e-6, 4: 8e-6}
# first_satellite_multipliers = [0.0, 0.2]
# markers = {2: "o", 3: "*", 4: "v"}
# colors = [color_list[i] for i in [0,2]]
# for i, theta in thetas.items():
#     out_path = os.path.join(result_path, "divergence_theta", str(i))
#     for multiplier, color in zip(first_satellite_multipliers, colors):
#         output_path = os.path.join(out_path, "%.3f_first_sat" % multiplier)
#         try:
#             df = pd.read_csv(os.path.join(output_path, "result.csv"), index_col=0)
#         except FileNotFoundError:
#             continue
#         x = df.index / 1000
#         y = np.real_if_close(np.array(df["key_per_time"], dtype=complex)) / 2
#         if y[-1] < 0:
#             x = x[:-1]
#             y = y[:-1]
#         plt.scatter(x, y, marker=markers[i], s=2, label=f"$\\theta={int(theta*1e6)}µ$rad, pos = {multiplier}", c=color)
#     # # compare to one satellite
#     # path = os.path.join("results", "one_satellite", "divergence_theta", str(i))
#     # try:
#     #     df = pd.read_csv(os.path.join(path, "result.csv"), index_col=0)
#     #     x = df.index / 1000
#     #     y = np.real_if_close(np.array(df["key_per_time"], dtype=complex)) / 2
#     #     plt.scatter(x, y, marker=markers[i], s=2, label=f"1 Satellite, ${int(theta*1e6)}\\mu$rad", c="C2")
#     # except FileNotFoundError:
#     #     pass
#     plt.plot(xx / 1000, [e91_rate(i, divergence_half_angle=theta) for i in xx], linestyle="dashed", color="gray")#, label=f"E91 20MHz, {int(theta*1e6)}µrad")
#     # # plt.plot(xx / 1000, yy, linestyle="dashed", color="gray", label="E91 20MHz")
# plt.yscale("log")
# plt.ylim(0.3e-1, 0.3e5)
# # plt.legend(loc='upper right', ncol=2)
# plt.grid()
# plt.xlabel("Ground distance $d$ [km]", color=font_color)
# plt.ylabel("Key / time [Hz]", color=font_color)
# # plt.title(f"{scenario_str}: T_DP=0.1s, num_memories=1000, theta=2µrad")
# plt.tight_layout()
# # save the plot
# figure_name = "divergence_thetas"
# figure_name = figure_name + ".pdf"
# plt.savefig(os.path.join(result_path, figure_name))
# print(f"Plot saved as {figure_name}")
#
# # third orbital heights plot
# fig = formatter.figure(width_ratio=1.0, wide=False)
# out_path = os.path.join(result_path, "sat_positions")
# orbital_heights = [400e3, 600e3, 1000e3, 1500e3, 2000e3]
# out_path = os.path.join(result_path, "orbital_heights")
# for i, orbital_height in enumerate(orbital_heights):
#     output_path = os.path.join(out_path, "%d_orbital_height" % int(orbital_height / 1000))
#     try:
#         df = pd.read_csv(os.path.join(output_path, "result.csv"), index_col=0)
#     except FileNotFoundError:
#         continue
#     x = df.index / 1000
#     y = np.real_if_close(np.array(df["key_per_time"], dtype=complex)) / 2
#     # yerr = np.real_if_close(np.array(df["key_per_time_std"], dtype=complex)) / 2
#     plt.scatter(x, y, c=color_list[i], marker="o", s=1, label=f"orbital_height={int(orbital_height / 1000)}km")
#     # plt.errorbar(x, y, yerr, marker="o", label=f"orbital_height={int(orbital_height / 1000)}km")
#     # # compare to one satellite
#     # path = os.path.join("results", "one_satellite", "memories", str(i), "100_t_dp")
#     # try:
#     #     df = pd.read_csv(os.path.join(path, "result.csv"), index_col=0)
#     #     x = df.index / 1000
#     #     y = np.real_if_close(np.array(df["key_per_time"], dtype=complex)) / 2
#     #     plt.scatter(x, y, marker="o", s=10, label="1 Satellite, t_dp=100ms")
#     # except FileNotFoundError:
#     #     pass
# xx = np.linspace(0, 44e5, num=10)
# yy = [e91_rate(i) for i in xx]
# plt.plot(xx / 1000, yy, linestyle="dashed", color="gray", label="E91 20MHz")
# plt.yscale("log")
# plt.ylim(1e-1, 1e5)
# # plt.legend(loc='upper right', ncol=2)
# plt.grid()
# plt.xlabel("Ground distance $d$ [km]", color=font_color)
# plt.ylabel("Key / time [Hz]", color=font_color)
# # plt.title(f"{scenario_str}: T_DP=0.1s, num_memories=1000, theta=2µrad")
# plt.tight_layout()
# # save the plot
# figure_name = "orbital_heights"
# figure_name = figure_name + ".pdf"
# plt.savefig(os.path.join(result_path, figure_name))
# print(f"Plot saved as {figure_name}")
#
# # forth memories
# fig = formatter.figure(width_ratio=1.0, wide=False)
# memories = {6: 1000}
# dephasing_times = [2e-3, 3e-3, 4e-3, 5e-3, 10e-3, 50e-3, 100e-3, 1.0]  # [10e-3, 50e-3, 100e-3, 1.0]
# dephasing_times.reverse()
# for i, num_memories in memories.items():
#     out_path = os.path.join(result_path, "memories", str(i))
#     for idx, t_dp in enumerate(dephasing_times):
#         output_path = os.path.join(out_path, "%d_t_dp" % int(t_dp * 1000))
#         try:
#             df = pd.read_csv(os.path.join(output_path, "result.csv"), index_col=0)
#         except FileNotFoundError:
#             continue
#         x = df.index / 1000
#         y = np.real_if_close(np.array(df["key_per_time"], dtype=complex)) / 2
#         plt.scatter(x, y, c=color_list[idx], marker="o", s=1, label=f"t_dp={t_dp * 1e3}ms")
# #     # # compare to one satellite
# #     # path = os.path.join("results", "one_satellite", "memories", str(i), "100_t_dp")
# #     # try:
# #     #     df = pd.read_csv(os.path.join(path, "result.csv"), index_col=0)
# #     #     x = df.index / 1000
# #     #     y = np.real_if_close(np.array(df["key_per_time"], dtype=complex)) / 2
# #     #     plt.scatter(x, y, marker="o", s=1, label="1 Satellite, t_dp=100ms")
# #     # except FileNotFoundError:
# #     #     pass
#     xx = np.linspace(0, 44e5, num=10)
#     yy = [e91_rate(i) for i in xx]
#     plt.plot(xx / 1000, yy, linestyle="dashed", color="gray", label="E91 20MHz")
# plt.yscale("log")
# plt.ylim(1e-1, 1e5)
# # plt.legend(loc='upper right', ncol=2)
# plt.grid()
# plt.xlabel("Ground distance $d$ [km]", color=font_color)
# plt.ylabel("Key / time [Hz]", color=font_color)
# # plt.title(f"{scenario_str}: T_DP=0.1s, num_memories=1000, theta=2µrad")
# plt.tight_layout()
# # save the plot
# figure_name = "memories_1000"
# figure_name = figure_name + ".pdf"
# plt.savefig(os.path.join(result_path, figure_name))
# print(f"Plot saved as {figure_name}")
#
# # plot cutoff time plot
# fig = formatter.figure(width_ratio=1.0, wide=False)
# out_path = os.path.join(result_path, "cutoff_times")
# cutoff_multipliers = [None, 0.5, 0.2, 0.1, 0.05, 0.02]
# for idx, cutoff_multiplier in enumerate(cutoff_multipliers):
#     try:
#         dir_prefix = "%d" % int(cutoff_multiplier * 100)
#         label = dir_prefix + "ms"
#     except TypeError as e:
#         if cutoff_multiplier is None:
#             dir_prefix = "None"
#             label = dir_prefix
#         else:
#             raise e
#     output_path = os.path.join(out_path, dir_prefix + "_cutoff_multiplier")
#     try:
#         df = pd.read_csv(os.path.join(output_path, "result.csv"), index_col=0)
#     except FileNotFoundError:
#         continue
#     x = df.index / 1000
#     y = np.real_if_close(np.array(df["key_per_time"], dtype=complex)) / 2
#     while y[-1] < 0:
#         x = x[:-1]
#         y = y[:-1]
#     plt.scatter(x, y, c=color_list[idx], marker="o", s=1, label=label)
# xx = np.linspace(0, 44e5, num=10)
# yy = [e91_rate(i, divergence_half_angle=6e-6) for i in xx]
# plt.plot(xx / 1000, yy, linestyle="dashed", color="gray", zorder=0)
# plt.yscale("log")
# plt.ylim(1e-1, 1e4)
# plt.legend(loc='lower left', fontsize=6, title="Cutoff time $t_\mathrm{cut}", title_fontsize=6)
# plt.grid()
# plt.xlabel("Ground distance $d$ [km]", color=font_color)
# plt.ylabel("Key / time [Hz]", color=font_color)
# # plt.title(f"{scenario_str}: T_DP=0.1s, num_memories=1000, theta=2µrad")
# plt.tight_layout()
# # save the plot
# figure_name = "cutoff_times"
# figure_name = figure_name + ".pdf"
# plt.savefig(os.path.join(result_path, figure_name))
# print(f"Plot saved as {figure_name}")
# plt.close()

# case 9 with changing satellite postitions
xx = np.linspace(0, 0.5, num=100)
fig = formatter.figure(aspect_ratio=0.8, width_ratio=1.0, wide=False)
out_path = os.path.join(result_path, "satellite_path")
configurations = [0, 1, 2, -1]
for idx, configuration in enumerate(configurations):
    output_path = os.path.join(out_path, f"{configuration}_configuration")
    df = pd.read_csv(os.path.join(output_path, "result.csv"), index_col=0)
    x = df.index
    ground_distance = 4400e3
    orbital_height = base_params["ORBITAL_HEIGHT"]
    angles = x * ground_distance / R_E
    orbital_period = 2 * np.pi * np.sqrt((R_E + orbital_height)**3 / (M * G))
    times = angles / (2 * np.pi) * orbital_period
    y = np.real_if_close(np.array(df["key_per_time"], dtype=complex)) / 2
    if configuration == -1:
        color = "gray"
    else:
        color = color_list[idx]
    plt.scatter(times, y, c=color, marker="o", s=1, label=f"{configuration=}")
orbital_heights = [600e3, 1000e3, 1500e3, 2000e3]
# for orbital_height in orbital_heights:
#     yy = [e91_rate_pos(length=ground_distance, orbital_height=orbital_height, satellite_position=0.5 + i) for i in xx]
#     plot_x = np.concatenate([-xx[:0:-1], xx])
#     angles = plot_x * ground_distance / R_E
#     orbital_period = 2 * np.pi * np.sqrt((R_E + orbital_height)**3 / (M * G))
#     times = angles / (2 * np.pi) * orbital_period
#     plot_y = list(reversed(yy[1:])) + yy
#     print(f"{orbital_height=} finished calculating")
#     plt.plot(times, plot_y, linestyle="dashed", color="gray", zorder=0)
plt.yscale("log")
plt.ylim(1e0, 1e4)
plt.grid()
# plt.legend()
plt.xlabel("Time [s]", color=font_color)
plt.ylabel("Key / time [Hz]", color=font_color)
plt.tight_layout()
figure_name = "configurations"
figure_name = figure_name + ".pdf"
plt.savefig(os.path.join(result_path, figure_name))
print(f"Plot saved as {figure_name}")

# # rsmf stuff example
# fig = formatter.figure(width_ratio=.8, wide=False)

# # begin... some example plotting
# x = np.linspace(400, 800, 128)

# for i in range(8):
#     y = 1+np.sin(.013*x+.4*i)
#     plt.plot(x, y, color=color_list[i%len(color_list)])

# plt.xlabel('$\\lambda$ [nm]', color=color_list[1])
# plt.ylabel('$I$ [a.u.]', color=color_list[2])
# plt.tight_layout()
# # end... some example plotting

# # save example plot
# figure_name = "example_wide"
# figure_name = figure_name + ".pdf"
# save_path = os.path.join("results", "plot_results")
# plt.savefig(os.path.join(save_path, figure_name))

# print(f"Plot saved as {figure_name}.pdf")
