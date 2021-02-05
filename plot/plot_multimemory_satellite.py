import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scenarios.multi_memory_satellite import sat_dist_curved, elevation_curved, eta_atm, eta_dif


def db(x):
    return 10 * np.log10(x)


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


result_path = os.path.join("results", "multimemory_satellite_cutoff")
for cutoff_multiplier in [0.001, 0.005, 0.010, 0.020, 0.030, 0.050, 0.100, 0.250, 0.500]:
    df = pd.read_csv(os.path.join(result_path, "%.3f_cutoff" % cutoff_multiplier, "result.csv"), index_col=0)
    x = df.index / 1000
    y = df["key_per_resource"] / 2
    yerr = df["key_per_resource_std"] / 2
    # plt.errorbar(x, y, yerr=yerr, fmt="o", label="cutoff_multiplier=%.3f" % cutoff_multiplier)
    plt.scatter(x, y, marker="o", s=5, label="cutoff_multiplier=%.3f" % cutoff_multiplier)

xx = np.linspace(0, 4e6, num=500)
yy = [e91_eta(i) for i in xx]
plt.plot(xx / 1000, yy, linestyle="dashed", color="gray", label="E91")

plt.yscale("log")
plt.ylim(1e-9, 1e-2)
# plt.xlim(0, 300)
plt.legend()
plt.xlabel("ground distance [km]")
plt.ylabel("key per resource")
plt.title("dephasing time = 7.5s, memory modes = 1000")
plt.grid()
plt.savefig(os.path.join(result_path, "per_resource.png"))
plt.show()

result_path = os.path.join("results", "multimemory_satellite_cutoff")
for cutoff_multiplier in [0.001, 0.005, 0.010, 0.020, 0.030, 0.050, 0.100, 0.250, 0.500]:
    df = pd.read_csv(os.path.join(result_path, "%.3f_cutoff" % cutoff_multiplier, "result.csv"), index_col=0)
    x = sat_dist_curved(df.index / 2, 400e3) / 1000
    y = df["key_per_resource"] / 2
    yerr = df["key_per_resource_std"] / 2
    # plt.errorbar(x, y, yerr=yerr, fmt="o", label="cutoff_multiplier=%.3f" % cutoff_multiplier)
    plt.scatter(x, y, marker="o", s=5, label="cutoff_multiplier=%.3f" % cutoff_multiplier)

xx = np.linspace(0, 4e6, num=500)
yy = [e91_eta(i) for i in xx]
plt.plot(sat_dist_curved(xx / 2, 400e3) / 1000, yy, linestyle="dashed", color="gray", label="E91")

plt.yscale("log")
# plt.ylim(1e-7, 1e-2)
# plt.xlim(0, 300)
plt.legend()
plt.xlabel("ground-satellite distance [km]")
plt.ylabel("key per resource")
plt.title("dephasing time = 7.5s, memory modes = 1000")
plt.grid()
plt.show()


for cutoff_multiplier in [0.001, 0.005, 0.010, 0.020, 0.030, 0.050, 0.100, 0.250, 0.500]:
    df = pd.read_csv(os.path.join(result_path, "%.3f_cutoff" % cutoff_multiplier, "result.csv"), index_col=0)
    x = df.index / 1000
    y = df["key_per_time"] / 2
    yerr = df["key_per_time_std"] / 2
    # plt.errorbar(x, y, yerr=yerr, fmt="o", label="cutoff_multiplier=%.3f" % cutoff_multiplier)
    plt.scatter(x, y, marker="o", s=5, label="cutoff_multiplier=%.3f" % cutoff_multiplier)

xx = np.linspace(0, 4e6, num=500)
yy = [e91_rate(i) for i in xx]
plt.plot(xx / 1000, yy, linestyle="dashed", color="gray", label="E91")

plt.yscale("log")
# plt.ylim(1e-7, 1e-2)
# plt.xlim(0, 300)
plt.legend()
plt.xlabel("ground distance [km]")
plt.ylabel("key per time")
plt.title("dephasing time = 7.5s, memory modes = 1000")
plt.grid()
plt.savefig(os.path.join(result_path, "per_time.png"))
plt.show()

for cutoff_multiplier in [0.001, 0.005, 0.010, 0.020, 0.030, 0.050, 0.100, 0.250, 0.500]:
    df = pd.read_csv(os.path.join(result_path, "%.3f_cutoff" % cutoff_multiplier, "result.csv"), index_col=0)
    x = sat_dist_curved(df.index / 2, 400e3) / 1000
    y = df["key_per_time"] / 2
    yerr = df["key_per_time_std"] / 2
    # plt.errorbar(x, y, yerr=yerr, fmt="o", label="cutoff_multiplier=%.3f" % cutoff_multiplier)
    plt.scatter(x, y, marker="o", s=5, label="cutoff_multiplier=%.3f" % cutoff_multiplier)

plt.plot(sat_dist_curved(xx / 2, 400e3) / 1000, yy, linestyle="dashed", color="gray", label="E91")

plt.yscale("log")
plt.ylim(1e-4, 1e4)
# plt.xlim(0, 300)
plt.legend()
plt.xlabel("ground-satellite distance [km]")
plt.ylabel("key per time")
plt.title("dephasing time = 7.5s, memory modes = 1000")
plt.grid()
plt.show()

# ================ DEPHASING PLOT ============
result_path = os.path.join("results", "multimemory_satellite_dephasing")
for t_dp in np.logspace(np.log10(100e-3), np.log10(7.5), num=10):
    df = pd.read_csv(os.path.join(result_path, "%.2f_dephasing" % t_dp, "result.csv"), index_col=0)
    x = df.index / 1000
    y = df["key_per_resource"] / 2
    yerr = df["key_per_resource_std"] / 2
    # plt.errorbar(x, y, yerr=yerr, fmt="o", label="dephasing_time=%.2f" % t_dp)
    plt.scatter(x, y, marker="o", s=5, label="dephasing_time=%.2f" % t_dp)

xx = np.linspace(0, 4e6, num=500)
yy = [e91_eta(i) for i in xx]
plt.plot(xx / 1000, yy, linestyle="dashed", color="gray", label="E91")

plt.yscale("log")
plt.ylim(1e-9, 1e-2)
# plt.xlim(0, 300)
plt.legend()
plt.xlabel("ground distance [km]")
plt.ylabel("key per resource")
plt.title("memory modes = 1000, cutoff_time = 0.05 * dephasing_time")
plt.grid()
plt.savefig(os.path.join(result_path, "per_resource.png"))
plt.show()

for t_dp in np.logspace(np.log10(100e-3), np.log10(7.5), num=10):
    df = pd.read_csv(os.path.join(result_path, "%.2f_dephasing" % t_dp, "result.csv"), index_col=0)
    x = df.index / 1000
    y = df["key_per_time"] / 2
    yerr = df["key_per_time_std"] / 2
    # plt.errorbar(x, y, yerr=yerr, fmt="o", label="dephasing_time=%.2f" % t_dp)
    plt.scatter(x, y, marker="o", s=5, label="dephasing_time=%.2f" % t_dp)

xx = np.linspace(0, 4e6, num=500)
yy = [e91_rate(i) for i in xx]
plt.plot(xx / 1000, yy, linestyle="dashed", color="gray", label="E91")

plt.yscale("log")
plt.ylim(1e-5, 1e4)
# plt.xlim(0, 300)
plt.legend()
plt.xlabel("ground distance [km]")
plt.ylabel("key per time")
plt.title("memory modes = 1000, cutoff_time = 0.05 * dephasing_time")
plt.grid()
plt.savefig(os.path.join(result_path, "per_time.png"))
plt.show()
