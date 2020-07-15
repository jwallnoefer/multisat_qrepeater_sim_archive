import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
import matplotlib.pyplot as plt

result_path = os.path.join("results", "whitepaper")

L_ATT = 22 * 10**3 / 1000 # attenuation length

x_base = np.arange(1000, 401000, 1000) / 1000
eta = np.exp(-x_base/L_ATT)
y_repeaterless = 10 * np.log10(-np.log2(1-eta))
y_optimal = 10 * np.log10(np.sqrt(eta))
y_realistic_repeaterless1 = 10 * np.log10(0.7 * eta / 2)
y_realistic_repeaterless2 = 10 * np.log10(0.1 * eta / 2)

name_list = ["NV", "SiV", "Qdot", "Ca", "Rb"]

# first plot available values
plt.plot(x_base, y_repeaterless, color="black")
plt.plot(x_base, y_optimal, color="gray")
plt.fill_between(x_base, y_repeaterless, y_optimal, facecolor="lightgray")
plt.plot(x_base, y_realistic_repeaterless1, color="black", linestyle="dashed")
plt.plot(x_base, y_realistic_repeaterless2, color="black", linestyle="dashed")
for name in name_list:
    path = os.path.join(result_path, "available", name)
    x = np.loadtxt(os.path.join(path, "length_list.txt")) / 1000
    y = 10 * np.log10(np.loadtxt(os.path.join(path, "key_per_resource_list.txt"), dtype=np.complex))
    plt.scatter(x, y, label=name)
plt.xlim((0,400))
plt.ylim((-60, 0))
plt.legend()
plt.grid()
plt.xlabel("L [km]")
plt.ylabel("secrate key rate per channel use [dB]")
plt.show()

# now plot future values
plt.plot(x_base, y_repeaterless, color="black")
plt.plot(x_base, y_optimal, color="gray")
plt.fill_between(x_base, y_repeaterless, y_optimal, facecolor="lightgray")
plt.plot(x_base, y_realistic_repeaterless1, color="black", linestyle="dashed")
plt.plot(x_base, y_realistic_repeaterless2, color="black", linestyle="dashed")
for name in name_list:
    path = os.path.join(result_path, "future", name)
    x = np.loadtxt(os.path.join(path, "length_list.txt")) / 1000
    y = 10 * np.log10(np.loadtxt(os.path.join(path, "key_per_resource_list.txt"), dtype=np.complex))
    plt.scatter(x, y, label=name)
plt.xlim((0,400))
plt.ylim((-60, 0))
plt.legend()
plt.grid()
plt.xlabel("L [km]")
plt.ylabel("secrate key rate per channel use [dB]")
plt.show()
