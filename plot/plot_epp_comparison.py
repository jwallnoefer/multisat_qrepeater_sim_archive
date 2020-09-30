import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
import matplotlib.pyplot as plt


x_base = np.arange(1000, 401000, 1000) / 1000
L_ATT = 22 * 10**3 / 1000  # attenuation length in km
eta = np.exp(-x_base / L_ATT)
y_repeaterless = 10 * np.log10(-np.log2(1 - eta))
y_optimal = 10 * np.log10(np.sqrt(eta))
y_realistic_repeaterless1 = 10 * np.log10(0.7 * eta / 2)
y_realistic_repeaterless2 = 10 * np.log10(0.1 * eta / 2)

result_path = os.path.join("results", "two_link_epp")

path_without = os.path.join(result_path, "without_epp")
length_list_without = np.loadtxt(os.path.join(path_without, "length_list.txt")) / 1000
skr_without = 10 * np.log10(np.loadtxt(os.path.join(path_without, "key_per_resource_list.txt"), dtype=np.complex) / 2)

path_with = os.path.join(result_path, "with_epp")
length_list_with = np.loadtxt(os.path.join(path_with, "length_list.txt")) / 1000
skr_with = 10 * np.log10(np.loadtxt(os.path.join(path_with, "key_per_resource_list.txt"), dtype=np.complex) / 2)


plt.plot(x_base, y_repeaterless, color="black")
plt.plot(x_base, y_optimal, color="gray")
plt.fill_between(x_base, y_repeaterless, y_optimal, facecolor="lightgray")
plt.plot(x_base, y_realistic_repeaterless1, color="black", linestyle="dashed")
plt.plot(x_base, y_realistic_repeaterless2, color="black", linestyle="dashed")

plt.scatter(length_list_without, skr_without, label="without epp")
plt.scatter(length_list_with, skr_with, label="with 1 epp-step")
plt.xlim((0, 400))
plt.ylim((-60, 0))
plt.grid()
plt.legend()
plt.xlabel("L [km]")
plt.ylabel("secret key rate per channel use [dB]")
plt.title("EPP comparison with bad memories and no cutoff_time")
plt.savefig(os.path.join(result_path, "epp_comparison.png"))
plt.show()
