import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
import matplotlib.pyplot as plt

result_path = os.path.join("results", "whitepaper", "available", "m_test", "Rb")

L_ATT = 22 * 10**3 / 1000 # attenuation length

x_base = np.arange(1000, 401000, 1000) / 1000
eta = np.exp(-x_base/L_ATT)
y_repeaterless = 10 * np.log10(-np.log2(1-eta))
y_optimal = 10 * np.log10(np.sqrt(eta))
y_realistic_repeaterless1 = 10 * np.log10(0.7 * eta / 2)
y_realistic_repeaterless2 = 10 * np.log10(0.1 * eta / 2)


plt.plot(x_base, y_repeaterless, color="black")
plt.plot(x_base, y_optimal, color="gray")
plt.fill_between(x_base, y_repeaterless, y_optimal, facecolor="lightgray")
plt.plot(x_base, y_realistic_repeaterless1, color="black", linestyle="dashed")
plt.plot(x_base, y_realistic_repeaterless2, color="black", linestyle="dashed")
ms = [50, 200, 500, 1000, 2000, 5000, 10000, 20000]
for m in ms:
    x = np.loadtxt(os.path.join(result_path, "length_list_%d.txt" % m)) / 1000
    y = 10 * np.log10(np.loadtxt(os.path.join(result_path, "key_per_resource_list_%d.txt" % m), dtype=np.complex))
    plt.plot(x, y, label="m=%d" % m)
plt.xlim((0,400))
plt.ylim((-60, 0))
plt.legend()
plt.grid()
plt.xlabel("L [km]")
plt.ylabel("secrate key rate per channel use [dB]")
plt.savefig(os.path.join(result_path, "m_compare.png"))
plt.show()
