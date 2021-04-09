import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

result_path = os.path.join("results", "three_satellites", "fourlink")
for first_satellite_multiplier in np.linspace(0, 0.5, num=9):
    try:
        df = pd.read_csv(os.path.join(result_path, "%.3f_first_sat" % first_satellite_multiplier, "result.csv"), index_col=0)
    except FileNotFoundError:
        continue
    x = df.index / 1000
    y = df["key_per_resource"] / 2
    yerr = df["key_per_resource_std"] / 2
    plt.scatter(x, y, marker="o", s=5, label="first_sat_pos=%.3f" % first_satellite_multiplier)

plt.yscale("log")
plt.ylim(1e-4, 1e-0)
plt.xlabel("Total ground distance [km]")
plt.ylabel("Key per resource")
plt.grid()
plt.legend()
plt.show()


for first_satellite_multiplier in np.linspace(0, 0.5, num=9):
    try:
        df = pd.read_csv(os.path.join(result_path, "%.3f_first_sat" % first_satellite_multiplier, "result.csv"), index_col=0)
    except FileNotFoundError:
        continue
    x = df.index / 1000
    y = df["key_per_time"] / 2
    yerr = df["key_per_time_std"] / 2
    plt.scatter(x, y, marker="o", s=5, label="first_sat_pos=%.3f" % first_satellite_multiplier)


plt.yscale("log")
plt.ylim(1e1, 1e5)
plt.xlabel("Total ground distance [km]")
plt.ylabel("Key per time [Hz]")
plt.grid()
plt.legend()
plt.show()
