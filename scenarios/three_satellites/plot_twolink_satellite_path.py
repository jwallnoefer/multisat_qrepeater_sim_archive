import os, sys; sys.path.insert(0, os.path.abspath("."))
import pandas as pd
import matplotlib.pyplot as plt

result_path = os.path.join("results", "three_satellites", "twolink_downlink", "satellite_path")

configurations = [-1, 0, 1, 2]

for configuration in configurations:
    load_path = os.path.join(result_path, f"{configuration}_configuration")
    df = pd.read_csv(os.path.join(load_path, "result.csv"), index_col=0)
    x = df.index
    y = df["key_per_time"]
    plt.plot(x, y, label=f"{configuration=}")
plt.yscale("log")
plt.ylabel("key per time [Hz]")
plt.xlabel("offset [ground_dist]")
plt.grid()
plt.legend()
plt.show()
