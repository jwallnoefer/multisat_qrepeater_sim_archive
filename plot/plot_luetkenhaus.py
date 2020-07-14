import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
import matplotlib.pyplot as plt

result_path = os.path.join("results", "luetkenhaus")

x1 = np.loadtxt(os.path.join(result_path, "length_list_seq.txt"), dtype=np.complex)
y1 = np.loadtxt(os.path.join(result_path, "key_per_resource_list_seq.txt"), dtype=np.complex) / 2  # half because we compare it to keyrate per mode
y4 = np.loadtxt(os.path.join(result_path, "key_per_resource_list_sim.txt"), dtype=np.complex) / 2
x2 = np.loadtxt(os.path.join(result_path, "onerep_length.txt"), dtype=np.complex)
y2 = np.loadtxt(os.path.join(result_path, "onerep_sequential.txt"), dtype=np.complex)
y3 = np.loadtxt(os.path.join(result_path, "onerep_simultaneous.txt"), dtype=np.complex)

plt.scatter(x1, y1, label="sequential, simulation")
plt.plot(x2, y2, label="sequential, analytical")
plt.scatter(x1, y4, label="simultaneous, simulation")
plt.plot(x2, y3, label="simultaneous, analytical")
plt.yscale("log")
plt.xlabel("total length")
plt.ylabel("key rate per channel use")
plt.grid()
plt.legend()
plt.savefig(os.path.join(result_path, "comparison.png"))
plt.show()
