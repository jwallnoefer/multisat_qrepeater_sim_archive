import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
import matplotlib.pyplot as plt

result_path = os.path.join("results", "multimemory_variant_cutoff")

for num_memories in [1, 5, 10, 50, 100, 400]:
    memory_path = os.path.join(result_path, "%d_memories" % num_memories)
    x = np.loadtxt(os.path.join(memory_path, "length_list.txt")) / 1000
    y = np.loadtxt(os.path.join(memory_path, "key_per_resource_list.txt"), dtype=np.complex) / 2
    plt.scatter(x, y, label="num_memories=%d" % num_memories)

plt.yscale("log")
plt.ylim(1e-7, 1e-2)
plt.xlim(0, 300)
plt.legend()
plt.xlabel("L [km]")
plt.ylabel("key per resource")
plt.grid()
plt.show()


for num_memories in [1, 5, 10, 50, 100, 400]:
    memory_path = os.path.join(result_path, "%d_memories" % num_memories)
    x = np.loadtxt(os.path.join(memory_path, "length_list.txt")) / 1000
    y = np.loadtxt(os.path.join(memory_path, "key_per_time_list.txt"), dtype=np.complex) / 2
    plt.scatter(x, y, label="num_memories=%d" % num_memories)

plt.yscale("log")
plt.ylim(5e-3, 1e5)
plt.xlim(0, 300)
plt.legend()
plt.xlabel("L [km]")
plt.ylabel("key per time")
plt.grid()
plt.show()
