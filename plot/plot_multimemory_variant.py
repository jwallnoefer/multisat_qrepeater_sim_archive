import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
import matplotlib.pyplot as plt

# first: fixed cutoff, variable length and memories
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
plt.title("fixed cutoff 3 * expected_time")
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
plt.title("fixed cutoff 3 * expected_time")
plt.grid()
plt.show()

# second: fixed cutoff, x-axis=memories
result_path = os.path.join("results", "multimemory_variant_memories")
for length in [50e3, 100e3, 150e3, 200e3]:
    memory_path = os.path.join(result_path, "%d_km" % (length / 1000))
    x = np.loadtxt(os.path.join(memory_path, "memories_list.txt"))
    y = np.loadtxt(os.path.join(memory_path, "key_per_resource_list.txt"), dtype=np.complex) / 2
    plt.scatter(x, y, label="length=%d km" % (length / 1000))

plt.yscale("log")
plt.ylim(1e-6, 1e-3)
# plt.xlim(0, 300)
plt.legend()
plt.xlabel("number of memories")
plt.ylabel("key per resource")
plt.title("fixed cutoff 3 * expected_time")
plt.grid()
plt.show()

for length in [50e3, 100e3, 150e3, 200e3]:
    memory_path = os.path.join(result_path, "%d_km" % (length / 1000))
    x = np.loadtxt(os.path.join(memory_path, "memories_list.txt"))
    y = np.loadtxt(os.path.join(memory_path, "key_per_time_list.txt"), dtype=np.complex) / 2
    plt.scatter(x, y, label="length=%d km" % (length / 1000))

plt.yscale("log")
plt.ylim(1e-3, 1e4)
# plt.xlim(0, 300)
plt.legend()
plt.xlabel("number of memories")
plt.ylabel("key per time")
plt.title("fixed cutoff 3 * expected_time")
plt.grid()
plt.show()

# third: fixed length, x-axis=cutoff_times
result_path = os.path.join("results", "multimemory_variant_by_cutoff")
for num_memories in [1, 5, 10, 50, 100, 400]:
    memory_path = os.path.join(result_path, "%d_memories" % num_memories)
    x = np.loadtxt(os.path.join(memory_path, "cutoff_times.txt"))
    y = np.loadtxt(os.path.join(memory_path, "key_per_resource_list.txt"), dtype=np.complex) / 2
    plt.scatter(x, y, label="num_memories=%d" % num_memories)

plt.yscale("log")
plt.ylim(1e-6, 1e-4)
# plt.xlim(0, 300)
plt.legend()
plt.xlabel("cutoff_time")
plt.ylabel("key per resource")
plt.title("fixed length 150km")
plt.grid()
plt.show()

for num_memories in [1, 5, 10, 50, 100, 400]:
    memory_path = os.path.join(result_path, "%d_memories" % num_memories)
    x = np.loadtxt(os.path.join(memory_path, "cutoff_times.txt"))
    y = np.loadtxt(os.path.join(memory_path, "key_per_time_list.txt"), dtype=np.complex) / 2
    plt.scatter(x, y, label="num_memories=%d" % num_memories)

plt.yscale("log")
plt.ylim(1e-3, 1e2)
# plt.xlim(0, 300)
plt.legend()
plt.xlabel("cutoff_time")
plt.ylabel("key per time")
plt.title("fixed length 150km")
plt.grid()
plt.show()


# fourth: fixed memories, x-axis=length
result_path = os.path.join("results", "multimemory_variant_fixed_mem")

for cutoff_multiplier in [0.001, 0.005, 0.01]:
    try:
        memory_path = os.path.join(result_path, "%.3f_cutoff" % cutoff_multiplier)
        x = np.loadtxt(os.path.join(memory_path, "length_list.txt")) / 1000
        y = np.loadtxt(os.path.join(memory_path, "key_per_resource_list.txt"), dtype=np.complex) / 2
        plt.scatter(x, y, label="cutoff_multiplier=%.3f" % cutoff_multiplier)
    except OSError:
        pass

for cutoff_multiplier in np.concatenate([[0.02, 0.03, 0.05, 0.10], [0.25, 0.5]]):  # np.arange(0.25, 3.25, 0.25)]):
    try:
        memory_path = os.path.join(result_path, "%.2f_cutoff" % cutoff_multiplier)
        x = np.loadtxt(os.path.join(memory_path, "length_list.txt")) / 1000
        y = np.loadtxt(os.path.join(memory_path, "key_per_resource_list.txt"), dtype=np.complex) / 2
        plt.scatter(x, y, label="cutoff_multiplier=%.2f" % cutoff_multiplier)
    except OSError:
        pass

plt.yscale("log")
plt.ylim(1e-8, 1e-2)
# plt.xlim(0, 300)
plt.legend()
plt.xlabel("L [km]")
plt.ylabel("key per resource")
plt.title("fixed 400 memories")
plt.grid()
plt.show()

for cutoff_multiplier in [0.001, 0.005, 0.01]:
    try:
        memory_path = os.path.join(result_path, "%.3f_cutoff" % cutoff_multiplier)
        x = np.loadtxt(os.path.join(memory_path, "length_list.txt")) / 1000
        y = np.loadtxt(os.path.join(memory_path, "key_per_time_list.txt"), dtype=np.complex) / 2
        plt.scatter(x, y, label="cutoff_multiplier=%.2f" % cutoff_multiplier)
    except OSError:
        pass

for cutoff_multiplier in np.concatenate([[0.02, 0.03, 0.05, 0.10], [0.25, 0.5]]):  # np.arange(0.25, 3.25, 0.25)]):
    try:
        memory_path = os.path.join(result_path, "%.2f_cutoff" % cutoff_multiplier)
        x = np.loadtxt(os.path.join(memory_path, "length_list.txt")) / 1000
        y = np.loadtxt(os.path.join(memory_path, "key_per_time_list.txt"), dtype=np.complex) / 2
        plt.scatter(x, y, label="cutoff_multiplier=%.2f" % cutoff_multiplier)
    except OSError:
        pass

plt.yscale("log")
plt.ylim(1e-2, 1e6)
# plt.xlim(0, 300)
plt.legend()
plt.xlabel("L [km]")
plt.ylabel("key per time")
plt.title("fixed 400 memories")
plt.grid()
plt.show()
