import numpy as np
import matplotlib.pyplot as plt
import os
result_path = os.path.join("../../results", "verificator","epp_two_link.txt")
arr = np.loadtxt(result_path)
x = arr.T[0]
y = 10*np.log10(arr.T[1]/2)
er = 10/(np.log(10)*np.abs(arr.T[1]))*arr.T[2]
plt.errorbar(x/1000, y, yerr=er, fmt='.')
plt.grid()
plt.show()

fx = np.loadtxt("../../results/verificator/fx_list.txt")
fz = np.loadtxt("../../results/verificator/fz_list.txt")
plt.plot(x/1000, fx, label="fx")
plt.plot(x/1000, fz, label="fz")
plt.legend()
plt.grid()
plt.show()
