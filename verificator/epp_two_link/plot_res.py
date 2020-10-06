import numpy as np
import matplotlib.pyplot as plt
import os
result_path = os.path.join("../../results", "verificator")
arr = np.loadtxt(os.path.join("../../results", "verificator","epp_two_link.txt"))
x = arr.T[0]
y = 10*np.log10(arr.T[1]/2)
er = 10/(np.log(10)*np.abs(arr.T[1]))*arr.T[2]
plt.figure(figsize=(12,12))
plt.errorbar(x/1000, y, yerr=0, fmt='.',label='One step of epp')
plt.plot(x/1000,10*np.log10(arr.T[3]/2),'.', label = 'No epp, analytic')
plt.legend()
plt.xlabel('Total distance in km')
plt.ylabel('Key-Rate per channnel use per mode in decibel')
plt.minorticks_on()
plt.savefig(os.path.join(result_path, "epp_luet_compare.png"))
plt.show()

fx = np.loadtxt("../../results/verificator/fx_list.txt")
fz = np.loadtxt("../../results/verificator/fz_list.txt")
plt.plot(x/1000, fx, label="fx")
plt.plot(x/1000, fz, label="fz")
plt.legend()
plt.grid()
plt.show()
