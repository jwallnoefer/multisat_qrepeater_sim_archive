import numpy as np
import matplotlib.pyplot as plt
import os
result_path = os.path.join("../../results", "verificator")
arr = np.loadtxt(os.path.join("../../results", "verificator","multi_link_epp4.txt"))
x = arr.T[0]
y = 10*np.log10(arr.T[1]/4)
er = 10/(np.log(10)*np.abs(arr.T[1]))*arr.T[2]
plt.figure(figsize=(12,12))
plt.errorbar(x/2000, y, yerr=er, fmt='.',label='One step of epp, four links')
plt.plot(x/2000,10*np.log10(arr.T[3]/4),'.', label = 'No epp, analytic')
plt.legend()
plt.xlabel('Total distance in km')
plt.ylabel('Key-Rate per channnel use per mode in decibel')
plt.minorticks_on()
plt.savefig(os.path.join(result_path, "epp_luet_compare4.png"))
plt.show()
