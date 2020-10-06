import numpy as np
import matplotlib.pyplot as plt
import os
result_path = os.path.join("../../results", "verificator")
arr = np.loadtxt(os.path.join("../../results", "verificator","multi_link_epp2.txt"))
arr2 = np.loadtxt(os.path.join("../../results", "verificator","epp_two_link.txt"))
x = arr.T[0]
y = 10*np.log10(arr.T[1]/2)
er = 10/(np.log(10)*np.abs(arr.T[1]))*arr.T[2]
plt.figure(figsize=(12,12))
x_base = np.arange(1000, 401000, 1000) / 1000
L_ATT = 22 * 10**3 / 1000  # attenuation length in km
eta = np.exp(-x_base / L_ATT)
y_repeaterless = 10 * np.log10(-np.log2(1 - eta))
y_optimal = 10 * np.log10(np.sqrt(eta))
y_realistic_repeaterless1 = 10 * np.log10(0.7 * eta / 2)
y_realistic_repeaterless2 = 10 * np.log10(0.1 * eta / 2)
plt.plot(x_base, y_repeaterless, color="black")
plt.plot(x_base, y_optimal, color="gray")
plt.fill_between(x_base, y_repeaterless, y_optimal, facecolor="lightgray")
plt.plot(x_base, y_realistic_repeaterless1, color="black", linestyle="dashed")
plt.plot(x_base, y_realistic_repeaterless2, color="black", linestyle="dashed")
plt.errorbar(x/1000, y, yerr=0, fmt='.',label='One step of epp, multi_link_scripts')
plt.plot(x/1000,10*np.log10(arr.T[3]/2),'.', label = 'No epp, analytic')
x = arr2.T[0]
y = 10*np.log10(arr2.T[1]/2)
er = 10/(np.log(10)*np.abs(arr2.T[1]))*arr2.T[2]
plt.errorbar(x/1000, y, yerr=0, fmt='.',label='One step of epp, two_link_script')
plt.legend()
plt.xlabel('Total distance in km')
plt.ylabel('Key-Rate per channnel use per mode in decibel')
plt.minorticks_on()
plt.grid(which = 'both')
plt.savefig(os.path.join(result_path, "epp_luet_compare2.png"))
plt.show()
