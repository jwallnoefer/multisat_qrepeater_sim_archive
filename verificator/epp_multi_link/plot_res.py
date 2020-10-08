import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('../../')
sys.path.append('../')
import luet
"""
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
"""
kwargs_tuple = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (5, 0), (5, 1),
                (5, 2), (6, 0), (6, 1), (6, 2), (7, 0), (7, 1), (7, 2), (8, 0), (8, 1), (8, 2)]
result_path = os.path.join("../../results", "verificator")
plt.figure(figsize=(18, 18))
x_base = np.arange(10000, 510000, 10000) / 1000
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
for n, k in kwargs_tuple:
    if k == 2 or k == 2:
        continue
    arr = np.loadtxt(os.path.join("../../results", "verificator",
                                  "multi_link_epp" + str(n) + "_" + str(k) + ".txt"))
    x = arr.T[0]
    y = 10 * np.log10(arr.T[1] / n)
    plt.errorbar(x / 1000, y, yerr=0, fmt='.', label=str(k) +
                 ' steps of epp, ' + str(n) + ' links')
y = [luet.lower_bound(l * 500) for l in x_base]
plt.plot(x_base, 10 * np.log10(np.array(y) / 2),
         '.', label='No epp, analytic')
plt.legend()
plt.xlabel('Total distance in km')
plt.ylabel('Key-Rate per channnel use per mode in decibel')
plt.minorticks_on()
plt.grid(which='both')
plt.savefig(os.path.join(result_path, "epp_multi_compare_0_1.png"))
plt.show()
