import numpy as np
import matplotlib.pyplot as plt

arr = np.loadtxt('res_dist_test.txt')
x = arr.T[0]
y = 10*np.log10(arr.T[1]/2)
er = 10/(np.log(10)*np.abs(arr.T[1]))*arr.T[2]
plt.errorbar(x/1000, y, yerr=0, fmt='.')
plt.show()
