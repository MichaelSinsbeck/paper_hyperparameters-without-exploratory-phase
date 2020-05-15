"""
Plotting the results of Experiment 4
"""

import numpy as np
import matplotlib.pyplot as plt

def harmonic_mean(error, axis=0):
    return np.exp(np.mean( np.log(error), axis=axis))

n_eval = np.arange(31)

e_map = np.load('output/error_map.npy')
e_random = np.load('output/error_random.npy')
e_miracle = np.load('output/error_miracle.npy')

e_mean_map = harmonic_mean(e_map)
e_0_map = e_map.min(axis=0)
e_100_map = e_map.max(axis=0)

e_mean_random = harmonic_mean(e_random)
e_0_random = e_random.min(axis=0)
e_100_random = e_random.max(axis=0)

e_mean_miracle = harmonic_mean(e_miracle)
e_0_miracle = e_miracle.min(axis=0)
e_100_miracle = e_miracle.max(axis=0)

plt.semilogy(n_eval, e_mean_map)
plt.semilogy(n_eval, e_mean_miracle)
plt.semilogy(n_eval, e_mean_random)

plt.fill_between(n_eval, e_0_map, e_100_map, color='lightblue')
#plt.fill_between(n_eval, e_0_miracle, e_100_miracle, color='pink')
#plt.fill_between(n_eval, e_0_random, e_100_random, color='lightgreen')

plotdata = np.full((31,10), np.nan)

plotdata[:,0] = n_eval

plotdata[:,1] = e_mean_map
plotdata[:,2] = e_0_map
plotdata[:,3] = e_100_map

plotdata[:,4] = e_mean_miracle
plotdata[:,5] = e_0_miracle
plotdata[:,6] = e_100_miracle

plotdata[:,7] = e_mean_random
plotdata[:,8] = e_0_random
plotdata[:,9] = e_100_random

np.savetxt('output/exp4.data', plotdata, '%2i %1.3e %1.3e %1.3e %1.3e %1.3e %1.3e %1.3e %1.3e %1.3e')
