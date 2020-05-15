#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creating plots for experiment 2
"""
import numpy as np
import bbi
import matplotlib.pyplot as plt
import pickle

def harmonic_mean(error, axis=0):
    return np.exp(np.mean( np.log(error), axis=axis))

# my main method (iterative map estimation)
content = np.load('output/02_main.npz')
n_eval = content['n_eval']
errors_m = content['errors_m']
t_m = content['t_m']
e_mean_map = harmonic_mean(errors_m)
e_0_map = errors_m.min(axis=0)
e_100_map = errors_m.max(axis=0)

# the miracle solution
errors_mir = np.load('output/02_error_miracle.npy')
e_mean_miracle = harmonic_mean(errors_mir)
e_0_miracle = errors_mir.min(axis=0)
e_100_miracle = errors_mir.max(axis=0)

# random guessing
content = np.load('output/02_21_random_se.npz')
errors_random = content['errors_gpe']
e_mean_random = harmonic_mean(errors_random)
e_0_random= errors_random.min(axis=0)
e_100_random= errors_random.max(axis=0)


plt.semilogy(n_eval, e_mean_map)
plt.semilogy(n_eval, e_mean_miracle, color = 'red')

plt.fill_between(n_eval, e_0_random, e_100_random, color = 'lightgray')
plt.semilogy(n_eval, e_mean_random, color = 'black')

# with exploratory phase

n_eval_pre = pickle.load(open('output/02_pre20_n_eval.pkl', 'rb'))
errors_f = pickle.load(open('output/02_pre_errors_f.pkl', 'rb'))
errors_r = pickle.load(open('output/02_pre_errors_r.pkl', 'rb'))


for e,n in zip(errors_r, n_eval_pre):
    e_r = np.exp(np.mean( np.log(e), axis=0))
    plt.semilogy(n, e_r, color = 'lightgray')


for e,n in zip(errors_f, n_eval_pre):
    e_f = np.exp(np.mean( np.log(e), axis=0))
    plt.semilogy(n,e_f, color = 'black')
    
plt.show()

plotdata = np.zeros((len(n_eval), 10))
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

np.savetxt('output/exp2.data', plotdata, '%3i %1.3e %1.3e %1.3e %1.3e %1.3e %1.3e %1.3e %1.3e %1.3e')


for e_r, e_f, n in zip(errors_r, errors_f, n_eval_pre):
        n_init = n[0]
        filename = 'output/exp2_{}.data'.format(n_init)
        plotdata = np.zeros((len(n), 7))
        plotdata[:,0] = n
        plotdata[:,1] = np.exp(np.mean( np.log(e_f), axis=0))
        plotdata[:,2] = e_f.min(axis=0)
        plotdata[:,3] = e_f.max(axis=0)
        plotdata[:,4] = np.exp(np.mean( np.log(e_r), axis=0))
        plotdata[:,5] = e_r.min(axis=0)
        plotdata[:,6] = e_r.max(axis=0)
        np.savetxt(filename, plotdata, '%3i %1.3e %1.3e %1.3e %1.3e %1.3e %1.3e')
