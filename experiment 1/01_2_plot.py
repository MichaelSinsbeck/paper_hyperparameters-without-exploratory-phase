# ====================================================================
# This code will load the results/outputs from 01_1.py and produce the
# plots for the paper
# ====================================================================

import numpy as np
import bbi
from matplotlib import pyplot as plt
import pickle
plt.rcParams.update({'font.size': 14})

def harmonic_mean(error, axis=0):
    return np.exp(np.mean( np.log(error), axis=axis))

# general: load reference likelihood and data needed in all plots
content = np.load('input/01_model.npz')
grid = content['grid']

ll_true = np.load('output/01_reference_ll.npy')

# load result of computation step 1)
content = np.load('output/01_main_ll.npz', allow_pickle = True)
n_eval = content['n_eval']
ll_m = content['ll_m']
ll_l = content['ll_l']
ll_a = content['ll_a']

nodes_m = content['nodes_m'].item()
nodes_l = content['nodes_l'].item()
nodes_a = content['nodes_a'].item()

content = pickle.load( open('output/01_pre_nodes_f.pkl', 'rb'))             
nodes_lhs = content[2][0] # first index = 2 means third run which is initial sample = 15

content = np.load('output/01_miracle.npz', allow_pickle = True)
ll_1 = content['ll_1']

# compute errors from loglikelihoods
errors_m = bbi.compute_errors(ll_true, ll_m)
errors_l = bbi.compute_errors(ll_true, ll_l)
errors_a = bbi.compute_errors(ll_true, ll_a)
errors_1 = bbi.compute_errors(ll_true, ll_1)

np.savetxt('output/exp1_nodes_m.txt', grid[nodes_m.idx], fmt= '%g', delimiter = ' ')
np.savetxt('output/exp1_nodes_l.txt', grid[nodes_l.idx], fmt= '%g', delimiter = ' ')
np.savetxt('output/exp1_nodes_a.txt', grid[nodes_a.idx], fmt= '%g', delimiter = ' ')
np.savetxt('output/exp1_nodes_lhs15.txt', grid[nodes_lhs.idx], fmt= '%g', delimiter = ' ')

f = plt.figure(figsize=(16,16))
x = np.linspace(0,1,51)
img = plt.contourf(x,x,np.exp(ll_true).reshape(51,51))

img.set_cmap('Blues')

#f.savefig("figures/01_fig_0_solution.pdf", bbox_inches='tight')

plt.axis('off')
#plt.savefig("figures/01_fig_0_solution.png", bbox_inches='tight')

# %% Plot 1: Comparison of new methods with random picking

plt.rcParams.update({'font.size': 14})
# load results of 20 random gpes
content = np.load('output/01_20_random_matern.npz', allow_pickle = True)
nodes_gpe = content['nodes_gpe']
#ll_gpe = content['ll_gpe']
errors_gpe = content['errors_gpe']

error_average = np.exp(np.mean( np.log(errors_gpe), axis=0))
e_mean_random = harmonic_mean(errors_gpe)
e_0_random = errors_gpe.min(axis=0)
e_100_random = errors_gpe.max(axis=0)

#error_sorted = np.sort(errors_gpe, axis = 0)
#error_0 = error_sorted[0,:]
#error_5 = error_sorted[1,:]
#error_95 = error_sorted[19,:]
#error_100 = error_sorted[-1,:]

#error_50 = error_sorted[10,:]
 
# plot
f = plt.figure(figsize=(12,8))
plt.xlim(0,30)
plt.ylim(1e-6,1e2)
plt.xlabel('Number of model evaluations')
plt.ylabel('Error (KL-divergence)')

plt.semilogy(n_eval, errors_m, label = 'dynamic MAP estimate',linewidth = 2.5)
plt.semilogy(n_eval, errors_a, label = 'average criterion',linewidth = 2.5)
plt.semilogy(n_eval, errors_l, label = 'linearization',linewidth = 2.5)
plt.semilogy(n_eval, errors_1, label = 'miracle',linewidth = 2.5)

plt.fill_between(n_eval, e_0_random, e_100_random, label = '5% to 95% percentiles', color = 'lightgray')
plt.semilogy(n_eval, e_mean_random, label = 'mean of random hyper parameters', color = 'black',linewidth = 2.5)

plt.legend(loc=3, frameon = False).set_zorder(-1)

plt.show()

#f.savefig("figures/01_fig_1_random.pdf", bbox_inches='tight')

plot_data = np.full((31,8), np.nan)
plot_data[:,0] = n_eval

plot_data[:,1] = errors_m
plot_data[:,2] = errors_a
plot_data[:,3] = errors_l
plot_data[:,4] = errors_1
plot_data[:,5] = e_mean_random
plot_data[:,6] = e_0_random
plot_data[:,7] = e_100_random

np.savetxt('output/exp1_fig1.data', plot_data, '%2i %1.3e %1.3e %1.3e %1.3e %1.3e %1.3e %1.3e')

#%% Plot 2: Comparison with separate sampling phase approach


plt.rcParams.update({'font.size': 14})
# load data

n_eval_pre =     pickle.load( open('output/01_pre50_lhs_n_eval.pkl','rb'))
errors_raw_fix = pickle.load( open('output/01_pre_errors_f.pkl','rb'))
errors_raw_re = pickle.load( open('output/01_pre_errors_r.pkl','rb'))

e_mean_fix = []
e_0_fix = []
e_100_fix = []
e_mean_re = []
e_0_re = []
e_100_re = []
    
for e in errors_raw_fix:
    e_mean_fix.append(harmonic_mean(e))
    e_0_fix.append(e.min(axis=0))
    e_100_fix.append(e.max(axis=0))
    
for e in errors_raw_re:
    e_mean_re.append(harmonic_mean(e))
    e_0_re.append(e.min(axis=0))
    e_100_re.append(e.max(axis=0))

f = plt.figure(figsize=(12,8))
plt.xlabel('Number of model evaluations')
plt.ylabel('Error (KL-divergence)')

plt.semilogy(n_eval, errors_m, label = 'dynamic MAP estimate',linewidth = 1)
plt.semilogy(n_eval, errors_a, label = 'average criterion',linewidth = 1)
plt.semilogy(n_eval, errors_l, label = 'linearization',linewidth = 1)

for i, (e, n) in enumerate(zip(e_mean_fix,n_eval_pre)):
    if i == 0:
        plt.semilogy(n, e, 'k', label = 'exploratory phase, fixed hyper parameters',linewidth = 2.5)    
    else:
        plt.semilogy(n, e, 'k',linewidth = 2.5)    

for i, (e, n) in enumerate(zip(e_mean_re,n_eval_pre)):
    if i == 0:
        plt.semilogy(n, e, 'C3', label = 'exploratory phase, re-estimate',linewidth = 2.5)
    else:
        plt.semilogy(n, e, 'C3',linewidth = 2.5)
        

error_new = np.array([errors_m, errors_l, errors_a])
error_new = np.sort(error_new, axis = 0)
error_upper = error_new[0,:]
error_lower = error_new[2,:]

plt.xlim(0,30)
plt.legend(loc=3, frameon = False).set_zorder(-1)
plt.show()

#f.savefig("figures/01_fig_2_presampled.pdf", bbox_inches='tight')

#for (e1,e2,n) in zip(e_0_fix, e_mean_re, n_eval_pre):
for i,n in enumerate(n_eval_pre):
    prefix = n[0]
    filename = "output/exp1_fig2_{}.data".format(prefix)
    plot_data = np.full((n.size, 7), np.nan)
    plot_data[:,0] = n
    plot_data[:,1] = e_mean_fix[i]
    plot_data[:,2] = e_0_fix[i]
    plot_data[:,3] = e_100_fix[i]
    plot_data[:,4] = e_mean_re[i]
    plot_data[:,5] = e_0_re[i]
    plot_data[:,6] = e_100_re[i]
    np.savetxt(filename, plot_data, '%2i %1.3e %1.3e %1.3e %1.3e %1.3e %1.3e')

#%% Plot 3: Investigate sensitivity to field parameter prior

ll_true = np.load('output/01_reference_ll.npy', allow_pickle = True)

content = np.load('output/01_main_ll.npz', allow_pickle = True)
n_eval = content['n_eval']
ll_m = content['ll_m']

content = np.load('output/01_prior_sensitivity_ll.npz', allow_pickle = True)
ll_p1 = content['ll_p1']
ll_p2 = content['ll_p2']
ll_p3 = content['ll_p3']

# compute errors from loglikelihoods
errors_m = bbi.compute_errors(ll_true, ll_m)
errors_p1 = bbi.compute_errors(ll_true, ll_p1)
errors_p2 = bbi.compute_errors(ll_true, ll_p2)
errors_p3 = bbi.compute_errors(ll_true, ll_p3)

# plot
f = plt.figure(figsize=(12,8))
plt.xlabel('Number of model evaluations')
plt.ylabel('Error (KL-divergence)')
plt.xlim(0,30)
plt.semilogy(n_eval, errors_m, label = 'default',linewidth = 2.5)

plt.semilogy(n_eval, errors_p1, label = 'wide upper and lower',linewidth = 2.5)
plt.semilogy(n_eval, errors_p2, label = 'wide upper',linewidth = 2.5)
plt.semilogy(n_eval, errors_p3, label = 'narrow',linewidth = 2.5)

plt.legend()
plt.show()

#f.savefig("figures/01_fig_3_sensitivity.pdf", bbox_inches='tight')

plot_data = np.full((31,5), np.nan)
plot_data[:,0] = n_eval

plot_data[:,1] = errors_m
plot_data[:,2] = errors_p1
plot_data[:,3] = errors_p2
plot_data[:,4] = errors_p3

np.savetxt('output/exp1_fig3.data', plot_data, '%2i %1.3e %1.3e %1.3e %1.3e')
