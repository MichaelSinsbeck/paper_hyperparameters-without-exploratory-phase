#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for experiment 2
"""

import bbi
import matplotlib.pyplot as plt
import scipy.io as io
import numpy as np
import time
import pickle

def scale_input(array):
    return (array - array.mean(axis=0)[np.newaxis,:]) / array.std(axis=0)[np.newaxis,:]

n_days = 20

std_rel = 0.05
std_abs = 2e-7 * 1e7
d = np.load('data.npy')
data = bbi.Data(0, (d*std_rel + std_abs)**2)
n_output = data.value.size

grid = scale_input(np.load('sorption_input.npy'))

model_y = np.load('sorption_output.npy') - d.T
n_sample = 51000

problem = bbi.Problem(grid, model_y, data)

n_subsample = 500

field = bbi.MixSquaredExponential([0.1, 10], [0.01, 1e4], n_output, anisotropy = 6)


ll_true = problem.compute_loglikelihood()

n_iterations = 70
n_repetitions = 51
n_random = 21
starting_points = [10,30]

#%% Run seq-des with expl-phase-free methods
np.random.seed(0)

errors_m = []
ll_m = []
nodes_m = []

t_start = time.time()
for i_iter in range(n_repetitions):
    ll, nodes, n_eval = bbi.design_map(problem, field, n_iterations, n_subsample=n_subsample)
    errors_m.append(bbi.compute_errors(ll_true, ll))
    ll_m.append(ll)
    nodes_m.append(nodes)
    
t_m = time.time() - t_start

#ll, nodes, n_eval = bbi.design_linearized(problem, field, n_iterations, n_subsample=n_subsample)
#errors_m.append(bbi.compute_errors(ll_true, ll))

#ll, nodes, n_eval = bbi.design_map(problem, field, n_iterations, n_subsample=n_subsample)
#errors_m.append(bbi.compute_errors(ll_true, ll))

np.savez('output/02_main_ll.npz',
         n_eval = n_eval,
         ll_m = ll_m
         )

np.savez('output/02_main.npz',
         n_eval = n_eval,
         nodes_m = nodes_m, 
         errors_m = errors_m, 
         t_m = t_m,
         )

#%% Run 21 random squared-exponential-gpes

np.random.seed(0)


nodes_gpe = []
ll_gpe = []
errors_gpe = []

for i in range(n_random):
    this_gpe = field.draw_gpe()
    this_ll, this_nodes, _ = bbi.design_linearized(problem, this_gpe, n_iterations, n_subsample=n_subsample)
    nodes_gpe.append(this_nodes)
    ll_gpe.append(this_ll)
    errors_gpe.append(bbi.compute_errors(ll_true, this_ll))
    

np.savez('output/02_21_random_se.npz', 
         nodes_gpe = nodes_gpe, 
         #ll_gpe = ll_gpe, 
         errors_gpe = errors_gpe)

#%% Run design with exploratory phase (with and without re-estimation)

np.random.seed(0)

# f - fixed hyperparameters, r - re-estimation
errors_f = []
errors_r = []
nodes_f = []
nodes_r = []
n_eval_pre = []


for i_n, n0 in enumerate(starting_points):
    print('Round {} with {} starting points'.format(i_n+1,n0))
    
    inner_error_list_f = []
    inner_error_list_r = []
    inner_nodes_f = []
    inner_nodes_r = []
    
    for i_iter in range(n_repetitions):
        print('Starting repetition no {}'.format(i_iter))
        # generate initial design (here random)
        nodes1 = bbi.Nodes()
        nodes2 = bbi.Nodes()
        #TODO
        if n0 > 0:
            idx = np.random.choice(n_sample, n0, replace=False)
            nodes1.append(idx, model_y[idx,:])
            nodes2.append(idx, model_y[idx,:])
        
        # find corresponding map field
        field = bbi.MixSquaredExponential([0.1, 10], [0.01, 1e4], n_output, anisotropy = 6)
        this_gpe = field.get_map_field(nodes1, grid)
        
        # start sequential design with fixed hyper parameters
        this_ll_f, this_nodes_f, this_n_eval_f = bbi.design_linearized(problem, this_gpe, n_iterations-n0, nodes1, n_subsample = n_subsample)
        # start sequential design with re-estimation
        this_ll_r, this_nodes_r, this_n_eval_r = bbi.design_map(problem, field, n_iterations-n0, nodes2, n_subsample = n_subsample)
        # append errors to errors-list
        this_error_f = bbi.compute_errors(ll_true, this_ll_f)
        inner_error_list_f.append(this_error_f)
        inner_nodes_f.append(this_nodes_f)
        
        this_error_r = bbi.compute_errors(ll_true, this_ll_r)
        inner_error_list_r.append(this_error_r)
        inner_nodes_r.append(this_nodes_r)
    
    
    
    # n_eval, nodes, errors
    
    n_eval_pre.append(this_n_eval_f)
    errors_f.append(np.array(inner_error_list_f))
    errors_r.append(np.array(inner_error_list_r))
    nodes_f.append(inner_nodes_f)
    nodes_r.append(inner_nodes_r)
    
    # filename = 'output/02_pre_{}.npz'.format(n0)
    # np.savez(filename, 
    #          n_eval_pre = this_n_eval_f,
    #          nodes_f = inner_nodes_f,
    #          nodes_r = inner_nodes_r,
    #          errors_f = np.array(inner_error_list_f),
    #          errors_r = np.array(inner_error_list_r))

# save
pickle.dump( n_eval_pre, open('output/02_pre20_n_eval.pkl','wb'))
pickle.dump( errors_f, open('output/02_pre_errors_f.pkl','wb'))
pickle.dump( errors_r, open('output/02_pre_errors_r.pkl','wb'))
#pickle.dump( nodes_f, open('output/02_pre_nodes_f.pkl', 'wb'))
#pickle.dump( nodes_r, open('output/02_pre_nodes_r.pkl', 'wb'))


#%% Determine miracle-parameters by hand

np.random.seed(0)

errors_mir = []
ll_mir = []
nodes_mir = []

for i_iter in range(n_repetitions):
    
    gpe = bbi.GpeSquaredExponential([40, 350, 150, 22, 8, 1.5], 530, n_output)

    ll, nodes, _ = bbi.design_linearized(problem, gpe, n_iterations, n_subsample = n_subsample)
    errors_mir.append(bbi.compute_errors(ll_true, ll))

errors_mir = np.array(errors_mir)

np.save('output/02_error_miracle.npy', errors_mir)
