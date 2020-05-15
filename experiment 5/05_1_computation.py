# ===========================================================
# This code will do the computations for the heat experiment B.
#
# Note: Measurement noise is generated inside this script.
# In Experiment 1, this is not the case (for reproducability)
#
# 1) Define the problem
# 2) Run miracle
# 3) Run new methods (MAP, average, linearization)
# 4) Exploratory phase
#
# ===========================================================

import sys
sys.path.insert(0, '../')

import numpy as np
import bbi
import time
import pickle

import matplotlib.pyplot as plt

#%% 1) Define problem

np.random.seed(1)

# Define problem
sigma_squared = 0.01 # same as in paper

content = np.load('input/05_model.npz')
grid = content['grid']
model_y = content['model_y']
y_true = content['y_true']

noise = np.random.randn(y_true.size)
data_value = y_true + noise * np.sqrt(sigma_squared)

# center model around data
model_y = model_y - data_value[np.newaxis,:]
data_value = data_value - data_value

# define subindex to only use some of the measurement
# this allows us to create a "bimodal posterior"

#simple_idx = np.array([0,4,8,36,40,44,72,76,80]) # recreate previous experiment
simple_idx = np.arange(4,77,9) # nine wells on the horizontal center

# note: results in 05_model.npz are on a 9x9-grid. Add 81 to get
# horizontal center in second timestep
subidx = np.concatenate((simple_idx, simple_idx+81)) 
#subidx = np.array([1,4,7,10,13,16])
data_value = data_value[subidx]
model_y = model_y[:,subidx]

data = bbi.Data(data_value, sigma_squared)
problem = bbi.Problem(grid, model_y, data)
ll_true = problem.compute_loglikelihood()
#np.save('output/05_reference_ll.npy', ll_true)

f = plt.figure(figsize=(10,10))
x = np.linspace(0,1,51)
img = plt.contourf(x,x,np.exp(ll_true).reshape(51,51))
img.set_cmap('Blues')


#%% # 2) Run with "miracle"-Solution (find hyper parameters from other run)
n_iterations = 50
n_output = data_value.size
mix = bbi.MixMatern([0.01, 1], [0.01, 10], [0.5, 10], grid, n_output = n_output)

# the very first map-run returned these parameters:
map_xi = np.array([ 1.05820584, -1.2937138, 1.21238416]);

(l, sigma_squared, nu) = mix.xi_to_parameters(map_xi) # map-parameters

t_start = time.time()
field = bbi.GpeMatern(l, sigma_squared, nu, grid, n_output=n_output)
ll_1, nodes_1, n_eval = bbi.design_linearized(problem, field, n_iterations)
t_end = time.time()

print('Miracle - Time elapsed: {}'.format(t_end-t_start))

np.savez('output/05_miracle.npz',
         n_eval = n_eval,
         ll_1 = ll_1, nodes_1 = nodes_1,
         )

errors_1 = bbi.compute_errors(ll_true, ll_1)
#plt.semilogy(n_eval, errors_1)
#plt.show()

#f = plt.figure(figsize=(6,6))
#x = np.linspace(0,1,51)
#img = plt.contourf(x,x,np.exp(ll_true).reshape(51,51))
##img = plt.contourf(x,x,(ll_true).reshape(51,51))
#plt.plot(grid[nodes_1.idx,0], grid[nodes_1.idx,1],'k.')
#
#img.set_cmap('Blues')
#plt.show()
#
#
#f = plt.figure(figsize=(6,6))
#x = np.linspace(0,1,51)
#img = plt.contourf(x,x,np.exp(ll_1[:,-1]).reshape(51,51))
##img = plt.contourf(x,x,(ll_1[:,30]).reshape(51,51))
#
#img.set_cmap('Blues')
#plt.show()
#%% 3) Run the three new methods: map, linearized and average

# Define gpe and start sequential design

n_iterations = 50
mix = bbi.MixMatern([0.01, 1], [0.01, 10], [0.5, 10], grid, n_output = n_output)

t_start = time.time()
ll_m, nodes_m, n_eval = bbi.design_map(problem, mix, n_iterations)
t_end = time.time()
t_m = t_end-t_start
print('Design MAP - Time elapsed: {}'.format(t_m))

t_start = time.time()
ll_l, nodes_l, _      = bbi.design_linearized(problem, mix, n_iterations)
t_end = time.time()
t_l = t_end-t_start
print('Design Linearized - Time elapsed: {}'.format(t_l))

t_start = time.time()
ll_a, nodes_a, _      = bbi.design_average(problem, mix, n_iterations)
t_end = time.time()
t_a = t_end-t_start
print('Design Average - Time elapsed: {}'.format(t_a))


errors_m = bbi.compute_errors(ll_true, ll_m)
#plt.semilogy(n_eval,errors_m, label = 'MAP')
#plt.semilogy(n_eval,errors_1, label = 'miracle')
#plt.legend()
#plt.show()

np.savez('output/05_main_ll.npz',
         n_eval = n_eval,
         ll_m = ll_m, nodes_m = nodes_m, t_m = t_m,
         ll_l = ll_l, nodes_l = nodes_l, t_l = t_l,
         ll_a = ll_a, nodes_a = nodes_a, t_a = t_a,
         )


#f = plt.figure(figsize=(6,6))
#x = np.linspace(0,1,51)
#img = plt.contourf(x,x,np.exp(ll_m[:,-1]).reshape(51,51))
##img = plt.contourf(x,x,(ll_true).reshape(51,51))
#plt.plot(grid[nodes_m.idx,0], grid[nodes_1.idx,1],'k.')
#
#img.set_cmap('Blues')
#plt.show()

#%% 3) Run design with exploratory phase (with and without re-estimation)

np.random.seed(0)


n_iterations = 50
n_repetitions = 21

starting_points = [10,20]

t_start = time.time()

def lhs(size, resolution = 51): # create lhs-sample on grid (gridsize 51 is hard-coded)
    bounds = np.floor(np.linspace(0, resolution, size+1))
    idx_x = []
    idx_y = []
    for i in range(size):
        idx_x.append(np.random.randint(bounds[i], bounds[i+1]))
        idx_y.append(np.random.randint(bounds[i], bounds[i+1]))            
    
    idx_x = np.array(idx_x)
    idx_y = np.array(idx_y)
    
    np.random.shuffle(idx_x)
    return idx_x + resolution * idx_y

errors_pre_r = []
n_eval_pre = []
nodes_r = []

for i_n, n0 in enumerate(starting_points):
    print('Round {}, with {} starting points'.format(i_n+1, n0))
    
    inner_error_list_r = []
    inner_nodes_r = []
    for i_iter in range(n_repetitions):
        print('Starting repetition no {}.'.format(i_iter))
    
        # generate LHS-design of size n0 -> results in a "nodes" object
        nodes_1 = bbi.Nodes()
        if n0 > 0:
            idx_lhs = lhs(n0)
            nodes_1.append(idx_lhs, model_y[idx_lhs,:])       

        # find corresponding map field        
        mix = bbi.MixMatern([0.01, 1], [0.01, 10], [0.5, 10], grid, n_output = 18)

        # start bbi.design_map(), with re-estimation
        this_ll_r, this_nodes_r, this_n_eval_r = bbi.design_map(problem, mix, n_iterations-n0, nodes_1)
        
        # append errors to errors-list
        
        this_error_r = bbi.compute_errors(ll_true, this_ll_r)
        inner_error_list_r.append(this_error_r)
        inner_nodes_r.append(this_nodes_r)
        
            
    n_eval_pre.append(this_n_eval_r) # after the loop, save n_eval (do only once per n0)
    errors_pre_r.append(np.array(inner_error_list_r)) # save complete list of errors
    nodes_r.append(inner_nodes_r)    
    
t_end = time.time()

print('Presampled gpes - Time elapsed: {}'.format(t_end-t_start))        

# save
pickle.dump( n_eval_pre, open('output/05_pre50_lhs_n_eval.pkl','wb'))
pickle.dump( errors_pre_r, open('output/05_pre_errors_r.pkl','wb'))

#%% 4) Run 21 random matern-gpes


np.random.seed(0)
n_iterations = 50

n_gpe = 21
nodes_gpe = []
ll_gpe = []
errors_gpe = []

t_start = time.time()

for i in range(n_gpe):
    this_gpe = mix.draw_gpe()
    this_ll, this_nodes, _ = bbi.design_linearized(problem, this_gpe, n_iterations)
    nodes_gpe.append(this_nodes)
    ll_gpe.append(this_ll)
    errors_gpe.append(bbi.compute_errors(ll_true, this_ll))
    

t_end = time.time()

print('21 Random gpes - Time elapsed: {}'.format(t_end-t_start))    
    
np.savez('output/05_20_random_matern.npz', nodes_gpe = nodes_gpe, errors_gpe = errors_gpe)

