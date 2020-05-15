# ===========================================================
# This code will do the computations for the heat experiment.
#
# 1) Run the three new methods: map, linearized and average
# 2) Run 20 random matern-gpes
# 3) Run pre-sampled gpes
# 4) Check sensitivity to the prior
#
# ===========================================================

import numpy as np
import bbi
import time
import pickle


#%% 1) Run the three new methods: map, linearized and average

np.random.seed(0)

# Define problem

content = np.load('input/01_model.npz')
data_value = np.load('input/01_data.npy') # data imported from matlab (from earlier computations, same as in previous paper)

grid = content['grid']
model_y = content['model_y']
data = bbi.Data(data_value, 0.01)
    
problem = bbi.Problem(grid, model_y, data)

ll_true = problem.compute_loglikelihood()
np.save('output/01_reference_ll.npy', ll_true)

# Define gpe and start sequential design

n_iterations = 30
mix = bbi.MixMatern([0.01, 1], [0.01, 10], [0.5, 10], grid, n_output = 18)

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

np.savez('output/01_main_ll.npz',
         n_eval = n_eval,
         ll_m = ll_m, nodes_m = nodes_m, t_m = t_m,
         ll_l = ll_l, nodes_l = nodes_l, t_l = t_l,
         ll_a = ll_a, nodes_a = nodes_a, t_a = t_a,
         )

#%% 2) Run 21 random matern-gpes

np.random.seed(0)

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
    
np.savez('output/01_20_random_matern.npz', nodes_gpe = nodes_gpe, errors_gpe = errors_gpe)

#%% 3) Run design with exploratory phase (with and without re-estimation)

np.random.seed(0)


content = np.load('input/01_model.npz')
data_value = np.load('input/01_data.npy')

grid = content['grid']
model_y = content['model_y']
data = bbi.Data(data_value, 0.01)
    
problem = bbi.Problem(grid, model_y, data)

ll_true = problem.compute_loglikelihood()
np.save('output/01_reference_ll.npy', ll_true)

n_iterations = 30

n_repetitions = 21

starting_points = [5,10,15,20]

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

errors_pre_f = []
errors_pre_r = []
n_eval_pre = []
nodes_f = []
nodes_r = []

for i_n, n0 in enumerate(starting_points):
    print('Round {}, with {} starting points'.format(i_n+1, n0))
    
    inner_error_list_f = []
    inner_error_list_r = []
    inner_nodes_f = []
    inner_nodes_r = []
    for i_iter in range(n_repetitions):
        print('Starting repetition no {}.'.format(i_iter))
    
        # generate LHS-design of size n0 -> results in a "nodes" object
        nodes1 = bbi.Nodes()
        nodes2 = bbi.Nodes()
        if n0 > 0:
            idx_lhs = lhs(n0)
            nodes1.append(idx_lhs, model_y[idx_lhs,:])       
            nodes2.append(idx_lhs, model_y[idx_lhs,:])       

        # find corresponding map field        
        mix = bbi.MixMatern([0.01, 1], [0.01, 10], [0.5, 10], grid, n_output = 18)
        this_gpe = mix.get_map_field(nodes1)

        # start bbi.design_linearized(...) with fixed hyper parameters
        this_ll_f, this_nodes_f, this_n_eval_f = bbi.design_linearized(problem, this_gpe, n_iterations-n0, nodes1)
        # start bbi.design_map(), with re-estimation
        this_ll_r, this_nodes_r, this_n_eval_r = bbi.design_map(problem, mix, n_iterations-n0, nodes2)
        
        # append errors to errors-list
        this_error_f = bbi.compute_errors(ll_true, this_ll_f)
        inner_error_list_f.append(this_error_f)
        inner_nodes_f.append(this_nodes_f)
        
        this_error_r = bbi.compute_errors(ll_true, this_ll_r)
        inner_error_list_r.append(this_error_r)
        inner_nodes_r.append(this_nodes_r)
        
            
    n_eval_pre.append(this_n_eval_f) # after the loop, save n_eval (do only once per n0)
    errors_pre_f.append(np.array(inner_error_list_f)) # save complete list of errors
    errors_pre_r.append(np.array(inner_error_list_r)) # save complete list of errors
    nodes_f.append(inner_nodes_f)
    nodes_r.append(inner_nodes_r)    
    
t_end = time.time()

print('Presampled gpes - Time elapsed: {}'.format(t_end-t_start))        

# save
pickle.dump( n_eval_pre, open('output/01_pre50_lhs_n_eval.pkl','wb'))
pickle.dump( errors_pre_f, open('output/01_pre_errors_f.pkl','wb'))
pickle.dump( errors_pre_r, open('output/01_pre_errors_r.pkl','wb'))
pickle.dump( nodes_f, open('output/01_pre_nodes_f.pkl', 'wb'))
pickle.dump( nodes_r, open('output/01_pre_nodes_r.pkl', 'wb'))

#%% # 4) Check sensitivity to the prior

np.random.seed(0)

# Define problem

content = np.load('input/01_model.npz')
data_value = np.load('input/01_data.npy')

grid = content['grid']
model_y = content['model_y']
data = bbi.Data(data_value, 0.01)
    
problem = bbi.Problem(grid, model_y, data)

# Define gpe and start sequential design

n_iterations = 30
# mix 1: Wider bounds (by factor of 10 for both upper and lower bounds)
mix1 = bbi.MixMatern([0.001, 10], [0.001, 100], [0.5, 10], grid, n_output = 18)
# mix 2: Wider upper bounds (multiply upper bounds by factor of 10)
mix2 = bbi.MixMatern([0.01, 10], [0.01, 100], [0.5, 10], grid, n_output = 18)
# mix 3: Very narrow bounds, possibly a bit off
mix3 = bbi.MixMatern([0.01, 0.1], [0.01, 0.1], [0.5, 10], grid, n_output = 18)

t_start = time.time()
ll_p1, nodes_p1, n_eval = bbi.design_map(problem, mix1, n_iterations)
ll_p2, nodes_p2, _      = bbi.design_map(problem, mix2, n_iterations)
ll_p3, nodes_p3, _      = bbi.design_map(problem, mix3, n_iterations)
t_end = time.time()

print('Prior Sensitivity - Time elapsed: {}'.format(t_end-t_start))

np.savez('output/01_prior_sensitivity_ll.npz',
         n_eval = n_eval,
         ll_p1 = ll_p1, nodes_p1 = nodes_p3,
         ll_p2 = ll_p2, nodes_p2 = nodes_p2,
         ll_p3 = ll_p3, nodes_p3 = nodes_p1,
         )

#%% # 5) Compare with "miracle"-Solution (find hyper parameters from other run)

# the very first map-run returned these parameters:
map_xi = np.array([ 1.05820584, -1.2937138, 1.21238416]);

(l, sigma_squared, nu) = mix.xi_to_parameters(map_xi) # map-parameters

t_start = time.time()
field = bbi.GpeMatern(l, sigma_squared, nu, grid, n_output=18)
ll_1, nodes_1, _ = bbi.design_linearized(problem, field, n_iterations)
t_end = time.time()

print('Miracle - Time elapsed: {}'.format(t_end-t_start))

np.savez('output/01_miracle.npz',
         n_eval = n_eval,
         ll_1 = ll_1, nodes_1 = nodes_1,
         )
