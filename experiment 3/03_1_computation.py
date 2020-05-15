"""
Experiment 3: curve fitting.

This is the curve-fitting problem given by Schöniger et al:
    
A. Schöniger, T. Wöhling, L. Samaniego, and W. Nowak, Model selection on 
solid ground: Rigorous comparison of nine ways to evaluate Bayesian model 
evidence, Water Resources Research, 50 (2014),pp. 9484–9513

From this article, we only consider the nonlinear model (the cosine), not
the linear model.
"""

import bali as bl
import numpy as np
import matplotlib.pyplot as plt
import time

n_iter = 60
n_repetitions = 51
n_random = 21

np.random.seed(0)

# load data from disc and define model functions
content = np.load('input/curve_fitting.npz')
#x_linear = content['x_linear']
x_cosine = content['x_cosine']
data = content['data']
variance = content['variance']
grid = content['grid']

# Nonlinear model y = a*cos(b*x+c)+d
def f_cosine(p):
    return p[0] * np.cos(p[1] * grid + p[2]) + p[3]

model_cosine = bl.Model(x_cosine, f_cosine, variance)

problem = bl.Problem(model_cosine, data)

#%%
np.random.seed(0)

error_map = []
p_list = []
for i in range(n_repetitions):

    
    gpe = problem.suggest_gpe()
    
    sd = bl.SequentialDesign(problem, gpe)
    
    #sd.iterate(n_iter)
    for i in range(n_iter):
        sd.iterate(1)
        xi = sd.gpe[0].xi_list[0]
        p = sd.gpe[0].xi_to_parameters(xi)
        #print(p)
        p_list.append(p)
    
    error_map.append(sd.error_ll())
    
p_list = np.array(p_list)
error_map = np.array(error_map)


np.save('output/error_map.npy', error_map)

# %% Random guessing

np.random.seed(0)

error_random = []

field = problem.suggest_gpe()

for i_repetition in range(n_random): 
    params = field.xi_to_parameters(np.random.randn(5))
    
    gpe = bl.GpeSquaredExponential(params[:4], params[4], 15)
    
    sd = bl.SequentialDesign(problem, gpe)
    sd.iterate(n_iter)
    
    error_random.append(sd.error_ll())
    
error_random = np.array(error_random)
plt.semilogy(error_random.T)

np.save('output/error_random.npy', error_random)

# %% Miracle case for comparison
np.random.seed(0)

l = [[7., 7.], [0.88, 0.88], [2.9, 2.9], [11.5, 11.5]]
sigma2 = 15

error_miracle = []
all_params = []
for i_repetition in range(n_repetitions):
 
    
    gpe = bl.GpeSquaredExponential(l, sigma2, 15)
    
    sd = bl.SequentialDesign(problem, gpe)
    sd.iterate(n_iter)
    error_miracle.append(sd.error_ll())
    
error_miracle = np.array(error_miracle)

np.save('output/error_miracle.npy', error_miracle)

#%% Exploratory phases

np.random.seed(0)

error_expl = []

pre_sample = [20, 40]

for n_pre in pre_sample:
    for i_repetition in range(n_repetitions):
        gpe = problem.suggest_gpe()
        
        sd = bl.SequentialDesign(problem, gpe)
    
        sd.set_acquisition_function('min_variance')
        
        sd.iterate(n_pre)
        
        sd.set_acquisition_function('inverse')
        
    