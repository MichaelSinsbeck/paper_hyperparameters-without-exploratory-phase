"""
Run CO2-example using bali
"""

import bali as bl
import numpy as np
from scipy.io import loadmat

n_repetitions = 51
n_iter = 30
n_random = 21

pascal_per_bar = 1e5

content = np.load('input/co2_model.npz')
x = content['grid']
y = content['model_y'] / pascal_per_bar

content = loadmat('input/co2_data')
data_value = content['data'] /pascal_per_bar
data_std = content['std_measure'] / pascal_per_bar
data_var = data_std**2

data_value = data_value[:,1:].flatten()

model = bl.Model(x, y, data_var)
problem = bl.Problem(model, data_value)


#%% Run map-method
np.random.seed(0)
gpe = problem.suggest_gpe()

error_map = []
p_list = []
for i in range(n_repetitions):
    sd = bl.SequentialDesign(problem, gpe)
    #sd.iterate(n_iter)
    for i in range(n_iter):
        sd.iterate(1)
        xi = sd.gpe[0].xi_list[0]
        p = sd.gpe[0].xi_to_parameters(xi)
        #print(p)
        p_list.append(p)    
    
    error_map.append(sd.error_ll())
    
error_map = np.array(error_map)
p_list = np.array(p_list)
plt.semilogy(p_list)
#np.save('output/error_map.npy', error_map)

#%% Random guessing

np.random.seed(0)

error_random = []
field = problem.suggest_gpe()

for i_repetition in range(n_random): 
    params = field.xi_to_parameters(np.random.randn(3))
    
    gpe = bl.GpeSquaredExponential(params[:2], params[2], 10)
    
    sd = bl.SequentialDesign(problem, gpe)
    sd.iterate(n_iter)
    
    error_random.append(sd.error_ll())
    
error_random = np.array(error_random)
np.save('output/error_random.npy', error_random)

#%% Miracle case for comparison
np.random.seed(0)

l = [[1., 1.], [60, 60]]
sigma2 = 2e5


error_miracle = []

for i_repetition in range(n_repetitions):
    
    gpe = bl.GpeSquaredExponential(l, sigma2, 10)
    
    sd = bl.SequentialDesign(problem, gpe)
    sd.iterate(n_iter)
    
    error_miracle.append(sd.error_ll())
    
error_miracle = np.array(error_miracle)


np.save('output/error_miracle.npy', error_miracle)


#%% Miracle case with hand-picked parameters

np.random.seed(0)

n_x = x.shape[0]

l = [[1., 1.], [60, 60]]
sigma2 = 2e5

error_miracle = []


all_params = []
#for i_repetition in range(n_repetitions):   
gpe = bl.GpeSquaredExponential(l, sigma2, 10)
sd = bl.SequentialDesign(problem, gpe)
sd.iterate(n_iter)
error_miracle.append(sd.error_ll())
    
error_miracle = np.array(error_miracle)
plt.semilogy(error_miracle[0])
plt.semilogy(e_mean_map)
#all_params = np.array(all_params)

np.save('output/error_miracle.npy', error_miracle)
#np.save('output/params_miracle.npy', all_params)


