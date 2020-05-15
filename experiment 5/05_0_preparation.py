# ===========================================================
# This code evaluates the heat-equation-model on a 51x51-grid
# and saves the outputs in a matrix
# As opposed to the 01_0_preperation.py, this one saves the
# output on more position (on a 9x9 grid)
#
# Run time: Around two hours.
# ===========================================================

import sys
sys.path.insert(0, '../')

import heat_eq as he
import numpy as np
import time

#%%

grid_size = 51

x1 = np.linspace(0,1,grid_size)
x2 = np.linspace(0,1,grid_size)
xx1, xx2 = np.meshgrid(x1,x2)
grid = np.column_stack((xx1.flatten(), xx2.flatten()))

x_true = np.array([0.25, 0.25])
model = he.heat_equation
output_resolution = 9 # measurement on a 9x9 grid, results in 9x9x2=162 values

y_true = model(x_true, output_resolution)

print('Starting computations')
t_start = time.time()
model_y = np.array( [model(point, output_resolution) for point in grid])
t_end = time.time()
print('Done.')
np.savez('input/05_model.npz', model_y=model_y, grid=grid, y_true = y_true)

t_elapsed = t_end - t_start
print(t_elapsed)

# This took 28 minutes (private computer)
