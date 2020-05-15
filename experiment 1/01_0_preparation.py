# ===========================================================
# This code evaluates the heat-equation-model on a 51x51-grid
# and saves the outputs in a matrix
#
# Run time: Around two hours.
# ===========================================================

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

print('Starting computations')
t_start = time.time()
model_y = np.array( [model(point) for point in grid])
t_end = time.time()
print('Done.')
np.savez('01_model.npz', model_y=model_y, grid=grid)

t_elapsed = t_end - t_start
print(t_elapsed)

# This took about two hours!
