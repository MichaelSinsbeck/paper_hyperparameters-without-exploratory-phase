#%% Check sensitivity to the prior
# This is not part of the git-repository


np.random.seed(0)

mix1 = bbi.MixSquaredExponential([0.01, 100], [0.001, 1e5], n_output, anisotropy = 6)

mix2 = bbi.MixSquaredExponential([0.1, 100], [0.01, 1e5], n_output, anisotropy = 6)

mix3 = bbi.MixSquaredExponential([0.1, 10], [0.01, 1e4], n_output, anisotropy = 6)


t_start = time.time()
ll_p1, nodes_p1, n_eval = bbi.design_map(problem, mix1, n_iterations, n_subsample=n_subsample)
ll_p2, nodes_p2, _      = bbi.design_map(problem, mix2, n_iterations, n_subsample=n_subsample)
ll_p3, nodes_p3, _      = bbi.design_map(problem, mix3, n_iterations, n_subsample=n_subsample)
t_end = time.time()

e1 = bbi.compute_errors(ll_true, ll_p1)
e2 = bbi.compute_errors(ll_true, ll_p2)
e3 = bbi.compute_errors(ll_true, ll_p3)

np.savez('output/02_prior_sensitivity_ll.npz',
         n_eval = n_eval,
         ll_p1 = ll_p1, 
         ll_p2 = ll_p2, 
         ll_p3 = ll_p3,
         )

plt.semilogy(n_eval,e1,n_eval,e2,n_eval,e3)
