#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for random fields
Currently contains classes for three field types:
    1) Gpe - Gaussian Process Emulator. Comes with two inherited classes:
       - GpeMatern
       - GpeSquaredExponential
    2) FieldCollection - Gaussian Mixture model of finite number of Gpes
    3) Mix - A continuous mixture model of Gpes. Based on an abstract base
             class, there are two usable subclasses:
       - MixMatern
       - MixSquaredExponential
"""
import numpy as np
import emcee
from scipy import special
from scipy import linalg
from scipy.optimize import minimize, basinhopping
from scipy.stats import norm
from scipy.stats import multivariate_normal


class Gpe:
    # note: Gpe assumes that all components have the same covariance matrix, so
    # the covariance matrix is "flat".
    # The Case of different covariance matrices for all components only occur
    # inside the design_linear and can only be caused by a FieldCollection.
    # At least currently
    def __init__(self, m, c):
        self.m = m
        self.c = c
        self.std = np.sqrt(np.diag(c))
        self.n_sample = m.shape[0]
        self.n_output = m.shape[1]

    @classmethod
    def squared_exponential(cls, l, sigma_squared, grid, n_output=1):
        m = zero_mean(grid, n_output)
        c = covariance_squared_exponential(l, sigma_squared, grid)
        return cls(m, c)
    
    @classmethod
    def squared_exponential_offset(cls, l, sigma_squared, offset, grid, n_output=1):
        m = zero_mean(grid, n_output)
        c = covariance_squared_exponential_offset(l, sigma_squared, offset, grid)
        return cls(m, c)

    @classmethod
    def matern(cls, l, sigma_squared, nu, grid, n_output=1):
        m = zero_mean(grid, n_output)
        c = covariance_matern(l, sigma_squared, nu, grid)
        return cls(m, c)

    def draw_realization(self):
        zero_mean = np.zeros(self.n_sample)
        realization = np.random.multivariate_normal(zero_mean, self.c, size = self.n_output).T
        realization += self.m       
#        realization = np.zeros((self.n_sample, self.n_output))
#        for i in range(self.n_output):
#            this_m = self.m[:, i].flatten()
#            this_c = self.c
#            this_component = np.random.multivariate_normal(this_m, this_c)
#            realization[:, i] = this_component
        return realization
    
    def draw_realization_at_index(self, idx, size=1):
        return np.random.randn(size, self.n_output) * self.std[idx] + self.m[[idx]]

    def quadrature_at_index(self, idx, size=1):
        # Warning: by resetting the seed, the criterion will get smooth,
        # but it might have unforseen bad consequences. Better fix this in the
        # future

        # Warning: This function returns a symmetric realization: So that the
        # mean of the sample is guaranteed to be zero (antithetic sampling)
        half_size = size//2
        np.random.seed(0)
        random_numbers = np.random.randn(
            half_size, self.n_output) * self.std[idx]
        part_0 = self.m[[idx]]  # add mean itself as well
        part_1 = random_numbers + self.m[[idx]]
        part_2 = -random_numbers + self.m[[idx]]

        weights = np.ones(2*half_size+1)

        return np.concatenate((part_0, part_1, part_2)), weights

    def y_list(self, idx, size=51):
        uniform_grid = np.linspace(1/(2*size), 1-1/(2*size), size)
        normal_grid = norm.ppf(uniform_grid)[:, np.newaxis]
        y_list = normal_grid * self.std[idx] + self.m[np.newaxis, idx]
        return y_list

    def condition_to(self, nodes):
        n_points = nodes.idx.size
        if n_points == 0:
            return Gpe(self.m, self.c)

        Q = self.c[nodes.idx[:, np.newaxis], nodes.idx]
        q = self.c[:, nodes.idx]
        deviation = nodes.y.reshape(n_points, -1) - self.m[nodes.idx]

        R = linalg.cholesky(Q)
        r = np.linalg.solve(R.T, q.T).T

        c = self.c - r@r.T
        m = self.m + np.linalg.solve(R, r.T).T @ deviation

        # some correction for numerical errors
        # 1) remove negative diagonal entries from c
        n_sample = c.shape[0]
        diag_index = np.eye(n_sample, dtype=bool)
        negative_index = (c < 0)
        c[diag_index & negative_index] = 0
        # 2) set rows and colums of updated points to zero
        c[nodes.idx, :] = 0
        c[:, nodes.idx] = 0
        # 3) enforce interpolation
        m[nodes.idx, :] = nodes.y

        return Gpe(m, c)

    def compute_node_loglikelihood(self, nodes):
        n_nodes = nodes.idx.size
        if n_nodes == 0:
            return 0

        n_output = self.m.shape[1]
        loglikelihood = 0.
        for i_output in range(n_output):
            this_y = nodes.y[:, i_output]
            this_m = self.m[nodes.idx, i_output]
            this_c = self.c[nodes.idx[:, np.newaxis], nodes.idx]
            loglikelihood = loglikelihood + log_mvnpdf(this_y, this_m, this_c)
            #loglikelihood = loglikelihood + multivariate_normal.logpdf(this_y, this_m, this_c)
        return loglikelihood

    def estimate_componentwise_likelihood(self, data):
        return estimate_componentwise_likelihood_gpe(self, data)

    def estimate_likelihood(self, data):
        return estimate_likelihood_gpe(self, data)

    def estimate_likelihood_linearized(self, data):
        return estimate_likelihood_gpe(self, data)

    def estimate_loglikelihood(self, data):
        return estimate_loglikelihood_gpe(self, data)


def estimate_componentwise_likelihood_gpe(field, data):
    var_field = extract_variance(field.c)
    var_nu = var_field + data.var
    likelihoods = np.full((field.n_sample, field.n_output), np.nan)
    for i_output in range(field.n_output):
        this_l = 1./np.sqrt(2*np.pi*var_nu[:, i_output]) * \
            np.exp(-(field.m[:, i_output]-data.value[i_output])
                   ** 2/(2*var_nu[:, i_output]))
        likelihoods[:, i_output] = this_l
    return likelihoods


def estimate_likelihood_gpe(field, data):
    var_field = extract_variance(field.c)
    var_nu = var_field + data.var

    likelihood = 1./np.sqrt(np.prod(2*np.pi*var_nu, axis=1)) * \
        np.exp(-np.sum((field.m-data.value)**2/(2*var_nu), axis=1))
    return likelihood


def estimate_loglikelihood_gpe(field, data):
    var_field = extract_variance(field.c)
    var_nu = var_field + data.var

    loglikelihood = -0.5*np.log(np.prod(2*np.pi*var_nu, axis=1)) - \
        np.sum((field.m-data.value)**2/(2*var_nu), axis=1)
    return loglikelihood


def extract_variance(c):
    c_is_2d = (c.ndim == 2)
    if c_is_2d:
        var_field = np.diag(c)[:, np.newaxis]
    else:
        n_sample = c.shape[0]
        n_output = c.shape[2]
        var_field = np.full((n_sample, n_output), np.nan)
        for i_output in range(n_output):
            var_field[:, i_output] = np.diag(c[:, :, i_output])
    return var_field


def zero_mean(grid, n_output):
    n_sample = grid.shape[0]
    return np.zeros([n_sample, n_output])


def add_nugget(c, variance):
    n_sample = c.shape[0]
    c += np.eye(n_sample) * variance


def log_mvnpdf(x, m, c):
    xc = x-m
    d = m.size
    const = -0.5*d*np.log(2*np.pi)
    term1 = -0.5 * np.sum(xc @ np.linalg.solve(c, xc))
    term2 = const - 0.5 * log_det(c)
    return term1 + term2


def log_det(A):
    U = linalg.cholesky(A)
    return 2*np.sum(np.log(np.diag(U)))


def covariance_squared_exponential(l, sigma_squared, grid):
    # l may be a scalar or a vector of size n_input
    squared_distances = (((grid[:, np.newaxis, :] - grid)/l)**2).sum(2)
    c = np.exp(- squared_distances) * sigma_squared
    
    add_nugget(c, sigma_squared * 1e-9)
    #add_nugget(c, sigma_squared * 1e-7) # use this, if optimizing the parameters leads to
    # an error
    return c


def covariance_squared_exponential_offset(l, sigma_squared, offset, grid):
    c = covariance_squared_exponential(l, sigma_squared, grid)
    c += offset * sigma_squared
    return c

def covariance_squared_exponential_linear(l, sigma_squared, offset, slope, grid, centers = None):
    if centers is None:
        centers = np.mean(grid, axis = 0)
    c = covariance_squared_exponential_offset(l, sigma_squared, offset, grid)    
    line = slope * (grid-centers)/l
    c += sigma_squared * line@line.T
    return c

def covariance_squared_exponential_squared(l, sigma_squared, offset, slope, square_trend, grid, centers= None):
    if centers is None:
        centers = np.mean(grid, axis =0)
    c = covariance_squared_exponential_linear(l, sigma_squared, offset, slope, grid, centers)
    coord_normalized = (grid-centers)/l
    n_dim = grid.shape[1]
    parabolas = []
    for i in range(n_dim):
        for j in range(i, n_dim):
            parabolas.append(coord_normalized[:, i]*coord_normalized[:,j])
    parabolas = np.array(parabolas)
    c += sigma_squared * square_trend * parabolas.T@parabolas
    return c

def covariance_matern(l, sigma_squared, nu, grid):
    n_sample = grid.shape[0]

    distances = np.sqrt((((grid[:, np.newaxis, :] - grid)/l)**2).sum(2))
    d_unique, indices = np.unique(distances, return_inverse=True)
    # np.unique helps speed this up, because special.kv is expensive

    c_unique = np.full_like(d_unique, sigma_squared)

    mask = (d_unique != 0)
    C1 = 1 / (special.gamma(nu) * 2**(nu-1))
    C2 = np.sqrt(2*nu) * d_unique[mask]
    c_unique[mask] = C1 * (C2**nu) * special.kv(nu, C2) * sigma_squared

    c = c_unique[indices.reshape(n_sample, n_sample)]
    #add_nugget(c, sigma_squared * 1e-9)
    add_nugget(c, sigma_squared * 1e-7)
    return c


class GpeMatern(Gpe):
    """
    A Gpe with Matern covariance. Requires three parameters:
        1) l - correlation length
        2) sigma_squared - variance of the field
        3) nu - smoothness parameter
    """

    def __init__(self, l, sigma_squared, nu, grid, n_output=1):
        m = zero_mean(grid, n_output)
        c = covariance_matern(l, sigma_squared, nu, grid)
        super().__init__(m, c)


class GpeSquaredExponential(Gpe):
    """
    A Gpe with Squred Exponential (Gaussian Bell, Radial Basis function) covariance.
    Requires two parameters:
        1) l - correlation length
        2) sigma_squared - variance of the field
    """

    def __init__(self, l, sigma_squared, grid, n_output=1):
        m = zero_mean(grid, n_output)
        c = covariance_squared_exponential(l, sigma_squared, grid)
        super().__init__(m, c)


class GpeSquaredExponentialOffset(Gpe):
    """
    A Gpe with Squred Exponential (Gaussian Bell, Radial Basis function) covariance.
    Requires two parameters:
        1) l - correlation length
        2) sigma_squared - variance of the field
    """

    def __init__(self, l, sigma_squared, offset, grid, n_output=1):
        m = zero_mean(grid, n_output)
        c = covariance_squared_exponential_offset(l, sigma_squared, offset, grid)
        super().__init__(m, c)


class GpeSquaredExponentialLinear(Gpe):
    """
    A Gpe with Squred Exponential (Gaussian Bell, Radial Basis function) covariance.
    Requires two parameters:
        1) l - correlation length
        2) sigma_squared - variance of the field
    """

    def __init__(self, l, sigma_squared, offset, slope, grid, n_output=1):
        m = zero_mean(grid, n_output)
        c = covariance_squared_exponential_linear(l, sigma_squared, offset, slope, grid)
        super().__init__(m, c)

class GpeSquaredExponentialSquare(Gpe):
    """
    A Gpe with Squred Exponential (Gaussian Bell, Radial Basis function) covariance.
    Requires two parameters:
        1) l - correlation length
        2) sigma_squared - variance of the field
    """

    def __init__(self, l, sigma_squared, offset, slope, square_trend, grid, n_output=1):
        m = zero_mean(grid, n_output)
        c = covariance_squared_exponential_squared(l, sigma_squared, offset, slope, square_trend, grid)
        super().__init__(m, c)

class FieldCollection:
    """
    A Field mix is a collection of GPEs with weights.
    They have a number of uses:
        1) In design_vanilla they are used to linearize a gaussian mixture
           model. This is done by using the overall m and c over the mix.
        2) In design_map, they are used as a collection of fields.
           If no weights are given, map reverts to ml
        3) In design_average, they are again used as a collection of
           fields.
    """

    def __init__(self, subfields, weights=None):
        self.n_fields = len(subfields)
        self.subfields = subfields
        if weights is None:
            weights = np.ones(self.n_fields)
        else:
            weights = np.array(weights)
        self.weights = weights / weights.sum()
        self.update_m_and_c()

        self.n_sample = self.m.shape[0]
        self.n_output = self.m.shape[1]

    def update_m_and_c(self):
        self.m = np.zeros_like(self.subfields[0].m)
        for field, weight in zip(self.subfields, self.weights):
            self.m += weight * field.m

        n_output = self.m.shape[1]
        n_sample = self.m.shape[0]
        self.c = np.zeros((n_sample, n_sample, n_output))

        for i_output in range(n_output):
            for field, weight in zip(self.subfields, self.weights):
                mean_difference = field.m[:, i_output] - self.m[:, i_output]
                self.c[:, :, i_output] += weight * field.c
                self.c[:, :, i_output] += weight * \
                    np.outer(mean_difference, mean_difference)

    def draw_gpe(self):
        sum_weights = self.weights.cumsum()
        i_field = (sum_weights < np.random.uniform()).sum()
        return self.subfields[i_field]

    def draw_realization(self):
        return self.draw_gpe().draw_realization()

    def quadrature_at_index(self, idx, size=1):
        y_list = [self.draw_gpe().draw_realization_at_index(idx)
                  for i in range(size)]
        y_list = np.array(y_list)

        weights = np.ones(size)
        return y_list, weights
        # idea for later:
        # sort subfields by weight
        # pick largest weighted subfields up until cumsum is 0.99
        # run "draw_realization_at_index" on each subfield
        # collect everything and save weights
        pass

    def draw_realization_at_index2(self, idx, size=1):
        # Warning: by resetting the seed, the criterion will get smooth,
        # but it might have unforseen bad consequences. Better fix this in the
        # future

        # Warning: This function returns a symmetric realization: So that the
        # mean of the sample is guaranteed to be zero (antithetic sampling)
        half_size = size//2
        np.random.seed(0)
        random_numbers = np.random.randn(
            half_size, self.n_output) * self.std[idx]
        part_0 = self.m[[idx]]  # add mean itself as well
        part_1 = random_numbers + self.m[[idx]]
        part_2 = -random_numbers + self.m[[idx]]

        weights = np.ones(2*half_size+1)

        return np.concatenate((part_0, part_1, part_2)), weights

    def condition_to(self, nodes):
        new_subfields = [field.condition_to(nodes) for field in self.subfields]
        log_weights = [field.compute_node_loglikelihood(
            nodes) for field in self.subfields]

        log_weights = np.array(log_weights)
        log_weights = log_weights - log_weights.max()
        new_weights = np.exp(log_weights)

        return FieldCollection(new_subfields, new_weights)

    def estimate_likelihood(self, data):
        likelihoods = [field.estimate_likelihood(
            data) for field in self.subfields]
        likelihoods = np.array(likelihoods).T
        return likelihoods @ self.weights

    def estimate_likelihood_linearized(self, data):
        return estimate_likelihood_gpe(self, data)

    def estimate_loglikelihood(self, data):
        return np.log(self.estimate_likelihood(data))

    def get_map_field(self, nodes=None):
        if nodes is None:
            posterior_weights = self.weights
        else:
            loglikelihoods = [field.compute_node_loglikelihood(
                nodes) for field in self.subfields]
            loglikelihoods = np.array(loglikelihoods)
            loglikelihoods -= loglikelihoods.max()
            posterior_weights = self.weights * np.exp(loglikelihoods)
        idx_map = posterior_weights.argmax()
        return self.subfields[idx_map]

    def is_almost_gpe(self, threshold=0.95):
        return self.weights.max() > threshold


class AbstractMix:
    def __init__(self, grid, n_output, lb, ub, optimizer = 'local'):
        self.grid = grid
        self.n_output = n_output
        self.n_sample = grid.shape[0]
        self.n_input = grid.shape[1]
        
        self.mu = (0.5 * (lb + ub)).flatten()
        self.sigma = ((ub - lb)/4).flatten()
        self.n_parameters = lb.size
        self.reset_starting_point()
        self.optimizer = optimizer
        
        self.n_walkers = 24

    def process_anisotropy(self, l, anisotropic, grid):
        l = np.atleast_2d(l).T
        self.anisotropic = anisotropic
        if anisotropic:
            # if l is "flat", blow it up into matrix shape
            n_input = grid.shape[1]
            l = l * np.ones((2,n_input))
            
        self.n_l = l.shape[1]
        return l

    def condition_to(self, nodes):
        n_walkers = self.n_walkers
        log_posterior = self.create_log_posterior(nodes)
        sampler = emcee.EnsembleSampler(
            n_walkers, self.n_parameters, log_posterior)

        # starting positions around MAP-value
        #xi_map = self.get_map_xi(nodes)
        #self.xi_map = xi_map
        #noise = np.random.normal(
        #    scale=0.001, size=(n_walkers, self.n_parameters))
        #p0 = xi_map[np.newaxis, :] + noise
        
        # completely random starting positions
        p0 = norm.rvs(size=(n_walkers, self.n_parameters))

        print('Starting MCMC...')
        pos, prob, state = sampler.run_mcmc(p0, 1000)
        sampler.reset()
        sampler.run_mcmc(pos, 1)
        print('done')
        self.xi_sample = sampler.flatchain

        m = zero_mean(self.grid, self.n_output)
        subfields = [Gpe(m, self.cov(xi, self.grid)) for xi in self.xi_sample]

        cond_subfields = [field.condition_to(nodes) for field in subfields]

        
        return FieldCollection(cond_subfields)

    def get_map_xi(self, nodes, start_from_previous = False):

        log_posterior_fun = self.create_log_posterior(nodes)

        def obj_fun(xi):
            return -log_posterior_fun(xi)
        
        if start_from_previous:
            starting_point = self.previous_starting_point
        else:
            starting_point = self.get_prior_center()        
        
        if self.optimizer == 'global':
            opt = basinhopping
            print('Using global optimization algorithm')
        else:
            opt = minimize
        
        result = opt(obj_fun, starting_point)
        #result = basinhopping(obj_fun, starting_point)
        #result = minimize(obj_fun, starting_point)
        x = result.x
        self.previous_starting_point = x
        return x

    def get_map_field(self, nodes, start_from_previous = True):
        map_xi = self.get_map_xi(nodes, start_from_previous)
        #print(map_xi)
        c = self.cov(map_xi, self.grid)
        m = zero_mean(self.grid, self.n_output)
        return Gpe(m, c)

    def get_prior_center(self):
        return np.zeros(self.n_parameters)
    
    def reset_starting_point(self):
        self.previous_starting_point = self.get_prior_center()

    def create_log_posterior(self, nodes):
        n_nodes = nodes.idx.size
        if n_nodes == 0:
            def log_posterior(xi):
                return self.log_prior(xi)
        else:
            def log_posterior(xi):
                try:
                    log_likelihood = self.node_loglikelihood(xi, nodes)
                    log_prior = self.log_prior(xi)
                    value = log_likelihood + log_prior
                except (np.linalg.LinAlgError, ValueError):
                    value = -np.inf
                return value

        return log_posterior

    def log_prior(self, xi):
        return np.sum(norm.logpdf(xi))

    def draw_realization(self):
        this_gpe = self.draw_gpe()
        return this_gpe.draw_realization()

    def draw_gpe(self):
        this_xi = norm.rvs(size=self.n_parameters)
        this_c = self.cov(this_xi, self.grid)
        this_m = np.zeros((self.grid.shape[0], self.n_output))
        return Gpe(this_m, this_c)

    def node_loglikelihood(self, xi, nodes):
        subgrid = self.grid[nodes.idx, :]
        c = self.cov(xi, subgrid)
        m = np.zeros(nodes.idx.size)
        # geht das hier?
        # loglikelihood = np.sum([log_mvnpdf(y, m, c) for y in nodes.y.T])

        loglikelihood = 0
        # todo: Das hier testen:
        for y in nodes.y.T:
            loglikelihood += log_mvnpdf(y, m, c)
            #loglikelihood += multivariate_normal.logpdf(y, m, c)
        # for i_output in range(self.n_output):
        #    y = nodes.y[:, i_output]
        #    loglikelihood += log_mvnpdf(y, m, c)
        return loglikelihood

    def parameters_to_xi(self, parameters):
        return (np.log(parameters) - self.mu) / self.sigma

    def xi_to_parameters(self, xi):
        return np.exp(self.sigma * xi + self.mu)

    def cov(self, xi, subgrid):
        raise NotImplementedError(
            'Function cov not implemented. Please use MixMatern or MixSquaredExponential')


class MixMatern(AbstractMix):
    def __init__(self, l, sigma_squared, nu, grid, n_output, anisotropic = False, optimizer = 'local'):
        
        l = self.process_anisotropy(l, anisotropic, grid)
        sigma_squared = np.array(sigma_squared)
        nu = np.array(nu)

        lb = np.log(np.column_stack((l[[0]], sigma_squared[0], nu[0])))
        ub = np.log(np.column_stack((l[[1]], sigma_squared[1], nu[1])))
        
        super().__init__(grid, n_output, lb, ub, optimizer)

    def cov(self, xi, subgrid):
        parameters = self.xi_to_parameters(xi)
        l = parameters[:self.n_l]
        sigma_squared, nu = parameters[self.n_l:]

        return covariance_matern(l, sigma_squared, nu, subgrid)


class MixSquaredExponential(AbstractMix):
    def __init__(self, l, sigma_squared, grid, n_output, anisotropic = False, optimizer = 'local'):

        l = self.process_anisotropy(l, anisotropic, grid)
        sigma_squared = np.array(sigma_squared)

        lb = np.log(np.column_stack((l[[0]], sigma_squared[0])))
        ub = np.log(np.column_stack((l[[1]], sigma_squared[1])))

        super().__init__(grid, n_output, lb, ub, optimizer)

    def cov(self, xi, subgrid):
        #l, sigma_squared = self.xi_to_parameters(xi)
        parameters = self.xi_to_parameters(xi)
        l = parameters[:self.n_l]
        sigma_squared = parameters[self.n_l:]
        
        return covariance_squared_exponential(l, sigma_squared, subgrid)


class MixSquaredExponentialLinear(AbstractMix):
    def __init__(self, l, sigma_squared, offset, slope, grid, n_output, anisotropic = False):

        l = self.process_anisotropy(l, anisotropic, grid)            
        sigma_squared = np.array(sigma_squared)
        offset = np.array(offset)
        slope = np.array(slope)

        lb = np.log(np.column_stack((l[[0]], sigma_squared[0], offset[0], slope[0])))
        ub = np.log(np.column_stack((l[[1]], sigma_squared[1], offset[1], slope[1])))

        super().__init__(grid, n_output, lb, ub)
        
        self.centers = np.mean(grid, axis = 0) # used for centering the linear trend


    def cov(self, xi, subgrid):
        parameters = self.xi_to_parameters(xi)
        l = parameters[:self.n_l]
        sigma_squared, offset, slope = parameters[self.n_l:]
        return covariance_squared_exponential_linear(l, sigma_squared, offset, slope, subgrid, self.centers)

    # estimate_likelihood
    # estimate_likelihood_linearized
    # estimate_loglikelihood
    # get_map_field
