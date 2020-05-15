#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module helpers

Contains the following classes, mostly for shorter notation:
    1) Nodes - Sampling Points and the corresponding model response
    2) Data - Data plus Variance, part of an inverse problem
    3) Problem - An inverse problem, consists of grid, model and data
"""
import numpy as np


class Nodes:
    def __init__(self, idx=None, y=None):
        self.idx = np.empty((0), dtype=int)
        if not idx is None and not y is None:
            self.append(idx, y)

    def append(self, idx, y):
        if self.idx.size == 0:  # difference procedure, if nodes is empty
            n_entries = np.array(idx).size
            self.idx = np.append(self.idx, idx)
            self.y = np.array(y).reshape(n_entries, -1)
            self.n_output = self.y.shape[1]
        else:
            self.idx = np.append(self.idx, idx)
            self.y = np.append(self.y, y.reshape(-1, self.n_output), axis=0)

    def first(self, n):
        # returns the first n nodes for quick conditioning
        # e.g. field.conditionTo(nodes.first(4))
        subset = Nodes()
        if n > 0:
            subset.append(self.idx[0:n], self.y[0:n, :])
        return subset


class Data:
    def __init__(self, value, var):
        # both value and variance are made into 1d-vector
        self.value = np.array(value).flatten()
        n_output = self.value.size
        self.var = np.ones(n_output) * np.array(var).flatten()


class Problem:
    def __init__(self, grid, model, data):
        self.grid = grid
        self.model = model
        self.data = data

    def evaluate_model(self, index):
        if callable(self.model):
            input_value = self.grid[index, :]
            y = self.model(input_value)
        else:
            y = self.model[index, :]
        return y

    def compute_likelihood(self):
        if callable(self.model):
            model_y = self.run_model_everywhere()
        else:
            model_y = self.model

        data = self.data
        likelihood = 1/np.sqrt(np.prod(2*np.pi*data.var)) * \
            np.exp(-np.sum((model_y-data.value)**2/(2*data.var), axis=1))
        return likelihood

    def compute_loglikelihood(self):
        if callable(self.model):
            model_y = self.run_model_everywhere()
        else:
            model_y = self.model

        data = self.data
        loglikelihood = -0.5*np.log(np.prod(2*np.pi*data.var)) - \
            np.sum((model_y-data.value)**2/(2*data.var), axis=1)
        return loglikelihood

    def run_model_everywhere(self):
        n_sample = self.grid.shape[0]
        n_output = self.data.value.size
        model_y = np.full((n_sample, n_output), np.nan)
        for i in range(n_sample):
            this_x = self.grid[i, :]
            this_y = self.model(this_x)
            model_y[i, :] = this_y
        return model_y
