"""
Module bbi
for Bayesian Bayesian Inverse Problems solving
"""
from .field import Gpe, \
                   GpeMatern, \
                   GpeSquaredExponential, \
                   GpeSquaredExponentialOffset, \
                   GpeSquaredExponentialLinear, \
                   GpeSquaredExponentialSquare
from .field import MixMatern, \
                   MixSquaredExponential, \
                   MixSquaredExponentialLinear
from .field import FieldCollection
from .design import design_linearized, \
                    design_map, \
                    design_average, \
                    design_sampled, \
                    design_hybrid, \
                    design_random, \
                    design_heuristic, \
                    design_min_variance
from .mini_classes import Nodes, Data, Problem
from .mini_functions import compute_errors

del field
#del design
del mini_functions
del mini_classes
