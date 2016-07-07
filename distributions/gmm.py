__author__ = 'jeremyma'
import numpy as np
from numba import jit
import sklearn.mixture
from time import time
from .distribution import Distribution


class GMM(Distribution):
    """Gaussian Mixture Model Distribution"""

    def __init__(self, means, weights, covariances):
        pass

    def sample(self, n_samples):
        pass

    def log_likelihood(self, X):
        pass

class Normal(Distribution):
    """multivariate normal"""
    def __init__(self, means, covariances):
        pass

    def sample(self, n_samples):
        pass

    def log_likelihood(self, X):
        pass

class Dirichlet(Distribution):
    """Dirichlet distribution"""
    def __init__(self, importances):
        pass

    def sample(self, n_samples):
        pass

    def log_likelihood(self, X):
        pass

def log_likelihood(X, n_mixtures, means, diagCovs, weights):
    evaluator = sklearn.mixture.GMM(n_components = n_mixtures)
    evaluator.weights_ = weights
    evaluator.covars_ = diagCovs
    evaluator.means_ = means

    return np.sum(evaluator.score(X))