__author__ = 'jeremyma'
import numpy as np
from numba import jit
import sklearn.mixture
from time import time
import sklearn

def log_likelihood(X, n_mixtures, means, diagCovs, weights):
    evaluator = sklearn.mixture.GMM(n_components = n_mixtures)
    evaluator.weights_ = weights
    evaluator.covars_ = diagCovs
    evaluator.means_ = means

    return np.sum(evaluator.score(X))