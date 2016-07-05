__author__ = 'jeremyma'
import numpy as np
from numba import jit
import sklearn.mixture
from time import time

@jit
def log_likelihood_numba(X, num_components, means, diagCovs, weights):
    evaluator = sklearn.mixture.GMM(n_components = num_components)
    evaluator.weights_ = weights
    evaluator.covars_ = diagCovs
    evaluator.means_ = means

    return np.sum(evaluator.score(X))

def log_likelihood_sklearn(X, n_mixtures, means, diagCovs, weights):
    evaluator = sklearn.mixture.GMM(n_components = n_mixtures)
    evaluator.weights_ = weights
    evaluator.covars_ = diagCovs
    evaluator.means_ = means

    return np.sum(evaluator.score(X))


if __name__ == '__main__':

    X = np.random.random((100000,1000))

    num_components = 1000
    means = np.random.random((1000,1000))
    diagCovs = np.random.random((1000,1000))
    weights = np.random.random(1000)

    start = time()
    a = log_likelihood_numba(X, num_components, means, diagCovs, weights)
    end = time()
    print end-start

    start = time()
    b = log_likelihood_sklearn(X, num_components, means, diagCovs, weights)
    end = time()
    print end-start