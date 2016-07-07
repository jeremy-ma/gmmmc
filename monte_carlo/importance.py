import numpy as np
from distributions import gmm
import scipy.stats

def gmm_mean_sample(X, true_weights, true_covars, n_runs=10000, n_mixtures=32):

    samples = np.array([np.random.uniform(low=-1, high=1, size=true_covars.shape) for _ in xrange(n_runs)])

    weights = np.zeros((n_runs))
    for run in xrange(n_runs):
        # because of uniform distribution
        weights[run] = gmm.log_likelihood(X, n_mixtures, samples[run], true_covars, true_weights)

    return samples, weights

