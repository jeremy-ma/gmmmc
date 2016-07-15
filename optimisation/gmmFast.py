import numpy as np
import sklearn.mixture
from scipy.stats import multivariate_normal
from scipy.misc import logsumexp
from fast_likelihood import gmm_likelihood
import multiprocessing

"""
cdef extern from 'math.h':
    double log(double x)

cdef double gmm_log_likelihood_diag(np.ndarray[double, ndim=2] X,
                             np.ndarray[double, ndim=2] means,
                             np.ndarray[double, ndim=1] weights,
                             np.ndarray[double, ndim=2] covars):
    cdef int i
    cdef int n_samples = X.shape[0]
    cdef int n_mixtures = means.shape[0]
    cdef np.ndarray[double, ndim=1] logprobs = np.empty(n_samples, dtype=np.double)
    cdef np.ndarray[double, ndim=1] logprobs_mixtures = np.empty(n_mixtures, dtype=np.double)
    cdef double prob
    for i in xrange(n_samples):
        for j in xrange(n_mixtures):
            logprobs_mixtures[j] = log(weights[j]) + multivariate_normal(mean=means[j], cov=covars[j]).logpdf(X[i])
        prob = logsumexp(logprobs_mixtures)
        logprobs[i] = prob
    return np.sum(logprobs)
"""
class GMM:
    """Gaussian Mixture Model Distribution"""

    def __init__(self, means, weights, covariances):
        if len(covariances.shape) == 2:
            self.covariance_type = 'diag'
        else:
            self.covariance_type = 'full'
        self.gmm = sklearn.mixture.GMM(n_components=len(weights), covariance_type=self.covariance_type)
        self.gmm.weights_ = weights
        self.gmm.covars_ = covariances
        self.gmm.means_ = means
        self.n_mixtures = len(weights)

    @property
    def means(self):
        return self.gmm.means_

    @property
    def covars(self):
        return self.gmm.covars_

    @property
    def weights(self):
        return self.gmm.weights_

    @property
    def n_mixtures(self):
        return self.n_mixtures

    def sample(self, n_samples):
        return self.gmm.sample(n_samples)

    def log_likelihood(self, X, n_jobs=1):
        if n_jobs == 0 or n_jobs < -1:
            raise ValueError('n_jobs must be valid')

        if n_jobs == -1:
            # use all available cores
            n_jobs = multiprocessing.cpu_count()

        return gmm_likelihood(X, self.means, self.covars, self.weights, n_jobs)