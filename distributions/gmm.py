import numpy as np
import sklearn.mixture
from optimisation import fast_likelihood
import multiprocessing

class GMM():
    """Gaussian Mixture Model Distribution"""

    def __init__(self, means, weights, covariances, n_jobs=1):
        if len(covariances.shape) == 2:
            self.covariance_type = 'diag'
        else:
            self.covariance_type = 'full'
        self.gmm = sklearn.mixture.GMM(n_components=len(weights), covariance_type=self.covariance_type)
        self.gmm.weights_ = weights
        self.gmm.covars_ = covariances
        self.gmm.means_ = means
        self.n_mixtures = len(weights)
        try:
            self.n_features = means.shape[1]
        except:
            raise ValueError("Means array must be 2 dimensional")


    @property
    def means(self):
        return self.gmm.means_

    @property
    def covars(self):
        return self.gmm.covars_

    @property
    def weights(self):
        return self.gmm.weights_


    def sample(self, n_samples):
        return self.gmm.sample(n_samples)

    def log_likelihood(self, X, n_jobs=1):

        if n_jobs == 0:
            raise ValueError("n_jobs==0 has no meaning")
        elif n_jobs < 0:
            n_jobs = multiprocessing.cpu_count()
        else:
            n_jobs = n_jobs

        if n_jobs == 1:
            return np.sum(self.gmm.score(X))
        else:
            #
            return fast_likelihood.gmm_likelihood(X, self.means, self.covars, self.weights, n_jobs=n_jobs)

