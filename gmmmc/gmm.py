import numpy as np
import sklearn.mixture
from gmmmc.fastgmm import gmm_likelihood
import multiprocessing

def create_gmm(n_components, means, covariances, weights):
    gmm = sklearn.mixture.GMM(n_components)
    gmm.weights_ = weights
    gmm.covars_ = covariances
    gmm.means_ = means
    return gmm

class GMM():
    """Gaussian Mixture Model Distribution"""
    # TODO: input verification
    def __init__(self, means, covariances, weights, n_jobs=1):
        if len(covariances.shape) == 2:
            self.covariance_type = 'diag'
        else:
            raise NotImplementedError('Only diagonal covariance matrices supported')
        self.gmm = sklearn.mixture.GMM(n_components=len(weights))
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

    @means.setter
    def means(self, means):
        # must create GMM object again so that sklearn sample method will work correctly
        self.gmm = create_gmm(self.n_mixtures, means, self.gmm.covars_, self.gmm.weights_)

    @covars.setter
    def covars(self, covars):
        self.gmm = create_gmm(self.n_mixtures, self.gmm.means_, covars, self.gmm.weights_)

    @weights.setter
    def weights(self, weights):
        self.gmm = create_gmm(self.n_mixtures, self.gmm.means_, self.gmm.covars_, weights)

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
            return gmm_likelihood(X, self.means, self.covars, self.weights, n_jobs=n_jobs)

