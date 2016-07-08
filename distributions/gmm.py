import numpy as np
import sklearn.mixture

class GMM():
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

    def log_likelihood(self, X):
        return np.sum(self.gmm.score(X))