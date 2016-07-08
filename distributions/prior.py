import numpy as np
import scipy
from .gmm import GMM

# TODO: Input verification
class GMMPrior():
    def __init__(self, means_prior, covars_prior, weights_prior):
        self.means_prior = means_prior
        self.covars_prior = covars_prior
        self.weights_prior = weights_prior

    def log_likelihood(self, gmm):
        logprob = 0
        logprob += self.means_prior.log_likelihood(gmm.means)
        logprob += self.covars_prior.log_likelihood(gmm.covars)
        logprob += self.weights_prior.log_likelihood(gmm.weights)
        return logprob

    def sample(self):
        return GMM(self.means_prior.sample(), self.weights_prior.sample(), self.covars_prior.sample())

class GMMMeansGaussianPrior():
    def __init__(self, prior_means, covariances):
        # shape should be (n_mixtures, n_features)
        self.means = prior_means
        self.covars = covariances

    def log_likelihood(self, means):
        # logsumexp probabilities
        pass

    def sample(self):
        # one at a time
        pass

class GMMMeansUniformPrior():
    """Uniform prior for means"""
    def __init__(self, low, high, n_mixtures, n_features):
        self.low = low
        self.high = high
        self.n_mixtures = n_mixtures
        self.n_features = n_features

    def log_likelihood(self, means):
        # just return some constant value
        return -0.5

    def sample(self):
        # sample means
        return np.random.uniform(self.low, self.high, size=(self.n_mixtures, self.n_features))

class GMMDiagCovarsUniformPrior():
    """Uniform Prior for diagonal covariances"""
    def __init__(self, low, high, n_mixtures, n_features):
        self.low = low
        self.high = high
        self.n_mixtures = n_mixtures
        self.n_features = n_features

    def log_likelihood(self, covars):
        if (covars < 0).any():
            return -np.inf
        else:
            # return some constant value
            return -0.5

    def sample(self):
        return np.random.uniform(self.low, self.high, size=(self.n_mixtures, self.n_features))

class GMMWeightsUniformPrior():
    """Uniform Prior for Weights (Dirichlet distribution)"""
    def __init__(self, n_mixtures):
        self.alpha = [1 for _ in xrange(n_mixtures)]

    def log_likelihood(self, weights):
        if np.isclose(np.sum(weights), 1.0):
            #return some constant value
            return -0.5
        else:
            return -np.inf

    def sample(self):
        return np.random.dirichlet(self.alpha, 1)[0]

class GMMCovarsStaticPrior():
    """ Assume that covariance is fixed """
    def __init__(self, prior_covars):
        self.prior_covars = prior_covars

    def log_likelihood(self, covariances):
        # assign probability of 1 to the static covariance prior
        if np.allclose(self.prior_covars, covariances):
            return 0.0
        else:
            return -np.inf

    def sample(self):
        return np.array(self.prior_covars)

class GMMWeightsStaticPrior():
    """ Assume Weights are fixed """
    def __init__(self, prior_weights):
        self.prior_weights = prior_weights

    def log_likelihood(self, covariances):
        # check that weights are the same
        return 0

    def sample(self):
        return np.array(self.prior_weights)