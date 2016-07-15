import numpy as np
import scipy.stats
from scipy.misc import logsumexp
from gmm import GMM
import abc

# TODO: Input verification
class GMMPrior():
    def __init__(self, means_prior, covars_prior, weights_prior):
        """
        Class containing prior distributions for GMM means, weights covariances
        :param means_prior: Class for priors of GMM means
        :param covars_prior: Class for priors of GMM covariances
        :param weights_prior: Class fo priors of GMM weights
        """
        self.means_prior = means_prior
        self.covars_prior = covars_prior
        self.weights_prior = weights_prior

    def log_prob(self, gmm):
        """
        Compute prior probability of parameters in gmm
        :param gmm: GMM object containing parameters
        :return: log prior probability
        """
        logprob = 0
        logprob += self.means_prior.log_prob(gmm.means)
        logprob += self.covars_prior.log_prob(gmm.covars)
        logprob += self.weights_prior.log_prob(gmm.weights)
        return logprob

    def sample(self):
        return GMM(self.means_prior.sample(), self.weights_prior.sample(), self.covars_prior.sample())

class MeansGaussianPrior():
    def __init__(self, prior_means, covariances):
        """
        Gaussian Prior for GMM means
        :param prior_means: Expected means of GMM
        :param covariances: Covariances of the means of the GMM
        """
        # shape should be (n_mixtures, n_features)
        self.means = prior_means
        self.covars = covariances

        self.distributions = [scipy.stats.multivariate_normal(self.means[i], self.covars[i])\
                              for i in xrange(len(self.means))]

    def log_prob(self, means):
        """
        Compute the prior probability of the means of a GMM
        :param means:
        :return:
        """
        # logsumexp probabilities
        return logsumexp([normal.pdf(means) for normal in self.distributions])

    def sample(self):
        # one at a time
        return np.array([normal.rvs() for normal in self.distributions])

class MeansUniformPrior():
    """Uniform prior for means"""
    def __init__(self, low, high, n_mixtures, n_features):
        self.low = low
        self.high = high
        self.n_mixtures = n_mixtures
        self.n_features = n_features

    def log_prob(self, means):
        # just return some constant value
        return -0.5

    def sample(self):
        # sample means
        return np.random.uniform(self.low, self.high, size=(self.n_mixtures, self.n_features))

class DiagCovarsUniformPrior():
    """Uniform Prior for diagonal covariances"""
    def __init__(self, low, high, n_mixtures, n_features):
        self.low = low
        self.high = high
        self.n_mixtures = n_mixtures
        self.n_features = n_features

    def log_prob(self, covars):
        if (covars < 0).any():
            return -np.inf
        else:
            # return some constant value
            return -0.5

    def sample(self):
        return np.random.uniform(self.low, self.high, size=(self.n_mixtures, self.n_features))

class CovarsInvWishartPrior():
    """Inverse Wishart Prior for covariance matrix"""
    def __init__(self, degrees_freedom, scale):
        """
        :param degrees_freedom: degrees of freedom, must be greater than or equal to dimension of matrix
        :param scale: scale matrix, can be taken as a covariance matrix
        """
        self.df = degrees_freedom
        self.scale = scale
        self.distribution = scipy.stats.invWishart(df=self.df, scale=self.scale)

    def log_prob(self, covariances):
        log_probs = [self.distribution.logpdf(covariance) for covariance in covariances]


class CovarsStaticPrior():
    """ Assume that covariance is fixed """
    def __init__(self, prior_covars):
        self.prior_covars = prior_covars

    def log_prob(self, covariances):
        # assign probability of 1 to the static covariance prior
        if np.allclose(self.prior_covars, covariances):
            return 0.0
        else:
            return -np.inf

    def sample(self):
        return np.array(self.prior_covars)

class WeightsUniformPrior():
    """Uniform Prior for Weights (Dirichlet distribution)"""
    def __init__(self, n_mixtures):
        self.alpha = [1 for _ in xrange(n_mixtures)]

    def log_prob(self, weights):
        if np.isclose(np.sum(weights), 1.0) and np.logical_and(0 <= weights, weights <= 1).all():
            #return some constant value
            return -0.5
        else:
            return -np.inf

    def sample(self):
        return np.random.dirichlet(self.alpha, 1)[0]


class WeightsStaticPrior():
    """ Assume Weights are fixed """
    def __init__(self, prior_weights):
        self.prior_weights = prior_weights

    def log_prob(self, covariances):
        # check that weights are the same
        return 0

    def sample(self):
        return np.array(self.prior_weights)

class WeightsDirichletPrior():
    """ Dirichlet prior for weights of GMM """
    def __init__(self, alpha):
        """
        Use Dirichlet prior for weights.
        :param alphas: Parameters of dirichlet distribution
        """
        self.alpha = alpha

    def log_prob(self, weights):
        return scipy.stats.dirichlet.logpdf(weights, self.alpha)

    def sample(self):
        return np.random.dirichlet(self.alpha)