import abc
import numpy as np
import scipy.stats
import xxhash
from expiringdict import ExpiringDict
from gmmmc import GMM

class GMMPrior():
    def __init__(self, means_prior, covars_prior, weights_prior):
        """
        Class containing prior priors for GMM means, weights covariances
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

class GMMParameterPrior:

    @abc.abstractmethod
    def log_prob(self, params):
        """Compute log probability of entire space of parameters (means/covariances/weights)"""
        pass

    @abc.abstractmethod
    def log_prob_single(self, param, mixture_num):
        """Compute log probability of a single set of parameters (mean/covariance/weight) vector"""
        pass

class MeansGaussianPrior(GMMParameterPrior):
    def __init__(self, prior_means, covariances):
        """
        Gaussian Prior for GMM means
        :param prior_means: Expected means of GMM
        :param covariances: Covariances of the means of the GMM
        """
        # shape should be (n_mixtures, n_features)
        self.means = prior_means
        self.covars = covariances
        self.cache = ExpiringDict(max_len=1024, max_age_seconds=100)
        self.distributions = [scipy.stats.multivariate_normal(self.means[i], self.covars[i])\
                              for i in xrange(len(self.means))]
        try:
            self.n_features = prior_means.shape[1]
        except:
            raise ValueError("Means must be 2d")

    def log_prob(self, means):
        """
        Compute the prior probability of the means of a GMM
        :param means:
        :return:
        """
        log_prob = 0
        for i, normal in enumerate(self.distributions):
            hashval = xxhash.xxh32(means[i]).intdigest()
            result = self.cache.get(hashval)
            if result is None:
                log_prob_mean = self.log_prob_single(means[i], i)
                self.cache[hashval] = (log_prob_mean, means[i])
            else:
                log_prob_mean, mean = result
                if not np.array_equal(mean, means[i]):
                    log_prob_mean = self.log_prob_single(means[i], i)
                    self.cache[hashval] = (log_prob_mean, means[i])
            log_prob += log_prob_mean

        return log_prob

    def log_prob_single(self, mean, mixture_num):
        """
        compute the log probability of the means for a specific mixture
        :param mean:
        :param mixture_num:
        :return:
        """
        return self.distributions[mixture_num].logpdf(mean)

    def sample(self):
        # one at a time
        return np.array([[normal.rvs()] if self.n_features == 1 else normal.rvs() for normal in self.distributions])

class MeansUniformPrior(GMMParameterPrior):
    """Uniform prior for means"""
    def __init__(self, low, high, n_mixtures, n_features):
        self.low = low
        self.high = high
        self.n_mixtures = n_mixtures
        self.n_features = n_features

    def log_prob(self, means):
        # just return some constant value
        if (means < self.low).any() or (means > self.high).any():
            return -np.inf
        else:
            return 0.0

    def log_prob_single(self, mean, mixture):
        if (mean < self.low).any() or (mean > self.high).any():
            # if invalid value
            return -np.inf
        else:
            return 0.0

    def sample(self):
        # sample means
        return np.random.uniform(self.low, self.high, size=(self.n_mixtures, self.n_features))

class DiagCovarsUniformPrior(GMMParameterPrior):
    """Uniform Prior for diagonal covariances"""
    def __init__(self, low, high, n_mixtures, n_features):
        self.low = low
        self.high = high
        self.n_mixtures = n_mixtures
        self.n_features = n_features

    def log_prob(self, covars):
        if (covars < self.low).any() or (covars > self.high).any():
            return -np.inf
        else:
            # return some constant value
            return 0.0

    def log_prob_single(self, covar, mixture_num):
        if (covar < self.low).any() or (covar > self.high).any():
            return -np.inf
        else:
            # return some constant value
            return 0.0

    def sample(self):
        return np.random.uniform(self.low, self.high, size=(self.n_mixtures, self.n_features))

class CovarsInvWishartPrior(GMMParameterPrior):
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
        raise NotImplementedError()

    def log_prob_single(self, param, mixture_num):
        raise NotImplementedError()


class CovarsStaticPrior(GMMParameterPrior):
    """ Assume that covariance is fixed """
    def __init__(self, prior_covars):
        self.prior_covars = prior_covars

    def log_prob(self, covariances):
        # assign probability of 1 to the static covariance prior
        if np.allclose(self.prior_covars, covariances):
            return 0.0
        else:
            return -np.inf

    def log_prob_single(self, covariance, mixture_num):
        """Assign probability to single covariance matrix"""
        if np.allclose(self.prior_covars[mixture_num], covariance):
            return 0.0
        else:
            return -np.inf

    def sample(self):
        return np.array(self.prior_covars)

class WeightsUniformPrior(GMMParameterPrior):
    """Uniform Prior for Weights (Dirichlet distribution)"""
    def __init__(self, n_mixtures):
        self.alpha = [1 for _ in xrange(n_mixtures)]

    def log_prob(self, weights):
        if np.isclose(np.sum(weights), 1.0) and np.logical_and(0 <= weights, weights <= 1).all():
            #return some constant value
            return 0.0
        else:
            return -np.inf

    def log_prob_single(self, weights, mixture_num):
        """Not needed"""
        raise NotImplementedError()

    def sample(self):
        return np.random.dirichlet(self.alpha, 1)[0]

class WeightsStaticPrior(GMMParameterPrior):
    """ Assume Weights are fixed """
    def __init__(self, prior_weights):
        self.prior_weights = prior_weights

    def log_prob(self, weights):
        # check that weights are the same
        if np.all(np.isclose(weights, self.prior_weights)):
            return 0.0
        else:
            return -np.inf

    def log_prob_single(self, weights, mixture_num):
        return self.log_prob(weights)

    def sample(self):
        return np.array(self.prior_weights)

class WeightsDirichletPrior(GMMParameterPrior):
    """ Dirichlet prior for weights of GMM """
    def __init__(self, alpha):
        """
        Use Dirichlet prior for weights.
        :param alphas: Parameters of dirichlet distribution
        """
        self.alpha = alpha

    def log_prob(self, weights):
        return scipy.stats.dirichlet.logpdf(weights, self.alpha)

    def log_prob_single(self, weights, mixture_num):
        return self.log_prob(weights)

    def sample(self):
        return np.random.dirichlet(self.alpha)