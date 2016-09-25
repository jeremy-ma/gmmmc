import abc
import numpy as np
import scipy.stats
from gmmmc import GMM


class GMMPrior():
    def __init__(self, means_prior, covars_prior, weights_prior):
        """
        Containing class for the priors of a GMM. Initialised with priors of means, weights & covariances.

        Parameters
        ----------
        means_prior : GMMParameterPrior
            Prior for the means of the GMM
        covars_prior : GMMParameterPrior
            Prior for the covariances of the GMM
        weights_prior : GMMParameterPrior
            Prior for the weights of the GMM
        """
        self.means_prior = means_prior
        self.covars_prior = covars_prior
        self.weights_prior = weights_prior

    def log_prob(self, gmm):
        """

        Parameters
        ----------
        gmm : GMM
            object containing parameters

        Returns
        -------
            : double
            log prior probability of the parameters of the input GMM
        """

        logprob = 0
        logprob += self.means_prior.log_prob(gmm.means)
        logprob += self.covars_prior.log_prob(gmm.covars)
        logprob += self.weights_prior.log_prob(gmm.weights)
        return logprob

    def sample(self):
        """
        Compute a sample from the prior distribution of the GMM's parameters.

        Returns
        -------
            : GMM
            GMM object containing the sampled parameters
        """
        return GMM(self.means_prior.sample(), self.covars_prior.sample(), self.weights_prior.sample())

class GMMParameterPrior:

    @abc.abstractmethod
    def log_prob(self, params):
        """
        Calculate the log prior probability.

        Parameters
        ----------
        params : array_like of varying shape
            parameters for a GMM e.g means weights covariances

        Returns
        -------
            : double
            log prior probability of the parameters

        """
        pass

    @abc.abstractmethod
    def log_prob_single(self, param, mixture_num):
        """
        Compute log probability of a single set of parameters (mean/covariance/weight) vector

        Parameters
        ----------
        param : 1-D array_like vector of parameters for a single mixture
        mixture_num : Mixture index for the parameters.

        Returns
        -------
            : double
            log prior probability of parameters
        """

        pass

class MeansGaussianPrior(GMMParameterPrior):
    def __init__(self, prior_means, covariances):
        """
        Multivariate Gaussian prior distribution for GMM means.
        Each mean vector for the means has its own Gaussian prior. Diagonal covariances for the means.

        Parameters
        ----------
        prior_means : 2-D array_like of shape (n_mixtures, n_features)
            Expected means for each mixture of the GMM
        covariances : 2-D array_like of shape (n_mixtures, n_features)
            Diagonal covariances of the expected means of the GMM. Alters the 'width' of the prior.
        """
        # shape should be (n_mixtures, n_features)
        self.means = prior_means
        self.covars = covariances
        self.distributions = [scipy.stats.multivariate_normal(self.means[i], self.covars[i])\
                              for i in xrange(self.means.shape[0])]
        try:
            self.n_features = prior_means.shape[1]
        except:
            raise ValueError("Means must be 2d")

    def log_prob(self, means):
        """Compute the log prior probability of the means of a GMM according to Gaussian priors.

        Parameters
        ----------
        means : 2-D array_like, of shape (n_mixtures, n_features)
            mean vectors for the GMM

        Returns
        -------
            : double
            Proportional to the log probability of the means of the GMM
        """
        log_prob = np.sum([self.log_prob_single(means[i], i) for i in xrange(len(self.distributions))])

        return log_prob

    def log_prob_single(self, mean, mixture_num):
        """Compute the log probability of the means for a specific mixture.

        Parameters
        ----------
        mean : 1-D array_like of length n_features
            Single mean vector from a single mixture of the GMM.
        mixture_num : int
            Index of the mixture for the mean.

        Returns
        -------
            : double
            Proportional to the log prior probability for means
        """

        return self.distributions[mixture_num].logpdf(mean)

    def sample(self):
        """
        Draw a sample from the Gaussian prior distributions of the mean vectors.

        Returns
        -------
            : 2-D array_like of shape (n_mixtures, n_features)
            Return a complete set of mean vectors for a GMM
        """
        return np.array([[normal.rvs()] if self.n_features == 1 else normal.rvs() for normal in self.distributions])

class MeansUniformPrior(GMMParameterPrior):
    def __init__(self, low, high, n_mixtures, n_features):
        """
        Uniform prior for the means of a GMM.

        Parameters
        ----------
        low : double
            Upper limit for uniform distribution.
        high : double
            Lower limit for uniform distribution.
        n_mixtures : int
            Number of mixtures in GMM.
        n_features : int
            Dimensionality of the feature space.
        """
        self.low = low
        self.high = high
        self.n_mixtures = n_mixtures
        self.n_features = n_features

    def log_prob(self, means):
        """
        Compute the log prior probability of the means of a GMM. Since this will be used for monte carlo simulations,
        we care only that it is proportional to the true probability. For a uniform distribution we can
        use any value.

        Parameters
        ----------
        means : 2-D array_like, of shape (n_mixtures, n_features)
            mean vectors for the GMM

        Returns
        -------
            : double
            Proportional to the log probability of the means of the GMM.
            0.0 if the means are within the bounds of the uniform prior, -inf otherwise.
        """
        # only care about proportional.
        if (means < self.low).any() or (means > self.high).any():
            return -np.inf
        else:
            return 0.0

    def log_prob_single(self, mean, mixture):
        """
        Compute the log probability of the means for a specific mixture.

        Parameters
        ----------
        mean : 1-D array_like of length n_features
            Single mean vector from a single mixture of the GMM.
        mixture_num : int
            Index of the mixture for the mean.

        Returns
        -------
            : double
            Proportional to the log prior probability for means
            0.0 if the means are within the bounds of the uniform prior, -inf otherwise.
        """
        if (mean < self.low).any() or (mean > self.high).any():
            # if invalid value
            return -np.inf
        else:
            return 0.0

    def sample(self):
        # sample means
        return np.random.uniform(self.low, self.high, size=(self.n_mixtures, self.n_features))


class DiagCovarsWishartPrior(GMMParameterPrior):
    def __init__(self, df, scale_matrices):
        """
        Inverse Wishart prior for diagonal covariance matrix
        Parameters
        ----------
        df : int
            Degrees of freedom
        scale : ndarray of shape (d,d)
            scale matrix
        """
        self.df = df
        self.distributions = [scipy.stats.invwishart(self.df, scale_matrices[i]) \
                              for i in xrange(len(scale_matrices))]

    def log_prob(self, covars):
        """
        Compute the log prior probability of the covariances of a GMM.
        Since this will be used for monte carlo simulations, we care only that it is proportional to
        the true probability.

        Parameters
        ----------
        covars : 2-D array_like, of shape (n_mixtures, n_features)
            covariance vectors for the GMM

        Returns
        -------
            : double
            Proportional to the log probability of the covariance of the GMM.
            w.r.t an inverse wishart distribution
        """

        if (covars < 0).any():
            return -np.inf
        else:
            log_prob = np.sum([self.log_prob_single(covars[i], i) for i in xrange(len(self.distributions))])
            return log_prob


    def log_prob_single(self, covar, mixture_num):
        """Compute the log probability of the means for a specific mixture.

        Parameters
        ----------
        mean : 1-D array_like of length n_features
            Single mean vector from a single mixture of the GMM.
        mixture_num : int
            Index of the mixture for the mean.

        Returns
        -------
            : double
            Proportional to the log prior probability for means
        """
        # convert diagonal covariance into full matrix
        covar = np.diag(covar)
        return self.distributions[mixture_num].logpdf(covar)

    def sample(self):
        """
        Draw a sample from the inverse wishart prior distribution.

        Returns
        -------
            : 2-D array_like of shape (n_mixtures, n_features)
            Return a complete set of mean vectors for a GMM
        """
        raise NotImplementedError

class DiagCovarsUniformPrior(GMMParameterPrior):
    def __init__(self, low, high, n_mixtures, n_features):
        """
        Uniform prior for a diagonal covariance matrix in a GMM.

        Parameters
        ----------
        low : double
            Lower limit for uniform distribution. Must be greater than 0.
        high : double
            Upper limit for uniform distribution.
        n_mixtures : int
            Number of mixtures in GMM.
        n_features : int
            Dimensionality of the feature space.
        """
        self.low = low
        self.high = high
        self.n_mixtures = n_mixtures
        self.n_features = n_features

    def log_prob(self, covars):
        """
        Compute the log prior probability of the covariances of a GMM.
        Since this will be used for monte carlo simulations, we care only that it is proportional to
        the true probability. For a uniform distribution we can use any value.

        Parameters
        ----------
        covars : 2-D array_like, of shape (n_mixtures, n_features)
            covariance vectors for the GMM

        Returns
        -------
            : double
            Proportional to the log probability of the covariance of the GMM.
            0.0 if the means are within the bounds of the uniform prior, -inf otherwise.eans
        """

        if (covars < self.low).any() or (covars > self.high).any():
            return -np.inf
        else:
            # return some constant value
            return 0.0

    def log_prob_single(self, covar, mixture_num):
        """
        Compute the log probability of the covariances for a specific mixture.

        Parameters
        ----------
        covar : 1-D array_like of length n_features
            Single diagonal covariance from a single mixture of the GMM.
        mixture_num : int
            Index of the mixture for the covariance matrix.

        Returns
        -------
        double
            Proportional to the log prior probability for the covariance.
            0.0 if the means are wimeansthin the bounds of the uniform prior, -inf otherwise.
        """
        if (covar < self.low).any() or (covar > self.high).any():
            return -np.inf
        else:
            # return some constant value
            return 0.0

    def sample(self):
        """
        Draw a sample for the diagonals of a covariance matrix assuming a uniform prior over each individual element.

        Returns
        -------
            : 2-D array_like of shape (n_mixtures, n_features)
            array of diagonal covariances for a GMM.
        """
        return np.random.uniform(self.low, self.high, size=(self.n_mixtures, self.n_features))

class CovarsStaticPrior(GMMParameterPrior):
    def __init__(self, prior_covars):
        """
        Prior for covariance assuming that we know its true value.
        i.e a probability of 1 at prior_covars and 0 otherwise.

        Parameters
        ----------
        prior_covars : 2-D array_like of shape (n_mixtures, n_features)
            Assumed true values of covariance matrices for GMM
        """
        self.prior_covars = prior_covars

    def log_prob(self, covariances):
        """
        Log probability of a covariance given we know what its true value 'should' be.

        Parameters
        ----------
        covariances : covariance matrices of GMM distribution

        Returns
        -------
            : double
            0 if covariances are identical to their true values, -inf otherwise.
        """
        # assign probability of 1 to the static covariance prior
        if np.allclose(self.prior_covars, covariances):
            return 0.0
        else:
            return -np.inf

    def log_prob_single(self, covariance, mixture_num):
        """
        Log probability of a covariance matrix of a single mixture given we know what its true value should be.

        Parameters
        ----------
        covariance : 1-D array_like of shape (n_features)
            covariance matrix for a specific mixture in GMM
        mixture_num : int
            Index of mixture in GMM

        Returns
        -------
            : double
            0 if covariance is identical to its true value, -inf otherwise
        """
        if np.allclose(self.prior_covars[mixture_num], covariance):
            return 0.0
        else:
            return -np.inf

    def sample(self):
        return np.array(self.prior_covars)

class WeightsUniformPrior(GMMParameterPrior):
    def __init__(self, n_mixtures):
        """
        Uniform Prior for Weights (Dirichlet distribution with all parameters equal to 1)

        Parameters
        ----------
        n_mixtures : int
            number of mixtures in the GMM
        """

        self.alpha = [1 for _ in xrange(n_mixtures)]
        self.dirichlet = scipy.stats.dirichlet(self.alpha)

    def log_prob(self, weights):
        """
        Returns a log probability according to uniform dirichlet prior.

        Parameters
        ----------
        weights : 1-D array_like of shape (n_mixtures)
            Weight vector for mixture. Must lie on the normal simplex.

        Returns
        -------
            : double
            log probability under uniform prior.
        """
        return self.dirichlet.logpdf(weights)

    def log_prob_single(self, weights, mixture_num):
        """
        Functionally the same as log_prob

        Parameters
        ----------
        weights : 1-D array_like of shape (n_mixtures)
            Weight vector for mixture. Must lie on the normal simplex.
        mixture_num : not used in this context

        Returns
        -------
            : double
            log probability under uniform prior.
        """
        return self.dirichlet.logpdf(weights)

    def sample(self):
        """
        Draw sample from dirichlet distribution

        Returns
        -------
            : 1-D array_like of shape (n_mixtures)
            Set of weight parameters for a GMM.
        """
        return np.random.dirichlet(self.alpha, 1)[0]

class WeightsStaticPrior(GMMParameterPrior):
    def __init__(self, prior_weights):
        """
        Prior for Weights  assuming it has a true fixed value.

        Parameters
        ----------
        n_mixtures : int
            number of mixtures in the GMM
        """
        self.prior_weights = prior_weights

    def log_prob(self, weights):
        """
        Returns a log probability assuming weights have a true fixed value.

        Parameters
        ----------
        weights : 1-D array_like of shape (n_mixtures)
            Weight vector for mixture. Must lie on the normal simplex.

        Returns
        -------
            : double
            0 if weights are close to true values, -inf otherwise
        """
        if np.all(np.isclose(weights, self.prior_weights)):
            return 0.0
        else:
            return -np.inf

    def log_prob_single(self, weights, mixture_num):
        """
        Functionally the same as log_prob

        Parameters
        ----------
        weights : 1-D array_like of shape (n_mixtures)
            Weight vector for mixture. Must lie on the normal simplex.
        mixture_num : int
            Not used.

        Returns
        -------
            : double
            0 if weights are close to true values, -inf otherwise
        """
        return self.log_prob(weights)

    def sample(self):
        """
        Sample true weight vector

        Returns
        -------
            : 1-D array_like with shape (n_mixtures)
            true weights.
        """
        return np.array(self.prior_weights)

class WeightsDirichletPrior(GMMParameterPrior):
    def __init__(self, alpha):
        """
        Dirichlet prior for the weights of a GMM

        Parameters
        ----------
        alpha : 1-D array_like of shape (n_mixtures)
            Parameters of the dirichlet distribution.

        Notes
        ----------
        Alphas of the dirichlet distribution can be set to the expected weights. (e.g those found by ML estimation)
        """
        self.alpha = alpha

    def log_prob(self, weights):
        """
        Calculate log probability of weight vector according to dirichlet prior.

        Parameters
        ----------
        weights : 1-D array_like of shape (n_mixtures)

        Returns
        -------
            : double
            log probability under distribution
        """
        return scipy.stats.dirichlet.logpdf(weights, self.alpha)

    def log_prob_single(self, weights, mixture_num):
        """
        Identical to log_prob

        Parameters
        ----------
        weights : 1-D array_like of shape (n_mixtures)
        mixture_num : unused

        Returns
        -------
            : double
            log probability under distribution
        """
        return self.log_prob(weights)

    def sample(self):
        """
        Sample from dirichlet distribution.

        Returns
        -------
            : 1-D array like of shape (n_mixtures)
            Sample weights from dirichlet distribution.
        """
        return np.random.dirichlet(self.alpha)