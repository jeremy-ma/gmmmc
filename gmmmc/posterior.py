
class GMMPosteriorTarget:
    """Posterior distribution (targets distribution)"""
    def __init__(self, prior, beta = 1):
        """
        A posterior target distribution. Calculated by adding the log likelihood and log prior probability.
        Parameters
        ----------
        prior : GMMPrior object
            prior distribution of the GMM parameters
        beta : double
            power of the likelihood component e.g P(X|parameters)^beta * P(parameters). Used for annealing.
        """
        self.prior = prior
        self.beta = beta

    def log_prob(self, X, gmm, n_jobs):
        """

        Parameters
        ----------
        X : 2-D array_like of shape (n_samples, n_features)
            Feature vectors
        gmm : GMM object
            GMM parameters for the calculation of the prior probability
        n_jobs : int
            Number of cores to use in the calculation.

        Returns : double
            log probability of the posterior up to a constant factor.
        -------

        """
        return self.beta * gmm.log_likelihood(X, n_jobs) + self.prior.log_prob(gmm)