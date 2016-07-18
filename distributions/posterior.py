
class GMMPosteriorTarget:
    """Posterior distribution (targets distribution)"""
    def __init__(self, prior, beta = 1):
        """
        :param prior: prior distributions used
        :param beta: power of the likelihood component e.g P(X|parameters)^beta * P(parameters). Used for annealing.
        """
        self.prior = prior
        self.beta = beta

    def log_prob(self, X, gmm):
        return self.beta * gmm.log_likelihood(X) + self.prior.log_prob(gmm)