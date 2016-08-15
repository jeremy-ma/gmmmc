import abc

class Proposal(object):

    def __init__(self):
        """
        Base class for a proposal function
        """
        self.count_proposed = 0.0
        self.count_accepted = 0.0
        self.count_illegal = 0.0

    def get_acceptance(self):
        """
        Calculate and return the acceptance rate of the proposal.

        Returns
        -------
            : double
            The acceptance rate of the proposal function
        """
        return self.count_accepted / self.count_proposed

    def get_illegal(self):
        """
        Calculate and return the illegal proposal rate of this proposal.
        (Proposing values outside the support of the parameter space e.g covariances < 0)

        Returns
        -------
            : double
            The illegal proposal rate of the proposal function
        """
        return self.count_illegal / self.count_proposed

    """ generic proposal function"""
    @abc.abstractmethod
    def propose(self, X, gmm, target, n_jobs=1):
        """

        Parameters
        ----------
        X : 2-D array_like
            Observed data or evidence.
        gmm : GMM object

        target
        n_jobs

        Returns
        -------

        """
        pass


class GMMBlockMetropolisProposal(Proposal):

    def __init__(self, propose_mean=None, propose_covars=None, propose_weights=None, propose_iterations=1):
        """
        Blocked Metropolis Proposal function for a GMM.
        The parameters of the means, covariances and weights are updated separately.

        Parameters
        ----------
        propose_mean : Proposal object
            Proposal object
        propose_covars
        propose_weights
        propose_iterations
        """
        self.propose_mean = propose_mean
        self.propose_covars = propose_covars
        self.propose_weights = propose_weights
        self.propose_iterations = propose_iterations

    def propose(self, X, gmm, target, n_jobs=1):
        """
        Propose a new set of gmm parameters. Calls each proposal function one after another.

        Parameters
        ----------
        X : 2-D array_like of shape (n_samples, n_features)
            Feature vectors
        gmm : GMM object
            Current GMM parameters in the markov chain
        target : GMMPosteriorTarget object
            Target distribution
        n_jobs : int
            Number of cpus to use. -1 to use all available cores.

        Returns
        -------
            : GMM Object
            The next state in the Markov Chain.

        """
        new_gmm = gmm
        for _ in xrange(self.propose_iterations):
            if self.propose_mean is not None:
                new_gmm = self.propose_mean.propose(X, new_gmm, target, n_jobs)

            if self.propose_covars is not None:
                new_gmm = self.propose_covars.propose(X, new_gmm, target, n_jobs)

            if self.propose_weights is not None:
                new_gmm = self.propose_weights.propose(X, new_gmm, target, n_jobs)
        return new_gmm