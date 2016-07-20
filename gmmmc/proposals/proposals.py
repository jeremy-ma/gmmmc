import abc

class Proposal(object):
    def __init__(self):
        self.count_proposed = 0.0
        self.count_accepted = 0.0
        self.count_illegal = 0.0

    def get_acceptance(self):
        return self.count_accepted / self.count_proposed

    def get_illegal(self):
        return self.count_illegal / self.count_proposed

    """ generic proposal function"""
    @abc.abstractmethod
    def propose(self, X, gmm, target, n_jobs=1):
        pass


class GMMBlockMetropolisProposal(Proposal):

    def __init__(self, propose_mean=None, propose_covars=None, propose_weights=None, propose_iterations=1):
        self.propose_mean = propose_mean
        self.propose_covars = propose_covars
        self.propose_weights = propose_weights
        self.propose_iterations = propose_iterations

    def propose(self, X, gmm, target, n_jobs=1):
        new_gmm = gmm
        for _ in xrange(self.propose_iterations):
            if self.propose_mean is not None:
                new_gmm = self.propose_mean.propose(X, new_gmm, target, n_jobs)

            if self.propose_covars is not None:
                new_gmm = self.propose_covars.propose(X, new_gmm, target, n_jobs)

            if self.propose_weights is not None:
                new_gmm = self.propose_weights.propose(X, new_gmm, target, n_jobs)
        return new_gmm