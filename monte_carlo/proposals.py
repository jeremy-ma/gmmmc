import abc
import numpy as np
from distributions.gmm import GMM

class Proposal(object):
    """generic proposal function"""
    @abc.abstractmethod
    def propose(self, X, target, gmm):
        pass

class GaussianStepMeansProposal(Proposal):
    def __init__(self, step_size=0.001):
        self.count_proposed = 0
        self.count_accepted = 0
        self.step_size = step_size

    def propose(self, X, target, gmm):
        new_means = np.array(gmm.means)
        previous_prob = target.log_prob(X, gmm)
        for mixture in xrange(gmm.n_mixtures):
            self.count_proposed += 1
            # propose new means
            new_mixture_means = np.random.multivariate_normal(gmm.means[mixture], self.step_size * np.eye(X.shape[1]))
            # new_mixture_means = np.random.uniform(low=-1, high=1, size=X.shape[1])

            # try out the new means
            proposed_means = np.array(new_means)
            proposed_means[mixture] = new_mixture_means

            proposed_gmm = GMM(proposed_means, np.array(gmm.weights), np.array(gmm.covars))

            # distributions
            proposed_prob = target.log_prob(X, proposed_gmm)

            # ratio
            ratio = proposed_prob - previous_prob
            if ratio > 0 or ratio > np.log(np.random.uniform()):
                # accept proposal
                new_means = proposed_means
                previous_prob = proposed_prob
                self.count_accepted += 1

        return GMM(new_means, np.array(gmm.weights), np.array(gmm.covars))

    def get_acceptance(self):
        return self.count_accepted / float(self.count_proposed)

class GaussianStepCovarProposal(Proposal):
    pass

class GaussianStepWeightsProposal(Proposal):
    pass

class GMMBlockMetropolisProposal(Proposal):

    def __init__(self, propose_mean=None, propose_covars=None, propose_weights=None):
        self.propose_mean = propose_mean
        self.propose_covars = propose_covars
        self.propose_weights = propose_weights

    def propose(self, X, target, gmm):
        new_gmm = gmm
        if self.propose_mean is not None:
            new_gmm = self.propose_mean.propose(X, target, new_gmm)

        if self.propose_covars is not None:
            new_gmm = self.propose_covars.propose(X, target, new_gmm)

        if self.propose_weights is not None:
            new_gmm = self.propose_weights.propose(X, target, new_gmm)

        return new_gmm

    def get_means_acceptance(self):
        return self.propose_mean.get_acceptance()