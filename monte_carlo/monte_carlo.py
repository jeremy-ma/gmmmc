import abc
import logging

class MonteCarloBase:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def sample(self, X, n_samples):
        """
        Start the GMM Monte Carlo Sampler
        :param n_samples:
        :return: list of GMM parameter samples
        """
        return

class MarkovChain(MonteCarloBase):

    def __init__(self, proposal, target, initial_gmm):
        self.proposal = proposal
        self.target = target
        self.initial_gmm = initial_gmm

    def sample(self, X, n_samples):
        samples = []
        current_gmm = self.initial_gmm
        for run in xrange(n_samples):
            logging.info('Run: {0}'.format(run))
            current_gmm = self.proposal.propose(X, self.target, current_gmm)
            samples.append(current_gmm)
        return samples

class AnnealedImportanceSampling(MonteCarloBase):

    def __init__(self, proposer, posterior):
        pass
