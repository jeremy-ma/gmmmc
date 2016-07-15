import abc
import logging
from distributions.posterior import GMMPosteriorTarget
import numpy as np

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

    def __init__(self, proposal, priors, betas):
        self.targets = [GMMPosteriorTarget(priors, beta) for beta in betas]
        self.proposal = proposal
        self.priors = priors

    def sample(self, X, n_samples):
        samples = []
        weights = []
        for run in xrange(n_samples):
            logging.info('Run: {0}'.format(run))
            sample, weight = self.anneal(X)
            samples.append(sample)
            weights.append(weight)

        return (samples, weights)

    def anneal(self, X):
        # draw from prior
        cur_gmm = self.priors.sample()
        samples = []
        samples.append(cur_gmm)

        for anneal_run, target in enumerate(self.targets):
            if anneal_run == 0 or anneal_run == len(self.targets) - 2:
                continue # skip the prior only target
            samples.append(cur_gmm)
            cur_gmm = self.proposal.propose(X, target, cur_gmm)

        numerator = 0
        denominator = 0
        for run, sample in enumerate(samples):
            numerator += self.targets[run+1].log_prob(X, sample)
            denominator += self.targets[run].log_prob(X, sample)

        weight = numerator - denominator

        return (cur_gmm, weight)

