import abc
import logging
from gmmmc.posterior import GMMPosteriorTarget


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

    def __init__(self, proposal, prior, initial_gmm):
        self.proposal = proposal
        self.target = GMMPosteriorTarget(prior, beta=1.0)
        self.initial_gmm = initial_gmm

    def sample(self, X, n_samples, n_jobs=1):
        samples = []
        current_gmm = self.initial_gmm
        for run in xrange(n_samples):
            logging.info('Run: {0}'.format(run))
            current_gmm = self.proposal.propose(X, current_gmm, self.target, n_jobs)
            samples.append(current_gmm)
        return samples

class AnnealedImportanceSampling(MonteCarloBase):

    def __init__(self, proposal, priors, betas):
        """

        :param proposal:
        :param priors:
        :param betas: array of betas in ascending order
        """
        self.targets = [GMMPosteriorTarget(priors, beta,) for beta in betas]
        self.proposal = proposal
        self.priors = priors

    def sample(self, X, n_samples, n_jobs=1):
        samples = []
        weights = []
        for run in xrange(n_samples):
            logging.info('Run: {0}'.format(run))
            sample, weight = self.anneal(X, n_jobs)
            samples.append(sample)
            weights.append(weight)

        return (samples, weights)

    def anneal(self, X, n_jobs):
        # draw from prior
        cur_gmm = self.priors.sample()
        samples = []
        samples.append(cur_gmm)

        for anneal_run, target in enumerate(self.targets):
            if anneal_run == 0 or anneal_run == len(self.targets) - 2:
                continue # skip the prior only beta and the last one
            samples.append(cur_gmm)
            cur_gmm = self.proposal.propose(X, cur_gmm, target, n_jobs)

        numerator = 0
        denominator = 0
        for run, sample in enumerate(samples):
            numerator += self.targets[run + 1].log_prob(X, sample, n_jobs)
            denominator += self.targets[run].log_prob(X, sample, n_jobs)

        weight = numerator - denominator

        return (cur_gmm, weight)

