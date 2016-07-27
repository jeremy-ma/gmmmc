import abc
import logging
from gmmmc.posterior import GMMPosteriorTarget
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
        betas = np.array(betas)
        #if (betas > 1).any() or (betas < 0).any():
        #    raise ValueError('betas must be between 0 and 1')
        self.targets = [GMMPosteriorTarget(priors, beta,) for beta in betas]
        self.proposal = proposal
        self.priors = priors

    def sample(self, X, n_samples, n_jobs=1, diagnostics=None):
        """

        Parameters
        ----------
        X
        n_samples
        n_jobs
        diagnostics

        Returns
        -------
            : list of tuples (GMM, double)
            A list of GMM samples with their corresponding weight.
        """
        samples = []
        for run in xrange(n_samples):
            logging.info('Run: {0}'.format(run))
            if diagnostics is not None:
                diagnostics[run] = {}
                sample, weight = self.anneal(X, n_jobs, diagnostics[run])
            else:
                sample, weight = self.anneal(X, n_jobs)
            samples.append((sample, weight))

        return samples

    def anneal(self, X, n_jobs, diagnostics=None):
        """
        A single run of AIS
        :param X: Data
        :param n_jobs:
        :param diagnostics:
        :return:
        """
        # draw from prior
        cur_gmm = self.priors.sample()
        samples = []
        for anneal_run, target in enumerate(self.targets[1:-1]):
            # skip first T_n (prior only) and last transition T_0 (posterior) (not necessary)
            samples.append(cur_gmm)
            cur_gmm = self.proposal.propose(X, cur_gmm, target, n_jobs)
            logging.info('Beta:{0}'.format(target.beta))

        samples.append(cur_gmm)

        if diagnostics is not None:
            diagnostics['intermediate_log_weights'] = []
            diagnostics['intermediate_betas'] = []

        numerator = 0
        denominator = 0
        for run, sample in enumerate(samples):
            numerator += self.targets[run + 1].log_prob(X, sample, n_jobs)
            denominator += self.targets[run].log_prob(X, sample, n_jobs)
            if diagnostics is not None:
                diagnostics['intermediate_log_weights'].append(numerator - denominator)
                diagnostics['intermediate_betas'].append(self.targets[run].beta)

        weight = numerator - denominator

        if diagnostics is not None:
            diagnostics['intermediate_samples'] = samples
            diagnostics['final_sample'] = cur_gmm
            diagnostics['final_weight'] = weight

        return (cur_gmm, weight)

