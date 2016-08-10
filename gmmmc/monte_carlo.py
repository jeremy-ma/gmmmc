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
        """
        Base algorithm for Markov Chain Monte Carlo
        Parameters
        ----------
        proposal : Proposal Object
            The proposal function to be used.
        prior : GMMPrior Object
            The prior distribution on the parameters of the GMM.
        initial_gmm : GMM Object
            The starting point for the MCMC simulation.
        """
        self.proposal = proposal
        self.target = GMMPosteriorTarget(prior, beta=1.0)
        self.initial_gmm = initial_gmm

    def sample(self, X, n_samples, n_jobs=1):
        """
        Sample from the posterior distribution of the GMM parameters
        Parameters
        ----------
        X : 2-D array_like of shape (n_feature_vectors, n_features)
            Feature vectors used as the data for the underlying model.
        n_samples : int
            Number of Monte Carlo samples to be drawn from the posterior distribution.
        n_jobs : int
            Number of cpu cores to utilise during the Monte Carlo simulation.

        Returns
        -------
        samples : List of GMM Objects
            Returns a list of GMMs (GMM parameters) which are the samples drawn from the distribution.

        Notes
        -----
        It is beneficial to discard a number of samples from the beginning of the chain (burn-in) as well as
        to use only every nth sample (lag) to reduce the correlation between succesive samples of the posterior.
        """
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
        Base algorithm for Annealed Importance Sampling.
        Parameters
        ----------
        proposal : Proposal Object
            The proposal function to be used.
        prior : GMMPrior Object
            The prior distribution on the parameters of the GMM.
        betas : 1-D array_like of shape (n_betas,)
            Betas used for the annealing process. A series of doubles from 0 to 1.

        References
        ----------
        Neal, Radford M. "Annealed importance sampling." Statistics and Computing 11.2 (2001): 125-139.
        """
        betas = np.array(betas)
        self.targets = [GMMPosteriorTarget(priors, beta,) for beta in betas]
        self.proposal = proposal
        self.priors = priors

    def sample(self, X, n_samples, n_jobs=1, diagnostics=None):
        """
        Generate samples from the posterior distribution of the parameters.
        Parameters
        ----------
        X : 2-D array_like of shape (n_feature_vectors, n_features)
            Feature vectors used as the data for the underlying model.
        n_samples : int
            Number of Monte Carlo samples to be drawn from the posterior distribution.
        n_jobs : int
            Number of cpu cores to utilise during the Monte Carlo simulation.
        diagnostics : empty dictionary, optional
            If included, the dictionary passed to the function will contain diagnostic information
            from each annealing run of AIS. TODO: expand on this

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
        A single annealing run from AIS.
        Parameters
        ----------
        X : 2-D array_like of shape (n_feature_vectors, n_features)
            Feature vectors used as the data for the underlying model.
        n_jobs : int
            Number of cpu cores to utilise during the Monte Carlo simulation.
        diagnostics : empty dictionary, optional
            If included, the dictionary passed to the function will contain diagnostic information
            from each annealing run of AIS. TODO: expand on this

        Returns
        -------
            : tuple (GMM Object, double)
            A single GMM sample from AIS and its corresponding weight.
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

