__author__ = 'jeremyma'
import numpy as np
from likelihood import gmm_likelihood
from scipy.stats import multivariate_normal
import pdb
from numba import jit

@jit
def block_means_proposal(X, current_means, prior_weights, prior_means, prior_covars, n_mixtures, step_size=0.001):
    acceptance_count = 0
    new_means = current_means
    for mixture in xrange(n_mixtures):
        # propose new means
        new_mixture_means = np.random.multivariate_normal(current_means[mixture], step_size * np.eye(X.shape[1]))
        #new_mixture_means = np.random.uniform(low=-1, high=1, size=X.shape[1])

        # try out the new means
        proposed_means = np.array(new_means)
        proposed_means[mixture] = new_mixture_means

        # likelihood
        previous_likelihood = gmm_likelihood.log_likelihood_sklearn(X, n_mixtures, new_means, prior_covars, prior_weights)
        proposed_likelihood = gmm_likelihood.log_likelihood_sklearn(X, n_mixtures, proposed_means, prior_covars, prior_weights)

        # prior (only need to calculate the priors of the single mixture that is changed)
        # previous_prior = multivariate_normal.logpdf(new_means[mixture], prior_means[mixture], prior_covars[mixture])
        # proposed_prior = multivariate_normal.logpdf(proposed_means[mixture], prior_means[mixture], prior_covars[mixture])

        previous_prior = 0
        proposed_prior = 0

        # posterior
        previous_posterior = previous_likelihood + previous_prior
        proposed_posterior = proposed_likelihood + proposed_prior

        #ratio
        ratio = proposed_posterior - previous_posterior
        if ratio > 0 or ratio > np.log(np.random.uniform()):
            # accept proposal
            new_means = proposed_means
            acceptance_count += 1

    return new_means, acceptance_count


# Metropolis hastings with priors
# only means for now
@jit
def gmm_mcmc(X, initial_weights, initial_means, initial_covars, n_runs=10000, n_mixtures=32):
    mean_samples = np.zeros((n_runs, n_mixtures, X.shape[1]))
    acceptance_count = 0
    current_means = initial_means
    for run in xrange(n_runs):
        current_means, count = block_means_proposal(X, current_means, initial_weights, initial_means, initial_covars, n_mixtures)
        print count
        acceptance_count += count
        print run
        mean_samples[run] = current_means

    print(acceptance_count / float(n_mixtures * n_runs))
    return np.array(mean_samples)

