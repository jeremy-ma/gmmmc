__author__ = 'jeremyma'
import logging

import numpy as np
from numba import jit
from trial import gmm_test

@jit
def block_means_proposal(X, current_means, current_weights, current_covars, n_mixtures, step_size=0.001):
    """

    :param X:
    :param current_means:
    :param current_weights:
    :param prior_means:
    :param current_covars:
    :param n_mixtures:
    :param step_size:
    :return:
    """
    acceptance_count = 0
    new_means = current_means
    previous_likelihood = gmm_test.log_likelihood(X, n_mixtures, new_means, current_covars, current_weights)
    for mixture in xrange(n_mixtures):
        # propose new means
        new_mixture_means = np.random.multivariate_normal(current_means[mixture], step_size * np.eye(X.shape[1]))
        #new_mixture_means = np.random.uniform(low=-1, high=1, size=X.shape[1])

        # try out the new means
        proposed_means = np.array(new_means)
        proposed_means[mixture] = new_mixture_means

        # distributions
        proposed_likelihood = gmm_test.log_likelihood(X, n_mixtures, proposed_means, current_covars, current_weights)

        # posterior
        previous_posterior = previous_likelihood
        proposed_posterior = proposed_likelihood

        #ratio
        ratio = proposed_posterior - previous_posterior
        if ratio > 0 or ratio > np.log(np.random.uniform()):
            # accept proposal
            new_means = proposed_means
            previous_likelihood = proposed_likelihood
            acceptance_count += 1

    return new_means, acceptance_count

def block_weights_proposal(X, current_means, current_weights, current_covars, n_mixtures, step_size=0.001):

    accepted = 0

    if n_mixtures > 1:
        # grab n-1 points on the plane x1 + x2 + x3 ..... =1
        points =  np.random.dirichlet([1 for i in xrange(n_mixtures)], size=n_mixtures-1)
        points = points.T

        plane_origin = np.ones((n_mixtures)) / float(n_mixtures)

        # get vectors parallel to plane from its center (1/n,1/n,....)
        parallel = points - np.ones(points.shape) / float(n_mixtures)

        # do gramm schmidt to get mutually orthonormal vectors (basis)
        e, _ = np.linalg.qr(parallel)

        current_weights_transformed = np.dot(e.T, current_weights - plane_origin)
        proposed_weights_transformed = np.random.multivariate_normal(current_weights_transformed,
                                                                     np.eye(n_mixtures-1) * step_size)
        proposed_weights = plane_origin + np.dot(e, proposed_weights_transformed)

        new_weights = current_weights

        if np.logical_and(0 <= proposed_weights, proposed_weights <= 1).all():
            previous_likelihood = gmm_test.log_likelihood(X, n_mixtures, current_means, current_covars, current_weights)
            proposed_likelihood = gmm_test.log_likelihood(X, n_mixtures, current_means, current_covars, new_weights)

            # posterior
            previous_posterior = previous_likelihood
            proposed_posterior = proposed_likelihood

            # ratio
            ratio = proposed_posterior - previous_posterior
            if ratio > 0 or ratio > np.log(np.random.uniform()):
                # accept proposal
                new_weights = proposed_weights
                accepted = 1
    else:
        new_weights = current_weights

    return new_weights, accepted

@jit
def block_covariance_proposal(X, current_means, current_weights, current_covars, n_mixtures, step_size=0.001):
    # diagonal covariances
    """
    :param X:
    :param current_means:
    :param current_weights:
    :param prior_means:
    :param current_covars:
    :param n_mixtures:
    :param step_size:
    :return:
    """
    acceptance_count = 0
    new_covars = np.array(current_covars)
    previous_likelihood = gmm_test.log_likelihood(X, n_mixtures, current_means, current_covars, current_weights)

    for mixture in xrange(n_mixtures):
        # propose new means
        new_mixture_covars = np.random.multivariate_normal(current_covars[mixture], step_size * np.eye(X.shape[1]))
        if (new_mixture_covars > 0).all():
            # try out the new covars
            proposed_covars = np.array(new_covars)
            proposed_covars[mixture] = new_mixture_covars

            # distributions
            proposed_likelihood = gmm_test.log_likelihood(X, n_mixtures, current_means, proposed_covars, current_weights)

            # posterior
            previous_posterior = previous_likelihood
            proposed_posterior = proposed_likelihood

            #ratio
            ratio = proposed_posterior - previous_posterior
            if ratio > 0 or ratio > np.log(np.random.uniform()):
                # accept proposal
                new_covars = proposed_covars
                previous_likelihood = proposed_likelihood
                acceptance_count += 1

    return new_covars, acceptance_count


def gmm_mcmc(X, initial_weights, initial_means, initial_covars, n_runs=10000, n_mixtures=32):
    """
    Metropolis hastings algorithm for GMMs
    :param X:
    :param initial_weights:
    :param initial_means:
    :param initial_covars:
    :param n_runs:
    :param n_mixtures:
    :return:
    """
    mean_samples = np.zeros((n_runs, n_mixtures, X.shape[1]))
    covar_samples = np.zeros((n_runs, n_mixtures, X.shape[1]))
    weight_samples = np.zeros((n_runs, n_mixtures))


    current_means = initial_means
    current_weights = initial_weights
    current_covars = initial_covars

    means_acceptance_count = 0
    weights_acceptance_count = 0
    covars_acceptance_count = 0
    for run in xrange(n_runs):
        logging.info("Run: {0}".format(run))
        current_means, means_acceptance = block_means_proposal(X, current_means, current_weights,
                                                               current_covars, n_mixtures,
                                                               step_size=0.0007)
        current_weights, weights_acceptance = block_weights_proposal(X, current_means, current_weights,
                                                                     current_covars, n_mixtures,
                                                                     step_size=0.003)
        current_covars, covars_acceptance = block_covariance_proposal(X, current_means, current_weights,
                                                                     current_covars, n_mixtures,
                                                                     step_size=0.0003)
        means_acceptance_count += means_acceptance
        weights_acceptance_count += weights_acceptance
        covars_acceptance_count += covars_acceptance
        mean_samples[run] = current_means
        covar_samples[run] = current_covars
        weight_samples[run] = current_weights


    logging.info("Means Acceptance Rate: {0}".format(means_acceptance_count / float(n_mixtures * n_runs)))
    logging.info("Weights Acceptance Rate: {0}".format(weights_acceptance_count / float(n_runs)))
    logging.info("Covariance Acceptance Rate: {0}".format(covars_acceptance_count / float(n_mixtures * n_runs)))

    return (mean_samples, covar_samples, weight_samples)

