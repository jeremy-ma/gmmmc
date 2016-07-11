__author__ = 'jeremyma'
import logging

import numpy as np
from scipy.misc import logsumexp
from sklearn.mixture import GMM

from trial import importance_test, metropolis_hastings_test, gmm_test


def evaluate_metropolis_hastings(n_mixtures=1, n_features=1, n_runs=10000):
    # single mixture gmm
    truth_gmm = GMM(n_components=n_mixtures, covariance_type='diag')
    truth_gmm.weights_ = np.random.dirichlet(np.ones((n_mixtures)))
    truth_gmm.covars_ = np.random.uniform(low=0, high=1, size=(n_mixtures, n_features))
    truth_gmm.means_ = np.random.uniform(low=-1, high=1, size=(n_mixtures, n_features))

    # draw samples from the true distribution
    X = truth_gmm.sample(n_samples=1000)

    # get test sample
    sample = np.array([X[0]])

    # randomise starting value
    initial_means = np.random.uniform(low=-1, high=1, size=(n_mixtures, n_features))
    initial_weights = np.random.dirichlet(np.ones(n_mixtures))
    initial_covars = np.random.uniform(low=0, high=1, size=(n_mixtures, n_features))
    # get parameter samples
    means_monte, covar_samples, weight_samples = metropolis_hastings_test.gmm_mcmc(X,
                                                                                   initial_weights,
                                                                                   initial_means,
                                                                                   initial_covars,
                                                                                   n_runs,
                                                                                   n_mixtures)

    # discard samples
    means_monte = means_monte[10000::50]
    covar_samples = covar_samples[10000::50]
    weight_samples = weight_samples[10000::50]

    # calculate estimated likelihood with mean samples
    likelihoods = []
    for i in xrange(weight_samples.shape[0]):
        likelihoods.append(
            gmm_test.log_likelihood(sample, n_mixtures, means_monte[i], covar_samples[i], truth_gmm.weights_))

    likelihood_monte = logsumexp(likelihoods) - np.log(len(means_monte))

    means_importance, logweights_importance = importance_test.gmm_mean_sample(X,
                                                                              truth_gmm.weights_,
                                                                              truth_gmm.covars_,
                                                                              n_runs,
                                                                              n_mixtures)


    # calculate estimated likelihood with importance mean samples
    numerator = [logweights_importance[i] + \
                 gmm_test.log_likelihood(sample, n_mixtures, means_sample, truth_gmm.covars_, truth_gmm.weights_) \
                 for i, means_sample in enumerate(means_importance)]
    numerator = logsumexp(numerator)
    denominator = logsumexp(logweights_importance)
    likelihood_importance = numerator - denominator

    # calculate true likelihood
    likelihood_true = np.sum(truth_gmm.score(sample))

    # calculate ML estimate likelihood
    gmm_ml = GMM(n_components=n_mixtures, covariance_type='diag')
    gmm_ml.fit(X)
    likelihood_ml = np.sum(gmm_ml.score(sample))

    logging.info("True Likelihood: {0}".format(likelihood_true))
    logging.info("Metropolis Monte Carlo likelihood: {0}".format(likelihood_monte))
    logging.info("Importance Sampling Monte Carlo likelihood: {0}".format(likelihood_importance))
    logging.info("Maximum Likelihood Estimation likelihood: {0}".format(likelihood_ml))


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    evaluate_metropolis_hastings(n_mixtures=1, n_features=1, n_runs=20000)
    # naive importance sampling
    # profile distributions stuff

    # gamma distribution->prior for gammma
    # wishart distribution proposal for covariance