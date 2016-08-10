import cPickle
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from scipy.misc import logsumexp
import sys
from gmmmc.priors import *
from gmmmc.proposals import *
from gmmmc import *
from collections import defaultdict


def create_data(n_mixtures, n_features, n_samples):
    # single mixture gmm
    truth_gmm = GMM(means=np.random.uniform(low=-1, high=1, size=(n_mixtures, n_features)),
                    covariances=np.random.uniform(low=0, high=1, size=(n_mixtures, n_features)),
                    weights=np.random.dirichlet(np.ones((n_mixtures))))

    # draw samples from the true distribution
    X = truth_gmm.sample(n_samples)
    return (X, truth_gmm)


def evaluate_mcmc( X, truth_gmm, n_mixtures, n_runs, n_jobs=1):

    ################ ML Estimate #####################
    gmm_ml = sklearn.mixture.GMM(n_components=n_mixtures, covariance_type='diag', n_iter=10000)
    gmm_ml.fit(X)
    print "finished ML fit"
    ########### MCMC ##################################
    # setup monte carlo sampler
    prior = GMMPrior(MeansUniformPrior(-1, 1, n_mixtures, X.shape[1]),
                     DiagCovarsUniformPrior(0.01, 1, n_mixtures, X.shape[1]),
                     WeightsUniformPrior(n_mixtures))

    proposal = GMMBlockMetropolisProposal(propose_mean=GaussianStepMeansProposal(step_sizes=[0.01]),
                                          propose_covars=GaussianStepCovarProposal(step_sizes=[0.0001]),
                                          propose_weights=GaussianStepWeightsProposal(n_mixtures,
                                                                                      step_sizes=[0.0]))

    initial_gmm = GMM(means=gmm_ml.means_, covariances=gmm_ml.covars_, weights=gmm_ml.weights_)

    mc = MarkovChain(proposal, prior, initial_gmm)
    # make samples
    gmm_samples = mc.sample(X, n_samples=n_runs, n_jobs=n_jobs)

    # discard gmm samples
    gmm_samples[int(n_runs / 2)::50]

    #################################################################################################################
    test_samples = truth_gmm.sample(10)
    likelihood_ml = [np.sum(gmm_ml.score(np.array([sample]))) for sample in test_samples]
    markov_chain_likelihood = [logsumexp([gmm.log_likelihood(np.array([sample]), n_jobs=-1) for gmm in gmm_samples]) - np.log(len(gmm_samples))\
                               for sample in test_samples]
    true_likelihood = [truth_gmm.log_likelihood(np.array([sample])) for sample in test_samples]

    logging.info('Means Acceptance: {0}'.format(proposal.propose_mean.get_acceptance()))
    logging.info('Covars Acceptance: {0}'.format(proposal.propose_covars.get_acceptance()))
    logging.info('Weights Acceptance: {0}'.format(proposal.propose_weights.get_acceptance()))
    logging.info('MCMC Likelihood: {0}'.format(str(markov_chain_likelihood)))
    logging.info('ML Estimate Likelihood: {0}'.format(str(likelihood_ml)))
    logging.info('True Likelihood: {0}'.format(str(true_likelihood)))



if __name__ == '__main__':
    X, truth_gmm = create_data(n_mixtures=16, n_features=64, n_samples=2000)
    start = time.time()
    evaluate_mcmc(X, truth_gmm, n_mixtures=16, n_runs=100, n_jobs=-1)
    print time.time() - start
