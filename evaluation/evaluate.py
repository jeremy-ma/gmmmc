import cPickle
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from scipy.misc import logsumexp
import sys
sys.path.append('/home/jeremy/Documents/gmmmc/')
from gmmmc.priors import *
from gmmmc.proposals import *
from gmmmc import *


def create_data(n_mixtures, n_features, n_samples):
    # single mixture gmm
    truth_gmm = GMM(means=np.random.uniform(low=-1, high=1, size=(n_mixtures, n_features)),
                    weights=np.random.dirichlet(np.ones((n_mixtures))),
                    covariances=np.random.uniform(low=0, high=1, size=(n_mixtures, n_features)))

    # draw samples from the true distribution
    X = truth_gmm.sample(n_samples)
    with open('/home/jeremy/Documents/gmmmc/evaluation/pickledgmm_n_mixtures{0}_n_features{1}'.format(n_mixtures, n_features),'w') as fp:
        cPickle.dump((truth_gmm, X), fp)

def load_data(n_mixtures, n_features):
    with open('/home/jeremy/Documents/gmmmc/evaluation/pickledgmm_n_mixtures{0}_n_features{1}'.format(n_mixtures, n_features)) as fp:
        truth_gmm, X = cPickle.load(fp)
    return (truth_gmm, X)

def evaluate_mcmc( X, truth_gmm, n_mixtures, n_runs, n_jobs=1):

    ################ ML Estimate #####################

    gmm_ml = sklearn.mixture.GMM(n_components=n_mixtures, covariance_type='diag')
    gmm_ml.fit(X)

    ########### MCMC ##################################
    # setup monte carlo sampler
    #pdb.set_trace()

    scale = 1.0
    prior = GMMPrior(#MeansGaussianPrior(gmm_ml.means_, covariances=np.ones((n_mixtures, X.shape[1]))*scale),
                     MeansUniformPrior(-1, 1, n_mixtures, X.shape[1]),
                     CovarsStaticPrior(gmm_ml.covars_),
                     WeightsStaticPrior(gmm_ml.weights_))
    proposal = GMMBlockMetropolisProposal(propose_mean=GaussianStepMeansProposal(step_sizes=[0.001, 0.01, 0.1]))

    """

    prior = GMMPrior(MeansUniformPrior(-1, 1, n_mixtures, X.shape[1]),
                     DiagCovarsUniformPrior(0.01, 1, n_mixtures, X.shape[1]),
                     WeightsUniformPrior(n_mixtures))

    target = GMMPosteriorTarget(prior)

    proposal = GMMBlockMetropolisProposal(propose_mean=GaussianStepMeansProposal(step_sizes=[0.05, 0.15, 0.5]),
                                          propose_covars=GaussianStepCovarProposal(step_sizes=[0.0001]),
                                          propose_weights=GaussianStepWeightsProposal(n_mixtures,
                                                                                      step_sizes=[0.01, 0.1]))
    """
    initial_gmm = GMM(means=gmm_ml.means_,
                      weights=gmm_ml.weights_,
                      covariances=gmm_ml.covars_)

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
    #logging.info('Covars Acceptance: {0}'.format(proposal.propose_covars.get_acceptance()))
    #logging.info('Weights Acceptance: {0}'.format(proposal.propose_weights.get_acceptance()))
    logging.info('MCMC Likelihood: {0}'.format(str(markov_chain_likelihood)))
    logging.info('ML Estimate Likelihood: {0}'.format(str(likelihood_ml)))
    logging.info('True Likelihood: {0}'.format(str(true_likelihood)))

    covars = np.array([sample.covars for sample in gmm_samples])
    means = np.array([sample.means for sample in gmm_samples])
    plt.scatter(means[:, 0], np.ones(means.shape[0]), color = 'red')
    plt.scatter(means[:, 1], np.ones(means.shape[0]), color = 'black')
    plt.scatter(gmm_ml.means_[0], 1.5, color='blue')
    plt.scatter(gmm_ml.means_[1], 1.5, color='blue')
    plt.scatter(truth_gmm.means[0], 2, color='green')
    plt.scatter(truth_gmm.means[1], 2, color='green')
    plt.show()

def evaluate_ais(X, truth_gmm, n_mixtures = 1,  n_samples = 10000, n_jobs=1):

    # get test sample
    sample = np.array([X[0]])

    ################ ML Estimate #####################

    gmm_ml = sklearn.mixture.GMM(n_components=n_mixtures, covariance_type='diag', n_iter=100000)
    gmm_ml.fit(X)

    ################ AIS ####################################
    scale = 5
    prior_ais = GMMPrior(#MeansGaussianPrior(prior_means=gmm_ml.means_, covariances=np.ones((n_mixtures, X.shape[1]))*scale),
                         MeansUniformPrior(-1,1,n_mixtures,X.shape[1]),
                         DiagCovarsUniformPrior(0.01,1,n_mixtures, X.shape[1]),
                         WeightsUniformPrior(n_mixtures))
    proposal_ais = GMMBlockMetropolisProposal(propose_mean=GaussianStepMeansProposal(step_sizes=[0.05, 0.15, 0.5]),
                                              propose_covars=GaussianStepCovarProposal(step_sizes=[0.1]),
                                              propose_weights=GaussianStepWeightsProposal(n_mixtures,step_sizes=[0.01,0.1]),
                                              propose_iterations=5)
    ais_sampler = AnnealedImportanceSampling(proposal_ais, prior_ais, betas=np.logspace(0, 1, 50))

    ais_samples, logweights = ais_sampler.sample(X, n_samples, n_jobs)

    ##############################################
    test_samples = truth_gmm.sample(10)

    # calculate estimated likelihood with importance mean samples
    numerator = [[logweights[i] + gmm.log_likelihood(np.array([sample])) for i, gmm in enumerate(ais_samples)] for sample in test_samples]
    numerator = np.array(numerator)
    numerator = logsumexp(numerator, axis=1)
    denominator = logsumexp(logweights)
    ais_likelihood = numerator - denominator
    ais_likelihood = [x for x in ais_likelihood]

    likelihood_ml = [np.sum(gmm_ml.score(np.array([sample]))) for sample in test_samples]
    true_likelihood = [truth_gmm.log_likelihood(np.array([sample])) for sample in test_samples]

    logging.info('AIS Means Acceptance: {0}'.format(proposal_ais.propose_mean.get_acceptance()))
    logging.info('AIS Covars Acceptance: {0}'.format(proposal_ais.propose_covars.get_acceptance()))
    logging.info('AIS Weights Acceptance: {0}'.format(proposal_ais.propose_weights.get_acceptance()))
    logging.info('AIS Likelihood:  {0}'.format(str(ais_likelihood)))
    logging.info('ML Likelihood:   {0}'.format(str(likelihood_ml)))
    logging.info('True Likelihood: {0}'.format(str(true_likelihood)))

    means = np.array([sample.means for sample in ais_samples])
    plt.scatter(means[:, 0], np.ones(means.shape[0]), color = 'red')
    plt.scatter(means[:, 1], np.ones(means.shape[0]), color = 'black')
    plt.scatter(gmm_ml.means_[0], 1.5, color='blue')
    plt.scatter(gmm_ml.means_[1], 1.5, color='blue')
    plt.scatter(truth_gmm.means[0], 2, color='green')
    plt.scatter(truth_gmm.means[1], 2, color='green')
    plt.show()


if __name__=='__main__':
    logging.getLogger().setLevel(logging.INFO)
    # create_data(n_mixtures=2, n_features=1, n_samples=1000)

    truth_gmm, X = load_data(n_mixtures=2, n_features=1)

    print truth_gmm.means
    print truth_gmm.covars
    print truth_gmm.weights

    #start = time.time()
    #evaluate_mcmc( X, truth_gmm, n_mixtures=2, n_runs=10000, n_jobs=-1)
    #print time.time() - start

    start = time.time()
    evaluate_ais( X, truth_gmm, n_mixtures=2, n_samples=100, n_jobs=-1)
    print time.time() - start


