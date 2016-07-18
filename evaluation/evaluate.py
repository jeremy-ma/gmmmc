from distributions.posterior import GMMPosteriorTarget
from distributions.prior import *
from distributions.gmm import GMM
from monte_carlo.proposals import *
from monte_carlo.monte_carlo import MarkovChain, AnnealedImportanceSampling
from scipy.misc import logsumexp
import multiprocessing
import numpy as np
import logging
import sklearn
import cPickle
import time

def create_data(n_mixtures=1, n_features=1, n_samples=1000):
    # single mixture gmm
    truth_gmm = GMM(means=np.random.uniform(low=-1, high=1, size=(n_mixtures, n_features)),
                    weights=np.random.dirichlet(np.ones((n_mixtures))),
                    covariances=np.random.uniform(low=0, high=1, size=(n_mixtures, n_features)))

    # draw samples from the true distribution
    X = truth_gmm.sample(n_samples)
    with open('pickledgmm_n_mixtures{0}_n_features{1}'.format(n_mixtures, n_features),'w') as fp:
        cPickle.dump((truth_gmm, X), fp)

def evaluate_mcmc( X, truth_gmm, n_mixtures = 1, n_runs = 10000, n_jobs=1):

    # get test sample
    sample = np.array([X[0]])

    ################ ML Estimate #####################

    gmm_ml = sklearn.mixture.GMM(n_components=n_mixtures, covariance_type='diag')
    gmm_ml.fit(X)
    likelihood_ml = np.sum(gmm_ml.score(sample))

    ########### MCMC ##################################
    # setup monte carlo sampler
    #pdb.set_trace()
    scale = 1.0
    prior = GMMPrior(MeansGaussianPrior(gmm_ml.means_, covariances=np.ones((n_mixtures, X.shape[1]))*scale),
                     CovarsStaticPrior(gmm_ml.covars_),
                     WeightsStaticPrior(gmm_ml.weights_))
    # DiagCovarsUniformPrior(low=0, high=1,n_mixtures=n_mixtures, n_features=n_features),
    # WeightsUniformPrior(n_mixtures=n_mixtures))
    target = GMMPosteriorTarget(prior)
    proposal = GMMBlockMetropolisProposal(propose_mean=GaussianStepMeansProposal(step_size=0.001), n_jobs=n_jobs)
    # propose_weights=GaussianStepWeightsProposal(n_mixtures, step_size=0.0005),
    # propose_covars=GaussianStepCovarProposal(step_size=0.001))
    initial_gmm = GMM(means=gmm_ml.means_, weights=gmm_ml.weights_, covariances=gmm_ml.covars_)
    mc = MarkovChain(proposal, target, initial_gmm)
    # make samples
    gmm_samples = mc.sample(X, n_samples=n_runs)
    # discard gmm samples
    gmm_samples[int(n_runs / 2)::50]
    markov_chain_likelihood = logsumexp([gmm.log_likelihood(sample) for gmm in gmm_samples]) - np.log(len(gmm_samples))
    true_likelihood = truth_gmm.log_likelihood(sample)

    logging.info('MCMC Means Acceptance: {0}'.format(proposal.propose_mean.get_acceptance()))
    logging.info('MCMC Likelihood: {0}'.format(markov_chain_likelihood))
    logging.info('ML Estimate Likelihood: {0}'.format(likelihood_ml))
    logging.info('True Likelihood: {0}'.format(true_likelihood))


def evaluate_ais(X, truth_gmm, n_mixtures = 1,  n_samples = 200, n_jobs=1):

    # get test sample
    sample = np.array([X[0]])

    ################ ML Estimate #####################

    gmm_ml = sklearn.mixture.GMM(n_components=n_mixtures, covariance_type='diag')
    gmm_ml.fit(X)
    likelihood_ml = np.sum(gmm_ml.score(sample))

    ################ AIS ####################################
    scale = 0.5
    prior_ais = GMMPrior(MeansGaussianPrior(prior_means=gmm_ml.means_, covariances=np.ones((n_mixtures, X.shape[1])*scale)),
                         CovarsStaticPrior(prior_covars=gmm_ml.covars_),
                         WeightsStaticPrior(prior_weights=gmm_ml.weights_))
    proposal_ais = GMMBlockMetropolisProposal(propose_mean=GaussianStepMeansProposal(step_size=0.0001), propose_iterations=10, n_jobs=n_jobs)
    ais_sampler = AnnealedImportanceSampling(proposal_ais, prior_ais, betas=np.linspace(0, 1, 100))

    ais_samples, logweights = ais_sampler.sample(X, n_samples)
    # calculate estimated likelihood with importance mean samples
    numerator = [logweights[i] + gmm.log_likelihood(sample) for i, gmm in enumerate(ais_samples)]
    numerator = logsumexp(numerator)
    denominator = logsumexp(logweights)
    ais_likelihood = numerator - denominator

    true_likelihood = truth_gmm.log_likelihood(sample)

    logging.info('AIS Means Acceptance: {0}'.format(proposal_ais.get_means_acceptance()))
    logging.info('AIS Likelihood: {0}'.format(ais_likelihood))
    logging.info('ML Estimate Likelihood: {0}'.format(likelihood_ml))
    logging.info('True Likelihood: {0}'.format(true_likelihood))

def load_data(n_mixtures=32, n_features=12):
    with open('pickledgmm_n_mixtures{0}_n_features{1}'.format(n_mixtures, n_features)) as fp:
        truth_gmm, X = cPickle.load(fp)
    return (truth_gmm, X)

if __name__=='__main__':
    logging.getLogger().setLevel(logging.INFO)
    create_data(n_mixtures=128, n_features=64, n_samples=1000)
    truth_gmm, X = load_data(n_mixtures=128, n_features=64)
    start = time.time()
    evaluate_mcmc( X, truth_gmm, n_mixtures=128, n_runs=100, n_jobs=-1)
    print time.time() - start
    #start = time.time()
    #evaluate_ais( X, truth_gmm, n_mixtures=32, n_samples=2, n_jobs=2)
    #print time.time() - start


