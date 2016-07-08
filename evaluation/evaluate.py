from distributions.posterior import GMMPosteriorTarget
from distributions.prior import GMMPrior, GMMMeansUniformPrior, GMMCovarsStaticPrior, GMMWeightsStaticPrior
from distributions.gmm import GMM
from monte_carlo.proposals import GMMBlockMetropolisProposal, GaussianStepCovarProposal, GaussianStepMeansProposal, GaussianStepWeightsProposal
from monte_carlo.monte_carlo import MarkovChain
from scipy.misc import logsumexp
import numpy as np
import logging

def evaluate(n_mixtures = 1, n_features = 1, n_runs = 10000):
    # single mixture gmm
    truth_gmm = GMM(means=np.random.uniform(low=-1, high=1, size=(n_mixtures, n_features)),
                    weights=np.random.dirichlet(np.ones((n_mixtures))),
                    covariances=np.random.uniform(low=0, high=1, size=(n_mixtures, n_features)))

    # draw samples from the true distribution
    X = truth_gmm.sample(n_samples=1000)

    # get test sample
    sample = np.array([X[0]])

    # setup monte carlo sampler
    prior = GMMPrior(GMMMeansUniformPrior(low=-1, high=1, n_mixtures=n_mixtures, n_features=n_features),
                     GMMCovarsStaticPrior(prior_covars=truth_gmm.covars),
                     GMMWeightsStaticPrior(prior_weights=truth_gmm.weights))
    target = GMMPosteriorTarget(prior)
    proposal = GMMBlockMetropolisProposal(GaussianStepMeansProposal(step_size=0.0005))
    initial_gmm = GMM(means=np.random.uniform(low=-1, high=1, size=(n_mixtures, n_features)),
                      weights=np.array(truth_gmm.weights), covariances=np.array(truth_gmm.covars))
    mc = MarkovChain(proposal, target, initial_gmm)

    # make samples
    gmm_samples = mc.sample(X, n_samples=n_runs)

    # discard gmm samples
    gmm_samples[int(n_runs/2)::50]

    markov_chain_likelihood = logsumexp([gmm.log_likelihood(sample) for gmm in gmm_samples]) - np.log(len(gmm_samples))
    true_likelihood = truth_gmm.log_likelihood(sample)

    logging.info('Means Acceptance: {0}'.format(proposal.get_means_acceptance()))
    logging.info('MCMC Likelihood: {0}'.format(markov_chain_likelihood))
    logging.info('True Likelihood: {0}'.format(true_likelihood))

if __name__=='__main__':
    logging.getLogger().setLevel(logging.INFO)
    evaluate(n_mixtures=64, n_features=60, n_runs=20000)