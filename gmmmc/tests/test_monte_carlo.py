import logging
from unittest import TestCase

from scipy.misc import logsumexp

from gmmmc.priors.prior import *
from gmmmc.monte_carlo import MarkovChain
from gmmmc.posterior import GMMPosteriorTarget


class TestMarkovChain(TestCase):

    def setUp(self):
        np.random.seed(0)
        self.n_mixtures = 4
        self.n_features = 4
        self.n_samples = 1000
        self.truth_gmm = GMM(means=np.random.uniform(low=-1, high=1, size=(self.n_mixtures, self.n_features)),
                             covariances=np.random.uniform(low=0, high=1, size=(self.n_mixtures, self.n_features)),
                             weights=np.random.dirichlet(np.ones(self.n_mixtures)))
        # draw samples from the true distribution
        self.X = self.truth_gmm.sample(self.n_samples)
        logging.getLogger().setLevel(logging.INFO)

    def test_sample(self):
        pass

    def test_simple_means_only(self):
        prior = GMMPrior(MeansUniformPrior(-1, 1, self.n_mixtures, self.n_features),
                         CovarsStaticPrior(self.truth_gmm.covars),
                         WeightsStaticPrior(self.truth_gmm.weights))
        target = GMMPosteriorTarget(prior)
        proposal = GMMBlockMetropolisProposal(propose_mean=GaussianStepMeansProposal(step_size=0.003))
        initial_gmm = GMM(means=np.random.uniform(low=-1, high=1, size=(self.n_mixtures, self.n_features)),
                          covariances=self.truth_gmm.covars, weights=self.truth_gmm.weights)
        mc = MarkovChain(proposal, target, initial_gmm)
        # make samples
        gmm_samples = mc.sample(self.X, n_samples=10000, n_jobs=-1)
        # discard gmm samples
        gmm_samples = gmm_samples[int(10000 / 2)::50]
        sample = self.X[0]
        mcmc_likelihood = logsumexp([gmm.log_likelihood(np.array([sample]), n_jobs=-1) for gmm in gmm_samples])
        mcmc_likelihood = mcmc_likelihood - np.log(len(gmm_samples))
        true_likelihood = self.truth_gmm.log_likelihood(np.array([sample]))
        logging.info('MCMC Means Acceptance: {0}'.format(proposal.propose_mean.get_acceptance()))
        logging.info('MCMC Likelihood: {0}'.format(str(mcmc_likelihood)))
        logging.info('True Likelihood: {0}'.format(str(true_likelihood)))
        if abs(mcmc_likelihood - true_likelihood) > 2:
            self.fail("MCMC failed to converge")
