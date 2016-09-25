import gmmmc
import numpy as np
import bob.bio.gmm
from gmmmc.priors import MeansUniformPrior, CovarsStaticPrior, WeightsStaticPrior, GMMPrior, MeansGaussianPrior,\
    DiagCovarsWishartPrior, WeightsUniformPrior
from gmmmc.proposals import GMMBlockMetropolisProposal, GaussianStepCovarProposal, GaussianStepWeightsProposal, GaussianStepMeansProposal
from gmmmc import MarkovChain
import logging
import matplotlib.pyplot as plt
import pdb

np.random.seed(3)
logging.getLogger().setLevel(logging.INFO)

covars = np.array([[0.01], [0.01]])
weights = np.array([0.5, 0.5])

prior_means = np.array([[-0.5], [-0.1]])
data_means = np.array([[0.8], [0.5]])

data_gmm = gmmmc.GMM(data_means, covars, weights)

artificial_data = data_gmm.sample(1000)

relevance_factor = 150

df = relevance_factor

scale = relevance_factor * covars

prior = GMMPrior(MeansGaussianPrior(prior_means, covars/relevance_factor),
                 DiagCovarsWishartPrior(df, scale),
                 WeightsUniformPrior(n_mixtures=2))

proposal = GMMBlockMetropolisProposal(propose_mean=GaussianStepMeansProposal(step_sizes=[0.0003, 0.001]),
                                      propose_covars=GaussianStepCovarProposal(step_sizes=[0.001]),
                                      propose_weights=GaussianStepWeightsProposal(2, step_sizes=[0.0001]))

initial_gmm = gmmmc.GMM(prior_means, covars, weights)

mcmc = MarkovChain(proposal, prior, initial_gmm)

samples = mcmc.sample(artificial_data, 20000, n_jobs=-1)

print "means acceptance: {0}".format(proposal.propose_mean.get_acceptance())
print "covars acceptance: {0}".format(proposal.propose_covars.get_acceptance())
print "weights acceptance: {0}".format(proposal.propose_weights.get_acceptance())

final = samples[-1]

mc_means = [[s.means[0][0], s.means[1][0]] for s in samples[::10]]
mc_means = np.array(mc_means)

mcmc_means = plt.scatter(mc_means[:,0], mc_means[:,1], color= 'b')
true_means = plt.scatter(data_means[0][0], data_means[1][0], color='g', s=500)
prior_means = plt.scatter(prior_means[0][0], prior_means[1][0], color= 'y', s=500)

plt.title('Samples from Posterior Distribution of GMM Means', fontsize=22)
plt.xlabel('Mixture 1 mean', fontsize=22)
plt.ylabel('Mixture 2 mean', fontsize=22)
plt.legend((mcmc_means, prior_means, true_means),
           ('Monte Carlo Samples', 'Prior Means', 'Data Means'),
           scatterpoints=1,
           loc='lower left',
           ncol=2,
           fontsize=22)

plt.show()

mc_covars = np.array([[s.covars[0][0], s.means[1][0]] for s in samples[::10]])
mcmc_covars = plt.scatter(mc_covars[:,0], mc_means[:,1], color='b')
true_covars = plt.scatter(covars[0][0], covars[1][0], color='g', s=500)

plt.title('Samples from Posterior Distribution of GMM Means', fontsize=22)
plt.xlabel('Mixture 1 covar', fontsize=22)
plt.ylabel('Mixture 2 covar', fontsize=22)
plt.legend((mcmc_covars, true_covars),
           ('Monte Carlo Samples', 'True Covars'),
           scatterpoints=1,
           loc='lower left',
           ncol=2,
           fontsize=22)

plt.show()

mc_weights = np.array([[s.weights[0], s.weights[1]] for s in samples[::10]])
mcmc_weights = plt.scatter(mc_weights[:,0], mc_weights[:,1], color='b')
true_covars = plt.scatter(weights[0], weights[1], color='g', s=500)

plt.title('Samples from Posterior Distribution of GMM Means', fontsize=22)
plt.xlabel('Mixture 1 weight', fontsize=22)
plt.ylabel('Mixture 2 weight', fontsize=22)
plt.legend((mcmc_covars, true_covars),
           ('Monte Carlo Samples', 'True Weights'),
           scatterpoints=1,
           loc='lower left',
           ncol=2,
           fontsize=22)

plt.show()