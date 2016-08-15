import gmmmc
import numpy as np
import bob.bio.gmm
from gmmmc.priors import MeansUniformPrior, CovarsStaticPrior, WeightsStaticPrior, GMMPrior, MeansGaussianPrior
from gmmmc.proposals import GMMBlockMetropolisProposal, GaussianStepCovarProposal, GaussianStepWeightsProposal, GaussianStepMeansProposal
from gmmmc import MarkovChain
import logging
import matplotlib.pyplot as plt

np.random.seed(3)
logging.getLogger().setLevel(logging.INFO)

covars = np.array([[0.01], [0.01]])
weights = np.array([0.5, 0.5])

prior_means = np.array([[-0.5], [-0.1]])
data_means = np.array([[0.8], [0.5]])

data_gmm = gmmmc.GMM(data_means, covars, weights)

artificial_data = data_gmm.sample(1000)

prior = GMMPrior(MeansGaussianPrior(prior_means, covars/2),
                 CovarsStaticPrior(covars),
                 WeightsStaticPrior(weights))

proposal = GMMBlockMetropolisProposal(propose_mean=GaussianStepMeansProposal(step_sizes=[0.0003, 0.001]),
                                      propose_covars=None,
                                      propose_weights=None)

initial_gmm = gmmmc.GMM(prior_means, covars, weights)

mcmc = MarkovChain(proposal, prior, initial_gmm)

samples = mcmc.sample(artificial_data, 20000, n_jobs=-1)

print proposal.propose_mean.get_acceptance()

final = samples[-1]

mc_means = [[s.means[0][0], s.means[1][0]] for s in samples[::10]]
mc_means = np.array(mc_means)

mcmc = plt.scatter(mc_means[:,0], mc_means[:,1], color= 'b')
true = plt.scatter(data_means[0][0], data_means[1][0], color='g', s=500)
prior = plt.scatter(prior_means[0][0], prior_means[1][0], color= 'y', s=500)
plt.title('Samples from Posterior Distribution of GMM Means', fontsize=22)
plt.xlabel('Mixture 1 mean', fontsize=22)
plt.ylabel('Mixture 2 mean', fontsize=22)

plt.legend((mcmc, prior, true),
           ('Monte Carlo Samples', 'Prior Means', 'Data Means'),
           scatterpoints=1,
           loc='lower left',
           ncol=2,
           fontsize=22)

plt.show()