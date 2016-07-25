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
from collections import defaultdict
import pdb


def create_data(n_mixtures, n_features, n_samples):
    # single mixture gmm
    truth_gmm = GMM(means=np.random.uniform(low=-1, high=1, size=(n_mixtures, n_features)),
                    covariances=np.random.uniform(low=0, high=1, size=(n_mixtures, n_features)),
                    weights=np.random.dirichlet(np.ones((n_mixtures))))

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

    gmm_ml = sklearn.mixture.GMM(n_components=n_mixtures, covariance_type='diag', n_iter=10000, n_init=100)
    gmm_ml.fit(X)

    ########### MCMC ##################################
    # setup monte carlo sampler
    #pdb.set_trace()

    scale = 1.0
    """
    prior = GMMPrior(#MeansGaussianPrior(gmm_ml.means_, covariances=np.ones((n_mixtures, X.shape[1]))*scale),
                     MeansUniformPrior(-1, 1, n_mixtures, X.shape[1]),
                     CovarsStaticPrior(gmm_ml.covars_),
                     WeightsStaticPrior(gmm_ml.weights_))
    proposal = GMMBlockMetropolisProposal(propose_mean=GaussianStepMeansProposal(step_sizes=[0.001, 0.01, 0.1]))

    """

    prior = GMMPrior(MeansUniformPrior(-1, 1, n_mixtures, X.shape[1]),
                     DiagCovarsUniformPrior(0.01, 1, n_mixtures, X.shape[1]),
                     WeightsUniformPrior(n_mixtures))

    proposal = GMMBlockMetropolisProposal(propose_mean=GaussianStepMeansProposal(step_sizes=[0.01, 0.05]),
                                          propose_covars=GaussianStepCovarProposal(step_sizes=[0.0001]),
                                          propose_weights=GaussianStepWeightsProposal(n_mixtures,
                                                                                      step_sizes=[0.01, 0.1]))

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

    """
    weights = np.array([sample.weights for sample in gmm_samples])

    plt.scatter(weights[:, 0], np.ones(weights.shape[0]), color = 'red')
    plt.scatter(weights[:, 1], np.ones(weights.shape[0]), color = 'black')
    plt.scatter(gmm_ml.weights_[0], 1.5, color='blue')
    plt.scatter(gmm_ml.weights_[1], 1.5, color='blue')
    plt.scatter(truth_gmm.weights[0], 2, color='green')
    plt.scatter(truth_gmm.weights[1], 2, color='green')


    """

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

    gmm_ml = sklearn.mixture.GMM(n_components=n_mixtures, covariance_type='diag', n_iter=100000, n_init=10, tol=1e-4, verbose=1)
    gmm_ml.fit(X)

    ################ AIS ####################################
    scale = 5
    prior_ais = GMMPrior(#MeansGaussianPrior(prior_means=gmm_ml.means_, covariances=np.ones((n_mixtures, X.shape[1]))*scale),
                         MeansUniformPrior(-1,1,n_mixtures,X.shape[1]),
                         DiagCovarsUniformPrior(0.01,1,n_mixtures, X.shape[1]),
                         WeightsUniformPrior(n_mixtures))
    proposal_ais = GMMBlockMetropolisProposal(propose_mean=GaussianStepMeansProposal(step_sizes=[0.05, 0.15]),
                                              propose_covars=GaussianStepCovarProposal(step_sizes=[0.05]),
                                              propose_weights=GaussianStepWeightsProposal(n_mixtures,step_sizes=[0.05]),
                                              propose_iterations=5)
    betas = [0.0]
    betas.extend(np.logspace(-2, 0, 100))
    #betas=np.linspace(0,1,100)
    ais_sampler = AnnealedImportanceSampling(proposal_ais, prior_ais, betas)

    diagnostics = {}

    ais_samples, logweights = ais_sampler.sample(X, n_samples, n_jobs, diagnostics)

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
    true_likelihood = [truth_gmm.log_likelihood(np.array([sample]), n_jobs=-1) for sample in test_samples]

    logging.info('AIS Means Acceptance: {0}'.format(proposal_ais.propose_mean.get_acceptance()))
    logging.info('AIS Covars Acceptance: {0}'.format(proposal_ais.propose_covars.get_acceptance()))
    logging.info('AIS Weights Acceptance: {0}'.format(proposal_ais.propose_weights.get_acceptance()))
    logging.info('AIS Likelihood:  {0}'.format(str(ais_likelihood)))
    logging.info('ML Likelihood:   {0}'.format(str(likelihood_ml)))
    logging.info('True Likelihood: {0}'.format(str(true_likelihood)))

    return diagnostics


def process_diagnostics(diagnostics):
    beta_v_mean = []
    beta_v_logweight = []
    mean_v_logweight = []
    beta_v_var_logweight = defaultdict(list)
    ais_samples = []
    weights = []
    for sample_dict in diagnostics.values():
        for i, beta in enumerate(sample_dict['intermediate_betas']):
            if i % 5 != 0:
                continue
            beta_v_mean.append((beta, sample_dict['intermediate_samples'][i].means[0]))
            beta_v_logweight.append((beta, sample_dict['intermediate_log_weights'][i]))
        mean_v_logweight.append((sample_dict['final_sample'].means[0], sample_dict['final_weight']))
        ais_samples.append(sample_dict['final_sample'])
        weights.append(sample_dict['final_sample'])

    for beta, logweight in beta_v_logweight:
        beta_v_var_logweight[beta].append(logweight)

    for beta, logweights in beta_v_var_logweight.iteritems():
        beta_v_var_logweight[beta] = np.var(logweights)

    beta_v_var_logweight = np.array([(beta, var) for beta, var in beta_v_var_logweight.iteritems()])

    beta_v_mean = np.array(beta_v_mean)
    beta_v_logweight = np.array(beta_v_logweight)
    mean_v_logweight = np.array(mean_v_logweight)


    weights = np.array([sample.covars for sample in ais_samples])
    plt.figure(5)
    plt.title('weights')
    plt.scatter(weights[:, 0], np.ones(weights.shape[0]), color = 'red')
    plt.scatter(weights[:, 1], np.ones(weights.shape[0]), color = 'black')
    plt.scatter(gmm_ml.weights_[0], 1.5, color='blue')
    plt.scatter(gmm_ml.weights_[1], 1.5, color='blue')
    plt.scatter(truth_gmm.weights[0], 2, color='green')
    plt.scatter(truth_gmm.weights[1], 2, color='green')


    covars = np.array([sample.covars for sample in ais_samples])
    plt.figure(6)
    plt.title('covars')
    plt.scatter(covars[:, 0], np.ones(covars.shape[0]), color = 'red')
    plt.scatter(covars[:, 1], np.ones(covars.shape[0]), color = 'black')
    plt.scatter(gmm_ml.covars_[0], 1.5, color='blue')
    plt.scatter(gmm_ml.covars_[1], 1.5, color='blue')
    plt.scatter(truth_gmm.covars[0], 2, color='green')
    plt.scatter(truth_gmm.covars[1], 2, color='green')


    plt.figure(0)
    plt.title('means')
    means = np.array([sample.means for sample in ais_samples])
    plt.scatter(means[:, 0], np.ones(means.shape[0]), color = 'red')
    plt.scatter(means[:, 1], np.ones(means.shape[0]), color = 'black')
    plt.scatter(gmm_ml.means_[0], 1.5, color='blue')
    plt.scatter(gmm_ml.means_[1], 1.5, color='blue')
    plt.scatter(truth_gmm.means[0], 2, color='green')
    plt.scatter(truth_gmm.means[1], 2, color='green')

    plt.draw()

    plt.figure(1)
    plt.title('beta v logweight')
    plt.scatter(beta_v_logweight[:,0],beta_v_logweight[:,1])
    plt.draw()

    plt.figure(2)
    plt.title('beta v mean component')
    plt.scatter(beta_v_mean[:,0],beta_v_mean[:,1])
    plt.draw()

    plt.figure(3)
    plt.title('beta vs variance of log weights')
    plt.scatter(beta_v_var_logweight[:,0], beta_v_var_logweight[:,1])
    plt.draw()

    plt.figure(4)
    plt.title('mean component vs logweight')
    plt.scatter(mean_v_logweight[:,0],mean_v_logweight[:,1])
    plt.show()



if __name__=='__main__':
    logging.getLogger().setLevel(logging.INFO)
    create_data(n_mixtures=16, n_features=64, n_samples=1000)

    truth_gmm, X = load_data(n_mixtures=16, n_features=64)

    #np.random.seed(5)
    #truth_gmm = GMM(np.array([[0.0], [0.5]]), np.array([[0.01], [0.5]]), np.array([0.3, 0.7]))
    #X = truth_gmm.sample(1000)

    #start = time.time()
    #evaluate_mcmc( X, truth_gmm, n_mixtures=2, n_runs=10000, n_jobs=-1)
    #print time.time() - start

    start = time.time()
    diagnostics = evaluate_ais( X, truth_gmm, n_mixtures=16, n_samples=10, n_jobs=-1)
    print time.time() - start

    #process_diagnostics(diagnostics)
