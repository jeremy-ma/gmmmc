__author__ = 'jeremyma'
from trial import metropolis_hastings
from sklearn.mixture import GMM
import numpy as np
import pdb
from numba import jit
from trial.blockedMetropolisHastings import MCMCRun

@jit
def evaluate_metropolis_hastings(n_mixtures=1, n_features=1, n_runs=10000):
    # single mixture gmm
    truth_gmm = GMM(n_components=n_mixtures, covariance_type='diag')
    truth_gmm.weights_ = np.random.dirichlet(np.ones((n_mixtures)))
    truth_gmm.covars_ = np.random.uniform(low=0, high=1, size=(n_mixtures, n_features))
    truth_gmm.means_ = np.random.uniform(low=-1, high=1, size=(n_mixtures, n_features))

    # draw samples
    X = truth_gmm.sample(n_samples=1000)
    initial_means = np.random.uniform(low=-1, high=1, size=(n_mixtures, n_features))
    print initial_means
    print truth_gmm.means_

    X_monte = metropolis_hastings.gmm_mcmc(X,
                                           truth_gmm.weights_,
                                           initial_means,
                                           truth_gmm.covars_,
                                           n_runs,
                                           n_mixtures)
    print np.mean(X_monte,axis=0)


if __name__ == '__main__':
    evaluate_metropolis_hastings(n_mixtures=128, n_features=64, n_runs=200)
    # naive importance sampling
    # profile likelihood stuff

    # gamma distribution->prior for gammma
    # wishart distribution proposal for covariance