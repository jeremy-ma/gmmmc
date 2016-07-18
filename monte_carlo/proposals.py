import abc
import numpy as np
import pdb
from distributions.gmm import GMM


class Proposal(object):
    def __init__(self):
        self.count_proposed = 0
        self.count_accepted = 0
        self.count_illegal = 0

    def get_acceptance(self):
        return self.count_accepted / float(self.count_proposed)

    def get_illegal(self):
        return self.count_illegal / float(self.count_proposed)

    """ generic proposal function"""
    @abc.abstractmethod
    def propose(self, X, target, gmm):
        pass

class GaussianStepMeansProposal(Proposal):
    def __init__(self, step_size=0.001):
        super(GaussianStepMeansProposal, self).__init__()
        self.step_size = step_size

    def propose(self, X, target, gmm, n_jobs=1):
        new_means = np.array(gmm.means)
        previous_prob = target.log_prob(X, gmm)
        for mixture in xrange(gmm.n_mixtures):
            self.count_proposed += 1
            # propose new means
            new_mixture_means = np.random.multivariate_normal(gmm.means[mixture], self.step_size * np.eye(X.shape[1]))

            # try out the new means
            proposed_means = np.array(new_means)
            proposed_means[mixture] = new_mixture_means

            proposed_gmm = GMM(proposed_means, np.array(gmm.weights), np.array(gmm.covars), n_jobs)

            # distributions
            proposed_prob = target.log_prob(X, proposed_gmm)

            # ratio
            ratio = proposed_prob - previous_prob

            if ratio > 0 or ratio > np.log(np.random.uniform()):
                # accept proposal
                new_means = proposed_means
                previous_prob = proposed_prob
                self.count_accepted += 1

        return GMM(new_means, np.array(gmm.weights), np.array(gmm.covars))

class GaussianStepCovarProposal(Proposal):
    def __init__(self, step_size=0.001):
        super(GaussianStepCovarProposal, self).__init__()
        self.step_size = step_size

    def propose(self, X, target, gmm, n_jobs=1):
        new_covars = np.array(gmm.covars)
        previous_prob = target.log_prob(X, gmm)

        for mixture in xrange(gmm.n_mixtures):
            self.count_proposed += 1
            # propose new covars
            new_mixture_covars = np.random.multivariate_normal(gmm.covars[mixture], self.step_size * np.eye(X.shape[1]))
            if (new_mixture_covars > 0).all(): # check covariances are valid
                # try out the new covars
                proposed_covars = np.array(new_covars)
                proposed_covars[mixture] = new_mixture_covars
                proposed_gmm = GMM(np.array(gmm.means), np.array(gmm.weights), proposed_covars, n_jobs)

                # distributions
                proposed_prob = target.log_prob(X, proposed_gmm)

                # ratio
                ratio = proposed_prob - previous_prob
                if ratio > 0 or ratio > np.log(np.random.uniform()):
                    # accept proposal
                    new_covars = proposed_covars
                    previous_prob = proposed_prob
                    self.count_accepted += 1
            else:
                self.count_illegal += 1

        return GMM(np.array(gmm.means), np.array(gmm.weights), np.array(new_covars))

class GaussianStepWeightsProposal(Proposal):
    def __init__(self,  n_mixtures, step_size=0.001):
        super(GaussianStepWeightsProposal, self).__init__()
        self.step_size = step_size
        self.n_mixtures = n_mixtures

        if n_mixtures > 1:
            # get change of basis matrix mapping n dim coodinates to n-1 dim coordinates on simplex
            # x1 + x2 + x3 ..... =1
            points = np.random.dirichlet([1 for i in xrange(n_mixtures)], size=n_mixtures - 1)
            points = points.T
            self.plane_origin = np.ones((n_mixtures)) / float(n_mixtures)
            # get vectors parallel to plane from its center (1/n,1/n,....)
            parallel = points - np.ones(points.shape) / float(n_mixtures)
            # do gramm schmidt to get mutually orthonormal vectors (basis)
            self.e, _ = np.linalg.qr(parallel)

    def transformSimplex(self, weights):
        # project onto the simplex
        return np.dot(self.e.T, weights - self.plane_origin)

    def invTransformSimplex(self, simplex_coords):
        return self.plane_origin + np.dot(self.e, simplex_coords)

    def propose(self, X, target, gmm, n_jobs):
        self.count_proposed += 1
        accepted = False
        if gmm.n_mixtures > 1:
            current_weights_transformed = self.transformSimplex(gmm.weights)
            proposed_weights_transformed = np.random.multivariate_normal(current_weights_transformed,
                                                                         np.eye(self.n_mixtures - 1) * self.step_size)
            proposed_weights = self.invTransformSimplex(proposed_weights_transformed)
            if np.logical_and(0 <= proposed_weights, proposed_weights <= 1).all()\
                and np.isclose(np.sum(proposed_weights), 1.0):
                previous_prob = target.log_prob(X, gmm)
                proposed_gmm = GMM(np.array(gmm.means), proposed_weights, np.array(gmm.covars), n_jobs)
                proposed_prob = target.log_prob(X, proposed_gmm)
                ratio = proposed_prob - previous_prob
                if ratio > 0 or ratio > np.log(np.random.uniform()):
                    # accept proposal
                    self.count_accepted += 1
                    accepted = True
            else:
                self.count_illegal += 1

        if accepted is True:
            return proposed_gmm
        else:
            return GMM(np.array(gmm.means), np.array(gmm.weights), np.array(gmm.covars))

class GMMBlockMetropolisProposal(Proposal):

    def __init__(self, propose_mean=None, propose_covars=None, propose_weights=None, propose_iterations=1, n_jobs=1):
        self.propose_mean = propose_mean
        self.propose_covars = propose_covars
        self.propose_weights = propose_weights
        self.propose_iterations = propose_iterations
        self.n_jobs = n_jobs

    def propose(self, X, target, gmm):
        new_gmm = gmm
        for _ in xrange(self.propose_iterations):
            if self.propose_mean is not None:
                new_gmm = self.propose_mean.propose(X, target, new_gmm, self.n_jobs)

            if self.propose_covars is not None:
                new_gmm = self.propose_covars.propose(X, target, new_gmm, self.n_jobs)

            if self.propose_weights is not None:
                new_gmm = self.propose_weights.propose(X, target, new_gmm, self.n_jobs)

        return new_gmm

    def get_means_acceptance(self):
        return self.propose_mean.get_acceptance()