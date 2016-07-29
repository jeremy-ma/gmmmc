import numpy as np
from gmmmc.gmm import GMM
from gmmmc.proposals.proposals import Proposal
import pdb

class GaussianStepMeansProposal(Proposal):
    """Gaussian Proposal distribution for means of a GMM"""

    def __init__(self, step_sizes=(0.001,)):
        """
        Gaussian proposal distribution for the means. The multivariate Gaussian is centered at the means of the current
        state in the Markov Chain and has covariance given by step_sizes. Multiple step sizes can be specified.
        The proposal algorithm will take these steps in the sequence specified in step_sizes.
        Parameters
        ----------
        step_sizes : 1-D array_like
            Iterable containing the sequence of step sizes (covariances of the Gaussian proposal distribution"
        """
        super(GaussianStepMeansProposal, self).__init__()
        self.step_sizes = step_sizes
        self.count_accepted = np.zeros((len(step_sizes),))
        self.count_illegal =  np.zeros((len(step_sizes),))
        self.count_proposed = np.zeros((len(step_sizes),))

    def propose(self, X, gmm, target, n_jobs=1):
        """
        Propose a new set of GMM means.
        Parameters
        ----------
        X : 2-D array like of shape (n_samples, n_features)
            The observed data or evidence.
        gmm : GMM object
            The current state (set of gmm parameters) in the Markov Chain
        target : GMMPosteriorTarget object
            The target distribution to be found.
        n_jobs : int
            Number of cpu cores to use in the calculation of log probabilities.

        Returns
        -------
            : GMM
            A new GMM object initialised with new mean parameters.
        """
        new_means = np.array(gmm.means)
        beta = target.beta
        prior = target.prior
        steps = [np.random.multivariate_normal(np.zeros(gmm.n_features),
                                              step_size * np.eye(gmm.n_features),
                                              size=gmm.n_mixtures)
                 for step_size in self.step_sizes]

        # calculation of prior probabilities of only the means, since only means will change
        log_priors = np.array([prior.means_prior.log_prob_single(gmm.means[mixture], mixture) for mixture in xrange(gmm.n_mixtures)])
        log_prob_priors = np.sum(log_priors)
        previous_prob = beta * gmm.log_likelihood(X, n_jobs) + np.sum(log_priors)
        for i, step in enumerate(steps):
            for mixture in xrange(gmm.n_mixtures):
                self.count_proposed[i] += 1
                # propose new means
                new_mixture_means = gmm.means[mixture] + step[mixture]

                # try out the new means
                proposed_means = np.array(new_means)
                proposed_means[mixture] = new_mixture_means
                proposed_gmm = GMM(proposed_means, np.array(gmm.covars), np.array(gmm.weights))

                # calculate new prior
                new_log_prob_mixture = prior.means_prior.log_prob_single(new_mixture_means, mixture)
                new_log_prob_priors = log_prob_priors - log_priors[mixture] + new_log_prob_mixture

                # priors
                proposed_prob = beta * proposed_gmm.log_likelihood(X, n_jobs) + new_log_prob_priors
                # ratio

                ratio = proposed_prob - previous_prob
                if ratio > 0 or ratio > np.log(np.random.uniform()):
                    # accept proposal
                    new_means = proposed_means
                    previous_prob = proposed_prob
                    # update prior probability calculation
                    log_prob_priors = new_log_prob_priors
                    log_priors[mixture] = new_log_prob_mixture
                    self.count_accepted[i] += 1

        return GMM(new_means, np.array(gmm.covars), np.array(gmm.weights))

class GaussianStepCovarProposal(Proposal):

    def __init__(self, step_sizes=(0.001,)):
        """
        Gaussian proposal function for the covariances of the GMM.
        Parameters
        ----------
        step_sizes : array_like
            Array of covariance values for the Gaussian proposal.
        """
        super(GaussianStepCovarProposal, self).__init__()
        self.step_sizes = step_sizes
        self.count_accepted = np.zeros((len(step_sizes),))
        self.count_illegal =  np.zeros((len(step_sizes),))
        self.count_proposed = np.zeros((len(step_sizes),))

    def propose(self, X, gmm, target, n_jobs=1):
        """
        Propose a new set of GMM covariances (diagonal only).
        Parameters
        ----------
        X : 2-D array like of shape (n_samples, n_features)
            The observed data or evidence.
        gmm : GMM object
            The current state (set of gmm parameters) in the Markov Chain
        target : GMMPosteriorTarget object
            The target distribution to be found.
        n_jobs : int
            Number of cpu cores to use in the calculation of log probabilities.

        Returns
        -------
            : GMM
            A new GMM object initialised with new covariance parameters.
        """
        new_covars = np.array(gmm.covars)
        beta = target.beta
        prior = target.prior
        previous_prob = beta * gmm.log_likelihood(X, n_jobs) + prior.log_prob(gmm)
        steps = [np.random.multivariate_normal(np.zeros(gmm.n_features),
                                               step_size * np.eye(gmm.n_features),
                                               size=gmm.n_mixtures) for step_size in self.step_sizes]

        log_priors = np.array([prior.means_prior.log_prob_single(gmm.means[mixture], mixture) for mixture in xrange(gmm.n_mixtures)])
        log_prob_priors = np.sum(log_priors)
        for i, step in enumerate(steps):
            for mixture in xrange(gmm.n_mixtures):
                self.count_proposed[i] += 1
                # propose new covars
                new_mixture_covars = gmm.covars[mixture] + step[mixture]

                if (new_mixture_covars > 0).all(): # check covariances are valid
                    # try out the new covars
                    proposed_covars = np.array(new_covars)
                    proposed_covars[mixture] = new_mixture_covars
                    proposed_gmm = GMM(np.array(gmm.means), proposed_covars, np.array(gmm.weights))

                    # calculate desired distribution
                    new_log_prob_mixture = prior.covars_prior.log_prob_single(new_mixture_covars, mixture)
                    new_log_prob_priors = log_prob_priors - log_priors[mixture] + new_log_prob_mixture
                    proposed_prob = beta * proposed_gmm.log_likelihood(X, n_jobs) + new_log_prob_priors

                    # ratio
                    ratio = proposed_prob - previous_prob
                    if ratio > 0 or ratio > np.log(np.random.uniform()):
                        # accept proposal
                        new_covars = proposed_covars
                        previous_prob = proposed_prob
                        log_prob_priors = new_log_prob_priors
                        log_priors[mixture] = new_log_prob_mixture
                        self.count_accepted[i] += 1
                else:
                    self.count_illegal[i] += 1

        return GMM(np.array(gmm.means), np.array(new_covars), np.array(gmm.weights))

class GaussianStepWeightsProposal(Proposal):

    def __init__(self,  n_mixtures, step_sizes=(0.001,)):
        """
        Gaussian proposal function for the weights of a GMM.
        Parameters
        ----------
        n_mixtures
        step_sizes

        Notes
        ----------
        The proposal function works by projecting the weight vector w onto the simplex defined by
        w_1 + w_2 + ..... w_n = 1 , 0<=w_i<=1. The change of basis matrix is found by finding n-1 vectors lying on the plane
        and using gramm schmidt to get an orthonormal basis. A Gaussian proposal function in (n-1)-d space is
        used to find the next point on the simplex.
        """
        super(GaussianStepWeightsProposal, self).__init__()
        self.step_sizes = step_sizes
        self.n_mixtures = n_mixtures
        self.count_accepted = np.zeros((len(step_sizes),))
        self.count_illegal =  np.zeros((len(step_sizes),))
        self.count_proposed = np.zeros((len(step_sizes),))


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
        """
        Project weight vector onto the normal simplex.
        Parameters
        ----------
        weights : array_like of shape (n_mixtures,)
            vector of weights for each gaussian component

        Returns
        -------
            : array_like of shape (n_mixtures-1,)
            vector of weights projected onto the simplex plane
        """
        # project onto the simplex
        return np.dot(self.e.T, weights - self.plane_origin)

    def invTransformSimplex(self, simplex_coords):
        """
        Transforms a point on the simplex to the original vector space.
        Parameters
        ----------
        simplex_coords : array_like of shape (n_mixtures - 1,)
            Coordinates of a weight vector on the simplex.

        Returns
        -------
            : array_like of shape(n_mixtures,)
            vector of weights.
        """
        return self.plane_origin + np.dot(self.e, simplex_coords)

    def propose(self, X, gmm, target, n_jobs=1):
        """
        Propose a new set of weight vectors.
        Parameters
        ----------
        X : 2-D array like of shape (n_samples, n_features)
            The observed data or evidence.
        gmm : GMM object
            The current state (set of gmm parameters) in the Markov Chain
        target : GMMPosteriorTarget object
            The target distribution to be found.
        n_jobs : int
            Number of cpu cores to use in the calculation of log probabilities.
        Returns
        -------
            : GMM
            A new GMM object initialised with new covariance parameters.
        """
        accepted = False
        cur_gmm = gmm
        if gmm.n_mixtures > 1:
            for i, step_size in enumerate(self.step_sizes):
                self.count_proposed[i] += 1
                current_weights_transformed = self.transformSimplex(cur_gmm.weights)
                proposed_weights_transformed = np.random.multivariate_normal(current_weights_transformed,
                                                                             np.eye(self.n_mixtures - 1) * step_size)
                proposed_weights = self.invTransformSimplex(proposed_weights_transformed   )
                if np.logical_and(0 <= proposed_weights, proposed_weights <= 1).all()\
                    and np.isclose(np.sum(proposed_weights), 1.0):
                    previous_prob = target.log_prob(X, cur_gmm, n_jobs)
                    proposed_gmm = GMM(np.array(cur_gmm.means), np.array(cur_gmm.covars), proposed_weights)
                    proposed_prob = target.log_prob(X, proposed_gmm, n_jobs)
                    ratio = proposed_prob - previous_prob
                    if ratio > 0 or ratio > np.log(np.random.uniform()):
                        # accept proposal
                        self.count_accepted[i] += 1
                        accepted = True
                        cur_gmm = proposed_gmm
                else:
                    self.count_illegal[i] += 1

        if accepted is True:
            return cur_gmm
        else:
            return GMM(np.array(gmm.means), np.array(gmm.covars), np.array(gmm.weights))

