from unittest import TestCase
from distributions.prior import *
import numpy as np
from distributions.gmm import GMM


class TestGMMPrior(TestCase):
    def setUp(self):
        self.covars = np.array([[3, 5], [2, 1]])
        means = np.array([[1.0, 2], [3, 4]])
        self.weights = np.array([0.3, 0.7])
        prior_width = 1.0 * np.ones((2, 2))
        self.prior = GMMPrior(MeansGaussianPrior(means, prior_width),
                         CovarsStaticPrior(self.covars), WeightsStaticPrior(self.weights))

    def test_log_prob_gaussian_means(self):
        test_gmm = GMM(np.array([[4, 5], [5, 3]]), self.weights, self.covars)
        test = self.prior.log_prob(test_gmm)
        real = -15.175754132818689
        self.assertAlmostEqual(test, real)

    def test_sample(self):
        pass


class TestMeansGaussianPrior(TestCase):
    def setUp(self):
        means = np.array([[1.0, 2], [3, 4]])
        prior_width = 1.0 * np.ones((2, 2))
        self.prior = MeansGaussianPrior(means, prior_width)
    def test_log_prob(self):
        test = self.prior.log_prob(np.array([[4, 5], [5, 3]]))
        real = -15.175754132818689
        self.assertAlmostEqual(test, real)

    def test_cache_log_prob(self):
        test = self.prior.log_prob(np.array([[4, 5], [5, 3]]))
        real = -15.175754132818689
        self.assertAlmostEqual(test, real)
        test = self.prior.log_prob(np.array([[4, 5], [5, 3]]))
        real = -15.175754132818689
        self.assertAlmostEqual(test, real)
        test = self.prior.log_prob(np.array([[4, 5], [5, 3]]))
        real = -15.175754132818689
        self.assertAlmostEqual(test, real)

    def test_sample(self):
        pass