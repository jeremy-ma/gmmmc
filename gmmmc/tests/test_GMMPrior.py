from unittest import TestCase

from gmmmc.priors.prior import *


class TestGMMPrior(TestCase):
    def setUp(self):
        self.covars = np.array([[3, 5], [2, 1]])
        means = np.array([[1.0, 2], [3, 4]])
        self.weights = np.array([0.3, 0.7])
        prior_width = 1.0 * np.ones((2, 2))
        self.prior = GMMPrior(MeansGaussianPrior(means, prior_width),
                              CovarsStaticPrior(self.covars), WeightsStaticPrior(self.weights))

    def test_log_prob_gaussian_means(self):
        test_gmm = GMM(np.array([[4, 5], [5, 3]]), self.covars, self.weights)
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


class TestMeansUniformPrior(TestCase):
    def setUp(self):
        self.prior = MeansUniformPrior(-1.0, 1.0, n_mixtures=5, n_features=10)

    def test_log_prob(self):
        means = np.ones((5, 10))
        self.assertEqual(self.prior.log_prob(means), 0.0)
        self.assertEqual(self.prior.log_prob(means * 2), -np.inf)
        means[3][2] = -10
        self.assertEqual(self.prior.log_prob(means), -np.inf)

    def test_log_prob_single(self):
        mean = np.ones((10))
        self.assertEqual(self.prior.log_prob_single(mean, 3), 0.0)
        self.assertEqual(self.prior.log_prob_single(mean * 2, 4), -np.inf)
        mean[3] = -10
        self.assertEqual(self.prior.log_prob_single(mean, 3), -np.inf)

    def test_sample(self):
        for i in xrange(100):
            sample = self.prior.sample()
            self.assertTrue((sample > -1.0).all() and (sample < 1.0).all())


class TestDiagCovarsUniformPrior(TestCase):
    def setUp(self):
        self.prior = DiagCovarsUniformPrior(0.001, 1.0, 5, 10)

    def test_log_prob(self):
        covars = np.ones((5, 10)) * 0.5
        self.assertEqual(self.prior.log_prob(covars), 0.0)
        self.assertEqual(self.prior.log_prob(covars * 4), -np.inf)
        covars[3][2] = -10
        self.assertEqual(self.prior.log_prob(covars), -np.inf)

    def test_log_prob_single(self):
        covar = np.ones((10))
        self.assertEqual(self.prior.log_prob_single(covar, 3), 0.0)
        self.assertEqual(self.prior.log_prob_single(covar * 2, 4), -np.inf)
        covar[3] = -10
        self.assertEqual(self.prior.log_prob_single(covar, 3), -np.inf)

    def test_sample(self):
        for i in xrange(100):
            sample = self.prior.sample()
            self.assertTrue((sample >= 0.001).all() and (sample <= 1.0).all())


class TestWeightsUniformPrior(TestCase):
    def setUp(self):
        self.prior = WeightsUniformPrior(10)

    def test_log_prob(self):
        weights = np.array([0.1 for i in xrange(10)])
        self.assertEqual(self.prior.log_prob(weights), 0.0)
        weights[3] = 0.5
        self.assertEqual(self.prior.log_prob(weights), -np.inf)

    def test_sample(self):
        for _ in xrange(100):
            sample = self.prior.sample()
            self.assertTrue((sample > 0.0).all())
            self.assertAlmostEqual(np.sum(sample), 1.0)

