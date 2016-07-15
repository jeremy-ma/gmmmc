from unittest import TestCase
import numpy as np
from distributions.gmm import GMM
import sklearn.mixture
import time

class TestGMM(TestCase):

    def setUp(self):
        self.n_mixtures, self.n_features = 32, 16
        means = np.array(np.random.uniform(-1,1,size=(self.n_mixtures, self.n_features)), dtype=np.float64)
        weights = np.array(np.random.dirichlet([1 for _ in xrange(self.n_mixtures)]), dtype=np.float64)
        covars = np.array(np.random.uniform(0,1,size=(self.n_mixtures, self.n_features)), dtype=np.float64)
        self.gmm = GMM(means, weights, covars, n_jobs=-1)
        self.true_gmm = sklearn.mixture.GMM(n_components=self.n_mixtures)
        self.true_gmm.means_ = means
        self.true_gmm.covars_ = covars
        self.true_gmm.weights_ = weights

    def test_sample(self):
        self.assertTrue(True)

    def test_log_likelihood(self):
        data = np.array(np.random.uniform(-1, 1, (2, self.n_features)), dtype=np.float64)
        start = time.time()
        mine = self.gmm.log_likelihood(data)
        print time.time() - start
        start = time.time()
        sklearns = np.sum(self.true_gmm.score(data))
        print time.time() - start
        self.assertAlmostEqual(mine, sklearns)

    def test_scaling(self):
        store_C = []
        store_sklearn = []
        suite = [10, 100, 1000, 10000, 100000, 1000000]
        for n_samples in suite:
            data = np.array(np.random.uniform(-1, 1, (n_samples, self.n_features)), dtype=np.float64)
            start = time.time()
            mine = self.gmm.log_likelihood(data)
            finish_C = time.time() - start
            store_C.append(str(finish_C))
            start = time.time()
            sklearns = np.sum(self.true_gmm.score(data))
            finish_sklearn = time.time() - start
            store_sklearn.append(str(finish_sklearn))
            self.assertAlmostEqual(mine, sklearns, places=3)




        print 'C'
        print '\n'.join(store_C)
        print 'sklearn'
        print '\n'.join(store_sklearn)
