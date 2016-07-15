#from gmmFast import GMM
import numpy as np
import sklearn
import time
from fast_likelihood import testing

testing()
"""
n_mixtures, n_features = 128, 64
#means = np.array(np.random.uniform(-1, 1, size=(n_mixtures, n_features)), dtype=np.float64)
#weights = np.array(np.random.dirichlet([1 for _ in xrange(n_mixtures)]), dtype=np.float64)
#covars = np.array(np.random.uniform(0, 1, size=(n_mixtures, n_features)), dtype=np.float64)
means = np.array([[0.1, 0.1, 0.1, 0.1],[0.2, 0.2, 0.2, 0.2]], dtype=np.float64)
covars = np.array([[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2]])
weights = np.array([0.5, 0.5])

gmm = GMM(means, weights, covars)
true_gmm = sklearn.mixture.GMM(n_components=n_mixtures)
true_gmm.means_ = means
true_gmm.covars_ = covars
true_gmm.weights_ = weights

store_C = []
store_sklearn = []
for n_samples in [150]:
    #data = np.array(np.random.uniform(-1, 1, (n_samples, n_features)), dtype=np.float64)
    data = np.array([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]])
    start = time.time()
    mine = gmm.log_likelihood(data)
    finish_C = time.time() - start
    store_C.append(str(finish_C))
    start = time.time()
    sklearns = np.sum(true_gmm.score(data))
    finish_sklearn = time.time() - start
    store_sklearn.append(str(finish_sklearn))
print 'C'
print '\n'.join(store_C)
print 'sklearn'
print '\n'.join(store_sklearn)
"""