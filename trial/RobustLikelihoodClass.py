__author__ = 'jeremyma'
import numpy as np
from sklearn.mixture import GMM as GMMEval

class LikelihoodEvaluator():
    def __init__(self, Xpoints, numMixtures):
        assert(Xpoints.ndim==2)

        self.Xpoints = Xpoints
        self.numPoints, self.dim = Xpoints.shape
        self.numMixtures = numMixtures

    def loglikelihood(self, means, diagCovs, weights):
        raise NotImplementedError

    __call__ = loglikelihood

class SingleCoreLL(LikelihoodEvaluator):

    def __init__(self, Xpoints, numMixtures):
        print "Single Core Implementation Chosen"
        LikelihoodEvaluator.__init__(self, Xpoints, numMixtures)

    def __str__(self):
        return "Single Core Implementation"



    def loglikelihood(self, means, diagCovs, weights):
        numMixtures = self.numMixtures

        #update if need be
        assert(means.shape == (numMixtures, self.dim))
        assert(diagCovs.shape == (numMixtures, self.dim))
        assert(len(weights)== numMixtures)

        numMixtures = len(weights)
        ll = np.zeros(self.numPoints)

        constMulti = self.dim / 2.0 * np.log(2 * np.pi)

        CovDet = np.zeros(numMixtures)

        for i in xrange(numMixtures):
            CovDet[i] = 1.0 / np.sqrt(np.prod(diagCovs[i]))

        for i in xrange(self.numPoints):
            for mixes in xrange(numMixtures):
                multiVal = 1

                temp = np.dot((self.Xpoints[i] - means[mixes]) / diagCovs[mixes], (self. Xpoints[i] - means[mixes]))
                temp *= -0.5
                ll[i] += weights[mixes] * np.exp(temp) * CovDet[mixes]

            ll[i] = np.log(ll[i]) - constMulti

        return np.sum(ll)


class scikitLL(LikelihoodEvaluator):
    """
    Fastest Single Core Version so far!
    """


    def __init__(self, Xpoints, numMixtures):
        #print "Scikits Learn Implementation Chosen"
        LikelihoodEvaluator.__init__(self, Xpoints, numMixtures)
        self.evaluator = GMMEval(n_components=numMixtures)
        self.Xpoints = Xpoints


    def __str__(self):
        return "SciKit's learn implementation Implementation"


    def loglikelihood(self, means, diagCovs, weights):
        self.evaluator.weights_ = weights
        self.evaluator.covars_ = diagCovs
        self.evaluator.means_ = means

        return np.sum(self.evaluator.score(self.Xpoints))

Likelihood = scikitLL
