from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
import pystan
import time

NUM_MIXTURE_COMPONENTS = 4
NUM_DIMENSIONS = 4


def create_data(num_samples):
    """
    Creates data to train model parameters
    """
    weights = np.random.random(NUM_MIXTURE_COMPONENTS)
    weights = (weights / weights.sum()).tolist()
    feature_vectors, labels = make_classification(
        n_samples=num_samples, n_features=NUM_DIMENSIONS, n_informative=NUM_DIMENSIONS, n_redundant=0,
        n_classes=NUM_MIXTURE_COMPONENTS, n_clusters_per_class=1, weights=weights)
    #plt.scatter(feature_vectors[:, 0], feature_vectors[:, 1], marker='o')
    #plt.show()

    return feature_vectors


def main():
    print "Make data"
    feature_vectors = create_data(1000)

    # create stan model
    print "Create Stan model"
    compiled_model = pystan.StanModel(file='multivariate_gmm_diagonal.stan')

    print "Train Stan"
    # training
    training_data = dict(N=len(feature_vectors), D=NUM_DIMENSIONS, M=NUM_MIXTURE_COMPONENTS, X=feature_vectors)

    starttime = time.time()

    samples = compiled_model.sampling(training_data)

    endtime = time.time()

    print endtime - starttime
    results = samples.extract(permuted=True)
    print(results)


if __name__ == '__main__':
    main()