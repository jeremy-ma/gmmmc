.. gmmmc documentation master file, created by
   sphinx-quickstart on Tue Aug  9 17:48:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to gmmmc's documentation!
=================================

.. toctree::
   :maxdepth: 2

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

Examples
========
.. code-block:: python
    n_mixtures, n_features = 2, 16
    # single mixture gmm
    truth_gmm = GMM(means=np.random.uniform(low=-1, high=1, size=(n_mixtures, n_features)),
                    covariances=np.random.uniform(low=0, high=1, size=(n_mixtures, n_features)),
                    weights=np.random.dirichlet(np.ones((n_mixtures))))

    # draw samples from the true distribution
    X = truth_gmm.sample(n_samples)

    gmm_ml = sklearn.mixture.GMM(n_components=n_mixtures, covariance_type='diag', n_iter=10000)
    gmm_ml.fit(X)
    print "finished ML fit"
    ########### MCMC ##################################
    # setup monte carlo sampler
    scale = 1.0
    proposal = GMMBlockMetropolisProposal(propose_mean=GaussianStepMeansProposal(step_sizes=[0.001, 0.01, 0.1]))


    prior = GMMPrior(MeansUniformPrior(-1, 1, n_mixtures, X.shape[1]),
                     DiagCovarsUniformPrior(0.01, 1, n_mixtures, X.shape[1]),
                     WeightsUniformPrior(n_mixtures))

    proposal = GMMBlockMetropolisProposal(propose_mean=GaussianStepMeansProposal(step_sizes=[0.0001]),
                                          propose_covars=GaussianStepCovarProposal(step_sizes=[0.00001]),
                                          propose_weights=GaussianStepWeightsProposal(n_mixtures,
                                                                                      step_sizes=[0.001]))

    initial_gmm = GMM(means=gmm_ml.means_, covariances=gmm_ml.covars_, weights=gmm_ml.weights_)

    mc = MarkovChain(proposal, prior, initial_gmm)
    # make gmm samples
    gmm_samples = mc.sample(X, n_samples=n_runs, n_jobs=n_jobs)
    # discard gmm samples
    gmm_samples[int(n_runs / 2)::50]

    test_samples = truth_gmm.sample(10)

    markov_chain_likelihood = [logsumexp([gmm.log_likelihood(np.array([sample]), n_jobs=-1) for gmm in gmm_samples]) - np.log(len(gmm_samples))\
                               for sample in test_samples]
    true_likelihood = [truth_gmm.log_likelihood(np.array([sample])) for sample in test_samples]


    logging.info('Means Acceptance: {0}'.format(proposal.propose_mean.get_acceptance()))
    logging.info('Covars Acceptance: {0}'.format(proposal.propose_covars.get_acceptance()))
    logging.info('Weights Acceptance: {0}'.format(proposal.propose_weights.get_acceptance()))
    logging.info('MCMC Likelihood: {0}'.format(str(markov_chain_likelihood)))
    logging.info('ML Estimate Likelihood: {0}'.format(str(likelihood_ml)))
    logging.info('True Likelihood: {0}'.format(str(true_likelihood)))


API Documentation
=================

Monte Carlo Algorithms
----------------------

.. automodule:: gmmmc.monte_carlo
    :members:
    :undoc-members:
    :show-inheritance:

Proposal Functions
------------------

.. automodule:: gmmmc.proposals.proposals
    :members:
    :undoc-members:
    :show-inheritance:

Gaussian Proposals
------------------

.. automodule:: gmmmc.proposals.gaussian_proposals
    :members:
    :undoc-members:
    :show-inheritance:

Priors
------
.. automodule:: gmmmc.priors.prior
    :members:
    :undoc-members:
    :show-inheritance:

GMM Representation
------------------

.. automodule:: gmmmc.gmm
    :members:
    :undoc-members:
    :show-inheritance:

Target Distributions
--------------------

.. automodule:: gmmmc.posterior
    :members:
    :undoc-members:
    :show-inheritance:
