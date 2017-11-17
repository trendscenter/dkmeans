Distributed Kmeans
===============
.. contents::


Introduction
---------------
Currently this directory contains proof of concept scripts for various approaches to distributed k-means and distributed clustering in general. There are six scripts which instantiate particular algorithms for distributed k-means, two utility scripts, and one script for running experiments.

Algorithms
---------------

The six scripts currently implemented for distributed k-means take different approaches to the problems of decentralization and optimization. There are two 'Single-Shot' scripts, which implement local optimization techniques and global merging strategies, and three 'Multi-Shot' scripts which implement global optimizations which depend on communication betweeen nodes during the optimization itself. 

Three different optimization algorithms are implemented

**LLoyd's Algorithm** ::

  Lloyd, Stuart. 
  "Least squares quantization in PCM." 
  IEEE transactions on information theory 28.2 (1982): 129-137.
      
**Gradient Descent for K-Means** ::


  Bottou, Leon, and Yoshua Bengio. 
  "Convergence properties of the k-means algorithms." 
  Advances in neural information processing systems. 1995.
       
**Expectation Maximiation for Gaussian Mixtures** :: 
  
  Rasmussen, Carl Edward, and Christopher KI Williams. 
  Gaussian processes for machine learning. 
  Vol. 1. Cambridge: MIT press, 2006.

Some of the strategies for **single-shot decentralization** were inspired by ::

  Jagannathan, Geetha, Krishnan Pillaipakkamnatt, and Rebecca N. Wright. 
  "A new privacy-preserving distributed k-clustering algorithm." 
  Proceedings of the 2006 SIAM International Conference on Data Mining. Society for Industrial and Applied Mathematics, 2006.
  
Namely, cluster merging as a decentralization strategy comes from Jagannathan, et. al 2006.

The strategies for **multi-shot decentralization** were partially inspired by ::

  Jagannathan, Geetha, and Rebecca N. Wright. 
  "Privacy-preserving distributed k-means clustering over arbitrarily partitioned data."
  Proceedings of the eleventh ACM SIGKDD international conference on Knowledge discovery in data mining. ACM, 2005.

Currently only Lloyd's Algorithm and Gradient Descent converge correctly. The Gaussian Mixture still requires work w.r.t computing the global Multivariate Standard Deviations in particular. 

Single-Shot Decentralized LLoyd
_________________

*Algorithm Flow*  ::

    1: On each site, initialize Random Centroids
    2: On each site, compute a clustering C with k-many clusters
    3: On each site, compute a local mean for each cluster in C
    4: On each site, recompute centroids as equal to local means
    5: On each site,
        if change in centroids below some epsilon, STOP, report STOPPED
        else GOTO step 3
    6: On each site, broadcast local centroids to aggregator
    7: On the aggregator, compute merging of clusters according to
        least merging error (e.g. smallest distance betweeen centroids)
    8: Broadcast merged centroids to all sites

Multi-Shot Decentralized LLoyd
_________________

*Algorithm Flow* ::

    1: On the aggregator, initialize random Centroids
        (either entirely remotely computed or shared between local sites)
    2: Broadcast Centroids to all Sites
    3: On each site, compute a clustering C with k-many clusters
    4: On each site, compute a local mean for each cluster in C
    5: On each site, broadcast local mean to the aggregator
    6: On the aggregator, compute the global means for each Cluster
    7: On the aggregator, recompute centroids as equal to global means
    8: On the aggregator,
        if change in centroids below some epsilon, broadcast STOP
        else broadcast new centroids, GOTO step 3

Single-Shot Decentralized Gradient Descent
_________________

*Algorithm Flow* ::

    1: On each site, initialize Random Centroids
    2: On each site, compute a clustering C with k-many clusters
    3: On each site, compute a local gradient for each cluster in C
    4: On each site, update centroids via gradient descent
    5: On each site,
        if change in centroids below some epsilon, STOP, report STOPPED
        else GOTO step 3
    6: On each site, broadcast local centroids to aggregator
    7: On the aggregator, compute merging of clusters according to
        least merging error (e.g. smallest distance betweeen centroids)
    8: Broadcast merged centroids to all sites


Multi-Shot Decentralized Gradient Descent
_________________

*Algorithm Flow* ::

    1: On the aggregator, initialize random Centroids 
        (either entirely remotely computed or shared between local sites)
    2: Broadcast Centroids to all Sites
    3: On each site, compute a clustering C with k-many clusters
    4: On each site, compute a local gradient for each cluster in C
    5: On each site, broadcast local gradient to the aggregator
    6: On the aggregator, compute the global gradients for each Cluster
    7: On the aggregator, update centroids according to gradient descent
    8: On the aggregator,
        if change in centroids below some epsilon, broadcast STOP
        else broadcast new centroids, GOTO step 3

Multi-Shot Gaussian Mixture Model with Expectation Maximization
___________________

*Algorithm Flow* ::

    1: On the aggregator, initialize random normal distributions, Theta
    2: Broadcast Theta to all sites
    3: all sites, compute weights for each cluster according to local data
    4: all sites, compute partial Nk 
    5: all sites, broadcast partial Nk and weights to aggregator
    6: Aggregator, compute mu for each cluster k, broadcast to sites
    7: All sites, compute partial sigma_k pass to aggregator
    8: Aggregator, compute sigma_k, broadcast to all sites
    9: All sites, locally compute partial log-likelihood
    10: Aggregator check change in log-likelihood
            if below epsilon, broadcast STOP
            else GOTO 3


Running the Implementation
---------------

The dkmeans filenames are formatted as follows ::
  dkmeans_\<DECENTRALIZATION\>_\<OPTIMIZATION\>.py

And can be run either individually, by importing the script, and running the main function

  >>> import dkmeans_ss_lloyd as ss_lloyd
  >>> import nump as np
  >>> X = np.random(100, 2)
  >>> ss_lloyd.main(X, 2, ep=0.001)

or can be run in the experiments script, dkmeans_experiments.py

  >>> import dkmeans_experiments as exp
  >>> exp.main()

Experiments
_______________

The dkmeans_experiments.py file currently runs the following experiments::
  
  1. Test all methods on gaussian data with known number of clusters.
  2. Test all methods on gaussian data, iris data set, simulated fMRI, and real fMRI,
      increasing the number of clusters, keeping the number of samples constant
  3. Test all methods with best guess number of clusters, increasing the number
      of samples in the data **TODO**
  4.  Test fMRI data with increasing number of subjects **TODO**
  5.  Test variations in the subject/sites distrubtions **TODO**
  6.  Test drop-out behavior, when one or multiple nodes drop out during an iteration **TODO**
