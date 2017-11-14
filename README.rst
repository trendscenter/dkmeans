dKMeans
===============
.. contents::

Introduction
---------------

This directory contains files for decentralized k-means and other clustering.

Algorithms
---------------

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

Running The Implementation
---------------
See the README in the pycode folder for more information about running the scripts.

Results/Visualizations
---------------
Look at the ipython notebook for the latest figures measuring silhouette scores of the different implementations
Other raw results have been stored in the results folder, as numpy files containing the following variables ::

  w - the K computed centroids
  C - the cluster labels for every instance in the data set
  X - the training data set, which has been shuffled according to the random distrubtion over nodes
  delta - the record of the changes in the centroids over each iteration
  iter - the total number of iterations
  name - the name of the model

TODO
-----------------
  
1. Finish the decentralized Gaussian Mixture Model with Expectation Maximization
2. Fix the initialization of the different algorithms so it aligns with Kmeans++::
    
    Arthur, David, and Sergei Vassilvitskii.
    "k-means++: The advantages of careful seeding."
    Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms.
    Society for Industrial and Applied Mathematics, 2007.
3. Add additional Metrics for evaluation:

  Rand, William M.
  "Objective criteria for the evaluation of clustering methods."
  Journal of the American Statistical association 66.336 (1971): 846-850.

4. Add more experiments to test the behavior of the different algorithms
