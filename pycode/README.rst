Code for DKmeans
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
