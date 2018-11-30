#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 11:39:40 2018

@author: bbaker
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from dkmeans.data import get_dataset
RUNNAME = 'N314_s2'
res = '/export/mialab/users/bbaker/projects/dkmeans/results/fbirn_%s_k5_multishot_lloyd_res.npy' % RUNNAME
#res = '/export/mialab/users/bbaker/projects/dkmeans/d_results/fbirn_k5_res.npy'
meths = ['multishot_lloyd']
print("Loading results")
NUM=314
a = np.load(res)
a = a.item()
#f, axes = plt.subplots(len(meths), 5, figsize=(30, 10))
#print("Grabbing data")
#X, subject_indices = get_dataset(314, dataset='real_fmri', 
#                      dfnc_window=22)
for r, meth in enumerate(meths):
    subject_indices = a[meth][0]['subjects']
    cluster_labels = a[meth][0]['cluster_labels']
    print(np.unique(cluster_labels, return_counts=True))
    data = a[meth][0]['X']
    #print(len(subject_indices), len(cluster_labels))
    C = a[meth][0]['centroids']
    #axes[r].set_ylabel(meth)
    #[ax.imshow(C[k], interpolation='none', vmin=-0.5, vmax=0.5, cmap='jet')
    #    for k, ax in enumerate(axes[r])]
    subject_statevect = [[] for i in range(NUM)]
    subject_data = [[] for i in range(NUM)]
    for i, c in enumerate(cluster_labels):
        si = subject_indices[i]
        subject_statevect[si].append(c)
        subject_data[si].append(data[i])
    subject_statevect = np.array(subject_statevect)
    sio.savemat('results/fbirn_%s_k5_%s.mat' % (RUNNAME, meth),
                {'centroids': C,
		 'subject_data':subject_data,
                 'subject_statevect': subject_statevect,
                 'cluster_labels': cluster_labels,
                })
#plt.savefig('fbirn_pooled_vs_decentarlized_lloyd.pdf')
#plt.show()
