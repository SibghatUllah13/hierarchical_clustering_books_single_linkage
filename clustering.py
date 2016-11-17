# -*- coding: utf-8 -*-
"""
@author: Marco Bressan
"""

import sys
import libcluster as lc
import numpy as np

# 1. compute the bag of words
files = sys.argv[1:]
threshold = 10
bags = [lc.bag(lc.wordcount(f), threshold) for f in files]
print(sorted([len(b) for b in bags], reverse=True))

# 2. compute the Jaccard distance matrix
J = np.zeros((len(bags), len(bags)))
for i in range(len(bags)):
    for j in range(i+1,len(bags)):
        J[i, j] = 1 - lc.jaccard(bags[i], bags[j])
J = J + J.T # (this is to fill under the diagonal)
np.set_printoptions(precision=2)
print(J)
print(J.mean())

# 3. perform the actual clustering
clus = lc.single_linkage(J, k=4)
print([list(np.where(clus == c)[0]) for c in np.unique(clus)])
