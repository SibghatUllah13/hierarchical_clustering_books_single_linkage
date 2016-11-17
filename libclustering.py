# -*- coding: utf-8 -*-
"""
@author: Marco Bressan
"""
import numpy as np


def wordcount(filename):
    """Compute the word count of a given text file.
    This function returns a dictionary mapping each word appearing in the input files
    to the number of times it occurs.
    """
    wc = {}
    punct = ',.<>/?;:"[]{}«»|=+-_()*&^%$#@!`~\t\n\r\\\''
    with open(filename, 'r') as f:
        for line in f:
            for c in punct: # convert punctuation into whitespaces (i.e. delimiters)
                line = line.replace(c,' ') 
            list = line.lower().split()
            for word in list: # increase bag['word'] by 1 (care of inexistent keys)
                wc[word] = wc[word] + 1 if word in wc else 1
    return wc


def bag(wc, threshold = 1):
    """Compute a bag of words from a word count.
    This function returns the set of keys in the dictionary wc whose associated value is at least threshold"""
    return set([key for key in wc if wc[key] >= threshold])


def jaccard(A, B):
    """Compute the Jaccard similarity between two sets."""
    sA, sB = set(A), set(B)
    return len(sA.intersection(sB))/len(sA.union(sB))


def single_linkage(D, k=2):
    """Compute a single-linkage clustering from a distance matrix."""
    # 1. set-up
    M = D.copy()  # avoid altering the original matrix
    n = M.shape[0]
    M[np.arange(n), np.arange(n)] = np.inf # make sure to never merge single elements with themselves
    cluster = np.arange(n) # cluster[j] is element j's cluster ID (we start with a cluster for each element)
    # 2. iterative cluster merging
    for i in np.arange(n - k):
        # 2.1. find the next two clusters to be merged
        pair = np.unravel_index(M.argmin(), M.shape) # find the two closest elements
        c1, c2 = cluster[pair[0]], cluster[pair[1]]  # get their cluster IDs
        # 2.2. merge the two clusters
        M[np.ix_(cluster == c1, cluster == c2)] = np.inf # set infinite intra-distance
        M[np.ix_(cluster == c2, cluster == c1)] = np.inf # this is a symmetric matrix...
        cluster[cluster == max(c1, c2)] = min(c1, c2)   # move all elements to the cluster with lower ID
    return cluster
