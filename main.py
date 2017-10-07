#!/usr/bin/env python3
import sys
import os.path
import numpy as np
import numpy.random
import scipy.sparse as sparse
import fileio


if __name__ == '__main__':
    dset_name = 'nips'
    verbose = True
    #  np.random.seed(1)

    # load data
    DATA_ROOT = '/media/hdd1/projects/examples/bag-of-words'
    vocab_file = os.path.join(DATA_ROOT, 'vocab.{!s}.txt'.format(dset_name))
    docwords_file = os.path.join(DATA_ROOT, 'docword.{!s}.txt.gz'.format(dset_name))
    vocab = fileio.load_vocabulary(vocab_file, verbose)
    D, W, NNZ, spcounts = fileio.load_docwords(docwords_file, verbose)

    # hyperparameter settings
    hp_p1 = hp_p2 = 1  # HPs for beta prior over labels
    hp_th = np.ones((W,))  # HPs for dirichlet prior over vocabulary

    # initialize latent parameters
    #   each LP is stored in a two element list
    #   element [0] is running total (for computing expectations)
    #   element [1] is used in sampling from conditionals at each iter.
    lp_pi  = [ np.int(0), np.random.beta(hp_p1, hp_p2) ]  # label prior
    lp_th1 = [ np.zeros((W,)), np.random.dirichlet(hp_th) ]  # vocabulary prior - class 1
    lp_th2 = [ np.zeros((W,)), np.random.dirichlet(hp_th) ]  # vocabulary prior - class 2
    lp_l   = [ np.zeros((D,)), np.random.binomial(1, lp_pi[1], size=(D,)) ]  # latent labels


    # Bookkeeping
    _, doccounts = np.unique(lp_l[1], return_counts=True)



    # Sampling
    max_iter = 10


