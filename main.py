#!/usr/bin/env python3
import sys
import os.path
import argparse
import numpy as np
import numpy.random
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import fileio
import timeit


if __name__ == '__main__':
    # arg defaults
    default_maxiter = 10
    default_dset = 'kos'
    default_ftype = 'float32'
    default_verbose = 2

    parser = argparse.ArgumentParser(description='Gibbs sampler for predicting document class labeling in bayesian bag-of-words model',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose', '-v', action='count', default=default_verbose, help='increase verbosity by 1 for each flag')
    parser.add_argument('--maxiter', type=int, default=default_maxiter)
    parser.add_argument('--dset', type=str, default=default_dset, choices=['enron', 'kos', 'nips', 'nytimes', 'pubmed'], help='set dataset name')
    parser.add_argument('--ftype', type=str, choices=['float32', 'float64'], default=default_ftype, help='set floating point bit-depth')
    # parse args
    args = parser.parse_args()
    dset_name = args.dset
    ftype = args.ftype
    verbose = args.verbose
    maxiter = args.maxiter


    data_root = '/media/hdd1/projects/examples/bag-of-words'
    np.random.seed(20)
    def float(x):
        return np.dtype(ftype).type(x)

    # load data
    vocab_file = os.path.join(data_root, 'vocab.{!s}.txt'.format(dset_name))
    docwords_file = os.path.join(data_root, 'docword.{!s}.txt.gz'.format(dset_name))
    vocab = fileio.load_vocabulary(vocab_file, verbose)
    D, W, NNZ, spcounts = fileio.load_docwords(docwords_file, verbose)

    # hyperparameter settings
    hp_pi  = [1, 1]         # HPs for beta prior over labels
    hp_th = np.ones((W,))  # HPs for dirichlet prior over vocabulary

    # initialize latent parameters
    #   each LP is stored in a two element list
    #   element [0] is running total (for computing expectations)
    #   element [1] is used in sampling from conditionals at each iter.
    lp_pi  = [ float(0), float(np.random.beta(*hp_pi)) ]                                  # label prior
    lp_th  = [[ np.zeros((W,), dtype=ftype), np.random.dirichlet(hp_th).astype(ftype) ],  # vocabulary prior - class 1
              [ np.zeros((W,), dtype=ftype), np.random.dirichlet(hp_th).astype(ftype) ]]  # vocabulary prior - class 2
    lp_l   = [ np.zeros((D,)), np.random.binomial(1, lp_pi[1], size=(D,)) ]               # latent labels

    # Bookkeeping
    _, doccounts = np.unique(lp_l[1], return_counts=True)
    num_class = len(doccounts)
    # divide counts into separate class-exclusive counts
    spcounts = spcounts.toarray() # make dense array
    wordcounts = []
    for i in range(num_class):
        mask = np.where(lp_l[1]==i, 1, 0)
        wordcounts.append(np.sum(spcounts[mask], axis=0))

    # convenience
    def cond_label(this_wordcounts):
        """ computes conditional for L_j by sampling from bernoulli with parameter set by current class word counts """
        log_res = [None, None]
        for i in range(num_class):
            # use log-probability for numerical stability
            log_res[i] = np.log(doccounts[i]+hp_pi[i]-1) - np.log(D+np.sum(hp_pi)-1) + np.sum(np.multiply(this_wordcounts, np.log(lp_th[i][1])))
        bern_param = log_res[i] / (np.sum(log_res))
        if verbose >=3: print('bern_param', bern_param)
        return np.random.binomial(1, bern_param)

    # Sampling
    ss_iter = 0
    num_flip = []
    while ss_iter<=maxiter:
        ss_iter+=1

        # iterate documents
        doc_iter = 0
        iter_num_flip = 0
        for d in range(D):
            doc_iter+=1
            # check for label (training doc) - NOTYETIMPLEMENTED

            # remove this document from conditional
            label_was = lp_l[1][d]
            this_wordcounts = spcounts[label_was, :].reshape((W,))
            #   remove word counts from class "label_was"
            wordcounts[label_was] -= this_wordcounts
            #   remove doc count
            doccounts[label_was] -= 1

            # sample new most-likely label
            label_is = cond_label(this_wordcounts)
            lp_l[0][i] += label_is  # add to running expectation tally
            lp_l[1][i] = label_is   # assign to current state space vector

            if verbose >= 3: print('label_was: {:d}, label_is: {:d}'.format(label_was, label_is))

            # add document to conditional
            #   add word counts to class "label_is"
            wordcounts[label_is] += this_wordcounts
            #   add doc count
            doccounts[label_is] += 1

            # track performance
            iter_num_flip += int(label_is!=label_was)

        # sample thetas
        #   augment class wordcounts with hyperparam pseudo-counts
        for i in range(num_class):
            t = wordcounts[i] + hp_th
            lp_th[i][1] = np.random.dirichlet(t)
            lp_th[i][0] += lp_th[i][1]

        # track performance
        num_flip.append(iter_num_flip)
        if verbose >= 2: print('iter: {:d} || num_flip: {:d}'.format(ss_iter, iter_num_flip))


    # show results
    plt.plot(num_flip)
    plt.show()
