import gzip
import numpy as np
import scipy.sparse as sparse

def load_vocabulary(fname, verbose=False):
    """
    load an indexed list of vocabulary words present across a number of documents in dset
    """
    with open(fname, 'r') as f:
        vocab = [line.strip('\n') for line in f]
    if verbose: print('loaded {:d} words from \"{!s}\"'.format(len(vocab), fname))
    return vocab


def load_docwords(fname, verbose=False):
    """
    load D, W, N, and counts from dset. File is organized according to:
        D
        W
        NNZ
        DOCID WORDID COUNT
        : for all docs, for all words with COUNT>0
        DOCID WORDID COUNT
        \EOF
    Note: that D, W are 1-indexed in-file and will be converted to 0-indexed in-memory

    Args:
        fname (str): path to docwords.txt.gz file

    Returns:
        D (int): number of documents
        W (int): number of words in vocabulary
        NNZ (int): number of nonzero counts in bag-of-words (number Non-zero elements in sparse matrix)
        spmat: DxW sparse numpy array where value at (d, w) is count of word w in document d
    """
    with gzip.open(fname, 'r') as f:
        def get_int(bytes): return int(bytes.decode('utf-8').strip('\n'))
        D = get_int(f.readline())
        W = get_int(f.readline())
        NNZ = get_int(f.readline())
        r = []
        c = []
        v = []
        for line in f.readlines():
            rcv = line.decode('utf-8').strip('\n').split(' ')
            # convert from 1-indexed to 0-indexed
            r.append(int(rcv[0])-1)
            c.append(int(rcv[1])-1)
            v.append(int(rcv[2]))
        spmat = sparse.coo_matrix((v, (r, c)), shape=(D, W), dtype=np.int8)
    if verbose: print('loaded {:d} documents; {:d} word vocabulary; {:d} unique words'.format(D, W, NNZ))
    return (D, W, NNZ, spmat)
