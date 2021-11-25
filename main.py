import itertools

import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.spatial import distance


# Hyperparameters (fixed)
JS_SIMILARITY_THRESHOLD = .5
CS_SIMILARITY_THRESHOLD = .73
DCS_SIMILARITY_THRESHOLD = .73

# Hyperparemeters (tunable)
SIGNATURE_LENGTH = 128


# For now... eventually we will allow
# flags that specify the file path as
# required.
umr = np.load('user_movie_rating.npy')


def jaccard_similarity(u1, u2):
    return distance.jaccard(u1, u2)


def test_jaccard_similarity(u1, u2):
    assert jaccard_similarity(u1, u2) == len(np.intersect1d(u1, u2))/len(np.union1d(u1, u2))


def cosine_similarity(u1, u2):
    return distance.cosine(u1, u2)


# Same as cosine similarity, just with truncated u1 and u2
def discrete_cosine_similarity(u1, u2):
    return distance.cosine(u1, u2)


def jaccard_main():
    """Find user pairs u1 and u2 such that JS(u1, u2) > JS_SIMILARITY_THRESHOLD"""
    # note: the .sign() performs truncation for us.
    input_matrix = csr_matrix((umr[:, 2], (umr[:, 0], umr[:, 1]))).sign()
    signature_matrix = np.empty([SIGNATURE_LENGTH, input_matrix.shape[1]])

    # Blocking
    ...

    # Minhash block 1...B ?
    ...


def cosine_main():
    pass


def discrete_cosine_main():
    pass


def main():
    pass


if __name__ == '__main__':
    main()
