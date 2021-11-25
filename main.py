import itertools

import numpy as np
import numpy.random as npr
import scipy
from scipy.sparse import csr_matrix
from scipy.spatial import distance


# Hyperparameters (fixed)
JS_SIMILARITY_THRESHOLD = .5
CS_SIMILARITY_THRESHOLD = .73
DCS_SIMILARITY_THRESHOLD = .73
SEED = 1

# Hyperparemeters (tunable)
SIGNATURE_LENGTH = 128
N_BANDS = 16


# For now... eventually we will allow
# flags that specify the file path as
# required.
umr = np.load('user_movie_rating.npy')
umr[:, :2] -= 1  # Ensure rows and columns start at 0

# We probably don't even need these helper functions:
# we can directly use scipy.distance to reduce the
# number of function invocation (read: overhead).


def jaccard_similarity(u1, u2):
    return distance.jaccard(u1, u2)


def test_jaccard_similarity(u1, u2):
    assert jaccard_similarity(u1, u2) == len(np.intersect1d(u1, u2))/len(np.union1d(u1, u2))


def cosine_similarity(u1, u2):
    return distance.cosine(u1, u2)


# Same as cosine similarity, just with truncated u1 and u2
def discrete_cosine_similarity(u1, u2):
    return distance.cosine(u1, u2)


def jaccard_main(toy=None):
    """Find user pairs u1 and u2 such that JS(u1, u2) > JS_SIMILARITY_THRESHOLD

    toy: reducing the data"""

    input_matrix = csr_matrix((umr[:toy, 2], (umr[:toy, 1], umr[:toy, 0]))).sign()
    signature_matrix = np.zeros([SIGNATURE_LENGTH, input_matrix.shape[1]], dtype=int)

    # Minhashing to fill the signature matrix
    rng = npr.default_rng(SEED)
    m = input_matrix.copy()
    indices = list(range(input_matrix.shape[0]))
    for i in range(SIGNATURE_LENGTH):
        rng.shuffle(indices)
        m = m[indices, :]
        signature_matrix[i, :] = m.argmax(0)

    print(signature_matrix)
    print(signature_matrix.shape)

    ...

    # Banding the signature matrix
    ...


def cosine_main():
    pass


def discrete_cosine_main():
    pass


def main():
    pass


if __name__ == '__main__':
    main()
    jaccard_main(10000)

