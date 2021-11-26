from collections import defaultdict
import itertools
import time

import numpy as np
import numpy.random as npr
from scipy.sparse import csr_matrix
from scipy.spatial import distance


# Hyperparameters
# Fixed
JS_SIMILARITY_THRESHOLD = .5
CS_SIMILARITY_THRESHOLD = .73
DCS_SIMILARITY_THRESHOLD = .73
SEED = 1

# Tuneable
SIGNATURE_LENGTH = 128
N_ROWS_PER_BAND = 8
N_BANDS = SIGNATURE_LENGTH // N_ROWS_PER_BAND


# TODO: allow fp to be specified by flag
umr = np.load('user_movie_rating.npy')
umr[:, :2] -= 1  # Ensure rows and columns start at 0
assert umr.min() == 0
input_matrix = None


# TODO: make toy dataset for hyperpar tuning
# TODO: write grid search routine on toy dataset


def cosine_similarity(u1, u2):
    return distance.cosine(u1, u2)


# Same as cosine similarity, just with truncated u1 and u2
def discrete_cosine_similarity(u1, u2):
    return distance.cosine(u1, u2)


def jaccard_main(toy=None, verbose=True):
    """Find user blocked_pairs u1 and u2 such that JS(u1, u2) > JS_SIMILARITY_THRESHOLD

    toy: reducing the data"""

    # Constructing the signature matrix
    t0 = time.time()
    global input_matrix
    input_matrix = csr_matrix((umr[:toy, 2], (umr[:toy, 1], umr[:toy, 0]))).sign()
    signature_matrix = np.full(([SIGNATURE_LENGTH, input_matrix.shape[1]]), np.inf)

    # TODO: parallelize
    rng = npr.default_rng(SEED)
    m = input_matrix.copy()
    indices = list(range(input_matrix.shape[0]))
    for i in range(SIGNATURE_LENGTH):
        rng.shuffle(indices)
        m = m[indices, :]
        signature_matrix[i, :] = m.argmax(0)
    s = signature_matrix.copy()
    t1 = time.time()
    if verbose:
        print(f'minhashing time: {t1-t0:.2f}')

    # Banding the signature matrix
    hash_buckets = defaultdict(set)
    bands = np.split(signature_matrix, N_BANDS)

    for i, band in enumerate(bands):
        for user, column in enumerate(band.T):
            hash_buckets[(i,) + tuple(column)].add(user)
    # We only care about the buckets with at least two users
    hash_buckets = set(tuple(sorted(v)) for v in hash_buckets.values() if len(v) > 1)

    # Empty hash_buckets
    blocked_pairs = set()
    for bucket in hash_buckets:
        for pair in itertools.combinations(bucket, 2):
            blocked_pairs.add(pair)

    # Calculate signature similarity
    # TODO: parallelize
    t2 = time.time()
    candidate_pairs = set()
    for u1, u2 in blocked_pairs:
        if np.sum(s[:, u1] == s[:, u2])/SIGNATURE_LENGTH >= JS_SIMILARITY_THRESHOLD:
            candidate_pairs.add((u1, u2))

    # Calculate JS
    pairs = set()
    for u1, u2 in candidate_pairs:
        intersection = (m1 := m.getcol(u1)).T.dot((m2 := m.getcol(u2)))[0, 0]
        union = m1.sum() + m2.sum() - intersection
        if (d := intersection / union) >= JS_SIMILARITY_THRESHOLD:
            pairs.add((u1, u2))
    t3 = time.time()
    if verbose:
        print(f'JS calculation time: {(t3-t2):.2f}')

    print(len(pairs))

    # TODO: Update result.txt if > .5


def cosine_main():
    pass


def discrete_cosine_main():
    pass


def main():
    pass


if __name__ == '__main__':
    main()
    jaccard_main()


# Output for with toy=None:
# >>> minhashing time: 159.16
# >>> JS calculation time: fast
