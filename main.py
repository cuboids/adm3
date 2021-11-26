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
N_BANDS = 16


# TODO: allow fp to be specified by flag
umr = np.load('user_movie_rating.npy')
umr[:, :2] -= 1  # Ensure rows and columns start at 0
assert umr.min() == 0
input_matrix = None


def cosine_similarity(u1, u2):
    return distance.cosine(u1, u2)


# Same as cosine similarity, just with truncated u1 and u2
def discrete_cosine_similarity(u1, u2):
    return distance.cosine(u1, u2)


def jaccard_main(toy=None, verbose=True):
    """Find user pairs u1 and u2 such that JS(u1, u2) > JS_SIMILARITY_THRESHOLD

    toy: reducing the data"""

    # Constructing the signature matrix
    t0 = time.time()
    global input_matrix
    input_matrix = csr_matrix((umr[:toy, 2], (umr[:toy, 1], umr[:toy, 0]))).sign()
    signature_matrix = np.full(([SIGNATURE_LENGTH, input_matrix.shape[1]]), np.inf)

    rng = npr.default_rng(SEED)
    m = input_matrix.copy()
    indices = list(range(input_matrix.shape[0]))
    for i in range(SIGNATURE_LENGTH):
        rng.shuffle(indices)
        m = m[indices, :]
        signature_matrix[i, :] = m.argmax(0)
    t1 = time.time()
    if verbose:
        print(f'minhashing time: {t1-t0:.2f}')

    # Banding the signature matrix
    t2 = time.time()
    hash_buckets = defaultdict(set)
    bands = np.split(signature_matrix, N_BANDS)
    temp = bands[0]

    for band in range(N_BANDS):
        for user, column in enumerate(temp.T):
            hash_buckets[(band,) + tuple(column)].add(user)
    # We only care about the buckets with at least two users
    hash_buckets = set(tuple(sorted(v)) for v in hash_buckets.values() if len(v) > 1)
    t3 = time.time()
    if verbose:
        print(f'banding time: {(t3-t2):.2f}')

    # Empty hash_buckets
    t4 = time.time()
    pairs = set()
    for bucket in hash_buckets:
        for pair in itertools.combinations(bucket, 2):
            pairs.add(pair)
    t5 = time.time()
    if verbose:
        print(f'emptying time: {(t5-t4):.2f}')

    # Apply JS distance and update result.txt if > .5
    t6 = time.time()

    similar_users = set()
    for u1, u2 in pairs:
        intersection = (m1 := m.getcol(u1)).T.dot((m2 := m.getcol(u2)))[0, 0]
        union = m1.sum() + m2.sum() - intersection
        if (d := intersection/union) > JS_SIMILARITY_THRESHOLD:
            similar_users.add((u1, u2))
    t7 = time.time()
    if verbose:
        print(f'JS calculation time: {(t7-t6):.2f}')
        print()

    print(f'Number of similar users found: {len(similar_users)}')


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
# >>> minhashing time: 161.89
# >>> banding time: 3.86
# >>> emptying time: 0.26
# >>> JS calculation time: 233.55
