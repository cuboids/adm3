from collections import defaultdict
import itertools
import time

import numpy as np
import numpy.random as npr
import scipy
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

# We'll probably directly use scipy.distance to reduce overhead


def jaccard_similarity(u1, u2):
    return distance.jaccard(u1, u2)


def jaccard_similarity2(u1, u2):
    return sum(np.intersect1d(u1, u2))/sum(np.union1d(u1, u2))


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

    # Constructing the signature matrix
    t0 = time.time()
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
    for buck in hash_buckets:
        print(buck)
    t3 = time.time()

    # Empty hash_buckets
    t4 = time.time()
    pairs = set()
    for bucket in hash_buckets:
        for pair in itertools.combinations(bucket, 2):
            pairs.add(pair)
    t5 = time.time()

    # Apply JS distance and update result.txt if > .5
    t6 = time.time()
    similar_users = set()
    for u1, u2 in pairs:
        if (d := distance.jaccard(m.getcol(u1), m.getcol(u2))) > JS_SIMILARITY_THRESHOLD:
            similar_users.add((u1, u2))
            print(d)  # TODO: fix Jaccard distance metric, now broken
    t7 = time.time()

    print(similar_users)

    print(f'minhashing time: {t1-t0:.2f}')
    print(f'banding time: {(t3-t2):.2f}')
    print(f'emptying time: {(t5-t4):.2f}')
    print(f'JS calculation time: {(t7-t6):.2f}')


def cosine_main():
    pass


def discrete_cosine_main():
    pass


def main():
    pass


if __name__ == '__main__':
    main()
    jaccard_main(10_000_000)


# Output for with toy=None:
# >>> minhashing time: 161.89
# >>> banding time: 3.86
# >>> emptying time: 0.26
# >>> JS calculation time: 233.55

