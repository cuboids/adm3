import argparse
from collections import defaultdict
import itertools
import time

import numpy as np
import numpy.linalg
import numpy.random as npr
from scipy.sparse import csr_matrix


# Process command line options
parser = argparse.ArgumentParser()
parser.add_argument('-d', help='data file path')
parser.add_argument('-s', type=int, help='random seed')
parser.add_argument('-m', help='similarity measure ("js", "cs", or "dcs")')
parser.add_argument('-v', type=bool, help='verbose')
args = parser.parse_args()


def main(fp=None, seed=None, similarity_measure=None, verbose=False):

    # Setting defaults
    fp = fp if fp is not None else 'user_movie_rating.npy'
    seed = seed if seed is not None else 0
    similarity_measure = similarity_measure if similarity_measure is not None else 'js'

    lsh = LSH(fp, seed, similarity_measure, verbose)
    lsh.main()


class LSH:
    """Implementation of Locality Sensitive Hashing for the Netflix Challenge data"""

    # Hyperparameters
    THRESHOLDS = {'js': .5, 'cs': .73, 'dcs': .73}
    SIGNATURE_LENGTHS = {'js': 256, 'cs': 320, 'dcs': 320}
    ROWS_PER_BAND = {'js': 8, 'cs': 16, 'dcs': 16}
    BANDS = {'js': 32, 'cs': 20, 'dcs': 20}
    THRESHOLD_EASING = {'js': .01, 'cs': .065, 'dcs': .065}

    def __init__(self, fp, seed, similarity_measure, verbose=False):

        self._fp = fp
        self._seed = seed
        self._sim = similarity_measure
        self._verbose = verbose

        assert self._sim in ['js', 'cs', 'dcs']

        try:
            self._umr = np.load(fp)
        except FileNotFoundError:
            # download from URL
            pass

        self._umr[:, :2] -= 1  # Ensure rows and columns start at 0
        assert self._umr.min() == 0

        self._input_matrix = csr_matrix((self._umr[:, 2], (self._umr[:, 1], self._umr[:, 0])))
        self._input_matrix_truncated = self._input_matrix.sign()

        # Initialize output containers
        self.pairs = {'js': set(), 'cs': set(), 'dcs': set()}

    def _compute_blocked_pairs(self, signature_matrix):
        """Compute the blocked pairs of a signature matrix Σ
        given a distance measure"""

        # Banding the signature matrix
        hash_buckets = defaultdict(set)
        bands = np.split(signature_matrix, self.BANDS[self._sim])

        for i, band in enumerate(bands):
            for user, column in enumerate(band.T):
                hash_buckets[(i,) + tuple(column)].add(user)

        # We only care about the buckets with at least two users
        # Also, we will now store hash_buckets as a set of tuples:
        # we will not need the keys anymore.
        hash_buckets = set(tuple(sorted(v)) for v in hash_buckets.values() if len(v) > 1)

        # Calculate the blocked pairs, that is, the pairs (u1, u2)
        # such that u1 and u2 are hashed into the same block.
        blocked_pairs = set()
        for bucket in hash_buckets:
            for pair in itertools.combinations(bucket, 2):
                blocked_pairs.add(pair)

        return blocked_pairs

    def _jaccard_main(self):
        """Find user pairs (u1, u2) such that JS(u1, u2) < JS_THRESHOLD"""

        t0 = time.time()
        # We will first construct the signature matrix
        signature_matrix = s = np.full(([self.SIGNATURE_LENGTHS['js'], self._input_matrix.shape[1]]), np.inf)

        # Permute the rows of m SIGNATURE_LENGTH times
        rng = npr.default_rng(self._seed)
        m = self._input_matrix_truncated.copy()
        indices = list(range(m.shape[0]))
        for i in range(self.SIGNATURE_LENGTHS[self._sim]):
            rng.shuffle(indices)
            m = m[indices, :]
            signature_matrix[i, :] = m.argmax(0)
        t1 = time.time()

        if self._verbose:
            print(f'minhashing time: {t1 - t0:.2f}')

        blocked_pairs = self._compute_blocked_pairs(signature_matrix)
        if self._verbose:
            print(f'number of blocked pairs: {len(blocked_pairs)}')

        # Calculate the candidate pairs, that is, the pairs for
        # which JS(s1, s2) > JS_THRESHOLD for signatures s1 and s2
        # corresponding to u1 and u2 respectively
        t2 = time.time()
        candidate_pairs = set()
        for u1, u2 in blocked_pairs:
            jsim = np.sum(s[:, u1] == s[:, u2]) / self.SIGNATURE_LENGTHS['js']
            if jsim >= self.THRESHOLDS['js'] - self.THRESHOLD_EASING['js']:
                candidate_pairs.add((u1, u2))
        if self._verbose:
            print(f'number of candidate pairs: {len(candidate_pairs)}')

        # Postprocessing: calculate the Jaccard distance for the candidate pairs
        for u1, u2 in candidate_pairs:
            intersection = (m1 := m.getcol(u1)).T.dot((m2 := m.getcol(u2)))[0, 0]
            union = m1.sum() + m2.sum() - intersection
            if intersection / union >= self.THRESHOLDS['js']:
                self.pairs['js'].add((u1, u2))
        t3 = time.time()
        if self._verbose:
            print(f'JS calculation time: {t3 - t2:.2f}')
            print(f'Number of pairs found: {len(self.pairs["js"])}')

    @staticmethod
    def _cosine_similarity(v, w, sparse_columns=False):

        # Sparse dot products require "unpacking" the resulting scalar value
        dot_product = v.T.dot(w)[0, 0] if sparse_columns else v.T.dot(w)
        return 1 - np.arccos(dot_product / np.linalg.norm(v) / np.linalg.norm(w)) / np.pi

    def _cosine_main(self, discrete=False):
        """Find user pairs u1, u2 such that CS(u1, u2) < CS_THRESHOLD or DSC(u1, u2) < DCS_THRESHOLD

        Args:
            discrete: use DSC instead of SC
        """

        t0 = time.time()

        # Use random hyperplanes to create signatures
        rng = npr.default_rng(self._seed)
        m = self._input_matrix_truncated.copy() if discrete else self._input_matrix.copy()

        # Create random matrix with SIGNATURE_LENGTH random hyperplanes
        hyperplanes = 2 * rng.integers(2, size=(m.shape[0], self.SIGNATURE_LENGTHS[self._sim])) - 1
        signature_matrix = s = np.sign(m.T.dot(hyperplanes) + .5).T
        t1 = time.time()

        if self._verbose:
            print(f'Signature calculation time: {t1 - t0:.2f}')

        blocked_pairs = self._compute_blocked_pairs(signature_matrix)
        if self._verbose:
            print(f'Number of blocked pairs: {len(blocked_pairs)}')

        # Calculate the candidate pairs, that is, the pairs for
        # which CS(s1, s2) > CS_THRESHOLD for signatures s1 and s2
        # corresponding to u1 and u2 respectively
        t2 = time.time()
        candidate_pairs = set()
        for u1, u2 in blocked_pairs:
            cos_sim = self._cosine_similarity(s[:, u1], s[:, u2])
            if cos_sim >= self.THRESHOLDS[self._sim] - self.THRESHOLD_EASING[self._sim]:
                candidate_pairs.add((u1, u2))

        if self._verbose:
            print(f'number of candidate pairs: {len(candidate_pairs)}')
        # Postprocessing: Calculate actual cosine distance for the candidate pairs
        for u1, u2 in candidate_pairs:
            cos_sim = self._cosine_similarity(m.getcol(u1).toarray(), m.getcol(u2).toarray(), sparse_columns=True)
            if cos_sim >= self.THRESHOLDS[self._sim]:
                self.pairs[self._sim].add((u1, u2))
        t3 = time.time()
        if self._verbose:
            print(f'(D)CS calculation time: {t3 - t2:.2f}')
            print(f'Number of pairs found: {len(self.pairs[self._sim])}')

    def _write_results(self):
        with open(self._sim + '.txt') as f:
            for u1, u2 in sorted(self.pairs[self._sim]):
                f.write(f'{u1}, {u2}')
        pass

    def main(self):
        if self._sim == 'cs':
            self._cosine_main(discrete=False)
        elif self._sim == 'dsc':
            self._cosine_main(discrete=True)
        elif self._sim == 'js':
            self._jaccard_main()
        self._write_results()


if __name__ == '__main__':
    params = vars(args)
    main(params['d'], params['s'], params['m'], params['v'])
