import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.spatial import distance


# For now... eventually we will allow
# flags that specify the file path as
# required.
user_movie_rating = np.load('user_movie_rating.npy')

# Getting a feeling for CSR :)
user_movie = user_movie_rating.copy()
user_movie[:, 2] = 1
m = csr_matrix((user_movie[:, 2], (user_movie[:, 0], user_movie[:, 1])))
print(m)


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
    pass


def cosine_main():
    pass


def discrete_cosine_main():
    pass


def main():
    pass


if __name__ == '__main__':
    main()
