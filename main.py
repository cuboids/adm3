import numpy as np
import scipy
from scipy.sparse import csr_matrix


# For now... eventually we will allow
# flags that specify the file path as
# required.
data = np.load('user_movie_rating.npy')

# Trying to figure out how to make the CSR...
# m = csr_matrix(data[:1000, 2], (data[:1000, 0], data[:1000, 1]))
# print(m)


def jaccard_similarity(u1, u2):
    pass


def cosine_similarity(u1, u2):
    pass


def discrete_cosine_similarity(u1, u2):
    pass


def main():
    pass


if __name__ == '__main__':
    main()
