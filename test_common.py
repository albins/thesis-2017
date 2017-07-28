import math

import common

import numpy as np

def make_matrix(n, m):
    a = np.arange(1, n * m + 1)
    return a.reshape(n, m)


def assert_keep_portion_split(m, keep_portion):
    length, height = m.shape
    first_portion = math.ceil(length * keep_portion)
    last_portion = length - first_portion

    first, second = common.sample_matrix(m, keep_portion=keep_portion)
    assert first.shape == (first_portion, height)
    assert second.shape == (last_portion, height)
    return first, second


def test_sample_matrix():
    for matrix_size in range(1, 1000, 5):
        m = make_matrix(matrix_size, 4)
        for percent in range(1, 100):
            frst, last = assert_keep_portion_split(m, percent/100)
            joined = np.append(frst, last, axis=0)
            assert np.array_equal(joined, m)
