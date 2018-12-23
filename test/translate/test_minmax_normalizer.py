"""Tests for the minmax_normalizer translator.
"""

import pytest
import math

import numpy as np

from tf_rnn.translate import MinMaxNormalizer


MAX_INT = 178
MIN_INT = 50

MAX_FLOAT = MAX_INT / 3.0
MIN_FLOAT = MIN_INT / 3.0

INT_DATA = np.random.randint(MIN_INT, MAX_INT+1, (100, 100))
FLOAT_DATA = INT_DATA / 3.0


class TestCreate():
    def test_should_create_normalizer_with_correct_min_and_max(self):
        norm = MinMaxNormalizer.create(INT_DATA)
        assert norm.maximum == MAX_INT
        assert norm.minimum == MIN_INT

        norm = MinMaxNormalizer.create(FLOAT_DATA)
        assert norm.maximum == MAX_FLOAT
        assert norm.minimum == MIN_FLOAT

class TestNormalize():
    def test_should_normalize_between_0_and_abs_1_with_int_data(self):
        norm = MinMaxNormalizer.create(INT_DATA)
        normalized = norm.to_rnn_matrix(INT_DATA)

        assert math.isclose(np.max(normalized), 1.0)
        assert math.isclose(np.min(normalized), -1.0)

    def test_should_normalize_between_0_and_abs_1_with_float_data(self):
        norm = MinMaxNormalizer.create(FLOAT_DATA)
        normalized = norm.to_rnn_matrix(FLOAT_DATA)

        assert math.isclose(np.max(normalized), 1.0)
        assert math.isclose(np.min(normalized), -1.0)
    
class TestDenormalize():
    def test_should_get_back_original_int_values(self):
        norm = MinMaxNormalizer.create(INT_DATA)
        normalized = norm.to_rnn_matrix(INT_DATA)
        denormalized = norm.to_human_matrix(normalized)

        assert np.array_equal(INT_DATA.astype(float), denormalized)
        assert np.array_equal(INT_DATA, denormalized.astype(int))

    def test_should_get_back_original_float_values(self):
        norm = MinMaxNormalizer.create(FLOAT_DATA)
        normalized = norm.to_rnn_matrix(FLOAT_DATA)
        denormalized = norm.to_human_matrix(normalized)

        assert np.allclose(FLOAT_DATA, denormalized)
