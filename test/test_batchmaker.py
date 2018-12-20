"""Tests for tf_rnn.batchmaker.
"""
import pytest
import shutil
from multiprocessing import Manager

from tf_rnn import batchmaker
from tf_rnn.batchmaker import *
from tf_rnn.logger import Logger

from .test_data import *
from . import utils


NEW_BATCHED_SCRAMBLED_DATA_2 = [(item, item) for item in BATCHED_SCRAMBLED_DATA_2]
NEW_BATCHED_SCRAMBLED_DATA_3 = [(item, item) for item in BATCHED_SCRAMBLED_DATA_3]

def setup_module(module):
    """Initializes logger with a log directory, so the batchmaker class does not break on logger initialization.
    """
    Logger('test-Logger')


def teardown_module(module):
    """Deletes the test-Logger directory after tests have completed.
    """
    shutil.rmtree('./test-Logger')


class TestSortByLength():
    def test_should_not_change_data_when_it_is_empty(self):
        assert sort_by_length([]) == []

    def test_should_not_change_data_when_it_is_already_sorted(self):
        assert sort_by_length([SORTED_DATA]) == [SORTED_DATA]

    def test_should_sort_scrambled_data(self):
        assert sort_by_length([SCRAMBLED_DATA]) == [SORTED_DATA]

    def test_should_sort_all_data(self):
        assert sort_by_length([SORTED_DATA, SCRAMBLED_DATA, SCRAMBLED_DATA, SORTED_DATA]) == [
            SORTED_DATA, SORTED_DATA, SORTED_DATA, SORTED_DATA]

class TestGroupIntoBatches():
    def test_should_raise_value_error_when_data_is_empty(self):
        with pytest.raises(ValueError):
            group_into_batches([], 4, Manager())

    def test_should_raise_value_error_when_batch_size_is_less_than_one(self):
        with pytest.raises(ValueError):
            group_into_batches([SCRAMBLED_DATA, SCRAMBLED_DATA], 0, Manager())

    def test_should_not_raise_value_error_when_batch_size_is_one(self):
        group_into_batches([SCRAMBLED_DATA, SCRAMBLED_DATA], 1, Manager())
        pass

    def test_should_raise_value_error_when_only_one_set_of_data_is_provided(self):
        with pytest.raises(ValueError):
            group_into_batches([SCRAMBLED_DATA], 2, Manager())

    def test_should_create_batches_out_of_multiple_sets_of_data(self):
        q = group_into_batches([SCRAMBLED_DATA, SCRAMBLED_DATA], 2, Manager())
        assert utils.equivalent(utils.qdata(q), NEW_BATCHED_SCRAMBLED_DATA_2)
        q = group_into_batches([SCRAMBLED_DATA, SCRAMBLED_DATA], 3, Manager())
        assert utils.equivalent(utils.qdata(q), NEW_BATCHED_SCRAMBLED_DATA_3)

class TestTruncateBatches():
    def test_should_raise_value_error_when_data_is_empty(self):
        batchmaker.TRUNCATE_LENGTH = 2
        with pytest.raises(ValueError):
            assert truncate_batch([]) == []

    def test_should_raise_value_error_when_truncate_is_less_than_one(self):
        with pytest.raises(ValueError):
            truncate_batch([])

    def test_should_not_raise_value_error_when_truncate_is_one(self):
        batchmaker.TRUNCATE_LENGTH = 1
        truncate_batch(BATCHED_SCRAMBLED_DATA_2)
        pass

    def test_should_return_same_data_when_truncate_is_equal_to_longest_example(self):
        batchmaker.TRUNCATE_LENGTH = 8
        assert truncate_batch(BATCHED_SCRAMBLED_DATA_2[0]) == [[True] + BATCHED_SCRAMBLED_DATA_2[0] + [True]]

    def test_should_return_batches_with_length_less_than_or_equal_to_truncate(self):
        for i in range(1, 10):
            batchmaker.TRUNCATE_LENGTH = i
            truncated_data = truncate_batch(BATCHED_SCRAMBLED_DATA_2[0])
            for batch in truncated_data:
                for example in batch[1:-1]:
                    assert len(example) <= i

    def test_should_correctly_truncate_data(self):
        batchmaker.TRUNCATE_LENGTH = 4
        assert truncate_batch(BATCHED_SCRAMBLED_DATA_2[0]) == TRUNCATED_SCRAMBLED_BATCHES_4[:2]
        assert truncate_batch(BATCHED_SORTED_DATA_3[0]) == TRUNCATED_SORTED_BATCHES_3_4[:2]
        assert truncate_batch(BATCHED_SORTED_DATA_2[0]) == TRUNCATED_SORTED_BATCHES_4[:2]

class TestGetRowLengths():
    def test_should_return_empty_list_when_data_is_empty(self):
        assert get_row_lengths([]) == []

    def test_should_return_row_lengths_for_one_piece_of_data(self):
        assert get_row_lengths(TRUNCATED_SORTED_BATCHES_4) == SIZES_TRUNCATED_SORTED_BATCHES_4
        assert get_row_lengths(TRUNCATED_SCRAMBLED_BATCHES_4) == SIZES_TRUNCATED_SCRAMBLED_BATCHES_4
        assert get_row_lengths(TRUNCATED_SORTED_BATCHES_3_4) == SIZES_TRUNCATED_SORTED_BATCHES_3_4

class TestPadBatches():
    def test_should_return_empty_list_when_data_is_empty(self):
        assert pad_batches([], 2, PAD).tolist() == []

    def test_should_throw_value_error_when_truncate_is_less_than_one(self):
        with pytest.raises(ValueError):
            pad_batches([], 0, PAD)

    def test_should_not_throw_value_error_when_truncate_is_one(self):
        pad_batches([], 1, PAD)
        pass

    def test_should_correctly_pad_data(self):
        assert pad_batches(TRUNCATED_SCRAMBLED_BATCHES_4, 4, PAD).tolist() == PADDED_SCRAMBLED_BATCHES_4
        assert pad_batches(TRUNCATED_SORTED_BATCHES_4, 4, PAD).tolist() == PADDED_SORTED_BATCHES_4
        assert pad_batches(TRUNCATED_SORTED_BATCHES_3_4, 4, PAD).tolist() == PADDED_SORTED_BATCHES_3_4

class TestMakeBatches():
    def test_should_do_nothing_when_data_is_empty(self):
        x_batches, y_batches, lengths = make_batches([], [], 2, 4, PAD, PAD)
        assert x_batches.tolist() == []
        assert y_batches.tolist() == []
        assert lengths == []

    def test_should_raise_value_error_when_batch_size_is_less_than_one(self):
        with pytest.raises(ValueError):
            make_batches([], [], -1, 4, PAD, PAD)

    def test_should_not_raise_value_error_when_batch_size_is_one(self):
        make_batches([], [], 1, 4, PAD, PAD)
        pass

    def test_should_raise_value_error_when_truncate_is_less_than_one(self):
        with pytest.raises(ValueError):
            make_batches([], [], 2, -1, PAD, PAD)

    def test_should_not_raise_value_error_when_truncate_is_one(self):
        make_batches([], [], 2, 1, PAD, PAD)
        pass

    def test_should_correctly_create_batches(self):
        x_batches, y_batches, lengths = make_batches(SCRAMBLED_DATA, SORTED_DATA, 2, 4, PAD, PAD)
        for batch in x_batches:
            for example in batch:
                assert len(example) == 4
        for batch in y_batches:
            for example in batch:
                assert len(example) <= 4
        assert x_batches.tolist() == PADDED_SORTED_BATCHES_4
        assert y_batches.tolist() == PADDED_SORTED_BATCHES_4
        assert lengths == SIZES_TRUNCATED_SORTED_BATCHES_4
