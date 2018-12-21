"""Tests for the DataPartition class.

@since 0.7.0
"""

import pytest
import shutil
import os

from tf_rnn.batchmaker import Batch
from tf_rnn.dataset import *


INPUT_PLACEHOLDER = 1
OUTPUT_PLACEHOLDER = 2

NUM_BATCHES_SMALL = 10
NUM_ROWS_SMALL = 10
ROW_LEN_SMALL = 5

NUM_BATCHES_MEDIUM = 1000
NUM_ROWS_MEDIUM = 100
ROW_LEN_MEDIUM = 20

SMALL_BATCH = Batch(
    [[INPUT_PLACEHOLDER for timestep in range(ROW_LEN_SMALL)] for row in range(NUM_ROWS_SMALL)],
    [[OUTPUT_PLACEHOLDER for timestep in range(ROW_LEN_SMALL)] for row in range(NUM_ROWS_SMALL)],
    ROW_LEN_SMALL,
    True,
    True
)

SMALL_BATCHES = [SMALL_BATCH] * NUM_BATCHES_SMALL

MEDIUM_BATCH = Batch(
    [[INPUT_PLACEHOLDER for timestep in range(ROW_LEN_MEDIUM)] for row in range(NUM_ROWS_MEDIUM)],
    [[OUTPUT_PLACEHOLDER for timestep in range(ROW_LEN_MEDIUM)] for row in range(NUM_ROWS_MEDIUM)],
    ROW_LEN_MEDIUM,
    True,
    True
)

MEDIUM_BATCHES = [MEDIUM_BATCH] * NUM_BATCHES_MEDIUM


class TestDataPartition():

    def setup_method(self, method):
        """Sets up stuff for data partition testing.
        """
        print('testing...')

    def teardown_method(self, method):
        """Deletes the partition data after method is run.
        """
        os.remove('./partition.pkl')

    def test_should_save_the_partition(self):
        DataPartition(SMALL_BATCHES, './partition.pkl', NUM_ROWS_SMALL)
        assert os.path.isfile('./partition.pkl')
    
    def test_should_have_no_file_handler_upon_creation(self):
        partition = DataPartition(SMALL_BATCHES, './partition.pkl', NUM_ROWS_SMALL)
        assert partition._file_handler == None
        assert partition.index == 0

    def test_should_have_closed_file_handler_after_all_batches_are_done(self):
        partition = DataPartition(SMALL_BATCHES, './partition.pkl', NUM_ROWS_SMALL)
        [None for _ in partition]
        assert partition._file_handler.closed
        assert partition.index == 0
    
    def test_should_retrieve_correct_batches(self):
        partition = DataPartition(MEDIUM_BATCHES, './partition.pkl', NUM_BATCHES_MEDIUM)
        for index, batch in enumerate(partition):
            assert batch.x == MEDIUM_BATCHES[index].x
            assert batch.y == MEDIUM_BATCHES[index].y
            assert batch.sequence_lengths == MEDIUM_BATCHES[index].sequence_lengths
            assert batch.beginning == MEDIUM_BATCHES[index].beginning
            assert batch.ending == MEDIUM_BATCHES[index].ending
