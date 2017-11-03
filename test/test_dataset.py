import pytest
from ..dataset import *

INPUT_PLACEHOLDER = 1
OUTPUT_PLACEHOLDER = 2

NUM_BATCHES_SMALL = 10
NUM_ROWS_SMALL = 10
ROW_LEN_SMALL = 5

NUM_BATCHES_MEDIUM = 1000
NUM_ROWS_MEDIUM = 100
ROW_LEN_MEDIUM = 20

BATCHED_INPUTS_SMALL = [[[INPUT_PLACEHOLDER for timestep in range(ROW_LEN_SMALL)]
        for row in range(NUM_ROWS_SMALL)]
        for batch in range(NUM_BATCHES_SMALL)]
BATCHED_OUTPUTS_SMALL = [[[INPUT_PLACEHOLDER for timestep in range(ROW_LEN_SMALL)]
        for row in range(NUM_ROWS_SMALL)]
        for batch in range(NUM_BATCHES_SMALL)]
BATCHED_SIZES_SMALL_ = [[True] + [len(row) for row in batch] + [True] for batch in BATCHED_INPUTS_SMALL]
BATCHED_SIZES_SMALL = [[len(row) for row in batch] for batch in BATCHED_INPUTS_SMALL]

BATCHED_INPUTS_MEDIUM = [[[INPUT_PLACEHOLDER for timestep in range(ROW_LEN_MEDIUM)]
        for row in range(NUM_ROWS_MEDIUM)]
        for batch in range(NUM_BATCHES_MEDIUM)]
BATCHED_OUTPUTS_MEDIUM = [[[INPUT_PLACEHOLDER for timestep in range(ROW_LEN_MEDIUM)]
        for row in range(NUM_ROWS_MEDIUM)]
        for batch in range(NUM_BATCHES_MEDIUM)]
BATCHED_SIZES_MEDIUM_ = [[True] + [len(row) for row in batch] + [True] for batch in BATCHED_INPUTS_MEDIUM]
BATCHED_SIZES_MEDIUM = [[len(row) for row in batch] for batch in BATCHED_INPUTS_MEDIUM]

class TestDataPartition():
    def test_should_correctly_calculate_the_batch_length(self):
        partition = DataPartition(BATCHED_INPUTS_SMALL, BATCHED_OUTPUTS_SMALL, BATCHED_SIZES_SMALL_)
        assert partition.x == BATCHED_INPUTS_SMALL
        assert partition.y == BATCHED_OUTPUTS_SMALL
        assert partition.sizes == BATCHED_SIZES_SMALL
        assert partition.num_batches == NUM_BATCHES_SMALL

        partition = DataPartition(BATCHED_INPUTS_MEDIUM, BATCHED_OUTPUTS_MEDIUM, BATCHED_SIZES_MEDIUM_)
        assert partition.x == BATCHED_INPUTS_MEDIUM
        assert partition.y == BATCHED_OUTPUTS_MEDIUM
        assert partition.sizes == BATCHED_SIZES_MEDIUM
        assert partition.num_batches == NUM_BATCHES_MEDIUM
