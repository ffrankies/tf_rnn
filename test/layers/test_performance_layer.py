import pytest
import numpy as np

from ...layers.performance_layer import *
from ..test_data import *

MAX_LENGTH = 8

COMBINED_SCRAMBLED_BATCHES_4 = [
    [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2, PAD, PAD, PAD, PAD, PAD],
        [0, 1, PAD, PAD, PAD, PAD, PAD, PAD],
        [0, 1, 2, 3, 4, PAD, PAD, PAD],
        [0, 1, 2, 3, PAD, PAD, PAD, PAD],
        [0, PAD, PAD, PAD, PAD, PAD, PAD, PAD],
        [0, 1, 2, 3, 4, 5, 6, PAD],
        [0, 1, 2, 3, 4, 5, PAD, PAD]
    ]
]

SIZES_COMBINED_SCRAMBLED_BATCHES_4 = [8, 3, 2, 5, 4, 1, 7, 6]

class TestAddBatchToPerformanceData():
    def test_should_correctly_add_batch_to_performance_data(self):
        shape = np.shape(PADDED_SCRAMBLED_BATCHES_4[0])
        var = PerformanceVariables(MAX_LENGTH, (shape, shape), (np.float, np.int), PAD)
        for index, batch in enumerate(PADDED_SCRAMBLED_BATCHES_4):
            x = batch
            y = batch
            sizes = SIZES_TRUNCATED_SCRAMBLED_BATCHES_4[index][1:-1]
            beginning = SIZES_TRUNCATED_SCRAMBLED_BATCHES_4[index][0]
            ending = SIZES_TRUNCATED_SCRAMBLED_BATCHES_4[index][-1]
            var.add_batch(x, y, sizes, beginning, ending)
        assert var._inputs == COMBINED_SCRAMBLED_BATCHES_4
        assert var._labels == COMBINED_SCRAMBLED_BATCHES_4
        assert var._sizes == SIZES_COMBINED_SCRAMBLED_BATCHES_4
