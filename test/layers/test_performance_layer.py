import pytest
import numpy as np

from ...layers.performance_layer import *
from ..test_data import *

MAX_LENGTH = 8

COMBINED_SCRAMBLED_BATCHES_4 = [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [0, 1, 2, PAD, PAD, PAD, PAD, PAD],
    [0, 1, PAD, PAD, PAD, PAD, PAD, PAD],
    [0, 1, 2, 3, 4, PAD, PAD, PAD],
    [0, 1, 2, 3, PAD, PAD, PAD, PAD],
    [0, PAD, PAD, PAD, PAD, PAD, PAD, PAD],
    [0, 1, 2, 3, 4, 5, 6, PAD],
    [0, 1, 2, 3, 4, 5, PAD, PAD]
]

SIZES_COMBINED_SCRAMBLED_BATCHES_4 = [8, 3, 2, 5, 4, 1, 7, 6]

class TestAppendBatchToPerformanceData():
    def test_should_correctly_append_batch_to_empty_performance_data(self):
        shape = np.shape(PADDED_SCRAMBLED_BATCHES_4[0])
        var = PerformanceVariables(MAX_LENGTH, (shape, shape), (np.int, np.int), PAD)
        batch = PADDED_SCRAMBLED_BATCHES_4[0]
        sizes = SIZES_TRUNCATED_SCRAMBLED_BATCHES_4[0][1:-1]
        var.append_batch(batch, batch, sizes)
        var.complete()
        assert var.inputs == PADDED_SCRAMBLED_BATCHES_4[0]
        assert var.labels == PADDED_SCRAMBLED_BATCHES_4[0]
        assert var.sizes == SIZES_TRUNCATED_SCRAMBLED_BATCHES_4[0][1:-1]
    
class TestExtendPerformanceDataWithBatch():
    def test_should_raise_value_error_if_sizes_dont_match(self):
        x1 = [list(range(4)) for l in list(range(5))]
        x2 = [list(range(4)) for l in list(range(6))]
        shape = np.shape(PADDED_SCRAMBLED_BATCHES_4[0])
        var = PerformanceVariables(MAX_LENGTH, (shape, shape), (np.int, np.int), PAD)
        with pytest.raises(ValueError):
            var.extend_batch(x1, x2, x2)
        with pytest.raises(ValueError):
            var.extend_batch(x1, x1, x2)
        with pytest.raises(ValueError):
            var.extend_batch(x1, x2, x1)

    def test_should_pass_if_sizes_match(self):
        x1 = [list(range(4)) for l in list(range(2))]
        x2 = [list(range(4, 0, -1)) for l in list(range(2))]
        x3 = [4 for l in list(range(2))]
        shape = np.shape(PADDED_SCRAMBLED_BATCHES_4[0])
        var = PerformanceVariables(MAX_LENGTH, (shape, shape), (np.int, np.int), PAD)
        batch = PADDED_SCRAMBLED_BATCHES_4[0]
        sizes = SIZES_TRUNCATED_SCRAMBLED_BATCHES_4[0][1:-1]
        var.append_batch(batch, batch, sizes)
        var.extend_batch(x1, x2, x3)

    def test_should_correctly_extend_previous_batch(self):
        shape = np.shape(PADDED_SCRAMBLED_BATCHES_4[0])
        var = PerformanceVariables(MAX_LENGTH, (shape, shape), (np.int, np.int), PAD)
        for index, batch in enumerate(PADDED_SCRAMBLED_BATCHES_4[:2]):
            x = batch
            y = batch
            sizes = SIZES_TRUNCATED_SCRAMBLED_BATCHES_4[index][1:-1]
            if index == 0: var.append_batch(x, y, sizes)
            if index == 1: var.extend_batch(x, y, sizes)
        var.complete()
        assert var.inputs == COMBINED_SCRAMBLED_BATCHES_4[:2]
        assert var.labels == COMBINED_SCRAMBLED_BATCHES_4[:2]
        assert var.sizes == SIZES_COMBINED_SCRAMBLED_BATCHES_4[:2]

class TestAddBatchToPerformanceVariables():
    def test_should_correctly_add_batch(self):
        shape = np.shape(PADDED_SCRAMBLED_BATCHES_4[0])
        var = PerformanceVariables(MAX_LENGTH, (shape, shape), (np.int, np.int), PAD)
        for index, batch in enumerate(PADDED_SCRAMBLED_BATCHES_4):
            x = batch
            y = batch
            sizes = SIZES_TRUNCATED_SCRAMBLED_BATCHES_4[index][1:-1]
            beginning = SIZES_TRUNCATED_SCRAMBLED_BATCHES_4[index][0]
            ending = SIZES_TRUNCATED_SCRAMBLED_BATCHES_4[index][-1]
            var.add_batch(x, y, sizes, beginning, ending)
        var.complete()
        assert var.inputs == COMBINED_SCRAMBLED_BATCHES_4
        assert var.labels == COMBINED_SCRAMBLED_BATCHES_4
        assert var.sizes == SIZES_COMBINED_SCRAMBLED_BATCHES_4
