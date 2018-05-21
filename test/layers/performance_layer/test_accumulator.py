import pytest
import logging
import numpy as np

from ....layers.performance_layer import *
from ...test_data import *

LOGGER = logging.getLogger('TEST')

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

PREDICTIONS = COMBINED_SCRAMBLED_BATCHES_4

LABELS = COMBINED_SCRAMBLED_BATCHES_4 = [
    [0, 1, 2, 3, 5, 5, 1, 7],
    [0, 1, 2, PAD, PAD, PAD, PAD, PAD],
    [0, 1, PAD, PAD, PAD, PAD, PAD, PAD],
    [0, 1, 2, 3, 5, PAD, PAD, PAD],
    [1, 2, 3, 4, PAD, PAD, PAD, PAD],
    [1, PAD, PAD, PAD, PAD, PAD, PAD, PAD],
    [1, 2, 3, 4, 5, 5, 6, PAD],
    [1, 2, 3, 4, 5, 5, PAD, PAD]
]

SIZES = SIZES_COMBINED_SCRAMBLED_BATCHES_4

#
# Accumulator tests
#

def make_data_list(batch, size, loss, accuracy, t_accuracy):
    """Creates the data for updating an accumulator.
    """
    max_size = max(size)
    t_size = [0] * max_size
    for i in size:
        for index, j in enumerate(range(i)):
            t_size[index] += 1
    print('t_size = ', t_size)
    data = [
        loss,  # loss
        accuracy,  # accuracy
        sum(size),  # number of elements in batch
        t_accuracy,  # timestep accuracies
        t_size,  # timestep lengths
        batch,  # predictions
        batch,  # labels
        size  # length of each sequence
    ]
    return data
# End of make_data_list()


class TestUpdate():
    def test_should_correctly_update_with_beginning_batch(self):
        batch = PADDED_SCRAMBLED_BATCHES_4[0]
        sizes = SIZES_TRUNCATED_SCRAMBLED_BATCHES_4[0]
        accumulator = Accumulator(LOGGER, MAX_LENGTH)
        t_accuracy = [0.5 for t in range(max(sizes))]
        data = make_data_list(batch, sizes, 0.2, 0.5, t_accuracy)
        accumulator.update(data, True, False)
        assert accumulator.loss == 0.2
        assert accumulator.accuracy == 0.5
        assert accumulator.elements == sum(sizes)
        assert not accumulator.timestep_accuracies
        assert not accumulator.timestep_elements
        assert not accumulator.confusion_matrix.is_empty()
        assert accumulator.losses == []
        assert accumulator.accuracies == []
        assert accumulator.best_accuracy == 0.0
        assert accumulator.next_timestep_accuracies == t_accuracy
        assert not accumulator.is_best_accuracy
        assert not accumulator.latest_timestep_accuracies
        assert accumulator.latest_confusion_matrix.is_empty()

    def test_should_correctly_merge_timestep_data(self):
        batches = PADDED_SCRAMBLED_BATCHES_4[:2]
        sizes = SIZES_TRUNCATED_SCRAMBLED_BATCHES_4[:2]
        accumulator = Accumulator(LOGGER, MAX_LENGTH)
        t_accuracy = [0.5 for t in range(max(sizes))]
        data = make_data_list(batches[0], sizes[0], 0.2, 0.5, t_accuracy)
        accumulator.update(data, True, False)
        data = make_data_list(batches[1], sizes[1], 0.2, 0.5, t_accuracy)
        accumulator.update(data, False, True)
        assert accumulator.loss == 0.0
        assert accumulator.accuracy == 0.0
        assert accumulator.elements == 0
        assert accumulator.timestep_accuracies == t_accuracy * 2
        assert accumulator.timestep_elements
        assert not accumulator.confusion_matrix.is_empty()
        assert accumulator.losses == [0.2]
        assert accumulator.accuracies == [0.5]
        assert accumulator.best_accuracy == 0.5
        assert accumulator.is_best_accuracy
        assert accumulator.latest_timestep_accuracies == [0.5] * MAX_LENGTH
        assert not accumulator.latest_confusion_matrix.is_empty()

    def test_should_correctly_move_to_next_epoch(self):
        batches = PADDED_SCRAMBLED_BATCHES_4[:2]
        sizes = SIZES_TRUNCATED_SCRAMBLED_BATCHES_4[:2]
        accumulator = Accumulator(LOGGER, MAX_LENGTH)
        data = make_data_list(batches[0], sizes[0], 0.2, 0.5, [0.5 for t in range(max(sizes[0]))])
        accumulator.update(data, True, False)
        data = make_data_list(batches[1], sizes[1], 0.2, 0.5, [0.5 for t in range(max(sizes[1]))])
        accumulator.update(data, False, True)
        accumulator.next_epoch()
        assert accumulator.loss == 0.0
        assert accumulator.accuracy == 0.0
        assert accumulator.elements == 0
        assert not accumulator.timestep_accuracies
        assert not accumulator.timestep_elements
        assert accumulator.confusion_matrix.is_empty()
        assert accumulator.losses == [0.2]
        assert accumulator.accuracies == [0.5]
        assert accumulator.best_accuracy == 0.5
        assert accumulator.is_best_accuracy
        assert accumulator.latest_timestep_accuracies == [0.5] * MAX_LENGTH
        assert not accumulator.latest_confusion_matrix.is_empty()


class TestExtendAccumulatorWithBatch():
    def test_should_raise_value_error_if_sizes_dont_match(self):
        x1 = [list(range(4)) for l in list(range(5))]
        x2 = [list(range(4)) for l in list(range(6))]
        shape = np.shape(PADDED_SCRAMBLED_BATCHES_4[0])
        var = Accumulator(LOGGER, MAX_LENGTH)
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
        var = Accumulator(LOGGER, MAX_LENGTH)
        batch = PADDED_SCRAMBLED_BATCHES_4[0]
        sizes = SIZES_TRUNCATED_SCRAMBLED_BATCHES_4[0][1:-1]
        var.append_batch(batch, batch, sizes)
        var.extend_batch(x1, x2, x3)

    def test_should_correctly_extend_previous_batch(self):
        shape = np.shape(PADDED_SCRAMBLED_BATCHES_4[0])
        var = Accumulator(LOGGER, MAX_LENGTH)
        for index, batch in enumerate(PADDED_SCRAMBLED_BATCHES_4[:2]):
            x = batch
            y = batch
            sizes = SIZES_TRUNCATED_SCRAMBLED_BATCHES_4[index][1:-1]
            if index == 0: var.append_batch(x, y, sizes)
            if index == 1: var.extend_batch(x, y, sizes)
        var.next_epoch()
        assert var.inputs == COMBINED_SCRAMBLED_BATCHES_4[:2]
        assert var.labels == COMBINED_SCRAMBLED_BATCHES_4[:2]
        assert var.sizes == SIZES_COMBINED_SCRAMBLED_BATCHES_4[:2]

class TestAddBatchToAccumulator():
    def test_should_correctly_add_batch(self):
        shape = np.shape(PADDED_SCRAMBLED_BATCHES_4[0])
        var = Accumulator(LOGGER, MAX_LENGTH)
        for index, batch in enumerate(PADDED_SCRAMBLED_BATCHES_4):
            x = batch
            y = batch
            sizes = SIZES_TRUNCATED_SCRAMBLED_BATCHES_4[index][1:-1]
            beginning = SIZES_TRUNCATED_SCRAMBLED_BATCHES_4[index][0]
            ending = SIZES_TRUNCATED_SCRAMBLED_BATCHES_4[index][-1]
            var.update(x, y, sizes, beginning, ending)
        var.next_epoch()
        assert var.inputs == COMBINED_SCRAMBLED_BATCHES_4
        assert var.labels == COMBINED_SCRAMBLED_BATCHES_4
        assert var.sizes == SIZES_COMBINED_SCRAMBLED_BATCHES_4
