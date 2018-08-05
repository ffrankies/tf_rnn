"""Test for the accumulator module.
"""
import pytest
import logging
import numpy as np

from tf_rnn.layers.utils import Accumulator, AccumulatorData

from ...test_data import PAD, PADDED_SCRAMBLED_BATCHES_4, SIZES_TRUNCATED_SCRAMBLED_BATCHES_4

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

def make_data_list(batch, size, loss, t_accuracy):
    """Creates the data for updating an accumulator.
    """
    max_size = max(size)
    t_size = [0] * max_size
    for i in size:
        for index, _ in enumerate(range(i)):
            t_size[index] += 1
    data = AccumulatorData(loss, sum(size), t_accuracy, t_size, batch, batch, size)
    return data
# End of make_data_list()


class TestUpdate():
    def test_should_correctly_update_with_beginning_batch(self):
        batch = PADDED_SCRAMBLED_BATCHES_4[0]
        sizes = SIZES_TRUNCATED_SCRAMBLED_BATCHES_4[0]
        accumulator = Accumulator(MAX_LENGTH)
        t_accuracy = [0.5 for t in range(max(sizes))]
        data = make_data_list(batch, sizes, 0.2, t_accuracy)
        accumulator.update(data, False)
        assert accumulator.loss == 0.2
        assert accumulator.accuracies() == []  # Next epoch hasn't been called
        assert accumulator.counts == sum(sizes)
        assert not accumulator.timestep_accuracies.timestep_accuracy_list[0]
        assert not accumulator.timestep_accuracies.timestep_count_list[0]
        assert not accumulator.confusion_matrix.is_empty()
        assert accumulator.losses == []
        assert accumulator.best_accuracy() == -1.0
        assert accumulator.timestep_accuracies._incoming_timestep_accuracies == t_accuracy
        assert not accumulator.is_best_accuracy()
        assert accumulator.latest_confusion_matrix.is_empty()

    def test_should_correctly_merge_timestep_data(self):
        batches = PADDED_SCRAMBLED_BATCHES_4[:2]
        sizes = SIZES_TRUNCATED_SCRAMBLED_BATCHES_4[:2]
        accumulator = Accumulator(MAX_LENGTH)
        t_accuracy0 = [0.5 for t in range(max(sizes[0]))]
        t_accuracy1 = [0.5 for t in range(max(sizes[1]))]
        data = make_data_list(batches[0], sizes[0], 0.2, t_accuracy0)
        accumulator.update(data, False)
        data = make_data_list(batches[1], sizes[1], 0.2, t_accuracy1)
        accumulator.update(data, True)
        assert accumulator.loss == 0.2
        assert accumulator.accuracies() == []  # Next epoch hasn't been called
        assert accumulator.counts == sum(sizes[0]) + sum(sizes[1])
        assert accumulator.timestep_accuracies.timestep_accuracy_list == [t_accuracy0 + t_accuracy1]
        assert accumulator.timestep_accuracies.timestep_count_list[0]  # Assert that it isn't empty
        assert not accumulator.confusion_matrix.is_empty()
        assert accumulator.losses == []
        assert accumulator.best_accuracy() == -1.0
        assert not accumulator.is_best_accuracy()
        assert accumulator.latest_confusion_matrix.is_empty()

    def test_should_correctly_move_to_next_epoch(self):
        batches = PADDED_SCRAMBLED_BATCHES_4[:2]
        sizes = SIZES_TRUNCATED_SCRAMBLED_BATCHES_4[:2]
        accumulator = Accumulator(MAX_LENGTH)
        data = make_data_list(batches[0], sizes[0], 0.2, [0.5 for t in range(max(sizes[0]))])
        accumulator.update(data, False)
        data = make_data_list(batches[1], sizes[1], 0.2, [0.5 for t in range(max(sizes[1]))])
        accumulator.update(data, True)
        accumulator.next_epoch()
        assert accumulator.loss == None
        assert accumulator.accuracies() == [1.0]
        assert accumulator.counts == 0
        assert not accumulator.timestep_accuracies._incoming_timestep_accuracies
        assert accumulator.timestep_accuracies._epoch == 1
        assert accumulator.confusion_matrix.is_empty()
        assert accumulator.losses == [0.2]
        assert accumulator.accuracies() == [1.0]
        assert accumulator.best_accuracy() == 1.0
        assert accumulator.is_best_accuracy
        assert accumulator.get_timestep_accuracies() == [0.5] * MAX_LENGTH
        assert not accumulator.latest_confusion_matrix.is_empty()
