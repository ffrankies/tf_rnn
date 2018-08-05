"""Tests the TimestepAccuracies class.

@since 0.6.1
"""

import pytest
from collections import namedtuple

from tf_rnn.layers.utils.accumulator import TimestepAccuracies, AccumulatorData


Dummy = namedtuple('Dummy', ['accuracies', 'counts', 'ending'])

MAX_LENGTH = 8

DUMMY1 = Dummy([0.4, 0.4, 0.4, 0.4], [5, 5, 5, 5], False)

DUMMY2 = Dummy([0.5, 0.5, 0.5, 0.5], [5, 4, 4, 4], True)

DUMMY3 = Dummy([0.8, 0.1, 0.3, 0], [5, 5, 4, 3], False)

DUMMY4 = Dummy([1.0, 0, 0, 0], [3, 3, 1, 0], True)

AVERAGE_ACCURACIES = [0.6, 0.25, 16/45, 0.25, 0.6875, 2/7, 0.4, 0.5]

TOTAL_COUNTS = [10, 10, 9, 8, 8, 7, 5, 4]

class TestUpdate():

    def setup_method(self):
        self.ta = TimestepAccuracies(MAX_LENGTH)

    def test_should_correctly_update_incoming_variables(self):
        self.ta.update(DUMMY1.accuracies, DUMMY1.counts, DUMMY1.ending)
        assert self.ta.timestep_accuracy_list == [[]]
        assert self.ta.timestep_count_list == [[]]
        assert self.ta._incoming_timestep_accuracies == DUMMY1.accuracies
        assert self.ta._incoming_timestep_counts == DUMMY1.counts

    def test_should_correctly_copy_incoming_variables(self):
        self.ta.update(DUMMY1.accuracies, DUMMY1.counts, DUMMY1.ending)
        self.ta.update(DUMMY2.accuracies, DUMMY2.counts, DUMMY2.ending)
        assert self.ta.timestep_accuracy_list == [DUMMY1.accuracies + DUMMY2.accuracies]
        assert self.ta.timestep_count_list == [DUMMY1.counts + DUMMY2.counts]
        assert self.ta._incoming_timestep_accuracies == []
        assert self.ta._incoming_timestep_counts == []
        assert self.ta._epoch == 0

    def test_should_correctly_update_running_average(self):
        self.ta.update(DUMMY1.accuracies, DUMMY1.counts, DUMMY1.ending)
        self.ta.update(DUMMY2.accuracies, DUMMY2.counts, DUMMY2.ending)
        self.ta.update(DUMMY3.accuracies, DUMMY3.counts, DUMMY3.ending)
        self.ta.update(DUMMY4.accuracies, DUMMY4.counts, DUMMY4.ending)
        assert self.ta.timestep_accuracy_list == [AVERAGE_ACCURACIES]
        assert self.ta.timestep_count_list == [TOTAL_COUNTS]
        assert self.ta._incoming_timestep_accuracies == []
        assert self.ta._incoming_timestep_counts == []
        assert self.ta._epoch == 0


class TestNextEpoch():

    def setup_method(self):
        self.ta = TimestepAccuracies(MAX_LENGTH)

    def test_should_correctly_start_new_epoch(self):
        self.ta.update(DUMMY1.accuracies, DUMMY1.counts, DUMMY1.ending)
        self.ta.update(DUMMY2.accuracies, DUMMY2.counts, DUMMY2.ending)
        self.ta.next_epoch()
        assert self.ta.timestep_accuracy_list == [DUMMY1.accuracies + DUMMY2.accuracies]
        assert self.ta.timestep_count_list == [DUMMY1.counts + DUMMY2.counts]
        assert self.ta._incoming_timestep_accuracies == []
        assert self.ta._incoming_timestep_counts == []
        assert self.ta._epoch == 1

    def test_should_correctly_update_after_advancing_epoch(self):
        self.ta.update(DUMMY1.accuracies, DUMMY1.counts, DUMMY1.ending)
        self.ta.update(DUMMY2.accuracies, DUMMY2.counts, DUMMY2.ending)
        self.ta.next_epoch()
        self.ta.update(DUMMY1.accuracies, DUMMY1.counts, DUMMY1.ending)
        self.ta.update(DUMMY2.accuracies, DUMMY2.counts, DUMMY2.ending)
        assert self.ta.timestep_accuracy_list == [DUMMY1.accuracies + DUMMY2.accuracies, 
                                                  DUMMY1.accuracies + DUMMY2.accuracies]
        assert self.ta.timestep_count_list == [DUMMY1.counts + DUMMY2.counts, DUMMY1.counts + DUMMY2.counts]
        assert self.ta._incoming_timestep_accuracies == []
        assert self.ta._incoming_timestep_counts == []
        assert self.ta._epoch == 1

