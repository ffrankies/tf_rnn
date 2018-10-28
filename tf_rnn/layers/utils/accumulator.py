"""The performance accumulator class.

@since 0.6.1
"""

from copy import copy
from collections import namedtuple

from .confusion_matrix import ConfusionMatrix


AccumulatorData = namedtuple('AccumulatorData',
                             ['loss', 'size', 'timestep_accuracies', 'timestep_counts', 'predictions',
                              'labels', 'sequence_lengths'])


class TimestepAccuracies(object):
    """Keeps track of timestep accuracies.
    """

    def __init__(self, max_sequence_length: int):
        """Creates a new TimestepAccuracies object.

        Params:
        - max_sequence_length (int): The maximum length of a sequence
        """
        self.max_sequence_length = max_sequence_length
        self.timestep_accuracy_list = list()
        self.timestep_count_list = list()
        self._incoming_timestep_accuracies = list()
        self._incoming_timestep_counts = list()
        self._epoch = 0
    # End of __init__()

    def next_epoch(self):
        """Advances the current epoch by 1.
        """
        self._epoch += 1
    # End of next_epoch()

    def update(self, accuracies: list, counts: list, ending: bool):
        """Updates the timestep accuracies with info from a new minibatch.

        Params:
        - accuracies (list<float>): The timestep accuracies for a new minibatch
        - counts (list<int>): The timestep counts for a new minibatch
        - ending (bool): Whether or not the minibatch is the ending of a sequence
        """
        self._extend_timesteps(accuracies, counts)
        if ending:
            self._merge_timesteps()
    # End of update()

    def _extend_timesteps(self, accuracies: list, counts: list):
        """Appends the timestep accuracies and timestep counts to the next_timestep_counts list.

        Params:
        - accuracies (list<float>): The timestep accuracies for a new minibatch
        - counts (list<int>): The timestep counts for a new minibatch
        """
        if len(self.timestep_accuracy_list) < self._epoch + 1:
            self.timestep_accuracy_list.append(list())
            self.timestep_count_list.append(list())
        self._incoming_timestep_accuracies.extend(accuracies)
        self._incoming_timestep_counts.extend(counts)
    # End of extend_timesteps()

    def _merge_timesteps(self):
        """Updates the cumulative timestep accuracies for the current epoch with a running average that includes the
        latest completed sequences.
        """
        self._incoming_timestep_accuracies = self._incoming_timestep_accuracies[:self.max_sequence_length]
        self._incoming_timestep_counts = self._incoming_timestep_counts[:self.max_sequence_length]
        if self.timestep_accuracy_list[self._epoch]:
            self._update_running_average()
        else:
            self.timestep_accuracy_list[self._epoch] = copy(self._incoming_timestep_accuracies)
            self.timestep_count_list[self._epoch] = copy(self._incoming_timestep_counts)
        self._incoming_timestep_accuracies = list()
        self._incoming_timestep_counts = list()
    # End of _merge_timesteps()

    def _update_running_average(self):
        """Updates the running average accuracy for each timestep using the incoming timestep accuracies and counts.
        """
        for index in range(len(self._incoming_timestep_accuracies)):
            old_avg = self.timestep_accuracy_list[self._epoch][index]
            old_count = self.timestep_count_list[self._epoch][index]
            new_avg = self._incoming_timestep_accuracies[index]
            new_count = self._incoming_timestep_counts[index]
            self.timestep_accuracy_list[self._epoch][index] = update_average(old_avg, old_count, new_avg, new_count)
            self.timestep_count_list[self._epoch][index] += new_count
    # End of _update_running_average()
# End of TimestepAccuracies()


class Accumulator(object):
    """Stores the data needed to evaluate the performance of the model on a given partition of the dataset.

    Instance Variables:
    - max_sequence_length (int): The maximum sequence length for this dataset
    - loss (float): The cumulative average loss for every minibatch
    - counts (float): The cumulative total number of valid elements seen so far
    - timestep_accuracies (TimestepAccuracies): The cumulative average accuracy for each timestep for every epoch
    - losses (list): List of losses for every epoch
    - metrics (list<layers.PerformanceMetrics>): List of performance metrics for every epoch
    - confusion_matrix (ConfusionMatrix): The confusion matrix for visualizing prediction accuracies
    - latest_confusion_matrix (ConfusionMatrix): The confusion matrix for the latest completed epoch
    """

    def __init__(self, max_sequence_length: int):
        """Creates a new PerformanceData object.

        Params:
        - max_sequence_length (int): The maximum sequence length for this dataset
        """
        self.max_sequence_length = max_sequence_length
        self.confusion_matrix = ConfusionMatrix()
        self.latest_confusion_matrix = None
        self.losses = list()
        self.metrics = list()
        self.timestep_accuracies = TimestepAccuracies(max_sequence_length)
        self.loss = None
        self.counts = 0
        self._reset_metrics()
    # End of __init__()

    def update(self, data: AccumulatorData, ending: bool):
        """Adds the performance data from a given minibatch to the PerformanceData object.

        Params:
        - data (AccumulatorData): The performance data for the given minibatch
        - ending (boolean): True if this minibatch marks the end of a sequence
        """
        if self.loss is None:
            self.loss = data.loss
        else:
            self.loss = update_average(self.loss, self.counts, data.loss, data.size)
        self.counts += data.size
        self.timestep_accuracies.update(data.timestep_accuracies, data.timestep_counts, ending)
        self.confusion_matrix.update(data.predictions, data.labels, data.sequence_lengths)
    # End of update()

    def next_epoch(self):
        """Creates space for storing performance data for the next epoch. Also resets metrics for the next epoch.
        """
        metrics = self.confusion_matrix.performance_metrics()
        self.losses.append(self.loss)
        self.metrics.append(metrics)
        self.timestep_accuracies.next_epoch()
        self.loss = None
        self.counts = 0
        self._reset_metrics()
    # End of next_epoch()

    def _reset_metrics(self):
        """Resets the performance metrics for the next epoch.

        Creates the following instance variables, if they haven't already been created:
        - loss (float): The cumulative average loss for every minibatch
        - accuracy (float): The cumulative average accuracy for every minibatch
        - counts (float): The cumulative total number of valid elements seen so far
        """
        self.latest_confusion_matrix = self.confusion_matrix.copy()
        self.confusion_matrix = ConfusionMatrix()
    # End of _reset_metrics()

    def accuracies(self):
        """Returns the list of accuracies from the list of metrics.

        Returns:
        - accuracies (list<float>): The list of accuracies for completed epochs
        """
        return [metric.accuracy for metric in self.metrics]
    # End of accuracies()

    def precisions(self):
        """Returns the list of precision values from the list of metrics.

        Returns:
        - precisions (list<float>): The list of precision values for completed epochs
        """
        return [metric.precision for metric in self.metrics]
    # End of precisions()

    def recalls(self):
        """Returns the list of recall values from the list of metrics.

        Returns:
        - recalls (list<float>): The list of recall values for completed epochs
        """
        return [metric.recall for metric in self.metrics]
    # End of recalls()

    def f1_scores(self):
        """Returns the list of f1_scores from the list of metrics.

        Returns:
        - f1_scores (list<float>): The list of f1_scores for completed epochs
        """
        return [metric.f1_score for metric in self.metrics]
    # End of f1_scores()

    def best_accuracy(self):
        """Returns the best attained accuracy.

        Returns:
        - accuracy (float): The best attained accuracy. If the accumulator has no accuracies yet, returns -1.0
        """
        accuracies = self.accuracies()
        if accuracies:
            accuracy = max(accuracies)
        else:
            accuracy = -1.0
        return accuracy
    # End of best_accuracy()

    def is_best_accuracy(self):
        """Indicates whether or not the latest accuracy is the best attained accuracy.

        Returns:
        - indicator (bool): True if the latest accuracy is the best attained accuracy.
        """
        accuracies = self.accuracies()
        if accuracies:
            latest = accuracies[-1]
            best = max(accuracies)
        else:
            return False
        return latest >= best
    # End of is_best_accuracy()

    def get_timestep_accuracies(self, index: int = -1):
        """Returns the latest timestep accuracies, or the timestep accuracies for any given epoch.

        Params:
        - index (int): The index of the epoch to retrieve. Defaults to -1, which retrieves the last epoch
        """
        return self.timestep_accuracies.timestep_accuracy_list[index]
    # End of get_timestep_accuracies()
# End of PerformanceData()


def update_average(old_avg: float, old_num: int, new_avg: float, new_num: int):
    """Updates the old average with new data.

    Params:
    - old_avg (float): The current average value
    - old_num (int): The number of elements contributing to the current average
    - new_avg (float): The new average value
    - new_num (int): The number of elements contributing to the new average
    """
    old_sum = old_avg * old_num
    new_sum = new_avg * new_num
    updated_sum = old_sum + new_sum
    updated_num = old_num + new_num
    updated_avg = updated_sum / updated_num
    return updated_avg
# End of update_average()
