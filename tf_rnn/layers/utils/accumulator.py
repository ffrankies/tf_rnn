"""The performance accumulator class.

@since 0.6.1
"""
from copy import deepcopy

from .confusion_matrix import ConfusionMatrix

# The following imports are only used for type hinting
from ...logger import Logger

class Accumulator(object):
    """Stores the data needed to evaluate the performance of the model on a given partition of the dataset.

    Instance Variables:
    - logger (logging.Logger): The logger used by the RNN model
    - max_sequence_length (int): The maximum sequence length for this dataset
    - loss (float): The cumulative average loss for every minibatch
    - accuracy (float): The cumulative average accuracy for every minibatch
    - best_accuracy (float): The best accuracy produced so far
    - is_best_accuracy (boolean): True if the latest produced accuracy is the best accuracy
    - elements (float): The cumulative total number of valid elements seen so far
    - timestep_accuracies (list): The cumulative average accuracy for each timestep
    - timestep_elements (list): The cumulative number of valid elements for each timestep
    - losses (list): List of losses for every epoch
    - accuracies (list): List of accuracies for every epoch
    - latest_timestep_accuracies (list): The latest timestep accuracies
    - confusion_matrix (ConfusionMatrix): The confusion matrix for visualizing prediction accuracies
    - latest_confusion_matrix (ConfusionMatrix): The confusion matrix for the latest completed epoch
    - Temporary instance variables:
      - next_timestep_accuracies (list): Incoming average accuracies per timestep
      - next_timestep_elements (list): Incoming number of valid elements per timestep
    """

    def __init__(self, logger: Logger, max_sequence_length: int):
        """Creates a new PerformanceData object.

        Params:
        - logger (logging.Logger): The logger from the model
        - max_sequence_length (int): The maximum sequence length for this dataset
        """
        self.logger = logger
        self.logger.debug('Creating a PerformanceData object')
        self.max_sequence_length = max_sequence_length
        self.confusion_matrix = ConfusionMatrix(logger)
        self.latest_confusion_matrix = None
        self.best_accuracy = 0.0
        self.is_best_accuracy = False
        self.losses = list()
        self.accuracies = list()
        self.latest_timestep_accuracies = list()
        self.next_timestep_accuracies = list()
        self.next_timestep_elements = list()
        self._reset_metrics()
    # End of __init__()

    def update(self, data: list, beginning: bool, ending: bool):
        """Adds the performance data from a given minibatch to the PerformanceData object.

        Params:
        - data (tuple/list): The performance data for the given minibatch
          - loss (float): The average loss for the given minibatch
          - accuracy (float): The average accuracy for the given minibatch
          - size (int): The number of valid elements in this minibatch
          - timestep_accuracies (list): The average accuracy for each timestep in this minibatch
          - timestep_elements (list): The number of valid elements for each timestep in this minibatch
          - predictions (list): The predictions made at every timestep, in token format
          - labels (list): The correct predictions for the minibatch
          - sequence_lengths (list): The lengths of each sequence in the minibatch
        - beginning (boolean): True if this minibatch marks the start of a sequence
        - ending (boolean): True if this minibatch marks the end of a sequence
        """
        loss, accuracy, size, timestep_accuracies, timestep_elements, predictions, labels, sequence_lengths = data
        # self.logger.debug("Minibatch loss: %.2f | Minibatch accuracy: %.2f" % (loss, accuracy))
        self.loss = self._update_average(self.loss, self.elements, loss, size)
        self.accuracy = self._update_average(self.accuracy, self.elements, accuracy, size)
        self.elements += size
        self.extend_timesteps(timestep_accuracies, timestep_elements)
        if ending is True:
            self._merge_timesteps()
        self.confusion_matrix.update(predictions, labels, sequence_lengths)
        # self.logger.debug("Updated loss: %.2f | Updated accuracy: %.2f" % (self.loss, self.accuracy))
    # End of update()

    def _update_average(self, old_avg: float, old_num: int, new_avg: float, new_num: int):
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
    # End of _update_average()

    def extend_timesteps(self, accuracies: list, sizes: list):
        """Appends the timestep accuracies and timestep sizes to the next_timestep_elements list.

        Params:
        - accuracies (list): The list of accuracies for each timestep in the minibatch
        - sizes (list): The list of the number of valid elements for each timestep in the minibatch
        """
        # self.logger.debug('Extending incoming timestep accuracies')
        if len(accuracies) != len(sizes):
            error_msg = ("Timestep accuracies and elements for each minibatch must be of same size."
                         "Accuracies: %d, Elements: %d" % (len(accuracies), len(sizes)))
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        self.next_timestep_accuracies.extend(accuracies)
        self.next_timestep_elements.extend(sizes)
    # End of extend_timesteps()

    def _merge_timesteps(self):
        """Updates the cumulative timestep accuracies.
        """
        timestep_accuracies = self.next_timestep_accuracies[:self.max_sequence_length]
        timestep_elements = self.next_timestep_elements[:self.max_sequence_length]
        if not self.timestep_accuracies:
            self._copy_timestep_accuracy_info(timestep_accuracies, timestep_elements)
        else:
            self._update_timestep_accuracy_info(timestep_accuracies, timestep_elements)
        self.next_timestep_accuracies = list()
        self.next_timestep_elements = list()
    # End of _merge_timesteps()

    def _copy_timestep_accuracy_info(self, timestep_accuracies: list, timestep_elements: list):
        """Copies over the next timestep accuracy and elements info into the timestep_accuracies and timestep_elements
        variables.

        Params:
        - timestep_accuracies (list<float>): The incoming timestep accuracies
        - timestep_elements (list<int>): The incoming timestep lengths
        """
        self.timestep_accuracies = deepcopy(timestep_accuracies)
        self.timestep_elements = deepcopy(timestep_elements)
    # End of _copy_timestep_accuracy_info()

    def _update_timestep_accuracy_info(self, timestep_accuracies: list, timestep_elements: list):
        """Updates the timestep accuracy information by keeping a running average over all the data presented.

        Params:
        - timestep_accuracies (list<float>): The incoming timestep accuracies
        - timestep_elements (list<int>): The incoming timestep lengths
        """
        for index, _ in enumerate(timestep_accuracies):
            old_avg = self.timestep_accuracies[index]
            old_num = self.timestep_elements[index]
            new_avg = timestep_accuracies[index]
            new_num = timestep_elements[index]
            self.timestep_accuracies[index] = self._update_average(old_avg, old_num, new_avg, new_num)
            self.timestep_elements[index] += new_num
    # End of _update_timestep_accuracy_info()

    def next_epoch(self):
        """Creates space for storing performance data for the next epoch. Also resets metrics for the next epoch.
        """
        if self.accuracy >= self.best_accuracy:
            self.best_accuracy = self.accuracy
            self.is_best_accuracy = True
        else:
            self.is_best_accuracy = False
        self.losses.append(self.loss)
        self.accuracies.append(self.accuracy)
        self.latest_timestep_accuracies = deepcopy(self.timestep_accuracies)
        self._reset_metrics()
    # End of next_epoch()

    def _reset_metrics(self):
        """Resets the performance metrics for the next epoch.

        Creates the following instance variables, if they haven't already been created:
        - loss (float): The cumulative average loss for every minibatch
        - accuracy (float): The cumulative average accuracy for every minibatch
        - elements (float): The cumulative total number of valid elements seen so far
        - timestep_accuracies (list): The cumulative average accuracy for each timestep
        - timestep_elements (list): The cumulative number of valid elements for each timestep
        """
        self.loss = 0.0
        self.accuracy = 0.0
        self.elements = 0
        self.timestep_accuracies = None
        self.timestep_elements = None
        self.latest_confusion_matrix = self.confusion_matrix.copy()
        self.confusion_matrix = ConfusionMatrix(self.logger)
    # End of _reset_metrics()
# End of PerformanceData()