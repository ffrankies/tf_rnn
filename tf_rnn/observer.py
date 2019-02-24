"""An object for observing network predictions against the actual labels.

@since 0.7.0
"""

import random

import numpy as np

from . import constants
from .utils import create_directory
from .dataset import DataPartition
from .translate.tokenizer import Tokenizer
from .layers.utils.accumulator import AccumulatorData


class Observer(object):
    """A singleton Observer class for observing and evaluating the predictions made by the network during training.
    It is a Singleton so that the object doesn't have to be passed around the training functions.
    """

    _current_epoch = None
    _current_batch = None
    _current_sequence = None
    _sequence_indexes = None
    _observer_path = None

    @classmethod
    def init(cls, partition: DataPartition, num_sequences: int, run_dir: str) -> type:
        """Figures out which sequences to observe. Does so by randomly selecting num_sequences from all sequences in
        the partition, and storing their indexes in sequence_indexes.

        Params:
        - partition (DataPartition): The data partition containing the sequences
        - num_sequences (int): The number of sequences to observe
        - run_dir (str): The directory in which the model is being run

        Returns:
        - class (type): The Observer class type, for chaining function calls (in case that's needed)
        """
        random.seed()  # seeds with current time or other system source of randomness
        cls._sequence_indexes = random.sample(range(partition.num_sequences), num_sequences)
        cls._observer_path = run_dir + constants.OBSERVER_FILE
        create_directory(cls._observer_path)
        return cls
    # End of init()

    @classmethod
    def set_epoch(cls, epoch: int) -> type:
        """Sets the current epoch to epoch. Also resets the batch count back to 0.

        Params:
        - epoch (int): The current epoch number

        Returns:
        - class (type): The Observer class type, for chaining function calls (in case that's needed)
        """
        cls._current_epoch = epoch
        cls._current_batch = 0
        cls._current_sequence = 0
        cls._print_epoch()
        return cls
    # End of set_epoch()

    @classmethod
    def observe(cls, data: AccumulatorData, indexer: Tokenizer) -> type:
        """Observes predictions made by the RNN. This method is meant to be called after every batch.

        Params:
            data (AccumulatorData): The data passed to the accumulator
            indexer (Tokenizer): The indexer, for translating indexes back to tokens

        Returns:
            type: The Observer class type, for chaining function calls (in case that's needed)
        """
        cls._current_batch += 1
        for index, sequence_length in enumerate(data.sequence_lengths):
            if sequence_length > 0:  # Ignore empty sequences
                cls._current_sequence += 1
            if cls._current_sequence in cls._sequence_indexes:
                predicted = data.predictions[index]
                predicted = indexer.to_human_vector(predicted)
                actual = data.labels[0][index].astype(np.int32)
                actual = indexer.to_human_vector(actual)
                cls._print_sample(predicted, np.squeeze(actual))
        return cls
    # End of observe()

    @classmethod
    def _print_epoch(cls):
        """Prints the current epoch number to the observer file.
        """
        with open(cls._observer_path, mode="at") as observer_file:
            epoch_str = "Epoch: {:d}\n".format(cls._current_epoch)
            observer_file.write(epoch_str)
    # End of _print_epoch()

    @classmethod
    def _print_sample(cls, predicted: list, actual: list):
        """Prints the current sample prediction comparison to the observer file.
        """
        predicted_str = [str(value) for value in predicted]
        actual_str = [str(value) for value in actual]
        max_value_length = max(map(len, predicted_str+actual_str))
        # Right-justify strings so they're easier to compare
        predicted_str = " ".join(["{0:>{1}}".format(value, max_value_length) for value in predicted_str])
        actual_str = " ".join(["{0:>{1}}".format(value, max_value_length) for value in actual_str])
        with open(cls._observer_path, mode="at") as observer_file:
            sample_str    = "\tSequence number: {:d} in batch: {:d}\n".format(cls._current_sequence, cls._current_batch)
            predicted_str = "\t\tPredicted: {}\n".format(predicted_str)
            actual_str    = "\t\tActual   : {}\n".format(actual_str)
            observer_file.writelines([sample_str, predicted_str, actual_str])
    # End of _print_sample()
# End of Observer
