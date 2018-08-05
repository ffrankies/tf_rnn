"""An object for storing a dataset for training.

@since 0.6.0
"""
import random
import math
from typing import Any

from . import dataset_utils
from . import batchmaker
from . import constants
from . import indexer
from .logger import info, debug, trace

# These variables are only imported for type hinting
from .logger import Logger
from .settings import SettingsNamespace


class DataPartition(object):
    """Stores a portion of a dataset.

    Instance Variables:
    - x (list): The inputs of this partition, as padded batches
    - y (list): The outputs/labels of this partition, as padded batches
    - sizes (list): The sizes of the examples in this partition
    - beginning (boolean): True if this batch contains the beginning of a sequence
    - ending (boolean): True if this batch contains the ending of a sequence
    - num_batches (int): The number of batches in this partition
    """

    def __init__(self, inputs: list, labels: list, sizes: list, num_sequences: int = None):
        """Creates a DataPartition object.

        Params:
        - inputs (list): The inputs in this partition
        - labels (list): The labels in this partition
        - sizes (list): The size of each input in this partition
        - num_sequences (int): The total number of sequences in this partition
        """
        self.x = inputs
        self.y = labels
        self.sizes = [size_of_batch[1:-1] for size_of_batch in sizes]
        self.beginning = [size_of_batch[0] for size_of_batch in sizes]
        self.ending = [size_of_batch[-1] for size_of_batch in sizes]
        self.num_batches = len(self.sizes)
        self.num_sequences = num_sequences
    # End of __init__()

    def get_batch(self, batch_num: int) -> tuple:
        """Obtains the inputs, labels and sizes for a particular batch in the partition.

        Params:
        - batch_num (int): The index of the requested batch

        Return:
        - inputs (list): The inputs for the requested batch
        - labels (list): The labels for the requested batch
        - sizes (list): The sizes for the requested batch
        """
        x = self.x[batch_num]
        y = self.y[batch_num]
        sizes = self.sizes[batch_num]
        return x, y, sizes
    # End of get_batch()
# End of DataPartition()


class DatasetBase(object):
    """Base Dataset class.

    Instance variables:
    - General:
      - logger (logger.Logger): The logger for this class
      - num_features (int): The number of features in the dataset
      - shuffle_seed (float): The seed for shuffling the dataset
      - settings (settings.SettingsNamespace): The training settings, containing batch_size and truncate values
    - Info on dataset:
      - data_type (string): The type of data stored in this dataset
      - token_level (string): The level at which the data was tokenized
      - indexer (indexer.Indexer): Converts between indexes and tokens
      - vocabulary_size (int): The total number of tokens in the dataset
      - max_length (int): The length (number of time-steps in) the longest example in the dataset
    """

    def __init__(self, logger: Logger, rnn_settings: SettingsNamespace, train_settings: SettingsNamespace):
        """Creates a Batches object.

        Params:
        - logger (logger.Logger): The logger to be used by this class
        - rnn_settings (settings.SettingsNamespace): The settings containing number of features and shuffle seed
        - train_settings (settings.SettingsNamespace): The settings containing truncate and batch_size values
        """
        self.logger = logger
        self.settings = train_settings
        self.num_features = rnn_settings.num_features
        self.shuffle_seed = rnn_settings.shuffle_seed
    # End of __init__()

    @debug()
    def load_dataset(self, dataset_name: str) -> tuple:
        """Loads the specified dataset. Instantiates variables for the class.
        Creates the following fields:

        Params:
        - dataset_name (string): The name of the dataset to load

        Return:
        - inputs (list): The list of inputs from the dataset
        - labels (list): The list of outputs from the dataset
        """
        dataset_params = dataset_utils.load_dataset(self.logger, dataset_name)
        self.data_type = dataset_params[0]
        self.token_level = dataset_params[1]
        # Skip vocabulary - we don't really need it
        index_to_token = dataset_params[3]
        token_to_index = dataset_params[4]
        self.indexer = indexer.Indexer(self.num_features, index_to_token, token_to_index)
        self.vocabulary_size = self.extract_vocabulary_size(index_to_token)
        self.max_length = self.longest_example(dataset_params[6])
        return dataset_params[5], dataset_params[6]
    # End of load_dataset()

    @debug()
    def extract_vocabulary_size(self, index_to_token: list) -> Any:
        """Finds the size of the vocabulary based on the number of input features in the dataset.

        Params:
        - index_to_token (list): The list of tokens used to convert indexes to tokens

        Returns:
        - vocabulary_size (list or int): The vocabulary size if there is only one feature, or the list of vocabulary
            sizes if there are multiple features
        """
        if self.num_features == 1:
            vocabulary_size = len(index_to_token)
        else:
            vocabulary_size = [len(feature_index_to_token) for feature_index_to_token in index_to_token]
        return vocabulary_size
    # End of extract_vocabulary_size()

    @debug()
    def longest_example(self, labels: list) -> int:
        """Finds the length (number of time-steps) of the longest example in the dataset. This can be done by looking
        at the labels only.

        Params:
        - labels (list): The list of outputs from the dataset

        Return:
        - length (int): The length of the longest example
        """
        longest_length = -1
        for label in labels:
            label_length = len(label)
            if label_length > longest_length:
                longest_length = label_length
        return longest_length
    # End of longest_example()

    @debug()
    def shuffle(self, inputs: list, labels: list) -> tuple:
        """Shuffles the inputs and labels to remove any ordering present in the dataset.
        The inputs and labels are joined together before they are shuffled, to ensure that they still correctly
        correspond to each other.

        Params:
        - inputs (list): The inputs from the dataset
        - labels (list): The labels from the dataset

        Return:
        - shuffled inputs (list): The shuffled inputs
        - shuffled lables (list): The shuffled labels
        """
        dataset = zip(inputs, labels)  # Returns an iterable
        dataset = list(dataset)
        random.shuffle(dataset, lambda: self.shuffle_seed)
        shuffled_inputs, shuffled_labels = ([a for a,b in dataset], [b for a,b in dataset])
        return shuffled_inputs, shuffled_labels
    # End of shuffle()

    @trace()
    def make_partition(self, inputs: list, labels: list, num_sequences: int = None) -> DataPartition:
        """Creates a partition out of the given portion of the dataset. The partition will contain data broken into
        batches.

        Params:
        - inputs (list): The inputs for the partition
        - labels (list): The labels for the partition
        - num_sequences (int): The number of sequences that will be present in the partition

        Return:
        - partition (DataPartition): The partition containing data in batch format
        """
        if constants.END_TOKEN in self.indexer.token_to_index:
            x_pad_token = constants.END_TOKEN
            y_pad_token = constants.END_TOKEN
        else:
            x_pad_token = inputs[0][-1]
            y_pad_token = labels[0][-1]
        input_batches, label_batches, sizes = batchmaker.make_batches(
            inputs, labels, self.settings.batch_size, self.settings.truncate, x_pad_token, y_pad_token)
        return DataPartition(input_batches, label_batches, sizes, num_sequences)
    # End of make_partition()
# End of DatasetBase


class SimpleDataset(DatasetBase):
    """A simple dataset that uses a 1:2:7 dataset partition. Stores the dataset in batches for easy access.

    @see DatasetBase

    Instance variables:
    - Partitions:
      - test (DataPartition): The partition to be used for testing the performance of the trained model
      - valid (DataPartition): The current partition of the dataset to be used for validation
      - train (DataPartition): The current partition of the dataset to be used for training
    """

    def __init__(self, logger: Logger, rnn_settings: SettingsNamespace, train_settings: SettingsNamespace):
        """Creates a SimpleDataset object.

        Params:
        - logger (logger.Logger): The logger to be used by this class
        - rnn_settings (settings.SettingsNamespace): The settings containing the dataset name, number of features and
            the shuffle seed
        - train_settings (settings.SettingsNamespace): The settings containing truncate and batch_size values
        """
        DatasetBase.__init__(self, logger, rnn_settings, train_settings)
        inputs, labels = self.load_dataset(rnn_settings.dataset)
        self.train, self.valid, self.test = self.extract_partitions(inputs, labels)
    # End of __init__()

    @info('Creating dataset partitions')
    @debug()
    def extract_partitions(self, inputs: list, labels: list) -> tuple:
        """Extracts the training, validation and testing partition from the dataset. The size of the partitions
        follows the following ratio: 70:20:10

        Params:
        - inputs (list): The inputs from the dataset
        - labels (list): The labels from the dataset

        Return:
        - train (DataPartition): The namespace containing the training inputs, labels and sizes
        - valid (DataPartition): The namespace containing the validation inputs, labels and sizes
        - test (DataPartition): The namespace containing the test inputs, labels and sizes
        """
        data = self.shuffle(inputs, labels)
        test_cutoff = math.floor(len(inputs) * 0.1)
        valid_cutoff = math.floor(len(inputs) * 0.3)
        test_data = data[0][:test_cutoff], data[1][:test_cutoff]
        valid_data = data[0][test_cutoff:valid_cutoff], data[1][test_cutoff:valid_cutoff]
        train_data = data[0][valid_cutoff:], data[1][valid_cutoff:]
        train = self.make_partition(train_data[0], train_data[1], len(train_data))
        valid = self.make_partition(valid_data[0], valid_data[1], len(valid_data))
        test = self.make_partition(test_data[0], test_data[1], len(test_data))
        return train, valid, test
    # End of extract_partitions()
# End of CrossValidationDataset()


class CrossValidationDataset(DatasetBase):
    """A dataset that uses cross-validation. Stores the dataset in batches for easy access.

    @see DatasetBase

    Instance variables:
    - Partitions:
      - test (DataPartition): The partition to be used for testing the performance of the trained model
      - valid (DataPartition): The current partition of the dataset to be used for validation
      - train (DataPartition): The current partition of the dataset to be used for training
    - Cross-validation:
      - current (int): The current section of inputs and labels being used for validation
      - num_sections (int): The total number of sections being used for cross-validation
    """

    def __init__(self, logger: Logger, rnn_settings: SettingsNamespace, train_settings: SettingsNamespace):
        """Creates a CrossValidationDataset object.

        Params:
        - logger (logger.Logger): The logger to be used by this class
        - rnn_settings (settings.SettingsNamespace): The settings containing dataset name, number of features, and
            shuffle seed
        - train_settings (settings.SettingsNamespace): The settings containing truncate and batch_size values
        """
        DatasetBase.__init__(self, logger, rnn_settings, train_settings)
        inputs, labels = self.load_dataset(rnn_settings.dataset)
        self.inputs, self.labels, self.test = self.extract_test_partition(inputs, labels)
        # Instantiate cross-validation parameters
        # The section of the training data that is currently being used as the validation set
        # Initialized to -1 so that the first call to next_iteration() starts the cross-validation loop
        self.current = -1
        self.num_sections = 10  # The total number of sections the dataset will be broken into
    # End of __init__()

    @info('Creating test data partition')
    @debug()
    def extract_test_partition(self, inputs: list, labels: list) -> tuple:
        """Samples 10 percent of inputs and labels as test data. The inputs and labels are first shuffled to make sure
        that test data does not follow any original ordering.

        Params:
        - inputs (list): The inputs from the dataset
        - labels (list): The labels from the dataset

        Return:
        - inputs (list): The inputs to be used for training
        - labels (list): The labels to be used for training
        - test (DataPartition): The namespace containing the test inputs, labels and sizes
        """
        self.logger.info('Creating test data partition')
        inputs, labels = self.shuffle(inputs, labels)
        test_cutoff = math.floor(len(inputs) * 0.1)
        test_inputs = inputs[:test_cutoff]
        test_labels = labels[:test_cutoff]
        test = self.make_partition(test_inputs, test_labels, len(test_labels))
        return inputs[test_cutoff:], labels[test_cutoff:], test
    # End of extract_test_partition()

    @info('Creating validation and training data partitions')
    def extract_validation_partition(self):
        """Extracts a section of the training data to use as a validation set, and assigns the rest of it to the
        training set.
        """
        section_length = math.floor(len(self.inputs) / self.num_sections)
        valid_start = self.current * section_length
        valid_end = valid_start + section_length
        self.valid = self.make_partition(self.inputs[valid_start:valid_end], self.labels[valid_start:valid_end])
        train_x = self.inputs[:valid_start] + self.inputs[valid_end:]
        train_y = self.labels[:valid_start] + self.labels[valid_end:]
        self.train = self.make_partition(train_x, train_y)
    # End of extract_validation_partition()

    @trace()
    def next_iteration(self):
        """Increments the current section number, and extracts the validation set corresponding to that section
        number from the dataset.
        """
        self.logger.debug('Updating validation set to next cross-validation section')
        self.current += 1
        self.current = self.current % self.num_sections
        self.extract_validation_partition()
    # End of next_iteration()
# End of CrossValidationDataset()