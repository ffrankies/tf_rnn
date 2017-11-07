"""
An object for storing a dataset for training.

Date: 7 November 2017
"""
import random
import math

from . import dataset_utils
from . import batchmaker
from . import constants

class DataPartition(object):
    """
    Stores a portion of a dataset.

    Instance Variables:
    x (list): The inputs of this partition, as padded batches
    y (list): The outputs/labels of this partition, as padded batches
    sizes (list): The sizes of the examples in this partition
    beginning (boolean): True if this batch contains the beginning of a sequence
    ending (boolean): True if this batch contains the ending of a sequence
    num_batches (int): The number of batches in this partition
    """

    def __init__(self, inputs, labels, sizes, num_sequences=None):
        """
        Creates a DataPartition object.

        Params:
        inputs (list): The inputs in this partition
        labels (list): The labels in this partition
        sizes (list): the size of each input in this partition
        """
        self.x = inputs
        self.y = labels
        self.sizes = [size_of_batch[1:-1] for size_of_batch in sizes]
        self.beginning = [size_of_batch[0] for size_of_batch in sizes]
        self.ending = [size_of_batch[-1] for size_of_batch in sizes]
        self.num_batches = len(self.sizes)
        self.num_sequences = num_sequences
    # End of __init__()

    def get_batch(self, batch_num):
        """
        Obtains the inputs, labels and sizes for a particular batch in the partition.

        Params:
        batch_num (int): The index of the requested batch

        Return:
        inputs (list): The inputs for the requested batch
        labels (list): The labels for the requested batch
        sizes (list): The sizes for the requested batch
        metadata (pair): Metadata showing whether this batch is the start and/or end of a sequence
        """
        x = self.x[batch_num]
        y = self.y[batch_num]
        sizes = self.sizes[batch_num]
        return x, y, sizes
    # End of get_batch()
# End of DataPartition()

class Dataset(object):
    """
    Stores the dataset in batches for easy access.

    Instance variables:
    - General:
        - logger (logging.Logger): The logger for this class
        - settings (settings.SettingsNamespace): The training settings, containing batch_size and truncate values
    - Data:
        - inputs (list): The inputs to be used for training and validation
        - labels (list): The labels to be used for training and validation
    - Partitions:
        - test (DataPartition): The partition to be used for testing the performance of the trained model
        - valid (DataPartition): The current partition of the dataset to be used for validation
        - train (DataPartition): The current partition of the dataset to be used for training
    - Cross-validation:
        - current (int): The current section of inputs and labels being used for validation
        - num_sections (int): The total number of sections being used for cross-validation
    - Info on dataset:
        - data_type (string): The type of data stored in this dataset
        - token_level (string): The level at which the data was tokenized
        - index_to_token (list): Converts an index to a token
        - token_to_index (dict): Converts a token to an index
        - vocabulary_size (int): The total number of tokens in the dataset
        - max_length (int): The length (number of time-steps in) the longest example in the dataset
    """

    def __init__(self, logger, dataset_name, train_settings):
        """
        Creates a Batches object.

        Params:
        logger (logging.Logger): The logger to be used by this class
        dataset_name (string): The name of the dataset to load
        train_settings (settings.SettingsNamespace): The settings containing truncate and batch_size values
        """
        self.logger = logger
        self.settings = train_settings
        inputs, labels = self.load_dataset(dataset_name)
        self.inputs, self.labels, self.test = self.extract_test_partition(inputs, labels)
        # Instantiate cross-validation parameters
        self.current = -1; # The section of the training data that is currently being used as the validation set
                           # Initialized to -1 so that the first call to next_iteration() starts the cross-validation
                           # loop.
        self.num_sections = 10; # The total number of sections the dataset will be broken into
    # End of __init__()

    def load_dataset(self, dataset_name):
        """
        Loads the specified dataset. Instantiates variables for the class.
        Creates the following fields:
            - data_type (string): The type of data stored in this dataset
            - token_level (string): The level at which the data was tokenized
            - index_to_token (list): Converts an index to a token
            - token_to_index (dict): Converts a token to an index
            - vocabulary_size (int): The total number of tokens in the dataset
            - max_length (int): The length (number of time-steps in) the longest example in the dataset

        Params:
        dataset_name (string): The name of the dataset to load

        Return:
        inputs (list): The list of inputs from the dataset
        labels (list): The list of outputs from the dataset
        """
        dataset_params = dataset_utils.load_dataset(self.logger, dataset_name)
        self.data_type = dataset_params[0]
        self.token_level = dataset_params[1]
        # Skip vocabulary - we don't really need it
        self.index_to_token = dataset_params[3]
        self.token_to_index = dataset_params[4]
        self.vocabulary_size = len(self.index_to_token)
        self.max_length = self.longest_example(dataset_params[6])
        return dataset_params[5], dataset_params[6]
    # End of load_dataset()

    def longest_example(self, labels):
        """
        Finds the length (number of time-steps) of the longest example in the dataset. This can be done by looking
        at the labels only.

        Params:
        labels (list): The list of outputs from the dataset

        Return:
        length (int): The length of the longest example
        """
        self.logger.info("Finding longest example in the dataset")
        longest_length = -1
        for label in labels:
            label_length = len(label)
            if label_length > longest_length:
                longest_length = label_length
        return longest_length
    # End of longest_example()

    def shuffle(self, inputs, labels):
        """
        Shuffles the inputs and labels to remove any ordering present in the dataset.
        The inputs and labels are joined together before they are shuffled, to ensure that they still correctly
        correspond to eacfrom random import shuffleh other.

        Params:
        inputs (list): The inputs from the dataset
        labels (list): The labels from the dataset

        Return:
        shuffled inputs (list): The shuffled inputs
        shuffled lables (list): The shuffled labels
        """
        self.logger.info("Shuffling dataset")
        random.seed(None) # Seed with system time / OS-specific randomness sources
        dataset = zip(inputs, labels) # Returns an iterable
        dataset = list(dataset)
        random.shuffle(dataset)
        shuffled_inputs, shuffled_labels = ([a for a,b in dataset], [b for a,b in dataset])
        return shuffled_inputs, shuffled_labels
    # End of shuffle()

    def extract_test_partition(self, inputs, labels):
        """
        Samples 10 percent of inputs and labels as test data. The inputs and labels are first shuffled to make sure
        that test data does not follow any original ordering.

        Params:
        inputs (list): The inputs from the dataset
        labels (list): The labels from the dataset

        Return:
        inputs (list): The inputs to be used for training
        labels (list): The labels to be used for training
        test (DataPartition): The namespace containing the test inputs, labels and sizes
        """
        self.logger.info("Creating test data partition")
        inputs, labels = self.shuffle(inputs, labels)
        test_cutoff = math.floor(len(inputs) * 0.1)
        test_inputs = inputs[:test_cutoff]
        test_labels = labels[:test_cutoff]
        test = self.make_partition(test_inputs, test_labels, len(test_labels))
        return inputs[test_cutoff:], labels[test_cutoff:], test
    # End of extract_test_partition()

    def extract_validation_partition(self):
        """
        Extracts a section of the training data to use as a validation set, and assigns the rest of it to the
        training set.
        Creates the following fields:
            - valid (DataPartition): The part of the dataset to be used for validation
            - train (DataPartition): The part of the dataset to be used for training
        """
        self.logger.debug("Creating validation data partition")
        section_length = math.floor(len(self.inputs) / self.num_sections)
        valid_start = self.current * section_length
        valid_end = valid_start + section_length
        self.valid = self.make_partition(self.inputs[valid_start:valid_end], self.labels[valid_start:valid_end])
        train_x = self.inputs[:valid_start] + self.inputs[valid_end:]
        train_y = self.labels[:valid_start] + self.labels[valid_end:]
        self.train = self.make_partition(train_x, train_y)
    # End of extract_validation_partition()

    def next_iteration(self):
        """
        Increments the current section number, and extracts the validation set corresponding to that section
        number from the dataset.
        """
        self.logger.debug("Updating validation set to next cross-validation section")
        self.current += 1
        self.current = self.current % self.num_sections
        self.extract_validation_partition()
    # End of next_iteration()

    def make_partition(self, inputs, labels, num_sequences=None):
        """
        Creates a partition out of the given portion of the dataset. The partition will contain data broken into
        batches.

        Params:
        inputs (list): The inputs for the partition
        labels (list): The labels for the partition

        Return:
        partition (DataPartition): The partition containing data in batch format
        """
        x, y, sizes = batchmaker.make_batches(inputs, labels, self.settings.batch_size, self.settings.truncate,
                self.token_to_index[constants.END_TOKEN])
        return DataPartition(x, y, sizes, num_sequences)
    # End of make_partition()
# End of Batches()
