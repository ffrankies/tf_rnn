"""
An object for storing a dataset for training.

Date: 25 October 2017
"""
import random
import math

from . import dataset_utils
from . import settings
from . import batchmaker
from . import constants

class Dataset(object):
    """
    Stores the dataset in batches for easy access.
    """

    def __init__(self, logger, dataset_name, train_settings):
        """
        Creates a Batches object.

        Params:
        logger (logging.Logger): The loreturn self.shuffle(dataset_params[5], dataset_params[6])gger to be used by this classreturn self.shuffle(dataset_params[5], dataset_params[6])
        dataset_name (string): The name of the dataset to load
        train_settings (settings.Settings): The settings containing truncate and batch_size values
        """
        self.logger = logger
        inputs, labels = self.load_dataset(dataset_name, train_settings)
        self.inputs, self.labels = self.extract_test_data(inputs, labels, train_settings)
        # Instantiate cross-validation parameters
        self.current = -1; # The numberreturn self.shuffle(dataset_params[5], dataset_params[6]) of the current section used for validation
        self.k = 10; # The total number of sections the dataset will be broken into
    # End of __init__()

    def load_dataset(self, dataset_name):
        """
        Loads the specified dataset. Instantiates variables for the class.
        Creates the following fields:
            - data_type
            - token_level
            - index_to_token
            - index_to_index
            - vocabulary_size
            - inputs
            - labels

        Params:
        dataset_name (string): The name of the dataset to load

        Return:
        inputs (list): The list of inputs from the dataset
        labels (list): The list of outputs from the dataset
        """
        dataset_params = dataset_utils.load_dataset(self.logger, self.settings.rnn.dataset)
        self.data_type = dataset_params[0]
        self.token_level = dataset_params[1]
        # Skip vocabulary - we don't really need it
        self.index_to_token = dataset_params[3]
        self.token_to_index = dataset_params[4]
        self.vocabulary_size = len(self.index_to_token)
        return dataset_params[5], dataset_params[6]
    # End of load_dataset()

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
        shuffled_inputs, shuffled_lables = ([a for a,b in dataset], [b for a,b in dataset])
        return shuffled_inputs, shuffled_labels
    # End of shuffle()

    def extract_test_data(self, inputs, labels, train_settings):
        """
        Samples 10 percent of inputs and labels as test data. The inputs and labels are first shuffled to make sure that
        test data does not follow any original ordering.

        Params:
        inputs (list): The inputs from the dataset
        labels (list): The labels from the dataset
        train_settings (settings.Settings): The settings containing truncate and batch_size values
        """
        self.logger.info("Creating test data")
        inputs, labels = self.shuffle(inputs, labels)
        test_cutoff = math.floor(len(inputs) * 0.1)
        test_inputs = inputs[:test_cutoff]
        test_labels = labels[:test_cutoff]
        x, y, sizes = batchmaker.make_batches(test_inputs, test_labels, train_settings.batch_size,
                train_settings.truncate, self.token_to_index[constants.END_TOKEN])
        self.test = settings.SettingsNamespace({"x" : x, "y" : y, "sizes" : sizes})
        return inputs[test_cutoff:], labels[test_cutoff:]
    # End of extract_test_data()

    def extract_validation_set(self):
        """
        """
        return None

    def next_iteration(self):
        """
        """
        return None
# End of Batches()
