"""An object for storing a dataset for training.

@since 0.6.2
"""
import random
import math
from typing import Any

import dill

from . import dataset_utils
from . import batchmaker
from . import constants
from . import indexer
from .logger import info, debug, trace
from .utils import create_directory

# These variables are only imported for type hinting
from io import TextIOWrapper
from .logger import Logger
from .settings import SettingsNamespace


class DataPartition(object):
    """Contains references to a dataset partition. To prevent loading large files into memory, the partition only
    provides sequential access to the batches, and loads a single batch at a time.
    """

    def __init__(self, batches: list, path: str, num_sequences: int = None):
        """Creates a DataPartition object.

        Params:
        - batches (list<Batch>): The list of batches that make up the partition
        - path (str): The path to the dataset partition file
        - num_sequences (int): The total number of sequences in this partition
        """
        self.num_batches = len(batches)
        self.num_sequences = num_sequences
        self.path = path
        self.index = 0
        self._file_handler = None
        self._save_partition(batches)
    # End of __init__()

    def _save_partition(self, batches: list):
        """Saves the partition's batches as individual objects into a file.

        Params:
        - batches (list<Batch>): The list of batches that make up the partition
        """
        with open(self.path, 'wb') as partition_file:
            for batch in batches:
                dill.dump(batch, partition_file, protocol=dill.HIGHEST_PROTOCOL) 
    # End of _save_partition()

    def __iter__(self):
        """Turns this class into an Iterable object.
        """
        return self
    # End of __iter__()

    def __next__(self):
        """The 'iterating' method for this class.

        Returns:
        - batch (Batch): The next batch in the partition
        """
        if self.index == 0:
            self._file_handler = open(self.path, 'rb')
        if self.index == self.num_batches:
            self.index = 0
            self._file_handler.close()
            raise StopIteration
        batch = self.next_batch()
        self.index += 1
        return batch
    # End of __next__()

    def next_batch(self) -> tuple:
        """Loads the next batch from the opened partition file using dill.

        Return:
        - batch (Batch): The next batch in the partition
        """
        batch = dill.load(self._file_handler)
        return batch
    # End of next_batch()
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

    def __init__(self, logger: Logger, train_settings: SettingsNamespace, model_path: str):
        """Creates a Batches object.

        Params:
        - logger (logger.Logger): The logger to be used by this class
        - train_settings (settings.SettingsNamespace): The settings containing truncate and batch_size values
        - model_path (str): The path to the model directory
        """
        self.logger = logger
        self.settings = train_settings
        self.data_path = model_path + constants.MODEL_DATA_DIR
        create_directory(self.data_path)
    # End of __init__()

    @debug()
    def load_dataset(self, dataset_name: str) -> tuple:
        """Loads the specified dataset. Instantiates variables for the class.

        Params:
        - dataset_name (string): The name of the dataset to load

        Return:
        - dataset_file (TextIOWrapper): The opened file pointer to the dataset file
        """
        self.logger.info('Loading saved dataset info')
        dataset_path = constants.DATASETS_DIR + dataset_name
        dataset_file = open(dataset_path, 'rb')
        meta_info = dill.load(dataset_file)
        self._set_meta(meta_info)
        return dataset_file
    # End of load_dataset()

    def _set_meta(self, meta_info: tuple):
        """Sets the meta parameters of this class from the loaded dataset.

        Params:
        - meta_info (tuple<Any>): The meta info for the loaded dataset
        """
        self.data_type = meta_info[0]
        self.token_level = meta_info[1]
        self.num_features = meta_info[2]
        index_to_token = meta_info[4]
        token_to_index = meta_info[5]
        self.indexer = indexer.Indexer(self.num_features, index_to_token, token_to_index)
        self.vocabulary_size = self.extract_vocabulary_size(index_to_token)
        self.max_length = meta_info[6]
    # End of _set_meta()

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

    @trace()
    def make_partition(self, partition_data: tuple, path: str) -> DataPartition:
        """Creates a partition out of the given portion of the dataset. The partition will contain data broken into
        batches.

        Params:
        - partition_data (tuple<list<Any>,list<Any>>): The partition inputs and labels
        - path (str): The path to the partition file

        Return:
        - partition (DataPartition): The partition containing data in batch format
        """
        inputs = partition_data[0]
        labels = partition_data[1]
        num_sequences = len(inputs)
        if constants.END_TOKEN in self.indexer.token_to_index:
            x_pad_token = self.indexer.to_index(constants.END_TOKEN)
            y_pad_token = self.indexer.to_index(constants.END_TOKEN)
        else:
            x_pad_token = inputs[0][-1]
            y_pad_token = labels[0][-1]
        batches = batchmaker.make_batches(inputs, labels, self.settings.batch_size, self.settings.truncate,
                                          x_pad_token, y_pad_token)
        return DataPartition(batches, path, num_sequences)
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

    def __init__(self, logger: Logger, train_settings: SettingsNamespace, model_path: str, dataset: str):
        """Creates a SimpleDataset object.

        Params:
        - logger (logger.Logger): The logger to be used by this class
        - train_settings (settings.SettingsNamespace): The settings containing truncate and batch_size values
        - model_path (str): The path to the model directory
        """
        DatasetBase.__init__(self, logger, train_settings, model_path)
        dataset_file = self.load_dataset(dataset)
        self.train, self.valid, self.test = self.extract_partitions(dataset_file)
    # End of __init__()

    @info('Creating dataset partitions')
    @debug()
    def extract_partitions(self, dataset_file: TextIOWrapper) -> tuple:
        """Extracts the training, validation and testing partition from the dataset. The size of the partitions
        follows the following ratio: 70:20:10

        Params:
        - dataset_file (TextIOWrapper): The file pointer to the opened dataset file

        Return:
        - train (DataPartition): The namespace containing the training inputs, labels and sizes
        - valid (DataPartition): The namespace containing the validation inputs, labels and sizes
        - test (DataPartition): The namespace containing the test inputs, labels and sizes
        """
        partition_data = dill.load(dataset_file)  # test partition
        test = self.make_partition(partition_data, self.data_path + constants.PART_TEST)
        partition_data = dill.load(dataset_file)  # validation partition
        valid = self.make_partition(partition_data, self.data_path + constants.PART_VALID)
        partition_data = dill.load(dataset_file)  # training partition
        train = self.make_partition(partition_data, self.data_path + constants.PART_TRAIN)
        dataset_file.close()
        return train, valid, test
    # End of extract_partitions()
# End of SimpleDataset()
