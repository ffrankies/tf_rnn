"""Contains functions that handle the shaping of input into batches.
The batches are converted into numpy arrays towards the end for them to play nice with tensorflow
(i.e. avoid the "ValueError: setting an array element with a sequence" error)

@since 0.6.2
"""
import math
import multiprocessing
from collections import namedtuple
from typing import Any, Optional, List
from queue import Queue

import numpy as np

from .logger import trace
from .utils import Singleton


TRUNCATE_LENGTH = None
X_PAD_TOKEN = None
Y_PAD_TOKEN = None


Batch = namedtuple('Batch', ['x', 'y', 'sequence_lengths', 'beginning', 'ending'])


class BatchConstants(object, metaclass=Singleton):
    """Contains the batch constants.
    """

    def __init__(self, truncate: Optional[int] = None, input_pad: Optional[Any] = None, 
        label_pad: Optional[Any] = None) -> None:
        """Initializes a BatchConstants object.

        Params:
            truncate (int): The length to which to truncate batches
            input_pad (Any): The input padding
            label_pad (Any): The label padding
        """
        if truncate:  # If parameters are provided
            if truncate < 1:
                raise ValueError('The value of truncate cannot be less than one')
            self.truncate = truncate
            self.input_pad = input_pad
            self.label_pad = label_pad
    # End of __init__()

    def __str__(self) -> str:
        """String representation of BatchConstants.
        """
        return str(self.__dict__)
# End of BatchConstants()


@trace()
def make_batches(input_data: list, labels: list, batch_size: int, truncate_length: int, x_pad_token: Any,
                 y_pad_token: Any) -> List[Batch]:
    """Converts both the input data and labels into batches of size 'batch_size'.

    Params:
        input_data (list): The input data to be made into batches
        labels (list): The labels to be made into batches
        batch_size (int): The size of the batches
        truncate_length (int): The maximum width or length of the batches
        x_pad_token (list or int): The token with which to pad the input data batches
        y_pad_token (list or int): The token with which to pad the label data batches

    Returns:
        List[Batch]: The list of Batches produced
    """
    BatchConstants(truncate_length, x_pad_token, y_pad_token)
    data = sort_by_length([input_data, labels])
    manager = multiprocessing.Manager()
    batch_queue = group_into_batches(data, batch_size, manager)
    processed_batch_queue = manager.Queue()
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() * 2) as thread_pool:
        thread_pool.apply(process_batch, (batch_queue, processed_batch_queue))
    batches = combine_batch_lists(processed_batch_queue)
    return batches
# End of make_batches()


@trace()
def sort_by_length(data: list) -> list:
    """Sorts the provided data objects in descending order by length.

    Params:
        data (List[List[Any]]): The list of data to be sorted

    Returns:
        List[List[Any]]: The data, sorted in descending order by length
    """
    sorted_data = list()
    for item in data:
        sorted_item = sorted(item, key=len, reverse=True)
        sorted_data.append(sorted_item)
    return sorted_data
# End of sort_by_length()


@trace()
def group_into_batches(data: list, batch_size: int, manager: multiprocessing.Manager) -> multiprocessing.Queue:
    """Group each item in data into batches of size 'batch_size'.

    Params:
        data (List[List[List[numeric]]]): The data to be grouped into batches
        batch_size (int): The size of the data batches
        manager (multiprocessing.Manager): The manager from which to create a Queue

    Returns:
        multiprocessing.Queue: A managed Queue holding batches to be processed

    Raises:
        ValueError when the batch size is less than 1
    """
    if batch_size < 1:
        raise ValueError('The size of the batches cannot be less than 1.')
    if not data or len(data) != 2:
        raise ValueError('Must provide some input data and labels.')
    batch_queue = manager.Queue()  # type: ignore
    inputs, labels = data
    for index in range(0, len(inputs), batch_size):
        batch_inputs = inputs[index:index+batch_size]
        batch_labels = labels[index:index+batch_size]
        while len(batch_inputs) < batch_size:  # Pad last batch if there aren't enough sequences
            batch_inputs.append([])
            batch_labels.append([])
        batch_queue.put((batch_inputs, batch_labels))
    return batch_queue
# End of group_into_batches()


def process_batch(batch_queue: multiprocessing.Queue, processed_batch_queue: multiprocessing.Queue):
    """Creates a batch out of input and label data.

    Params:
        batch_queue (multiprocessing.Queue): The managed Queue containing the batches to be processed
        processed_batch_queue (multiprocessing.Queue): The managed Queue into which to put processed batches
    """
    batch_constants = BatchConstants()
    while not batch_queue.empty():
        input_data, label_data = batch_queue.get()
        batch_input = truncate_batch(input_data)
        batch_labels = truncate_batch(label_data)
        sequence_lengths = get_sequence_lengths(batch_labels)[:3]
        batch_input = pad_batches(batch_input, batch_constants.input_pad)
        batch_labels = pad_batches(batch_labels, batch_constants.label_pad)
        batch_input = reshape_batches(batch_input)
        batch_labels = reshape_batches(batch_labels)
        for index in range(len(batch_input)):
            batch = Batch(batch_input[index], batch_labels[index], sequence_lengths[index][1:-1], 
                        sequence_lengths[index][0], sequence_lengths[index][-1])
            processed_batch_queue.put(batch)
# End of process_batch()


def truncate_batch(batch_data: List[List[Any]]) -> list:
    """Truncates each sequence in the given batch. This may result in multiple batches.

    Params:
        batch_data (List[List[Any]]): The batch data to truncate.

    Returns:
        List[List[Any]]: The truncated batch
    """
    batch_constants = BatchConstants()
    if batch_constants.truncate < 1:
        raise ValueError('Truncate length cannot be less than 1')
    max_length = max(map(len, batch_data))
    truncated_batches = list()
    times_to_truncate = math.ceil(max_length / batch_constants.truncate)
    for i in range(times_to_truncate):
        truncated_batch = _truncate_batch(i, times_to_truncate-1, batch_data)
        truncated_batches.append(truncated_batch)
    return truncated_batches
# End of truncate_batch()


def _truncate_batch(index: int, last_index: int, batch: list) -> list:
    """Truncates each example in the batch, so that no row is longer than 'truncate_length' values long.
    Also surrounds the batch with two boolean indicators:
    - The indicator at the start of the batch is true if it is the beginning of the example sequence
    - The indicator at the end of the batch is true if it is the ending of the example sequence

    Params:
        index (int): The index of the resulting batch partition (dictates at what point in the batch the truncation
        starts)
        last_index (int): The index of the last batch partition
        batch (list): The full-length batch on which to perform the truncation

    Returns:
        List[Tuple[bool, List[List[Any]], bool]]: The truncated section of the batch
    """
    batch_constants = BatchConstants()
    start = index * batch_constants.truncate
    end = start + batch_constants.truncate
    beginning = [True] if index == 0 else [False]
    ending = [True] if index == last_index else [False]
    truncated_batch_section = [example[start:end] for example in batch]
    truncated_batch_section = beginning + truncated_batch_section + ending
    return truncated_batch_section
# End of _truncate_batch()


def get_sequence_lengths(data: list) -> List[List[int]]:
    """Returns the lengths of every row in the given data. The data must be arranged in batches or the function will
    fail.

    Params:
        data (list): The list of data (arranged in batches) whose length is to be found

    Returns:
        timestep_lengths (List[List[int]]): The lengths of the sequences of every batch in the given data
    """
    batch_lengths = list()
    for batch in data:
        item_row_lengths = [batch[0]]
        item_row_lengths += [len(row) for row in batch[1:-1]]
        item_row_lengths += [batch[-1]]
        batch_lengths.append(item_row_lengths)
    return batch_lengths
# End of get_sequence_lengths()


def pad_batches(data: np.array, pad_token: Any) -> np.array:
    """Pads every row in the data batches so that it's the same length as the value of 'truncate_length'.

    Params:
        data (np.array): The batches of data to be padded
        pad_token (list or int): The token with which to pad the input data batches

    Returns:
        padded_batches (np.array): The padded batches of data
    """
    padded_batches = list()
    pad_token = get_pad_token(data, pad_token)
    for batch in data:
        padded_batch = list()
        for sequence in batch[1:-1]:
            padded_sequence = pad_sequence(sequence, pad_token)
            padded_batch.append(padded_sequence)
        padded_batches.append(padded_batch)
    return np.array(padded_batches, dtype=data[0][1].dtype)
# End of pad_batches()


def reshape_batches(data: np.array) -> np.array:
    """Tensorflow needs simple numpy arrays, while data is a structured array. This reshapes the array into a 
    simpler format, where everything is cast to a float datatype.

    Params:
        data (np.array): Numpy structured array
    
    Returns:
        np.array: Array of Numpy array of size (num_features, batch_size..)
    """
    reshaped_batches = list()
    for batch in data:
        reshaped_batch = list()
        for feature in batch.dtype.names:
            reshaped_batch.append(batch[feature].astype(np.float32))
        reshaped_batches.append(np.asarray(reshaped_batch))
    np_reshaped_batches = np.asarray(reshaped_batches, dtype=np.float32)
    return np.asarray(reshaped_batches)
# End of reshape_batches()


def get_pad_token(batch: list, pad_token: Any) -> Any:
    """Determines the correct padding token. If there are multiple inputs, the pad token is replicated for each feature.

    Params:
        batch (list): The data to be padded
        pad_token (numeric): The padding token, to be replicated if there are multiple inputs

    Returns:
        pad_token (numeric/list): The padding token, replicated if there are multiple inputs
    """
    if not batch or not batch[0] or batch[0][1].size == 0:
        return batch
    data_item = batch[0][1][0]
    if isinstance(data_item, (tuple, list)):
        if not isinstance(pad_token, (tuple, list)):
            pad_token = [pad_token for item in data_item]
    return pad_token
# End of get_pad_token()


def pad_sequence(sequence: list, pad_token: Any) -> np.array:
    """Pads a single sequence with the given pad token.

    Params:
        sequence (list): The sequence to pad
        pad_token (numeric/list): The token to be used for padding

    Return:
        np.array: The padded sequence
    """
    batch_constants = BatchConstants()
    padded_sequence = list(sequence)
    while len(padded_sequence) < batch_constants.truncate:
        padded_sequence.append(pad_token)
    return np.array(padded_sequence)
# End of pad_sequence()


def combine_batch_lists(processed_batch_queue: Queue) -> List[Batch]:
    """Combines processed batches produced my the multiprocessing module into a single batch list.

    Params:
        processed_batch_queue (queue.Queue): The Queue of processed batches, provided by the multiprocessing Manager

    Returns:
        List[Batch]: The combined list of batches
    """
    batches = list()
    while not processed_batch_queue.empty():
        batches.append(processed_batch_queue.get())
    return batches
# End of combine_batch_lists()
