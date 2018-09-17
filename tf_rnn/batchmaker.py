"""Contains functions that handle the shaping of input into batches.
The batches are converted into numpy arrays towards the end for them to play nice with tensorflow
(i.e. avoid the "ValueError: setting an array element with a sequence" error)

@since 0.6.2
"""
import math
import multiprocessing
from collections import namedtuple
from typing import Any

import numpy as np

from .logger import trace


TRUNCATE_LENGTH = None
X_PAD_TOKEN = None
Y_PAD_TOKEN = None


Batch = namedtuple('Batch', ['x', 'y', 'sequence_lengths', 'beginning', 'ending'])


@trace()
def make_batches(input_data: list, labels: list, batch_size: int, truncate_length: int, x_pad_token: Any,
                 y_pad_token: Any) -> tuple:
    """Converts both the input data and labels into batches of size 'batch_size'.

    Params:
    - input_data (list): The input data to be made into batches
    - labels (list): The labels to be made into batches
    - batch_size (int): The size of the batches
    - truncate_length (int): The maximum width or length of the batches
    - x_pad_token (list or int): The token with which to pad the input data batches
    - y_pad_token (list or int): The token with which to pad the label data batches

    Returns:
    - batches (list<Batch>): The list of Batches produced
    """
    global TRUNCATE_LENGTH, X_PAD_TOKEN, Y_PAD_TOKEN
    TRUNCATE_LENGTH = truncate_length
    X_PAD_TOKEN = x_pad_token
    Y_PAD_TOKEN = y_pad_token
    data = sort_by_length([input_data, labels])
    data = group_into_batches(data, batch_size)
    batch_data = zip(data[0], data[1])  # create tuples of input, label data pairs
    thread_pool = multiprocessing.Pool(processes=5)
    batches = thread_pool.starmap(process_batch, batch_data)
    batches = combine_batch_lists(batches)
    return batches
# End of make_batches()


@trace()
def sort_by_length(data: list) -> list:
    """Sorts the provided data objects in descending order by length.

    Params:
    - data (list): The list of data to be sorted

    Returns:
    - sorted_data (list): The data, sorted in descending order by length
    """
    sorted_data = list()
    for item in data:
        sorted_item = sorted(item, key=len, reverse=True)
        sorted_data.append(sorted_item)
    return sorted_data
# End of sort_by_length()


@trace()
def group_into_batches(data: list, batch_size: int) -> list:
    """Group each item in data into batches of size 'batch_size'.

    Params:
    - data (list): The data to be grouped into batches
    - batch_size (int): The size of the data batches

    Returns:
    - batches (list): The data items, in batch form

    Raises:
    - ValueError when the batch size is less than 1
    """
    if batch_size < 1:
        raise ValueError('The size of the batches cannot be less than 1.')
    batched_data = list()
    for item in data:
        batched_data_item = [item[i:i+batch_size] for i in range(0, len(item), batch_size)]
        if len(batched_data_item) is not 0:
            while len(batched_data_item[-1]) < batch_size:
                batched_data_item[-1].append([])
        batched_data.append(batched_data_item)
    return batched_data
# End of group_into_batches()


def process_batch(input_data: list, label_data: list) -> list:
    """Creates a batch out of input and label data.

    Params:
    - input_data (list): The input data for the batch
    - label_data (list): The label data for the batch

    Returns:
    - batches (list<Batch>): The list of truncated batches created out of the given data
    """
    global X_PAD_TOKEN, Y_PAD_TOKEN
    batch_input = truncate_batch(input_data)
    batch_labels = truncate_batch(label_data)
    sequence_lengths = get_sequence_lengths(batch_labels)[:3]
    batch_input = pad_batches(batch_input, X_PAD_TOKEN)
    batch_labels = pad_batches(batch_labels, Y_PAD_TOKEN)
    batches = list()
    for index in range(len(batch_input)):
        batch = Batch(batch_input[index], batch_labels[index], sequence_lengths[index][1:-1], 
                      sequence_lengths[index][0], sequence_lengths[index][-1])
        batches.append(batch)
    return batches
# End of process_batch()


def truncate_batch(batch_data: list) -> list:
    """Truncates each sequence in the given batch. This may result in multiple batches.

    Params:
    - batch_data (list<list<Any>>): The batch data to truncate.

    Returns:
    - truncated_batch (list<list<Any>>): The truncated batch
    """
    global TRUNCATE_LENGTH
    max_length = max(map(len, batch_data))
    truncated_batches = list()
    times_to_truncate = math.ceil(max_length / TRUNCATE_LENGTH)
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
    - index (int): The index of the resulting batch partition (dictates at what point in the batch the truncation
        starts)
    - last_index (int): The index of the last batch partition
    - batch (list): The full-length batch on which to perform the truncation

    Returns:
    - truncated_batch_section (list): The truncated section of the batch
    """
    global TRUNCATE_LENGTH
    start = index * TRUNCATE_LENGTH
    end = start + TRUNCATE_LENGTH
    beginning = [True] if index == 0 else [False]
    ending = [True] if index == last_index else [False]
    truncated_batch_section = [example[start:end] for example in batch]
    truncated_batch_section = beginning + truncated_batch_section + ending
    return truncated_batch_section
# End of _truncate_batch()


def get_sequence_lengths(data: list) -> list:
    """Returns the lengths of every row in the given data. The data must be arranged in batches or the function will
    fail.

    Params:
    - data (list): The list of data (arranged in batches) whose length is to be found

    Returns:
    - timestep_lengths (list): The lengths of the sequences of every batch in the given data
    """
    batch_lengths = list()
    for batch in data:
        item_row_lengths = [batch[0]]
        item_row_lengths += [len(row) for row in batch[1:-1]]
        item_row_lengths += [batch[-1]]
        batch_lengths.append(item_row_lengths)
    return batch_lengths
# End of get_sequence_lengths()


def pad_batches(data: list, pad_token: Any) -> np.array:
    """Pads every row in the data batches so that it's the same length as the value of 'truncate_length'.

    Params:
    - data (list): The batches of data to be padded
    - pad_token (list or int): The token with which to pad the input data batches

    Returns:
    - padded_batches (np.array): The padded batches of data
    """
    padded_batches = list()
    pad_token = get_pad_token(data, pad_token)
    for batch in data:
        padded_batch = list()
        for sequence in batch[1:-1]:
            padded_sequence = pad_sequence(sequence, pad_token)
            padded_batch.append(padded_sequence)
        padded_batches.append(np.array(padded_batch))
    return np.array(padded_batches)
# End of pad_batches()


def get_pad_token(data: list, pad_token: Any) -> Any:
    """Determines the correct padding token. If there are multiple inputs, the pad token is replicated for each feature.

    Params:
    - data (list): The data to be padded
    - pad_token (numeric): The padding token, to be replicated if there are multiple inputs

    Returns:
    - pad_token (numeric/list): The padding token, replicated if there are multiple inputs
    """
    if not data or not data[0] or not data[0][1]:
        return data
    data_item = data[0][1][0]
    if isinstance(data_item, (tuple, list)):
        if not isinstance(pad_token, (tuple, list)):
            pad_token = [pad_token for item in data_item]
    return pad_token
# End of get_pad_token()


def pad_sequence(sequence: list, pad_token: Any) -> np.array:
    """Pads a single sequence with the given pad token.

    Params:
    - sequence (list): The sequence to pad
    - pad_token (numeric/list): The token to be used for padding

    Return:
    - padded_sequence (np.array): The padded sequence
    """
    global TRUNCATE_LENGTH
    padded_sequence = list(sequence)
    while len(padded_sequence) < TRUNCATE_LENGTH:
        padded_sequence.append(pad_token)
    return np.array(padded_sequence)
# End of pad_sequence()


def combine_batch_lists(batch_lists: list) -> np.array:
    """Combines lists of batches produced my the multiprocessing module into a single batch list.

    Params:
    - batch_lists (list<list<Batch>>): The list of lists of batches

    Returns:
    - batches (list<Batch>): The combined list of batches
    """
    batches = list()
    for batch_list in batch_lists:
        batches.extend(batch_list)
    return batches
# End of combine_batch_lists()
