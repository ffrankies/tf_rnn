'''
Contains functions that handle the shaping of input into batches.
The batches are converted into numpy arrays towards the end for them to play nice with tensorflow
(i.e. avoid the "ValueError: setting an array element with a sequence" error)

Copyright (c) 2017 Frank Derry Wanye

Date: 25 November, 2017
'''
import math
import numpy as np

def make_batches(input_data, labels, batch_size, truncate, pad_token):
    '''
    Converts both the input data and labels into batches of size 'batch_size'.

    Params:
    - input_data (list): The input data to be made into batches
    - labels (list): The labels to be made into batches
    - batch_size (int): The size of the batches
    - truncate (int): The maximum width or length of the batches
    - pad_token (numeric): The token with which to pad the input data batches

    Return:
    - inputs (list): The padded input data batches
    - outputs (list): The padded batches of output labels
    - timestep_lengths (list): The true timestep lengths of the lables batches
    '''
    data = sort_by_length([input_data, labels])
    x_data, y_data = group_into_batches(data, batch_size)
    x_data = truncate_batches(x_data, truncate)
    y_data = truncate_batches(y_data, truncate)
    lengths = get_row_lengths(y_data)
    x_data = pad_batches(x_data, truncate, pad_token)
    y_data = pad_batches(y_data, truncate, pad_token)
    return (x_data, y_data, lengths)
# End of make_batches()

def sort_by_length(data):
    '''
    Sorts the provided data objects in descending order by length.

    Params:
    - data (list): The list of data to be sorted

    Return:
    - sorted_data (list): The data, sorted in descending order by length
    '''
    sorted_data = list()
    for item in data:
        sorted_item = sorted(item, key=len, reverse=True)
        sorted_data.append(sorted_item)
    return sorted_data
# End of sort_by_length()

def group_into_batches(data, batch_size):
    '''
    Group each item in data into batches of size 'batch_size'.

    Params:
    - data (list): The data to be grouped into batches
    - batch_size (list): The size of the data batches

    Return:
    - batches (list): The data items, in batch form
    '''
    if batch_size < 1:
       raise ValueError("The size of the batches cannot be less than 1.")
    batched_data = list()
    for item in data:
        batched_data_item = [item[i:i+batch_size] for i in range(0, len(item), batch_size)]
        if len(batched_data_item) is not 0:
            while(len(batched_data_item[-1]) < batch_size):
                batched_data_item[-1].append([])
        batched_data.append(batched_data_item)
    return batched_data
# End of group_into_batches()

def truncate_batches(data, truncate):
    '''
    Truncates each example in each batch in data, so that no row is longer than 'truncate' values long. The truncated
    parts become new batches in the data.

    Params:
    - data (list): The data batches to be truncated
    - truncate (truncate): The length to which the batches should be truncated

    Return:
    - truncated_batches (list): The truncated batches of data
    '''
    if truncate < 1:
        raise ValueError("The length of each batch cannot be less than 1.")
    truncated_batches = list()
    for batch in data:
        max_length = max(map(len, batch))
        times_to_truncate = math.ceil(max_length / truncate)
        for i in range(times_to_truncate):
            truncated_batch_section = truncate_batch(i, times_to_truncate-1, truncate, batch)
            truncated_batches.append(truncated_batch_section)
    return truncated_batches
# End of truncate_batches()

def truncate_batch(index, last_index, truncate, batch):
    '''
    Truncates each example in the batch, so that no row is longer than 'truncate' values long.
    Also surrounds the batch with two boolean indicators:
    - The indicator at the start of the batch is true if it is the beginning of the example sequence
    - The indicator at the end of the batch is true if it is the ending of the example sequence

    Params:
    - index (int): The index of the resulting batch partition (dictates at what point in the batch the truncation starts)
    - last_index (int): The index of the last batch partition
    - truncate (int): The length to which the batches should be truncated
    - batch (list): The full-length batch on which to perform the truncation

    Return:
    - truncated_batch_section (list): The truncated section of the batch
    '''
    start = index * truncate
    end = start + truncate
    beginning = [True] if index == 0 else [False]
    ending = [True] if index == last_index else [False]
    truncated_batch_section = [example[start:end] for example in batch]
    truncated_batch_section = beginning + truncated_batch_section + ending
    return truncated_batch_section
# End of truncate_batch()

def get_row_lengths(data):
    '''
    Returns the lengths of every row in the given data. The data must be arranged in batches or the function will fail.

    Params:
    - data (list): The list of data (arranged in batches) whose length is to be found

    Return:
    - timestep_lengths (list): The lengths of the sequences of every batch in the given data
    '''
    batch_lengths = list()
    for batch in data:
        item_row_lengths = [batch[0]]
        item_row_lengths += [len(row) for row in batch[1:-1]]
        item_row_lengths += [batch[-1]]
        batch_lengths.append(item_row_lengths)
    return batch_lengths
# End of get_row_lengths()

def pad_batches(data, truncate, pad_token):
    '''
    Pads every row in the data batches so that it's the same length as the value of 'truncate'.

    Params:
    - data (list): The batches of data to be padded
    - truncate (int): The size to which every batch should be padded
    - pad_token (numeric): The token with which to pad the input data batches

    Return:
    - padded_batches (np.array): The padded batches of data
    '''
    if truncate < 1:
        raise ValueError("The length of each batch cannot be less than 1.")
    padded_batches = list()
    pad_token = get_pad_token(data, pad_token)
    for batch in data:
        padded_batch = list()
        for sequence in batch[1:-1]:
            padded_sequence = pad_sequence(sequence, truncate, pad_token)
            padded_batch.append(padded_sequence)
        padded_batches.append(np.array(padded_batch))
    return np.array(padded_batches)
# End of pad_batches()

def get_pad_token(data, pad_token):
    '''
    Determines the correct padding token. If there are multiple inputs, the pad token is replicated for each input.

    Params:
    - data (list): The data to be padded
    - pad_token (numeric): The padding token, to be replicated if there are multiple inputs

    Return:
    - pad_token (numeric/list): The padding token, replicated if there are multiple inputs
    '''
    data_item = data[0][1][0]
    if (type(data_item) is tuple) or (type(data_item) is list):
        pad_token = [pad_token for item in data_item]
    return pad_token
# End of get_pad_token()

def pad_sequence(sequence, truncate, pad_token):
    '''
    Pads a single sequence with the given pad token.

    Params:
    - sequence (list): The sequence to pad
    - truncate (int): The maximum length of every sequence
    - pad_token (numeric/list): The token to be used for padding

    Return:
    - padded_sequence (np.array): The padded sequence
    '''
    padded_sequence = list(sequence)
    while len(padded_sequence) < truncate:
        padded_sequence.append(pad_token)
    return np.array(padded_sequence)
# End of pad_sequence()
