"""
Contains functions that handle the shaping of input into batches.

Copyright (c) 2017 Frank Derry Wanye

Date: 1 October, 2017
"""
import math

def make_batches(input_data, labels, batch_size, truncate, pad_token):
    """
    Converts both the input data and labels into batches of size 'batch_size'.

    Params:
    input_data (list): The input data to be made into batches
    labels (list): The labels to be made into batches
    batch_size (int): The size of the batches
    truncate (int): The maximum width or length of the batches
    pad_token (numeric): The token with which to pad the input data batches

    Return:
    (list, list, list): The padded input data batches, the labels batches, and the row sizes of the lables batches
    """
    data = sort_by_length([input_data, labels])
    data = group_into_batches(data, batch_size)
    x_data, y_data = truncate_batches(data, truncate)
    lengths = get_row_lengths(y_data)
    x_data = pad_batches(x_data, truncate, pad_token)
    return (x_data, y_data, lengths)
# End of make_batches()

def sort_by_length(data):
    """
    Sorts the provided data objects in descending order by length.

    Params:
    data (list): The list of data to be sorted

    Return:
    list: The data, sorted in descending order by length
    """
    sorted_data = list()
    for item in data:
        sorted_item = sorted(item, reverse=True)
        sorted_data.append(sorted_item)
    return sorted_data
# End of sort_by_length()

def get_row_lengths(data):
    """
    Returns the lengths of every row in the given data. The data must be arranged in batches or the function will fail.

    Params:
    data (list): The list of data (arranged in batches) whose length is to be found

    Return:
    list: The lengths of the rows of every batch in the given data
    """
    batch_lengths = list()
    for batch in data:
        item_row_lengths = [len(row) for row in batch]
        batch_lengths.append(item_row_lengths)
    return batch_lengths
# End of get_row_lengths()

def group_into_batches(data, batch_size):
    """
    Group each item in data into batches of size 'batch_size'.

    Params:
    data (list): The data to be grouped into batches
    batch_size (list): The size of the data batches

    Return:
    list: The data items, in batch form
    """
    if batch_size < 1:
       raise ValueError("The size of the batches cannot be less than 1.") 
    batched_data = list()
    for item in data:
        batched_data_item = [item[i:i+batch_size] for i in range(0, len(item), batch_size)]
        batched_data.append(batched_data_item)
    return batched_data
# End of group_into_batches()

def truncate_batches(data, truncate):
    """
    Truncates each row in each item in data, so that no row is longer than 'truncate' values long. The truncated 
    parts become new rows in the data.

    Params:
    data (list): The data batches to be truncated
    truncate (truncate): The length to which the batches should be truncated

    Return:
    list: The truncated batches of data
    """
    if truncate < 1:
        raise ValueError("The length of each batch cannot be less than 1.")
    truncated_data = list()
    for item in data:
        item_batches = list()
        for batch in item:
            max_length = max(map(len, batch))
            times_to_truncate = math.ceil(max_length / truncate)
            for i in range(times_to_truncate):
                truncated_batch = list()
                start = i * truncate
                end = start + truncate
                for example in batch:
                    if start > len(example):
                        truncated_batch.append([])
                    else:
                        truncated_batch.append(example[start:end])
                item_batches.append(truncated_batch)
        truncated_data.append(item_batches)
    return truncated_data
# End of truncate_batches()

def pad_batches(x_data, truncate, pad_token):
    """
    Pads every row in the data batches so that it's the same length as the value of 'truncate'.

    Params:
    x_data (list): The batches of input data to be padded
    truncate (int): The size to which every batch should be padded
    pad_token (numeric): The token with which to pad the input data batches

    Return:
    list: The padded batches of input data
    """
    if truncate < 1:
        raise ValueError("The length of each batch cannot be less than 1.")
    padded_batches = list()
    for batch in x_data:
        padded_batch = list()
        for example in batch:
            padded_example = list()
            padded_example.extend(example)
            while len(padded_example) < truncate:
                padded_example.append(pad_token)
            padded_batch.append(padded_example)
        padded_batches.append(padded_batch)
    return padded_batches
# End of pad_batches()