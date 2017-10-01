"""
Contains functions that handle the shaping of input into batches.

Copyright (c) 2017 Frank Derry Wanye

Date: 30 September, 2017
"""
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
    sorted_data = tuple(sorted_data)
    return sorted_data
# End of sort_by_length()

def make_batches(input_data, labels, batch_size, truncate):
    """
    Converts both the input data and labels into batches of size 'batch_size'.

    Params:
    input_data (list): The input data to be made into batches
    labels (list): The labels to be made into batches
    batch_size (int): The size of the batches
    truncate (int): The maximum width or length of the batches

    Return:
    ((list, list), (list, list)): input data and labels, in batch form, with the original size of each row before
                                  padding
    """
    sorted_data = sort_by_length([input_data, labels])
    return (sorted_data[0], []), (sorted_data[1], [])
# End of make_batches()