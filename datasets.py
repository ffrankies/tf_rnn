"""
This project is licensed under the MIT license:
https://github.com/ffrankies/Terry/blob/master/LICENSE

This project uses the Theano library:
https://arxiv.org/pdf/1605.02688.pdf

This project is based on the following tutorial:
https://goo.gl/DPf37h

Copyright (c) 2017 Frank Derry Wanye

Date: 24 September, 2017
"""

###############################################################################
# Source: rnn tutorial from www.wildml.com
# This script closely follows the tutorial, repurposing it to work with python3.
# This part of the code creates a dataset from a given csv file. The csv file
# should contain only one column, start with a column heading, and contain
# text data in sentence format. The script will break the data down into
# sentences, paragraphs or stories, tokenize them, and then save the dataset in
# a file of the user's choice.
# The file will contain the following items, in the same order:
#   the vocabulary of the training set
#   the vector used to convert token indexes to words
#   the dictionary used to convert words to token indexes
#   the input for training, in tokenized format (as indexes)
#   the output for training, in tokenized format (as indexes)
#   the start token
#   the end token
#
# Author: Frank Wanye
# Date: 24 September, 2017
###############################################################################

# Specify documentation format
__docformat__ = 'restructedtext en'

try:
    import _pickle as cPickle
except Exception:
    import cPickle
import re #regex library for re.split()
import os
import io
import numpy as np
# import operator
import csv
import itertools
import nltk
import logging
import logging.handlers
import argparse
from . import constants
from . import setup
from . import settings

def run():
    """
    A simplified method for creating a dataset.
    """
    settings_obj = get_settings()
    logger = setup.setup_logger(settings_obj.logging, settings_obj.logging.log_dir)
    save_dataset(logger, settings_obj.data)
    load_dataset(logger, settings_obj.data.dataset_name)
# End of run()

def get_settings():
    """
    Parses command-line arguments into a settings Object. 
    If non-dataset arguments are provided, prints error and exits script.

    Return:
    settings.Settings: The settings needed for creating a dataset
    """
    settings_obj = settings.Settings()
    if constants.DATA_ARGS.keys() != vars(settings_obj.data).keys():
        print("ERROR: Need to pass in dataset arguments to run this script.")
        print("Correct usage: python dataset_maker.py dataset [args...]")
        exit(-1)
    return settings_obj
# End of parse_arguments()

def save_dataset(logger, settings):
    """
    Saves the created dataset to a specified file.

    Params:
    logger (logging.Logger): The logger used in this run of the script
    settings (settings.SettingsNamespace): The dataset creation settings
    """
    path = constants.DATASETS_DIR
    filename = settings.dataset_name

    setup.create_dir(path)
    dataset = create_dataset(logger, settings)
    with open(path + filename, "wb") as dataset_file:
        cPickle.dump(dataset, dataset_file, protocol=2)
# End of save_dataset()

def create_dataset(logger, settings):
    """
    Creates a dataset using tokenized data.

    Params:
    logger (logging.Logger): The logger used in this run of the script
    settings (settings.SettingsNamespace): The dataset creation settings

    Return:
    tuple: (vocabulary as List, index_to_word as List, token_to_index as Dict, x_train as List, y_train as List)
    """
    dataset = create_text_dataset(logger, settings, None)
    dataset = create_numeric_dataset(logger, settings, dataset)
    return dataset
# End of create_dataset()

def create_text_dataset(logger, settings, dataset):
    """
    Creates a dataset based on text data. If the settings chosen do not specify a text dataset, returns the value 
    of the dataset parameter, unchanged.

    Params:
    logger (logging.Logger): The logger used in this run of the script
    settings (settings.SettingsNamespace): The dataset creation settings
    dataset (tuple): The previously created dataset, if any

    Return:
    tuple: (type, token_level, vocabulary: List, index_to_word: List, token_to_index: Dict, x_train: List, 
            y_train: List)
    """
    if settings.type != constants.TYPE_CHOICES[0]: # type = 'text'
        return dataset   
    data = tokenize_data(logger, settings)
    data = normalize_examples(logger, settings, data)
    vocabulary = create_vocabulary(logger, settings, data)
    index_to_token = [token[0] for token in vocabulary]
    index_to_token.append(constants.UNKNOWN)
    token_to_index = dict((token, index) for index, token in enumerate(index_to_token))
    x_train, y_train = create_training_data(logger, settings, data, token_to_index)
    return (constants.TYPE_CHOICES[0], 
            constants.TOKEN_LEVEL_CHOICES[0], 
            vocabulary, 
            index_to_token, 
            token_to_index, 
            x_train, 
            y_train)
# End of create_text_dataset

def create_numeric_dataset(logger, settings, dataset):
    """
    Creates a dataset based on numeric data. If the settings chosen do not specify a numeric dataset, returns the value 
    of the dataset parameter, unchanged.

    Params:
    logger (logging.Logger): The logger used in this run of the script
    settings (settings.SettingsNamespace): The dataset creation settings
    dataset (tuple): The previously created dataset, if any

    Return:
    tuple: either 
    ("word_dataset", vocabulary: List, index_to_word: List, token_to_index: Dict, x_train: List, y_train: List)
           or
    ("char_dataset", vocabulary: List, x_train: List, y_train: List)
    """
    if settings.type != constants.TYPE_CHOICES[1]: # type = 'number'
        return dataset
    # TODO: Implement this function
    return dataset
# End of create_numeric_dataset()

def tokenize_data(logger, settings):
    """
    Creates a dataset using tokenized data.

    Params:
    logger (logging.Logger): The logger used in this run of the script
    settings (settings.SettingsNamespace): The dataset creation settings

    Return:
    list: The tokenized data
    """
    mode = settings.mode
    if mode == 'sentences':
        data = tokenize_sentences(logger, settings)
    elif mode == 'paragraphs':
        data = tokenize_paragraphs(logger, settings)
    else: # mode == 'stories'
        data = tokenize_stories(logger, settings)
    return data
# End of tokenize_data()

def tokenize_sentences(logger, settings):
    """
    Uses the nltk library to break comments down into sentences, and then
    tokenizes the words in the sentences. Also appends the sentence start and
    end tokens to each sentence.

    Params:
    logger (logging.Logger): The logger used in this run of the script
    settings (settings.SettingsNamespace): The dataset creation settings

    Return:
    list: Tokenized sentence strings
    """
    comments = read_csv(logger, settings)

    logger.info("Breaking comments down into sentences.")
    sentences = itertools.chain(*[nltk.sent_tokenize(comment.lower()) for comment in comments])
    sentences = list(sentences)
    logger.info("%d sentences found in dataset." % len(sentences))

    return sentences
# End of tokenize_sentences()

def tokenize_paragraphs(logger, settings):
    """
    Uses the nltk library to break comments down into paragraphs, and then
    tokenizes the words in the paragraphs. Also appends the paragraph start and
    end tokens to each paragraph.

    Params:
    logger (logging.Logger): The logger used in this run of the script
    settings (settings.SettingsNamespace): The dataset creation settings

    Return:
    list: Tokenized paragraph strings
    """
    comments = read_csv(logger, settings)

    paragraphs = list()
    logger.info("Breaking comments down into paragraphs.")
    for comment in comments:
        paragraphs.extend(re.split('\n+', comment.lower()))
    logger.info("%d comments were broken down into %d paragraphs." % (len(comments), len(paragraphs)))

    return paragraphs
# End of tokenize_paragraphs()

def tokenize_stories(logger, settings):
    """
    Uses the nltk library to word tokenize entire comments, assuming that
    each comment is its own story. Also appends the story start and end tokens
    to each story.

    Params:
    logger (logging.Logger): The logger used in this run of the script
    settings (settings.SettingsNamespace): The dataset creation settings

    Return:
    list: Tokenized story strings
    """
    comments = read_csv(logger, settings)

    logger.info("Retrieving stories from data.")
    stories = [comment.lower() for comment in comments]
    logger.info("Found %d stories in the dataset." % len(stories))
    
    return stories
# End of tokenize_stories()

def read_csv(logger, settings):
    """
    Reads the given csv file and extracts data from it into the comments array.
    Empty data cells are not included in the output.

    Params:
    logger (logging.Logger): The logger used in this run of the script
    settings (settings.SettingsNamespace): The dataset creation settings

    Return:
    list: The rows in the data that can be tokenized
    """
    path = constants.RAW_DATA_DIR + settings.raw_data

    # Encoding breaks when using python2.7 for some reason.
    comments = list()
    logger.info("Reading the csv data file at: %s" % path)
    with open(path, "r", encoding='utf-8') as datafile:
        reader = csv.reader(datafile, skipinitialspace=True)
        try:
            reader.__next__() # Skips over table heading in Python 3.2+
        except Exception:
            reader.next() # For older versions of Python
        for item in reader:
            if len(item) > 0 and len(item[0]) > 0:
                comments.append(item[0])
                num_seen = len(comments)
                if settings.num_rows <= num_seen:
                    break
    logger.info("%d examples kept for creating training dataset." % num_seen)
    # End with
    return comments
# End of read_csv()

def normalize_examples(logger, settings, examples):
    """
    Normalizes tokenized examples. 
    - Removes invalid examples
    - Replaces invalid tokens with valid ones
    - Reduces number of examples to the requested number
    - Adds start and end tokens to the examples

    Params:
    logger (logging.Logger): The logger used in this run of the script
    settings (settings.SettingsNamespace): The dataset creation settings
    examples (list): Tokenized examples

    Return:
    list: The normalized examples
    """
    examples = preprocess_data(logger, examples)
    examples = examples[:settings.num_examples]
    examples = low_level_tokenize(logger, settings, examples)
    return list(examples)
# End of normalize_examples()

def preprocess_data(logger, data_array):
    """
    Pre-processes data in data_array so that it is more or less modular.

    :type logger: logging.Logger()
    :param logger: the logger to which to write log output.

    :type data_array: list()
    :param data_array: the list of Strings to be preprocessed

    :type return: list()
    :param return: the list of preprocessed Strings.
    """
    logger.info("Preprocessing data")
    num_skipped = 0
    preprocessed_data = []
    for item in data_array:
        if "[" in item or "]" in item:
            num_skipped += 1
            continue
        item = item.replace("\n", " %s " % constants.CARRIAGE_RETURN)
        item = item.replace("\'\'", "\"")
        item = item.replace("``", "\"")
        preprocessed_data.append(item)
    logger.info("Skipped %d items in data." % num_skipped)
    return preprocessed_data
# End of preprocess_data()

def low_level_tokenize(logger, settings, examples):
    """
    Tokenizes examples into either words or letters.

    Params:
    logger (logging.Logger): The logger used in this run of the script
    settings (settings.SettingsNamespace): The dataset creation settings
    examples (list): Examples to be tokenized

    Return:
    list: The tokenized examples
    """
    logger.info("Adding start and end tokens to examples.")
    if settings.token_level == constants.TOKEN_LEVEL_CHOICES[0]: # Words
        examples = ["%s %s %s" % (constants.START_TOKEN, example, constants.END_TOKEN) for example in examples]
        logger.info("Tokenizing words in examples.")
        examples = [nltk.word_tokenize(example.lower()) for example in examples]
    else:
        examples = ["%s%s%s" % (constants.START_TOKEN, example, constants.END_TOKEN) for example in examples]
        logger.info("Tokenizes characters in examples.")
        examples = [list(example) for example in examples]
    return examples
# End of low_level_tokenize()

def create_vocabulary(logger, settings, data):
    """
    Creates the vocabulary list out of the given tokenized data.

    Params:
    logger (logging.Logger): The logger used in this run of the script
    settings (settings.SettingsNamespace): The dataset creation settings
    data (list): Tokenized data

    Return:
    list: The most common vocabulary words in the tokenized data
    """
    logger.info("Obtaining word frequency disribution.")
    word_freq = nltk.FreqDist(itertools.chain(*data))
    logger.info("Found %d unique words." % len(word_freq.items()))

    vocabulary = word_freq.most_common(settings.vocab_size - 1)

    logger.info("Calculating percent of words captured...")
    total = 0
    for word in vocabulary:
        total += word_freq.freq(word[0])
    logger.info("Percent of total words captured: %f" % (total * 100))
    return vocabulary
# End of create_vocabulary()

def create_training_data(logger, settings, data, token_to_index):
    """
    Creates the inputs and labels for training.

    Params:
    logger (logging.Logger): The logger used in this run of the script
    settings (settings.SettingsNamespace): The dataset creation settings
    data (list): The data to break into inputs and labels
    token_to_index (dict): The dictionary used to convert words to indexes

    Return:
    tuple: (inputs, labels)
    """
    logger.info("Replace all words not in vocabulary with unkown token.")
    for index, sentence in enumerate(data):
        data[index] = [word if word in token_to_index else constants.UNKNOWN for word in sentence]

    logger.info("Creating training data.")
    x_train = np.asarray([[token_to_index[word] for word in item[:-1]] for item in data])
    y_train = np.asarray([[token_to_index[word] for word in item[1:]] for item in data])
    return x_train, y_train
# End of create_training_data()

def load_dataset(logger, dataset):
    """
    Loads a saved dataset.

    Params:
    logger (logging.Logger): The logger to be used for logging function results
    dataset (string): The filename of the pickled dataset to load

    Return:
    tuple: (data_type, token_level, vocabulary, index_to_word, token_to_index, x_train, y_train)
    """
    path = constants.DATASETS_DIR + dataset

    logger.info("Loading saved dataset.")
    with open(path, "rb") as dataset_file:
        data = cPickle.load(dataset_file)
        dataset_type = data[0]
        token_level = data[1]
        vocabulary = data[2]
        index_to_token = data[3]
        token_to_index = data[4]
        x_train = data[5]
        y_train = data[6]

        logger.info("The dataset type is: %s" % dataset_type)
        logger.info("The tokenizing level is: %s" % token_level)
        logger.info("Size of vocabulary is: %d" % len(vocabulary))
        logger.info("Some words from vocabulary: \n%s" % index_to_token[:50])
        logger.info("Number of examples: %d" % len(x_train))
        logger.info("Sample training input: \n%s" % x_train[:5])
        logger.info("Sample training labels: \n%s" % y_train[:5])
    # End with
    return data
# End of load_dataset()

if __name__ == "__main__":
    run()