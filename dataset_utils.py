'''
Utility class for creating, saving and loading datasets.

Date: 24 November, 2017
'''

# Specify documentation format
__docformat__ = 'restructedtext en'

import dill
import re #regex library for re.split()
import os
import io
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
    '''
    A simplified method for creating a dataset. The file will contain the following items, in the same order:
    - the vocabulary of the training set
    - the vector used to convert token indexes to words
    - the dictionary used to convert words to token indexes
    - the input for training, in tokenized format (as indexes)
    - the output for training, in tokenized format (as indexes)
    - the start token
    - the end token
    '''
    settings_obj = get_settings()
    logger = setup.setup_logger(settings_obj.logging, settings_obj.logging.log_dir)
    save_dataset(logger, settings_obj.data)
    load_dataset(logger, settings_obj.data.dataset_name)
# End of run()

def get_settings(dataset_only=False):
    '''
    Parses command-line arguments into a settings Object.
    If non-dataset arguments are provided, prints error and exits script.

    Params:
    - dataset_only (bool): True if only settings for the dataset should be provided. Default: False

    Return:
    - settings (settings.Settings): The settings needed for creating a dataset
    '''
    settings_obj = settings.Settings(dataset_only)
    required_keys = constants.DATA_ARGS.keys()
    received_keys = vars(settings_obj.data).keys()
    for key in required_keys:
        if key not in received_keys:
            print('ERROR: Need to pass in dataset arguments to run this script.')
            print('Correct usage: python dataset_maker.py dataset [args...]')
            exit(-1)
    return settings_obj
# End of parse_arguments()

def save_dataset(logger, settings):
    '''
    Saves the created dataset to a specified file.

    Params:
    - logger (logging.Logger): The logger used in this run of the script
    - settings (settings.SettingsNamespace): The dataset creation settings
    '''
    path = constants.DATASETS_DIR
    filename = settings.dataset_name

    setup.create_dir(path)
    dataset = create_dataset(logger, settings)
    with open(path + filename, 'wb') as dataset_file:
        dill.dump(dataset, dataset_file, protocol=2)
# End of save_dataset()

def create_dataset(logger, settings):
    '''
    Creates a dataset using tokenized data.

    Params:
    - logger (logging.Logger): The logger used in this run of the script
    - settings (settings.SettingsNamespace): The dataset creation settings

    Return:
    - dataset (tuple):
      - type (string): The type of dataset ('text' or 'number')
      - token_level (string): The level of tokenization ('sentences', 'paragraphs', or 'stories')
      - vocabulary (list): The vocabulary used in the dataset
      - index_to_token (list): Converts indexes to tokens
      - token_to_index (dict): Converts tokens to indexes
      - x_train (list): Training inputs
      - y_train (list): Training outputs
    '''
    dataset = create_text_dataset(logger, settings, None)
    dataset = create_numeric_dataset(logger, settings, dataset)
    return dataset
# End of create_dataset()

def create_text_dataset(logger, settings, dataset):
    '''
    Creates a dataset based on text data. If the settings chosen do not specify a text dataset, returns the value
    of the dataset parameter, unchanged.

    Params:
    - logger (logging.Logger): The logger used in this run of the script
    - settings (settings.SettingsNamespace): The dataset creation settings
    - dataset (tuple): The previously created dataset, if any

    Return:
    - dataset (tuple):
      - type (string): The type of dataset. 'text', in this case
      - token_level (string): The level of tokenization ('sentences', 'paragraphs', or 'stories')
      - vocabulary (list): The vocabulary used in the dataset
      - index_to_token (list): Converts indexes to tokens
      - token_to_index (dict): Converts tokens to indexes
      - x_train (list): Training inputs
      - y_train (list): Training outputs
    '''
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
    '''
    Creates a dataset based on numeric data. If the settings chosen do not specify a numeric dataset, returns the value
    of the dataset parameter, unchanged.

    Params:
    - logger (logging.Logger): The logger used in this run of the script
    - settings (settings.SettingsNamespace): The dataset creation settings
    - dataset (tuple): The previously created dataset, if any

    Return:
    - dataset (tuple):
      - type (string): 'number'
      - x_train (list): The input data
      - y_train (list): The output data
    '''
    if settings.type != constants.TYPE_CHOICES[1]: # type = 'number'
        return dataset
    # TODO: Implement this function
    return dataset
# End of create_numeric_dataset()

def tokenize_data(logger, settings):
    '''
    Creates a dataset using tokenized data.

    Params:
    - logger (logging.Logger): The logger used in this run of the script
    - settings (settings.SettingsNamespace): The dataset creation settings

    Return:
    - tokenized_data (list): The tokenized data
    '''
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
    '''
    Uses the nltk library to break comments down into sentences, and then
    tokenizes the words in the sentences. Also appends the sentence start and
    end tokens to each sentence.

    Params:
    - logger (logging.Logger): The logger used in this run of the script
    - settings (settings.SettingsNamespace): The dataset creation settings

    Return:
    - tokenized_sentences (list): Tokenized sentence strings
    '''
    comments = read_csv(logger, settings)
    logger.info('Breaking comments down into sentences.')
    sentences = itertools.chain(*[nltk.sent_tokenize(comment.lower()) for comment in comments])
    sentences = list(sentences)
    logger.info("%d sentences found in dataset." % len(sentences))
    return sentences
# End of tokenize_sentences()

def tokenize_paragraphs(logger, settings):
    '''
    Uses the nltk library to break comments down into paragraphs, and then
    tokenizes the words in the paragraphs. Also appends the paragraph start and
    end tokens to each paragraph.

    Params:
    - logger (logging.Logger): The logger used in this run of the script
    - settings (settings.SettingsNamespace): The dataset creation settings

    Return:
    - tokenized_paragraphs (list): Tokenized paragraph strings
    '''
    comments = read_csv(logger, settings)

    paragraphs = list()
    logger.info('Breaking comments down into paragraphs.')
    for comment in comments:
        paragraphs.extend(re.split('\n+', comment.lower()))
    logger.info("%d comments were broken down into %d paragraphs." % (len(comments), len(paragraphs)))

    return paragraphs
# End of tokenize_paragraphs()

def tokenize_stories(logger, settings):
    '''
    Uses the nltk library to word tokenize entire comments, assuming that
    each comment is its own story. Also appends the story start and end tokens
    to each story.

    Params:
    - logger (logging.Logger): The logger used in this run of the script
    - settings (settings.SettingsNamespace): The dataset creation settings

    Return:
    - tokenized_stories (list): Tokenized story strings
    '''
    comments = read_csv(logger, settings)
    logger.info('Retrieving stories from data.')
    stories = [comment.lower() for comment in comments]
    logger.info("Found %d stories in the dataset." % len(stories))
    return stories
# End of tokenize_stories()

def read_csv(logger, settings):
    '''
    Reads the given csv file and extracts data from it into the comments array.
    Empty data cells are not included in the output.

    Params:
    - logger (logging.Logger): The logger used in this run of the script
    - settings (settings.SettingsNamespace): The dataset creation settings

    Return:
    - csv_rows (list): The rows in the data that can be tokenized
    '''
    path = constants.RAW_DATA_DIR + settings.raw_data

    # Encoding breaks when using python2.7 for some reason.
    csv_rows = list()
    logger.info("Reading the csv data file at: %s" % path)
    with open(path, 'r', encoding='utf-8') as datafile:
        reader = csv.reader(datafile, skipinitialspace=True)
        try:
            reader.__next__() # Skips over table heading in Python 3.2+
        except Exception:
            reader.next() # For older versions of Python
        for item in reader:
            if len(item) > 0 and len(item[0]) > 0:
                csv_rows.append(item[0])
                num_seen = len(csv_rows)
                if settings.num_rows <= num_seen:
                    break
    logger.info("%d examples kept for creating training dataset." % num_seen)
    # End with
    return csv_rows
# End of read_csv()

def normalize_examples(logger, settings, examples):
    '''
    Normalizes tokenized examples.
    - Removes invalid examples
    - Replaces invalid tokens with valid ones
    - Reduces number of examples to the requested number
    - Adds start and end tokens to the examples

    Params:
    - logger (logging.Logger): The logger used in this run of the script
    - settings (settings.SettingsNamespace): The dataset creation settings
    - examples (list): Tokenized examples

    Return:
    - normalized_examples (list): The normalized examples
    '''
    examples = preprocess_data(logger, examples)
    examples = examples[:settings.num_examples]
    examples = low_level_tokenize(logger, settings, examples)
    return list(examples)
# End of normalize_examples()

def preprocess_data(logger, data_array):
    '''
    Pre-processes data in data_array so that it is more or less modular.

    Params:
    - logger (logging.Logger): The logger to which to write log output.
    - data_array (list): The list of Strings to be preprocessed

    Return:
    - preprocessed_data (list): The list of preprocessed strings.
    '''
    logger.info('Preprocessing data')
    num_skipped = 0
    preprocessed_data = []
    for item in data_array:
        if '[' in item or ']' in item:
            num_skipped += 1
            continue
        item = item.replace('\n', " %s " % constants.CARRIAGE_RETURN)
        item = item.replace('`', '\'')
        item = item.replace('\'\'', '"')
        preprocessed_data.append(item)
    logger.info("Skipped %d items in data." % num_skipped)
    return preprocessed_data
# End of preprocess_data()

def low_level_tokenize(logger, settings, examples):
    '''
    Tokenizes examples into either words or letters.

    Params:
    - logger (logging.Logger): The logger used in this run of the script
    - settings (settings.SettingsNamespace): The dataset creation settings
    - examples (list): Examples to be tokenized

    Return:
    - tokenized_examples (list): The tokenized examples
    '''
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
    '''
    Creates the vocabulary list out of the given tokenized data.

    Params:
    - logger (logging.Logger): The logger used in this run of the script
    - settings (settings.SettingsNamespace): The dataset creation settings
    - data (list): Tokenized data

    Return:
    - vocabulary (list): The most common vocabulary words in the tokenized data
    '''
    logger.info('Obtaining word frequency disribution.')
    word_freq = nltk.FreqDist(itertools.chain(*data))
    logger.info("Found %d unique words." % len(word_freq.items()))

    if settings.vocab_size == None:
        vocabulary = word_freq.most_common(None)
    else:
        vocabulary = word_freq.most_common(settings.vocab_size - 1)

    logger.info('Calculating percent of words captured...')
    total = 0
    for word in vocabulary:
        total += word_freq.freq(word[0])
    logger.info("Percent of total words captured: %f" % (total * 100))
    return vocabulary
# End of create_vocabulary()

def create_training_data(logger, settings, data, token_to_index):
    '''
    Creates the inputs and labels for training.

    Params:
    - logger (logging.Logger): The logger used in this run of the script
    - settings (settings.SettingsNamespace): The dataset creation settings
    - data (list): The data to break into inputs and labels
    - token_to_index (dict): The dictionary used to convert words to indexes

    Return:
    - training_data (tuple):
      - inputs (list): The training inputs
      - labels (list): The training labels (outputs)
    '''
    if type(data[0][0]) is str: # Currently not needed for other types of data
        logger.info('Replace all words not in vocabulary with unkown token.')
        for index, sentence in enumerate(data):
            data[index] = [word if word in token_to_index else constants.UNKNOWN for word in sentence]

    logger.info('Creating training data.')
    data = tokens_to_indexes(data, token_to_index)
    x_train = [item[:-1] for item in data]
    y_train = [item[1:] for item in data]
    if (type(y_train[0][0]) is tuple) or (type(y_train[0][0]) is list):
        y_train = [[word[0] for word in row] for row in y_train] # Training labels only have one outcome
    return x_train, y_train
# End of create_training_data()

def tokens_to_indexes(data, token_to_index):
    '''
    Replaces the tokens in the data with the token's indexes in the vocabulary.

    Params:
    - data (list): The data to break into inputs and labels
    - token_to_index (dict): The dictionary used to convert words to indexes

    Return:
    - modified_data (list): The data with the tokens replaced with their indexes in the vocabulary
    '''
    new_data = list()
    for row in data:
        if type(row[0]) is str:
            new_row = [token_to_index[word] for word in row]
        elif (type(row[0]) is tuple) or (type(row[0]) is list):
            new_row = [[token_to_index[char] for char in word] for word in row]
        new_data.append(new_row)
    return new_data
# End of tokens_to_indexes()

def load_dataset(logger, dataset):
    '''
    Loads a saved dataset.

    Params:
    - logger (logging.Logger): The logger to be used for logging function results
    - dataset (string): The filename of the pickled dataset to load

    Return:
    - dataset (tuple):
      - type (string): The type of dataset ('text' or 'number')
      - token_level (string): The level of tokenization ('sentences', 'paragraphs', or 'stories')
      - vocabulary (list): The vocabulary used in the dataset
      - index_to_token (list): Converts indexes to tokens
      - token_to_index (dict): Converts tokens to indexes
      - x_train (list): Training inputs
      - y_train (list): Training outputs
    '''
    path = constants.DATASETS_DIR + dataset

    logger.info("Loading saved dataset.")
    with open(path, "rb") as dataset_file:
        data = dill.load(dataset_file)
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
