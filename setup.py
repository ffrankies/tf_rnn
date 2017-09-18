"""
Utility class for setting up an RNN.

Copyright (c) 2017 Frank Derry Wanye

Date: 15 September, 2017
"""

# Specify documentation format
__docformat__ = 'restructedtext en'

import argparse
import logging
import logging.handlers
import os
import sys

from . import constants

def parse_arguments():
    """
    Parses the command line arguments and returns the namespace with those
    arguments.

    There are three modes for this:
    - config: Specify a config place in place of command-line options
    - options: Specify command-line options for all settings
    - dataset: Specify command-line options for creating a dataset, or a config file to do the same

    Return:
    argparse.Namespace: The Namespace containing the values of all passed-in command-line arguments
    """
    arg_parse = argparse.ArgumentParser()
    subparsers = arg_parse.add_subparsers(help="Sub-command help.")
    config_parser = subparsers.add_parser("config", help="Pick a config file for setting up the network.")
    config_parser.add_argument(constants.CONFIG_FILE_STR, help="The name of the config file holding network settings.")
    add_options_parser(subparsers)
    add_dataset_parser(subparsers)
    return arg_parse.parse_args()
# End of parse_arguments()

def add_options_parser(subparsers):
    """
    Adds network settings as command_line arguments.

    Params:
    subparser (argparse.Namespace): Container for the subparser Namespace objects
    """
    options_parser = subparsers.add_parser("options", help="Provide network arguments as command arguments.")
    add_general_arguments(options_parser)
    add_log_arguments(options_parser)
    add_rnn_arguments(options_parser)
    add_train_arguments(options_parser)
# End of add_options_parser()

def add_dataset_parser(subparsers):
    """
    Adds a subparser containing arguments for creating a dataset.
    Arguments added:
    - raw_data
    - dataset_name
    - source_type
    - vocab_size
    - num_comments
    - num_examples
    - mode

    Params:
    subparser (argparse.Namespace): Container for the subparser Namespace objects
    """
    parser = subparsers.add_parser('dataset', help="Provide arguments for creating a dataset.")
    parser.add_argument("-c", "--config_file", default=constants.CONFIG_FILE,
                        help="The config file to ")
    parser.add_argument("-r", "--raw_data", default=constants.RAW_DATA,
                        help="The name of the file containing raw (unprocessed) data.")
    parser.add_argument("-d", "--dataset_name", default=constants.DATASET_NAME,
                        help="The name of the saved dataset.")
    parser.add_argument("-s", "--source_type", default=constants.SOURCE_TYPE,
                        help="The type of source data [currently only the csv data type is supported].")
    parser.add_argument("-v", "--vocab_size", default=constants.VOCAB_SIZE, type=int,
                        help="The size of the dataset vocabulary.")
    parser.add_argument("-w", "--num_rows", type=int, default=constants.NUM_ROWS,
                        help="The number of rows of data to be read.")
    parser.add_argument("-n", "--num_examples", type=int, default=constants.NUM_EXAMPLES,
                        help="The number of sentence examples to be saved.")
    parser.add_argument("-m", "--mode", default=constants.MODE, choices=constants.MODE_CHOICES,
                        help="Selects what constitutes an example in the dataset.")
    parser.add_argument("-t", "--token_level", default=constants.TOKEN_LEVEL, choices=constants.TOKEN_LEVEL_CHOICES,
                        help="Selects on what level to break down the training data.")
# End of add_dataset_parser()

def add_general_arguments(parser):
    """
    Adds general model arguments to the given argument parser.
    Arguments added:
    --model_name

    :type parser: argparse.ArgumentParser
    :param parser: The argument parser to which to add the logger arguments.
    """
    parser.add_argument("-m", "--model_name", default=constants.MODEL_NAME,
                        help="The previously trained model to load on init.")
# End of add_general_arguments()

def add_log_arguments(parser):
    """
    Adds arguments for setting up the logger to the given argument parser.
    Arguments added:
    --log_name
    --log_filename
    --log_dir

    :type parser: argparse.ArgumentParser
    :param parser: The argument parser to which to add the logger arguments.
    """
    parser.add_argument("-ln", "--log_name", default=constants.LOG_NAME,
                        help="The name of the logger to be used. Defaults to %s" % constants.LOG_NAME)
    parser.add_argument("-lf", "--log_filename", default=constants.LOG_FILENAME,
                        help="The name of the file to which the logging will be done.")
    parser.add_argument("-ld", "--log_dir", default=constants.LOG_DIR,
                        help="The path to the directory where the log file will be stored.")
# End of add_log_arguments()

def add_rnn_arguments(parser):
    """
    Adds arguments for setting up an RNN to the given argument parser.
    Arguments added:
    --dataset
    --hidden_size
    --embed_size
    --model

    :type parser: argparse.ArgumentParser
    :param parser: The argument parser to which to add the logger arguments.
    """
    parser.add_argument("-d", "--dataset", default=constants.DATASET,
                        help="The path to the dataset to be used for training.")
    parser.add_argument("-hs", "--hidden_size", type=int, default=constants.HIDDEN_SIZE,
                        help="The size of the hidden layers in the RNN.")
    parser.add_argument("-es", "--embed_size", type=int, default=constants.EMBED_SIZE,
                        help="The size of the embedding layer in the RNN.")
# End of add_rnn_arguments()

def add_train_arguments(parser):
    """
    Adds arguments for training an RNN to the given argument parser.
    Arguments added:
    --epochs
    --patience
    --test
    --learning_rate
    --anneal
    --truncate
    --batch_size

    :type parser: argparse.ArgumentParser
    :param parser: The argument parser to which to add the logger arguments.
    """
    parser.add_argument("-e", "--epochs", default=constants.EPOCHS, type=int,
                        help="The number of epochs for which to train the RNN.")
    parser.add_argument("-p", "--patience", default=constants.PATIENCE, type=int,
                        help="The number of examples to train before evaluating loss.")
    parser.add_argument("-l", "--learn_rate", default=constants.LEARN_RATE, type=float,
                        help="The learning rate to be used in training.")
    parser.add_argument("-a", "--anneal", type=float, default=constants.ANNEAL,
                        help="The minimum possible learning rate.")
    parser.add_argument("-t", "--truncate", type=int, default=constants.TRUNCATE,
                        help="The backpropagate truncate value.")
    parser.add_argument("-b", "--batch_size", type=int, default=constants.BATCH_SIZE,
                        help="The size of the batches into which to split the training data.")
# End of add_train_arguments()

def create_dir(dirPath):
    """
    Creates a directory if it does not exist.

    :type dirPath: string
    :param dirPath: the path of the directory to be created.
    """
    try:
        if os.path.dirname(dirPath) != "":
            os.makedirs(os.path.dirname(dirPath), exist_ok=True) # Python 3.2+
    except TypeError:
        try: # Python 3.2-
            os.makedirs(dirPath)
        except OSError as exception:
            if exception.errno != 17:
                raise
# End of create_dir()

def get_arg(settings, argument, asInt=False, asBoolean=False, asFloat=False, checkNone=False):
    """
    Retrieves the argument from the given settings namespace, or asks the user to enter one.

    :type settings: argparse.Namespace() object
    :param settings: The parsed command-line arguments to the program.

    :type argument: String
    :param argument: the argument to retrieve from the settings.

    :type return: Any
    :param return: The value of the returned setting.
    """
    arg = None
    if argument in settings:
        arg = vars(settings)[argument]
    if checkNone:
        if arg == None:
            arg = input("Specify the value for %s" % argument)
    if asInt:
        arg = int(arg)
    if asBoolean:
        if arg.lower() == 'true' or arg.lower() == 't':
            arg = True
        if arg.lower() == 'false' or arg.lower() == 'f':
            arg = False
    if asFloat:
        arg = float(arg)
    return arg
# End of get_arg()

def setup_logger(args, logger_dir):
    """
    Sets up a logger

    Params:
    args (settings.SettingsNamespace): The logging settings
    logger_dir (string): The directory where the logs will be saved

    Return:
    logging.Logger: The logger created with the given settings
    """
    logger = logging.getLogger(args.log_name)
    logger.setLevel(logging.INFO)
    create_dir(logger_dir)

    # Logger will use up to 5 files for logging, 'rotating' the data between them as they get filled up.
    handler = logging.handlers.RotatingFileHandler(
        filename=logger_dir + args.log_filename,
        maxBytes=1024*512,
        backupCount=5
    )

    formatter = logging.Formatter(
        "%(asctime)s-%(name)s-%(levelname)s-%(message)s"
    )

    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.info("Logger successfully set up.")
    return logger
# End of setup_logger
