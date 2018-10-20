"""Utility module for parsing command-line arguments.

@since 0.6.3
"""

import argparse

from . import constants

# Specify documentation format
__docformat__ = 'restructedtext en'


def parse_arguments(dataset_only: bool = False) -> argparse.Namespace:
    """Parses the command line arguments and returns the namespace with those
    arguments.

    There are three modes for this:
    - config: Specify a config place in place of command-line options
    - options: Specify command-line options for all settings
    - dataset: Specify command-line options for creating a dataset, or a config file to do the same

    Params:
    - dataset_only (bool): True if only the dataset arguments should be added to the options parser

    Return:
    - args (argparse.Namespace): The Namespace containing the values of all passed-in command-line arguments
    """
    arg_parse = argparse.ArgumentParser()
    subparsers = arg_parse.add_subparsers(help='Sub-command help.')
    config_parser = subparsers.add_parser('config', help='Pick a config file for setting up the network.')
    config_parser.add_argument('config_file', help='The name of the config file holding network settings.')
    add_options_parser(subparsers, dataset_only)
    return arg_parse.parse_args()
# End of parse_arguments()


def add_options_parser(subparsers: argparse.Namespace, dataset_only: bool = False):
    """Adds network settings as command_line arguments.

    Params:
    - subparsers (argparse.Namespace): Container for the subparser Namespace objects
    """
    options_parser = subparsers.add_parser('options', help='Provide network arguments as command arguments.')
    add_dataset_arguments(options_parser)
    if not dataset_only:
        add_general_arguments(options_parser)
        add_log_arguments(options_parser)
        add_rnn_arguments(options_parser)
        add_train_arguments(options_parser)
# End of add_options_parser()


def add_dataset_arguments(parser: argparse.ArgumentParser):
    """Adds a subparser containing arguments for creating a dataset.

    Arguments added:
    - raw_data
    - dataset_name
    - source_type
    - vocab_size
    - num_comments
    - num_examples
    - mode

    Params:
    - parser (argparse.ArgumentParser): The argument parser to which to add the dataset arguments
    """
    group = parser.add_argument_group('Dataset Args')
    group.add_argument('-dc', '--config_file', default=constants.CONFIG_FILE,
                       help='The config file to ')
    group.add_argument('-dr', '--raw_data', default=constants.RAW_DATA,
                       help='The name of the file containing raw (unprocessed) data.')
    group.add_argument('-dd', '--dataset_name', default=constants.DATASET_NAME,
                       help='The name of the saved dataset.')
    group.add_argument('-ds', '--source_type', default=constants.SOURCE_TYPE,
                       help='The type of source data [currently only the csv data type is supported].')
    group.add_argument('-dv', '--vocab_size', default=constants.VOCAB_SIZE, type=int,
                       help='The size of the dataset vocabulary.')
    group.add_argument('-dw', '--num_rows', type=int, default=constants.NUM_ROWS,
                       help='The number of rows of data to be read.')
    group.add_argument('-dn', '--num_examples', type=int, default=constants.NUM_EXAMPLES,
                       help='The number of sentence examples to be saved.')
    group.add_argument('-dt', '--type', default=constants.TYPE, choices=constants.TYPE_CHOICES,
                       help='The type of the dataset.')
    group.add_argument('-dm', '--mode', default=constants.MODE, choices=constants.MODE_CHOICES,
                       help='Selects what constitutes an example in the dataset.')
    group.add_argument('-dl', '--token_level', default=constants.TOKEN_LEVEL, choices=constants.TOKEN_LEVEL_CHOICES,
                       help='Selects on what level to break down the training data.')
# End of add_dataset_arguments()


def add_general_arguments(parser: argparse.ArgumentParser):
    """Adds general model arguments to the given argument parser.

    Arguments added:
    - model_name
    - new_model
    - best_model

    Params:
    - parser (argparse.ArgumentParser): The argument parser to which to add the general arguments
    """
    group = parser.add_argument_group('General Args')
    group.add_argument('-m', '--model_name', default=constants.MODEL_NAME,
                       help='The previously trained model to load on init.')
    group.add_argument('-n', '--new_model', action='store_true',
                       help='Provide this option if you want to train the same model from the beginning')
    group.add_argument('-o', '--best_model', action='store_true',
                       help='Provide this option if you want to load the weights that produced the best accuracy')
# End of add_general_arguments()


def add_log_arguments(parser: argparse.ArgumentParser):
    """Adds arguments for setting up the logger to the given argument parser.

    Arguments added:
    - log_name
    - log_filename
    - log_dir
    - log_level

    Params:
    - parser (argparse.ArgumentParser): The argument parser to which to add the logger arguments
    """
    group = parser.add_argument_group('Logging Args')
    group.add_argument('-ln', '--log_name', default=constants.LOG_NAME,
                       help="The name of the logger to be used. Defaults to %s" % constants.LOG_NAME)
    group.add_argument('-lf', '--log_filename', default=constants.LOG_FILENAME,
                       help='The name of the file to which the logging will be done.')
    group.add_argument('-ld', '--log_dir', default=constants.LOG_DIR,
                       help='The path to the directory where the log file will be stored.')
    group.add_argument('-ll', '--log_level', default=constants.LOG_LEVEL,
                       help='The level at which the logger logs data.')
# End of add_log_arguments()


def add_rnn_arguments(parser: argparse.ArgumentParser):
    """
    Adds arguments for setting up an RNN to the given argument parser.

    Arguments added:
    - dataset
    - hidden_size
    - embed_size
    - layers
    - dropout

    Params:
    - parser (argparse.ArgumentParser): The argument parser to which to add the RNN arguments
    """
    group = parser.add_argument_group('RNN Args')
    group.add_argument('-d', '--dataset', default=constants.DATASET,
                       help='The path to the dataset to be used for training.')
    group.add_argument('-s', '--hidden_size', type=int, default=constants.HIDDEN_SIZE,
                       help='The size of the hidden layers in the RNN.')
    group.add_argument('-z', '--embed_size', type=int, default=constants.EMBED_SIZE,
                       help='The size of the embedding layer in the RNN.')
    group.add_argument('-y', '--layers', type=int, default=constants.LAYERS,
                       help='The number of layers in the RNN.')
    group.add_argument('-r', '--dropout', type=float, default=constants.DROPOUT,
                       help='The dropout to be applied at each RNN layer.')
# End of add_rnn_arguments()


def add_train_arguments(parser: argparse.ArgumentParser):
    """Adds arguments for training an RNN to the given argument parser.

    Arguments added:
    - epochs
    - patience
    - test
    - learning_rate
    - anneal
    - truncate
    - batch_size

    Params:
    - parser (argparse.ArgumentParser): The argument parser to which to add the training arguments
    """
    group = parser.add_argument_group('Training Args')
    group.add_argument('-e', '--epochs', default=constants.EPOCHS, type=int,
                       help='The number of epochs for which to train the RNN.')
    group.add_argument('-p', '--patience', default=constants.PATIENCE, type=int,
                       help='The number of examples to train before evaluating loss.')
    group.add_argument('-l', '--learn_rate', default=constants.LEARN_RATE, type=float,
                       help='The learning rate to be used in training.')
    group.add_argument('-a', '--anneal', type=float, default=constants.ANNEAL,
                       help='The minimum possible learning rate.')
    group.add_argument('-t', '--truncate', type=int, default=constants.TRUNCATE,
                       help='The backpropagate truncate value.')
    group.add_argument('-b', '--batch_size', type=int, default=constants.BATCH_SIZE,
                       help='The size of the batches into which to split the training data.')
    group.add_argument('-u', '--num_sequences_to_observe', type=int, default=constants.NUM_SEQUENCES_TO_OBSERVE,
                       help='The number of sequences for the Observer to observe.')
# End of add_train_arguments()


def get_arg(settings: argparse.Namespace, argument: str, as_int: bool = False, as_bool: bool = False,
            as_float: bool = False, check_none: bool = False):
    """
    Retrieves the argument from the given settings namespace, or asks the user to enter one.

    Params:
    - settings (argparse.Namespace): The parsed command-line arguments to the program.
    - argument (str): The argument to retrieve from the settings.
    - as_nt (bool): If True, cast argument to int
    - as_bool (bool): If True, cast argument to bool
    - as_float (bool): If True, cast argument to float
    - check_none (bool): If True, force user to enter argument when none are passed

    Returns:
    - arg (Any): The parsed argument
    """
    arg = None
    if argument in settings:
        arg = vars(settings)[argument]
    if check_none:
        if arg is None:
            arg = input("Specify the value for %s" % argument)
    if as_int:
        arg = int(arg)
    if as_bool:
        if arg.lower() == 'true' or arg.lower() == 't':
            arg = True
        if arg.lower() == 'false' or arg.lower() == 'f':
            arg = False
    if as_float:
        arg = float(arg)
    return arg
# End of get_arg()
