"""A Python3 collection of Namespaces for the different model settings.

@since 0.6.3
"""
import argparse  # I want to see if I can get rid of this - it's only use is type hinting

import yaml

from . import constants
from . import cmd_arg_parser
from .utils import Singleton


class SettingsNamespace(object):
    """An extensible Namespace object for easy accessibility to inner variables.
    """

    def __init__(self, dictionary: dict):
        """
        Creates a SettingsNamespace out of the given dictionary object.

        Params:
        dictionary (dict): The dictionary to convert to a Namespace
        """
        self.__dict__.update(dictionary)
    # End of __init__()
# End of SettingsNamespace


class GeneralSettings(SettingsNamespace):
    """A namespace object for the general RNN settings.
    """

    def __init__(self, parameters: dict):
        """Creates default parameters for general settings, and then updates them with the given dictionary.
        
        TODO:
        - Change parameters names to better ones
        
        Params:
        - parameters (dict[str, Any]): The parameters with which to update the general settings 
        """
        self.model_name: str = constants.MODEL_NAME
        self.new_model: bool = constants.NEW_MODEL  # TODO: rename
        self.best_model: bool = constants.BEST_MODEL  # TODO: rename
    # End of __init__()
# End of GeneralSettings


class LoggingSettings(SettingsNamespace):
    """A namespace object for storing the settings for the RNN's logger.
    """

    def __init__(self, parameters: dict):
        """Creates default parameters for logging settings, and then updates them with the given dictionary.

        TODO:
        - Get rid of unused logging parameters

        Params:
        - parameters (dict[str, Any]): The parameters with which to update the logging settings 
        """
        self.log_name: str = constants.LOG_NAME  # TODO: possibly remove
        self.log_dir: str = constants.LOG_DIR  # TODO: possibly remove
        self.log_filename: str = constants.LOG_FILENAME  # TODO: possibly remove
        self.log_level: str = constants.LOG_LEVEL  # TODO: possibly remove
    # End of __init__()
# End of LoggingSettings


class RNNSettings(SettingsNamespace):
    """A namespace object for storing the RNN settings.
    """

    def __init__(self, parameters: dict):
        """Creates default parameters for the RNN settings, and then updates them with the given dictionary.

        TODO:
        - Get rid of unused RNN logging parameters

        Params:
        - parameters (dict[str, Any]): The parameters with which to update the RNN settings 
        """
        self.dataset: str = constants.DATASET
        self.embed_size: int = constants.EMBED_SIZE
        self.hidden_size: int = constants.HIDDEN_SIZE
        self.layers: int = constants.LAYERS  # TODO: possibly rename
        self.dropout: float = constants.DROPOUT  # TODO: possibly rename
        self.input_names: list = constants.INPUT_NAMES
        self.shuffle_seed: float = constants.SHUFFLE_SEED  # TODO: remove
    # End of __init__()
# End of RNNSettings


class TrainingSettings(SettingsNamespace):
    """A namespace object for RNN training settings.
    """

    def __init__(self, parameters: dict):
        """Creates default parameters for RNN training settings, and then updates them with the given dictionary.

        TODO:
        - Get rid of unused training parameters

        Params:
        - parameters (dict[str, Any]): The parameters with which to update the training settings 
        """
        self.batch_size: int = constants.BATCH_SIZE
        self.patience: int = constants.PATIENCE
        self.learn_rate: float = constants.LEARN_RATE
        self.epochs: int = constants.EPOCHS
        self.anneal: float = constants.ANNEAL  # TODO: possibly remove
        self.truncate: int = constants.TRUNCATE
    # End of __init__()
# End of TrainingSettings


class DatasetSettings(SettingsNamespace):
    """A namespace object for logging dataset settings.
    """

    def __init__(self, parameters: dict):
        """Creates default parameters for dataset settings, and then updates them with the given dictionary.

        TODO:
        - Get rid of unused dataset parameters
        - Figure out why the config file is one of the dataset settings

        Params:
        - parameters (dict[str, Any]): The parameters with which to update the dataset settings 
        """
        self.config_file: str = constants.CONFIG_FILE  # Why is this here?
        self.raw_data: str = constants.RAW_DATA
        self.dataset_name: str = constants.DATASET_NAME
        self.source_type: str = constants.SOURCE_TYPE
        self.vocab_size: int = constants.VOCAB_SIZE
        self.num_rows: int = constants.NUM_ROWS  # TODO: rename
        self.num_examples: int = constants.NUM_EXAMPLES  # TODO: possibly rename
        self.type: str = constants.TYPE  # TODO: rename
        self.mode: str = constants.MODE  # TODO: rename
        self.token_level: str = constants.TOKEN_LEVEL
        self.add_start_token: bool = constants.ADD_START_TOKEN
        self.add_end_token: bool = constants.ADD_END_TOKEN
    # End of __init__()
# End of DatasetSettings


class Settings(object, metaclass=Singleton):
    """Collection of Namespaces that separates settings into groups based on their function.
    Settings are either created from a YAML config file, or from an argparse.Namespace().
    Either way, the YAML file is, for simplicity, passed into the program as a command_line argument.

    The object is a singleton to ensure that only one instance of settings is ever active.
    """

    def __init__(self, dataset_only: bool = False):
        """Creates the Settings class.
        The class is created from the provided config file if it is supplied.
        If 'options' is passed in as a command-line argument, then the class is created from the command-line
        arguments passed into it.

        Params:
        - dataset_only (bool): True if only settings for the dataset should be provided
        """
        config_dicts = self.get_config_dicts(dataset_only)
        self.general = GeneralSettings(config_dicts[0])
        self.logging = LoggingSettings(config_dicts[1])
        self.rnn = RNNSettings(config_dicts[2])
        self.train = TrainingSettings(config_dicts[3])
        self.data = DatasetSettings(config_dicts[4])
    # End of __init__()

    def __str__(self) -> str:
        """Creates a string representation of the Settings object.

        Return:
        - settings_string (str): A string representation of the Namespaces comprising this Settings object.
        """
        settings_dicts = {
            'general': vars(self.general),
            'logging': vars(self.logging),
            'rnn': vars(self.rnn),
            'train': vars(self.train),
            'data': vars(self.data)
            }
        return str(settings_dicts)
    # End of __str__()

    def get_config_dicts(self, dataset_only: bool) -> tuple:
        """Obtains the configuration dictionaries from either the config file or the command-line arguments.

        Params:
        - dataset_only (bool): True if only settings for the dataset should be provided

        Return:
        - config_dicts (tuple): Separate dictionaries for each configuration category.
        """
        args = cmd_arg_parser.parse_arguments(dataset_only)
        if 'config_file' in args and args.config_file is not None:
            config_dicts = self.parse_config_yml(args.config_file)
            config_dicts = self.set_defaults(config_dicts)
        else:
            config_dicts = self.parse_config_args(args)
        return config_dicts
    # End of get_config_dicts()

    def parse_config_yml(self, config_file: str) -> str:
        """Parses the YAML config file into multiple dictionaries.

        Params:
        - config_file (string): The path to the config file from which to load settings

        Return:
        - config_dicts (tuple<dict>): A list of dictionaries, each representing a different section of the settings
        """
        yaml_settings = self.read_yml(config_file)
        # Break settings into categories
        general = yaml_settings.get('general')
        logging = yaml_settings.get('logging')
        rnn = yaml_settings.get('rnn')
        train = yaml_settings.get('train')
        data = yaml_settings.get('data')
        return general, logging, rnn, train, data
    # End of parse_config_yml()

    def read_yml(self, yml_file: str) -> dict:
        """Reads the contents of a YAML file and returns the file contents as a dictionary.

        Params:
        - yml_file (string): The path to the YAML file

        Return:
        - yml_contents (dict<string, any>): A dictionary containing the contents of the YAML file
        """
        with open(yml_file, 'r') as stream:
            yml_contents = yaml.safe_load(stream)
        return yml_contents
    # End of read_yml()

    def set_defaults(self, dictionaries: tuple) -> tuple:
        """Sets None values in dictionaries to default values.
        This is unnecessary when command-line options are passed in, since defaults get set automatically for those.

        Params:
        - dictionaries (tuple<dict>): The dictionaries whose defaults should be set

        Return:
        - dictionaries_with_defaults (tuple<dict>): The dictionaries with default values in place of None values
        """
        general = self.set_default_values(dictionaries[0], constants.GENERAL_ARGS)
        logging = self.set_default_values(dictionaries[1], constants.LOGGING_ARGS)
        rnn = self.set_default_values(dictionaries[2], constants.RNN_ARGS)
        train = self.set_default_values(dictionaries[3], constants.TRAIN_ARGS)
        data = self.set_default_values(dictionaries[4], constants.DATA_ARGS)
        return general, logging, rnn, train, data
    # End of set_defaults()

    def set_default_values(self, user_dict: dict, default_dict: dict) -> dict:
        """Sets None values in user_dict to default values from default_dict.
        Also adds default values for parameters that haven't been named in the user_dict.

        Params:
        - user_dict (dict<string, any>): The dictionary containing user-provided values
        - default_dict (dict<string, any>): The dictionary containing default values

        Return:
        - user_dict_with_defaults (dict<string,any>): The user_dict with None values replaced with defaults
        """
        changed_dict = dict()
        if user_dict is not None:
            for key, value in user_dict.items():
                if value is None:
                    changed_dict[key] = default_dict[key]
                else:
                    changed_dict[key] = value
            for key, value in default_dict.items():
                if key not in changed_dict.keys():
                    changed_dict[key] = value
        return changed_dict
    # End of set_default_values()

    def parse_config_args(self, args: argparse.Namespace) -> tuple:
        """Parses the contents of the command-line arguments into multiple dictionaries.

        Params:
        - args (argparse.Namespace): The command-line arguments passed in to the program

        Return:
        - arg_dictionaries (tuple<dict>): A list of dictionaries, each representing a different section of the settings
        """
        # Create dicts for individual categories
        general = self.get_arg_subset(args, constants.GENERAL_ARGS.keys())
        logging = self.get_arg_subset(args, constants.LOGGING_ARGS.keys())
        rnn = self.get_arg_subset(args, constants.RNN_ARGS.keys())
        train = self.get_arg_subset(args, constants.TRAIN_ARGS.keys())
        data = self.get_arg_subset(args, constants.DATA_ARGS.keys())
        return general, logging, rnn, train, data
    # End of parse_config_args()

    def get_arg_subset(self, args: argparse.Namespace, arg_keys: list) -> dict:
        """Creates a dictionary containing the given keys from the command_line args.

        Params:
        - args (argparse.Namespace): The command-line arguments passed in to the program
        - arg_keys (list<str>): The keys that form this argument group

        Return:
        - arg_subset(dict<string, any>): The general settings passed into the program
        """
        arg_dict = vars(args)
        new_dict = dict()
        for arg_key in arg_keys:
            new_dict[arg_key] = arg_dict.get(arg_key)
        return new_dict
    # End of get_arg_subset()
# End of Settings()
