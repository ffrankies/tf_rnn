"""
A Python3 collection of Namespaces for the different model settings.

Copyright (c) 2017 Frank Derry Wanye

Date: 16 September, 2017
"""

import yaml
from . import setup
from . import constants

class SettingsNamespace(object):
    """
    A Namespace object for easy accessibility to inner variables.
    """

    def __init__(self, dictionary):
        """
        Creates a SettingsNamespace out of the given dictionary object.

        Params:
        dictionary (dict): The dictionary to convert to a Namespace
        """
        self.__dict__.update(dictionary)
    # End of __init__()
# End of SettingsNamespace

class Settings(object):
    """
    Collection of Namespaces that separates settings into groups based on their function.
    Settings are either created from a YAML config file, or from an argparse.Namespace().
    Either way, the YAML file is, for simplicity, passed into the program as a command_line argument.
    """

    def __init__(self):
        """
        Creates the Settings class.
        The class is created from the provided config file if it is supplied.
        If 'options' is passed in as a command-line argument, then the class is created from the command-line 
        arguments passed into it.
        """
        config_dicts = self.get_config_dicts()
        self.general = SettingsNamespace(config_dicts[0])
        self.logging = SettingsNamespace(config_dicts[1])
        self.rnn = SettingsNamespace(config_dicts[2])
        self.train = SettingsNamespace(config_dicts[3])
        self.data = SettingsNamespace(config_dicts[4])
    # End of __init__()

    def __str__(self):
        """
        Creates a string representation of the Settings object.

        Return:
        string: A string representation of the Namespaces comprising this Settings object.
        """
        settings_dicts = {
            'general' : vars(self.general),
            'logging' : vars(self.logging),
            'rnn' : vars(self.rnn),
            'train' : vars(self.train),
            'data' : vars(self.data) }
        return str(settings_dicts)
    # End of __str__()

    def get_config_dicts(self):
        """
        Obtains the configuration dictionaries from either the config file or the command-line arguments.
        
        Return:
        tuple(dict): Separate dictionaries for each configuration category.
        """
        args = setup.parse_arguments()
        if 'config_file' in args:
            config_dicts = self.parse_config_yml(args.config_file)
            config_dicts = self.set_defaults(config_dicts)
        else:
            config_dicts = self.parse_config_args(args)
        return config_dicts
    # End of get_config_dicts()

    def parse_config_yml(self, config_file):
        """
        Parses the YAML config file into multiple dictionaries.

        Params:
        config_file (string): The path to the config file from which to load settings

        Return:
        tuple(dict): A list of dictionaries, each representing a different section of the settings
        """
        yaml_settings = self.read_yml(config_file)
        # Break settings into categories
        general = yaml_settings.get("general")
        logging = yaml_settings.get("logging")
        rnn = yaml_settings.get("rnn")
        train = yaml_settings.get("train")
        data = yaml_settings.get("data")
        return general, logging, rnn, train, data
    # End of parse_config_yml()

    def read_yml(self, yml_file):
        """
        Reads the contents of a YAML file and returns the file contents as a dictionary.

        Params:
        yml_file (string): The path to the YAML file

        Return:
        dict(string, any): A dictionary containing the contents of the YAML file
        """
        with open(yml_file, "r") as stream:
            yml_contents = yaml.load(stream)
        return yml_contents
    # End of read_Yml()

    def set_defaults(self, dictionaries):
        """
        Sets None values in dictionaries to default values.
        This is unnecessary when command-line options are passed in, since defaults get set automatically for those.

        Params:
        dictionaries (tuple(dict)): The dictionaries whose defaults should be set

        Return:
        tuple(dict): The dictionaries with default values in place of None values
        """
        general = self.set_default_values(dictionaries[0], constants.GENERAL_ARGS)
        logging = self.set_default_values(dictionaries[1], constants.LOGGING_ARGS)
        rnn = self.set_default_values(dictionaries[2], constants.RNN_ARGS)
        train = self.set_default_values(dictionaries[3], constants.TRAIN_ARGS)
        data = self.set_default_values(dictionaries[4], constants.DATA_ARGS)
        return general, logging, rnn, train, data
    # End of set_defaults()

    def set_default_values(self, user_dict, default_dict):
        """
        Sets None values in user_dict to default values from default_dict.
        Also adds default values for parameters that haven't been named in the user_dict.

        Params:
        user_dict (dict): The dictionary containing user-provided values
        default_dict (dict): The dictionary containing default values

        Return:
        dict: The user_dict with None values replaced with defaults
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

    def parse_config_args(self, args):
        """
        Parses the contents of the command-line arguments into multiple dictionaries.

        Params:
        args (argparse.Namespace): The command-line arguments passed in to the program

        Return:
        tuple(dict): A list of dictionaries, each representing a different section of the settings
        """
        # Create dicts for individual categories
        general = self.get_arg_subset(args, constants.GENERAL_ARGS.keys())
        logging = self.get_arg_subset(args, constants.LOGGING_ARGS.keys())
        rnn = self.get_arg_subset(args, constants.RNN_ARGS.keys())
        train = self.get_arg_subset(args, constants.TRAIN_ARGS.keys())
        data = self.get_arg_subset(args, constants.DATA_ARGS.keys())
        return general, logging, rnn, train, data
    # End of parse_config_args()

    def get_arg_subset(self, args, arg_keys):
        """
        Creates a dictionary containing the given keys from the command_line args.

        Params:
        args (argparse.Namespace): The command-line arguments passed in to the program
        arg_keys (list): The keys that form this argument group

        Return:
        dict(string, any): The general settings passed into the program
        """
        arg_dict = vars(args)
        new_dict = dict()
        for arg_key in arg_keys:
            new_dict[arg_key] = arg_dict.get(arg_key)
        return new_dict
    # End of get_arg_subset
# End of Settings