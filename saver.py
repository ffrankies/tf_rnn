"""
Provides an interface for saving and loading various aspects of the 
tensorflow model to file.

Copyright (c) 2017 Frank Derry Wanye

Date: 14 November, 2017
"""

import tensorflow as tf
import pickle
import os

from . import constants
from . import setup

class MetaInfo(object):
    '''
    Contains the Meta Info for each model, to be saved.

    Instance Variables:
    - run (int): The current run (variant) of the model
    - epoch (int): The number of epochs for which the model has been trained
    - model_path (string): The path to the model's directory
    - run_dir (string): The directory in which to save off the model
    '''

    def __init__(self, settings):
        '''
        Instantiates a MetaInfo object.

        Params:
        settings (settings.SettingsNamespace): The settings needed for saving and loading the model
        '''
        self.run = 0
        self.epoch = 0
        self.model_path = self.create_model_dir(settings.model_name)
        self.increment_run() # Set run to 1
    # End of __init__()

    def create_model_dir(self, model_name):
        """
        Creates the directory in which to save the model.

        Params:
        model_name (string): The name of the model (gets set to a timestamp if a name is not given)

        Return:
        model_path (string): The path to the created directory
        """
        model_path = constants.MODEL_DIR + model_name + "/"
        setup.create_dir(model_path)
        return model_path
    # End of create_model_dir()

    def increment_run(self):
        '''
        Increments the run number for this model, and updates the run_dir with the new run number.

        Creates the following instance variables (if they don't already exist):
        - run_dir (string): The directory in which to save the model's weights and graphs
        '''
        self.run += 1
        self.run_dir = self.model_path + 'run_' + str(self.run) + '/'
        setup.create_dir(self.run_dir)
    # End of increment_run()
# End of MetaInfo()

class Saver(object):
    '''
    Class for saving and loading the RNN model.
    '''

    def __init__(self, logger, settings, variables):
        '''
        Instantiates a Saver object. Has to be called after the model's graph has been created.

        Params:
        logger (logging.Logger): The RNN's model
        settings (settings.SettingsNamespace): The settings needed for saving and loading the model
        variables (ray.experimental.TensorFlowVariables): The tensorflow variables present in the model's graph
        '''
        self.logger = logger
        self.logger.debug('Creating a saver object')
        self.settings = settings
        self.meta_path = constants.MODEL_DIR + self.settings.model_name + '/' + constants.META
        self.variables = variables
        self.meta = self.load_meta(settings)
    # End of __init__()

    def load_meta(self, settings):
        '''
        Loads meta information about the model.

        Return:
        meta_info (saver.MetaInfo): The meta info for this model
        '''
        if os.path.isfile(self.meta_path):
            with open(meta_path, 'rb') as meta_file: # Read current meta info from file
                meta_info = pickle.load(meta_file)
        else: 
            meta_info = MetaInfo(self.settings) # New model, so create new meta info
            self.save_meta(0, meta_info)
        return meta_info
    # End of load_meta()

    def save_meta(self, epoch, best_weights=False):
        '''
        Saves the current meta info to the model's current run directory. 
        If the model's current iteration has the best weights, also saves it to the model's home directory.

        Params:
        epoch (int): The number of epochs for which the model has trained
        best_weights (boolean): True if the weights correspond to the best accuracy trained so far
        '''
        self.meta.epoch = epoch
        with open(meta_info.run_dir + constants.META, 'wb') as meta_file:
            pickle.dump(self.meta, meta_file)
        if best_weights is True:
            with open(meta_path, 'wb') as meta_file:
                pickle.dump(self.meta, meta_file)
    # End of save_meta()

    def save_model(self, epoch, best_weights=False):
        '''
        Save the current model's weights in the models/ directory.

        Params:
        epoch (int): The number of epochs for which the model has trained
        best_weights (boolean): True if the weights correspond to the best accuracy trained so far
        '''
        self.save_meta(epoch, best_weights)
        weights = self.variables.get_weights()
        if best_weights is True:
            with open(self.meta.model_path + constants.LATEST_WEIGHTS, 'wb') as weights_file:
                pickle.dump(weights, weights_file)
        with open(self.meta.run_dir + constants.WEIGHTS, 'wb') as weights_file:
            pickle.dump(weights, weights_file)
    # End of save_model()

    def load_model(self, best_weights=False):
        """
        Load the given model, and updates the meta info accordingly.

        Params:
        best_weights (boolean): True if the model should load the best weights, as opposed to the latest weights
        """
        if best_weights is True:
            weights_path = self.meta.model_path + constants.LATEST_WEIGHTS
        elif best_weights is False:
            weights_path = self.meta.run_dir + constants.WEIGHTS
        with open(weights_path, "rb") as weights_file:
            weights = pickle.load(weights_file)
            self.variables.set_weights(weights)
    # End of load_model()
# End of Saver()
    