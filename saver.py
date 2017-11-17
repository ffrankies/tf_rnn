'''
Provides an interface for saving and loading various aspects of the
tensorflow model to file.

Copyright (c) 2017 Frank Derry Wanye

Date: 17 November, 2017
'''

import tensorflow as tf
import pickle
import os
import dill

from . import constants
from . import setup
from .layers.performance_layer import Accumulator

class MetaInfo(object):
    '''
    Contains the Meta Info for each model, to be saved.

    Instance Variables:
    - run (int): The current run (variant) of the model
    - model_path (string): The path to the model's directory
    - run_info (dict): Dictionary mapping runs to their information. Dictionary contains the following information:
        - 'dir' (string): The directory in which to save off the model
        - 'epoch' (int): The number of epochs for which the model has been trained
        - 'train_accumulator' (layers.PerformanceLayer.Accumulator): Accumulator for the training partition performance
                                                                     metrics
        - 'valid_accumulator' (layers.PerformanceLayer.Accumulator): Accumulator for the validation partition
                                                                     performance metrics
        - 'test_accumulator' (layers.PerformanceLayer.Accumulator): Accumulator for the test partition performance
                                                                    metrics
    '''

    def __init__(self, logger, settings, max_length):
        '''
        Instantiates a MetaInfo object.

        Params:
        logger (logging.Logger): The logger for the model
        settings (settings.SettingsNamespace): The settings needed for saving and loading the model
        max_length (int): The maximum sequence length for the model's dataset
        '''
        self.run = 0
        self.model_path = create_model_dir(settings.model_name)
        self.run_info = dict()
        self.increment_run(logger, max_length) # Set run to 1
    # End of __init__()

    def latest(self):
        '''
        Grabs the information for the latest run. If the latest run has no information associated with it, creates
        an empty dictionary for it in run_info.

        Return:
        latest_info (dict): The information for the latest run
        '''
        if self.run not in self.run_info.keys():
            self.run_info[self.run] = dict()
        return self.run_info[self.run]
    # End of latest()

    def increment_run(self, logger, max_length):
        '''
        Increments the run number for this model, creates a directory for saving off weights for the new run, and
        creates an initial dictionary for the run information.

        Params:
        logger (logging.Logger): The logger for the model
        max_length (int): The maximum sequence length for the model's dataset
        '''
        self.run += 1
        latest = self.latest()
        latest[constants.DIR] = self.model_path + 'run_' + str(self.run) + '/'
        setup.create_dir(latest[constants.DIR])
        train_accumulator = Accumulator(logger, max_length)
        valid_accumulator = Accumulator(logger, max_length)
        test_accumulator = Accumulator(logger, max_length)
        self.update((-1, train_accumulator, valid_accumulator, test_accumulator))
    # End of increment_run()

    def update(self, new_info):
        '''
        Updates meta info with latest info.

        Params:
        new_info (list/tuple): The new info with which to update the meta info
        - epoch (int): The number of epochs for which the model has been trained
        - train_accumulator (layers.PerformanceLayer.Accumulator): Accumulator for the training partition performance
                                                                     metrics
        - valid_accumulator (layers.PerformanceLayer.Accumulator): Accumulator for the validation partition
                                                                     performance metrics
        - test_accumulator (layers.PerformanceLayer.Accumulator): Accumulator for the test partition performance
                                                                    metrics
        '''
        latest = self.latest()
        epoch, train_acc, valid_acc, test_acc = new_info
        latest[constants.EPOCH] = epoch
        latest[constants.TRAIN] = train_acc
        latest[constants.VALID] = valid_acc
        latest[constants.TEST] = test_acc
        print("Latest run info: %s" % self.latest())
    # End of update()
# End of MetaInfo()

class Saver(object):
    '''
    Class for saving and loading the RNN model.
    '''

    def __init__(self, logger, settings, variables, max_length):
        '''
        Instantiates a Saver object. Has to be called after the model's graph has been created.

        Params:
        logger (logging.Logger): The RNN's model
        settings (settings.SettingsNamespace): The settings needed for saving and loading the model
        variables (ray.experimental.TensorFlowVariables): The tensorflow variables present in the model's graph
        max_length (int): The maximum sequence size in the model's dataset
        '''
        self.logger = logger
        self.logger.debug('Creating a saver object')
        self.settings = settings
        self.meta_path = constants.MODEL_DIR + self.settings.model_name + '/' + constants.META
        self.variables = variables
        self.meta = self.load_meta(settings, max_length)
        if settings.new_model is False: self.load_model(settings.best_model)
    # End of __init__()

    def load_meta(self, settings, max_length):
        '''
        Loads meta information about the model.

        Return:
        meta_info (saver.MetaInfo): The meta info for this model
        max_length (int): The maximum sequence size in the model's dataset
        '''
        if os.path.isfile(self.meta_path):
            with open(self.meta_path, 'rb') as meta_file: # Read current meta info from file
                meta_info = dill.load(meta_file)
        else:
            meta_info = MetaInfo(self.logger, self.settings, max_length) # New model, so create new meta info
        return meta_info
    # End of load_meta()

    def save_meta(self, new_info):
        '''
        Updates meta info with latest info, and saves it.

        Params:
        new_info (list/tuple): The new info with which to update the meta info
        - epoch (int): The number of epochs for which the model has been trained
        - train_accumulator (layers.PerformanceLayer.Accumulator): Accumulator for the training partition performance
                                                                     metrics
        - valid_accumulator (layers.PerformanceLayer.Accumulator): Accumulator for the validation partition
                                                                     performance metrics
        - test_accumulator (layers.PerformanceLayer.Accumulator): Accumulator for the test partition performance
                                                                    metrics
        '''
        self.meta.update(new_info)
        with open(self.meta_path, 'wb') as meta_file:
            dill.dump(obj=self.meta, file=meta_file)
    # End of save_meta()

    def save_model(self, meta_info, best_weights=False):
        '''
        Save the current model's weights in the models/ directory.

        Params:
        meta_info (list/tuple): The new info with which to update the meta info
        - epoch (int): The number of epochs for which the model has been trained
        - train_accumulator (layers.PerformanceLayer.Accumulator): Accumulator for the training partition performance
                                                                     metrics
        - valid_accumulator (layers.PerformanceLayer.Accumulator): Accumulator for the validation partition
                                                                     performance metrics
        - test_accumulator (layers.PerformanceLayer.Accumulator): Accumulator for the test partition performance
                                                                    metrics
        best_weights (boolean): True if the weights correspond to the best accuracy trained so far
        '''
        self.save_meta(meta_info)
        weights = self.variables.get_weights()
        run_dir = self.meta.latest()[constants.DIR]
        if best_weights == True:
            with open(run_dir + constants.BEST_WEIGHTS, 'wb') as weights_file:
                pickle.dump(weights, weights_file)
        with open(run_dir + constants.LATEST_WEIGHTS, 'wb') as weights_file:
            pickle.dump(weights, weights_file)
    # End of save_model()

    def load_model(self, best_weights=False):
        '''
        Load the given model, and updates the meta info accordingly.

        Params:
        best_weights (boolean): True if the model should load the best weights, as opposed to the latest weights
        '''
        if best_weights is True:
            self.logger.info('Loading the weights that produced the best accuracy')
            weights_path = self.meta.latest()[constants.DIR] + constants.BEST_WEIGHTS
        elif best_weights is False:
            self.logger.info('Loading the latest saved weights')
            weights_path = self.meta.latest()[constants.DIR] + constants.LATEST_WEIGHTS
        if os.path.isfile(weights_path) is True:
            with open(weights_path, 'rb') as weights_file:
                weights = pickle.load(weights_file)
                self.variables.set_weights(weights)
        else:
            self.logger.info('Could not load weights: Weights not found')
    # End of load_model()
# End of Saver()

def create_model_dir(model_name):
    '''
    Creates the directory in which to save the model.

    Params:
    model_name (string): The name of the model (gets set to a timestamp if a name is not given)

    Return:
    model_path (string): The path to the created directory
    '''
    model_path = constants.MODEL_DIR + model_name + "/"
    setup.create_dir(model_path)
    return model_path
# End of create_model_dir()
