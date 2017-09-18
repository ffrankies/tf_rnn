"""
Provides an interface for saving and loading various aspects of the 
tensorflow model to file.

Copyright (c) 2017 Frank Derry Wanye

Date: 10 September, 2017
"""

import tensorflow as tf
import pickle
import os

from .constants import MODEL_DIR
from . import constants
from . import setup

def create_model_dir(model_name):
    """
    Creates the directory in which to save the model.

    Params:
    model_name (string): The name of the model (gets set to a timestamp if a name is not given)

    Return:
    string: The path to the created directory
    """
    model_path = MODEL_DIR + model_name + "/"
    setup.create_dir(model_path)
    return model_path
# End of create_model_dir()

def save_model(model):
    """
    Save the current model's weights in the models/ directory.

    :type model: RNNModel()
    :param model: The model to save.
    """
    weights = model.variables.get_weights()
    with open(model.model_path + constants.LATEST_WEIGHTS, "wb") as weights_file:
        pickle.dump(weights, weights_file)
    with open(model.model_path + model.run_dir + "/" + constants.WEIGHTS, "wb") as weights_file:
        pickle.dump(weights, weights_file)
# End of save_model()

def load_model(model, model_name_or_timestamp):
    """
    Load the model contained in the given directory.

    :type model: RNNModel()
    :param model: The model to which to restore previous data.

    :type model_name_or_timestamp: String
    :param model_name_or_timestamp: the name or timestamp of the model to load
    """
    weights_path = MODEL_DIR + model_name_or_timestamp + "/" + constants.LATEST_WEIGHTS
    with open(weights_path, "rb") as weights_file:
        weights = pickle.load(weights_file)
        model.variables.set_weights(weights)
# End of load_model()

def load_meta(model_path):
    """
    Loads meta information about the model. Currently, it only loads in the latest run folder.

    :type model_path: String
    :param model_path: the path to the directory where model information will be saved.

    :type return: String
    :param return: the save directory for snapshot model data
    """
    meta_path = model_path + constants.META
    if os.path.isfile(meta_path):
        # Read current run_num from meta file
        with open(meta_path, "rb") as meta_file:
            run_num = pickle.load(meta_file)
    else: 
        run_num = 1
    # Write next run_num to meta file
    with open(meta_path, "wb") as meta_file:
        pickle.dump(run_num+1, meta_file)
    run_dir = "run_" + str(run_num)
    setup.create_dir(model_path + run_dir + "/")
    return run_dir
# End of load_meta()
    