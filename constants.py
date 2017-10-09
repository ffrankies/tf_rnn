"""
Contains constants for use within the project.

Copyright (c) 2017 Frank Derry Wanye

Date: 8 October, 2017
"""

import time
import math

##########################################
# TOKENS
##########################################
UNKNOWN = "UNKNOWN_TOKEN"
PAD = "PADDING_TOKEN"
SENT_START = "SENTENCE_START"
SENT_END = "SENTENCE_END"
PARA_START = "PARAGRAPH_START"
PARA_END = "PARAGRAPH_END"
STORY_START = "STORY_START"
STORY_END = "STORY_END"
CARRIAGE_RETURN = "CARRIAGE_RETURN"
SPACE = "SPACE_TOKEN"
START_TOKEN = '['
END_TOKEN = ']'

#########################################
# DEFAULT DIRECTORIES
#########################################
MODEL_DIR = "models/"
DATASETS_DIR = "datasets/"
TENSORBOARD = "tensorboard/"
RAW_DATA_DIR = "raw_data/"

#########################################
# VARIABLE SCOPES FOR TENSORBOARD
#########################################
INPUT = "input_layer"
HIDDEN = "hidden_layers"
OUTPUT = "output_layer"
PERFORMANCE = "performance_evaluation"
LOSS_CALC = "loss_calculation"
ACCURACY = "accuracy_calculation"
TRAINING = "network_training"

#########################################
# TENSORBOARD SUMMARY NAMES
#########################################


#########################################
# DEFAULT FILENAMES
#########################################
LATEST_WEIGHTS = "latest_weights.pkl"
WEIGHTS = "/weights.pkl"
PLOT = "/loss_plot.png"
META = "meta.pkl"

#########################################
# ARG KEY NAMES
#########################################
# GENERAL
MODEL_NAME_STR = 'model_name'
# LOGGING
LOG_NAME_STR = 'log_name'
LOG_DIR_STR = 'log_dir'
LOG_FILENAME_STR = 'log_filename'
# RNN
DATASET_STR = 'dataset'
EMBED_SIZE_STR = 'embed_size'
HIDDEN_SIZE_STR = 'hidden_size'
# TRAIN
BATCH_SIZE_STR = 'batch_size'
PATIENCE_STR = 'patience'
LEARN_RATE_STR = 'learn_rate'
EPOCHS_STR = 'epochs'
ANNEAL_STR = 'anneal'
TRUNCATE_STR = 'truncate'
# DATA
CONFIG_FILE_STR = 'config_file'
RAW_DATA_STR = 'raw_data'
DATASET_NAME_STR = 'dataset_name'
SOURCE_TYPE_STR = 'source_type'
VOCAB_SIZE_STR = 'vocab_size'
NUM_ROWS_STR = 'num_rows'
NUM_EXAMPLES_STR = 'num_examples'
TYPE_STR = 'type'
MODE_STR = 'mode'
TOKEN_LEVEL_STR = 'token_level'

#########################################
# ARG CHOICES
#########################################
TYPE_CHOICES = ['text', 'number']
MODE_CHOICES = ['sentences', 'paragraphs', 'stories']
TOKEN_LEVEL_CHOICES = ['words', 'characters']

#########################################
# ARG DEFAULTS
#########################################
# GENERAL
MODEL_NAME = time.strftime("%d%m%y%H")
# LOGGING
LOG_NAME = 'TERRY'
LOG_DIR = 'logging/'
LOG_FILENAME = 'logging.log'
# RNN
DATASET = 'test.pkl'
EMBED_SIZE = 100
HIDDEN_SIZE = 100
# TRAIN
BATCH_SIZE = 5
PATIENCE = 100000 # Probably going to be deprecated
LEARN_RATE = 0.005
EPOCHS = 10
ANNEAL = 0.00001
TRUNCATE = 10
# DATA
CONFIG_FILE = None
RAW_DATA = 'stories.csv'
DATASET_NAME = 'stories.pkl'
SOURCE_TYPE = 'csv'
VOCAB_SIZE = 200
NUM_ROWS = math.inf
NUM_EXAMPLES = None # list[:None] returns all elements in list
TYPE = TYPE_CHOICES[0]
MODE = MODE_CHOICES[0]
TOKEN_LEVEL = TOKEN_LEVEL_CHOICES[0]

#########################################
# ARG DEFAULTS
#########################################
GENERAL_ARGS = {
    MODEL_NAME_STR : MODEL_NAME }
LOGGING_ARGS = {
    LOG_NAME_STR : LOG_NAME,
    LOG_DIR_STR : LOG_DIR,
    LOG_FILENAME_STR : LOG_FILENAME }
RNN_ARGS = {
    DATASET_STR : DATASET,
    EMBED_SIZE_STR : EMBED_SIZE,
    HIDDEN_SIZE_STR : HIDDEN_SIZE }
TRAIN_ARGS = {
    BATCH_SIZE_STR : BATCH_SIZE,
    PATIENCE_STR : PATIENCE,
    LEARN_RATE_STR : LEARN_RATE,
    EPOCHS_STR : EPOCHS,
    ANNEAL_STR : ANNEAL,
    TRUNCATE_STR : TRUNCATE }
DATA_ARGS = {
    # Config file not added here because it's not needed to create the dataset
    RAW_DATA_STR : RAW_DATA,
    DATASET_NAME_STR : DATASET_NAME,
    SOURCE_TYPE_STR : SOURCE_TYPE,
    VOCAB_SIZE_STR : VOCAB_SIZE,
    NUM_ROWS_STR : NUM_ROWS,
    NUM_EXAMPLES_STR : NUM_EXAMPLES,
    TYPE_STR : TYPE,
    MODE_STR : MODE,
    TOKEN_LEVEL_STR : TOKEN_LEVEL }