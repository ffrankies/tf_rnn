"""Contains constants for use within the project.

@since 0.6.1
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
# PLOT NAMES
#########################################
PLT_TRAIN_LOSS = "training_loss.png"
PLT_TRAIN_ACCURACY = "training_accuracy.png"
PLT_TIMESTEP_ACCURACY = "timestep_accuracy.png"
PLT_CONFUSION_MATRIX = "confusion_matrix.png"
PLT_F1_SCORE = "f1_score.png"

#########################################
# META INFO KEYS
#########################################
EPOCH = 'epoch'
DIR = 'dir'
TRAIN = 'training_accumulator'
VALID = 'validation_accumulator'
TEST = 'test_accumulator'
METRICS = 'performance_metrics'

#########################################
# VARIABLE SCOPES FOR TENSORBOARD
#########################################
INPUT = "input_layer"
HIDDEN = "hidden_layers"
OUTPUT = "output_layer"
LOSS_LAYER = "loss_layer"
TRAINING = "training_layer"
TRAINING_PERFORMANCE = "training_performance"
VALIDATION_PERFORMANCE = "validation_performance"
TEST_PERFORMANCE = "test_performance"
BATCH_LOSS_CALC = "minibatch_loss_calculation"
LOSS_CALC = "loss_calculation"
PREDICTIONS_MASK = "predict_and_mask"
ACCURACY = "accuracy_calculation"
TIMESTEP_ACCURACY = "timestep_accuracy_calculation"

#########################################
# TENSORBOARD SUMMARY NAMES
#########################################
TBOARD_SUMMARY = 'summaries'
TBOARD_LOSS = 'loss'
TBOARD_ACCURACY = 'accuracy'
TBOARD_TIMESTEP_ACCURACY = 'accuracy_at_timestep'
TBOARD_TRAIN = 'training'
TBOARD_VALID = 'validation'
TBOARD_TEST = 'test'

#########################################
# DEFAULT FILENAMES
#########################################
LATEST_WEIGHTS = "latest_weights.pkl"
BEST_WEIGHTS = "best_weights.pkl"
PLOT = "loss_plot.png"
META = "meta.pkl"

#########################################
# ARG KEY NAMES
#########################################
# GENERAL
MODEL_NAME_STR = 'model_name'
NEW_MODEL_STR = 'new_model'
BEST_MODEL_STR = 'best_model'
# LOGGING
LOG_NAME_STR = 'log_name'
LOG_DIR_STR = 'log_dir'
LOG_FILENAME_STR = 'log_filename'
LOG_LEVEL_STR = 'log_level'
# RNN
DATASET_STR = 'dataset'
EMBED_SIZE_STR = 'embed_size'
HIDDEN_SIZE_STR = 'hidden_size'
LAYERS_STR = 'layers'
DROPOUT_STR = 'dropout'
NUM_FEATURES_STR = 'num_features'
INPUT_NAMES_STR = 'input_names'
SHUFFLE_SEED_STR = 'shuffle_seed'
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
ADD_START_TOKEN_STR = 'add_start_token'
ADD_END_TOKEN_STR = 'add_end_token'

#########################################
# LOGGING LEVELS
#########################################
ERROR = 'Error'
INFO = 'Info'
DEBUG = 'Debug'
TRACE = 'Trace'

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
NEW_MODEL = False
BEST_MODEL = False
# LOGGING
LOG_NAME = 'TERRY'
LOG_DIR = 'logging/'
LOG_FILENAME = 'logging.log'
LOG_LEVEL = 'info'
# RNN
DATASET = 'test.pkl'
EMBED_SIZE = 100
HIDDEN_SIZE = 100
LAYERS = 2
DROPOUT = 0.5
NUM_FEATURES = 1
INPUT_NAMES = ['token']
# TRAIN
BATCH_SIZE = 5
PATIENCE = 5
LEARN_RATE = 0.05
EPOCHS = 10
ANNEAL = 0.5
TRUNCATE = 10
# DATA
CONFIG_FILE = None
RAW_DATA = 'stories.csv'
DATASET_NAME = 'stories.pkl'
SOURCE_TYPE = 'csv'
VOCAB_SIZE = None
NUM_ROWS = math.inf
NUM_EXAMPLES = None  # list[:None] returns all elements in list
TYPE = TYPE_CHOICES[0]
MODE = MODE_CHOICES[0]
TOKEN_LEVEL = TOKEN_LEVEL_CHOICES[0]
ADD_START_TOKEN = False
ADD_END_TOKEN = False
SHUFFLE_SEED = 0.2345

#########################################
# ARG DEFAULTS
#########################################
GENERAL_ARGS = {
    MODEL_NAME_STR: MODEL_NAME,
    NEW_MODEL_STR: NEW_MODEL,
    BEST_MODEL_STR: BEST_MODEL
    }
LOGGING_ARGS = {
    LOG_NAME_STR: LOG_NAME,
    LOG_DIR_STR: LOG_DIR,
    LOG_FILENAME_STR: LOG_FILENAME,
    LOG_LEVEL_STR: LOG_LEVEL
    }
RNN_ARGS = {
    DATASET_STR: DATASET,
    EMBED_SIZE_STR: EMBED_SIZE,
    HIDDEN_SIZE_STR: HIDDEN_SIZE,
    LAYERS_STR: LAYERS,
    DROPOUT_STR: DROPOUT,
    NUM_FEATURES_STR: NUM_FEATURES,
    INPUT_NAMES_STR: INPUT_NAMES
    }
TRAIN_ARGS = {
    BATCH_SIZE_STR: BATCH_SIZE,
    PATIENCE_STR: PATIENCE,
    LEARN_RATE_STR: LEARN_RATE,
    EPOCHS_STR: EPOCHS,
    ANNEAL_STR: ANNEAL,
    TRUNCATE_STR: TRUNCATE
    }
DATA_ARGS = {
    # Config file not added here because it's not needed to create the dataset
    RAW_DATA_STR: RAW_DATA,
    DATASET_NAME_STR: DATASET_NAME,
    SOURCE_TYPE_STR: SOURCE_TYPE,
    VOCAB_SIZE_STR: VOCAB_SIZE,
    NUM_ROWS_STR: NUM_ROWS,
    NUM_EXAMPLES_STR: NUM_EXAMPLES,
    TYPE_STR: TYPE,
    MODE_STR: MODE,
    TOKEN_LEVEL_STR: TOKEN_LEVEL,
    ADD_START_TOKEN_STR: ADD_START_TOKEN,
    ADD_END_TOKEN_STR: ADD_END_TOKEN,
    SHUFFLE_SEED_STR: SHUFFLE_SEED
    }
