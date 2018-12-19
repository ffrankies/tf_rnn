"""Contains constants for use within the project.

@since 0.6.3
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
MODEL_DATA_DIR = "data/"
DATASETS_DIR = "datasets/"
TENSORBOARD = "tensorboard/"
RAW_DATA_DIR = "raw_data/"

#########################################
# DATA PARTITION FILES
#########################################
PART_META = "meta_partition.pkl"
PART_TRAIN = "train_partition.pkl"
PART_VALID = "valid_partition.pkl"
PART_TEST = "test_partition.pkl"

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
OBSERVER_FILE = "observer.txt"

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
INPUT_NAMES = ['token']
# TRAIN
BATCH_SIZE = 5
PATIENCE = 5
LEARN_RATE = 0.05
EPOCHS = 10
ANNEAL = 0.5
TRUNCATE = 10
NUM_SEQUENCES_TO_OBSERVE = 10
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
NUM_FEATURES = 1
OUTPUT_INDEXES = [0]
