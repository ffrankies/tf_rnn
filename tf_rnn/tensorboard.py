"""Utility class for creating tensorboard summaries.

@since 0.6.1
"""

import tensorflow as tf

from . import constants

# These imports are only used for type hinting
from .model import RNNBase
from .layers import PerformancePlaceholders


def init_tensorboard(model: RNNBase) -> tuple:
    """Initialize tensorboard event writer, and add variable summaries to it.

    Params:
    - model (model.RNNModel): The model for which summaries should be made.

    Return:
    - summary_writer (tf.summary.FileWriter): The tensorboard summary writer
    - summary_ops (tf.Tensor): The tensorflow operations that produce the tensorboard summaries
    """
    tensorboard_dir = model.run_dir
    with tf.variable_scope(constants.TBOARD_SUMMARY):
        max_timesteps = model.dataset.max_length
        summarize_partition(constants.TBOARD_TRAIN, model.train_performance, max_timesteps)
        summarize_partition(constants.TBOARD_VALID, model.validation_performance, max_timesteps)
        # summarize_partition(constants.TBOARD_TEST, model.test_performance, max_timesteps)
    merged_summary_ops = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir, graph=model.session.graph)
    return writer, merged_summary_ops
# End of init_tensorboard()


def summarize_partition(name: str, performance_placeholders: PerformancePlaceholders, max_timesteps: int):
    """Summarizes the performance evaluations for a given partition.

    Params:
    - name (string): The name of the partition
    - performance_placeholders (layers.PerformancePlaceholders): The placeholders for the performance evaluations
    - max_timesteps (int): The maximum sequence length in the dataset
    """
    with tf.variable_scope(name):
        tf.summary.scalar(constants.TBOARD_LOSS, performance_placeholders.average_loss)
        # tf.summary.scalar(constants.TBOARD_ACCURACY, performance_placeholders.average_accuracy)
        summarize_timesteps(performance_placeholders.timestep_accuracies, max_timesteps)
# End of summarize_partition()


def summarize_timesteps(timestep_accuracy_op: tf.placeholder, max_timesteps: int):
    """Summarizes the average accuracy for each timestep for a given partition.

    Params:
    - timestep_accuracy_op (tf.placeholder): The average accuracies for each timestep
    - max_timesteps (int): The maximum sequence length in the dataset
    """
    timestep_accuracies = tf.unstack(timestep_accuracy_op)
    timestep_accuracies = timestep_accuracies[:max_timesteps]
    for timestep, accuracy in enumerate(timestep_accuracies):
        tf.summary.scalar(constants.TBOARD_TIMESTEP_ACCURACY + str(timestep+1), accuracy)
# End of summarize_timesteps()
