"""
Utility class for creating tensorboard summaries.

9 November, 2017
"""
import tensorflow as tf
from . import constants

def init_tensorboard(model):
    """
    Initialize tensorboard event writer, and add variable summaries to it.

    :type model: RNNModel()
    :param model: the model for which summaries should be made.

    :type return: tuple(tf.summary.FileWriter(), tf.Tensor(dtype=String))
    :param return: the tensorboard summary writer and a Tensor containing summarized tensorflow operations
    """
    tensorboard_dir = model.model_path + constants.TENSORBOARD + model.run_dir
    with tf.variable_scope("summaries"):
        tf.summary.scalar("validation_loss", model.validation_loss_op)
        tf.summary.scalar("validation_accuracy", model.validation_accuracy_op)
        summarize_timesteps("validation_accuracy", model.validation_timestep_accuracy_op)
        tf.summary.scalar("test_loss", model.test_loss_op)
        tf.summary.scalar("test_accuracy", model.test_accuracy_op)
        summarize_timesteps("test_accuracy", model.test_timestep_accuracy_op)
        # summarize_variable(model.out_weights, "output_weights")
        # summarize_variable(model.out_bias, "output_bias")
        # summarize_variable(model.accuracy, "output_accuracy")
    merged_summary_ops = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir, graph=model.session.graph)
    return writer, merged_summary_ops
# End of init_tensorboard()

def summarize_timesteps(name, timestep_accuracy_op):
    for timestep, accuracy in enumerate(timestep_accuracy_op):
        tf.summary.scalar(name + '_timestep_' + str(timestep+1), accuracy)
