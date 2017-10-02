"""
Utility class for creating tensorboard summaries.

10 September, 2017
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
        tf.summary.scalar("total_loss", model.total_loss_op)
        summarize_variable(model.out_weights, "output_weights")
        summarize_variable(model.out_bias, "output_bias")
        summarize_variable(model.accuracy, "output_accuracy")
    merged_summary_ops = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir, graph=model.session.graph)
    return writer, merged_summary_ops
# End of init_tensorboard()

def summarize_variable(variable, variable_name):
    with tf.variable_scope(variable_name):
        mean = tf.reduce_mean(variable)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(variable - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(variable))
        tf.summary.scalar('min', tf.reduce_min(variable))
        tf.summary.histogram('histogram', variable)