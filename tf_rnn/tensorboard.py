'''
Utility class for creating tensorboard summaries.
18 November, 2017
'''
import tensorflow as tf
from . import constants

def init_tensorboard(model):
    '''
    Initialize tensorboard event writer, and add variable summaries to it.
    Params:
    model (model.RNNModel): The model for which summaries should be made.
    Return:
    summary_writer (tf.summary.FileWriter): The tensorboard summary writer
    summary_ops (tf.Tensor): The tensorflow operations that produce the tensorboard summaries
    '''
    tensorboard_dir = model.run_dir
    with tf.variable_scope('summaries'):
        max_timesteps = model.dataset.max_length
        summarize_partition('training', model.train_performance, max_timesteps)
        summarize_partition('validation', model.validation_performance, max_timesteps)
        summarize_partition('test', model.test_performance, max_timesteps)
    merged_summary_ops = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir, graph=model.session.graph)
    return writer, merged_summary_ops
# End of init_tensorboard()

def summarize_partition(name, performance_placeholders, max_timesteps):
    """
    Summarizes the performance evaluations for a given partition.
    Params:
    name (string): The name of the partition
    performance_placeholders (layers.performance_layer.PerformancePlaceholders): The placeholders for the performance
                                                                                 evaluations
    max_timesteps (int): The maximum sequence length in the dataset
    """
    with tf.variable_scope(name):
        tf.summary.scalar("loss", performance_placeholders.average_loss)
        tf.summary.scalar("accuracy", performance_placeholders.average_accuracy)
        summarize_timesteps(performance_placeholders.timestep_accuracies, max_timesteps)
# End of summarize_partition()

def summarize_timesteps(timestep_accuracy_op, max_timesteps):
    """
    Summarizes the average accuracy for each timestep for a given partition.
    Params:
    timestep_accuracy_op (tf.placeholder): The average accuracies for each timestep
    max_timesteps (int): The maximum sequence length in the dataset
    """
    timestep_accuracies = tf.unstack(timestep_accuracy_op)
    timestep_accuracies = timestep_accuracies[:max_timesteps]
    for timestep, accuracy in enumerate(timestep_accuracies):
        tf.summary.scalar('accuracy_at_timestep_' + str(timestep+1), accuracy)
# End of summarize_timesteps()