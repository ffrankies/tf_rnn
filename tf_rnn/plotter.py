"""Contains code for generating plots describing the neural network's performance.

@since 0.6.1
"""

# pylint: disable=C0413
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
from typing import Iterable, Any

import numpy as np  # noqa

from . import constants  # noqa
from .logger import debug, trace  # noqa

# These things are only imported for type checking
from .model import RNNBase  # noqa
from .layers.utils import Accumulator  # noqa
from .layers.utils import ConfusionMatrix  # noqa
from .translate.tokenizer import Tokenizer  # noqa

plt.style.use('ggplot')


@debug('Plotting training results')
def plot(model: RNNBase, train_accumulator: Accumulator, valid_accumulator: Accumulator,
         test_accumulator: Accumulator):
    """Plots a graphical representation of the model's training performance over time. Saves the plot to file in
    <model_path>/run_<run_number>/graph.png.

    Params:
    - model (model.RNNModel): The model containing the path where the figure will be saved
    - train_accumulator (layers.utils.Accumulator): The accumulator for training performance
    - valid_accumulator (layers.utils.Accumulator): The accumulator for validation performance
    - test_accumulator (layers.utils.Accumulator): The accumulator for test performance
    """
    model.logger.info('Plotting results for visualization')
    directory = model.saver.meta.latest()[constants.DIR]
    plot_training_loss(directory, train_accumulator.losses, valid_accumulator.losses)
    plot_training_accuracy(directory, train_accumulator.accuracies(), valid_accumulator.accuracies())
    plot_f1_score(directory, valid_accumulator)
    plot_timestep_accuracy(directory, test_accumulator)
    plot_confusion_matrix(directory, test_accumulator.latest_confusion_matrix,
                          model.dataset.translators.output_translators[0])
# End of plot()


@trace()
def setup_plot(title: str, x_label: str, y_label: str) -> tuple:
    """Sets up the plot with the given parameters.

    Params:
    - title (str): The title of the plot
    - x_label (str): The x-axis label
    - y_label (str): The y-axis label

    Returns:
    - figure (matplotlib.figure.Figure): The figure containing the plot
    - axes (matplotlib.axes.Axes): The plot axes
    """
    plt.close('all')
    figure, axes = plt.subplots()
    axes.set_title(title)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    return figure, axes
# End of setup_plot()


@trace()
def plot_training_loss(directory: str, training_loss: list, validation_loss: list):
    """Plots the training and validation losses on a sublot.

    Params:
    - directory (str): The directory in which to save the plot
    - training_loss (list): The list of training losses
    - validation_loss (list): The list of validation losses
    """
    figure, axes = setup_plot('Training Loss Over Time', 'Epoch', 'Loss')
    x_range = range(0, len(training_loss))
    axes.plot(x_range, training_loss, label="Training Loss (final=%.2f)" % training_loss[-1])
    axes.plot(x_range, validation_loss, label="Validation Loss (final=%.2f)" % validation_loss[-1])
    axes.legend(loc='upper right')
    figure.savefig(directory + constants.PLT_TRAIN_LOSS)
# End of plot_training_loss()


@trace()
def plot_training_accuracy(directory: str, training_accuracy: list, validation_accuracy: list):
    """Plots the training and validation prediction accuracy on a sublot.

    Params:
    - directory (str): The directory in which to save the plot
    - training_accuracy (list): The list of training accuracies
    - validation_accuracy (list): The list of validation accuracies
    """
    figure, axes = setup_plot('Training Accuracy Over Time', 'Epoch', 'Accuracy')
    x_range = range(0, len(training_accuracy))
    axes.plot(x_range, training_accuracy, label="Training Accuracy (final=%.2f)" % training_accuracy[-1])
    axes.plot(x_range, validation_accuracy, label="Validation Accuracy (final=%.2f)" % validation_accuracy[-1])
    axes.legend(loc='lower right')
    figure.savefig(directory + constants.PLT_TRAIN_ACCURACY)
# End of plot_training_accuracy()


@trace()
def plot_f1_score(directory: str, valid_accumulator: Accumulator):
    """Plots the f1_score (together with precision and recall) for the validation partition.

    Params:
    - directory (str): The directory in which to save the plot
    - valid_accumulator (Accumulator): The accumulator that contains the f1_score metrics
    """
    figure, axes = setup_plot('F1 Score During Validation Over Time', 'Epoch', 'Value')
    f1_scores = valid_accumulator.f1_scores()
    precisions = valid_accumulator.precisions()
    recalls = valid_accumulator.recalls()
    x_range = range(0, len(f1_scores))
    axes.plot(x_range, f1_scores, label="F1 Score (final=%.2f)" % f1_scores[-1])
    axes.plot(x_range, precisions, label="Precision (final=%.2f)" % precisions[-1])
    axes.plot(x_range, recalls, label="Recall(final=%.2f)" % recalls[-1])
    axes.legend(loc='lower right')
    figure.savefig(directory + constants.PLT_F1_SCORE)
# End of plot_f1_score()


@trace()
def plot_timestep_accuracy(directory: str, accumulator: Accumulator, timestep_labels: list = None):
    """Plots the average accuracy for each timestep as a bar chart.

    Params:
    - directory (str): The directory in which to save the plot
    - timestep_accuracy (list): The average accuracy of predictions for each timestep
    - timestep_labels (list): The labels for the timesteps
    """
    if not accumulator.losses:
        return
    figure, axes = setup_plot(
        "Avg. Accuracy at Each Timestep\nAvg. Overall Accuracy = {:.2f}".format(accumulator.best_accuracy()),
        'Timestep', 'Avg. Accuracy')
    timestep_accuracy = [ratio * 100.0 for ratio in accumulator.get_timestep_accuracies()]
    axes.set_ylim(0, 120)
    bar_chart = axes.bar(range(1, len(timestep_accuracy) + 1), timestep_accuracy)

    # Add labels to the bar chart
    # pylint: disable=C0102
    for bar in bar_chart:
        x_pos = bar.get_x() + bar.get_width()/2.
        height = bar.get_height()
        y_pos = height + 5
        axes.text(x_pos, y_pos, "%.1f" % height, ha='center', va='bottom', rotation=90)

    # Replace default labels with timestep_labels
    if timestep_labels is not None:
        axes.xaxis.set(ticklabels=timestep_labels)

    figure.savefig(directory + constants.PLT_TIMESTEP_ACCURACY)
# End of plot_timestep_accuracy()


@trace()
def plot_confusion_matrix(directory: str, confusion_matrix: ConfusionMatrix, indexer: Tokenizer = None):
    """Plots a confusion matrix a more accurate breakdown of the predictions made.

    Params:
    - directory (str): The directory in which to save the plot
    - confusion_matrix (layers.utils.ConfusionMatrix): The confusion matrix
    - indexer (translate.Tokenizer): Converts the label indexes to tokens
    """
    if confusion_matrix.is_empty():
        return  # Don't do anything if confusion matrix is empty

    figure, axes = setup_plot('Predictions Breakdown', 'Prediction', 'Correct Label')

    # Get labels
    labels = confusion_matrix.all_labels()  # type: Iterable[Any]
    if indexer is not None:
        labels = indexer.to_human_vector(labels)

    normalized_cm = confusion_matrix.to_normalized_array()

    # Adjust size of figure to fit in whole plot
    cell_size = 0.15  # size is in inches
    num_used_labels = len(labels)
    figure.set_size_inches(1.15*num_used_labels*cell_size, num_used_labels*cell_size)
    grid = axes.pcolormesh(normalized_cm, cmap=plt.cm.YlGnBu)  # pylint: disable=E1101
    figure.colorbar(grid)

    # Center labels: credit to https://stackoverflow.com/a/24193138
    axes.yaxis.set(ticks=np.arange(0.5, num_used_labels), ticklabels=labels)
    axes.xaxis.set(ticks=np.arange(0.5, num_used_labels), ticklabels=labels)
    plt.xticks(rotation=90)

    figure.tight_layout()
    figure.savefig(directory + constants.PLT_CONFUSION_MATRIX)
# End of plot_confusion_matrix()
