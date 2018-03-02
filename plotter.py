'''
Contains code for generating plots describing the neural network's performance.
@since 0.4.1
'''
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import seaborn as sns # for heatmap

from . import constants


def plot(model, train_accumulator, valid_accumulator, test_accumulator):
    '''
    Plots a graphical representation of the model's training performance over time. Saves the plot to file in
    <model_path>/run_<run_number>/graph.png.

    Params:
    - model (model.RNNModel): The model containing the path where the figure will be saved
    - train_accumulator (layers.performance_layer.Accumulator): The accumulator for training performance
    - valid_accumulator (layers.performance_layer.Accumulator): The accumulator for validation performance
    - test_accumulator (layers.performance_layer.Accumulator): The accumulator for test performance
    '''
    model.logger.info('Plotting results for visualization')
    directory = model.saver.meta.latest()[constants.DIR]
    plot_training_loss(directory, train_accumulator.losses, valid_accumulator.losses)
    plot_training_accuracy(directory, train_accumulator.accuracies, valid_accumulator.accuracies)
    plot_timestep_accuracy(directory, test_accumulator.latest_timestep_accuracies)
    plot_confusion_matrix(directory, test_accumulator.latest_confusion_matrix, model.dataset.indexer)
# End of plot()


def setup_plot(title, x_label, y_label):
    '''
    Sets up the plot with the given parameters.

    Params:
    - title (str): The title of the plot
    - x_label (str): The x-axis label
    - y_label (str): The y-axis label

    Return:
    - figure (matplotlib.figure.Figure): The figure containing the plot
    - plot (matplotlib.axes.Axes): The plot axes
    '''
    plt.close('all')
    figure, plot = plt.subplots()
    plot.set_title(title)
    plot.set_xlabel(x_label)
    plot.set_ylabel(y_label)
    return figure, plot
# End of setup_plot()


def plot_training_loss(directory, training_loss, validation_loss):
    '''
    Plots the training and validation losses on a sublot.

    Params:
    - directory (str): The directory in which to save the plot
    - training_loss (list): The list of training losses
    - validation_loss (list): The list of validation losses
    '''
    figure, plot = setup_plot('Training Loss Over Time', 'Epoch', 'Loss')
    x = range(0, len(training_loss))
    plot.plot(x, training_loss, label="Training Loss (final=%.2f)" % training_loss[-1])
    plot.plot(x, validation_loss, label="Validation Loss (final=%.2f)" % validation_loss[-1])
    plot.legend(loc='upper right')
    figure.savefig(directory + constants.PLT_TRAIN_LOSS)
# End of plot_training_loss()


def plot_training_accuracy(directory, training_accuracy, validation_accuracy):
    '''
    Plots the training and validation prediction accuracy on a sublot.

    Params:
    - directory (str): The directory in which to save the plot
    - training_accuracy (list): The list of training accuracies
    - validation_accuracy (list): The list of validation accuracies
    '''
    figure, plot = setup_plot('Training Accuracy Over Time', 'Epoch', 'Accuracy')
    x = range(0, len(training_accuracy))
    plot.plot(x, training_accuracy, label="Training Accuracy (final=%.2f)" % training_accuracy[-1])
    plot.plot(x, validation_accuracy, label="Validation Accuracy (final=%.2f)" % validation_accuracy[-1])
    plot.legend(loc='lower right')
    figure.savefig(directory + constants.PLT_TRAIN_ACCURACY)
# End of plot_training_accuracy()


def plot_timestep_accuracy(directory, timestep_accuracy, timestep_labels=None):
    '''
    Plots the average accuracy for each timestep as a bar chart.

    Params:
    - directory (str): The directory in which to save the plot
    - timestep_accuracy (list): The average accuracy of predictions for each timestep
    - timestep_labels (list): The labels for the timesteps
    '''
    if timestep_accuracy is None:
        return
    figure, plot = setup_plot('Avg. Accuracy at Each Timestep', 'Timestep', 'Avg. Accuracy')
    timestep_accuracy = [ratio * 100.0 for ratio in timestep_accuracy]
    plot.set_ylim(0, 120)
    bar_chart = plot.bar(range(1, len(timestep_accuracy) + 1), timestep_accuracy)

    # Add labels to the bar chart
    for bar in bar_chart:
        x_pos = bar.get_x() + bar.get_width()/2.
        height = bar.get_height()
        y_pos = height + 5
        plot.text(x_pos, y_pos, "%.1f" % height, ha='center', va='bottom', rotation=90)

    # Replace default labels with timestep_labels
    if timestep_labels is not None:
        plot.xaxis.set(ticklabels=timestep_labels)

    figure.savefig(directory + constants.PLT_TIMESTEP_ACCURACY)
# End of plot_timestep_accuracy()


def plot_confusion_matrix(directory, confusion_matrix, indexer=None):
    '''
    Plots a confusion matrix a more accurate breakdown of the predictions made.

    Params:
    - directory (str): The directory in which to save the plot
    - confusion_matrix (layers.performance_layer.ConfusionMatrix): The confusion matrix
    - indexer (indexer.Indexer): Converts the label indexes to tokens
    '''
    print('Plotting confusion matrix')
    if confusion_matrix.is_empty():
        return  # Don't do anything if confusion matrix is empty

    figure, plot = setup_plot('Predictions Breakdown', 'Prediction', 'Correct Label')

    # Get labels
    labels = confusion_matrix.all_labels()
    # print('Labels: ', labels)
    if indexer is not None:
        labels = indexer.to_tokens(labels)

    normalized_cm = confusion_matrix.to_normalized_array()
    print('Got normalized matrix array')

    # Adjust size of figure to fit in whole plot
    cell_size = 0.15  # size is in inches
    num_used_labels = len(labels)
    print('Got length of used labels')
    figure.set_size_inches(num_used_labels*cell_size, num_used_labels*cell_size)
    print('Set size of figure')
    grid = plot.pcolormesh(normalized_cm, cmap=plt.cm.YlGnBu)
    print('Plotted confusion matrix')
    figure.colorbar(grid)
    print('Added color grid')

    # Center labels: credit to https://stackoverflow.com/a/24193138
    plot.yaxis.set(ticks=np.arange(0.5, num_used_labels), ticklabels=labels)
    print('Set xticks')
    plot.xaxis.set(ticks=np.arange(0.5, num_used_labels), ticklabels=labels)
    print('Set yticks')
    plt.xticks(rotation=90)
    print('Added rotation')

    figure.savefig(directory + constants.PLT_CONFUSION_MATRIX)
    print('Saved figure')
# End of plot_confusion_matrix()
