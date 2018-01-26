'''
Contains code for generating plots describing the neural network's performance.
Copyright (c) 2017 Frank Derry Wanye
Date: 24 January, 2018
'''
import numpy as np
import bokeh
import bokeh.plotting as bplot
import bokeh.models as bmodels

# Selenium support for PhantomJS warning is annoying
import warnings
warnings.filterwarnings('ignore')

import seaborn as sns

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
    # plt.clf() # Clear the current figure
    # plt.figure(num=1, figsize=(10, 20))
    directory = model.saver.meta.latest()[constants.DIR]
    plot_loss(directory, train_accumulator.losses, valid_accumulator.losses)
    plot_accuracy_line(directory, train_accumulator.accuracies, valid_accumulator.accuracies)
    plot_bar_chart(directory, test_accumulator.latest_timestep_accuracies)
    # plot_accuracy_line((train_accumulator.accuracies, valid_accumulator.accuracies), 512)
    # if len(test_accumulator.accuracies) > 0:
    #     plot_accuracy_pie_chart(test_accumulator.accuracies[-1], 513)
    # plot_bar_chart(test_accumulator.latest_timestep_accuracies, 514)
    # print('CM: ', test_accumulator.confusion_matrix.to_array())
    # print('Latest CM: ', test_accumulator.latest_confusion_matrix.to_array())
    # plot_confusion_matrix(test_accumulator.latest_confusion_matrix, 515)
    # plt.tight_layout()
    # plt.savefig(model.saver.meta.latest()[constants.DIR] + constants.PLOT)
    # plt.show()
# End of plot()

def plot_comparison():
    # TODO()
    return None

def setup_plot(title, x_label, y_label, width=600):
    '''
    Sets up the figure and axis for a plot.

    Params:
    - title (str): The title for the plot
    - x_label (str): The label for the x axis
    - y_label (str): The label for the y axis
    - width (int): The width of the plot to generate

    Return:
    - plot (bokeh.plotting.Figure): The plot figure
    '''
    plot = bplot.figure(title=title, x_axis_label=x_label, y_axis_label=y_label, tools='', toolbar_location=None, 
        plot_width=width)
    plot.min_border_left = 10
    plot.min_border_right = 10
    plot.min_border_top = 10
    plot.min_border_bottom = 10
    return plot
# End of setup_plot()

def plot_loss(directory, training_losses, validation_losses):
    '''
    Plots the training and validation losses on a sublot.
    Params:
    - directory (str): The directory in which to save the plot
    - training_losses (list): The training losses to plot
    - validation_losses (list): The validation losses to plot
    '''
    x = list(range(0, len(training_losses)))
    plot = setup_plot('Training Loss Over Time', 'Epoch', 'Loss')
    plot.line(x, training_losses, legend="Training Loss (final=%.2f)" % training_losses[-1], color='blue', 
        line_width=2)
    plot.line(x, validation_losses, legend="Validation Loss (final=%.2f)" % validation_losses[-1], color='red',
        line_width=2)
    bokeh.io.export_png(plot, filename=directory + constants.PLT_TRAIN_LOSS)
# End of plot_loss()

def plot_accuracy_line(directory, training_accuracies, validation_accuracies):
    '''
    Plots the average accuracy during training as a line graph a separate subplot.

    Params:
    - directory (str): The directory in which to save the plot
    - training_accuracies (list): The training accuracies to plot
    - validation_accuracies (list): The validation accuracies to plot
    '''
    x = list(range(0, len(training_accuracies)))
    plot = setup_plot('Training Accuracy Over Time', 'Epoch', 'Accuracy')
    plot.line(x, training_accuracies, legend="Training Accuracy (final=%.2f)" % training_accuracies[-1], 
        color='blue', line_width=2)
    plot.line(x, validation_accuracies, legend="Validation Accuracy (final=%.2f)" % validation_accuracies[-1], 
        color='red', line_width=2)
    plot.legend.location = 'bottom_right'
    bokeh.io.export_png(plot, filename=directory + constants.PLT_TRAIN_ACCURACY)
# End of plot_accuracy_line()

def plot_accuracy_pie_chart(accuracy, axis):
    '''
    Plots the average accuracy on the test partition as a pie chart on a separate subplot.

    Params:
    - accuracy (float): The average accuracy of predictions for the test dataset partition
    - axis (int): The axis on which to plot the validation loss
    '''
    axis = plt.subplot(axis)
    axis.clear()
    accuracy = accuracy * 100
    axis.pie([accuracy, 100 - accuracy], labels=['Correct predictions', 'Incorrect predictions'], autopct='%.2f%%')
    axis.axis('equal')
    axis.set_title('Average accuracy for the test partition')
# End of plot_accuracy_pie_chart()

def plot_bar_chart(directory, timestep_accuracy):
    '''
    Plots the average accuracy for each timestep as a bar chart on a separate subplot.
    
    Params:
    - timestep_accuracy (list): The average accuracy of predictions for each timestep on the test dataset partition
    - axis (int): The axis on which to plot the validation loss
    '''
    timestep_accuracy = [x * 100.0 for x in timestep_accuracy]
    timesteps = list(range(len(timestep_accuracy)))
    plot_width = len(timesteps) * 30
    if plot_width == 0:
        plot_width = 600
    plot = setup_plot('Accuracy of Predictions at Each Timestep', 'Timestep', 'Average Accuracy (%)', plot_width)
    source = bmodels.ColumnDataSource(data=dict( 
        timesteps=timesteps,
        timestep_accuracy=timestep_accuracy,
        labels=[" %.1f" % accuracy for accuracy in timestep_accuracy]))
    plot.vbar(x='timesteps', top='timestep_accuracy', width=0.9, source=source)
    plot.y_range = bmodels.Range1d(0, 120)
    labels = bmodels.LabelSet(x='timesteps', y='timestep_accuracy', text='labels', level='glyph', text_baseline='middle', 
        source=source, render_mode='canvas', angle=90, angle_units='deg')
    plot.add_layout(labels)
    bokeh.io.export_png(plot, directory + constants.PLT_TEST_ACCURACY_BAR)
# End of plot_bar_chart()

def plot_confusion_matrix(confusion_matrix, axis):
    '''
    Plots a confusion matrix at the given axis in the figure.

    Params:
    - confusion_matrix (list): The confusion matrix, as a list of lists
    - axis (int): The axis on which to plot the confusion matrix
    '''
    axis = plt.subplot(axis)
    if confusion_matrix.is_empty(): return # Don't do anything if confusion matrix is empty
    num_used_labels = len(confusion_matrix.row_labels)
    num_used_predictions = len(confusion_matrix.col_labels)
    print('Length of labels: ', num_used_labels)
    print('Length of predictions: ', num_used_predictions)
    # plt.yticks(range(height), alphabet[:height])
    axis.imshow(confusion_matrix.to_array(), cmap=plt.cm.jet, interpolation='nearest')
    plt.yticks(range(num_used_labels), confusion_matrix.row_labels)
    plt.xticks(range(num_used_predictions), confusion_matrix.col_labels)
# End of plot_confusion_matrix()