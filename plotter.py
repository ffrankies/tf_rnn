'''
Contains code for generating plots describing the neural network's performance.
Copyright (c) 2017 Frank Derry Wanye
Date: 21 January, 2018
'''
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    plt.figure(num=1, figsize=(10, 20))
    plot_loss((train_accumulator.losses, valid_accumulator.losses), 511)
    plot_accuracy_line((train_accumulator.accuracies, valid_accumulator.accuracies), 512)
    if len(test_accumulator.accuracies) > 0:
        plot_accuracy_pie_chart(test_accumulator.accuracies[-1], 513)
    plot_bar_chart(test_accumulator.latest_timestep_accuracies, 514)
    print('CM: ', test_accumulator.confusion_matrix.to_array())
    print('Latest CM: ', test_accumulator.latest_confusion_matrix.to_array())
    plot_confusion_matrix(test_accumulator.latest_confusion_matrix, 515)
    plt.tight_layout()
    plt.savefig(model.saver.meta.latest()[constants.DIR] + constants.PLOT)
    plt.show()
# End of plot()

def plot_comparison():
    # TODO()
    return None

def plot_final():
    # TODO()
    return None

def plot_loss(loss_list, axis):
    '''
    Plots the training and validation losses on a sublot.
    Params:
    loss_list (tuple):
    - training_losses (list): The training losses to plot
    - validation_losses (list): The validation losses to plot
    axis (int): The axis on which to plot the validation loss
    '''
    axis = plt.subplot(axis)
    # axis.clear()
    x = range(0, len(loss_list[0]))
    axis.plot(x, loss_list[0], '-b', label="Training Loss (final=%.2f)" % loss_list[0][-1])
    axis.plot(x, loss_list[1], '-r', label="Validation Loss (final=%.2f)" % loss_list[1][-1])
    axis.legend(loc='upper right')
    axis.set_xlabel('Epoch')
    axis.set_ylabel('Average Loss')
    axis.set_title('Visualizing loss during training')
# End of plot_loss()

def plot_accuracy_line(accuracy_list, axis):
    '''
    Plots the average accuracy during training as a line graph a separate subplot.

    Params:
    accuracy_list (tuple): 
    - training_accuracies (list): The training accuracies to plot
    - validation_accuracies (list): The validation accuracies to plot
    axis (int): The axis on which to plot the validation loss
    '''
    axis = plt.subplot(axis)
    # axis.clear()
    x = range(0, len(accuracy_list[0]))
    axis.plot(x, accuracy_list[0], '-b', label="Training Accuracy (final=%.2f)" % accuracy_list[0][-1])
    axis.plot(x, accuracy_list[1], '-r', label="Validation Accuracy (final=%.2f)" % accuracy_list[1][-1])
    axis.legend(loc='upper left')
    axis.set_xlabel('Epoch')
    axis.set_ylabel('Average Accuracy of Predictions')
    axis.set_title('Visualizing accuracy of predictions during training')
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

def plot_bar_chart(timestep_accuracy, axis):
    '''
    Plots the average accuracy for each timestep as a bar chart on a separate subplot.
    
    Params:
    - timestep_accuracy (list): The average accuracy of predictions for each timestep on the test dataset partition
    - axis (int): The axis on which to plot the validation loss
    '''
    axis = plt.subplot(axis)
    timestep_accuracy = [x * 100.0 for x in timestep_accuracy]
    bar_chart = axis.bar(range(1, len(timestep_accuracy) + 1), timestep_accuracy)
    axis.set_xlabel('Timestep')
    axis.set_ylabel('Average Accuracy (%)')
    axis.set_title('Average accuracy of each timestep for the test partition')
    # from https://matplotlib.org/gallery/api/barchart.html#sphx-glr-gallery-api-barchart-py
    # Couldn't find a better way to label the bar chart, unfortunately
    for bar, accuracy in zip(bar_chart, timestep_accuracy):
        x_pos = bar.get_x() + bar.get_width()/2.
        height = bar.get_height()
        y_pos = height - 20.0
        if height <= 30.0 : y_pos = height + 10.0
        axis.text(x_pos, y_pos, "%.1f" % accuracy, ha='center', va='bottom', rotation=90)
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