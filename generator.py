"""
Tensorflow implementation of methods for generating RNN output.

Copyright (c) 2017 Frank Derry Wanye

Date: 24 September, 2017
"""

import numpy as np
import tensorflow as tf
import logging
import math

from . import constants

def generate_output(model, num_tokens=30, num_outputs=10):
    """
    Generates output from the RNN.

    Params: 
    model (rnn.RNNModel): The model to be used to generate output
    num_tokens (int): The number of tokens to generate. If set to infinity, the output is generated until an END token 
                      is reached
    num_outputs (int): The number of outputs to generate.

    Return:
    list: A list of generated outputs in string format
    """
    model.logger.info("Generating output.")
    outputs = []
    for i in range(1, num_outputs+1):
        sentence = generate_single_output(model, num_tokens)
        sentence = " ".join([model.index_to_token[word] for word in sentence])
        outputs.append(sentence)
    for sentence in outputs:
        print("%s\n\n" % sentence)
    model.logger.info("Generated outputs: %s" % outputs)
    return outputs
# End of generate_output()

def generate_single_output(model, num_tokens=math.inf):
    """
    Generates a single output from the model.

    Params: 
    model (rnn.RNNModel): The model to be used to generate output
    num_tokens (int): The number of tokens to generate. If set to infinity, the output is generated until an END token 
                      is reached

    Return:
    np.ndarray: The generated output as a numpy array
    """
    sentence = np.array([model.token_to_index[constants.START_TOKEN]])
    current_state = np.zeros((model.settings.train.batch_size, model.settings.rnn.hidden_size))
    num_generated = 0
    while num_tokens > num_generated:
        output, new_current_state = predict(model, sentence, current_state)
        sentence = np.append(sentence, sample_output_token(model, output))
        if sentence[-1] == model.token_to_index[constants.END_TOKEN] : break
        num_generated += 1
    # End of while loop
    return sentence
# End of generate_single_output()

def predict(model, sentence, current_state):
    """
    Pass the input sentence through the RNN and retrieve the predictions.

    :type model: RNNModel()
    :param model: the model through which to pass the sentence.

    :type sentence: list of int()
    :param sentence: the sentence to pass throught the RNN.

    :type current_state: np.array()
    :param current_state: the current hidden_state of the RNN.

    :type return: (np.array(), np.array())
    :param return: the predictions and the hidden state after the sentence is passed through.
    """
    input_batch = sentence_to_batch_array(model.settings.train, sentence)

    predictions, final_hidden_state = model.session.run(
        [model.predictions_series, model.current_state], 
        feed_dict={
            model.batch_x_placeholder:input_batch, 
            model.hidden_state_placeholder:current_state   
        })

    position = (len(sentence)-1) % model.settings.train.truncate
    return predictions[position][0], final_hidden_state
# End of predict()

def sentence_to_batch_array(train_settings, sentence):
    """
    Convert sentence to batch array.

    Params: 
    train_settings (settings.SettingsNamespace): The training settings
    sentence (list): The input sentence as a list of integers

    Return:
    np.ndarray: The correct slice of the input sentence, replicated across all batches
    """
    batch_array = np.zeros((train_settings.truncate))
    start_position = math.floor((len(sentence)-1)/train_settings.truncate) * train_settings.truncate
    for index, word in enumerate(sentence[start_position:start_position+train_settings.truncate]):
        batch_array[index] = word
    # Make all rows the same (so it doesn't matter what batch is accessed afterwards)
    batch_array = np.tile(batch_array, (train_settings.batch_size, 1))
    return batch_array
# End of __sentence_to_batch_array__()

def sample_output_token(model, probabilities):
    """
    Returns the probable next word in sentence. Some randomization is included to make sure 
    that not all the sentences produced are the same.

    :type model: RNNModel()
    :param model: the model on which the probabilities were calculated.

    :type probabilities: np.array()
    :param probabilities: the probabilities for the next word in the sentence.

    :type return: int()
    :param return: the index of the next word in the sentence.
    """
    return np.argmax(probabilities)
    # output_word = model.token_to_index[constants.UNKNOWN]
    # while output_word == model.token_to_index[constants.UNKNOWN]:
    #     while sum(probabilities[:-1]) > 1.0 : 
    #         model.logger.error("Sum of word probabilities (%f) > 1.0" % sum(probabilities[:-1]))
    #         probabilities = softmax(probabilities)
    #     samples = np.random.multinomial(1, probabilities)
    #     output_word = np.argmax(samples)
    # return output_word
# End of sample_output_token()

def softmax(probabilities):
    """
    Compute softmax values for each sets of scores in the given array.
    It is needed because python's floats are dumb. 
    There are times when, say, 1.00000 does not equal 1
    Credit: https://stackoverflow.com/q/34968722/5760608

    :type probabilities: an array of floats.
    :param probabilities: an array of probabilities.

    :type return: an array of floats.
    :param return: an array of probabilities that sum up to 1.
    """
    return np.exp(probabilities) / np.sum(np.exp(probabilities), axis=0)
# End of softmax