#!/usr/local/bin/python

"""
Let this branch be the optimization of number of epochs
# Number of training samples = 5250
# Number of test samples = 2107

history graphs show that around 250 epochs the model starts overfitting
so I am setting num of epochs to 250
In addition to that it shows that l2 regularization behaves funny
Checkout Optimizations google sheet for more details
"""


import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

input_size = 1000
test_fp = "test_1000.csv"
train_fp = "train_1000.csv"
BATCH_SIZE = 5250  # this should perfectly divide number of training samples
EPOCHS = 250


def get_data(fp):
    file_data = []
    file_labels = []
    with open(fp) as f:
        for line in f:
            splitted_line = line.strip('\n').split(',')
            sample_data = []
            sample_label = []
            for i, line_arr in enumerate(splitted_line):
                if i == 0:
                    sample_label.extend(line_arr)
                if i > 1:
                    sample_data.extend(line_arr)
            file_data.extend(np.array([sample_data]))
            file_labels.extend(np.array([sample_label]))
    return np.array(file_data), np.array(file_labels)

train_data, train_labels = get_data(train_fp)
test_data, test_labels = get_data(test_fp)


def get_model(l2=False, dropout=False):
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, momentum=0.9)
    hidden_nodes = [245, 225]

    baseline_model = keras.Sequential()
    for hidden_layer in hidden_nodes:
        model = keras.layers.Dense(hidden_layer, activation=tf.nn.relu)
        if l2:
            model = keras.layers.Dense(hidden_layer, activation=tf.nn.relu,
                                       kernel_regularizer=keras.regularizers.l2(0.001))
        baseline_model.add(model)
        if dropout:
            baseline_model.add(keras.layers.Dropout(0.5))

    baseline_model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    baseline_model.compile(optimizer=optimizer,
                           loss='binary_crossentropy',
                           metrics=['accuracy', 'binary_crossentropy'])
    return baseline_model


def get_history_of(model):
    return model.fit(train_data,
                      train_labels,
                      epochs=EPOCHS,
                      batch_size=BATCH_SIZE,
                      validation_data=(test_data, test_labels),
                      verbose=2)


baseline_history = get_history_of(get_model())

l2_history = get_history_of(get_model(l2=True))
dropout_history = get_history_of(get_model(dropout=True))
all_history = get_history_of(get_model(l2=True, dropout=True))


def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16, 10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_' + key],
                       '--', label=name.title() + ' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title() + ' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])


plot_history([('baseline', baseline_history),
              ('l2', l2_history),
              ('dropout', dropout_history),
              ('all', all_history)])
print('done')
