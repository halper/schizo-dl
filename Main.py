#!/usr/local/bin/python

"""
Let this branch be the optimization of number of hidden layers
# Number of training samples = 5250
# Number of test samples = 2107

3 layers seems to be better with 240 epochs
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
EPOCHS = 240


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


def get_history_of(model):
    return model.fit(train_data,
                      train_labels,
                      epochs=EPOCHS,
                      batch_size=BATCH_SIZE,
                      validation_data=(test_data, test_labels),
                      verbose=2)


def plot_history(histories_to_be_plotted, key='binary_crossentropy'):
    plt.figure(figsize=(16, 10))

    for name, history in histories_to_be_plotted:
        val = plt.plot(history.epoch, history.history['val_' + key],
                       '--', label=name.title() + ' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title() + ' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])

optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, momentum=0.9)

histories = []
accuracies = []
for k in range(1, 6):
    my_model = keras.Sequential()
    for i in range(k):
        hidden_layer = 256
        layer = keras.layers.Dense(hidden_layer, activation=tf.nn.relu)

        my_model.add(layer)
        my_model.add(keras.layers.Dropout(0.5))

    my_model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    my_model.compile(optimizer=optimizer,
                     loss='binary_crossentropy',
                     metrics=['accuracy', 'binary_crossentropy'])
    model_hist = get_history_of(my_model)
    histories.append(('{} layers'.format(k), model_hist))
    metrics = my_model.evaluate(test_data, test_labels, batch_size=BATCH_SIZE)
    accuracies.append('Evaluation acc for {} layers: {}'.format(k, metrics[1]))


plot_history(histories)
for acc in accuracies:
    print(acc)
print('done')
