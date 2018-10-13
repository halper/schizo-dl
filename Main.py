#!/usr/local/bin/python
# This is the base model

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

input_size = 1000
test_fp = "test_1000.csv"
train_fp = "train_1000.csv"


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


optimizer = tf.train.AdamOptimizer(0.001)
hidden_nodes = [245, 225]

baseline_model = keras.Sequential()
for hidden_layer in hidden_nodes:
    baseline_model.add(keras.layers.Dense(hidden_layer, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.001)))

baseline_model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

baseline_model.compile(optimizer=optimizer,
                       loss='binary_crossentropy',
                       metrics=['accuracy', 'binary_crossentropy'])

train_data, train_labels = get_data(train_fp)
test_data, test_labels = get_data(test_fp)


baseline_history = baseline_model.fit(train_data,
                                      train_labels,
                                      epochs=60,
                                      batch_size=256,
                                      validation_data=(test_data, test_labels),
                                      verbose=2)


def plot_history(histories, key='acc'):
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


plot_history([('baseline', baseline_history)])
print('done')