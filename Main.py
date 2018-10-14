#!/usr/local/bin/python

"""
Let this branch be the optimization of number of hidden nodes for each layer
# Number of training samples = 5250
# Number of test samples = 2107

Using early stopping for the optimization
Checkout Optimizations google sheet for more details
"""


import tensorflow as tf
from tensorflow import keras

import numpy as np

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
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3)
    return model.fit(train_data,
                     train_labels,
                     epochs=EPOCHS,
                     batch_size=BATCH_SIZE,
                     validation_data=(test_data, test_labels),
                     verbose=0,
                     callbacks=[callback])


optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, momentum=0.9)

histories = []
accuracies = []
hidden_nodes = [i*50 for i in range(1, 21)]

for h1 in hidden_nodes:
    for h2 in hidden_nodes:
        for h3 in hidden_nodes:
            my_model = keras.Sequential()

            for nodes in (h1, h2, h3):
                layer = keras.layers.Dense(nodes, activation=tf.nn.relu)

                my_model.add(layer)
                my_model.add(keras.layers.Dropout(0.5))

            my_model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

            my_model.compile(optimizer=optimizer,
                             loss='binary_crossentropy',
                             metrics=['accuracy', 'binary_crossentropy'])
            model_hist = get_history_of(my_model)
            model_acc = 'Acc for {}, {}, {}: {} - Max is {}'.format(h1, h2, h3,
                                                                    model_hist.history['val_acc'][-1],
                                                                    max(model_hist.history['val_acc']))
            print(model_acc)

