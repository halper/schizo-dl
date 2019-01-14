#!/usr/local/bin/python

"""
Let this branch be the optimization of all parameters
# Number of training samples = 5250
# Number of test samples = 2107

Using early stopping for the optimization
Checkout Optimizations google sheet for more details
Adadelta
"""

from tensorflow import keras
import itertools
from matplotlib import pyplot

import numpy as np

prefix = 'pval1.5k_ahp100_enc_'
suffix = 1600
TEST_FP = "{}test_{}.csv".format(prefix, suffix)
TRAIN_FP = "{}train_{}.csv".format(prefix, suffix)
LOAD_WEIGHTS = True

BATCH_SIZE = 646  # this should perfectly divide number of training samples
EPOCHS = 10000
HIDDEN_NODES = (100, ) * 6

L2_WP = 5e-6#5e-6
L2_BP = 5e-4#5e-4

LEARNING_RATE = 5e-3#5e-3

BEST_WEIGHTS = 'best/model-{}-0.7{}.hdf5'.format("02", 2)
BEST_WEIGHTS = 'my-model.hdf5'
BEST_MODEL = 'best/model-0.71.hdf5'
LOAD_MODEL = False

OPTIMIZER = keras.optimizers.Adadelta() # Adadelta
ONE_HOT_ENCODING = True


def plot_history():
    # plot training history
    pyplot.plot(model_hist.history['loss'], label='train_loss')
    pyplot.plot(model_hist.history['val_loss'], label='test_loss')
    #pyplot.plot(model_hist.history['acc'], label='train_acc')
    #pyplot.plot(model_hist.history['val_acc'], label='test_acc')
    pyplot.legend()
    pyplot.show()


def calc_precision(tp, fp):
    return tp / (tp + fp)


def calc_recall(tp, fn):
    return tp / (tp + fn)


def calc_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


def make_predictions():
    CASE = 1
    CONTROL = 0
    predictions = my_model.predict_classes(test_data)
    control_num = 0
    case_num = 0
    true_control_num = 0
    true_case_num = 0
    # round predictions
    rounded = [round(x[0]) for x in predictions]
    for i in range(len(rounded)):
        true_lab = test_labels[i]
        prediction = rounded[i]
        if true_lab == CONTROL:
            control_num += 1
            if prediction == true_lab:
                true_control_num += 1
        elif true_lab == CASE:
            case_num += 1
            if prediction == true_lab:
                true_case_num += 1
    precision = calc_precision(true_case_num, control_num - true_control_num)
    recall = calc_recall(true_case_num, case_num - true_case_num)
    f1_score = calc_f1_score(precision, recall)
    acc = (true_case_num + true_control_num) / (case_num + control_num)
    print('\n')
    print('TP (Correctly labeled Scz - cases)\tTN (Correctly labeled controls)\tAcc\tF1 Score')
    print('{}\t{}\t{:.3f}\t{:.3f}'.format(true_case_num, true_control_num, acc*100, f1_score*100))
    print('Total number of cases: {} '.format(case_num))
    print('Total number of controls: {}'.format(control_num))

def get_data(fp):
    file_data = []
    file_labels = []
    with open(fp) as f:
        for line in f:
            splitted_line = line.strip('\n').split(',')
            sample_data = splitted_line[1:]
            sample_label = int(splitted_line[0])
            file_data.append(sample_data)
            file_labels.append(sample_label)
    print('Done reading file {}'.format(fp))
    if ONE_HOT_ENCODING:
        return np.array(file_data), np.array(file_labels)
    else:
        file_data = keras.utils.to_categorical(file_data)
        flattened_file_data = []
        for data in file_data:
            flattened_file_data.append(list(itertools.chain.from_iterable(data)))
        return np.array(flattened_file_data), np.array(file_labels)


train_data, train_labels = get_data(TRAIN_FP)
test_data, test_labels = get_data(TEST_FP)
INPUT_SIZE = len(train_data[0])


# Loads model and print metrics
def load_model(weights=BEST_WEIGHTS):
    new_model = keras.models.load_model(weights)

    loss, acc, binary_cross = new_model.evaluate(test_data, test_labels)
    print("Restored model with test data, accuracy: {:5.2f}% ".format(100*acc))

if LOAD_MODEL:
    load_model()
    print()


def get_history_of(model):
    filepath = "model-{epoch:02d}-{val_acc:.2f}.hdf5"
    filepath = "my-model.hdf5"
    monitor = "val_loss"
    checkpointer = keras.callbacks.ModelCheckpoint(filepath, monitor=monitor, verbose=1,
                                                   save_best_only=True, mode='min', period=1)
    early_stopping = keras.callbacks.EarlyStopping(monitor=monitor, min_delta=0, patience=75,
                                                   mode='min')
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.1,
                                                  patience=8, min_lr=1e-7)

    if LOAD_WEIGHTS:
        model.load_weights(BEST_WEIGHTS)
    return model.fit(train_data,
                     train_labels,
                     epochs=EPOCHS,
                     batch_size=BATCH_SIZE,
                     #validation_data=(test_data, test_labels),
                     validation_split=0.33,
                     verbose=0,
                     callbacks=[early_stopping, checkpointer, reduce_lr])


my_model = keras.Sequential()

if LOAD_MODEL:
    my_model = keras.models.load_model(BEST_MODEL)
    print("Loaded model: {}".format(BEST_MODEL))
else:
    for layer_no, nodes in enumerate(HIDDEN_NODES):
        # Required for connecting inputs to the network
        layer = keras.layers.Dense(nodes)
        if layer_no == 0:
            layer = keras.layers.Dense(nodes, input_dim=INPUT_SIZE)

        layer.kernel_initializer = keras.initializers.uniform()
        layer.kernel_regularizer = keras.regularizers.l2(L2_WP)
        layer.bias_regularizer = keras.regularizers.l2(L2_BP)
        layer.activation = keras.activations.linear
        my_model.add(layer)
        my_model.add(keras.layers.LeakyReLU())
        my_model.add(keras.layers.BatchNormalization())
        my_model.add(keras.layers.Dropout(0.65))


    # Final layer
    my_model.add(keras.layers.Dense(1, activation=keras.activations.sigmoid))

    my_model.summary()
    my_model.compile(optimizer=OPTIMIZER,
                     loss='binary_crossentropy',
                     metrics=['accuracy', 'binary_crossentropy'])

model_hist = None

try:
    model_hist = get_history_of(my_model)
    model_acc = 'Acc: {} - Max is {}'.format(model_hist.history['val_acc'][-1],
                                             max(model_hist.history['val_acc']))
    print(model_acc)
except Exception:
    pass
finally:
    load_model('my-model.hdf5')

    make_predictions()
if model_hist:
    plot_history()
