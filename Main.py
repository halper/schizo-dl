#!/usr/local/bin/python

"""
Let this branch be the optimization of all parameters

Using early stopping for the optimization
Checkout Optimizations google sheet for more details
Adadelta
"""

from tensorflow import keras
from utils import utils
import sys

c_parser = utils.c_parser
prefix = c_parser.get('COMMON', 'prefix')
suffix = c_parser.get('CNN', 'suffix')
TEST_FP = c_parser.get('CNN', 'TEST_FP').format(prefix, suffix)

TRAIN_FP = c_parser.get('CNN', 'TRAIN_FP').format(prefix, suffix)
LOAD_WEIGHTS = c_parser.getboolean('CNN', 'LOAD_WEIGHTS')

BATCH_SIZE = c_parser.getint('CNN', 'BATCH_SIZE')
EPOCHS = c_parser.getint('CNN', 'EPOCHS')
HIDDEN_NODES = [nodes for nodes in c_parser.get('CNN', 'HIDDEN_NODES').split(',')]

L2_WP = c_parser.getfloat('CNN', 'L2_WP')
L2_BP = c_parser.getfloat('CNN', 'L2_BP')

LEARNING_RATE = c_parser.getfloat('CNN', 'LEARNING_RATE')

BEST_WEIGHTS = c_parser.get('CNN', 'BEST_WEIGHTS')
BEST_MODEL = c_parser.get('CNN', 'BEST_MODEL')
LOAD_MODEL = c_parser.getboolean('CNN', 'LOAD_MODEL')

OPTIMIZER = getattr(keras.optimizers, c_parser.get('CNN', 'OPTIMIZER'))()


train_data, train_labels = utils.get_data(TRAIN_FP)
test_data, test_labels = utils.get_data(TEST_FP)
INPUT_SIZE = len(train_data[0])


# Loads model and print metrics
def load_model(weights=BEST_WEIGHTS):
    new_model = keras.models.load_model(weights)

    loss, acc, binary_cross = new_model.evaluate(test_data, test_labels)
    print("Restored model with test data, accuracy: {:5.2f}% ".format(100*acc))


def get_history_of(model):
    filepath = "model-{epoch:02d}-{val_acc:.2f}.hdf5"
    filepath = "my-model.hdf5"
    monitor = "val_loss"
    checkpointer = keras.callbacks.ModelCheckpoint(filepath, monitor=monitor, verbose=1,
                                                   save_best_only=True, mode='min', period=1)
    early_stopping = keras.callbacks.EarlyStopping(monitor=monitor, min_delta=0, patience=40,
                                                   mode='min')
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.2, mode='min',
                                                  patience=5, min_lr=1e-8, cooldown=1)

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
    utils.make_predictions(my_model, test_data, test_labels)
    sys.exit(0)
else:
    for layer_no, nodes in enumerate(HIDDEN_NODES):
        # Required for connecting inputs to the network
        nodes = int(nodes)
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

    utils.make_predictions(my_model, test_data, test_labels)
    if model_hist:
        utils.plot_history(model_hist)
        history_file = 'results/history/{}{}'.format(prefix, suffix)
        history = model_hist.history
        with open(history_file, 'w') as fw:
            fw.write('Epoch\tLoss\tValidation Loss\tAccuracy\tValidation Acc\n')
            for i in range(len(history['loss'])):
                loss = history['loss'][i]
                val_loss = history['val_loss'][i]
                acc = history['acc'][i]
                val_acc = history['val_acc'][i]
                fw.write('{:d}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(i+1, loss, val_loss, acc, val_acc))
