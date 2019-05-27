from tensorflow import keras
import itertools
from matplotlib import pyplot
from configparser import ConfigParser
import os

import numpy as np

c_parser = ConfigParser(inline_comment_prefixes=('#', ))
thisfolder = os.path.dirname(os.path.abspath(__file__))
initfile = os.path.join(thisfolder, '../config/my-config')
c_parser.read(initfile)
ONE_HOT_ENCODING = c_parser.getboolean('CNN', 'ONE_HOT_ENCODING')
PATH = "/Users/alper/Documents/tez/analysis/" # "/Volumes/Untitled/Tez Data/data/"
MASTER_THESIS_PATH = '/Users/alper/Dropbox/Google Drive/AydınSon Lab/Master Tezi/'


def get_data(fp):
    file_data = []
    file_labels = []
    with open(fp) as f:
        for line in f:
            splitted_line = line.strip('\n').split(',')
            # this is 1: because the first column indicates label
            sample_data = splitted_line[1:]
            sample_label = int(splitted_line[0])
            file_data.append(sample_data)
            file_labels.append(sample_label)
    print('Done reading file {}'.format(fp))
    # if the file is already encoded
    # Or if you don't want to encode categorical numeric data
    if ONE_HOT_ENCODING:
        return np.array(file_data), np.array(file_labels)
    # else apply one hot encoding to categorical data
    else:
        file_data = keras.utils.to_categorical(file_data)
        flattened_file_data = []
        for data in file_data:
            flattened_file_data.append(list(itertools.chain.from_iterable(data)))
        return np.array(flattened_file_data), np.array(file_labels)


def plot_history(model_hist):
    # plot training history
    pyplot.plot(model_hist.history['loss'], label='train_loss')
    pyplot.plot(model_hist.history['val_loss'], label='val_loss')
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


def make_predictions(model, test_data, test_labels):
    CASE = 1
    CONTROL = 0
    predictions = model.predict_classes(test_data)
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


def get_rsid_map():
    conversion_file = PATH + 'old_analysis/conversion_map'
    my_map = {}
    with open(conversion_file) as cf:
        for line in cf:
            if '#' in line or 'Probe Set ID' in line:
                continue
            splitted_line = line.replace('"', '').split()
            rsid = splitted_line[1]
            affy_name = splitted_line[0]
            my_map[rsid] = affy_name

    #add missing rsid conversions
    conversion_file = MASTER_THESIS_PATH + 'SNPmaster1-2-1.txt'
    with open(conversion_file) as cf:
        for i, line in enumerate(cf):
            if i == 0:
                continue
            splitted_line = line.split()
            rsid = splitted_line[1]
            if rsid not in my_map:
                affy_name = splitted_line[2]
                my_map[rsid] = affy_name
    return my_map
