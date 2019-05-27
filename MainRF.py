#!/usr/local/bin/python

"""
This script is going to be used to find the importance of features utilizing RF
The aim is to reduce the dataset from 86K to a lower value without losing accuracy with DNN
"""

from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
from utils import utils
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
import sys


c_parser = utils.c_parser
prefix = c_parser.get('COMMON', 'prefix')
suffix = c_parser.get('CNN', 'suffix')
TEST_FP = c_parser.get('CNN', 'TEST_FP').format(prefix, suffix)

TRAIN_FP = c_parser.get('CNN', 'TRAIN_FP').format(prefix, suffix)

train_data, train_labels = utils.get_data(TRAIN_FP)
test_data, test_labels = utils.get_data(TEST_FP)

print('Training Features Shape:', train_data.shape)
print('Training Labels Shape:', train_labels.shape)
print('Validation Features Shape:', test_data.shape)
print('Validation Labels Shape:', test_labels.shape)

# Instantiate model with n decision trees
# entropy 1000 ~.60
# gini 1000 ~.60
rf = RandomForestClassifier(n_estimators=5000,
                            criterion='gini',
                            n_jobs=-1,
                            min_samples_leaf=75,
                            min_samples_split=5,
                            oob_score=True,
                            max_features="auto",
                            random_state=0)

param_grid = {
    'n_estimators': [2000, 5000, 7500, 10000],
    #'min_samples_leaf': [75, 100, 200],
    #'min_samples_split': [5, 10, 25, 50]
}

#CV_rfc = GridSearchCV(estimator=rf, param_grid=param_grid)

# Train the model
rf.fit(train_data, train_labels)
#CV_rfc.fit(train_data, train_labels)
#print(CV_rfc.best_params_)
#sys.exit(-1)

# Make predictions
predictions = rf.predict(test_data)

print(confusion_matrix(test_labels, predictions))
print(classification_report(test_labels, predictions))
print(accuracy_score(test_labels, predictions))

# Calculate the absolute errors
errors = abs(predictions - test_labels)

truth_count = len(list(filter(lambda x: x == 0, errors)))

# Calculate and display accuracy
accuracy = 100.0 * truth_count / len(test_labels)
print('Accuracy: {:.2f}%'.format(accuracy))


def print_importances():
    # Get numerical feature importances
    importances = list(rf.feature_importances_)

    # Since it is an encoded data sequential importances should be averaged to find the importance of the feature
    averaged_importances = [(importances[i] + importances[i+1]) / 2 for i in range(0, len(importances), 2)]

    feature_list = ['SNP_{:d}'.format(n+1) for n in range(int(suffix))]

    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance*100, 4)) for feature, importance in zip(feature_list, averaged_importances)]

    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

    # Print out the feature and importances
    with open('results/importances', 'w') as fw:
        for i, pair in enumerate(feature_importances):
            result = '{}\t{}'.format(*pair)
            fw.write(result + '\n')
            if i < 20 or i > len(feature_importances) - 20:
                print(result)


print_importances()




