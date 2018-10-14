#!/usr/bin/env python3

import argparse
import copy
import cv2
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

def create_dataset(data, color_space, predicted_color=None):
    buffered_data = data[['image', 'seed_number', 'color']].copy()
    channels = data[['blue', 'green', 'red']].values
    channels = channels.astype(np.uint8).reshape(1, channels.shape[0], channels.shape[1])
    if color_space == 'lab':
        channels = cv2.cvtColor(channels, cv2.COLOR_BGR2Lab)
    elif color_space == 'hsv':
        channels = cv2.cvtColor(channels, cv2.COLOR_BGR2HSV)
    buffered_data['channel1'] = channels[0, :, 0]
    buffered_data['channel2'] = channels[0, :, 1]
    buffered_data['channel3'] = channels[0, :, 2]

    if predicted_color is None:
        features = buffered_data.groupby(['image', 'seed_number'])[['channel1', 'channel2', 'channel3']]\
                .median().sort_index()
        answers = buffered_data.groupby(['image', 'seed_number'])['color'].min().sort_index()
        if color_space == 'rgb':
            print("%d seeds" % (len(features)))
            print("Purple: %d, red: %d, white %d" % (sum(answers == 'purple'), sum(answers == 'red'), sum(answers == 'white')))
        features['answer'] = 0
        features.loc[answers == 'red', 'answer'] = 1
        features.loc[answers == 'white', 'answer'] = 2
    else:
        features = buffered_data.groupby(['image', 'seed_number'])[['channel1', 'channel2', 'channel3']]\
                .median().sort_index()
        answers = buffered_data.groupby(['image', 'seed_number'])['color'].min().sort_index()
        features['answer'] = 1
        features.loc[answers != predicted_color, 'answer'] = 0

    answers = features.answer.values.astype(np.int32)
    features = features.drop(['answer'], axis=1).values.astype(np.float32)

    return features, answers

def filter_data(data, photo_info, type):
    result = data[data.type == type].copy()
    result['image'] = result.file.apply(lambda x: x.split('/')[-1])
    result = result.join(photo_info.set_index('image'), on=['image'], how='inner')
    result = result[['red', 'green', 'blue', 'image', 'seed_number', 'color']]
    return result

def grid_search(data, color=None):
    kfold = KFold(n_splits=10, shuffle=True)
    best_accuracy = 0.0
    best_params = {}

    # lab or hsv can also be used
    for color_space in ['rgb', 'lab', 'hsv']:
        X, y = create_dataset(data, color_space, color)
        for model, classifier in iterate_classifier():
            accuracy = []
            for train_index, test_index in kfold.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                classifier.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
                answers = classifier.predict(X_test)[1].ravel()
                accuracy.append((answers == y_test).mean())
            mean_accuracy = np.mean(accuracy)
            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                best_params['color_space'] = color_space
                best_params['model'] = model
                if model == 'svm':
                    best_params['C'] = classifier.getC()
                    best_params['gamma'] = classifier.getGamma()
                else:
                    best_params['K'] = classifier.getDefaultK()

    if best_params['model'] == 'svm':
        print('Color space: %s, SVM, C: %f, gamma: %f, cross-val accuracy: %.2f' \
              % (best_params['color_space'], best_params['C'], best_params['gamma'], best_accuracy * 100.0))
    else:
        print('Color space: %s, KNN, K: %f, cross-val accuracy: %.2f' \
              % (best_params['color_space'], best_params['K'], best_accuracy * 100.0))

    return best_params

def iterate_svm():
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    for C in np.logspace(-2, 2, 20):
        for gamma in np.logspace(-3, 0, 20):
            svm.setC(C)
            svm.setGamma(gamma)
            yield svm

def iterate_knn():
    knn = cv2.ml.KNearest_create()
    for k in range(1, 30):
        knn.setDefaultK(k)
        yield knn

def iterate_classifier():
    for svm in iterate_svm():
        yield 'svm', svm
    for knn in iterate_knn():
        yield 'knn', knn

parser = argparse.ArgumentParser(description='Train the seed classifier, choose the best parameters')
parser.add_argument('--photo-info', dest='photo_info',
        required=True, help='Tsv file containing images\' filenames with colors')
parser.add_argument('--seed-data', dest='seed_data',
        required=True, help='Tsv file containing seed pixels')

args = parser.parse_args()
photo_info = pd.read_csv(args.photo_info, sep='\t')
seed_data = pd.read_csv(args.seed_data, sep='\t')

source_data = filter_data(seed_data, photo_info, 'source')
calibrated_data = filter_data(seed_data, photo_info, 'calibrated')

print('Source')
grid_search(source_data)
print('Calibrated')
grid_search(calibrated_data)
print('Source, purple')
grid_search(source_data, 'purple')
print('Calibrated, purple')
grid_search(calibrated_data, 'purple')
print('Source, red')
grid_search(source_data, 'red')
print('Calibrated, red')
grid_search(calibrated_data, 'red')
print('Source, white')
grid_search(source_data, 'white')
print('Calibrated, white')
grid_search(calibrated_data, 'white')
