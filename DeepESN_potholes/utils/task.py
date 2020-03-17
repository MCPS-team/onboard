'''
Task configuration file

----

This file is a part of the DeepESN Python Library (DeepESNpy)

Luca Pedrelli
luca.pedrelli@di.unipi.it
lucapedrelli@gmail.com

Department of Computer Science - University of Pisa (Italy)
Computational Intelligence & Machine Learning (CIML) Group
http://www.di.unipi.it/groups/ciml/

----
'''

import functools
import os
import numpy as np
import pandas as pd
import glob
import errno
import random
import itertools
from scipy.io import loadmat
import json


class Struct(object):
    pass


def select_indexes(data, indexes, transient=0):
    # da decommentare con dataset MG altrimenti commenta
    # if len(data) == 1:
    #     return [data[0][:, indexes][:, transient:]]

    return [data[i][:, transient:] for i in indexes]


def load_pianomidi(path, metric_function):

    data = loadmat(os.path.join(path, 'pianomidi.mat'))  # load dataset

    dataset = Struct()
    dataset.name = data['dataset'][0][0][0][0]
    dataset.inputs = data['dataset'][0][0][1][0].tolist()
    dataset.targets = data['dataset'][0][0][2][0].tolist()

    # input dimension
    Nu = dataset.inputs[0].shape[0]

    # function used for model evaluation
    error_function = functools.partial(metric_function, 0.5)

    # select the model that achieves the maximum accuracy on validation set
    optimization_problem = np.argmax

    # 124 traces, every trace have 88 inputs and a variable length
    # indexes for training, validation and test set in Piano-midi.de task
    TR_indexes = range(87)
    VL_indexes = range(87, 99)
    TS_indexes = range(99, 124)

    return dataset, Nu, error_function, optimization_problem, TR_indexes, VL_indexes, TS_indexes


def load_MG(path, metric_function):

    data = loadmat(os.path.join(path, 'MG.mat'))  # load dataset

    dataset = Struct()
    dataset.name = data['dataset'][0][0][0][0]
    dataset.inputs = data['dataset'][0][0][1][0]
    dataset.targets = data['dataset'][0][0][2][0]

    # input dimension
    Nu = dataset.inputs[0].shape[0]

    # function used for model evaluation
    error_function = functools.partial(metric_function, 0.5)

    # select the model that achieves the maximum accuracy on validation set
    optimization_problem = np.argmin

    # Only 1 trace, with 1 input and 10000 points
    # indexes for training, validation and test set in Piano-midi.de task
    TR_indexes = range(4000)
    VL_indexes = range(4000, 5000)
    TS_indexes = range(5000, 9999)

    return dataset, Nu, error_function, optimization_problem, TR_indexes, VL_indexes, TS_indexes


def load_PH(path, metric_function):

    df = pd.read_csv(path,  names=['timestamp', 'x', 'y', 'z', 'target'])
    df['target'] = [1.0 if x else -1.0 for x in df['target']]
    print(df)

    dataset = Struct()
    dataset.name = path
    dataset.inputs = np.expand_dims(
        df.drop(["timestamp", 'target'], axis=1).to_numpy().T, axis=0)
    dataset.targets = np.expand_dims([df['target'].to_numpy()], axis=0)

    # input dimension
    Nu = len(dataset.inputs[0])

    # function used for model evaluation
    error_function = functools.partial(metric_function, 0)

    # select the model that achieves the maximum f1-score on validation set
    optimization_problem = np.argmax

    print("# DATASET INPUT SIZE: {}, DATASET TRAGET SIZE: {}".format(
        dataset.inputs.shape, dataset.targets.shape))
    # many traces, with 3 inputs and a variable length
    # indexes for training, validation and test set in Earthquake task
    X_train, X_test = split_dataset(dataset.inputs, 0.8)
    Y_train, Y_test = split_dataset(dataset.targets, 0.8)

    print("#-> CONSIDERED SLICE SIZE: {}".format(len(X_train) + len(Y_train)))
    print("-" * 50)
    X_train, X_test = normalize_input(X_train, X_test)

    return dataset, Nu, error_function, optimization_problem, X_train, Y_train, X_test, Y_test


def normalize_input(X_train, X_test):
    '''
    Normalize the dataset, by using only the training elements to choose
    a normalization coefficient, and applying the same coefficient to all
    the data.
    '''

    TR_data=X_train
    Nu=len(TR_data[0])

    # Valuate Mean, Max and Min of every input for training data
    flat_inp=[list(itertools.chain.from_iterable(X_train[:, i]))
                for i in range(Nu)]
    means=np.mean(flat_inp, axis=1)
    stds=np.std(flat_inp, axis=1)

    print("NORMALIZATION mean={}, std={}".format(means, stds))

    for i in range(Nu):
        X_train[:, i]=(X_train[:, i] -
                         means[i]) / (10 * stds[i])
        X_test[:, i]=(X_test[:, i] -
                        means[i]) / (10 * stds[i])

    return X_train, X_test


def split_dataset(data, split_perc=0.7):
    length=data.shape[-1]
    train_len=int(length*split_perc)
    return data[:, :, :train_len], data[:, :, train_len:]

def split_timeseries(data, chunk_len=100, step=1):
    new_data = []
    length = data.shape[-1]
    offset = length%chunk_len
    for i in range(offset, length, step):
        new_data.append(data[:, : , i:i+chunk_len])
        if i+chunk_len >= length:
            break
    return np.concatenate(new_data, axis=0)
