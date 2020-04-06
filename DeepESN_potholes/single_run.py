import os
import numpy as np
from DeepESN import DeepESN_skl
from model_selection import model_selection
from sklearn.metrics import classification_report
from utils import ACC_eq, F1_score, config_PH, load_PH, select_indexes, resample_dataset, plot_timeseries_clf, plot_min_max_mean_acc_prediction, find_roc_threshold, split_timeseries
from sklearn.model_selection import ParameterGrid
from NNReadout import ReadoutModel
from copy import deepcopy
import time

STEP = 10
CHUNK_LEN = 100

# fix a seed for the reproducibility of results
np.random.seed(7)

def main():

    # dataset path
    path = "datasets"
    path = os.path.join(path, 'accelerometer_2020-04-03T160405505Z.csv')

    print("Loading data from {}".format(path))

    dataset, Nu, error_function, optimization_problem, _X_train, y_train, _X_test, y_test, unorm_X_train = load_PH(path, F1_score)

    unorm_X_train = unorm_X_train[:,:,:STEP*(_X_train.shape[-1]//STEP)]
    _X_train = _X_train[:,:,:STEP*(_X_train.shape[-1]//STEP)]
    y_train = y_train[:,:,:STEP*(y_train.shape[-1]//STEP)]
    _X_test =  _X_test[:,:,:STEP*(_X_test.shape[-1]//STEP)]
    y_test = y_test[:,:,:STEP*(y_test.shape[-1]//STEP)]

    y_original_true = np.copy(y_train)

    _X_train = split_timeseries(_X_train, chunk_len=CHUNK_LEN, step=STEP)
    y_train = split_timeseries(y_train, chunk_len=CHUNK_LEN, step=STEP)
    _X_test = _X_train
    y_test = y_train

    print("----------")
    print("_X_train:", _X_train.shape)
    print("y_train:", y_train.shape)
    print("_X_test:", _X_test.shape)
    print("y_test:", y_test.shape)
    print("----------")

    # original_len per salvare la lunghezza originale della serie temporale
    # per poi poter ricostruire il dataset
    original_len = _X_test.shape[-1]
    # Il dataste viene concatenato in un unica  series temporale per migliorare le performance
    # Il risultaro è un unica serier lunga original_len * lunghezza
  

    print("DATASET TR_LEN={}, TS_LEN={}".format(len(_X_train), len(_X_test)))

    # load configuration for Earthquake task
    _configs = config_PH(list(range(_X_test.shape[0])), Nu)

    print(_X_train.shape)

    # plt = plot_timeseries_clf(_X_train[:3], y_train[:3], save=False)
    # plt.show()

    grid = {"Nl":[4], "rhos": [0.7], "iss": [0.8], "lis": [1.0], "Nr": [30], "input_mul": [1]}
    params = list(ParameterGrid(grid))
    print("Tested parameters: ", params)
    print("Estimated running time: {} min.".format(len(params) * 5))
    for param in params:
        print("Testing params: {}".format(param))
        configs = deepcopy(_configs)
        for key in param:
            if hasattr(configs, key):
                setattr(configs, key, param[key])
            else:
                print("Config has not param={}".format(key))
            if key == "input_mul":
                X_train = _X_train * param["input_mul"]
                X_test = _X_test * param["input_mul"]

        print("\nOn validation:\n")        
        print("Training...")
        deep_esn = DeepESN_skl(configs=configs.to_dict(), error_function=error_function, **param)
        deep_esn.fit(X_train, y_train)
        # deep_esn.save_model('./pretraineds/model_1_apr_4.h5f')
        print("BEST Nl=", deep_esn._deepESN.Nl)
        print("BEST rhos=", deep_esn._deepESN.rhos)
        print("BEST Nr=", deep_esn._deepESN.Nr)

        print("Predicting on train...")
        start = time.time()
        y_train_pred = deep_esn.predict(X_train, verbose=1)
        y_train_true = deep_esn.remove_transient(y_train)
        print("BEST_ESN TRAIN SCORE: {} in {} sec.".format(error_function(y_train_pred, y_train_true), time.time() - start))

        print("Predicting on test...")
        start = time.time()
        y_pred = deep_esn.predict(X_test, verbose=1)
        y_true = deep_esn.remove_transient(y_test)
        print("BEST_ESN TEST SCORE: {} in {} sec.".format(error_function(y_pred, y_true), time.time() - start))

        y_true = y_original_true
        result = []
        for index, i in enumerate(y_pred.reshape(y_pred.shape[-1]//CHUNK_LEN,CHUNK_LEN)):
            if index == 0:
                result = [i]
            else:
                result.append(i[-STEP:])
        y_pred = np.expand_dims(np.concatenate(result), 0)

        print("Out shapes y_true {} and y_pred {}".format(y_true.shape, y_pred.shape))


        threshold = find_roc_threshold(np.concatenate(y_true, 1), y_pred, plot=False)

        # y_pred = np.concatenate(y_pred, 1)
        y_true = y_true[0]
        print("YTRUE SHAPE", y_true.shape)
        # for classification result, threshold = 0
        y_pred[y_pred > threshold] = 1
        y_pred[y_pred <= threshold] = -1
        print(classification_report(y_true.squeeze(), y_pred.squeeze()))
        print("-" * 50)

        # reshape_timeseriesEQ spezza la lunga serie temporale nei pezzettini originali, il pezzo compromesso dal transient viene rimosso
        # viene rimossa la prima serie temporale perchè tagliata dal transient
        print(unorm_X_train.shape)
        print(y_true.shape)
        print(y_pred.shape)
        plt = plot_timeseries_clf(unorm_X_train, y_true, configs.transient)
        plt = plot_timeseries_clf(unorm_X_train, y_pred, configs.transient)
        plt.show()

if __name__ == "__main__":
    main()
