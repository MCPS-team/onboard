import os
import numpy as np
from DeepESN import DeepESN_skl
from utils import F1_score, best_config_PH, load_PH, plot_timeseries_clf, split_timeseries
import time
# fix a seed for the reproducibility of results
np.random.seed(7)

PRETAINED_MODEL_PATH = './pretraineds/model_1_apr_4.h5f'
STEP = 10
CHUNK_LEN = 50


def plot_potholes(y_pred):
    for y in y_pred:
        print(''.join(["-" if a == -1 else '#' for a in y.squeeze()]))


def main():

    # dataset path
    path = "datasets"
    path = os.path.join(path, 'accelerometer_2020-04-03T160405505Z.csv')

    print("Loading data from {}".format(path))

    dataset, Nu, error_function, optimization_problem, _X, y, _, _, unorm_X_train = load_PH(
        path, F1_score)

    # remove exceded datat for step
    print("_X", _X.shape)
    _X = _X[:,:,:STEP*(_X.shape[-1]//STEP)]
    unorm_X_train = unorm_X_train[:,:,:STEP*(unorm_X_train.shape[-1]//STEP)]
    print("_X", _X.shape)

    origin_X = _X
    X = split_timeseries(_X, chunk_len=CHUNK_LEN, step=STEP)
    # y = split_timeseries(y, chunk_len=CHUNK_LEN, step=STEP)
    print("----------")
    print("X:", X.shape)
    print("----------")

    print("DATASET TR_LEN={}, TS_LEN={}".format(len(X), len(y)))

    # load configuration for Earthquake task
    _configs = best_config_PH(list(range(X.shape[0])), Nu)

    result = []

    deep_esn = DeepESN_skl(configs=_configs.to_dict(),
                           error_function=error_function)
    deep_esn.restore_model(PRETAINED_MODEL_PATH)
    print("Predicting on test...")
    start = time.time()
    states = None
    for _X in X:
        y_pred = deep_esn.predict([_X], verbose=0)
        threshold = -0.555764
        y_pred[y_pred > threshold] = 1
        y_pred[y_pred <= threshold] = -1
        if len(result)==0:
            result.append(y_pred[:,:])
        else:
            result.append(y_pred[:, -STEP:])

    print("Time elapsed : {} sec.".format(time.time()-start))

    # reshape_timeseriesEQ spezza la lunga serie temporale nei pezzettini originali, il pezzo compromesso dal transient viene rimosso
    # viene rimossa la prima serie temporale perchè tagliata dal transient
    result = np.concatenate(result, axis=1)
    print(result.shape)
    plot_potholes(result)
    plt = plot_timeseries_clf(unorm_X_train, result, transient=0)
    plt.show()

    print("After clustering")
    result = np.array([cluster_ts(r, 3, 3) for r in result])
    plot_potholes(result)
    plt = plot_timeseries_clf(
        unorm_X_train, result, transient=0)
    plt.show()


def cluster_ts(X, min_samples=3, max_gap=3, nested=False):
    X = X.squeeze()
    out = np.zeros((len(X),)) - 1
    last_pos_index = 0
    group_indexes = []
    for i in range(len(X)):
        if X[i] == 1:
            last_pos_index = i
            group_indexes.append(i)
        else:
            # Se finisce l'iterazione
            if i > last_pos_index+max_gap:
                if len(group_indexes) >= min_samples:
                    for z in range(group_indexes[0], group_indexes[-1]):
                        out[z] = 1
                group_indexes = []
    if len(group_indexes) >= min_samples:
        for z in range(group_indexes[0], group_indexes[-1]):
            out[z] = 1
    if nested:
        out = cluster_ts(out, min_samples, max_gap*2, nested=False)
    return out


if __name__ == "__main__":
    main()
