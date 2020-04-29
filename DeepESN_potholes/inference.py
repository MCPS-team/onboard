import numpy as np
import time
import os
from .DeepESN import DeepESN_skl
from .utils import best_config_PH
from .utils import F1_score, best_config_PH, load_PH, plot_timeseries_clf, split_timeseries

PRETAINED_MODEL_PATH = './DeepESN_potholes/pretraineds/model_1_apr_4.h5f'
MIN_SAMPLES = 3
# MAX_GAP > config.detect_delay => MAX_GAP ignored
MAX_GAP = 11
NU = 3

_configs = best_config_PH([], NU)
deep_esn = DeepESN_skl(configs=_configs.to_dict())
deep_esn.restore_model(PRETAINED_MODEL_PATH)

def ascii_plot_potholes(y_pred):
    for y in y_pred:
        print(''.join(["-" if a == -1 else '#' for a in y.squeeze()]))


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

def inference(unnorm_X, config, verbose=0):
    if verbose:
        print("Detecting potholes...")
        start = time.time()
    X = np.zeros(unnorm_X.shape)
    for i in range(X.shape[0]):
        X[i, :] = (unnorm_X[i, :] -
                   config.data_normalization_mean[i]) / (10 * config.data_normalization_std[i])
    y_pred = deep_esn.predict([X], verbose=0)
    y_pred = y_pred.squeeze()[-config.detect_delay:]
    threshold = -0.55
    y_pred[y_pred > threshold] = 1
    y_pred[y_pred <= threshold] = -1

    # out = y_pred
    out = cluster_ts(y_pred, min_samples=MIN_SAMPLES, max_gap=MAX_GAP)

    if verbose:
        print("Time elapsed : {} sec.".format(time.time()-start))

    if verbose > 1:
        print("Potholes preview")
        ascii_plot_potholes(out)

    return out


def inference_all_data(origin_X, config,  window_size=200, detect_delay=20):
    norm_X = np.zeros(origin_X.shape)
    for i in range(origin_X.shape[0]):
        norm_X[i, :] = (origin_X[i, :] -
                   config.data_normalization_mean[i]) / (10 * config.data_normalization_std[i])
    X = split_timeseries(norm_X, chunk_len=window_size, step=detect_delay)
    print("split time series shape", X.shape)

    result = [np.zeros((1, window_size-detect_delay))]
    for _X in X:
        y_pred = deep_esn.predict([_X], verbose=0)
        threshold = 0
        y_pred[y_pred > threshold] = 1
        y_pred[y_pred <= threshold] = -1
        result.append(y_pred[:, -detect_delay:])

    result = np.concatenate(result, axis=1)
    result = np.array([cluster_ts(r, min_samples=MIN_SAMPLES, max_gap=MAX_GAP)
           for r in result])
    print(norm_X.shape, result.shape)
    plt = plot_timeseries_clf(norm_X, result, transient=0)
    plt.show()
