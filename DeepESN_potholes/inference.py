import numpy as np
import time
import os 
from .DeepESN import DeepESN_skl
from .utils import best_config_PH

PRETAINED_MODEL_PATH = './DeepESN_potholes/pretraineds/model_0_mar_6.h5f'
MIN_SAMPLES=10
MAX_GAP=5
NU = 3

_configs = best_config_PH([], NU)
deep_esn = DeepESN_skl(configs=_configs.to_dict())
deep_esn.restore_model(PRETAINED_MODEL_PATH)

def ascii_plot_potholes(y_pred):
    for y in y_pred:
        print(''.join(["-" if a == -1 else '#' for a in y.squeeze()]))

def cluster_ts(X, min_samples=10, max_gap=5):
    X = X.squeeze()
    out = np.zeros((len(X),)) - 1
    last_pos_index = 0
    consecutive_pos = 0
    group_indexs = []
    for i in range(len(X)):
        if X[i] == 1:
            last_pos_index = i
            group_indexs.append(i)
        else:
            # Se finisce l'iterazione
            if i >= last_pos_index+max_gap:
                if len(group_indexs) >= min_samples:
                    for z in range(group_indexs[0], group_indexs[-1]):
                        out[z] = 1
                group_indexs = []
    if len(group_indexs) >= min_samples:
        for z in range(group_indexs[0], group_indexs[-1]):
            out[z] = 1
    return out


def inference(X, verbose=0):
    if verbose:
        print("Detecting potholes...")
        start = time.time()
    y_pred = deep_esn.predict([X], verbose=0)
      
    threshold = 0
    y_pred[y_pred > threshold] = 1
    y_pred[y_pred <= threshold] = -1

    out = [cluster_ts(x, min_samples=MIN_SAMPLES, max_gap=MAX_GAP) for x in y_pred]

    if verbose:
        print("Time elapsed : {} sec.".format(time.time()-start))

    if verbose > 1:
        print("Potholes preview")
        ascii_plot_potholes(out)
    
    return out
    


    
