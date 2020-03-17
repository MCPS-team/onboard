
import os
import pickle
from datetime import datetime
import numpy as np


def save_grid_search_history(cv_results, path="./"):
    check_dir(path)
    path = os.path.join(path, "grid-search-{}.pkl".format(strtime()))
    with open(path, 'wb') as handle:
        pickle.dump(cv_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Grid Search results saved in {}".format(path))
    return path


def restore_grid_search_history(path):
    with open(path, 'rb') as handle:
        pkl_file = pickle.load(handle)
    return pkl_file


def save_predictions(y_out, path="./"):
    check_dir(path)
    path = os.path.join(path, "predictions-{}.pkl".format(strtime()))
    with open(path, 'wb') as handle:
        pickle.dump(y_out, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Output preditcions saved in {}".format(path))
    return path


def save_statistic_report(X, y, y_out, params, cv_results, path="./log"):
    ''' Call this for save all useful results '''
    save_predictions(y_out)
    save_grid_search_history(cv_results)

    # save tested params
    path = os.path.join(path, "tested params-{}.pkl".format(strtime()))
    with open(path, 'wb') as handle:
        pickle.dump(y_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return True


def strtime():
    now = datetime.now()
    strtime = now.strftime('%Y%m%d_%H:%M')
    return strtime


def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == "__main__":
    file_path = save_grid_search_history({"aa": {"a": "aaa", "b": 0.1}})
    obj = restore_grid_search_history(file_path)
