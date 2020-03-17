from __future__ import print_function
import sys
from DeepESN import DeepESN_skl
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from utils import save_grid_search_history


def model_selection(configs, X_train, y_train, params=None, error_function=None, k_fold=3, outer_k_fold=0, n_jobs=2, random_state=420, shuffle=True, save_dir=None, verbose=0):
    ''' Perform model selection
    @param configs: model's configs
    @param params: dict of list of model parameters (see sklearn GridSearchCV), optional in inner_k_fold <= 0
    @param error_function: error function given by configs
    @param k_fold: outer k fold for double cv
    @param inner_k_folds: inner_k_fold for double cv, if it is >= 0 only cv will be performed using the configs settting
    @param n_jobs: parallelized jobs (see sklearn)
    @param randon_state: random state
    @param verbose: see sklearn

    @returns: dict with best_parameters, scores on outer cv, Grid search object (or DeepESN_skl object if inner_kv_fold<=0)
    '''
    inner_cv = KFold(n_splits=k_fold, shuffle=shuffle, random_state=random_state)
    if outer_k_fold > 0:
        outer_cv = KFold(n_splits=outer_k_fold, shuffle=shuffle, random_state=random_state)

    configs_dict = configs.to_dict()
    base_model = DeepESN_skl(configs=configs_dict, error_function=error_function)
    print(base_model)

    p_grid = {"configs": [configs_dict], "error_function": [error_function]}
    p_grid.update(params)
    gs = GridSearchCV(estimator=base_model, param_grid=p_grid, cv=inner_cv, n_jobs=n_jobs, refit=False, error_score=-1, verbose=verbose, return_train_score=True)
    gs.fit(X_train, y_train)
    best_params = gs.best_params_

    if outer_k_fold > 0:
        nested_score = cross_val_score(gs, X_train, y_train, cv=outer_cv, n_jobs=n_jobs)
    else:
        nested_score = gs.cv_results_['mean_test_score']

    if verbose > 0:
        print("\nBest parameters set found on development set:\n")
        print(gs.best_params_)
        print("-" * 50)
        print("Grid scores on development set:\n")
        means = gs.cv_results_['mean_test_score']
        stds = gs.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, gs.cv_results_['params']):
            print("mean: %0.4f , std: (+/-%0.04f) for %r"
                  % (mean, std * 2, params))
        print("-" * 50)
        if outer_k_fold > 0:
            print("Double cross validation scores:")
            print("mean: %0.4f , std: (+/-%0.04f)"
                  % (nested_score.mean(), nested_score.std()))
        print("end" + "." * 10)

    if save_dir is not None:
        save_grid_search_history(gs, save_dir)

    return best_params, nested_score, gs


if __name__ == '__main__':
    import numpy as np
    from utils import ACC_eq, F1_score, config_EQ, load_EQ, select_indexes, restore_grid_search_history, resample_dataset, plot_timeseries_clf, plot_min_max_mean_acc_prediction, plot_grid_search_validation_curve
    from sklearn.metrics import classification_report

    path = './datasets/pezzetto/*'
    print("Loading data from {}".format(path))

    dataset, Nu, error_function, optimization_problem, TR_indexes, VL_indexes, TS_indexes = load_EQ(path, F1_score)
    error_function = ACC_eq

    X_train = np.array([dataset.inputs[i] for i in TR_indexes] + [dataset.inputs[i] for i in VL_indexes])
    y_train = np.array([dataset.targets[i] for i in TR_indexes] + [dataset.targets[i] for i in VL_indexes])
    X_test = np.array([dataset.inputs[i] for i in TS_indexes]) 
    y_test = np.array([dataset.targets[i] for i in TS_indexes])

    X_train, y_train = resample_dataset(X_train, y_train, int(X_train.shape[-1] / 2))
    X_test, y_test = resample_dataset(X_test, y_test, int(X_test.shape[-1] / 2))
    original_len = X_train.shape[-1]
    print("DATASET TR_LEN={}, TS_LEN={}".format(len(X_train), len(X_test)))

    # load configuration for Earthquake task
    configs = config_EQ(list(range(X_test.shape[0])), Nu)

    plt = plot_timeseries_clf(X_train[:3], y_train[:3], save=False)
    plt.show()

    params = {"design_deep": [False], "Nl": [4], "rhos": [0.5, 0.7], "iss": [0.01, 0.1, 1], "lis": [0.01, 0.1, 0.3], "Nr": [50]}
    best_params, cv_scores, gs = model_selection(configs, X_train, y_train, params, error_function, save_dir="./log/",
                                                 k_fold=3, outer_k_fold=0, shuffle=False, n_jobs=2, verbose=5)

    print("#--> DoubleCV score: {}".format(cv_scores))

    print("\nOn validation:\n")
    # # best_params['design_deep'] = True
    # # best_params['min_layers'] = best_params['Nl'] - 1
    # # best_params['max_layers'] = best_params['Nl'] + 5
    # best_params = {"Nl": 3, "rhos": 0.9, "iss": 100, "lis": 0.5, "Nr": 100}
    deep_esn = DeepESN_skl(**best_params)
    deep_esn.fit(X_train, y_train)
    print("BEST Nl=", deep_esn._deepESN.Nl)
    print("BEST rhos=", deep_esn._deepESN.rhos)
    print("BEST Nr=", deep_esn._deepESN.Nr)


    y_train_pred = deep_esn.predict(X_train)
    y_train_true = deep_esn.remove_transient(y_train)
    print("BEST_ESN TRAIN SCORE:{}".format(error_function(np.concatenate(y_train_pred, 1), y_train_true)))

    y_pred = deep_esn.predict(X_test)
    y_true = deep_esn.remove_transient(y_test)
    print("BEST_ESN TEST SCORE:{}".format(error_function(np.concatenate(y_pred, 1), y_true)))

    plt = plot_min_max_mean_acc_prediction(X_test, y_pred, y_true, configs.transient, error_function, save=True)
    plt.show()

    # y_pred = np.concatenate(y_pred, 1)
    y_true = np.concatenate(y_true, 1)
    # for classification result, threshold = 0
    
    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0] = -1
    print(classification_report(y_true.squeeze(), y_pred.squeeze()))
    print("-" * 50)
