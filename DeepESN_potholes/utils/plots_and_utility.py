import matplotlib.pyplot as plt
import numpy as np
from .save_restore import check_dir
from datetime import datetime
import matplotlib.transforms as mtransforms
import os
import pandas as pd
import numbers
import pickle
from scipy import signal

now = datetime.now()
strtime = now.strftime('%Y%m%d_%H%M')


def read_pickle(path='./log/gs_params_tested.pkl'):
    data = []
    with open(path, 'rb') as fr:
        try:
            while True:
                data.append(pickle.load(fr))
        except EOFError:
            pass
    return data


def resample_dataset(X, y, num=5000):
    ''' Resample data to num length '''
    print(X.shape, y.shape, np.unique(y))
    data = np.concatenate((X, y), axis=1)
    resampled_data = signal.resample(data, num, axis=-1)
    X = resampled_data[:, :-1, :]
    y = resampled_data[:, -1, :]
    y = np.expand_dims(np.where(y > 0, 1.0, -1.0), 1)
    print(X.shape, y.shape, np.unique(y))
    return X, y


def reshape_timeseriesEQ(X, shape, transient=0, remove_initial=0):
    ''' Adjust net output to original shape, removing first timeseries truncated by esn transient'''
    if transient > 0:
        # Da sistemare, supporta solo array 1 dimensionali
        X = np.concatenate((np.zeros(transient), np.array(X).squeeze()))
    else:
        X = np.array(X).squeeze()
    X = X.reshape(-1, shape[-2], shape[-1])[remove_initial:]
    print("reshaped to", X.shape)
    return X


def speed_concat_data(X, select_range=0):
    return np.expand_dims(np.concatenate(X, 1), axis=0)


def plot_parallel_gs(gs, params, title="", dir="./log/"):
    df_cv_results = pd.DataFrame(gs.cv_results_)
    train_scores_mean = df_cv_results['mean_train_score']
    valid_scores_mean = df_cv_results['mean_test_score']
    print(df_cv_results)


def plot_grid_search_validation_curve(grid, param_to_vary,
                                      title='Validation Curve', ylim=None,
                                      xlim=None, log=None, save=False):
    """Plots train and cross-validation scores from a GridSearchCV instance's
    best params while varying one of those params.
    https://matthewbilyeu.com/blog/2019-02-05/validation-curve-plot-from-gridsearchcv-results
    """

    df_cv_results = pd.DataFrame(grid.cv_results_)
    train_scores_mean = df_cv_results['mean_train_score']
    valid_scores_mean = df_cv_results['mean_test_score']
    train_scores_std = df_cv_results['std_train_score']
    valid_scores_std = df_cv_results['std_test_score']

    param_cols = [c for c in df_cv_results.columns if c[:6] == 'param_']
    param_ranges = [grid.param_grid[p[6:]] for p in param_cols]
    param_ranges_lengths = [len(pr) for pr in param_ranges]

    train_scores_mean = np.array(
        train_scores_mean).reshape(*param_ranges_lengths)
    valid_scores_mean = np.array(
        valid_scores_mean).reshape(*param_ranges_lengths)
    train_scores_std = np.array(
        train_scores_std).reshape(*param_ranges_lengths)
    valid_scores_std = np.array(
        valid_scores_std).reshape(*param_ranges_lengths)

    param_to_vary_idx = param_cols.index('param_{}'.format(param_to_vary))

    slices = []
    for idx, param in enumerate(grid.best_params_):
        if (idx == param_to_vary_idx):
            slices.append(slice(None))
            continue
        best_param_val = grid.best_params_[param]
        idx_of_best_param = 0
        if isinstance(param_ranges[idx], np.ndarray):
            idx_of_best_param = param_ranges[idx].tolist().index(
                best_param_val)
        else:
            idx_of_best_param = param_ranges[idx].index(best_param_val)
        slices.append(idx_of_best_param)

    train_scores_mean = train_scores_mean[tuple(slices)]
    valid_scores_mean = valid_scores_mean[tuple(slices)]
    train_scores_std = train_scores_std[tuple(slices)]
    valid_scores_std = valid_scores_std[tuple(slices)]

    plt.clf()

    plt.title(title)
    plt.xlabel(param_to_vary)
    plt.ylabel('Score')

    if (ylim is None):
        plt.ylim(0.0, 1.1)
    else:
        plt.ylim(*ylim)

    if (not (xlim is None)):
        plt.xlim(*xlim)

    lw = 2

    plot_fn = plt.plot
    if log:
        plot_fn = plt.semilogx

    param_range = param_ranges[param_to_vary_idx]
    if (not isinstance(param_range[0], numbers.Number)):
        param_range = [str(x) for x in param_range]
    plot_fn(param_range, train_scores_mean, label='Training score', color='r',
            lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color='r', lw=lw)
    plot_fn(param_range, valid_scores_mean, label='Cross-validation score',
            color='b', lw=lw)
    plt.fill_between(param_range, valid_scores_mean - valid_scores_std,
                     valid_scores_mean + valid_scores_std, alpha=0.1,
                     color='b', lw=lw)

    plt.legend(loc='lower right')
    if save:
        plt.savefig(
            "./log/plot_grid_search_validation_curve{}.svg".format(strtime))
    return plt


def plot_timeseries_clf(X, y, transient=None, save=False):
    ''' Plot the timeseries given by senesors and the classification target '''
    if transient:
        y = np.concatenate((np.zeros((y.shape[0], transient)), y), axis=1)
    fig, axs = plt.subplots(3, 1, figsize=(16, 10))
    for i, x in enumerate(X):
        trans = mtransforms.blended_transform_factory(
            axs[i].transData, axs[i].transAxes)
        for j, x_s in enumerate(x):
            axs[i].plot(range(len(x_s)), x_s, label='Sensor_{}'.format(j))
        axs[i].fill_between(range(len(x_s)), 0, 1, where=y[i] > 0,
                            facecolor='red', alpha=0.3, transform=trans, label="Target")
        axs[i].set_ylabel('magnitude')
        axs[i].set_xlabel('time')
        if i == 0:
            axs[i].legend()
    fig.suptitle('Time Series labeling')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    if save:
        plt.savefig("./log/plot_timeseries_clf_{}.svg".format(strtime))
    return plt


def plot_predicted_timeseries_clf(X, y_pred, y_true, transient, save=False):
    ''' Plot the timesreshape_timeseriesEQeries given by senesors and the classification target '''
    fig, axs = plt.subplots(3, 1, figsize=(16, 10))
    for i, x in enumerate(X):
        trans = mtransforms.blended_transform_factory(
            axs[i].transData, axs[i].transAxes)
        for j, x_s in enumerate(x):
            axs[i].plot(range(len(x_s)), x_s, label='Sensor_{}'.format(j))
        axs[i].fill_between(range(transient, len(x[0])), 0, 1, where=y_true[i] > 0,
                            facecolor='red', alpha=0.3, transform=trans, label="Target")
        axs[i].fill_between(range(transient, len(x[0])), 0, 1, where=y_pred[i] > 0,
                            facecolor='blue', alpha=0.3, transform=trans, label="Predicted")
        axs[i].set_ylabel('magnitude')
        axs[i].set_xlabel('time')
        if i == 0:
            axs[i].legend()
    fig.suptitle('Time Series labeling')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    if save:
        plt.savefig("./log/plot_timeseries_clf_{}.svg".format(strtime))
    return plt


def plot_min_max_mean_acc_prediction(X, y_pred, y_true, transient, error_function, save=False):
    score_list = []
    score_list.append(error_function(y_pred, [y_true]))
    index_max_score = np.argmax(score_list)
    index_min_score = np.argmin(score_list)
    index_mean_score = score_list.index(
        np.percentile(score_list, 50, interpolation='nearest'))
    print("(index, value) MIN SCORE={}, MAX_SCORE={}, MEAN_SCORE={}".format((index_min_score, score_list[index_min_score]), (
        index_max_score, score_list[index_max_score]), (index_mean_score, score_list[index_mean_score])))

    for j in [index_min_score, index_max_score, index_mean_score]:
        x_3 = [X[index_min_score], X[index_max_score], X[index_mean_score]]
        y_pred_3 = [y_pred[index_min_score],
                    y_pred[index_max_score], y_pred[index_mean_score]]
        y_true_3 = [y_true[index_min_score],
                    y_true[index_max_score], y_true[index_mean_score]]

    plt = plot_predicted_timeseries_clf(
        x_3, y_pred_3, y_true_3, transient, save=save)
    return plt


def plot_network_outputs(inputs, outputs):

    input_s = np.concatenate(inputs, 1)

    plot_len = -1
    n = range(plot_len)
    plt.plot(outputs[0, 0:plot_len])
    plt.plot(input_s[0, 0:plot_len])
    plt.plot(input_s[1, 0:plot_len])
    plt.plot(input_s[2, 0:plot_len])
    plt.plot(np.sign(outputs[0, 0:plot_len]))
    plt.ylabel('net output')
    plt.xlabel('time')
    plt.show()


def plot_states(all_states):
    plt.plot(all_states)
    plt.show()
