import os
import numpy as np
from DeepESN import DeepESN_skl, DeepESN
from utils import F1_score, config_EQ, load_EQ
from model_selection import model_selection
from sklearn.metrics import classification_report
from utils import F1_score, config_EQ, load_EQ, select_indexes, restore_grid_search_history, plot_timeseries_clf, plot_min_max_mean_acc_prediction

# fix a seed for the reproducibility of results
np.random.seed(7)


def main():

    # dataset path
    path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(path, 'datasets/pezzetto/*')

    print("Loading data from {}".format(path))

    dataset, Nu, error_function, optimization_problem, TR_indexes, VL_indexes, TS_indexes = load_EQ(path, F1_score)

    X_train = np.array([dataset.inputs[i] for i in TR_indexes[-int(len(TR_indexes) * 1):]] + [dataset.inputs[i] for i in VL_indexes])
    y_train = np.array([dataset.targets[i] for i in TR_indexes[-int(len(TR_indexes) * 1):]] + [dataset.targets[i] for i in VL_indexes])
    X_test = np.array([dataset.inputs[i] for i in TS_indexes[:int(len(TS_indexes) * 1)]])
    y_test = np.array([dataset.targets[i] for i in TS_indexes[:int(len(TS_indexes) * 1)]])
    print("DATASET TR_LEN={}, TS_LEN={}".format(len(X_train), len(X_test)))

    # load configuration for Earthquake task
    configs = config_EQ(list(range(X_test.shape[0])), Nu)

    print(X_train.shape)

    # plt = plot_timeseries_clf(X_train[:3], y_train[:3], save=False)
    # plt.show()

    # params = {"design_deep": [True], "eta": [0.01], "rho": [0.9], "iss": [10], "Nr": [20, 50, 100], "min_layers": [5], "max_layers": [15]}
    # best_params, cv_scores, gs = model_selection(configs, X_train, y_train, params, error_function, save_dir="./log/",
    #                                              k_fold=3, outer_k_fold=0, shuffle=True, n_jobs=4, verbose=10)

    # print("#--> DoubleCV score: {}".format(cv_scores.mean()))

    print("\nOn validation:\n")
    # deep_esn = deep_esn_skl(configs=configs.to_dict(), Nl:5, rho:0.9, design_deep=False)
    configs.Nl = 4
    configs.Nr = 100
    configs.rhos = 1.0
    configs.lis = 1.0
    configs.iss = 10
    deep_esn = DeepESN(configs)
    print("BEST Nl=", deep_esn.Nl)
    print("BEST rhos=", deep_esn.rhos)
    print("BEST Nr=", deep_esn.Nr)

    # Training
    states = deep_esn.computeState(X_train, deep_esn.IPconf.DeepIP)
    indexes = np.arange(len(X_train))
    train_states = select_indexes(states, indexes, configs.transient)
    train_targets = np.asarray(select_indexes(y_train, indexes, configs.transient))

    if configs.rebalance_states:
        train_states, train_targets = deep_esn.rebalanceStates(train_states, train_targets)

    deep_esn.trainReadout(train_states, train_targets, configs.reg)

    # Evaluate Training
    states = deep_esn.computeState(X_train, deep_esn.IPconf.DeepIP)
    test_states = select_indexes(states, indexes, configs.transient)
    y_train_pred = deep_esn.computeOutput(test_states)
    y_train_true = select_indexes(y_train, indexes, configs.transient)
    print("BEST_ESN TRAIN SCORE:   ", error_function(y_train_pred, y_train_true))

    # Evaluate Test
    states = deep_esn.computeState(X_test, deep_esn.IPconf.DeepIP)
    indexes = np.arange(len(X_test))
    test_states = select_indexes(states, indexes, configs.transient)
    y_pred = deep_esn.computeOutput(test_states)
    y_true = select_indexes(y_test, indexes, configs.transient)
    print("BEST_ESN TEST SCORE:   ", error_function(y_pred, y_true))
    print(np.array(y_true).shape)
    plt = plot_min_max_mean_acc_prediction(np.array([X_test[i] for i in range(0, 3)]), np.array([[y_pred[0][(5000 - configs.transient) * i:(5000 - configs.transient) * (i + 1)]] for i in range(0, 3)]),
                                           np.array([y_true[i] for i in range(0, 3)]), configs.transient, error_function, save=True)
    plt.show()
    
    y_true = np.concatenate(y_true, 1)
    # for classification result, threshold = 0
    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0] = -1
    print(classification_report(y_true.squeeze(), y_pred.squeeze()))
    print("-" * 50)


if __name__ == "__main__":
    main()
