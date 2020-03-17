import numpy as np
from copy import deepcopy
from DeepESN import DeepESN
from utils import select_indexes, plot_network_outputs
from itertools import chain


def train_val(dataset, configs, indexes, error_function):

    TR_indexes, VL_indexes, TS_indexes = indexes
    deepESN = DeepESN(configs)

    states = deepESN.computeState((dataset.inputs), deepESN.IPconf.DeepIP)

    train_states = select_indexes(states, list(TR_indexes) + list(VL_indexes), configs.transient)
    train_targets = np.asarray(select_indexes(dataset.targets, list(TR_indexes) + list(VL_indexes), configs.transient))

    test_states = select_indexes(states, TS_indexes, configs.transient)
    test_targets = np.asarray(select_indexes(dataset.targets, TS_indexes, configs.transient))

    if configs.rebalance_states:
        train_states, train_targets = deepESN.rebalanceStates(train_states, train_targets)

    deepESN.trainReadout(train_states, train_targets, configs.reg)

    train_outputs = deepESN.computeOutput(train_states)
    train_error = error_function(train_outputs, train_targets)

    test_outputs = deepESN.computeOutput(test_states)
    test_error = error_function(test_outputs, test_targets)
    print('Test ACC: ', np.mean(test_error), '\n')

    #plot_network_outputs(select_indexes(dataset.inputs, list(TR_indexes) + list(VL_indexes), configs.transient), train_outputs)
    #plot_network_outputs(select_indexes(dataset.inputs, list(TS_indexes), configs.transient), test_outputs)

    return test_error, deepESN
