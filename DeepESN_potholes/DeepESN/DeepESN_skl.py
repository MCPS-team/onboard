import numpy as np
import os
import sys
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from . import DeepESN
from tqdm import tqdm
from copy import deepcopy
import pickle
from datetime import datetime
import time


now = datetime.now()
strtime = now.strftime('%Y%m%d_%H%M')


class DeepESN_skl(BaseEstimator, ClassifierMixin):

    def __init__(self, configs={}, error_function=None, design_deep=False, eta=0.01, min_layers=1, max_layers=10, readout_model=None, **kwargs):
        '''
        Wrapper of DeepESN for sklearn support.
        @param configs: Configurations dict (same as object but duct, use it for sklearn compatibility)
        @param deepESN_: Original model from DeepESN
        '''
        self.configs = configs
        self.error_function = error_function
        self.eta = eta
        self.max_layers = max_layers
        self.min_layers = min_layers
        self.design_deep = design_deep
        self.trained_ = False
        self.kwargs = dict(kwargs)
        self.y_true_ = []
        self._deepESN = None

        self._readout = None
        if readout_model:
            self._readout = readout_model

        # Compatibility with estimator sklean model
        if kwargs:
            self.configs.update(dict(kwargs))

        self._update_model_configs()

    def get_params(self, deep=True):
        ''' return list of parameters accepted by the model's class '''
        params = {"configs": self.configs, "design_deep": self.design_deep, "error_function": self.error_function,
                  "eta": self.eta, "max_layers": self.max_layers, "min_layers": self.min_layers}
        params.update(self.configs)
        params.update(self.kwargs)
        return params

    def set_params(self, **parameters):
        ''' function called by GridSearch to set new parameters '''
        for key in dict(parameters):
            if key in ["eta", "max_layers", "min_layers", "design_deep"]:
                setattr(self, key, parameters[key])
        self.configs.update(dict(parameters))
        self._update_model_configs()
        return self

    def _update_model_configs(self):
        ''' Reupdate model on paramters changes '''
        self._configs = Struct(self.configs)

    def generate_esn(self):
        self._deepESN = DeepESN(self._configs)

    def fit(self, X, y):
        # Create deepESN
        if not self._deepESN:
            if self.design_deep:
                Nl = self.designDeep(
                    X, min_layers=self.min_layers, max_layers=self.max_layers, eta=self.eta, verbose=1)
                self.configs['Nl'] = Nl
                if hasattr(self, 'Nl'):
                    setattr(self, 'Nl', Nl)
                params_to_save = self.get_params()
                save_params_status(params_to_save)
                self._update_model_configs()
                print("|DD| Optimal Nl={}".format(Nl))
                self._deepESN = DeepESN(self._configs)
            else:
                self._deepESN = DeepESN(self._configs)

        states = self._deepESN.computeState(X, self._deepESN.IPconf.DeepIP)
        indexes = np.arange(len(X))
        train_states = select_indexes(states, indexes, self._configs.transient)
        train_targets = np.asarray(select_indexes(
            y, indexes, self._configs.transient))

        if self._configs.rebalance_states:
            train_states, train_targets = self._deepESN.rebalanceStates(
                train_states, train_targets)

        self._deepESN.trainReadout(
            train_states, train_targets, self._configs.reg)

        if self._readout is not None:
            x = train_states
            # x = np.reshape(self._deepESN.computeOutput(train_states), (-1, 1, train_targets.shape[-1]))
            # x = train_states
            print("train_states SHAPE, states", np.array(
                x).shape, np.array(train_targets).shape)
            # y = np.concatenate(train_targets, 1)
            # x_train = []
            # y_train = []
            # for i in range(0, x.shape[-1], self._readout.in_features):
            #     x_train.append(x[0][i:i + self._readout.in_features])
            #     y_train.append(y[0][i:i + self._readout.in_features])
            # x_train = np.array(x_train)
            # y_train = np.array(y_train)
            # print(x_train.shape, y_train.shape, x.shape, y.shape)
            self._readout.fit(np.expand_dims(x, 1), np.expand_dims(
                train_targets, 1), epochs=100)
        self.trained_ = True
        return self

    def remove_transient(self, data):
        indexes = np.arange(len(data))
        targets = np.asarray(select_indexes(
            data, indexes, self._configs.transient))
        return targets

    def predict(self, X, verbose=False):
        input = X
        indexes = np.arange(len(input))
        # Check is fit had been called, if train_==True

        start = time.time()
        states = self._deepESN.computeState(input, self._deepESN.IPconf.DeepIP)
        if verbose:
            print("STATE OUT {} in sec {}".format(
                np.array(states).shape, time.time() - start))
        start = time.time()
        test_states = select_indexes(states, indexes, self._configs.transient)
        if verbose:
            print("STATE OUT AFTER TRANSIENT {} in sec {}".format(
                np.array(test_states).shape, time.time() - start))
        start = time.time()
        out = self._deepESN.computeOutput(test_states)
        if verbose:
            print("LINEAR OUT {} in sec {}".format(
                out.shape, time.time() - start))

        if verbose:
            print("OUT SHAPE {}".format(out.shape))

        return out

    def online_predict(self, X, states=None, verbose=False):
        input = X
        indexes = np.arange(len(input))
        # Check is fit had been called, if train_==True
        start = time.time()
        states = self._deepESN.onlineComputeState(input, _states=states, DeepIP=self._deepESN.IPconf.DeepIP)
        print(np.array(states).shape)
        if verbose:
            print("STATE OUT {} in sec {}".format(
                np.array(states).shape, time.time() - start))
        start = time.time()
        test_states = np.array(select_indexes(states, indexes, self._configs.transient))
        if verbose:
            print("STATE OUT AFTER TRANSIENT {} in sec {}".format(
                np.array(test_states).shape, time.time() - start))
        start = time.time()
        out = self._deepESN.computeOutput(test_states[:,:,-len(input):])
        if verbose:
            print("LINEAR OUT {} in sec {}".format(
                out.shape, time.time() - start))

        if verbose:
            print("OUT SHAPE {}".format(out.shape))

        return out, states

    def score(self, X, y):
        ''' Used to evaluate the score of the model prediction'''
        indexes = np.arange(len(X))
        y_targets = np.asarray(select_indexes(
            y, indexes, self._configs.transient))
        preds = self.predict(X)
        score = self.error_function(preds, y_targets)
        return score

    # -------------------------------------------------------- #
    # ------------------    DESIGN DEEP     ------------------ #
    # -------------------------------------------------------- #

    def designDeep(self, X, max_layers=10, eta=0.01, min_layers=1, plot=False, verbose=0):
        ''' Procedure Design of DeepESN described in "Design of deep echo state networks - C Gallicchio, A Micheli, L Pedrelli"

        Parameters:
        theta (list): model's parameters
        computeStateFn (function: theta,l,X -> o):  function that generate and return the new model based on given parameters and data
        eta (float): 0 < eta < 1
        max_layer (int): network's layers upperbound
        plot: save plot of mu,sigma on increasing layers

        Returns:
        (int): optimal number of layers
        '''
        mu_ls = []
        sigma_ls = []
        threshold_hist = []

        config = deepcopy(Struct(self.configs))
        optimal_l = 0

        for l in range(min_layers, max_layers + 1):
            if verbose:
                print("|DD| Trying with Nl={}...".format(l))
            config.Nl = l
            deepESN = DeepESN(config)
            states = deepESN.computeState(X, deepESN.IPconf.DeepIP)
            indexes = np.arange(len(X))
            selected_states = select_indexes(
                states, indexes, self._configs.transient)

            p, f = self.FFT(selected_states)
            p_sum = np.sum(p, dtype=np.float)

            mu_l = np.sum(np.prod([p, f], axis=0, dtype=np.float)) / p_sum

            sigma_l = np.sqrt(np.sum(np.prod(
                [p, (np.subtract(f, mu_l, dtype=np.float))**2], axis=0, dtype=np.float)) / p_sum)

            mu_ls += [mu_l]
            sigma_ls += [sigma_l]

            if l > min_layers + 1:
                threshold_hist += [np.absolute(mu_ls[-1] -
                                               mu_ls[-2]) - sigma_ls[-2] * eta]
                if threshold_hist[-1] <= 0:
                    # equal to --> np.absolute(mu_ls[-1] - mu_ls[-2]) <= sigma_ls[-2] * eta
                    # balanced states obtained, end of the loop
                    optimal_l = l
                    break

        if optimal_l == 0:
            optimal_l = max_layers

        if verbose:
            print('|DD| MU', mu_ls)
            print('|DD| SIGMA', sigma_ls)
            print('|DD| THRESHOLDS: {}, eta:{}'.format(threshold_hist, eta))

        if plot:
            import matplotlib.pyplot as plt
            plt.plot(range(start_from, l + 1), mu_ls, 'r--', label="mu")
            plt.plot(range(start_from, l), sigma_ls[:-1], 'b', label="sigma")
            plt.plot(range(start_from, l), threshold_hist,
                     'g', label="thresholds hist")
            plt.title("mu e sigma al variare dei layers")
            plt.rc('grid', linestyle="-", color='grey')
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.show()

        return l

    def FFT(self, X):
        ''' FFT layer state signal described in "Design of deep echo state networks - C Gallicchio, A Micheli, L Pedrelli"

        Parameters:
        X (numpy.array): Computed state of the network

        Returns:
        (float, float): normalized frequencies, magnitudes of the components
        '''
        comps_g = []
        _timesteps = 0
        timesteps = None

        for guess in range(0, len(X)):  # numero di sample nell'input
            comps_u = []

            # sample output per ogni unità della rete
            for unit in range(0, len(X[guess])):
                signal = X[guess][unit]
                # timestamp calcolato una sola volta poichè tutti i sample hanno lunghezza uguale
                if not bool(_timesteps):
                    _timesteps = len(signal)
                timesteps = _timesteps
                comps = np.fft.fft(signal)[1:]
                comps_u += [np.absolute(comps[:int(timesteps / 2)]).tolist()]

            comps_g += [np.mean(comps_u, axis=0)]

        p = np.mean(comps_g, axis=0)
        f = np.array(list(range(0, int(timesteps / 2))),
                     dtype=np.float) / timesteps
        return (p, f)

    def save_model(self, path):
        self._deepESN.saveModel(path)

    def restore_model(self, path):
        if not self._deepESN:
            self._deepESN = DeepESN(self._configs)
        self._deepESN.restoreModel(path)


def save_params_status(params, path="./log/gs_params_tested"):
    path = path + '_' + strtime + '.pkl'
    with (open(path, 'wb')) as f:
        pickle.dump(dict(params), f)


def select_indexes(data, indexes, transient=0):
    # da decommentare con dataset MG altrimenti commenta
    # if len(data) == 1:
    #     return [data[0][:, indexes][:, transient:]]

    return [data[i][:, transient:] for i in indexes]

class Struct(object):
    def __init__(self, d=None):
        if d:
            # print("PRENORM", d)
            d = self._normalize_dict(d)
            # print("AFTERNORM", d)
            for a, b in d.items():
                if isinstance(b, (list, tuple)):
                    setattr(self, a, [Struct(x) if isinstance(x, dict) else x for x in b])
                else:
                    setattr(self, a, Struct(b) if isinstance(b, dict) else b)

    def to_dict(self, obj=None):
        if obj is None:
            obj = self
        if not hasattr(obj, "__dict__"):
            return obj
        result = {}
        for key, val in obj.__dict__.items():
            if key.startswith("_"):
                continue
            element = []
            if isinstance(val, list):
                for item in val:
                    element.append(self.to_dict(item))
            else:
                element = self.to_dict(val)
            result[key] = element
        return result

    def _normalize_dict(self, obj):
        newobj = {}
        for k in obj.keys():
            if '.' in k:
                steps = k.split('.')
                val = {}
                tmp = val
                for i in range(len(steps)):
                    tmp[steps[i]] = {}
                    if i < len(steps)-1:
                        tmp = tmp[steps[i]]
                tmp[steps[-1]] = obj[k]
                newobj = self._update_attr(newobj, val)

        obj = self._update_attr(obj, newobj)
        return obj

    def _update_attr(self, obj, newobj):
        if type(newobj) is not dict:
            return newobj
        
        for k in newobj.keys():
            if k in obj:
                obj[k] = self._update_attr(obj[k], newobj[k])
            else:
                obj[k] = newobj[k]
        return obj
# def predict(self, X):
#     # Check is fit had been called, if train_==True
#     preds = []
#     self.y_true_ = []
#     for input in X:
#         input = np.expand_dims(input, axis=0)
#         indexes = np.arange(len(input))

#         states = self._deepESN.computeState(input, self._deepESN.IPconf.DeepIP)
#         test_states = select_indexes(states, indexes, self._configs.transient)
#         out = self._deepESN.computeOutput(test_states)
#         preds.append(out)

#     preds = np.array(preds)
#     return preds
