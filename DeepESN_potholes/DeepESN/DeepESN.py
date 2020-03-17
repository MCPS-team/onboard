import numpy as np
import numpy.matlib
import scipy as sc
import random
import sys
from collections import Counter
import h5py
from concurrent.futures import ProcessPoolExecutor as PoolExecutor
from time import time


class DeepESN():

    '''
    Deep Echo State Network (DeepESN) class:
    this class implement the DeepESN model suitable for
    time-serie prediction and sequence classification.

    Reference paper for DeepESN model:
    C. Gallicchio, A. Micheli, L. Pedrelli, "Deep Reservoir Computing: A
    Critical Experimental Analysis", Neurocomputing, 2017, vol. 268, pp. 87-99

    Reference paper for the design of DeepESN model in multivariate time-series prediction tasks:
    C. Gallicchio, A. Micheli, L. Pedrelli, "Design of deep echo state networks",
    Neural Networks, 2018, vol. 108, pp. 33-47

    ----

    This file is a part of the DeepESN Python Library (DeepESNpy)

    Luca Pedrelli
    luca.pedrelli@di.unipi.it
    lucapedrelli@gmail.com

    Department of Computer Science - University of Pisa (Italy)
    Computational Intelligence & Machine Learning (CIML) Group
    http://www.di.unipi.it/groups/ciml/

    ----
    '''

    def __init__(self, configs):
        # initialize the DeepESN model
        sys.stdout.write('init DeepESN...')
        sys.stdout.flush()

        # spectral radius (maximum absolute eigenvalue)
        rhos = np.array(configs.rhos)
        lis = np.array(configs.lis)  # leaky rate
        iss = np.array(configs.iss)  # input scale
        IPconf = configs.IPconf  # configuration for Deep Intrinsic Plasticity
        reservoirConf = configs.reservoirConf  # reservoir configurations

        if len(rhos.shape) == 0:
            rhos = np.matlib.repmat(rhos, 1, configs.Nl)[0]

        if len(lis.shape) == 0:
            lis = np.matlib.repmat(lis, 1, configs.Nl)[0]

        if len(iss.shape) == 0:
            iss = np.matlib.repmat(iss, 1, configs.Nl)[0]

        self.W = {}  # recurrent weights
        self.Win = {}  # recurrent weights
        self.Gain = {}  # activation function gain
        self.Bias = {}  # activation function bias

        self.Nu = configs.Nu  # number of inputs
        self.Nr = configs.Nr  # number of units per layer
        self.Nl = configs.Nl  # number of layers
        self.rhos = rhos.tolist()  # list of spectral radius
        self.lis = lis  # list of leaky rate
        self.iss = iss  # list of input scale

        self.IPconf = IPconf

        self.readout = configs.readout

        # sparse recurrent weights init
        if reservoirConf.connectivity < 1:
            for layer in range(self.Nl):
                self.W[layer] = np.zeros((self.Nr, self.Nr))
                for row in range(self.Nr):
                    number_row_elements = round(
                        reservoirConf.connectivity * self.Nr)
                    row_elements = random.sample(
                        range(self.Nr), number_row_elements)
                    self.W[layer][row, row_elements] = np.random.uniform(
                        -1, +1, size=(1, number_row_elements))

        # full-connected recurrent weights init
        else:
            for layer in range(self.Nl):
                self.W[layer] = np.random.uniform(-1, +1,
                                                  size=(self.Nr, self.Nr))

        # layers init
        for layer in range(self.Nl):

            target_li = lis[layer]
            target_rho = rhos[layer]
            input_scale = iss[layer]

            if layer == 0:
                self.Win[layer] = np.random.uniform(
                    -input_scale, input_scale, size=(self.Nr, self.Nu + 1))
            else:
                self.Win[layer] = np.random.uniform(
                    -input_scale, input_scale, size=(self.Nr, self.Nr + 1))

            Ws = (1 - target_li) * np.eye(self.W[layer].shape[0],
                                          self.W[layer].shape[1]) + target_li * self.W[layer]
            eig_value, eig_vector = np.linalg.eig(Ws)
            actual_rho = np.max(np.absolute(eig_value))

            Ws = (Ws * target_rho) / actual_rho
            self.W[layer] = (target_li**-1) * (Ws - (1. - target_li)
                                               * np.eye(self.W[layer].shape[0], self.W[layer].shape[1]))

            self.Gain[layer] = np.ones((self.Nr, 1))
            self.Bias[layer] = np.zeros((self.Nr, 1))

        print('done.')
        sys.stdout.flush()

    def onlineComputeLayerState(self, input, layer, _state=None, is_first=False, DeepIP=0):
        if _state is not None:
            state = _state
        else:
            state = np.zeros((self.Nr, len(input[0])))

        input = self.Win[layer][:, 0:-1].dot(input[:,-1:]) + np.expand_dims(self.Win[layer][:, -1], 1)

        # IS FIRST
        if _state is None:
            if DeepIP:
                state_net = np.zeros((self.Nr, len(input[0])))
                state_net[:, 0:1] = input
                state[:, 0:1] = self.lis[layer] * \
                    np.tanh(np.multiply(
                        self.Gain[layer], state_net[:, 0:1]) + self.Bias[layer])
            else:
                state[:, 0:1] = self.lis[layer] * \
                    np.tanh(np.multiply(
                        self.Gain[layer], input) + self.Bias[layer])
            return state

        t = state.shape[-1] -1
        if DeepIP:
            state_net[:, t:t +
                      1] = self.W[layer].dot(state[:, t - 1:t]) + input
            state[:, t:t + 1] = (1 - self.lis[layer]) * state[:, t - 1:t] + self.lis[layer] * np.tanh(
                np.multiply(self.Gain[layer], state_net[:, t:t + 1]) + self.Bias[layer])

            eta = self.IPconf.eta
            mu = self.IPconf.mu
            sigma2 = self.IPconf.sigma**2

            # IP learning rule
            deltaBias = -eta * ((-mu / sigma2) + np.multiply(state[:, t:t + 1], (2 * sigma2 + 1 - (
                state[:, t:t + 1]**2) + mu * state[:, t:t + 1]) / sigma2))
            deltaGain = eta / np.matlib.repmat(
                self.Gain[layer], 1, state_net[:, t:t + 1].shape[1]) + deltaBias * state_net[:, t:t + 1]

            # update gain and bias of activation function
            self.Gain[layer] = self.Gain[layer] + deltaGain
            self.Bias[layer] = self.Bias[layer] + deltaBias

        else:
            state[:, t:t + 1] = (1 - self.lis[layer]) * state[:, t - 1:t] + self.lis[layer] * np.tanh(np.multiply(
                self.Gain[layer], self.W[layer].dot(state[:, t - 1:t]) + input) + self.Bias[layer])
        
        return state

    def computeLayerState(self, input, layer, DeepIP=0):
        # compute the state of a layer with pre-training if DeepIP == 1

        state = np.zeros((self.Nr, len(input[0])))

        input = self.Win[layer][:, 0:-
                                1].dot(input) + np.expand_dims(self.Win[layer][:, -1], 1)
        if DeepIP:
            state_net = np.zeros((self.Nr, len(input[0])))
            state_net[:, 0:1] = input[:, 0:1]
            state[:, 0:1] = self.lis[layer] * \
                np.tanh(np.multiply(
                    self.Gain[layer], state_net[:, 0:1]) + self.Bias[layer])
        else:
            state[:, 0:1] = self.lis[layer] * \
                np.tanh(np.multiply(
                    self.Gain[layer], input[:, 0:1]) + self.Bias[layer])

        for t in range(1, state.shape[1]):
            if DeepIP:
                state_net[:, t:t +
                          1] = self.W[layer].dot(state[:, t - 1:t]) + input[:, t:t + 1]
                state[:, t:t + 1] = (1 - self.lis[layer]) * state[:, t - 1:t] + self.lis[layer] * np.tanh(
                    np.multiply(self.Gain[layer], state_net[:, t:t + 1]) + self.Bias[layer])

                eta = self.IPconf.eta
                mu = self.IPconf.mu
                sigma2 = self.IPconf.sigma**2

                # IP learning rule
                deltaBias = -eta * ((-mu / sigma2) + np.multiply(state[:, t:t + 1], (2 * sigma2 + 1 - (
                    state[:, t:t + 1]**2) + mu * state[:, t:t + 1]) / sigma2))
                deltaGain = eta / np.matlib.repmat(
                    self.Gain[layer], 1, state_net[:, t:t + 1].shape[1]) + deltaBias * state_net[:, t:t + 1]

                # update gain and bias of activation function
                self.Gain[layer] = self.Gain[layer] + deltaGain
                self.Bias[layer] = self.Bias[layer] + deltaBias

            else:
                # print(input[:, t:t + 1].shape) (30, 1)
                state[:, t:t + 1] = (1 - self.lis[layer]) * state[:, t - 1:t] + self.lis[layer] * np.tanh(np.multiply(
                    self.Gain[layer], self.W[layer].dot(state[:, t - 1:t]) + input[:, t:t + 1]) + self.Bias[layer])
        return state

    def computeDeepIntrinsicPlasticity(self, inputs):
        # we incrementally perform the pre-training (deep intrinsic plasticity) over layers

        len_inputs = range(len(inputs))
        states = []

        for i in len_inputs:
            states.append(np.zeros((self.Nr * self.Nl, inputs[i].shape[1])))

        for layer in range(self.Nl):

            for epoch in range(self.IPconf.Nepochs):
                Gain_epoch = self.Gain[layer]
                Bias_epoch = self.Bias[layer]

                if len(inputs) == 1:
                    self.computeLayerState(
                        inputs[0][:, self.IPconf.indexes], layer, 1)
                else:
                    for i in self.IPconf.indexes:
                        self.computeLayerState(inputs[i], layer, 1)

                if (np.linalg.norm(self.Gain[layer] - Gain_epoch, 2) < self.IPconf.threshold) and (np.linalg.norm(self.Bias[layer] - Bias_epoch, 2) < self.IPconf.threshold):
                    sys.stdout.write(str(epoch + 1))
                    sys.stdout.write('.')
                    sys.stdout.flush()
                    break

                if epoch + 1 == self.IPconf.Nepochs:
                    sys.stdout.write(str(epoch + 1))
                    sys.stdout.write('.')
                    sys.stdout.flush()

            inputs2 = []
            for i in range(len(inputs)):
                inputs2.append(self.computeLayerState(inputs[i], layer))

            for i in range(len(inputs)):
                states[i][(layer) * self.Nr: (layer + 1)
                          * self.Nr, :] = inputs2[i]

            inputs = inputs2

        return states

    def computeState(self, inputs, DeepIP=0):
        # compute the global state of DeepESN with pre-training if DeepIP == 1

        if self.IPconf.DeepIP and DeepIP:
            # sys.stdout.write('compute state with DeepIP...')
            # sys.stdout.flush()
            states = self.computeDeepIntrinsicPlasticity(inputs)
        else:
            # sys.stdout.write('compute state...')
            # sys.stdout.flush()
            states = []
            for inp in inputs:
                states.append(self.computeGlobalState(inp))
            # with PoolExecutor(max_workers=None) as executor:
            #     results = executor.map(
            #         self.computeGlobalState, inputs, chunksize=5)
            # states = list(results)
        # print('done.')
        # sys.stdout.flush()
        return states

    def onlineComputeState(self, input, _states=None, DeepIP=0):
        # compute the global state of DeepESN with pre-training if DeepIP == 1
        if _states is None:
            states = []
        else:
            states = _states
        
        states = [self.onlineComputeGlobalState(input[0], _state= states if _states else _states)]
        return states

    def computeGlobalState(self, input):
        # compute the global state of DeepESN

        state = np.zeros((self.Nl * self.Nr, len(input[0])))
        
        for layer in range(self.Nl):
            state[(layer) * self.Nr: (layer + 1) * self.Nr,
                  :] = self.computeLayerState(input, layer)
            input = state[(layer) * self.Nr: (layer + 1) * self.Nr, :]
        return state


    def onlineComputeGlobalState(self, input, _state=None):
        # compute the global state of DeepESN
        if _state:
            state = np.concatenate(
                [_state[0], np.zeros((self.Nl * self.Nr, len(input[0])))], axis=1)
        else:
            state = np.zeros((self.Nl * self.Nr, len(input[0])))
            

        for layer in range(self.Nl):
            layer_state = state[layer * self.Nr: (layer+1) * self.Nr] if _state else None
            state[(layer) * self.Nr: (layer + 1) * self.Nr,:] = self.onlineComputeLayerState(input, layer, _state=layer_state)
            input = state[(layer) * self.Nr: (layer + 1) * self.Nr, :]
        return state

    def rebalanceStates(self, states, targets):
        # rebalance the states in order to rebalance target classes and
        # improve scores that requires high accuracy in all classes

        new_targets = targets.tolist()
        for ind, tar in enumerate(targets):
            # Find the positive class / negative class target rateo for this event
            classes = Counter(tar[0])
            num_rep = int(classes[-1] / classes[1]) - 1

            # Find the positive states
            pos_states = states[ind][:, np.where(tar[0] == 1)]

            # Reshape the states matrix from (50, 1, 300) to (50, 300)
            pos_states = pos_states.reshape(pos_states.shape[0:3:2])

            # Repeat the positive states to make them equal in number to the negative states
            pos_states = np.tile(pos_states, num_rep)

            # Append them to the states structure
            states[ind] = np.concatenate((states[ind], pos_states), axis=1)
            new_targets[ind][0] = np.concatenate(
                (targets[ind][0], np.ones(num_rep * classes[1])))

        return np.asarray(states), np.asarray(new_targets)

    def trainReadout(self, trainStates, trainTargets, lb):
        # train the readout of DeepESN

        trainStates = np.concatenate(trainStates, 1)
        trainTargets = np.concatenate(trainTargets, 1)

        # add bias
        X = np.ones((trainStates.shape[0] + 1, trainStates.shape[1]))
        X[:-1, :] = trainStates
        trainStates = X

        sys.stdout.write('train readout...')
        sys.stdout.flush()

        if self.readout.trainMethod == 'SVD':  # SVD, accurate method
            U, s, V = np.linalg.svd(trainStates, full_matrices=False)
            s = s / (s**2 + lb)

            self.Wout = trainTargets.dot(np.multiply(
                V.T, np.expand_dims(s, 0)).dot(U.T))

        else:  # NormalEquation, fast method
            B = trainTargets.dot(trainStates.T)
            A = trainStates.dot(trainStates.T)

            self.Wout = np.linalg.solve(
                (A + np.eye(A.shape[0], A.shape[1]) * lb), B.T).T

        print('done.')
        sys.stdout.flush()

    def computeOutput(self, state):
        # compute a linear combination between the global state and the output weights
        # print(state.shape)
        state = np.concatenate(state, 1)
        # Wout product + add bias
        return self.Wout[:, 0:-1].dot(state) + np.expand_dims(self.Wout[:, -1], 1)

    def saveModel(self, save_path):
        ''' save Wout to h5 file '''
        h5f = h5py.File(save_path, 'w')
        for layer in range(self.Nl):
            h5f.create_dataset('W_'+str(layer), data=self.W[layer])
            h5f.create_dataset('Win_'+str(layer), data=self.Win[layer])
            h5f.create_dataset('Gain_'+str(layer), data=self.Gain[layer])
            h5f.create_dataset('Bias_'+str(layer), data=self.Bias[layer])
        h5f.create_dataset('Wout', data=self.Wout)
        h5f.close()
        print('ESN saved in {}'.format(save_path))

    def restoreModel(self, save_path):
        ''' restore Wout from h5 file '''
        h5f = h5py.File(save_path, 'r')
        for layer in range(self.Nl):
            self.W[layer] = h5f['W_'+str(layer)][:]
            self.Win[layer] = h5f['Win_'+str(layer)][:]
            self.Gain[layer] = h5f['Gain_'+str(layer)][:]
            self.Bias[layer] = h5f['Bias_'+str(layer)][:]
        self.Wout = h5f['Wout'][:]
        h5f.close()
        print('ESN restored from {}'.format(save_path))
