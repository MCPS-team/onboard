'''
Model configuration file

Reference paper for DeepESN model:
C. Gallicchio, A. Micheli, L. Pedrelli, "Deep Reservoir Computing: A Critical Experimental Analysis",
Neurocomputing, 2017, vol. 268, pp. 87-99

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

import numpy as np


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




def config_pianomidi(IP_indexes, Nu):

    configs = Struct()

    configs.rhos = 0.1  # set spectral radius 0.1 for all recurrent layers
    configs.lis = 0.7  # set li 0.7 for all recurrent layers
    configs.iss = 0.1  # set input scale 0.1 for all recurrent layers

    # Be careful with memory usage
    configs.Nu = Nu
    configs.Nr = 100  # number of recurrent units
    configs.Nl = 5  # number of recurrent layers
    configs.reg = 10.0**-2
    configs.transient = 5

    configs.IPconf = Struct()
    configs.IPconf.DeepIP = 0  # activate pre-train
    configs.IPconf.threshold = 0.1  # threshold for gradient descent in pre-train algorithm
    configs.IPconf.eta = 10**-5  # learning rate for IP rule
    configs.IPconf.mu = 0  # mean of target gaussian function
    configs.IPconf.sigma = 0.1  # std of target gaussian function
    configs.IPconf.Nepochs = 10  # maximum number of epochs
    configs.IPconf.indexes = IP_indexes  # perform the pre-train on these indexes

    configs.reservoirConf = Struct()
    configs.reservoirConf.connectivity = 1  # connectivity of recurrent matrix

    configs.readout = Struct()
    configs.readout.trainMethod = 'NormalEquations'  # train with normal equations (faster)
    configs.readout.regularizations = 10.0**np.array(range(-4, -1, 1))

    configs.rebalance_states = False  # rebalance network states

    return configs


def config_MG(IP_indexes, Nu):

    configs = Struct()

    configs.rhos = 0.9  # set spectral radius 0.9 for all recurrent layers
    configs.lis = 1.0  # set li 1.0 for all recurrent layers
    configs.iss = 0.1  # set input scale 0.1 for all recurrent layers

    # Be careful with memory usage
    configs.Nu = Nu
    configs.Nr = 100  # number of recurrent units
    configs.Nl = 5  # number of recurrent layers
    configs.reg = 0.0
    configs.transient = 100

    configs.IPconf = Struct()
    configs.IPconf.DeepIP = 0  # deactivate pre-train

    configs.reservoirConf = Struct()
    configs.reservoirConf.connectivity = 1  # connectivity of recurrent matrix

    configs.readout = Struct()
    configs.readout.trainMethod = 'SVD'  # train with singular value decomposition (more accurate)
    configs.readout.regularizations = 10.0**np.array(range(-16, -1, 1))

    configs.rebalance_states = False  # rebalance network states

    return configs


def config_PH(IP_indexes, Nu):

    configs = Struct()

    configs.rhos = 0.9  # set spectral radius 0.9 for all recurrent layers
    configs.lis = 1.0  # set li 1.0 for all recurrent layers
    configs.iss = 0.1  # set input scale 0.1 for all recurrent layers

    # Be careful with memory usage
    configs.Nu = Nu
    configs.Nr = 100  # number of recurrent units
    configs.Nl = 5  # number of recurrent layers
    configs.reg = 1e-7
    configs.transient = 0

    configs.IPconf = Struct()
    configs.IPconf.DeepIP = 0  # deactivate pre-train
    configs.IPconf.threshold = 0.1  # threshold for gradient descent in pre-train algorithm
    configs.IPconf.eta = 10**-5  # learning rate for IP rule
    configs.IPconf.mu = 0  # mean of target gaussian function
    configs.IPconf.sigma = 0.1  # std of target gaussian function
    configs.IPconf.Nepochs = 10  # maximum number of epochs
    configs.IPconf.indexes = IP_indexes  # perform the pre-train on these indexes

    configs.reservoirConf = Struct()
    configs.reservoirConf.connectivity = 1  # connectivity of recurrent matrix

    configs.readout = Struct()
    configs.readout.trainMethod = 'NormalEquations'  # train with singular value decomposition (more accurate)
    configs.readout.regularizations = 10.0**np.array(range(-16, -16, 1))

    configs.rebalance_states = False  # rebalance network states

    return configs
    

def best_config_PH(IP_indexes, Nu):
    configs = Struct()

    configs.rhos = 0.7  # set spectral radius 0.9 for all recurrent layers
    configs.lis = 1.0  # set li 1.0 for all recurrent layers
    configs.iss = 0.8  # set input scale 0.1 for all recurrent layers

    # Be careful with memory usage
    configs.Nu = Nu
    configs.Nr = 30  # number of recurrent units
    configs.Nl = 4  # number of recurrent layers
    configs.reg = 1e-7
    configs.transient = 0

    configs.IPconf = Struct()
    configs.IPconf.DeepIP = 0  # deactivate pre-train
    configs.IPconf.threshold = 0.1  # threshold for gradient descent in pre-train algorithm
    configs.IPconf.eta = 10**-5  # learning rate for IP rule
    configs.IPconf.mu = 0  # mean of target gaussian function
    configs.IPconf.sigma = 0.1  # std of target gaussian function
    configs.IPconf.Nepochs = 10  # maximum number of epochs
    configs.IPconf.indexes = IP_indexes  # perform the pre-train on these indexes

    configs.reservoirConf = Struct()
    configs.reservoirConf.connectivity = 1  # connectivity of recurrent matrix

    configs.readout = Struct()
    configs.readout.trainMethod = 'NormalEquations'  # train with singular value decomposition (more accurate)
    configs.readout.regularizations = 10.0**np.array(range(-16, -16, 1))

    configs.rebalance_states = True  # rebalance network states

    return configs
    

