'''
Metrics functions

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
from sklearn.metrics import f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd

# Accuracy function used to evaluate the prediction in polyphonic music tasks: true positive/(true positive + false positive + false negative)
def computeMusicAccuracy(threshold, Y_output, Y_target):
    Y_target = np.concatenate(Y_target, 1)
    Nsys = np.sum(Y_output > threshold, axis=0)
    Nref = np.sum(Y_target > threshold, axis=0)
    Ncorr = np.sum((Y_output > threshold) * (Y_target > threshold), axis=0)

    TP = np.sum(Ncorr)
    FP = np.sum(Nsys - Ncorr)
    FN = np.sum(Nref - Ncorr)
    ACCURACY = TP / float(TP + FP + FN)
    return ACCURACY


# Mean Squared Error
def MSE(threshold, Y_output, Y_target):
    Y_target = np.concatenate(Y_target, 1)
    return np.mean((Y_output - Y_target)**2)


def F1_score(threshold, Y_output, Y_target):
    Y_target = np.concatenate(Y_target, 1)
    return f1_score(Y_target[0] > threshold, Y_output[0] > threshold, average='macro')


def ACC_eq(Y_output, Y_target, threshold=0, start_offset=100):
    out_offset = 0
    scores = []
    for y_target in Y_target:
        y_target = np.squeeze(y_target)
        # estrai da serie concatenata interval corrispondente a target
        y_out = Y_output[0, out_offset: out_offset + y_target.shape[-1]]
        offset_eq = max(np.where(y_target == 1)[0][0] - start_offset, 1)
        # corretti negativi prima di offset_eq / offset_eq
        pre_score = len(np.where(y_out[:offset_eq] <= threshold)[0]) / offset_eq
        # f1 score su quelli restanti
        post_score = f1_score(y_out[offset_eq:] > threshold, y_target[offset_eq:] > threshold, average='weighted')
        scores.append((pre_score**2) * post_score)
        # percentuale di corretti negativi su tratto prima della classificazione - start_offset * f1_score sulla parte rimanente
        out_offset = out_offset + y_target.shape[-1]
    return np.mean(scores)


def find_roc_threshold(y_true, y_pred, plot=False):
    y_pred = y_pred.squeeze()
    y_true = y_true.squeeze()

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : %f" % roc_auc)

    ####################################
    # The optimal cut off would be where tpr is high and fpr is low
    # tpr - (1-fpr) is zero or near to zero is the optimal cut off point
    ####################################
    i = np.arange(len(tpr))  # index for df
    roc = pd.DataFrame({'fpr': pd.Series(fpr, index=i), 'tpr': pd.Series(tpr, index=i), '1-fpr': pd.Series(
        1-fpr, index=i), 'tf': pd.Series(tpr - (1-fpr), index=i), 'thresholds': pd.Series(thresholds, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    print(roc_t)
    threshold = list(roc_t['thresholds'])[0]

    if plot:
        # Plot tpr vs 1-fpr
        fig, ax = plt.subplots()
        plt.plot(roc['tpr'])
        plt.plot(roc['1-fpr'], color='red')
        plt.xlabel('1-False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        ax.set_xticklabels([])
        plt.show()

    return threshold