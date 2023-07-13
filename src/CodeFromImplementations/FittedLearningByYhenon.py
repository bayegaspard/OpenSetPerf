#code from https://github.com/yhenon/fitted-learning/blob/master/fitted_learning.py
#From paper: https://arxiv.org/pdf/1609.02226
import numpy as np

def build_label(class_idx, n_classes, DOO):
    # returns the target for a training instance
    label = np.zeros((n_classes * DOO, ))
    for ii in range(DOO):
        label[ii * n_classes + class_idx] = 1.0 / DOO
    return label


def infer(probs, DOO, n_classes):
    # infer from a test instance
    out = np.ones((n_classes,))
    for ii in range(DOO):
        for jj in range(n_classes):
            out[jj] = out[jj] * probs[jj + ii * n_classes] * DOO
    return out

