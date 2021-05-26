import os
import numpy as np


def iou_oa(y_true, y_pred, n_class):
    iou = []

    # calculate iou per class
    tp = 0
    for c in range(n_class):
        TP = np.sum((y_true == c) & (y_pred == c))
        FP = np.sum((y_true != c) & (y_pred == c))
        FN = np.sum((y_true == c) & (y_pred != c))

        n = TP
        d = float(TP + FP + FN + 1e-12)

        iou.append(np.divide(n, d))
        tp += TP

    oa =  TP / len(y_true)

    return (np.mean(iou), oa)


if __name__ == "__main__":

    with open('y_true', 'r') as f:
        y_true = f.read()
    # assume result file
    with open('y_pred', 'r') as f:
        y_pred = f.read()

    iou, oa = iou_oa(y_true, y_pred, 4)

    