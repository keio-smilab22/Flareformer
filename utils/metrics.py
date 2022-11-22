import math

import numpy as np
from sklearn import metrics


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
        np.sum((true - true.mean()) ** 2)
    )


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def calc_score(y_pred, y_true, climatology):
    """
    Compute ACC, TSS, BSS, and GMGS
    """
    score = {}
    y_predl = []
    for y in y_pred:
        y_predl.append(np.argmax(y))

    score["TSS-M"] = TSS(y_predl, y_true, 2)
    score["BSS-M"] = BSS(y_pred, y_true, climatology)
    score["GMGS"] = GMGS(y_predl, y_true)

    return score


def TSS(y_predl, y_true, flare_class):
    """
    Compute TSS
    """

    tn, fp, fn, tp = calc_tp_4(
        metrics.confusion_matrix(y_true, y_predl, labels=[0, 1, 2, 3]), flare_class
    )
    tss = (tp / (tp + fn)) - (fp / (fp + tn))
    if math.isnan(tss):
        return 0
    return float(tss)


def GMGS(y_predl, y_true):
    """
    Compute GMGS (simplified version)
    """
    tss_x = TSS(y_predl, y_true, 3)
    tss_m = TSS(y_predl, y_true, 2)
    tss_c = TSS(y_predl, y_true, 1)
    print("x: ", tss_x)
    print("m: ", tss_m)
    print("c: ", tss_c)
    return (tss_x + tss_m + tss_c) / 3


def BSS(y_pred, y_true, climatology):
    """
    Compute BSS >= M
    """
    y_truel = []
    for y in y_true:
        y_truel.append(convert_2_one_hot_2class(y))

    y_pred2 = np.reshape(np.array(y_pred).ravel(), (-1, 2))
    y_pred2 = np.reshape(np.sum(y_pred2, axis=1), (-1, 2))
    bs = 0
    bsc = 0
    for p, t in zip(y_pred2.tolist(), np.array(y_truel).tolist()):
        bs += (p[1] - t[1]) ** 2
    bs = bs / len(y_true)
    bsc = climatology[0] * climatology[1]
    bss = (bsc - bs) / bsc

    return bss


def convert_2_one_hot_2class(binary_value):
    """
    return 2-dimentional 1-of-K vector
    """
    if binary_value == 3 or binary_value == 2:
        return [0, 1]
    return [1, 0]


def calc_cm4(y_pred, y_true):
    """
    return confusion matrix for 4 class
    """
    return metrics.confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])


def calc_acc4(y_pred, y_true):
    """
    Compute classification accuracy for 4 class
    """
    cm = calc_cm4(y_pred, y_true)
    print(cm)
    acc4 = (cm[0, 0] + cm[1, 1] + cm[2, 2] + cm[3, 3]) / len(y_pred)
    return acc4


def calc_tp_4(four_class_matrix, flare_class):
    """
    Convert 4 class output to 2 class
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for true in range(4):
        for pred in range(4):
            if true >= flare_class:
                if pred >= flare_class:
                    tp += four_class_matrix[true][pred]
                else:
                    fn += four_class_matrix[true][pred]
            else:
                if pred >= flare_class:
                    fp += four_class_matrix[true][pred]
                else:
                    tn += four_class_matrix[true][pred]
    return tn, fp, fn, tp


def regression_to_class(pred: np.ndarray) -> np.ndarray:
    """
    Convert regression output to class
    """
    pred_class = np.zeros((pred.shape[0], 4))
    print(pred_class.shape)
    for i in range(len(pred)):
        if pred[i, 0] >= 2:
            pred_class[i, 3] = 1
        elif pred[i, 0] >= 1 and pred[i, 0] < 2:
            pred_class[i, 2] = 1
        elif pred[i, 0] >= 0 and pred[i, 0] < 1:
            pred_class[i, 1] = 1
        else:
            pred_class[i, 0] = 1
    return pred_class


def metric(pred, true):

    # from 24 hours to 48 hours after
    pred_48 = pred[:, 24:, :]
    true_48 = true[:, 24:, :]

    mae = MAE(pred_48, true_48)
    mse = MSE(pred_48, true_48)
    rmse = RMSE(pred_48, true_48)
    mape = MAPE(pred_48, true_48)
    mspe = MSPE(pred_48, true_48)

    pred_max = np.max(pred_48, axis=1)
    true_max = np.max(true_48, axis=1)

    # classification
    pred_class = regression_to_class(pred_max)
    true_class = regression_to_class(true_max)

    pred_class_l = []
    for y in pred_class:
        pred_class_l.append(np.argmax(y))
    pred_class_l = np.array(pred_class_l)

    true_class_l = []
    for y in true_class:
        true_class_l.append(np.argmax(y))
    true_class_l = np.array(true_class_l)

    tss_m = TSS(pred_class_l, true_class_l, 2)
    bss_m = BSS(pred_class, true_class_l, [0.9053, 0.0947])
    gmgs = GMGS(pred_class_l, true_class_l)

    # up to 24 hours later
    pred_24 = pred[:, :24, :]
    true_24 = true[:, :24, :]

    mae_24 = MAE(pred_24, true_24)
    mse_24 = MSE(pred_24, true_24)
    rmse_24 = RMSE(pred_24, true_24)
    mape_24 = MAPE(pred_24, true_24)
    mspe_24 = MSPE(pred_24, true_24)

    pred_max_24 = np.max(pred_24, axis=1)
    true_max_24 = np.max(true_24, axis=1)

    pred_class_24 = regression_to_class(pred_max_24)
    true_class_24 = regression_to_class(true_max_24)

    pred_class_l_24 = []
    for y in pred_class_24:
        pred_class_l_24.append(np.argmax(y))
    pred_class_l_24 = np.array(pred_class_l_24)

    true_class_l_24 = []
    for y in true_class_24:
        true_class_l_24.append(np.argmax(y))
    true_class_l_24 = np.array(true_class_l_24)

    tss_m_24 = TSS(pred_class_l_24, true_class_l_24, 2)
    bss_m_24 = BSS(pred_class_24, true_class_l_24, [0.9053, 0.0947])
    gmgs_24 = GMGS(pred_class_l_24, true_class_l_24)

    confusion_matrix = calc_cm4(pred_class_l, true_class_l)
    print(f'48h confusion matrix\n{confusion_matrix[0]}\n{confusion_matrix[1]}\n{confusion_matrix[2]}\n{confusion_matrix[3]}\n')
    confusion_matrix_24 = calc_cm4(pred_class_l_24, true_class_l_24)
    print(f'24h confusion matrix\n{confusion_matrix_24[0]}\n{confusion_matrix_24[1]}\n{confusion_matrix_24[2]}\n{confusion_matrix_24[3]}\n')

    return (
        mae,
        mse,
        rmse,
        mape,
        mspe,
        tss_m,
        bss_m,
        gmgs,
        mae_24,
        mse_24,
        rmse_24,
        mape_24,
        mspe_24,
        tss_m_24,
        bss_m_24,
        gmgs_24,
    )

if __name__ == "__main__":
    cm = \
    [[1979, 419,   21,    2],
     [ 335, 1554,  453,   11],
     [  23,  379,  495,  43],
     [   0,   19,   82,   29]]

    # for i in range(len(pred)):
    #     if pred[i, 0] >= 2:
    #         pred_class[i, 3] = 1
    #     elif pred[i, 0] >= 1 and pred[i, 0] < 2:
    #         pred_class[i, 2] = 1
    #     elif pred[i, 0] >= 0 and pred[i, 0] < 1:
    #         pred_class[i, 1] = 1
    #     else:
    #         pred_class[i, 0] = 1

    # confusion matrix to prediction and true label
    pred = []
    true = []
    for i in range(4):
        for j in range(4):
            for k in range(cm[i][j]):
                pred.append(j-1)
                true.append(i-1)

    pred = np.array(pred)
    true = np.array(true)
    pred = pred.reshape(-1, 1)
    true = true.reshape(-1, 1)
    # [N, 1] -> [N, 24, 1]
    pred = np.repeat(pred, 24, axis=1)
    true = np.repeat(true, 24, axis=1)
    pred = pred.reshape(-1, 24, 1)
    true = true.reshape(-1, 24, 1)
    print(pred[0])

    pred_max = np.max(pred, axis=1)
    true_max = np.max(true, axis=1)

    # classification
    pred_class = regression_to_class(pred_max)
    true_class = regression_to_class(true_max)

    pred_class_l = []
    for y in pred_class:
        pred_class_l.append(np.argmax(y))
    pred_class_l = np.array(pred_class_l)

    true_class_l = []
    for y in true_class:
        true_class_l.append(np.argmax(y))
    true_class_l = np.array(true_class_l)

    tss_m = TSS(pred_class_l, true_class_l, 2)
    bss_m = BSS(pred_class, true_class_l, [0.9053, 0.0947])
    gmgs = GMGS(pred_class_l, true_class_l)

    print(gmgs)