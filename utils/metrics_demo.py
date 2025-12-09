import numpy as np


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


def smape(y_true, y_pred, eps=1e-5):
    return float(100.0 * np.mean((2.0 * np.abs(y_true - y_pred) + eps) / (np.abs(y_true) + np.abs(y_pred) + eps)))


def metric(y_true, y_pred):
    """
    Return a tuple to mimic your original printing style:
    (mae, mse, rmse, smape)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return mae(y_true, y_pred), mse(y_true, y_pred), rmse(y_true, y_pred), smape(y_true, y_pred)
