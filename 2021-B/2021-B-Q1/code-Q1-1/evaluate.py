# =========================
# Q1-1：拟合优度评价：R2 与 RMSE
# =========================

import numpy as np


def r_squared(y_true, y_pred):
    if np.any(np.isnan(y_pred)):
        return np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot


def rmse(y_true, y_pred):
    if np.any(np.isnan(y_pred)):
        return np.nan
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def evaluate(y_true, y_pred):
    return {"R2": r_squared(y_true, y_pred), "RMSE": rmse(y_true, y_pred)}
