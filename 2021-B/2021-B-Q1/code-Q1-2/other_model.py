# =========================
# Q1-2：其余指标候选模型定义
# =========================
"""
M1: y = a*T + b                       线性
M2: y = a*T^3 + b*T^2 + c*T + d       三次多项式
M3: y = a*exp(b*T) + c                指数
M4: y = k / (1 + exp(-r*(T-t0))) + c  S型(Logistic)模型
M5: y = c                             平稳(常数)模型
"""

import numpy as np
from scipy.optimize import curve_fit

CUBIC_POWER = 3


def linear_model(T, a, b):
    return a * T + b


def cubic_model(T, a, b, c, d):
    return a * np.power(T, CUBIC_POWER) + b * np.power(T, 2) + c * T + d


def exp_model(T, a, b, c):
    return a * np.exp(b * T) + c


def fit_linear(T, y):
    a, b = np.polyfit(T, y, 1)
    y_pred = linear_model(T, a, b)
    return {"a": a, "b": b}, y_pred


def fit_cubic(T, y):
    a, b, c, d = np.polyfit(T, y, CUBIC_POWER)
    y_pred = cubic_model(T, a, b, c, d)
    return {"a": a, "b": b, "c": c, "d": d}, y_pred


def fit_exp(T, y):
    init_guesses = [
        (1.0, 0.005, y.min()),
        (1.0, 0.01, y.min()),
        (0.1, 0.02, y.min()),
        (10.0, -0.005, y.max()),
    ]
    bounds = ([-1e4, -0.2, -1e4], [1e4, 0.2, 1e4])

    best_params, best_pred, best_sse = None, None, np.inf
    for p0 in init_guesses:
        try:
            popt, _ = curve_fit(exp_model, T, y, p0=p0, bounds=bounds, maxfev=20000)
            y_pred = exp_model(T, *popt)
            sse = np.sum((y - y_pred) ** 2)
            if sse < best_sse:
                best_sse = sse
                best_params = popt
                best_pred = y_pred
        except RuntimeError:
            continue

    if best_params is None:
        return {"a": np.nan, "b": np.nan, "c": np.nan}, np.full_like(y, np.nan)

    a, b, c = best_params
    return {"a": a, "b": b, "c": c}, best_pred


def constant_model(t, c):
    # 平稳(常数)模型，拟合值就是均值，用来体现随时间基本稳定、无明显趋势
    return np.full_like(np.asarray(t, dtype=float), c)


def fit_constant(t, y):
    c = float(np.mean(y))
    y_pred = np.full_like(y, c)
    return {"c": c}, y_pred


def cubic_model(t, a, b, c, d):
    return a * t**3 + b * t**2 + c * t + d


def fit_cubic(t, y):
    # 三次多项式同样是线性最小二乘问题，直接用polyfit求闭式解，不需要初值
    a, b, c, d = np.polyfit(t, y, 3)
    y_pred = cubic_model(t, a, b, c, d)
    return {"a": a, "b": b, "c": c, "d": d}, y_pred


def logistic_model(T, k, r, t0, c):
    return k / (1 + np.exp(-r * (T - t0))) + c


def fit_logistic(T, y):
    # S型(Logistic)模型，专门给C4烯烃选择性里呈S形变化的催化剂单独拟合用。
    # 4个参数在5~7个点上很敏感，尤其r和t0，多组初值分别尝试，取残差平方和最小的一组。
    t_range = T.max() - T.min()
    y_range = y.max() - y.min()
    if t_range == 0:
        t_range = 1.0

    init_guesses = [
        (y_range, 4 / t_range, np.median(T), y.min()),
        (y_range, 8 / t_range, np.median(T), y.min()),
        (y_range * 1.2, 2 / t_range, T.mean(), y.min()),
        (y_range, 6 / t_range, T[len(T) // 2], y.min()),
    ]
    bounds = (
        [-1e4, -5, T.min() - t_range, -1e4],
        [1e4, 5, T.max() + t_range, 1e4],
    )

    best_params, best_pred, best_sse = None, None, np.inf
    for p0 in init_guesses:
        try:
            popt, _ = curve_fit(logistic_model, T, y, p0=p0, bounds=bounds, maxfev=20000)
            y_pred = logistic_model(T, *popt)
            sse = np.sum((y - y_pred) ** 2)
            if sse < best_sse:
                best_sse = sse
                best_params = popt
                best_pred = y_pred
        except RuntimeError:
            continue

    if best_params is None:
        return {"k": np.nan, "r": np.nan, "t0": np.nan, "c": np.nan}, np.full_like(y, np.nan)

    k, r, t0, c = best_params
    return {"k": k, "r": r, "t0": t0, "c": c}, best_pred


# 模型名 -> 拟合函数，模型名 -> 计算函数，两处保持同样的key方便main.py统一取用
MODEL_FIT_REGISTRY = {
    "M1_线性": fit_linear,
    "M2_三次": fit_cubic,
    "M3_指数": fit_exp,
    "M4_S型_Logistic": fit_logistic,
    "M5_平稳(常数)": fit_constant,
}

MODEL_FUNC_REGISTRY = {
    "M1_线性": linear_model,
    "M2_三次": cubic_model,
    "M3_指数": exp_model,
    "M4_S型_Logistic": logistic_model,
    "M5_平稳(常数)": constant_model,
}