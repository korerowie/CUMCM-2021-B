# =========================
# Q1-1：候选模型的定义与拟合
# =========================
"""
M1: y = a*T + b                           线性
M2: y = a*T^2 + b*T + c                   二次多项式
M3: y = a*exp(b*T) + c                    指数
特殊: y = k / (1 + exp(-r*(T-t0))) + c     S型(Logistic)模型

注：M2 的指数按最常见的二次(p=2)处理，
    三个模型按"线性/二次/指数"依次递进。
    若确认不是二次，改下面的 QUAD_POWER 常量即可，不用动其余代码。
    用S型模型替换B6/A13/A3的拟合结果。
"""
import numpy as np
from scipy.optimize import curve_fit

QUAD_POWER = 2


def linear_model(T, a, b):
    return a * T + b


def quad_model(T, a, b, c):
    return a * np.power(T, QUAD_POWER) + b * T + c


def exp_model(T, a, b, c):
    return a * np.exp(b * T) + c


def fit_linear(T, y):
    # 线性最小二乘有闭式解，直接用polyfit，稳定不需要初值
    a, b = np.polyfit(T, y, 1)
    y_pred = linear_model(T, a, b)
    return {"a": a, "b": b}, y_pred


def fit_quad(T, y):
    # 二次多项式同样是线性最小二乘问题，用polyfit求解后转成a,b,c三个系数
    a, b, c = np.polyfit(T, y, QUAD_POWER)
    y_pred = quad_model(T, a, b, c)
    return {"a": a, "b": b, "c": c}, y_pred


def fit_exp(T, y):
    # 指数模型没有闭式解，用curve_fit，温度跨度较大时容易不收敛或发散
    # 这里取几组不同的初值分别尝试，取残差平方和最小的一组作为最终结果
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
        # 极少数情况下指数模型完全无法收敛，返回NaN，后续步骤会自动把它从候选中剔除
        return {"a": np.nan, "b": np.nan, "c": np.nan}, np.full_like(y, np.nan)

    a, b, c = best_params
    return {"a": a, "b": b, "c": c}, best_pred


def logistic_model(T, k, r, t0, c):
    return k / (1 + np.exp(-r * (T - t0))) + c
 
 
def fit_logistic(T, y):
    # S型(Logistic)模型，专门给C4烯烃选择性里呈S形变化的催化剂单独拟合用
    # 4个参数在5~7个点上很敏感，尤其r和t0，多组初值分别尝试，取残差平方和最小的一组
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
    "M2_二次": fit_quad,
    "M3_指数": fit_exp,
}

MODEL_FUNC_REGISTRY = {
    "M1_线性": linear_model,
    "M2_二次": quad_model,
    "M3_指数": exp_model,
}
