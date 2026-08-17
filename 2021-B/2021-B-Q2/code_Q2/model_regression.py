# =========================
# Q2：最终岭回归模型
# =========================
"""
使用附件1全部实验数据建立最终岭回归模型。

模型特征：
    X1
    X2
    X3
    X3^2
    X1X2
    X1X3

其中：
    X1 = Co绝对质量 / 乙醇流速
    X2 = 乙醇流速 × 装料总质量
    X3 = 温度

建模流程：
    1. 全部数据进行5折交叉验证
    2. 确定最优lambda
    3. 使用全部数据重新拟合岭回归
    4. 输出最终模型系数
    5. 输出全部数据的预测结果
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    r2_score
)


# =========================
# 特征名称
# =========================

FEATURE_NAMES = [
    "X1",
    "X2",
    "X3",
    "X3^2",
    "X1X2",
    "X1X3"
]


# =========================
# 构造特征
# =========================

def build_features(df):
    """
    根据data_process.py得到的X1、X2、X3
    构造最终模型所需的6个特征。
    """

    X1 = df["X1_Co质量比流速"].to_numpy(dtype=float)
    X2 = df["X2_流速x总质量"].to_numpy(dtype=float)
    X3 = df["X3_温度"].to_numpy(dtype=float)

    X = np.column_stack([
        X1,
        X2,
        X3,
        X3 ** 2,
        X1 * X2,
        X1 * X3
    ])

    return X


# =========================
# 岭回归5折交叉验证
# =========================

def ridge_cross_validation(
    X,
    y,
    n_splits=5,
    random_state=2021
):
    """
    使用全部数据进行5折交叉验证，
    确定最终岭回归lambda。

    每一折中：
        训练数据 → 标准化 → 岭回归
        验证数据 → 使用训练折的标准化参数 → 预测

    测试集已经不再参与，
    因为现在进入最终模型阶段。
    """

    alphas = np.logspace(-6, 6, 49)

    kfold = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    results = []

    for alpha in alphas:

        fold_mse = []

        for train_idx, valid_idx in kfold.split(X):

            X_train = X[train_idx]
            X_valid = X[valid_idx]

            y_train = y[train_idx]
            y_valid = y[valid_idx]

            model = Pipeline([
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "ridge",
                    Ridge(
                        alpha=alpha,
                        fit_intercept=True
                    )
                )
            ])

            model.fit(
                X_train,
                y_train
            )

            y_pred = model.predict(
                X_valid
            )

            mse = mean_squared_error(
                y_valid,
                y_pred
            )

            fold_mse.append(mse)

        mean_mse = np.mean(
            fold_mse
        )

        results.append({
            "lambda": alpha,
            "CV_MSE": mean_mse,
            "CV_RMSE": np.sqrt(mean_mse)
        })

    cv_df = pd.DataFrame(results)

    best_index = cv_df["CV_MSE"].idxmin()

    best_lambda = cv_df.loc[
        best_index,
        "lambda"
    ]

    return best_lambda, cv_df


# =========================
# 最终岭回归
# =========================

def fit_final_ridge(
    X,
    y,
    best_lambda
):
    """
    使用全部数据拟合最终岭回归模型。
    """

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "ridge",
            Ridge(
                alpha=best_lambda,
                fit_intercept=True
            )
        )
    ])

    model.fit(
        X,
        y
    )

    return model


# =========================
# 计算原始变量尺度下的系数
# =========================

def get_original_coefficients(model):
    """
    岭回归内部进行了标准化。

    将标准化变量空间中的系数
    转换回原始变量尺度。

    最终得到：

        y = β0
            + β1 X1
            + β2 X2
            + β3 X3
            + β4 X3^2
            + β5 X1X2
            + β6 X1X3
    """

    scaler = model.named_steps["scaler"]

    ridge = model.named_steps["ridge"]

    coef_standardized = ridge.coef_

    mean = scaler.mean_

    scale = scaler.scale_

    # 转换到原始变量尺度
    coef_original = (
        coef_standardized / scale
    )

    intercept_original = (
        ridge.intercept_
        - np.sum(
            coef_standardized
            * mean
            / scale
        )
    )

    result = {
        "截距": intercept_original
    }

    for name, coef in zip(
        FEATURE_NAMES,
        coef_original
    ):
        result[name] = coef

    return result


# =========================
# 最终模型评价
# =========================

def evaluate_final_model(
    model,
    X,
    y
):
    """
    使用全部数据评价最终模型。
    """

    y_pred = model.predict(X)

    residual = y - y_pred

    sse = np.sum(
        residual ** 2
    )

    mse = np.mean(
        residual ** 2
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y,
        y_pred
    )

    return {
        "SSE": sse,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }, y_pred


# =========================
# 单个因变量建立最终模型
# =========================

def build_final_model(
    df,
    y_col,
    random_state=2021
):
    """
    对一个因变量建立最终岭回归模型。
    """

    # -------------------------
    # 构造特征
    # -------------------------

    X = build_features(df)

    y = df[
        y_col
    ].to_numpy(
        dtype=float
    )

    # -------------------------
    # 5折CV确定lambda
    # -------------------------

    best_lambda, cv_df = (
        ridge_cross_validation(
            X,
            y,
            n_splits=5,
            random_state=random_state
        )
    )

    # -------------------------
    # 全部数据拟合
    # -------------------------

    model = fit_final_ridge(
        X,
        y,
        best_lambda
    )

    # -------------------------
    # 模型评价
    # -------------------------

    metrics, y_pred = (
        evaluate_final_model(
            model,
            X,
            y
        )
    )

    # -------------------------
    # 原始变量尺度系数
    # -------------------------

    coefficients = (
        get_original_coefficients(
            model
        )
    )

    coefficient_df = pd.DataFrame({
        "变量": list(
            coefficients.keys()
        ),
        "回归系数": list(
            coefficients.values()
        )
    })

    # -------------------------
    # 全部数据预测结果
    # -------------------------

    prediction_df = df[
        [
            "催化剂组合编号",
            "温度"
        ]
    ].copy()

    prediction_df[
        "实际值"
    ] = y

    prediction_df[
        "预测值"
    ] = y_pred

    prediction_df[
        "残差"
    ] = y - y_pred

    return {
        "model": model,
        "best_lambda": best_lambda,
        "cv": cv_df,
        "coefficients": coefficient_df,
        "predictions": prediction_df,
        "metrics": metrics
    }


# =========================
# 两个因变量统一建立最终模型
# =========================

def build_all_final_models(
    df,
    y_cols,
    random_state=2021
):
    """
    分别建立：
        1. 乙醇转化率最终模型
        2. C4烯烃选择性最终模型
    """

    results = {}

    for y_col in y_cols:

        results[y_col] = (
            build_final_model(
                df,
                y_col,
                random_state
            )
        )

    return results