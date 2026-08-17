# =========================
# Q2：多元多项式回归与岭回归
# =========================
"""
模型：

    y = β0
        + β1 X1
        + β2 X2
        + β3 X3
        + β4 X3^2
        + β5 X1X2
        + β6 X1X3

其中：
    X1 = Co绝对质量 / 乙醇流速
    X2 = 乙醇流速 × 装料总质量
    X3 = 温度

数据划分：
    110条数据中随机抽取10条作为测试集
    其余100条作为训练集

模型：
    1. 普通多元多项式回归
    2. 岭回归

岭回归：
    在训练集内部进行5折交叉验证，
    从候选lambda中选择验证误差最小的lambda。

注意：
    测试集只用于最终模型评价，
    不参与训练和lambda选择。
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error


# 特征设置
BASE_FEATURES = [
    "X1_Co质量比流速",
    "X2_流速x总质量",
    "X3_温度"
]

FEATURE_NAMES = [
    "X1",
    "X2",
    "X3",
    "X3^2",
    "X1X2",
    "X1X3"
]


# 构造多项式特征
def build_features(df):
    """
    根据X1、X2、X3构造多元多项式回归所需特征。
    最终特征：
        X1
        X2
        X3
        X3^2
        X1X2
        X1X3
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


# 数据划分
def split_train_test(df, test_size=10, random_state=2021):
    """
    随机抽取test_size条数据作为测试集，
    其余数据作为训练集。
    """

    if len(df) <= test_size:
        raise ValueError("数据量不足，无法划分训练集和测试集。")

    rng = np.random.default_rng(random_state)

    test_indices = rng.choice(
        len(df),
        size=test_size,
        replace=False
    )

    test_indices = np.sort(test_indices)

    test_mask = np.zeros(len(df), dtype=bool)
    test_mask[test_indices] = True

    train_df = df.loc[~test_mask].copy()
    test_df = df.loc[test_mask].copy()

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df


# 普通多元多项式回归
def fit_polynomial_regression(X_train, y_train):
    """
    普通最小二乘多元回归。
    """

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


# 岭回归交叉验证
def ridge_cross_validation(
    X_train,
    y_train,
    n_splits=5,
    random_state=2021
):
    """
    在训练集内部使用5折交叉验证选择最优lambda。

    岭回归前进行标准化：
        不同特征量纲差异较大，
        标准化可以避免正则化受到量纲影响。

    候选lambda采用对数尺度。
    """

    # 候选lambda
    alphas = np.logspace(-6, 6, 49)

    kfold = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    cv_results = []

    for alpha in alphas:

        fold_errors = []

        for train_idx, valid_idx in kfold.split(X_train):

            X_tr = X_train[train_idx]
            X_val = X_train[valid_idx]

            y_tr = y_train[train_idx]
            y_val = y_train[valid_idx]

            model = Pipeline([
                ("scaler", StandardScaler()),
                (
                    "ridge",
                    Ridge(
                        alpha=alpha,
                        fit_intercept=True
                    )
                )
            ])

            model.fit(X_tr, y_tr)

            y_pred = model.predict(X_val)

            mse = mean_squared_error(
                y_val,
                y_pred
            )

            fold_errors.append(mse)

        mean_mse = np.mean(fold_errors)

        cv_results.append({
            "lambda": alpha,
            "CV_MSE": mean_mse,
            "CV_RMSE": np.sqrt(mean_mse)
        })

    cv_df = pd.DataFrame(cv_results)

    best_index = cv_df["CV_MSE"].idxmin()

    best_lambda = cv_df.loc[
        best_index,
        "lambda"
    ]

    return best_lambda, cv_df


# 拟合岭回归
def fit_ridge_regression(
    X_train,
    y_train,
    best_lambda
):
    """
    使用交叉验证得到的最优lambda，
    在完整训练集上重新拟合岭回归。
    """

    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "ridge",
            Ridge(
                alpha=best_lambda,
                fit_intercept=True
            )
        )
    ])

    model.fit(X_train, y_train)

    return model


# 模型评价
def evaluate_model(model, X, y):
    """
    计算：
        SSE
        MSE
        RMSE
        R²
    """

    y_pred = model.predict(X)

    residual = y - y_pred

    sse = np.sum(residual ** 2)

    mse = np.mean(residual ** 2)

    rmse = np.sqrt(mse)

    r2 = r2_score(y, y_pred)

    return {
        "SSE": sse,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }, y_pred


# 提取模型系数
def get_polynomial_coefficients(model):
    """
    获取普通多元多项式回归系数。
    """

    result = {
        "截距": model.intercept_
    }

    for name, coef in zip(
        FEATURE_NAMES,
        model.coef_
    ):
        result[name] = coef

    return result


def get_ridge_coefficients(model):
    """
    获取岭回归系数。

    由于岭回归前进行了标准化，
    这里保存的是标准化特征空间中的系数。
    """

    ridge_model = model.named_steps["ridge"]

    result = {
        "截距": ridge_model.intercept_
    }

    for name, coef in zip(
        FEATURE_NAMES,
        ridge_model.coef_
    ):
        result[name] = coef

    return result


# =========================
# 单个因变量完整建模
# =========================

def run_single_target(
    train_df,
    test_df,
    y_col,
    random_state=2021
):
    """
    对一个因变量完成：

        1. 构造训练/测试特征
        2. 普通多元多项式回归
        3. 岭回归CV
        4. 岭回归最终拟合
        5. 训练集评价
        6. 测试集评价
        7. 保存预测结果
    """

    # -------------------------
    # 构造特征
    # -------------------------

    X_train = build_features(train_df)
    X_test = build_features(test_df)

    y_train = train_df[y_col].to_numpy(dtype=float)
    y_test = test_df[y_col].to_numpy(dtype=float)

    # -------------------------
    # 普通多项式回归
    # -------------------------

    poly_model = fit_polynomial_regression(
        X_train,
        y_train
    )

    poly_train_metrics, poly_train_pred = evaluate_model(
        poly_model,
        X_train,
        y_train
    )

    poly_test_metrics, poly_test_pred = evaluate_model(
        poly_model,
        X_test,
        y_test
    )

    # -------------------------
    # 岭回归
    # -------------------------

    best_lambda, cv_df = ridge_cross_validation(
        X_train,
        y_train,
        n_splits=5,
        random_state=random_state
    )

    ridge_model = fit_ridge_regression(
        X_train,
        y_train,
        best_lambda
    )

    ridge_train_metrics, ridge_train_pred = evaluate_model(
        ridge_model,
        X_train,
        y_train
    )

    ridge_test_metrics, ridge_test_pred = evaluate_model(
        ridge_model,
        X_test,
        y_test
    )

    # -------------------------
    # 模型比较
    # -------------------------

    comparison = pd.DataFrame([
        {
            "因变量": y_col,
            "模型": "多元多项式回归",
            "lambda": np.nan,
            "训练集SSE": poly_train_metrics["SSE"],
            "训练集RMSE": poly_train_metrics["RMSE"],
            "训练集R2": poly_train_metrics["R2"],
            "测试集SSE": poly_test_metrics["SSE"],
            "测试集RMSE": poly_test_metrics["RMSE"],
            "测试集R2": poly_test_metrics["R2"]
        },
        {
            "因变量": y_col,
            "模型": "岭回归",
            "lambda": best_lambda,
            "训练集SSE": ridge_train_metrics["SSE"],
            "训练集RMSE": ridge_train_metrics["RMSE"],
            "训练集R2": ridge_train_metrics["R2"],
            "测试集SSE": ridge_test_metrics["SSE"],
            "测试集RMSE": ridge_test_metrics["RMSE"],
            "测试集R2": ridge_test_metrics["R2"]
        }
    ])

    # -------------------------
    # 测试集预测结果
    # -------------------------

    prediction_df = test_df[
        ["催化剂组合编号", "温度"]
    ].copy()

    prediction_df["实际值"] = y_test

    prediction_df["多项式回归预测值"] = poly_test_pred

    prediction_df["多项式回归残差"] = (
        y_test - poly_test_pred
    )

    prediction_df["岭回归预测值"] = ridge_test_pred

    prediction_df["岭回归残差"] = (
        y_test - ridge_test_pred
    )

    # -------------------------
    # 模型系数
    # -------------------------

    poly_coef = get_polynomial_coefficients(
        poly_model
    )

    ridge_coef = get_ridge_coefficients(
        ridge_model
    )

    coef_df = pd.DataFrame({
        "变量": list(poly_coef.keys()),
        "多元多项式回归系数": list(poly_coef.values()),
        "岭回归系数": [
            ridge_coef.get(name, np.nan)
            for name in poly_coef.keys()
        ]
    })

    return {
        "comparison": comparison,
        "prediction": prediction_df,
        "cv": cv_df,
        "coefficients": coef_df,
        "best_lambda": best_lambda
    }


# =========================
# 两个因变量统一建模
# =========================

def run_all_models(
    df,
    y_cols,
    test_size=10,
    random_state=2021
):
    """
    对两个因变量统一进行建模。

    注意：
        两个因变量使用完全相同的训练集和测试集，
        保证模型比较公平。
    """

    # -------------------------
    # 划分训练集和测试集
    # -------------------------

    train_df, test_df = split_train_test(
        df,
        test_size=test_size,
        random_state=random_state
    )

    all_results = {}

    for y_col in y_cols:

        result = run_single_target(
            train_df,
            test_df,
            y_col,
            random_state=random_state
        )

        all_results[y_col] = result

    return (
        train_df,
        test_df,
        all_results
    )