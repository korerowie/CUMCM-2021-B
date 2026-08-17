# =========================
# Q2：最终模型结果可视化
# =========================

import os

import numpy as np

import matplotlib.pyplot as plt


plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS"
]

plt.rcParams["axes.unicode_minus"] = False


# =========================
# 构造模型特征
# =========================

def build_features_from_values(
    x1,
    x2,
    x3
):
    """
    根据X1、X2、X3构造：
        X1
        X2
        X3
        X3²
        X1X2
        X1X3
    """

    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    x3 = np.asarray(x3)

    return np.column_stack([
        x1,
        x2,
        x3,
        x3 ** 2,
        x1 * x2,
        x1 * x3
    ])


# =========================
# 单因素影响曲线
# =========================

def plot_one_factor(
    df,
    model,
    y_col,
    factor,
    save_path
):
    """
    固定另外两个变量为中位数，
    只改变当前因素，
    绘制模型预测值变化曲线。
    """

    x1_median = df[
        "X1_Co质量比流速"
    ].median()

    x2_median = df[
        "X2_流速x总质量"
    ].median()

    x3_median = df[
        "X3_温度"
    ].median()

    # -------------------------
    # 生成变化范围
    # -------------------------

    if factor == "X1":

        values = np.linspace(
            df[
                "X1_Co质量比流速"
            ].min(),
            df[
                "X1_Co质量比流速"
            ].max(),
            200
        )

        x1 = values

        x2 = np.full_like(
            values,
            x2_median
        )

        x3 = np.full_like(
            values,
            x3_median
        )

        xlabel = "X1：Co绝对质量 / 乙醇流速"

    elif factor == "X2":

        values = np.linspace(
            df[
                "X2_流速x总质量"
            ].min(),
            df[
                "X2_流速x总质量"
            ].max(),
            200
        )

        x1 = np.full_like(
            values,
            x1_median
        )

        x2 = values

        x3 = np.full_like(
            values,
            x3_median
        )

        xlabel = "X2：乙醇流速 × 装料总质量"

    elif factor == "X3":

        values = np.linspace(
            df[
                "X3_温度"
            ].min(),
            df[
                "X3_温度"
            ].max(),
            200
        )

        x1 = np.full_like(
            values,
            x1_median
        )

        x2 = np.full_like(
            values,
            x2_median
        )

        x3 = values

        xlabel = "X3：温度 / ℃"

    else:

        raise ValueError(
            "factor必须为X1、X2或X3"
        )

    # -------------------------
    # 构造特征
    # -------------------------

    X = build_features_from_values(
        x1,
        x2,
        x3
    )

    # -------------------------
    # 模型预测
    # -------------------------

    y_pred = model.predict(
        X
    )

    # -------------------------
    # 绘图
    # -------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        values,
        y_pred,
        linewidth=2
    )

    plt.xlabel(
        xlabel,
        fontsize=12
    )

    plt.ylabel(
        y_col,
        fontsize=12
    )

    plt.title(
        f"{factor}对{y_col}的影响",
        fontsize=14
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =========================
# 三因素影响图
# =========================

def plot_factor_effects(
    df,
    model,
    y_col,
    figure_dir
):
    """
    分别绘制：
        X1影响
        X2影响
        X3影响
    """

    name = (
        y_col
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )

    factors = [
        "X1",
        "X2",
        "X3"
    ]

    for factor in factors:

        save_path = os.path.join(
            figure_dir,
            f"{name}_{factor}影响.png"
        )

        plot_one_factor(
            df,
            model,
            y_col,
            factor,
            save_path
        )

        print(
            f"[完成] {save_path}"
        )


# =========================
# 实际值-预测值散点图
# =========================

def plot_actual_vs_predicted(
    prediction_df,
    y_col,
    figure_dir
):
    """
    绘制实际值与模型预测值之间的关系。
    """

    actual = prediction_df[
        "实际值"
    ].to_numpy()

    predicted = prediction_df[
        "预测值"
    ].to_numpy()

    min_value = min(
        actual.min(),
        predicted.min()
    )

    max_value = max(
        actual.max(),
        predicted.max()
    )

    margin = (
        max_value - min_value
    ) * 0.05

    line_min = (
        min_value - margin
    )

    line_max = (
        max_value + margin
    )

    # -------------------------
    # 绘图
    # -------------------------

    plt.figure(
        figsize=(7, 6)
    )

    plt.scatter(
        actual,
        predicted,
        s=45
    )

    plt.plot(
        [line_min, line_max],
        [line_min, line_max],
        linestyle="--",
        linewidth=1.5
    )

    plt.xlabel(
        "实际值",
        fontsize=12
    )

    plt.ylabel(
        "预测值",
        fontsize=12
    )

    plt.title(
        f"{y_col}：实际值与预测值",
        fontsize=14
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    name = (
        y_col
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )

    save_path = os.path.join(
        figure_dir,
        f"{name}_实际值预测值.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[完成] {save_path}"
    )