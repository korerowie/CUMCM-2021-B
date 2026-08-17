# =========================
# Q2：最终岭回归模型
# =========================

import os

import pandas as pd

from data_process import load_data

from model_regression import (
    build_all_final_models
)

from visualization_q2 import (
    plot_factor_effects,
    plot_actual_vs_predicted
)


# =========================
# 路径设置
# =========================

DATA_PATH = "../2021-B/附件1.xlsx"

TABLE_DIR = "tables"

FIGURE_DIR = "figures"


# =========================
# 因变量
# =========================

Y_COLS = [
    "乙醇转化率(%)",
    "C4烯烃选择性(%)"
]


# =========================
# 参数
# =========================

RANDOM_STATE = 2021


# =========================
# 文件名处理
# =========================

def safe_name(name):

    return (
        name
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


# =========================
# 主程序
# =========================

def main():

    # -------------------------
    # 创建文件夹
    # -------------------------

    os.makedirs(
        TABLE_DIR,
        exist_ok=True
    )

    os.makedirs(
        FIGURE_DIR,
        exist_ok=True
    )

    # -------------------------
    # 读取数据
    # -------------------------

    print("=" * 60)
    print("Q2：最终岭回归模型")
    print("=" * 60)

    df = load_data(
        DATA_PATH
    )

    print(
        f"\n实验数据数量：{len(df)}"
    )

    # -------------------------
    # 建立最终模型
    # -------------------------

    results = (
        build_all_final_models(
            df,
            Y_COLS,
            RANDOM_STATE
        )
    )

    # -------------------------
    # 输出两个最终模型
    # -------------------------

    for y_col in Y_COLS:

        result = results[
            y_col
        ]

        name = safe_name(
            y_col
        )

        print("\n" + "=" * 60)

        print(
            f"因变量：{y_col}"
        )

        print("=" * 60)

        # -------------------------
        # lambda
        # -------------------------

        print(
            f"\n最终 lambda："
            f"{result['best_lambda']:.6g}"
        )

        # -------------------------
        # 模型评价
        # -------------------------

        metrics = result[
            "metrics"
        ]

        print("\n全部数据拟合结果：")

        print(
            f"SSE  = {metrics['SSE']:.6f}"
        )

        print(
            f"RMSE = {metrics['RMSE']:.6f}"
        )

        print(
            f"R²   = {metrics['R2']:.6f}"
        )

        # -------------------------
        # 输出模型方程系数
        # -------------------------

        print("\n最终模型系数：")

        print(
            result[
                "coefficients"
            ].to_string(
                index=False,
                float_format=lambda x:
                f"{x:.10g}"
            )
        )

        # -------------------------
        # 保存模型系数
        # -------------------------

        coefficient_path = os.path.join(
            TABLE_DIR,
            f"{name}_最终岭回归模型系数.xlsx"
        )

        result[
            "coefficients"
        ].to_excel(
            coefficient_path,
            index=False
        )

        # -------------------------
        # 保存CV结果
        # -------------------------

        cv_path = os.path.join(
            TABLE_DIR,
            f"{name}_最终岭回归CV.xlsx"
        )

        result[
            "cv"
        ].to_excel(
            cv_path,
            index=False
        )

        # -------------------------
        # 保存预测结果
        # -------------------------

        prediction_path = os.path.join(
            TABLE_DIR,
            f"{name}_最终模型预测结果.xlsx"
        )

        result[
            "predictions"
        ].to_excel(
            prediction_path,
            index=False
        )

        print(
            f"\n[完成] 系数："
            f"{coefficient_path}"
        )

        print(
            f"[完成] CV："
            f"{cv_path}"
        )

        print(
            f"[完成] 预测："
            f"{prediction_path}"
        )

        # -------------------------
        # 因素影响图
        # -------------------------

        plot_factor_effects(
            df,
            result["model"],
            y_col,
            FIGURE_DIR
        )

        # -------------------------
        # 实际值-预测值图
        # -------------------------

        plot_actual_vs_predicted(
            result["predictions"],
            y_col,
            FIGURE_DIR
        )

    print("\n" + "=" * 60)
    print("Q2最终模型及可视化全部完成")
    print("=" * 60)


# =========================
# 程序入口
# =========================

if __name__ == "__main__":
    main()