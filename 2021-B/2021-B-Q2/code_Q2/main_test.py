# =========================
# Q2：多元回归模型
# =========================
"""
研究不同催化剂组合及温度对：

    1. 乙醇转化率
    2. C4烯烃选择性

的影响。

模型：
    1. 多元多项式回归
    2. 岭回归

数据：
    110条实验数据
    随机10条作为测试集
    100条作为训练集

注意：
    测试集只用于最终模型评价，
    不参与模型训练和岭回归参数选择。
"""

import os
import pandas as pd

from data_process import load_data
from model_regression_test import run_all_models


# 路径设置
DATA_PATH = "../2021-B/附件1.xlsx"
TABLE_DIR = "tables"
FIGURE_DIR = "figures"

# 因变量
Y_COLS = [
    "乙醇转化率(%)",
    "C4烯烃选择性(%)"
]

# 参数设置
TEST_SIZE = 10
RANDOM_STATE = 2026

# =========================
# 主程序
# =========================
def main():
    # 创建结果文件夹
    os.makedirs(
        TABLE_DIR,
        exist_ok=True
    )

    os.makedirs(
        FIGURE_DIR,
        exist_ok=True
    )

    # 读取并处理数据
    print("=" * 60)
    print("Q2：数据读取与多元回归建模")
    print("=" * 60)

    df = load_data(DATA_PATH)

    print(f"\n数据总量：{len(df)} 条")

    # 建立模型
    (
        train_df,
        test_df,
        all_results
    ) = run_all_models(
        df,
        Y_COLS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # 输出训练集/测试集信息
    print("\n" + "=" * 60)
    print("数据集划分")
    print("=" * 60)

    print(f"训练集：{len(train_df)} 条")

    print(f"测试集：{len(test_df)} 条")

    print("\n测试集数据：")

    print(
        test_df[
            ["催化剂组合编号", "温度"]
        ].to_string(index=False)
    )

    # 保存训练集和测试集
    train_path = os.path.join(
        TABLE_DIR,
        "训练集.xlsx"
    )

    test_path = os.path.join(
        TABLE_DIR,
        "测试集.xlsx"
    )

    train_df.to_excel(
        train_path,
        index=False
    )

    test_df.to_excel(
        test_path,
        index=False
    )

    print(
        f"\n[完成] 训练集已保存：{train_path}"
    )

    print(
        f"[完成] 测试集已保存：{test_path}"
    )

    # 分别处理两个因变量
    comparison_list = []

    for y_col in Y_COLS:

        print("\n" + "=" * 60)
        print(f"因变量：{y_col}")
        print("=" * 60)

        result = all_results[y_col]

        # 模型比较结果
        comparison_df = result["comparison"]

        comparison_list.append(
            comparison_df
        )

        print("\n模型比较：")

        print(
            comparison_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}"
            )
        )

        # 最优lambda
        print(
            f"\n最优岭回归 lambda = "
            f"{result['best_lambda']:.6g}"
        )

        # 保存模型比较结果
        safe_name = (
            y_col
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
        )

        comparison_path = os.path.join(
            TABLE_DIR,
            f"{safe_name}_模型比较.xlsx"
        )

        comparison_df.to_excel(
            comparison_path,
            index=False
        )

        # 保存测试集预测结果
        prediction_path = os.path.join(
            TABLE_DIR,
            f"{safe_name}_测试集预测.xlsx"
        )

        result["prediction"].to_excel(
            prediction_path,
            index=False
        )

        # 保存交叉验证结果
        cv_path = os.path.join(
            TABLE_DIR,
            f"{safe_name}_岭回归交叉验证.xlsx"
        )

        result["cv"].to_excel(
            cv_path,
            index=False
        )

        # 保存模型系数
        coef_path = os.path.join(
            TABLE_DIR,
            f"{safe_name}_模型系数.xlsx"
        )

        result["coefficients"].to_excel(
            coef_path,
            index=False
        )

        print(
            f"\n[完成] 模型比较结果：{comparison_path}"
        )

        print(
            f"[完成] 测试集预测结果：{prediction_path}"
        )

        print(
            f"[完成] 岭回归CV结果：{cv_path}"
        )

        print(
            f"[完成] 模型系数：{coef_path}"
        )

    # 汇总两个因变量
    all_comparison_df = pd.concat(
        comparison_list,
        ignore_index=True
    )

    all_comparison_path = os.path.join(
        TABLE_DIR,
        "两个因变量模型比较汇总.xlsx"
    )

    all_comparison_df.to_excel(
        all_comparison_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("全部建模完成")
    print("=" * 60)

    print(
        f"\n[完成] 汇总结果："
        f"{all_comparison_path}"
    )


# =========================
# 程序入口
# =========================

if __name__ == "__main__":
    main()