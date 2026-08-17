# =========================
# Q2：单因素方差分析
# =========================
"""
探讨不同催化剂组合及温度对乙醇转化率、C4烯烃选择性的影响
先用单因素方差分析(eta^2)初筛三个因素的相对重要性，再决定用什么多元回归模型
"""
import os
import pandas as pd

from data_process import load_data
from model_anova import one_way_anova
from visualization import plot_eta_comparison

DATA_PATH = "../2021-B/附件1.xlsx"
TABLE_DIR = "tables"
FIGURE_DIR = "figures"

FACTOR_COLS = ["X3_温度", "X1_Co质量比流速", "X2_流速x总质量"]
Y_COLS = ["乙醇转化率(%)", "C4烯烃选择性(%)"]


def main():
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    df = load_data(DATA_PATH)

    results = []
    for factor in FACTOR_COLS:
        for y_col in Y_COLS:
            results.append(one_way_anova(df, factor, y_col))
    results_df = pd.DataFrame(results)

    table_path = os.path.join(TABLE_DIR, "单因素方差分析结果.xlsx")
    results_df.to_excel(table_path, index=False)
    print(f"[完成] 方差分析结果表已保存: {table_path}")
    print(results_df[["因素", "因变量", "分组数k", "F值", "P值", "eta^2(SSA/SST)"]].to_string(index=False))

    fig_path = os.path.join(FIGURE_DIR, "各因素影响程度对比.png")
    plot_eta_comparison(results_df, fig_path)
    print(f"\n[完成] 影响程度对比图已保存: {fig_path}")


if __name__ == "__main__":
    main()