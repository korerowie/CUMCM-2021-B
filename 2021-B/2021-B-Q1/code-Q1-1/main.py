# =========================
# Q1-1：模型筛选主程序
# =========================
"""
流程：
    读取附件1数据
    -> 对每个催化剂组合、每个目标变量(乙醇转化率/C4烯烃选择性)分别拟合三种候选模型
    -> 用R2、RMSE评价拟合优度
    -> 选出每组的最优模型
    -> 输出拟合明细表、最优模型汇总表、拟合结果图
"""
import os
import pandas as pd

from data_process import load_data, get_catalyst_series, CATALYST_ORDER
from model import MODEL_FIT_REGISTRY, MODEL_FUNC_REGISTRY, fit_logistic, logistic_model
from evaluate import evaluate
from visualization import plot_fit_grid

DATA_PATH = "../2021-B/附件1.xlsx"
TABLE_DIR = "tables"
FIGURE_DIR = "figures"

TARGET_COLS = ["乙醇转化率(%)", "C4烯烃选择性(%)"]

# C4烯烃选择性中，B6/A13/A3 三个催化剂呈S形变化，M1~M3效果不好
# 单独用S型(Logistic)模型替换掉原来选出的最优模型
SPECIAL_TARGET = "C4烯烃选择性(%)"
SPECIAL_CATALYSTS = ["B6", "A13", "A3"]
HIGHLIGHT_COLORS = {
    "B6": "#6CBA47",
    "A13": "#6CBA47",
    "A3": "#6CBA47", 
}

def run_model_selection(df, target_col):
    # 对某一个目标变量，逐催化剂拟合三种候选模型并选出最优模型
    detail_rows = []
    best_results = {}

    for catalyst in CATALYST_ORDER:
        T, y = get_catalyst_series(df, catalyst, target_col)

        catalyst_fits = {}
        for model_name, fit_func in MODEL_FIT_REGISTRY.items():
            params, y_pred = fit_func(T, y)
            metrics = evaluate(y, y_pred)
            catalyst_fits[model_name] = {"params": params, "metrics": metrics}

            detail_rows.append({
                "催化剂组合编号": catalyst,
                "目标变量": target_col,
                "模型": model_name,
                "R2": metrics["R2"],
                "RMSE": metrics["RMSE"],
                **params,
            })

        # 选R^2最大/RMSE最小的模型作为最优模型；指数模型若拟合失败(NaN)会被自动排除
        valid_fits = {k: v for k, v in catalyst_fits.items() if pd.notna(v["metrics"]["R2"])}
        best_model_name = max(valid_fits, key=lambda k: valid_fits[k]["metrics"]["R2"])
        best_fit = valid_fits[best_model_name]

        best_results[catalyst] = {
            "T": T,
            "y": y,
            "best_model_name": best_model_name,
            "best_params": best_fit["params"],
            "best_metrics": best_fit["metrics"],
            "model_func": MODEL_FUNC_REGISTRY[best_model_name],
        }

    return detail_rows, best_results

def apply_special_logistic_fits(df, target_col, best_results):
    # 把 SPECIAL_CATALYSTS(B6/A13/A3) 在 target_col 上的最优模型结果，
    # 替换成图中给出的S型(Logistic)模型；同时单独生成这三个催化剂的拟合数据表。
    # 只在 target_col == SPECIAL_TARGET 时调用
    special_rows = []
 
    for catalyst in SPECIAL_CATALYSTS:
        T, y = get_catalyst_series(df, catalyst, target_col)
        params, y_pred = fit_logistic(T, y)
        metrics = evaluate(y, y_pred)
 
        best_results[catalyst] = {
            "T": T,
            "y": y,
            "best_model_name": "S型_Logistic",
            "best_params": params,
            "best_metrics": metrics,
            "model_func": logistic_model,
        }
 
        special_rows.append({
            "催化剂组合编号": catalyst,
            "目标变量": target_col,
            "模型": "S型_Logistic(手动指定替换)",
            "R2": metrics["R2"],
            "RMSE": metrics["RMSE"],
            **params,
        })
 
    special_df = pd.DataFrame(special_rows)
    special_path = os.path.join(TABLE_DIR, "C4烯烃选择性_S型拟合结果_B6_A13_A3.xlsx")
    special_df.to_excel(special_path, index=False)
    print(f"[完成] B6/A13/A3 的S型拟合表已单独保存: {special_path}")
 
    return special_rows
 

def main():
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    df = load_data(DATA_PATH)

    all_detail_rows = []
    summary_rows = []

    for target_col in TARGET_COLS:
        detail_rows, best_results = run_model_selection(df, target_col)
        all_detail_rows.extend(detail_rows)

        highlight_colors = None
        if target_col == SPECIAL_TARGET:
            # 用S型模型替换B6/A13/A3的拟合结果，
            # M1~M3的原始拟合结果仍保留在detail表里，方便对比看出S型确实更合适
            special_rows = apply_special_logistic_fits(df, target_col, best_results)
            all_detail_rows.extend(special_rows)
            highlight_colors = HIGHLIGHT_COLORS

        for catalyst, info in best_results.items():
            summary_rows.append({
                "催化剂组合编号": catalyst,
                "目标变量": target_col,
                "最优模型": info["best_model_name"],
                "R2": info["best_metrics"]["R2"],
                "RMSE": info["best_metrics"]["RMSE"],
                **info["best_params"],
            })

        target_short = "乙醇转化率" if "乙醇" in target_col else "C4烯烃选择性"
        fig_path = os.path.join(FIGURE_DIR, f"温度-{target_short}-最优模型拟合.png")
        plot_fit_grid(best_results, target_col, fig_path, highlight_colors=highlight_colors)
        print(f"[完成] {target_col} 拟合图已保存: {fig_path}")

    detail_df = pd.DataFrame(all_detail_rows)
    summary_df = pd.DataFrame(summary_rows)

    detail_path = os.path.join(TABLE_DIR, "模型拟合明细_全部候选模型.xlsx")
    summary_path = os.path.join(TABLE_DIR, "模型筛选结果_最优模型汇总.xlsx")
    detail_df.to_excel(detail_path, index=False)
    summary_df.to_excel(summary_path, index=False)

    print(f"[完成] 拟合明细表已保存: {detail_path}")
    print(f"[完成] 最优模型汇总表已保存: {summary_path}")

    print()
    print("各模型作为最优模型的次数统计：")
    print(summary_df.groupby(["目标变量", "最优模型"]).size())


if __name__ == "__main__":
    main()
