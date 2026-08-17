# =========================
# Q1-2：附件2指标模型拟合
# 乙醇转化率(%)                   指数衰减
# 碳数4-12脂肪醇选择性(%)          指数衰减
# 乙烯选择性(%)                   线性
# 乙醛选择性(%)                   S型(Logistic)
# 甲基苯甲醛和甲基苯甲醇选择性(%)   三次多项式
# C4烯烃选择性(%)                 平稳(常数，取均值)
# 其他选择性(%) 不参与建模，跳过
# =========================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from other_model import (
    linear_model, fit_linear,
    exp_model, fit_exp,
    logistic_model, fit_logistic,
    cubic_model, fit_cubic,
    constant_model, fit_constant,
)
from evaluate import evaluate

_preferred_fonts = ["Microsoft YaHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei"]
_available = {f.name for f in fm.fontManager.ttflist}
_font = next((f for f in _preferred_fonts if f in _available), None)
if _font:
    plt.rcParams["font.sans-serif"] = [_font]
plt.rcParams["axes.unicode_minus"] = False

DATA_PATH = "../2021-B/附件2.xlsx"
TABLE_DIR = "tables"
FIGURE_DIR = "figures"

POINT_COLOR = "#3274A1"
FIT_COLOR = "#E1812C"

# 变量名 -> (模型显示名, 拟合函数, 计算函数)
VARIABLE_MODEL_MAP = {
    "乙醇转化率(%)": ("指数衰减", fit_exp, exp_model),
    "碳数4-12脂肪醇选择性(%)": ("指数衰减", fit_exp, exp_model),
    "乙烯选择性(%)": ("线性", fit_linear, linear_model),
    "乙醛选择性(%)": ("S型_Logistic", fit_logistic, logistic_model),
    "甲基苯甲醛和甲基苯甲醇选择性(%)": ("三次多项式", fit_cubic, cubic_model),
    "C4烯烃选择性(%)": ("平稳(常数)", fit_constant, constant_model),
}


def load_attachment2(path):
    df = pd.read_excel(path, sheet_name="稳定性测试", header=None)
    data = df.iloc[3:, :].reset_index(drop=True)
    data.columns = [
        "时间(min)", "乙醇转化率(%)", "乙烯选择性(%)", "C4烯烃选择性(%)",
        "乙醛选择性(%)", "碳数4-12脂肪醇选择性(%)", "甲基苯甲醛和甲基苯甲醇选择性(%)", "其他选择性(%)",
    ]
    return data.apply(pd.to_numeric)


def fit_all_variables(data):
    # 对VARIABLE_MODEL_MAP里的每个指标做拟合，返回结果字典和汇总表
    t = data["时间(min)"].to_numpy(dtype=float)
    results = {}
    rows = []

    for col, (model_name, fit_func, model_func) in VARIABLE_MODEL_MAP.items():
        y = data[col].to_numpy(dtype=float)
        params, y_pred = fit_func(t, y)
        metrics = evaluate(y, y_pred)

        results[col] = {
            "t": t,
            "y": y,
            "model_name": model_name,
            "params": params,
            "metrics": metrics,
            "model_func": model_func,
        }

        rows.append({
            "指标": col,
            "模型": model_name,
            "R2": metrics["R2"],
            "RMSE": metrics["RMSE"],
            **params,
        })

    return results, pd.DataFrame(rows)


def plot_fit_results(results, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for ax, (col, info) in zip(axes.flat, results.items()):
        t, y = info["t"], info["y"]

        ax.scatter(t, y, s=45, color=POINT_COLOR, zorder=3, label="实验数据")

        t_dense = np.linspace(t.min(), t.max(), 200)
        y_dense = info["model_func"](t_dense, **info["params"])
        ax.plot(t_dense, y_dense, color=FIT_COLOR, lw=1.8, zorder=2, label=info["model_name"])

        r2 = info["metrics"]["R2"]
        ax.set_title(f"{col}\n{info['model_name']}  R²={r2:.3f}", fontsize=10)
        ax.set_xlabel("时间 (min)")
        ax.set_ylabel(col)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("附件2：各指标随时间的模型拟合结果", fontsize=16)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    data = load_attachment2(DATA_PATH)
    results, summary_df = fit_all_variables(data)

    table_path = os.path.join(TABLE_DIR, "附件2_指标拟合结果.xlsx")
    summary_df.to_excel(table_path, index=False)
    print(f"[完成] 拟合结果表已保存: {table_path}")
    print(summary_df.to_string(index=False))

    fig_path = os.path.join(FIGURE_DIR, "附件2-指标拟合结果.png")
    plot_fit_results(results, fig_path)
    print(f"[完成] 拟合图已保存: {fig_path}")