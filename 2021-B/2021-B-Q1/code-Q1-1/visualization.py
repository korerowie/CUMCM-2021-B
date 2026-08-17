# =========================
# Q1-1：拟合结果可视化
# =========================
"""
沿用之前散点图的3x7网格布局，把每个催化剂组合的最优拟合曲线叠加在散点上，
方便和已有的散点图直接对照，也方便整体放进论文。
"""
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.font_manager as fm

_preferred_fonts = ["Microsoft YaHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei"]
_available = {f.name for f in fm.fontManager.ttflist}
_font = next((f for f in _preferred_fonts if f in _available), None)
if _font:
    plt.rcParams["font.sans-serif"] = [_font]
plt.rcParams["axes.unicode_minus"] = False

CATALYST_ORDER = [f"A{i}" for i in range(1, 15)] + [f"B{i}" for i in range(1, 8)]


def plot_fit_grid(best_results, target_name, save_path, highlight_colors=None):
    """
    best_results: dict，key为催化剂编号，value为dict，包含
        T, y, best_model_name, best_params, best_metrics, model_func
    highlight_colors: dict，可选，{催化剂编号: 颜色}，用于把个别单独处理的催化剂
        (比如用S型模型单独拟合的B6/A13/A3)用不同颜色的曲线区分出来
    """
    highlight_colors = highlight_colors or {}
    fig, axes = plt.subplots(3, 7, figsize=(20, 9))

    for ax, catalyst in zip(axes.flat, CATALYST_ORDER):
        info = best_results[catalyst]
        T, y = info["T"], info["y"]

        ax.scatter(T, y, s=30, color="#3274A1", zorder=3, label="实验数据")

        T_dense = np.linspace(T.min(), T.max(), 200)
        y_dense = info["model_func"](T_dense, **info["best_params"])
        curve_color = highlight_colors.get(catalyst, "#E1812C")
        line_width = 2.2 if catalyst in highlight_colors else 1.8
        ax.plot(T_dense, y_dense, color=curve_color, lw=line_width, zorder=2)

        r2 = info["best_metrics"]["R2"]
        ax.set_title(f"{catalyst}  {info['best_model_name']}  R²={r2:.3f}", fontsize=9)
        ax.set_xlabel("温度 (℃)", fontsize=8)
        ax.set_ylabel(target_name, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(f"温度—{target_name} 最优模型拟合结果", fontsize=16)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
