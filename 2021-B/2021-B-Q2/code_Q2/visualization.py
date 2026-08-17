# =========================
# Q2：可视化：三个因素在两个因变量上的eta^2对比
# =========================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_preferred_fonts = ["Microsoft YaHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei"]
_available = {f.name for f in fm.fontManager.ttflist}
_font = next((f for f in _preferred_fonts if f in _available), None)
if _font:
    plt.rcParams["font.sans-serif"] = [_font]
plt.rcParams["axes.unicode_minus"] = False

COLORS = ["#3274A1", "#E1812C", "#6CBA47"]

FACTOR_LABELS = {
    "X3_温度": "温度",
    "X1_Co质量比流速": "Co绝对质量/流速",
    "X2_流速x总质量": "流速x装料总质量",
}


def plot_eta_comparison(results_df, save_path):
    factors = list(FACTOR_LABELS.keys())
    factor_labels = [FACTOR_LABELS[f] for f in factors]
    targets = results_df["因变量"].unique()

    x = np.arange(len(factors))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, target in enumerate(targets):
        sub = results_df[results_df["因变量"] == target].set_index("因素").reindex(factors)
        offset = (i - (len(targets) - 1) / 2) * width
        bars = ax.bar(x + offset, sub["eta^2(SSA/SST)"], width, label=target, color=COLORS[i % len(COLORS)])
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(factor_labels)
    ax.set_ylabel("η² (组间平方和 / 总平方和)")
    ax.set_ylim(0, 1.05)
    ax.set_title("各因素对乙醇转化率、C4烯烃选择性的影响程度对比")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)