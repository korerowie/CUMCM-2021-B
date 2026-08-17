# =========================
# Q1-2：附件2中除C4烯烃选择性外，其他各项指标随时间的变化 - 散点图
# =========================
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_preferred_fonts = ["Microsoft YaHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei"]
_available = {f.name for f in fm.fontManager.ttflist}
_font = next((f for f in _preferred_fonts if f in _available), None)
if _font:
    plt.rcParams["font.sans-serif"] = [_font]
plt.rcParams["axes.unicode_minus"] = False

DATA_PATH = "../2021-B/附件2.xlsx"
FIGURE_DIR = "figures"

COLORS = ["#3274A1", "#3274A1", "#3274A1"]

# 附件2中除C4烯烃选择性外要画的指标
TARGET_COLS = [
    "乙醇转化率(%)",
    "乙烯选择性(%)",
    "乙醛选择性(%)",
    "碳数4-12脂肪醇选择性(%)",
    "甲基苯甲醛和甲基苯甲醇选择性(%)",
    "其他选择性(%)",
]


def load_attachment2(path):
    df = pd.read_excel(path, sheet_name="稳定性测试", header=None)
    data = df.iloc[3:, :].reset_index(drop=True)
    data.columns = [
        "时间(min)", "乙醇转化率(%)", "乙烯选择性(%)", "C4烯烃选择性(%)",
        "乙醛选择性(%)", "碳数4-12脂肪醇选择性(%)", "甲基苯甲醛和甲基苯甲醇选择性(%)", "其他选择性(%)",
    ]
    return data.apply(pd.to_numeric)


def plot_other_indicators(data, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    t = data["时间(min)"]

    for ax, col, color in zip(axes.flat, TARGET_COLS, COLORS * 2):
        ax.scatter(t, data[col], s=45, color=color, zorder=3)
        ax.plot(t, data[col], color=color, alpha=0.35, lw=1, zorder=2)  # 辅助线
        ax.set_title(col, fontsize=11)
        ax.set_xlabel("时间 (min)")
        ax.set_ylabel(col)
        ax.grid(alpha=0.3)

    fig.suptitle("其他指标随时间的变化", fontsize=16)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    import os
    os.makedirs(FIGURE_DIR, exist_ok=True)

    data = load_attachment2(DATA_PATH)
    fig_path = os.path.join(FIGURE_DIR, "其他指标随时间变化散点图.png")
    plot_other_indicators(data, fig_path)
    print(f"[完成] 散点图已保存: {fig_path}")