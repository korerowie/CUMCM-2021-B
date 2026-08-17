# =========================
# Q1-1：乙醇转化率、C4烯烃的选择性与温度的关系 - 散点图
# =========================

import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

file_path = "附件1.xlsx"

df = pd.read_excel(file_path)

df["催化剂组合编号"] = df["催化剂组合编号"].ffill()

# 按催化剂编号排序
order = [f"A{i}" for i in range(1, 15)] + [f"B{i}" for i in range(1, 8)]
df["催化剂组合编号"] = pd.Categorical(
    df["催化剂组合编号"],
    categories=order,
    ordered=True
)

df = df.sort_values(["催化剂组合编号", "温度"])


# 绘制温度—乙醇转化率散点图
fig, axes = plt.subplots(3, 7, figsize=(18, 8))

for ax, catalyst in zip(axes.flat, order):

    data = df[df["催化剂组合编号"] == catalyst]

    ax.scatter(
        data["温度"],
        data["乙醇转化率(%)"],
        s=35
    )

    ax.set_title(catalyst)
    ax.set_xlabel("温度 (℃)")
    ax.set_ylabel("转化率 (%)")
    ax.grid(alpha=0.3)

plt.suptitle(
    "温度—乙醇转化率",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    "2021-B-Q1/温度-乙醇转化率-散点图.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# 绘制温度—C4烯烃选择性散点图
fig, axes = plt.subplots(3, 7, figsize=(18, 8))

for ax, catalyst in zip(axes.flat, order):

    data = df[df["催化剂组合编号"] == catalyst]

    ax.scatter(
        data["温度"],
        data["C4烯烃选择性(%)"],
        s=35
    )

    ax.set_title(catalyst)
    ax.set_xlabel("温度 (℃)")
    ax.set_ylabel("C4选择性 (%)")
    ax.grid(alpha=0.3)

plt.suptitle(
    "温度—C4烯烃选择性",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    "2021-B-Q1/温度-C4烯烃选择性-散点图.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()