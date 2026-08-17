# =========================
# Q4：新增实验方案基础数据计算
# =========================

import os
import re
import pandas as pd


# =========================
# 1. 参数设置
# =========================

DATA_PATH = "../2021-B/附件1.xlsx"

TABLE_DIR = "tables"

os.makedirs(TABLE_DIR, exist_ok=True)


# =========================
# 2. 读取数据
# =========================

df = pd.read_excel(DATA_PATH)

df["催化剂组合编号"] = (
    df["催化剂组合编号"].ffill()
)

df["催化剂组合"] = (
    df.groupby("催化剂组合编号")["催化剂组合"]
    .ffill()
)


# =========================
# 3. 解析催化剂组合
# =========================

PATTERN = re.compile(
    r"(?P<co_sio2_mass>\d+\.?\d*)mg\s*"
    r"(?P<co_wt>\d+\.?\d*)wt%Co/SiO2[+\-]\s*"
    r"(?P<second_mass>\d+\.?\d*)mg\s*"
    r"(?P<second_type>HAP|石英砂).*?"
    r"乙醇浓度(?P<flow_rate>\d+\.?\d*)ml/min"
)


def parse_catalyst_text(text):

    m = PATTERN.search(text)

    if not m:
        raise ValueError(
            f"无法解析催化剂组合：{text}"
        )

    d = m.groupdict()

    return pd.Series({
        "Co负载量": float(d["co_wt"]),
        "Co_SiO2质量": float(d["co_sio2_mass"]),
        "第二载体质量": float(d["second_mass"]),
        "乙醇流速": float(d["flow_rate"])
    })


parsed = (
    df["催化剂组合"]
    .apply(parse_catalyst_text)
)

df = pd.concat(
    [df, parsed],
    axis=1
)


# =========================
# 4. 五个原始自变量
# =========================

VARIABLES = [
    "Co负载量",
    "Co_SiO2质量",
    "第二载体质量",
    "乙醇流速",
    "温度"
]


# =========================
# 5. 中位数实验
# =========================

median_values = (
    df[VARIABLES]
    .median()
)


# =========================
# 6. 平均数实验
# =========================

mean_values = (
    df[VARIABLES]
    .mean()
)


# =========================
# 7. 整理实验基础条件
# =========================

result = pd.DataFrame({

    "变量": [
        "Co负载量(wt%)",
        "Co/SiO2质量(mg)",
        "第二载体质量(mg)",
        "乙醇流速(ml/min)",
        "温度(℃)"
    ],

    "中位数实验":
        median_values.values,

    "平均数实验":
        mean_values.values
})


# =========================
# 8. 输出结果
# =========================

print("\n")
print("=" * 60)
print("Q4：新增实验基础条件")
print("=" * 60)

print()

print(
    result.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# =========================
# 9. 保存 Excel
# =========================

save_path = os.path.join(
    TABLE_DIR,
    "Q4_实验设计基础条件.xlsx"
)

result.to_excel(
    save_path,
    index=False
)

print()

print(
    f"[完成] 实验基础条件已保存："
    f"{save_path}"
)
