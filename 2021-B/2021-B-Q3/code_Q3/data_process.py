# =========================
# Q3：数据读取与特征构造
# =========================
"""
沿用Q2的三个自变量：
    X1 = Co绝对质量 / 乙醇流速
       = (Co负载量(%) / 100 × Co/SiO2质量(mg)) / 流速(ml/min)
    X2 = 乙醇流速(ml/min) × 装料总质量(mg)
       = 流速 × (Co/SiO2质量 + 第二载体质量)
       （A11无HAP，用石英砂质量代替第二载体）
    X3 = 温度(℃)（附件1原始温度列）

同时计算C4烯烃收率 = 乙醇转化率(%) × C4烯烃选择性(%) / 100
"""

import re
import pandas as pd
import numpy as np

# 催化剂组合文本解析正则
_PATTERN = re.compile(
    r"(?P<co_sio2_mass>\d+\.?\d*)mg\s*(?P<co_wt>\d+\.?\d*)wt%Co/SiO2[+\-]?\s*"
    r"(?P<second_mass>\d+\.?\d*)mg\s*(?P<second_type>HAP|石英砂).*?"
    r"乙醇浓度(?P<flow_rate>\d+\.?\d*)ml/min"
)


def parse_catalyst_text(text):
    # 从催化剂组合描述文本解析四个原始数值
    m = _PATTERN.search(text)
    if not m:
        raise ValueError(f"无法解析催化剂组合文本: {text}")
    d = m.groupdict()
    return pd.Series({
        "Co_SiO2质量": float(d["co_sio2_mass"]),
        "Co负载量": float(d["co_wt"]),
        "第二载体质量": float(d["second_mass"]),
        "流速": float(d["flow_rate"]),
    })


def load_and_process_data(file_path):
    # 读取附件1，构造特征，返回DataFrame
    df = pd.read_excel(file_path)

    # 前向填充催化剂编号和组合文本
    df["催化剂组合编号"] = df["催化剂组合编号"].ffill()
    df["催化剂组合"] = df.groupby("催化剂组合编号")["催化剂组合"].ffill()

    # 解析催化剂组合文本
    parsed = df["催化剂组合"].apply(parse_catalyst_text)
    df = pd.concat([df, parsed], axis=1)

    # 构造自变量
    df["Co绝对质量"] = df["Co负载量"] / 100 * df["Co_SiO2质量"]
    df["装料总质量"] = df["Co_SiO2质量"] + df["第二载体质量"]

    df["X1_Co质量比流速"] = df["Co绝对质量"] / df["流速"]
    df["X2_流速x总质量"] = df["流速"] * df["装料总质量"]
    df["X3_温度"] = df["温度"]

    # C4烯烃收率
    df["C4烯烃收率"] = df["乙醇转化率(%)"] * df["C4烯烃选择性(%)"] / 100

    df = df.sort_values(["催化剂组合编号", "温度"]).reset_index(drop=True)
    return df


# 自变量列名
X_COLS = ["X1_Co质量比流速", "X2_流速x总质量", "X3_温度"]
Y_COLS = ["乙醇转化率(%)", "C4烯烃选择性(%)"]