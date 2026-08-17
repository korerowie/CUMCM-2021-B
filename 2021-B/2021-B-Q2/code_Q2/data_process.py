# =========================
# Q2：数据读取与特征构造
# =========================
"""
从"催化剂组合"文本里解析出Co/SiO2质量、Co负载量、第二载体质量(HAP或石英砂)、乙醇流速，
构造出三个自变量：
    X1 = Co绝对质量(Co负载量 x Co/SiO2质量) / 乙醇流速
    X2 = 乙醇流速 x 装料总质量(Co/SiO2质量 + 第二载体质量)
    X3 = 温度(附件1原始温度列)
"""
import re
import pandas as pd

CATALYST_ORDER = [f"A{i}" for i in range(1, 15)] + [f"B{i}" for i in range(1, 8)]

_PATTERN = re.compile(
    r"(?P<co_sio2_mass>\d+\.?\d*)mg\s*(?P<co_wt>\d+\.?\d*)wt%Co/SiO2[+\-]\s*"
    r"(?P<second_mass>\d+\.?\d*)mg\s*(?P<second_type>HAP|石英砂).*?"
    r"乙醇浓度(?P<flow_rate>\d+\.?\d*)ml/min"
)


def parse_catalyst_text(text):
    # 从催化剂组合描述文本里解析出四个原始数值，解析不出来直接报错
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


def load_data(file_path):
    # 读取附件1，前向填充催化剂编号/组合文本，构造三个自变量
    df = pd.read_excel(file_path)
    df["催化剂组合编号"] = df["催化剂组合编号"].ffill()
    df["催化剂组合"] = df.groupby("催化剂组合编号")["催化剂组合"].ffill()

    parsed = df["催化剂组合"].apply(parse_catalyst_text)
    df = pd.concat([df, parsed], axis=1)

    df["Co绝对质量"] = df["Co负载量"] / 100 * df["Co_SiO2质量"]
    df["装料总质量"] = df["Co_SiO2质量"] + df["第二载体质量"]

    df["X1_Co质量比流速"] = df["Co绝对质量"] / df["流速"]
    df["X2_流速x总质量"] = df["流速"] * df["装料总质量"]
    df["X3_温度"] = df["温度"]

    df = df.sort_values(["催化剂组合编号", "温度"]).reset_index(drop=True)
    return df