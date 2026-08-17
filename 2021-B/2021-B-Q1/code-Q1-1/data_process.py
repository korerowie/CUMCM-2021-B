# =========================
# Q1-1：数据读取与预处理
# =========================

import pandas as pd

# 附件1中催化剂组合的编号顺序（装料方式I: A1~A14，装料方式II: B1~B7）
CATALYST_ORDER = [f"A{i}" for i in range(1, 15)] + [f"B{i}" for i in range(1, 8)]


def load_data(file_path):
    # 按 编号+温度 排序
    df = pd.read_excel(file_path)
    df["催化剂组合编号"] = df["催化剂组合编号"].ffill()
    df = df.sort_values(["催化剂组合编号", "温度"]).reset_index(drop=True)
    return df


def get_catalyst_series(df, catalyst, target_col):
    # 按温度升序排列
    sub = df[df["催化剂组合编号"] == catalyst].sort_values("温度")
    T = sub["温度"].to_numpy(dtype=float)
    y = sub[target_col].to_numpy(dtype=float)
    return T, y
