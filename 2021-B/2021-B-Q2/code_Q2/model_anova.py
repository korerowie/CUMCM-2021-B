# =========================
# Q2：单因素方差分析
# =========================
"""
单因素方差分析(One-way ANOVA)，用来初步比较各个自变量对因变量的影响程度

对某个分组变量factor_col，把因变量y_col的数据按factor_col的取值(完全相等才算一组)分组：
    SST(总平方和)   = sum((y_i - y_bar)^2)                    全部数据对总均值的离差平方和
    SSA(组间平方和) = sum(n_k * (y_k_bar - y_bar)^2)          每组均值对总均值的离差平方和
    SSE(组内平方和) = SST - SSA                               每组内部对本组均值的离差平方和
    eta^2 = SSA / SST      该因素能解释的方差比例，越大说明这个因素对结果的影响越大

注意：这里是逐个因素单独分组比较，没有控制其他因素的影响，组内噪声里混杂了其他自变量的
效应，只能作为筛选各因素相对重要性的粗略参考，不能替代后面的多元回归模型。
"""
import numpy as np
import pandas as pd
from scipy import stats


def one_way_anova(df, factor_col, y_col, alpha=0.05):
    groups = df.groupby(factor_col)[y_col]

    y_bar = df[y_col].mean()
    n_total = len(df)
    k = groups.ngroups

    sst = float(np.sum((df[y_col] - y_bar) ** 2))
    ssa = float(groups.apply(lambda g: len(g) * (g.mean() - y_bar) ** 2).sum())
    sse = sst - ssa

    df_between = k - 1
    df_within = n_total - k

    msa = ssa / df_between if df_between > 0 else np.nan
    mse = sse / df_within if df_within > 0 else np.nan

    if mse and mse > 0 and not np.isnan(msa):
        f_value = msa / mse
        p_value = stats.f.sf(f_value, df_between, df_within)
        f_crit = stats.f.ppf(1 - alpha, df_between, df_within)
    else:
        f_value, p_value, f_crit = np.nan, np.nan, np.nan

    eta_sq = ssa / sst if sst > 0 else np.nan

    return {
        "因素": factor_col,
        "因变量": y_col,
        "分组数k": k,
        "总样本量n": n_total,
        "SST": sst,
        "SSA": ssa,
        "SSE": sse,
        "df组间": df_between,
        "df组内": df_within,
        "MSA": msa,
        "MSE": mse,
        "F值": f_value,
        "P值": p_value,
        "F临界值(0.05)": f_crit,
        "eta^2(SSA/SST)": eta_sq,
    }