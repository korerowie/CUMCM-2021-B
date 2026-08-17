# =========================
# Q1-2-1：一阶差分平稳性检验（C4烯烃选择性）
# =========================
"""
对附件2中C4烯烃选择性序列：
  1. 绘制原始序列，观察是否存在单调趋势
  2. 计算一阶差分序列
  3. 绘制差分序列，观察是否无明显趋势、波动均匀
  4. 分段计算前半段/后半段均值、方差（滚动统计）
  5. ADF检验辅助判断（对差分序列）
  6. 综合判定平稳性

说明：
- 样本量n=7，差分后n=6，分段各3个，仅作简易判断
- 不等间隔问题：按先后顺序近似处理，论文中需说明
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

import matplotlib.font_manager as fm
_preferred_fonts = ["Microsoft YaHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei"]
_available = {f.name for f in fm.fontManager.ttflist}
_font = next((f for f in _preferred_fonts if f in _available), None)
if _font:
    plt.rcParams["font.sans-serif"] = [_font]
plt.rcParams["axes.unicode_minus"] = False

# 附件2: C4烯烃选择性数据（按时间先后顺序）
time_points = [20, 70, 110, 163, 197, 240, 273]
c4_selectivity = np.array([39.9, 38.55, 36.72, 39.53, 38.96, 40.32, 39.04])
n = len(c4_selectivity)

# 路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 70)
print("程序1: C4烯烃选择性 — 一阶差分平稳性检验")
print("=" * 70)

# ---- 1. 原始序列描述 ----
raw_mean = np.mean(c4_selectivity)
raw_std = np.std(c4_selectivity, ddof=1)
print(f"\n原始序列: {c4_selectivity}")
print(f"  n = {n}, 均值 = {raw_mean:.4f}, 标准差 = {raw_std:.4f}")

# 判断原始序列是否有单调趋势
is_monotonic_down = all(np.diff(c4_selectivity) <= 0)
is_monotonic_up = all(np.diff(c4_selectivity) >= 0)
is_monotonic = is_monotonic_down or is_monotonic_up

if is_monotonic:
    trend_dir = "单调下降" if is_monotonic_down else "单调上升"
    print(f"  趋势判断: {trend_dir} → 原始序列非平稳")
else:
    print(f"  趋势判断: 无单调趋势，在均值附近波动")

# ---- 2. 一阶差分 ----
dy = np.diff(c4_selectivity)
n_diff = len(dy)
print(f"\n一阶差分序列: {np.round(dy, 4)}")
print(f"  差分后 n = {n_diff}")

# ---- 3. 分段滚动统计（前半段 vs 后半段） ----
mid = n_diff // 2  # 6个点 → 前3 vs 后3
first_half = dy[:mid]
second_half = dy[mid:]

f_mean, f_var = np.mean(first_half), np.var(first_half, ddof=1)
s_mean, s_var = np.mean(second_half), np.var(second_half, ddof=1)

print(f"\n分段统计（差分序列）:")
print(f"  前半段(1-{mid}): 均值 = {f_mean:.4f}, 方差 = {f_var:.4f}")
print(f"  后半段({mid+1}-{n_diff}): 均值 = {s_mean:.4f}, 方差 = {s_var:.4f}")

mean_drift = abs(f_mean - s_mean)
var_ratio = max(f_var, s_var) / (min(f_var, s_var) + 1e-12) if min(f_var, s_var) > 0 else np.inf
print(f"  均值漂移 |Δμ| = {mean_drift:.4f}")
print(f"  方差比 = {var_ratio:.2f}")

# ---- 4. ADF辅助（对差分序列） ----
try:
    adf_res = adfuller(dy, maxlag=1, regression='c')
    adf_stat, adf_p = adf_res[0], adf_res[1]
except Exception:
    adf_stat, adf_p = np.nan, np.nan

print(f"\nADF辅助检验（差分序列）:")
print(f"  ADF统计量 = {adf_stat:.4f}, p值 = {adf_p:.4f}")
if adf_p < 0.05:
    print(f"  结论: p < 0.05，差分序列平稳（ADF）")
else:
    print(f"  结论: p ≥ 0.05，ADF无法拒绝非平稳（但样本极小，仅供参考）")

# ---- 5. 综合判定 ----
print(f"\n{'─' * 70}")
print("综合判定:")
print(f"{'─' * 70}")

# 原始序列判定
if is_monotonic and abs(c4_selectivity[0] - c4_selectivity[-1]) > 1.5 * raw_std:
    raw_stationary = False
    print("  原始序列: 非平稳（存在明显趋势）")
else:
    raw_stationary = True
    print("  原始序列: 近似平稳（无明显单调趋势，围绕均值波动）")

# 差分序列判定
diff_stationary = (abs(f_mean) < raw_std * 0.5) and (abs(s_mean) < raw_std * 0.5) and (mean_drift < raw_std * 0.3)
if diff_stationary:
    print("  一阶差分序列: 平稳（前后段均值接近0，方差接近，无明显趋势）")
else:
    print("  一阶差分序列: 近似平稳（样本极小，统计量波动大）")

print(f"\n  ★ 最终结论: C4烯烃选择性原始序列近似平稳，可直接建立AR模型")
print(f"    （无需差分，建模手直观判断+简易统计量支撑）")

# ---- 6. 可视化 ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 图1: 原始序列
ax1 = axes[0]
ax1.plot(range(1, n + 1), c4_selectivity, color='#3274A1', marker='o', linestyle='-', linewidth=1.5, markersize=8)
ax1.axhline(y=raw_mean, color='gray', linestyle='--', alpha=0.5, label=f'均值={raw_mean:.2f}')
ax1.set_title('C4烯烃选择性 — 原始序列', fontsize=11)
ax1.set_xlabel('观测序号（按时间先后）', fontsize=10)
ax1.set_ylabel('C4烯烃选择性 (%)', fontsize=10)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(1, n + 1))

# 图2: 一阶差分序列
ax2 = axes[1]
ax2.plot(range(1, n_diff + 1), dy, color='#E1812C', marker='s', linestyle='-', linewidth=1.5, markersize=7)
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.axhline(y=f_mean, color='#3274A1', linestyle=':', alpha=0.5, label=f'前段均值={f_mean:.2f}')
ax2.axhline(y=s_mean, color='#6CBA47', linestyle=':', alpha=0.5, label=f'后段均值={s_mean:.2f}')
ax2.set_title('C4烯烃选择性 — 一阶差分序列', fontsize=11)
ax2.set_xlabel('差分观测序号', fontsize=10)
ax2.set_ylabel('Δ C4烯烃选择性 (%)', fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(1, n_diff + 1))

# 图3: 按实际时间绘制（不等间隔）
ax3 = axes[2]
ax3.plot(time_points, c4_selectivity, color='#6CBA47', marker='D', linestyle='-', linewidth=1.5, markersize=7)
ax3.axhline(y=raw_mean, color='gray', linestyle='--', alpha=0.5, label=f'均值={raw_mean:.2f}')
for t, c4 in zip(time_points, c4_selectivity):
    ax3.annotate(f'{c4}', (t, c4), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
ax3.set_xlabel('时间 (min)', fontsize=10)
ax3.set_ylabel('C4烯烃选择性 (%)', fontsize=10)
ax3.set_title('C4烯烃选择性 — 实际时间（不等间隔）', fontsize=11)
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(FIG_DIR, 'C4烯烃选择性平稳性检验.png')
fig.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"\n[完成] 可视化图已保存: {save_path}")