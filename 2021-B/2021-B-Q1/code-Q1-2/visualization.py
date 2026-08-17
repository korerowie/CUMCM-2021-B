# =========================
# Q1-2-4：数据可视化（C4烯烃选择性）
# =========================
"""
包含：
  1. 原始序列时序图（按观测序号）
  2. 一阶差分序列图
  3. 原始序列按实际时间绘制（不等间隔）
  4. AR(1)拟合效果图
  5. AR(2)拟合效果图
  6. AIC模型选择对比
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
import warnings
warnings.filterwarnings('ignore')

import matplotlib.font_manager as fm
_preferred_fonts = ["Microsoft YaHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei"]
_available = {f.name for f in fm.fontManager.ttflist}
_font = next((f for f in _preferred_fonts if f in _available), None)
if _font:
    plt.rcParams["font.sans-serif"] = [_font]
plt.rcParams["axes.unicode_minus"] = False

# 路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# 数据
time_points = [20, 70, 110, 163, 197, 240, 273]
c4_selectivity = np.array([39.9, 38.55, 36.72, 39.53, 38.96, 40.32, 39.04])
n = len(c4_selectivity)

# 一阶差分
dy = np.diff(c4_selectivity)
n_diff = len(dy)

# 分段均值
mid = n_diff // 2
f_mean = np.mean(dy[:mid])
s_mean = np.mean(dy[mid:])

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# ── 第一行 ──

# 图1: 原始序列（按观测序号）
ax1 = axes[0, 0]
ax1.plot(range(1, n + 1), c4_selectivity, color='#3274A1', marker='o', linestyle='-', linewidth=1.5, markersize=8)
ax1.axhline(y=np.mean(c4_selectivity), color='gray', linestyle='--', alpha=0.5,
            label=f'均值={np.mean(c4_selectivity):.2f}')
ax1.set_xlabel('观测序号（按时间先后）', fontsize=10)
ax1.set_ylabel('C4烯烃选择性 (%)', fontsize=10)
ax1.set_title('(a) 原始序列', fontsize=11, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(1, n + 1))

# 图2: 一阶差分序列
ax2 = axes[0, 1]
ax2.plot(range(1, n_diff + 1), dy, color='#E1812C', marker='s', linestyle='-', linewidth=1.5, markersize=7)
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.axhline(y=f_mean, color='#3274A1', linestyle=':', alpha=0.5, label=f'前段均值={f_mean:.2f}')
ax2.axhline(y=s_mean, color='#6CBA47', linestyle=':', alpha=0.5, label=f'后段均值={s_mean:.2f}')
ax2.set_xlabel('差分观测序号', fontsize=10)
ax2.set_ylabel('Δ C4烯烃选择性 (%)', fontsize=10)
ax2.set_title('(b) 一阶差分序列', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(1, n_diff + 1))

# 图3: 按实际时间绘制（不等间隔）
ax3 = axes[0, 2]
ax3.plot(time_points, c4_selectivity, color='#6CBA47', marker='D', linestyle='-', linewidth=1.5, markersize=7)
ax3.axhline(y=np.mean(c4_selectivity), color='gray', linestyle='--', alpha=0.5,
            label=f'均值={np.mean(c4_selectivity):.2f}')
for t, c4 in zip(time_points, c4_selectivity):
    ax3.annotate(f'{c4}', (t, c4), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
ax3.set_xlabel('时间 (min)', fontsize=10)
ax3.set_ylabel('C4烯烃选择性 (%)', fontsize=10)
ax3.set_title('(c) 实际时间序列（不等间隔）', fontsize=11, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# ── 第二行 ──

# 图4: AR(1)拟合效果
ax4 = axes[1, 0]
model1 = AutoReg(c4_selectivity, lags=1, trend='c').fit()
pred1 = model1.fittedvalues
ax4.plot(range(1, n + 1), c4_selectivity, color='#3274A1', marker='o', linestyle='-',
         label='观测值', linewidth=1.5, markersize=8)
ax4.plot(range(2, n + 1), pred1, color='#E1812C', marker='s', linestyle='--',
         label='AR(1)拟合值', linewidth=1.5, markersize=7)
ax4.set_xlabel('观测序号', fontsize=10)
ax4.set_ylabel('C4烯烃选择性 (%)', fontsize=10)
ax4.set_title(f'(d) AR(1)拟合效果 (AIC={model1.aic:.2f})', fontsize=11, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_xticks(range(1, n + 1))

# 图5: AR(2)拟合效果
ax5 = axes[1, 1]
model2 = AutoReg(c4_selectivity, lags=2, trend='c').fit()
pred2 = model2.fittedvalues
ax5.plot(range(1, n + 1), c4_selectivity, color='#3274A1', marker='o', linestyle='-',
         label='观测值', linewidth=1.5, markersize=8)
ax5.plot(range(3, n + 1), pred2, color='#6CBA47', marker='s', linestyle='--',
         label='AR(2)拟合值', linewidth=1.5, markersize=7)
ax5.set_xlabel('观测序号', fontsize=10)
ax5.set_ylabel('C4烯烃选择性 (%)', fontsize=10)
ax5.set_title(f'(e) AR(2)拟合效果 (AIC={model2.aic:.2f})', fontsize=11, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)
ax5.set_xticks(range(1, n + 1))

# 图6: AIC对比与残差
ax6 = axes[1, 2]
ax6.bar(['AR(1)', 'AR(2)'], [model1.aic, model2.aic], color=['#E1812C', '#6CBA47'], edgecolor='black')
ax6.set_ylabel('AIC', fontsize=10)
ax6.set_title('(f) AIC模型选择', fontsize=11, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')
for i, v in enumerate([model1.aic, model2.aic]):
    ax6.text(i, v + 0.3, f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')

# 标注最优
best_p = 1 if model1.aic < model2.aic else 2
ax6.annotate(f'★ 最优: AR({best_p})', xy=(best_p-1, [model1.aic, model2.aic][best_p-1]),
             xytext=(best_p-1, [model1.aic, model2.aic][best_p-1] - 2),
             ha='center', fontsize=11, color='red', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='red'))

plt.tight_layout()
save_path = os.path.join(FIG_DIR, 'C4烯烃选择性平稳性的AR拟合.png')
fig.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"[完成] 可视化图已保存: {save_path}")