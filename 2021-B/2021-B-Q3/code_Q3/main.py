# =========================
# Q3：催化剂组合与温度优化 — 粒子群优化(PSO)
# =========================
"""
目标：
    在相同实验条件下，使C4烯烃收率尽可能高。
    C4烯烃收率 = 乙醇转化率(%) × C4烯烃选择性(%) / 100

方法：
    1. 沿用Q2的岭回归模型（二次多项式特征 + 岭回归），
       分别预测乙醇转化率和C4烯烃选择性；
    2. 以收率为目标函数，用PSO在三维空间(X1, X2, X3)中搜索最优；
    3. 分两种场景：
        (a) 无温度限制；
        (b) 温度低于350℃。

PSO参数：
    - 粒子数: 10
    - 维度: 3 (X1, X2, X3)
    - 学习因子: c1 = c2 = 2
    - 惯性权重: w = 0.8
    - 最大迭代: 200
    - 初始位置: 从原始数据中随机选取10组
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RidgeCV

from data_process import load_and_process_data, X_COLS, Y_COLS
from pso_optimizer import PSO

import matplotlib.font_manager as fm
_preferred_fonts = ["Microsoft YaHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei"]
_available = {f.name for f in fm.fontManager.ttflist}
_font = next((f for f in _preferred_fonts if f in _available), None)
if _font:
    plt.rcParams["font.sans-serif"] = [_font]
plt.rcParams["axes.unicode_minus"] = False

# 路径设置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', '附件1.xlsx')
TABLE_DIR = os.path.join(SCRIPT_DIR, '..', 'tables')
FIGURE_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# =========================
# 主程序
# =========================
def main():
    print("=" * 70)
    print("Q3：催化剂组合与温度优化 — 粒子群优化(PSO)")
    print("=" * 70)

    # ---------- 1. 数据读取与特征构造 ----------
    print("\n[1/5] 数据读取与特征构造...")
    df = load_and_process_data(DATA_PATH)
    print(f"      数据总量: {len(df)} 条")
    print(f"      自变量: {X_COLS}")
    print(f"      因变量: {Y_COLS}")
    print(f"      收率范围: [{df['C4烯烃收率'].min():.4f}, {df['C4烯烃收率'].max():.4f}]")

    # ---------- 2. 训练岭回归模型（二次多项式） ----------
    print("\n[2/5] 训练岭回归预测模型...")

    X = df[X_COLS].values
    y_conv = df["乙醇转化率(%)"].values
    y_sel = df["C4烯烃选择性(%)"].values

    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X)

    alphas = np.logspace(-3, 1, 50)

    # 乙醇转化率模型
    ridge_conv = RidgeCV(alphas=alphas, cv=5)
    ridge_conv.fit(X_poly, y_conv)
    print(f"      乙醇转化率 — 最优alpha={ridge_conv.alpha_:.4f}, "
          f"训练R²={ridge_conv.score(X_poly, y_conv):.4f}")

    # C4烯烃选择性模型
    ridge_sel = RidgeCV(alphas=alphas, cv=5)
    ridge_sel.fit(X_poly, y_sel)
    print(f"      C4烯烃选择性 — 最优alpha={ridge_sel.alpha_:.4f}, "
          f"训练R²={ridge_sel.score(X_poly, y_sel):.4f}")

    # ---------- 3. 构造PSO目标函数 ----------
    print("\n[3/5] 构造PSO目标函数...")

    def objective_func(X_array):
        """
        目标函数：最小化负收率（等价于最大化收率）
        输入: (n_particles, 3) 的X数组
        输出: (n_particles,) 的函数值
        """
        X_poly_i = poly.transform(X_array)
        conv = ridge_conv.predict(X_poly_i)
        sel = ridge_sel.predict(X_poly_i)
        yield_rate = conv * sel / 100.0

        # 惩罚项：防止转化率/选择性超出合理范围[0, 100]
        penalty = 0.0
        penalty += np.sum(np.maximum(0, -conv) ** 2) * 100
        penalty += np.sum(np.maximum(0, conv - 100) ** 2) * 100
        penalty += np.sum(np.maximum(0, -sel) ** 2) * 100
        penalty += np.sum(np.maximum(0, sel - 100) ** 2) * 100

        return -(yield_rate - penalty)

    # 搜索边界（基于原始数据范围适当扩展）
    bounds = [
        (df["X1_Co质量比流速"].min() * 0.8, df["X1_Co质量比流速"].max() * 1.2),
        (df["X2_流速x总质量"].min() * 0.8, df["X2_流速x总质量"].max() * 1.2),
        (df["X3_温度"].min() * 0.9, df["X3_温度"].max() * 1.1),
    ]
    print(f"      X1范围: [{bounds[0][0]:.4f}, {bounds[0][1]:.4f}]")
    print(f"      X2范围: [{bounds[1][0]:.4f}, {bounds[1][1]:.4f}]")
    print(f"      X3(温度)范围: [{bounds[2][0]:.4f}, {bounds[2][1]:.4f}]")

    # ---------- 4. 场景(a): 无温度限制 ----------
    print("\n[4/5] 场景(a): 无温度限制，PSO优化中...")

    pso_a = PSO(
        objective_func=objective_func,
        dim=3,
        bounds=bounds,
        n_particles=10,
        max_iter=200,
        w=0.8,
        c1=2.0,
        c2=2.0,
        seed=2026
    )
    # 从全部原始数据中随机选10组初始化
    init_pool_a = df[X_COLS].values
    pso_a.initialize_from_data(init_pool_a)
    print(f"      初始粒子选自全部原始数据随机10组")
    print(f"      初始最优收率: {-pso_a.gbest_val:.4f}")

    gbest_a, gbest_val_a = pso_a.optimize()
    best_yield_a = -gbest_val_a
    X_poly_a = poly.transform(gbest_a.reshape(1, -1))
    best_conv_a = ridge_conv.predict(X_poly_a)[0]
    best_sel_a = ridge_sel.predict(X_poly_a)[0]

    print(f"      优化完成!")
    print(f"      ★ 最优收率 = {best_yield_a:.4f}")
    print(f"        对应转化率 = {best_conv_a:.4f}%")
    print(f"        对应选择性 = {best_sel_a:.4f}%")
    print(f"        X1 = {gbest_a[0]:.6f}")
    print(f"        X2 = {gbest_a[1]:.6f}")
    print(f"        X3(温度) = {gbest_a[2]:.2f}℃")

    # ---------- 5. 场景(b): 温度 < 350℃ ----------
    print("\n[5/5] 场景(b): 温度<350℃，PSO优化中...")

    bounds_b = [
        (df["X1_Co质量比流速"].min() * 0.8, df["X1_Co质量比流速"].max() * 1.2),
        (df["X2_流速x总质量"].min() * 0.8, df["X2_流速x总质量"].max() * 1.2),
        (df["X3_温度"].min() * 0.9, 350.0),  # 温度上限350
    ]

    pso_b = PSO(
        objective_func=objective_func,
        dim=3,
        bounds=bounds_b,
        n_particles=10,
        max_iter=200,
        w=0.8,
        c1=2.0,
        c2=2.0,
        seed=2026
    )
    # 从温度<350℃的原始数据中随机选10组初始化
    init_pool_b = df[df["X3_温度"] < 350][X_COLS].values
    pso_b.initialize_from_data(init_pool_b)
    print(f"      初始粒子选自温度<350℃原始数据随机10组")
    print(f"      初始最优收率: {-pso_b.gbest_val:.4f}")

    gbest_b, gbest_val_b = pso_b.optimize()
    best_yield_b = -gbest_val_b
    X_poly_b = poly.transform(gbest_b.reshape(1, -1))
    best_conv_b = ridge_conv.predict(X_poly_b)[0]
    best_sel_b = ridge_sel.predict(X_poly_b)[0]

    print(f"      优化完成!")
    print(f"      ★ 最优收率 = {best_yield_b:.4f}")
    print(f"        对应转化率 = {best_conv_b:.4f}%")
    print(f"        对应选择性 = {best_sel_b:.4f}%")
    print(f"        X1 = {gbest_b[0]:.6f}")
    print(f"        X2 = {gbest_b[1]:.6f}")
    print(f"        X3(温度) = {gbest_b[2]:.2f}℃")

    # ---------- 6. 结果汇总与保存 ----------
    print("\n" + "=" * 70)
    print("优化结果汇总")
    print("=" * 70)

    result_df = pd.DataFrame({
        "场景": ["无温度限制", "温度<350℃"],
        "最优收率": [best_yield_a, best_yield_b],
        "转化率(%)": [best_conv_a, best_conv_b],
        "选择性(%)": [best_sel_a, best_sel_b],
        "X1_Co质量比流速": [gbest_a[0], gbest_b[0]],
        "X2_流速x总质量": [gbest_a[1], gbest_b[1]],
        "X3_温度(℃)": [gbest_a[2], gbest_b[2]],
    })
    print(result_df.to_string(index=False))

    # 保存表格
    result_path = os.path.join(TABLE_DIR, "Q3_PSO优化结果汇总.xlsx")
    result_df.to_excel(result_path, index=False)
    print(f"\n[完成] 结果已保存: {result_path}")

    # 保存迭代历史
    hist_a = pso_a.get_history_df()
    hist_b = pso_b.get_history_df()
    hist_a["场景"] = "无温度限制"
    hist_b["场景"] = "温度<350℃"
    hist_all = pd.concat([hist_a, hist_b], ignore_index=True)
    hist_path = os.path.join(TABLE_DIR, "Q3_PSO迭代历史.xlsx")
    hist_all.to_excel(hist_path, index=False)
    print(f"[完成] 迭代历史已保存: {hist_path}")


    # ---------- 7. 可视化 ----------
    print("\n[可视化] 绘制PSO收敛曲线...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 场景(a)
    ax1 = axes[0]
    iters_a = hist_a["迭代次数"].values
    gbest_a_vals = [-v for v in hist_a["全局最优值"].values]  # 转回收率
    ax1.plot(iters_a, gbest_a_vals, color="#3274A1", linewidth=1.5)
    ax1.axhline(y=best_yield_a, color="#E1812C", linestyle="--",
                label=f"最优收率={best_yield_a:.2f}")
    ax1.set_xlabel("迭代次数", fontsize=11)
    ax1.set_ylabel("C4烯烃收率", fontsize=11)
    ax1.set_title("(a) 无温度限制 — PSO收敛曲线", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 场景(b)
    ax2 = axes[1]
    iters_b = hist_b["迭代次数"].values
    gbest_b_vals = [-v for v in hist_b["全局最优值"].values]
    ax2.plot(iters_b, gbest_b_vals, color="#6CBA47", linewidth=1.5)
    ax2.axhline(y=best_yield_b, color="#E1812C", linestyle="--",
                label=f"最优收率={best_yield_b:.2f}")
    ax2.set_xlabel("迭代次数", fontsize=11)
    ax2.set_ylabel("C4烯烃收率", fontsize=11)
    ax2.set_title("(b) 温度<350℃ — PSO收敛曲线", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIGURE_DIR, "Q3_PSO收敛曲线.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[完成] 收敛曲线已保存: {fig_path}")


    # ---------- 8. 最优粒子三维运动轨迹 ----------
    print("\n[可视化] 绘制最优粒子(gbest)三维运动轨迹...")

    fig = plt.figure(figsize=(14, 6))

    # 场景(a): 无温度限制
    ax1 = fig.add_subplot(121, projection='3d')
    traj_a = np.array([h['gbest_pos'] for h in pso_a.history])  # (200, 3)
    n_iter = len(traj_a)

    # 颜色渐变：深蓝(cool) → 红(hot)，表示迭代进程
    colors = plt.cm.coolwarm(np.linspace(0, 1, n_iter))

    for i in range(n_iter - 1):
        ax1.plot(traj_a[i:i+2, 0], traj_a[i:i+2, 1], traj_a[i:i+2, 2],
                 color=colors[i], linewidth=1.5, alpha=0.7)

    # 起点和终点标记
    ax1.scatter(*traj_a[0], color='#3274A1', s=80, marker='o',
                label='起点', edgecolors='black')
    ax1.scatter(*traj_a[-1], color='#E1812C', s=80, marker='s',
                label='终点', edgecolors='black')

    ax1.set_xlabel('X1: Co质量/流速', fontsize=9)
    ax1.set_ylabel('X2: 流速×总质量', fontsize=9)
    ax1.set_zlabel('X3: 温度(℃)', fontsize=9)
    ax1.set_title('(a) 无温度限制 — gbest三维轨迹', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)

    # 场景(b): 温度<350℃
    ax2 = fig.add_subplot(122, projection='3d')
    traj_b = np.array([h['gbest_pos'] for h in pso_b.history])
    n_iter_b = len(traj_b)
    colors_b = plt.cm.coolwarm(np.linspace(0, 1, n_iter_b))

    for i in range(n_iter_b - 1):
        ax2.plot(traj_b[i:i+2, 0], traj_b[i:i+2, 1], traj_b[i:i+2, 2],
                 color=colors_b[i], linewidth=1.5, alpha=0.7)

    ax2.scatter(*traj_b[0], color='#3274A1', s=80, marker='o',
                label='起点', edgecolors='black')
    ax2.scatter(*traj_b[-1], color='#E1812C', s=80, marker='s',
                label='终点', edgecolors='black')

    ax2.set_xlabel('X1: Co质量/流速', fontsize=9)
    ax2.set_ylabel('X2: 流速×总质量', fontsize=9)
    ax2.set_zlabel('X3: 温度(℃)', fontsize=9)
    ax2.set_title('(b) 温度<350℃ — gbest三维轨迹', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    traj_path = os.path.join(FIGURE_DIR, "Q3_gbest三维轨迹.png")
    fig.savefig(traj_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[完成] 三维轨迹图已保存: {traj_path}")


    print("\n" + "=" * 70)
    print("Q3 全部完成")
    print("=" * 70)


if __name__ == "__main__":
    main()