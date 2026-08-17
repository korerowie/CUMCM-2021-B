# =========================
# Q3：粒子群优化(PSO)算法
# =========================
"""
参数设置：
    - 粒子数: 10
    - 维度: 3 (X1, X2, X3)
    - 学习因子: c1 = c2 = 2
    - 惯性权重: w = 0.8（常数）
    - 最大迭代次数: 200
    - 初始位置: 从原始数据中随机选取10组

PSO更新公式:
    V_{id}^{k+1} = w·V_{id}^k + c1·r1·(P_{id}^k - X_{id}^k) + c2·r2·(P_{gd}^k - X_{id}^k)
    X_{id}^{k+1} = X_{id}^k + V_{id}^{k+1}

其中:
    P_{id}: 粒子i自身历史最优位置的第d维
    P_{gd}: 整个种群当前最优位置的第d维
    r1, r2: [0,1]内均匀分布的随机数
"""

import numpy as np


class PSO:
    """
    粒子群优化器
    """

    def __init__(self, objective_func, dim, bounds, n_particles=10, max_iter=200,
                 w=0.8, c1=2.0, c2=2.0, seed=2026):
        """
        Parameters
        ----------
        objective_func : callable
            目标函数，输入形状 (n_particles, dim) 的数组，
            输出形状 (n_particles,) 的函数值（PSO最小化该值）
        dim : int
            搜索空间维度
        bounds : list of tuple
            每维的搜索边界 [(low1, high1), (low2, high2), ...]
        n_particles : int
            粒子数量
        max_iter : int
            最大迭代次数
        w : float
            惯性权重
        c1, c2 : float
            个体学习因子、社会学习因子
        seed : int
            随机种子，保证结果可复现
        """
        self.obj_func = objective_func
        self.dim = dim
        self.bounds = np.array(bounds, dtype=float)  # (dim, 2)
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.rng = np.random.RandomState(seed)

        # 速度限制：每维搜索范围的20%
        range_per_dim = self.bounds[:, 1] - self.bounds[:, 0]
        self.v_max = 0.2 * range_per_dim
        self.v_min = -self.v_max

        # 状态变量
        self.X = None          # 位置 (n_particles, dim)
        self.V = None          # 速度 (n_particles, dim)
        self.pbest = None      # 个体最优位置 (n_particles, dim)
        self.pbest_val = None  # 个体最优值 (n_particles,)
        self.gbest = None      # 全局最优位置 (dim,)
        self.gbest_val = None  # 全局最优值
        self.history = []      # 迭代历史记录

    def initialize_from_data(self, data_points):
        """
        从给定数据点中随机选取初始化粒子位置

        Parameters
        ----------
        data_points : ndarray, shape (n_data, dim)
            候选初始位置池（如原始实验数据）
        """
        n_data = len(data_points)
        if n_data >= self.n_particles:
            idx = self.rng.choice(n_data, self.n_particles, replace=False)
        else:
            idx = self.rng.choice(n_data, self.n_particles, replace=True)
        self.X = data_points[idx].copy().astype(float)

        # 速度随机初始化
        self.V = self.rng.uniform(self.v_min, self.v_max,
                                  (self.n_particles, self.dim))

        # 评估初始位置
        vals = self.obj_func(self.X)
        self.pbest = self.X.copy()
        self.pbest_val = vals.copy()

        gbest_idx = np.argmin(self.pbest_val)
        self.gbest = self.pbest[gbest_idx].copy()
        self.gbest_val = self.pbest_val[gbest_idx]

    def optimize(self):
        """
        执行PSO优化，返回 (全局最优位置, 全局最优值)
        """
        for t in range(self.max_iter):
            # 随机数
            r1 = self.rng.rand(self.n_particles, self.dim)
            r2 = self.rng.rand(self.n_particles, self.dim)

            # 速度更新
            self.V = (self.w * self.V
                      + self.c1 * r1 * (self.pbest - self.X)
                      + self.c2 * r2 * (self.gbest - self.X))

            # 速度限制
            self.V = np.clip(self.V, self.v_min, self.v_max)

            # 位置更新
            self.X = self.X + self.V

            # 边界处理：越界则拉回边界，速度反向衰减
            for d in range(self.dim):
                mask_low = self.X[:, d] < self.bounds[d, 0]
                mask_high = self.X[:, d] > self.bounds[d, 1]
                self.X[mask_low, d] = self.bounds[d, 0]
                self.V[mask_low, d] = -self.V[mask_low, d] * 0.5
                self.X[mask_high, d] = self.bounds[d, 1]
                self.V[mask_high, d] = -self.V[mask_high, d] * 0.5

            # 评估新位置
            vals = self.obj_func(self.X)

            # 更新个体最优
            improved = vals < self.pbest_val
            self.pbest[improved] = self.X[improved].copy()
            self.pbest_val[improved] = vals[improved]

            # 更新全局最优
            gbest_idx = np.argmin(self.pbest_val)
            if self.pbest_val[gbest_idx] < self.gbest_val:
                self.gbest = self.pbest[gbest_idx].copy()
                self.gbest_val = self.pbest_val[gbest_idx]

            # 记录历史
            self.history.append({
                'iter': t + 1,
                'gbest_val': float(self.gbest_val),
                'gbest_pos': self.gbest.copy(),
                'mean_val': float(np.mean(vals)),
            })

        return self.gbest.copy(), float(self.gbest_val)

    def get_history_df(self):
        # 返回迭代历史为DataFrame
        import pandas as pd
        records = []
        for h in self.history:
            records.append({
                '迭代次数': h['iter'],
                '全局最优值': h['gbest_val'],
                '平均目标值': h['mean_val'],
            })
        return pd.DataFrame(records)