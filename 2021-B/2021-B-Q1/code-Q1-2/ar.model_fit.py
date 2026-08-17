# =========================
# Q1-2-2：AR模型拟合与AIC选阶（C4烯烃选择性）
# =========================
"""
C4烯烃选择性原始序列经一阶差分检验判定为近似平稳，
直接对原始序列拟合AR(1)和AR(2)，按AIC选最优阶数。

说明：
- 样本量n=7，仅尝试AR(1)和AR(2)，更高阶数无意义
- 使用statsmodels的AutoReg进行拟合
- 不等间隔按先后顺序近似处理
"""

import numpy as np
from statsmodels.tsa.ar_model import AutoReg
import warnings
warnings.filterwarnings('ignore')

# 数据
c4_selectivity = np.array([39.9, 38.55, 36.72, 39.53, 38.96, 40.32, 39.04])

print("=" * 70)
print("程序2: C4烯烃选择性 — AR模型拟合与AIC选阶")
print("=" * 70)
print(f"\n原始序列: {c4_selectivity}")
print(f"样本量 n = {len(c4_selectivity)}")
print(f"序列均值 = {np.mean(c4_selectivity):.4f}")
print(f"序列标准差 = {np.std(c4_selectivity, ddof=1):.4f}")

results = {}

# 分别拟合AR(1)和AR(2)
for p in [1, 2]:
    print("\n" + "-" * 70)
    print(f"AR({p}) 模型拟合结果:")
    print("-" * 70)
    
    try:
        model = AutoReg(c4_selectivity, lags=p, trend='c')
        fitted = model.fit()
        results[p] = fitted
        
        print(f"  模型形式: X(t) = c + φ₁·X(t-1)" + (" + φ₂·X(t-2)" if p == 2 else "") + " + ε(t)")
        print(f"\n  系数估计:")
        
        # 兼容处理参数名
        if hasattr(fitted.params, 'index'):
            param_names = fitted.params.index
            param_values = fitted.params.values
        else:
            param_names = fitted.param_names if hasattr(fitted, 'param_names') else [f'param_{i}' for i in range(len(fitted.params))]
            param_values = fitted.params
        
        if hasattr(fitted, 'bse') and fitted.bse is not None:
            if hasattr(fitted.bse, 'values'):
                bse_values = fitted.bse.values
            else:
                bse_values = fitted.bse
        else:
            bse_values = [None] * len(param_values)
        
        for name, val, se in zip(param_names, param_values, bse_values):
            se_str = f"(标准误: {se:.6f})" if se is not None else ""
            print(f"    {name:<12} = {val:>10.6f}  {se_str}")
        
        # 关键统计量
        print(f"\n  模型评价:")
        print(f"    AIC        = {fitted.aic:.4f}")
        print(f"    BIC        = {fitted.bic:.4f}")
        print(f"    Log-Likelihood = {fitted.llf:.4f}")
        print(f"    残差标准差  = {np.std(fitted.resid, ddof=p+1):.6f}")
        
        # φ系数
        phi = param_values[1:]  # 去掉常数项
        print(f"\n  φ系数:")
        for i, ph in enumerate(phi, 1):
            print(f"    φ_{i} = {ph:.6f}")
            
    except Exception as e:
        print(f"  AR({p})拟合失败: {e}")
        results[p] = None

# AIC选阶
print("\n" + "=" * 70)
print("AIC选阶结果:")
print("=" * 70)

aic_values = {p: results[p].aic for p in [1, 2] if results[p] is not None}
if aic_values:
    best_p = min(aic_values, key=aic_values.get)
    print(f"\n  AR(1) AIC = {aic_values[1]:.4f}")
    print(f"  AR(2) AIC = {aic_values[2]:.4f}")
    print(f"\n  ★ 最优模型: AR({best_p}) (AIC最小)")
    
    best_model = results[best_p]
    
    # 重新获取参数
    if hasattr(best_model.params, 'index'):
        param_names = best_model.params.index
        param_values = best_model.params.values
    else:
        param_names = best_model.param_names if hasattr(best_model, 'param_names') else [f'param_{i}' for i in range(len(best_model.params))]
        param_values = best_model.params
    
    print(f"\n  最优模型系数:")
    for name, val in zip(param_names, param_values):
        print(f"    {name:<12} = {val:.6f}")
    
    phi_best = param_values[1:]
    print(f"\n  最优模型φ系数:")
    for i, ph in enumerate(phi_best, 1):
        print(f"    φ_{i} = {ph:.6f}")
    
    # 保存最优模型结果供后续程序使用
    print(f"\n  最优模型方程:")
    c = param_values[0]
    if best_p == 1:
        print(f"    X(t) = {c:.6f} + {phi_best[0]:.6f}·X(t-1) + ε(t)")
    else:
        print(f"    X(t) = {c:.6f} + {phi_best[0]:.6f}·X(t-1) + {phi_best[1]:.6f}·X(t-2) + ε(t)")
else:
    print("  模型拟合失败，无法选阶")

print("\n" + "=" * 70)
print("说明:")
print("  - 由于n=7，AR(2)只有5个自由度，参数估计可靠性有限")
print("  - 所有结果需在论文中注明'基于小样本的近似分析'")
print("  - 不等间隔问题：未代入具体时间间隔，仅按先后顺序处理")
print("=" * 70)