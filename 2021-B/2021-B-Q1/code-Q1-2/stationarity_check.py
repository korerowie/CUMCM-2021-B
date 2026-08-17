# =========================
# Q1-2-3：AR模型平稳性验证（特征根检验）（C4烯烃选择性）
# =========================
"""
通过AR特征方程的根判断模型是否满足平稳性条件。
平稳性条件：AR(p)模型的特征方程 Φ(B) = 1 - φ₁·B - φ₂·B² - ... - φ_p·B^p = 0
的所有根都在单位圆外（|B| > 1）。
"""

import numpy as np
from statsmodels.tsa.ar_model import AutoReg
import warnings
warnings.filterwarnings('ignore')

# C4烯烃选择性数据
c4_selectivity = np.array([39.9, 38.55, 36.72, 39.53, 38.96, 40.32, 39.04])

print("=" * 70)
print("程序3: C4烯烃选择性 — AR模型平稳性验证（特征根检验）")
print("=" * 70)

for p in [1, 2]:
    print("\n" + "-" * 70)
    print(f"AR({p}) 特征根分析:")
    print("-" * 70)
    
    try:
        model = AutoReg(c4_selectivity, lags=p, trend='c')
        fitted = model.fit()
        
        # 获取参数值
        if hasattr(fitted.params, 'values'):
            param_values = fitted.params.values
        else:
            param_values = fitted.params
        
        phi = param_values[1:]  # φ系数（去掉常数项）
        
        # 特征多项式: Φ(B) = 1 - φ₁·B - φ₂·B² - ... - φ_p·B^p = 0
        # 改写为: -φ_p·B^p - φ_{p-1}·B^{p-1} - ... - φ₁·B + 1 = 0
        # np.roots 需要系数从高次到低次: [-φ_p, -φ_{p-1}, ..., -φ₁, 1]
        poly_coeffs = list(-phi[::-1]) + [1]
        roots = np.roots(poly_coeffs)
        
        print(f"  φ系数: {phi}")
        print(f"  特征多项式: 1", end="")
        for i, ph in enumerate(phi, 1):
            sign = "-" if ph > 0 else "+"
            print(f" {sign} {abs(ph):.6f}·B^{i}", end="")
        print(" = 0")
        
        print(f"\n  特征根:")
        all_outside = True
        for i, root in enumerate(roots, 1):
            modulus = np.abs(root)
            status = "✓ 单位圆外（平稳）" if modulus > 1 else "✗ 单位圆内/上（非平稳）"
            if modulus <= 1:
                all_outside = False
            print(f"    根{i} = {root:.6f}, 模 = {modulus:.6f}  {status}")
        
        print(f"\n  平稳性结论: ", end="")
        if all_outside:
            print(f"AR({p})模型满足平稳性条件（所有特征根在单位圆外）")
        else:
            print(f"AR({p})模型不满足平稳性条件")
            
        # AR(1)显式条件
        if p == 1:
            print(f"\n  AR(1)显式平稳条件检验:")
            print(f"    |φ₁| = {abs(phi[0]):.6f}")
            if abs(phi[0]) < 1:
                print(f"    |φ₁| < 1, 满足平稳性条件 ✓")
            else:
                print(f"    |φ₁| ≥ 1, 不满足平稳性条件")
                
        # AR(2)显式条件
        if p == 2:
            print(f"\n  AR(2)显式平稳条件检验:")
            print(f"    条件1: φ₂ + φ₁ < 1  →  {phi[1] + phi[0]:.6f} < 1 ? {'是 ✓' if phi[1] + phi[0] < 1 else '否'}")
            print(f"    条件2: φ₂ - φ₁ < 1  →  {phi[1] - phi[0]:.6f} < 1 ? {'是 ✓' if phi[1] - phi[0] < 1 else '否'}")
            print(f"    条件3: |φ₂| < 1     →  {abs(phi[1]):.6f} < 1 ? {'是 ✓' if abs(phi[1]) < 1 else '否'}")
            cond1 = phi[1] + phi[0] < 1
            cond2 = phi[1] - phi[0] < 1
            cond3 = abs(phi[1]) < 1
            if cond1 and cond2 and cond3:
                print(f"    三个条件均满足，AR(2)平稳 ✓")
            else:
                print(f"    有条件不满足，AR(2)非平稳")
                
    except Exception as e:
        print(f"  AR({p})分析失败: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("说明:")
print("  - 特征根检验是判断AR模型平稳性的标准方法")
print("  - 平稳性要求所有特征根都在复平面的单位圆之外（|B| > 1）")
print("  - 小样本下系数估计有误差，平稳性结论需谨慎解读")
print("=" * 70)