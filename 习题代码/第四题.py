import numpy as np
import matplotlib.pyplot as plt

# 神经元群体与观测参数
N = 50
s_true = 90.0  # 真实的外部物理刺激角度
s_a = np.linspace(0, 180, N)
sigma = 15.0
T = 1.0  
trials = 1000

def tuning_curve(s, sa, sig):
    return np.exp(-((s - sa)**2) / (2 * sig**2))

# ==========================================
# 1. 计算理论 Fisher 信息量与 CRLB 下界
# ==========================================
# (1) 离散求和的精确 CRLB (考虑了 N=50 的空间离散性)
f_a_vals = tuning_curve(s_true, s_a, sigma)
J_s_discrete = T * np.sum(((s_true - s_a)**2 / sigma**4) * f_a_vals)
CRLB_discrete = 1.0 / J_s_discrete

# (2) 连续积分近似的 CRLB 
CRLB_continuous = (180.0 * sigma) / (T * N * np.sqrt(2 * np.pi))

print(f"理论 CRLB 方差下界 (离散求和): {CRLB_discrete:.4f}")
print(f"理论 CRLB 方差下界 (连续积分): {CRLB_continuous:.4f}")

# ==========================================
# 2. 蒙特卡洛随机采样解码
# ==========================================
estimates = np.zeros(trials)
for i in range(trials):
    lambdas = T * tuning_curve(s_true, s_a, sigma)
    k_a = np.random.poisson(lambdas) # 独立泊松过程
    
    if np.sum(k_a) > 0:
        # 使用质心公式 (近似 MVUB) 进行估计
        estimates[i] = np.sum(k_a * s_a) / np.sum(k_a)
    else:
        estimates[i] = s_true 

sim_variance = np.var(estimates)
print(f"数值模拟经验估计方差 (蒙特卡洛): {sim_variance:.4f}")

# ==========================================
# 3. 绘制估计量的概率密度分布直方图
# ==========================================
plt.figure(figsize=(8, 5))
plt.hist(estimates, bins=30, density=True, alpha=0.7, color='green', edgecolor='black')
plt.axvline(s_true, color='red', linestyle='dashed', linewidth=2, label=f'True Stimulus Angle (s={s_true})')
plt.title('Probability Density of MVUB Estimated Angle')
plt.xlabel('Estimated Angle (degrees)')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
