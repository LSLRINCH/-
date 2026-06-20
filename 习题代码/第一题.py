import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import erfcx  # <-- 引入专门解决指数溢出的函数
import math

# 常量定义
gL, Vth, Vr, tau_ref, tau_syn, gsyn = 0.1, 20.0, 0.0, 2.0, 3.0, 1.0
m_target, v_target = 30.0, 100.0  # 目标统计量

def D_minus(u):
    # 使用 erfcx(-u) 完美替代原先的 exp(u**2)*(1+erf(u))，根治积分发散问题！
    return (np.sqrt(np.pi) / 2) * erfcx(-u)

def theoretical_moments(mu, sigma):
    a = gL; b = gsyn * mu; c = gsyn * sigma
    u_th = (Vth - b/a) / (c / np.sqrt(a))
    u_r = (Vr - b/a) / (c / np.sqrt(a))
    
    # 加入 limit=200 提升复杂积分的容错度
    mean_T0, _ = quad(D_minus, u_r, u_th, limit=200)
    mean_T0 *= (2.0 / a)
    
    def var_integrand(u):
        val, _ = quad(lambda v: np.exp(-v**2) * D_minus(v)**2, -10, u, limit=200) 
        return np.exp(u**2) * val
        
    var_T0, _ = quad(var_integrand, u_r, u_th, limit=200)
    var_T0 *= (8.0 / a**2)
    return mean_T0 + tau_ref, var_T0

def objective(params):
    mu, sigma = params
    if sigma <= 0.1: return 1e6 # 边界条件约束
    try:
        m_calc, v_calc = theoretical_moments(mu, sigma)
        return ((m_calc - m_target)/m_target)**2 + ((v_calc - v_target)/v_target)**2
    except:
        return 1e6

# 1. 理论求解最优参数
res = minimize(objective, [2.5, 1.0], method='Nelder-Mead')
best_mu, best_sigma = res.x
print(f"优化求解参数: mu = {best_mu:.3f}, sigma = {best_sigma:.3f}")

# 2. SDE 纯白噪声近似数值验证 (严格对齐Siegert理论公式)
# ==========================================
dt = 0.01  
sqrt_dt = math.sqrt(dt)

def simulate_isi_white_noise(mu, sigma, N_isi=2000): # 样本数提高到 2000，结果会极度稳定
    V = Vr
    t, last_spike, t_ref_end = 0.0, 0.0, 0.0
    isi = []
    
    # 初始分配一个大数组（足够跑几千毫秒）
    dW_array = np.random.normal(0, 1, size=2000000) * sqrt_dt
    step = 0
    
    # 移除 step < len(dW_array) 限制，强制收集够足够的样本
    while len(isi) < N_isi + 50:
        if step >= len(dW_array):
            dW_array = np.random.normal(0, 1, size=2000000) * sqrt_dt
            step = 0
            
        dW = dW_array[step]
        t += dt
        step += 1
        
        # 绝对不应期判定
        if t < t_ref_end:
            V = Vr
            continue
            
        # 核心修改：1D 纯白噪声扩散过程
        V += (-gL * V + gsyn * mu) * dt + gsyn * sigma * dW
        
        # 达到阈值放电
        if V >= Vth:
            if last_spike > 0:
                isi.append(t - last_spike)
            last_spike = t
            t_ref_end = t + tau_ref
            V = Vr
            
    return isi[50:50+N_isi] 

# 运行验证
print("\n[开始进行 SDE 大样本蒙特卡洛验证... (约需几秒钟)]")
isi_final = simulate_isi_white_noise(best_mu, best_sigma, N_isi=2000)
m_final, v_final = np.mean(isi_final), np.var(isi_final)

print(f"均值 = {m_final:.2f} ms (理论目标 {m_target})")
print(f"方差 = {v_final:.2f} ms^2 (理论目标 {v_target})")
