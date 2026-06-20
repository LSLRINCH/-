import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 系统参数
tau_E = 10.0
M_EE = 2.0; M_EI = 2.0; M_IE = 2.0; M_II = 1.0
h_E = 1.0; h_I = 1.0

# 理论分叉边界计算
tau_I_critical = tau_E * (M_II + 1) / (M_EE - 1)
print(f"理论计算的临界 tau_I = {tau_I_critical}")

def relu(x):
    return np.maximum(0, x)

def network_dynamics(y, t, tau_I, h_E):
    v_E, v_I = y
    dvE_dt = (-v_E + relu(M_EE * v_E - M_EI * v_I + h_E)) / tau_E
    dvI_dt = (-v_I + relu(M_IE * v_E - M_II * v_I + h_I)) / tau_I
    return [dvE_dt, dvI_dt]

t = np.linspace(0, 500, 5000)
y0 = [0.1, 0.1]

# 模拟分叉前后的状态演化
tau_I_stable = 15.0  # < critical
tau_I_osc = 25.0     # > critical

sol_stable = odeint(network_dynamics, y0, t, args=(tau_I_stable, h_E))
sol_osc = odeint(network_dynamics, y0, t, args=(tau_I_osc, h_E))

# 绘制结果图
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t, sol_stable[:, 0], label='v_E (Excitatory)', color='red')
plt.plot(t, sol_stable[:, 1], label='v_I (Inhibitory)', color='blue')
plt.title(f'Stable State (tau_I = {tau_I_stable})')
plt.xlabel('Time'); plt.ylabel('Firing Rate'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t, sol_osc[:, 0], label='v_E (Excitatory)', color='red')
plt.plot(t, sol_osc[:, 1], label='v_I (Inhibitory)', color='blue')
plt.title(f'Limit Cycle Oscillation (tau_I = {tau_I_osc})')
plt.xlabel('Time'); plt.ylabel('Firing Rate'); plt.legend()
plt.tight_layout()
plt.show()
