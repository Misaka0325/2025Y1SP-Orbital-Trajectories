import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anm
from collections import deque

# 物理常数
G = 6.67430e-11  # 引力常数 (m³ kg⁻¹ s⁻²)

# 天体参数
m1 = 1.989e30    # 太阳质量(kg)
m2 = 5.972e24    # 地球质量(kg)

# 初始条件
r_initial = 1.496e11  # 初始距离 (m)
v_initial = 2.98e4    # 初始切向速度 (m/s)  地球轨道速度

# 模拟参数
dt = 3600 * 24        # 时间步长 (s)
num_steps = 365 * 4   # 总步数 (年)

# 初始化数组 (相同)
x = np.zeros(num_steps)
y = np.zeros(num_steps)
vx = np.zeros(num_steps)
vy = np.zeros(num_steps)

# 设置初始条件 (相同)
x[0] = r_initial
y[0] = 0.0
vx[0] = 0.0
vy[0] = v_initial

# 给动画部分统计边界条件（相同）
xmax = x[0]
ymax = y[0]
# 动画播放参数
frame_interv = 10 # 帧率（毫秒）

# 定义运动方程函数
def f(state, t):
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)
    ax = -G * m1 * x / r**3
    ay = -G * m1 * y / r**3
    return np.array([vx, vy, ax, ay])

# Leap-frog法数值积分

# 初始条件：在轨道前一步的速度
r_start = np.sqrt(x[0]**2 + y[0]**2)
ax_start = -G * m1 * x[0] / r_start**3
ay_start = -G * m1 * y[0] / r_start**3
vx_half = vx[0] + 0.5*ax_start * dt
vy_half = vy[0] + 0.5*ay_start * dt


for i in range(num_steps - 1):
    current_state = np.array([x[i], y[i], vx[i], vy[i]])
    r = np.sqrt(x[i]**2 + y[i]**2) # 更新当前位置

    # 中心差分
    x[i+1] = x[i] + vx_half * dt
    y[i+1] = y[i] + vy_half * dt
    ax = -G * m1 * x[i+1] / r**3
    ay = -G * m1 * y[i+1] / r**3
    vx_half += ax * dt
    vy_half += ay * dt

    #更新边界
    if abs(x[i+1]) > xmax:
        xmax = abs(x[i+1])
    if abs(y[i+1]) > ymax:
        ymax = abs(y[i+1])

# 计算能量函数 (相同)
def calculate_energy(x, y, vx, vy):
    r = np.sqrt(x**2 + y**2)
    v_sq = vx**2 + vy**2
    kinetic = 0.5 * m2 * v_sq
    potential = -G * m1 * m2 / r
    return kinetic + potential

# 初始能量能量相对误差 (相同)
E0 = calculate_energy(x[0], y[0], vx[0], vy[0])
# 所有时间步的能量
E = calculate_energy(x, y, vx, vy)
energy_error = np.abs((E - E0) / E0)

# 创建图形 (相同)
plt.figure(figsize=(15, 10))

# 1. 行星轨道
plt.subplot(1, 2, 1)
plt.plot(x, y, 'b-', linewidth=1)
plt.plot(0, 0, 'yo', markersize=15, label=f'm1={m1:.2e} kg)')
plt.plot(x[0], y[0], 'go', markersize=5, label=f'initial position')
plt.plot(x[-1], y[-1], 'ro', markersize=5, label=f'final position')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title(f'Euler Orbital Trajectory (m2={m2:.2e} kg)\n step length={dt/3600:.1f}hours, total time={num_steps*dt/(3600*24*365):.1f}years')
plt.axis('equal')
plt.grid(True)
plt.legend(loc = "upper right")

# 2. 能量守恒验证
plt.subplot(1, 2, 2)
plt.plot(np.arange(num_steps) * dt / (3600 * 24), energy_error, 'r-')
plt.yscale('log')
plt.xlabel('time (days)')
plt.ylabel('relative energy error')
plt.title('energy conservation error')
plt.grid(True)

plt.tight_layout()
plt.savefig('LF_sun-earth.png', dpi=300)

# 动画

fig = plt.figure() #创建一个新的图形（Figure）对象，作为整个图表的容器
ax = fig.add_subplot(1, 1, 1) #在图形中添加一个子图（Axes）。参数(1, 1, 1)表示创建1x1网格中的第1个（也是唯一一个）子图
line, = plt.plot([], [], "r-", animated=True) #创建一个空的线条对象 [], []：初始x和y数据为空
x_anim = []
y_anim = [] # 初始化两个空列表，用于存储动画中不断增加的x和y坐标数据

def init():
    ax.set_xlim(-xmax * 1.1, xmax * 1.1) # 设置x轴范围
    ax.set_ylim(-ymax * 1.1, ymax * 1.1) # 设置y轴范围
    return line,


# 拖尾轨迹：定义轨迹长度（显示最近的点数）
trail_length = 100
# 初始化动画轨迹列表
x_anim = deque(maxlen=trail_length)  # 使用双端队列限制长度
y_anim = deque(maxlen=trail_length)

def update(n): #第n帧
    x_anim.append(x[n]) # 将当前帧值添加到x列表
    y_anim.append(y[n])
    line.set_data(x_anim, y_anim) # 更新线条数据    
    #line.set_data(x[:n+1], y[:n+1]) #显示n+1个点（左闭右开区间
    return line,

ani = anm.FuncAnimation(fig # 使用创建的Figure对象
    ,update # 指定更新函数
    ,frames= num_steps # 总帧数
    ,interval=frame_interv # 帧间隔10毫秒（展示帧率
    ,init_func=init # 指定初始化函数
    ,blit=True  #使用blitting优化（只重绘变化部分）
    ,repeat=False
    )

plt.plot(0, 0, 'yo', markersize=15, label=f'm1={m1:.2e} kg)')
plt.plot(x[0], y[0], 'go', markersize=5, label=f'initial position')
#plt.plot(x[-1], y[-1], 'ro', markersize=5, label=f'final position')
plt.grid(True)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Cartesian coordinate position')
plt.legend()
plt.show()
ani.save("2LF_anime.gif", fps=20, writer="imagemagick")