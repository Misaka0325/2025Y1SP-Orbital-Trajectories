import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anm

G = 6.67430e-11  # 引力常数 (m³ kg⁻¹ s⁻²)
# 太阳和地球的近似值
m1 = 1.989e30    # 太阳质量(kg)
m2 = 5.972e24    # 地球质量(kg)

# 初始条件
r_initial = 1.496e11  # 地球到太阳的平均距离(m)
v_initial = 2.98e4    # 地球轨道速度(m/s)

# 模拟参数
dt = 3600 * 6       # 时间步长 (秒) - 6小时
num_steps = 365 * 5  # 总步数 (5年)

# 初始化数组
x = np.zeros(num_steps)
y = np.zeros(num_steps)
vx = np.zeros(num_steps)
vy = np.zeros(num_steps)

# 设置初始条件 (行星在x轴上，初始速度在y方向)
x[0] = r_initial
y[0] = 0.0
vx[0] = 0.0
vy[0] = v_initial

# 给动画部分统计边界条件
xmax = x[0]
ymax = y[0]

# 欧拉法数值积分
for i in range(num_steps - 1):
    # 计算当前位置到恒星的距离
    r = np.sqrt(x[i]**2 + y[i]**2)
    
    # 计算引力加速度 (牛顿万有引力定律)
    ax = -G * m1 * x[i] / r**3
    ay = -G * m1 * y[i] / r**3
    
    # 欧拉法更新速度和位置
    vx[i+1] = vx[i] + ax * dt
    vy[i+1] = vy[i] + ay * dt
    x[i+1] = x[i] + vx[i] * dt
    y[i+1] = y[i] + vy[i] * dt

    #更新边界
    if abs(x[i+1]) > xmax:
        xmax = abs(x[i+1])
    if abs(y[i+1]) > ymax:
        ymax = abs(y[i+1])

# 计算能量以验证精度
def calculate_energy(x, y, vx, vy):
    """计算系统总能量"""
    r = np.sqrt(x**2 + y**2)
    v_sq = vx**2 + vy**2
    kinetic = 0.5 * m2 * v_sq
    potential = -G * m1 * m2 / r
    return kinetic + potential

# 初始能量
E0 = calculate_energy(x[0], y[0], vx[0], vy[0])
# 所有时间步的能量
E = calculate_energy(x, y, vx, vy)
# 相对能量误差
energy_error = np.abs((E - E0) / E0)

# 创建图形
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
plt.legend()

# 2. 能量守恒验证
plt.subplot(1, 2, 2)
plt.plot(np.arange(num_steps) * dt / (3600 * 24), energy_error, 'r-')
plt.yscale('log')
plt.xlabel('time (days)')
plt.ylabel('relative energy error')
plt.title('energy conservation error')
plt.grid(True)

plt.tight_layout()
plt.savefig('Eu_sun-earth.png', dpi=300)

# 动画
fig = plt.figure() #创建一个新的图形（Figure）对象，作为整个图表的容器
ax = fig.add_subplot(1, 1, 1) #在图形中添加一个子图（Axes）。参数(1, 1, 1)表示创建1x1网格中的第1个（也是唯一一个）子图
line, = plt.plot([], [], "r-", animated=True) #创建一个空的线条对象 [], []：初始x和y数据为空
x = []
y = [] # 初始化两个空列表，用于存储动画中不断增加的x和y坐标数据

def init():
    ax.set_xlim(-xmax, xmax) # 设置x轴范围
    ax.set_ylim(-ymax, ymax) # 设置y轴范围
    return line,

'''
def update(frame): #动画更新函数（每帧调用） frame：从frames参数传入的当前帧值
    x.append(frame) # 将当前帧值添加到x列表
    y.append(np.sin(frame))
    line.set_data(x, y) # 更新线条数据
    return line,
'''
def update(n): #第n帧
    x.append(x[n]) # 将当前帧值添加到x列表
    y.append(y[n])
    line.set_data(x, y) # 更新线条数据    
    #line.set_data(x[:n+1], y[:n+1]) #显示n+1个点（左闭右开区间
    return line,

ani = anm.FuncAnimation(fig # 使用创建的Figure对象
    ,update # 指定更新函数
    ,frames= num_steps # 总帧数
    ,interval=10 # 帧间隔10毫秒（展示帧率
    ,init_func=init # 指定初始化函数
    ,blit=True  #使用blitting优化（只重绘变化部分）
    )

plt.show()
ani.save("animation.gif", fps=25, writer="imagemagick")