import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib.animation as anm
from matplotlib.animation import PillowWriter
from PIL import Image

G = 1.0  # gravitational constant
m1 = m2 = 1.0

def two_body_derivatives(t, y):
    r1 = y[0:2]
    v1 = y[2:4]
    r2 = y[4:6]
    v2 = y[6:8]

    r12 = r2 - r1
    dist = np.linalg.norm(r12)
    a1 = G * m2 * r12 / dist**3
    a2 = -G * m1 * r12 / dist**3

    return np.concatenate([v1, a1, v2, a2])

def two_body_energy(y):
    r1, v1 = y[:2], y[2:4]
    r2, v2 = y[4:6], y[6:8]
    KE = 0.5 * m1 * np.dot(v1, v1) + 0.5 * m2 * np.dot(v2, v2)
    PE = -G * m1 * m2 / np.linalg.norm(r1 - r2)
    return KE + PE

def shoot_residual_two_body(guess):
    T = guess[-1]
    y0 = guess[:-1]

    sol = solve_ivp(two_body_derivatives, [0, T], y0, t_eval=[0, T], rtol=1e-12, atol=1e-14)
    if not sol.success:
        return np.ones_like(y0) * 1e6
    yT = sol.y[:, -1]
    return np.append(yT - y0, 0.0)  # pad to match 9 variables

# Ideal circular orbit
a = 1.0
# Orbital speed for circular motion in reduced units:
# v = sqrt(G * M / r), here M = m_total/2, r = a
v = np.sqrt(0.5)  # Since a = 1, G = 1, m = 1

r1 = np.array([0, 0])
v1 = np.array([0, 0])
r2 = np.array([a, 0])
v2 = np.array([0, -v])

y0 = np.concatenate([r1, v1, r2, v2])
T_guess = 2 * np.pi

guess = np.concatenate([y0, [T_guess]])

sol = root(shoot_residual_two_body, guess, method='lm', tol=1e-10)


if sol.success:
    print("Shooting converged.")
    y0_corrected = sol.x[:-1]
    T_corrected = sol.x[-1]

    E0 = two_body_energy(y0_corrected)
    sol2 = solve_ivp(two_body_derivatives, [0, T_corrected], y0_corrected, t_eval=np.linspace(0, T_corrected, 500),
                     rtol=1e-12, atol=1e-14)
    r1 = sol2.y[0:2].T
    r2 = sol2.y[4:6].T

    
    # Extract start and end positions
    r1_start = sol2.y[0:2, 0]
    r2_start = sol2.y[4:6, 0]
    r1_end = sol2.y[0:2, -1]
    r2_end = sol2.y[4:6, -1]

    # Plot starting positions (green)
    plt.scatter([r1_start[0], r2_start[0]],
                [r1_start[1], r2_start[1]],
                color='green', label='Start', zorder=5)

    # Plot ending positions (red X)
    plt.scatter([r1_end[0], r2_end[0]],
                [r1_end[1], r2_end[1]],
                color='red', marker='x', label='End', zorder=6)

    plt.plot(r1[:, 0], r1[:, 1], label="Body 1")
    plt.plot(r2[:, 0], r2[:, 1], label="Body 2")
    plt.scatter([r1[0, 0], r2[0, 0]], [r1[0, 1], r2[0, 1]], color='black', label='Start')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Two-Body Periodic Orbit via Shooting")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()
    

    res = np.linalg.norm(sol2.y[:, -1] - sol2.y[:, 0])
    E1 = two_body_energy(sol2.y[:, -1])
    print(f"Residual norm = {res:.2e}")
    print(f"Energy drift  = {abs(E1 - E0):.2e}")
else:
    print(" Shooting failed to converge.")
 #%%   
   
print('T_corrected',T_corrected)
print('y0_corrected',y0_corrected)

    
#%%    
#PLOTTING ORBIT OF INITAL CONDITIONS 
#THIS CELL ANIMATIONS THE ORBITS ACCORDING TO THE INITIAL CONDITIONS (OUR GUESSES)

sol = solve_ivp(two_body_derivatives, [0, 20], y0, t_eval=np.linspace(0, 20, 20000),
                rtol=1e-12, atol=1e-14)

r1 = sol.y[0:2].T
r2 = sol.y[4:6].T

com = (m1 * r1 + m2 * r2) / (m1 + m2)
plt.plot(com[:, 0], com[:, 1], '--', label="Center of Mass")


plt.plot(r1[:, 0], r1[:, 1], label="Body 1")
plt.plot(r2[:, 0], r2[:, 1], label="Body 2")
plt.axis("equal")
plt.legend()
plt.grid(True)
plt.title("Two-Body Orbit")
plt.show()
    
step = 100  # Only take every 100th point to speed up animation
x_data1 = r1[::step, 0]
y_data1 = r1[::step, 1]
x_data2 = r2[::step, 0]
y_data2 = r2[::step, 1]

x_anim1, y_anim1 = [], []
x_anim2, y_anim2 = [], []

fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')

# One line for each body
line1, = ax.plot([], [], 'r-', label="Body 1")
line2, = ax.plot([], [], 'b-', label="Body 2")
ax.legend()

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2

def update(n):
    x_anim1.append(x_data1[n])
    y_anim1.append(y_data1[n])
    x_anim2.append(x_data2[n])
    y_anim2.append(y_data2[n])

    line1.set_data(x_anim1, y_anim1)
    line2.set_data(x_anim2, y_anim2)
    return line1, line2


ani = anm.FuncAnimation(
    fig,
    update,
    frames=len(x_data1),  # use the actual length
    init_func=init,
    blit=True,
    interval=0.1
)

plt.show()

#%% 
#THIS CELL ANIMATES USING THE INITAL CONDITIONS FOUND BY SHOOTING

# Use the corrected initial conditions and period
T_anim = T_corrected
t_eval = np.linspace(0, T_anim, 2000)
sol = solve_ivp(two_body_derivatives, [0, T_anim], y0_corrected, t_eval=t_eval,
                rtol=1e-12, atol=1e-14)

r1 = sol.y[0:2].T
r2 = sol.y[4:6].T

plt.plot(r1[:, 0], r1[:, 1], label="Body 1")
plt.plot(r2[:, 0], r2[:, 1], label="Body 2")
plt.axis("equal")
plt.legend()
plt.grid(True)
plt.title("Two-Body Orbit")
plt.show()
    

# Extract position data
x_data1, y_data1 = r1[:, 0], r1[:, 1]
x_data2, y_data2 = r2[:, 0], r2[:, 1]

# Setup figure
fig, ax = plt.subplots()
ax.set_xlim(-3,6)
ax.set_ylim(-4, 2)
ax.set_aspect('equal')

body1_line, = ax.plot([], [], 'r-', lw=1, label='Body 1 trail')
body2_line, = ax.plot([], [], 'b-', lw=1, label='Body 2 trail')
body1_dot, = ax.plot([], [], 'ro')  # current position
body2_dot, = ax.plot([], [], 'bo')

ax.legend()

# Trail length
trail_length = 100

def init():
    body1_line.set_data([], [])
    body2_line.set_data([], [])
    body1_dot.set_data([], [])
    body2_dot.set_data([], [])
    return body1_line, body2_line, body1_dot, body2_dot

def update(frame):
    i0 = max(0, frame - trail_length)
    body1_line.set_data(x_data1[i0:frame], y_data1[i0:frame])
    body2_line.set_data(x_data2[i0:frame], y_data2[i0:frame])
    body1_dot.set_data([x_data1[frame]], [y_data1[frame]])  
    body2_dot.set_data([x_data2[frame]], [y_data2[frame]])
    return body1_line, body2_line, body1_dot, body2_dot

ani = anm.FuncAnimation(
    fig,
    update,
    frames=len(t_eval),
    init_func=init,
    blit=True,
    interval=1
)


plt.show()
