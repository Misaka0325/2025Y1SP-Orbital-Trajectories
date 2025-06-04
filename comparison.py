# -*- coding: utf-8 -*-
"""
Created on Sat May 17 15:23:09 2025

@author: dpazd
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants 
G = 6.674e-11
dt = 1e5
frames = 1000
N = 5

# Using sun and orbiting planets as an exmaple, we could use any set of initial conitions
SUN_MASS = 1.989e30
# Initial Conditions
np.random.seed(0)  # For reproducibility

planet_masses = np.random.uniform(1e24, 5e25, N)
planet_distances = np.linspace(5e10, 1e11, N)

planet_positions = np.zeros((N, 3))
planet_velocities = np.zeros((N, 3))

for i in range(N):
    angle = 2 * np.pi * np.random.rand()
    d = planet_distances[i]
    planet_positions[i] = [d * np.cos(angle), d * np.sin(angle), 0]
    speed = np.sqrt(G * SUN_MASS / d)
    planet_velocities[i] = [-speed * np.sin(angle), speed * np.cos(angle), 0]

# Combine with Sun
masses = np.concatenate(([SUN_MASS], planet_masses))
positions0 = np.vstack((np.zeros(3), planet_positions))
velocities0 = np.vstack((np.zeros(3), planet_velocities))

def compute_forces(masses, positions):
    N = len(masses)
    forces = np.zeros_like(positions)
    for i in range(N):
        for j in range(N):
            if i != j:
                r_vec = positions[j] - positions[i]
                r_mag = np.linalg.norm(r_vec) + 1e-10
                F = G * masses[i] * masses[j] / r_mag**2
                forces[i] += F * r_vec / r_mag
    return forces

def kinetic_energy(masses, velocities):
    return 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))

def potential_energy(masses, positions):
    PE = 0
    N = len(masses)
    for i in range(N):
        for j in range(i + 1, N):
            r = np.linalg.norm(positions[i] - positions[j]) + 1e-10
            PE -= G * masses[i] * masses[j] / r
    return PE
#%%
#  Euler Method 
positions_euler = positions0.copy()
velocities_euler = velocities0.copy()
initial_E_euler = kinetic_energy(masses, velocities_euler) + potential_energy(masses, positions_euler)

euler_energy = []
euler_error = []

for _ in range(frames):
    KE = kinetic_energy(masses, velocities_euler)
    PE = potential_energy(masses, positions_euler)
    E = KE + PE
    euler_energy.append(E)
    euler_error.append(abs((E - initial_E_euler) / initial_E_euler))

    forces = compute_forces(masses, positions_euler)
    accelerations = forces / masses[:, np.newaxis]
    velocities_euler += accelerations * dt
    positions_euler += velocities_euler * dt
#%%
#  Velocity Verlet Method 
positions_vv = positions0.copy()
velocities_vv = velocities0.copy()
accelerations = compute_forces(masses, positions_vv) / masses[:, np.newaxis]
initial_E_vv = kinetic_energy(masses, velocities_vv) + potential_energy(masses, positions_vv)

vv_energy = []
vv_error = []

for _ in range(frames):
    KE = kinetic_energy(masses, velocities_vv)
    PE = potential_energy(masses, positions_vv)
    E = KE + PE
    vv_energy.append(E)
    vv_error.append(abs((E - initial_E_vv) / initial_E_vv))

    positions_vv += velocities_vv * dt + 0.5 * accelerations * dt**2
    new_forces = compute_forces(masses, positions_vv)
    new_accelerations = new_forces / masses[:, np.newaxis]
    velocities_vv += 0.5 * (accelerations + new_accelerations) * dt
    accelerations = new_accelerations
#%%
# RK4 Method
def rk4_step(masses, positions, velocities, dt):
    def acceleration(pos):
        return compute_forces(masses, pos) / masses[:, np.newaxis]

    k1_v = acceleration(positions)
    k1_x = velocities

    k2_v = acceleration(positions + 0.5 * dt * k1_x)
    k2_x = velocities + 0.5 * dt * k1_v

    k3_v = acceleration(positions + 0.5 * dt * k2_x)
    k3_x = velocities + 0.5 * dt * k2_v

    k4_v = acceleration(positions + dt * k3_x)
    k4_x = velocities + dt * k3_v

    new_positions = positions + dt * (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
    new_velocities = velocities + dt * (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6

    return new_positions, new_velocities

positions_rk4 = positions0.copy()
velocities_rk4 = velocities0.copy()
initial_E_rk4 = kinetic_energy(masses, velocities_rk4) + potential_energy(masses, positions_rk4)

rk4_energy = []
rk4_error = []

for _ in range(frames):
    KE = kinetic_energy(masses, velocities_rk4)
    PE = potential_energy(masses, positions_rk4)
    E = KE + PE
    rk4_energy.append(E)
    rk4_error.append(abs((E - initial_E_rk4) / initial_E_rk4))

    positions_rk4, velocities_rk4 = rk4_step(masses, positions_rk4, velocities_rk4, dt)

#%%
# Plot Total Energy Comparison 
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(euler_energy, label="Euler", color='r')
plt.plot(vv_energy, label="Velocity Verlet", color='g')
plt.plot(rk4_energy, label="RK4", color='b')
plt.title("Total Energy Over Time")
plt.xlabel("Time Step")
plt.ylabel("Total Energy (Joules)")
plt.legend()
plt.grid(True)

# --Plot Relative Error
plt.subplot(1, 2, 2)
plt.plot(euler_error, label="Euler", color='r')
plt.plot(vv_error, label="Velocity Verlet", color='g')
plt.plot(rk4_error, label="RK4", color='b')
plt.yscale("log")
plt.title("Relative Energy Error")
plt.xlabel("Time Step")
plt.ylabel("Relative Error (log scale)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
