#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.integrate import odeint
import sympy as smp
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import PillowWriter

t, m, g = smp.symbols('t m g')
theta = smp.symbols(r'\theta', cls=smp.Function)
theta = theta(t)

d_theta  = smp.diff(theta, t)
dd_theta = smp.diff(d_theta, t)

x, y = smp.symbols('x y', cls=smp.Function)
x, y = x(theta), y(theta)

local = 'taut'
if local == 'parabola':
    x = theta
    y = theta ** 2
elif local == 'taut':
    x = smp.sin(2 * theta) + 2 * theta
    y = 1 - smp.cos(2 * theta)
x_func = smp.lambdify(theta, x)
y_func = smp.lambdify(theta, y)

T = 0.5 * m * (smp.diff(x, t)**2 + smp.diff(y, t)**2)
V = m * g * y
L = T - V

LEQ = smp.diff(L, theta) - smp.diff(smp.diff(L, d_theta), t)

second_d = smp.solve(LEQ, dd_theta)[0]
first_d = d_theta
second_d_func = smp.lambdify((g, theta, d_theta), second_d)
first_d_func = smp.lambdify(d_theta, d_theta)

def dSdt(S, t):
    return [
        first_d_func(S[1]), second_d_func(g, S[0], S[1])
    ]
    
t = np.linspace(0, 20, 1000)
g = 9.81

if local == 'parabola':
    answer = odeint(dSdt, y0 = [2, 0], t = t)
elif local == 'taut':
    answer = odeint(dSdt, y0 = [np.pi/4, 0], t = t)
    
def get_xy(theta):
    return x_func(theta), y_func(theta)

x, y = get_xy(answer.T[0])

def animate(i):
    ln.set_data([x[i]], [y[i]])

fig, ax = plt.subplots()

ax.grid(True, linestyle='--', alpha=0.6)
ln, = plt.plot([], [], 'ro', label='Pendulum Position')

ax.set_title('Pendulum Motion Simulation', fontsize=16, fontweight='bold')
ax.set_xlabel('Horizontal Position (m)', fontsize=12)
ax.set_ylabel('Vertical Position (m)', fontsize=12)

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-1, 5)

ax.legend(loc='upper right')

ani = animation.FuncAnimation(fig, animate, frames = 1000, interval = 50)
ani.save(f'PENDULUM_{local}.gif', writer='pillow', fps = 50)
