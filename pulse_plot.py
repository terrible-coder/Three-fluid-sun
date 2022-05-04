import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

params = pd.read_csv("nparams.txt", header=None, sep="=")

params	= params[1]
Nx		= int(params[0])
Nt		= int(params[1])
dx, dt	= params[2:4]

x0_idx	= int(params[4])
B0		= params[5]
L,t0,VA0= params[6:9]
Om_e	= params[9]
Om_i	= params[10]

dz = dx * L
dt = dt * t0

# B = np.loadtxt("B.csv", delimiter=",") * B0
v = np.loadtxt("v.csv", delimiter=",") * VA0
# u = np.loadtxt("u.csv", delimiter=",") * VA0

z = np.arange(-600, -600+L-dz, dz)

fig, ax = plt.subplots()
ax.grid()
ax.set_ylim([-2, 2])
ax.set_xlabel(r"$z$")
# ax.set_ylabel(r"$B$")
ax.set_ylabel(r"$v$")
# ax.set_ylabel(r"$u$")

# line, = ax.plot(z, B[10])
line, = ax.plot(z, v[0])
# line, = ax.plot(z, u[10])

def update(frame):
	# line.set_ydata(B[frame])
	line.set_ydata(v[frame])
	# line.set_ydata(u[frame])
	return line,

movie = anim.FuncAnimation(fig, update, frames=range(40), repeat=True, interval=100, blit=True, save_count=0)

plt.show()
