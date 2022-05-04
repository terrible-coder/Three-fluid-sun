from time import time
import pandas as pd
import numpy as np
import scipy.sparse as sp

from pulse_invert import A_inv

data = pd.read_csv("ndata.csv", header=0)
params = pd.read_csv("nparams.txt", header=None, sep="=")

params	= params[1]
Nx		= int(params[0])
Nt		= int(params[1])
dx, dt	= params[2:4]

x0_idx	= int(params[4])
VA0		= params[8]
Om_e	= params[9]
Om_i	= params[10]

alpha	= data["alpha"]
beta	= data["beta"]
nu_in	= data["nu_in"]
nu_en	= data["nu_en"]
n_e		= data["n_e"]
rho		= data["rho"]

lam = 2 * dt / dx

a1 = lambda j: lam
a2 = lambda j: (lam / (4*dx)) * ( (beta[j+1] if j<Nx-1 else 0) - (beta[j-1] if j>0 else 0) ) / Om_i
a3 = lambda j: (lam / dx) * ( beta[j] - 1/n_e[j] ) / Om_i
a4 = lambda j: -(lam / (4*dx)) * ( (n_e[j+1] if j<Nx-1 else 0) - (n_e[j-1] if j>0 else 0) ) / (Om_i * n_e[j])

b1 = lambda j: 2*dt * alpha[j] * nu_in[j]
b2 = lambda j: lam * alpha[j] * (nu_en[j] - nu_in[j]) / ( rho[j] * Om_e )

c1 = lambda j: lam / rho[j]
c2 = lambda j: 2*dt * nu_in[j]
c3 = lambda j: lam * (nu_en[j] - nu_in[j]) / (rho[j] * Om_e)

alpha1	= lambda j: a2(j) - a3(j) - a4(j)
alpha2	= lambda j: a1(j)
alpha3	= lambda j: 3+2*a3(j)
alpha4	= lambda j: -a1(j)
alpha5	= lambda j: -a2(j) - a3(j) + a4(j)

beta1	= lambda j: -b2(j)
beta2	= lambda j: -b1(j)
beta3	= lambda j: 3+b1(j)
beta4	= lambda j: b2(j)

gamma1	= lambda j: c1(j) + c3(j)
gamma2	= lambda j: 3+c2(j)
gamma3	= lambda j: -c2(j)
gamma4	= lambda j: -c1(j) - c3(j)

u0 = 0.350
u0_b = u0 / VA0

Vi = dx/2
Ui = dx/2

def rhs(n):
	b = np.zeros(3*Nx)
	b[0] = 3 * grid[n][0+0] - alpha4(0) * Vi
	b[1] = Vi
	b[2] = Ui

	for j in range(3, 3*Nx, 3):
		b[j]	= 4 * grid[n][j+0] - grid[n-1][j+0]
		b[j+1]	= 4 * grid[n][j+2] - grid[n-1][j+2]
		b[j+2]	= 4 * grid[n][j+1] - grid[n-1][j+1]

	if n < t80_idx:
		b[3*x0_idx+1] += -beta3(x0_idx) * u0_b # b[3*x0_idx+2] - (beta3(x0_idx) + gamma3(x0_idx)) * u0_b
		b[3*x0_idx+2]  = (n!=t80_idx-1) * u0_b

	return b

grid = np.zeros((Nt+1, Nx, 3))
grid[0,:,1] = Vi
grid[0,:,2] = Ui
grid = grid.reshape((Nt+1, 3*Nx))
t80_idx = int(10 / dt)

grid[:t80_idx-1, 3*x0_idx+2] = u0_b

start = time()

print(f"u0_b = {u0_b} at n = {t80_idx}")

A_inv = sp.load_npz("pulse_inverse_1.npz")
for n in range(1, t80_idx):
	if n % 100 == 0:
		print(n)
	b = rhs(n)
	sol = A_inv.dot(b)
	grid[n+1] = np.copy(sol)

A_inv = sp.load_npz("pulse_inverse_2.npz")
for n in range(t80_idx, Nt):
	if n % 100 == 0:
		print(n)
	b = rhs(n)
	sol = A_inv.dot(b)
	grid[n+1] = np.copy(sol)

end = time()

print(f"Time taken = {end - start} s")

print("Saving solution...")
grid = grid.reshape((Nt+1, Nx, 3))
B = grid[:,:,0]
v = grid[:,:,1]
u = grid[:,:,2]

np.savetxt("B.csv", B, delimiter=",")
np.savetxt("v.csv", v, delimiter=",")
np.savetxt("u.csv", u, delimiter=",")
print("Done.")
