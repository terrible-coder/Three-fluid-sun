import time
import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as lin

data = pd.read_csv("ndata.csv", header=0)
params = pd.read_csv("nparams.txt", header=None, sep="=")

print(params)

alpha	= data["alpha"]
beta	= data["beta"]
nu_in	= data["nu_in"]
nu_en	= data["nu_en"]
n_e		= data["n_e"]
rho		= data["rho"]

params	= params[1]
Nx		= int(params[0])
Nt		= int(params[1])
dx, dt	= params[2:4]

x0_idx	= int(params[4])
Om_e	= params[9]
Om_i	= params[10]

period_idx = 30

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

def mm1(j):
	return np.array([
	[	alpha1(j),	alpha2(j),	0	],
	[	0, 				0,		0	],
	[	0,				0,		0	]
	])

def m(j):
	return np.array([
	[	alpha3(j),		alpha4(j),			0		],
	[	beta1(j),		beta2(j),		beta3(j)	],
	[	gamma1(j),		gamma2(j),		gamma3(j)	]
	])

def mp1(j):
	return np.array([
	[	alpha5(j),		0,		0	],
	[	beta4(j),		0,		0	],
	[	gamma4(j),		0,		0	]
	])

def M_first():
	return np.array([
		[alpha3(0),	0,	0,	alpha1(0)+alpha5(0)	],
		[	0,		1,	0,		0				],
		[	0,		0,	1,		0				]
	])

def M_last():
	return np.array([
		[alpha1(Nx-1),	alpha2(Nx-1),	0,	alpha3(Nx-1),	alpha4(Nx-1),	0				],
		[0,				0,				0,	beta1(Nx-1),	beta2(Nx-1),	beta3(Nx-1)		],
		[0,				0,				0,	gamma1(Nx-1),	gamma2(Nx-1),	gamma3(Nx-1)	],
	])

def generateMatrix(above):
	A = sp.lil_matrix((3*Nx, 3*Nx))
	A[:3, :4]			= M_first()
	A[3*Nx-3:, 3*Nx-6:]	= M_last()
	for i in range(3, 3*Nx-3, 3):
		A[i:i+3, i-3:i  ]	= mm1(i // 3)
		A[i:i+3, i  :i+3]	= m(i // 3)
		A[i:i+3, i+3:i+6]	= mp1(i // 3)

	A[3*Nx-3:, 3*(Nx - period_idx)] = np.array([ alpha5(Nx-1), beta4(Nx-1), gamma4(Nx-1) ])

	if not above:
		idx = 3*x0_idx
		# A[idx+1] += A[idx+2]
		A[:, idx+2] = 0
		A[idx+2, :] = 0
		A[idx+2, idx+2] = 1
	return sp.csc_matrix(A)


print("For T < 80s\n===============")

start = time.time()
A = generateMatrix(False)
print("Matrix generated. Calculating inverse...")
A_inv = lin.inv(A)
sp.save_npz("pulse_inverse_1.npz", A_inv)
end = time.time()
print(f"Time taken = {end - start}s")


print("For T > 80s\n===============")

start = time.time()
A = generateMatrix(True)
print("Matrix generated. Calculating inverse...")
A_inv = lin.inv(A)
sp.save_npz("pulse_inverse_2.npz", A_inv)
end = time.time()
print(f"Time taken = {end - start}s")

