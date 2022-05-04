import pandas as pd
import numpy as np
from scipy import interpolate as intp
import matplotlib.pyplot as plt

params = pd.read_csv("param.txt", header=None, sep="=")
print(params)

B0 = params[1][0]
dx = params[1][3]
m_e, m_i, e = params[1][5:]
# B0, dx, dt, m_e, m_i, e = params[1]

data = np.genfromtxt("C7_model.csv", delimiter=",")

print(data.shape)

z = data[14:,1] # in km
# rho = data[14:,2] # in g cm^-2
T = data[14:,3] # in K
Nn = data[14:,8] # in cm^-3
Ne = data[14:,9] # in cm^-3

x = np.arange(0, 3000, dx)
T = intp.interp1d(z, T, kind="cubic")(x)
Nn = intp.interp1d(z, Nn, kind="cubic")(x)
Ne = intp.interp1d(z, Ne, kind="cubic")(x)
# rho = intp.interp1d(z, rho, kind="cubic")(x)

# collision frequencies
nu_in = 7.4E-11  * Nn * T**0.5
nu_en = 1.95E-10 * Nn * T**0.5
nu_ei = 3.759E-6 * Ne * T**-1.5

def ghost_p(ax, A):
	return np.ones(len(ax)) * A[-1]

def ghost_n(ax, A):
	A_sl = (A[1] - A[0]) / (ax[1] - ax[0])
	return A[0] + (ax) * A_sl

def concat(n, a, p):
	return np.concatenate((n, a, p))

# ghost region 3000 to 5000
x_ghp = np.arange(3000, 5000, dx)
T_ghp = ghost_p(x_ghp, T)
Nn_ghp = ghost_p(x_ghp, Nn)
Ne_ghp = ghost_p(x_ghp, Ne)
nu_in_ghp = 7.4E-11  * Nn_ghp * T_ghp**0.5
nu_en_ghp = 1.95E-10 * Nn_ghp * T_ghp**0.5
nu_ei_ghp = 3.759E-6 * Ne_ghp * T_ghp**-1.5

# ghost region -600 to 0
x_ghn = np.arange(-600, 0, dx)
T_ghn = ghost_n(x_ghn, T)
Nn_ghn = ghost_n(x_ghn, Nn)
Ne_ghn = ghost_n(x_ghn, Ne)
nu_in_ghn = 7.4E-11  * Nn_ghn * T_ghn**0.5
nu_en_ghn = 1.95E-10 * Nn_ghn * T_ghn**0.5
nu_ei_ghn = 3.759E-6 * Ne_ghn * T_ghn**-1.5

Omega_e = 1.76E7 * B0
Omega_e = np.ones(len(x)) * Omega_e
Omega_i = 9.58E3 * B0
Omega_i = np.ones(len(x)) * Omega_i

plt.subplot(121)
plt.plot(np.log10(T), x, "b-", label="T")
plt.plot(np.log10(T_ghp), x_ghp, "b-.")
plt.plot(np.log10(T_ghn), x_ghn, "b-.")
plt.plot(np.log10(Nn), x, "r-", label=r"$N_n$")
plt.plot(np.log10(Nn_ghp), x_ghp, "r-.")
plt.plot(np.log10(Nn_ghn), x_ghn, "r-.")
plt.plot(np.log10(Ne), x, "g-", label=r"$N_e$")
plt.plot(np.log10(Ne_ghp), x_ghp, "g-.")
plt.plot(np.log10(Ne_ghn), x_ghn, "g-.")
plt.ylabel("height (in km)")
plt.xlabel("(in powers of 10)")
plt.legend()

plt.subplot(122)
plt.plot(np.log10(nu_ei), x, "b-", label=r"$\nu_{ei}$")
plt.plot(np.log10(nu_ei_ghp), x_ghp, "b-.")
plt.plot(np.log10(nu_ei_ghn), x_ghn, "b-.")
plt.plot(np.log10(nu_en), x, "r-", label=r"$\nu_{en}$")
plt.plot(np.log10(nu_en_ghp), x_ghp, "r-.")
plt.plot(np.log10(nu_en_ghn), x_ghn, "r-.")
plt.plot(np.log10(nu_in), x, "g-", label=r"$\nu_{in}$")
plt.plot(np.log10(nu_in_ghp), x_ghp, "g-.")
plt.plot(np.log10(nu_in_ghn), x_ghn, "g-.")
plt.plot(np.log10(Omega_e), x, label=r"$\Omega_e$")
plt.plot(np.log10(Omega_i), x, label=r"$\Omega_i$")
plt.ylabel("height (in km)")
plt.xlabel("(in Hz)(in powers of 10)")
plt.legend()

plt.show()

x = concat(x_ghn, x, x_ghp)
T = concat(T_ghn, T, T_ghp)
Nn = concat(Nn_ghn, Nn, Nn_ghp)
Ne = concat(Ne_ghn, Ne, Ne_ghp)
nu_in = concat(nu_in_ghn, nu_in, nu_in_ghp)
nu_ei = concat(nu_ei_ghn, nu_ei, nu_ei_ghp)
nu_en = concat(nu_en_ghn, nu_en, nu_en_ghp)

table = pd.DataFrame({
	"z"		: x,
	"T"		: T,
	"Nn"	: Nn,
	"Ne"	: Ne,
	"nu_in"	: nu_in,
	"nu_ei"	: nu_ei,
	"nu_en"	: nu_en
})

table.to_csv("data.csv", index=True)
print("Data saved in data.csv")
