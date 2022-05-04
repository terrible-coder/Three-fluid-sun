import pandas as pd
import numpy as np

params = pd.read_csv("param.txt", header=None, sep="=")
data = pd.read_csv("data.csv", header=0, index_col=0)

B0, L, T, dx, dt, m_e, m_i, e = params[1]

Ne			= np.array(data["Ne"])
Nn			= np.array(data["Nn"])
nu_en		= np.array(data["nu_en"])
nu_in		= np.array(data["nu_in"])
nu_ei		= np.array(data["nu_ei"])
Omega_e0	= 1.76E7 * B0
Omega_i0	= 9.58E3 * B0
rho			= Ne * (m_i + m_e)

x0_idx	= int(600 / dx)
Ne0		= Ne[x0_idx]
rho0	= rho[x0_idx]
VA0		= 1E-5/np.sqrt(4*np.pi) * B0 / np.sqrt(rho0)
t0		= L / VA0

print(f"x0_idx = {x0_idx}")
print(VA0, "km/s")

Ne_b = Ne / Ne0
rho_b = rho / rho0
nu_en_b = nu_en * t0
nu_in_b = nu_in * t0
nu_ei_b = nu_ei * t0
Omega_eb = Omega_e0 * t0
Omega_ib = Omega_i0 * t0
beta = (nu_en_b + nu_ei_b) / (Omega_eb * Ne_b)
alpha = (Ne * m_i) / (Nn * (m_i + m_e))

table = pd.DataFrame({
	"alpha": alpha,
	"beta": beta,
	"nu_in": nu_in_b,
	"nu_ei": nu_ei_b,
	"nu_en": nu_en_b,
	"n_e": Ne_b,
	"rho": rho_b
})

table.to_csv("ndata.csv", index=False)

nparams = open("nparams.txt", "w")
nparams.write("\n".join([
	f"Nx = {int(L / dx)}",
	f"Nt = {int(T / dt)}",
	f"dz = {dx / L}",
	f"dt = {dt / t0}",
	"",
	f"x0_idx = {x0_idx}",
	f"B0 = {B0}",
	f"L = {L}",
	f"t0 = {t0}",
	f"VA0 = {VA0}",
	f"Omega_e = {Omega_eb}",
	f"Omega_i = {Omega_ib}"
]))
nparams.close()