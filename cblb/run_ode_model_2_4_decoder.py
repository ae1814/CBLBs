from scipy.integrate import ode
import matplotlib.pyplot as plt

from models import *
from parameters import *

rho_x = 0
rho_y = 0

params = delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, rho_x, gamma_x, theta_x, r_X

# A0, A1
#A = np.array([0, 0]) # D0 = 1, D1 = 0, D2 = 0, D3 = 0
#A = np.array([0, 1]) # D0 = 0, D1 = 0, D2 = 1, D3 = 0
#A = np.array([1, 0]) # D0 = 0, D1 = 1, D2 = 0, D3 = 0
A = np.array([1, 1]) # D0 = 0, D1 = 0, D2 = 0, D3 = 1

# simulation parameters
t_end = 1500
N = t_end


Y0 = np.zeros(30)
Y0[14:26] = 1 # number of cells

Y0[:2] = A


T = np.linspace(0, t_end, N)

t1 = t_end
dt = t_end/N
T = np.arange(0,t1+dt,dt)
Y = np.zeros([1+N,30])
Y[0,:] = Y0

r = ode(DECODER_2_4_model_ODE).set_integrator('zvode', method='bdf')
r.set_initial_value(Y0, T[0]).set_f_params(params)

i = 1
while r.successful() and r.t < t1:
    Y[i,:] = r.integrate(r.t+dt)
    i += 1

# D3
out = Y[:,-1]
plt.plot(T,out)
plt.title("D3")
plt.show()
# D2
out = Y[:,-2]
plt.plot(T,out)
plt.title("D2")
plt.show()
# D1
out = Y[:,-3]
plt.plot(T,out)
plt.title("D1")
plt.show()
# D0
out = Y[:,-4]
plt.plot(T,out)
plt.title("D0")
plt.show()

"""
# Y = a, b, N_A
a = Y[:,0]
b = Y[:,1]
N_A = Y[:,2]

ax1 = plt.subplot(211)
ax1.plot(T,a)
ax1.plot(T,b)
ax1.legend(["a", "b"])

ax2 = plt.subplot(212)
ax2.plot(T,N_A)
ax2.legend(["N_A"])


plt.show()
"""