from scipy.integrate import ode
import matplotlib.pyplot as plt

from models import *
from parameters import *


"""
[[(S1, I1)], []]
"""
"""
states = [([0,0], [0,0,0,0]), ([0,0], [1,0,0,0]), 
          ([1,0], [0,0,0,0]), ([1,0], [0,1,0,0]), 
          ([0,1], [0,0,0,0]), ([0,1], [0,0,1,0]), 
          ([1,1], [0,0,0,0]), ([1,1], [0,0,0,1])]
"""        

"""
states = [([0,0], [0,0,0,0]), ([0,0], [1,0,0,0]), 
          ([1,0], [1,0,0,0]), ([1,0], [1,1,0,0]), 
          ([0,1], [1,1,0,0]), ([0,1], [1,1,1,0]), 
          ([1,1], [1,1,1,0]), ([1,1], [1,1,1,1])]

"""
states = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]



# simulation parameters (for a single state)
t_end = 500
N = t_end

rho_x = 0
rho_y = 0

"""
rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b, rho_I2_a, rho_I2_b, rho_I3_a, rho_I3_b = 0, 5, 5, 0, 5, 0, 5, 0

params = (delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, delta_y, rho_x, rho_y, gamma_x, theta_x, r_X, r_Y, 
         rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b, rho_I2_a, rho_I2_b, rho_I3_a, rho_I3_b)
"""

Y0 = np.zeros(40)

# number of cells: toggle switches
N_A0 = np.array([1,1])
N_A1 = np.array([1,1])

Y0[4:6] = N_A0
Y0[10:12] = N_A1

#Y0[14-2+12:26-2+12] = 1
Y0[24:36] = 1


"""
simulations
"""

for iteration, state in enumerate(states):

    A0 = state[0]
    A1 = state[1]


    if iteration > 0 and states[iteration-1] == state:
        # Za dekoder se ne bo nikoli zgodilo to!
        #rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b, rho_I2_a, rho_I2_b, rho_I3_a, rho_I3_b = (1-I0) * 5, I0*5, (1-I1)*5, I1*5, (1-I2)*5, I2*5, (1-I3)*5, I3*5
        rho_A0_a, rho_A0_b, rho_A1_a, rho_A1_b = 0, 0, 0, 0
    else:
        rho_A0_a, rho_A0_b, rho_A1_a, rho_A1_b = (1-A0) * 5, A0*5, (1-A1)*5, A1*5
        
    
    rho_x, rho_y = 0,0
    params = (delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, delta_y, rho_x, rho_y, gamma_x, theta_x, r_X, r_Y, 
         rho_A0_a, rho_A0_b, rho_A1_a, rho_A1_b)

    if iteration:
        Y0 = Y_last[-1,:]
    
    #Y0[24:26] = S

    # initialization

    T = np.linspace(0, t_end, N)

    t1 = t_end
    dt = t_end/N
    T = np.arange(0,t1+dt,dt)
    Y = np.zeros([1+N,40])
    Y[0,:] = Y0


    # simulation
    r = ode(CLB_model_2_4_Decoder_ODE).set_integrator('zvode', method='bdf')
    r.set_initial_value(Y0, T[0]).set_f_params(params)

    i = 1
    while r.successful() and r.t < t1:
        Y[i,:] = r.integrate(r.t+dt)
        i += 1

        # hold the state after half of the simulation time!
        if r.t > t1/2:
            params = (delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, delta_y, rho_x, rho_y, gamma_x, theta_x, r_X, r_Y, 
            0, 0, 0, 0)
            r.set_f_params(params)

    Y_last = Y
    if not iteration:
        Y_full = Y
        T_full = T
    else:
        Y_full = np.append(Y_full, Y, axis = 0)
        T_full = np.append(T_full, T + iteration * t_end, axis = 0)

Y = Y_full
T = T_full

#S0, S1 = Y[:,24], Y[:,25]

A0_a, A0_b = Y[:,2], Y[:,3]
A1_a, A1_b = Y[:,8], Y[:,9]

D3 = Y[:,-1]
D2 = Y[:,-2]
D1 = Y[:,-3]
D0 = Y[:,-4]
# plot

ax1 = plt.subplot(611)
ax1.plot(T, A0_a, color="#800000ff", alpha=0.75)
ax1.plot(T, A0_b, color="#999999ff", alpha=0.75)
ax1.legend(["$A_0$", "$\\overline{A_0}$"])
#ax1.set_title('$I_0$ toggle')
ax1.set_xlabel("Time [min]")
ax1.set_ylabel("Conc. [nM]")


ax2 = plt.subplot(612)
ax2.plot(T, A1_a, color = "#00ff00ff", alpha=0.75)
ax2.plot(T, A1_b, color = "#666666ff")#, alpha=0.75)
ax2.legend(["$A_1$", "$\\overline{A_1}$"])
#ax2.set_title('$I_1$ toggle')
ax2.set_xlabel("Time [min]")
ax2.set_ylabel("Conc. [nM]")


ax3 = plt.subplot(613)
ax3.plot(T,D0, color = "#8080805a", alpha=0.75)
#ax6.set_title('out')
ax3.legend(['D0'])
ax3.set_xlabel("Time [min]")
ax3.set_ylabel("Conc. [nM]")

ax4 = plt.subplot(614)
ax4.plot(T,D1, color = "#8080805a", alpha=0.75)
#ax6.set_title('out')
ax4.legend(['D1'])
ax4.set_xlabel("Time [min]")
ax4.set_ylabel("Conc. [nM]")

ax5 = plt.subplot(615)
ax5.plot(T,D2, color = "#8080805a", alpha=0.75)
#ax6.set_title('out')
ax5.legend(['D2'])
ax5.set_xlabel("Time [min]")
ax5.set_ylabel("Conc. [nM]")

ax6 = plt.subplot(616)
ax6.plot(T,D3, color = "#8080805a", alpha=0.75)
#ax6.set_title('out')
ax6.legend(['D3'])
ax6.set_xlabel("Time [min]")
ax6.set_ylabel("Conc. [nM]")


#plt.suptitle("$out = \\overline{S}_1 \\overline{S}_0 I_0 \\vee \\overline{S}_1 S_0 I_1 \\vee S_1 \\overline{S}_0 I_2 \\vee S_1 S_0 I_3$")
plt.gcf().set_size_inches(15,10)
plt.savefig("figs\\CBLB_2_4_Decoder_ode.pdf", bbox_inches = 'tight')

plt.show()  


