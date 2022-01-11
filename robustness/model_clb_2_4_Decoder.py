from scipy.integrate import ode
import matplotlib.pyplot as plt

from models import *
from parameters import *


def_parameter_values = {
            "gamma_x": {"min": 0.01, "max": 10},
            "omega_x": {"min": 0.01, "max": 10**3},
            "eta_x": {"min": 0.01, "max": 10},
            "m_x": {"min": 1, "max": 5},
            "n_y": {"min": 1, "max": 5},
            "delta_x": {"min": 0.01, "max": 10},
            "theta_x": {"min": 0.01, "max": 10**3},
            "rho_x" :  {"min": 0.1, "max": 10}}

param_references = (delta_L, 
                    gamma_L_X, 
                    n_y, 
                    theta_L_X, 
                    eta_x, 
                    omega_x, 
                    m_x, 
                    delta_x, 
                    gamma_x, 
                    theta_x, 
                    rho_x)


param_names = ["delta_x",
               "gamma_x",
               "n_y",
               "theta_x",
               "eta_x",
               "omega_x",
               "m_x",
               "delta_x",
               "gamma_x",
               "theta_x",
               "rho_x"]



class model_clb_2_4_Decoder:
    def __init__(self, params=param_names, parameter_values=def_parameter_values, threshold = 0.75, NM = 0.5):
        
        self.nParams = len(params)  
        self.params = params #model parameter names
        self.parameter_values = parameter_values #allowed parameter ranges  
        self.modes = [self.eval]  
        
        
        
        self.states =  [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]


        # simulation parameters (for a single state)
        self.t_end = 500
        self.N = self.t_end

        # optimization parameters
        self.threshold = threshold
        self.NM = NM


    def eval(self, candidate):
        D3, D2, D1, D0 = self.simulate(candidate)
        return self.getFitness([D3, D2, D1, D0])

    def isViable(self, candidate, fitness = None):
        if fitness == None:
            fitness = self.eval(candidate)
        
        return fitness[0] == 1.0

    def getFitness(self, signal):
        threshold_low = self.threshold - self.NM
        threshold_high = self.threshold + self.NM

        n = len(self.states)
        step = int(self.N)

        S = signal[step::step]
        #print(S)

        O = np.zeros(n)
        O[1::2] = 1
        #print(O)        



        fit = 0
        for o,s in zip(O,S):
            if o: # if high
                if s > threshold_high:
                    fit += 1
            else:
                if s < threshold_low:
                    fit += 1
        
        fit /= n

        print(fit)
        return (fit,)


        
    def getTotalVolume(self):
        vol = 1.0
        for param in self.params:       
            vol = vol*(self.parameter_values[param]["max"] - self.parameter_values[param]["min"])
        return vol    


    def simulate(self, candidate, plot_on=False):


        delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, gamma_x, theta_x, rho = candidate
        delta_y = delta_x
        rho_x = 0
        rho_y = 0

        states = self.states
        t_end = self.t_end
        N = self.N
        
        
        """
        rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b, rho_I2_a, rho_I2_b, rho_I3_a, rho_I3_b = 0, 5, 5, 0, 5, 0, 5, 0

        params = (delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, delta_y, rho_x, rho_y, gamma_x, theta_x, r_X, r_Y, 
                rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b, rho_I2_a, rho_I2_b, rho_I3_a, rho_I3_b)
        """

        Y0 = np.zeros(40)

        # number of cells: toggle switches
        N_A0 = np.array([1, 1])
        N_A1 = np.array([1, 1])

        Y0[4:6] = N_A0
        Y0[10:12] = N_A1

        # Y0[14-2+12:26-2+12] = 1
        Y0[24:36] = 1



        """
        simulations
        """

        for iteration, state in enumerate(states):

            A0 = state[0]
            A1 = state[1]

            if iteration > 0 and states[iteration - 1] == state:
                # Za dekoder se ne bo nikoli zgodilo to!
                # rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b, rho_I2_a, rho_I2_b, rho_I3_a, rho_I3_b = (1-I0) * 5, I0*5, (1-I1)*5, I1*5, (1-I2)*5, I2*5, (1-I3)*5, I3*5
                rho_A0_a, rho_A0_b, rho_A1_a, rho_A1_b = 0, 0, 0, 0
            else:
                rho_A0_a, rho_A0_b, rho_A1_a, rho_A1_b = (1 - A0) * 5, A0 * 5, (1 - A1) * 5, A1 * 5

            rho_x, rho_y = 0, 0
            params = (
            delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, delta_y, rho_x, rho_y, gamma_x, theta_x,
            r_X, r_Y,
            rho_A0_a, rho_A0_b, rho_A1_a, rho_A1_b)

            if iteration:
                Y0 = Y_last[-1, :]

            # Y0[24:26] = S

            # initialization

            T = np.linspace(0, t_end, N)

            t1 = t_end
            dt = t_end / N
            T = np.arange(0, t1 + dt, dt)
            Y = np.zeros([1 + N, 40])
            Y[0, :] = Y0

            # simulation
            r = ode(CLB_model_2_4_Decoder_ODE).set_integrator('zvode', method='bdf')
            r.set_initial_value(Y0, T[0]).set_f_params(params)

            i = 1
            while r.successful() and r.t < t1:
                Y[i, :] = r.integrate(r.t + dt)
                i += 1

                # hold the state after half of the simulation time!
                if r.t > t1 / 2:
                    params = (
                    delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, delta_y, rho_x, rho_y, gamma_x,
                    theta_x, r_X, r_Y,
                    0, 0, 0, 0)
                    r.set_f_params(params)

            Y_last = Y
            if not iteration:
                Y_full = Y
                T_full = T
            else:
                Y_full = np.append(Y_full, Y, axis=0)
                T_full = np.append(T_full, T + iteration * t_end, axis=0)

        Y = Y_full
        T = T_full

        # S0, S1 = Y[:,24], Y[:,25]

        A0_a, A0_b = Y[:, 2], Y[:, 3]
        A1_a, A1_b = Y[:, 8], Y[:, 9]

        D3 = Y[:, -1]
        D2 = Y[:, -2]
        D1 = Y[:, -3]
        D0 = Y[:, -4]
        # plot
        """
        ax1 = plt.subplot(611)
        ax1.plot(T, A0_a, color="#800000ff", alpha=0.75)
        ax1.plot(T, A0_b, color="#999999ff", alpha=0.75)
        ax1.legend(["$A_0$", "$\\overline{A_0}$"])
        # ax1.set_title('$I_0$ toggle')
        ax1.set_xlabel("Time [min]")
        ax1.set_ylabel("Conc. [nM]")

        ax2 = plt.subplot(612)
        ax2.plot(T, A1_a, color="#00ff00ff", alpha=0.75)
        ax2.plot(T, A1_b, color="#666666ff")  # , alpha=0.75)
        ax2.legend(["$A_1$", "$\\overline{A_1}$"])
        # ax2.set_title('$I_1$ toggle')
        ax2.set_xlabel("Time [min]")
        ax2.set_ylabel("Conc. [nM]")

        ax3 = plt.subplot(613)
        ax3.plot(T, D0, color="#8080805a", alpha=0.75)
        # ax6.set_title('out')
        ax3.legend(['D0'])
        ax3.set_xlabel("Time [min]")
        ax3.set_ylabel("Conc. [nM]")

        ax4 = plt.subplot(614)
        ax4.plot(T, D1, color="#8080805a", alpha=0.75)
        # ax6.set_title('out')
        ax4.legend(['D1'])
        ax4.set_xlabel("Time [min]")
        ax4.set_ylabel("Conc. [nM]")

        ax5 = plt.subplot(615)
        ax5.plot(T, D2, color="#8080805a", alpha=0.75)
        # ax6.set_title('out')
        ax5.legend(['D2'])
        ax5.set_xlabel("Time [min]")
        ax5.set_ylabel("Conc. [nM]")

        ax6 = plt.subplot(616)
        ax6.plot(T, D3, color="#8080805a", alpha=0.75)
        # ax6.set_title('out')
        ax6.legend(['D3'])
        ax6.set_xlabel("Time [min]")
        ax6.set_ylabel("Conc. [nM]")

        # plt.suptitle("$out = \\overline{S}_1 \\overline{S}_0 I_0 \\vee \\overline{S}_1 S_0 I_1 \\vee S_1 \\overline{S}_0 I_2 \\vee S_1 S_0 I_3$")
        plt.gcf().set_size_inches(15, 10)
        plt.savefig("figs\\CBLB_2_4_Decoder_ode.pdf", bbox_inches='tight')

        plt.show()
        """

        return [D3, D2, D1, D0]

    """
    def simulateStochastic(self, candidate): 
        multiplier = 1
        omega = 1
        
        delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, delta_y, gamma_x, theta_x, rho = candidate
                
        states = self.states
        t_end = self.t_end
        N = self.N
        
        rho_x = 0
        rho_y = 0

        Y0 = np.zeros(59)

        # number of cells: toggle switches
        N_I0 = np.array([1,1])
        N_I1 = np.array([1,1])
        N_I2 = np.array([1,1])
        N_I3 = np.array([1,1])

        Y0[4:6] = N_I0
        Y0[10:12] = N_I1
        Y0[16:18] = N_I2
        Y0[22:24] = N_I3

        # number of cells: mux
        #Y0[22-4+24:38-4+24] = 1 # number of cells
        Y0[42:58] = 1 # number of cells
         
        Y_conc = np.array(Y0*omega).astype(int) 
        print(y_conc) 

        Y_total = []
        Y_total.append(y_conc)
        t = 0 
        T = [] 
        T.append(t) 
   

        N = np.zeros(59)
        
        
        for i in range(multiplier):
            #flip flops 
            #a
            N[i*4 + 0,i*12 + 0] = 1  
            N[i*4 + 0,i*12 + 1] = 1  
            N[i*4 + 0,i*12 + 2] = -1  
            #not a
            N[i*4 + 1,i*12 + 3] = 1  
            N[i*4 + 1,i*12 + 4] = 1  
            N[i*4 + 1,i*12 + 5] = -1  
            #q 
            N[i*4 + 2,i*12 + 6] = 1  
            N[i*4 + 2,i*12 + 7] = 1   
            N[i*4 + 2,i*12 + 8] = -1            
            #not q 
            N[i*4 + 3,i*12 + 9] = 1    
            N[i*4 + 3,i*12 + 10] = 1    
            N[i*4 + 3,i*12 + 11] = -1 
            #instructions               
            N[multiplier*4 + i*2+ 0, multiplier*12 + i*4 + 0] = 1  
            N[multiplier*4 + i*2 + 0, multiplier*12 + i*4 + 1] = -1     
            N[multiplier*4 + i*2 + 1, multiplier*12 + i*4 + 2] = 1  
            N[multiplier*4 + i*2 + 1, multiplier*12 + i*4 + 3] = -1             
            
        
        
        while t < self.T: 
            #choose two random numbers 
            r = np.random.uniform(size=2)
            r1 = r[0]
            r2 = r[1]           

            #get clk    
            clk = int(get_clock(t)*omega)       
            #get propensities               
            a = np.zeros(16*multiplier) 
            
            if self.model_mode == one_bit_processor_ext:
                ds = [y_conc[3]] #not q1   
                a[multiplier*12:] = addressing_stochastic_one_bit_model(y_conc, t, params_addr, omega)
                
            elif self.model_mode == two_bit_processor_ext:
                ds = [y_conc[7],y_conc[2]] #not q2, q1     
                a[multiplier*12:] = addressing_stochastic_two_bit_model(y_conc, t, params_addr, omega)
                
            else: #three_bit_processor_ext  
                ds = [y_conc[11],y_conc[2],y_conc[6]] #not q3, q1, q2       
                a[multiplier*12:] = addressing_stochastic_three_bit_model(y_conc, t, params_addr, omega) 

            for i in range(multiplier):
                y = y_conc[i*4:i*4+4] 
                y = np.append(y, ds[i]) #to do 
                y = np.append(y, clk) 
                a[i*12:i*12+12] = ff_stochastic_model(y, t, params_ff, omega)    
                    
            asum = np.cumsum(a)
            a0 = np.sum(a)  
            #get tau
            tau = (1.0/a0)*np.log(1.0/r1)    
        
            #print(t)       
            #select reaction 
            reaction_number = np.argwhere(asum > r2*a0)[0,0] #get first element         
        
            #update concentrations
            y_conc = y_conc + N[:,reaction_number]      
            Y_total.append(y_conc) 
            #update time
            t = t + tau  
            T.append(t)

            
        T = np.array(T)
        Y_total = np.array(Y_total) 
        return T, Y_total       
    """

if __name__ == "__main__":
    clb = model_clb()

    rho = 5
    #candidate = delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, delta_y, gamma_x, theta_x, rho
    candidate = delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, gamma_x, theta_x, rho

    D3, D2, D1, D0 = clb.simulate(candidate, plot_on=True)
    print(clb.getFitness([D3, D2, D1, D0]))

    #print(clb.eval(candidate))