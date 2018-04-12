# create balanced networks P-B and P-S

from brian2 import *
import utilities as util
import matplotlib.pyplot as plt
import numpy as np
start_scope()

# params


# Neuron Model

tau = 20*ms      # Membrane time constant
Vt = -50*mV      # Spiking threshold
Vr = -60*mV      # Resting potential and reset potential
Ve = 0*mV        # Excitatory reversal potential
Vi = -80*mV      # Inhibitory reversal potential
gl = 10*nsiemens # Leak conductance
Ib = 200*pA      # Background current to each cell
tau_ref = 2*ms   # Absolute refractory period
C = 200*pF       # Membrane capacitance

# Synapse Model

tauP = 5*ms      # Decay constant of AMPA-type conductance - P cells
tauB = 4*ms      # Decay constant of GABA-type conductance - B cells
tauS = 8*ms      # Decay constant of GABA-type conductance - S cells

# Plasticity Model

tauSTDP = 20*ms         # Decay constant of (pre and post) synaptic trace
Wmin = 0                # Minimum inhibitory synaptic weight
Wmax = 300              # Maximum inhibitory synaptic weight
rho0 = 5*Hz            # Target frequency
alpha = 2*rho0*tauSTDP  # Presynaptic offset
eta = 1e-2              # Learning rate

# Network parameters

N_P = 8000           # Size of excitatory population P
N_S = 1000           # Size of inhibitory population S
N_B = 1000           # Size of inhibitory population B
epsilon = 0.02       # Probability of any connection

# Simulation parameters

runtime = 60*second # Running time (neuron time)
record_neurons = 50   # for ICI-CV and correlations
measure_time = 1000*ms # Time window to compute different measures

# Define equations

# define model of neurons

neuron_model='''dV/dt= -(V-Vr)/tau + (- gP*(V-Ve) - gS*(V-Vi) - gB*(V-Vi) + Ib)/(gl*tau) : volt (unless refractory)
dgP/dt = -gP/tauP : siemens
dgB/dt = -gB/tauB : siemens
dgS/dt = -gS/tauS : siemens'''

# define plasticity model

inh_plast = '''dXpr/dt=-Xpr/tauSTDP : 1 (event-driven)
dXpo/dt=-Xpo/tauSTDP : 1 (event-driven)
W : 1'''

# implement

# implement neurons

neurons = NeuronGroup(N_P+N_B+N_S,neuron_model,'euler',threshold='V>Vt',reset='V=Vr',refractory=tau_ref)
P = neurons[:N_P]
S = neurons[N_P:N_P+N_S]
B = neurons[N_P+N_S:]

# define synapses

B_to_P = Synapses(B,P,inh_plast,on_pre='''Xpr += 1
                                 W = clip(W + eta*(Xpo-alpha), Wmin, Wmax)
                                 gB += W*30*nS''',on_post='''Xpo += 1
                                 W = clip(W + Xpr*eta, Wmin, Wmax)''')
S_to_P = Synapses(S,P,inh_plast,on_pre='''Xpr += 1
                                 W = clip(W + eta*(Xpo-alpha), Wmin, Wmax)
                                 gS += W*30*nS''',on_post='''Xpo += 1
                                 W = clip(W + Xpr*eta, Wmin, Wmax)''')
S_to_S = Synapses(S,S,on_pre='gS += 30*nS')
B_to_B = Synapses(B,B,on_pre='gB += 30*nS')
S_to_B = Synapses(S,B,on_pre='gS += 30*nS')
B_to_S = Synapses(B,S,on_pre='gB += 30*nS')
P_to_P = Synapses(P,P,on_pre='gP += 3*nS')
P_to_B = Synapses(P,B,on_pre='gP += 3*nS')
P_to_S = Synapses(P,S,on_pre='gP += 3*nS')

# implement random connectivity

B_to_P.connect(p=epsilon)
B_to_B.connect(p=epsilon)
P_to_P.connect(p=epsilon)
P_to_B.connect(p=epsilon)

# run simulations

act_neurons = neurons[0:N_P+N_B]
spikes = SpikeMonitor(act_neurons)
LFP = PopulationRateMonitor(act_neurons)
# pick excitatory neurons and record data from them
M = StateMonitor(act_neurons, True, record = randint(0, N_B+N_P, record_neurons))

# run with plasticity
net = Network(collect())
net.remove(S)
net.run(runtime, report = 'text')
net.store('balanced P-B')
st = spikes.spike_trains()

rates = LFP.smooth_rate(window='flat', width=200.1*ms)/Hz
print(rates[-10100:-10000])

# plot

fig,ax = plt.subplots(2,1,figsize=(10,15))
ax[0].plot(spikes.t/ms,spikes.i,'.',markersize=1)
ax[0].set_xlabel('time (ms)')
ax[0].set_ylabel('Neuron')
ax[0].set_xlim([(runtime-100*ms)/ms, runtime/ms])

ax[1].plot(LFP.t/ms, rates)
ax[1].set_xlabel('time (ms)')
ax[1].set_ylabel('Firing Rate (Hz)')
ax[1].set_title('Firing Rate')
plt.show()

# regularity
print('ICI CV with plasticity = %f' % util.regularity(st, runtime-measure_time))

# correlation
time = np.round(np.linspace(runtime/ms-measure_time/ms, runtime/ms, 10 * measure_time/ms+1), 1)
corrs = np.ones((record_neurons,1))

for k in range(record_neurons):
    i,j = np.random.choice(list(st.keys()), 2)
    cr = util.correlation(time, st[i], st[j])
    corrs[k] = cr
print('Correlation coefficient averaged for 50 neurons = %f ' % np.nanmean(corrs))