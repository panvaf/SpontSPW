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

#- \tau_{GABA_B} = 4 ms = GABAergic synaptic time constant(B cells) 
#- \tau_{GABA_S} = 8 ms = GABAergic synaptic time constant(S cells) 
#- \tau_{ampa} = 5 ms = Glutamatergic synaptic time constant(B cells) 

# Synapse Model

tauE = 5*ms      # Decay constant of AMPA-type conductance
tauI = 10*ms     # Decay constant of GABA-type conductance

# Plasticity Model

tauSTDP = 20*ms         # Decay constant of (pre and post) synaptic trace
Wmin = 0                # Minimum inhibitory synaptic weight
Wmax = 300              # Maximum inhibitory synaptic weight
rho0 = 20*Hz            # Target frequency
alpha = 2*rho0*tauSTDP  # Presynaptic offset
eta = 1e-2              # Learning rate

# Network parameters

Ne = 8000           # Size of excitatory population E
Ni = 2000           # Size of inhibitory population I
epsilon = 0.02       # Probability of any connection

# Simulation parameters

runtime = 3*60*second # Running time (neuron time)
record_neurons = 50   # for ICI-CV and correlations
measure_time = 1000*ms # Time window to compute different measures

# Define equations

# define model of neurons

neuron_model='''dV/dt= -(V-Vr)/tau + (- gE*(V-Ve) - gI*(V-Vi) + Ib)/(gl*tau) : volt (unless refractory)
dgE/dt = -gE/tauE : siemens
dgI/dt = -gI/tauI : siemens'''

# define plasticity model

inh_plast = '''dXpr/dt=-Xpr/tauSTDP : 1 (event-driven)
dXpo/dt=-Xpo/tauSTDP : 1 (event-driven)
W : 1'''

# implement

# implement neurons

neurons = NeuronGroup(Ne+Ni,neuron_model,'euler',threshold='V>Vt',reset='V=Vr',refractory=tau_ref)
E = neurons[:Ne]
I = neurons[Ne:]

# define synapses

i_to_e = Synapses(I,E,inh_plast,on_pre='''Xpr += 1
                                 W = clip(W + eta*(Xpo-alpha), Wmin, Wmax)
                                 gI += W*30*nS''',on_post='''Xpo += 1
                                 W = clip(W + Xpr*eta, Wmin, Wmax)''')
i_to_i = Synapses(I,I,on_pre='gI += 30*nS')
e_to_other = Synapses(E,neurons,on_pre='gE += 3*nS')

# implement random connectivity

i_to_e.connect(p=epsilon)
i_to_i.connect(p=epsilon)
e_to_other.connect(p=epsilon)

# run simulations

spikes = SpikeMonitor(neurons)
LFP = PopulationRateMonitor(neurons)
# pick excitatory neurons and record data from them
M = StateMonitor(neurons, True, record = randint(0, Ne, record_neurons)) 

# run with plasticity
net = Network(collect())
net.run(runtime, report = 'text')
net.store('learned')
st = spikes.spike_trains()

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


# plot

fig,ax = plt.subplots(2,1,figsize=(10,15))
ax[0].plot(spikes.t/ms,spikes.i,'.',markersize=1)
ax[0].set_xlabel('time (ms)')
ax[0].set_ylabel('Neuron')
ax[0].set_xlim([(runtime-100*ms)/ms, runtime/ms])

ax[1].plot(LFP.t/ms, LFP.smooth_rate(window='flat', width=200.1*ms)/Hz)
ax[1].set_xlabel('time (ms)')
ax[1].set_ylabel('Firing Rate (Hz)')
ax[1].set_title('Firing Rate')
plt.show()