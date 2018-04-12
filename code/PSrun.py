# used to run already trained network PS

import utilities as util
import matplotlib.pyplot as plt
import numpy as np
from brian2 import *

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

tauE = 5*ms      # Decay constant of AMPA conductance                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           stant of AMPA-type conductance
tauI = 8*ms      # Decay constant of GABA-type conductance for SOM cells
delay = 1.5*ms   # Synaptic delay, here modeled as transmission delay (same effect)

# Network parameters

Ne = 8000            # Size of excitatory population E
Ni = 1000            # Size of inhibitory population I

# Simulation parameters

runtime = 5*second     # Running time (neuron time)
trec = 100*ms          # Time window to measure spikes
LFP_width = 200.1*ms   # Window for smoothing LFP rates

# Define equations

# define model of neurons

neuron_model='''dV/dt= -(V-Vr)/tau + (- gE*(V-Ve) - gI*(V-Vi) + Ib)/(gl*tau) : volt (unless refractory)
dgE/dt = -gE/tauE : siemens
dgI/dt = -gI/tauI : siemens'''

# define plasticity model

inh_plast = 'W : 1'

# implement

# implement neurons

neurons = NeuronGroup(Ne+Ni,neuron_model,'euler',threshold='V>Vt',reset='V=Vr',refractory=tau_ref)
E = neurons[:Ne]
I = neurons[Ne:]

# define synapses

i_to_e = Synapses(I,E,inh_plast,delay=delay,on_pre='gI += W*.4*nS')
i_to_i = Synapses(I,I,delay=delay,on_pre='gI += .4*nS')
e_to_e = Synapses(E,E,delay=delay,on_pre='gE += .1*nS')
e_to_i = Synapses(E,I,delay=delay,on_pre='gE += .1*nS')

# implement connectivity

PScon = np.load('PSconnections.npz')
e_to_e.connect(i=PScon['PPi'],j=PScon['PPj'])
i_to_e.connect(i=PScon['PSi'],j=PScon['PSj'])
e_to_i.connect(i=PScon['SPi'],j=PScon['SPj'])
i_to_i.connect(i=PScon['SSi'],j=PScon['SSj'])

# initialize

i_to_e.W = PScon['W']
neurons.V = np.random.randint(Vr/mV,Vt/mV,Ne+Ni)*mV  # random initial conditions 
#neurons.V = PScon['V']*volt
#neurons.gE = PScon['gE']*siemens
#neurons.gI = PScon['gI']*siemens

# run simulations

spikes = SpikeMonitor(neurons)
LFPe = PopulationRateMonitor(neurons[:Ne])
LFPi = PopulationRateMonitor(neurons[Ne:])
# pick excitatory neurons and record data from them

# should we need a state monitor
# M = StateMonitor(neurons, True, record = randint(0, Ne, record_neurons)) 

# run with plasticity
net = Network(collect())
net.run(runtime+trec, report = 'text')
# net.store('learned')
#st = spikes.spike_trains()
weights = np.asarray(i_to_e.W)


# plot

fig,ax = plt.subplots(4,1,figsize=(10,15))
ax[0].plot(LFPe.t/ms/1000/60, LFPe.smooth_rate(window = 'flat',width=LFP_width)/Hz)
ax[0].set_xlabel('time (min)')
ax[0].set_ylabel('Freq (Hz)')
#ax[0].set_ylim([0,30])
ax[0].set_title('Firing Rate')

ax[1].plot(LFPi.t/ms/1000/60, LFPi.smooth_rate(window = 'flat',width=LFP_width)/Hz)
ax[1].set_xlabel('time (min)')
ax[1].set_ylabel('Freq (Hz)')
#ax[1].set_ylim([0,200])
ax[1].set_title('Firing Rate')

ax[2].hist(weights, bins=np.linspace(min(weights), max(weights), 100),normed='True')
ax[2].set_xlabel('weight values')
ax[2].set_title('Weights probability distribution')

ax[3].plot(spikes.t/ms,spikes.i,'.',markersize=1)
ax[3].set_xlabel('time (ms)')
ax[3].set_ylabel('Neuron')
ax[3].set_xlim([(runtime)/ms, (runtime + trec)/ms])

plt.show()