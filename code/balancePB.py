# create balanced network P-B

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

tauE = 5*ms      # Decay con                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            stant of AMPA-type conductance
tauI = 4*ms      # Decay constant of GABA-type conductance
delay = 1.5*ms   # Synaptic delay, here modeled as transmission delay (same effect)

# Plasticity Model

tauSTDP = 20*ms         # Decay constant of (pre and post) synaptic trace
Wmin = 0                # Minimum inhibitory synaptic weight
Wmax = 10               # Maximum inhibitory synaptic weight
rho0 = 20*Hz            # Target frequency
alpha = 2*rho0*tauSTDP  # Presynaptic offset
eta = 1e-4              # Learning rate

# Network parameters

Ne = 8000            # Size of excitatory population E
Ni = 1000            # Size of inhibitory population I
epsilon = 0.08       # Probability of any connection

# Simulation parameters

runtime = 10*60*second  # Running time (neuron time)
record_neurons = 50    # for ICI-CV and correlations
measure_time = 1000*ms # Time window to compute different measures (depracated)
trec = 100*ms          # Time window to measure spikes
LFP_width = 200.1*ms   # Window for smoothing LFP rates

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
                                 gI += W*.4*nS''',on_post='''Xpo += 1
                                 W = clip(W + Xpr*eta, Wmin, Wmax)''')
i_to_i = Synapses(I,I,on_pre='gI += .4*nS')
e_to_e = Synapses(E,E,on_pre='gE += .1*nS')
e_to_i = Synapses(E,I,on_pre='gE += .1*nS')

# implement random connectivity

i_to_e.connect(p=epsilon)
i_to_i.connect(p=0.2)
e_to_e.connect(p=0.04)
e_to_i.connect(p=epsilon)
neurons.V = np.random.randint(Vr/mV,Vt/mV,Ne+Ni)*mV

# run simulations

spikes = SpikeMonitor(neurons, record = 'False')   # do not record while learning
LFPe = PopulationRateMonitor(neurons[:Ne])
LFPi = PopulationRateMonitor(neurons[Ne:])
# pick excitatory neurons and record data from them

# should we need a state monitor
# M = StateMonitor(neurons, True, record = randint(0, Ne, record_neurons)) 

# run with plasticity
net = Network(collect())
net.run(runtime, report = 'text')
# net.store('learned')
#st = spikes.spike_trains()
weights = np.asarray(i_to_e.W)


# plot

fig,ax = plt.subplots(3,1,figsize=(10,15))
ax[0].plot(LFPe.t/ms/1000/60, LFPe.smooth_rate(window = 'flat',width=LFP_width)/Hz)
ax[0].set_xlabel('time (min)')
ax[0].set_ylabel('Freq (Hz)')
ax[0].set_ylim([0,30])
ax[0].set_title('Firing Rate')

ax[1].plot(LFPi.t/ms/1000/60, LFPi.smooth_rate(window = 'flat',width=LFP_width)/Hz)
ax[1].set_xlabel('time (min)')
ax[1].set_ylabel('Freq (Hz)')
ax[1].set_ylim([0,200])
ax[1].set_title('Firing Rate')

ax[2].hist(weights, bins=np.linspace(min(weights), max(weights), 100),normed='True')
ax[2].set_xlabel('weight values')
ax[2].set_ylabel('Occurences')
ax[2].set_title('Weights distribution')

plt.show()

# save network state to output file

PPi = e_to_e.i
PPj = e_to_e.j
PBi = i_to_e.i
PBj = i_to_e.j
BPi = e_to_i.i
BPj = e_to_i.j
BBi = i_to_i.i
BBj = i_to_i.j
W = i_to_e.W
V = neurons.V
gE = neurons.gE
gI = neurons.gI

np.savez('PBconnections',PPi=PPi,PPj=PPj,PBi=PBi,PBj=PBj,BPi=BPi,BPj=BPj,BBi=BBi,BBj=BBj,W=W,V=V,gE=gE,gI=gI)

'''
# record spiking behavior

spikes.record = True  # this restores all past spikes!
net.run(trec, report = 'text')

fig,ax = plt.subplots(1,1,figsize=(10,15))

ax.plot(spikes.t/ms,spikes.i,'.',markersize=1)
ax.set_xlabel('time (ms)')
ax.set_ylabel('Neuron')
ax.set_xlim([(runtime)/ms, (runtime + trec)/ms])

plt.show()
'''

'''
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
'''

# how often should SOM cells spike? 15 Hz and 6 Hz from poster, 

# pyramidals at 0.3 Hz (Priming of hippocampal population bursts by individual
# perisomatic-targeting interneurons)

# Chenkov: 5 Hz for exc rest, 100 peak, 20 for inh rest, 60 peak