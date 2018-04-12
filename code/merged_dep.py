# create full merged network and add depression mechanism in B to S

import sys
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

tauP = 5*ms      # Decay constant of AMPA-type conductance - P cells
tauB = 4*ms      # Decay constant of GABA-type conductance - B cells
tauS = 8*ms      # Decay constant of GABA-type conductance - S cells
delay = 1.5*ms   # Synaptic delay, here modeled as transmission delay (same effect)

# Network parameters

N_P = 8000           # Size of excitatory population P
N_S = 1000           # Size of inhibitory population S
N_B = 1000           # Size of inhibitory population B
epsilon = 0.08       # Probability of any connection

# Depression mechanisms

tauD = 100*ms      # Time constant of synaptic depression
delta = 0.1        # strength of depression
# 100, 0.2, 1, .6
# 150, 0.1, 1, .6 gg
# 100, 0.1, 1.05, .65
# oscillating: 150ms, 0.07, 0.6, 1
# stochastic nice: 100,0.1,0.65,1.05
# and also 150,0.08,0.65,1.05 aproximately 300 ms for the whole event, near the to for 100 ms for both
# nice in-between states: 92,0.1,0.65,1.05
# deterministic (kinda): 1000, 0.01, 0.65,1.05

# Simulation parameters

runtime = 5*second     # Running time (neuron time)
trec = 100*ms           # Time window to measure spikes
FR_width = 50.1*ms      # Window for smoothing firing rates
record_neurons = 100    # To compute correlation
measure_time = 5*second # Window to compute measures

# Define equations

# define model of neurons

neuron_model='''dV/dt= -(V-Vr)/tau + (- gP*(V-Ve) - gS*(V-Vi) - gB*(V-Vi) + Ib)/(gl*tau) : volt (unless refractory)
dgP/dt = -gP/tauP : siemens
dgB/dt = -gB/tauB : siemens
dgS/dt = -gS/tauS : siemens'''

# define plasticity model

inh_plast = 'W : 1'

# define depression mechanisms

syn_dep = 'dD/dt = (1 - D)/tauD : 1 (event-driven)'

# implement neurons

neurons = NeuronGroup(N_P+N_B+N_S,neuron_model,'euler',threshold='V>Vt',reset='V=Vr',refractory=tau_ref)
P = neurons[:N_P]
S = neurons[N_P:N_P+N_S]
B = neurons[N_P+N_S:]

# define synapses

B_to_P = Synapses(B,P,inh_plast,delay=delay,on_pre='gB += W*.4*nS')
S_to_P = Synapses(S,P,inh_plast,delay=delay,on_pre='gS += W*.4*nS')
S_to_S = Synapses(S,S,delay=delay,on_pre='gS += .4*nS')
B_to_B = Synapses(B,B,delay=delay,on_pre='gB += .4*nS')
P_to_P = Synapses(P,P,delay=delay,on_pre='gP += .1*nS')
P_to_B = Synapses(P,B,delay=delay,on_pre='gP += .1*nS')
P_to_S = Synapses(P,S,delay=delay,on_pre='gP += .1*nS')

# new synapses
S_to_B = Synapses(S,B,delay=delay,on_pre='gS += .65*nS')
B_to_S = Synapses(B,S,syn_dep,delay=delay,on_pre='''gB += D*1.05*nS
                  D = clip(D - delta*D,0,1)''')

# implement connectivity

PBcon = np.load('PBconnections.npz')
PScon = np.load('PSconnections.npz')
BS = np.load('BS.npz')
SB = np.load('SB.npz')

if not (np.array_equal(PBcon['PPi'],PScon['PPi']) and np.array_equal(PBcon['PPj'],PScon['PPj'])):
    sys.exit("Not the same connectivity in P cells in the 2 networks")

P_to_P.connect(i=PBcon['PPi'],j=PBcon['PPj'])  # taken from B, if everything's ok it's the same in S
S_to_P.connect(i=PScon['PSi'],j=PScon['PSj'])
P_to_S.connect(i=PScon['SPi'],j=PScon['SPj'])
S_to_S.connect(i=PScon['SSi'],j=PScon['SSj'])
B_to_P.connect(i=PBcon['PBi'],j=PBcon['PBj'])
P_to_B.connect(i=PBcon['BPi'],j=PBcon['BPj'])
B_to_B.connect(i=PBcon['BBi'],j=PBcon['BBj'])

# new connections
S_to_B.connect(p=epsilon)
B_to_S.connect(p=epsilon)

# initialize
S_to_P.W = PScon['W']
B_to_P.W = PBcon['W']
neurons.V = np.random.randint(Vr/mV,Vt/mV,N_P+N_S+N_B)*mV

# check if they can switch states. It works!

#neurons.V = BS['V']*volt
#neurons.gS = BS['gS']*siemens
#neurons.gP = BS['gP']*siemens
#neurons.gB = BS['gB']*siemens

fig,ax = plt.subplots(1,2,figsize=(10,5))

ax[0].hist(PScon['W'], bins=np.linspace(min(PScon['W']), max(PScon['W']), 100),normed='True')
ax[0].set_xlabel('weight values')
ax[0].set_title('S to P weights probability distribution')

ax[1].hist(PBcon['W'], bins=np.linspace(min(PBcon['W']), max(PBcon['W']), 100),normed='True')
ax[1].set_xlabel('weight values')
ax[1].set_title('B to P weights probability distribution')

plt.show()

# run simulation

spikes = SpikeMonitor(neurons, record = 'True')
FR_P = PopulationRateMonitor(neurons[:N_P])
FR_S = PopulationRateMonitor(neurons[N_P:N_P+N_S])
FR_B = PopulationRateMonitor(neurons[N_P+N_S:])

net = Network(collect())
net.run(runtime+trec, report = 'text')

fig,ax = plt.subplots(4,1,figsize=(10,15))
ax[0].plot(FR_P.t/ms/1000/60, FR_P.smooth_rate(window = 'flat',width=FR_width)/Hz)
ax[0].set_xlabel('time (min)')
ax[0].set_ylabel('Freq (Hz)')
#ax[0].set_ylim([0,30])
ax[0].set_title('Firing Rate of P cells')

ax[1].plot(FR_S.t/ms/1000/60, FR_S.smooth_rate(window = 'flat',width=FR_width)/Hz)
ax[1].set_xlabel('time (min)')
ax[1].set_ylabel('Freq (Hz)')
#ax[1].set_ylim([0,200])
ax[1].set_title('Firing Rate of S cells')

ax[2].plot(FR_B.t/ms/1000/60, FR_B.smooth_rate(window = 'flat',width=FR_width)/Hz)
ax[2].set_xlabel('time (min)')
ax[2].set_ylabel('Freq (Hz)')
#ax[2].set_ylim([0,200])
ax[2].set_title('Firing Rate of B cells')

ax[3].plot(spikes.t/ms,spikes.i,'.',markersize=1)
ax[3].set_xlabel('time (ms)')
ax[3].set_ylabel('Neuron')
ax[3].set_xlim([(runtime)/ms, (runtime + trec)/ms])

plt.show()

fr_P = FR_P.smooth_rate(window = 'flat',width=FR_width)/Hz
fr_B = FR_B.smooth_rate(window = 'flat',width=FR_width)/Hz
fr_S = FR_S.smooth_rate(window = 'flat',width=FR_width)/Hz
time = FR_P.t/ms/1000
np.savez('fr',fr_P=fr_P,fr_S=fr_S,fr_B=fr_B,time=time)

'''
# compute measures of synchronicity and regularity

st = spikes.spike_trains()

# ISI CV
print('ICI CV for pyramidal cells = %f' % util.regularity(st,[0,N_P],0))

# correlation
time = np.round(np.linspace(runtime/ms-measure_time/ms, runtime/ms, 10 * measure_time/ms+1), 1)
corrs = np.empty((record_neurons,1))
corrs[:] = nan
keys = np.arange(0,N_P)

for k in range(record_neurons):
    i,j = np.random.choice(keys, 2)
    cr = util.correlation(time, st[i], st[j])
    corrs[k] = cr
print('Correlation coefficient averaged for 100 pyramidal neuron pairs = %f ' % np.nanmean(corrs))
'''
