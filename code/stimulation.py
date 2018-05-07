# implements first part of question 2
# create different neuron populations in order to stimulate with currents
# use strict bistable network

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
I = 200*pA       # Background current to each cell
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

# Simulation parameters

runinit = 2*second     # Running time (neuron time)
runafter = 2*second    # Running time (neuron time)
trec = 100*ms          # Time window to measure spikes
FR_width = 50.1*ms    # Window for smoothing firing rates
record_neurons = 100   # To compute correlation
measure_time = 5*second
Ip = 0*pA              # stimulation currents
Ib = 0*pA
Is = 0*pA
stim_dur = 20*ms     # stimulation duration
eta = .02             # percentage of cells stimulated

# B cells: 90 nA, 100 ms, eta = .3
# P cells: 200 nA, 100 ms, eta = .05
# S cells: -140 nA, 100 ms, eta = .3 Here transient synchronicity is inevitable

# shorter stimulation duration
# really difficult to avoid synchronicity effects when using unimodal currents
# P cells: 850 nA, 20 ms, eta = .02, jumps a lot over. completely random. sometimes nothing,sometimes synchronous, sometimes switch
# 
# 

# avoiding synchonicity can be quite tricky. sometimes it does not elicit, sometimes
# in has quite a lot of synchonicity when you stimulate (its natural and inevitable, also expected, if it leaves afterwards its ok)
    
# Depression mechanisms

tauD = 100*ms      # Time constant of synaptic depression
delta = 0.01      # strength of depression
# 100, 0.2, 1, .6
# 150, 0.1, 1, .6 gg
# 100, 0.1, 1.05, .65
# slow transition: 100, .05 (.025 for P), 1, .8

# Define equations

# define model of neurons

neuron_P='''dV/dt= -(V-Vr)/tau + (- gP*(V-Ve) - gS*(V-Vi) - gB*(V-Vi) + I + w*Ip)/(gl*tau) : volt (unless refractory)
dgP/dt = -gP/tauP : siemens
dgB/dt = -gB/tauB : siemens
dgS/dt = -gS/tauS : siemens
w : 1'''

neuron_B='''dV/dt= -(V-Vr)/tau + (- gP*(V-Ve) - gS*(V-Vi) - gB*(V-Vi) + I + w*Ib)/(gl*tau) : volt (unless refractory)
dgP/dt = -gP/tauP : siemens
dgB/dt = -gB/tauB : siemens
dgS/dt = -gS/tauS : siemens
w : 1'''

neuron_S='''dV/dt= -(V-Vr)/tau + (- gP*(V-Ve) - gS*(V-Vi) - gB*(V-Vi) + I + w*Is)/(gl*tau) : volt (unless refractory)
dgP/dt = -gP/tauP : siemens
dgB/dt = -gB/tauB : siemens
dgS/dt = -gS/tauS : siemens
w : 1'''

# define plasticity model

inh_plast = 'W : 1'

# define depression mechanisms

syn_dep = 'dD/dt = (1 - D)/tauD : 1 (event-driven)'

# implement neurons

P = NeuronGroup(N_P,neuron_P,'euler',threshold='V>Vt',reset='V=Vr',refractory=tau_ref)
B = NeuronGroup(N_B,neuron_B,'euler',threshold='V>Vt',reset='V=Vr',refractory=tau_ref)
S = NeuronGroup(N_S,neuron_S,'euler',threshold='V>Vt',reset='V=Vr',refractory=tau_ref)

# define synapses

B_to_P = Synapses(B,P,inh_plast,delay=delay,on_pre='gB += W*.4*nS')
S_to_P = Synapses(S,P,inh_plast,delay=delay,on_pre='gS += W*.4*nS')
S_to_S = Synapses(S,S,delay=delay,on_pre='gS += .4*nS')
B_to_B = Synapses(B,B,delay=delay,on_pre='gB += .4*nS')
P_to_P = Synapses(P,P,delay=delay,on_pre='gP += .1*nS')
P_to_B = Synapses(P,B,delay=delay,on_pre='gP += .1*nS')
P_to_S = Synapses(P,S,delay=delay,on_pre='gP += .1*nS')

# new synapses
S_to_B = Synapses(S,B,delay=delay,on_pre='gS += .8*nS')
B_to_S = Synapses(B,S,syn_dep,delay=delay,on_pre='''gB += D*1*nS
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
P.V = np.random.randint(Vr/mV,Vt/mV,N_P)*mV
S.V = np.random.randint(Vr/mV,Vt/mV,N_S)*mV
B.V = np.random.randint(Vr/mV,Vt/mV,N_B)*mV
P.w = eta>np.random.rand(N_P)     # randomly choose which are stimulated

# initialize with other conditions, so that we can start with any state we desire

V = SB['V']*volt
gS = SB['gS']*siemens
gP = SB['gP']*siemens
gB = SB['gB']*siemens

[P.V, P.gS, P.gP, P.gB] = [V[:N_P], gS[:N_P], gP[:N_P], gB[:N_P]]
[S.V, S.gS, S.gP, S.gB] = [V[N_P:N_P+N_S], gS[N_P:N_P+N_S], gP[N_P:N_P+N_S], gB[N_P:N_P+N_S]]
[B.V, B.gS, B.gP, B.gB] = [V[N_P+N_S:], gS[N_P+N_S:], gP[N_P+N_S:], gB[N_P+N_S:]]

# plot weight distributions

fig,ax = plt.subplots(1,2,figsize=(10,5))

ax[0].hist(PScon['W'], bins=np.linspace(min(PScon['W']), max(PScon['W']), 100),normed='True')
ax[0].set_xlabel('weight values')
ax[0].set_title('S to P weights probability distribution')

ax[1].hist(PBcon['W'], bins=np.linspace(min(PBcon['W']), max(PBcon['W']), 100),normed='True')
ax[1].set_xlabel('weight values')
ax[1].set_title('B to P weights probability distribution')

plt.show()

# run simulation

spikes = SpikeMonitor(P, record = 'True')  # need to have 3 spike monitors
FR_P = PopulationRateMonitor(P)
FR_S = PopulationRateMonitor(S)
FR_B = PopulationRateMonitor(B)

net = Network(collect())
net.run(runinit, report = 'text')

Ip = 850*pA
net.run(stim_dur, report = 'text')  

Ip = 0*pA
net.run(runafter, report = 'text')

fig,ax = plt.subplots(4,1,figsize=(10,15))
ax[0].plot(FR_P.t/second, FR_P.smooth_rate(window = 'flat',width=FR_width)/Hz)
ax[0].set_ylabel('Freq (Hz)')
ax[0].set_title('Firing Rate of P cells')

ax[1].plot(FR_S.t/second, FR_S.smooth_rate(window = 'flat',width=FR_width)/Hz)
ax[1].set_ylabel('Freq (Hz)')
ax[1].set_title('Firing Rate of S cells')

ax[2].plot(FR_B.t/second, FR_B.smooth_rate(window = 'flat',width=FR_width)/Hz)
ax[2].set_ylabel('Freq (Hz)')
ax[2].set_title('Firing Rate of B cells')

ax[3].plot(spikes.t/second,spikes.i,'.',markersize=1)
ax[3].set_xlabel('time (sec)')
ax[3].set_ylabel('Neuron')
ax[3].set_xlim([(runinit)/second, (runinit + stim_dur)/second])
ax[3].set_title('Spiking of P cells during stimulation')

plt.show()

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