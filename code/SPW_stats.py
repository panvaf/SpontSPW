# compute statistics of SPW events

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats.stats import pearsonr
from scipy import signal

thres = 15   # Hz
dt = 0.0001  # sec

fr = np.load('fr.npz')
fr_B = fr['fr_B']
fr_P = fr['fr_P']
fr_S = fr['fr_S']
time = fr['time']

fig,ax = plt.subplots(3,1,figsize=(10,15))
ax[0].plot(time,fr_P)
ax[0].set_ylabel('Freq (Hz)')
#ax[0].set_xlabel('time (sec)')
ax[0].set_title('Firing Rate of P cells')
#ax[0].set_xlim([25,30])

ax[1].plot(time,fr_S)
ax[1].set_ylabel('Freq (Hz)')
ax[1].set_title('Firing Rate of S cells')

ax[2].plot(time,fr_B)
ax[2].set_xlabel('time (sec)')
ax[2].set_ylabel('Freq (Hz)')
ax[2].set_title('Firing Rate of B cells')

plt.show()

start = np.zeros(time.size, dtype=bool)
end = np.zeros(time.size, dtype=bool)

# start and end of event
sw1 = 1
sw2 = 0
for i in range(time.size):
    if fr_P[i]>thres:
        sw2 = 1
        if sw1 == 1:
            start[i] = True
            sw1 = 0
        else:
            continue
    else:
        sw1 = 1
        if sw2 == 1:
            end[i] = True
            sw2 = 0
        else:
            continue

# event times
time_s = time[start]
time_e = time[end]
time_peak = np.zeros(time_s.size)
peaks = np.zeros(time_s.size)

for i in range(time_s.size):
    peaks[i] = max(fr_P[int(time_s[i]/dt):int(time_e[i]/dt+1)])
    time_peak[i] = np.argmax(fr_P[int(time_s[i]/dt):int(time_e[i]/dt+1)])*dt

time_peak += time_s
duration = time_e - time_s

# throw away problematic measurements

mask = np.ones(len(peaks), dtype=bool)
# fluctuations in the firing rate values cause the FR to go up and down 15 Hz
# while ascending 
mask[peaks<20] = False
# some events can happen before the previous event has terminated if the overall
# activity is too high
mask[duration>0.45] = False
peaks = peaks[mask]
time_s = time_s[mask]
time_e = time_e[mask]
time_peak = time_peak[mask]
duration = duration[mask]

# time differences
ISI = [time_peak[i] - time_peak[i -1] for i in range(len(time_peak))]
ISI = ISI[1:]

fig,ax = plt.subplots(1,3,figsize=(15,5))

ax[0].hist(ISI, bins=np.linspace(0, max(ISI), 100),normed='True')
ax[0].set_xlabel('Interspike interval (sec)')
ax[0].set_title('ISI distribution')

ax[1].hist(peaks, bins=np.linspace(min(peaks), max(peaks), 100),normed='True')
ax[1].set_xlabel('Peak value (Hz)')
ax[1].set_title('Peak value distribution')

ax[2].hist(duration, bins=np.linspace(min(duration), max(duration), 100),normed='True')
ax[2].set_xlabel('Event duration (sec)')
ax[2].set_title('Event duration distribution')

plt.show()

# find correlations

dur_amp = pearsonr(duration,peaks)
ISIlag_dur = pearsonr(ISI,duration[0:len(duration)-1])
ISI_durlag = pearsonr(ISI,duration[1:])
ISIlag_amp = pearsonr(ISI,peaks[0:len(duration)-1])
ISI_amplag = pearsonr(ISI,peaks[1:])

# power spectral density of event occurences

time = np.round(np.arange(0,3000,.1),1)
events = np.asarray([i in np.round(time_peak,1) and 1 or 0 for i in time], dtype='float')
f, Pxx_den = signal.periodogram(events, 10**4)
plt.semilogy(f, Pxx_den)
plt.ylim([1e-10, 1e-4])
plt.xlim([0, 1e2])
plt.xlabel('frequency [Hz]')
plt.ylabel('Spectral density [1/Hz]')
plt.show()

'''
# fit distributions
x = np.arange(0,15,.1)
size = len(x)
h = plt.hist(fr_P, bins=range(50), color='w')

dist_names = ['gamma', 'lognorm', 'weibull_max', 'norm', 'pareto']

for dist_name in dist_names:
    dist = getattr(scipy.stats, dist_name)
    param = dist.fit(ISI)
    pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1]) * size
    plt.plot(pdf_fitted, label=dist_name)
    plt.xlim(0,15)
plt.legend(loc='upper right')
plt.show()
'''