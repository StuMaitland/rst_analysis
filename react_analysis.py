import pandas as pd
import numpy as np
import plotly.express as px
import sys
import getopt
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import signal

Fs = 2000

inputfile = '/Users/stuartbman/GitHub/rst_analysis/data/2021-07-22-Sruti/start_react_2021-07-22 15-38-05.481453.txt'
results = []
trial_types = []
emgs = []
t = 0


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def make_sweep(parts, Fs, ):
    sweep = list(map(int, parts))
    # sweep = [abs(x) for x in sweep]
    b_notch, a_notch = signal.iirnotch(50, 30, Fs)
    y_notched = signal.filtfilt(b_notch, a_notch, sweep)
    b, a = butter_bandpass(100, 500, Fs, order=5)
    y = signal.lfilter(b, a, y_notched)
    return y


def start_valid_island(a, thresh, window_size):
    m = a < thresh
    me = np.r_[False, m, False]
    idx = np.flatnonzero(me[:-1] != me[1:])
    lens = idx[1::2] - idx[::2]
    return idx[::2][(lens >= window_size).argmax()]


def window_rms(a, window_size):
    a2 = np.power(a, 2)
    window = np.ones(window_size) / float(window_size)
    return np.sqrt(np.convolve(a2, window, 'valid'))


file = open(inputfile)
for line in file:

    parts = line.split(',')
    div = parts.index('emg2')
    trialtype = parts[0]
    trial_types.append(trialtype)
    emg1 = make_sweep(parts[2:div], Fs)
    emg2 = make_sweep(parts[div + 1:], Fs)

    #emg = window_rms(emg2, 100)
    emg = emg1
    if t == 0:
        t = range(0, len(emg))
        t = [(a / Fs) - 0.1 for a in t]
    plt.plot(t,emg)
    emgs.append(emg)
    baseline = emg[0:int(0.1 * Fs)]
    min_dur = int(0.005 * Fs)  # Number of samples above threshold to count for a reaction time
    st_lim = np.mean(baseline) + np.std(baseline)
    result = start_valid_island(emg1, st_lim, min_dur)

    results.append(result)

plt.show()
df = pd.DataFrame()

df['trials'] = trial_types
df['rt'] = results

df.to_csv('test.csv')
