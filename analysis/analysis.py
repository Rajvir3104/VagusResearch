from analysis import data_loading
# import data_loading
import numpy as np
import matplotlib.pyplot as plt

df = data_loading.load_parquet()
fs = data_loading.calculate_freq(df)

def time():
    time = df["time"].to_numpy(float)

    return time
    
def breath_data():
    result = df["resp"].to_numpy(float)
    return result

def ecg_data():
    ecg = df["ecg"].to_numpy(float)

    return ecg

def nerve_data():
    nerve_filt = data_loading.pre_process(df, fs)
    thr = data_loading.compute_threshold(nerve_filt)
    
    spike_times = data_loading.detect_spikes(thr, nerve_filt, fs, t=df["time"].values)

    return spike_times


def get_aligned_metric_data(
    bin_width=0.05,
    max_time=600
):
    # Raw signals
    t_raw = df["time"].to_numpy(float)
    mask = t_raw <= max_time
    t_raw = t_raw[mask]

    respiration = df["resp"].to_numpy(float)[mask]
    ecg = df["ecg"].to_numpy(float)[mask]
    nerve = df["nerve"].to_numpy(float)[mask]

    # Spike detection
    nerve_filt = data_loading.pre_process(df, fs)
    thr = data_loading.compute_threshold(nerve_filt)
    spike_times = np.asarray(
        data_loading.detect_spikes(thr, nerve_filt, fs, t=df["time"].values),
        dtype=float
    )

    # Keep spikes in plotted range
    spike_times = spike_times[(spike_times >= t_raw[0]) & (spike_times <= t_raw[-1])]

    # Bin spikes into firing rate
    bins = np.arange(t_raw[0], t_raw[-1] + bin_width, bin_width)
    spike_counts, _ = np.histogram(spike_times, bins=bins)
    neuron_firing_rate = spike_counts / bin_width
    t_rate = (bins[:-1] + bins[1:]) / 2

    return t_raw, t_rate, neuron_firing_rate, respiration, ecg, nerve