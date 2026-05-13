import sys
from pathlib import Path
import numpy as np

# allow imports from analysis/
sys.path.append(str(Path(__file__).resolve().parents[1]))



from analysis.analysis import (
    epochs_ecg,
    epochs_breath_peaks,
    epochs_breath_troughs,
    obtain_spikes,
    combining_spike_and_epoch,
    psth_from_overlap,
    average_aligned_epochs
)
from analysis import data_loading


def run_dashboard_analysis(df):
    fs = 20000
    spikes = obtain_spikes(df, fs)
    spike_times = spikes

    ecg_epochs = epochs_ecg(df, fs)
    breath_peak_epochs = epochs_breath_peaks(df, fs)
    breath_trough_epochs = epochs_breath_troughs(df, fs)

    ecg_overlap = combining_spike_and_epoch(spikes, ecg_epochs)
    breath_peak_overlap = combining_spike_and_epoch(spikes, breath_peak_epochs)
    breath_trough_overlap = combining_spike_and_epoch(spikes, breath_trough_epochs)

    t_ecg, rate_ecg, sem_ecg = psth_from_overlap(ecg_overlap, ecg_epochs, bin_width=0.005)
    t_breath_peak, rate_breath_peak, sem_breath_peak = psth_from_overlap(breath_peak_overlap, breath_peak_epochs, bin_width=0.1)
    t_breath_trough, rate_breath_trough, sem_breath_trough = psth_from_overlap(breath_trough_overlap, breath_trough_epochs, bin_width=0.1)

    signal_t = df["time"].to_numpy(float)
    raw_ecg = df["ecg"].to_numpy(float)
    raw_resp = df["resp"].to_numpy(float)

    fs = data_loading.calculate_freq(df)

    ecglow = raw_ecg - np.mean(raw_ecg)
    ecg_filtered = data_loading.lowpass(ecglow, fs=fs, cutoff=40)

    ecg_rpeak_times = data_loading.detecting_rpeaks_ecg(fs, df)
    peak_indices = np.searchsorted(signal_t, ecg_rpeak_times)
    peak_indices = peak_indices[peak_indices < len(raw_ecg)]
    ecgr_peak_values = raw_ecg[peak_indices]

    t_ecg_signal, ecg_mean_ecg = average_aligned_epochs(raw_ecg, signal_t, ecg_epochs, bin_width=0.005)

    t_breath_peak_signal, ecg_mean_breath_peak = average_aligned_epochs(raw_ecg, signal_t, breath_peak_epochs, bin_width=0.1)
    _, resp_mean_breath_peak = average_aligned_epochs(raw_resp, signal_t, breath_peak_epochs, bin_width=0.1)

    t_breath_trough_signal, ecg_mean_breath_trough = average_aligned_epochs(raw_ecg, signal_t, breath_trough_epochs, bin_width=0.1)
    _, resp_mean_breath_trough = average_aligned_epochs(raw_resp, signal_t, breath_trough_epochs, bin_width=0.1)

    t_raw = signal_t
    ecg = raw_ecg

    window_half = 5.0
    center_time = t_raw[len(t_raw) // 2]
    window_mask = (t_raw >= center_time - window_half) & (t_raw <= center_time + window_half)

    t_window = t_raw[window_mask] - center_time
    ecg_window = ecg[window_mask]

    spike_window = spikes[
        (spikes >= center_time - window_half) &
        (spikes <= center_time + window_half)
    ] if t_window.size else np.array([])

    window_bin_width = 0.1
    window_bins = np.arange(
        center_time - window_half,
        center_time + window_half + window_bin_width,
        window_bin_width
    ) if t_window.size else np.array([])

    window_counts = np.histogram(spike_window, bins=window_bins)[0] if t_window.size else np.array([])
    window_rate = window_counts / window_bin_width if window_bins.size else np.array([])
    t_window_rate = ((window_bins[:-1] + window_bins[1:]) / 2) - center_time if window_bins.size else np.array([])
    window_sem = np.zeros_like(window_rate)

    return {
        "spike_times": spike_times,
        "t_ecg": t_ecg,
        "rate_ecg": rate_ecg,
        "sem_ecg": sem_ecg,
        "t_breath_peak": t_breath_peak,
        "rate_breath_peak": rate_breath_peak,
        "sem_breath_peak": sem_breath_peak,
        "t_breath_trough": t_breath_trough,
        "rate_breath_trough": rate_breath_trough,
        "sem_breath_trough": sem_breath_trough,
        "t_ecg_signal": t_ecg_signal,
        "ecg_mean_ecg": ecg_mean_ecg,
        "t_breath_peak_signal": t_breath_peak_signal,
        "resp_mean_breath_peak": resp_mean_breath_peak,
        "t_breath_trough_signal": t_breath_trough_signal,
        "resp_mean_breath_trough": resp_mean_breath_trough,
        "t_raw": t_raw,
        "ecg_filtered": ecg_filtered,
        "ecg_rpeak_times": ecg_rpeak_times,
        "ecgr_peak_values": ecgr_peak_values,
        "t_window": t_window,
        "t_window_rate": t_window_rate,
        "window_rate": window_rate,
        "window_sem": window_sem,
        "ecg_window": ecg_window,
    }