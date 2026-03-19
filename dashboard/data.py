import sys
from pathlib import Path

# allow imports from analysis/
sys.path.append(str(Path(__file__).resolve().parents[1]))



from analysis.analysis import (
    epochs_ecg,
    epochs_breath_peaks,
    epochs_breath_troughs,
    obtain_spikes,
    combining_spike_and_epoch,
    psth_from_overlap,
    breath_data,
    ecg_data,
    time,
    get_aligned_metric_data,
)


# Run analysis once
spikes = obtain_spikes()

ecg_epochs = epochs_ecg()
breath_peak_epochs = epochs_breath_peaks()
breath_trough_epochs = epochs_breath_troughs()

ecg_overlap = combining_spike_and_epoch(spikes, ecg_epochs)
breath_peak_overlap = combining_spike_and_epoch(spikes, breath_peak_epochs)
breath_trough_overlap = combining_spike_and_epoch(spikes, breath_trough_epochs)

t_ecg, rate_ecg, sem_ecg = psth_from_overlap(
    ecg_overlap, ecg_epochs, bin_width=0.005
)

t_breath_peak, rate_breath_peak, sem_breath_peak = psth_from_overlap(
    breath_peak_overlap, breath_peak_epochs, bin_width=0.1
)

t_breath_trough, rate_breath_trough, sem_breath_trough = psth_from_overlap(
    breath_trough_overlap, breath_trough_epochs, bin_width=0.1
)

# time, breath, ecg, nerve = triple_graph_data()
t_raw, t_rate, neuron_firing_rate, respiration, ecg, nerve = get_aligned_metric_data()