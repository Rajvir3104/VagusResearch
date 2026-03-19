import dash
from dash import html, dcc

from data import (
    t_raw,
    t_rate,
    neuron_firing_rate,
    respiration,
    ecg,
    nerve
)
from figures import aligned_metrics_figure

dash.register_page(__name__, path="/triple")



ecg_plot = ecg


layout = html.Div([
    html.H2("Triple Aligned Graph"),
    dcc.Graph(
        figure=aligned_metrics_figure(
            t_raw = t_raw,
            t_rate = t_rate,
            neuron_firing_rate = neuron_firing_rate,
            respiration = respiration,
            ecg = ecg,
            nerve = nerve,
            neuron_firing_rate_name="Neuron Firing Rate",
            respiration_name="Respiration Metric",
            ecg_name="ECG Metric",
            nerve_name="Nerve Activity",
            x_label="Time relative to alignment event (s)",
        )
    ),
])