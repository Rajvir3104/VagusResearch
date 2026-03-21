import dash
from dash import html, dcc

from data import (
    t_ecg,
    rate_ecg,
    sem_ecg,
    t_breath_peak,
    rate_breath_peak,
    sem_breath_peak,
    t_breath_trough,
    rate_breath_trough,
    sem_breath_trough,
)
from figures import psth_figure

dash.register_page(__name__, path="/")

layout = html.Div([
    html.H2("ECG-Aligned PSTH"),
    dcc.Graph(
        figure=psth_figure(
            t_ecg,
            rate_ecg,
            sem_ecg,
            title="ECG-aligned PSTH",
            xlabel="Time relative to R-peak (s)",
        )
    ),

    html.H2("Respiration-Aligned Peaks PSTH"),
    dcc.Graph(
        figure=psth_figure(
            t_breath_peak,
            rate_breath_peak,
            sem_breath_peak,
            title="Respiration-aligned PSTH",
            xlabel="Time relative to respiration peak (s)",
        )
    ),

    html.H2("Respiration-Aligned Troughs PSTH"),
    dcc.Graph(
        figure=psth_figure(
            t_breath_trough,
            rate_breath_trough,
            sem_breath_trough,
            title="Respiration-aligned PSTH",
            xlabel="Time relative to respiration trough (s)",
        )
    ),
])
