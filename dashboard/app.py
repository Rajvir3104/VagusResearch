# import sys
# from pathlib import Path
# import numpy as np

# # allow imports from analysis/
# sys.path.append(str(Path(__file__).resolve().parents[1]))

# from analysis.analysis import (
#     epochs_ecg,
#     epochs_breath,
#     obtain_spikes,
#     combining_spike_and_epoch,
#     psth_from_overlap,
# )

# import dash
# from dash import dcc, html
# import plotly.graph_objs as go

# #  Run analysis once 
# spikes = obtain_spikes()

# ecg_epochs = epochs_ecg()
# breath_epochs = epochs_breath()

# ecg_overlap = combining_spike_and_epoch(spikes, ecg_epochs)
# breath_overlap = combining_spike_and_epoch(spikes, breath_epochs)

# t_ecg, rate_ecg, sem_ecg = psth_from_overlap(ecg_overlap, ecg_epochs, bin_width=0.005)
# t_breath, rate_breath, sem_breath = psth_from_overlap(breath_overlap, breath_epochs, bin_width=0.1)


# # Dash App
# app = dash.Dash(__name__)

# def psth_figure(t, rate, sem, title, xlabel):
#     return go.Figure(
#         data=[
#             # Mean rate
#             go.Scatter(
#                 x=t,
#                 y=rate,
#                 mode="lines",
#                 name="Mean firing rate",
#                 line=dict(width=2),
#             ),
#             # SEM shading
#             go.Scatter(
#                 x=np.concatenate([t, t[::-1]]),
#                 y=np.concatenate([rate - sem, (rate + sem)[::-1]]),
#                 fill="toself",
#                 fillcolor="rgba(0,100,200,0.2)",
#                 line=dict(color="rgba(255,255,255,0)"),
#                 hoverinfo="skip",
#                 name="SEM",
#             ),
#         ],
#         layout=go.Layout(
#             title=title,
#             xaxis_title=xlabel,
#             yaxis_title="Events / second",
#             shapes=[
#                 dict(
#                     type="line",
#                     x0=0, x1=0,
#                     y0=0, y1=float(np.max(rate + sem)),
#                     line=dict(dash="dash", color="black")
#                 )
#             ],
#             margin=dict(l=60, r=20, t=60, b=50),
#         )
#     )


# app.layout = html.Div([
#     html.H1("Vagus PSTH Analysis"),

#     html.H2("ECG-Aligned PSTH"),
#     dcc.Graph(
#         figure=psth_figure(
#             t_ecg,
#             rate_ecg,
#             sem_ecg,
#             title="ECG-aligned PSTH",
#             xlabel="Time relative to R-peak (s)"
#         )
#     ),

#     html.H2("Respiration-Aligned PSTH"),
#     dcc.Graph(
#         figure=psth_figure(
#             t_breath,
#             rate_breath,
#             sem_breath,
#             title="Respiration-aligned PSTH",
#             xlabel="Time relative to respiration peak (s)"
#         )
#     ),
# ])

# if __name__ == "__main__":
#     app.run(debug=True)
import sys
from pathlib import Path
import numpy as np

# allow imports from analysis/
sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.analysis import (
    epochs_ecg,
    epochs_breath,
    obtain_spikes,
    combining_spike_and_epoch,
    psth_from_overlap,
)

import dash
from dash import dcc, html
import plotly.graph_objs as go
from plotly.subplots import make_subplots


# Run analysis once
spikes = obtain_spikes()

ecg_epochs = epochs_ecg()
breath_epochs = epochs_breath()

ecg_overlap = combining_spike_and_epoch(spikes, ecg_epochs)
breath_overlap = combining_spike_and_epoch(spikes, breath_epochs)

t_ecg, rate_ecg, sem_ecg = psth_from_overlap(ecg_overlap, ecg_epochs, bin_width=0.005)
t_breath, rate_breath, sem_breath = psth_from_overlap(breath_overlap, breath_epochs, bin_width=0.1)


from analysis import data_loading
import analysis


df = data_loading.load_any()

time = df["time"].values
respiration_signal = df["resp"].values
ecg_signal = df["ecg"].values

fs = data_loading.calculate_freq(df)
signal = data_loading.pre_process(df, fs)
spikes = data_loading.detect_spikes(
    data_loading.compute_threshold(signal),
    signal,
    fs,
    t=df["time"].values
)
nerve_signal = obtain_spikes()


# PSTH figure function
def psth_figure(t, rate, sem, title, xlabel):
    return go.Figure(
        data=[
            go.Scatter(
                x=t,
                y=rate,
                mode="lines",
                name="Mean firing rate",
                line=dict(width=2),
            ),
            go.Scatter(
                x=np.concatenate([t, t[::-1]]),
                y=np.concatenate([rate - sem, (rate + sem)[::-1]]),
                fill="toself",
                fillcolor="rgba(0,100,200,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="SEM",
            ),
        ],
        layout=go.Layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title="Events / second",
            shapes=[
                dict(
                    type="line",
                    x0=0,
                    x1=0,
                    y0=0,
                    y1=float(np.max(rate + sem)),
                    line=dict(dash="dash", color="black")
                )
            ],
            margin=dict(l=60, r=20, t=60, b=50),
        )
    )


def physio_figure(time, respiration, nerve, ecg):

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Respiration", "MSNA", "ECG")
    )

    # Respiration
    fig.add_trace(
        go.Scatter(
            x=time,
            y=respiration,
            mode="lines",
            line=dict(color="blue"),
            name="Respiration"
        ),
        row=1,
        col=1
    )

    # MSNA (line signal instead of spikes)
    fig.add_trace(
        go.Scatter(
            x=time,
            y=nerve,
            mode="lines",
            line=dict(color="green"),
            name="MSNA"
        ),
        row=2,
        col=1
    )

    # ECG
    fig.add_trace(
        go.Scatter(
            x=time,
            y=ecg,
            mode="lines",
            line=dict(color="red"),
            name="ECG"
        ),
        row=3,
        col=1
    )

    fig.update_layout(
        height=700,
        title="Respiration, MSNA, and ECG",
        showlegend=False
    )

    # Scrollable timeline
    fig.update_xaxes(
        title_text="Time (s)",
        rangeslider=dict(visible=True),
        row=3,
        col=1
    )

    return fig

# Dash App
app = dash.Dash(__name__)

app.layout = html.Div([

    html.H1("Vagus PSTH Analysis"),

    html.H2("Raw Physiological Signals"),
    dcc.Graph(
    figure=physio_figure(
        time,
        respiration_signal,
        spikes,
        ecg_signal
    ),
    config={"scrollZoom": True}

    ),

    html.H2("ECG-Aligned PSTH"),
    dcc.Graph(
        figure=psth_figure(
            t_ecg,
            rate_ecg,
            sem_ecg,
            title="ECG-aligned PSTH",
            xlabel="Time relative to R-peak (s)"
        )
    ),

    html.H2("Respiration-Aligned PSTH"),
    dcc.Graph(
        figure=psth_figure(
            t_breath,
            rate_breath,
            sem_breath,
            title="Respiration-aligned PSTH",
            xlabel="Time relative to respiration peak (s)"
        )
    ),
])


if __name__ == "__main__":
    app.run(debug=True)