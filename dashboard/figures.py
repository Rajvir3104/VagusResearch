import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
                    line=dict(dash="dash", color="black"),
                )
            ],
            margin=dict(l=60, r=20, t=60, b=50),
        ),
    )


from plotly.subplots import make_subplots
import plotly.graph_objects as go

def aligned_metrics_figure(
    t_raw,
    t_rate,
    neuron_firing_rate,
    respiration,
    ecg,
    nerve,
    neuron_firing_rate_name="Neuron Firing Rate",
    respiration_name="Respiration",
    ecg_name="ECG",
    nerve_name="Nerve Activity",
    x_label="Aligned time (s)",
):
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            neuron_firing_rate_name,
            respiration_name,
            ecg_name,
        ),
    )

    fig.add_trace(
        go.Scatter(x=t_rate, y=neuron_firing_rate, mode="lines", name=neuron_firing_rate_name),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t_raw, y=nerve, mode="lines", name=nerve_name),
        row=2,
        col=1,
    
    )
    fig.update_yaxes(range=[-1, 1], row=2, col=1)
    fig.add_trace(
        go.Scatter(x=t_raw, y=respiration, mode="lines", name=respiration_name),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t_raw, y=ecg, mode="lines", name=ecg_name),
        row=4,
        col=1,
    )

    fig.update_layout(
        height=800,
        title="Aligned Metrics",
        hovermode="x unified",
        showlegend=False,
        dragmode="pan",
    )

    fig.update_xaxes(title_text=x_label, row=4, col=1)
    fig.update_xaxes(range=[300, 600], row=4, col=1)
    fig.update_xaxes(rangeslider_visible=True, row=4, col=1)

    return fig