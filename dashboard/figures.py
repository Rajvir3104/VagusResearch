import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def time_series_figure(x, y, title, xlabel, ylabel, markers_x=None, markers_y=None, marker_name=None):
    data = [
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=ylabel,
            line=dict(color="red", width=1.5),
        )
    ]

    if markers_x is not None and markers_y is not None:
        data.append(
            go.Scatter(
                x=markers_x,
                y=markers_y,
                mode="markers",
                name=marker_name or "Markers",
                marker=dict(color="black", size=8, symbol="x"),
            )
        )

    return go.Figure(
        data=data,
        layout=go.Layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            margin=dict(l=60, r=20, t=60, b=50),
        ),
    )


def psth_figure(
    t,
    rate,
    sem,
    title,
    xlabel,
    signal_x=None,
    ecg=None,
    resp=None,
):
    has_signal = (signal_x is not None) and (ecg is not None or resp is not None)

    if has_signal:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.6, 0.4],
            subplot_titles=(title, "Aligned signal"),
        )
    else:
        fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=t,
            y=rate,
            name="Spike count",
            marker_color="lightgray",
            opacity=0.75,
            hovertemplate="%{x:.3f}s<br>%{y:.2f} spikes/s",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=rate,
            mode="lines",
            name="Mean firing rate",
            line=dict(color="black", width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([t, t[::-1]]),
            y=np.concatenate([rate - sem, (rate + sem)[::-1]]),
            fill="toself",
            fillcolor="rgba(0,100,200,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="SEM",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    if has_signal:
        if ecg is not None:
            fig.add_trace(
                go.Scatter(
                    x=signal_x,
                    y=ecg,
                    mode="lines",
                    name="ECG (mean)",
                    line=dict(color="red", width=1.5),
                ),
                row=2,
                col=1,
            )

        if resp is not None:
            fig.add_trace(
                go.Scatter(
                    x=signal_x,
                    y=resp,
                    mode="lines",
                    name="Respiration (mean)",
                    line=dict(color="blue", width=1.5, dash="dash"),
                ),
                row=2,
                col=1,
            )

        fig.update_yaxes(title_text="Events / second", row=1, col=1)
        fig.update_yaxes(title_text="Signal", row=2, col=1)
        fig.update_xaxes(title_text=xlabel, row=2, col=1)
        fig.update_layout(
            height=700,
            margin=dict(l=60, r=60, t=60, b=50),
            showlegend=True,
        )
    else:
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title="Events / second",
            margin=dict(l=60, r=60, t=60, b=50),
        )

    fig.add_shape(
        type="line",
        x0=0,
        x1=0,
        y0=0,
        y1=float(np.max(rate + sem)),
        line=dict(dash="dash", color="black"),
    )

    return fig

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