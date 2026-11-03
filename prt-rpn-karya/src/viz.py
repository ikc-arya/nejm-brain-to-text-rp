"""Interactive spike rasters (plotly WebGL): drag to zoom, double-click to reset, scroll the range slider."""
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# native plotly rendering in VS Code / JupyterLab (no HTML copy, keeps the notebook small)
pio.renderers.default = 'plotly_mimetype'


def raster(spike_times, spike_units, t0, t1, unit_groups=None, events=None, bin_size=0.02, title=''):
    """Raster of spikes in [t0, t1) over a population-rate trace.

    unit_groups: optional array (one label per unit, e.g. brain area). Units are sorted and coloured by group.
    events: optional dict {name: times}, drawn as vertical lines (e.g. stimulus onsets).
    """
    w = (spike_times >= t0) & (spike_times < t1)
    t, u = spike_times[w], spike_units[w]
    n_units = int(spike_units.max()) + 1

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.03)
    if unit_groups is None:
        fig.add_trace(go.Scattergl(x=t, y=u, mode='markers', marker=dict(size=2, color='black'),
                                   name='spikes', hovertemplate='t=%{x:.3f}s unit %{y}<extra></extra>'), 1, 1)
    else:
        groups = np.asarray(unit_groups)
        order = np.argsort(groups, kind='stable')  # unit -> row, grouped by label
        row = np.empty(n_units, int)
        row[order] = np.arange(n_units)
        for g in np.unique(groups):
            m = groups[u] == g
            fig.add_trace(go.Scattergl(x=t[m], y=row[u[m]], mode='markers', marker=dict(size=2), name=str(g),
                                       hovertemplate=f'{g}<br>t=%{{x:.3f}}s<extra></extra>'), 1, 1)

    edges = np.arange(t0, t1 + bin_size, bin_size)
    rate = np.histogram(t, edges)[0] / bin_size / n_units
    fig.add_trace(go.Scatter(x=edges[:-1], y=rate, mode='lines', line=dict(width=1, color='gray'),
                             name='pop. rate (Hz/unit)', showlegend=False), 2, 1)

    for name, times in (events or {}).items():
        for x in np.asarray(times)[(np.asarray(times) >= t0) & (np.asarray(times) < t1)]:
            fig.add_vline(x=x, line=dict(color='red', width=0.6, dash='dot'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='red', dash='dot'), name=name), 1, 1)

    fig.update_yaxes(title_text='unit', row=1, col=1)
    fig.update_yaxes(title_text='Hz/unit', row=2, col=1)
    fig.update_xaxes(title_text='time (s)', rangeslider=dict(visible=True, thickness=0.05), row=2, col=1)
    fig.update_layout(title=title, height=650, template='simple_white', legend=dict(itemsizing='constant'),
                      margin=dict(l=60, r=20, t=50, b=40))
    return fig
