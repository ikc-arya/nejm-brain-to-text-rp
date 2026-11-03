"""Loaders for the pretraining datasets + a common binned representation.

Every source is reduced to the same unit-level primitive:
    spike_times (float64, seconds) + spike_units (int, index into a units table)
and from there to a binned count matrix [T, N] (T = time bins, N = units of that session).

Sources
- IBL Brain-Wide Map (mouse, Neuropixels)      : ONE API, public openalyx server
- Allen Visual Coding Neuropixels (mouse)      : Allen API (api.brain-map.org), streamed NWB -> small subset
- Brain-to-Text 2025 (human, Utah arrays)      : local hdf5, already binned at 20 ms -> [T, 512]
"""
import os

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'data'))  # shared repo-level data/, never copied

BIN_SIZE = 0.02  # 20 ms, same as Brain-to-Text


# ------------------------------------------------------------------ common

def bin_spikes(spike_times, spike_units, n_units, bin_size=BIN_SIZE, t_start=None, t_end=None):
    """Spike trains -> count matrix [T, n_units] (uint16). spike_units must be in [0, n_units)."""
    t_start = spike_times.min() if t_start is None else t_start
    t_end = spike_times.max() if t_end is None else t_end
    keep = (spike_times >= t_start) & (spike_times < t_end)
    n_bins = int(np.ceil((t_end - t_start) / bin_size))
    t_idx = ((spike_times[keep] - t_start) / bin_size).astype(np.int64)
    flat = np.bincount(t_idx * n_units + spike_units[keep], minlength=n_bins * n_units)
    return flat.reshape(n_bins, n_units).astype(np.uint16)


def session_summary(name, spike_times, spike_units, n_units):
    duration = float(spike_times.max() - spike_times.min())
    return {
        'session': name,
        'n_units': n_units,
        'n_spikes': len(spike_times),
        'duration_min': duration / 60,
        'n_bins_20ms': int(duration / BIN_SIZE),
        'mean_rate_hz': len(spike_times) / duration / max(n_units, 1),
    }


# ------------------------------------------------------------------ IBL

IBL_URL = 'https://openalyx.internationalbrainlab.org'


def ibl_one(username='intbrainlab', password='international'):
    """ONE client on the public server. The shared account is being retired:
    register at https://openalyx.internationalbrainlab.org/signup and pass your own credentials."""
    from one.api import ONE
    return ONE(base_url=IBL_URL, username=username, password=password, silent=True,
               cache_dir=os.path.join(DATA_DIR, 'ibl'))


def ibl_probe_table(one):
    """One row per spike-sorted probe insertion (pykilosort, default revision), from the public cache tables."""
    one.load_cache()
    sessions, datasets = one._cache['sessions'], one._cache['datasets']
    st = datasets[datasets['rel_path'].str.match(r'^alf/probe\d\d/pykilosort/spikes\.times\.npy$')].reset_index()
    st['probe'] = st['rel_path'].str.split('/').str[1]
    st = st.merge(sessions.reset_index(), left_on='eid', right_on='id', how='left')
    st['spikes_times_mb'] = st['file_size'] / 1e6
    return st[['eid', 'probe', 'subject', 'lab', 'date', 'task_protocol', 'projects', 'spikes_times_mb']]


def load_ibl_probe(one, eid, probe):
    """Spikes of 'good' units (label == 1) for one probe + all-unit metrics + the session's trials table."""
    collection = f'alf/{probe}/pykilosort'
    spikes = one.load_object(eid, 'spikes', collection=collection, attribute=['times', 'clusters'])
    metrics = one.load_dataset(eid, 'clusters.metrics.pqt', collection=collection)
    trials = one.load_object(eid, 'trials', collection='alf').to_df()
    good = metrics.index[metrics['label'] == 1].to_numpy()
    remap = -np.ones(int(spikes['clusters'].max()) + 1, dtype=np.int64)
    remap[good] = np.arange(len(good))
    units = remap[spikes['clusters']]
    keep = units >= 0
    return {
        'spike_times': spikes['times'][keep],
        'spike_units': units[keep],
        'n_units': len(good),
        'units': metrics.loc[good].reset_index(drop=True),
        'all_units': metrics,
        'trials': trials,
    }


# ------------------------------------------------------------------ Allen (Visual Coding Neuropixels, Allen API)

ALLEN_API = 'https://api.brain-map.org/api/v2'
ALLEN_DIR = os.path.join(DATA_DIR, 'allen')
# Continuous, naturalistic viewing: closest to free-running dynamics; ~50 of ~160 min per session.
ALLEN_PRETRAIN_EPOCHS = ('spontaneous', 'natural_movie_one', 'natural_movie_three')


def _rma(criteria):
    import requests
    r = requests.get(f'{ALLEN_API}/data/query.json?criteria={criteria}', timeout=120).json()
    if not r.get('success', True):
        raise RuntimeError(r.get('msg'))
    return r['msg']


def allen_sessions():
    """All 58 Visual Coding Neuropixels sessions (32 brain_observatory_1.1 + 26 functional_connectivity)."""
    return pd.DataFrame(_rma('model::EcephysSession,rma::options[num_rows$eqall]'))[
        ['id', 'specimen_id', 'stimulus_name', 'date_of_acquisition']].sort_values('id').reset_index(drop=True)


def allen_good_units(session_id):
    """Good units of one session with their brain area (from the API, no spike download)."""
    rows = _rma(f"model::EcephysUnit,rma::criteria,[quality$eq'good'],"
                f"ecephys_channel(ecephys_probe[ecephys_session_id$eq{session_id}]),"
                f"rma::include,ecephys_channel(structure),rma::options[num_rows$eqall]")
    return pd.DataFrame({'id': [u['id'] for u in rows],
                         'area': [(u['ecephys_channel'].get('structure') or {}).get('acronym', '') for u in rows],
                         'firing_rate': [u['firing_rate'] for u in rows]})


def allen_session_units_table(stimulus_name='brain_observatory_1.1'):
    """Per-session good-unit counts (total + visual cortex), cached to data/allen/session_units.csv."""
    path = os.path.join(ALLEN_DIR, f'session_units_{stimulus_name}.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    rows = []
    for _, s in allen_sessions().query('stimulus_name == @stimulus_name').iterrows():
        u = allen_good_units(s['id'])
        rows.append({'session_id': s['id'], 'specimen_id': s['specimen_id'], 'n_good_units': len(u),
                     'n_visual_cortex': int(u['area'].str.startswith('VIS').sum()), 'n_areas': u['area'].nunique()})
    df = pd.DataFrame(rows).sort_values('n_good_units', ascending=False).reset_index(drop=True)
    os.makedirs(ALLEN_DIR, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _allen_nwb_url(session_id):
    import requests
    f = _rma(f"model::WellKnownFile,rma::criteria,[attachable_id$eq{session_id}],[attachable_type$eq'EcephysSession']")[0]
    return requests.head(f"{ALLEN_API}/well_known_file_download/{f['id']}", allow_redirects=True, timeout=60).url


def build_allen_subset(session_id, epochs=ALLEN_PRETRAIN_EPOCHS, area_prefix=None):
    """Stream one session NWB from the Allen API and keep only good units (optionally areas starting with
    area_prefix, e.g. 'VIS') and spikes inside `epochs`. Saves data/allen/pretrain_small/ses-<id>.npz (~0.1 GB)."""
    import remfile
    out = os.path.join(ALLEN_DIR, 'pretrain_small', f'ses-{session_id}.npz')
    if os.path.exists(out):
        return out
    units = allen_good_units(session_id)
    if area_prefix:
        units = units[units['area'].str.startswith(area_prefix)].reset_index(drop=True)
    with h5py.File(remfile.File(_allen_nwb_url(session_id)), 'r') as f:
        u = f['units']
        row = {uid: i for i, uid in enumerate(u['id'][:])}
        index = np.concatenate([[0], u['spike_times_index'][:]])
        times = u['spike_times'][:]  # all spikes, ~0.6 GB float64; units are contiguous blocks
        blocks = []
        for name in epochs:
            g = f['intervals'][f'{name}_presentations']
            blocks.append(pd.DataFrame({'stimulus': name, 'start': g['start_time'][:], 'stop': g['stop_time'][:]}))
    pres = pd.concat(blocks).sort_values('start').reset_index(drop=True)
    # merge back-to-back presentations (movie frames) into contiguous epochs
    new = pres['start'] > pres['stop'].shift().fillna(-np.inf) + 1.0
    pres['epoch'] = new.cumsum()
    ep = pres.groupby('epoch').agg(stimulus=('stimulus', 'first'), start=('start', 'min'), stop=('stop', 'max')).reset_index(drop=True)

    st, su = [], []
    for k, uid in enumerate(units['id']):
        t = times[index[row[uid]]:index[row[uid] + 1]]
        keep = np.zeros(len(t), bool)
        for a, b in zip(ep['start'], ep['stop']):
            keep |= (t >= a) & (t < b)
        st.append(t[keep]); su.append(np.full(keep.sum(), k, np.uint16))
    st, su = np.concatenate(st), np.concatenate(su)
    order = np.argsort(st, kind='stable')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, spike_times=st[order].astype(np.float32), spike_units=su[order],
                        unit_id=units['id'].to_numpy(), unit_area=units['area'].to_numpy().astype(str),
                        unit_rate=units['firing_rate'].to_numpy(), epochs=ep.to_records(index=False))
    return out


def load_allen_subset(session_id):
    """Spikes of the cached pretraining subset + units and epochs tables (see build_allen_subset)."""
    z = np.load(os.path.join(ALLEN_DIR, 'pretrain_small', f'ses-{session_id}.npz'), allow_pickle=True)
    units = pd.DataFrame({'id': z['unit_id'], 'area': z['unit_area'], 'firing_rate': z['unit_rate']})
    return {
        'spike_times': z['spike_times'].astype(np.float64),
        'spike_units': z['spike_units'].astype(np.int64),
        'n_units': len(units),
        'units': units,
        'epochs': pd.DataFrame(z['epochs']),
    }


# ------------------------------------------------------------------ Brain-to-Text 2025 (local)

B2T_DIR = os.path.join(DATA_DIR, 'hdf5_data_final')


def b2t_trial_table(split='train'):
    """One row per trial: session, n_time_steps (20 ms bins), phoneme seq_len, sentence."""
    rows = []
    for session in sorted(os.listdir(B2T_DIR)):
        p = os.path.join(B2T_DIR, session, f'data_{split}.hdf5')
        if not os.path.exists(p):
            continue
        with h5py.File(p, 'r') as f:
            for k in f:
                a = f[k].attrs
                rows.append({'session': session, 'trial': k, 'n_time_steps': int(a['n_time_steps']),
                             'seq_len': int(a.get('seq_len', -1)), 'sentence': a.get('sentence_label', '')})
    return pd.DataFrame(rows)


def load_b2t_trial(session, trial, split='train'):
    """input_features [T, 512] float32: 256 electrodes x (threshold crossings, spike-band power), 20 ms bins."""
    with h5py.File(os.path.join(B2T_DIR, session, f'data_{split}.hdf5'), 'r') as f:
        return f[trial]['input_features'][:]
