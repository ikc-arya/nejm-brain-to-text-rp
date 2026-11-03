"""Neural data sources for pretraining. Every source yields batches in one common format:

    x        float [B, T, C]  model input (20 ms bins; C may differ per session)
    target   float [B, T, C]  what the AR head predicts (raw z-scored features for B2T, raw counts for mouse)
    lengths  long  [B]        valid bins per row (rows are right-padded)
    session  long  [B]        session index within the source
    phonemes / phone_lens     B2T only, for Stage 2

and describes itself with `meta()` -> {'channels': [C per session], 'ar_output': 'linear' | 'log_rate',
'sessions': [names], optional 'mean_rate': [[rate per unit] per session]}, which is all the model needs to build
the source's stem/heads.
Add a dataset by writing a class with train_batches / val_batches / meta and registering it in SOURCES.
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import BrainToTextDataset, train_test_split_indicies
import neural_data


class B2TSource:
    """Brain-to-Text '25 (human, 256 Utah electrodes x {threshold crossings, spike-band power} = 512, z-scored)."""
    ar_output = 'linear'

    def __init__(self, name='b2t', dataset_dir='../data/hdf5_data_final', sessions='all', batch_size=64, days_per_batch=4,
                 seed=1, num_workers=4, **_):
        self.name = name
        self.dataset_dir = dataset_dir
        self.sessions = sorted(os.listdir(dataset_dir)) if sessions == 'all' else list(sessions)
        self.batch_size, self.days_per_batch = batch_size, days_per_batch
        self.seed, self.num_workers = seed, num_workers

    def meta(self):
        return {'channels': [512] * len(self.sessions), 'ar_output': self.ar_output, 'sessions': self.sessions}

    def _paths(self, split):
        return [os.path.join(self.dataset_dir, s, f'data_{split}.hdf5') for s in self.sessions]

    @staticmethod
    def _convert(b):
        return {'x': b['input_features'], 'target': b['input_features'], 'lengths': b['n_time_steps'].long(),
                'session': b['day_indicies'].long(), 'phonemes': b['seq_class_ids'], 'phone_lens': b['phone_seq_lens'],
                'transcriptions': b['transcriptions'], 'block_nums': b['block_nums'], 'trial_nums': b['trial_nums']}

    def train_batches(self, n_batches):
        trials, _ = train_test_split_indicies(self._paths('train'), test_percentage=0, seed=self.seed)
        ds = BrainToTextDataset(trials, n_batches, split='train', batch_size=self.batch_size,
                                days_per_batch=self.days_per_batch, random_seed=self.seed)
        loader = DataLoader(ds, batch_size=None, shuffle=False, num_workers=self.num_workers, pin_memory=False,
                            persistent_workers=self.num_workers > 0)
        return (self._convert(b) for b in loader)

    def val_batches(self, max_batches=None):
        """All trials of data_val.hdf5 (sessions without one are skipped): the baseline's 1426-trial val set.
        Streamed from disk (~2.6 GB as float). max_batches picks an evenly spaced subset (SSL val)."""
        _, trials = train_test_split_indicies(self._paths('val'), test_percentage=1, seed=self.seed)
        ds = BrainToTextDataset(trials, None, split='test', batch_size=self.batch_size, random_seed=self.seed)
        idx = range(len(ds)) if not max_batches else np.unique(np.linspace(0, len(ds) - 1, max_batches).astype(int))
        for i in idx:
            yield self._convert(ds[int(i)])


class AllenSource:
    """Allen Visual Coding Neuropixels (mouse), the cached pretraining subset in data/allen/pretrain_small/.

    Spikes of good units are binned at 20 ms per epoch (no windows across epoch gaps). Input = per-unit z-scored
    counts, target = raw counts (Poisson). The last `val_fraction` of every epoch is held out for validation.
    One session per batch, because unit counts differ per session.
    """
    ar_output = 'log_rate'

    def __init__(self, name='allen', session_ids='all', window_bins=512, batch_size=64, val_fraction=0.1,
                 max_val_windows=128, seed=1, cache_dir=None, **_):
        self.name = name
        cache = cache_dir or os.path.join(neural_data.ALLEN_DIR, 'pretrain_small')
        neural_data.ALLEN_DIR = os.path.dirname(os.path.abspath(cache))  # load_allen_subset reads <ALLEN_DIR>/pretrain_small
        if session_ids == 'all':
            session_ids = sorted(int(f[4:-4]) for f in os.listdir(cache) if f.startswith('ses-') and f.endswith('.npz'))
        self.sessions = [str(s) for s in session_ids]
        self.window_bins, self.batch_size = window_bins, batch_size
        self.rng = np.random.default_rng(seed)
        self.counts, self.train_segs, self.val_segs, self.mu, self.sd = [], [], [], [], []
        for sid in session_ids:
            d = neural_data.load_allen_subset(sid)
            mats, train, val, off = [], [], [], 0
            for _, ep in d['epochs'].iterrows():
                c = neural_data.bin_spikes(d['spike_times'], d['spike_units'], d['n_units'],
                                           t_start=ep['start'], t_end=ep['stop'])
                mats.append(np.minimum(c, 255).astype(np.uint8))
                n_val = int(len(c) * val_fraction)
                train.append((off, off + len(c) - n_val))
                val.append((off + len(c) - n_val, off + len(c)))
                off += len(c)
            counts = np.concatenate(mats)
            tr = np.concatenate([counts[a:b] for a, b in train]).astype(np.float32)
            self.mu.append(tr.mean(0)); self.sd.append(np.maximum(tr.std(0), 0.1))
            self.counts.append(counts)
            self.train_segs.append([s for s in train if s[1] - s[0] >= window_bins])
            self.val_segs.append([s for s in val if s[1] - s[0] >= window_bins])
        self.max_val_windows = max_val_windows

    def meta(self):
        return {'channels': [c.shape[1] for c in self.counts], 'ar_output': self.ar_output, 'sessions': self.sessions,
                'mean_rate': [m.tolist() for m in self.mu]}  # counts / bin, initialises the AR head bias

    def _batch(self, s, starts):
        raw = np.stack([self.counts[s][a:a + self.window_bins] for a in starts]).astype(np.float32)
        return {'x': torch.from_numpy((raw - self.mu[s]) / self.sd[s]), 'target': torch.from_numpy(raw),
                'lengths': torch.full((len(starts),), self.window_bins, dtype=torch.long),
                'session': torch.full((len(starts),), s, dtype=torch.long)}

    def _starts(self, segs, n, rng):
        seg = rng.integers(len(segs), size=n)
        return [int(rng.integers(segs[i][0], segs[i][1] - self.window_bins + 1)) for i in seg]

    def train_batches(self, n_batches):
        for _ in range(n_batches):
            s = int(self.rng.integers(len(self.sessions)))
            yield self._batch(s, self._starts(self.train_segs[s], self.batch_size, self.rng))

    def val_batches(self, max_batches=None):
        rng = np.random.default_rng(0)  # fixed val windows
        out = []
        for s in range(len(self.sessions)):
            starts = self._starts(self.val_segs[s], self.max_val_windows, rng)
            out += [self._batch(s, starts[i:i + self.batch_size]) for i in range(0, len(starts), self.batch_size)]
        return out[:max_batches] if max_batches else out


SOURCES = {'b2t': B2TSource, 'allen': AllenSource}
# 'ibl': IBLSource -- same contract as AllenSource (neural_data.load_ibl_probe + bin_spikes); not implemented yet.


def build_sources(source_cfgs):
    """[{type, weight, ...kwargs}] -> ({name: source}, {name: weight})."""
    sources, weights = {}, {}
    for cfg in source_cfgs:
        cfg = dict(cfg)
        typ, weight = cfg.pop('type'), float(cfg.pop('weight', 1.0))
        src = SOURCES[typ](**cfg)
        if src.name in sources:
            raise ValueError(f'duplicate source name {src.name}; give each source a unique `name`')
        sources[src.name], weights[src.name] = src, weight
    return sources, weights


def mixture(sources, weights, n_batches, seed=0):
    """Yields (source_name, batch) n_batches times, picking the source of every batch with probability ~ weight."""
    rng = np.random.default_rng(seed)
    names = list(sources)
    p = np.array([weights[n] for n in names]); p = p / p.sum()
    picks = rng.choice(len(names), size=n_batches, p=p)
    iters = {n: iter(sources[n].train_batches(int((picks == i).sum()))) for i, n in enumerate(names)}
    for i in picks:
        yield names[i], next(iters[names[i]])
