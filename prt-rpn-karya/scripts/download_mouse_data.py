"""Download/cache mouse pretraining sessions into the repo-level data/ dir.

    .venv-train/bin/python scripts/download_mouse_data.py --dataset ibl --n_sessions 5      # smallest BWM probes first
    .venv-train/bin/python scripts/download_mouse_data.py --dataset allen --n_sessions 3    # most good units first

IBL: data/ibl/ (ONE cache layout).
Allen: data/allen/pretrain_small/ses-<id>.npz: good units, spikes inside the continuous viewing epochs
(spontaneous + natural movies), ~0.1-0.3 GB per session.
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from neural_data import (ALLEN_PRETRAIN_EPOCHS, allen_session_units_table, build_allen_subset, ibl_one,
                         ibl_probe_table, load_allen_subset, load_ibl_probe)

parser = argparse.ArgumentParser(description='Cache mouse sessions for pretraining.')
parser.add_argument('--dataset', choices=['ibl', 'allen'], required=True)
parser.add_argument('--n_sessions', type=int, default=1)
parser.add_argument('--ibl_project', default='brainwide', help='substring of the IBL project name to keep')
parser.add_argument('--allen_epochs', nargs='+', default=list(ALLEN_PRETRAIN_EPOCHS))
parser.add_argument('--allen_area_prefix', default=None, help="keep only units whose area starts with this, e.g. 'VIS'")
args = parser.parse_args()

if args.dataset == 'ibl':
    one = ibl_one()
    probes = ibl_probe_table(one)
    probes = probes[probes['projects'].str.contains(args.ibl_project, na=False)].sort_values('spikes_times_mb')
    for _, r in probes.head(args.n_sessions).iterrows():
        s = load_ibl_probe(one, r['eid'], r['probe'])
        print(f"IBL {r['subject']} {r['date']} {r['probe']}: {s['n_units']} good units, {len(s['trials'])} trials")
else:
    sessions = allen_session_units_table()
    for sid in sessions['session_id'].head(args.n_sessions):
        path = build_allen_subset(sid, epochs=args.allen_epochs, area_prefix=args.allen_area_prefix)
        s = load_allen_subset(sid)
        minutes = (s['epochs']['stop'] - s['epochs']['start']).sum() / 60
        print(f"Allen ses-{sid}: {s['n_units']} units, {len(s['spike_times']):,} spikes, {minutes:.0f} min "
              f"-> {path} ({os.path.getsize(path) / 1e6:.0f} MB)")
