# prt-rpn-karya: pretraining experiments

Experiment workspace carved out of `nejm-brain-to-text-rp`, focused on **pretraining on neural data**.
Data is never copied: everything reads from and writes to the parent repo's `../data/`.

## Datasets

| Dataset | Access | Local path | Status |
|---|---|---|---|
| IBL Brain-Wide Map (mouse) | ONE API, public openalyx server | `../data/ibl/` | loader + 1 probe cached |
| Allen Visual Coding Neuropixels (mouse) | Allen API (`api.brain-map.org`), streamed NWB | `../data/allen/pretrain_small/` | 3-session pretraining subset: good units, spontaneous + natural-movie epochs, 217 MB |
| Brain-to-Text '25 (human) | local hdf5 | `../data/hdf5_data_final/` | all 45 sessions present |
| Brain-to-Text '24, Gesture (human) | — | — | not set up |

Start with `analyses/eda_pretraining_datasets.ipynb`. It covers dimensions per dataset, behaviour and stimulus tables, and how to batch variable-N sessions.

## Layout

```
analyses/   eda_pretraining_datasets.ipynb, figure_2/4.ipynb (paper figures)
configs/    baseline_rnn.yaml (B2T GRU hparams), t15_copyTaskData_description.csv
src/        neural_data.py (IBL / Allen / B2T loaders, bin_spikes), viz.py (interactive rasters), rnn_model.py, rnn_trainer.py, dataset.py, data_augmentations.py, nejm_b2txt_utils/
scripts/    download_mouse_data.py, download_data.py (B2T, Dryad), train.py (B2T GRU)
experiments/ one folder per run (git-ignored)
```

## Setup

```bash
./setup_env.sh   # uv env .venv-train from requirements.txt; puts src/ on the path
```

The notebook kernel is **prt-rpn-karya (.venv-train)**. The env is called `.venv-train` because something on this machine kept deleting a folder named `.venv`.

## Data

```bash
.venv-train/bin/python scripts/download_mouse_data.py --dataset ibl --n_sessions 5     # smallest BWM probes first
.venv-train/bin/python scripts/download_mouse_data.py --dataset allen --n_sessions 3   # most good units first, ~70 MB each
```

IBL uses the shared public account, which IBL is retiring. Register at https://openalyx.internationalbrainlab.org/signup
and pass your credentials to `neural_data.ibl_one(username, password)`.
Dryad (B2T) refuses anonymous API downloads (HTTP 401), so fetch files in a browser. `download_data.py` skips anything already present.

All scripts resolve paths from this directory, so they work from any cwd.

## Pretraining pipeline (SpikeGPT): Stage 1 SSL -> Stage 2 CTC -> PER

| Stage | Script | Configs (Mac / cluster) | Output |
|---|---|---|---|
| 1. causal AR SSL | `scripts/pretrain_ssl.py` | `ssl_human`, `ssl_mouse`, `ssl_human_mouse` / `ssl_human_gpu` | `experiments/<run>/checkpoint/best_checkpoint` |
| 2. phoneme CTC | `scripts/finetune_ctc.py` | `ctc_human` / `ctc_human_gpu` | val PER in `training_log`, `val_history.json`, per-day PER in `checkpoint/val_metrics.pkl` |
| 3. Whisper -> WER | `scripts/train_e2e.py` | placeholder (`src/stage3_whisper.py`) | - |
| GRU baseline | `scripts/train.py` | `baseline_rnn` | val PER (shipped checkpoint: 10.1%) |

Code: `src/spikegpt/` (vendored SpikeGPT, BSD-2; CUDA WKV kernel, Metal WKV + LIF kernels for Apple GPUs),
`src/neural_encoder.py` (per-source stems + shared backbone + heads), `src/pretrain/sources.py` (datasets),
`src/ssl_trainer.py`, `src/ctc_trainer.py`, `src/train_utils.py`. Tests: `tests/test_spikegpt_kernels.py`.

Every config value can be overridden on the CLI as `key=value` (OmegaConf dotlist). Main knobs:

- `data_root=/path` (or env `DATA_ROOT`): parent of `hdf5_data_final/` and `allen/pretrain_small/`
- `b2t_sessions=[t15.2023.08.13,t15.2023.08.18,...]`: B2T subset (default `all`). For the SSL stem to be reused, Stage 2 must use the same sessions as Stage 1; otherwise a new stem is trained (the log says which)
- which datasets: pick the SSL config (`ssl_human` / `ssl_mouse` / `ssl_human_mouse`), or edit its `sources:` list (type, weight, per-source batch size and augmentations)
- Stage 2 variants: `init=ssl init_checkpoint=...` | `init=random` (control) | `freeze_backbone_steps=-1` (linear probe)
- budget: `num_training_batches`, `lr_decay_steps`, `lr_warmup_steps`, `batches_per_val_step`; device: `device=cuda:1`
- CTC batch: `source.batch_size=16 source.days_per_batch=2`

```bash
# cluster pipeline
./setup_env.sh && .venv-train/bin/pip install ninja      # ninja + nvcc compile the WKV kernel
.venv-train/bin/python tests/test_spikegpt_kernels.py   # CUDA kernel vs reference
export DATA_ROOT=/path/to/data
.venv-train/bin/python scripts/pretrain_ssl.py --config configs/ssl_human_gpu.yaml --run_name ssl_human_gpu
.venv-train/bin/python scripts/finetune_ctc.py --config configs/ctc_human_gpu.yaml --run_name ctc_ssl_gpu
.venv-train/bin/python scripts/finetune_ctc.py --config configs/ctc_human_gpu.yaml --run_name ctc_random_gpu init=random
.venv-train/bin/python scripts/train.py --config configs/baseline_rnn.yaml --run_name rnn_baseline dataset.dataset_dir=$DATA_ROOT/hdf5_data_final
```
