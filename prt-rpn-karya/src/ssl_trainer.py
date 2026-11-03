"""Stage 1: causal autoregressive SSL of the shared SpikeGPT backbone on one or more neural sources.

Objective: the latent of token t (bins [tS, tS+P)) predicts the next H raw bins [tS+P, tS+P+H).
Input smoothing must be causal (train_utils.causal_smooth); an acausal kernel would leak the target into the input.
Loss: MSE for z-scored features (B2T), Poisson NLL on log-rates for spike counts (mouse).
Val reports the loss next to two trivial predictors the model has to beat:
    linear:   zero (= per-feature mean of z-scored data) and persistence (repeat the last observed bin)
    log_rate: per-unit mean training rate
"""
import json
import os
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from neural_encoder import build_model, n_tokens
from pretrain.sources import build_sources, mixture
import train_utils as tu


def ar_targets(target, P, S, H):
    """target [B, T, C] -> [B, T2, H, C] with row t = bins tS+P ... tS+P+H-1, and the last observed bin [B, T2, C]."""
    tgt = target[:, P:].unfold(1, H, S).permute(0, 1, 3, 2)  # [B, T2, H, C]
    last = target[:, P - 1::S][:, :tgt.shape[1]]  # bin tS+P-1
    return tgt, last


def ar_loss(pred, tgt, valid, kind):
    """pred, tgt [B, T2, H, C]; valid [B, T2] bool. Mean over valid tokens, horizon and channels."""
    if kind == 'linear':
        err = (pred - tgt) ** 2
    else:  # log_rate: Poisson NLL without the log(y!) constant
        err = torch.exp(pred) - tgt * pred
    err = err.mean(dim=(2, 3))
    return (err * valid).sum() / valid.sum().clamp(min=1)


class SSLTrainer:
    def __init__(self, args):
        self.args = args
        os.makedirs(args['output_dir'], exist_ok=False)
        os.makedirs(args['checkpoint_dir'], exist_ok=True)
        self.logger = tu.make_logger(args['output_dir'], 'ssl')
        self.device = tu.get_device(args.get('device', 'auto'))
        self.logger.info(f'Using device: {self.device}')
        tu.set_seed(args['seed'])
        self.rng = np.random.default_rng(args['seed'])

        source_cfgs = OmegaConf.to_container(args['sources'], resolve=True)
        self.transforms = {c.get('name', c['type']): c.pop('data_transforms', {}) for c in source_cfgs}
        self.sources, self.weights = build_sources(source_cfgs)
        self.meta = {n: s.meta() for n, s in self.sources.items()}
        for n, m in self.meta.items():
            self.logger.info(f"source {n}: {len(m['sessions'])} sessions, channels {sorted(set(m['channels']))}, "
                             f"weight {self.weights[n]}, loss {m['ar_output']}")

        m = args['model']
        self.P, self.S, self.H = m['patch_size'], m['patch_stride'], m['ar_horizon']
        self.model = build_model(m, self.meta, args.get('use_native_kernels', True),
                                 args.get('grad_checkpoint', False)).to(self.device)
        n_backbone = sum(p.numel() for p in self.model.backbone.parameters())
        n_adapt = sum(p.numel() for p in self.model.adapters.parameters())
        self.logger.info(f'SpikeGPT backbone: {n_backbone:,} params | stems + AR heads: {n_adapt:,} params')

        lrs = {'backbone': args['lr_max'], 'adapters': args['lr_max']}
        self.optimizer = tu.make_optimizer(self.model, lrs, args, self.device)
        self.scheduler = tu.cosine_scheduler(self.optimizer, args['lr_min'] / args['lr_max'], args['lr_decay_steps'],
                                             args['lr_warmup_steps'])
        self.best_val = float('inf')
        tu.save_args(args, os.path.join(args['output_dir'], 'args.yaml'))

    def step_loss(self, name, batch, train):
        kind = self.meta[name]['ar_output']
        tcfg = self.transforms[name]
        batch = tu.random_crop(batch, self.args['max_bins'], self.rng) if train else \
            tu.random_crop(batch, self.args['max_bins'], np.random.default_rng(0))
        x = batch['x'].to(self.device).float()
        target = batch['target'].to(self.device).float()
        lengths = batch['lengths'].to(self.device)
        session = batch['session'].to(self.device)
        x, _, _ = tu.transform(x, lengths, tcfg, train, self.device)  # no random_cut here: target must stay aligned
        x = tu.pad_time(x, self.args.get('pad_time_multiple', 128))
        target = tu.pad_time(target, self.args.get('pad_time_multiple', 128))

        with tu.autocast(self.device, self.args['use_amp']):
            pred = self.model.forward_ar(x, name, session)
        tgt, last = ar_targets(target, self.P, self.S, self.H)
        T2 = tgt.shape[1]
        pred = pred[:, :T2].float()
        valid = (torch.arange(T2, device=self.device) * self.S + self.P + self.H)[None] <= lengths[:, None]
        loss = ar_loss(pred, tgt, valid, kind)
        if train:
            return loss, None, None
        if kind == 'linear':
            base = {'zero': ar_loss(torch.zeros_like(tgt), tgt, valid, kind),
                    'persistence': ar_loss(last.unsqueeze(2).expand_as(tgt), tgt, valid, kind)}
        else:
            mu = torch.as_tensor(self.sources[name].mu[int(session[0])], device=self.device)
            base = {'mean_rate': ar_loss(torch.log(mu.clamp(min=1e-3)).expand_as(tgt), tgt, valid, kind)}
        return loss, {k: v.item() for k, v in base.items()}, valid.sum().item()

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        out = {}
        for name, src in self.sources.items():
            tot, n, base = 0., 0, {}
            for batch in src.val_batches(self.args.get('val_max_batches')):
                loss, b, nv = self.step_loss(name, batch, train=False)
                tot += loss.item() * nv; n += nv
                for k, v in b.items():
                    base[k] = base.get(k, 0.) + v * nv
            out[name] = {'loss': tot / n, **{k: v / n for k, v in base.items()}}
        self.model.train()
        return out

    def save(self, path, step, val):
        torch.save({'model_state_dict': self.model.state_dict(), 'sources_meta': self.meta,
                    'model_cfg': OmegaConf.to_container(self.args['model'], resolve=True),
                    'step': step, 'val': val, 'stage': 'ssl'}, path)
        self.logger.info(f'Saved checkpoint: {path}')

    def train(self):
        a = self.args
        self.model.train()
        history = []
        t_start = time.time()
        for i, (name, batch) in enumerate(mixture(self.sources, self.weights, a['num_training_batches'], a['seed'])):
            t0 = time.time()
            self.optimizer.zero_grad()
            loss, _, _ = self.step_loss(name, batch, train=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), a['grad_norm_clip_value'])
            self.optimizer.step()
            self.scheduler.step()

            if i % a['batches_per_train_log'] == 0:
                self.logger.info(f'Train batch {i} [{name}]: loss {loss.item():.4f} grad norm {grad_norm:.2f} '
                                 f'time {time.time() - t0:.3f}')
            if i % a['batches_per_val_step'] == 0 or i == a['num_training_batches'] - 1:
                t0 = time.time()
                val = self.validate()
                total = sum(self.weights[n] * v['loss'] for n, v in val.items()) / sum(self.weights.values())
                for n, v in val.items():
                    self.logger.info(f'Val batch {i} [{n}]: ' + ' '.join(f'{k} {x:.4f}' for k, x in v.items()))
                self.logger.info(f'Val batch {i}: weighted loss {total:.4f} time {time.time() - t0:.1f}s')
                history.append({'step': i, 'val': val, 'train_loss': loss.item()})
                if total < self.best_val:
                    self.best_val = total
                    self.save(os.path.join(a['checkpoint_dir'], 'best_checkpoint'), i, val)
                with open(os.path.join(a['output_dir'], 'val_history.json'), 'w') as f:
                    json.dump(history, f, indent=1)
        self.save(os.path.join(a['checkpoint_dir'], 'final_checkpoint'), i, history[-1]['val'])
        self.logger.info(f'Best weighted val loss {self.best_val:.4f}, total time {(time.time() - t_start) / 60:.1f} min')
        return history
