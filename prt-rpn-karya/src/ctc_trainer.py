"""Stage 2: phoneme CTC fine-tuning of the (SSL-pretrained) SpikeGPT backbone on B2T -> val PER.

init: 'ssl'    load the backbone (and the B2T stem, if Stage 1 trained one) from init_checkpoint
      'random' same architecture, no pretraining (the control that isolates the effect of SSL)
freeze_backbone_steps: N  backbone frozen (requires_grad=False) for the first N steps while the stem
                          (per-day scale/bias, w_in, patch_proj) and ctc_head warm up; then everything trains
                       -1  frozen for the whole run (linear probe of the SSL representation)
PER: greedy CTC decoding on the 1426 val trials, same definition as the GRU baseline (rnn_trainer.validation).
"""
import json
import os
import pickle
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from neural_encoder import build_model, n_tokens
from pretrain.sources import B2TSource
import train_utils as tu


class CTCTrainer:
    def __init__(self, args):
        self.args = args
        os.makedirs(args['output_dir'], exist_ok=False)
        os.makedirs(args['checkpoint_dir'], exist_ok=True)
        self.logger = tu.make_logger(args['output_dir'], 'ctc')
        self.device = tu.get_device(args.get('device', 'auto'))
        self.logger.info(f'Using device: {self.device}')
        tu.set_seed(args['seed'])

        src_cfg = OmegaConf.to_container(args['source'], resolve=True)
        self.transform_cfg = src_cfg.pop('data_transforms', {})
        src_cfg.pop('type', None)
        self.source = B2TSource(**src_cfg)
        self.name = self.source.name
        meta = {self.name: self.source.meta()}

        model_cfg = OmegaConf.to_container(args['model'], resolve=True)
        ckpt = None
        if args['init'] == 'ssl':
            ckpt = torch.load(args['init_checkpoint'], map_location='cpu', weights_only=False)
            model_cfg = {**ckpt['model_cfg'], 'input_dropout': model_cfg['input_dropout'],
                         'backbone_dropout': model_cfg['backbone_dropout']}  # architecture comes from Stage 1
        self.model_cfg = model_cfg
        self.P, self.S = model_cfg['patch_size'], model_cfg['patch_stride']
        self.model = build_model(model_cfg, meta, args.get('use_native_kernels', True),
                                 args.get('grad_checkpoint', False))
        if ckpt is not None:
            self._load_pretrained(ckpt)
        self.model.to(self.device)

        self.freeze_steps = int(args['freeze_backbone_steps'])
        lrs = {'backbone': args['lr_max_backbone'], 'adapters': args['lr_max'], 'ctc_head': args['lr_max']}
        self.optimizer = tu.make_optimizer(self.model, lrs, args, self.device)
        self.scheduler = tu.cosine_scheduler(self.optimizer, args['lr_min'] / args['lr_max'], args['lr_decay_steps'],
                                             args['lr_warmup_steps'])
        self._set_backbone_trainable(self.freeze_steps == 0)
        self.best_PER, self.best_loss = float('inf'), float('inf')
        tu.save_args(args, os.path.join(args['output_dir'], 'args.yaml'))

    def _load_pretrained(self, ckpt):
        sd = ckpt['model_state_dict']
        own = self.model.state_dict()
        load = {k: v for k, v in sd.items() if k.startswith('backbone.')}
        stem_prefix = f'adapters.{self.name}.'
        same_sessions = ckpt['sources_meta'].get(self.name, {}).get('sessions') == self.source.sessions
        if same_sessions:  # Stage 1 saw this source: keep its stem (day scale/bias, w_in, patch_proj)
            load.update({k: v for k, v in sd.items() if k.startswith(stem_prefix) and '.ar_head.' not in k})
        bad = [k for k, v in load.items() if own[k].shape != v.shape]
        assert not bad, f'shape mismatch: {bad[:5]}'
        missing, _ = self.model.load_state_dict(load, strict=False)
        self.logger.info(f"Loaded {len(load)} tensors from {self.args['init_checkpoint']} (SSL step {ckpt['step']}); "
                         f"stem {'loaded' if same_sessions else 'NEW (source not in Stage 1)'}; "
                         f"{len(missing)} tensors freshly initialised")

    def _set_backbone_trainable(self, flag):
        for p in self.model.backbone.parameters():
            p.requires_grad = flag
        n = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Backbone {'trainable' if flag else 'FROZEN'}; trainable params {n:,}")

    def _forward(self, batch, train):
        x = batch['x'].to(self.device).float()
        lengths = batch['lengths'].to(self.device)
        x, lengths, _ = tu.transform(x, lengths, self.transform_cfg, train, self.device)
        x = tu.pad_time(x, self.args.get('pad_time_multiple', 128))
        with tu.autocast(self.device, self.args['use_amp']):
            logits = self.model.forward_ctc(x, self.name, batch['session'].to(self.device))
        out_lens = n_tokens(lengths, self.P, self.S).clamp(min=1).to(torch.int32)
        loss = tu.ctc_loss(logits, batch['phonemes'].to(self.device), out_lens, batch['phone_lens'].to(self.device))
        return logits, out_lens, loss

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        tot_ed, tot_len, losses = 0, 0, []
        day = {}
        for batch in self.source.val_batches():
            logits, out_lens, loss = self._forward(batch, train=False)
            losses.append(loss.item())
            d = int(batch['session'][0])
            day.setdefault(d, [0, 0])
            for seq, true, n in zip(tu.greedy_decode(logits, out_lens), batch['phonemes'], batch['phone_lens']):
                ed = tu.edit_distance(seq, true[:n].numpy())
                tot_ed += ed; tot_len += int(n)
                day[d][0] += ed; day[d][1] += int(n)
        self.model.train()
        return {'avg_PER': tot_ed / tot_len, 'avg_loss': float(np.mean(losses)),
                'day_PERs': {self.source.sessions[d]: e / n for d, (e, n) in day.items()}}

    def save(self, path, step, val):
        torch.save({'model_state_dict': self.model.state_dict(), 'sources_meta': {self.name: self.source.meta()},
                    'model_cfg': self.model_cfg,
                    'step': step, 'val_PER': val['avg_PER'], 'val_loss': val['avg_loss'], 'stage': 'ctc'}, path)
        self.logger.info(f'Saved checkpoint: {path}')

    def train(self):
        a = self.args
        self.model.train()
        history = []
        t_start = time.time()
        for i, batch in enumerate(self.source.train_batches(a['num_training_batches'])):
            if i == self.freeze_steps and self.freeze_steps > 0:
                self._set_backbone_trainable(True)
            t0 = time.time()
            self.optimizer.zero_grad()
            _, _, loss = self._forward(batch, train=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],
                                                       a['grad_norm_clip_value'])
            self.optimizer.step()
            self.scheduler.step()

            if i % a['batches_per_train_log'] == 0:
                self.logger.info(f'Train batch {i}: loss {loss.item():.2f} grad norm {grad_norm:.2f} time {time.time() - t0:.3f}')
            if i % a['batches_per_val_step'] == 0 or i == a['num_training_batches'] - 1:
                t0 = time.time()
                val = self.validate()
                self.logger.info(f"Val batch {i}: PER (avg): {val['avg_PER']:.4f} CTC Loss (avg): {val['avg_loss']:.4f} "
                                 f"time: {time.time() - t0:.1f}s")
                history.append({'step': i, 'PER': val['avg_PER'], 'loss': val['avg_loss'], 'train_loss': loss.item()})
                if val['avg_PER'] < self.best_PER or (val['avg_PER'] == self.best_PER and val['avg_loss'] < self.best_loss):
                    self.logger.info(f"New best val PER {self.best_PER:.4f} --> {val['avg_PER']:.4f}")
                    self.best_PER, self.best_loss = val['avg_PER'], val['avg_loss']
                    self.save(os.path.join(a['checkpoint_dir'], 'best_checkpoint'), i, val)
                    with open(os.path.join(a['checkpoint_dir'], 'val_metrics.pkl'), 'wb') as f:
                        pickle.dump(val, f)
                with open(os.path.join(a['output_dir'], 'val_history.json'), 'w') as f:
                    json.dump(history, f, indent=1)
        self.logger.info(f'Best avg val PER achieved: {self.best_PER:.5f}')
        self.logger.info(f'Total training time: {(time.time() - t_start) / 60:.2f} minutes')
        return history
