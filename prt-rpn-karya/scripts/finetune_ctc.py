"""Stage 2: phoneme CTC fine-tuning on B2T -> val PER.

    .venv-train/bin/python scripts/finetune_ctc.py --run_name ctc_from_ssl_human init_checkpoint=experiments/ssl_human/checkpoint/best_checkpoint
    .venv-train/bin/python scripts/finetune_ctc.py --run_name ctc_random init=random
    .venv-train/bin/python scripts/finetune_ctc.py --run_name ctc_probe init_checkpoint=... freeze_backbone_steps=-1
"""
import argparse
# all relative paths (configs, ../data, experiments/) resolve from the prt-rpn-karya root, whatever the cwd
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from omegaconf import OmegaConf
from ctc_trainer import CTCTrainer

# guard: DataLoader workers re-import this file on macOS (spawn)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 2: phoneme CTC fine-tuning.')
    parser.add_argument('--config', type=str, default='configs/ctc_human.yaml')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Writes outputs to experiments/<run_name>/ (overrides output_dir/checkpoint_dir in the config).')
    parser.add_argument('overrides', nargs='*', help='OmegaConf dotlist overrides, e.g. num_training_batches=200 device=cpu')
    cli = parser.parse_args()

    args = OmegaConf.merge(OmegaConf.load(cli.config), OmegaConf.from_dotlist(cli.overrides))
    if cli.run_name:
        args.output_dir = os.path.join('experiments', cli.run_name)
        args.checkpoint_dir = os.path.join(args.output_dir, 'checkpoint')

    trainer = CTCTrainer(args)
    metrics = trainer.train()