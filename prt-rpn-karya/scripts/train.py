"""Train the baseline RNN phoneme decoder. Paths are relative to the prt-rpn-karya root (works from any cwd):

    .venv-train/bin/python scripts/train.py --config configs/baseline_rnn.yaml --run_name my_run
"""
import argparse
# all relative paths (configs, ../data, experiments/) resolve from the prt-rpn-karya root, whatever the cwd
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer

parser = argparse.ArgumentParser(description='Train the RNN phoneme decoder.')
parser.add_argument('--config', type=str, default='configs/baseline_rnn.yaml')
parser.add_argument('--run_name', type=str, default=None,
                    help='Writes outputs to experiments/<run_name>/ (overrides output_dir/checkpoint_dir in the config).')
parser.add_argument('overrides', nargs='*', help='OmegaConf dotlist overrides, e.g. num_training_batches=2000 gpu_number=1')
cli = parser.parse_args()

args = OmegaConf.merge(OmegaConf.load(cli.config), OmegaConf.from_dotlist(cli.overrides))
if cli.run_name:
    args.output_dir = os.path.join('experiments', cli.run_name)
    args.checkpoint_dir = os.path.join(args.output_dir, 'checkpoint')

trainer = BrainToTextDecoder_Trainer(args)
metrics = trainer.train()
