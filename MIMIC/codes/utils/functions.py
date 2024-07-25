import argparse
from omegaconf import OmegaConf
import os
from pathlib import Path


def setup_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str)
    parser.add_argument('--seed', type=int)
    _args = parser.parse_args()
    arg_seed = _args.seed
    args = OmegaConf.load(_args.config)
    if arg_seed is not None:
        args.seed = arg_seed
    return args

def setup_inf_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--hparams', type=str)
    _args = parser.parse_args()
    args = OmegaConf.load(_args.hparams)
    hparams_path = Path(_args.hparams)
    ckpt_path = hparams_path.parent / 'checkpoints'
    #find the latest checkpoint
    ckpt_path = max(ckpt_path.glob('*.ckpt'))
    args.ckpt_path = str(ckpt_path)
    return args
