import os
import sys

import torch
from torch import nn


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_BEATS_DIR = os.path.join(_REPO_ROOT, 'third_party', 'unilm', 'beats')

if _BEATS_DIR not in sys.path:
    sys.path.insert(0, _BEATS_DIR)

from BEATs import BEATs, BEATsConfig


class BEATsWrapper(nn.Module):
    def __init__(self, checkpoint_path):
        super(BEATsWrapper, self).__init__()

        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(_REPO_ROOT, checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        cfg = BEATsConfig(checkpoint['cfg'])

        if cfg.finetuned_model:
            raise ValueError(
                'Use a BEATs pre-trained checkpoint for encoder features.')

        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint['model'])
        self.output_dim = cfg.encoder_embed_dim
        self.num_attention_heads = cfg.encoder_attention_heads

    def forward(self, waveforms, padding_mask=None):
        if waveforms.dim() != 2:
            raise ValueError(
                'BEATs expects raw 16 kHz waveforms with shape B x T, '
                'got {}'.format(tuple(waveforms.shape)))

        return self.model.extract_features(
            waveforms, padding_mask=padding_mask)
