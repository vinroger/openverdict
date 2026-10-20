"""The simple interface: load once, then ask decisions in either mode.

    from openverdict import OpenVerdict
    model = OpenVerdict.load('path/to/adapter-folder')                  # adapter.safetensors (+ calibration.json if present)
    model.decide(case)                                                   # thinking mode, budget 80
    model.decide(case, thinking=False)                                   # non-thinking mode: one forward pass
"""
from pathlib import Path

import torch

from . import calibration as cal
from .decide import Encoder, non_thinking, think
from .format import parse_case
from .model import BASE_MODEL, load_adapter, load_base


class OpenVerdict:
    def __init__(self, backbone, encoder, calibration=None):
        self.backbone, self.encoder, self.calibration = backbone, encoder, calibration

    @classmethod
    def load(cls, adapter_dir, *, base_model=BASE_MODEL, device=None, dtype=torch.bfloat16, image_root='.'):
        device = device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        backbone, processor = load_base(base_model, device=device, dtype=dtype)
        load_adapter(backbone, adapter_dir)
        backbone.to(device).eval()
        calib_path = Path(adapter_dir) / 'calibration.json'
        return cls(backbone, Encoder(processor, image_root=image_root), cal.load(calib_path) if calib_path.exists() else None)

    def decide(self, case, *, thinking=True, budget=80):
        """Answer every question of a case. Returns {question id: {'probabilities': {...}, 'answer': option id, ...}}."""
        results = {}
        for q in parse_case(case, require_target=False):
            item = self.encoder.encode(case, q.id, thinking=thinking)
            if thinking:
                out = think(self.backbone, self.encoder, item, budget=budget)
                if self.calibration:
                    out['probabilities'] = cal.apply(out['probabilities'], q.kind, self.calibration)
            else:
                out = non_thinking(self.backbone, item)
            out.pop('logits', None)
            out['answer'] = max(out['probabilities'], key=out['probabilities'].get)
            results[q.id] = out
        return results
