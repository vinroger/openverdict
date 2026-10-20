"""Shared training pieces: config, data, question sampling, learning-rate schedule, model setup and checkpoints."""
from collections import Counter
from contextlib import contextmanager
import json
import math
from pathlib import Path
import random

import torch
import yaml

from openverdict.format import parse_case
from openverdict.model import (adapter_state, install_lora, load_adapter, load_adapter_state, load_base, save_adapter,
                               widen_lora)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class Questions:
    """Every (case, question) pair of a JSONL file, checked once at load time."""

    def __init__(self, path, *, require_target=True):
        self.cases = {}
        self.units = []
        for case in read_jsonl(path):
            for q in parse_case(case, require_target=require_target):
                self.units.append((case['id'], q.id, q.kind))
            self.cases[case['id']] = case
        if not self.units:
            raise ValueError(f'no questions in {path}')

    def type_factors(self):
        """Loss weights that give each question kind the same total weight.

        Example: 600 yes / no and 200 choice questions -> yes / no x 0.67, choice x 2.0 (each kind then adds up to half the loss).
        """
        counts = Counter(kind for _, _, kind in self.units)
        return {kind: len(self.units) / (len(counts) * n) for kind, n in counts.items()}


class Sampler:
    """Shuffled batches of question units; a new shuffle each epoch, reproducible from the seed."""

    def __init__(self, questions, *, batch_size, seed):
        self.units, self.batch_size, self.rng = list(questions.units), batch_size, random.Random(seed)
        self.order, self.position = [], 0

    def next_batch(self):
        batch = []
        while len(batch) < self.batch_size:
            if self.position >= len(self.order):
                self.order = list(self.units)
                self.rng.shuffle(self.order)
                self.position = 0
            batch.append(self.order[self.position])
            self.position += 1
        return batch


def learning_rate(step, *, peak, total, warmup, final_fraction):
    """Linear warm-up to `peak`, then a linear decay to final_fraction x peak at the last update."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    return peak * (1 - (1 - final_fraction) * progress)


def make_optimizer(params, cfg):
    return torch.optim.AdamW(params, lr=cfg['learning_rate'], betas=(0.9, 0.95), eps=1e-8, weight_decay=cfg.get('weight_decay', 0.0))


def setup_model(cfg, device):
    """Load the base model and the adapter: a fresh adapter (stage 1) or the previous stage's adapter, optionally widened."""
    dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[cfg.get('dtype', 'float32')]
    if device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32 = True     # float32 weights, TF32 matmuls
        torch.backends.cudnn.allow_tf32 = True
    backbone, processor = load_base(cfg.get('base_model', 'Qwen/Qwen3.5-4B'), device=device, dtype=dtype)
    lora = cfg['lora']
    if cfg.get('init_adapter'):
        load_adapter(backbone, cfg['init_adapter'])
        if lora.get('widen_to'):
            widen_lora(backbone, rank=lora['widen_to']['rank'], alpha=lora['widen_to']['alpha'])
    else:
        install_lora(backbone, rank=lora['rank'], alpha=lora['alpha'])
    backbone.to(device)
    if cfg.get('gradient_checkpointing', True):
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        backbone.config.use_cache = False
    params = [p for n, p in backbone.named_parameters() if p.requires_grad]
    return backbone, processor, params


def adapter_rank_alpha(backbone):
    """(rank, alpha) of the installed adapter, read from the model itself (alpha = scale x rank)."""
    from openverdict.model import LoRALinear
    for m in backbone.modules():
        if isinstance(m, LoRALinear):
            rank = m.lora_A.shape[0]
            return rank, m.scale * rank
    raise ValueError('no adapter installed')


def save(backbone, out_dir, step):
    rank, alpha = adapter_rank_alpha(backbone)
    path = Path(out_dir) / f'step-{step:06d}'
    save_adapter(backbone, path, rank=rank, alpha=alpha)
    return path


@contextmanager
def adapter_swapped(backbone, state):
    """Temporarily load other adapter weights (the frozen reference policy), then restore the current ones."""
    current = adapter_state(backbone)
    was_training = backbone.training
    try:
        load_adapter_state(backbone, state)
        backbone.eval()
        with torch.no_grad():
            yield
    finally:
        load_adapter_state(backbone, current)
        backbone.train(was_training)


def check_finite(params):
    if not all(torch.isfinite(p).all() for p in params):
        raise FloatingPointError('an update produced non-finite adapter weights')


def log(path, row):
    with open(path, 'a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def is_finite_number(x):
    return isinstance(x, (int, float)) and math.isfinite(x)
