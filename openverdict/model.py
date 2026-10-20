"""The model: a frozen base language model with one LoRA adapter, and how the answer is read from it.

The adapter is added to 12 kinds of linear layers inside the language model (attention, the linear-attention projections and the MLP);
the vision encoder is left untouched. Adapter weights are kept in float32 and added on top of the frozen layer:

    y = W x + (alpha / rank) * B (A x)          A: [rank, in], B: [out, rank], B starts at zero

Reading the answer: take the last hidden state at the answer position and multiply it by the output-embedding rows of the option codes.
That gives one logit per option, computed for all options in the same pass.
"""
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

BASE_MODEL = 'Qwen/Qwen3.5-4B'
ADAPTER_TARGETS = frozenset({'q_proj', 'k_proj', 'v_proj', 'o_proj', 'in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a',
                             'out_proj', 'gate_proj', 'up_proj', 'down_proj'})


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        out = self.base(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            delta = F.linear(F.linear(x.float(), self.lora_A), self.lora_B) * self.scale
        return out + delta.to(out.dtype)


def language_model(backbone):
    """The text decoder of a vision-language model, or the model itself for a text-only one."""
    inner = backbone.model
    return getattr(inner, 'language_model', inner)


def install_lora(backbone, *, rank=16, alpha=32):
    """Freeze every base weight and wrap the target linear layers of the language model. Returns the wrapped layer names."""
    lm = language_model(backbone)
    for p in backbone.parameters():
        p.requires_grad_(False)
    names = [n for n, m in lm.named_modules() if isinstance(m, nn.Linear) and n.rsplit('.', 1)[-1] in ADAPTER_TARGETS]
    if not names:
        raise ValueError('no adapter target layers found')
    for name in names:
        parent, attr = name.rsplit('.', 1)
        setattr(lm.get_submodule(parent), attr, LoRALinear(lm.get_submodule(name), rank, alpha))
    return names


def widen_lora(backbone, *, rank, alpha):
    """Grow every adapter to a larger rank without changing what the model computes.

    The new rows of A get the usual random start and the new columns of B are zero, so B A is unchanged; alpha / rank is kept
    constant (16 / 32 -> 32 / 64 keeps the scale at 2), so the output is identical right after widening. Only capacity grows.
    """
    for m in language_model(backbone).modules():
        if not isinstance(m, LoRALinear):
            continue
        old = m.lora_A.shape[0]
        if rank < old or not math.isclose(alpha / rank, m.scale, abs_tol=1e-12):
            raise ValueError('widening must keep alpha / rank and cannot lower the rank')
        if rank == old:
            continue
        with torch.no_grad():
            a = m.lora_A.new_empty(rank, m.lora_A.shape[1])
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            b = m.lora_B.new_zeros(m.lora_B.shape[0], rank)
            a[:old].copy_(m.lora_A)
            b[:, :old].copy_(m.lora_B)
        m.lora_A, m.lora_B = nn.Parameter(a), nn.Parameter(b)


def adapter_state(backbone):
    return {n: p.detach().cpu().clone() for n, p in backbone.named_parameters() if n.endswith(('lora_A', 'lora_B'))}


def load_adapter_state(backbone, state):
    params = {n: p for n, p in backbone.named_parameters() if n.endswith(('lora_A', 'lora_B'))}
    if set(state) != set(params):
        raise ValueError('adapter tensor names differ from the model')
    with torch.no_grad():
        for n, p in params.items():
            if state[n].shape != p.shape:
                raise ValueError(f'adapter shape differs: {n}')
            p.copy_(state[n].to(p))


def save_adapter(backbone, path, *, rank, alpha):
    from safetensors.torch import save_file
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_file(adapter_state(backbone), str(path / 'adapter.safetensors'), metadata={'rank': str(rank), 'alpha': str(alpha)})


def load_adapter(backbone, path):
    from safetensors import safe_open
    from safetensors.torch import load_file
    file = str(Path(path) / 'adapter.safetensors')
    with safe_open(file, 'pt') as f:
        meta = f.metadata()
    install_lora(backbone, rank=int(meta['rank']), alpha=float(meta['alpha']))
    load_adapter_state(backbone, load_file(file))


def load_base(model_name=BASE_MODEL, *, device='cpu', dtype=torch.bfloat16):
    """The base model and its processor. Image questions need the vision-language class; it also answers text-only questions."""
    from transformers import AutoProcessor, AutoModelForImageTextToText
    processor = AutoProcessor.from_pretrained(model_name)
    backbone = AutoModelForImageTextToText.from_pretrained(model_name, dtype=dtype).to(device).eval()
    return backbone, processor


def hidden_states(backbone, inputs):
    """Last-layer hidden states of the language model, [batch, tokens, hidden]."""
    return backbone.model(**inputs, use_cache=False, return_dict=True).last_hidden_state


def option_logits(backbone, hidden, code_ids):
    """The logits of the option letters: the hidden state at the answer position times their output-embedding rows."""
    weight = backbone.get_output_embeddings().weight
    codes = torch.as_tensor(code_ids, dtype=torch.long, device=weight.device)
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        return hidden.float() @ weight.index_select(0, codes).float().T
