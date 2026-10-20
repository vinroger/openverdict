"""End-to-end smoke test on CPU: a tiny, randomly initialised model of the same architecture, with the real tokenizer.

It checks that every training step runs and sends gradients only to the adapter. It says nothing about accuracy.
Skipped when the base model's tokenizer and config are not downloaded.
"""
import copy

import pytest
import torch

transformers = pytest.importorskip('transformers')

from openverdict.decide import Encoder, non_thinking, think                       # noqa: E402
from openverdict.model import LoRALinear, install_lora, widen_lora                 # noqa: E402
from train.common import adapter_swapped                                           # noqa: E402
from openverdict.model import adapter_state                                        # noqa: E402
from train.rl import blocked_token_ids, question_update                            # noqa: E402
from train.sft import question_loss                                                # noqa: E402

CASE = {'id': 'demo', 'state': 'Leda tells the truth. Shalonda says Leda lies.',
        'questions': [{'id': 'q', 'kind': 'noul', 'instructions': 'Does Shalonda tell the truth?',
                       'options': [{'id': 'true', 'description': 'yes'}, {'id': 'false', 'description': 'no'}],
                       'target': {'format': 'distribution', 'values': {'true': 0.0, 'false': 1.0}},
                       'cot': {'text': 'Leda=T. Shalonda=F.'}}]}


@pytest.fixture(scope='module')
def tiny():
    try:
        tok = transformers.AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B', local_files_only=True)
        cfg = transformers.AutoConfig.from_pretrained('Qwen/Qwen3.5-4B', local_files_only=True).text_config
    except OSError:
        pytest.skip('base model files not downloaded')
    cfg = copy.deepcopy(cfg)
    cfg.update(dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                    head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
                    layer_types=['linear_attention'] * 3 + ['full_attention']))
    torch.manual_seed(0)
    model = transformers.Qwen3_5ForCausalLM(cfg).float().eval()
    install_lora(model, rank=4, alpha=8)
    return model, Encoder(tok)


def test_adapter_is_the_only_trainable_part(tiny):
    model, _ = tiny
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(n.endswith(('lora_A', 'lora_B')) for n in trainable)
    assert any(isinstance(m, LoRALinear) for m in model.modules())


def test_widening_keeps_the_output(tiny):
    model, enc = tiny
    m = copy.deepcopy(model)
    with torch.no_grad():
        for p in m.parameters():
            if p.requires_grad:
                p.add_(0.01 * torch.randn_like(p))
    item = enc.encode(CASE, 'q', thinking=False)
    before = non_thinking(m, item)['logits']
    widen_lora(m, rank=8, alpha=16)
    after = non_thinking(m, item)['logits']
    assert torch.allclose(before, after, atol=1e-5)


def test_both_modes_return_a_distribution(tiny):
    model, enc = tiny
    out = think(model, enc, enc.encode(CASE, 'q', thinking=True), budget=6)
    assert abs(sum(out['probabilities'].values()) - 1) < 1e-5 and out['cot_tokens'] <= 6
    assert abs(sum(non_thinking(model, enc.encode(CASE, 'q', thinking=False))['probabilities'].values()) - 1) < 1e-5


def test_sft_loss_with_a_cot_backpropagates_to_the_adapter(tiny):
    model, enc = tiny
    model.train()
    model.zero_grad(set_to_none=True)
    loss, parts = question_loss(model, enc, CASE, 'q', 'cpu')
    loss.backward()
    assert set(parts) == {'direct', 'answer', 'cot'} and torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
    truncated = copy.deepcopy(CASE)
    truncated['questions'][0]['cot'] = {'text': 'Leda=T. Shalonda=F.', 'cut': 3}
    _, parts = question_loss(model, enc, truncated, 'q', 'cpu')
    assert set(parts) == {'direct', 'answer'}


def test_one_rl_question_runs(tiny):
    model, enc = tiny
    model.zero_grad(set_to_none=True)
    reference = adapter_state(model)
    cfg = {'rollouts_per_question': 2, 'budget': 6, 'completion_bonus': 0.25, 'kl_coef': 0.001}
    stats = {'rollouts': 0, 'right': 0, 'truncated': 0, 'cot_tokens': 0, 'kl': 0.0}
    question_update(model, enc, CASE, 'q', reference, cfg, 1.0, blocked_token_ids(enc.tokenizer, enc.stop_id), 'cpu', stats)
    assert stats['rollouts'] == 2 and abs(stats['kl']) < 1e-4          # policy == reference at the start: KL is zero
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad and p.grad is not None)


def test_reference_swap_restores_the_policy(tiny):
    model, _ = tiny
    before = adapter_state(model)
    other = {k: torch.zeros_like(v) for k, v in before.items()}
    with adapter_swapped(model, other):
        pass
    after = adapter_state(model)
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_saved_adapter_reloads_to_the_same_answers(tiny, tmp_path):
    from openverdict.model import load_adapter, save_adapter
    model, enc = tiny
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(0.01 * torch.randn_like(p))
    save_adapter(model, tmp_path, rank=4, alpha=8)
    fresh = transformers.Qwen3_5ForCausalLM(model.config).float().eval()
    fresh.load_state_dict({k: v for k, v in model.state_dict().items() if 'lora_' not in k and '.base.' not in k}, strict=False)
    base = {k.replace('.base.', '.'): v for k, v in model.state_dict().items() if '.base.' in k}
    fresh.load_state_dict(base, strict=False)
    load_adapter(fresh, tmp_path)
    item = enc.encode(CASE, 'q', thinking=False)
    assert torch.allclose(non_thinking(model, item)['logits'], non_thinking(fresh, item)['logits'], atol=1e-5)
