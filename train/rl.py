"""Stage 3: reinforcement learning with RLOO and a completion bonus.

For each question the model samples K CoTs (temperature 1, at most `budget` tokens) and answers after each one. Each rollout gets

    r = 1[right] + beta * 1[right and the CoT finished inside the budget]          (the completion bonus; beta = 0.25)
    R = r - kl_coef * KL,   KL = sum over the CoT's tokens of log pi(token) - log pi_ref(token)

where pi_ref is the adapter frozen at the start of RL. The advantage is leave-one-out (RLOO): A_i = R_i - mean of the other K - 1
returns. Example with beta = 0.25 and no KL: rollouts (right, finished), (right, truncated), (wrong), (wrong) get r = 1.25, 1, 0, 0
and A = +0.92, +0.58, -0.75, -0.75. A question whose K rollouts all score the same teaches nothing, so build the pool from questions
the model solves 1 to K-1 times out of K (a pass rate above 0 and below 1).

Loss per question (weights as in the SFT stages):
    1/2 * non-thinking answer cross-entropy
  + 1/K * sum over rollouts [ -A_i * log pi(CoT_i)  +  1/2 * answer cross-entropy after CoT_i if rollout i was right ]
plus a replay loss (weight 0.25) on direct-answer questions from earlier stages, so the non-thinking mode is kept.

Usage: python -m train.rl --config train/configs/stage3_rl_round2.yaml
"""
import argparse
import time

import torch
from torch.nn import functional as F

from openverdict.decide import Encoder, append_tokens, generate_cots, to_device
from openverdict.format import gold_indices, is_hard, loss_from_logits
from openverdict.model import adapter_state, hidden_states, option_logits
from train.common import (Questions, Sampler, adapter_swapped, check_finite, ensure_dir, load_config, log, make_optimizer, save,
                          setup_model)


def blocked_token_ids(tokenizer, stop_id):
    """Tokens a CoT may never contain: every special or template token except "</think>" (which ends the CoT)."""
    blocked = set(tokenizer.all_special_ids)
    for text, tid in tokenizer.get_added_vocab().items():
        if text.startswith('<|') or text.endswith('|>') or text == '<think>':
            blocked.add(tid)
    blocked.discard(stop_id)
    return sorted(blocked)


def rollout_forward(backbone, item, encoder, gen, blocked, device):
    """One pass over prompt + CoT + "</think>\\n\\n": per-token log-probabilities of the generated tokens, and the answer logits."""
    inputs = append_tokens(to_device(item.inputs, device), [*gen['cot_ids'], *encoder.suffix_ids])
    h = hidden_states(backbone, inputs)[0]
    answer = option_logits(backbone, h[-1], item.code_ids)
    start = item.inputs['input_ids'].shape[1] - 1
    ids = gen['generated_ids']
    weight = backbone.get_output_embeddings().weight
    logits = F.linear(h[start:start + len(ids)].float(), weight.float())
    if blocked:
        logits = logits.index_fill(-1, torch.tensor(blocked, device=logits.device), -torch.inf)
    logp = -F.cross_entropy(logits, torch.tensor(ids, device=logits.device), reduction='none')
    return logp, answer


def completion_reward(right, finished, beta):
    return float(right) + (beta if right and finished else 0.0)


def rloo_advantages(returns):
    r = torch.tensor(returns, dtype=torch.float32)
    return r - (r.sum() - r) / (len(r) - 1)


def question_update(backbone, encoder, case, qid, reference, cfg, scale, blocked, device, stats):
    """Sample, score and back-propagate one RL question. Gradients accumulate; the caller steps the optimizer."""
    item = encoder.encode(case, qid, thinking=True, require_target=True)
    if not is_hard(item.question):
        raise ValueError('RL questions need exactly one right answer (or an acceptable set)')
    gold = set(gold_indices(item.question))
    k = cfg['rollouts_per_question']
    gens = generate_cots(backbone, encoder, item, budget=cfg['budget'], samples=k, temperature=1.0, blocked_ids=blocked)
    returns, rights = [], []
    for g in gens:
        with torch.no_grad():
            old_logp, answer = rollout_forward(backbone, item, encoder, g, blocked, device)
        with adapter_swapped(backbone, reference):
            ref_logp, _ = rollout_forward(backbone, item, encoder, g, blocked, device)
        right = int(answer.argmax()) in gold and g['valid']
        kl = float((old_logp - ref_logp).sum())
        returns.append(completion_reward(right, not g['truncated'], cfg['completion_bonus']) - cfg['kl_coef'] * kl)
        rights.append(right)
        stats['rollouts'] += 1
        stats['right'] += right
        stats['truncated'] += g['truncated']
        stats['cot_tokens'] += len(g['cot_ids'])
        stats['kl'] += kl
    adv = rloo_advantages(returns)
    backbone.train()
    direct_item = encoder.encode(case, qid, thinking=False, require_target=True)
    h = hidden_states(backbone, to_device(direct_item.inputs, device))
    direct = loss_from_logits(option_logits(backbone, h[0, -1], direct_item.code_ids), direct_item.question)
    (scale * 0.5 * direct).backward()
    for g, a, right in zip(gens, adv, rights):
        logp, answer = rollout_forward(backbone, item, encoder, g, blocked, device)
        loss = -a.to(logp.device) * logp.sum()
        if right:
            loss = loss + 0.5 * loss_from_logits(answer, item.question)
        (scale / k * loss).backward()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg['output_dir'])
    torch.manual_seed(cfg.get('seed', 19))
    backbone, processor, params = setup_model(cfg, args.device)
    encoder = Encoder(processor, image_root=cfg.get('image_root', '.'))
    blocked = blocked_token_ids(encoder.tokenizer, encoder.stop_id)
    if cfg.get('reference_adapter'):
        from safetensors.torch import load_file
        reference = load_file(f"{cfg['reference_adapter']}/adapter.safetensors")
    else:
        reference = adapter_state(backbone)             # frozen copy of the policy at the start of RL
    train_set, replay = Questions(cfg['rl_train_file']), Questions(cfg['replay_file'])
    factors, replay_factors = train_set.type_factors(), replay.type_factors()
    sampler = Sampler(train_set, batch_size=cfg['batch_size'], seed=cfg.get('seed', 19))
    replay_sampler = Sampler(replay, batch_size=cfg['batch_size'], seed=cfg.get('seed', 19) + 1)
    optimizer = make_optimizer(params, cfg)                 # constant learning rate during RL
    for step in range(cfg['updates']):
        started = time.monotonic()
        stats = {'rollouts': 0, 'right': 0, 'truncated': 0, 'cot_tokens': 0, 'kl': 0.0}
        optimizer.zero_grad(set_to_none=True)
        batch = sampler.next_batch()
        for case_id, qid, kind in batch:
            question_update(backbone, encoder, train_set.cases[case_id], qid, reference, cfg, factors[kind] / len(batch), blocked,
                            args.device, stats)
        backbone.train()
        replay_batch = replay_sampler.next_batch()
        for case_id, qid, kind in replay_batch:
            item = encoder.encode(replay.cases[case_id], qid, thinking=False, require_target=True)
            h = hidden_states(backbone, to_device(item.inputs, args.device))
            loss = loss_from_logits(option_logits(backbone, h[0, -1], item.code_ids), item.question)
            (cfg['replay_weight'] * replay_factors[kind] / len(replay_batch) * loss).backward()
        norm = torch.nn.utils.clip_grad_norm_(params, cfg.get('clip_norm', 1.0), error_if_nonfinite=True)
        optimizer.step()
        check_finite(params)
        n = max(1, stats['rollouts'])
        log(out / 'train_log.jsonl', {'step': step + 1, 'accuracy': stats['right'] / n, 'truncated': stats['truncated'] / n,
                                      'mean_cot_tokens': stats['cot_tokens'] / n, 'mean_kl': stats['kl'] / n,
                                      'grad_norm': float(norm), 'seconds': time.monotonic() - started})
        if (step + 1) % cfg.get('save_every', 20) == 0:
            save(backbone, out, step + 1)
    print('saved', save(backbone, out, cfg['updates']))


if __name__ == '__main__':
    main()
