"""Supervised fine-tuning, used by stage 1 (decision format), stage 2 (rejection-sampling fine-tuning) and stage 4 (image SFT).

Every question trains the non-thinking answer. A question that also carries a chain-of-thought (field "cot") trains the thinking mode:

    complete CoT   {"cot": {"text": "Leda=T. Shalonda=F. ..."}}               loss = (direct + answer + CoT tokens) / 2
    truncated CoT  {"cot": {"text": "Leda=T. Shalonda=F. ...", "cut": 16}}    loss = (direct + answer) / 2

direct : cross-entropy of the non-thinking answer
answer : cross-entropy of the answer read after the CoT and "</think>\\n\\n"
CoT    : mean next-token cross-entropy of the CoT text plus its closing "</think>"
A truncated-CoT example (the CoT cut after `cut` tokens, closed the way the thinking budget closes it) teaches the model to answer
from an unfinished CoT. It never teaches the model to stop mid-thought, so it has no CoT term. Losses are weighted so that every question kind
(choice, yes / no, rating scale) has the same total weight, times an optional per-question "weight" (stage 1 weights harder questions up).

Usage: python -m train.sft --config train/configs/stage1_sft.yaml
"""
import argparse
import time

import torch
from torch.nn import functional as F

from openverdict.decide import Encoder, append_tokens, to_device
from openverdict.format import loss_from_logits
from openverdict.model import hidden_states, option_logits
from openverdict.prompts import check_trace_text
from train.common import (Questions, Sampler, check_finite, ensure_dir, learning_rate, load_config, log, make_optimizer, save,
                          setup_model)


def question_loss(backbone, encoder, case, qid, device):
    """The loss of one question (a scalar tensor) and the parts for logging."""
    raw = next(q for q in case['questions'] if q['id'] == qid)
    direct_item = encoder.encode(case, qid, thinking=False, require_target=True)
    h = hidden_states(backbone, to_device(direct_item.inputs, device))
    direct = loss_from_logits(option_logits(backbone, h[0, -1], direct_item.code_ids), direct_item.question)
    cot = raw.get('cot')
    if not cot:
        return direct, {'direct': float(direct.detach())}
    ids = encoder.tokenizer.encode(check_trace_text(cot['text']), add_special_tokens=False)
    item = encoder.encode(case, qid, thinking=True, require_target=True)
    cut = cot.get('cut')
    if cut is not None:
        if not 0 < cut < len(ids):
            raise ValueError('cut must fall inside the CoT')
        ids = ids[:cut]
    inputs = append_tokens(to_device(item.inputs, device), [*ids, *encoder.suffix_ids])
    h = hidden_states(backbone, inputs)[0]
    answer = loss_from_logits(option_logits(backbone, h[-1], item.code_ids), item.question)
    if cut is not None:
        return (direct + answer) / 2, {'direct': float(direct.detach()), 'answer': float(answer.detach())}
    # Next-token loss on the CoT and its closing "</think>": position p predicts token p + 1.
    targets = [*ids, encoder.stop_id]
    start = item.inputs['input_ids'].shape[1] - 1
    weight = backbone.get_output_embeddings().weight
    logits = F.linear(h[start:start + len(targets)].float(), weight.float())
    cot_loss = F.cross_entropy(logits, torch.tensor(targets, device=logits.device))
    return (direct + answer + cot_loss) / 2, {'direct': float(direct.detach()), 'answer': float(answer.detach()), 'cot': float(cot_loss.detach())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg['output_dir'])
    torch.manual_seed(cfg.get('seed', 17))
    backbone, processor, params = setup_model(cfg, args.device)
    encoder = Encoder(processor, image_root=cfg.get('image_root', '.'), max_pixels=cfg.get('max_pixels', 262_144))
    data = Questions(cfg['train_file'])
    factors = data.type_factors()
    sampler = Sampler(data, batch_size=cfg['batch_size'], seed=cfg.get('seed', 17))
    total = cfg.get('updates') or round(cfg.get('epochs', 1) * len(data.units) / cfg['batch_size'])
    warmup = cfg.get('warmup_updates', max(1, round(0.05 * total)))
    optimizer = make_optimizer(params, cfg)
    backbone.train()
    for step in range(total):
        started = time.monotonic()
        lr = learning_rate(step, peak=cfg['learning_rate'], total=total, warmup=warmup, final_fraction=cfg.get('final_lr_fraction', 0.1))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        batch, parts, total_loss = sampler.next_batch(), [], 0.0
        for case_id, qid, kind in batch:
            case = data.cases[case_id]
            loss, info = question_loss(backbone, encoder, case, qid, args.device)
            row_weight = next(q for q in case['questions'] if q['id'] == qid).get('weight', 1.0)
            (loss * factors[kind] * row_weight / len(batch)).backward()
            total_loss += float(loss.detach()) / len(batch)
            parts.append(info)
        norm = torch.nn.utils.clip_grad_norm_(params, cfg.get('clip_norm', 1.0), error_if_nonfinite=True)
        optimizer.step()
        check_finite(params)
        log(out / 'train_log.jsonl', {'step': step + 1, 'loss': total_loss, 'lr': lr, 'grad_norm': float(norm),
                                      'cot_rows': sum('cot' in p for p in parts), 'seconds': time.monotonic() - started})
        if (step + 1) % cfg.get('save_every', 500) == 0:
            save(backbone, out, step + 1)
    print('saved', save(backbone, out, total))


if __name__ == '__main__':
    main()
