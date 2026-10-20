"""Stage 2, rejection-sampling fine-tuning data: sample CoTs from the current model, keep the shortest correct one per question.

Step 1, sample: for every question, write `samples` CoTs under each prompt style (own, symbolic, draft), score the answer after each
CoT, and record whether it is right and whether it finished inside the budget.
Step 2, select: per question keep the SHORTEST CoT that is right and finished. Write SFT rows for train/sft.py:
    - the question with {"cot": {"text": ...}}                      (teaches a CoT that finishes)
    - truncated-CoT examples {"cot": {"text": ..., "cut": 16}}    (teach the model to answer from an unfinished CoT)
    - questions where no sample was right keep their direct answer only (no CoT is invented for them)

Example (one question, budget 64): own style 47 tokens right, symbolic 64 tokens truncated, draft 26 tokens right -> keep the
26-token CoT. The selected CoTs are trained under the normal thinking prompt, so the final model needs no style instruction.

The same sampling also builds the RL training set for stage 3 ("filter"): sample K CoTs per question in the model's own style at
temperature 1, then keep only the questions whose pass rate is above 0 and below 1 (right 1 to K-1 times out of K). A question the
model always or never solves gives every rollout the same reward, so RLOO learns nothing from it.

Usage:
  python -m train.rft sample --config train/configs/stage2_rft.yaml
  python -m train.rft select --config train/configs/stage2_rft.yaml
  python -m train.rft filter --config <config with sampling.styles [own], temperature 1.0, the RL budget>
"""
import argparse
import copy
import json

import torch

from openverdict.decide import Encoder, answer_logits, generate_cots
from openverdict.format import gold_indices, is_hard
from openverdict.prompts import STYLE_INSTRUCTIONS
from train.common import Questions, ensure_dir, load_config, read_jsonl


def sample(cfg, device):
    from openverdict.model import load_adapter, load_base
    dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[cfg.get('dtype', 'bfloat16')]
    backbone, processor = load_base(cfg.get('base_model', 'Qwen/Qwen3.5-4B'), device=device, dtype=dtype)
    load_adapter(backbone, cfg['init_adapter'])
    backbone.to(device).eval()
    encoder = Encoder(processor, image_root=cfg.get('image_root', '.'))
    data = Questions(cfg['questions_file'])
    s = cfg['sampling']
    torch.manual_seed(s.get('seed', 17))
    out = ensure_dir(cfg['output_dir']) / 'samples.jsonl'
    with open(out, 'w') as f:
        for case_id, qid, _ in data.units:
            case = data.cases[case_id]
            for style in s['styles']:
                item = encoder.encode(case, qid, thinking=True, require_target=True, instruction=STYLE_INSTRUCTIONS[style])
                if not is_hard(item.question):
                    continue
                gold = set(gold_indices(item.question))
                gens = generate_cots(backbone, encoder, item, budget=s['budget'], samples=s['samples'], temperature=s['temperature'],
                                     top_p=s.get('top_p', 1.0), top_k=s.get('top_k', 0))
                for g in gens:
                    with torch.no_grad():
                        logits = answer_logits(backbone, item, g['cot_ids'], encoder.suffix_ids)
                    right = int(logits.argmax()) in gold and g['valid']
                    f.write(json.dumps({'case_id': case_id, 'question_id': qid, 'style': style, 'right': right,
                                        'finished': not g['truncated'], 'tokens': len(g['cot_ids']),
                                        'text': encoder.tokenizer.decode(g['cot_ids'], skip_special_tokens=False)}) + '\n')
    print('wrote', out)


def select(cfg):
    data = Questions(cfg['questions_file'])
    samples = read_jsonl(ensure_dir(cfg['output_dir']) / 'samples.jsonl')
    best = {}
    for r in samples:
        if r['right'] and r['finished'] and r['text'].strip():
            key = (r['case_id'], r['question_id'])
            if key not in best or r['tokens'] < best[key]['tokens']:
                best[key] = r
    cuts = cfg['selection'].get('truncated_cot_lengths', [16, 32])
    rows, n_cot, n_cut = [], 0, 0
    for case_id, qid, _ in data.units:
        case = data.cases[case_id]
        chosen = best.get((case_id, qid))
        if chosen is None:
            rows.append(only_question(case, qid, None))
            continue
        rows.append(only_question(case, qid, {'text': chosen['text']}))
        n_cot += 1
        for cut in cuts:
            if cut < chosen['tokens']:
                c = only_question(case, qid, {'text': chosen['text'], 'cut': cut})
                c['id'] += f'~cut{cut}'
                rows.append(c)
                n_cut += 1
    path = ensure_dir(cfg['output_dir']) / 'rft_train.jsonl'
    with open(path, 'w') as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
    mean = sum(r['tokens'] for r in best.values()) / max(1, len(best))
    print(json.dumps({'questions': len(data.units), 'with_cot': n_cot, 'truncated_cot_examples': n_cut, 'mean_cot_tokens': round(mean, 1),
                      'file': str(path)}))


def filter_by_pass_rate(cfg):
    """Keep the questions with a pass rate above 0 and below 1 (right 1 to K-1 times out of K): the ones RLOO can learn from."""
    data = Questions(cfg['questions_file'])
    counts = {}
    for r in read_jsonl(ensure_dir(cfg['output_dir']) / 'samples.jsonl'):
        if r['style'] == 'own':
            right, total = counts.get((r['case_id'], r['question_id']), (0, 0))
            counts[(r['case_id'], r['question_id'])] = (right + r['right'], total + 1)
    keep = [only_question(data.cases[c], q, None) for (c, q), (right, total) in counts.items() if 0 < right < total]
    path = ensure_dir(cfg['output_dir']) / 'rl_train.jsonl'
    with open(path, 'w') as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in keep)
    print(json.dumps({'sampled_questions': len(counts), 'kept': len(keep), 'file': str(path)}))


def only_question(case, qid, cot):
    """A copy of the case that holds one question, with its CoT supervision (or none)."""
    c = copy.deepcopy(case)
    c['id'] = f"{case['id']}::{qid}"   # one row per question, so a case with several questions keeps them all
    c['questions'] = [q for q in c['questions'] if q['id'] == qid]
    if cot is not None:
        c['questions'][0]['cot'] = cot
    else:
        c['questions'][0].pop('cot', None)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['sample', 'select', 'filter'])
    ap.add_argument('--config', required=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.step == 'sample':
        sample(cfg, args.device)
    elif args.step == 'select':
        select(cfg)
    else:
        filter_by_pass_rate(cfg)


if __name__ == '__main__':
    main()
