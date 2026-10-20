"""Evaluate a model on a JSONL file of cases, in non-thinking mode, thinking mode, or both.

Writes one prediction row per question (both modes side by side) and prints a summary:
accuracy, expected calibration error, log loss, mean CoT length and the share of CoTs truncated at the budget.

Usage:
  python -m eval.evaluate --adapter checkpoints/openverdict --cases data/examples/toy_eval.jsonl --out runs/toy \\
      [--calibration checkpoints/openverdict/calibration.json] [--budget 80] [--modes thinking non_thinking]
"""
import argparse
import json
from pathlib import Path

import torch

from openverdict import calibration as cal
from openverdict.decide import Encoder, non_thinking, think
from openverdict.format import gold_indices, parse_case
from openverdict.metrics import accuracy, expected_calibration_error, log_loss
from openverdict.model import BASE_MODEL, load_adapter, load_base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adapter', required=True, help='folder with adapter.safetensors')
    ap.add_argument('--cases', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--calibration', help='calibration.json; applied to the thinking mode')
    ap.add_argument('--budget', type=int, default=80)
    ap.add_argument('--modes', nargs='+', default=['non_thinking', 'thinking'], choices=['non_thinking', 'thinking'])
    ap.add_argument('--base-model', default=BASE_MODEL)
    ap.add_argument('--image-root', default='.')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    backbone, processor = load_base(args.base_model, device=args.device)
    load_adapter(backbone, args.adapter)
    backbone.to(args.device).eval()
    encoder = Encoder(processor, image_root=args.image_root)
    calib = cal.load(args.calibration) if args.calibration else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = {m: [] for m in args.modes}
    with open(args.cases) as f, open(out / 'predictions.jsonl', 'w') as w:
        for line in f:
            case = json.loads(line)
            for q in parse_case(case, require_target=True):
                record = {'case_id': case['id'], 'question_id': q.id, 'kind': q.kind,
                          'gold': [q.option_ids[i] for i in gold_indices(q)]}
                if 'non_thinking' in args.modes:
                    r = non_thinking(backbone, encoder.encode(case, q.id, thinking=False))
                    record['non_thinking'] = r['probabilities']
                    rows['non_thinking'].append({'probabilities': r['probabilities'], 'gold': record['gold']})
                if 'thinking' in args.modes:
                    r = think(backbone, encoder, encoder.encode(case, q.id, thinking=True), budget=args.budget)
                    probs = cal.apply(r['probabilities'], q.kind, calib) if calib else r['probabilities']
                    record.update(thinking=probs, thinking_raw=r['probabilities'], cot=r['cot'], cot_tokens=r['cot_tokens'],
                                  truncated=r['truncated'])
                    rows['thinking'].append({'probabilities': probs, 'gold': record['gold'], 'tokens': r['cot_tokens'],
                                             'truncated': r['truncated']})
                w.write(json.dumps(record, ensure_ascii=False) + '\n')
    summary = {}
    for mode, rs in rows.items():
        summary[mode] = {'questions': len(rs), 'accuracy': accuracy(rs), 'ece': expected_calibration_error(rs), 'log_loss': log_loss(rs)}
        if mode == 'thinking':
            summary[mode].update(mean_cot_tokens=sum(r['tokens'] for r in rs) / len(rs),
                                 truncated_share=sum(r['truncated'] for r in rs) / len(rs))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
