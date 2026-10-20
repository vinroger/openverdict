"""Fit the calibration of the thinking mode on a validation set (questions never used for training).

Input: the predictions.jsonl that eval/evaluate.py writes when run WITHOUT --calibration on the validation set.
Output: calibration.json, for example {"choice": 4.55, "score": 4.35, "noul_cap": 0.86}.
No answer changes: the top option is the same before and after.

Usage: python -m train.calibrate --predictions runs/validation/predictions.jsonl --out checkpoints/openverdict/calibration.json
"""
import argparse
import json

from openverdict import calibration as cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    rows = []
    with open(args.predictions) as f:
        for line in f:
            r = json.loads(line)
            if 'thinking_raw' in r and len(r['gold']) == 1:
                rows.append({'kind': r['kind'], 'probabilities': r['thinking_raw'], 'gold': r['gold'][0]})
    fitted = cal.fit(rows)
    cal.save(fitted, args.out)
    for kind in ('choice', 'noul', 'score'):
        part = [r for r in rows if r['kind'] == kind]
        if part:
            before = cal.log_loss(part, lambda p: p)
            after = cal.log_loss(part, lambda p, k=kind: cal.apply(p, k, fitted))
            print(f'{kind}: {len(part)} questions, log loss {before:.3f} -> {after:.3f}')
    print(json.dumps(fitted))


if __name__ == '__main__':
    main()
