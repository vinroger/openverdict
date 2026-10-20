"""Post-hoc calibration: make the probabilities match how often the model is right, without changing any answer.

choice and score questions: temperature scaling, p_T(k) proportional to p(k) ** (1 / T), one T per question kind.
yes / no questions: the top probability is capped (for example at 0.86); a top probability below the cap is left as it is.

Example: a thinking-mode answer {"A": 0.995, "B": 0.004, "C": 0.001} with T = 4.55 becomes about {"A": 0.66, "B": 0.20, "C": 0.14}.
The top option stays A, so accuracy is identical.
"""
import json
import math

TEMPERATURE_GRID = [round(0.5 + 0.05 * i, 2) for i in range(151)]   # 0.5 ... 8.0
CAP_GRID = [round(0.5 + 0.01 * i, 2) for i in range(50)]             # 0.50 ... 0.99


def temperature_scale(probs, t):
    w = {k: math.exp(math.log(max(v, 1e-12)) / t) for k, v in probs.items()}
    z = sum(w.values())
    return {k: v / z for k, v in w.items()}


def cap_top(probs, cap):
    top = max(probs, key=probs.get)
    if probs[top] <= cap:
        return dict(probs)
    rest = sum(v for k, v in probs.items() if k != top)
    return {k: (cap if k == top else (1 - cap) * (v / rest if rest > 0 else 1 / (len(probs) - 1))) for k, v in probs.items()}


def apply(probs, kind, calibration):
    if kind == 'noul':
        return cap_top(probs, calibration['noul_cap'])
    return temperature_scale(probs, calibration[kind])


def log_loss(rows, fn):
    return sum(-math.log(max(fn(r['probabilities'])[r['gold']], 1e-12)) for r in rows) / len(rows)


def fit(rows):
    """Fit on predictions for a validation set (questions not used in training). rows: [{'kind': 'choice' | 'noul' | 'score', 'probabilities': {...}, 'gold': option id}].
    Each value is chosen on a grid by the lowest log loss on its own question kind."""
    out = {}
    for kind in ('choice', 'score'):
        part = [r for r in rows if r['kind'] == kind]
        out[kind] = min(TEMPERATURE_GRID, key=lambda t: log_loss(part, lambda p: temperature_scale(p, t))) if part else 1.0
    part = [r for r in rows if r['kind'] == 'noul']
    out['noul_cap'] = min(CAP_GRID, key=lambda c: log_loss(part, lambda p: cap_top(p, c))) if part else 0.99
    return out


def save(calibration, path):
    with open(path, 'w') as f:
        json.dump(calibration, f, indent=1)


def load(path):
    with open(path) as f:
        return json.load(f)
