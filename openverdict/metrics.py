"""Evaluation metrics: accuracy, expected calibration error, log loss, and paired differences with bootstrap intervals.

A prediction row is {'id': ..., 'probabilities': {option id: p}, 'gold': [option ids that count as right]}.
Accuracy counts a question as right when the top option is in 'gold'. A question with no prediction counts as wrong.
"""
import math
import random


def top(probs):
    return max(probs, key=probs.get)


def accuracy(rows):
    return sum(top(r['probabilities']) in r['gold'] for r in rows) / len(rows)


def expected_calibration_error(rows, bins=10):
    """Top-label ECE: bin questions by their top probability; average |confidence - accuracy| weighted by bin size."""
    buckets = [[] for _ in range(bins)]
    for r in rows:
        p = r['probabilities'][top(r['probabilities'])]
        buckets[min(bins - 1, int(p * bins))].append((p, top(r['probabilities']) in r['gold']))
    return sum(abs(sum(p for p, _ in b) / len(b) - sum(ok for _, ok in b) / len(b)) * len(b) for b in buckets if b) / len(rows)


def log_loss(rows):
    return sum(-math.log(max(sum(r['probabilities'].get(g, 0.0) for g in r['gold']), 1e-12)) for r in rows) / len(rows)


def paired_difference(right_a, right_b, *, draws=2000, seed=7):
    """Accuracy of A minus B in points on the questions both answered, with a 95% interval from resampling the questions.

    right_a, right_b: {question id: True / False}. Example: {'q1': True, 'q2': False} vs {'q1': True, 'q2': True} -> -50.0.
    """
    ids = [i for i in right_a if i in right_b]
    d = [int(right_a[i]) - int(right_b[i]) for i in ids]
    rng = random.Random(seed)
    n = len(d)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    return {'n': n, 'difference': 100 * sum(d) / n, 'low': 100 * boots[int(0.025 * draws)], 'high': 100 * boots[int(0.975 * draws) - 1]}
