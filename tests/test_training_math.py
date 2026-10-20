import math

import torch

from openverdict.calibration import apply, cap_top, fit, temperature_scale
from openverdict.metrics import accuracy, expected_calibration_error, paired_difference
from train.common import learning_rate
from train.rl import completion_reward, rloo_advantages


def test_completion_bonus_is_paid_only_to_right_finished_cots():
    assert completion_reward(True, True, 0.25) == 1.25
    assert completion_reward(True, False, 0.25) == 1.0
    assert completion_reward(False, True, 0.25) == 0.0


def test_rloo_advantages_match_the_readme_example():
    adv = rloo_advantages([1.25, 1.0, 0.0, 0.0]).tolist()
    assert [round(a, 2) for a in adv] == [0.92, 0.58, -0.75, -0.75]
    assert torch.all(rloo_advantages([1.0, 1.0, 1.0, 1.0]) == 0)     # same outcome everywhere: no learning signal


def test_learning_rate_warms_up_then_decays_to_ten_percent():
    lr = [learning_rate(s, peak=1.0, total=100, warmup=10, final_fraction=0.1) for s in range(100)]
    assert math.isclose(lr[9], 1.0) and math.isclose(lr[-1], 0.1) and lr[0] == 0.1


def test_temperature_scaling_keeps_the_top_answer():
    p = {'A': 0.995, 'B': 0.004, 'C': 0.001}
    q = temperature_scale(p, 4.55)
    assert max(q, key=q.get) == 'A' and abs(q['A'] - 0.659) < 0.001


def test_yes_no_cap():
    assert cap_top({'yes': 0.99, 'no': 0.01}, 0.86) == {'yes': 0.86, 'no': 0.14000000000000001}
    assert cap_top({'yes': 0.7, 'no': 0.3}, 0.86) == {'yes': 0.7, 'no': 0.3}


def test_fit_softens_an_overconfident_model():
    rows = [{'kind': 'choice', 'probabilities': {'A': 0.999, 'B': 0.001}, 'gold': 'A' if i % 4 else 'B'} for i in range(40)]
    fitted = fit(rows)
    assert fitted['choice'] > 1.0
    assert apply({'A': 0.999, 'B': 0.001}, 'choice', fitted)['A'] < 0.999


def test_metrics_and_paired_difference():
    rows = [{'probabilities': {'A': 0.9, 'B': 0.1}, 'gold': ['A']}, {'probabilities': {'A': 0.9, 'B': 0.1}, 'gold': ['B']}]
    assert accuracy(rows) == 0.5 and abs(expected_calibration_error(rows) - 0.4) < 1e-9
    d = paired_difference({'q1': True, 'q2': False}, {'q1': True, 'q2': True})
    assert d['difference'] == -50.0 and d['n'] == 2
