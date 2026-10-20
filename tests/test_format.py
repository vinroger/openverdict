import math

import pytest
import torch

from openverdict.format import gold_indices, is_hard, loss_from_logits, parse_case, parse_question

YES_NO = {'id': 'late', 'kind': 'noul', 'instructions': 'Is the order late?',
          'options': [{'id': 'true', 'description': 'yes'}, {'id': 'false', 'description': 'no'}],
          'target': {'format': 'distribution', 'values': {'true': 1.0, 'false': 0.0}}}


def test_yes_no_question_parses_with_true_false_ids():
    q = parse_question(YES_NO)
    assert q.kind == 'noul' and q.positive_index == 0 and is_hard(q) and gold_indices(q) == (0,)


def test_one_hot_loss_is_cross_entropy():
    q = parse_question(YES_NO)
    logits = torch.tensor([2.0, 0.0])
    assert math.isclose(float(loss_from_logits(logits, q)), -math.log(math.exp(2) / (math.exp(2) + 1)), rel_tol=1e-6)


def test_soft_target_is_not_hard_and_gold_is_its_top():
    raw = {**YES_NO, 'target': {'format': 'distribution', 'values': {'true': 0.3, 'false': 0.7}}}
    q = parse_question(raw)
    assert not is_hard(q) and gold_indices(q) == (1,)


def test_score_question_needs_levels_and_mean_target_uses_the_scale():
    raw = {'id': 's', 'kind': 'score', 'options': [{'id': 'low'}, {'id': 'mid'}, {'id': 'high'}], 'ordinal_values': [1, 2, 3],
           'target': {'format': 'mean', 'mean': 3}}
    q = parse_question(raw)
    perfect = torch.tensor([-50.0, -50.0, 50.0])
    assert float(loss_from_logits(perfect, q)) < 1e-6 and gold_indices(q) == (2,)
    with pytest.raises(ValueError):
        parse_question({**raw, 'ordinal_values': None})


def test_acceptable_set_counts_every_listed_option():
    raw = {'id': 'c', 'kind': 'choice', 'options': [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}],
           'target': {'format': 'acceptable_set', 'valid_ids': ['a', 'c']}}
    q = parse_question(raw)
    assert gold_indices(q) == (0, 2) and is_hard(q)


def test_case_rejects_duplicate_question_ids():
    with pytest.raises(ValueError):
        parse_case({'id': 'x', 'state': 's', 'questions': [YES_NO, YES_NO]})
