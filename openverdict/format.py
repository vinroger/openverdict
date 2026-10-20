"""The decision format: one case (a state and its questions) and the loss of one question.

A case is one JSON object. Example (a yes / no question):

    {"id": "demo-1",
     "state": "Order #812 shipped on 3 March. The customer reports it has not arrived on 20 March.",
     "questions": [{"id": "late", "kind": "noul", "instructions": "Is the order late (more than 14 days)?",
                    "options": [{"id": "true", "description": "yes"}, {"id": "false", "description": "no"}],
                    "target": {"format": "distribution", "values": {"true": 1.0, "false": 0.0}}}]}

Question kinds: "choice" (pick one option), "noul" (yes / no, exactly two options) and "score" (a rating scale with numeric levels).
Targets are optional at inference time and required for training. See data/FORMAT.md for every field.
"""
from dataclasses import dataclass
import math
from typing import Mapping

import torch
import torch.nn.functional as F

KINDS = ('choice', 'noul', 'score')


@dataclass(frozen=True)
class Question:
    id: str
    kind: str
    option_ids: tuple
    positive_index: int | None      # yes / no only: the index of the "yes" option
    levels: tuple                   # score only: the numeric value of each option
    target: dict | None             # normalised target, or None


def _ids(options):
    if not isinstance(options, (list, tuple)) or not all(isinstance(o, Mapping) for o in options):
        raise ValueError('options must be a list of objects')
    ids = tuple(o.get('id') for o in options)
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('every option needs a unique, non-empty string id')
    if not 2 <= len(ids) <= 255:
        raise ValueError('a question needs 2 to 255 options')
    return ids


def parse_question(raw, *, require_target=True):
    """Check one question and return it in normalised form. Raises ValueError with the reason when a field is wrong."""
    kind = str(raw.get('kind', '')).lower()
    if kind not in KINDS:
        raise ValueError(f'unsupported question kind: {kind!r}')
    qid = raw.get('id')
    if not isinstance(qid, str) or not qid:
        raise ValueError('a question needs a string id')
    ids = _ids(raw.get('options'))
    positive = None
    if kind == 'noul':
        if len(ids) != 2:
            raise ValueError('a yes / no question has exactly two options')
        pos_id = raw.get('positive_id', 'true' if set(ids) == {'true', 'false'} else None)
        if pos_id not in ids:
            raise ValueError('a yes / no question needs positive_id (or the option ids "true" / "false")')
        positive = ids.index(pos_id)
    levels = ()
    if kind == 'score':
        values = raw.get('ordinal_values')
        if isinstance(values, Mapping):
            values = [values.get(i) for i in ids]
        if (not isinstance(values, (list, tuple)) or len(values) != len(ids)
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values) or len(set(values)) != len(ids)):
            raise ValueError('a score question needs distinct numeric ordinal_values for every option')
        levels = tuple(float(v) for v in values)
    target = raw.get('target')
    if target is None:
        if require_target:
            raise ValueError('a training question needs a target')
        return Question(qid, kind, ids, positive, levels, None)
    fmt = target.get('format')
    if fmt == 'distribution':
        values = target.get('values')
        if not isinstance(values, Mapping) or set(values) != set(ids):
            raise ValueError('a distribution target gives a probability for every option id')
        probs = [float(values[i]) for i in ids]
        total = math.fsum(probs)
        if any(p < 0 for p in probs) or abs(total - 1) > 1e-6:
            raise ValueError('target probabilities must be non-negative and sum to 1')
        norm = {'format': fmt, 'probabilities': tuple(p / total for p in probs)}
    elif fmt == 'acceptable_set':
        valid = target.get('valid_ids')
        if not valid or not set(valid) <= set(ids):
            raise ValueError('valid_ids must be a non-empty subset of the option ids')
        norm = {'format': fmt, 'valid_indices': tuple(i for i, o in enumerate(ids) if o in valid)}
    elif fmt == 'mean':
        if kind != 'score':
            raise ValueError('a mean target needs a score question')
        mean = float(target.get('mean'))
        if not min(levels) <= mean <= max(levels):
            raise ValueError('the target mean lies outside the scale')
        norm = {'format': fmt, 'mean': mean}
    else:
        raise ValueError(f'unsupported target format: {fmt!r}')
    return Question(qid, kind, ids, positive, levels, norm)


def gold_indices(question):
    """The option indices that count as right: the top of a distribution target, or the acceptable set."""
    t = question.target
    if t is None:
        return ()
    if t['format'] == 'acceptable_set':
        return t['valid_indices']
    if t['format'] == 'distribution':
        top = max(t['probabilities'])
        return tuple(i for i, p in enumerate(t['probabilities']) if p == top)
    nearest = min(range(len(question.levels)), key=lambda i: abs(question.levels[i] - t['mean']))
    return (nearest,)


def is_hard(question):
    """True when the question has exactly one right option (RL needs a right / wrong answer)."""
    t = question.target
    if t is None:
        return False
    if t['format'] == 'acceptable_set':
        return True
    return t['format'] == 'distribution' and sorted(t['probabilities'])[-1] == 1.0


def loss_from_logits(logits, question):
    """Loss of one question from its option logits (shape [number of options]).

    distribution  : cross-entropy with the target distribution (a one-hot target gives the usual cross-entropy)
    acceptable_set: -log of the probability mass on the acceptable options
    mean          : squared error between the predicted and the target mean, on the scale mapped to [0, 1]
    """
    logp = F.log_softmax(logits.float(), dim=-1)
    t = question.target
    if t is None:
        raise ValueError('the loss needs a target')
    if t['format'] == 'distribution':
        return -(logp.new_tensor(t['probabilities']) * logp).sum()
    if t['format'] == 'acceptable_set':
        return -torch.logsumexp(logp[list(t['valid_indices'])], dim=-1)
    low, high = min(question.levels), max(question.levels)
    scale = logp.new_tensor([(v - low) / (high - low) for v in question.levels])
    return ((logp.exp() * scale).sum() - (t['mean'] - low) / (high - low)).square()


def parse_case(case, *, require_target=True):
    """Check a whole case and return its parsed questions in order."""
    if not isinstance(case.get('id'), str):
        raise ValueError('a case needs a string id')
    if not isinstance(case.get('state'), (str, dict, list)):
        raise ValueError('state must be text, a JSON object or a JSON array')
    questions = case.get('questions')
    if not isinstance(questions, list) or not questions:
        raise ValueError('a case needs a non-empty questions list')
    parsed = [parse_question(q, require_target=require_target) for q in questions]
    if len({q.id for q in parsed}) != len(parsed):
        raise ValueError('question ids must be unique inside a case')
    return parsed
