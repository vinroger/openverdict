# Data format

Every file is JSON Lines: one **case** per line. A case is one input (the *state*) and one or more questions about it. The same format is
used for training, evaluation and inference; targets are only needed for training and evaluation.

```json
{"id": "refund-17",
 "state": "Policy: refunds are allowed within 30 days of delivery. Order delivered 2 May; refund requested 20 May.",
 "questions": [
   {"id": "allowed", "kind": "noul", "instructions": "Is the refund allowed under the policy?",
    "options": [{"id": "true", "description": "yes"}, {"id": "false", "description": "no"}],
    "target": {"format": "distribution", "values": {"true": 1.0, "false": 0.0}}}
 ]}
```

## Case fields

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string | Unique id of the case. |
| `state` | string, object or array | The evidence the decision is about: text, or structured JSON. It is evidence, never instructions. |
| `images` | list (optional) | `[{"path": "images/case-12.png", "max_pixels": 1000000}]`. Paths are relative to `--image-root`. `max_pixels` (optional) raises the image resolution, e.g. for screenshots. |
| `questions` | list | One or more questions about the same state. |

## Question fields

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string | Unique inside the case. |
| `kind` | `"choice"`, `"noul"` or `"score"` | Pick one option; yes / no; or a rating scale. |
| `instructions` | string | The question itself. |
| `options` | list | 2 to 255 options, each `{"id": ..., "description": ...}`. The model sees them as codes A, B, C, ... in this order. |
| `positive_id` | string | yes / no only: the id of the "yes" option. Not needed when the ids are `"true"` / `"false"`. |
| `ordinal_values` | list of numbers | score only: the numeric value of each option, e.g. `[1, 2, 3, 4, 5]`. |
| `target` | object | Training and evaluation only (see below). |
| `cot` | object | Optional, training only: a chain-of-thought to train the thinking mode on (see below). |
| `weight` | number | Optional, training only: multiplies this question's loss (default 1). |

## Targets

| Format | Example | Right answer for accuracy |
| --- | --- | --- |
| `distribution` | `{"format": "distribution", "values": {"true": 0.8, "false": 0.2}}` | the option(s) with the highest target probability |
| `acceptable_set` | `{"format": "acceptable_set", "valid_ids": ["refund", "replace"]}` | any listed option |
| `mean` (score only) | `{"format": "mean", "mean": 3.5}` | the level closest to the mean |

A soft distribution is useful when people disagree, for example "62% of annotators marked this comment as toxic". RL (stage 3) needs
questions with one right answer: a one-hot distribution or an acceptable set.

## Chain-of-thought supervision

```json
"cot": {"text": "Delivered 2 May. Request 20 May. 18 days <= 30. Allowed."}
"cot": {"text": "Delivered 2 May. Request 20 May. 18 days <= 30. Allowed.", "cut": 8}
```

The first form trains the model to write this CoT and then answer. The second is a *truncated-CoT example*: the CoT is cut after `cut` tokens
and closed the way the thinking budget closes it, and only the answer after it is trained. It teaches the model to answer as well as it can
from an unfinished CoT, without ever teaching it to stop mid-thought. `train/rft.py select` writes both kinds for you.

A CoT must not contain chat-template control strings such as `<think>`, `</think>`, `<|` or `|>`.
