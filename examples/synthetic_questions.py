"""A template for making your own synthetic training questions.

Many useful decisions have no public dataset. One way to get training questions is to write a generator: a program (or a larger
language model you have access to) proposes a state, a question and options, and a CHECK decides the right answer independently.
Only questions whose answer is verified should be used for training, and never items from a test set you report on.

This file gives the skeleton only. Connect your own generator in `propose()`.
"""
import json
import random


def propose(rng):
    """Return one candidate case without a target.

    Connect a generator here: a rule-based template (as below) or a larger language model of your choice.
    """
    a, b = rng.randint(1, 28), rng.randint(1, 28)
    return {'id': f'refund-{a}-{b}', 'state': f'Policy: refunds within 14 days of delivery. Delivered on day {a}; request on day {a + b}.',
            'questions': [{'id': 'allowed', 'kind': 'noul', 'instructions': 'Is the refund allowed under the policy?',
                           'options': [{'id': 'true', 'description': 'yes'}, {'id': 'false', 'description': 'no'}]}],
            '_facts': {'days': b, 'limit': 14}}


def check(case):
    """Decide the right answer from the facts, independently of whoever wrote the question. Return None to reject the case."""
    facts = case.pop('_facts')
    right = 'true' if facts['days'] <= facts['limit'] else 'false'
    case['questions'][0]['target'] = {'format': 'distribution', 'values': {'true': float(right == 'true'), 'false': float(right == 'false')}}
    return case


if __name__ == '__main__':
    rng = random.Random(0)
    for _ in range(3):
        print(json.dumps(check(propose(rng))))
