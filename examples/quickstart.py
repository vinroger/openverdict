"""Ask one decision in both modes. Usage: python examples/quickstart.py path/to/adapter-folder"""
import json
import sys

from openverdict import OpenVerdict

case = {
    "id": "refund-17",
    "state": "Policy: refunds are allowed within 30 days of delivery. Order delivered 2 May; refund requested 20 May.",
    "questions": [{"id": "allowed", "kind": "noul", "instructions": "Is the refund allowed under the policy?",
                   "options": [{"id": "true", "description": "yes"}, {"id": "false", "description": "no"}]}],
}

model = OpenVerdict.load(sys.argv[1])
print(json.dumps(model.decide(case, thinking=False), indent=1))   # one forward pass
print(json.dumps(model.decide(case, thinking=True), indent=1))    # a short CoT (at most 80 tokens), then the answer
