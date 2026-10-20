"""Turn a public natural-language-inference dataset (SNLI) into OpenVerdict training cases.

It shows the conversion pattern for any labelled dataset: the premise becomes the state, the hypothesis becomes a three-option choice
question, the label becomes a one-hot target. Rows without a gold label (label -1) are skipped.
Usage: pip install datasets; python examples/convert_snli.py data/snli_train.jsonl
"""
import json
import sys

OPTIONS = [{'id': 'entailment', 'description': 'the premise implies the statement'},
           {'id': 'neutral', 'description': 'the premise neither implies nor contradicts the statement'},
           {'id': 'contradiction', 'description': 'the premise contradicts the statement'}]


def convert(row, i):
    """One SNLI row -> one case, or None when the row has no gold label."""
    if row['label'] not in (0, 1, 2):
        return None
    gold = OPTIONS[row['label']]['id']
    return {'id': f'snli-{i}', 'state': row['premise'],
            'questions': [{'id': 'relation', 'kind': 'choice',
                           'instructions': f'How does the premise relate to the statement "{row["hypothesis"]}"?',
                           'options': OPTIONS,
                           'target': {'format': 'distribution', 'values': {o['id']: float(o['id'] == gold) for o in OPTIONS}}}]}


if __name__ == '__main__':
    from datasets import load_dataset
    rows = load_dataset('stanfordnlp/snli', split='train')
    n = 0
    with open(sys.argv[1], 'w') as f:
        for i, r in enumerate(rows):
            case = convert(r, i)
            if case:
                f.write(json.dumps(case, ensure_ascii=False) + '\n')
                n += 1
    print('wrote', n, 'cases to', sys.argv[1])
