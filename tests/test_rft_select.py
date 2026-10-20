import json

from train.rft import filter_by_pass_rate, select

CASE = {'id': 'c', 'state': 's', 'questions': [{'id': 'q', 'kind': 'noul', 'instructions': 'ok?',
        'options': [{'id': 'true'}, {'id': 'false'}], 'target': {'format': 'distribution', 'values': {'true': 1.0, 'false': 0.0}}}]}


def write(tmp_path, samples):
    (tmp_path / 'q.jsonl').write_text(json.dumps(CASE) + '\n')
    (tmp_path / 'out').mkdir()
    (tmp_path / 'out' / 'samples.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in samples))
    return {'questions_file': str(tmp_path / 'q.jsonl'), 'output_dir': str(tmp_path / 'out'), 'selection': {'truncated_cot_lengths': [2, 50]}}


def sample(style, right, finished, tokens, text):
    return {'case_id': 'c', 'question_id': 'q', 'style': style, 'right': right, 'finished': finished, 'tokens': tokens, 'text': text}


def test_select_keeps_the_shortest_right_finished_cot_and_adds_truncated_cot_examples(tmp_path):
    cfg = write(tmp_path, [sample('own', True, True, 9, 'long right'), sample('draft', True, True, 4, 'short right'),
                           sample('symbolic', True, False, 3, 'cut'), sample('own', False, True, 2, 'wrong')])
    select(cfg)
    rows = [json.loads(l) for l in open(tmp_path / 'out' / 'rft_train.jsonl')]
    cots = [r['questions'][0]['cot'] for r in rows]
    assert cots == [{'text': 'short right'}, {'text': 'short right', 'cut': 2}]     # no copy at 50: longer than the CoT


def test_filter_keeps_pass_rates_strictly_between_0_and_1(tmp_path):
    cfg = write(tmp_path, [sample('own', True, True, 5, 'a'), sample('own', False, True, 5, 'b')])
    filter_by_pass_rate(cfg)
    assert len(open(tmp_path / 'out' / 'rl_train.jsonl').read().splitlines()) == 1
