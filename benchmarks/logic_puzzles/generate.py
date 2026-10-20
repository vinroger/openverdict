#!/usr/bin/env python3
"""Generate the 800 logic puzzles used in the OpenVerdict evaluation.

Three tasks in the sentence templates of BIG-Bench Hard (Suzgun et al. 2022), with longer reasoning chains than BIG-Bench Hard has,
to show at what chain length a single-pass model starts to fail:

* tracking             five people swap objects; depth = number of swaps                {3, 5, 7, 10, 13, 16, 20}
* web_of_lies          chain of truth-tellers and liars; depth = number of "says" links  {4, 6, 8, 10, 13, 16, 20}
* boolean_expressions  True / False with not, and, or and parentheses; depth = operators {4, 6, 8, 10, 13, 16}

40 puzzles per task and depth (800 in total). Multiple-choice answers are spread evenly over A-E (8 each); yes / no answers are
exactly 20 / 20. Every answer is computed twice: by a solver that works on the generator's own structure, and by a second solver
that re-parses the rendered text. Boolean puzzles get a third check with Python's own evaluator. Generation stops if any two disagree.

Standard library only, deterministic for a given --seed. Every puzzle is newly generated; none is copied from BIG-Bench Hard
(MIT licence, https://github.com/suzgunmirac/BIG-Bench-Hard).

Usage: python benchmarks/logic_puzzles/generate.py [--out benchmarks/logic_puzzles/puzzles.jsonl]
Output: one case per line in the format of data/FORMAT.md, plus "task", "depth" and "metadata" fields.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

SEED = 20261003
QUESTION_ID = 'decision'
ITEMS_PER_LEVEL = 40

TRACKING_DEPTHS = (3, 5, 7, 10, 13, 16, 20)
WOL_DEPTHS = (4, 6, 8, 10, 13, 16, 20)
BOOL_DEPTHS = (4, 6, 8, 10, 13, 16)

NOUL_OPTIONS = [{'id': 'false', 'description': 'No'}, {'id': 'true', 'description': 'Yes'}]
TRACKING_INSTRUCTIONS = 'Which option correctly completes the sentence "{stem} ..."?'
BOOLEAN_INSTRUCTIONS = 'Does this Boolean expression evaluate to True?'
BOOLEAN_PREFIX = 'Boolean expression: '


def subrng(seed: int, *parts) -> random.Random:
    """Independent stream per (task, depth) so changing one level never shifts another."""
    blob = json.dumps([seed, *parts], separators=(',', ':')).encode()
    return random.Random(int.from_bytes(hashlib.sha256(blob).digest()[:8], 'big'))


def oxford(items):
    return ', '.join(items[:-1]) + ', and ' + items[-1]


def sequenced(sentences):
    """BBH order words: First, Then, ..., Finally."""
    n = len(sentences)
    words = ['First' if i == 0 else 'Finally' if i == n - 1 else 'Then' for i in range(n)]
    return ' '.join(f'{w}, {s}' for w, s in zip(words, sentences))


def choice_question(instructions, descriptions, gold_index):
    letters = [chr(65 + i) for i in range(len(descriptions))]
    return {'id': QUESTION_ID, 'kind': 'choice', 'instructions': instructions,
            'options': [{'id': l, 'description': d} for l, d in zip(letters, descriptions)],
            'target': {'format': 'distribution', 'values': {l: float(i == gold_index) for i, l in enumerate(letters)}}}


def noul_question(instructions, answer: bool):
    return {'id': QUESTION_ID, 'kind': 'noul', 'instructions': instructions,
            'options': [dict(o) for o in NOUL_OPTIONS], 'positive_id': 'true',
            'target': {'format': 'distribution', 'values': {'false': float(not answer), 'true': float(answer)}}}


def make_case(group, task, depth, index, state, question, depth_definition, expected, expected_option, seed, extra):
    case_id = f'reasondepth:{group}:d{depth}:{index:03d}'
    return {'id': case_id, 'task': group, 'depth': depth, 'state': state, 'questions': [question],
            'metadata': {'template_task': task, 'depth_definition': depth_definition, 'expected': expected,
                         'expected_option': expected_option, 'seed': seed, **extra}}


# =============================================================== tracking (shuffled objects)

PEOPLE = ('Alice', 'Bob', 'Claire', 'Dave', 'Eve')  # BBH five_objects always uses these five
TRACKING_TASK = 'tracking_shuffled_objects_five_objects'

# Templates and object lists exactly as in BBH tracking_shuffled_objects_*.
SCENARIOS = {
    'books': dict(
        intro='{names} are friends and avid readers who occasionally trade books. At the start of the '
              'semester, they each buy one new book: {assign}.',
        assign='{p} gets {o}', middle='As the semester proceeds, they start trading around the new books.',
        swap='{a} and {b} swap books.', stem='At the end of the semester, {p} has',
        objects=('Catch-22', 'Frankenstein', 'Hound of the Baskervilles', 'Lolita', 'Moby Dick',
                 'The Fellowship of the Ring', 'The Great Gatsby', 'The Odyssey', 'The Pearl', 'Ulysses')),
    'balls': dict(
        intro='{names} are playing a game. At the start of the game, they are each holding a ball: {assign}.',
        assign='{p} has a {o}', middle='As the game progresses, pairs of players trade balls.',
        swap='{a} and {b} swap balls.', stem='At the end of the game, {p} has the',
        objects=('black ball', 'blue ball', 'brown ball', 'green ball', 'orange ball', 'pink ball',
                 'purple ball', 'red ball', 'white ball', 'yellow ball')),
    'gifts': dict(
        intro='{names} are holding a white elephant gift exchange. At the start of the event, they are each '
              'holding a present of a different color: {assign}.',
        assign='{p} has a {o}', middle='As the event progresses, pairs of people swap gifts.',
        swap='{a} and {b} swap their gifts.', stem='At the end of the event, {p} has the',
        objects=('black ball', 'blue present', 'brown present', 'green present', 'orange ball', 'pink ball',
                 'purple present', 'red present', 'white present', 'yellow present')),
    'dance': dict(
        intro='{names} are dancers at a square dance. At the start of a song, they each have a partner: {assign}.',
        assign='{p} is dancing with {o}', middle='Throughout the song, the dancers often trade partners.',
        swap='{a} and {b} switch partners.', stem='At the end of the dance, {p} is dancing with',
        objects=('Helga', 'Izzi', 'Jamie', 'Karl', 'Lola', 'Melissa', 'Ophelia', 'Patrick', 'Rodrigo', 'Sam')),
    'soccer': dict(
        intro='{names} are on the same team in a soccer match. At the start of the match, they are each '
              'assigned to a position: {assign}.',
        assign='{p} is playing {o}',
        middle='As the game progresses, pairs of players occasionally swap positions.',
        swap='{a} and {b} trade positions.', stem='At the end of the match, {p} is playing',
        objects=('benchwarmer', 'center midfielder', 'cheerleader', 'fullback', 'goalkeeper', 'left midfielder',
                 'left winger', 'right midfielder', 'right winger', 'striker')),
}


def render_tracking(scenario, objects, swaps, queried):
    sc = SCENARIOS[scenario]
    assign = oxford([sc['assign'].format(p=p, o=o) for p, o in zip(PEOPLE, objects)])
    state = (sc['intro'].format(names=oxford(list(PEOPLE)), assign=assign) + '\n' + sc['middle'] + ' '
             + sequenced([sc['swap'].format(a=a, b=b) for a, b in swaps]))
    stem = sc['stem'].format(p=queried)
    return state, stem


def solve_tracking_forward(objects, swaps):
    """Solver 1: simulate who holds what, swap by swap. Returns {person: object}."""
    holding = dict(zip(PEOPLE, objects))
    for a, b in swaps:
        holding[a], holding[b] = holding[b], holding[a]
    return holding


_ASSIGN_RE = re.compile(r'^(\w+) (?:gets|has a|is dancing with|is playing) (.+)$')
_SWAP_RE = re.compile(r'(?:First|Then|Finally), (\w+) and (\w+) '
                      r'(?:swap books|swap balls|swap their gifts|switch partners|trade positions)\.')
_STEM_RE = re.compile(r'At the end of the \w+, (\w+) (?:has the|has|is dancing with|is playing)$')


def solve_tracking_from_text(state, stem, options):
    """Solver 2 (independent): parse the rendered text, then trace the queried person BACKWARD
    through the swap list to find whose starting object they end with. Returns the option letter."""
    first_line, second_line = state.split('\n')
    listing = first_line.split(': ', 1)[1].rstrip('.')
    parts = re.split(r', and |, ', listing)
    initial = {}
    for part in parts:
        m = _ASSIGN_RE.match(part)
        if not m:
            raise ValueError(f'unparsable assignment {part!r}')
        initial[m.group(1)] = m.group(2)
    swaps = _SWAP_RE.findall(second_line)
    if len(re.findall(r'(?:First|Then|Finally), ', second_line)) != len(swaps):
        raise ValueError('swap parse mismatch')
    m = _STEM_RE.match(stem)
    if not m:
        raise ValueError(f'unparsable stem {stem!r}')
    holder = m.group(1)
    for a, b in reversed(swaps):
        if holder == a:
            holder = b
        elif holder == b:
            holder = a
    obj = initial[holder]
    letters = [o['id'] for o in options if o['description'] == obj]
    if len(letters) != 1:
        raise ValueError('answer object not uniquely among options')
    return letters[0], len(swaps)


def random_swaps(rng, n):
    swaps, prev = [], None
    while len(swaps) < n:
        a, b = rng.sample(PEOPLE, 2)
        if prev is not None and {a, b} == prev:
            continue  # an immediate repeat cancels itself; BBH-style chains do not do that on purpose
        swaps.append((a, b))
        prev = {a, b}
    return swaps


def answer_path_swaps(swaps, queried):
    """How many swaps actually move the answer object on its way to the queried person."""
    holder, n = queried, 0
    for a, b in reversed(swaps):
        if holder in (a, b):
            holder = b if holder == a else a
            n += 1
    return n


def gen_tracking(seed, depth, seen):
    rng = subrng(seed, 'tracking', depth)
    letters = [i for i in range(5) for _ in range(ITEMS_PER_LEVEL // 5)]
    scenarios = [s for s in SCENARIOS for _ in range(ITEMS_PER_LEVEL // len(SCENARIOS))]
    rng.shuffle(letters)
    rng.shuffle(scenarios)
    cases = []
    for index, (gold_index, scenario) in enumerate(zip(letters, scenarios)):
        while True:
            objects = rng.sample(SCENARIOS[scenario]['objects'], 5)
            swaps = random_swaps(rng, depth)
            final = solve_tracking_forward(objects, swaps)
            queried = next(p for p, o in final.items() if o == objects[gold_index])
            if not any(queried in pair for pair in swaps):
                continue  # queried person never swapped: answer would be readable without tracking
            state, stem = render_tracking(scenario, objects, swaps, queried)
            if state in seen:
                continue
            break
        seen.add(state)
        question = choice_question(TRACKING_INSTRUCTIONS.format(stem=stem), list(objects), gold_index)
        letter = chr(65 + gold_index)
        letter2, n_swaps_text = solve_tracking_from_text(state, stem, question['options'])
        assert letter2 == letter and n_swaps_text == depth, (letter, letter2, n_swaps_text, depth)
        cases.append(make_case(
            'tracking', TRACKING_TASK, depth, index, state, question, 'number of swaps', f'({letter})', letter, seed,
            {'n_objects': 5, 'n_swaps': depth, 'cloze_stem': stem, 'scenario': scenario,
             'answer_path_swaps': answer_path_swaps(swaps, queried),
             'answer_is_own_start_object': gold_index == PEOPLE.index(queried)}))
    return cases


# =============================================================== web of lies

WOL_NAMES = ('Alejandro', 'Alexis', 'Amberly', 'Andree', 'Antwan', 'Audrie', 'Bernita', 'Christie', 'Conception',
             'Crista', 'Dallas', 'Delbert', 'Delfina', 'Elanor', 'Fidel', 'Fletcher', 'Gwenn', 'Helene', 'Inga',
             'Jamey', 'Jaymie', 'Jerry', 'Jim', 'Ka', 'Kandi', 'Kristian', 'Leda', 'Lorine', 'Maybelle', 'Michael',
             'Michaela', 'Millicent', 'Millie', 'Osvaldo', 'Phoebe', 'Rashida', 'Raymond', 'Ryan', 'Sal',
             'Shalonda', 'Shaunda', 'Shenna', 'Sherrie', 'Sima', 'Tamika', 'Teressa', 'Vernell', 'Vina', 'Willian',
             'Yoland')  # the 50 names BBH web_of_lies uses


def render_wol(names, base_truth, claims):
    """claims[i] is True when names[i+1] says names[i] tells the truth, False when 'lies'."""
    parts = [f'{names[0]} tells the truth.' if base_truth else f'{names[0]} lies.']
    for i, claim in enumerate(claims):
        parts.append(f'{names[i + 1]} says {names[i]} {"tells the truth" if claim else "lies"}.')
    return ' '.join(parts), f'Does {names[-1]} tell the truth?'


def solve_wol_forward(base_truth, claims):
    """Solver 1: walk the chain. A speaker is truthful exactly when their claim matches reality."""
    truth = base_truth
    for claim in claims:
        truth = (claim == truth)
    return truth


_WOL_BASE = re.compile(r'^([A-Z][a-z]+) (tells the truth|lies)\.$')
_WOL_LINK = re.compile(r'^([A-Z][a-z]+) says ([A-Z][a-z]+) (tells the truth|lies)\.$')


def solve_wol_from_text(state, ask):
    """Solver 2 (independent): parse the text, check the chain is connected, then use parity:
    the last person tells the truth iff (first person truthful) XOR (odd number of 'lies' claims)."""
    sentences = re.findall(r'[^.]+\.', state)
    sentences = [s.strip() for s in sentences]
    m = _WOL_BASE.match(sentences[0])
    if not m:
        raise ValueError('bad first sentence')
    prev, base = m.group(1), m.group(2) == 'tells the truth'
    n_lies = 0
    for s in sentences[1:]:
        m = _WOL_LINK.match(s)
        if not m or m.group(2) != prev:
            raise ValueError(f'broken chain at {s!r}')
        n_lies += m.group(3) == 'lies'
        prev = m.group(1)
    if ask != f'Does {prev} tell the truth?':
        raise ValueError('question is not about the last speaker')
    return base ^ (n_lies % 2 == 1), len(sentences) - 1


def gen_wol(seed, depth, seen):
    rng = subrng(seed, 'web_of_lies', depth)
    targets = [True] * (ITEMS_PER_LEVEL // 2) + [False] * (ITEMS_PER_LEVEL // 2)
    rng.shuffle(targets)
    cases = []
    for index, target in enumerate(targets):
        while True:  # rejection sampling keeps every claim independent of the answer
            names = rng.sample(WOL_NAMES, depth + 1)
            base = rng.random() < 0.5
            claims = [rng.random() < 0.5 for _ in range(depth)]
            if solve_wol_forward(base, claims) != target:
                continue
            state, ask = render_wol(names, base, claims)
            if state in seen:
                continue
            break
        seen.add(state)
        answer2, links = solve_wol_from_text(state, ask)
        assert answer2 == target and links == depth
        cases.append(make_case(
            'web_of_lies', 'web_of_lies', depth, index, state, noul_question(ask, target),
            'number of "says" links in the truth chain', 'Yes' if target else 'No', 'true' if target else 'false',
            seed, {'n_people': depth + 1, 'n_lies_claims': sum(not c for c in claims),
                   'first_person_truthful': base}))
    return cases


# =============================================================== boolean expressions
# Tree nodes: ('lit', bool) | ('not', child) | ('and'|'or', left, right). Python precedence: not > and > or.

PREC = {'or': 1, 'and': 2, 'not': 3, 'lit': 4}


def random_tree(rng, n_ops, p_not=0.3):
    if n_ops == 0:
        return ('lit', rng.random() < 0.5)
    if rng.random() < p_not:
        return ('not', random_tree(rng, n_ops - 1, p_not))
    left = rng.randint(0, n_ops - 1)
    return (rng.choice(('and', 'or')), random_tree(rng, left, p_not), random_tree(rng, n_ops - 1 - left, p_not))


def eval_tree(node):
    """Solver 1: evaluate the generator's tree directly."""
    kind = node[0]
    if kind == 'lit':
        return node[1]
    if kind == 'not':
        return not eval_tree(node[1])
    left, right = eval_tree(node[1]), eval_tree(node[2])
    return (left and right) if kind == 'and' else (left or right)


def short_circuit_ops(node):
    """Operators a lazy left-to-right evaluator must actually process (Python short-circuit order).
    Guards against deep items that collapse early, e.g. 'True or <15 operators>'."""
    kind = node[0]
    if kind == 'lit':
        return node[1], 0
    if kind == 'not':
        value, work = short_circuit_ops(node[1])
        return (not value), work + 1
    value, work = short_circuit_ops(node[1])
    if (kind == 'and' and not value) or (kind == 'or' and value):
        return value, work + 1
    value2, work2 = short_circuit_ops(node[2])
    return value2, work + work2 + 1


def render_tree(node, rng, p_extra):
    """Tokens with the parentheses Python precedence needs, plus BBH-style redundant ones
    such as '( False )' or '( not True )'."""
    kind = node[0]
    if kind == 'lit':
        toks = ['True' if node[1] else 'False']
    elif kind == 'not':
        child = render_tree(node[1], rng, p_extra)
        if PREC[node[1][0]] < PREC['not'] and not _wrapped(child):
            child = ['('] + child + [')']
        toks = ['not'] + child
    else:
        toks = []
        for side, child_node in (('l', node[1]), ('r', node[2])):
            child = render_tree(child_node, rng, p_extra)
            if PREC[child_node[0]] < PREC[kind] and not _wrapped(child):
                child = ['('] + child + [')']
            toks += child
            if side == 'l':
                toks.append(kind)
    if rng.random() < p_extra:
        toks = ['('] + toks + [')']
    return toks


def _wrapped(toks):
    """True when the whole token list is one parenthesised group."""
    if not toks or toks[0] != '(' or toks[-1] != ')':
        return False
    level = 0
    for i, t in enumerate(toks):
        level += (t == '(') - (t == ')')
        if level == 0 and i < len(toks) - 1:
            return False
    return True


def parse_and_eval(text):
    """Solver 2 (independent): tokenise the rendered string and evaluate it with a recursive-descent
    parser that implements Python precedence (or < and < not < atom)."""
    toks = text.split()
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else None

    def take(expected=None):
        nonlocal pos
        t = toks[pos]
        if expected is not None and t != expected:
            raise ValueError(f'expected {expected!r}, got {t!r}')
        pos += 1
        return t

    def p_or():
        value = p_and()
        while peek() == 'or':
            take('or')
            rhs = p_and()
            value = value or rhs
        return value

    def p_and():
        value = p_not()
        while peek() == 'and':
            take('and')
            rhs = p_not()
            value = value and rhs
        return value

    def p_not():
        if peek() == 'not':
            take('not')
            return not p_not()
        return p_atom()

    def p_atom():
        t = take()
        if t == '(':
            value = p_or()
            take(')')
            return value
        if t in ('True', 'False'):
            return t == 'True'
        raise ValueError(f'unexpected token {t!r}')

    value = p_or()
    if pos != len(toks):
        raise ValueError('trailing tokens')
    return value


def python_eval(text):
    """Check 3: Python's own evaluator, after a strict token whitelist."""
    if not set(text.split()) <= {'True', 'False', 'and', 'or', 'not', '(', ')'}:
        raise ValueError('unexpected token')
    return bool(eval(text, {'__builtins__': {}}, {}))  # noqa: S307 - whitelisted tokens only


def paren_nesting(toks):
    level = nest = 0
    for t in toks:
        level += (t == '(') - (t == ')')
        nest = max(nest, level)
    return nest


def sensitive_literals(node):
    """How many single True/False flips would change the answer (0 = answer survives any one flip)."""
    leaves = []

    def collect(n, path):
        if n[0] == 'lit':
            leaves.append(path)
        else:
            for i, c in enumerate(n[1:], start=1):
                collect(c, path + (i,))

    def flip(n, path):
        if not path:
            return ('lit', not n[1])
        i = path[0]
        items = list(n)
        items[i] = flip(n[i], path[1:])
        return tuple(items)

    collect(node, ())
    base = eval_tree(node)
    return sum(eval_tree(flip(node, p)) != base for p in leaves), len(leaves)


def gen_boolean(seed, depth, seen):
    rng = subrng(seed, 'boolean_expressions', depth)
    targets = [True] * (ITEMS_PER_LEVEL // 2) + [False] * (ITEMS_PER_LEVEL // 2)
    rng.shuffle(targets)
    cases = []
    for index, target in enumerate(targets):
        while True:
            tree = random_tree(rng, depth)
            if eval_tree(tree) != target:
                continue
            n_sensitive, n_literals = sensitive_literals(tree)
            if n_sensitive == 0:
                continue  # answer would survive flipping any single literal: too insensitive to test depth
            sc_value, sc_work = short_circuit_ops(tree)
            assert sc_value == target
            if 2 * sc_work < depth:
                continue  # a lazy evaluator could skip more than half the operators: not really this deep
            toks = render_tree(tree, rng, p_extra=0.12)
            expr = ' '.join(toks)
            state = BOOLEAN_PREFIX + expr
            if state in seen:
                continue
            break
        seen.add(state)
        assert sum(t in ('not', 'and', 'or') for t in toks) == depth
        v2, v3 = parse_and_eval(expr), python_eval(expr)
        assert v2 == target and v3 == target, (expr, target, v2, v3)
        cases.append(make_case(
            'boolean_expressions', 'boolean_expressions', depth, index, state,
            noul_question(BOOLEAN_INSTRUCTIONS, target), 'number of Boolean operators (not/and/or)',
            'True' if target else 'False', 'true' if target else 'false', seed,
            {'n_operators': depth, 'paren_nesting': paren_nesting(toks), 'n_literals': n_literals,
             'n_sensitive_literals': n_sensitive, 'ops_evaluated_left_to_right': sc_work}))
    return cases


# =============================================================== checks

def balance_table(cases):
    rows = []
    by = collections.defaultdict(list)
    for c in cases:
        by[(c['task'], c['depth'])].append(c)
    for (group, depth), cs in sorted(by.items()):
        answers = collections.Counter(c['metadata']['expected_option'] for c in cs)
        row = {'task': group, 'depth': depth, 'n': len(cs), 'answers': dict(sorted(answers.items()))}
        ev = [c['metadata'] for c in cs]
        if group == 'tracking':
            row['scenarios'] = dict(sorted(collections.Counter(e['scenario'] for e in ev).items()))
            row['mean_answer_path_swaps'] = round(sum(e['answer_path_swaps'] for e in ev) / len(ev), 2)
            row['answer_is_own_start_object'] = sum(e['answer_is_own_start_object'] for e in ev)
        elif group == 'web_of_lies':
            row['mean_lies_claims'] = round(sum(e['n_lies_claims'] for e in ev) / len(ev), 2)
        else:
            row['paren_nesting'] = dict(sorted(collections.Counter(e['paren_nesting'] for e in ev).items()))
            row['mean_sensitive_literals'] = round(sum(e['n_sensitive_literals'] for e in ev) / len(ev), 2)
            row['mean_literals'] = round(sum(e['n_literals'] for e in ev) / len(ev), 2)
            row['mean_ops_evaluated_left_to_right'] = round(
                sum(e['ops_evaluated_left_to_right'] for e in ev) / len(ev), 2)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--out', default=str(Path(__file__).resolve().parent / 'puzzles.jsonl'))
    args = ap.parse_args()

    seen: set[str] = set()
    cases = []
    for depth in TRACKING_DEPTHS:
        cases += gen_tracking(args.seed, depth, seen)
    for depth in WOL_DEPTHS:
        cases += gen_wol(args.seed, depth, seen)
    for depth in BOOL_DEPTHS:
        cases += gen_boolean(args.seed, depth, seen)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')

    # every line parses; ids and states are unique; each target is one-hot on the expected option; answers are balanced
    reread = [json.loads(line) for line in out.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert len(reread) == len(cases)
    assert len({c['id'] for c in reread}) == len(reread) and len({c['state'] for c in reread}) == len(reread)
    for c in reread:
        q = c['questions'][0]
        assert sorted(q['target']['values']) == sorted(o['id'] for o in q['options'])
        assert sum(q['target']['values'].values()) == 1.0
        assert q['target']['values'][c['metadata']['expected_option']] == 1.0
    for row in balance_table(reread):
        assert len(set(row['answers'].values())) == 1, row
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(json.dumps({'puzzles': len(reread), 'by_task': dict(collections.Counter(c['task'] for c in reread)),
                      'file': str(out), 'sha256': digest}))


if __name__ == '__main__':
    sys.exit(main())
