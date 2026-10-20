# Logic puzzles (800)

Multi-step reasoning puzzles in the sentence templates of BIG-Bench Hard, with longer chains than BIG-Bench Hard has. We use them to
see at what chain length a model that answers in one forward pass starts to fail, and how much a short chain-of-thought helps.

| Task | What the model must do | Depths | Puzzles |
| --- | --- | --- | ---: |
| tracking | follow five people's objects through a series of swaps | 3, 5, 7, 10, 13, 16, 20 swaps | 280 |
| web_of_lies | follow a chain of truth-tellers and liars | 4, 6, 8, 10, 13, 16, 20 links | 280 |
| boolean_expressions | evaluate not / and / or with parentheses | 4, 6, 8, 10, 13, 16 operators | 240 |

Each depth has 40 puzzles, with the answers spread evenly over the options. Every answer is computed by two independent solvers (and,
for Boolean expressions, also by Python's own evaluator); generation stops if they disagree. Example (tracking, 3 swaps):

> Alice, Bob, Claire, Dave, and Eve are dancers at a square dance. At the start of a song, they each have a partner: Alice is dancing
> with Izzi, Bob is dancing with Lola, Claire is dancing with Helga, Dave is dancing with Sam, and Eve is dancing with Melissa.
> Throughout the song, the dancers often trade partners. First, Bob and Claire switch partners. Then, Alice and Dave switch partners.
> Finally, Bob and Dave switch partners. — *At the end of the dance, Claire is dancing with ...* (answer: Lola)

## Files

- `puzzles.jsonl`: the 800 puzzles, one case per line in the format of [data/FORMAT.md](../../data/FORMAT.md), with `task`, `depth`
  and per-puzzle `metadata` (for example the number of swaps on the path to the answer).
- `generate.py`: the generator (Python standard library only). `python benchmarks/logic_puzzles/generate.py` rebuilds exactly the same
  file (seed 20261003).

## Results (accuracy, %)

| System | Accuracy |
| --- | ---: |
| OpenVerdict, thinking mode | **63.5** |
| OpenVerdict, non-thinking mode | 48.4 |
| Clef-Flash 9B | 52.0 |
| Decision 4B v1.2 | 50.2 |
| Kev-4B | 50.1 |
| JevK5 | 49.8 |
| H2O-Lightning-4B v1.2.3 | 49.5 |
| Imajev-4B | 49.1 |
| Kev-27B (reference) | 55.9 |
| Jev 1.13.0, TypeSafe API (reference) | 65.6 |

Every model was run by us on the same puzzles with its own prompt format. The models that answer in one forward pass all land near
50%, whatever their training data; a short chain-of-thought is what moves the score.

```bash
python -m eval.evaluate --adapter path/to/openverdict-adapter --cases benchmarks/logic_puzzles/puzzles.jsonl --out runs/puzzles
```
