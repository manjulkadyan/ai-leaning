# Review: Days 1–13 (Week 1 and Week 2)

Reviewed on 2026-10-21 after reader feedback: "there are so many areas that you left, plus you didn't explain each and everything in detail".

The feedback is correct, and most of it lands on **Days 1–5**. Those were the first lessons written, before the detailed brief existed. They are also the lessons that every later agent brief named as the style reference for depth and length ("match Day 4/5, ~250–350 lines"). So the thinness was copied forward as the target. Days 8–13 are noticeably better (twice the length, prediction exercises, full solutions), but they share some of the same habits.

## What is missing (coverage)

A grep of Days 1–13 for the core topics a Python + NumPy foundation needs. **0** means the topic never appears in any of the 12 lessons.

| Topic | Where it should be | Found |
|---|---|---|
| Exceptions: `try` / `except` / `finally`, `raise`, reading a traceback | Day 3 | 0 lessons teach it (it appears only incidentally in Days 9–13) |
| Files: `open`, `with`, `pathlib`, reading and writing text | Day 3 | 0 |
| JSON (configs, API responses, label maps) | Day 3 | 0 |
| Iterators, generators, `yield` (how data loaders work) | Day 3 | 0 |
| String methods: `strip`, `join`, `replace`, `startswith`, indexing a string | Day 2 | 0 (only `split`, `lower` and `in`) |
| Type conversion: `int()`, `float()`, `str()`, `round`, `isinstance` | Day 2 | never explained |
| Loops in full: `for` over a range or a dict, `break`, `continue`, `while` | Day 2 | `break` and `continue` 0; `for` only in passing |
| List methods: `extend`, `insert`, `pop`, `remove`, `.sort()` vs `sorted()` | Day 2 | only `append` |
| Dict iteration (`items()`, `keys()`, `values()`), `KeyError`, `del` | Day 2 | 0 |
| Set operations (union, intersection, difference) | Day 2 | 0 |
| Hashability: why a list can't be a dict key or set member | Day 2 | 0 |
| Shallow vs deep copy (`copy.deepcopy`) | Day 2 | 0 (Exercise 3 hints at it) |
| Scope: local vs global, closures | Day 3 | 0 |
| Type hints for containers and optionals (`list[int]`, `X \| None`) | Day 3 | 0 |
| `@property`, `__eq__`, `__len__` with a working example | Day 3 | named only |
| Testing basics: `assert`, a first pytest test | Day 3 | 0 (first used, unexplained, on Day 6) |
| `np.where`, `np.clip`, `np.cumsum`, `np.unique`, `np.sort` / `argsort` | Day 4 | `np.where`, `cumsum` and `unique` 0; `argsort` first on Day 8 |
| `NaN` and `inf`: `np.nan`, `np.isnan`, `np.nanmean` | Day 4 | 0 |
| `astype`, and integer overflow (`uint8` 250 + 10 = 4) | Day 4 | `astype` first on Day 8; overflow never shown |
| `keepdims`, reductions over several axes | Day 4 | first on Day 9 |
| Comparing floats: `np.allclose`, `math.isclose`, why `==` fails | Day 2 / 4 | `allclose` first used, unexplained, on Day 6 |
| Random: `shuffle`, `permutation`, `choice` (train/test splits) | Day 4 | 0 in Days 1–5 |
| Saving arrays: `np.save` / `np.load` | Day 4 | 0 |
| Vectors: distance between two vectors in code, angle from cosine (`arccos`), unit vectors in code | Day 5 | prose only |

Python fundamentals that usually take six to eight lessons were squeezed into Days 2 and 3. That is the biggest structural gap.

## Where the explanations are too thin (depth)

1. **One line per concept.** Many ideas get a single sentence or a table row and nothing else, with no "why", no picture of what happens underneath, and no example of it going wrong. Examples:
    - decorators ("a function that takes a function and returns a modified one"), then a dense example with nested functions, `*args` and `functools.wraps` all at once;
    - `if __name__ == "__main__"` in one sentence, without explaining what `__name__` is;
    - truthiness, `is` vs `==`, f-string format specs like `:.1%`, `**kwargs` unpacking;
    - views vs copies: stated, but nothing on which operations make views and which make copies (fancy indexing copies);
    - the `axis` rule: given as a memory trick, with no step-by-step picture of a 3-D reduction.
2. **Tables used instead of teaching.** The Day 2 scalar and collection tables and the Day 1 terminology table carry most of the content. A table is a good summary after an explanation; it doesn't replace one.
3. **Outputs missing or approximate.** Some lines print nothing (Day 2's `Counter(...)`, Day 4's `rng.normal(size=3)` shown as "3 samples from a bell curve"). Timings are given as "~30 ms" with no note that they depend on the machine. Several exercise solutions show no output at all.
4. **Errors are never shown.** The lessons warn about mistakes but never show the real error message the reader will see (`TypeError: can only concatenate str (not "int") to str`, `KeyError`, `IndexError`, numpy's shape error). Learning to read error messages is never taught.
5. **No step-by-step traces.** A beginner learns comprehensions, slicing with negative steps, and mutable defaults by watching a trace ("call 1: layers is [] → ['conv']; call 2: the same list → ['conv', 'relu']"). The lessons state the result without walking through it.

## Exercises and solutions

- Day 1 has solutions for 2 of 4 exercises; Day 3 has 3 of 4 (the package exercise has none); Day 4 has 4 of 6 (3 and 5 have none).
- Solutions rarely explain *why*, and most don't show output.
- There is no graded progression (warm-up, then core, then challenge), and Days 1–5 have no "predict the output" or "find the bug" exercises (Day 8 does, which is why it reads better).

## Code that fails with the ▶ Run button

Running every block in page order, as the browser does:

- Day 2, the `match device:` block uses `device` before defining it: NameError.
- Day 2 and Day 4 solutions use `temps` / `scores` without defining them: NameError.
- Day 3, the `mlkit` package and `torch` blocks fail in the browser and are not labelled "run on your computer".

## Days 8–13

These are deeper and mostly fine. Shared weaknesses to fix in the same pass:

- missing outputs in a few places;
- no real error messages shown;
- `astype`, `keepdims` and `allclose` introduced later than they're first needed.

Days 15+ were written to a stricter brief and are not part of this review.

## Root causes

1. The brief named Days 4–5 as the depth reference and capped length at 250–350 lines, so agents compressed instead of explaining.
2. The brief said what sections a lesson must have, not what *topics* it must cover. No agent checked coverage against a reference list.
3. Nobody reviewed a lesson before it shipped except its own writer. Tests checked that code ran, not that the teaching was complete.

All three are fixed in [lesson-brief.md](lesson-brief.md) and [lesson-checklist.md](lesson-checklist.md).
