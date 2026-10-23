# Week 1 rewrite map

Days 1–5 are being rewritten under [lesson-brief.md](../lesson-brief.md) and [lesson-checklist.md](../lesson-checklist.md), because they skipped topics and under-explained ([review-week1.md](../review-week1.md)). Five extra lessons are added so the Python and NumPy foundations aren't crammed into five days. The extras get letter suffixes, so no later day number changes.

Rules for rewrites:

- Replace the whole file. Keep a diagram from the old version only if it still fits and passes the render check. You may redraw it.
- Keep the file name, so links from other lessons keep working.
- Everything taught in the old version stays covered somewhere in the new one.
- The **seed topics** below are a starting point, not the inventory. Build the full inventory from the docs, as the brief says.
- The Kotlin bridge is the heart of Week 1. Every Python construct is shown next to the Kotlin the reader already writes.
- The `**Next:**` lines below are links, because every target page will exist when this ships.

## The lessons

### Day 1: Your AI workstation and first Python program

- **File:** `day-01-setup.md`, ~120 min.
- **Seed topics:**
    - why Python for ML (the steering-wheel and native-libraries idea);
    - how CPython runs code (source → bytecode → interpreter, versus Kotlin → dex → ART);
    - installing Python per OS, including `python` vs `python3` vs `py`, PATH, and checking which interpreter runs (`which python`, `sys.executable`);
    - virtual environments explained underneath (what's in `.venv`, what activation changes), creating, activating, deactivating, deleting;
    - pip: install, uninstall, show, list, freeze, version pins, `requirements.txt`, why not to `sudo pip`; `uv` as an option;
    - running code four ways: a script, `python -m`, the REPL, Jupyter;
    - Jupyter in depth: cells, kernel, execution order, restart-and-run-all, magics (`%timeit`, `!`), and notebooks vs scripts;
    - the ▶ Run button on this site and what it can't do;
    - editors: VS Code or PyCharm with interpreter selection, and a debugger breakpoint;
    - reading your first traceback;
    - Colab: GPU runtimes, uploading files, and the free tier's limits;
    - a course repo with git: `.gitignore`, committing `requirements.txt`;
    - a troubleshooting table of real error messages and fixes ("command not found", "No module named numpy" inside or outside the venv, PowerShell execution policy, and so on).
- **Next:** `**Next:** [Day 2: Python's core types, through Kotlin eyes](day-02-python-types.md)`

### Day 2: Python's core types, through Kotlin eyes

- **File:** `day-02-python-types.md`, ~120 min.
- **Seed topics:**
    - names vs objects (the sticky-note model; `id`, `is` vs `==`), dynamic but strong typing, `type()`;
    - **numbers:**
        - `int` (arbitrary size) and `float` (IEEE-754);
        - `/`, `//`, `%` with negatives, `**`, `divmod`, `round` (banker's rounding);
        - why `0.1 + 0.2 != 0.3`, and `math.isclose`;
        - the `math` module, and `inf` and `nan`;
    - `bool` and truthiness, with the full falsy list and `and` / `or` returning operands;
    - `None` and the Optional idea versus Kotlin null-safety (`?.` and `?:` have no direct equivalent: show the idioms);
    - **type conversion:** `int()`, `float()`, `str()`, `bool()`, and their errors (`ValueError: invalid literal for int()`);
    - **strings:**
        - immutability, indexing and slicing;
        - common methods: `strip`, `split`, `join`, `replace`, `lower`, `upper`, `startswith`, `endswith`, `find`, `count`, `isdigit`;
        - f-strings and the format spec mini-language (`:.2f`, `:,`, `:.1%`, `:>8`);
        - escapes, raw strings, multi-line strings;
        - `in`, comparison, `len`;
        - Unicode and `encode` / `decode` briefly;
    - **control flow:**
        - `if` / `elif` / `else`, conditional expressions, comparison chaining (`0 < x < 1`);
        - `for` over `range`, strings and lists; `range` semantics; `enumerate`; `zip`;
        - `while`, `break`, `continue`, `for ... else`;
        - `match` versus Kotlin `when`;
    - `input` and `print` options (`sep`, `end`).
- **Not in this lesson:** collections in depth go to Day 2b. Only show a list or dict where a loop needs one.
- **Next:** `**Next:** [Day 2b: Lists, tuples, dicts and sets in depth](day-02b-collections.md)`

### Day 2b: Lists, tuples, dicts and sets in depth

- **File:** `day-02b-collections.md` (new), ~120 min. Header `**Phase 1 · Week 1 · ~120 minutes**`. Say at the top that it's the second half of Day 2, to do the same day or the next.
- **Seed topics:**
    - **list:**
        - every common method (`append`, `extend`, `insert`, `pop`, `remove`, `index`, `count`, `sort`, `reverse`, `clear`, `copy`);
        - `sort` vs `sorted`, with `key` and `reverse`;
        - concatenation and repetition (`+`, `*`), nested lists;
        - `IndexError`, and how a list is stored (a dynamic array of references; amortised append);
    - **tuple:** immutability, `(1,)`, packing and unpacking, `*rest` unpacking, swapping, named tuples briefly;
    - **dict:**
        - creating, reading (`[]` vs `get`), `KeyError`, `in`, `del`, `pop`, `update`, `|` merge, `setdefault`;
        - iterating over `keys`, `values` and `items`; insertion order is guaranteed;
        - `defaultdict` and `Counter`;
        - nested dicts and JSON-like data;
    - **set:** uniqueness, `add` and `discard`, union, intersection, difference, symmetric difference, `frozenset`;
    - **hashability:** why lists can't be keys, and what `TypeError: unhashable type: 'list'` means;
    - **copying:** shallow vs deep (`copy.copy`, `copy.deepcopy`), and the `[[0]*3]*3` trap traced step by step;
    - **slicing in depth** (start, stop, step, negative values, slice assignment);
    - **comprehensions:** list, dict and set comprehensions, nested ones, conditions, when a loop is clearer, generator expressions previewed (Day 3c covers them);
    - **choosing a collection:** a decision table placed *after* the teaching, plus `in` cost: a list is O(n), a set or dict O(1) on average, timed for real;
    - a Kotlin collections mapping (`listOf` / `mutableListOf` / `mapOf` / `setOf`; `map`, `filter`, `groupBy` and `associate` versus comprehensions and `Counter`).
- **Next:** `**Next:** [Day 3: Functions, classes and modules](day-03-functions-classes.md)`

### Day 3: Functions, classes and modules

- **File:** `day-03-functions-classes.md`, ~120 min.
- **Seed topics:**
    - **functions:**
        - `def`, `return` (multiple values as a tuple, and the implicit `None`), docstrings;
        - positional, keyword and default arguments; the mutable-default trap traced step by step;
        - `*args` and `**kwargs`, and unpacking at the call site; keyword-only (`*`) and positional-only (`/`) parameters;
        - scope (LEGB, `global`, `nonlocal`) and closures;
        - functions as values, `lambda`, `sorted` / `max` / `min` with `key`, `map` and `filter` versus comprehensions;
        - type hints: basics, `list[int]`, `dict[str, float]`, `X | None`, `Callable`, and what checks them (mypy, the IDE) versus what doesn't (the runtime);
    - **classes:**
        - `__init__`, `self`, instance vs class attributes, methods;
        - dunders with working examples: `__repr__` vs `__str__`, `__eq__`, `__len__`, `__getitem__`, `__call__`, `__iter__` (pointer to Day 3c);
        - `@property` with a setter, `@classmethod` (alternative constructors), `@staticmethod`;
        - inheritance, `super()`, method overriding; composition vs inheritance;
        - dataclasses: `field(default_factory=list)`, `frozen`, `order`;
        - the `nn.Module` shape, explained on a toy;
    - **decorators** built up gently: functions are objects → a function that returns a function → the wrapper → the `@` syntax as shorthand → `functools.wraps` → a decorator with arguments, briefly. Show `@torch.no_grad()` and `@lru_cache` as uses;
    - **modules and packages:**
        - what `import` does (runs the file once, caches it in `sys.modules`);
        - `__name__` and `if __name__ == "__main__"` explained;
        - `__init__.py`, absolute vs relative imports, `sys.path`, `python -m`;
        - circular imports briefly;
        - multi-file examples labelled "run on your computer".
- **Split (2026-10-23):** the rewrite came out at 2,634 lines, so it is split at Part 2. Day 3 (`day-03-functions-classes.md`, file name kept for links) is now "Functions, scope and type hints". Classes, dataclasses, the nn.Module shape, decorators and modules move to **Day 3a** (`day-03a-classes-modules.md`, new).
- **Not in this lesson:** errors, files and JSON go to Day 3b; iterators and generators to Day 3c.
- **Next:** Day 3 → `**Next:** [Day 3a: Classes, decorators and modules](day-03a-classes-modules.md)`; Day 3a → `**Next:** [Day 3b: Errors, files and JSON](day-03b-errors-files-json.md)`

### Day 3b: Errors, files and JSON

- **File:** `day-03b-errors-files-json.md` (new), ~120 min.
- **Seed topics:**
    - **reading a traceback** bottom-up, on a real multi-frame example;
    - the common built-in exceptions (`TypeError`, `ValueError`, `KeyError`, `IndexError`, `AttributeError`, `NameError`, `ZeroDivisionError`, `FileNotFoundError`, `ImportError`), each with a real message;
    - **handling exceptions:**
        - `try`, `except`, `else`, `finally`;
        - catching specific exceptions, and why a bare `except` is a bug;
        - `raise`, re-raising, `raise ... from ...`, custom exception classes;
        - versus Kotlin: no checked exceptions in either, `runCatching`, the EAFP vs LBYL style;
    - **assertions:** `assert` and when it's stripped (`python -O`);
    - **files:**
        - context managers and `with` (what `__enter__` and `__exit__` do, versus Kotlin `use {}`);
        - `open` modes (`r`, `w`, `a`, `b`), `encoding="utf-8"`, reading whole files vs line by line;
        - `pathlib.Path`: `/` joining, `exists`, `mkdir(parents=True)`, `glob`, `read_text` and `write_text`, `suffix` and `stem`;
    - **data formats:**
        - CSV with the `csv` module (pandas comes in Week 3);
        - JSON: `json.load` / `dump` / `loads` / `dumps`, `indent`, the type mapping, non-serialisable errors;
        - a model config and a `labels.json` example;
    - **logging** basics versus `print`;
    - **testing basics:** a first pytest test file, `assert` in tests, `pytest.approx`, and running pytest (labelled "on your computer").
    - **Browser note:** file blocks must work in the ▶ Run sandbox, which has a virtual in-memory filesystem. Pyodide supports `open`, `pathlib` and writing to the current directory, so write to a temporary folder (`tempfile.mkdtemp()`) and say why. Check this works; if anything can't run in the browser, label it.
- **Next:** `**Next:** [Day 3c: Iterators and generators](day-03c-iterators-generators.md)`

### Day 3c: Iterators and generators

- **File:** `day-03c-iterators-generators.md` (new), ~90 min.
- **Seed topics:**
    - iterable vs iterator, `iter` and `next`, `StopIteration`, what a `for` loop really does (desugared step by step);
    - writing an iterator class (`__iter__`, `__next__`);
    - generators and `yield`: execution pausing and resuming, traced; generator state; one-shot exhaustion (the classic bug);
    - generator expressions vs list comprehensions, and memory measured with `sys.getsizeof`;
    - lazy pipelines (read → clean → batch);
    - `yield from`;
    - `itertools` essentials: `islice`, `chain`, `count`, `cycle`, `product`, `groupby` (with its sort-first trap), `accumulate`, `batched` (Python 3.12+);
    - `enumerate`, `zip` (with `strict=True`) and `sorted` as iterator consumers;
    - **ML tie-in:** a mini `DataLoader` that shuffles and yields mini-batches from a list of samples with a generator (the bridge to PyTorch's DataLoader in Month 4) and an infinite training-data stream;
    - **Kotlin bridge:** `Sequence` and `sequence { yield() }`, `asSequence()` laziness, and Flow as a distant cousin.
- **Next:** `**Next:** [Day 4: NumPy arrays, and why ML runs on them](day-04-numpy-arrays.md)`

### Day 4: NumPy arrays, and why ML runs on them

- **File:** `day-04-numpy-arrays.md`, ~120 min.
- **Seed topics:**
    - why arrays are fast (contiguous memory, one dtype, SIMD), and the list vs ndarray memory picture (strides previewed);
    - **creating arrays:** `array` from nested lists, `zeros`, `ones`, `full`, `empty` (and its garbage values), `arange` vs `linspace` (with the float-step trap), `eye`, `*_like`;
    - shape, ndim, size, dtype, itemsize, nbytes;
    - **dtypes:**
        - the table; choosing a dtype; `astype`;
        - integer overflow (`np.uint8(250) + np.uint8(10)` → 4) and image clipping;
        - float32 vs float64 precision;
        - upcasting rules (int / int → float64; uint8 / 255.0);
    - **indexing:** basic indexing and slicing in 1-D, 2-D and 3-D; negative indices; integer index vs slice (and the shape it gives);
    - **boolean masks:** combining with `&`, `|` and `~` and the parentheses trap, and assigning through a mask;
    - **views vs copies:** which operations make which (basic slice → view; mask or fancy index → copy), `np.shares_memory`, `.copy()`;
    - **vectorised arithmetic and ufuncs:** `sqrt`, `exp`, `log`, `abs`, `maximum`; comparisons give booleans;
    - **reductions:** `sum`, `mean`, `std`, `min`, `max`, `argmin`, `argmax`, `any`, `all`; `axis` traced on a 3-D example; `keepdims`;
    - **the `%timeit` loop vs vectorised comparison**, with real numbers marked as machine-dependent.
- **Not in this lesson:** `where`, sorting, NaN, random and saving go to Day 4b.
- **Next:** `**Next:** [Day 4b: The NumPy toolkit](day-04b-numpy-toolkit.md)`

### Day 4b: The NumPy toolkit: where, sorting, NaN, random and saving

- **File:** `day-04b-numpy-toolkit.md` (new), ~90 min.
- **Seed topics:**
    - `np.where` with three arguments and with one; `np.clip`; `np.select` briefly;
    - **sorting:** `np.sort`, `argsort` (top-k), `np.unique` with `return_counts`, `np.searchsorted` briefly;
    - cumulative operations: `cumsum`, `cumprod`, `diff`;
    - **missing and special values:**
        - `nan` and `inf`, and how `nan` spreads through arithmetic;
        - `np.isnan`, `np.isfinite`, `nanmean`, `nansum`;
        - why `nan != nan`;
        - division by zero warnings, and `np.errstate`;
    - **comparing floats:** `np.allclose` and `np.isclose` (`rtol` / `atol` explained), `np.array_equal`;
    - **random numbers:**
        - `default_rng` and seeds (why the legacy `np.random.seed` is discouraged);
        - `random`, `integers`, `normal`, `uniform`, `choice` (with and without replacement, with `p`), `shuffle` vs `permutation`;
        - a reproducible train/test split by permutation;
    - **saving and loading:** `np.save` / `np.load`, `savez`, and `savetxt` / `loadtxt` for CSV-ish data (in a temporary folder, for the browser);
    - **useful one-liners:** `np.round`, `np.argwhere` / `nonzero`, `np.count_nonzero`, `np.percentile`, `np.histogram`.
- **Next:** `**Next:** [Day 5: Vectors, the atoms of machine learning](day-05-vectors.md)`

### Day 5: Vectors, the atoms of machine learning

- **File:** `day-05-vectors.md`, ~120 min.
- **Seed topics** (keep the vector playground widget, `<div class="vector-playground"></div>`, and its experiments):
    - a vector as a list, an arrow and a point; notation and ℝⁿ; the two meanings of "dimension";
    - addition and scalar multiplication (geometric and algebraic), subtraction as "the arrow from b to a", linear combinations previewed;
    - the norm by Pythagoras in 2-D, then 3-D, then n-D; distance between two points in code; unit vectors and normalising, with the zero-vector trap;
    - **the dot product:**
        - the algebraic definition, worked by hand;
        - the geometric meaning with cos θ, and a 2-D derivation sketch from the law of cosines;
        - its sign, orthogonality, and projection length as a preview;
    - **angles:** recovering an angle with `arccos` (with clipping for float error);
    - **cosine similarity:** the formula, range and interpretation, cosine vs Euclidean distance on normalised vectors (‖a − b‖² = 2 − 2 cos θ, shown);
    - **ML uses:** embeddings and search (toy embeddings with real outputs), a neuron as w·x + b, with features scaled first;
    - batch versions with a matrix of vectors, previewed (Day 10).
- **Next:** `**Next:** [Day 6: Week 1 review and mini-project](day-06-week1-review.md)`, then Week 2 begins with Day 8.

## After the rewrite

- Day 6 (the review) gets a pass so its recap, self-check and concept map cover Days 2b, 3b, 3c and 4b as well.
- Days 8–13 get a reviewer pass against the checklist. Fixes are patches, not rewrites, unless the reviewer finds a must-topic missing.
