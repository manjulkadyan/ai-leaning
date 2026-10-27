# Day 3d: Iterators and Generators

**Phase 1 · Week 1 · ~150 minutes**

This is the last part of Day 3, after [Day 3](day-03-functions-classes.md) (functions), [Day 3a](day-03a-classes-modules.md) (classes and modules), [Day 3b](day-03b-errors-files-json.md) (errors) and [Day 3c](day-03c-files-json.md) (files and JSON). It's long; a good break point is after Part 2, "Generators".

## Objectives

- [ ] Tell an **iterable** from an **iterator**, and drive one by hand with `iter`, `next` and `StopIteration`.
- [ ] Rewrite any `for` loop as the `while` loop Python really runs, and say which built-ins return one-shot iterators (`map`, `zip`, `enumerate`, files) and which return reusable iterables (`list`, `range`, `dict`).
- [ ] Write an iterator class with `__iter__` and `__next__`, and a reusable iterable class that hands out a fresh iterator each time.
- [ ] Write generator functions with `yield`, and trace exactly when their body runs, pauses and resumes.
- [ ] Recognise the "generator used twice" bug, and fix it three ways.
- [ ] Choose between a generator expression and a list comprehension, backed by a real `sys.getsizeof` measurement.
- [ ] Build a lazy pipeline (read → parse → clean → batch) from small generators, and delegate to another generator with `yield from`.
- [ ] Use the `itertools` essentials: `islice`, `count`, `cycle`, `repeat`, `chain`, `product`, `accumulate`, `pairwise`, `zip_longest`, `takewhile`, `batched` and `groupby` (sorting first).
- [ ] Predict how much of an iterator `zip(strict=True)`, `sorted`, `sum`, `any`, `in` and unpacking consume.
- [ ] Build a mini `DataLoader` that reshuffles every epoch and yields mini-batches, and an infinite training stream that you stop by step count.
- [ ] Translate Kotlin's `Sequence`, `sequence { yield() }` and `asSequence()` into Python, and say where Python's generators behave differently.

**Prerequisites.**

- [Day 2](day-02-python-types.md): `for` loops over `range`, strings and lists; `enumerate`, `zip` (and `strict=True`) and `reversed`; `while True` with `break`; why removing items from a list while looping over it skips items.
- [Day 2b](day-02b-collections.md): lists, tuples and `*` unpacking, dicts and their `keys` / `values` / `items`, `min` / `max` / `sum` / `any` / `all`, `sorted(..., key=...)`, `Counter`, `collections.deque` with `maxlen`, list comprehensions, and the generator-expression preview.
- [Day 3](day-03-functions-classes.md): functions as values, `lambda` and `key=`, closures (functions that remember variables), `*args` and `*` unpacking at a call site, `map` and `filter` returning lazy iterators, and type hints.
- [Day 3a](day-03a-classes-modules.md): classes, `self`, dunder methods, and the `ReviewDataset` class with `__len__` and `__getitem__`, which a `for` loop could already walk through.
- [Day 3b](day-03b-errors-files-json.md): `try` / `except` / `finally`, and catching a specific exception type.
- [Day 3c](day-03c-files-json.md): `with` and context managers, `@contextmanager` and its single `yield`, `open` and reading a file line by line, `tempfile.mkdtemp()` and `pathlib.Path`, and JSON Lines.

!!! info "How to read the code on this page"
    Every Python block has a **▶ Run** button, and blocks share their variables from top to bottom like notebook cells, so run them in order. A bare expression on its own line shows its value, exactly as the `# comment` beside or below it says. Lines that fail on purpose are wrapped in `try` / `except` so the rest of the block still runs, and the comment shows the real error message.

---

## The big idea

Day 2 told you that Python's `for` is always a for-each, and Day 3c showed that `for line in f:` reads a 10 GB file one line at a time. Today you learn what makes both work, and how to build your own.

There are two roles. An **iterable** is something that *can be walked*: a list, a string, a dict, a file. An **iterator** is *the walk itself*: an object that remembers where it is and hands out the next item each time you ask, until it runs out. Every `for` loop asks the iterable for an iterator, then asks the iterator for items one at a time.

**Analogy.** An iterable is a book. An iterator is a bookmark in that book. You can put as many bookmarks into one book as you like, and each one moves forward independently. A bookmark only moves forward; to read the book again, you take a new bookmark. And some things are *only* a bookmark, with no book behind them: a live radio broadcast, a stream of sensor readings, a file being read from disk. Once you've heard it, it's gone.

A **generator** is the easiest way to make your own bookmark. It's a function that, instead of building a whole list and returning it, **hands back one value, pauses, and waits** until someone asks for the next one. That's what lets a data loader feed a model millions of images without ever holding them all in memory, which is the reason this lesson sits in an ML course. PyTorch's `DataLoader` (Month 4) is an iterable that hands out a fresh iterator of mini-batches every epoch. By the end of today you'll have written a small one yourself.

You already know the Kotlin version of this split: `List` versus `Sequence`. A `List` holds everything; a `Sequence` computes items lazily when a terminal operation pulls them. Python's generators are Kotlin's `sequence { yield(x) }`, with one sharp difference you'll meet in Part 2: a Python generator can only be walked **once**.

---

**Part 1: The iteration protocol**

## Iterables and iterators

**What it is.** Python defines iteration with two small rules, called the **iteration protocol** (a *protocol* is a set of methods an object must have to work with some piece of syntax):

- An **iterable** is any object you can pass to the built-in `iter()`. `iter(x)` returns an iterator over `x`. Underneath, `iter(x)` calls the dunder method `x.__iter__()`.
- An **iterator** is an object you can pass to the built-in `next()`. Each call returns the next item. When there are no more items, `next()` raises the exception **`StopIteration`**. Underneath, `next(it)` calls `it.__next__()`. An iterator is also iterable: its `__iter__` returns the iterator itself.

**Why it exists.** One protocol lets a single `for` loop walk lists, strings, dicts, files, database cursors, network streams and objects that compute values on the fly. Code that consumes items (`for`, `sum`, `list`, `max`, unpacking) never needs to know where they come from. That's the hook you'll plug data loaders into.

**Kotlin comparison.** Kotlin has the same two roles:

```kotlin
val names = listOf("cat", "dog")      // Iterable<String>
val it = names.iterator()             // Iterator<String>
while (it.hasNext()) {
    println(it.next())                // cat, then dog
}
it.next()                             // throws NoSuchElementException
```

```python
names = ["cat", "dog"]       # an iterable
it = iter(names)             # an iterator over it
next(it)                     # 'cat'
next(it)                     # 'dog'
try:
    next(it)                 # no items left
except StopIteration:
    print("StopIteration: the iterator is used up")
# StopIteration: the iterator is used up
```

What's the same: an iterable gives out iterators, and an iterator moves forward only. What's different: Python has **no `hasNext()`**. You can't ask an iterator "is there more?"; you call `next()` and find out by catching `StopIteration`. That sounds clumsy, but you rarely do it by hand, because `for` does it for you (next section). Also, `iter` and `next` are built-in *functions* that call the dunder methods, the same pattern as `len(x)` calling `x.__len__()` in Day 3a.

**Example: `next` with a default.** `next(it, default)` returns `default` instead of raising when the iterator is used up, like Kotlin's `firstOrNull()` on what's left:

```python
it = iter(["cat"])
next(it, None)               # 'cat'
print(next(it, None))        # None   used up: the default comes back, no exception
next(it, "<end>")            # '<end>'
```

**How it works underneath.** A list iterator is a small separate object holding two things: a reference to the list, and a position (an index) that starts at 0. Each `next()` reads the item at the position and adds 1. When the position reaches the end, it raises `StopIteration`. The list itself never changes and knows nothing about who's walking it, which is why two iterators over one list move independently:

```python
names = ["cat", "dog", "fox"]
a = iter(names)
b = iter(names)              # a second, independent bookmark
next(a), next(a)             # ('cat', 'dog')
next(b)                      # 'cat'   b starts at the beginning on its own
type(a)                      # <class 'list_iterator'>
iter(a) is a                 # True    an iterator's iter() is itself
iter(names) is names         # False   a list is not its own iterator
```

That index explains Day 2's "removing items while looping skips one" bug: the iterator's position moves forward by one each time, but `remove` shifted the remaining items left underneath it. Dicts and sets protect themselves instead: their iterators notice a change in size and stop with `RuntimeError: dictionary changed size during iteration`, which Day 2b showed.

!!! warning "Common mistake: calling `next()` on a list"
    A list is iterable, but it isn't an iterator: it has no position of its own. `next()` needs an iterator:

    ```python
    try:
        next(["cat", "dog"])
    except TypeError as e:
        print("TypeError:", e)
    # TypeError: 'list' object is not an iterator
    ```

    **Fix:** `next(iter(names))` gets the first item (or `names[0]`, for a list). Use the `iter()` form when you have *any* iterable, such as a set or a dict, that may not support indexing.

## What a `for` loop really does

**What it is.** `for x in items:` is shorthand. Python calls `iter(items)` once, then calls `next()` on that iterator before every round, and stops when `next()` raises `StopIteration`. It catches that exception for you, so you never see it.

**Why it matters.** Once you know this, every looping surprise has an explanation: why a file can only be looped over once, why `zip` "empties", why a generator loop prints nothing the second time, and why `for` works on your own classes as soon as you add `__iter__`.

**Kotlin comparison.** Kotlin's `for (x in items)` compiles to the same thing: `val it = items.iterator(); while (it.hasNext()) { val x = it.next(); ... }`. Python's version has the same shape, with `try` / `except StopIteration` where Kotlin has `hasNext()`.

**Example.** These two pieces of code do the same thing:

```python
names = ["cat", "dog", "fox"]
for name in names:
    print(name)
# cat
# dog
# fox
```

```python
it = iter(names)             # 1. ask the iterable for an iterator, once
while True:
    try:
        name = next(it)      # 2. ask for the next item
    except StopIteration:    # 3. no more items: leave the loop
        break
    print(name)              # 4. the loop body, with name bound to the item
# cat
# dog
# fox
```

**How it works underneath, step by step.** Trace the second version:

1. `iter(names)` creates a `list_iterator` at position 0. It's created **once**, before the loop starts.
2. Round 1: `next(it)` returns `names[0]`, `'cat'`, and the position becomes 1. The body prints `cat`.
3. Round 2: `next(it)` returns `'dog'`; position 2. Round 3: `'fox'`; position 3.
4. Round 4: position 3 is past the end, so `next(it)` raises `StopIteration`. The `except` catches it and `break` ends the loop. A `for ... else` block (Day 2) would run now, because the loop ended by running out, not by a `break` in your code.

<figure class="diagram" markdown>
![Animation: on the left, the code "for name in names: print(name)". In the middle, a list of three boxes holding cat, dog and fox, with an orange iterator arrow below it that starts at index 0 and moves one box to the right each step. On the right, a log fills in line by line: iter(names) creates the iterator at position 0, next(it) returns 'cat', next(it) returns 'dog', next(it) returns 'fox', and finally next(it) raises StopIteration and the loop ends.](../../assets/img/phase-1/day03d-for-loop.svg)
<figcaption>Animated: a for loop calls iter() once, then next() before every round; StopIteration is how it learns to stop.</figcaption>
</figure>

The same protocol is used by everything that walks a collection: `list(x)`, `tuple(x)`, `set(x)`, `sum(x)`, `max(x)`, `"".join(x)`, `x in y`, unpacking (`a, b = x`), `*x` in a call, comprehensions, and `dict(x)`. All of them call `iter()` and then `next()` until `StopIteration`.

!!! warning "Common mistake: looping over something that isn't iterable"
    A number is a single value, not a collection, so it has no `__iter__`:

    ```python
    n_epochs = 3
    try:
        for epoch in n_epochs:
            print(epoch)
    except TypeError as e:
        print("TypeError:", e)
    # TypeError: 'int' object is not iterable
    ```

    "`X` object is not iterable" means `iter()` was called on something without `__iter__` (or `__getitem__`, below). **Fix:** loop over `range(n_epochs)`. The same error appears when a function you expected to return a list returned `None` or a number instead: `TypeError: 'NoneType' object is not iterable` usually means a missing `return`.

## Which built-ins give you an iterator

**What it is.** Some objects are **reusable iterables**: every `iter()` call gives a fresh iterator starting at the beginning, so you can loop over them as often as you like. Others **are iterators**: `iter()` returns the object itself, so there's only one position, and once it's used up it stays empty. These are called **one-shot**.

**Why it matters.** Passing a one-shot iterator to code that loops twice is one of the most common silent bugs in Python data code. You need to know which is which.

**Kotlin comparison.** Kotlin draws the same line between a `List` (iterate as often as you like) and an `Iterator` (once). Kotlin's `map` and `filter` on a list are **eager** and return a new `List`. Python's `map`, `filter`, `zip`, `enumerate` and `reversed` are **lazy** and return one-shot iterators, like calling them on `iterator.asSequence()` in Kotlin.

**Example.** The quick test is `iter(x) is x`: true for an iterator, false for a reusable iterable.

```python
pairs = zip(["cat", "dog"], [0.9, 0.1])
iter(pairs) is pairs         # True    zip returns an iterator
list(pairs)                  # [('cat', 0.9), ('dog', 0.1)]
list(pairs)                  # []      used up: one-shot

r = range(3)
iter(r) is r                 # False   range is a reusable iterable
list(r), list(r)             # ([0, 1, 2], [0, 1, 2])

doubled = map(lambda v: v * 2, [1, 2, 3])
sum(doubled)                 # 12
sum(doubled)                 # 0       map's iterator was used up by the first sum
```

The second `sum(doubled)` is 0 because `sum` of nothing is its start value, 0. No error, no warning: a wrong number.

**How it works underneath.** Here are the common types, sorted by role. Each line below was checked with `iter(x) is x`:

| Reusable iterable (`iter(x) is x` is `False`) | One-shot iterator (`iter(x) is x` is `True`) |
|---|---|
| `list`, `tuple`, `str`, `dict`, `set` | `map(...)`, `filter(...)`, `zip(...)`, `enumerate(...)`, `reversed(...)` |
| `range(...)` (lazy, but re-iterable) | an open file object (`open(...)`) and `csv.reader(...)` |
| dict views: `d.keys()`, `d.values()`, `d.items()` | every generator and generator expression (Part 2) |
| your class with an `__iter__` that returns a new iterator | anything returned by `iter(...)`, and the `itertools` functions (Part 4) |

`range` is worth a second look. It's lazy (Day 2: `range(10**9)` costs no memory), yet reusable, because it's a **sequence**: it stores only start, stop and step, supports `len` and indexing, and hands out a new `range_iterator` every time. Lazy and one-shot are different properties.

**Checking with `collections.abc`.** The `collections.abc` module defines abstract classes for these roles, and `isinstance` can test against them:

```python
from collections.abc import Iterable, Iterator

isinstance([1, 2], Iterable), isinstance([1, 2], Iterator)            # (True, False)
isinstance(zip([1], [2]), Iterable), isinstance(zip([1], [2]), Iterator)   # (True, True)
isinstance(range(3), Iterator)                                         # False
```

These checks look for the methods: `Iterable` means "has `__iter__`", `Iterator` means "has `__iter__` and `__next__`". They're also what you write in type hints, where they accept `[...]` like `list[int]` does (Day 3): `def mean(xs: Iterable[float]) -> float:` says "anything I can loop over", and `-> Iterator[int]` says "I return a one-shot iterator".

!!! warning "Common mistake: a file is one-shot too"
    A file object is an iterator over its lines, so a second loop over the same open file finds nothing. Using Day 3c's temporary folder:

    ```python
    import json, tempfile
    from pathlib import Path

    work = Path(tempfile.mkdtemp())
    labels_path = work / "labels.txt"
    labels_path.write_text("cat\ndog\nfox\n", encoding="utf-8")   # 12   characters written (Day 3c)

    with open(labels_path, encoding="utf-8") as f:
        n_lines = sum(1 for line in f)     # walks the file to the end
        first = [line.strip() for line in f]
    n_lines, first                         # (3, [])   the second pass started at the end
    ```

    `sum(1 for line in f)` adds 1 per line, a common way to count the items of an iterator. **Fix:** read the lines into a list once (`lines = f.read().splitlines()`) if the file fits in memory, or reopen the file for the second pass. (`f.seek(0)` also rewinds a file, but reopening is clearer.)

## Writing an iterator class

**What it is.** To make your own iterator, write a class with two methods: `__next__`, which returns the next item or raises `StopIteration`, and `__iter__`, which returns `self`.

**Why it exists.** It's how you give a `for` loop items that don't sit in a list: values computed on demand, rows read from a source, batches assembled from a dataset. In practice you'll usually write a generator instead (Part 2, four lines instead of fifteen), but the class shows exactly what a generator does for you.

**Kotlin comparison.** In Kotlin you implement `Iterator<Int>`:

```kotlin
class Countdown(start: Int) : Iterator<Int> {
    private var current = start
    override fun hasNext() = current > 0
    override fun next(): Int {
        if (!hasNext()) throw NoSuchElementException()
        return current--
    }
}
for (n in Countdown(3)) println(n)    // 3, 2, 1
```

Kotlin's `for` accepts an `Iterator` because the standard library adds `operator fun Iterator<T>.iterator() = this`: an iterator's iterator is itself. That's exactly Python's `__iter__` returning `self`.

```python
class Countdown:
    """Counts down from start to 1."""
    def __init__(self, start: int):
        self.current = start

    def __iter__(self):
        return self                  # an iterator is its own iterator

    def __next__(self) -> int:
        if self.current <= 0:
            raise StopIteration      # the "no more items" signal
        value = self.current
        self.current -= 1
        return value

for n in Countdown(3):
    print(n, end=" ")
print()
# 3 2 1
c = Countdown(2)
next(c), next(c)                     # (2, 1)
next(c, "done")                      # 'done'
```

What's different from Kotlin: there's no `hasNext()`, so `__next__` does both jobs: it checks whether there's an item, and either returns it or raises.

**How it works underneath.** The iterator's whole state is its attributes, here `self.current`. `for n in Countdown(3)` calls `iter()` on the new object, which returns the object itself, then calls `__next__` each round: `current` goes 3 → 2 → 1 → 0, returning 3, 2 and 1, and the fourth call raises `StopIteration`.

**A reusable iterable: hand out a fresh iterator.** Because `Countdown` is its own iterator, it's one-shot, like `zip`. To loop over the same thing twice, separate the two roles. The iterable holds the settings, and its `__iter__` builds a *new* iterator every time:

```python
class CountdownFrom:
    """A reusable iterable: every loop gets a fresh Countdown."""
    def __init__(self, start: int):
        self.start = start

    def __iter__(self):
        return Countdown(self.start)   # a new iterator, at the beginning

one_shot = Countdown(3)
list(one_shot), list(one_shot)         # ([3, 2, 1], [])
reusable = CountdownFrom(3)
list(reusable), list(reusable)         # ([3, 2, 1], [3, 2, 1])
```

That's the list / list-iterator split again: `CountdownFrom` is the book, each `Countdown` is a bookmark.

**Where Day 3a's `__getitem__` loop fits.** Day 3a's `ReviewDataset` had only `__len__` and `__getitem__`, and a `for` loop still worked. That's because `iter(x)` tries two things, in order:

1. if the class has `__iter__`, call it;
2. otherwise, if it has `__getitem__`, build an iterator that calls `x[0]`, `x[1]`, `x[2]`, ... and stops at the first `IndexError`.

Only if both are missing do you get "object is not iterable". The fallback exists for old code. Modern classes define `__iter__`, which is clearer and works for things that can't be indexed. (`isinstance(x, Iterable)` only checks for `__iter__`, so it says `False` for a `__getitem__`-only class even though a `for` loop works. The reliable test is to call `iter(x)` and catch `TypeError`.)

!!! warning "Common mistake: forgetting `StopIteration`, or forgetting `__iter__`"
    An iterator whose `__next__` never raises `StopIteration` never ends. Here a hand-written guard stops it:

    ```python
    class Forever:                     # bug: __next__ never raises StopIteration
        def __init__(self):
            self.n = 0
        def __iter__(self):
            return self
        def __next__(self):
            self.n += 1
            return self.n

    for x in Forever():
        if x > 3:
            print("still going at", x, "- stopping by hand")
            break
    # still going at 4 - stopping by hand
    ```

    Without the `if`, this loop runs until you press Ctrl+C. An infinite iterator can be exactly what you want (Part 4's `count` is one), but then the *consumer* must stop it. And a class with `__next__` but no `__iter__` can be driven with `next()` but not looped over:

    ```python
    class HalfDone:
        def __next__(self):
            return 1

    try:
        for x in HalfDone():
            pass
    except TypeError as e:
        print("TypeError:", e)
    # TypeError: 'HalfDone' object is not iterable
    ```

    **Fix:** always write both methods, with `__iter__` returning `self`.

---

**Part 2: Generators**

## Generator functions and `yield`

**What it is.** A **generator function** is a function whose body contains the keyword **`yield`**. Calling it does **not** run the body. It returns a **generator**, an iterator object. Each `next()` on the generator runs the body until it reaches a `yield`; the yielded value is what `next()` returns, and the function **pauses** right there, with all its local variables kept. The next `next()` **resumes** from that point. When the body finishes (falls off the end, or reaches `return`), the generator raises `StopIteration`.

**Why it exists.** The `Countdown` class above needed `__init__`, `__iter__`, `__next__`, an attribute for its state, and a manual `raise StopIteration`. A generator gets all of that from the language: the state lives in ordinary local variables, and "where am I?" is the line it paused on. You'll write generators for anything that produces a stream: lines of a file, records, batches, augmented images, or an endless training stream.

**Kotlin comparison.** Kotlin's closest construct is the `sequence` builder, where `yield` suspends the block:

```kotlin
fun countdown(start: Int) = sequence {
    var n = start
    while (n > 0) {
        yield(n)            // suspends here until the next item is requested
        n--
    }
}
countdown(3).toList()       // [3, 2, 1]
```

```python
def countdown(start: int) -> Iterator[int]:
    n = start
    while n > 0:
        yield n             # pauses here until the next item is requested
        n -= 1

list(countdown(3))          # [3, 2, 1]
for n in countdown(2):
    print(n, end=" ")
print()
# 2 1
```

The hint `-> Iterator[int]` (from `collections.abc`, imported in Part 1) says what the *caller* gets back: calling `countdown` returns an iterator of ints, not an int.

What's the same: the code reads like a normal loop, `yield` hands out one value and suspends, and nothing runs until a consumer pulls. What's different: Kotlin needs a builder block (`sequence { }`) and `yield` is a suspending function inside it; Python's `yield` is a keyword, and its mere presence anywhere in a `def` turns the whole function into a generator function. And, as the next section shows, Kotlin's sequence can be walked many times, while a Python generator object is one-shot.

**Example: watch the body pause and resume.** Prints inside the generator (indented by two spaces) show exactly when its code runs:

```python
def countdown_traced(n):
    print("  body starts")
    while n > 0:
        print(f"  about to yield {n}")
        yield n
        print(f"  resumed after {n}")
        n -= 1
    print("  body ends")

gen = countdown_traced(2)    # nothing printed: the body hasn't started
type(gen)                    # <class 'generator'>
```

```python
next(gen)
#   body starts
#   about to yield 2
# 2
```

```python
next(gen)
#   resumed after 2
#   about to yield 1
# 1
```

```python
try:
    next(gen)
except StopIteration:
    print("StopIteration")
#   resumed after 1
#   body ends
# StopIteration
```

**How it works underneath, step by step.**

1. `countdown_traced(2)` creates a generator object. Inside it is a **frame**: the function's local variables (`n = 2`) and a marker for where execution is. The marker is at the very start; no line has run.
2. The first `next(gen)` runs the body from the top: it prints "body starts", enters the loop, prints "about to yield 2", and reaches `yield n`. The generator hands `2` back to the caller and **freezes**: the frame, with `n = 2` and the marker on the `yield` line, is kept inside the generator object.
3. The second `next(gen)` unfreezes the frame and continues on the line *after* the `yield`: it prints "resumed after 2", sets `n = 1`, loops, and pauses again at `yield n`, handing back `1`.
4. The third `next(gen)` resumes, prints "resumed after 1", sets `n = 0`, the `while` test fails, it prints "body ends" and the function finishes. Finishing a generator raises `StopIteration` in the caller. The generator is now **closed**: its frame is released (`gen.gi_frame`, the attribute that held it, is now `None`), and every later `next()` raises `StopIteration` at once.

<figure class="diagram" markdown>
![Animation: two columns. On the left, the caller's code: gen = countdown_traced(2), then three next(gen) calls. On the right, the generator's body. A highlight steps through the caller's lines; for each next(gen), an orange highlight runs through the body lines until it stops on the yield line, where a badge says "paused, n = 2" and then "paused, n = 1"; the value 2 or 1 travels back to the caller. On the last next(gen) the highlight runs past the loop to the end of the body and the caller receives StopIteration. Below the caller, a state label changes from GEN_CREATED to GEN_RUNNING, GEN_SUSPENDED and finally GEN_CLOSED, next to the current value of n, which is replaced by "frame released" once the generator is closed.](../../assets/img/phase-1/day03d-generator-pause.svg)
<figcaption>Animated: each next() runs the generator's body only as far as the next yield, then the body pauses with its local variables intact.</figcaption>
</figure>

You can ask a generator which state it's in with `inspect.getgeneratorstate`. The `inspect` module is the standard library's toolbox for looking inside live objects:

```python
import inspect

g = countdown(2)
inspect.getgeneratorstate(g)   # 'GEN_CREATED'     made, body not started
next(g)                        # 2
inspect.getgeneratorstate(g)   # 'GEN_SUSPENDED'   paused at a yield
list(g)                        # [1]               list() takes the rest
inspect.getgeneratorstate(g)   # 'GEN_CLOSED'      finished
```

(There's a fourth state, `'GEN_RUNNING'`, which you only see from inside the generator while its code is executing.)

**A generator is a function whose locals survive.** Day 3's closures remembered *variables* between calls. A generator remembers variables **and its position in the code**. That's why the `while` loop in `countdown` can keep `n` as a plain local: between two `next()` calls, the frame holding `n` is parked inside the generator object instead of being thrown away, which is what happens to an ordinary function's frame when it returns.

**`return` in a generator.** A bare `return` ends a generator early, exactly like falling off the end. `return value` also ends it, and attaches `value` to the `StopIteration` as `.value`; ordinary consumers like `for` and `list` ignore it. You'll meet the one place it's used, `yield from`, later in this part.

```python
def first_positive(xs):
    for x in xs:
        if x > 0:
            yield x
            return               # stop after the first one
    print("  none found")

list(first_positive([-1, 3, 5]))  # [3]
list(first_positive([-1, -2]))
#   none found
# []
```

!!! warning "Common mistake: expecting a generator function to run when called"
    Calling a generator function only builds the generator. If it's never iterated, its body never runs, and any side effect inside it, like a `print` or a write, never happens:

    ```python
    def save_all(records):
        for r in records:
            print("  saving", r)
            yield r

    save_all(["a", "b"])         # <generator object save_all at 0x...>   no "saving" lines
    result = list(save_all(["a", "b"]))
    #   saving a
    #   saving b
    ```

    The first call prints no "saving" lines. The ▶ Run button only echoes the generator's repr, `<generator object save_all at 0x...>`, where the hexadecimal memory address differs on every run. If a function's job is to *do* something rather than produce values, it shouldn't contain `yield`.

!!! warning "Common mistake: treating a generator like a list"
    A generator has no length and no indexes: it doesn't know how many items it will produce until it has produced them.

    ```python
    g = countdown(3)
    try:
        len(g)
    except TypeError as e:
        print("TypeError:", e)
    try:
        g[0]
    except TypeError as e:
        print("TypeError:", e)
    # TypeError: object of type 'generator' has no len()
    # TypeError: 'generator' object is not subscriptable
    ```

    ("Subscriptable" means "supports `x[...]`".) **Fix:** if you need the length, an index or a second pass, convert once with `items = list(g)`. If you only need the first item, use `next(g)`.

## The one-shot trap

**What it is.** A generator object is an iterator, so it can be walked **once**. After that it's empty, and every consumer silently sees zero items. No error is raised.

**Why it matters.** This is the classic generator bug, and it produces wrong results rather than crashes. It appears whenever a function loops over its input twice, for example once to compute a mean and once to use it.

**Kotlin comparison.** Here Kotlin and Python really differ. A Kotlin `sequence { }` re-runs its block for every terminal operation, so it can be consumed many times:

```kotlin
val s = countdown(3)
s.toList()      // [3, 2, 1]
s.toList()      // [3, 2, 1]   the block ran again
```

A Python generator object is one paused function call; once that call has finished, it's finished:

```python
s = countdown(3)
list(s)         # [3, 2, 1]
list(s)         # []   the same generator object, already finished
```

The closest Kotlin analogue to Python's behaviour is `iterator.asSequence()`, whose result is constrained to be iterated only once: a second pass throws `IllegalStateException`. Python doesn't throw; it gives you nothing.

**Example: the bug.** This function centres values by subtracting their mean. It works on a list, and returns nothing on a generator:

```python
def centred(values):
    values_mean = sum(values) / 3        # first pass (3 items, to keep it short)
    return [v - values_mean for v in values]   # second pass

centred([1.0, 2.0, 6.0])                     # [-2.0, -1.0, 3.0]
centred(v for v in [1.0, 2.0, 6.0])          # []   the first pass used everything up
```

**How it works underneath.** `sum(values)` called `next()` on the generator until it raised `StopIteration`. The comprehension then called `iter()` on the same generator, which returned the same, finished object, whose first `next()` raised `StopIteration` immediately. Zero rounds, empty list.

The same thing happens with an `in` test, which consumes the iterator up to and including the match:

```python
nums = iter([1, 5, 2, 8, 3])
5 in nums                    # True
list(nums)                   # [2, 8, 3]   1 and 5 were consumed by the search
```

**Three fixes.**

1. **Materialise once.** Turn the input into a list at the top: `values = list(values)`. Simple and right whenever the data fits in memory. This is what most libraries do with an argument they need to read twice.
2. **Make a fresh generator for each pass.** Call the generator function again: `sum(read_values())` then `[... for v in read_values()]`. Right when re-reading is cheap and the data doesn't fit in memory (re-reading a file, say).
3. **Make a reusable iterable.** Write a class whose `__iter__` is a generator method. Every `for` loop calls `__iter__`, gets a brand-new generator, and starts at the beginning:

```python
class Values:
    def __init__(self, data):
        self.data = data

    def __iter__(self):                # a generator method: each call makes a new generator
        for v in self.data:
            yield v

vals = Values([1.0, 2.0, 6.0])
centred(vals)                          # [-2.0, -1.0, 3.0]   two passes, two fresh generators
iter(vals) is vals                     # False   reusable, like a list
```

Fix 3 is the idiom to remember: a class holding the settings, with `__iter__` written as a generator. It replaces the two-class `CountdownFrom` / `Countdown` pair from Part 1 with one short method, and it's exactly how a PyTorch `DataLoader` can be looped over once per epoch (Part 5).

**`itertools.tee`: two iterators from one.** You may see `tee` suggested for "using a generator twice". `tee(it)` returns two independent iterators over the same items:

```python
from itertools import tee

a, b = tee(countdown(3))
list(a), list(b)                       # ([3, 2, 1], [3, 2, 1])
```

!!! warning "Common mistake: `itertools.tee` is not a free second pass"
    `tee` works by storing every item one copy has seen and the other hasn't yet. Above, running `a` to the end before touching `b` stored all three items, the same memory as `list(gen)` with more overhead. Use `tee` only when the two copies advance roughly together (say, comparing each item with the next); otherwise use `list`.

## Generator expressions versus list comprehensions

**What it is.** Day 2b previewed it: `(expr for x in items if cond)` is a **generator expression**. It's a list comprehension with round brackets, and it builds a generator instead of a list. Underneath, it's a nameless generator function, defined and called on the spot.

**Why it exists.** When you only need to feed values to one consumer, such as `sum`, `max`, `any`, `"".join` or a `for` loop, building the whole list first wastes memory. A generator expression produces one value at a time and keeps only its own small frame.

**Kotlin comparison.** `listOf(...).map { }.filter { }` is eager, like a list comprehension. `asSequence().map { }.filter { }` is lazy, like a generator expression. As in Kotlin, the lazy form doesn't run until a terminal operation (here: a consumer such as `sum` or `list`) pulls values.

**Example: memory, measured.** `sys.getsizeof(x)` returns the size of the object `x` itself, in bytes:

```python
import sys

squares_list = [n * n for n in range(1_000_000)]
squares_gen = (n * n for n in range(1_000_000))
sys.getsizeof(squares_list)              # 4224364   about 4 MB in the browser
sys.getsizeof(squares_gen)               # 104       the same size for any length
sum(squares_list) == sum(squares_gen)    # True
```

Those are the numbers the ▶ Run button shows. Its Python is 32-bit WebAssembly (Python 3.14), where each reference takes 4 bytes. On 64-bit Python on your computer, references take 8 bytes, and the same lines give `8448728` (about 8 MB) and `200` (with Python 3.13 on my laptop). Either way the list grows with its length and the generator doesn't.

Two details make these numbers honest. `getsizeof` is **shallow**: the list's 4 MB is only its array of one million references (4 bytes each here, plus spare room for growth, Day 2b); the integer objects they point to are extra. And `1_000_000` is how Python lets you write `1000000` readably, with underscores (Day 2).

**How it works underneath.** `(n * n for n in range(1_000_000))` behaves like this:

```python
def _squares(iterable):          # roughly what Python builds for you
    for n in iterable:
        yield n * n

squares_gen2 = _squares(range(1_000_000))
sum(squares_gen2) == sum(squares_list)   # True
```

Note the argument: the expression after the **first** `in` (`range(1_000_000)`) is evaluated **immediately**, when the generator expression is created, and passed in. Everything else, the `n * n` and any `if` condition, runs **later**, one item at a time, when a consumer pulls. That split causes the mistake below.

**Which to choose.**

- Use a **list comprehension** when you need the result more than once, need `len` or indexing, want to print or inspect it, or the data is small. Lists are also slightly faster to loop over than generators, so laziness isn't a speed trick; it's a memory one.
- Use a **generator expression** when the result goes straight into one consumer, when the data is large or comes from a file or a stream, or when it may be infinite.

!!! warning "Common mistake: a generator expression reads variables when it runs, not when it's written"
    Because the condition runs lazily, it sees the *current* value of any variable it uses, as a closure does (Day 3's late binding):

    ```python
    scores = [0.2, 0.7, 0.9]
    threshold = 0.5
    confident = (s for s in scores if s > threshold)
    threshold = 0.8                      # changed before anything was pulled
    list(confident)                      # [0.9]   used 0.8, not 0.5
    ```

    The first iterable, by contrast, was captured at creation:

    ```python
    scores = [0.2, 0.7, 0.9]
    gen = (s for s in scores)
    scores = [1.0]                       # rebinding the name doesn't affect gen
    list(gen)                            # [0.2, 0.7, 0.9]
    ```

    **Fix:** consume a generator expression right where you create it, or use a list comprehension if the values must be fixed now. A generator that lives a long time between creation and use is a warning sign.

## Generators that hold resources: `finally` and `close()`

**What it is.** A generator can open a file (or any resource) and keep it open while it's paused. The resource is released when the generator finishes, or when it's **closed** early with `gen.close()`.

**Why it matters.** Reading files lazily is the main use of generators in data code (Part 3), so you need to know when the file actually closes. It also explains how Day 3c's `@contextmanager` works.

**Example.** `gen.close()` makes the paused `yield` raise a special exception, `GeneratorExit`, inside the generator. It travels up through any `with` block or `finally`, which run their clean-up, and then the generator is finished:

```python
def lines_of(path):
    print("  opening", path.name)
    with open(path, encoding="utf-8") as f:
        try:
            for line in f:
                yield line.rstrip("\n")
        finally:
            print("  closing", path.name)

g = lines_of(labels_path)
next(g)
#   opening labels.txt
# 'cat'
```

```python
g.close()                     # stop early: GeneratorExit at the paused yield
#   closing labels.txt
inspect.getgeneratorstate(g)  # 'GEN_CLOSED'
list(lines_of(labels_path))   # read to the end: closes by itself
#   opening labels.txt
#   closing labels.txt
# ['cat', 'dog', 'fox']
```

**How it works underneath.** While the generator is paused at `yield`, it's *inside* the `with` block, so the file is open. Three things end the pause for good: the loop runs out (the body finishes normally), `close()` (the `yield` raises `GeneratorExit`), or the generator object being garbage-collected, which calls `close()` for you. In all three, `with` and `finally` run. The `print` inside `finally` is only there to make the closing visible; the `with` alone would close the file.

**Kotlin comparison.** A Kotlin `sequence { }` can hold a resource in a `try` / `finally` too, but a sequence has no `close()`. If the consumer stops early, the suspended block is never resumed, so its `finally` never runs:

```kotlin
val lines = sequence {
    val reader = File("labels.txt").bufferedReader()
    try {
        while (true) yield(reader.readLine() ?: break)
    } finally {
        reader.close()          // never runs if the consumer stops early
    }
}
lines.first()                   // "cat"; the reader is left open
```

That's why Kotlin scopes the file from the outside with `File.useLines { }`. A Python generator can be closed from the outside, by `close()` or by garbage collection, and closing runs its `finally`.

**Breaking out of a loop doesn't close the generator.** `break` only stops pulling. If you still hold the generator in a variable, it stays paused, with the file open, until you close it, or until the last reference to it goes away and it's garbage-collected:

```python
g = lines_of(labels_path)
for line in g:
    break                           # stop after the first line
#   opening labels.txt
inspect.getgeneratorstate(g)        # 'GEN_SUSPENDED'   still paused: the file is still open
g.close()
#   closing labels.txt
```

To close it at a known point, wrap it in `contextlib.closing`, a ready-made context manager whose `__exit__` calls `close()`:

```python
from contextlib import closing

with closing(lines_of(labels_path)) as g:
    first_line = next(g)
#   opening labels.txt
#   closing labels.txt
first_line                          # 'cat'
```

**Now you can see how `@contextmanager` works.** Day 3c's `@contextmanager` builds a context manager from a generator with one `yield`. Its `__enter__` calls `next()` once, which runs your setup and stops at the `yield`; the yielded value is what `as` binds. Its `__exit__` calls `next()` again, which resumes after the `yield`, runs your teardown, and expects the generator to finish. Driving one by hand shows it:

```python
def managed(name):
    print(f"  setup {name}")
    yield name.upper()                # what `as` would bind
    print(f"  teardown {name}")

g = managed("gpu")
resource = next(g)                    # what __enter__ does
#   setup gpu
resource                              # 'GPU'
next(g, None)                         # what __exit__ does: resume, run to the end
#   teardown gpu
```

(When the `with` body raises, `__exit__` uses `gen.throw(exc)` instead, which raises the exception at the paused `yield`. That's why Day 3c put a `try` / `finally` around its `yield`.)

!!! warning "Common mistake: `next()` inside a generator becomes `RuntimeError`"
    If code inside a generator calls `next()` on an iterator that's empty, the `StopIteration` must not leak out, because it would look like "this generator finished" and silently cut your data short. Since Python 3.7, Python turns it into a `RuntimeError`:

    ```python
    def first_of_each(lists):
        for xs in lists:
            yield next(iter(xs))          # fails on an empty list

    try:
        list(first_of_each([[1, 2], [3], []]))
    except RuntimeError as e:
        print("RuntimeError:", e)
    # RuntimeError: generator raised StopIteration
    ```

    **Fix:** give `next` a default and decide what an empty input means, `next(iter(xs), None)`, or skip empty inputs with `if xs:`.

## `yield from`: delegating to another iterable

**What it is.** `yield from iterable` yields every item of `iterable`, one at a time, as if you'd written `for x in iterable: yield x`. The generator that contains it pauses on each of those items in turn.

**Why it exists.** It lets you build a generator out of other generators: chain several sources, or walk a nested structure recursively (nested lists, a folder tree, a JSON document). It also passes the inner generator's `return` value back out, which the plain `for` version can't.

**Kotlin comparison.** Inside Kotlin's `sequence { }`, `yieldAll(other)` does the same job:

```kotlin
val both = sequence {
    yieldAll(listOf("cat", "dog"))
    yield("fox")
}
both.toList()                   // [cat, dog, fox]
```

```python
def both():
    yield from ["cat", "dog"]   # any iterable works
    yield "fox"

list(both())                    # ['cat', 'dog', 'fox']
```

**Example: flattening nested lists.** A generator that calls itself on each sub-list. `isinstance(item, list)` asks "is this a list?" (Day 2):

```python
def flatten(nested):
    for item in nested:
        if isinstance(item, list):
            yield from flatten(item)    # hand out everything the inner call yields
        else:
            yield item

list(flatten([1, [2, [3, 4]], 5]))     # [1, 2, 3, 4, 5]
```

**How it works underneath, step by step.** `list(...)` calls `next()` on the outer generator. It yields `1`. Next, `item` is `[2, [3, 4]]`, so it creates a second generator, `flatten([2, [3, 4]])`, and passes each `next()` straight through to it. That one yields `2`, then meets `[3, 4]` and creates a third generator, which yields `3` and `4` and finishes. The second then finishes too, and control returns to the outer generator, which yields `5`. At the deepest point, three paused frames are stacked inside each other, each waiting on the next.

**The return value.** `result = yield from sub` receives the value `sub` returned (its `StopIteration.value`):

```python
def read_batch():
    yield "a"
    yield "b"
    return 2                     # how many items it produced

def read_all():
    n = yield from read_batch()
    print("  inner generator produced", n)

list(read_all())
#   inner generator produced 2
# ['a', 'b']
```

The type hint for a generator that returns a value is `Generator[Y, S, R]` from `collections.abc`: the type it yields, the type it can receive (Going further; `None` here) and the type it returns. `read_batch` is a `Generator[str, None, int]`. When nothing is returned, `Iterator[str]` says enough.

!!! warning "Common mistake: `yield` where you meant `yield from`"
    `yield flatten(item)` hands out the inner *generator object* as a single item, instead of its contents:

    ```python
    def flatten_bug(nested):
        for item in nested:
            if isinstance(item, list):
                yield flatten_bug(item)     # bug: yields the generator itself
            else:
                yield item

    [type(x).__name__ for x in flatten_bug([1, [2, 3], 4])]   # ['int', 'generator', 'int']
    ```

    `type(x).__name__` is the name of an object's class, used here because a generator's repr includes a memory address that changes on every run. If you see `<generator object ...>` in your data, look for a `yield` that should be `yield from`.

---

**Part 3: Lazy pipelines**

## A pipeline: read → parse → clean → batch

**What it is.** A **lazy pipeline** is a chain of generators, each of which takes an iterable, does one small job to each item, and yields the results. The last consumer pulls one item at a time through every stage.

**Why it exists.** It's the shape of every data-loading step in ML: read raw records from disk, parse them, drop or fix bad ones, normalise, group into mini-batches. Written as generators, each stage is a few lines you can test alone, and the whole chain uses memory for **one item (or one batch)**, however big the file is.

**Kotlin comparison.** Kotlin's `useLines` gives you a `Sequence<String>` over a file, and sequence operations chain lazily:

```kotlin
File("reviews.jsonl").useLines { lines ->
    lines.filter { it.isNotBlank() }
         .map { parse(it) }
         .chunked(2)
         .forEach { batch -> train(batch) }
}
```

Python writes each stage as a generator function and nests the calls. The dataset for this section is a small JSON Lines file (Day 3c) of app reviews, with a blank line and a broken record mixed in, as real data has:

```python
reviews_path = work / "reviews.jsonl"
raw_lines = [
    '{"app": "Snapseed", "text": "  Great   editor ", "stars": 5}',
    '{"app": "Keep", "text": "ok", "stars": 3}',
    '',
    '{"app": "Pocket Casts", "text": "Crashes on START", "stars": 1}',
    '{"app": "Lens", "text": "Love it', 
    '{"app": "Files", "text": "Fast and clean", "stars": 4}',
    '{"app": "Maps", "text": "Battery hog", "stars": 2}',
]
reviews_path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")   # 307   characters written
```

The fifth line is cut off (no closing quote or brace), so it isn't valid JSON. Now the four stages:

```python
def read_lines(path):
    """Yield the lines of a text file, one at a time."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield line

def parse(lines: Iterable[str]) -> Iterator[dict]:
    """Yield one dict per valid JSON line; skip blank and broken lines."""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue            # in real code, log it (Day 3c) and count the skips

def clean(records):
    """Yield (text, stars) with whitespace squashed and text lower-cased; drop very short texts."""
    for r in records:
        text = " ".join(r["text"].split()).lower()
        if len(text) >= 3:
            yield text, r["stars"]

def batch(items, size):
    """Yield lists of up to `size` items."""
    current = []
    for item in items:
        current.append(item)
        if len(current) == size:
            yield current
            current = []
    if current:                 # a last, shorter batch
        yield current

pipeline = batch(clean(parse(read_lines(reviews_path))), 2)
for b in pipeline:
    print(b)
# [('great editor', 5), ('crashes on start', 1)]
# [('fast and clean', 4), ('battery hog', 2)]
```

Line by line through `clean`: `r["text"].split()` with no argument splits on any run of whitespace and drops the empty pieces, so `"  Great   editor "` becomes `['Great', 'editor']`; `" ".join(...)` glues them back with single spaces (Day 2a). `"ok"` is shorter than 3 characters, so it's dropped. In `batch`, `current` collects items until it holds `size` of them, is yielded, and a new empty list starts; the `if current:` at the end yields whatever is left over.

**How it works underneath: who pulls whom.** Building `pipeline` ran *no* stage code: four generator objects were created, each holding the one inside it. Then the `for` loop calls `next()` on `batch`, which needs an item, so it calls `next()` on `clean`, which calls `next()` on `parse`, which calls `next()` on `read_lines`, which reads one line from the file. That line travels back up through every stage before the next line is read. A small helper, `tap`, makes the order visible; it passes every item through unchanged and prints it on the way (`!r` in an f-string shows the `repr`, Day 2a):

```python
def tap(label, items):
    for item in items:
        print(f"  {label:<6} {item!r}")
        yield item

traced = batch(tap("clean", clean(tap("parse", parse(read_lines(reviews_path))))), 2)
next(traced)
#   parse  {'app': 'Snapseed', 'text': '  Great   editor ', 'stars': 5}
#   clean  ('great editor', 5)
#   parse  {'app': 'Keep', 'text': 'ok', 'stars': 3}
#   parse  {'app': 'Pocket Casts', 'text': 'Crashes on START', 'stars': 1}
#   clean  ('crashes on start', 1)
# [('great editor', 5), ('crashes on start', 1)]
```

Read it as a timeline. Record 1 goes through `parse` and `clean` before record 2 is even read. Record 2 (`"ok"`) passes `parse` but `clean` drops it, so `clean` pulls again. The blank line never shows: `parse` skipped it internally. After two cleaned items, `batch` has a full batch and yields it, and everything pauses. The broken line and the rest of the file haven't been pulled through the pipeline yet. (The file object itself reads ahead from disk in fixed-size chunks, 8 KB by default, so for this small file all the bytes are already in memory. That's why a pipeline's memory stays constant rather than strictly one line.)

<figure class="diagram" markdown>
![Animation in two lanes. Top lane, "Eager: a list after every stage": all seven lines move from read into a full list, then all into the next full list, then the next, and a memory bar fills to the whole dataset before the first batch exists. Bottom lane, "Lazy: generators": one record at a time travels through read, parse, clean and into a batch slot; when two records fill the slot, the batch is handed to the training step on the right, and the memory bar never holds more than one record plus one batch.](../../assets/img/phase-1/day03d-pipeline.svg)
<figcaption>Animated: an eager pipeline stores every stage's full output; a generator pipeline moves one record through all the stages before reading the next.</figcaption>
</figure>

The eager version, with a list comprehension at each stage, gives the same batches but keeps a full copy of the data between every pair of stages. For six reviews that doesn't matter. For a 20 GB crawl of app reviews, the lazy version runs on a laptop and the eager one doesn't.

!!! warning "Common mistake: returning a generator from inside `with`"
    This looks like a tidy one-liner version of `read_lines`, but the file is already closed when anyone reads from it:

    ```python
    def read_lines_bug(path):
        with open(path, encoding="utf-8") as f:
            return (line for line in f)     # bug: the with block ends at this return

    lines = read_lines_bug(reviews_path)
    try:
        next(lines)
    except ValueError as e:
        print("ValueError:", e)
    # ValueError: I/O operation on closed file.
    ```

    `read_lines_bug` is an ordinary function (it has no `yield` of its own). It creates the generator expression, then `return` leaves the `with` block, which closes the file. The generator only tries to read later, from a closed file. **Fix:** put the `yield` *inside* the function, inside the `with`, as `read_lines` does. Then the function itself is a generator, and the `with` stays open while it's paused.

!!! warning "Common mistake: expecting errors where the pipeline is built"
    Laziness moves errors to the moment a value is pulled. Here the `clean` stage would fail on a record with no `"text"` key, but building the pipeline succeeds; only the loop fails, on the bad record, after the good ones have already gone through:

    ```python
    records = [{"text": "fine", "stars": 4}, {"stars": 2}]     # second record: no "text"
    batches = batch(clean(records), 1)                          # no error yet
    try:
        for b in batches:
            print("trained on", b)
    except KeyError as e:
        print("KeyError:", e)
    # trained on [('fine', 4)]
    # KeyError: 'text'
    ```

    The traceback points at the `for` loop and then into `clean`, not at the line that built `batches`. When a lazy pipeline fails, read the traceback bottom-up (Day 3b) to find the stage that raised. Validate records in an early stage, like `parse` does with bad JSON, so that problems surface close to where they come from.

---

**Part 4: `itertools` and iterator consumers**

## The `itertools` essentials

The standard library's `itertools` module is a toolbox of fast, lazy building blocks. Every function in it takes iterables and returns an **iterator** (one-shot). Each item below gets what it is, the Kotlin equivalent, and an example. `groupby` and `batched`, the two you'll use most in data code, get the full treatment.

```python
from itertools import (accumulate, batched, chain, count, cycle, groupby,
                       islice, pairwise, product, repeat, takewhile, dropwhile, zip_longest)
```

### `islice`: take a slice of any iterator

**What it is.** `islice(it, stop)` yields the first `stop` items; `islice(it, start, stop)` and `islice(it, start, stop, step)` work like list slicing, but on any iterator, lazily.

**Why it exists.** You can't write `gen[:5]` (a generator isn't subscriptable, Part 2). `islice` is how you take "the first 5 batches" of a loader, "every 10th line" of a file, or a finite piece of an infinite stream.

**Kotlin comparison.** `seq.take(5)`, `seq.drop(2).take(3)`, and so on, on a `Sequence`.

```python
list(islice(countdown(10), 3))         # [10, 9, 8]
list(islice("abcdefgh", 2, 6))         # ['c', 'd', 'e', 'f']
list(islice(range(20), 0, 20, 5))      # [0, 5, 10, 15]
```

**Underneath.** `islice` calls `next()` on its input only as often as it must: `islice(it, 3)` pulls three items and stops, leaving the rest of `it` unread. Skipping to `start` still pulls and throws away the skipped items, because an iterator can only move forward one step at a time.

**Goes wrong.** An iterator doesn't know its length, so negative indices are impossible:

```python
try:
    list(islice(countdown(5), -2, None))
except ValueError as e:
    print("ValueError:", e)
# ValueError: Indices for islice() must be None or an integer: 0 <= x <= sys.maxsize.
```

If you need "the last 2 items", you must walk the whole iterator anyway: use `list(it)[-2:]`, or `collections.deque(it, maxlen=2)` (Day 2b), which keeps only the most recent 2 items as it goes.

### `count`, `cycle` and `repeat`: infinite iterators

`count(start=0, step=1)` counts forever; `cycle(iterable)` repeats an iterable's items forever; `repeat(x, n)` gives `x` `n` times (or forever without `n`). Kotlin's nearest relatives are `generateSequence(0) { it + 1 }` and `generateSequence { x }`.

```python
list(islice(count(1), 4))              # [1, 2, 3, 4]
list(islice(count(0, 0.25), 4))        # [0, 0.25, 0.5, 0.75]
list(islice(cycle(["train", "val"]), 5))   # ['train', 'val', 'train', 'val', 'train']
list(repeat("<pad>", 3))               # ['<pad>', '<pad>', '<pad>']
```

An infinite iterator is fine as long as something stops it: `islice`, `zip` with a finite partner, `takewhile`, or a `break`. `list(count())` never finishes and eats memory until the program is killed. A common use of `count` is numbering a stream that `enumerate` would also number, but with a start or step of your choice; `zip(count(100), names)` labels items 100, 101, 102 and so on:

```python
list(zip(count(100), ["cat", "dog"]))      # [(100, 'cat'), (101, 'dog')]
```

One thing to know about `cycle`: to repeat its input, it **saves a copy of every item** during the first pass and replays the copies afterwards. That costs memory for a long input, and it changes behaviour with a loader that reshuffles (Part 5).

### `chain`: one iterator after another

`chain(a, b, c)` yields everything from `a`, then from `b`, then from `c`. `chain.from_iterable(list_of_lists)` does the same for an iterable of iterables, which flattens one level. Kotlin: `seqA + seqB`, and `flatten()`.

```python
train_files = ["a.jpg", "b.jpg"]
extra_files = ["c.jpg"]
list(chain(train_files, extra_files))              # ['a.jpg', 'b.jpg', 'c.jpg']
list(chain.from_iterable([[1, 2], [3], [4, 5]]))   # [1, 2, 3, 4, 5]
```

Unlike `train_files + extra_files`, `chain` builds no new list, and works on any iterables, including generators and files.

**Peeking with `chain`.** Sometimes you must look at a stream's first item before processing it, for example to check that the records have the fields you expect. `next` takes it out; `chain` puts it back in front:

```python
records = (r for r in [{"text": "fine", "stars": 4}, {"text": "slow", "stars": 2}])
first = next(records)
sorted(first)                              # ['stars', 'text']   check the schema
records = chain([first], records)          # the same stream, with first back in front
[r["stars"] for r in records]              # [4, 2]   nothing lost
```

### `product`: nested loops, and hyperparameter grids

`product(a, b)` yields every pair `(x, y)` with `x` from `a` and `y` from `b`, in the order of two nested `for` loops with `a` outer. With more arguments, it gives every combination. Kotlin has no built-in; you'd write `a.flatMap { x -> b.map { y -> x to y } }`.

**Why it matters in ML.** A **grid search** tries every combination of hyperparameters (settings you choose before training, like the learning rate):

```python
grid = {"lr": [0.1, 0.01], "batch_size": [16, 32], "dropout": [0.0]}
for values in product(*grid.values()):
    config = dict(zip(grid.keys(), values))
    print(config)
# {'lr': 0.1, 'batch_size': 16, 'dropout': 0.0}
# {'lr': 0.1, 'batch_size': 32, 'dropout': 0.0}
# {'lr': 0.01, 'batch_size': 16, 'dropout': 0.0}
# {'lr': 0.01, 'batch_size': 32, 'dropout': 0.0}
```

Line by line: `grid.values()` gives the three lists; `*` spreads them into three arguments (Day 3), so this is `product([0.1, 0.01], [16, 32], [0.0])`. Each `values` is a tuple like `(0.1, 16, 0.0)`; `zip(grid.keys(), values)` pairs each name with its value, and `dict(...)` turns the pairs into a config. The last list varies fastest, as in nested loops. The number of runs is the product of the list lengths (2 × 2 × 1 = 4), which grows fast: five settings with five values each is 3,125 training runs.

### `accumulate`: running totals

`accumulate(xs)` yields running sums: `xs[0]`, `xs[0] + xs[1]`, and so on. Give it a two-argument function to use instead of `+`, such as `max` for "best so far". `initial=` puts a start value in front. Kotlin: `runningReduce { a, b -> a + b }` and `runningFold(initial) { ... }`.

```python
list(accumulate([3, 1, 4, 1, 5]))              # [3, 4, 8, 9, 14]
val_acc = [0.61, 0.68, 0.66, 0.71, 0.70]
list(accumulate(val_acc, max))                 # [0.61, 0.68, 0.68, 0.71, 0.71]   best so far
list(accumulate([1, 2, 3], initial=10))        # [10, 11, 13, 16]
```

The "best so far" line is the check behind **early stopping**: if the best validation accuracy hasn't improved for a few epochs, stop training.

### `pairwise` and `zip_longest`

`pairwise(xs)` (Python 3.10+) yields overlapping pairs of neighbours, like Kotlin's `zipWithNext()`. It's the natural way to compute differences between consecutive values:

```python
losses = [0.92, 0.61, 0.48, 0.47]
list(pairwise(losses))                                # [(0.92, 0.61), (0.61, 0.48), (0.48, 0.47)]
[round(b - a, 2) for a, b in pairwise(losses)]        # [-0.31, -0.13, -0.01]   improvement per epoch
```

`zip_longest(a, b, fillvalue=x)` is `zip` that keeps going until the *longest* input ends, filling the gaps with `x` (default `None`). It's the third choice next to Day 2's plain `zip` (stops at the shortest) and `zip(strict=True)` (raises):

```python
list(zip_longest(["a", "b", "c"], [1], fillvalue=0))  # [('a', 1), ('b', 0), ('c', 0)]
```

### `takewhile` and `dropwhile`

`takewhile(pred, it)` yields items while `pred(item)` is true and stops at the first false one; `dropwhile(pred, it)` skips items while it's true and yields everything after. Kotlin has the same names on sequences. Unlike `filter`, they look only at the *front* of the stream:

```python
warmup_lrs = [0.001, 0.002, 0.004, 0.008, 0.004, 0.002]
list(takewhile(lambda lr: lr < 0.005, warmup_lrs))   # [0.001, 0.002, 0.004]
list(dropwhile(lambda lr: lr < 0.005, warmup_lrs))   # [0.008, 0.004, 0.002]
list(takewhile(lambda n: n < 4, count()))            # [0, 1, 2, 3]   stops an infinite iterator
```

`filter(lambda lr: lr < 0.005, warmup_lrs)` would also keep the last two values; `takewhile` stops at `0.008` and never looks further, which is what makes it safe on an infinite `count()`.

### `batched`: fixed-size chunks

**What it is.** `batched(iterable, n)` (Python 3.12+) yields **tuples** of `n` items, with a shorter last tuple if the items don't divide evenly.

**Why it exists.** Mini-batches. It's the library version of Part 3's `batch` generator, and the core of every data loader.

**Kotlin comparison.** `chunked(n)` on a `List` or `Sequence`, which gives `List`s and also leaves the last chunk short.

```python
list(batched(range(7), 3))             # [(0, 1, 2), (3, 4, 5), (6,)]
for b in batched(["a.jpg", "b.jpg", "c.jpg", "d.jpg"], 2):
    print(b)
# ('a.jpg', 'b.jpg')
# ('c.jpg', 'd.jpg')
```

**Underneath.** Each round, `batched` pulls up to `n` items from its input, roughly `tuple(islice(it, n))`, and stops when that comes back empty. It holds one batch at a time, so it works on files and infinite streams.

**Goes wrong.** A short last batch can break code that assumes a fixed batch size, such as reshaping into `(batch, ...)` for a model with a fixed input size. Python 3.13 added `strict=True`, which raises instead of yielding the short batch:

```python
try:
    list(batched(range(7), 3, strict=True))
except ValueError as e:
    print("ValueError:", e)
# ValueError: batched(): incomplete batch
```

On Python 3.12, `strict` doesn't exist (`TypeError`). The ML habit is different anyway: either accept the short last batch, or drop it, which PyTorch calls `drop_last=True` (Part 5).

### `groupby`: grouping consecutive items

**What it is.** `groupby(iterable, key=f)` walks the items and yields `(key, group)` pairs, one for each **run of consecutive items** with the same `f(item)`. Each `group` is itself an iterator over that run.

**Why it exists.** Grouping sorted records ("all reviews per app", "all frames per video", "all log lines per request id") in one lazy pass, without building a dict of lists. It's ideal for a file that's already sorted by the key.

**Kotlin comparison.** This is the trap. Kotlin's `groupBy { }` collects **all** items with the same key into one `Map` entry, wherever they are in the list. Python's `itertools.groupby` only groups **neighbours**. Its real equivalent in your toolkit is Day 2b's `defaultdict(list)` loop, not `groupby`.

**Example.** Ratings by category, already sorted by category:

```python
apps = [("GAME", "Chess", 4.5), ("GAME", "Go", 4.1),
        ("TOOLS", "Files", 4.6), ("TOOLS", "Clock", 4.0), ("TOOLS", "Torch", 3.9)]
for category, rows in groupby(apps, key=lambda a: a[0]):
    ratings = [r[2] for r in rows]
    print(category, ratings, round(sum(ratings) / len(ratings), 2))
# GAME [4.5, 4.1] 4.3
# TOOLS [4.6, 4.0, 3.9] 4.17
```

**Underneath.** `groupby` keeps one "current key". It pulls items one by one; while `key(item)` equals the current key, the item belongs to the current group. When the key changes, it starts a new group. It never looks back, and it never looks ahead past the next item, which is why it can work on a stream.

<figure class="diagram" markdown>
![Two rows of coloured boxes. Top row, unsorted input GAME, TOOLS, GAME: groupby draws a bracket around each run of equal neighbours, producing three groups, GAME, TOOLS and a second GAME, with the second GAME marked in red as a duplicate key. Bottom row, the same data after sorted(key=...): GAME, GAME, TOOLS, producing two groups, GAME with two items and TOOLS with one. A note says Kotlin's groupBy would give the bottom answer for either input.](../../assets/img/phase-1/day03d-groupby.svg)
<figcaption>groupby brackets runs of equal neighbours, so unsorted input gives the same key twice; sort by the same key first.</figcaption>
</figure>

!!! warning "Common mistake: `groupby` on unsorted data"
    With the same categories in a different order, `GAME` comes out twice:

    ```python
    unsorted_apps = [("GAME", "Chess", 4.5), ("TOOLS", "Files", 4.6), ("GAME", "Go", 4.1)]
    [(k, len(list(g))) for k, g in groupby(unsorted_apps, key=lambda a: a[0])]
    # [('GAME', 1), ('TOOLS', 1), ('GAME', 1)]
    ```

    If you then build a dict from these groups, the second `GAME` overwrites the first, and half the games vanish silently. **Fix:** sort by the *same* key first:

    ```python
    by_category = lambda a: a[0]
    [(k, len(list(g))) for k, g in groupby(sorted(unsorted_apps, key=by_category), key=by_category)]
    # [('GAME', 2), ('TOOLS', 1)]
    ```

    (Sorting needs all the data in memory. If the data isn't sorted and doesn't need to be streamed, a `defaultdict(list)` loop or a `Counter` from Day 2b is simpler.)

!!! warning "Common mistake: keeping the groups for later"
    Each group is a view into the *shared* underlying iterator. When `groupby` moves on to the next key, the previous group is used up. So collecting the pairs first and reading the groups afterwards gives empty groups:

    ```python
    pairs = list(groupby(apps, key=lambda a: a[0]))
    [(k, list(g)) for k, g in pairs]         # [('GAME', []), ('TOOLS', [])]
    ```

    **Fix:** consume each group inside the loop, as the example above did, or turn it into a list straight away: `{k: list(g) for k, g in groupby(...)}`.

## Iterator consumers: who reads how much

**What it is.** A **consumer** is anything that pulls items out of an iterator. Some pull everything, some stop early, and some are themselves lazy wrappers that pull nothing until *they* are pulled. With one-shot iterators, "how much did that consume?" decides what's left for the next line.

**Why it matters.** Most of the surprises with iterators come from a consumer that read more, or less, than you expected.

**Kotlin comparison.** Kotlin splits `Sequence` operations into **intermediate** ones (`map`, `filter`, `take`: lazy, return another `Sequence`) and **terminal** ones (`toList`, `sum`, `first`, `any`: pull values). Python has the same split without the names: `map`, `filter`, `zip`, `enumerate`, generator expressions and the `itertools` functions are intermediate; `list`, `sum`, `max`, `sorted`, `any`, `next` and a `for` loop are terminal.

**Example: the lazy wrappers.** `enumerate` and `zip` don't pull anything when called; they pull one item from each input per item they produce:

```python
def labels_gen():
    for name in ["cat", "dog"]:
        print(f"  producing {name}")
        yield name

e = enumerate(labels_gen())   # nothing produced yet
next(e)
#   producing cat
# (0, 'cat')
```

**`zip(strict=True)` raises late.** Day 2's `strict=True` turns a length mismatch into a `ValueError`, but with iterators it can only find out when one input runs dry, which is **after** it has handed out all the pairs it could make:

```python
pairs = zip(["a.jpg", "b.jpg", "c.jpg"], labels_gen(), strict=True)
try:
    for image, label in pairs:
        print("train on", image, label)
except ValueError as e:
    print("ValueError:", e)
#   producing cat
# train on a.jpg cat
#   producing dog
# train on b.jpg dog
# ValueError: zip() argument 2 is shorter than argument 1
```

Two training steps ran before the error. With lists, check `len(a) == len(b)` before you start; with streams, `strict=True` is still worth it, because a late error beats silently dropping data.

**Terminal consumers that stop early.** `any` stops at the first true item, `all` at the first false one, `next` after one item, and `in` at the first match. What they didn't read is still in the iterator:

```python
nums = iter([1, 5, 2, 8, 3])
any(n > 4 for n in nums)      # True    stopped at 5
list(nums)                    # [2, 8, 3]
```

**Unpacking.** `a, b = it` pulls one item per name, then pulls one more to check that the iterator is empty. If there is one, that extra item is consumed and you get `ValueError`. `first, *rest = it` (Day 2b) pulls everything:

```python
first, *rest = countdown(4)
first, rest                   # (4, [3, 2, 1])
try:
    a, b = countdown(3)
except ValueError as e:
    print("ValueError:", e)
# ValueError: too many values to unpack (expected 2)

it = iter([1, 2, 3, 4])
try:
    a, b = it
except ValueError:
    pass
list(it)                      # [4]   1 and 2 went to a and b; 3 was pulled by the check and lost
```

**Terminal consumers that read everything.** `list`, `tuple`, `set`, `dict`, `sum`, `max`, `min`, `"".join` and `sorted` read to the end. `sorted` is worth singling out: it accepts any iterable, but it must see every item before it can return the first, so it **builds a full list**. `sorted(huge_generator)` uses the same memory as `list(huge_generator)`, and `sorted(count())` never returns. `max` and `min` read everything too, but keep only the best item so far.

| Consumer | How much it pulls | What it returns |
|---|---|---|
| `map`, `filter`, `zip`, `enumerate`, `islice`, generator expressions | nothing until pulled; then one item per item produced | a lazy iterator |
| `next(it)` | one item | that item |
| `any`, `all`, `x in it` | up to the first item that decides the answer | `True` / `False` |
| `a, b = it` | two, plus one more to check (consumed if present) | — |
| `list`, `tuple`, `set`, `dict`, `sum`, `max`, `min`, `"".join` | everything | a value or a container |
| `sorted(it)` | everything, and keeps it all | a new list |

---

**Part 5: Iterators in ML code**

## A mini `DataLoader`

**What it is.** A **data loader** is a reusable iterable over a dataset. Each time you loop over it, which is once per **epoch** (one full pass over the training data), it yields **mini-batches**: small groups of samples that the model trains on in one step. When asked, it shuffles the order every epoch.

**Why it exists.** Training on one sample at a time is slow and noisy; training on the whole dataset at once doesn't fit in memory. Mini-batches of 16 to 512 samples are the standard compromise. Shuffling every epoch stops the model from learning the order of the file, and stops batches from being the same every time. PyTorch's `torch.utils.data.DataLoader` (Month 4) does exactly this, with extras such as parallel loading.

**Kotlin comparison.** In Kotlin you'd write a class implementing `Iterable<Batch>` whose `iterator()` returns `iterator { ... yield(batch) ... }`, building a new iterator, and therefore a new shuffle, for each `for` loop. Python's version is a class whose `__iter__` is a generator method: Part 2's "Fix 3".

**Example.** The dataset is Day 3a's `ReviewDataset` (`__len__` and `__getitem__`). `random.Random(seed)` creates a private random-number generator started from a fixed **seed**, so every run shuffles the same way; its `shuffle(list)` method reorders a list in place. (Day 4b covers NumPy's random generators.) `math.ceil(x)` rounds up to the next whole number:

```python
import math
import random

class ReviewDataset:
    def __init__(self, texts: list[str], labels: list[int]):
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]

class DataLoader:
    def __init__(self, dataset, batch_size: int, shuffle: bool = False,
                 drop_last: bool = False, seed: int | None = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        """Number of batches per epoch."""
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)

    def __iter__(self):
        order = list(range(len(self.dataset)))
        if self.shuffle:
            self.rng.shuffle(order)               # a new order on every epoch
        for idx in batched(order, self.batch_size):
            if self.drop_last and len(idx) < self.batch_size:
                return                            # skip the short last batch
            samples = [self.dataset[i] for i in idx]
            texts, labels = zip(*samples)         # [(t, l), (t, l)] -> (t, t), (l, l)
            yield list(texts), list(labels)

ds = ReviewDataset(["great app", "crashes a lot", "love it", "too many ads", "works offline"],
                   [1, 0, 1, 0, 1])
loader = DataLoader(ds, batch_size=2, shuffle=True, seed=0)
len(loader)                                       # 3
for epoch in range(2):
    print("epoch", epoch)
    for texts, labels in loader:
        print("  ", labels, texts)
# epoch 0
#    [1, 0] ['love it', 'crashes a lot']
#    [1, 1] ['great app', 'works offline']
#    [0] ['too many ads']
# epoch 1
#    [1, 1] ['great app', 'love it']
#    [0, 0] ['crashes a lot', 'too many ads']
#    [1] ['works offline']
```

**How it works underneath, step by step.**

1. `for texts, labels in loader:` calls `iter(loader)`, which calls `loader.__iter__()`. Because `__iter__` contains `yield`, that call returns a **new generator**; nothing has run yet.
2. The first `next()` runs `__iter__`'s body: it builds `order = [0, 1, 2, 3, 4]`, shuffles it with the loader's own random generator, and starts the `for` over `batched(order, 2)`.
3. With this seed, the shuffled order is `[2, 1, 0, 4, 3]`, so the first batch of indexes is `(2, 1)`. `[self.dataset[i] for i in idx]` calls `ds.__getitem__` twice, giving `[('love it', 1), ('crashes a lot', 0)]`.
4. `zip(*samples)` is Day 2b's "unzip": `*` spreads the two pairs into two arguments, and `zip` pairs up their first items and their second items, giving `('love it', 'crashes a lot')` and `(1, 0)`. The generator yields them as two lists, and pauses.
5. After three batches, the body finishes and the epoch's `for` loop ends. The second epoch calls `iter(loader)` again: a new generator, a new `order`, a new shuffle. `self.rng` is the same generator object, which has moved on, so the order is different, yet the whole run is reproducible because it started from `seed=0`.

With `drop_last=True`, the short third batch is skipped and `len(loader)` is `5 // 2 = 2`:

```python
strict_loader = DataLoader(ds, batch_size=2, drop_last=True)
len(strict_loader)                                # 2
[labels for _, labels in strict_loader]           # [[1, 0], [1, 0]]
```

`_` is the conventional name for a value you don't use. Without `shuffle`, the order is the dataset's own: indexes 0 and 1, then 2 and 3.

!!! warning "Common mistake: creating the random generator inside `__iter__`"
    For reproducibility it's tempting to seed inside `__iter__`. Then every epoch starts a new generator from the same seed, and every epoch gets the same order:

    ```python
    class SameOrderLoader(DataLoader):
        def __iter__(self):
            rng = random.Random(0)           # bug: a fresh generator, same seed, every epoch
            order = list(range(len(self.dataset)))
            rng.shuffle(order)
            for idx in batched(order, self.batch_size):
                yield [self.dataset[i][1] for i in idx]   # labels only, to keep the output short

    same = SameOrderLoader(ds, batch_size=2)
    [list(same) for epoch in range(2)]       # [[[1, 0], [1, 1], [0]], [[1, 0], [1, 1], [0]]]
    ```

    Nothing fails, and both epochs show `[1, 0] [1, 1] [0]`: "shuffled", but identically. The same happens with `random.seed(0)` inside the loop. **Fix:** create the generator once, in `__init__`, and keep it on `self`, as `DataLoader` does. It's seeded once, so the whole run is reproducible, and each `shuffle` call moves it on, so each epoch differs.

The same structure in PyTorch, for comparison (Month 4 explains every argument):

```python
# Run this on your computer (or in Colab), not in the browser.
from torch.utils.data import DataLoader

class ReviewDataset:                 # the same class as above: __len__ and __getitem__
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, i):
        return self.texts[i], self.labels[i]

ds = ReviewDataset(["great app", "crashes a lot", "love it", "too many ads", "works offline"],
                   [1, 0, 1, 0, 1])
loader = DataLoader(ds, batch_size=2, shuffle=True, drop_last=False)
for epoch in range(2):
    for texts, labels in loader:     # a new shuffled iterator every epoch
        print(epoch, labels, texts)  # labels arrives as a tensor, texts as a sequence of strings
```

!!! android "Android bridge"
    You've used one-shot, forward-only iterators on Android for years. A SQLite `Cursor` is one: `while (cursor.moveToNext()) { ... }` walks rows forward, `moveToNext()` returning `false` plays the role of `StopIteration`, and a `Cursor` must be closed (`cursor.use { }`), like a generator holding an open file. The Paging 3 library's `PagingSource.load()` is the data-loader idea for UI: it fetches one page of items when the list scrolls near the end, instead of loading the whole table. And Kotlin's `Flow` is a *cold* stream like a generator: its `flow { emit(x) }` block runs only when collected, and runs again for each collector. The difference is that `Flow` is asynchronous, built on suspending functions, while Python's plain generators are synchronous; Python's asynchronous cousin is the async generator (Going further).

## An infinite training stream

**What it is.** Many training scripts count **steps** (mini-batches) rather than epochs: "train for 10,000 steps, evaluate every 500". For that you want an endless stream of batches that moves on to a new, reshuffled epoch by itself.

**Why it exists.** It decouples the training loop from the dataset size: the loop asks for the next batch and never sees epoch boundaries. Large-scale training (and LLM pre-training) is almost always written this way.

**Example.** A generator with `while True` and `yield from`. Each pass of `yield from loader` calls `iter(loader)`, so each epoch reshuffles:

```python
def forever(loader):
    while True:
        yield from loader             # a fresh, reshuffled epoch each time round

loader = DataLoader(ds, batch_size=2, shuffle=True, seed=0)
stream = forever(loader)
for step, (texts, labels) in enumerate(islice(stream, 5), start=1):
    print(f"step {step}: {labels}")
# step 1: [1, 0]
# step 2: [1, 1]
# step 3: [0]
# step 4: [1, 1]
# step 5: [0, 0]
```

Line by line: `islice(stream, 5)` stops the infinite stream after five batches; `enumerate(..., start=1)` numbers them from 1; `(texts, labels)` unpacks each batch inside the `for` (Day 2b's nested unpacking). Steps 1 to 3 are epoch 0 (the third batch is the short one), and steps 4 and 5 start epoch 1, with the same orders as the loop above, because the loader was rebuilt from `seed=0`.

!!! warning "Common mistake: `cycle(loader)` repeats the first epoch forever"
    `itertools.cycle` looks like the same thing, but Part 4 said how it works: it saves every item from its *first* pass and replays the saved copies. It never calls `iter(loader)` again, so the shuffle never changes:

    ```python
    loader = DataLoader(ds, batch_size=2, shuffle=True, seed=0)
    [labels for _, labels in islice(cycle(loader), 6)]
    # [[1, 0], [1, 1], [0], [1, 0], [1, 1], [0]]
    ```

    Epoch 2 (the last three batches) is identical to epoch 1. Nothing crashes; the model sees the same batches in the same order every epoch, and also, `cycle` keeps a copy of the whole first epoch in memory. **Fix:** use the `forever` generator above, which asks the loader for a new iterator each time.

**A streaming dataset.** When a dataset is too big for `__getitem__` (a folder of JSON Lines files larger than memory, or a feed from a server), the dataset itself becomes a generator over records. PyTorch calls this an `IterableDataset`. Part 3's pipeline already is one:

```python
stream = batch(clean(parse(read_lines(reviews_path))), 3)
next(stream)                    # [('great editor', 5), ('crashes on start', 1), ('fast and clean', 4)]
```

A stream can't be shuffled with `order = list(range(len(...)))`, because it has no length and no indexes. Instead it's shuffled **approximately, within a buffer**: keep `size` items in a list; each time a new item arrives, hand out a randomly chosen item from the buffer and keep the new one in its place. Memory is `size` items. An item can come out at most `size - 1` places earlier than its position in the stream (though it can wait in the buffer much longer), so a bigger buffer mixes better. `rng.randrange(n)` returns a random whole number from 0 to `n - 1`:

```python
def shuffle_buffer(items, size, rng):
    buffer = []
    for item in items:
        buffer.append(item)
        if len(buffer) == size:
            i = rng.randrange(size)                    # pick a random slot
            buffer[i], buffer[-1] = buffer[-1], buffer[i]
            yield buffer.pop()                         # hand out what was in it
    rng.shuffle(buffer)                                # the stream ended: empty the buffer
    yield from buffer

list(shuffle_buffer(range(10), 3, random.Random(0)))   # [1, 2, 0, 3, 6, 5, 7, 8, 4, 9]
```

The swap moves the chosen item to the end, so `pop()` (Day 2b) removes it cheaply. `IterableDataset` doesn't do this for you: you write a generator like this one, or a library built on PyTorch, such as WebDataset, provides it.

## Kotlin to Python: a summary

Now that each piece has been taught, here is the mapping in one place:

| Kotlin | Python | Note |
|---|---|---|
| `Iterable<T>` / `iterator()` | an object with `__iter__` / `iter(x)` | same role |
| `Iterator<T>`: `hasNext()`, `next()` | `__next__` / `next(it)`, raising `StopIteration` | no `hasNext` in Python |
| `for (x in xs)` | `for x in xs:` | both desugar to iterator calls |
| `sequence { yield(x) }` | a generator function with `yield x` | Kotlin's re-runs per pass; Python's object is one-shot |
| `yieldAll(other)` | `yield from other` | `yield from` also passes back a return value |
| `list.asSequence().map { }.filter { }` | generator expression, `map`, `filter` | lazy until a terminal operation |
| `iterator.asSequence()` (constrained once) | any iterator | Kotlin throws on a 2nd pass; Python gives nothing |
| `take(n)`, `drop(n)` | `islice(it, n)`, `islice(it, n, None)` | |
| `generateSequence(0) { it + 1 }` | `count()` | infinite |
| `chunked(n)` | `batched(it, n)` | Python yields tuples |
| `zipWithNext()` | `pairwise(it)` | |
| `runningReduce`, `runningFold` | `accumulate(it, f, initial=...)` | |
| `takeWhile`, `dropWhile` | `takewhile`, `dropwhile` | |
| `groupBy { }` (collects all) | `defaultdict(list)` loop | `itertools.groupby` only groups neighbours |
| `File.useLines { }` | `with open(...) as f: for line in f:` | |
| `Flow` (`flow { emit() }`) | async generator (`async def` + `yield`) | asynchronous; not covered here |

## Going further

These are real and worth knowing exist, but aren't needed yet:

- **`send()` and `throw()`**: a generator can receive values at its `yield` (`x = yield y`), which is how generator-based coroutines worked before `async`: [yield expressions reference](https://docs.python.org/3/reference/expressions.html#yield-expressions) and [PEP 342](https://peps.python.org/pep-0342/).
- **Async generators and `async for`**, Python's counterpart of Kotlin's `Flow`: [PEP 525](https://peps.python.org/pep-0525/).
- **`iter(callable, sentinel)`**, the two-argument form, which calls a function until it returns the sentinel (handy for reading fixed-size chunks): [built-in `iter`](https://docs.python.org/3/library/functions.html#iter).
- **The rest of `itertools`**: `combinations`, `permutations`, `starmap`, `compress`, `filterfalse`, and the recipes at the end of the page: [itertools docs](https://docs.python.org/3/library/itertools.html).
- **`more-itertools`**, a third-party package with dozens more building blocks (`chunked`, `windowed`, `peekable`): [more-itertools docs](https://more-itertools.readthedocs.io/).
- **`functools.reduce`**, which folds an iterable into one value (Kotlin's `reduce`): [functools docs](https://docs.python.org/3/library/functools.html#functools.reduce).
- **`__reversed__`, `__contains__` and `__length_hint__`**, the dunders that make `reversed`, `in` and pre-sizing work on your classes: [Data model: emulating container types](https://docs.python.org/3/reference/datamodel.html#emulating-container-types).
- **The history and design** of generators and `yield from`: [PEP 255](https://peps.python.org/pep-0255/), [PEP 380](https://peps.python.org/pep-0380/), and why `StopIteration` inside a generator became `RuntimeError`, [PEP 479](https://peps.python.org/pep-0479/).
- **`ValueError: generator already executing`**: a generator can't be resumed while it's already running, for example from inside itself or from two threads at once, which is one reason PyTorch's `DataLoader` workers each get their own iterator: [generator-iterator methods](https://docs.python.org/3/reference/expressions.html#generator-iterator-methods).
- **PyTorch's `Dataset`, `IterableDataset` and `DataLoader`**, in full: Month 4, and [torch.utils.data](https://pytorch.org/docs/stable/data.html).

---

## Exercises

**Warm-ups**

1. **Drive a loop by hand.** Without a `for` loop, use `iter`, `next` and a `while True` with `try` / `except StopIteration` to print every character of `"GPU"` on its own line. Then use `next` with a default to get the first item of an empty list without an exception.
2. **Your first generator.** Write a generator function `evens(limit)` that yields the even numbers from 0 up to, but not including, `limit`. Print `list(evens(10))`, then get the first three even numbers of an *endless* version, `evens_forever()`, with `islice`.

**Core**

3. **Write `batched` yourself.** Write a generator `my_batched(iterable, n)` that yields lists of `n` items, with a shorter last list, and works on any iterable, including a generator and an infinite one. Use `iter()` once and `islice` to take each batch. Check it on `range(7)` and on `count()` (first three batches).
4. **A lazy log summary.** Write a JSON Lines file of request logs into a temporary folder, each line like `{"path": "/predict", "ms": 120}`, with one blank line and one broken line among them. Using a pipeline of generators (read → parse → filter), and without ever building a list of all records, find the number of `/predict` requests and their average time in milliseconds.
5. **Group, then summarise.** Given a list of `(app, stars)` review tuples in no particular order, use `sorted` and `groupby` to print each app's review count and mean stars, in alphabetical order. Then show what you'd have got without `sorted`.

**Predict the output**

6. Predict every line this prints, then run it:

    ```python
    def gen():
        print("A")
        yield 1
        print("B")
        yield 2
        print("C")

    g = gen()
    print("start")
    print(next(g))
    print(list(g))
    print(list(g))
    print(sum(x for x in [1, 2, 3] if x > 1))
    z = zip([1, 2, 3], "ab")
    print(list(z), list(z))
    ```

**Find and fix the bug**

7. This function should report each value as a percentage of the total. It works when called with a list, but a colleague calls it with a generator that reads values from a file, and gets an empty list. Explain why, and fix it so it works for both.

    ```python
    def as_percentages(values):
        total = sum(values)
        return [round(100 * v / total, 1) for v in values]

    as_percentages([1, 1, 2])                      # [25.0, 25.0, 50.0]
    as_percentages(v for v in [1, 1, 2])           # []
    ```

**Challenge**

8. **Smoothing a loss curve.** Raw training losses jump around from batch to batch, so dashboards plot a **moving average**: the mean of the last `k` values. Write a generator `moving_average(values, k)` that yields, for each new value, the mean of the last `k` values seen so far (fewer at the start). It must work on an infinite stream, so it may keep only `k` values in memory. Use `collections.deque(maxlen=k)`, a list-like container that, when full, drops its oldest item each time you `append` a new one. Test it on `[4.0, 2.0, 3.0, 1.0, 2.0]` with `k=3`, and on the first 4 values of an infinite stream of losses `1/step` for `step` = 1, 2, 3, …, with `k=2`. For a bonus, keep a running sum so each step costs the same however large `k` is.

??? success "Solutions"
    **1. Drive a loop by hand.**

    ```python
    it = iter("GPU")
    while True:
        try:
            ch = next(it)
        except StopIteration:
            break
        print(ch)
    # G
    # P
    # U
    print(next(iter([]), None))
    # None
    ```

    `iter("GPU")` gives a string iterator; each `next` returns one character and the fourth raises `StopIteration`, which the `except` turns into `break`. That is exactly what `for ch in "GPU":` does for you. `next(iter([]), None)` asks for the first item of an empty list and gets the default instead of an exception.

    **2. Your first generator.**

    ```python
    from itertools import islice

    def evens(limit):
        n = 0
        while n < limit:
            yield n
            n += 2

    def evens_forever():
        n = 0
        while True:
            yield n
            n += 2

    print(list(evens(10)))
    print(list(islice(evens_forever(), 3)))
    # [0, 2, 4, 6, 8]
    # [0, 2, 4]
    ```

    Each `next()` resumes after the `yield`, adds 2, and pauses at the next `yield`. `evens_forever` never finishes on its own, which is safe because `islice` pulls only three items and then stops asking. (`list(evens_forever())` would never return.)

    **3. Write `batched` yourself.**

    ```python
    from itertools import count, islice

    def my_batched(iterable, n):
        it = iter(iterable)               # one shared iterator, so each islice continues
        while True:
            chunk = list(islice(it, n))   # take up to n items
            if not chunk:                 # nothing left: stop
                return
            yield chunk

    print(list(my_batched(range(7), 3)))
    print(list(islice(my_batched(count(), 2), 3)))
    print(list(my_batched((x * x for x in range(5)), 2)))
    # [[0, 1, 2], [3, 4, 5], [6]]
    # [[0, 1], [2, 3], [4, 5]]
    # [[0, 1], [4, 9], [16]]
    ```

    The key line is `it = iter(iterable)`. Every `islice(it, n)` pulls from that *same* iterator, so each batch starts where the previous one stopped. Without it, `islice(range(7), 3)` on the reusable `range` would start again at 0 each time, and the loop would yield `[0, 1, 2]` forever. An empty `chunk` is falsy (Day 2), so `if not chunk: return` ends the generator.

    **4. A lazy log summary.**

    ```python
    import json, tempfile
    from pathlib import Path

    folder = Path(tempfile.mkdtemp())
    log_path = folder / "requests.jsonl"
    log_path.write_text(
        '{"path": "/predict", "ms": 120}\n'
        '{"path": "/health", "ms": 2}\n'
        '\n'
        '{"path": "/predict", "ms": 80}\n'
        '{"path": "/predict", "ms": \n'
        '{"path": "/predict", "ms": 100}\n',
        encoding="utf-8",
    )                                     # 153   write_text returns the characters written

    def read_lines(path):
        with open(path, encoding="utf-8") as f:
            yield from f

    def parse(lines):
        for line in lines:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass

    predict_ms = (r["ms"] for r in parse(read_lines(log_path)) if r["path"] == "/predict")
    count_, total = 0, 0
    for ms in predict_ms:
        count_ += 1
        total += ms
    print(count_, total / count_)
    # 3 100.0
    ```

    `yield from f` hands out the file's lines one at a time (a file is an iterator). `parse` skips the blank line and the cut-off one, and the generator expression keeps only `/predict` times. The final loop keeps two numbers, never a list, so this works the same on a 50 GB log. The count and the sum come from one pass, because a second pass over `predict_ms` would find it empty. (The name `count_` has a trailing underscore so it doesn't hide `itertools.count`, a common convention.)

    **5. Group, then summarise.**

    ```python
    from itertools import groupby

    reviews = [("Keep", 4), ("Lens", 5), ("Keep", 2), ("Files", 5), ("Lens", 3), ("Keep", 3)]
    app_of = lambda r: r[0]
    for app, rows in groupby(sorted(reviews, key=app_of), key=app_of):
        stars = [s for _, s in rows]
        print(f"{app:<6}{len(stars)} reviews, mean {sum(stars) / len(stars):.2f}")
    # Files 1 reviews, mean 5.00
    # Keep  3 reviews, mean 3.00
    # Lens  2 reviews, mean 4.00
    print([(app, len(list(rows))) for app, rows in groupby(reviews, key=app_of)])
    # [('Keep', 1), ('Lens', 1), ('Keep', 1), ('Files', 1), ('Lens', 1), ('Keep', 1)]
    ```

    `sorted(..., key=app_of)` puts each app's reviews next to each other, alphabetically, so `groupby` sees one run per app. Each `rows` group is turned into a list straight away, inside the loop, before `groupby` moves on. Without sorting, no two neighbours share an app, so every review becomes its own group of one.

    **6. Predict the output.**

    ```python
    def gen():
        print("A")
        yield 1
        print("B")
        yield 2
        print("C")

    g = gen()
    print("start")
    print(next(g))
    print(list(g))
    print(list(g))
    print(sum(x for x in [1, 2, 3] if x > 1))
    z = zip([1, 2, 3], "ab")
    print(list(z), list(z))
    # start
    # A
    # 1
    # B
    # C
    # [2]
    # []
    # 5
    # [(1, 'a'), (2, 'b')] []
    ```

    `g = gen()` runs nothing, so `start` comes first. `next(g)` runs up to the first `yield`, printing `A`, and returns 1. `list(g)` resumes: `B`, yields 2, then `C` and the body ends; `list` collects `[2]`, and `print` shows it after both letters, because `list` finishes before `print` runs. The second `list(g)` finds a closed generator: `[]`. The generator expression keeps 2 and 3, so `sum` is 5. The first `list(z)` uses up the `zip` iterator, stopping at the shorter `"ab"`; the second gets nothing.

    **7. Find and fix the bug.**

    ```python
    def as_percentages(values):
        values = list(values)          # materialise once: works for lists and generators
        total = sum(values)
        return [round(100 * v / total, 1) for v in values]

    print(as_percentages([1, 1, 2]))
    print(as_percentages(v for v in [1, 1, 2]))
    # [25.0, 25.0, 50.0]
    # [25.0, 25.0, 50.0]
    ```

    The function reads its input twice: `sum` walks the generator to the end, and the comprehension then finds it empty, so it returns `[]` without any error. `list(values)` at the top makes one copy that can be read any number of times. If the data were too big for memory, the alternative would be to take a function that *creates* the generator, and call it once per pass.

    **8. Smoothing a loss curve.**

    ```python
    from collections import deque
    from itertools import count, islice

    def moving_average_simple(values, k):
        window = deque(maxlen=k)          # keeps only the last k values
        for v in values:
            window.append(v)              # when full, the oldest value drops out
            yield sum(window) / len(window)

    print([round(m, 3) for m in moving_average_simple([4.0, 2.0, 3.0, 1.0, 2.0], k=3)])
    # [4.0, 3.0, 3.0, 2.0, 2.0]

    def moving_average(values, k):        # the bonus: a running sum
        window = deque(maxlen=k)
        running = 0.0
        for v in values:
            if len(window) == k:
                running -= window[0]      # the oldest value is about to drop out
            window.append(v)
            running += v
            yield running / len(window)

    print([round(m, 3) for m in moving_average([4.0, 2.0, 3.0, 1.0, 2.0], k=3)])
    losses = (1 / step for step in count(1))
    print([round(m, 3) for m in islice(moving_average(losses, k=2), 4)])
    # [4.0, 3.0, 3.0, 2.0, 2.0]
    # [1.0, 0.75, 0.417, 0.292]
    ```

    For the first list: after 4.0 the mean is 4.0; after 2.0 it's 3.0; after 3.0, (4 + 2 + 3) / 3 = 3.0; then 4.0 drops out, (2 + 3 + 1) / 3 = 2.0; then 2.0 drops out, (3 + 1 + 2) / 3 = 2.0. `deque(maxlen=k)` throws the oldest value away on `append` once it holds `k`, so memory stays at `k` values on an endless stream. The simple version re-adds `k` numbers per step. The bonus version's running sum subtracts the value that's about to fall out (`window[0]`, the oldest) *before* appending, so each step does a fixed amount of work instead of re-adding `k` numbers. `islice(..., 4)` stops the infinite stream: (1 + 1/2) / 2 = 0.75, (1/2 + 1/3) / 2 ≈ 0.417, (1/3 + 1/4) / 2 ≈ 0.292, shown rounded to three places by `round`. One caution for real use: a float running sum picks up a tiny rounding error at each subtraction, which adds up over millions of steps, so long-running code recomputes `sum(window)` every so often.

## Resources

- [Python tutorial §9.8–9.10: Iterators, Generators, Generator Expressions](https://docs.python.org/3/tutorial/classes.html#iterators): the official short introduction.
- [Functional Programming HOWTO](https://docs.python.org/3/howto/functional.html): iterators, generators and `itertools`, explained at length with examples.
- [itertools documentation](https://docs.python.org/3/library/itertools.html): every function, with a pure-Python equivalent of each, and the recipes section.
- [Kotlin docs: Sequences](https://kotlinlang.org/docs/sequences.html): the Kotlin side of today's comparison, eager versus lazy.
- [Real Python: How to Use Generators and yield in Python](https://realpython.com/introduction-to-python-generators/): more worked examples, including reading large files.
- [PyTorch tutorial: Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html): skim it now to spot `__len__`, `__getitem__`, `batch_size` and `shuffle`.

## Checklist

- [ ] I can tell an iterable from an iterator, and drive one by hand with `iter`, `next` and `StopIteration`.
- [ ] I can rewrite a `for` loop as the `while` loop Python runs, and I know which built-ins return one-shot iterators and which return reusable iterables.
- [ ] I can write an iterator class with `__iter__` and `__next__`, and a reusable iterable that hands out a new iterator each time.
- [ ] I can write a generator function and trace when its body runs, pauses and resumes.
- [ ] I can spot the "generator used twice" bug and fix it three ways.
- [ ] I can choose between a generator expression and a list comprehension, and I've measured the memory difference.
- [ ] I can build a lazy read → parse → clean → batch pipeline, and use `yield from`.
- [ ] I can use `islice`, `count`, `cycle`, `repeat`, `chain`, `product`, `accumulate`, `pairwise`, `zip_longest`, `takewhile`, `batched` and `groupby` (sorting first).
- [ ] I can predict how much of an iterator `zip(strict=True)`, `sorted`, `sum`, `any`, `in` and unpacking consume.
- [ ] I can build a data loader that reshuffles every epoch, and an infinite stream I stop by step count.
- [ ] I can translate Kotlin's `Sequence`, `sequence { yield() }` and `asSequence()` into Python, and I know where generators behave differently.

**Next:** [Day 4: NumPy arrays, and why ML runs on them](day-04-numpy-arrays.md)
