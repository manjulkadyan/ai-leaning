# Day 2b: Lists, Tuples, Dicts and Sets in Depth

**Phase 1 · Week 1 · ~170 minutes**

This is the last part of Day 2, after Day 2 itself and Day 2a. It's long: do it the same day if you have the energy, or split it over the next day (a good break point is after "Sets"). Day 2 gave you Python's single values (numbers, booleans, `None`) and control flow, and Day 2a gave you strings; today is about the four built-in containers that hold them, and the handful of `collections` tools built on top.

## Objectives

- [ ] Use every common list method (`append`, `extend`, `insert`, `pop`, `remove`, `index`, `count`, `sort`, `reverse`, `clear`, `copy`) and choose between `list.sort()` and `sorted()`, with `key` and `reverse`.
- [ ] Explain how a list is stored (a growable array of references) and why `append` is fast while `insert(0, ...)` is slow.
- [ ] Slice any sequence with start, stop and step, including negative values, and change a list through slice assignment.
- [ ] Create tuples (including the one-element tuple), pack and unpack them with `*rest`, swap two names, and use a named tuple.
- [ ] Read and change dicts safely (`[]` vs `get`, `in`, `del`, `pop`, `update`, `|`, `setdefault`), iterate over `keys`, `values` and `items`, and use `defaultdict` and `Counter`.
- [ ] Navigate nested, JSON-like dicts and lists.
- [ ] Use sets for uniqueness and set algebra (union, intersection, difference, symmetric difference), and `frozenset` when you need an immutable set.
- [ ] Explain what "hashable" means and fix `TypeError: unhashable type: 'list'`.
- [ ] Tell assignment, shallow copy and deep copy apart, and trace why `[[0] * 3] * 3` goes wrong.
- [ ] Write list, dict and set comprehensions, with conditions and nesting, and know when a plain loop is clearer.
- [ ] Choose the right collection for a job, backed by a real timing of `in` on a list versus a set.
- [ ] Translate everyday Kotlin collection code (`listOf`, `mapOf`, `map`, `filter`, `groupBy`, `associate`) into Python.

**Prerequisites.**

- [Day 1](day-01-setup.md): running Python, the REPL and the ▶ Run button on this site.
- [Day 2](day-02-python-types.md): names and objects (the sticky-note model, `id`, `is` vs `==`), numbers, truthiness, `and`/`or` returning an operand, conditional expressions (`a if cond else b`), `for` loops, `range`, `enumerate` and `zip`.
- [Day 2a](day-02a-strings.md): string indexing and slicing (the fence-post picture), string methods such as `split`, `strip`, `lower` and `join`, and f-strings with format specs such as `:.2f`, `:,`, `:.1%`, `:>8` and `!r`.

!!! info "How to read the code on this page"
    Every Python block has a **▶ Run** button, and blocks share their variables from top to bottom like notebook cells, so run them in order. A line that is a bare expression shows its value when run, exactly as the `# comment` beside it says.

    A few blocks define a small helper with `def name(args):`. That's Kotlin's `fun`: the indented lines are the body, and `return` hands back a value. Day 3 covers functions fully.

    Blocks that start with `>>>` are different: they are transcripts from the Python 3.14 REPL showing an error **on purpose**. They show what an error looks like in a terminal, full traceback included; if you type the same lines into a ▶ Run block, it shows the same final message. They have no Run button, so an error can't break the page. Type them into a `python3` prompt to see the error yourself. `<python-input-1>` in a traceback is the name the REPL gives the line you typed (numbered from 0). Day 2 wrapped its failing lines in `try`/`except` so they could run; here you see the whole traceback, because reading it is part of today's skill.

    A bare expression whose value is `None` shows nothing when run, so where a line's answer is `None` the page wraps it in `print(...)`.

---

## The big idea

Kotlin gives you collection *interfaces* and decides mutability by which one you ask for: `listOf` vs `mutableListOf`, `mapOf` vs `mutableMapOf`. Python does it with *types*: a `list` is always mutable and a `tuple` never is; a `set` is mutable and a `frozenset` isn't. A `dict` is always mutable.

| Python | Literal | Ordered? | Mutable? | Closest Kotlin |
|---|---|---|---|---|
| `list` | `[1, 2, 3]` | yes, by position | yes | `MutableList` / `ArrayList` |
| `tuple` | `(1, 2, 3)` | yes, by position | no | `Pair`, `Triple`, a small data class |
| `dict` | `{"lr": 0.01}` | yes, by insertion | yes | `MutableMap` / `LinkedHashMap` |
| `set` | `{1, 2, 3}` | no | yes | `MutableSet` / `HashSet` |

That table is a map, not the lesson. Each row gets its own section below.

The one rule that runs through every section is the one from Day 2: **a container holds references, not values.** A list of three strings is a row of three sticky notes, each pointing at a string object that lives somewhere else. Almost every surprise today (shared rows in a grid, a copy that isn't, a key that isn't allowed) comes from that rule.

**Analogy.** A list is a numbered row of coat hooks: hook 0, hook 1, hook 2, each holding a ticket that points at a coat. A dict is a cloakroom with a clever attendant: you hand over a ticket name, and they walk straight to the right hook without checking the others. A set is that same cloakroom with tickets and no coats.

## Lists

### What a list is

A **list** is an ordered, growable sequence of references to objects. "Ordered" means each item has a position, its **index**, counted from 0. "Sequence" is Python's word for any ordered collection you can index with integers: lists, tuples, strings and `range` are all sequences.

Why it exists: it is the default container for "some things, in order, and I may add more". In ML code that's a batch of file names, the losses from each epoch, the layers of a model.

In Kotlin you write:

```kotlin
val layers = mutableListOf("conv", "relu", "pool")
layers.add("dense")
println(layers[0])          // conv
println(layers.size)        // 4
println(layers.last())      // dense
```

In Python:

```python
layers = ["conv", "relu", "pool"]
layers.append("dense")
layers[0]          # 'conv'
len(layers)        # 4        a function, not a .size property
layers[-1]         # 'dense'  negative index: count from the end
layers[-2]         # 'pool'

layers[1] = "gelu"           # replace an item in place
layers                       # ['conv', 'gelu', 'pool', 'dense']

list("abc")                  # ['a', 'b', 'c']   any iterable → list
list(range(4))               # [0, 1, 2, 3]
[1, "two", 3.0, None]        # [1, 'two', 3.0, None]   mixed types are allowed
```

What's the same: square-bracket indexing from 0, and the list grows as you add. What's different:

- the length is `len(x)`, a built-in function that works on every container;
- `x[-1]` is the last item, `x[-2]` the one before it (Kotlin needs `last()` or `x[x.size - 1]`);
- `list(...)` builds a list from any **iterable**, which is anything a `for` loop can walk over (a string, a range, a dict, a file);
- a list can mix types. You rarely want that, but nothing stops you.

Indexing past the end raises an **exception**, Python's word for a runtime error (Day 3b teaches how to catch them):

```pycon
>>> layers = ["conv", "relu", "pool"]
>>> layers[3]
Traceback (most recent call last):
  File "<python-input-1>", line 1, in <module>
    layers[3]
    ~~~~~~^^^
IndexError: list index out of range
```

Read a traceback from the **bottom up**. The last line names the exception type (`IndexError`) and the reason. The lines above show where it happened, and the `~~~^^^` marks point at the exact expression. Valid indexes for a 3-item list are `0, 1, 2` and `-1, -2, -3`. Kotlin throws `IndexOutOfBoundsException` in the same situation. Kotlin's `getOrNull(i)` has no built-in Python twin; the idiom is a conditional expression:

```python
i = 5
print(layers[i] if i < len(layers) else None)    # None
```

### How a list is stored

A list object holds three things: its length, its **capacity** (how many slots are allocated), and a pointer to a separate block of memory, an array of **references** (memory addresses). The items themselves live elsewhere. A list of three strings is one small array of three addresses, plus three string objects.

<figure class="diagram" markdown>
![Animation: a list object with len 4 and capacity 4 points at an array of four slots, each slot an arrow to a separate string object. Appending a fifth item finds no free slot, so Python allocates a new array of 8 slots, copies the four arrows across, and writes the fifth arrow into slot 4; the list's capacity changes to 8](../../assets/img/phase-1/day02b-list-memory.svg)
<figcaption>Animated: a list is an array of arrows. When it's full, append moves the arrows into a bigger array, then writes the new one.</figcaption>
</figure>

When you `append` and there's a free slot, Python writes one address and adds 1 to the length: constant time, written **O(1)**. (O-notation describes how cost grows with the number of items `n`: O(1) means "doesn't grow", O(n) means "grows in proportion to n".) When the array is full, Python allocates a bigger one, copies every address across, and then writes the new one. That copy is O(n), but the new array has room to spare, so it happens rarely. You can watch the capacity grow:

```python
import sys       # sys: interpreter internals; getsizeof(obj) is the object's size in bytes
import struct    # struct.calcsize("P") is the size of one memory address in bytes

PTR = struct.calcsize("P")    # 8 on a 64-bit laptop (the browser's Python may use 4)
EMPTY = sys.getsizeof([])

def capacity(lst):
    return (sys.getsizeof(lst) - EMPTY) // PTR    # slots allocated, used or not

grow = []
changes = []
for i in range(30):
    if not changes or capacity(grow) != changes[-1][1]:
        changes.append((len(grow), capacity(grow)))
    grow.append(i)
changes    # [(0, 0), (1, 4), (5, 8), (9, 16), (17, 24), (25, 32)]
```

How the loop works: `capacity(grow)` asks how many slots the list has right now. `changes` records a `(length, capacity)` pair only when the capacity differs from the last pair recorded. `changes[-1]` is that last pair, and `[1]` picks its second item, the capacity (indexing a pair works like indexing a list; pairs are tuples, coming up later). `not changes` is true on the first pass, because an empty list is falsy (Day 2), so the first pair is always recorded. Then `grow.append(i)` adds one item and the loop checks again.

Read each pair as `(length, capacity)`: the empty list has no slots; the first append allocates 4; the fifth finds the array full and grows it to 8; the ninth grows it to 16; then 24, 32, and so on (CPython adds about 1/8 plus a little each time). Spread over many appends, the occasional copy costs a constant amount per append on average. That's called **amortised O(1)**, and it's why building a list with `append` in a loop is fine.

The flip side: anything that shifts items is O(n). `insert(0, x)` has to move every address one slot to the right, and `pop(0)` moves every address one slot left. On a list of a million items that's a million moves per call.

!!! android "Android bridge"
    This is the same design as `java.util.ArrayList`, which backs Kotlin's `mutableListOf()`: an internal `Object[]` array of references plus a size, grown by copying into a bigger array (about 1.5× on the JVM) when full. `add` is amortised O(1) and `add(0, x)` is O(n) in both languages. The big difference shows up on Day 4: an `IntArray` stores raw numbers side by side, while a Python list of ints stores addresses of separate int objects, each 28 bytes on a 64-bit machine. NumPy arrays exist to give Python the `IntArray` layout.

### Adding and removing items

| To do this | Kotlin | Python |
|---|---|---|
| add one item at the end | `add(x)` | `append(x)` |
| add every item of another collection | `addAll(xs)` | `extend(xs)` |
| add at a position | `add(i, x)` | `insert(i, x)` |
| remove by position, get the item back | `removeAt(i)` | `pop(i)` (`pop()` = last) |
| remove the first equal item | `remove(x)` | `remove(x)` |
| remove everything | `clear()` | `clear()` |

The table summarises; here's each one working, with the list shown after every step:

```python
batch = ["img_1.png", "img_2.png"]

batch.append("img_3.png")
batch            # ['img_1.png', 'img_2.png', 'img_3.png']

batch.extend(["img_4.png", "img_5.png"])
batch            # ['img_1.png', 'img_2.png', 'img_3.png', 'img_4.png', 'img_5.png']

batch.insert(0, "img_0.png")      # insert(position, item): goes BEFORE the item at that position
batch            # ['img_0.png', 'img_1.png', 'img_2.png', 'img_3.png', 'img_4.png', 'img_5.png']

batch.pop()      # 'img_5.png'   removes AND returns the last item
batch.pop(0)     # 'img_0.png'   removes and returns item 0 (O(n): everything shifts left)

batch.remove("img_3.png")         # by value; returns None
batch            # ['img_1.png', 'img_2.png', 'img_4.png']

del batch[0]     # del is a statement: delete by position, returns nothing
batch            # ['img_2.png', 'img_4.png']

backup = batch.copy()             # a NEW list with the same items (Kotlin's toMutableList())
batch.clear()
batch, backup    # ([], ['img_2.png', 'img_4.png'])   clearing batch left the copy alone
```

`copy()` matters because `backup = batch` would only add a second name to the same list (Day 2), and `clear()` would empty both. "Copying" below explains exactly what `copy()` does and doesn't copy.

`+` and `+=` also join lists. `a + b` builds a **new** list; `a += b` extends `a` in place (the same as `a.extend(b)`). `*` repeats:

```python
a = [1, 2]
b = a + [3]          # new list; a is unchanged
a, b                 # ([1, 2], [1, 2, 3])
a += [9]             # in place: a itself grows
a                    # [1, 2, 9]
[0] * 4              # [0, 0, 0, 0]
["pad"] * 2 + ["cat", "sat"]   # ['pad', 'pad', 'cat', 'sat']
```

(The `a, b` line builds a tuple of the two lists, which is why the output has round brackets. Tuples are coming up.)

!!! warning "Common mistake: `append` a list instead of `extend`"
    `append` adds its argument as **one** item, even when that argument is a list. You get a nested list instead of a longer one:

    ```python
    files = ["a.png", "b.png"]
    files.append(["c.png", "d.png"])
    files          # ['a.png', 'b.png', ['c.png', 'd.png']]
    len(files)     # 3, not 4
    ```

    Recognise it by a `[` inside your printed list, or a length that is too small. The fix is `files.extend(["c.png", "d.png"])` or `files += ["c.png", "d.png"]`.

`remove` and `pop` fail loudly when there's nothing to remove:

```pycon
>>> layers = ["conv", "relu"]
>>> layers.remove("dense")
Traceback (most recent call last):
  File "<python-input-1>", line 1, in <module>
    layers.remove("dense")
    ~~~~~~~~~~~~~^^^^^^^^^
ValueError: list.remove(x): x not in list
```

```pycon
>>> empty = []
>>> empty.pop()
Traceback (most recent call last):
  File "<python-input-1>", line 1, in <module>
    empty.pop()
    ~~~~~~~~~^^
IndexError: pop from empty list
```

Kotlin's `remove(x)` returns `false` when `x` isn't there; Python's raises `ValueError`. Check first with `in` (next section) when "not there" is a normal case.

!!! warning "Common mistake: `x = x.append(...)`"
    Methods that change a list in place (`append`, `extend`, `insert`, `remove`, `sort`, `reverse`, `clear`) return `None`, Python's "no value" object. Assigning the result throws your list away:

    ```pycon
    >>> x = [1, 2, 3]
    >>> x = x.append(4)
    >>> x.append(5)
    Traceback (most recent call last):
      File "<python-input-2>", line 1, in <module>
        x.append(5)
        ^^^^^^^^
    AttributeError: 'NoneType' object has no attribute 'append'
    ```

    An `AttributeError` mentioning `'NoneType'` almost always means "you stored the `None` some method returned". Call in-place methods on their own line, without `=`.

### Finding and counting

```python
preds = ["cat", "dog", "cat", "fox", "cat"]
"dog" in preds          # True    membership: scans until it finds an equal item
"owl" not in preds      # True
preds.index("cat")      # 0       position of the FIRST equal item
preds.index("cat", 1)   # 2       ...searching from position 1 onwards
preds.count("cat")      # 3
```

`index` raises if the item isn't there, like `remove`:

```pycon
>>> layers = ["conv", "relu"]
>>> layers.index("dense")
Traceback (most recent call last):
  File "<python-input-1>", line 1, in <module>
    layers.index("dense")
    ~~~~~~~~~~~~^^^^^^^^^
ValueError: list.index(x): x not in list
```

(Python 3.13 and earlier word this `ValueError: 'dense' is not in list`.) Kotlin's `indexOf` returns `-1` instead. In Python, `-1` is a valid index (the last item), so returning it would hide bugs; raising is safer.

### `min`, `max`, `sum`, `any` and `all`

These are **built-in functions** (always available, no import) that take any iterable and boil it down to one value: the smallest item, the largest, the total, or a yes/no answer. They work on lists, tuples, sets, ranges, strings and dict keys alike.

Why they exist: "best score", "worst loss", "total size", "is any prediction confident?" are one-liners you'll write constantly, and these functions do the loop in C, faster than a hand-written one.

In Kotlin these are extension functions on the collection:

```kotlin
val scores = listOf(0.12, 0.85, 0.40)
scores.min(); scores.max(); scores.sum()      // 0.12, 0.85, 1.37 (min/max throw on an empty list)
scores.minOrNull()                            // null on an empty list
scores.any { it > 0.8 }; scores.all { it > 0.1 }
```

In Python they are functions you call with the collection:

```python
scores = [0.12, 0.85, 0.40]
min(scores), max(scores)        # (0.12, 0.85)
sum(scores)                     # 1.37
sum(scores) / len(scores)       # 0.4566666666666667   the mean
sum(scores, 10)                 # 11.37   an optional start value (default 0)
min("banana")                   # 'a'     works on any iterable of comparable items
max([(4.5, "Keep"), (4.6, "Files")])   # (4.6, 'Files')   tuples compare item by item
```

**`key=`** works as it does for `sorted` (next section): it's a function called once per item, and the items are compared by what it returns. The answer is still the original item:

```python
words = ["cat", "mouse", "ox"]
max(words)                 # 'ox'      alphabetical: 'o' comes after 'm' and 'c'
max(words, key=len)        # 'mouse'   the longest
min(words, key=len)        # 'ox'      the shortest
max(["cat", "dog"], key=len)    # 'cat'   a tie: the FIRST of the largest wins
max(range(len(scores)), key=lambda i: scores[i])   # 1   the index of the best score (an argmax)
```

`key=len` passes the function `len` itself, without parentheses: `max` calls it for you on each item. `lambda i: scores[i]` is a small unnamed function: it takes `i` and returns `scores[i]`, like Kotlin's `{ i -> scores[i] }`. So the last line walks over the indexes `0, 1, 2` but compares their scores, and returns the index whose score is largest, Kotlin's `scores.indices.maxBy { scores[it] }`. Day 3 covers `lambda` fully.

`any` and `all` take an iterable of true/false values:

```python
any(s > 0.8 for s in scores)    # True    is at least one item true?
all(s > 0.1 for s in scores)    # True    are all of them?
any([]), all([])                # (False, True)   nothing is true; nothing is false
```

The `s > 0.8 for s in scores` inside is a **generator expression**, a lazy relative of the comprehensions taught later today: it hands `any` one `True`/`False` at a time.

How they work underneath: each is one pass, left to right. `max` keeps a "best so far" and replaces it only when an item is strictly larger, which is why ties keep the first. With `key=`, the key is computed once per item and compared, and the item itself is kept. `sum` starts from `start` (0 by default) and adds each item in turn: `sum(xs)` is `0 + xs[0] + xs[1] + ...`. `any` stops at the first true value and `all` at the first false one, so they can finish early (called **short-circuiting**). `all([])` is `True` because there's no item to make it false.

They go wrong in two common ways. An empty input has no smallest or largest item:

```pycon
>>> scores = []
>>> max(scores)
Traceback (most recent call last):
  File "<python-input-1>", line 1, in <module>
    max(scores)
    ~~~^^^^^^^^
ValueError: max() iterable argument is empty
```

This happens when a filter upstream removed everything (no predictions above the threshold, an empty batch). Pass `default=` to get a value instead, the way Kotlin's `maxOrNull()` gives `null`:

```python
print(max([], default=None))    # None
max([], default=0.0)            # 0.0
```

And `sum` only adds numbers, because it starts from the number 0:

```pycon
>>> sum(["cat", "dog"])
Traceback (most recent call last):
  File "<python-input-0>", line 1, in <module>
    sum(["cat", "dog"])
    ~~~^^^^^^^^^^^^^^^^
TypeError: unsupported operand type(s) for +: 'int' and 'str'
```

Read it as "`0 + 'cat'` isn't allowed". To glue strings, use `"".join(words)` (Day 2a).

### Sorting: `list.sort()` vs `sorted()`

Python has two ways to sort, and the difference matters:

- `xs.sort()` is a **method**. It reorders `xs` itself (in place) and returns `None`.
- `sorted(xs)` is a **built-in function**. It leaves `xs` alone and returns a new list. It accepts any iterable: a tuple, a set, a dict's keys, a string.

Kotlin draws the same line: `list.sort()` on a `MutableList` vs `list.sorted()` returning a new `List`.

```python
nums = [3, 1, 2]
new = sorted(nums)
new, nums             # ([1, 2, 3], [3, 1, 2])   nums untouched
nums.sort()
nums                  # [1, 2, 3]                 nums changed
sorted("banana")      # ['a', 'a', 'a', 'b', 'n', 'n']
sorted(nums, reverse=True)    # [3, 2, 1]
```

Both take the same two keyword arguments:

- `reverse=True` sorts high to low (Kotlin's `sortedDescending()`).
- `key=` is a function called **once per item**; items are ordered by what it returns, not by themselves. This is Kotlin's `sortedBy { }`.

```python
words = ["banana", "Cherry", "apple"]
sorted(words)                   # ['Cherry', 'apple', 'banana']   capitals sort before lower case
sorted(words, key=str.lower)    # ['apple', 'banana', 'Cherry']   compare lower-cased copies
sorted(words, key=len)          # ['apple', 'banana', 'Cherry']   by length: 5, 6, 6
```

`key=str.lower` and `key=len` pass the function itself, without parentheses: `sorted` calls it for you on each item. Without a key, strings compare by Unicode code point, and every capital letter (`'C'` is 67) comes before every lower-case one (`'a'` is 97).

Sorting **records** is where `key` earns its keep. Here are apps as `(name, category, rating)` tuples:

```python
apps = [
    ("Snapseed", "photo", 4.5),
    ("Lens", "photo", 4.6),
    ("Keep", "tools", 4.5),
    ("Files", "tools", 4.6),
    ("Gallery", "photo", 4.5),
]
by_rating = sorted(apps, key=lambda app: app[2], reverse=True)
for app in by_rating:
    print(app[0], app[2])
# Lens 4.6
# Files 4.6
# Snapseed 4.5
# Keep 4.5
# Gallery 4.5
```

`app[2]` is item 2 of each tuple, the rating; indexing a tuple works like indexing a list.

Look at the ties. `Snapseed`, `Keep` and `Gallery` all have 4.5, and they come out in their original order. Python's sort is **stable**: items with equal keys keep their relative order. Kotlin's `sortedBy` is stable too.

To sort by several things, return a **tuple** from the key. Tuples compare item by item, left to right, like words in a dictionary, so `(4.6, "Files")` < `(4.6, "Lens")` because the first items tie and `"Files" < "Lens"`. To make one part descending and another ascending, negate the number:

```python
sorted(apps, key=lambda app: (app[1], -app[2], app[0]))
# [('Lens', 'photo', 4.6), ('Gallery', 'photo', 4.5), ('Snapseed', 'photo', 4.5),
#  ('Files', 'tools', 4.6), ('Keep', 'tools', 4.5)]
```

Trace the key for `Lens`: `("photo", -4.6, "Lens")`. All photo apps sort before tools apps (`"photo" < "tools"`), inside each category `-4.6 < -4.5` puts the higher rating first, and ties on rating fall back to the name. In Kotlin this is `sortedWith(compareBy({ it.category }, { -it.rating }, { it.name }))`.

`reverse()` and `reversed()` follow the same method-versus-function split: `xs.reverse()` flips `xs` in place, and `reversed(xs)` returns a lazy reversed view that you can loop over or turn into a list.

```python
order = [1, 2, 3]
list(reversed(order)), order    # ([3, 2, 1], [1, 2, 3])
order.reverse()
order                           # [3, 2, 1]
```

How sorting works underneath: CPython uses **Timsort**, a merge sort that first finds runs that are already in order, so sorting nearly sorted data is close to O(n), and the worst case is O(n log n). The `key` function is called exactly once per item, and the results are cached during the sort, so an expensive key costs n calls, not n log n.

!!! warning "Common mistake: sorting things that can't be compared"
    Python 3 refuses to compare unrelated types, so a list that mixes them can't be sorted:

    ```pycon
    >>> mixed = [3, "a", 1.5]
    >>> mixed.sort()
    Traceback (most recent call last):
      File "<python-input-1>", line 1, in <module>
        mixed.sort()
        ~~~~~~~~~~^^
    TypeError: '<' not supported between instances of 'str' and 'int'
    ```

    You'll meet this when a column of numbers has a missing value stored as `None` or a placeholder string such as `"Varies with device"`. The fix is to clean the data first (convert or drop the odd values), or to sort with a `key` that maps every item to one comparable type, for example `key=str`:

    ```python
    sorted([3, "a", 1.5], key=str)       # [1.5, 3, 'a']   compared as '1.5', '3', 'a'
    sorted([10, 9, 100], key=str)        # [10, 100, 9]    but numbers now sort as TEXT
    ```

    The second line shows the cost: as text, `"10" < "9"` because `"1"` comes before `"9"`. `key=str` makes a mixed list sortable; it doesn't make it sensible. Cleaning the data is the real fix.

### Nested lists, equality and comparison

A list can hold other lists. That's how plain Python represents a table or a matrix before NumPy (Day 4):

```python
m = [[1, 2, 3],
     [4, 5, 6]]
m[1]          # [4, 5, 6]    row 1
m[1][2]       # 6            row 1, then item 2 of that row
len(m), len(m[0])            # (2, 3)   rows, columns

[1, 2, 3] == [1, 2, 3]       # True    == compares contents, item by item
[1, 2, 3] is [1, 2, 3]       # False   two different list objects
[1, 2, 3] < [1, 3]           # True    first difference decides: 2 < 3
```

`==` on lists is Kotlin's `==` on lists (structural equality), and `is` is Kotlin's `===` (same object), as on Day 2. `<` compares **lexicographically**: item by item until two differ. Kotlin lists don't define `<` at all.

## Slicing in depth

You met slices on strings in Day 2. Every sequence supports them, and lists add a twist: you can assign to a slice.

A **slice** `seq[start:stop:step]` picks the items at positions `start`, `start + step`, `start + 2·step`, and so on, stopping **before** `stop`. Each of the three numbers is optional:

- `start` defaults to the beginning (or the end, if `step` is negative);
- `stop` defaults to "past the end" (or "past the beginning" for a negative step);
- `step` defaults to 1.

It helps to picture the indexes as fence posts **between** items, the same picture Day 2a used for strings:

```text
 positive:   0     1     2     3     4     5     6
             |  a  |  b  |  c  |  d  |  e  |  f  |
 negative:  -6    -5    -4    -3    -2    -1
```

`x[1:4]` is "from post 1 to post 4": `b, c, d`. `x[-2:]` is "from post -2 to the end": `e, f`. A slice of a list is always a **new list** (a shallow copy, explained in "Copying" below).

```python
x = ["a", "b", "c", "d", "e", "f"]
x[1:4]       # ['b', 'c', 'd']
x[:2]        # ['a', 'b']        first two
x[-2:]       # ['e', 'f']        last two
x[:-1]       # ['a', 'b', 'c', 'd', 'e']   all but the last
x[::2]       # ['a', 'c', 'e']   every second item, from 0
x[1::2]      # ['b', 'd', 'f']   every second item, from 1
x[:]         # ['a', 'b', 'c', 'd', 'e', 'f']   a copy of the whole list
x[2:100]     # ['c', 'd', 'e', 'f']   a slice past the end is clipped, no error
x[4:2]       # []                start after stop with a positive step: empty
```

Kotlin's `subList(1, 4)` is the closest relative, with two differences: it throws if the indexes are out of range, and it returns a *view* that shares storage with the original. A Python slice never throws for out-of-range positions and always copies.

**Negative steps** walk backwards. Trace `x[4:1:-1]`: start at position 4 (`'e'`), step −1 to position 3 (`'d'`), then 2 (`'c'`), then the next position, 1, is the stop, so it isn't included.

```python
x[4:1:-1]    # ['e', 'd', 'c']
x[::-1]      # ['f', 'e', 'd', 'c', 'b', 'a']   start defaults to the END when step < 0
x[::-2]      # ['f', 'd', 'b']
x[-1:-4:-1]  # ['f', 'e', 'd']    the last three, newest first
```

**Slice assignment** replaces a whole stretch of a list with the items of another iterable. The lengths don't have to match, so the list can grow or shrink:

```python
y = [0, 1, 2, 3, 4, 5]
y[1:3] = ["a", "b", "c"]     # replace 2 items with 3: the list grows
y                            # [0, 'a', 'b', 'c', 3, 4, 5]
y[1:4] = []                  # replace 3 items with none: delete them
y                            # [0, 3, 4, 5]
y[1:1] = [1, 2]              # an empty slice at position 1: insert there
y                            # [0, 1, 2, 3, 4, 5]
y[::2] = [10, 20, 30]        # with a step, the lengths MUST match (3 slots, 3 items)
y                            # [10, 1, 20, 3, 30, 5]
del y[::2]                   # del works on slices too
y                            # [1, 3, 5]
```

```pycon
>>> nums = [1, 2, 3, 4]
>>> nums[::2] = [0]
Traceback (most recent call last):
  File "<python-input-1>", line 1, in <module>
    nums[::2] = [0]
    ~~~~^^^^^
ValueError: attempt to assign sequence of size 1 to extended slice of size 2
```

An **extended slice** is one with a step. It names fixed positions, so the replacement must fill exactly that many. A step of 0 is also an error (`ValueError: slice step cannot be zero`).

!!! tip "Slices in ML code"
    Splitting data and cutting batches are slices. With 10 samples and a batch size of 4, `data[i:i + 4]` for `i` in `range(0, 10, 4)` gives batches of 4, 4 and 2: the last slice is clipped at the end instead of raising. Day 4 uses exactly this syntax on NumPy arrays, one slice per axis.

    ```python
    data = list(range(10))
    train, test = data[:8], data[8:]
    train, test                                       # ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9])
    [data[i:i + 4] for i in range(0, len(data), 4)]   # [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    ```

## Tuples

### What a tuple is, and why it exists

A **tuple** is an ordered, **immutable** sequence: once created, you can't add, remove or replace its items. It supports everything a list does that doesn't change it: indexing, slicing, `len`, `in`, `count`, `index`, iteration, `+` and `*` (which build new tuples).

Why have an immutable list? Three reasons you'll hit this week:

1. **Fixed-shape records.** An image shape `(224, 224, 3)`, a 2-D point `(x, y)`, an RGB colour: the number of items is part of the meaning.
2. **Returning several values** from a function. `return mean, std` returns one tuple (Day 3).
3. **Dict keys and set members.** Only immutable things can be keys (see "Hashability"), so `(row, col)` can index a dict and `[row, col]` can't.

The Kotlin relatives are `Pair` and `Triple`, which are also immutable, and destructuring declarations:

```kotlin
val shape = Triple(224, 224, 3)
val (h, w, c) = shape
```

```python
shape = (224, 224, 3)
h, w, c = shape          # unpacking: Kotlin's destructuring declaration
h, w, c                  # (224, 224, 3)
shape[0], shape[-1]      # (224, 3)
shape[:2]                # (224, 224)   slicing a tuple gives a tuple
len(shape), 3 in shape   # (3, True)
shape + (1,)             # (224, 224, 3, 1)   a NEW tuple; shape is unchanged
```

Unlike `Triple`, a tuple can have any number of items, and they're read by position, not by `first`/`second`/`third`.

```pycon
>>> shape = (224, 224, 3)
>>> shape[0] = 128
Traceback (most recent call last):
  File "<python-input-1>", line 1, in <module>
    shape[0] = 128
    ~~~~~^^^
TypeError: 'tuple' object does not support item assignment
```

To "change" a tuple, build a new one: `(128,) + shape[1:]` gives `(128, 224, 3)`.

### The comma makes the tuple

The parentheses are mostly decoration. It's the **comma** that makes a tuple:

```python
t = 1, 2, 3          # a tuple, no parentheses needed
t                    # (1, 2, 3)
type((5))            # <class 'int'>     the int 5 in grouping parentheses
type((5,))           # <class 'tuple'>   the trailing comma makes a 1-tuple
type(())             # <class 'tuple'>   the empty tuple is the one exception
```

!!! warning "Common mistake: the one-element tuple"
    `(5)` is the number 5. A one-element tuple needs a trailing comma: `(5,)`. This bites with shapes: NumPy (Day 4) reports a 1-D array's shape as `(5,)`, and passing `(5)` where a shape tuple is expected gives you an `int`. The opposite slip also happens: a stray comma at the end of an assignment, `n = 5,`, silently makes `n` the tuple `(5,)`.

### Packing, unpacking and swapping

Writing `t = 1, 2, 3` **packs** three values into a tuple. Writing `a, b, c = t` **unpacks** it. Unpacking works on any iterable, not only tuples, and the counts must match:

```pycon
>>> h, w = (224, 224, 3)
Traceback (most recent call last):
  File "<python-input-0>", line 1, in <module>
    h, w = (224, 224, 3)
    ^^^^
ValueError: too many values to unpack (expected 2, got 3)
```

(Python 3.13 and earlier leave out the `, got 3` part.) A **starred name** collects "everything else" into a list, so the counts no longer have to match exactly:

```python
first, *rest = [10, 20, 30, 40]
first, rest                 # (10, [20, 30, 40])
*init, last = "abc"
init, last                  # (['a', 'b'], 'c')
batch_dim, *spatial, channels = (32, 224, 224, 3)
spatial                     # [224, 224]
```

There can be only one starred name, and it is always a list, possibly empty.

**Swapping** two names needs no temporary variable:

```python
a, b = 1, 2
a, b = b, a
a, b                        # (2, 1)
```

Here's what happens, step by step. Python evaluates the whole right-hand side first: `b, a` packs the current values into a new tuple `(2, 1)`. Only then does it unpack that tuple into `a` and `b`, left to right. Because the tuple was built before any assignment, nothing is overwritten too early. Kotlin's idiom for the same thing is `a = b.also { b = a }`.

Unpacking shines in `for` loops over sequences of pairs. Each item is unpacked as the loop takes it:

```python
pairs = [("cat", 0.91), ("dog", 0.07)]
for label, score in pairs:
    print(f"{label:>4}: {score:.2f}")
#  cat: 0.91
#  dog: 0.07

for i, (label, score) in enumerate(pairs):    # nested unpacking: (index, (label, score))
    print(i, label)
# 0 cat
# 1 dog
```

Unpacking also works inside list, tuple and set **literals**: `[*a, *b]` builds a list from the items of `a` followed by those of `b`.

```python
[*(1, 2), *[3, 4], 5]     # [1, 2, 3, 4, 5]
```

### A tuple holds references, and so can hold mutable things

A tuple's immutability is shallow: the tuple can't change *which* objects it points at, but if one of those objects is a list, the list can still change.

```python
record = ("run-7", [0.9, 0.7])
record[1].append(0.5)     # allowed: we change the list, not the tuple
record                    # ('run-7', [0.9, 0.7, 0.5])
```

This matters for hashability, below: a tuple that contains a list can't be a dict key.

### Named tuples

A **named tuple** is a tuple whose positions also have names, so `p.x` reads better than `p[0]`. `collections.namedtuple` creates the type (`collections` is a standard-library module of extra container types; `from collections import namedtuple` brings in one name from it):

```python
from collections import namedtuple

Prediction = namedtuple("Prediction", ["label", "score"])
p = Prediction("cat", 0.91)
p                       # Prediction(label='cat', score=0.91)
p.label, p[1]           # ('cat', 0.91)   by name or by position
label, score = p        # it is still a tuple, so it unpacks
p._replace(score=0.5)   # Prediction(label='cat', score=0.5)   a NEW tuple
isinstance(p, tuple)    # True
```

Underneath, `namedtuple("Prediction", ["label", "score"])` generates a brand-new class named `Prediction` that is a subclass of `tuple`, plus a read-only property for each field name (`p.label` is item 0). That's why it unpacks, indexes, compares and hashes exactly like a plain tuple, and costs no more memory than one. `isinstance(x, T)` asks "is `x` a `T`, or a subclass of `T`?". The leading underscore in `_replace` (and its siblings `_asdict` and `_fields`) isn't a "private" marker here: it's there so the method names can't clash with a field you might call `replace`.

Because it's a tuple, it's immutable, and assigning to a field fails:

```pycon
>>> from collections import namedtuple
>>> Prediction = namedtuple("Prediction", ["label", "score"])
>>> p = Prediction("cat", 0.91)
>>> p.score = 0.5
Traceback (most recent call last):
  File "<python-input-3>", line 1, in <module>
    p.score = 0.5
    ^^^^^^^
AttributeError: can't set attribute
```

The fix is to make a changed copy, `p = p._replace(score=0.5)`, or, if the record really needs to change in place, to use Day 3's dataclass.

It's the closest built-in thing to a Kotlin `data class` with `val` properties: readable fields, a useful printed form, equality by value, and `_replace` as the counterpart of `copy(score = 0.5)`. For records with defaults, methods or mutable fields, Day 3's **dataclasses** are the better tool.

## Dicts

### What a dict is

A **dict** (dictionary) maps **keys** to **values**. Each key appears at most once; looking up a value by its key takes the same time whether the dict has ten entries or ten million. Keys must be **hashable** (roughly: immutable, see "Hashability"); values can be anything.

Why it exists: any time you'd say "look X up by name". In ML, dicts are everywhere: a model config (`{"lr": 0.001, "epochs": 10}`), a label map (`{0: "cat", 1: "dog"}`), word counts, a parsed JSON response, the `state_dict` holding a PyTorch model's weights.

```kotlin
val config = mutableMapOf("lr" to 0.001, "epochs" to 10)
config["batch_size"] = 32
println(config["lr"])         // 0.001
println(config["dropout"])    // null
```

```python
config = {"lr": 0.001, "epochs": 10}       # literal: {key: value, ...}
config["batch_size"] = 32                   # add, or overwrite if the key exists
config                      # {'lr': 0.001, 'epochs': 10, 'batch_size': 32}
config["lr"]                # 0.001
len(config)                 # 3

dict(lr=0.001, epochs=10)   # {'lr': 0.001, 'epochs': 10}   keyword form: keys become strings
dict([("cat", 0), ("dog", 1)])          # {'cat': 0, 'dog': 1}   from (key, value) pairs
dict(zip(["cat", "dog"], [0, 1]))       # {'cat': 0, 'dog': 1}   zip pairs two lists up
dict.fromkeys(["a", "b"], 0)            # {'a': 0, 'b': 0}       every key, one shared value
```

The same as Kotlin: `d[key] = value` to add or overwrite, square brackets to read. The big difference is what happens when the key is missing.

### Reading safely: `[]`, `get` and `in`

In Kotlin, `map[key]` returns `null` for a missing key. In Python, `d[key]` **raises**:

```pycon
>>> stock = {"cat": 3}
>>> stock["dog"]
Traceback (most recent call last):
  File "<python-input-1>", line 1, in <module>
    stock["dog"]
    ~~~~~^^^^^^^
KeyError: 'dog'
```

`KeyError` prints the missing key, which is usually all you need: a typo (`'Dog'`), a key of the wrong type (`1` vs `'1'`), or a key that legitimately might not be there. For the last case, Python has two tools:

```python
print(config.get("dropout"))   # None   Kotlin's config["dropout"]
config.get("dropout", 0.0)     # 0.0    Kotlin's getOrDefault, or config["dropout"] ?: 0.0
"dropout" in config            # False  Kotlin's containsKey
"lr" in config                 # True
0.001 in config                # False  in checks KEYS, never values
0.001 in config.values()       # True   check values explicitly (this one is a slow scan)
```

Which to use: `d[key]` when a missing key is a bug you want to hear about (a required config field); `d.get(key, default)` when missing is normal (an optional setting). Don't write `d.get(key)` everywhere "to be safe": a silent `None` that crashes three functions later is harder to trace than a `KeyError` at the source.

### Changing a dict

```python
d = {"lr": 0.01, "epochs": 10, "seed": 0}
del d["seed"]                     # remove; KeyError if absent
d.pop("epochs")                   # 10     remove AND return the value
print(d.pop("momentum", None))    # None   with a default: no error if absent
d                                 # {'lr': 0.01}

d.update({"lr": 0.02, "epochs": 5})   # merge in place; existing keys are overwritten
d                                 # {'lr': 0.02, 'epochs': 5}

defaults = {"lr": 0.001, "epochs": 10, "batch_size": 32}
user = {"lr": 0.05}
defaults | user                   # {'lr': 0.05, 'epochs': 10, 'batch_size': 32}   NEW dict; right side wins
{**defaults, **user}              # {'lr': 0.05, 'epochs': 10, 'batch_size': 32}   the older spelling
defaults                          # {'lr': 0.001, 'epochs': 10, 'batch_size': 32}  unchanged
user |= {"seed": 42}              # |= merges in place, like update
user                              # {'lr': 0.05, 'seed': 42}
```

`a | b` (Python 3.9+) is Kotlin's `a + b` on maps: a new map where `b`'s values win on shared keys. `{**a, **b}` unpacks both dicts into a new literal and gives the same result.

`setdefault(key, default)` is Kotlin's `getOrPut`: if `key` is missing it stores `default` under it, and either way it returns the value now stored. Its classic use is grouping:

```python
groups = {}
for name, category in [("Lens", "photo"), ("Keep", "tools"), ("Snapseed", "photo")]:
    groups.setdefault(category, []).append(name)
groups          # {'photo': ['Lens', 'Snapseed'], 'tools': ['Keep']}
```

Trace it: first pass, `"photo"` is missing, so `setdefault` stores a new empty list and returns it, and `append` adds `"Lens"` to that stored list. Second pass, `"tools"` is missing: same thing. Third pass, `"photo"` exists, so `setdefault` returns the existing list and `"Snapseed"` joins `"Lens"`. (`defaultdict`, below, does this more cleanly.)

### Iterating: keys, values, items

```python
metrics = {"loss": 0.42, "accuracy": 0.88, "f1": 0.81}
for key in metrics:                    # looping over a dict gives its KEYS
    print(key)
# loss
# accuracy
# f1

list(metrics.keys())                   # ['loss', 'accuracy', 'f1']
list(metrics.values())                 # [0.42, 0.88, 0.81]
list(metrics.items())                  # [('loss', 0.42), ('accuracy', 0.88), ('f1', 0.81)]

for name, value in metrics.items():    # items() gives (key, value) tuples: unpack them
    print(f"{name:<8} {value:.2f}")
# loss     0.42
# accuracy 0.88
# f1       0.81
```

`items()` is Kotlin's `for ((name, value) in metrics)`. `keys()`, `values()` and `items()` return **views**: live windows onto the dict, not copies. If the dict changes, a view you saved earlier shows the change.

**Order is guaranteed.** Since Python 3.7, a dict remembers insertion order: iteration, printing and `list(d)` follow the order keys were first added. Overwriting a key's value keeps its position; deleting and re-adding moves it to the end. This makes a Python `dict` behave like Kotlin's default `mutableMapOf()`, which is a `LinkedHashMap`, not like a bare `HashMap`.

Order is kept, but it doesn't count for equality: two dicts are `==` when they hold the same key-value pairs, in any order (Kotlin's map `==` works the same way).

```python
{"a": 1, "b": 2} == {"b": 2, "a": 1}     # True
list({"a": 1, "b": 2}) == list({"b": 2, "a": 1})   # False   the key ORDER differs
```

To sort a dict by its values, sort its items with a key. `kv` is one `(key, value)` tuple, so `kv[1]` is the value:

```python
sorted(metrics.items(), key=lambda kv: kv[1], reverse=True)
# [('accuracy', 0.88), ('f1', 0.81), ('loss', 0.42)]
```

To find the key with the largest value (the best class, the best epoch), give `max` the dict and its own `get` method as the key:

```python
probs = {"cat": 0.2, "dog": 0.7, "fox": 0.1}
max(probs, key=probs.get)      # 'dog'
max(probs.values())            # 0.7    the value itself
```

Trace it: iterating `probs` gives its keys, `"cat"`, `"dog"`, `"fox"`. For each key, `max` calls `probs.get(key)` (the method passed without parentheses, like `key=len`), getting `0.2`, `0.7`, `0.1`, and returns the key whose value was largest. It's Kotlin's `probs.maxBy { it.value }.key`.

!!! warning "Common mistake: changing a dict's size while looping over it"
    Adding or removing keys while a `for` loop walks the same dict breaks the loop:

    ```pycon
    >>> sizes = {"small": 1, "large": 3}
    >>> for k in sizes:
    ...     sizes[k.upper()] = sizes[k]
    ...
    Traceback (most recent call last):
      File "<python-input-1>", line 1, in <module>
        for k in sizes:
                 ^^^^^
    RuntimeError: dictionary changed size during iteration
    ```

    (Day 2 showed the list version of this bug. Kotlin throws `ConcurrentModificationException` for the same thing.) Fix it by looping over a snapshot of the keys, `for k in list(sizes):`, or by building a new dict with a comprehension. Changing the **values** of existing keys during the loop is fine; only adding or removing keys is a problem.

### How a dict works underneath: a hash table

A dict is a **hash table**. A **hash** is an integer computed from a value by the built-in `hash()` function, with one promise: values that are equal (`==`) have equal hashes. To store `d["cat"] = 3`, Python computes `hash("cat")`, uses it to pick a slot in an internal table, and stores the key, its hash and the value there. To read `d["cat"]`, it recomputes the hash, jumps straight to that slot, and then checks with `==` that the key stored there really is `"cat"` (two different keys can land on the same slot; Python then probes a sequence of other slots). The cost doesn't depend on how many entries there are, so a lookup is **O(1) on average**.

<figure class="diagram" markdown>
![Animation in two halves. Left: checking whether "fox" is in a list of six strings, a highlight moves item by item and compares each one, six comparisons in total. Right: checking whether "fox" is in a set, hash("fox") is computed, which picks one slot of an eight-slot table, and a single comparison finds it](../../assets/img/phase-1/day02b-hash-lookup.svg)
<figcaption>Animated: <code>in</code> on a list compares item after item; on a set or dict the hash jumps straight to one slot.</figcaption>
</figure>

```python
hash(7)          # 7     small ints hash to themselves
hash(7.0)        # 7     7.0 == 7, so the hashes must agree
hash(True)       # 1     True == 1
{1: "a", 1.0: "b", True: "c"}    # {1: 'c'}   three spellings of ONE key
```

That last line surprises people. `1`, `1.0` and `True` are all equal and hash the same, so they're one key: the first spelling (`1`) stays as the key, and each later entry overwrites the value. Python's integer-valued floats and bools behave this way everywhere, in sets too.

String hashes are **randomised per process** (a defence against attackers who craft colliding keys), so `hash("cat")` gives a different number every time you start Python. Never store a string hash and expect it to match tomorrow.

Since Python 3.6, CPython keeps the entries themselves in a separate, compact array in insertion order, with the hash table holding only positions into that array. That's what makes the ordering guarantee cheap.

### `defaultdict` and `Counter`

Grouping and counting are so common that `collections` has a type for each.

A **`defaultdict`** is a dict that builds a value for a missing key instead of raising `KeyError`. You give it a **factory**, a function that it calls with no arguments to make that value: `list` makes `[]`, `int` makes `0`, `set` makes `set()`.

```python
from collections import defaultdict

by_category = defaultdict(list)          # missing key → by_category[key] = list() = []
for name, category in [("Lens", "photo"), ("Keep", "tools"), ("Snapseed", "photo")]:
    by_category[category].append(name)
by_category
# defaultdict(<class 'list'>, {'photo': ['Lens', 'Snapseed'], 'tools': ['Keep']})
dict(by_category)                        # {'photo': ['Lens', 'Snapseed'], 'tools': ['Keep']}
```

Notice `defaultdict(list)` passes the function `list`, not `list()`: the dict calls it later, once per new key. This is Kotlin's `groupBy`, written as a loop:

```kotlin
val byCategory = apps.groupBy({ it.second }, { it.first })   // Map<String, List<String>>
```

!!! warning "Common mistake: a `defaultdict` creates keys when you read them"
    Even a lookup adds the key: `by_category["games"]` returns `[]` **and stores it**. `"games" in by_category` is safe (it doesn't create anything), and so is `by_category.get("games")`. If you pass the result on, convert with `dict(...)` so that later typos raise `KeyError` again.

    ```python
    probe = defaultdict(int)
    probe["typo"]      # 0
    dict(probe)        # {'typo': 0}   the read created an entry
    ```

A **`Counter`** counts hashable things. Give it any iterable and it builds a dict-like object mapping each item to how many times it appeared:

```python
from collections import Counter

preds = ["cat", "dog", "cat", "fox", "cat", "dog"]
counts = Counter(preds)
counts                    # Counter({'cat': 3, 'dog': 2, 'fox': 1})
counts["cat"]             # 3
counts["owl"]             # 0     a missing item counts as 0, no KeyError (and no new key)
counts.most_common(2)     # [('cat', 3), ('dog', 2)]
counts.total()            # 6

counts.update(["owl", "cat"])          # add more observations
counts                    # Counter({'cat': 4, 'dog': 2, 'fox': 1, 'owl': 1})
Counter("hello") + Counter("world")    # Counter({'l': 3, 'o': 2, 'h': 1, 'e': 1, 'w': 1, 'r': 1, 'd': 1})
Counter(a=3, b=1) - Counter(a=1, b=2)  # Counter({'a': 2})
```

`+` adds the counts key by key (`'l'` is 2 in "hello" plus 1 in "world"). `-` subtracts key by key and then drops every count that isn't positive, which is why `'b'` (1 − 2 = −1) disappears. Underneath, a `Counter` is a dict subclass: keys are the items, values are their counts, and its one special rule is that a missing key reads as 0. Every dict method (`items()`, `keys()`, `in`, `del`) works on it.

`Counter(preds)` is Kotlin's `preds.groupingBy { it }.eachCount()`. Ties in `most_common` come out in first-seen order. You'll use `Counter` to check class balance in a dataset, to build a vocabulary (exercise 8), and to see which mistakes a model makes most.

!!! warning "Common mistake: counting a string's characters"
    A string is an iterable of characters, so passing a sentence counts letters, not words:

    ```python
    Counter("cat dog")            # Counter({'c': 1, 'a': 1, 't': 1, ' ': 1, 'd': 1, 'o': 1, 'g': 1})
    Counter("cat dog".split())    # Counter({'cat': 1, 'dog': 1})
    ```

    Recognise it by single letters and a `' '` key in the output. Split the text into words first (Day 2a's `split`).

### Nested dicts and JSON-like data

Configs and API responses arrive as **JSON**, a text format whose objects become dicts and whose arrays become lists when Python reads them (Day 3c covers the `json` module). Once loaded, you're navigating nested dicts and lists:

```python
response = {
    "model": "mobilenet_v3",
    "input": {"shape": [1, 224, 224, 3], "dtype": "float32"},
    "labels": ["cat", "dog", "fox"],
    "metrics": {"top1": 0.752, "top5": 0.925},
}
response["input"]["shape"]           # [1, 224, 224, 3]
response["input"]["shape"][1:3]      # [224, 224]
response["labels"][-1]               # 'fox'
response.get("training", {}).get("epochs", 10)   # 10   optional nested value with a default

for name, value in response["metrics"].items():
    print(f"{name}: {value:.1%}")
# top1: 75.2%
# top5: 92.5%
```

Read `response["input"]["shape"][1:3]` left to right: `response["input"]` is the inner dict, `["shape"]` is the list inside it, and `[1:3]` slices that list. The chained `.get("training", {})` returns an empty dict when `"training"` is missing, so the second `.get` has something to call on. It's the Python spelling of Kotlin's `response["training"]?.get("epochs") ?: 10`. Use it for optional fields only; a required field should fail loudly with `[]`.

!!! warning "Common mistake: the key exists, but holds `null`"
    JSON's `null` becomes Python's `None`. Real APIs often send `"training": null` rather than leaving the field out, and then the default in `.get("training", {})` isn't used, because the key **is** there:

    ```pycon
    >>> response = {"model": "mobilenet_v3", "training": None}
    >>> response.get("training", {}).get("epochs", 10)
    Traceback (most recent call last):
      File "<python-input-1>", line 1, in <module>
        response.get("training", {}).get("epochs", 10)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AttributeError: 'NoneType' object has no attribute 'get'
    ```

    The fix uses `or`, which returns its right operand when the left one is falsy (Day 2). Both a missing key and a `None` value then become `{}`:

    ```python
    nulls = {"model": "mobilenet_v3", "training": None}
    (nulls.get("training") or {}).get("epochs", 10)       # 10
    (response.get("training") or {}).get("epochs", 10)    # 10   still works when the key is missing
    ```

For a quick look at a big nested structure, `pprint` (pretty-print) wraps it over several lines:

```python
from pprint import pprint
pprint(response, width=50)
# {'input': {'dtype': 'float32',
#            'shape': [1, 224, 224, 3]},
#  'labels': ['cat', 'dog', 'fox'],
#  'metrics': {'top1': 0.752, 'top5': 0.925},
#  'model': 'mobilenet_v3'}
```

`pprint` sorts dict keys alphabetically by default; pass `sort_dicts=False` to keep insertion order.

!!! android "Android bridge"
    A label map is the Python twin of the `labels.txt` you ship next to a `.tflite` model: line `i` of the file is the name of output index `i`. In Kotlin you'd read it into a `List<String>` and index it with the `argmax` of the output array. In Python the same thing is `labels[i]` on a list, or `{i: name for i, name in enumerate(labels)}` when you want a dict (comprehensions are coming up). The inverse map, `{name: i ...}`, is what you use to turn training labels into the integers a loss function needs.

## Sets

### What a set is

A **set** is an unordered collection of **unique**, hashable items. It's a dict with keys and no values, built on the same hash table, so `in` is O(1) on average.

Why it exists: removing duplicates, fast membership tests ("have I seen this file?"), and comparing groups ("which test images also appear in the training set?").

```kotlin
val seen = mutableSetOf("cat", "dog")
seen.add("cat")                 // false: already there
val a = setOf(1, 2, 3); val b = setOf(2, 3, 4)
a union b; a intersect b; a subtract b
```

```python
labels = {"cat", "dog", "cat"}     # duplicates collapse
len(labels)                        # 2
"cat" in labels                    # True
set([3, 1, 3, 2])                  # {1, 2, 3}   from any iterable
set()                              # set()       the ONLY way to write an empty set
type({})                           # <class 'dict'>   {} is an empty DICT

labels.add("fox")                  # add one item (no effect if present)
labels.discard("owl")              # remove if present; silent if not
labels.remove("dog")               # remove; KeyError if absent
sorted(labels)                     # ['cat', 'fox']
```

**Sets have no order.** Iteration order depends on the hashes, and string hashes change every time Python starts, so a set of strings can print in a different order on every run. That's why this page shows `sorted(labels)` rather than `labels`: whenever you display a set, or depend on its order, sort it first.

```pycon
>>> seen = set()
>>> seen.remove("cat")
Traceback (most recent call last):
  File "<python-input-1>", line 1, in <module>
    seen.remove("cat")
    ~~~~~~~~~~~^^^^^^^
KeyError: 'cat'
```

Use `remove` when the item must be there, and `discard` when it may not be.

### Set algebra

Four operations compare two sets. Each has an operator, which needs both sides to be sets, and a method, which accepts any iterable as its argument:

| Operation | Operator | Method | Kotlin | Meaning |
|---|---|---|---|---|
| union | `a | b` | `a.union(b)` | `a union b` | in either |
| intersection | `a & b` | `a.intersection(b)` | `a intersect b` | in both |
| difference | `a - b` | `a.difference(b)` | `a subtract b` | in `a`, not in `b` |
| symmetric difference | `a ^ b` | `a.symmetric_difference(b)` | (none) | in exactly one |

Here they are on a real question: are any test images leaking into the training set?

```python
train = {"img_01", "img_02", "img_03", "img_04"}
test = {"img_04", "img_05", "img_06"}

sorted(train | test)     # ['img_01', 'img_02', 'img_03', 'img_04', 'img_05', 'img_06']
sorted(train & test)     # ['img_04']                      the leak
sorted(test - train)     # ['img_05', 'img_06']            clean test images
sorted(train ^ test)     # ['img_01', 'img_02', 'img_03', 'img_05', 'img_06']

{"img_01", "img_02"} <= train        # True    subset: every item is also in train
train.isdisjoint(test)               # False   disjoint = no items in common
train.intersection(["img_02", "img_09"])   # {'img_02'}   methods accept a list
```

The operator/method split matters when one side is a list:

```pycon
>>> train = {"img_01", "img_02"}
>>> train | ["img_09"]
Traceback (most recent call last):
  File "<python-input-1>", line 1, in <module>
    train | ["img_09"]
    ~~~~~~^~~~~~~~~~~~
TypeError: unsupported operand type(s) for |: 'set' and 'list'
```

```python
sorted(train.union(["img_09"]))      # ['img_01', 'img_02', 'img_03', 'img_04', 'img_09']
sorted(train | set(["img_09"]))      # ['img_01', 'img_02', 'img_03', 'img_04', 'img_09']   convert the list first
```

`&=`, `|=`, `-=` and `^=` update a set in place. Underneath, `a & b` loops over the smaller set and does a hash lookup in the other for each item, so it costs about `min(len(a), len(b))` lookups rather than `len(a) × len(b)` comparisons.

!!! tip "De-duplicating while keeping order"
    `list(set(xs))` removes duplicates but scrambles the order. Since dict keys are unique **and** ordered, `list(dict.fromkeys(xs))` removes duplicates and keeps the first occurrence of each:

    ```python
    files = ["b.png", "a.png", "b.png", "c.png", "a.png"]
    list(dict.fromkeys(files))     # ['b.png', 'a.png', 'c.png']
    ```

### `frozenset`

A **`frozenset`** is an immutable set: same operations, no `add`, `remove` or `discard`. Because it can't change, it's hashable, so it can be a dict key or live inside another set. Use it for "an unordered group of things" as a key, such as a pair of classes the model confuses, where `{cat, dog}` and `{dog, cat}` must be the same key:

```python
confusions = {frozenset({"cat", "dog"}): 14}
confusions[frozenset({"dog", "cat"})]     # 14   order doesn't matter
frozenset({1, 2}) | {3}                   # frozenset({1, 2, 3})   operations return new frozensets
```

Kotlin doesn't need a separate type: `setOf(...)` is a read-only view, and any `Set` (even a `MutableSet`) can be a map key or sit inside another set, with the mutation risk shown in "Hashability" below. Python splits the two jobs into `set` (changeable, not a key) and `frozenset` (fixed, usable as a key).

Two errors you'll meet. A plain set can't go inside a set, because it isn't hashable:

```pycon
>>> {{1, 2}, {3}}
Traceback (most recent call last):
  File "<python-input-0>", line 1, in <module>
    {{1, 2}, {3}}
TypeError: cannot use 'set' as a set element (unhashable type: 'set')
```

and a frozenset has no methods that change it:

```pycon
>>> classes = frozenset({"cat", "dog"})
>>> classes.add("fox")
Traceback (most recent call last):
  File "<python-input-1>", line 1, in <module>
    classes.add("fox")
    ^^^^^^^^^^^
AttributeError: 'frozenset' object has no attribute 'add'
```

The fixes: `{frozenset({1, 2}), frozenset({3})}` for a set of sets, and `classes | {"fox"}` to get a new, bigger frozenset.

## Hashability

### What it means and why lists aren't allowed

An object is **hashable** if it has a hash value that never changes during its lifetime and it can be compared with `==`. Every immutable built-in type is hashable: `int`, `float`, `str`, `bool`, `None`, `tuple` (if everything inside is hashable) and `frozenset`. The mutable ones, `list`, `dict` and `set`, are not.

Why the rule exists follows from the hash table above. A key is filed in the slot its hash picks. If you could put a list in as a key and then `append` to it, its contents (and so its hash) would change, but it would still sit in the slot chosen by its **old** hash. Looking up the new contents would search the wrong slot; looking up the old contents would find the slot but fail the `==` check. The entry would be lost while still taking up space. Python prevents this by refusing mutable keys up front:

```pycon
>>> cache = {}
>>> cache[[224, 224]] = "resized"
Traceback (most recent call last):
  File "<python-input-1>", line 1, in <module>
    cache[[224, 224]] = "resized"
    ~~~~~^^^^^^^^^^^^
TypeError: cannot use 'list' as a dict key (unhashable type: 'list')
```

Python 3.13 and earlier print the shorter `TypeError: unhashable type: 'list'`, the message you'll see quoted all over the internet. Both mean the same thing. A tuple is only as hashable as its contents:

```pycon
>>> pair_with_list = (1, [2, 3])
>>> {pair_with_list: "x"}
Traceback (most recent call last):
  File "<python-input-1>", line 1, in <module>
    {pair_with_list: "x"}
TypeError: cannot use 'tuple' as a dict key (unhashable type: 'list')
```

The fix is to convert the key to its immutable twin: a list becomes a tuple, a set becomes a frozenset.

```python
cache = {}
cache[tuple([224, 224])] = "resized"     # or write the tuple directly: (224, 224)
cache[(224, 224)]                        # 'resized'
hash((224, 224)) == hash((224, 224))     # True    equal tuples, equal hashes
```

Kotlin has no such guard, which makes the danger easy to see:

```kotlin
val key = mutableListOf(224, 224)
val cache = hashMapOf(key to "resized")
key.add(3)
cache[key]                     // null: looked in the bucket for [224, 224, 3]
cache[listOf(224, 224)]        // null: right bucket, but equals() fails against the changed key
```

The Java `Map` documentation warns that a map's behaviour is unspecified if a key changes in a way that affects `equals`. Python turns that warning into a `TypeError`. (Your own Python classes are hashable by identity unless you define `__eq__`; Day 3 shows how `__eq__` and `__hash__` work together.)

## Copying: assignment, shallow copy, deep copy

### Three levels of "copy"

Day 2 showed that `b = a` copies nothing: it puts a second name on the same object. When an object contains other objects, there are two more levels:

- A **shallow copy** creates a new outer container whose slots point at the **same** inner objects. `xs.copy()`, `list(xs)`, `xs[:]`, `d.copy()`, `dict(d)`, `set(s)` and `copy.copy(x)` are all shallow.
- A **deep copy** creates a new outer container **and**, recursively, new copies of everything inside it that is mutable. `copy.deepcopy(x)` does this.

Why it matters: a training script that tweaks a "copy" of a config, or augments a "copy" of a list of samples, and silently changes the original, produces results nobody can reproduce.

<figure class="diagram" markdown>
![Three panels. Alias: names config and alias both point at one dict, which points at one layers list. Shallow copy: config and shallow point at two different dicts, but both dicts point at the same layers list. Deep copy: config and deep point at two different dicts, each with its own layers list](../../assets/img/phase-1/day02b-copy-levels.svg)
<figcaption>Assignment shares everything; a shallow copy shares the insides; a deep copy shares nothing mutable.</figcaption>
</figure>

```python
import copy      # the standard-library module with copy() and deepcopy()

config = {"name": "mobilenet", "layers": [32, 64]}
alias = config
shallow = config.copy()           # same as dict(config) or copy.copy(config)
deep = copy.deepcopy(config)

config["name"] = "resnet"         # rebind a key in the ORIGINAL outer dict
config["layers"].append(128)      # mutate the inner list, which some copies share

alias      # {'name': 'resnet', 'layers': [32, 64, 128]}       same object as config
shallow    # {'name': 'mobilenet', 'layers': [32, 64, 128]}    own outer dict, shared list
deep       # {'name': 'mobilenet', 'layers': [32, 64]}         shares nothing mutable

alias is config, shallow is config, shallow["layers"] is config["layers"]   # (True, False, True)
deep["layers"] is config["layers"]                                          # False
```

Read the two changes separately. `config["name"] = "resnet"` points one slot of the original dict at a new string; `shallow` has its own outer dict, so its `"name"` slot still points at `"mobilenet"`. `config["layers"].append(128)` doesn't touch any dict: it changes the one list that both `config` and `shallow` point at. Only `deep` has a separate list.

In Kotlin, `toMutableList()`, `toMutableMap()` and a data class's `copy()` are all shallow, exactly like Python's `copy()`. Kotlin has no built-in deep copy; you write one by hand. A shallow copy is enough whenever the items inside are immutable (numbers, strings, tuples of those), which is why `nums[:]` is a perfectly good copy of a list of floats.

### The `[[0] * 3] * 3` trap, traced

Here is the most famous bug in Python collections:

```python
grid = [[0] * 3] * 3
grid[0][0] = 1
grid          # [[1, 0, 0], [1, 0, 0], [1, 0, 0]]   all three rows changed!
```

<figure class="diagram" markdown>
![Animation: on the left, the outer list of grid has three slots whose arrows all point to one inner list [0, 0, 0]; setting grid[0][0] = 1 changes that single inner list, so every row shows 1. On the right, a comprehension builds three separate inner lists, and the same assignment changes only the first](../../assets/img/phase-1/day02b-grid-trap.svg)
<figcaption>Animated: <code>* 3</code> copies the arrow, not the row. A comprehension makes a new row each time.</figcaption>
</figure>

Trace it step by step:

1. `[0] * 3` runs **once**. It creates one list object, call it `R`, holding three references to the int `0`: `R = [0, 0, 0]`.
2. `[R] * 3` builds the outer list by repeating its one **reference** three times. The outer list is `[R, R, R]`: three arrows to the same row.
3. `grid[0][0] = 1` follows arrow 0 to `R` and sets `R[0] = 1`.
4. Printing `grid` follows each of the three arrows, and each leads to `R`, which is now `[1, 0, 0]`.

`id` (Day 2) confirms there is one row, not three:

```python
[id(row) == id(grid[0]) for row in grid]    # [True, True, True]
```

Why was `[0] * 3` fine in step 1? It also repeats one reference, to the int `0`. But ints are immutable: `R[0] = 1` doesn't change the `0` object, it points slot 0 of `R` at a different object, `1`. Repetition with `*` is safe for immutable items and dangerous for mutable ones.

The fix is to create a new row for each position, with a comprehension (explained in the next section). The expression `[0] * 3` is then evaluated once per row:

```python
grid = [[0] * 3 for _ in range(3)]
grid[0][0] = 1
grid          # [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
```

!!! warning "Common mistake: the same trap in `dict.fromkeys`"
    `dict.fromkeys(keys, value)` stores the **same** value object under every key, so a mutable default is shared:

    ```python
    per_class = dict.fromkeys(["cat", "dog"], [])
    per_class["cat"].append(0.9)
    per_class           # {'cat': [0.9], 'dog': [0.9]}   one list, two keys
    ```

    The output looks like a data bug ("why does every class have the same scores?"), not a crash. Fix it with a comprehension, `{k: [] for k in ["cat", "dog"]}`, or with `defaultdict(list)`.

## Comprehensions

### List comprehensions

A **list comprehension** builds a new list by running an expression for each item of an iterable, optionally keeping only items that pass a test. It's Python's `map` and `filter` in one piece of syntax.

Why it exists: transforming and filtering collections is constant in data work, and a comprehension says "the list of X for each Y where Z" in one readable line, instead of four lines of `result = []`, `for`, `if`, `append`.

```kotlin
val squares = (1..6).filter { it % 2 == 0 }.map { it * it }    // [4, 16, 36]
```

```python
squares = [n * n for n in range(1, 7) if n % 2 == 0]
squares       # [4, 16, 36]
```

Read it in three parts: `n * n` is **what** to produce; `for n in range(1, 7)` is **where** the items come from; `if n % 2 == 0` is **which** ones to keep. It runs as if you had written this loop:

```python
squares = []
for n in range(1, 7):
    if n % 2 == 0:
        squares.append(n * n)
squares       # [4, 16, 36]
```

Trace: `n = 1`, `1 % 2` is 1, so the test fails and nothing is added. `n = 2`, test passes, `4` is appended. `n = 3`, skip. `n = 4`, append `16`. `n = 5`, skip. `n = 6`, append `36`. Done: `[4, 16, 36]`.

More shapes you'll write every day:

```python
names = ["  Cat", "dog ", " FOX "]
[s.strip().lower() for s in names]                 # ['cat', 'dog', 'fox']    map only
[s for s in ["a.png", "b.txt", "c.png"] if s.endswith(".png")]   # ['a.png', 'c.png']  filter only
[(i, w) for i, w in enumerate(["cat", "dog"])]     # [(0, 'cat'), (1, 'dog')]   unpack in the for
```

Unlike Kotlin's `filter { }.map { }` chain on a list, which builds an intermediate list at each step, a comprehension makes one pass and one list. (Kotlin's lazy version, `asSequence().filter { }.map { }`, corresponds to the generator expressions previewed at the end of this section.)

### `if` at the end vs `if ... else` at the front

There are two different `if`s, and where they go matters:

- `if` **at the end** filters: items that fail are dropped, so the result can be shorter.
- `a if cond else b` **at the front** is a conditional expression (Day 2): it chooses what to produce, and every item produces something, so the result has the same length.

```python
vals = [-2, 5, -1, 3]
[v for v in vals if v > 0]            # [5, 3]          filter: 2 items
[v if v > 0 else 0 for v in vals]     # [0, 5, 0, 3]    map with a choice: 4 items (a ReLU!)
```

!!! warning "Common mistake: `else` in the filter position"
    A filter can't have an `else`, because a dropped item has no value to produce:

    ```pycon
    >>> vals = [-2, 5, -1]
    >>> [v for v in vals if v > 0 else 0]
      File "<python-input-1>", line 1
        [v for v in vals if v > 0 else 0]
                                  ^^^^
    SyntaxError: invalid syntax
    ```

    A `SyntaxError` is reported before anything runs, and the `^^^^` points at the first word Python couldn't fit. If you want an `else`, move the whole `a if cond else b` to the front.

### Dict and set comprehensions

The same syntax in curly braces builds a dict (with `key: value`) or a set (with a single expression):

```python
labels = ["cat", "dog", "fox"]
label_to_idx = {name: i for i, name in enumerate(labels)}
label_to_idx                  # {'cat': 0, 'dog': 1, 'fox': 2}
idx_to_label = {i: name for name, i in label_to_idx.items()}    # invert a dict
idx_to_label                  # {0: 'cat', 1: 'dog', 2: 'fox'}

{len(w) for w in ["cat", "dog", "mouse"]}     # {3, 5}   a set: duplicates collapse
{k: v for k, v in {"lr": 0.1, "seed": None}.items() if v is not None}   # {'lr': 0.1}
```

`{name: i for ...}` is Kotlin's `labels.withIndex().associate { (i, name) -> name to i }`; inverting with `items()` is `entries.associate { (k, v) -> v to k }`. When inverting, duplicate values collapse: if two keys share a value, the later one wins.

### Nested comprehensions

A comprehension can have several `for` clauses. They nest **in the same order you'd write the loops**, outermost first:

```python
image = [[1, 2, 3],
         [4, 5, 6]]
flat = [px for row in image for px in row]
flat          # [1, 2, 3, 4, 5, 6]
```

That is exactly:

```python
flat = []
for row in image:          # first for = outer loop
    for px in row:         # second for = inner loop
        flat.append(px)
flat          # [1, 2, 3, 4, 5, 6]
```

!!! warning "Common mistake: `for` clauses in reading order"
    It's tempting to write "each pixel, for each pixel in the row, for each row" with the clauses in that order. But the first `for` runs first, and at that moment `row` doesn't exist yet:

    ```pycon
    >>> image = [[1, 2, 3], [4, 5, 6]]
    >>> [px for px in row for row in image]
    Traceback (most recent call last):
      File "<python-input-1>", line 1, in <module>
        [px for px in row for row in image]
                      ^^^
    NameError: name 'row' is not defined. Did you mean: 'pow'?
    ```

    It's worse in a notebook, or on this page, where an earlier loop left a `row` behind (a `for` loop's variable outlives the loop: `row` is still `[4, 5, 6]` from the block above). Then it runs and is silently wrong:

    ```python
    [px for px in row for row in image]    # [4, 4, 5, 5, 6, 6]   wrong: each px of the OLD row, once per row of image
    ```

    The rule: write the `for` clauses in the same order as the nested loops, outermost first.

That's Kotlin's `image.flatten()`, or `flatMap { it }`. A comprehension can also sit **inside** another one's expression, which builds nested lists. Here's a transpose:

```python
[[row[c] for row in image] for c in range(3)]    # [[1, 4], [2, 5], [3, 6]]
```

Trace from the outside in: the outer comprehension runs `c = 0, 1, 2`. For `c = 0`, the inner comprehension `[row[0] for row in image]` takes item 0 of each row: `[1, 4]`. For `c = 1` it gives `[2, 5]`, and for `c = 2`, `[3, 6]`. Each inner result becomes one row of the outer list. (NumPy's `.T` does the same without a loop, on Day 8.)

You'll also see the transpose written with `zip`. `zip(a, b)` (Day 2) pairs up item 0 of each argument, then item 1, and so on. A `*` in front of an argument at a call site spreads a list out into separate arguments (Day 3 covers it), so `zip(*image)` is `zip([1, 2, 3], [4, 5, 6])`:

```python
list(zip(*image))                  # [(1, 4), (2, 5), (3, 6)]   columns, as tuples
app_names, app_ratings = zip(*[("Lens", 4.6), ("Keep", 4.5)])   # "unzip" pairs into two tuples
app_names, app_ratings             # (('Lens', 'Keep'), (4.6, 4.5))
```

### When a loop is clearer

A comprehension is for **building a collection**. Prefer a regular `for` loop when:

- you need more than one `if` and a nested `for`, or anything you have to read twice to follow;
- the body has side effects such as `print`, writing a file or updating a counter (a comprehension used only for its side effects builds a list of `None` you throw away);
- you need `break`, or several statements per item.

```python
# Hard to read: three clauses and a condition on both loop variables
pairs = [(a, b) for a in range(4) for b in range(4) if a < b if (a + b) % 2 == 1]
pairs         # [(0, 1), (0, 3), (1, 2), (2, 3)]
```

That works, but a loop with a named condition, or a small helper function (Day 3), would be kinder to the next reader.

A comprehension's loop variable stays **inside** it. A name defined before is untouched:

```python
n = "before"
[n * n for n in range(3)]    # [0, 1, 4]   inside, n is 0, 1, 2
n                            # 'before'    outside, the old n is untouched
for n in range(3):
    pass                     # pass is a statement that does nothing
n                            # 2           a for loop's variable keeps its last value
```

The comprehension's `n` lives in its own small scope that disappears when it finishes. A plain `for` loop assigns to the surrounding scope, which is the leak Day 2 showed and the reason the nested-comprehension mistake above ran silently on this page.

### Generator expressions, a preview

Swap the square brackets for parentheses and you get a **generator expression**: it produces items one at a time, when asked, instead of building the whole list in memory. When the generator expression is a call's **only** argument, the call's own parentheses are enough:

```python
sum(n * n for n in range(1, 7) if n % 2 == 0)    # 56   no list is built
max(len(w) for w in ["cat", "mouse", "ox"])      # 5
```

With any other argument next to it, the generator needs its own parentheses:

```pycon
>>> xs = [3, -1, 2]
>>> sorted(x * 2 for x in xs, key=abs)
  File "<python-input-1>", line 1
    sorted(x * 2 for x in xs, key=abs)
           ^^^^^^^^^^^^^^^^^
SyntaxError: Generator expression must be parenthesized
```

```python
xs = [3, -1, 2]
sorted((x * 2 for x in xs), key=abs)    # [-2, 4, 6]
```

You've already used these with `any` and `all`, and with `join` in Day 2a. Day 3d explains how they work (iterators, `yield`, and why a generator can only be used once). For now: use one when you only feed the values to a function that consumes them, such as `sum`, `max`, `any`, `all`, `sorted` or `"".join`.

## Choosing a collection

### What `in` really costs, measured

You've seen why `in` is O(n) on a list (compare item after item) and O(1) on average for a set or dict (one hash, one slot). Here are real numbers. `timeit.timeit(stmt, globals=..., number=k)` runs the code in the string `stmt` `k` times and returns the total time in seconds; `globals=globals()` lets that code see this page's variables.

```python
import timeit

n = 100000
ids_list = list(range(n))
ids_set = set(ids_list)
target = n - 1                     # the worst case for a list: the very last item

reps = 200
t_list = timeit.timeit("target in ids_list", globals=globals(), number=reps) / reps
t_set = timeit.timeit("target in ids_set", globals=globals(), number=reps) / reps
print(f"list: {t_list * 1e6:9.1f} µs per lookup")
print(f"set:  {t_set * 1e6:9.3f} µs per lookup")
print(f"the set is about {t_list / t_set:,.0f}× faster")
# list:     502.2 µs per lookup
# set:      0.033 µs per lookup
# the set is about 15,288× faster
```

Those timings come from the machine this lesson was written on; yours will differ, and the browser's Python is slower again. The **shape** won't change: double `n` and the list time doubles while the set time stays flat. The usual pattern is to build the set once, then test against it many times. Converting a list to a set costs one O(n) pass, so it only pays off if you do more than a few lookups.

The same reasoning applies to removing from the front of a list. If you need a queue (add at one end, take from the other), use `collections.deque`, a double-ended queue with O(1) `append`, `appendleft`, `pop` and `popleft`. With `maxlen` it keeps only the newest items, which makes a neat running window over training losses. It's Kotlin's `ArrayDeque` (`addFirst`, `addLast`, `removeFirst`, `removeLast`), with `maxlen` as an extra:

```python
from collections import deque

recent = deque(maxlen=3)
for loss in [1.0, 0.75, 0.5, 0.375, 0.25]:
    recent.append(loss)          # once full, the oldest item drops off the left
recent                           # deque([0.5, 0.375, 0.25], maxlen=3)
sum(recent) / len(recent)        # 0.375   moving average of the last 3 losses
recent.popleft()                 # 0.5
```

### The decision table

Now that each type has been taught, here's the summary:

| You need… | Use | Why |
|---|---|---|
| items in order, added over time, read by position | `list` | amortised O(1) `append`, O(1) index |
| a fixed record, a shape, or a dict key made of several parts | `tuple` | immutable and hashable |
| a record with readable field names | `namedtuple` (or Day 3's dataclass) | `p.score` instead of `p[1]` |
| look something up by a key | `dict` | O(1) average `[]`, `get`, `in` |
| group items by a key | `defaultdict(list)` | no "is this key new?" check |
| count things | `Counter` | counts, `most_common`, missing = 0 |
| unique items, or fast "have I seen this?" | `set` | O(1) average `in`, set algebra |
| an unordered group as a dict key | `frozenset` | hashable set |
| add and remove at both ends | `deque` | O(1) at both ends; list is O(n) at the front |
| millions of numbers to do maths on | NumPy array (Day 4) | compact memory, fast vectorised maths |

And the cost of common operations (average case; `n` is the collection's size):

| Operation | `list` | `dict` / `set` |
|---|---|---|
| `x in c` | O(n) | O(1) |
| `c[i]` / `d[key]` | O(1) | O(1) |
| `append` / `d[key] = v` / `add` | O(1) amortised | O(1) |
| `insert(0, x)` / `pop(0)` | O(n) | n/a |
| `pop()` from the end | O(1) | n/a |
| `sort()` | O(n log n) | n/a |

### Kotlin collections, mapped

Every row below was taught above; this is the phrasebook for when you're translating.

| Kotlin | Python |
|---|---|
| `listOf(1, 2)` / `mutableListOf(1, 2)` | `(1, 2)` or `[1, 2]` / `[1, 2]` |
| `mapOf("a" to 1)` / `mutableMapOf(...)` | `{"a": 1}` |
| `setOf(1, 2)` / `mutableSetOf(...)` | `frozenset({1, 2})` or `{1, 2}` / `{1, 2}` |
| `emptyList()` / `emptyMap()` / `emptySet()` | `[]` / `{}` / `set()` |
| `list.size`, `list.isEmpty()` | `len(xs)`, `not xs` |
| `list.map { it * 2 }` | `[x * 2 for x in xs]` |
| `list.filter { it > 0 }` | `[x for x in xs if x > 0]` |
| `list.flatMap { it }` / `flatten()` | `[y for x in xs for y in x]` |
| `list.groupBy { it.category }` | `defaultdict(list)` and a loop |
| `list.groupingBy { it }.eachCount()` | `Counter(xs)` |
| `list.associate { it.id to it.name }` | `{x.id: x.name for x in xs}` |
| `list.associateWith { it.length }` | `{x: len(x) for x in xs}` |
| `list.withIndex()` | `enumerate(xs)` |
| `a.zip(b)` | `zip(a, b)` |
| `list.sortedBy { it.score }` | `sorted(xs, key=lambda x: x.score)` |
| `list.sortedByDescending { ... }` | `sorted(xs, key=..., reverse=True)` |
| `list.distinct()` | `list(dict.fromkeys(xs))` |
| `list.any { ... }` / `all { ... }` / `count { ... }` | `any(...)` / `all(...)` / `sum(1 for x in xs if ...)` |
| `list.sum()` / `maxOrNull()` / `maxBy { ... }` | `sum(xs)` / `max(xs, default=None)` / `max(xs, key=...)` |
| `map.getOrDefault(k, d)` / `map[k] ?: d` | `d.get(k, default)` |
| `map.getOrPut(k) { mutableListOf() }` | `d.setdefault(k, [])` |
| `a union b`, `a intersect b`, `a subtract b` | `a | b`, `a & b`, `a - b` |
| `list.subList(1, 4)` (a view) | `xs[1:4]` (a copy) |
| `val (a, b) = pair` | `a, b = pair` |

`max(xs, default=None)` returns the `default` instead of raising `ValueError` on an empty sequence, which matches `maxOrNull()`.

## Going further

- [`bisect`](https://docs.python.org/3/library/bisect.html): binary search and insertion into a sorted list in O(log n).
- [`heapq`](https://docs.python.org/3/library/heapq.html): a priority queue on a plain list; `heapq.nlargest(k, xs)` for top-k.
- [`collections.OrderedDict` and `ChainMap`](https://docs.python.org/3/library/collections.html): an order-aware dict with `move_to_end` (handy for an LRU cache), and layered lookups across several dicts.
- [`operator.itemgetter`](https://docs.python.org/3/library/operator.html#operator.itemgetter): a faster, named alternative to `key=lambda x: x[2]`.
- [`array`](https://docs.python.org/3/library/array.html): compact arrays of one numeric type; in practice you'll use NumPy (Day 4) instead.
- [Dict views as sets](https://docs.python.org/3/library/stdtypes.html#dictionary-view-objects): `d1.keys() & d2.keys()` finds the keys two dicts share.
- [Time complexity of built-in operations](https://wiki.python.org/moin/TimeComplexity): the full cost table for list, deque, dict and set.
- Coming in this course: type hints such as `list[int]` and `dict[str, float]` and dataclasses ([Day 3](day-03-functions-classes.md)); the `json` module ([Day 3c](day-03c-files-json.md)); iterators and generators ([Day 3d](day-03d-iterators-generators.md)).

---

## Exercises

1. **Warm-up: list methods.** Start from `queue = ["img_03.png", "img_01.png", "img_02.png"]`. In order: add `"img_04.png"` at the end; add `"img_00.png"` at the front; remove `"img_02.png"` by value; take the last item off and print it; sort the list in place. Print the list after each step.
2. **Warm-up: counting two ways.** Count the labels in `preds = ["cat", "dog", "cat", "fox", "cat", "dog"]` first with a plain dict and `get`, then with `Counter`. Print the most common label and its count.
3. **Label maps.** From `labels = ["cat", "dog", "fox"]`, build `label_to_idx` and `idx_to_label` with dict comprehensions. Encode `["dog", "dog", "fox", "cat"]` to integers and decode the result back. Then encode `["dog", "owl"]`, mapping any unknown label to `-1` instead of crashing.
4. **Leakage check.** `train_ids = [101, 102, 103, 104, 105, 103]` and `test_ids = [105, 106, 107, 101, 107]`. Find: the ids duplicated inside `train_ids`; the ids that appear in both lists; `test_ids` de-duplicated with its order kept; and that de-duplicated test list with every leaked id removed.
5. **Group and rank.** Given the `apps` list of `(name, category, rating)` tuples below, group app names by category with `defaultdict`, then for each category in alphabetical order print its best app (highest rating, ties broken by name A→Z) and its average rating to two decimals.

    ```python
    apps = [
        ("Snapseed", "photo", 4.5), ("Lens", "photo", 4.6), ("Keep", "tools", 4.5),
        ("Files", "tools", 4.6), ("Gallery", "photo", 4.5), ("Clock", "tools", 4.2),
        ("Maps", "travel", 4.3),
    ]
    ```

6. **Predict the output.** Write down what each `print` shows **before** running it. (This block has no Run button on purpose; the solution has a runnable copy.)

    ```text
    a = [1, 2, 3]
    b = a
    c = a[:]
    b.append(4)
    c.append(5)
    grid = [[0] * 2] * 2
    grid[1][0] = 7
    t = (a, 9)
    a.append(6)
    print(a, b, c)
    print(grid)
    print(t)
    print(len({1, 1.0, True, "1"}))
    ```

7. **Find and fix the bug.** This function should return each class's scores, sorted high to low. Run it, explain the output, and fix it (there are two bugs).

    ```python
    def per_class_scores(results, classes):
        """results: list of (label, score). Returns {label: [scores, high to low]}."""
        scores = dict.fromkeys(classes, [])
        for label, score in results:
            scores[label].append(score)
        for label in scores:
            scores[label] = scores[label].sort(reverse=True)
        return scores

    results = [("cat", 0.9), ("dog", 0.4), ("cat", 0.7), ("dog", 0.8)]
    per_class_scores(results, ["cat", "dog"])    # {'cat': None, 'dog': None}
    ```

8. **Challenge: a tiny tokenizer vocabulary.** Text models turn words into integer ids. Given

    ```python
    sentences = ["the cat sat", "the dog sat", "a cat ran", "the cat ran off fast"]
    ```

    build a vocabulary: count every word with `Counter`; keep words that appear at least twice; order them by count (high to low), breaking ties alphabetically; and give them ids starting at 2, because id 0 is reserved for `"<pad>"` and id 1 for `"<unk>"` (unknown). Then encode each sentence to a list of exactly 4 ids: unknown words become 1, short sentences are padded with 0 at the end, and long ones are cut to 4.

??? success "Solutions"
    **1. List methods.**

    ```python
    queue = ["img_03.png", "img_01.png", "img_02.png"]
    queue.append("img_04.png")
    print(queue)     # ['img_03.png', 'img_01.png', 'img_02.png', 'img_04.png']
    queue.insert(0, "img_00.png")
    print(queue)     # ['img_00.png', 'img_03.png', 'img_01.png', 'img_02.png', 'img_04.png']
    queue.remove("img_02.png")
    print(queue)     # ['img_00.png', 'img_03.png', 'img_01.png', 'img_04.png']
    last = queue.pop()
    print(last)      # img_04.png
    print(queue)     # ['img_00.png', 'img_03.png', 'img_01.png']
    queue.sort()
    print(queue)     # ['img_00.png', 'img_01.png', 'img_03.png']
    ```

    Each method changes `queue` in place and (except `pop`) returns `None`, which is why every step is its own statement and the result is never assigned. `pop` is the one that hands back the item it removed. Zero-padded numbers (`01`, `02`) make alphabetical order match numeric order; without the padding, `"img_10"` would sort before `"img_2"`.

    **2. Counting two ways.**

    ```python
    from collections import Counter

    preds = ["cat", "dog", "cat", "fox", "cat", "dog"]
    counts = {}
    for p in preds:
        counts[p] = counts.get(p, 0) + 1
    print(counts)                       # {'cat': 3, 'dog': 2, 'fox': 1}

    c = Counter(preds)
    print(c)                            # Counter({'cat': 3, 'dog': 2, 'fox': 1})
    label, n = c.most_common(1)[0]
    print(label, n)                     # cat 3
    ```

    `counts.get(p, 0)` reads the current count, or 0 the first time a label appears, so no `KeyError` is possible. `Counter` does the same loop for you. `most_common(1)` returns a list with one `(item, count)` tuple, so `[0]` takes that tuple and the assignment unpacks it.

    **3. Label maps.**

    ```python
    labels = ["cat", "dog", "fox"]
    label_to_idx = {name: i for i, name in enumerate(labels)}
    idx_to_label = {i: name for name, i in label_to_idx.items()}
    print(label_to_idx)                 # {'cat': 0, 'dog': 1, 'fox': 2}
    print(idx_to_label)                 # {0: 'cat', 1: 'dog', 2: 'fox'}

    encoded = [label_to_idx[w] for w in ["dog", "dog", "fox", "cat"]]
    print(encoded)                      # [1, 1, 2, 0]
    print([idx_to_label[i] for i in encoded])     # ['dog', 'dog', 'fox', 'cat']
    print([label_to_idx.get(w, -1) for w in ["dog", "owl"]])   # [1, -1]
    ```

    `enumerate` supplies `(index, name)` pairs, and the comprehension flips them into `name: index`. Inverting with `items()` is safe here because the indexes are unique. With `label_to_idx[w]`, an unknown `"owl"` would raise `KeyError: 'owl'`; `get(w, -1)` turns it into a sentinel value you can check for.

    **4. Leakage check.**

    ```python
    from collections import Counter

    train_ids = [101, 102, 103, 104, 105, 103]
    test_ids = [105, 106, 107, 101, 107]

    dupes = sorted(i for i, n in Counter(train_ids).items() if n > 1)
    print(dupes)                        # [103]
    leaked = set(train_ids) & set(test_ids)
    print(sorted(leaked))               # [101, 105]
    test_unique = list(dict.fromkeys(test_ids))
    print(test_unique)                  # [105, 106, 107, 101]
    clean_test = [i for i in test_unique if i not in leaked]
    print(clean_test)                   # [106, 107]
    ```

    `Counter` finds duplicates by counting, and the generator expression keeps the ids counted more than once. The intersection `&` finds ids in both sets, with one hash lookup per id. `dict.fromkeys` keeps the first occurrence of each id in order, which `set` wouldn't. The final filter uses `not in leaked` on a **set**, so each check is O(1).

    **5. Group and rank.**

    ```python
    from collections import defaultdict

    apps = [
        ("Snapseed", "photo", 4.5), ("Lens", "photo", 4.6), ("Keep", "tools", 4.5),
        ("Files", "tools", 4.6), ("Gallery", "photo", 4.5), ("Clock", "tools", 4.2),
        ("Maps", "travel", 4.3),
    ]
    by_cat = defaultdict(list)
    for name, category, rating in apps:
        by_cat[category].append((name, rating))

    for category in sorted(by_cat):
        members = by_cat[category]
        best_name, best_rating = sorted(members, key=lambda m: (-m[1], m[0]))[0]
        avg = sum(r for _, r in members) / len(members)
        print(f"{category:<7} best: {best_name} ({best_rating})  avg: {avg:.2f}")
    # photo   best: Lens (4.6)  avg: 4.53
    # tools   best: Files (4.6)  avg: 4.43
    # travel  best: Maps (4.3)  avg: 4.30
    ```

    `defaultdict(list)` removes the "is this category new?" check. `sorted(by_cat)` iterates the keys alphabetically. The key `(-rating, name)` sorts the highest rating first and breaks ties by name; `[0]` takes the best. `sum(r for _, r in members)` unpacks each `(name, rating)` pair, ignores the name, and adds the ratings.

    **6. Predict the output.**

    ```python
    a = [1, 2, 3]
    b = a
    c = a[:]
    b.append(4)
    c.append(5)
    grid = [[0] * 2] * 2
    grid[1][0] = 7
    t = (a, 9)
    a.append(6)
    print(a, b, c)
    print(grid)
    print(t)
    print(len({1, 1.0, True, "1"}))
    # [1, 2, 3, 4, 6] [1, 2, 3, 4, 6] [1, 2, 3, 5]
    # [[7, 0], [7, 0]]
    # ([1, 2, 3, 4, 6], 9)
    # 2
    ```

    `b` is a second name for `a`'s list, so both see `4` and `6`. `c = a[:]` is a copy made before either append, so it only gets its own `5`. Both rows of `grid` are one list, so setting `grid[1][0]` shows in row 0 too. The tuple `t` holds a reference to `a`'s list, not a snapshot, so the later `append(6)` shows through. Finally `1`, `1.0` and `True` are equal with equal hashes, so the set keeps one of them, plus the string `"1"`: two items.

    **7. Find and fix the bug.**

    ```python
    def per_class_scores(results, classes):
        """results: list of (label, score). Returns {label: [scores, high to low]}."""
        scores = {c: [] for c in classes}            # fix 1: a new list per class
        for label, score in results:
            scores[label].append(score)
        for label in scores:
            scores[label] = sorted(scores[label], reverse=True)   # fix 2: sorted() returns the list
        return scores

    results = [("cat", 0.9), ("dog", 0.4), ("cat", 0.7), ("dog", 0.8)]
    per_class_scores(results, ["cat", "dog"])    # {'cat': [0.9, 0.7], 'dog': [0.8, 0.4]}
    ```

    Bug 2 is the one you see: `list.sort()` sorts in place and returns `None`, and that `None` was stored under every key. Fixing only that bug reveals bug 1: every class would get `[0.9, 0.8, 0.7, 0.4]`, because `dict.fromkeys(classes, [])` puts **one** shared list under every key, so all four scores went into it. The comprehension creates a separate list per class. (Replacing values for existing keys inside `for label in scores` is allowed; only adding or removing keys during the loop is an error.)

    **8. Challenge: a tiny tokenizer vocabulary.**

    ```python
    from collections import Counter

    sentences = ["the cat sat", "the dog sat", "a cat ran", "the cat ran off fast"]
    counts = Counter(w for s in sentences for w in s.split())
    print(counts.most_common())
    # [('the', 3), ('cat', 3), ('sat', 2), ('ran', 2), ('dog', 1), ('a', 1), ('off', 1), ('fast', 1)]

    kept = [w for w, n in counts.items() if n >= 2]
    kept.sort(key=lambda w: (-counts[w], w))      # count high→low, then A→Z
    vocab = {"<pad>": 0, "<unk>": 1} | {w: i for i, w in enumerate(kept, start=2)}
    print(vocab)
    # {'<pad>': 0, '<unk>': 1, 'cat': 2, 'the': 3, 'ran': 4, 'sat': 5}

    MAX_LEN = 4
    def encode(sentence):
        ids = [vocab.get(w, vocab["<unk>"]) for w in sentence.split()]
        ids = ids[:MAX_LEN]                          # cut long sentences
        return ids + [vocab["<pad>"]] * (MAX_LEN - len(ids))   # pad short ones

    for s in sentences:
        print(f"{s!r:24} -> {encode(s)}")
    # 'the cat sat'            -> [3, 2, 5, 0]
    # 'the dog sat'            -> [3, 1, 5, 0]
    # 'a cat ran'              -> [1, 2, 4, 0]
    # 'the cat ran off fast'   -> [3, 2, 4, 1]
    ```

    The nested generator expression walks every word of every sentence. Sorting by `(-count, word)` puts `cat` and `the` (3 each) first, alphabetically, then `ran` and `sat`. `enumerate(kept, start=2)` numbers from 2, and `|` merges that dict after the two reserved tokens. In `encode`, `vocab.get(w, 1)` maps unknown words to `<unk>`, the slice `[:MAX_LEN]` truncates (a slice never fails when the list is shorter), and `[0] * (MAX_LEN - len(ids))` pads: when nothing is missing, that's `[0] * 0`, an empty list. `{s!r:24}` formats the sentence with its quotes (`!r` uses `repr`) in a 24-character column. This is a miniature of what real tokenizers do before text reaches a model.

## Resources

- [Python tutorial: Data Structures](https://docs.python.org/3/tutorial/datastructures.html): lists, tuples, sets, dicts, comprehensions and looping techniques, from the official tutorial.
- [Built-in Types: sequences, sets and mappings](https://docs.python.org/3/library/stdtypes.html): the reference for every method on `list`, `tuple`, `set` and `dict`.
- [`collections`: container datatypes](https://docs.python.org/3/library/collections.html): `defaultdict`, `Counter`, `deque`, `namedtuple` and more.
- [`copy`: shallow and deep copy operations](https://docs.python.org/3/library/copy.html): the exact rules for `copy.copy` and `copy.deepcopy`.
- [Sorting Techniques](https://docs.python.org/3/howto/sorting.html): the official how-to on `key`, stability and multi-key sorts.
- [Python Wiki: TimeComplexity](https://wiki.python.org/moin/TimeComplexity): the cost of every list, deque, dict and set operation.
- [Kotlin: Collections overview](https://kotlinlang.org/docs/collections-overview.html): the Kotlin side of today's mapping, to compare against.

## Checklist

- [ ] I can use `append`, `extend`, `insert`, `pop`, `remove`, `index`, `count`, `sort`, `reverse`, `clear` and `copy`, and I know `sort()` returns `None` while `sorted()` returns a new list.
- [ ] I can explain why `append` is amortised O(1) and `insert(0, x)` is O(n).
- [ ] I can predict any slice with negative indexes or steps, and I can grow, shrink and delete with slice assignment.
- [ ] I can write a one-element tuple, unpack with `*rest`, swap two names, and use a `namedtuple`.
- [ ] I know when to use `d[k]` vs `d.get(k, default)`, I can merge dicts with `|`, and I group with `defaultdict` and count with `Counter`.
- [ ] I can pull a value out of nested JSON-like data, with defaults for optional fields.
- [ ] I can use union, intersection, difference and symmetric difference, and I know `{}` is a dict.
- [ ] I can explain why a list can't be a dict key and fix the `unhashable type` error.
- [ ] I can tell alias, shallow copy and deep copy apart, and explain what `[[0] * 3] * 3` does wrong.
- [ ] I can write list, dict and set comprehensions with filters, conditional expressions and two `for` clauses, and I know when a loop is clearer.
- [ ] I can choose a collection for a task and justify it with the cost of `in`.
- [ ] I can translate `map`, `filter`, `groupBy`, `associate` and `eachCount` from Kotlin to Python.

**Next:** [Day 3: Functions, classes and modules](day-03-functions-classes.md)
