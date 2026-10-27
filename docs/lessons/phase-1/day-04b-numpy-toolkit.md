# Day 4b: The NumPy Toolkit

**Phase 1 · Week 1 · ~160 minutes**

This lesson builds on [Day 4](day-04-numpy-arrays.md), which built the array itself: creating arrays, `dtype`, `shape`, indexing, boolean masks, views, vectorised arithmetic and reductions. Today is the first half of the set of tools you reach for every day once you have an array: choosing values by a condition, finding and counting matches, sorting, running totals, missing values and comparing floats. [Day 4c](day-04c-random-io-stats.md) is the second half: random numbers, saving arrays, and statistics.

## Objectives

- [ ] Choose values element by element with `np.where(cond, a, b)`, explain why both `a` and `b` are computed in full, and use `np.select` for more than two cases.
- [ ] Find the positions where a condition holds with one-argument `np.where`, `np.nonzero` and `np.argwhere`, count them with `np.count_nonzero`, and know which of these can be used as an index.
- [ ] Limit values to a range with `np.clip`, and use it before converting floats to `uint8`.
- [ ] Sort with `np.sort` and `.sort()`, sort along the right axis, and sort in descending order.
- [ ] Use `np.argsort` to find the top-k items, to sort one array by another, to sort the rows of a table by a column, and to compute ranks.
- [ ] Count categories with `np.unique(..., return_counts=True)`, and encode labels as integers with `return_inverse=True`.
- [ ] Put values into buckets with `np.searchsorted`, and explain why its input must be sorted.
- [ ] Compute running totals and differences with `np.cumsum`, `np.cumprod` and `np.diff`, and predict their output shape.
- [ ] Explain where `nan` and `inf` come from, how `nan` spreads, and why `nan != nan`; detect them with `np.isnan`, `np.isinf` and `np.isfinite`; and summarise data that has gaps with `np.nanmean` and friends.
- [ ] Control NumPy's division-by-zero warnings with `np.errstate`, including turning them into errors while debugging.
- [ ] Compare float arrays with `np.isclose` and `np.allclose`, explain `rtol` and `atol`, and know when to use `np.array_equal` instead.

**Prerequisites.**

- [Day 2](day-02-python-types.md): `float`, `math.isclose` and its `abs_tol`, `math.inf` and `math.nan`, `nan == nan` being `False`, and integer division by zero raising `ZeroDivisionError`.
- [Day 2b](day-02b-collections.md): `sorted(..., key=..., reverse=True)` vs `list.sort()`, `Counter`, `dict(zip(...))`, one-element tuples, and the `[::-1]` slice.
- [Day 3](day-03-functions-classes.md): writing functions with default arguments and docstrings.
- [Day 3b](day-03b-errors-files-json.md): `try` / `except`, and catching a specific exception type.
- [Day 3c](day-03c-files-json.md): `with` and context managers.
- [Day 4](day-04-numpy-arrays.md): `np.array`, `arange`, `zeros_like`, `dtype` and `astype` (and that out-of-range casts aren't safe), `uint8` overflow, indexing and slicing, boolean masks with `&`, `|` and `~`, fancy indexing (indexing with an array of integers, which makes a copy), ufuncs such as `np.log` and `np.maximum`, reductions with `axis=`, `argmax`, and summing a boolean mask to count its `True`s.

!!! info "How to read the code on this page"
    Every Python block has a **▶ Run** button, and the blocks share their variables from top to bottom like notebook cells, so run them in order. A bare expression on its own line shows its value, exactly as the `# comment` under it says. Lines that fail on purpose are wrapped in `try` / `except` so the rest of the block still runs, and the comment shows the real error message. NumPy **warnings** (not errors) appear in the output above the value, as `<cell>:2: RuntimeWarning: ...`; the comments show them as `# RuntimeWarning: ...`.

    The outputs on this page come from 64-bit Python. The browser's Python is 32-bit, so NumPy's default integer there is `int32`: where this page shows `np.int64(3)`, as it does for counts and positions, ▶ Run shows `np.int32(3)`. The values are the same; only the type name differs. Float results are identical, and `print(...)` output looks the same everywhere.

---

## The big idea

Day 4 gave you the array and the habit of replacing loops with whole-array operations. But a real data script needs more than arithmetic. It needs to say "use this value where the reading is valid, and that one where it isn't"; "give me the five best scores and their names"; "how many of each label do I have?"; "this column has holes in it"; and "are these two results the same, give or take rounding?".

Each of those is one NumPy function. Each replaces a loop you'd otherwise write in Kotlin with an `if`, a `sortedByDescending`, a `groupingBy`, or a null check. Learning them is like learning the Kotlin standard library's collection functions: once you know `map`, `filter`, `sortedBy` and `groupBy`, you stop writing loops for them, and the same thing happens here.

**Analogy.** Day 4 gave you a spreadsheet's cells and formulas. Today you get its toolbar: the IF function, the Sort button, the COUNTIF, and the "blank cell" handling. None of them is deep on its own, but every one has a trap, and the traps are what this lesson spends its time on. In ML code they are silent: a wrong sort axis, a `nan` that turns your loss into `nan`, or a "missing value" check that never matches doesn't crash. It trains a model on garbage.

The lesson has five parts:

1. Choosing values by a condition: `where`, `clip`, `select`.
2. Sorting and counting: `sort`, `argsort`, `unique`, `searchsorted`.
3. Running totals and differences: `cumsum`, `cumprod`, `diff`.
4. Missing and special values: `nan`, `inf`, the `nan*` functions, `errstate`.
5. Comparing floats: `isclose`, `allclose`, `array_equal`.

[Day 4c](day-04c-random-io-stats.md) finishes the toolkit: random numbers and train/test splits, saving arrays to disk, rounding, percentiles and histograms, and a cleaning pipeline that uses all of it.

---

**Part 1: Choosing values by a condition**

## `np.where` with three arguments

**What it is.** `np.where(cond, a, b)` builds a new array. In each position, it takes the value from `a` where `cond` is `True`, and from `b` where `cond` is `False`. `cond` is a boolean array (Day 4's masks); `a` and `b` are arrays of the same shape, or anything that **broadcasts** to it. Broadcasting gets its own lesson on Day 9; for now you need one case: a single number, such as `0.0`, counts as "that number in every position".

**Why it exists.** It's the vectorised form of `if / else`. You'll use it to replace invalid readings with a fallback, to apply a different formula on each side of a threshold (ReLU, clipping a loss, the Huber loss later in the course), and to turn a sentinel value like `-1` into a proper "missing" marker.

**Kotlin comparison.** In Kotlin you'd map over the list with an `if` expression:

```kotlin
val x = listOf(0.5, -1.2, 3.0, -0.1, 2.2)
val cleaned = x.map { if (it > 0) it else 0.0 }   // [0.5, 0.0, 3.0, 0.0, 2.2]
```

The Python version has the same three parts (a condition, a "then" value and an "else" value), but they are three whole arrays instead of one element at a time, and there's no lambda:

```python
import numpy as np

x = np.array([0.5, -1.2, 3.0, -0.1, 2.2])
np.where(x > 0, x, 0.0)
# array([0.5, 0. , 3. , 0. , 2.2])
np.where(x > 0, "pos", "neg")
# array(['pos', 'neg', 'pos', 'neg', 'pos'], dtype='<U3')
```

The second line shows that the result can hold anything an array can: here, strings. `dtype='<U3'` means "Unicode strings of up to 3 characters" (the `<` is the byte order, which you can ignore for text).

**How it works underneath.** Read it slot by slot. Here `a` and `b` are both full arrays, so you can see the choice in every position:

```python
cond = np.array([True, False, True, False])
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])
np.where(cond, a, b)
# array([ 1, 20,  3, 40])
```

Position 0: `cond[0]` is `True`, so the result is `a[0]`, 1. Position 1: `False`, so `b[1]`, 20. And so on. The animation shows the first example, `np.where(x > 0, x, 0.0)`, one slot at a time:

<figure class="diagram" markdown>
![Animation: a row of True and False values for x > 0 sits above the x row; below them is an empty result row and, at the bottom, a row of 0.0 values. Slot by slot, a red arrow brings the value down from x where the condition is True, or up from 0.0 where it is False, until the result reads 0.5, 0.0, 3.0, 0.0, 2.2](../../assets/img/phase-1/day04b-where.svg)
<figcaption>Animated: in each slot, True takes the value from x and False takes it from the fallback.</figcaption>
</figure>

Two facts follow from this picture, and both matter:

1. **Nothing is modified.** `np.where` returns a new array; `x` is unchanged. (To change `x` itself, assign through a mask, as on Day 4: `x[x < 0] = 0.0`.)
2. **`a` and `b` are finished arrays before `np.where` starts.** Python evaluates every argument of a function call before the call. So `np.where(cond, f(x), g(x))` computes `f` on *every* element and `g` on *every* element, then picks. It is not lazy like Kotlin's `if`. You'll see why that matters in a moment.

It works the same in 2-D, and `a` and `b` can be expressions:

```python
M = np.array([[ 3, -1],
              [-4,  5]])
np.where(M < 0, 0, M)
# array([[3, 0],
#        [0, 5]])
np.where(M < 0, -M, M)
# array([[3, 1],
#        [4, 5]])
```

The second line is a hand-made absolute value (`np.abs` does it for you): where `M` is negative, take `-M`, otherwise `M`.

**The result's dtype.** The result needs one dtype that can hold values from both `a` and `b`. NumPy picks it with the same promotion rules as arithmetic (Day 4): ints with ints stay ints, but one float anywhere makes the whole result float.

```python
np.where(x > 0, 1, 0)
# array([1, 0, 1, 0, 1])
np.where(x > 0, 1, 0.5)
# array([1. , 0.5, 1. , 0.5, 1. ])
```

**How it goes wrong (1): writing `if` on an array.** The first thing a Kotlin developer types is an `if`:

```python
try:
    if x > 0:
        print("positive")
except ValueError as e:
    print(e)
# The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
```

`x > 0` is an array of five booleans, and `if` needs one. NumPy refuses to guess whether you meant "all of them" or "any of them". **Recognise it** by that message; **fix it** by saying what you mean: `np.where` for "per element", or `(x > 0).all()` / `(x > 0).any()` (Day 4) for one answer about the whole array.

**How it goes wrong (2): both branches run.** You want `log(v)` where `v > 0`, and `0` elsewhere:

```python
v = np.array([4.0, 0.0, 1.0])
np.where(v > 0, np.log(v), 0.0)
# RuntimeWarning: divide by zero encountered in log
# array([1.38629436, 0.        , 0.        ])
```

The answer is right, but a warning appears, because `np.log(v)` was computed for all three elements, including `log(0)`, before `np.where` threw that value away. With a `0` it's a warning; with a function that raises an error on bad input, the whole line would fail. **Recognise it** by a warning (or error) about values your condition should have excluded. Two **fixes**:

```python
with np.errstate(divide="ignore"):
    safe = np.where(v > 0, np.log(v), 0.0)
safe
# array([1.38629436, 0.        , 0.        ])
```

`np.errstate` silences that one kind of warning inside the `with` block (Part 4 explains it fully). The second fix computes `log` only where it's allowed. Most NumPy maths functions (they're called **ufuncs**, "universal functions", on Day 4) accept `where=` and `out=` arguments: `out=` is an existing array to write the results into, and `where=` says which positions to compute. Positions not computed keep whatever `out` held:

```python
out = np.zeros_like(v)
np.log(v, out=out, where=v > 0)
# array([1.38629436, 0.        , 0.        ])
```

`np.zeros_like(v)` (Day 4) is an array of zeros with `v`'s shape and dtype, so the skipped positions stay `0.0`. Always pass `out=` together with `where=`: without it, the skipped positions hold whatever was in freshly allocated memory, like Day 4's `np.empty`.

## `np.where` with one argument: where is it true?

**What it is.** With only a condition, `np.where(cond)` doesn't choose values. It returns the **positions** where `cond` is `True`, as a tuple with one array of indices per axis. It's exactly the same function as `np.nonzero(cond)` (next section).

**Why it exists.** Sometimes you need *where* something happens, not the values: which samples were misclassified, which pixels are saturated, at which step the loss became `nan`.

**Kotlin comparison.** Kotlin's `indices.filter { x[it] > 0 }`, or `withIndex().filter { it.value > 0 }.map { it.index }`. NumPy gives you the index list directly, with one difference: it's wrapped in a tuple.

```python
np.where(x > 0)
# (array([0, 2, 4]),)
np.where(x > 0)[0]
# array([0, 2, 4])
```

**How it works underneath.** The result has one index array **per axis** of `cond`. A 1-D condition has one axis, so the tuple has one element: `(array([0, 2, 4]),)`. The trailing comma is how Python writes a one-element tuple (Day 2b). In 2-D you get two arrays, the row numbers and the column numbers of each `True`, read in pairs:

```python
np.where(M < 0)
# (array([0, 1]), array([1, 0]))
rows, cols = np.where(M < 0)
rows, cols
# (array([0, 1]), array([1, 0]))
M[rows, cols]
# array([-1, -4])
```

Pair them up: `(0, 1)` and `(1, 0)`. Those are exactly where `M` holds `-1` and `-4`. Passing both arrays back as an index, `M[rows, cols]`, picks those elements: fancy indexing with one array per axis (Day 4; Day 8 has the full rules). The tuple shape is designed for that.

**How it goes wrong: forgetting the `[0]`.** The tuple has one element, so code that treats it as the index array silently does the wrong thing:

```python
idx = np.where(x > 0)
len(idx)
# 1
for i in np.where(x > 0)[0]:
    print(i, x[i])
# 0 0.5
# 2 3.0
# 4 2.2
```

`len(idx)` is `1` (one axis), not `3` (three matches). **Recognise it** by a count of 1, or by a loop that runs once and prints a whole array. **Fix it** with `[0]` for 1-D data, or use `np.count_nonzero(x > 0)` when all you want is the count (next section).

## Finding and counting: `np.nonzero`, `np.argwhere`, `np.count_nonzero`

**What they are.**

- `np.nonzero(a)` returns the indices of the non-zero elements of `a` (for a boolean array, the `True` ones), as a tuple with one array per axis. It's the same as one-argument `np.where`.
- `np.argwhere(a)` returns the same positions arranged differently: an array of shape `(number of matches, number of axes)`, one row per match.
- `np.count_nonzero(a, axis=None)` counts the non-zero (or `True`) elements, overall or along an axis.

**Why.** `nonzero` gives index arrays ready to use for fancy indexing. `argwhere` gives one coordinate per row, which is what you want to print or loop over ("pixel at row 1, column 2"). `count_nonzero` answers "how many?" directly and quickly.

**Kotlin comparison.** For a grid stored as `List<List<Int>>`, the two layouts are two ways of collecting the same matches. `argwhere` is a list of pairs; `nonzero` is that list unzipped into one list per axis; `count_nonzero` is `count { }`:

```kotlin
val pairs = img.flatMapIndexed { r, row -> row.indices.filter { row[it] != 0 }.map { c -> r to c } }
// [(0, 1), (1, 0), (1, 2)]                      like np.argwhere
val (rows, cols) = pairs.unzip()                 // [0, 1, 1] and [1, 0, 2], like np.nonzero
img.flatten().count { it != 0 }                  // 3, like np.count_nonzero
```

```python
img = np.array([[0, 7, 0],
                [3, 0, 9]])
np.nonzero(img)
# (array([0, 1, 1]), array([1, 0, 2]))
np.argwhere(img)
# array([[0, 1],
#        [1, 0],
#        [1, 2]])
np.argwhere(img > 5)
# array([[0, 1],
#        [1, 2]])
```

**Underneath.** Both describe the same three positions, `(0, 1)`, `(1, 0)` and `(1, 2)`. `nonzero` gives the row numbers in one array and the column numbers in another (read them down in pairs); `argwhere` is that pair of arrays turned on its side, one `[row, col]` per line. So `img[np.nonzero(img)]` gives the values `[7, 3, 9]`: one index array per axis, which is what fancy indexing expects.

**How it goes wrong: indexing with `argwhere`.** Pass the `argwhere` result as an index and NumPy reads it as a single array of **row** numbers, so every number in it, column numbers included, is treated as a row:

```python
try:
    img[np.argwhere(img)]
except IndexError as e:
    print(e)
# index 2 is out of bounds for axis 0 with size 2
```

The 2 is a column number from `[1, 2]`, read as row 2 of a 2-row array. On a larger array there might be a row 2, and you'd get a wrong array of whole rows instead of an error. **Fix:** index with `np.nonzero(...)` (or the mask itself, `img[img > 0]`), and keep `argwhere` for printing or looping over coordinates: `for r, c in np.argwhere(img): ...`.

```python
np.count_nonzero(img)
# np.int64(3)
np.count_nonzero(img > 2)
# np.int64(3)
np.count_nonzero(img, axis=0)
# array([1, 1, 1])
```

`np.count_nonzero(mask)` gives the same number as `mask.sum()`, but says what you mean. With `axis=0` it counts down each column: one non-zero in each.

## `np.clip`: keep values inside a range

**What it is.** `np.clip(a, lo, hi)` returns a copy of `a` where every value below `lo` becomes `lo` and every value above `hi` becomes `hi`. Values in between are unchanged. Pass `None` for a side you don't want limited.

**Why it exists.** Pixel values must stay within 0 to 255 after brightening an image; probabilities must stay strictly between 0 and 1 before you take a `log`; gradients are "clipped" in training so one huge step can't wreck a model; and quantising a model to `int8` clips to -128 to 127 (Day 4c's Android bridge).

**Kotlin comparison.** It's `coerceIn` applied to every element:

```kotlin
val pixels = listOf(-12.0, 40.5, 300.0, 255.0, 128.9)
pixels.map { it.coerceIn(0.0, 255.0) }    // [0.0, 40.5, 255.0, 255.0, 128.9]
pixels.map { it.coerceAtLeast(0.0) }      // lower limit only
```

```python
pixels = np.array([-12.0, 40.5, 300.0, 255.0, 128.9])
np.clip(pixels, 0, 255)
# array([  0. ,  40.5, 255. , 255. , 128.9])
np.clip(pixels, 0, None)
# array([  0. ,  40.5, 300. , 255. , 128.9])
np.clip(pixels, None, 100)
# array([-12. ,  40.5, 100. , 100. , 100. ])
```

`np.clip(pixels, 0, None)` is Kotlin's `coerceAtLeast(0.0)`, and `None` for the lower bound is `coerceAtMost`.

**How it works underneath.** `np.clip(a, lo, hi)` is the same as `np.minimum(np.maximum(a, lo), hi)`: first raise everything below `lo` up to `lo`, then lower everything above `hi` down to `hi`. (Day 4's `np.maximum` takes the larger of two values element by element.) It returns a new array, like `np.where`; the method form `a.clip(lo, hi)` does the same thing and also leaves `a` alone:

```python
pixels
# array([-12. ,  40.5, 300. , 255. , 128.9])
pixels.clip(0, 255)
# array([  0. ,  40.5, 255. , 255. , 128.9])
pixels
# array([-12. ,  40.5, 300. , 255. , 128.9])
```

A realistic ML use: cross-entropy loss (Phase 2) takes `log(p)`, and `log(0)` is `-inf`. Clipping the probabilities a tiny distance away from 0 and 1 keeps the loss finite. `1e-7` is scientific notation for \(10^{-7}\), 0.0000001:

```python
probs = np.array([0.0, 0.3, 1.0])
np.clip(probs, 1e-7, 1 - 1e-7)
# array([1.000000e-07, 3.000000e-01, 9.999999e-01])
np.log(np.clip(probs, 1e-7, 1 - 1e-7))
# array([-1.61180957e+01, -1.20397280e+00, -1.00000005e-07])
```

NumPy switched the whole array to scientific notation because the values span many orders of magnitude: `-1.61180957e+01` is -16.118, and the last value is a tiny negative number, not an error.

**How it goes wrong: converting to `uint8` before clipping.** You brighten an image by 20% and convert it back to bytes:

```python
brightened = pixels * 1.2
brightened
# array([-14.4 ,  48.6 , 360.  , 306.  , 154.68])
brightened.astype(np.uint8)
# array([242,  48, 104,  50, 154], dtype=uint8)
np.clip(brightened, 0, 255).astype(np.uint8)
# array([  0,  48, 255, 255, 154], dtype=uint8)
```

The middle line is garbage: 360 became 104 and -14.4 became 242. Day 4 warned that an out-of-range cast "isn't guaranteed to give 255 or anything sensible"; here is what it looks like. `astype` doesn't clip; a value outside 0 to 255 has no `uint8` form, and converting an out-of-range float to an integer type is *undefined* in C, the language underneath NumPy. On my machine it wrapped around, like Day 4's `uint8` overflow; another machine, or the browser's ▶ Run, may print different garbage, and NumPy may or may not warn. Either way, the bright pixels came out dark. **Recognise it** by bright areas turning dark (or dark areas bright) in an image after processing. **Fix it** by clipping first, as in the last line, and by rounding with `np.round` (Day 4c) if you want nearest rather than truncation.

## `np.select`: more than two cases

**What it is.** `np.select(conditions, choices, default=...)` is `np.where` for several cases. `conditions` is a list of boolean arrays, `choices` a list of the same length. In each position, the result takes the choice belonging to the **first** condition that's `True` there, or `default` if none is.

**Why it exists.** Bucketing values into categories ("freezing", "cold", "mild", "hot"), or a piecewise formula with three pieces, without nesting `np.where` inside `np.where`.

**Kotlin comparison.** It's a `when` without a subject, evaluated for every element:

```kotlin
temps.map { t -> when { t < 0 -> "freezing"; t < 15 -> "cold"; t < 25 -> "mild"; else -> "hot" } }
```

```python
temps = np.array([-5.0, 3.0, 12.0, 21.0, 30.0, 38.0])
conditions = [temps < 0, temps < 15, temps < 25]
choices = ["freezing", "cold", "mild"]
np.select(conditions, choices, default="hot")
# array(['freezing', 'cold', 'cold', 'mild', 'hot', 'hot'], dtype='<U8')
```

**Underneath and going wrong.** Like Kotlin's `when`, the first match wins, so put the most specific condition first. -5.0 is also `< 15` and `< 25`, but `temps < 0` comes first. Get the order wrong and the later cases never fire:

```python
grades = np.array([95, 82, 71, 55])
np.select([grades >= 70, grades >= 90], ["pass", "distinction"], default="fail")
# array(['pass', 'pass', 'pass', 'fail'], dtype='<U11')
np.select([grades >= 90, grades >= 70], ["distinction", "pass"], default="fail")
# array(['distinction', 'pass', 'pass', 'fail'], dtype='<U11')
```

In the first line, 95 matches `>= 70` first, so nobody gets a distinction. There's no error. **Recognise it** by a category that never appears; **fix it** by ordering conditions from narrowest to widest.

Underneath, you can picture `np.select` as a chain of `np.where` calls applied from the **last** condition backwards: start with an array full of `default`; then `np.where(conditions[-1], choices[-1], result)`; then the second-to-last, and so on up to the first. Each earlier condition overwrites the later ones, which is why the first match wins. As with `np.where`, every condition and every choice is computed in full before the selection.

---

**Part 2: Sorting and counting**

## `np.sort` and `.sort()`

**What it is.** `np.sort(a)` returns a **new**, sorted copy of `a`, in ascending order. The method `a.sort()` sorts `a` **in place** and returns `None`.

**Why it exists.** Sorted data is how you find medians and percentiles, look at the extremes of a dataset, and check results by eye.

**Kotlin comparison.** This is the same split as Kotlin's `sorted()` (returns a new list) and `sort()` (on a `MutableList`, sorts in place, returns `Unit`), and Python's own `sorted(lst)` vs `lst.sort()` from Day 2b:

```kotlin
val scores = mutableListOf(0.72, 0.15, 0.93, 0.41, 0.15)
val s = scores.sorted()        // new list; scores unchanged
scores.sort()                  // scores itself is now sorted; returns Unit
scores.sortedDescending()      // descending
```

```python
scores = np.array([0.72, 0.15, 0.93, 0.41, 0.15])
np.sort(scores)
# array([0.15, 0.15, 0.41, 0.72, 0.93])
scores
# array([0.72, 0.15, 0.93, 0.41, 0.15])
```

```python
result = scores.sort()
print(result)
# None
scores
# array([0.15, 0.15, 0.41, 0.72, 0.93])
```

Writing `scores = scores.sort()` is the same bug as Day 2b's `lst = lst.sort()`: `scores` becomes `None`.

**Descending order.** There's no `reverse=True` argument. Reverse the sorted copy with the `[::-1]` slice (a step of -1, Day 2b), or sort the negated values and negate back:

```python
scores = np.array([0.72, 0.15, 0.93, 0.41, 0.15])
np.sort(scores)[::-1]
# array([0.93, 0.72, 0.41, 0.15, 0.15])
-np.sort(-scores)
# array([0.93, 0.72, 0.41, 0.15, 0.15])
```

`[::-1]` on an array is a view (Day 4), so it costs nothing extra.

**How it works underneath: the axis.** On a 2-D array, `np.sort` sorts along **one axis**, and the default is `axis=-1`, the **last** axis. For a table, that means each **row** is sorted on its own:

```python
T = np.array([[3, 8, 2],
              [9, 1, 7]])
np.sort(T)                 # axis=-1: each row on its own
# array([[2, 3, 8],
#        [1, 7, 9]])
np.sort(T, axis=0)         # each column on its own
# array([[3, 1, 2],
#        [9, 8, 7]])
np.sort(T, axis=None)      # flatten to 1-D, then sort
# array([1, 2, 3, 7, 8, 9])
```

Trace the first one: row 0 `[3, 8, 2]` becomes `[2, 3, 8]`; row 1 `[9, 1, 7]` becomes `[1, 7, 9]`. The rows never mix. With `axis=0`, each column is sorted from top to bottom: column 0 `[3, 9]` stays; column 1 `[8, 1]` becomes `[1, 8]`; column 2 `[2, 7]` stays. `axis=None` means "no axis": flatten everything, then sort.

!!! warning "Common mistake: sorting a table with `np.sort` scrambles the records"
    Each row of a data table is one record, such as an app's id and its rating. `np.sort` with any axis sorts each column (or row) **independently**, so the values no longer belong together:

    ```python
    # columns: app id, rating
    apps = np.array([[101, 4.5],
                     [102, 3.9],
                     [103, 4.8]])
    np.sort(apps, axis=0)
    # array([[101. ,   3.9],
    #        [102. ,   4.5],
    #        [103. ,   4.8]])
    ```

    App 101 now appears to have a rating of 3.9, which belongs to app 102. No error, no warning. **Fix:** to sort records by one column, use `argsort` on that column and index the rows with it (next section).

**`nan` sorts last.** A missing value (`np.nan`, Part 4) can't be compared with anything, so NumPy puts it at the end:

```python
np.sort(np.array([3.0, np.nan, 1.0, 2.0]))
# array([ 1.,  2.,  3., nan])
```

So `np.sort(a)[-1]` is not "the maximum" if the data has gaps. The same goes for the `arg` functions: `np.argmax` returns the position of the first `nan`, because `nan` propagates through the comparison (Part 4 explains why). `np.nanargmax` skips it:

```python
gappy = np.array([1.0, np.nan, 3.0])
np.argmax(gappy)
# np.int64(1)
np.nanargmax(gappy)
# np.int64(2)
```

## `np.argsort`: the positions that would sort

**What it is.** `np.argsort(a)` doesn't return sorted values. It returns the **indices** that would sort `a`: `order[0]` is the position of the smallest element, `order[1]` the position of the next one, and so on. Indexing `a` with that array, `a[order]`, gives the sorted values.

**Why it exists.** Because indices let you carry other things along. Sort names by their scores, find the top-k predictions of a model and their class labels, or sort the rows of a table by one column while keeping every row intact. In ML it's everywhere: "top-5 accuracy", nearest neighbours, ranking search results.

**Kotlin comparison.** Kotlin has no direct `argsort`, but you can build one, and it shows exactly what the NumPy function does:

```kotlin
val scores = listOf(0.72, 0.15, 0.93, 0.41, 0.15)
val order = scores.indices.sortedBy { scores[it] }   // [1, 4, 3, 0, 2]
val names = listOf("cat", "dog", "fox", "owl", "pig")
order.map { names[it] }                             // [dog, pig, owl, cat, fox]
```

In everyday Kotlin you'd rather zip the two lists into pairs and `sortedBy` the score. NumPy works with separate parallel arrays, so the index list is the tool:

```python
scores = np.array([0.72, 0.15, 0.93, 0.41, 0.15])
order = np.argsort(scores)
order
# array([1, 4, 3, 0, 2])
scores[order]
# array([0.15, 0.15, 0.41, 0.72, 0.93])
```

**How it works underneath.** Read `order` as "where each sorted value came from". The smallest score, 0.15, lives at index 1, so `order[0]` is 1. The other 0.15 is at index 4, so `order[1]` is 4. Then 0.41 at index 3, 0.72 at index 0 and 0.93 at index 2. `scores[order]` is fancy indexing: it builds a new array whose element `k` is `scores[order[k]]`.

<figure class="diagram" markdown>
![Animation: the scores array 0.72, 0.15, 0.93, 0.41, 0.15 with indices 0 to 4 sits at the top. One at a time, an orange arrow runs from a score down to its slot in the sorted row, and the index it came from appears below in the order row, until scores[order] reads 0.15, 0.15, 0.41, 0.72, 0.93 and order reads 1, 4, 3, 0, 2; finally the text "top 3: order[::-1][:3] = [2, 0, 3]" appears](../../assets/img/phase-1/day04b-argsort.svg)
<figcaption>Animated: argsort returns positions; indexing with them gathers the values in sorted order.</figcaption>
</figure>

**Sorting one array by another.** Because `order` holds positions, you can apply it to any array of the same length:

```python
names = np.array(["cat", "dog", "fox", "owl", "pig"])
names[order]
# array(['dog', 'pig', 'owl', 'cat', 'fox'], dtype='<U3')
```

**Top-k.** For the *largest* k, reverse the order and take the first k:

```python
top3 = np.argsort(scores)[::-1][:3]
top3
# array([2, 0, 3])
names[top3], scores[top3]
# (array(['fox', 'cat', 'owl'], dtype='<U3'), array([0.93, 0.72, 0.41]))
```

Step by step: `np.argsort(scores)` is `[1, 4, 3, 0, 2]` (smallest first); `[::-1]` reverses it to `[2, 0, 3, 4, 1]` (largest first); `[:3]` keeps `[2, 0, 3]`. Two equivalent spellings you'll see in other people's code:

```python
np.argsort(scores)[-3:][::-1]
# array([2, 0, 3])
np.argsort(-scores, kind="stable")[:3]
# array([2, 0, 3])
```

The first takes the last three (the three largest, smallest of them first) and then reverses. The second sorts the negated scores, so the largest score comes first. (`kind="stable"` is explained in the next paragraph.)

**Ties and `kind="stable"`.** A **stable** sort keeps equal elements in their original order. NumPy's default algorithm doesn't promise that; `kind="stable"` does. It matters for descending order, because reversing an ascending result also reverses the order of the ties:

```python
np.argsort(scores)[::-1]
# array([2, 0, 3, 4, 1])
np.argsort(-scores, kind="stable")
# array([2, 0, 3, 1, 4])
```

The two 0.15s are at indices 1 and 4. Reversing puts 4 before 1; the stable sort of `-scores` keeps 1 before 4. If you show users "the first app with this score wins", or you need results to match between two runs or two libraries, use `np.argsort(-a, kind="stable")`.

**Sorting the rows of a table by one column.** This is the fix for the scrambled-records mistake above. Take the column, `argsort` it, and use the result to reorder whole rows:

```python
apps[np.argsort(apps[:, 1])]
# array([[102. ,   3.9],
#        [101. ,   4.5],
#        [103. ,   4.8]])
apps[np.argsort(apps[:, 1])[::-1]]
# array([[103. ,   4.8],
#        [101. ,   4.5],
#        [102. ,   3.9]])
```

`apps[:, 1]` is the rating column; its argsort is `[1, 0, 2]`; `apps[[1, 0, 2]]` picks whole rows in that order. Every id stays with its own rating. (pandas, in Week 3, does this with `sort_values`.)

**Along an axis.** `argsort` takes `axis=` like `sort`, with the same default of the last axis. A model's output for a batch has shape `(batch, classes)`, so per-sample top-2 classes are:

```python
logits = np.array([[2.0, 0.1, 5.0, 1.0],
                   [0.3, 4.0, 0.2, 3.5]])
np.argsort(logits, axis=1)[:, ::-1][:, :2]
# array([[2, 0],
#        [1, 3]])
```

Read it in three steps: `np.argsort(logits, axis=1)` sorts the class indices within each row; `[:, ::-1]` reverses every row (all rows, columns backwards); `[:, :2]` keeps the first two columns. Sample 0's best classes are 2 (score 5.0) and 0 (2.0); sample 1's are 1 (4.0) and 3 (3.5).

**How it goes wrong: reading argsort as ranks.** It's tempting to think `argsort` tells you each element's rank ("the tallest person is number..."). It doesn't, and on many inputs the answer looks plausible:

```python
heights = np.array([170, 182, 155])
np.argsort(heights)
# array([2, 0, 1])
np.argsort(np.argsort(heights))
# array([1, 2, 0])
```

`np.argsort(heights)` says "the shortest is at index 2, then index 0, then index 1". It does **not** say person 0 has rank 2. The ranks (0 = shortest) are `[1, 2, 0]`: person 0 (170) is second shortest, person 1 (182) is tallest, person 2 (155) is shortest. Applying `argsort` twice gives ranks, because the second argsort asks "at which position in the sorted order does each original element land?". **Recognise it** by ranks that don't match the data when you check one by hand. **Fix it** with the double argsort.

## `np.unique`: distinct values and how often they appear

**What it is.** `np.unique(a)` returns the distinct values of `a`, **sorted**. With `return_counts=True` it also returns how many times each one appears; with `return_inverse=True`, for each element of `a`, the position of its value in the unique array; with `return_index=True`, where each value first appears.

**Why it exists.** "How many examples of each class do I have?" is the first question about any classification dataset: a 95%/5% split changes how you train and how you judge accuracy. `return_inverse` turns string labels into the integers a model needs.

**Kotlin comparison.** Kotlin's `groupingBy { it }.eachCount()`, or Python's `Counter` from Day 2b:

```kotlin
val labels = listOf("cat", "dog", "cat", "bird", "dog", "cat")
labels.groupingBy { it }.eachCount()    // {cat=3, dog=2, bird=1}  (first-seen order)
labels.distinct().sorted()              // [bird, cat, dog]
```

```python
labels = np.array(["cat", "dog", "cat", "bird", "dog", "cat"])
np.unique(labels)
# array(['bird', 'cat', 'dog'], dtype='<U4')
values, counts = np.unique(labels, return_counts=True)
values, counts
# (array(['bird', 'cat', 'dog'], dtype='<U4'), array([1, 3, 2]))
values[np.argmax(counts)]
# np.str_('cat')
```

The two arrays are parallel: `values[i]` appears `counts[i]` times. `np.argmax(counts)` is the position of the largest count (Day 4), so `values[...]` of it is the most common label. `np.str_('cat')` is how a single string taken out of a NumPy array displays; it behaves like a normal `str`. Single numbers taken out of arrays work the same way: they display as `np.int64(...)` or `np.float64(...)`, as on Day 4. (In the browser's 32-bit Python, integer results such as counts and positions show as `np.int32(...)`; see the box at the top of the page.)

For a plain dict, convert both arrays to Python lists with `.tolist()` and zip them:

```python
dict(zip(values.tolist(), counts.tolist()))
# {'bird': 1, 'cat': 3, 'dog': 2}
```

**Encoding labels as integers.** A model can't train on the strings `"cat"` and `"dog"`; it needs class numbers. `return_inverse=True` gives you exactly that:

```python
values, inverse = np.unique(labels, return_inverse=True)
values
# array(['bird', 'cat', 'dog'], dtype='<U4')
inverse
# array([1, 2, 1, 0, 2, 1])
values[inverse]
# array(['cat', 'dog', 'cat', 'bird', 'dog', 'cat'], dtype='<U4')
```

`inverse[0]` is 1 because `labels[0]` is `"cat"`, which is `values[1]`. So `values` is your class list (index → name) and `inverse` is the encoded label array. `values[inverse]` decodes it back, which proves the round trip works.

**How it works underneath.** Conceptually, `np.unique` sorts a copy of the array, then keeps each element that differs from the one before it. (Recent NumPy versions can use a hash table for the plain case, and have a `sorted=` argument, but the result is still sorted by default.) That's why the result comes out sorted, and why counting is cheap: equal values are neighbours after sorting, so the count is the distance between where each value starts.

**How it goes wrong: expecting first-appearance order.** Kotlin's `distinct()` and Python's `dict` keep the order in which values first appeared. `np.unique` doesn't, so a class list built from it is alphabetical, which matters if another part of your app assumed "first label seen is class 0". If you need first-appearance order, `return_index=True` tells you where each value first occurs; sorting those positions recovers the order:

```python
values, first = np.unique(labels, return_index=True)
first
# array([3, 0, 1])
labels[np.sort(first)]
# array(['cat', 'dog', 'bird'], dtype='<U4')
```

`"bird"` first appears at index 3, `"cat"` at 0, `"dog"` at 1. Sorted, those positions are `[0, 1, 3]`, and the labels there are cat, dog, bird. Whichever order you choose, **save the class list next to the model** (Day 4c shows how, or a `labels.json` as on Day 3c) so the app decodes predictions with the same mapping.

**2-D arrays.** Without `axis`, `np.unique` flattens first. With `axis=0` it finds unique **rows**, which is how you find duplicate samples:

```python
grid = np.array([[1, 2],
                 [3, 4],
                 [1, 2]])
np.unique(grid)
# array([1, 2, 3, 4])
np.unique(grid, axis=0)
# array([[1, 2],
#        [3, 4]])
```

## `np.searchsorted`: where would this value go?

**What it is.** `np.searchsorted(sorted_arr, v)` returns the index at which `v` would have to be inserted into the **already sorted** `sorted_arr` to keep it sorted. It works on many values at once, and uses binary search, so it's fast on large arrays.

**Why it exists.** Bucketing: given the edges of some bins (age groups, price bands), find which bin each value falls into. It's also how `rng.choice` with probabilities picks a value (Day 4c).

**Kotlin comparison.** `List.binarySearch()` in Kotlin returns the index if found, or `-(insertion point) - 1` if not. NumPy always returns the insertion point. The `side=` argument decides where equal values go: `"left"` (default) inserts before an equal element, `"right"` after it.

```python
edges = np.array([0, 18, 35, 60])
ages = np.array([5, 18, 34, 35, 70])
np.searchsorted(edges, ages, side="right")
# array([1, 2, 2, 3, 4])
```

Read the result as "number of edges ≤ the age". 5 is past one edge (0), so bucket 1, "0 to 17". 18 is past 0 and 18 (`side="right"` puts an equal value after the edge), so bucket 2, "18 to 34". 70 is past all four edges, so bucket 4, "60 and over".

**How binary search works underneath.** It halves the search range at each step, so a million edges need only about 20 comparisons. A trace for inserting 34 into `[0, 18, 35, 60]` with `side="right"`:

```
lo = 0, hi = 4   middle 2: edges[2] = 35 > 34    → the answer is at 2 or before: hi = 2
lo = 0, hi = 2   middle 1: edges[1] = 18 <= 34   → the answer is after 1:       lo = 2
lo = hi = 2                                        → insert at 2
```

The answer always lies between `lo` and `hi`, and each step throws away the half that can't contain it. That only works if the array is sorted.

**How it goes wrong: an unsorted input.** Binary search assumes sorted input and doesn't check:

```python
np.searchsorted(np.array([1, 3, 5, 7]), 6)
# np.int64(3)
np.searchsorted(np.array([7, 1, 5, 3]), 6)
# np.int64(4)
```

The second answer is meaningless, and nothing warns you. **Recognise it** by bucket numbers that disagree with a hand check; **fix it** by sorting the edges first, `np.searchsorted(np.sort(edges), ...)`. `np.histogram` (Day 4c) counts how many values fall in each bin, which is often what you wanted in the first place.

---

**Part 3: Running totals and differences**

## `np.cumsum`, `np.cumprod` and `np.diff`

**What they are.**

- `np.cumsum(a)` is the **cumulative sum**, a running total: element `i` of the result is `a[0] + a[1] + ... + a[i]`.
- `np.cumprod(a)` is the **cumulative product**: element `i` is `a[0] * a[1] * ... * a[i]`.
- `np.diff(a)` is the opposite of `cumsum`: the difference between each element and the one before it, `a[i+1] - a[i]`. Its result is **one element shorter** than `a`.

**Why they exist.** Daily installs become total installs (`cumsum`); daily growth rates become overall growth (`cumprod`); a cumulative counter, such as total steps or total downloads, becomes per-period changes (`diff`). In training, `diff` of the loss curve shows whether it's still falling, and `cumsum` of class counts divided by the total gives the cumulative share you'll use for sampling.

**Kotlin comparison.** Kotlin's `runningReduce` and `zipWithNext`:

```kotlin
val installs = listOf(120, 340, 90, 410, 260)
installs.runningReduce { acc, x -> acc + x }   // [120, 460, 550, 960, 1220]
val totals = listOf(120, 460, 550, 960, 1220)
totals.zipWithNext { a, b -> b - a }          // [340, 90, 410, 260]
```

Same results, same lengths: the running version keeps the length, the pairwise one loses one element.

```python
installs = np.array([120, 340, 90, 410, 260])
np.cumsum(installs)
# array([ 120,  460,  550,  960, 1220])
```

```python
growth = np.array([1.10, 0.95, 1.20])
np.cumprod(growth)
# array([1.1  , 1.045, 1.254])
1000 * np.cumprod(growth)
# array([1100., 1045., 1254.])
```

Read the growth one as: a value of 1,000 rises 10%, falls 5%, then rises 20%. After step 1 it's 1,000 × 1.10 = 1,100; after step 2, 1,100 × 0.95 = 1,045; after step 3, 1,045 × 1.20 = 1,254.

```python
totals = np.array([120, 460, 550, 960, 1220])
np.diff(totals)
# array([340,  90, 410, 260])
np.diff(totals, prepend=0)
# array([120, 340,  90, 410, 260])
```

**How it works underneath.** A trace makes the relationship clear. `cumsum` keeps a running total and writes it after each element:

```
a            120   340    90   410   260
running sum  120   460   550   960  1220      each = previous + a[i]
```

`diff` looks at neighbouring pairs, so 5 elements give 4 pairs: (120, 460) → 340, (460, 550) → 90, and so on. That's why it loses one element, and why `prepend=0` (pretend there was a 0 before the first element) gives back the original daily installs exactly: `diff` undoes `cumsum`. `n=2` applies `diff` twice, "the change of the change":

```python
loss = np.array([2.30, 1.10, 0.70, 0.55, 0.52, 0.51])
np.diff(loss)
# array([-1.2 , -0.4 , -0.15, -0.03, -0.01])
np.diff(loss, n=2)
# array([0.8 , 0.25, 0.12, 0.02])
```

Every step of the loss is negative (it's falling), and the steps shrink: -1.2, -0.4, ... -0.01. The training is converging, which is what "the improvements get smaller" means in numbers.

Two more facts. On 2-D arrays they take `axis=` like a reduction; without it, `cumsum` flattens (row by row) first:

```python
week = np.array([[1, 2, 3],
                 [4, 5, 6]])
np.cumsum(week)
# array([ 1,  3,  6, 10, 15, 21])
np.cumsum(week, axis=0)
# array([[1, 2, 3],
#        [5, 7, 9]])
np.cumsum(week, axis=1)
# array([[ 1,  3,  6],
#        [ 4,  9, 15]])
```

With `axis=0` the running total goes down each column (1, then 1 + 4 = 5); with `axis=1` it goes along each row (4, 4 + 5 = 9, 9 + 6 = 15). Unlike `sum`, the axis doesn't disappear: the shape stays `(2, 3)`.

And the last element of a `cumsum` is the total, so dividing by the total gives the running share:

```python
np.cumsum(installs)[-1] == installs.sum()
# np.True_
np.cumsum(installs) / installs.sum()
# array([0.09836066, 0.37704918, 0.45081967, 0.78688525, 1.        ])
```

`np.True_` is NumPy's boolean scalar, as on Day 4; it behaves like `True`.

**How it goes wrong (1): `diff` on unsigned integers.** Images are `uint8`, and the difference between neighbouring pixels is how edge detectors start. A drop in brightness should be negative, but `uint8` has no negative numbers:

```python
row = np.array([10, 200, 30], dtype=np.uint8)
np.diff(row)
# array([190,  86], dtype=uint8)
np.diff(row.astype(np.int16))
# array([ 190, -170], dtype=int16)
```

30 - 200 is -170, which wrapped around to 256 - 170 = 86, the same overflow as Day 4's `uint8(250) + uint8(10)`. **Recognise it** by differences that are all positive when the data clearly goes down. **Fix it** by converting to a signed type with room (`int16`, or `float32`) before subtracting.

**How it goes wrong (2): forgetting that `diff` is shorter.** Percentage change is "change divided by the previous value". Divide by the whole array and the lengths don't match:

```python
prices = np.array([5.0, 5.5, 5.2, 6.0])
try:
    returns = np.diff(prices) / prices
except ValueError as e:
    print(e)
# operands could not be broadcast together with shapes (3,) (4,)
np.diff(prices) / prices[:-1]
# array([ 0.1       , -0.05454545,  0.15384615])
```

The error message names the two shapes, `(3,)` and `(4,)`: that's the hint. ("Broadcast" is the Day 9 word for combining arrays of different shapes; here they don't match.) **Fix it** by dividing by `prices[:-1]`, every price except the last, which lines each change up with the price it started from: 0.5 / 5.0 = 0.1, -0.3 / 5.5 = -0.0545, 0.8 / 5.2 = 0.1538.

---

**Part 4: Missing and special values**

## `nan` and `inf` in arrays

**What they are.** IEEE-754 floats (Day 2) have three special values. `inf` and `-inf` stand for "larger (or smaller) than any finite number". `nan`, "not a number", stands for "no meaningful value": the result of `0/0`, `inf - inf`, or `sqrt(-1)` on real numbers. NumPy spells them `np.inf` and `np.nan`. They're ordinary `float64` (or `float32`) values, so they can sit in any float array.

**Why they matter.** In data, `nan` is the standard marker for a **missing value**: a sensor that didn't report, an app with no rating yet. pandas (Week 3) uses it the same way. In training, `inf` and `nan` are how numerical trouble shows up: one `log(0)` or one overflow and the loss becomes `inf`, then `nan`, and every weight it touches becomes `nan` on the next update. Spotting them early is a debugging skill.

**Kotlin comparison.** Kotlin has the same values, because the JVM uses the same IEEE-754 floats: `Double.NaN`, `Double.POSITIVE_INFINITY`, `x.isNaN()`, `x.isInfinite()`, `x.isFinite()`. What differs is where they come from. In Kotlin, `1.0 / 0.0` is `Infinity` silently, while `1 / 0` on `Int` throws `ArithmeticException`. In plain Python, `1.0 / 0.0` *raises* `ZeroDivisionError` (Day 2). NumPy behaves like neither: it returns `inf` or `nan` **and prints a warning**. Day 4 showed this for single ufunc calls (`np.log(0.0)`, `np.sqrt(-1.0)`); here are the sources side by side.

```python
x = np.array([1.0, 0.0, -1.0])
x / 0.0
# RuntimeWarning: divide by zero encountered in divide
# RuntimeWarning: invalid value encountered in divide
# array([ inf,  nan, -inf])
```

1/0 is `inf`, 0/0 is `nan` (NumPy calls this an "invalid value"), and -1/0 is `-inf`. The other common sources:

```python
np.log(np.array([1.0, 0.0, -1.0]))
# RuntimeWarning: divide by zero encountered in log
# RuntimeWarning: invalid value encountered in log
# array([  0., -inf,  nan])
np.sqrt(np.array([4.0, -4.0]))
# RuntimeWarning: invalid value encountered in sqrt
# array([ 2., nan])
np.float32(3e38) * np.float32(10)
# RuntimeWarning: overflow encountered in scalar multiply
# np.float32(inf)
```

The last one is **overflow**: `float32`'s largest value is about \(3.4 \times 10^{38}\), so \(3 \times 10^{39}\) doesn't fit and becomes `inf`. That's one reason training in 16-bit floats (Phase 6) needs care: `float16` overflows above 65,504.

**How `nan` spreads.** Any arithmetic involving `nan` gives `nan`, and that includes reductions. One missing value poisons the whole sum:

```python
data = np.array([4.0, np.nan, 3.0, 5.0])
data.sum()
# np.float64(nan)
data.mean()
# np.float64(nan)
data.max()
# np.float64(nan)
data + 1
# array([ 5., nan,  4.,  6.])
```

<figure class="diagram" markdown>
![Two panels. On the left, data.sum() adds 4.0, NaN, 3.0 and 5.0 into a running total: the total is 4.0, then NaN, and stays NaN for the rest, giving nan. On the right, np.nansum(data) greys out and skips the NaN, so the running total goes 4.0, 4.0, 7.0, 12.0 and gives 12.0](../../assets/img/phase-1/day04b-nan-spread.svg)
<figcaption>One NaN turns a running total into NaN for good. The nan* functions skip it instead.</figcaption>
</figure>

This is deliberate. A `nan` that disappeared quietly would give you a wrong average; a `nan` result tells you something is missing. `inf` has its own rules, which follow from "a number larger than any other":

```python
np.inf + 1
# inf
np.inf - np.inf
# nan
0 * np.inf
# nan
1 / np.inf
# 0.0
```

These four show as plain `inf`, `nan` and `0.0` (no `np.float64(...)`) because `np.inf` is a plain Python `float`, so the arithmetic is Python's.

**How it goes wrong: `nan` in an integer array.** `nan` is a float. Integer arrays can't hold it:

```python
counts = np.array([3, 5, 7])
try:
    counts[1] = np.nan
except ValueError as e:
    print(e)
# cannot convert float NaN to integer
counts = counts.astype(float)
counts[1] = np.nan
counts
# array([ 3., nan,  7.])
```

**Recognise it** by `cannot convert float NaN to integer`. **Fix it** by converting to float first, as above, or by choosing an integer sentinel such as `-1` and handling it explicitly (and documenting it, because the next person will forget).

## Why `nan != nan`, and how to test for it

**What it is.** IEEE-754 defines every comparison involving `nan` as false, except `!=`, which is true. So `nan == nan` is `False`, and comparing an array to `np.nan` is `False` everywhere.

**Why.** `nan` means "no meaningful value". Two missing ratings are not known to be equal, so "equal" can't be true. Making `nan` unequal to itself also gives a portable test: `x != x` is true only for `nan`, which is how C code checked for it before `isnan` existed.

**Kotlin comparison.** Kotlin follows IEEE-754 when the types are statically `Double`: `Double.NaN == Double.NaN` is `false`. But when the values are boxed or typed as `Any?` (for example inside a `List<Double>`'s `equals` or `contains`), Kotlin uses `equals()`, where `NaN` *is* equal to itself. NumPy has no such second mode: `==` on arrays is always IEEE.

```python
np.nan == np.nan
# False
data == np.nan
# array([False, False, False, False])
(data == np.nan).sum()
# np.int64(0)
```

That last line is the classic bug: "count the missing values" with `==` always reports zero. Summing a boolean array counts its `True`s (Day 4), and there are none.

**The right tools.** `np.isnan(a)` is `True` where `a` is `nan`. `np.isinf(a)` is `True` for `inf` and `-inf`. `np.isfinite(a)` is `True` for ordinary numbers, and is the one to use as a guard, since it rejects all three special values at once:

```python
np.isnan(data)
# array([False,  True, False, False])
np.isnan(data).sum()
# np.int64(1)
data[~np.isnan(data)]
# array([4., 3., 5.])
```

`~` flips the mask (Day 4), so `data[~np.isnan(data)]` keeps the values that are present.

```python
mixed = np.array([1.0, np.nan, np.inf, -np.inf, 0.0])
np.isnan(mixed)
# array([False,  True, False, False, False])
np.isinf(mixed)
# array([False, False,  True,  True, False])
np.isfinite(mixed)
# array([ True, False, False, False,  True])
```

!!! warning "Common mistake: checking for missing values with `== np.nan`"
    `x == np.nan`, `x[x == np.nan] = 0` and `if value == np.nan:` never match anything, and raise no error: `(data == np.nan).sum()` is `np.int64(0)` even when `data` has gaps. **Fix:** `np.isnan(x)` for arrays, `math.isnan(v)` for a single number (Day 2). In a training loop, `if not np.isfinite(loss): ...` catches both `nan` and `inf`.

## The `nan*` functions: summaries that skip gaps

**What they are.** For most reductions NumPy has a version that ignores `nan`: `np.nansum`, `np.nanmean`, `np.nanmax`, `np.nanmin`, `np.nanstd`, `np.nanmedian`, `np.nanargmax`, and more. They compute the same thing over the values that are present.

**Why.** Real data has gaps. You want "the average rating of the apps that have a rating", not `nan`.

**Kotlin comparison.** In Kotlin you'd filter first: `ratings.filterNot { it.isNaN() }.average()`. The `nan*` functions are that filter and the reduction in one call, and they work along an axis of a 2-D array.

```python
np.nansum(data)
# np.float64(12.0)
np.nanmean(data)
# np.float64(4.0)
np.nanmax(data)
# np.float64(5.0)
np.nanmin(data), np.nanstd(data)
# (np.float64(3.0), np.float64(0.816496580927726))
```

`np.nanmean(data)` is (4 + 3 + 5) / 3 = 4.0: the divisor is 3, the number of values present, not 4. They take `axis=` like the ordinary reductions; here each column's mean uses only that column's present values:

```python
ratings = np.array([[4.0, np.nan],
                    [3.0, 5.0],
                    [np.nan, 4.0]])
np.nanmean(ratings, axis=0)
# array([3.5, 4.5])
ratings.mean(axis=0)
# array([nan, nan])
```

**Filling gaps.** Often you want to *replace* missing values instead of skipping them, because a model needs a number in every cell. Filling with the mean or median of the present values is called **imputation**. `np.where` from Part 1 does it in one line:

```python
filled = np.where(np.isnan(data), np.nanmean(data), data)
filled
# array([4., 4., 3., 5.])
```

`np.nan_to_num` replaces all three special values at once. Its defaults (0 for `nan`, the largest float for `inf`) are rarely what you want, so pass your own:

```python
np.nan_to_num(mixed)
# array([ 1.00000000e+000,  0.00000000e+000,  1.79769313e+308,
#        -1.79769313e+308,  0.00000000e+000])
np.nan_to_num(mixed, nan=0.0, posinf=1e6, neginf=-1e6)
# array([ 1.e+00,  0.e+00,  1.e+06, -1.e+06,  0.e+00])
```

**How it goes wrong: a slice that's entirely missing.** If every value is `nan`, there's nothing to average. The result is `nan`, with a warning:

```python
np.nanmean(np.array([np.nan, np.nan]))
# RuntimeWarning: Mean of empty slice
# np.float64(nan)
```

On a whole table this happens to one column (a feature nobody filled in) and the warning is easy to miss in a long log. **Recognise it** by "Mean of empty slice" (or "All-NaN slice encountered" from `nanmax` and `nanmin`). **Fix it** by counting the missing values per column first, `np.isnan(X).sum(axis=0)`, and dropping or filling the empty columns before summarising.

## Division by zero warnings and `np.errstate`

**What it is.** When NumPy divides by zero or produces `nan`, it doesn't raise an error by default; it prints a `RuntimeWarning` and carries on. `np.errstate(...)` is a context manager (Day 3c) that changes this behaviour **inside its `with` block only**. Each kind of problem has its own setting:

- `divide`: dividing a non-zero number by zero (gives `inf`), and `log(0)`;
- `invalid`: a result with no meaningful value (gives `nan`), such as `0/0` or `sqrt(-1)`;
- `over` and `under`: overflow and underflow;
- `all`: all four.

Each can be `"ignore"`, `"warn"` (the default), `"raise"` (turn it into a `FloatingPointError` exception), and a few other options.

**Why.** Two opposite needs. Sometimes the `inf` or `nan` is expected and you'll handle it on the next line, so the warning is noise. And sometimes, while debugging a training run that ends in `nan`, you want the program to **stop at the first** bad operation so the traceback (Day 3b) points at it.

**Kotlin comparison.** Kotlin has no switch like this: `Double` division silently gives `Infinity` or `NaN`, and `Int` division throws. `errstate` lets one block of NumPy code behave like either. The `with` block works like Kotlin's `use { }`: when the block ends, the old settings are restored, even if an exception was raised.

```python
nums = np.array([1.0, 2.0, 0.0])
dens = np.array([2.0, 0.0, 0.0])
with np.errstate(divide="ignore", invalid="ignore"):
    ratio = nums / dens
ratio
# array([0.5, inf, nan])
```

No warnings this time. Turning on `"raise"` makes the same division fail loudly instead:

```python
try:
    with np.errstate(all="raise"):
        nums / dens
except FloatingPointError as e:
    print("FloatingPointError:", e)
# FloatingPointError: divide by zero encountered in divide
```

**Underneath.** The CPU sets status flags when a float operation divides by zero or produces `nan`. After each operation NumPy checks those flags and does what the current error state says. `np.errstate` changes that state and puts it back when the block ends. (`np.seterr` changes it for the rest of the program; prefer `errstate`, so the change can't leak into code that didn't ask for it.)

**How it goes wrong (1): integers are different.** Integer division by zero in NumPy doesn't give `inf` (integers have no `inf`). It gives `0`, with a warning; plain Python raises:

```python
np.array([6, 3]) // np.array([0, 1])
# RuntimeWarning: divide by zero encountered in floor_divide
# array([0, 3])
try:
    6 // 0
except ZeroDivisionError as e:
    print("ZeroDivisionError:", e)
# ZeroDivisionError: division by zero
```

(That is Python 3.14's wording, which the ▶ Run button uses; Python 3.13 and older say `integer division or modulo by zero`.)

A count divided by a zero count quietly becomes 0, which looks like a real value. **Recognise it** by that warning; **fix it** by masking out zero denominators with `np.where` before dividing.

**How it goes wrong (2): silencing warnings for the whole program.** It's tempting to put `np.seterr(all="ignore")` (or `warnings.filterwarnings("ignore")`) at the top of a notebook to get rid of the noise. Then a real `nan` appears in training and nothing tells you where. Keep `errstate` blocks small, around the one line where you *expect* the problem.

---

**Part 5: Comparing floats**

## `np.isclose` and `np.allclose`

**What they are.** `np.isclose(a, b)` compares two arrays element by element and returns `True` where they are equal **within a tolerance**. `np.allclose(a, b)` returns a single `True` if every element is close. Both use this test for each pair:

\[
|a - b| \le \text{atol} + \text{rtol} \times |b|
\]

Symbol by symbol: \(|a - b|\) is the gap between the two values. **`atol`** is the **absolute tolerance**, a fixed allowed gap, `1e-08` by default. **`rtol`** is the **relative tolerance**, an allowed gap that grows with the size of the numbers, `1e-05` (0.001%) by default. The allowed gap is the two added together.

**Why.** Floats are rounded (Day 2), so two calculations of the same quantity often differ in the last digits. Checking a vectorised rewrite against a loop, a `float32` model against a `float64` one, or your gradient against a numerical estimate (Phase 2) needs "equal, give or take rounding". You'll write `assert np.allclose(mine, expected)` in tests constantly.

**Kotlin comparison.** JUnit's `assertEquals(expected, actual, delta)` uses an absolute tolerance only: `|expected - actual| <= delta`. NumPy combines an absolute and a relative part. And compared with Day 2's `math.isclose`, there are three differences: NumPy's `atol` is **not** zero by default, its default `rtol` is much looser (`1e-05` vs `1e-09`), and the relative part uses only `|b|`, not the larger of the two.

```python
total = np.array([0.1, 0.2]).sum()
total
# np.float64(0.30000000000000004)
total == 0.3
# np.False_
np.isclose(total, 0.3)
# np.True_
```

```python
a = np.array([1.0, 2.0, 3.0])
b = a + np.array([1e-9, -1e-9, 2e-6])
a == b
# array([False, False, False])
np.isclose(a, b)
# array([ True,  True,  True])
np.allclose(a, b)
# True
```

**A worked example by hand.** Is 1000.009 close to 1000.0 with the defaults? The gap is 0.009. The allowed gap is \(10^{-8} + 10^{-5} \times 1000.009 = 0.00000001 + 0.01000009 = 0.0100001\). 0.009 is less, so yes. For 1000.02 the gap is 0.02, which is more than the 0.0100002 allowed, so no:

```python
np.allclose(1000.0, 1000.009)
# True
np.allclose(1000.0, 1000.02)
# False
```

**How it goes wrong (1): `atol` makes all tiny numbers "close".** With small values, such as gradients or probabilities near zero, the default `atol` of \(10^{-8}\) is bigger than the numbers themselves:

```python
np.allclose(1e-9, 3e-9)
# True
np.allclose(1e-9, 3e-9, atol=0)
# False
```

\(3 \times 10^{-9}\) is three times \(10^{-9}\), yet the default test passes, because the gap, \(2 \times 10^{-9}\), is below `atol`. A gradient check built this way would pass for a gradient that's off by a factor of 3. **Recognise it** when the values you compare are around \(10^{-6}\) or smaller. **Fix it** by setting `atol` to a size that makes sense for your data, or `atol=0` to rely on `rtol` only.

**How it goes wrong (2): it isn't symmetric.** Because only \(|b|\) is scaled, `isclose(a, b)` and `isclose(b, a)` can disagree. With the defaults the difference is negligible; with a loose `rtol` you can see it. Here `rtol=0.1` allows a 10% gap:

```python
np.isclose(10.0, 11.05, rtol=0.1, atol=0), np.isclose(11.05, 10.0, rtol=0.1, atol=0)
# (np.True_, np.False_)
```

The gap is 1.05. Measured against `b = 11.05`, 10% is 1.105, so it passes; against `b = 10.0`, 10% is 1.0, so it fails. **Fix:** put the trusted reference value second (`np.isclose(actual, expected)`), which is the convention NumPy's own testing tools use.

**How it goes wrong (3): shapes that don't match still pass.** `isclose` and `allclose` broadcast their inputs (Day 9), so a single value is compared with every element:

```python
np.allclose(np.array([1.0, 1.0, 1.0]), np.array([1.0]))
# True
np.array_equal(np.array([1.0, 1.0, 1.0]), np.array([1.0]))
# False
```

If your function was supposed to return three values and returned one, the `allclose` test still passes. **Fix:** check the shape too (`assert result.shape == expected.shape`), or use `np.array_equal` when exact equality is right.

## `np.array_equal`: exact equality, shape included

**What it is.** `np.array_equal(a, b)` returns one `True` if `a` and `b` have the **same shape** and **exactly the same** elements. `equal_nan=True` treats `nan` in the same position as equal; `np.allclose` takes the same argument.

**Kotlin comparison.** It's `DoubleArray.contentEquals(other)`: element-by-element, exact. Kotlin's `==` on two arrays compares *references* (are they the same object?), and NumPy's `==` gives an array of booleans; neither answers "same contents?", which is why both languages have a separate function for it.

**When to use which.** Integers, booleans, indices and labels: `array_equal`, since there's no rounding to forgive. Floats that came out of arithmetic: `allclose`. Floats that were only copied, saved and loaded: `array_equal` is fine, and stricter.

```python
np.array_equal(np.array([1, 2]), np.array([1, 2]))
# True
np.array_equal(np.array([1, np.nan]), np.array([1, np.nan]))
# False
np.array_equal(np.array([1, np.nan]), np.array([1, np.nan]), equal_nan=True)
# True
np.allclose(np.array([1, np.nan]), np.array([1, np.nan]), equal_nan=True)
# True
```

The second line is `False` for the same reason as `nan == nan`.


---

## Going further

These exist and are worth knowing about; you won't need them this month.

- [`np.argpartition`](https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html): the top-k indices without sorting everything, faster than `argsort` on large arrays (the k results come out unordered).
- [`np.lexsort`](https://numpy.org/doc/stable/reference/generated/numpy.lexsort.html): sort by several keys (by category, then by rating).
- [Set routines](https://numpy.org/doc/stable/reference/routines.set.html): `np.union1d`, `np.intersect1d`, `np.setdiff1d` and friends (Day 4c uses `intersect1d` and `isin`).
- [`np.bincount`](https://numpy.org/doc/stable/reference/generated/numpy.bincount.html) and [`np.digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html): fast counting of small integer labels, and bin numbers for values.
- [Masked arrays, `numpy.ma`](https://numpy.org/doc/stable/reference/maskedarray.html): an alternative to `nan` for missing data, which also works for integer arrays.
- [Floating-point error handling](https://numpy.org/doc/stable/reference/routines.err.html): `np.seterr`, `np.geterr` and the other `errstate` modes (`"call"`, `"print"`, `"log"`).
- [`np.testing`](https://numpy.org/doc/stable/reference/routines.testing.html): `assert_allclose` and `assert_array_equal`, which print exactly which elements differ; use them in pytest (Day 3c).

---

## Exercises

Each solution re-creates its own inputs, so you can run it on its own.

**1. Warm-up: clean sensor readings.** A sensor should report values from 0 to 100: `r = np.array([12.5, -3.0, 7.25, 101.0, 55.0, -0.5])`. (a) Replace the negative readings with 0 using `np.where`, leaving the others alone. (b) Clip all readings to 0 to 100 with `np.clip`. (c) Count how many readings were outside the range, with `np.count_nonzero`, and print their positions.

**2. Warm-up: label counts.** `labels = np.array(["dog", "cat", "cat", "bird", "dog", "cat", "fish", "cat"])`. Print the count of each label as a dict, print the most common label, and encode the labels as integers. Decode your integers back and check they match the original labels.

**3. Core: a column with gaps.** `ratings = np.array([4.5, np.nan, 3.0, 4.0, np.nan, 5.0, 2.5])`. Print: how many are missing; the mean of the ratings that are present; and the array with the gaps filled by the **median** of the present ratings (`np.nanmedian`). Then show that `ratings.mean()` is `nan` while the filled array's mean is a number, and say in one sentence why they differ.

**4. Core: leaderboard.** `apps = np.array(["Maps", "Chat", "Notes", "Camera", "Music", "Clock"])` and `downloads = np.array([500, 1200, 300, 1200, 800, 50])`. (a) Print the top-3 app names by downloads, most downloads first; when two apps tie, the one listed first comes first. (b) Give every app its rank, where 1 means most downloads (ties broken the same way).

**5. Core: a launch week.** An app's daily installs for its first ten days are `daily = np.array([40, 120, 260, 180, 90, 300, 210, 100, 60, 30])`. (a) Compute the running total of installs. (b) Find the first day (counting from day 1) on which the running total reached 1,000, using `np.searchsorted` on the running total. (c) Compute the day-over-day change, and find the day with the biggest drop and how big it was. (d) Check that `np.diff` with `prepend=0` on your running total gives back `daily` exactly.

**6. Predict the output.** Write down what each line prints *before* running it. Then run it and explain every one you got wrong.

```python
import numpy as np

print((np.array([1.0, np.nan]) == np.nan).sum())
print(np.diff(np.array([5, 1], dtype=np.uint8)))
print(np.sort(np.array([[3, 1], [2, 4]])))
print(np.allclose(1e-10, 0.0))
print(len(np.where(np.array([0, 3, 0, 5]))))
print(np.cumsum(np.array([1, 2, 3]))[-1])
print(np.unique(np.array(["b", "a", "b"])))
print(np.where(np.array([1, -1]) > 0, 1, 0.5))
```

**7. Find and fix the bug.** This function should report how many ratings are missing and the two best-rated apps, each with its own id. It runs without an error, but it has **two** bugs. Find them, explain what each one does to the output, and fix them.

```python
import numpy as np

def report(table):
    """table columns: app id, rating (the rating may be nan)."""
    ratings = table[:, 1]
    n_missing = (ratings == np.nan).sum()
    ranked = np.sort(table, axis=0)[::-1]
    return n_missing, ranked[:2]

table = np.array([[101, 4.1],
                  [102, np.nan],
                  [103, 4.7],
                  [104, 3.2]])
n_missing, best = report(table)
print(n_missing)
print(best)
# 0
# [[104.    nan]
#  [103.    4.7]]
```

**8. Challenge: a float comparison helper.** Write `compare(actual, expected, rtol=1e-5, atol=1e-8)` for checking a model's outputs in tests. It should return a short report string: `"shape mismatch: (3,) vs (2, 2)"` if the shapes differ; `"nan mismatch at [...]"` with the positions (from `np.argwhere`) if `nan` appears in different places; otherwise `"ok"` if every element is close (with `nan` in the same place counting as equal), or `"N mismatches, worst at [i, j]: actual vs expected"` naming the element with the largest absolute gap. Test it on: identical arrays; a `(2, 2)` array against a `(3,)` one; `nan` in different places; and two `(2, 3)` arrays where two elements are off.

??? success "Solutions"

    **1. Clean sensor readings.**

    ```python
    import numpy as np

    r = np.array([12.5, -3.0, 7.25, 101.0, 55.0, -0.5])
    np.where(r < 0, 0.0, r)
    # array([ 12.5 ,   0.  ,   7.25, 101.  ,  55.  ,   0.  ])
    np.clip(r, 0, 100)
    # array([ 12.5 ,   0.  ,   7.25, 100.  ,  55.  ,   0.  ])
    out_of_range = (r < 0) | (r > 100)
    print(np.count_nonzero(out_of_range))
    print(np.nonzero(out_of_range)[0])
    # 3
    # [1 3 5]
    ```

    `np.where` only replaces the positions where its condition is `True`, so 101.0 survives (a); `np.clip` limits both ends (b). For (c), `|` combines the two masks element by element (Day 4), and the parentheses are required because `|` binds more tightly than `<`. The out-of-range readings are -3.0, 101.0 and -0.5, at positions 1, 3 and 5; `[0]` takes the one index array out of the tuple `np.nonzero` returns.

    **2. Label counts.**

    ```python
    import numpy as np

    labels = np.array(["dog", "cat", "cat", "bird", "dog", "cat", "fish", "cat"])
    values, counts = np.unique(labels, return_counts=True)
    print(dict(zip(values.tolist(), counts.tolist())))
    print(values[np.argmax(counts)])
    values, codes = np.unique(labels, return_inverse=True)
    print(codes)
    # {'bird': 1, 'cat': 4, 'dog': 2, 'fish': 1}
    # cat
    # [2 1 1 0 2 1 3 1]
    np.array_equal(values[codes], labels)
    # True
    ```

    `np.unique` returns the distinct labels sorted alphabetically, with parallel counts; `argmax` of the counts picks the most common. `return_inverse` gives each label's position in `values`, which is its integer code (bird 0, cat 1, dog 2, fish 3), so `values[codes]` decodes them back exactly.

    **3. A column with gaps.**

    ```python
    import numpy as np

    ratings = np.array([4.5, np.nan, 3.0, 4.0, np.nan, 5.0, 2.5])
    print(np.isnan(ratings).sum())
    # 2
    np.nanmean(ratings)
    # np.float64(3.8)
    filled = np.where(np.isnan(ratings), np.nanmedian(ratings), ratings)
    filled
    # array([4.5, 4. , 3. , 4. , 4. , 5. , 2.5])
    ratings.mean(), filled.mean()
    # (np.float64(nan), np.float64(3.857142857142857))
    ```

    `np.isnan` finds the gaps (`== np.nan` would find none). The present values are 4.5, 3.0, 4.0, 5.0, 2.5: their mean is 19 / 5 = 3.8, and their median, the middle of 2.5, 3.0, 4.0, 4.5, 5.0, is 4.0. `ratings.mean()` is `nan` because one `nan` makes any sum `nan`; the filled array has no gaps, so its mean is (19 + 4 + 4) / 7 = 3.857.

    **4. Leaderboard.**

    ```python
    import numpy as np

    apps = np.array(["Maps", "Chat", "Notes", "Camera", "Music", "Clock"])
    downloads = np.array([500, 1200, 300, 1200, 800, 50])
    order = np.argsort(-downloads, kind="stable")
    apps[order[:3]]
    # array(['Chat', 'Camera', 'Music'], dtype='<U6')
    ranks = np.argsort(order) + 1
    dict(zip(apps.tolist(), ranks.tolist()))
    # {'Maps': 4, 'Chat': 1, 'Notes': 5, 'Camera': 2, 'Music': 3, 'Clock': 6}
    ```

    Sorting `-downloads` ascending puts the largest downloads first, and `kind="stable"` keeps Chat (index 1) before Camera (index 3) at their tie of 1200. `np.argsort(downloads)[::-1]` would have reversed the tie and put Camera first. `order` says which app is at each place; `np.argsort(order)` inverts that, giving each app its place (0-based), and `+ 1` makes the best rank 1.

    **5. A launch week.**

    ```python
    import numpy as np

    daily = np.array([40, 120, 260, 180, 90, 300, 210, 100, 60, 30])
    total = np.cumsum(daily)
    print(total)
    day = np.searchsorted(total, 1000) + 1
    print(day)
    change = np.diff(daily)
    print(change)
    worst = np.argmin(change)
    print(worst + 2, change[worst])
    # [  40  160  420  600  690  990 1200 1300 1360 1390]
    # 7
    # [  80  140  -80  -90  210  -90 -110  -40  -30]
    # 8 -110
    np.array_equal(np.diff(total, prepend=0), daily)
    # True
    ```

    The running total only grows (installs are never negative), so it's sorted and `searchsorted` is allowed on it. With the default `side="left"`, `searchsorted(total, 1000)` is the first position whose total is at least 1,000: the totals pass 990 at position 5 and reach 1,200 at position 6, so the answer is position 6, which is day 7 counting from 1. `np.diff(daily)` has 9 elements: `change[i]` is day `i + 2` minus day `i + 1`, which is why the day of the biggest drop is `worst + 2`. `argmin` finds the most negative change. Finally, `diff` with `prepend=0` undoes `cumsum`, so the round trip is exact.

    **6. Predict the output.**

    ```python
    import numpy as np

    print((np.array([1.0, np.nan]) == np.nan).sum())
    print(np.diff(np.array([5, 1], dtype=np.uint8)))
    print(np.sort(np.array([[3, 1], [2, 4]])))
    print(np.allclose(1e-10, 0.0))
    print(len(np.where(np.array([0, 3, 0, 5]))))
    print(np.cumsum(np.array([1, 2, 3]))[-1])
    print(np.unique(np.array(["b", "a", "b"])))
    print(np.where(np.array([1, -1]) > 0, 1, 0.5))
    # 0
    # [252]
    # [[1 3]
    #  [2 4]]
    # True
    # 1
    # 6
    # ['a' 'b']
    # [1.  0.5]
    ```

    Line by line: `nan` equals nothing, so the mask is all `False` and sums to 0. 1 - 5 = -4 wraps to 256 - 4 = 252 in `uint8`. `np.sort` on 2-D sorts each row (the last axis) separately: `[3, 1]` becomes `[1, 3]`, `[2, 4]` stays. `allclose` passes because the gap, \(10^{-10}\), is under the default `atol` of \(10^{-8}\). One-argument `np.where` returns a tuple with one array per axis, and a 1-D input has one axis, so its length is 1. The last element of a `cumsum` is the total, 6. `np.unique` returns the distinct values **sorted**, so `['a' 'b']`, not first-seen order. And `np.where` with an int and a float choice promotes the result to float: `[1.  0.5]`.

    **7. Find and fix the bug.**

    ```python
    import numpy as np

    def report(table):
        """table columns: app id, rating (the rating may be nan)."""
        ratings = table[:, 1]
        n_missing = np.isnan(ratings).sum()                 # fix 1: was ratings == np.nan
        ranked = table[np.argsort(-ratings, kind="stable")] # fix 2: reorder whole rows
        return n_missing, ranked[:2]

    table = np.array([[101, 4.1],
                      [102, np.nan],
                      [103, 4.7],
                      [104, 3.2]])
    n_missing, best = report(table)
    print(n_missing)
    print(best)
    # 1
    # [[103.    4.7]
    #  [101.    4.1]]
    ```

    Bug 1: `ratings == np.nan` is `False` everywhere, because `nan` isn't equal to anything, so the report said 0 missing ratings when there is one. `np.isnan` finds it. Bug 2: `np.sort(table, axis=0)` sorts the id column and the rating column **independently**, so the "best" rows paired id 104 with a `nan` rating and id 103 with 4.7 only by coincidence; the records were scrambled. `argsort` of the rating column, used to index whole rows, keeps every id with its own rating. Negating the ratings sorts the largest first, and `-nan` is still `nan`, which sorts last, so the app without a rating can't come out on top.

    **8. Challenge: a float comparison helper.**

    ```python
    import numpy as np

    def compare(actual, expected, rtol=1e-5, atol=1e-8):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.shape != expected.shape:
            return f"shape mismatch: {actual.shape} vs {expected.shape}"
        nan_a, nan_e = np.isnan(actual), np.isnan(expected)
        if not np.array_equal(nan_a, nan_e):
            where = np.argwhere(nan_a != nan_e).tolist()
            return f"nan mismatch at {where}"
        close = np.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
        if close.all():
            return "ok"
        gap = np.where(np.isnan(actual), 0.0, np.abs(actual - expected))
        worst = np.unravel_index(np.argmax(gap), gap.shape)
        worst = [int(i) for i in worst]
        return (f"{np.count_nonzero(~close)} mismatches, worst at {worst}: "
                f"{actual[tuple(worst)]} vs {expected[tuple(worst)]}")

    a = np.array([[1.0, 2.0], [3.0, np.nan]])
    print(compare(a, a.copy()))
    print(compare(a, np.array([1.0, 2.0, 3.0])))
    print(compare(a, np.array([[1.0, np.nan], [3.0, 4.0]])))
    exp = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    act = exp + np.array([[0.0, 1e-9, 0.02], [0.0, -0.5, 0.0]])
    print(compare(act, exp))
    # ok
    # shape mismatch: (2, 2) vs (3,)
    # nan mismatch at [[0, 1], [1, 1]]
    # 2 mismatches, worst at [1, 1]: 0.0 vs 0.5
    ```

    `np.asarray(x)` returns `x` unchanged if it's already an array and converts it if it's a list, so the function accepts both. The checks go from the cheapest to the most detailed. Shapes come first, because `isclose` broadcasts: a single value would be compared with every element and pass, and two incompatible shapes would raise a broadcasting error instead of a readable report. Then `nan` positions: comparing the two `isnan` masks with `!=` marks exactly the cells where one side has `nan` and the other doesn't, and `argwhere` lists them as `[row, col]` pairs, which is the readable form for a report. Then `isclose` with `equal_nan=True`, so matching `nan`s pass. For the worst element, the gaps are computed with `nan` positions set to 0 (they matched, so they're not mismatches), `argmax` finds the largest gap's position in the flattened array (Day 4: without `axis`, it counts through the elements row by row), and `np.unravel_index(flat_position, shape)` turns that flat position back into a `(row, col)` pair: in a `(2, 3)` array, flat position 4 is row 1, column 1. The `int(i)` conversion makes the report print `[1, 1]` rather than NumPy integer reprs, and `tuple(worst)` turns the list back into an index. The 1e-9 difference is within tolerance; the 0.02 and -0.5 are not, and -0.5 is the worst.

## Resources

- [NumPy reference: Sorting, searching, and counting](https://numpy.org/doc/stable/reference/routines.sort.html): `sort`, `argsort`, `searchsorted`, `nonzero`, `argwhere`, `count_nonzero` and more, each with examples.
- [NumPy reference: `numpy.where`](https://numpy.org/doc/stable/reference/generated/numpy.where.html): both forms, with the note recommending `nonzero` for the one-argument case.
- [NumPy reference: `isclose`](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html): the exact tolerance formula and its notes on choosing `atol`.
- [NumPy reference: `numpy.unique`](https://numpy.org/doc/stable/reference/generated/numpy.unique.html): every `return_*` option, with examples.
- [Python Data Science Handbook, Chapter 2](https://jakevdp.github.io/PythonDataScienceHandbook/02.00-introduction-to-numpy.html): free online; "Comparisons, Masks, and Boolean Logic" and "Sorting Arrays" cover Parts 1 and 2 with more examples.
- [Python for Data Analysis, Chapter 4](https://wesmckinney.com/book/numpy-basics): free online; conditional logic with `where`, sorting, and unique values.

## Checklist

- [ ] I can choose values per element with `np.where(cond, a, b)`, I know both `a` and `b` are computed in full, and I can use `np.select` for several cases.
- [ ] I can find where a condition holds with one-argument `np.where`, `np.nonzero` and `np.argwhere`, count it with `np.count_nonzero`, and I know `argwhere` isn't an index.
- [ ] I can limit values with `np.clip`, and I clip before converting floats to `uint8`.
- [ ] I can sort with `np.sort` and `.sort()`, choose the axis, and sort in descending order.
- [ ] I can use `np.argsort` for top-k, to sort one array by another, to sort table rows by a column, and to compute ranks.
- [ ] I can count categories with `np.unique(..., return_counts=True)` and encode labels with `return_inverse=True`.
- [ ] I can bucket values with `np.searchsorted`, and I sort its input first.
- [ ] I can compute running totals and differences with `np.cumsum`, `np.cumprod` and `np.diff`, and predict their shapes.
- [ ] I know where `nan` and `inf` come from, why `nan != nan`, how to detect them, and how to summarise data with gaps using the `nan*` functions.
- [ ] I can silence or raise NumPy's floating-point warnings with `np.errstate`.
- [ ] I can compare float arrays with `np.isclose` / `np.allclose`, explain `rtol` and `atol`, and know when `np.array_equal` is right.

**Next:** [Day 4c: Random numbers, saving arrays and statistics](day-04c-random-io-stats.md)
