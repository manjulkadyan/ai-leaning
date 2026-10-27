# Day 8: Shapes, Reshaping and Indexing

**Phase 1 · Week 2 · ~150 minutes**

Day 4 built the array: one block of memory plus a header (`dtype`, `shape`, `strides`). Today is everything you do to the *header*: give the data a new shape, add and remove axes, reorder axes, pick elements by a list of positions, and glue arrays together or cut them apart. It's long; a good break point is after Part 3, "Reordering axes".

## Objectives

- [ ] Explain why `reshape` is usually free, compute the flat position of any element with \( i \cdot C + j \), and convert between positions and `(row, col)` with `np.ravel_multi_index` and `np.unravel_index`.
- [ ] Use `-1` in `reshape`, read its two error messages, and say when `reshape` and `ravel` have to copy (and how `copy=False` catches it).
- [ ] Tell `flatten` (always a copy) from `ravel` (a view when it can be).
- [ ] Add a batch axis with `None`, `np.newaxis` or `np.expand_dims`, and remove size-1 axes with `squeeze(axis=...)`, avoiding the 0-d array trap.
- [ ] Reorder axes with `np.transpose`, `.T`, `np.moveaxis` and `np.swapaxes`; convert HWC ↔ CHW and NHWC (LiteRT) ↔ NCHW (PyTorch); check contiguity and fix it with `np.ascontiguousarray`; flip RGB ↔ BGR.
- [ ] Pick elements with integer-array ("fancy") indexing, predict the result's shape, take a block with `np.ix_`, gather one value per row with paired indices and `np.take_along_axis`, and avoid the chained-assignment and repeated-index traps.
- [ ] Join arrays with `np.stack` and `np.concatenate` (and know `vstack`, `hstack`, `column_stack`), cut them with `np.split` and `np.array_split`, copy with `np.repeat` and `np.tile`, and build a batch without copying it over and over.

**Prerequisites.**

- [Day 2b](day-02b-collections.md): tuples, slicing with a step (including `[::-1]`), and list comprehensions such as `[f(x) for x in items]`.
- [Day 3b](day-03b-errors-files-json.md): reading an error message, and `try` / `except ... as e`.
- [Day 4](day-04-numpy-arrays.md): the buffer-plus-header model (`dtype`, `shape`, `strides`, row-major order), `np.arange`, `np.zeros`, `np.eye`, the rule "an integer index removes an axis, a slice keeps it", `...` (Ellipsis), views versus copies and `np.shares_memory`, fancy indexing as a copy, `astype` and type promotion, `argmax`, and how NumPy scalars display (`np.int64(6)`).
- [Day 4b](day-04b-numpy-toolkit.md): `np.argsort` with `axis=` and `np.array_equal`.
- [Day 4c](day-04c-random-io-stats.md): `np.random.default_rng`, `rng.random`, `rng.integers(..., dtype=...)`, `rng.permutation`, and `np.column_stack` (met once there; today explains it).

!!! info "How to read the code on this page"
    Every Python block has a **▶ Run** button, and blocks share their variables from top to bottom like notebook cells, so run them in order. A bare expression on its own line shows its `repr()`, exactly as the `# comment` beside or below it says: arrays as `array([...])` with commas, single numbers as `np.int64(9)` or `np.float64(0.4)`. Lines that fail on purpose are wrapped in `try` / `except` so the rest of the block still runs, and the comment shows the real error message. The outputs come from 64-bit Python; in the browser NumPy's default integer is `int32`, so ▶ Run shows `np.int32(9)` and `dtype('int32')` where this page shows `int64`, and integer arrays report half the strides ([How lessons work](../../how-lessons-work.md) explains why).

---

## The big idea

On Day 4 you learned that an array is one flat block of memory plus a **shape**. Today's lesson takes that seriously: most of the "plumbing" in ML code is changing the shape of an array so it fits what the next function or model expects. An image arrives as `(224, 224, 3)`; a LiteRT model wants `(1, 224, 224, 3)`; PyTorch wants `(1, 3, 224, 224)`; the model returns `(1, 1000)` and you want `(1000,)`.

**Analogy.** Think of an egg carton holding 12 eggs in one long row. You can label the carton "3 rows of 4" or "4 rows of 3" without touching a single egg. You agree on a new way to count them. That's `reshape`. Taking the eggs out and arranging them in a different order (say, all brown eggs first) is a different operation: that's `transpose`, and it really does change which egg comes next.

Three questions decide every operation on this page, and the lesson asks them each time:

1. What **shape** comes out?
2. Is the result a **view** of the same memory, or a **copy**?
3. Did the **meaning** of each axis survive?

---

**Part 1: Reshaping**

## Reshape: same data, new shape

**What it is.** `a.reshape(new_shape)` returns an array with the same elements, in the same reading order, arranged into a different shape. The total number of elements (the **size**) must stay the same: 12 elements can become `(3, 4)`, `(4, 3)`, `(2, 2, 3)` or `(12,)`, but not `(5, 3)`.

**Why it exists.** Data rarely arrives in the shape a function wants. A flat list of 784 pixel values is really a 28 × 28 image. A batch of images `(32, 28, 28)` has to become `(32, 784)` before a fully connected layer can take it. A model returns 12 numbers that are really 4 boxes × 3 values each. Reshape changes the description of the data without moving it.

**Kotlin comparison.** Kotlin has no reshape; you do the index arithmetic yourself over a flat array:

```kotlin
val flat = IntArray(12) { it }                     // 0, 1, ..., 11
fun at(i: Int, j: Int, cols: Int) = flat[i * cols + j]
at(2, 1, cols = 4)    // 9   read as 3 rows of 4
at(3, 0, cols = 3)    // 9   read as 4 rows of 3: same slot
```

What's the same: one flat block, and the "shape" is only a rule for turning `(i, j)` into a position. What's different: NumPy stores that rule in the array's header, so `g[2, 1]` does the arithmetic for you, and every NumPy function respects the shape.

**Example.**

<figure class="diagram" markdown>
![Animation: twelve cells numbered 0 to 11 sit in a row with shape (12,), then move into a 3 by 4 grid, then into a 4 by 3 grid, always filling row by row in the same order, while a memory strip below holding 0 to 11 never changes](../../assets/img/phase-1/day08-reshape-flow.svg)
<figcaption>Animated: reshape relabels the same block of memory. Numbers stay in reading order; only the row length changes.</figcaption>
</figure>

```python
import numpy as np

a = np.arange(12)
a.shape            # (12,)

g = a.reshape(3, 4)
g
# array([[ 0,  1,  2,  3],
#        [ 4,  5,  6,  7],
#        [ 8,  9, 10, 11]])

h = a.reshape(4, 3)
h
# array([[ 0,  1,  2],
#        [ 3,  4,  5],
#        [ 6,  7,  8],
#        [ 9, 10, 11]])

np.shares_memory(a, g), np.shares_memory(a, h)
# (True, True)
```

`np.shares_memory` (Day 4) says both results use `a`'s bytes: no data was copied. So writing through one is visible through the others:

```python
g[0, 0] = 100
a[0]               # np.int64(100)    g is a view onto a's memory
a[0] = 0           # put it back for the rest of the page
```

(`a[0]` is a single element, a NumPy scalar, which echoes as `np.int64(...)`. In the browser it shows `np.int32(...)`.)

The rule NumPy uses to "fill" the new shape is **row-major order**, also called **C order** (the C language lays out 2-D arrays this way): fill row 0 from left to right, then row 1, and so on. The *last* axis changes fastest.

**How it works underneath: from a position to an address.** Say the array has shape \( (R, C) \): \( R \) rows and \( C \) columns. The element in row \( i \), column \( j \) (both counted from 0) lives at flat position

\[
\text{position}(i, j) = i \cdot C + j
\]

In words: skip \( i \) full rows of \( C \) elements each, then step \( j \) more.

Worked example by hand. In `g` (shape \( (3, 4) \), so \( C = 4 \)), element `g[2, 1]` is at \( 2 \cdot 4 + 1 = 9 \). In `h` (shape \( (4, 3) \), \( C = 3 \)), position 9 is row \( 9 \div 3 = 3 \) remainder \( 0 \), so `h[3, 0]`. Both are the value 9, because both point at the same slot. For three axes, shape \( (A, B, C) \), the same idea stacks up: \( \text{position}(i, j, k) = i \cdot BC + j \cdot C + k \).

NumPy computes this for you in both directions. `np.ravel_multi_index(index, shape)` turns a `(row, col)` tuple into a flat position, and `np.unravel_index(position, shape)` turns a flat position back into one index per axis:

```python
g[2, 1], h[3, 0], a[9]
# (np.int64(9), np.int64(9), np.int64(9))
np.ravel_multi_index((2, 1), (3, 4))
# np.int64(9)
np.unravel_index(9, (4, 3))
# (np.int64(3), np.int64(0))
```

You'll use `unravel_index` whenever `argmax` on a 2-D array gives you a flat position (it does, unless you pass `axis=`) and you want the row and column:

```python
scores = np.array([[0.1, 0.7, 0.2],
                   [0.9, 0.3, 0.4]])
scores.argmax()                              # np.int64(3)   position in the flattened array
np.unravel_index(scores.argmax(), scores.shape)
# (np.int64(1), np.int64(0))                 row 1, column 0: the 0.9
```

Day 4 introduced **strides**: the number of bytes to jump to get to the next element along each axis. Reshape's real work is computing new strides for the new shape. The strides are the position formula multiplied by the item size. This block uses an explicit `int64` array so the numbers are the same in the browser:

```python
s = np.arange(12, dtype=np.int64)
s.itemsize                      # 8               bytes per element
s.reshape(3, 4).strides         # (32, 8)         next column +8 bytes, next row +4 × 8
s.reshape(4, 3).strides         # (24, 8)         rows are 3 long now: +3 × 8
s.reshape(2, 2, 3).strides      # (48, 24, 8)     B·C·8, C·8, 8
```

So reshape writes a new header (a shape tuple and a strides tuple) and points it at the same buffer. That's why its cost doesn't depend on the array's size.

**How it goes wrong: the sizes don't match.** The new shape must hold exactly the same number of elements:

```python
try:
    a.reshape(5, 3)
except ValueError as e:
    print("ValueError:", e)
# ValueError: cannot reshape array of size 12 into shape (5,3)
```

The message gives both numbers: the size you have (12) and the shape you asked for (5 × 3 = 15). It usually means an upstream step produced a different size than you assumed, such as an image that was resized to 223 pixels, or a dataset with one extra column. Print the `.shape` of the input before the reshape.

!!! android "Android bridge"
    You've done this by hand. `Bitmap.getPixels()` fills a flat `IntArray`, and you read pixel `(x, y)` as `pixels[y * width + x]`. That is exactly \( i \cdot C + j \) with \( i = y \) and \( C = \) `width`. A NumPy shape is that index arithmetic, built in. Likewise `ByteBuffer.asFloatBuffer()` gives a new *view* of the same bytes without copying, the way `reshape` does.

### `-1`: "you work it out"

**What it is.** One dimension in `reshape` can be `-1`, meaning "whatever makes the total come out right". NumPy computes it as total size ÷ product of the known sizes, so `a.reshape(3, -1)` gives \( 12 \div 3 = 4 \).

**Why it exists.** You often know every axis but one. "Keep the batch axis, squash everything else into one long feature vector" is `x.reshape(len(x), -1)`, and it works for any image size.

```python
a.reshape(3, -1).shape      # (3, 4)
a.reshape(-1, 6).shape      # (2, 6)
a.reshape(2, -1, 3).shape   # (2, 2, 3)
a.reshape(-1).shape         # (12,)    flatten to 1-D

batch = np.zeros((32, 28, 28))
batch.reshape(len(batch), -1).shape   # (32, 784)   28 × 28 = 784 features per image
```

**How it goes wrong.** Two `-1`s leave the answer undetermined, and a `-1` that doesn't divide evenly has no answer:

```python
try:
    a.reshape(-1, -1)
except ValueError as e:
    print("ValueError:", e)
# ValueError: can only specify one unknown dimension

try:
    a.reshape(5, -1)
except ValueError as e:
    print("ValueError:", e)
# ValueError: cannot reshape array of size 12 into shape (5,newaxis)
```

The second message writes the unknown dimension as `newaxis`; 12 isn't divisible by 5.

## When reshape has to copy

**What it is.** `reshape` returns a view only when the new shape can be described by **one stride per axis** over the existing buffer. When it can't, `reshape` silently makes a copy in the new order and returns that. `ravel` (next section) follows the same rule.

**Why it matters.** Two reasons. A copy of a large array costs memory and time, which you don't expect from an "instant" operation. And writes through a copy don't reach the original, so code that relied on the view behaviour silently stops working.

**Example.** The usual way to get there is reshaping something that was transposed first. `.T` swaps rows and columns (Part 3 covers it); like reshape, it's a view with new strides.

```python
m = np.arange(6).reshape(2, 3)
m
# array([[0, 1, 2],
#        [3, 4, 5]])
m.T
# array([[0, 3],
#        [1, 4],
#        [2, 5]])
m.T.reshape(-1)
# array([0, 3, 1, 4, 2, 5])
np.shares_memory(m, m.T), np.shares_memory(m, m.T.reshape(-1))
# (True, False)
```

`m.T` shares memory; `m.T.reshape(-1)` doesn't. Reshape had to copy.

**How it works underneath.** In memory, `m`'s buffer holds `0 1 2 3 4 5` in that order. Reading `m.T` row by row, as reshape must, visits `0 3 1 4 2 5`: memory slots 0, 3, 1, 4, 2, 5. A view of shape `(6,)` has one stride, a single fixed jump between neighbours, but these jumps are +3, −2, +3, −2, +3. No single stride walks that order, so NumPy copies the elements into a new buffer in reading order. Reshaping the *original* `m` to `(6,)` needs stride 8 (one element), which works, so that's a view.

The rule of thumb: reshaping a freshly created or sliced-by-rows array is a view; reshaping after `.T`, `transpose`, or a step slice like `[:, ::2]` may copy.

**Asking for a guarantee: `copy=False`.** Since NumPy 2.1, `reshape` takes `copy=`. With `copy=False` it raises instead of copying, so you find out:

```python
try:
    m.T.reshape(-1, copy=False)
except ValueError as e:
    print("ValueError:", e)
# ValueError: Unable to avoid creating a copy while reshaping.

m.reshape(-1, copy=False)
# array([0, 1, 2, 3, 4, 5])           fine: this one can be a view
```

**How it goes wrong: writing through a reshaped array that was a copy.**

```python
flat = m.T.reshape(-1)     # a copy, silently
flat[0] = 100
m[0, 0]                    # np.int64(0)    m didn't change
```

If later code expected `m` to change, it won't, and there's no error. Recognise it with `np.shares_memory(original, result)`; fix it by doing the write on the original's indices, or by using `copy=False` where you depend on a view.

!!! warning "Common mistake: assigning to `.shape`"
    Old code reshapes in place with `arr.shape = (6,)`. It fails when a view is impossible, and NumPy 2.5 deprecates it altogether (the browser's NumPy 2.4 doesn't warn yet). This transcript is from a laptop:

    ```pycon
    >>> b = np.arange(6).reshape(2, 3).T
    >>> b.shape = (6,)
    <stdin-2>:1: DeprecationWarning: Setting the shape on a NumPy array has been deprecated in NumPy 2.5.
    As an alternative, you can create a new view using np.reshape (with copy=False if needed).
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
        b.shape = (6,)
        ^^^^^^^
    AttributeError: Incompatible shape for in-place modification. Use `.reshape()` to make a copy with the desired shape.
    ```

    **Fix:** `b = b.reshape(6)`, or `b.reshape(6, copy=False)` if a copy would be a bug.

## `flatten` vs `ravel`

**What they are.** Both turn any array into 1-D, in row-major order. `flatten()` **always** returns a new copy. `ravel()` returns a view when it can (the previous section's rule) and a copy only when it must. `a.reshape(-1)` behaves exactly like `ravel()`.

**Why both exist.** `ravel` is for reading: passing data to a function that wants 1-D, such as a histogram, costs nothing. `flatten` is for when you'll modify the result and must not touch the original.

**Kotlin comparison.** Kotlin's `flatten()` always builds a new list, like NumPy's `flatten()`:

```kotlin
val rows = listOf(listOf(0, 1, 2), listOf(3, 4, 5))
val flat = rows.flatten()     // [0, 1, 2, 3, 4, 5], a new list
```

Kotlin has no flattening *view*; a `List<List<Int>>` is separate objects, so there's no single buffer to view. NumPy's 2-D array is already one buffer, which is what makes `ravel` free.

**Example.** Each write is on its own line, followed by what it did to `m2`:

```python
m2 = np.arange(6).reshape(2, 3)
f = m2.flatten()     # always a copy
r = m2.ravel()       # a view here

f[0] = 99
m2[0, 0]             # np.int64(0)     m2 untouched
r[0] = 77
m2[0, 0]             # np.int64(77)    m2 changed through r
np.shares_memory(m2, f), np.shares_memory(m2, r)
# (False, True)
```

**How it goes wrong.** The mistake is assuming `ravel` gives you a private copy. You "clean" the flattened values in place and the original image changes with them, as `m2` did above. **Fix:** use `flatten()`, or `ravel().copy()`, when you intend to modify the result. (And the reverse surprise: `ravel()` after a transpose is a copy, `np.shares_memory(m, m.T.ravel())` is `False`, so writes don't go back.)

---

**Part 2: Adding and removing axes**

## Adding a size-1 axis: `None`, `np.newaxis`, `np.expand_dims`

**What it is.** A **size-1 axis** is an axis of length 1. Adding one changes the shape, `(3,)` → `(1, 3)` or `(3, 1)`, without changing the number of elements. You add it by putting `None` (or its alias `np.newaxis`) in the index at the position you want, or with `np.expand_dims(a, axis=k)`.

**Why it exists.** A model rarely takes one example. It takes a **batch**, with the batch size as the first axis. LiteRT (Google's on-device ML runtime, formerly TensorFlow Lite or TFLite) image classifiers typically declare an input of `[1, 224, 224, 3]`: one image, 224 high, 224 wide, 3 colour channels. Your photo is `(224, 224, 3)`, so you need one extra axis of size 1 at the front. Size-1 axes are also how you turn a vector into a row or a column, which Day 9's broadcasting relies on.

**Kotlin comparison.** Making a one-item batch in Kotlin means wrapping: `listOf(image)` or `arrayOf(pixels)`. That creates a new outer container holding a reference to the same inner data. `img[None]` is the same idea, and also shares the data.

**Example.**

```python
rng = np.random.default_rng(0)
img = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)   # a random "photo"
img.shape                          # (224, 224, 3)

img[np.newaxis].shape              # (1, 224, 224, 3)
img[None, ...].shape               # (1, 224, 224, 3)
np.expand_dims(img, axis=0).shape  # (1, 224, 224, 3)

np.newaxis is None                 # True    np.newaxis is another name for None
np.shares_memory(img, img[None])   # True    adding a size-1 axis copies nothing
```

`img[None, ...]` reads "a new axis, then all the existing axes" (`...` from Day 4). `None` can go anywhere in the index; it inserts a size-1 axis at that position:

```python
v = np.array([1, 2, 3])
v[:, None]
# array([[1],
#        [2],
#        [3]])                     a column, shape (3, 1)
v[None, :]
# array([[1, 2, 3]])               a row, shape (1, 3)
np.expand_dims(v, axis=1).shape    # (3, 1)    the same as v[:, None]
np.expand_dims(v, axis=-1).shape   # (3, 1)    -1 means "at the end"
```

`expand_dims`' `axis` is the position the new axis will have *in the result*.

**How it works underneath.** The new axis has length 1, so its stride never gets used. NumPy sets it to 0:

```python
img[None].strides                  # (0, 672, 3, 1)
```

The other three strides are unchanged (`uint8`, so 1 byte per element; a row is 224 × 3 = 672 bytes). Only the header changed.

**How it goes wrong (1): an axis number out of range.** `axis` counts positions in the result, which has one more axis than the input, so for a 2-D input the valid values are 0, 1 and 2 (or −1 to −3):

```python
try:
    np.expand_dims(np.zeros((2, 3)), axis=3)
except np.exceptions.AxisError as e:
    print("AxisError:", e)
# AxisError: axis 3 is out of bounds for array of dimension 3
```

**How it goes wrong (2): the right number of axes in the wrong place.** This one doesn't raise. `img[:, None]` puts the new axis *after* the first:

```python
img[:, None].shape                 # (224, 1, 224, 3)   not a batch
```

A model expecting `(1, 224, 224, 3)` rejects that shape, if you're lucky; a function that only checks `ndim == 4` accepts it and computes nonsense. **Fix:** put `None` first (`img[None]`), or use `np.expand_dims(img, 0)`, and assert the full shape: `assert x.shape == (1, 224, 224, 3)`.

## Removing size-1 axes: `squeeze`

**What it is.** `a.squeeze()` removes axes of size 1. With no argument it removes **all** of them; with `axis=k` (or a tuple of axes) it removes only those, and raises an error if one isn't size 1.

**Why it exists.** Model outputs come back with a batch axis you usually don't want: one image in, `(1, 1000)` scores out, and you want `(1000,)` to find the top class.

**Kotlin comparison.** The Android equivalent is `output[0]`: with LiteRT's `Interpreter` you allocate `Array(1) { FloatArray(1000) }`, call `run`, and read `output[0]`. In NumPy, `out[0]` works too, and so does `squeeze(axis=0)`, which also checks that the axis really had size 1.

**Example.**

```python
out = np.zeros((1, 1000), dtype=np.float32)   # e.g. 1000 class scores for 1 image
out.squeeze().shape            # (1000,)
np.squeeze(out, axis=0).shape  # (1000,)   safer: say which axis
out[0].shape                   # (1000,)   an integer index also removes the axis

x = np.zeros((1, 3, 1, 5))
x.squeeze().shape              # (3, 5)      removes EVERY size-1 axis
x.squeeze(axis=2).shape        # (1, 3, 5)
x.squeeze(axis=(0, 2)).shape   # (3, 5)

try:
    x.squeeze(axis=1)
except ValueError as e:
    print("ValueError:", e)
# ValueError: cannot select an axis to squeeze out which has size not equal to one
```

**How it works underneath.** Like adding an axis, squeezing only rewrites the header: the size-1 axis and its stride are dropped. `np.shares_memory(out, out.squeeze())` is `True`.

**A 0-d array.** Squeezing an array whose axes are *all* size 1 leaves an array with **no axes at all**: shape `()`, `ndim` 0, exactly one element. It's called a **0-d array**. It prints as `array(0.)`, it isn't a NumPy scalar or a Python float, and you can't index it with `[0]` because there's no axis to index:

```python
one = np.zeros((1, 1)).squeeze()
one                    # array(0.)
one.shape, one.ndim    # ((), 0)
one.item()             # 0.0       .item() (Day 4) gets the plain Python number out
```

!!! warning "Common mistake: bare `squeeze()` on a batch of one"
    `squeeze()` with no axis removes *all* size-1 axes. Your code works on a batch of 32 and breaks on the day the batch has one item, or a model has one output:

    ```python
    probs = np.array([[0.83]])        # (batch=1, outputs=1)
    p = probs.squeeze()
    p.shape                           # ()
    try:
        p[0]
    except IndexError as e:
        print("IndexError:", e)
    # IndexError: too many indices for array: array is 0-dimensional, but 1 were indexed
    ```

    **Fix:** always pass `axis=`: `probs.squeeze(axis=1)` has shape `(1,)` whatever the batch size, and `squeeze(axis=0)` raises a clear error if the batch isn't 1.

---

**Part 3: Reordering axes**

## Transpose: changing the order of axes

**What it is.** `reshape` keeps reading order. Sometimes you need to change the *meaning* of the axes, for example from height-width-channels (**HWC**) to channels-height-width (**CHW**). That's a **transpose** (also called a **permutation of axes**): you give the new order of the old axes.

**Why it exists.** Different libraries expect different axis orders (the NHWC/NCHW section below). Converting between them is a transpose, and getting it wrong is one of the most common silent bugs in ML preprocessing.

**Kotlin comparison.** Transposing a `List<List<Int>>` in Kotlin builds new lists:

```kotlin
val rows = listOf(listOf(0, 1, 2), listOf(3, 4, 5))
val cols = (0 until 3).map { j -> rows.map { it[j] } }   // [[0, 3], [1, 4], [2, 5]]
```

That copies every element. NumPy's transpose copies nothing: it reorders the shape and the strides in the header, so the same bytes are read in a different order.

**Example.** `img` from Part 2 is HWC:

```python
chw = np.transpose(img, (2, 0, 1))   # new axis 0 = old axis 2 (C), then old 0 (H), then old 1 (W)
chw.shape                    # (3, 224, 224)

img[10, 20]                  # array([  2, 130, 173], dtype=uint8)   pixel (10, 20): R, G, B
chw[:, 10, 20]               # array([  2, 130, 173], dtype=uint8)   the same three values

back = np.transpose(chw, (1, 2, 0))
back.shape, np.array_equal(back, img)
# ((224, 224, 3), True)
```

Read `(2, 0, 1)` as "take old axis 2 first, then old axis 0, then old axis 1". To undo it, you need the inverse order, `(1, 2, 0)`: the new position of each old axis.

There are two ways to call it, and you'll see both: the function with a tuple, `np.transpose(img, (2, 0, 1))`, and the method with separate arguments, `img.transpose(2, 0, 1)`. They do the same thing (the method also accepts a tuple).

Two helpers name the move instead of the whole order:

```python
np.moveaxis(img, -1, 0).shape        # (3, 224, 224)   move the last axis to the front
np.moveaxis(chw, 0, -1).shape        # (224, 224, 3)   and back
np.swapaxes(img, 0, 1).shape         # (224, 224, 3)   swap H and W (a 90° flip of the image)
np.array_equal(np.moveaxis(img, -1, 0), chw)
# True
```

`np.moveaxis(a, source, destination)` moves one axis and keeps the others in order, which is easier to read than a full permutation on a 4-D or 5-D array. `np.swapaxes(a, i, j)` exchanges two axes.

**How it works underneath.** The transpose permutes the strides the same way it permutes the shape:

```python
img.strides                  # (672, 3, 1)     next row +672 bytes, next pixel +3, next channel +1
chw.strides                  # (1, 672, 3)     the same three numbers, reordered
chw.flags['C_CONTIGUOUS']    # False
```

`flags['C_CONTIGUOUS']` says whether the array is stored in row-major order with no gaps, as if it had been created in this shape. `chw` isn't: walking it in reading order hops around the buffer. NumPy doesn't mind, but code that reads the raw bytes does, such as a runtime that copies the buffer into a tensor, or a `ByteBuffer` you fill from `arr.tobytes()`. When you need the bytes in CHW order for real, ask for a contiguous copy:

```python
c = np.ascontiguousarray(chw)
c.strides                    # (50176, 224, 1)   a plain (3, 224, 224) layout
c.flags['C_CONTIGUOUS']      # True
np.shares_memory(c, img)     # False             the data really moved
```

(`np.ascontiguousarray` returns the input unchanged if it's already contiguous, so it's safe to call by habit before handing data over.)

**How it goes wrong (1): the permutation doesn't match the array.** The order must name every axis exactly once:

```python
try:
    np.transpose(img, (0, 1))
except ValueError as e:
    print("ValueError:", e)
# ValueError: axes don't match array
```

**How it goes wrong (2): `reshape` where you meant `transpose`.**

!!! warning "Common mistake: reshape instead of transpose"
    `img.reshape(3, 224, 224)` has the right *shape* and no error, but it's garbage: it re-cuts the interleaved R,G,B,R,G,B… stream into three chunks, mixing channels. See it on a tiny image:

    ```python
    small = np.arange(12).reshape(2, 2, 3)   # H=2, W=2, C=3
    small.transpose(2, 0, 1)[0]
    # array([[0, 3],
    #        [6, 9]])          the red plane: element 0 of every pixel
    small.reshape(3, 2, 2)[0]
    # array([[0, 1],
    #        [2, 3]])          R, G, B of pixel 0 plus R of pixel 1: nonsense
    ```

    A model fed this still returns confident predictions. They're wrong. Rule: **if the meaning of the axes changes, use `transpose`; if you're only grouping or splitting axes, use `reshape`.**

### A quick word on `.T`

For a 2-D array, `.T` swaps rows and columns: shape `(2, 3)` becomes `(3, 2)`. It's shorthand for `transpose()` with the axes reversed, so on a 3-D array it reverses all three axes, `(H, W, C)` → `(C, W, H)`, which is rarely what you want for images; on a 1-D array it does nothing. Day 11 covers what transposing means for matrices.

```python
np.arange(6).reshape(2, 3).T
# array([[0, 3],
#        [1, 4],
#        [2, 5]])
img.T.shape          # (3, 224, 224)   C, W, H: W and H are swapped too
np.arange(3).T.shape # (3,)            a 1-D array has nothing to swap
```

### NHWC vs NCHW

With a batch axis **N** in front, there are two conventions for images:

<figure class="diagram" markdown>
![Diagram of a 2 by 2 RGB image in two layouts. NHWC, used by LiteRT and TFLite, stores each pixel's R, G, B values next to each other: R0 G0 B0 R1 G1 B1 and so on. NCHW, used by PyTorch, stores a full plane of red, then green, then blue: R0 R1 R2 R3 G0 G1 G2 G3 B0 B1 B2 B3. An animated highlight shows pixel 0's three values are neighbours in NHWC but 4 apart in NCHW](../../assets/img/phase-1/day08-nhwc-nchw.svg)
<figcaption>Animated: the same image, two orders in memory. The highlight follows pixel 0's red, green and blue values.</figcaption>
</figure>

| Layout | Shape of a batch | Who uses it |
|---|---|---|
| **NHWC** "channels last" | `(N, H, W, C)`, e.g. `(1, 224, 224, 3)` | TensorFlow, Keras, LiteRT/TFLite, Android `Bitmap` pixels |
| **NCHW** "channels first" | `(N, C, H, W)`, e.g. `(1, 3, 224, 224)` | PyTorch (`nn.Conv2d` expects it) |

Why two? Partly history, partly hardware. PyTorch inherited channels-first from early GPU libraries such as NVIDIA's cuDNN. TensorFlow's mobile runtime went channels-last, which matches how cameras and `Bitmap`s deliver pixels (R, G, B together per pixel) and suits many CPU and mobile **kernels** (the low-level compiled routines that do one operation, such as a convolution) that process all channels of a pixel at once. Neither is "right"; you convert at the boundary, with the batch axis staying in place:

```python
nhwc = np.zeros((32, 224, 224, 3), dtype=np.float32)
nchw = nhwc.transpose(0, 3, 1, 2)
nchw.shape                           # (32, 3, 224, 224)   NHWC → NCHW
nchw.transpose(0, 2, 3, 1).shape     # (32, 224, 224, 3)   NCHW → NHWC
```

Read `(0, 3, 1, 2)` as: N stays first, then old axis 3 (C), then H, then W.

!!! android "Android bridge"
    When you fill a `ByteBuffer` for a LiteRT `Interpreter`, the usual loop goes pixel by pixel and writes R, G, B for each one. That *is* NHWC. If a model was exported from PyTorch it may still declare an NCHW input such as `[1, 3, 224, 224]`, and the same loop would feed it scrambled data. Check `interpreter.getInputTensor(0).shape()` before writing preprocessing code, and match it. In Python, the matching step is `np.ascontiguousarray(batch.transpose(0, 3, 1, 2))`: transpose for the order, contiguous so the bytes are really in that order.

### Flipping channels: RGB ↔ BGR

**What it is.** Some libraries store colour channels in the order blue, green, red (**BGR**): OpenCV's `cv2.imread` does, so do some older models. Converting is reversing the last axis: `img[..., ::-1]`.

**Why it matters.** A model trained on RGB that's fed BGR still runs, and its accuracy quietly drops (Day 9 warns about it in its normalisation example). One line fixes it, once you know to check.

**Example.** `[..., ::-1]` is "all the leading axes in full, and the last axis with step −1" (Day 2b's reversed slice, applied to one axis):

```python
pixel = np.array([[[255, 128, 0]]], dtype=np.uint8)   # one orange pixel, RGB, shape (1, 1, 3)
pixel[..., ::-1]
# array([[[  0, 128, 255]]], dtype=uint8)             the same pixel as BGR
np.flip(pixel, axis=-1)
# array([[[  0, 128, 255]]], dtype=uint8)             np.flip names the axis instead
```

**How it works underneath.** A reversed slice is a view with a **negative stride**: start at the last channel and step −1 byte.

```python
bgr = img[..., ::-1]
bgr.strides                    # (672, 3, -1)
np.shares_memory(bgr, img)     # True
```

**How it goes wrong.** Some consumers can't read negative strides. The best known is PyTorch's `torch.from_numpy`, which refuses the array with a `ValueError` saying at least one stride is negative and suggesting `array.copy()`. **Fix:** `np.ascontiguousarray(img[..., ::-1])` before handing the array over; it copies into ordinary positive strides.

---

**Part 4: Fancy indexing**

## Picking with a list of indices

**What it is.** Day 4's slices (`a[1:4]`) pick a regular range. **Fancy indexing** (NumPy's docs call it *integer array indexing*) passes an array or list of integer positions instead, in any order, with repeats allowed. The result is always a new array, a copy (Day 4 showed why: an arbitrary list of positions can't be described with strides).

**Why it exists.** Reordering (`a[order]` after an `argsort`, Day 4b), sampling a batch (`X[batch_idx]`), shuffling with a permutation (Day 4b), looking up rows of an embedding table by token id (`E[token_ids]`, Month 5), and turning labels into one-hot vectors (Exercise 5).

**Kotlin comparison.** Kotlin's `slice` with a list of indices is exactly this, and it also returns a new list:

```kotlin
val a = listOf(10, 20, 30, 40, 50)
a.slice(listOf(0, 2, 4))     // [10, 30, 50]
a.slice(listOf(4, 0, 0))     // [50, 10, 10]
```

What's different: NumPy's version works on any axis of a multi-dimensional array, and the index can itself be multi-dimensional.

**Example.**

```python
a = np.array([10, 20, 30, 40, 50])
a[[0, 2, 4]]        # array([10, 30, 50])
a[[4, 0, 0]]        # array([50, 10, 10])     any order, repeats allowed
a[[-1, -2]]         # array([50, 40])          negative positions count from the end

m = np.arange(12).reshape(3, 4)
m[[2, 0]]
# array([[ 8,  9, 10, 11],
#        [ 0,  1,  2,  3]])          rows 2 then 0
m[:, [3, 1]]
# array([[ 3,  1],
#        [ 7,  5],
#        [11,  9]])                  columns 3 then 1, every row
```

**How it works underneath: the shape of the result.** Day 4's rule, extended. For each axis, a **slice keeps** the axis, an **integer removes** it, and an **index array puts its own shape** in place of the axis. So the index's shape decides the result's shape:

```python
a[np.array([[0, 1],
            [3, 4]])]
# array([[10, 20],
#        [40, 50]])                  a (2, 2) index gives a (2, 2) result
m[[2, 0]].shape        # (2, 4)      axis 0 replaced by the index's (2,); axis 1 kept
m[:, [3, 1]].shape     # (3, 2)      axis 0 kept; axis 1 replaced by (2,)
m[:, [3]].shape        # (3, 1)      a one-element list keeps the axis, like a slice
m[:, 3].shape          # (3,)        an integer removes it
```

Underneath, NumPy allocates a new array of the result shape and copies `a[i]` into it for every `i` in the index: a gather loop in C.

**How it goes wrong.** Three errors you'll meet:

```python
try:
    a[[1, 5]]
except IndexError as e:
    print("IndexError:", e)
# IndexError: index 5 is out of bounds for axis 0 with size 5

idx = np.array([1, 2]) * 1.0     # positions that came out of a calculation as floats
try:
    a[idx]
except IndexError as e:
    print("IndexError:", e)
# IndexError: arrays used as indices must be of integer (or boolean) type
a[idx.astype(int)]               # array([20, 30])   fix: convert to integers (Day 4)
```

The first means a position past the end; the whole indexing fails, not only that element. The second happens when the positions came from arithmetic like `n * 0.8` or `np.linspace` (which returns floats); NumPy refuses to guess how to round them. Convert with `.astype(int)` after deciding how to round (`np.round` first, if truncation isn't what you want).

The third error is about two index arrays, next.

## Two index arrays: pairs, not a block

**What it is.** With an index array on two axes, NumPy pairs them up **element by element**: `m[[0, 1, 2], [3, 2, 1]]` picks `m[0, 3]`, `m[1, 2]` and `m[2, 1]`. It does *not* take the block of rows 0–2 × columns 3, 2, 1.

**Why it exists.** Pairs are exactly what you need for "for each example, take the probability of its true label": the core of the cross-entropy loss (the number a classifier is trained to make small), which you'll meet in Phase 3.

<figure class="diagram" markdown>
![Two copies of the 3 by 4 grid m holding 0 to 11. On the left, m[[0, 1, 2], [3, 2, 1]] highlights the three cells (0,3), (1,2) and (2,1), holding 3, 6 and 9, and an arrow leads to a 1-D result of shape (3,): 3, 6, 9. On the right, m[np.ix_([0, 2], [1, 3])] highlights the four cells where rows 0 and 2 cross columns 1 and 3, holding 1, 3, 9 and 11, and an arrow leads to a 2 by 2 result](../../assets/img/phase-1/day08-fancy-pairs.svg)
<figcaption>Two index arrays pick pairs (left); np.ix_ turns them into a block of rows × columns (right).</figcaption>
</figure>

**Example.**

```python
m[[0, 1, 2], [3, 2, 1]]
# array([3, 6, 9])                   pairs (0,3), (1,2), (2,1)

probs = np.array([[0.1, 0.2, 0.7],
                  [0.6, 0.3, 0.1],
                  [0.2, 0.5, 0.3]])  # 3 examples × 3 classes
labels = np.array([2, 0, 1])         # the true class of each example
probs[np.arange(3), labels]
# array([0.7, 0.6, 0.5])             probs[0, 2], probs[1, 0], probs[2, 1]
```

`np.arange(3)` is `[0, 1, 2]`, one row number per example, paired with that example's label.

**Taking a block instead: `np.ix_`.** When you *do* want rows 0 and 2 crossed with columns 1 and 3, either index in two steps or use `np.ix_`, which builds index arrays shaped as a column and a row:

```python
m[[0, 2]][:, [1, 3]]
# array([[ 1,  3],
#        [ 9, 11]])
m[np.ix_([0, 2], [1, 3])]
# array([[ 1,  3],
#        [ 9, 11]])
np.ix_([0, 2], [1, 3])
# (array([[0],
#        [2]]), array([[1, 3]]))
```

The two-step version makes an intermediate copy (the selected rows) and then indexes that. `np.ix_` does it in one step, and unlike the two-step version it also works on the left of `=`. Why the shapes `(2, 1)` and `(1, 2)` produce a `(2, 2)` block is broadcasting, Day 9's topic: the two index arrays are stretched against each other before pairing.

**How it goes wrong: index arrays of different lengths.**

```python
try:
    m[[0, 1, 2], [1, 2]]
except IndexError as e:
    print("IndexError:", e)
# IndexError: shape mismatch: indexing arrays could not be broadcast together with shapes (3,) (2,)
```

Three row positions and two column positions can't be paired. If you meant a block, this error is the hint: use `np.ix_`.

## One value per row: `np.take_along_axis`

**What it is.** `np.take_along_axis(a, idx, axis=1)` picks, from each row of `a`, the columns listed in the same row of `idx`. `idx` has the same number of axes as `a`, so row 0 of `idx` holds the column numbers to take from row 0 of `a`, and so on.

**Why it exists.** Day 4b found the top-2 classes of each sample with `np.argsort(logits, axis=1)`. That gives *positions*; to get the matching *scores*, you need a per-row gather, and plain `logits[top2]` does something else entirely.

**Example.** The same logits as Day 4b:

```python
logits = np.array([[2.0, 0.1, 5.0, 1.0],
                   [0.3, 4.0, 0.2, 3.5]])
top2 = np.argsort(logits, axis=1)[:, ::-1][:, :2]
top2
# array([[2, 0],
#        [1, 3]])
np.take_along_axis(logits, top2, axis=1)
# array([[5. , 2. ],
#        [4. , 3.5]])
```

Row 0 takes columns 2 and 0 (5.0 and 2.0); row 1 takes columns 1 and 3 (4.0 and 3.5).

**How it works underneath.** It's the pairing rule with a row number for every element of `idx`. Spelled out by hand:

```python
rows = np.arange(2)[:, None]       # array([[0], [1]]): one row number per row, as a column
logits[rows, top2]
# array([[5. , 2. ],
#        [4. , 3.5]])
```

`rows` has shape `(2, 1)` and `top2` has shape `(2, 2)`; broadcasting (Day 9) stretches `rows` to `[[0, 0], [1, 1]]`, and then each position is paired as in the previous section. `take_along_axis` writes this for you and works along any axis.

**How it goes wrong: indexing with the positions directly.**

```python
try:
    logits[top2]
except IndexError as e:
    print("IndexError:", e)
# IndexError: index 2 is out of bounds for axis 0 with size 2
```

A single index array applies to axis 0 only, so `logits[top2]` treats `2, 0, 1, 3` as *row* numbers, and `logits` has only 2 rows. With a real batch, which usually has more rows than you have top-k positions, it's worse: there's no error, and you get whole rows arranged in `top2`'s shape. A batch of 32 with 10 classes gives `logits[top2].shape == (32, 2, 10)` instead of `(32, 2)`. The shape is the clue. Fix: `np.take_along_axis(logits, top2, axis=1)`.

## Assigning through a fancy index

**What it is.** A fancy index on the left of `=` writes into the original array at those positions. That's different from reading, which copies.

```python
x = np.array([1, 2, 3])
x[[0, 2]] = 5
x                          # array([5, 2, 5])
x[[0, 2]] = [7, 9]         # one value per position
x                          # array([7, 2, 9])
```

**How it goes wrong (1): chained indexing.** Day 4 showed this with masks; the fancy-index version is the same bug. `x[[0, 2]][0] = 5` first *reads* `x[[0, 2]]`, a copy, then writes into the copy, which is thrown away:

```python
x = np.array([1, 2, 3])
x[[0, 2]][0] = 5           # writes into a temporary copy
x                          # array([1, 2, 3])   unchanged
```

**Fix:** one indexing step on the left of `=`: `x[0] = 5`, or `x[[0, 2]] = ...`.

**How it goes wrong (2): repeated positions with `+=`.** This is a classic silent bug when counting:

```python
counts = np.zeros(4)
counts[[0, 0, 1]] += 1     # "add 1 for each occurrence"
counts                     # array([1., 1., 0., 0.])   position 0 got 1, not 2
```

`counts[[0, 0, 1]] += 1` runs as `counts[[0, 0, 1]] = counts[[0, 0, 1]] + 1`: it reads `[0, 0, 0]`, adds 1 to get `[1, 1, 1]`, and writes those three values back, so position 0 is written twice with the same `1`. **Fix:** `np.add.at`, which really does one addition per index:

```python
counts = np.zeros(4)
np.add.at(counts, [0, 0, 1], 1)
counts                     # array([2., 1., 0., 0.])
```

(For counting small integer labels, `np.bincount` from Day 4b's "Going further" is faster still.)

---

**Part 5: Joining and splitting**

## Stacking and concatenating

**What they are.** Two ways to glue a list of arrays together. The difference is whether you get a **new axis**:

- `np.concatenate(arrays, axis=k)` joins along an **existing** axis `k` (default 0). The inputs must match on every other axis.
- `np.stack(arrays, axis=k)` adds a **new** axis at position `k` (default 0) and lines the inputs up along it. The inputs must all have exactly the same shape.

**Why they exist.** Building a batch from separate images (`stack`), appending more samples to a dataset (`concatenate` on axis 0), adding a feature column (`concatenate` on axis 1).

**Kotlin comparison.** For lists, `a + b` is concatenate: one longer list. `listOf(a, b)` is stack: a list of two lists, one new level of nesting. Both build a new list, as NumPy's versions do.

<figure class="diagram" markdown>
![Two 2 by 3 arrays a and b. Concatenating along axis 0 gives a 4 by 3 array, along axis 1 gives a 2 by 6 array, and stacking gives a 2 by 2 by 3 array, shown as b sitting behind a](../../assets/img/phase-1/day08-stack-concat.svg)
<figcaption>concatenate grows an existing axis; stack adds a new one.</figcaption>
</figure>

**Example.**

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.concatenate([a, b])           # array([1, 2, 3, 4, 5, 6])   shape (6,)
np.stack([a, b])
# array([[1, 2, 3],
#        [4, 5, 6]])             shape (2, 3): the new axis 0 is "which input"
np.stack([a, b], axis=1)
# array([[1, 4],
#        [2, 5],
#        [3, 6]])                shape (3, 2): the new axis is last

p = np.ones((2, 3))
q = np.zeros((2, 3))
np.concatenate([p, q], axis=0).shape   # (4, 3)
np.concatenate([p, q], axis=1).shape   # (2, 6)
np.stack([p, q]).shape                 # (2, 2, 3)
```

Building a batch from separate images is the classic use of `stack`:

```python
imgs = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(4)]
np.stack(imgs).shape          # (4, 224, 224, 3)   a batch of 4: correct
np.concatenate(imgs).shape    # (896, 224, 3)      one very tall image: wrong
```

**How it works underneath.** Both functions **always allocate a new array and copy every input into it**. They can't return a view: the inputs live in separate buffers, and a view can only describe one buffer.

```python
np.shares_memory(a, np.concatenate([a, b]))   # False
```

**How it goes wrong (1): shapes that don't fit.**

```python
try:
    np.stack([np.zeros(3), np.zeros(4)])
except ValueError as e:
    print("ValueError:", e)
# ValueError: all input arrays must have the same shape

try:
    np.concatenate([np.zeros((2, 3)), np.zeros((2, 4))])
except ValueError as e:
    print("ValueError:", e)
# ValueError: all the input array dimensions except for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 3 and the array at index 1 has size 4
```

Read the second message slowly: joining on axis 0 needs axis 1 to match, and input 0 has 3 columns while input 1 has 4. Joining on `axis=1` would work (`(2, 7)`).

**How it goes wrong (2): concatenating inside a loop.** Because every call copies everything, growing an array one piece at a time copies the first row once per iteration. For \( n \) rows that's about \( n^2 / 2 \) row copies instead of \( n \):

```python
rows = [np.ones(100) for _ in range(1000)]

def grow():
    batch = np.empty((0, 100))                         # start with 0 rows
    for r in rows:
        batch = np.concatenate([batch, r[None]])       # copies everything so far, every time
    return batch

def collect():
    return np.stack(rows)                              # one allocation, one copy of each row

np.array_equal(grow(), collect())   # True: same result
%timeit grow()
%timeit collect()
# 28.1 ms per loop (best of 3, 10 loops each)      grow: on my laptop; yours, and the browser's, will differ
# 423 µs per loop (best of 3, 500 loops each)     collect
```

`np.empty((0, 100))` is an array with 0 rows and 100 columns, a valid empty start; `r[None]` makes each row `(1, 100)` so it can be concatenated on axis 0. The loop is over 60 times slower here, and the gap grows with \( n \): ten times the rows is about a hundred times the work. **Fix:** append to a Python list (cheap: Day 2b) and call `np.stack` or `np.concatenate` **once** at the end. Kotlin has the same trap with `list = list + x` in a loop, and the same fix, a `MutableList` that you `add` to.

## `vstack`, `hstack` and `column_stack`

**What they are.** Shortcuts you'll see in other people's code, all built on `concatenate`. They aren't deprecated (only an alias, `np.row_stack`, was removed in NumPy 2), so you need to read them:

- `np.vstack` ("vertical") joins along axis 0, but first turns any 1-D input of length N into a row `(1, N)`.
- `np.hstack` ("horizontal") joins along axis 1, or axis 0 for 1-D inputs.
- `np.column_stack` turns 1-D inputs into columns `(N, 1)`, then joins along axis 1. Day 4b used it to show each sample next to its label.

```python
np.vstack([a, b])
# array([[1, 2, 3],
#        [4, 5, 6]])            same as np.stack([a, b]) for 1-D inputs
np.hstack([a, b])               # array([1, 2, 3, 4, 5, 6])   same as concatenate for 1-D
np.column_stack([a, b])
# array([[1, 4],
#        [2, 5],
#        [3, 6]])               same as np.stack([a, b], axis=1)
```

**The one difference that matters: appending a row to a matrix.** `vstack` accepts a 2-D matrix and a 1-D row together; `concatenate` doesn't, because their numbers of axes differ:

```python
M = np.ones((2, 3))
try:
    np.concatenate([M, a])
except ValueError as e:
    print("ValueError:", e)
# ValueError: all the input arrays must have same number of dimensions, but the array at index 0 has 2 dimension(s) and the array at index 1 has 1 dimension(s)
np.vstack([M, a])
# array([[1., 1., 1.],
#        [1., 1., 1.],
#        [1., 2., 3.]])
np.concatenate([M, a[None]])     # the explicit version: make a a (1, 3) row first
# array([[1., 1., 1.],
#        [1., 1., 1.],
#        [1., 2., 3.]])
```

Note the result is float: `M` is float and `a` is int, so type promotion (Day 4) picks `float64`.

## Splitting: `np.split` and `np.array_split`

**What they are.** The inverse of `concatenate`: cut one array into a list of pieces along an axis. `np.split(x, n)` cuts into `n` **equal** pieces; `np.split(x, [i, j])` cuts *at* positions `i` and `j`. `np.array_split(x, n)` allows unequal pieces.

**Why they exist.** Cutting a dataset into mini-batches, or into train/validation/test parts at known positions.

**Example.**

```python
X = np.arange(20).reshape(10, 2)          # 10 samples, 2 features
parts = np.split(X, [6, 8])               # cut before rows 6 and 8
[p.shape for p in parts]                  # [(6, 2), (2, 2), (2, 2)]

try:
    np.split(X, 3)
except ValueError as e:
    print("ValueError:", e)
# ValueError: array split does not result in an equal division

batches = np.array_split(X, 3)
[bt.shape for bt in batches]              # [(4, 2), (3, 2), (3, 2)]
np.shares_memory(X, batches[0])           # True    the pieces are views
```

`np.split` refuses 10 rows into 3 equal parts; `array_split` gives the first piece the extra row. The pieces are slices of `X`, so they're views: modifying a batch in place modifies `X` (Day 4's slicing bug).

## Repeating: `np.repeat` and `np.tile`

**What they are.** `np.repeat(x, k, axis=...)` repeats each element `k` times in place. `np.tile(x, reps)` repeats the whole array, like laying tiles.

**Why they exist.** Turning a grayscale image into a 3-channel one for a model that expects RGB, duplicating samples to balance classes, or building test patterns. Both **build a new, bigger array**; Day 9 shows how broadcasting gets the same effect for arithmetic without allocating anything.

```python
v = np.array([1, 2, 3])
np.repeat(v, 2)                 # array([1, 1, 2, 2, 3, 3])   each element twice
np.tile(v, 2)                   # array([1, 2, 3, 1, 2, 3])   the whole array twice

gray = np.zeros((224, 224), dtype=np.uint8)
rgb = np.repeat(gray[..., None], 3, axis=-1)   # add a channel axis, then repeat along it
rgb.shape                       # (224, 224, 3)
rgb.nbytes // gray.nbytes       # 3           three times the memory
```

**How it goes wrong: `repeat` without `axis`.** With no `axis`, `repeat` flattens first:

```python
np.repeat(np.array([[1, 2],
                    [3, 4]]), 2)
# array([1, 1, 2, 2, 3, 3, 4, 4])
np.repeat(np.array([[1, 2],
                    [3, 4]]), 2, axis=0)
# array([[1, 2],
#        [1, 2],
#        [3, 4],
#        [3, 4]])
```

If you get a 1-D result from a 2-D input, you forgot `axis=`.

## Summary

Now that each one has been taught:

| Operation | Shape | View or copy |
|---|---|---|
| `reshape`, `ravel`, `reshape(-1)` | same size, new shape | view, unless the order can't be walked with strides |
| `flatten` | 1-D | always a copy |
| `None` / `expand_dims` / `squeeze` | adds or removes size-1 axes | view |
| `transpose`, `.T`, `moveaxis`, `swapaxes`, `[..., ::-1]` | reordered axes | view (not contiguous) |
| fancy indexing `a[[...]]`, `take_along_axis` | the index's shape | copy |
| `stack`, `concatenate`, `vstack`, `hstack`, `repeat`, `tile` | joined or bigger | always a copy |
| `split`, `array_split` | list of pieces | views |

---

## Going further

These exist and are worth knowing about; you won't need them this month.

- **`order="F"`** (Fortran, column-major) in `reshape`, `ravel` and `np.zeros`: fill by columns instead of rows. [numpy.reshape](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html).
- **`np.lib.stride_tricks.sliding_window_view`** and `as_strided`: views made by choosing strides yourself, such as every window of 3 samples in a time series. [sliding_window_view](https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html).
- **`np.einsum`**, and the `einops` package's `rearrange`: name each axis with a letter and describe reshapes, transposes and sums in one string. [numpy.einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) and [einops](https://einops.rocks/).
- **`np.take` and `np.put`**: fancy indexing as functions, with an `axis` argument and options for out-of-range positions. [numpy.take](https://numpy.org/doc/stable/reference/generated/numpy.take.html).
- **`np.add.at`** and the other unbuffered ufunc `.at` methods, for accumulating at repeated indices. [numpy.ufunc.at](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html).
- **`np.atleast_1d`, `np.atleast_2d`**: add axes only if the input has too few, handy for functions that accept a single sample or a batch. [numpy.atleast_2d](https://numpy.org/doc/stable/reference/generated/numpy.atleast_2d.html).
- **Combining slices and index arrays in one index**, and the rule that moves the indexed axes to the front when index arrays are separated by a slice. [Indexing on ndarrays: combining advanced and basic indexing](https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing).
- **`np.pad`, `np.roll`, `np.flip`'s other axes**, `np.rot90`: padding images for convolutions, shifting sequences, rotating. [Array manipulation routines](https://numpy.org/doc/stable/reference/routines.array-manipulation.html).
- **PyTorch's `channels_last` memory format**: NCHW shape with NHWC strides, the same "shape vs memory order" idea as `C_CONTIGUOUS`. [Channels last memory format tutorial](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html).
- **`ndarray.view(dtype)`**: read the same bytes as a different type, such as four `uint8` values as one `uint32` ARGB pixel. [numpy.ndarray.view](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.view.html).

---

## Exercises

Each solution re-creates its own inputs, so you can run it on its own.

**1. Warm-up: where is the maximum?** Make `scores = np.random.default_rng(7).random((4, 5))`. Find the largest score's flat position with `scores.argmax()`, convert it to `(row, col)` with `np.unravel_index`, and check it against the formula by hand: row = position // 5, col = position % 5. Confirm that `scores[row, col]` equals `scores.max()`.

**2. Warm-up: a model output with extra axes.** A classifier returns `out` with shape `(1, 1, 10)`; make it with `np.random.default_rng(1).random((1, 1, 10)).astype(np.float32)`. Get a `(10,)` vector of scores and the predicted class, in two different ways.

**3. Core: preprocessing for LiteRT.** You have a grayscale digit `digit` of shape `(28, 28)`, `uint8` (make one with `np.random.default_rng(2).integers(0, 256, size=(28, 28), dtype=np.uint8)`). A LiteRT model wants `float32` input of shape `(1, 28, 28, 1)` with values from 0 to 1. Produce it in one line, and check the shape, dtype and maximum.

**4. Core: NCHW to NHWC, proved.** A PyTorch pipeline gives you `nchw` of shape `(8, 3, 32, 32)` (make it with `np.random.default_rng(3).random((8, 3, 32, 32)).astype(np.float32)`). Convert it to NHWC. Prove it's right by checking that `nchw[5, :, 10, 20]` equals the converted array's `[5, 10, 20, :]`. Then show that `reshape(8, 32, 32, 3)` fails the same check.

**5. Core: one-hot labels.** A **one-hot** encoding turns label `k` out of `C` classes into a vector of `C` zeros with a 1 at position `k`; classifiers are trained against these. For `labels = np.array([2, 0, 1, 2])` and 3 classes: (a) build the `(4, 3)` one-hot matrix by fancy-indexing the rows of `np.eye(3)`; (b) build it again by starting from `np.zeros((4, 3))` and assigning 1s with paired index arrays; (c) get the labels back with `argmax`. Is the result of (a) a view of `np.eye(3)`?

**6. Predict the shape.** Write each answer down *before* running anything, then check with `.shape`. Start with `x = np.zeros((2, 3, 4))`.

1. `x.reshape(6, -1)`
2. `x.reshape(-1)`
3. `x[None]`
4. `x[:, None]`
5. `np.transpose(x, (2, 0, 1))`
6. `np.stack([x, x])`
7. `np.concatenate([x, x], axis=1)`
8. `x[:, [0, 2]]`
9. `x[..., 0]` (`...` from Day 4)
10. `np.stack([x, x], axis=-1)`

**7. Find and fix the bug.** This function should turn a list of `(H, W, 3)` RGB photos into a `float32` NCHW batch for a PyTorch model. It runs without an error, and the shape is right, but the model's predictions are nonsense. It has **two** bugs, and the second one hides the first one's wrong shape. Find them, explain what each does to the data, and fix the function.

```python
def make_batch(images):
    h, w, _ = images[0].shape
    batch = np.concatenate(images)                  # join the photos
    batch = batch.reshape(len(images), 3, h, w)     # to NCHW
    return batch.astype(np.float32) / 255

photos = [np.random.default_rng(i).integers(0, 256, size=(4, 5, 3), dtype=np.uint8)
          for i in range(2)]
bad = make_batch(photos)
bad.shape                                              # (2, 3, 4, 5)
np.allclose(bad[1, :, 2, 3], photos[1][2, 3] / 255)   # False: pixel (2, 3) of photo 1 is wrong
```

**8. Challenge: image patches (a Vision Transformer preview).** Split `img4 = np.arange(16).reshape(4, 4)` into four 2×2 patches, shape `(4, 2, 2)`, in the order top-left, top-right, bottom-left, bottom-right, using only `reshape` and `transpose`. Then explain why `img4.reshape(4, 2, 2)` alone gives the wrong patches.

??? success "Solutions"
    **1.**

    ```python
    import numpy as np

    scores = np.random.default_rng(7).random((4, 5))
    pos = scores.argmax()
    pos                                     # np.int64(16)
    row, col = np.unravel_index(pos, scores.shape)
    row, col                                # (np.int64(3), np.int64(1))
    pos // 5, pos % 5                       # (np.int64(3), np.int64(1))   the formula, by hand
    scores[row, col] == scores.max()        # np.True_
    ```

    Why: `argmax()` without `axis` works on the array read in row-major order, so it returns a position \( i \cdot C + j \) with \( C = 5 \): here \( 16 = 3 \cdot 5 + 1 \). Dividing by the row length gives the row, and the remainder gives the column, which is exactly what `unravel_index` computes.

    **2.**

    ```python
    import numpy as np

    out = np.random.default_rng(1).random((1, 1, 10)).astype(np.float32)
    s1 = np.squeeze(out, axis=(0, 1))       # way 1: name the axes to remove
    s2 = out.reshape(-1)                    # way 2: flatten
    s1.shape, s2.shape                      # ((10,), (10,))
    s1.argmax(), s2.argmax()                # (np.int64(1), np.int64(1))
    out[0, 0].shape                         # (10,)    a third way: two integer indices
    ```

    Why: all three only rewrite the header, because the two leading axes have size 1. Bare `out.squeeze()` would give the same answer here, but `axis=(0, 1)` is safer: for a model with a single output, shape `(1, 1, 1)`, bare `squeeze()` would leave a 0-d array, while `squeeze(axis=(0, 1))` still gives shape `(1,)`.

    **3.**

    ```python
    import numpy as np

    digit = np.random.default_rng(2).integers(0, 256, size=(28, 28), dtype=np.uint8)
    inp = digit[None, :, :, None].astype(np.float32) / 255.0
    inp.shape, inp.dtype                    # ((1, 28, 28, 1), dtype('float32'))
    float(inp.max()) <= 1.0                 # True
    np.array_equal(inp, digit.reshape(1, 28, 28, 1).astype(np.float32) / 255.0)
    # True
    ```

    Why: `None` at the front adds the batch axis and `None` at the end adds the channel axis. Dividing the `float32` array by the Python float `255.0` keeps `float32`, because a plain Python number doesn't raise the precision of an array (Day 4's promotion rule); dividing the `uint8` array directly would have given `float64`. `reshape(1, 28, 28, 1)` gives the same result, because adding size-1 axes never changes the reading order.

    **4.**

    ```python
    import numpy as np

    nchw = np.random.default_rng(3).random((8, 3, 32, 32)).astype(np.float32)
    nhwc = nchw.transpose(0, 2, 3, 1)
    nhwc.shape                                               # (8, 32, 32, 3)
    np.array_equal(nchw[5, :, 10, 20], nhwc[5, 10, 20, :])   # True
    wrong = nchw.reshape(8, 32, 32, 3)
    np.array_equal(nchw[5, :, 10, 20], wrong[5, 10, 20, :])  # False
    ```

    Why: `transpose(0, 2, 3, 1)` keeps N first and moves the channel axis (old axis 1) to the end, so every pixel keeps its own three values. `reshape` keeps the memory order instead: it cuts the red plane (1,024 reds in a row) into chunks of 3 and calls each chunk "a pixel", so the first third of `wrong`'s "pixels" hold three neighbouring red values, the next third greens, and the last third blues.

    **5.**

    ```python
    import numpy as np

    labels = np.array([2, 0, 1, 2])
    eye = np.eye(3)
    onehot = eye[labels]                    # (a) row k of the identity is the one-hot vector for k
    onehot
    # array([[0., 0., 1.],
    #        [1., 0., 0.],
    #        [0., 1., 0.],
    #        [0., 0., 1.]])

    oh2 = np.zeros((4, 3))                  # (b) paired indices: (0,2), (1,0), (2,1), (3,2)
    oh2[np.arange(4), labels] = 1
    np.array_equal(onehot, oh2)             # True

    onehot.argmax(axis=1)                   # (c) array([2, 0, 1, 2])
    np.shares_memory(onehot, eye)           # False
    ```

    Why: `eye[labels]` is fancy indexing with one index array on axis 0, so it gathers whole rows, and row `k` of the identity matrix has its single 1 at column `k`; the repeated label 2 picks row 2 twice. Like every fancy index read, it's a copy, so changing `onehot` can't damage `eye`. In (b), the index arrays pair each row number with that row's label, and assignment writes into `oh2` itself. `argmax(axis=1)` finds the position of the 1 in each row, undoing the encoding.

    **6.**

    ```python
    import numpy as np

    x = np.zeros((2, 3, 4))
    x.reshape(6, -1).shape               # (6, 4)
    x.reshape(-1).shape                  # (24,)
    x[None].shape                        # (1, 2, 3, 4)
    x[:, None].shape                     # (2, 1, 3, 4)
    np.transpose(x, (2, 0, 1)).shape     # (4, 2, 3)
    np.stack([x, x]).shape               # (2, 2, 3, 4)
    np.concatenate([x, x], axis=1).shape # (2, 6, 4)
    x[:, [0, 2]].shape                   # (2, 2, 4)
    x[..., 0].shape                      # (2, 3)
    np.stack([x, x], axis=-1).shape      # (2, 3, 4, 2)
    ```

    Why, one by one: (1) 24 elements ÷ 6 = 4. (2) all 24 in one axis. (3) `None` first adds a new axis 0. (4) `x[:, None]` keeps axis 0 (the `:`) and *then* inserts the new axis, so it lands at position 1. (5) new order: old axis 2 (size 4), old 0 (2), old 1 (3). (6) `stack` adds a new axis 0 of length 2, the number of inputs. (7) `concatenate` on axis 1 adds the lengths, 3 + 3 = 6, and leaves the other axes alone. (8) an index array of length 2 replaces axis 1. (9) an integer on the last axis removes it. (10) the new axis goes at the end.

    **7.**

    ```python
    import numpy as np

    def make_batch_fixed(images):
        batch = np.stack(images)                  # (N, H, W, 3): one photo per index of axis 0
        batch = batch.transpose(0, 3, 1, 2)       # (N, 3, H, W): move channels, keep each pixel's values
        return np.ascontiguousarray(batch).astype(np.float32) / 255

    photos = [np.random.default_rng(i).integers(0, 256, size=(4, 5, 3), dtype=np.uint8)
              for i in range(2)]
    good = make_batch_fixed(photos)
    good.shape                                               # (2, 3, 4, 5)
    np.allclose(good[1, :, 2, 3], photos[1][2, 3] / 255)    # True
    good.flags['C_CONTIGUOUS']                               # True
    ```

    Why the original was wrong: bug 1, `np.concatenate(images)` joins on the existing axis 0, giving one tall `(8, 5, 3)` image instead of a batch; it should be `np.stack`, which adds the batch axis. Bug 2, `reshape(N, 3, h, w)` keeps the memory order R, G, B, R, G, B… and cuts it into planes, mixing the channels; changing the meaning of axes needs `transpose`. The two hide each other because both keep the total size (2 × 4 × 5 × 3 = 120), so the final shape looks right. (`astype` already returns a contiguous copy; `np.ascontiguousarray` makes the intent explicit and costs nothing extra when the array is already contiguous.)

    **8.**

    ```python
    import numpy as np

    img4 = np.arange(16).reshape(4, 4)
    patches = img4.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 2, 2)
    patches
    # array([[[ 0,  1],
    #         [ 4,  5]],
    #
    #        [[ 2,  3],
    #         [ 6,  7]],
    #
    #        [[ 8,  9],
    #         [12, 13]],
    #
    #        [[10, 11],
    #         [14, 15]]])
    img4.reshape(4, 2, 2)[0]
    # array([[0, 1],
    #        [2, 3]])                 the first row cut in half: not a 2×2 patch
    ```

    Why it works: `reshape(2, 2, 2, 2)` reads the axes as (patch row, row inside patch, patch column, column inside patch), because a row of 4 splits into 2 patch columns × 2 columns. `transpose(0, 2, 1, 3)` brings the two "which patch" axes together in front, and the final `reshape` merges them into one patch index (that last reshape copies, since the transposed order can't be walked with strides). A plain `reshape(4, 2, 2)` keeps reading order, so its first "patch" is the first row of the image cut into two halves. Vision Transformers cut images into patches exactly like the working version.

## Resources

- [NumPy: Indexing on ndarrays](https://numpy.org/doc/stable/user/basics.indexing.html): the official guide to slicing, `newaxis`, and integer-array (fancy) indexing, including the rules for combining them.
- [NumPy: Copies and views](https://numpy.org/doc/stable/user/basics.copies.html): when `reshape` and `ravel` return a view and when they copy.
- [NumPy reference: Array manipulation routines](https://numpy.org/doc/stable/reference/routines.array-manipulation.html): every reshape, transpose, join, split and repeat function on one page.
- [NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html): the sections on reshaping and adding axes have good pictures.
- [Jay Alammar: A Visual Intro to NumPy and Data Representation](https://jalammar.github.io/visual-numpy/): shapes of images and text, drawn.
- [Dive into Deep Learning: Data Manipulation](https://d2l.ai/chapter_preliminaries/ndarray.html): the same operations on tensors, in PyTorch and other frameworks.
- [PyTorch: `torch.permute`](https://pytorch.org/docs/stable/generated/torch.permute.html): PyTorch's name for transposing axes, the call you'll use to go NCHW ↔ NHWC.

## Checklist

- [ ] I can explain why `reshape` is usually free, compute a flat position with \( i \cdot C + j \), and turn it back into `(row, col)` with `np.unravel_index`.
- [ ] I use `-1` confidently, I know when `reshape` and `ravel` copy, and I can use `copy=False` to catch it.
- [ ] I know `flatten` always copies while `ravel` is a view when it can be.
- [ ] I can turn `(224, 224, 3)` into `(1, 224, 224, 3)` three ways, I pass `axis=` to `squeeze`, and I know what a 0-d array is.
- [ ] I know when to `transpose` instead of `reshape`, I can convert NHWC ↔ NCHW and RGB ↔ BGR, and I know when to call `np.ascontiguousarray`.
- [ ] I can predict the shape of a fancy-indexing result, I know `m[[0, 1], [2, 3]]` picks pairs while `np.ix_` picks a block, and I can gather per row with `np.take_along_axis`.
- [ ] I can choose between `stack` and `concatenate` by asking "do I want a new axis?", split with `array_split`, and I build batches with one `np.stack` call instead of a loop.
- [ ] I got at least 8 of the 10 shape predictions in Exercise 6 right before checking.

**Next:** [Day 9: Broadcasting](day-09-broadcasting.md)
