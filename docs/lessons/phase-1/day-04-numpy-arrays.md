# Day 4: NumPy Arrays, and Why ML Runs on Them

**Phase 1 · Week 1 · ~180 minutes**

This is the first of three NumPy days. Today is the array itself: how it's stored, how to make one, its dtypes, indexing, masks, views, element-wise maths and reductions. [Day 4b](day-04b-numpy-toolkit.md) is the toolkit you use on top (`where`, sorting, NaN, comparing floats), and [Day 4c](day-04c-random-io-stats.md) adds random numbers, saving arrays and statistics. A good break point today is after Part 3, "dtypes".

## Objectives

- [ ] Explain, with real byte counts, why a NumPy array of numbers is smaller and much faster than a Python list of the same numbers.
- [ ] Describe what an `ndarray` is underneath (one data buffer plus `dtype`, `shape` and `strides`) and read `shape`, `ndim`, `size`, `dtype`, `itemsize` and `nbytes`.
- [ ] Create arrays with `np.array`, `zeros`, `ones`, `full`, `empty`, `arange`, `linspace`, `eye` and the `*_like` functions, and avoid the `arange` float-step trap.
- [ ] Choose a dtype, convert with `astype`, and predict integer overflow (`uint8` 250 + 10 → 4), float32 rounding and type promotion (`uint8 / 255.0` → `float64`).
- [ ] Index and slice 1-D, 2-D and 3-D arrays, and predict the shape an integer index or a slice gives you.
- [ ] Pick and reorder elements and rows with integer-array ("fancy") indexing, including `M[rows, cols]` pairs, and know it returns a copy.
- [ ] Select and assign with boolean masks combined with `&`, `|` and `~`, avoiding the parentheses and `and` traps.
- [ ] Tell a view from a copy (slice → view; mask → copy), check with `np.shares_memory`, and use `.copy()` where it matters.
- [ ] Replace loops with element-wise arithmetic and ufuncs (`sqrt`, `exp`, `log`, `abs`, `maximum`), and know `np.maximum` from `np.max`.
- [ ] Reduce with `sum`, `mean`, `std`, `min`, `max`, `argmin`, `argmax`, `any` and `all`, along any `axis` of a 3-D array, with and without `keepdims`.
- [ ] Time a Python loop against the vectorised version with `%timeit`, and explain the gap.

**Prerequisites.**

- [Day 2](day-02-python-types.md): names and objects, `int` never overflowing, `float` as an IEEE-754 double, why `0.1 + 0.2 != 0.3`, `**`, `/` versus `//`, `inf` and `nan`, and `type()`.
- [Day 2a](day-02a-strings.md): f-strings, including format specs like `:,` and `:.2f`.
- [Day 2b](day-02b-collections.md): how a list stores **references** (addresses) to separate objects, `sys.getsizeof`, slicing with start, stop and step, nested lists as a "matrix", shallow copies, and `timeit`.
- [Day 3](day-03-functions-classes.md): functions, and passing an object to a function (the function gets the same object, not a copy).
- [Day 3a](day-03a-classes-modules.md): dunder methods: `a + b` calls `a.__add__(b)`, `a[i]` calls `a.__getitem__(i)`. NumPy uses exactly these hooks.
- [Day 3b](day-03b-errors-files-json.md): reading an error message, and `try` / `except`.
- [Day 3c](day-03c-files-json.md): the note that a `float32` 0.91 comes back as `0.9100000262260437`; today explains why.
- [Day 3d](day-03d-iterators-generators.md): `for` loops pull items one at a time through `iter` and `next`.

!!! info "How to read the code on this page"
    Every Python block has a **▶ Run** button, and blocks share their variables from top to bottom like notebook cells, so run them in order. A bare expression on its own line shows its `repr()`, as the `# comment` beside or below it says. The comments come from 64-bit Python on a laptop; the few integer-dtype differences you'll see in the browser are explained where they first occur. Lines that fail on purpose are wrapped in `try` / `except` so the rest of the block still runs, and the comment shows the real error message.

---

## The big idea

Everything in machine learning is numbers in grids. A photo is a grid of pixel values, height × width × 3 colours. A sentence becomes a grid of token embeddings, tokens × 768. A neural-network layer is a grid of weights. Training is arithmetic on those grids, billions of multiply-adds per second.

Plain Python can't do that fast. A Python list of numbers is a list of *addresses* of separate number objects (Day 2b), and every `+` on an element goes through the interpreter: look up the object, check its type, find the right `__add__`, allocate a new object for the result. That's fine for ten numbers and hopeless for ten million.

**NumPy** (Numerical Python) fixes this with one type, the **`ndarray`** (n-dimensional array): a single block of raw machine numbers, all of the same type, plus a small header saying how to read it. Operations on an `ndarray` run as compiled C loops over that block, so `arr * 2` on a million numbers is one call from Python, not a million.

**Analogy.** A Python list is a shelf of labelled cards, each card giving the warehouse location of a box that holds one number. To add up the numbers you walk to every box. A NumPy array is an egg carton: every slot the same size, side by side, and a machine can sweep along the row in one pass.

Why this matters beyond NumPy: PyTorch tensors, TensorFlow and LiteRT tensors, and JAX arrays all copy NumPy's design and most of its function names. `shape`, `dtype`, slicing, masks, `axis=`, `sum`, `mean`, `argmax` work almost identically in all of them. Learn NumPy properly and you've learned the core API of every ML framework.

---

**Part 1: What an array is**

## Why a list of numbers is slow

**What it is.** A Python list of floats is three layers in memory: the list object; an array of 8-byte **references** (memory addresses); and one separate `float` object per number, each a full Python object with a reference count, a pointer to its type and the 8-byte value itself. A NumPy array of floats is a small header plus one block of raw 8-byte values, side by side.

**Why it matters.** Memory size, and speed. Every extra layer is bytes you store and pointers the CPU must follow.

**Kotlin comparison.** You already know this split. Kotlin's `List<Float>` holds **boxed** `java.lang.Float` objects, each on the heap; `FloatArray` compiles to a primitive `float[]`, raw 4-byte values side by side.

```kotlin
val boxed: List<Float> = listOf(1.5f, 2.0f, 3.25f, 4.0f)   // 4 references to 4 Float objects
val raw: FloatArray = floatArrayOf(1.5f, 2.0f, 3.25f, 4.0f) // 16 bytes of floats in a row
```

A Python list is always the `List<Float>` kind. NumPy gives Python the `FloatArray` kind. The difference: Kotlin can sometimes avoid boxing in a compiled loop; Python never can, because every Python number is an object.

<figure class="diagram" markdown>
![Animation in two halves. Top: a Python list object points to an array of four address slots, and each address has an arrow to a float object placed somewhere else in memory; a red dot hops from a slot to its object and back to the next slot, four times. Bottom: a NumPy array is a small header (dtype float64, shape (4,)) and one block of four 8-byte values at bytes 0, 8, 16 and 24; an orange highlight sweeps across the block in one pass and the label changes to "one sweep: done"](../../assets/img/phase-1/day04-list-vs-array.svg)
<figcaption>Animated: reading a list chases one address per number. Reading an array streams through one block.</figcaption>
</figure>

**Example: measure it.** `sys.getsizeof(obj)` (Day 2b) gives one object's own size in bytes, *not* counting the objects it refers to. For a list, add up the items too. For an array, `.nbytes` is the size of the data block.

```python
import sys
import numpy as np                      # the universal convention: always "np"

xs = [float(i) for i in range(1000)]    # 1,000 floats in a list
list_bytes = sys.getsizeof(xs) + sum(sys.getsizeof(x) for x in xs)
list_bytes                              # 32856

arr = np.array(xs)                      # the same 1,000 floats in an array
arr.nbytes                              # 8000   1,000 × 8 bytes, nothing else
round(list_bytes / arr.nbytes, 1)       # 4.1
```

**How it works underneath.** The list object itself is 8 856 bytes: a 56-byte header, 8 000 bytes for 1 000 addresses, and 800 bytes of spare slots (a comprehension grows the list with appends, and appends over-allocate, Day 2b). Then come 24 000 bytes for 1 000 float objects at 24 bytes each: 8 for the reference count, 8 for the type pointer, 8 for the value. The array stores only the 8 000 value bytes; its header is a fixed hundred-odd bytes however long the array is. Less than a third of the list's memory is the numbers themselves, which is why the array is about a quarter of the size.

(`sys.getsizeof` numbers depend on the build. The browser's 32-bit Python uses 4-byte addresses, so ▶ Run prints smaller list numbers; the array's 8 000 doesn't change, because a `float64` is 8 bytes everywhere.)

!!! warning "Common mistake: looping over an array in Python"
    Putting numbers in an array doesn't make a Python `for` loop over them fast. It makes it *slower*, because each `arr[i]` has to box the raw value into a brand-new NumPy scalar object on the way out:

    ```python
    import timeit

    big_list = [float(i) for i in range(100_000)]
    big_arr = np.array(big_list)
    t_list = timeit.timeit("sum(big_list)", globals=globals(), number=20) / 20
    t_py_on_arr = timeit.timeit("sum(big_arr)", globals=globals(), number=20) / 20
    t_numpy = timeit.timeit("big_arr.sum()", globals=globals(), number=20) / 20
    print(f"built-in sum over a list:   {t_list * 1e3:7.3f} ms")
    print(f"built-in sum over an array: {t_py_on_arr * 1e3:7.3f} ms")
    print(f"array.sum():                {t_numpy * 1e3:7.3f} ms")
    # built-in sum over a list:     0.781 ms
    # built-in sum over an array:   6.442 ms
    # array.sum():                  0.031 ms
    ```

    These timings are from the machine this lesson was written on; yours will differ, and the browser is slower again. The *order* won't change: Python's built-in `sum` walks the array one boxed element at a time and loses even to the list, by about 8×, while the array's own `.sum()` is about 25× faster than the list. **Fix:** once data is in an array, use NumPy's operations (`.sum()`, `np.sqrt`, `*`) on the whole array, never a Python loop over its elements.

## The three reasons NumPy is fast

**What it is.** A NumPy operation like `arr * 2` is one Python call that hands the whole data block to a loop written in C. Three things make that loop fast:

1. **Contiguous memory.** **Contiguous** means "in one unbroken block". The CPU doesn't read memory one number at a time; it pulls in 64-byte chunks called **cache lines**. With 8-byte floats side by side, one fetch brings in eight numbers, and the CPU's prefetcher sees the straight-line pattern and fetches ahead. A list's float objects are wherever the heap put them, so every number can be a separate fetch.
2. **One type for every element.** The array's **dtype** (data type, explained fully in Part 3) says "every element is a float64". The C loop is compiled for float64 and never checks types. Python, adding two objects, must look up both types and find the right `__add__` every single time.
3. **SIMD.** *Single Instruction, Multiple Data*: modern CPUs have instructions that multiply 2, 4, 8 or more numbers at once (NEON on the ARM chip in your phone, SSE and AVX on laptop chips). NumPy's inner loops are written to use them where the hardware has them.

**Why it matters.** This is the reason the whole Python ML ecosystem works: Python is the steering wheel, and compiled loops over contiguous buffers are the engine. PyTorch and TensorFlow do the same thing, with GPU kernels as the engine.

**Example.** In Jupyter (and in the ▶ Run button on this site), `%timeit statement` runs a statement many times and reports the time per run: IPython prints the mean ± standard deviation over 7 runs, the ▶ Run button the best of 3. It's a notebook command, not Python syntax (Day 1), so it doesn't work in a `.py` script; there you'd use `timeit.timeit`, as in the box above.

```python
xs = list(range(1_000_000))
arr = np.arange(1_000_000)             # the integers 0 … 999,999 as an array (more in Part 2)

%timeit [x * 2 for x in xs]
%timeit arr * 2
# 33.6 ms ± 2.33 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 581 μs ± 15.5 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
```

(μs is microseconds, a thousandth of a millisecond; "std. dev." is the spread between runs, a statistic defined in Part 8.) Those two lines are from IPython (the engine inside Jupyter) on the machine this lesson was written on: the array version is about 58× faster. In the browser the ▶ Run button prints its own format, ending in `[browser timing]`, and both numbers are slower. Your numbers will differ; the gap of one to two orders of magnitude won't.

**How it works underneath.** The list comprehension runs a million rounds of Python bytecode: fetch the next address, follow it to an `int` object, check its type, call `int.__mul__`, allocate a new `int` for the result, store its address in the new list. `arr * 2` calls `arr.__mul__(2)` once (Day 3a's operator hooks), which runs a C loop: read 8 bytes, multiply, write 8 bytes, a million times, with SIMD doing several per instruction and no objects created.

!!! android "Android bridge"
    This is the difference between a `List<Float>` and a `FloatArray`, and you already pick the second when performance matters. Feeding a LiteRT (formerly TensorFlow Lite) `Interpreter`, you use a `FloatArray` or a direct `ByteBuffer` in native byte order, not a `List<Float>`, because the runtime wants one raw block of numbers it can hand straight to native code. A NumPy array *is* that raw block, plus a shape and a dtype. That's why handing NumPy arrays to PyTorch, TensorFlow or LiteRT's Python API is usually cheap.

## What an `ndarray` is: a buffer plus a header

**What it is.** An `ndarray` object has two parts:

- a **data buffer**: one contiguous block of raw bytes holding the values;
- a **header** that says how to read those bytes:
    - **`dtype`**: what each element is, for example `float64` (8-byte float) or `uint8` (1-byte unsigned integer);
    - **`shape`**: a tuple with the length of each **axis** (an *axis* is one direction you can index along: a 1-D array has one axis, a table has two, rows and columns);
    - **`strides`**: for each axis, how many bytes to jump to get to the next element along it.

**Why it exists.** Separating the bytes from the description of the bytes is what makes NumPy cheap: changing the shape, taking a slice or transposing usually writes a new header over the *same* buffer, with no copying (Part 6 and Day 8).

**Example.** The attributes you'll check constantly:

```python
m = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])   # a nested list → a 2-D array

m.shape       # (2, 3)            2 rows, 3 columns: one number per axis
m.ndim        # 2                 the number of axes, len(m.shape)
m.size        # 6                 the number of elements, 2 × 3
m.dtype       # dtype('float64')  every element is an 8-byte float
m.itemsize    # 8                 bytes per element
m.nbytes      # 48                size × itemsize: the data buffer
m.strides     # (24, 8)           next column: +8 bytes; next row: +3 × 8 = 24 bytes
len(m)        # 2                 len() is the length of the first axis only
```

**How it works underneath.** The buffer holds `1.0, 2.0, 3.0, 4.0, 5.0, 6.0` as 48 bytes in a row, row after row (called **row-major** or **C order**). To find `m[1, 2]`, NumPy computes `1 × 24 + 2 × 8 = 40` and reads the 8 bytes starting at byte 40: `6.0`. That arithmetic is the whole trick of indexing, and Day 8 spends its first half on it. Today you need the picture: *one block of bytes, and a header that says how to walk it.*

Shapes you'll meet constantly in ML:

```python
np.zeros((1000, 8)).shape             # (1000, 8)            a table: 1,000 samples × 8 features
np.zeros((224, 224, 3)).shape         # (224, 224, 3)        one RGB photo: height × width × channels
np.zeros((32, 224, 224, 3)).shape     # (32, 224, 224, 3)    a batch of 32 photos
np.zeros((12, 768)).shape             # (12, 768)            12 tokens, each a 768-number embedding
```

(`np.zeros(shape)` makes an array of that shape filled with 0.0; it's the next section's first function.) The conventions are worth memorising: the **first axis is usually the batch or sample axis**, and for images in TensorFlow and LiteRT the channels come last.

!!! warning "Common mistake: `len()` on a multi-dimensional array"
    `len(m)` is the length of the *first* axis, like `len()` of a list of lists. On a `(32, 224, 224, 3)` batch it's `32`, not the number of pixels:

    ```python
    batch = np.zeros((32, 224, 224, 3))
    len(batch)       # 32
    batch.size       # 4816896   32 × 224 × 224 × 3
    ```

    **Fix:** use `.shape` to see every axis and `.size` for the element count. Print `.shape` by habit: most bugs in your first months of ML will be shape bugs.

---

**Part 2: Creating arrays**

## `np.array`: from Python lists

**What it is.** `np.array(data)` builds a new array from a list, a nested list or a tuple, copying the values into a fresh buffer. It works out the shape from the nesting and picks a dtype that can hold every value, unless you pass `dtype=`.

**Why it exists.** It's the bridge from plain Python data (parsed JSON, CSV rows, a list you built in a loop) into NumPy.

**Kotlin comparison.** Kotlin has a separate constructor per element type and builds 2-D arrays as arrays of arrays:

```kotlin
val v = intArrayOf(1, 2, 3)
val grid = Array(2) { r -> IntArray(3) { c -> r * 3 + c } }   // an array of 2 separate IntArrays
```

Python has one function for every type and every number of axes, and a 2-D NumPy array is *one* block, not an array of row objects:

```python
v = np.array([1, 2, 3])
v                     # array([1, 2, 3])
v.dtype               # dtype('int64')

grid = np.array([[0, 1, 2],
                 [3, 4, 5]])
grid.shape            # (2, 3)

np.array([1, 2.5])                       # array([1. , 2.5])          one float: everything becomes float64
np.array([True, False])                  # array([ True, False])      dtype bool
np.array([1, 2, 3], dtype=np.float32)    # array([1., 2., 3.], dtype=float32)
```

Note how the repr shows the dtype only when it can't be guessed from the values: `float32` is printed; `float64`, the default integer type and `bool` aren't. `1.` is NumPy's short way of printing the float `1.0`.

**How it works underneath.** `np.array` walks the nested lists once to find the shape (the outer list has 2 items, each inner list has 3, so `(2, 3)`), then picks the "smallest" dtype that holds every value: all `bool` → `bool`; all ints → the default integer; any float → `float64`. Then it allocates the buffer and converts each Python object into raw bytes. That's a full copy: changing `grid` later never changes the original list, and the other way round.

The default integer type is `int64` on a normal 64-bit computer. In the browser's 32-bit Python (the ▶ Run button) NumPy's default integer is `int32`, so there `v.dtype` shows `dtype('int32')`. Whenever the byte size of an integer array matters, pass `dtype=` explicitly.

It works the other way round too. An array that is `int64` *on purpose* (made with `astype(np.int64)` or `dtype=np.int64`, returned by `rng.integers`, or holding a number too big for `int32`) isn't the default in the browser, so there its repr gains `, dtype=int64)` at the end, where this page shows none. The values are always the same; only the type label differs. `print(...)` never shows a dtype, so printed output looks identical everywhere.

!!! warning "Common mistake: ragged lists and mixed types"
    Every row must have the same length. A **ragged** list (rows of different lengths) can't be one rectangular block:

    ```python
    try:
        np.array([[1, 2, 3], [4, 5]])
    except ValueError as e:
        print("ValueError:", e)
    # ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (2,) + inhomogeneous part.
    ```

    Read it as "after the first axis (2 rows), the lengths didn't agree". **Fix:** pad the rows to one length, or keep ragged data in a list of arrays.

    Mixing numbers and strings doesn't fail; it turns *everything* into strings, so later maths fails far from the cause:

    ```python
    mixed = np.array([1, 2, "3"])
    mixed                 # array(['1', '2', '3'], dtype='<U21')
    ```

    `<U21` means "Unicode string of up to 21 characters" (`<` is the byte order, little-endian: least significant byte first, which is what every phone and laptop uses). **Fix:** convert while parsing (`int(s)`, Day 2), or convert the whole array with `astype` (Part 3).

## Arrays filled with a value: `zeros`, `ones`, `full`, `empty`, `eye`

**What it is.** Functions that build an array of a given **shape** without a list: `np.zeros(shape)`, `np.ones(shape)`, `np.full(shape, value)`, `np.empty(shape)` and `np.eye(n)`. A shape is a tuple like `(2, 3)`; for 1-D you can pass a plain int. They all take `dtype=`; the default is `float64`.

**Why it exists.** You allocate first and fill later all the time: a results buffer, a weight matrix to initialise, a mask of `False`, an image canvas.

**Kotlin comparison.** `FloatArray(6)` (all zeros), `FloatArray(6) { 1f }`, `Array(2) { FloatArray(3) }`. NumPy does the same with a shape tuple for any number of axes.

**Example.**

```python
np.zeros(3)                         # array([0., 0., 0.])
np.zeros((2, 3))                    # array([[0., 0., 0.],
                                    #        [0., 0., 0.]])
np.ones((2, 2), dtype=np.uint8)     # array([[1, 1],
                                    #        [1, 1]], dtype=uint8)
np.full((2, 3), 7)                  # array([[7, 7, 7],
                                    #        [7, 7, 7]])
np.full(3, 0.5)                     # array([0.5, 0.5, 0.5])
np.eye(3)                           # array([[1., 0., 0.],
                                    #        [0., 1., 0.],
                                    #        [0., 0., 1.]])
```

`np.eye(n)` is the **identity matrix**: ones on the diagonal, zeros elsewhere (Day 11 explains why it matters). `np.full` takes its dtype from the value: `7` gives integers, `0.5` gives floats.

**How it works underneath: `empty`.** `zeros` asks the operating system for memory and makes sure every byte is 0. `np.empty` asks for memory and **doesn't touch it**, so the array contains whatever bytes happened to be there: leftovers from earlier arrays, or zeros if the memory was fresh. It's slightly faster, and only safe when you'll overwrite every element before reading any:

```python
scratch = np.empty(3)
scratch.shape                       # (3,)   the shape is real; the values are whatever was in memory
```

Printing `scratch` shows garbage that differs from run to run and machine to machine, for example `array([4.9e-324, 9.9e-324, 1.5e-323])` once on mine; that's why this page doesn't echo it. Use `np.zeros` unless you've measured that it matters.

!!! warning "Common mistake: passing the shape as separate arguments"
    The shape is *one* argument, a tuple. Write `np.zeros((2, 3))`, with two pairs of parentheses. With one pair, `3` lands in the *second* parameter, which is `dtype`:

    ```python
    try:
        np.zeros(2, 3)
    except TypeError as e:
        print("TypeError:", e)
    # TypeError: Cannot interpret '3' as a data type
    ```

    The message mentions data types because `3` was taken as the dtype. (`reshape`, which you'll meet below, accepts both styles; the creation functions don't.)

## Number ranges: `arange` and `linspace`

**What it is.** `np.arange(start, stop, step)` is `range()` returning an array: from `start` up to but **not including** `stop`, in steps of `step`. `np.linspace(start, stop, num)` gives `num` evenly spaced values from `start` to `stop`, **including** `stop` by default.

**Why it exists.** Indices, time steps, x-values for a plot, a grid of learning rates to try.

**Kotlin comparison.** `arange(0, 10, 2)` is `(0 until 10 step 2)`. Kotlin has no `linspace`; you'd write `List(5) { i -> i / 4.0 }`.

**Example.**

```python
np.arange(5)                # array([0, 1, 2, 3, 4])
np.arange(2, 10, 3)         # array([2, 5, 8])
np.arange(0, 1, 0.25)       # array([0.  , 0.25, 0.5 , 0.75])    floats work too
np.linspace(0, 1, 5)        # array([0.  , 0.25, 0.5 , 0.75, 1.  ])   5 points, both ends included
np.linspace(0, 1, 5, endpoint=False)   # array([0. , 0.2, 0.4, 0.6, 0.8])
```

**How it works underneath.** `arange` first computes how many elements to make, `ceil((stop − start) / step)`, then fills element `i` with `start + i × step`. `linspace` fixes the count and computes the step as `(stop − start) / (num − 1)`.

!!! warning "Common mistake: `arange` with a float step"
    Because the count is computed from `(stop − start) / step` in floating point, a float step can produce one element too many. `0.1` isn't exactly representable in binary (Day 2), so `(1.3 − 1) / 0.1` comes out as `3.0000000000000004`, which rounds *up* to 4 elements:

    ```python
    np.arange(1, 1.3, 0.1)      # array([1. , 1.1, 1.2, 1.3])   1.3 was supposed to be excluded
    len(np.arange(1, 1.3, 0.1)) # 4
    ```

    No error, and an extra value that sneaks into your plot or grid. **Fix:** for floats, use `linspace` and say how many points you want: `np.linspace(1, 1.2, 3)` gives exactly `array([1. , 1.1, 1.2])`. Keep `arange` for integers.

## Copying a shape: `zeros_like`, `ones_like`, `full_like`, `empty_like`

**What it is.** `np.zeros_like(a)` makes a new array with the same **shape and dtype** as `a`, filled with zeros. `ones_like`, `full_like(a, value)` and `empty_like` work the same way.

**Why it exists.** "An output buffer shaped like the input" comes up constantly: gradients have the same shape as weights, a mask has the same shape as an image.

```python
img = np.array([[10, 20], [30, 40]], dtype=np.uint8)
np.zeros_like(img)          # array([[0, 0],
                            #        [0, 0]], dtype=uint8)
np.full_like(img, 255)      # array([[255, 255],
                            #        [255, 255]], dtype=uint8)
```

!!! warning "Common mistake: `*_like` keeps an integer dtype"
    The copy keeps the dtype, and assigning a float into an integer array truncates it silently:

    ```python
    counts = np.array([3, 1, 4])
    fractions = np.zeros_like(counts)   # integer zeros, because counts is integer
    fractions[0] = 0.75
    fractions                           # array([0, 0, 0])   0.75 was truncated to 0
    ```

    **Fix:** say the dtype you need: `np.zeros_like(counts, dtype=np.float64)`.

## NumPy scalars, and getting back to Python

**What it is.** Taking one element out of an array gives a **NumPy scalar**, such as `np.float64` or `np.int64`: a single number that still carries its dtype. When echoed, it shows as `np.float64(2.5)` rather than `2.5`. That's this page's convention from here on: a bare expression that gives one number shows `np.float64(...)`, `np.int64(...)`, `np.True_` and so on.

**Why it matters.** A NumPy scalar behaves like a number (`+`, comparisons, f-strings all work) but keeps NumPy's rules, including fixed-size integer overflow (Part 3). Some libraries, `json` among them (Day 3c), don't accept NumPy scalars.

```python
a = np.array([1.5, 2.5, 4.0])
a[1]                     # np.float64(2.5)
type(a[1])               # <class 'numpy.float64'>
a.sum()                  # np.float64(8.0)
print(a.sum())           # 8.0                      print shows the plain value
a[1].item()              # 2.5                      .item() gives a plain Python float
a.tolist()               # [1.5, 2.5, 4.0]          the whole array as a Python list
np.array([1, 2])[0]      # np.int64(1)              in the browser: np.int32(1)
```

`.item()` converts one element to the matching Python type, and `.tolist()` converts the whole array, nested lists and all. Use them at the boundary, when handing results to `json`, a database or an API.

**`print` versus the echo.** A bare expression is echoed with `repr()`, the unambiguous form with commas, `array(...)` and any non-default dtype. `print()` uses `str()` instead, the compact form: spaces between elements, no `array(...)`, no commas and never a dtype. Both appear on this page, and in the exercises you'll predict `print` output, so learn to read both:

```python
print(np.array([1, 2, 3]))                 # [1 2 3]
print(np.array([1.5, 2.0]))                # [1.5 2. ]
print(np.arange(6).reshape(2, 3))          # [[0 1 2]
                                           #  [3 4 5]]
np.array([1.5, 2.0])                       # array([1.5, 2. ])   the echo: repr
```

**How it works underneath.** A NumPy scalar is a small Python object that holds the raw bytes of one element plus its dtype, so `a[1]` has to create a new object each time you index (that's the boxing cost from Part 1). A Python `float` is a different class, and code that checks `isinstance(x, float)` or only knows Python's types can reject a NumPy scalar. (`np.float64` happens to be a subclass of Python's `float`, but `np.int64` is not a subclass of `int`, and `np.float32` isn't a `float`.)

!!! warning "Common mistake: NumPy scalars in JSON"
    The `json` module (Day 3c) only knows Python's own types, so a NumPy integer from `argmax` or `sum` breaks serialisation:

    ```python
    import json
    best = np.array([0.2, 0.7, 0.1]).argmax()     # a NumPy integer, not an int
    try:
        json.dumps({"best_class": best})
    except TypeError as e:
        print("TypeError:", e)
    # TypeError: Object of type int64 is not JSON serializable
    json.dumps({"best_class": best.item()})       # '{"best_class": 1}'
    ```

    (In the browser the message says `int32`.) **Fix:** convert with `.item()` (one value) or `.tolist()` (a whole array) before handing results to `json`, an HTTP client or a database.

## A random generator, for test data

You'll want realistic-looking data to practise on. NumPy's random numbers come from a **generator** object; Day 4c covers it in depth. For now, three calls:

```python
rng = np.random.default_rng(42)      # a generator with seed 42: the same numbers on every run
rng.random(3)                        # array([0.77395605, 0.43887844, 0.85859792])    floats from 0 (included) to 1 (excluded)
print(rng.integers(0, 10, size=5))   # [0 6 2 0 5]          ints from 0 up to 9
rng.normal(size=3)                   # array([ 0.1278404 , -0.31624259, -0.01680116])   a bell curve around 0
```

`rng.integers` also takes `dtype=`, for example `dtype=np.uint8` to make fake pixel values. The **seed** (`42`) fixes the sequence, so these lines print the same numbers every time you run the page from the top. Each call moves the generator along, so running one line twice gives different numbers. ML results depend on random initialisation and shuffling, and seeding is how you make an experiment repeatable.

---

**Part 3: dtypes**

## The dtype families

**What it is.** A **dtype** describes how to turn the bytes of one element into a number: what kind of number, and how many bytes. There are four families you'll use:

- **`bool`**: 1 byte, `True` or `False`. What comparisons and masks give you.
- **Signed integers** `int8`, `int16`, `int32`, `int64`: 1, 2, 4, 8 bytes, holding negatives and positives. The number in the name is bits.
- **Unsigned integers** `uint8`, `uint16`, `uint32`, `uint64`: the same sizes, no sign, so from 0 up to twice as high. `uint8` (0 to 255) is the type of image pixels.
- **Floats** `float16`, `float32`, `float64`: 2, 4, 8 bytes, IEEE-754 like Python's `float` (Day 2), which is `float64`.

**Why it exists.** Python has one `int` (unlimited) and one `float` (64-bit). Hardware has many sizes, and choosing the size is how you trade memory and speed against range and precision. ML cares a lot: a model's weights in `float32` take half the memory of `float64`, and in `int8` a quarter of `float32`.

**Kotlin comparison.** These are Kotlin's primitives, with explicit sizes: `int8` is `Byte`, `int16` is `Short`, `int32` is `Int`, `int64` is `Long`, `float32` is `Float`, `float64` is `Double`. Kotlin's `UByte` is `uint8`. One difference you know well: Kotlin's `Byte` is signed (-128 to 127), so on Android you write `b.toInt() and 0xFF` to read a pixel byte as 0 to 255. NumPy has a real `uint8`, so pixels read as 0 to 255 directly.

**Example: ranges and sizes.** `np.iinfo(dtype)` describes an integer dtype and `np.finfo(dtype)` a float dtype:

```python
np.iinfo(np.uint8)              # iinfo(min=0, max=255, dtype=uint8)
np.iinfo(np.int8)               # iinfo(min=-128, max=127, dtype=int8)
np.iinfo(np.int32).max          # 2147483647
np.finfo(np.float32).max        # np.float32(3.4028235e+38)
np.dtype(np.float16).itemsize   # 2
np.zeros(1000, dtype=np.float32).nbytes   # 4000
np.zeros(1000, dtype=np.int8).nbytes      # 1000
```

You can name a dtype as an attribute (`np.float32`) or as a string (`dtype="float32"`); both mean the same.

**How it works underneath.** The dtype is how NumPy reads the bytes. The same 4 bytes mean completely different numbers as a `float32` or as an `int32`. You'll rarely reinterpret bytes on purpose, but it explains why a dtype mismatch at a boundary (a model expecting `float32`, given `uint8`) produces garbage rather than an error in lower-level runtimes.

Now that each family is taught, here they are side by side:

| dtype | Bytes | Range or precision | Typical use |
|---|---|---|---|
| `bool` | 1 | `True` / `False` | masks, comparison results |
| `uint8` | 1 | 0 to 255 | image pixels; quantized models |
| `int8` | 1 | -128 to 127 | quantized model weights |
| `int32` / `int64` | 4 / 8 | ±2.1 billion / ±9.2 × 10¹⁸ | indices, labels, counts |
| `float16` | 2 | about 3 significant digits | GPU training and inference |
| `float32` | 4 | about 7 significant digits | neural networks (PyTorch's default) |
| `float64` | 8 | about 16 significant digits | NumPy's default; statistics |

`bfloat16` ("brain float", 2 bytes with `float32`'s range but fewer digits) is common on TPUs (Google's chips built for neural networks) and newer GPUs. NumPy doesn't have it built in; PyTorch does, and it's listed under Going further.

!!! android "Android bridge"
    This table is the heart of on-device AI. A model with 100 million weights is 400 MB in `float32`, 200 MB in `float16` and 100 MB in `int8`. Storing weights in fewer bytes is called **quantization**: it's how models fit in a phone's memory and run on its NPU (neural processing unit, a chip block built for the multiply-adds of neural networks). A quantized LiteRT model often wants `uint8` or `int8` input tensors, while a float model wants `float32`; check `interpreter.getInputTensor(0).dataType()` and feed exactly that. You'll quantize a model for real in Phase 6. The arithmetic starts with `.nbytes`.

## Converting with `astype`

**What it is.** `arr.astype(dtype)` returns a **new array** (a copy) with every element converted to `dtype`.

**Why it exists.** Every pipeline converts: pixels arrive as `uint8` and the model wants `float32`; labels are parsed as floats and indexing wants ints; boolean masks become 0/1 counts.

**Kotlin comparison.** `intArray.map { it.toFloat() }.toFloatArray()`, or `toInt()` on each element. The truncation rule is the same as Kotlin's `Double.toInt()`: toward zero.

**Example.**

```python
prices = np.array([-1.7, 0.5, 2.9])
prices.astype(np.int64)            # array([-1,  0,  2])    truncates toward zero (browser adds dtype=int64)

flags = np.array([True, False, True])
flags.astype(np.int64)             # array([1, 0, 1])
print(flags.sum())                 # 2              True counts as 1, so sum() counts them

text = np.array(["3.5", "2", "-1e3"])
text.astype(np.float64)            # array([    3.5,     2. , -1000. ])

pix = np.array([0, 128, 255], dtype=np.uint8)
pix.astype(np.float32) / 255       # array([0.       , 0.5019608, 1.       ], dtype=float32)
```

Notice `-1.7` became `-1`, not `-2`: truncation toward zero, like `int(-1.7)` in Python. To round first, use `np.round` (Day 4c), then `astype`.

**How it works underneath.** `astype` allocates a new buffer of `size × new itemsize` bytes and converts each element in C. The original is untouched. That copy is real work, so in a hot loop you convert once, up front.

!!! warning "Common mistake: casting something that doesn't fit"
    `astype` doesn't check that values fit. A `nan` (Day 2's "not a number", Day 4b's topic) has no integer value at all, and casting it gives a meaningless number with only a warning:

    ```python
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")       # collect warnings instead of printing them
        bad = np.array([1.0, np.nan]).astype(np.int64)
    print(bad)             # [                   1 -9223372036854775808]
    str(caught[0].message) # 'invalid value encountered in cast'
    ```

    (`warnings.catch_warnings(record=True)` collects warnings into a list so this page can show the exact text; in normal code you'd see a `RuntimeWarning` line printed.) Values out of an integer type's range are equally unsafe: `np.array([300.0]).astype(np.uint8)` isn't guaranteed to give 255 or anything sensible. **Fix:** clean `nan`s and clip to the target range first (`np.clip`, Day 4b), then cast.

## Integer overflow: `uint8` 250 + 10 is 4

**What it is.** A fixed-size integer has a fixed number of bits, so it can only hold a fixed range. When a result goes past the end, the extra high bits are dropped and the value **wraps round** to the other end. For `uint8`, arithmetic is effectively "modulo 256": 250 + 10 = 260, and 260 − 256 = 4.

**Why it matters.** Python's own `int` never overflows (Day 2), so Python habits say "integer maths is safe". NumPy integers are machine integers and do overflow, usually **silently**. The classic victim is image processing, where pixels are `uint8`.

**Kotlin comparison.** Here your Kotlin intuition is right again: `Int.MAX_VALUE + 1` is `-2147483648` with no error, and `(250 + 10).toUByte()` is `4u`. NumPy arrays behave like Kotlin primitives, not like Python ints.

**Example.**

```python
pixels = np.array([100, 250], dtype=np.uint8)
pixels + 10                  # array([110,   4], dtype=uint8)    250 + 10 wrapped to 4
pixels * 2                   # array([200, 244], dtype=uint8)    500 − 256 = 244

big = np.array([2**62])      # int64: the default here; in the browser too, since 2**62 doesn't fit int32
print(big * 4)               # [0]   2⁶⁴ doesn't fit in 64 bits: every bit that's left is 0
```

<figure class="diagram" markdown>
![Animation: on the left, a dial numbered 0, 64, 128 and 192 round the edge, with marks at 250 and 4. A red pointer starts at 250 and turns clockwise past 255 and 0 to stop at 4. On the right, binary addition appears line by line: 250 is 1111 1010, plus 10 is 0000 1010, equals 260, 1 0000 0100; the leading 1, a ninth bit, is boxed in red with the note that it has nowhere to go and is dropped, leaving 0000 0100, which is 4](../../assets/img/phase-1/day04-uint8-wrap.svg)
<figcaption>Animated: a uint8 is an 8-bit counter. Adding past 255 drops the ninth bit, so 250 + 10 lands on 4.</figcaption>
</figure>

**How it works underneath.** 250 in binary is `1111 1010`, and 10 is `0000 1010`. Their sum, 260, needs nine bits: `1 0000 0100`. A `uint8` element has eight bits of storage, so the ninth is discarded and `0000 0100` = 4 is stored. The CPU's add instruction does exactly this; NumPy's array loops don't check for it, because checking every element would cost speed.

Single NumPy *scalars* are slower anyway, so NumPy does check those and warns. And a plain Python int that doesn't fit the array's dtype at all is an error, not a wrap:

```python
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    s = np.uint8(250) + np.uint8(10)
s                               # np.uint8(4)
str(caught[0].message)          # 'overflow encountered in scalar add'

try:
    pixels + 300                # 300 can't even be a uint8
except OverflowError as e:
    print("OverflowError:", e)
# OverflowError: Python integer 300 out of bounds for uint8
```

**Reductions are the exception.** `sum` and `prod` don't wrap small integers: they accumulate in the platform's full-size integer (64-bit here), and `mean` always works in `float64`. So the total of a `uint8` image is safe even though adding two pixels isn't:

```python
two = np.array([200, 100], dtype=np.uint8)
print(two + two)          # [144 200]     element-wise: 400 wrapped to 144
two.sum()                 # np.uint64(300)   a reduction: widened, no wrap (np.uint32 in the browser)
two.mean()                # np.float64(150.0)
```

The rule to remember: **element-wise operations keep the input dtype; `sum`, `prod` and `mean` widen.**

!!! warning "Common mistake: brightening an image in `uint8`"
    Adding brightness to a `uint8` image wraps the brightest pixels round to near-black, the bright-sky-turns-dark-speckles bug:

    ```python
    photo = np.array([[10, 120], [200, 250]], dtype=np.uint8)
    photo + 60                  # array([[ 70, 180],
                                #        [  4,  54]], dtype=uint8)   200 and 250 wrapped
    ```

    No error and no warning. **Fix:** do the arithmetic in a wider type, cap the result, and convert back. `np.minimum(a, b)` (Part 7) takes the element-wise smaller of two values:

    ```python
    wide = photo.astype(np.int16) + 60          # int16 holds up to 32,767: no wrap
    np.minimum(wide, 255).astype(np.uint8)      # array([[ 70, 180],
                                                #        [255, 255]], dtype=uint8)
    ```

    Or convert to `float32` in 0–1 first, which is what ML preprocessing does anyway (below).

## `float32` versus `float64`: precision

**What it is.** A `float64` keeps about 16 significant decimal digits; a `float32` keeps about 7. Both are IEEE-754 floats like Day 2's, with fewer bits of **mantissa** (the stored digits) in `float32`: 23 bits instead of 52.

**Why it matters.** Neural networks run in `float32` (or smaller): half the memory and much faster on GPUs and phone NPUs, and 7 digits is plenty for weights that are noisy estimates anyway. But you'll compare `float32` results with `float64` ones and see "different" numbers, and you need to know why.

**Kotlin comparison.** Exactly Kotlin's `Float` versus `Double`. `0.1f.toDouble()` in Kotlin prints `0.10000000149011612`, the same digits as below.

**Example.**

```python
f32 = np.float32(0.1)
f32                        # np.float32(0.1)        NumPy prints the shortest digits that round-trip
f32.item()                 # 0.10000000149011612    the value actually stored, shown as a 64-bit float
np.float64(0.1).item()     # 0.1

np.float32(16_777_216) + np.float32(1)   # np.float32(1.6777216e+07)   16777216 again: the +1 is lost
np.finfo(np.float32).eps   # np.float32(1.1920929e-07)
np.finfo(np.float64).eps   # np.float64(2.220446049250313e-16)
```

**How it works underneath.** **`eps`** (machine epsilon) is the gap between 1.0 and the next representable number: about 1.2 × 10⁻⁷ for `float32` and 2.2 × 10⁻¹⁶ for `float64`. Gaps scale with size: at 16 777 216 (2²⁴) the gap between `float32` neighbours is 2, so adding 1 rounds back to where it started. That explains Day 3c's `0.9100000262260437` too: 0.91 stored as `float32` is the nearest of these spaced-out values, and widening to a Python float shows all its digits.

!!! warning "Common mistake: comparing float32 and float64 results with `==`"
    The same calculation in two precisions gives slightly different answers, and `==` says they differ:

    ```python
    third64 = np.float64(1) / 3
    third32 = np.float32(1) / np.float32(3)
    third64 == third32          # np.False_
    float(third64 - third32)    # -9.934107481068821e-09
    ```

    The difference is around the 8th digit, exactly `float32`'s limit. **Fix:** compare floats with a tolerance: `math.isclose` from Day 2, or `np.allclose` and `np.isclose`, which Day 4b teaches. Never test floats for exact equality.

## Type promotion: what dtype does a result get?

**What it is.** When an operation mixes dtypes, NumPy **promotes** both to a common dtype that can represent both, and the result has that dtype. The rules you need:

1. Same kind, different sizes → the bigger one: `int8 + int16` → `int16`.
2. Integer and float → a float big enough: `int64 + float32` → `float64`.
3. Signed and unsigned → a signed type that holds both ranges: `uint8 + int8` → `int16`.
4. `/` (true division) always gives a float, even `int / int`, like Python's `/` (Day 2).
5. A plain Python number is "weak": it adapts to the array's dtype instead of forcing its own. `float32_array * 2.0` stays `float32`. But a Python float with an *integer* array still needs a float result, and the default float is `float64`.

**Why it matters.** Promotion decides memory (a silent `float64` doubles it) and whether a model accepts the input at all.

**Example.** `np.result_type(a, b)` tells you the promoted dtype without computing anything:

```python
np.result_type(np.int8, np.int16)       # dtype('int16')
np.result_type(np.int64, np.float32)    # dtype('float64')
np.result_type(np.uint8, np.int8)       # dtype('int16')

counts = np.array([1, 2, 3])
(counts / 2).dtype                      # dtype('float64')   rule 4
(counts // 2).dtype                     # dtype('int64')     // stays integer (int32 in the browser)

w = np.ones(3, dtype=np.float32)
(w * 2.0).dtype                         # dtype('float32')   rule 5: the Python float adapts

img8 = np.array([0, 128, 255], dtype=np.uint8)
(img8 / 255.0).dtype                    # dtype('float64')   rule 5: integer array + Python float
(img8 / np.float32(255)).dtype          # dtype('float32')   a float32 scalar decides it
```

**How it works underneath.** Before running the C loop, NumPy picks one loop for the pair of input dtypes, the "add two float64s" loop for example, converts both inputs to that dtype on the fly, and allocates an output of that dtype. The loop itself never mixes types.

!!! warning "Common mistake: `/ 255.0` silently doubles your image's memory"
    The standard preprocessing step "scale pixels to 0–1" written the natural way produces `float64`, eight times the size of the `uint8` image and twice what a model wants:

    ```python
    photo = np.zeros((224, 224, 3), dtype=np.uint8)
    photo.nbytes                                # 150528
    (photo / 255.0).nbytes                      # 1204224   float64: 8 bytes per value
    (photo.astype(np.float32) / 255).nbytes     # 602112    float32: what the model expects
    ```

    A LiteRT float model given `float64` data refuses it with a type-mismatch error, and a PyTorch layer given `float64` input raises a dtype-mismatch `RuntimeError`. **Fix:** `astype(np.float32)` first, then divide.

!!! warning "Common mistake: in-place operators can't change the dtype"
    `a += x` writes the result back into `a`'s existing buffer, so the result must fit `a`'s dtype. Adding a float into an integer array in place is refused:

    ```python
    steps = np.array([1, 2, 3])
    try:
        steps += 0.5
    except TypeError as e:          # UFuncTypeError is a kind of TypeError
        print(type(e).__name__ + ":", e)
    # UFuncTypeError: Cannot cast ufunc 'add' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'
    ```

    (In the browser the message says `int32`.) **Fix:** `steps = steps + 0.5` makes a new `float64` array and rebinds the name; or create the array as float in the first place.

---

**Part 4: Indexing and slicing**

## 1-D: like a list

**What it is.** A 1-D array indexes like a Python list (Day 2b): `a[i]` for one element, counting from 0; negative indices count from the end; `a[start:stop:step]` for a slice.

**Why it exists.** Same syntax, so nothing new to learn for one axis. The difference is what a slice *is* (a view, Part 6).

**Example.**

```python
a = np.array([10, 20, 30, 40, 50])
a[0]          # np.int64(10)
a[-1]         # np.int64(50)
a[1:4]        # array([20, 30, 40])
a[::2]        # array([10, 30, 50])
a[::-1]       # array([50, 40, 30, 20, 10])    reversed
a[3:100]      # array([40, 50])                a slice past the end is trimmed, as with lists
```

(In the browser, default-integer scalars show as `np.int32(...)` and `dtype('int32')`; from here on this page shows the 64-bit form for integer scalars and reprs alike.)

!!! warning "Common mistake: an index past the end"
    A single index out of range is an `IndexError`, and the message names the axis and its size:

    ```python
    try:
        a[5]
    except IndexError as e:
        print("IndexError:", e)
    # IndexError: index 5 is out of bounds for axis 0 with size 5
    ```

    The valid indices are 0 to 4 (or -5 to -1). "axis 0" matters on bigger arrays: it tells you *which* index was wrong.

## 2-D: one index per axis

**What it is.** A 2-D array takes one index per axis inside **one** pair of brackets, separated by commas: `m[row, col]`. Each position can be an integer or a slice. `:` alone means "everything along this axis".

**Why it exists.** Rows and columns are the everyday operations of data work: `X[:, 2]` is feature 2 for every sample; `X[:100]` is the first 100 samples.

**Kotlin comparison.** With `Array<IntArray>`, you write `grid[1][2]`: first fetch row 1 (a separate `IntArray` object), then element 2 of it. There's no way to take a *column* without a loop: `grid.map { it[1] }`. NumPy indexes all axes at once and takes columns as easily as rows.

**Example.** `reshape` gives an existing array a new shape over the same values, in reading order; `np.arange(12).reshape(3, 4)` is the numbers 0 to 11 laid out as 3 rows of 4. Day 8 covers `reshape` in full; here it's a quick way to make a grid.

```python
m = np.arange(12).reshape(3, 4)
m                # array([[ 0,  1,  2,  3],
                 #        [ 4,  5,  6,  7],
                 #        [ 8,  9, 10, 11]])
m[1, 2]          # np.int64(6)                  row 1, column 2
m[0]             # array([0, 1, 2, 3])          a whole row: missing trailing indices mean ":"
m[:, 1]          # array([1, 5, 9])             column 1 of every row
m[:2, 1:3]       # array([[1, 2],
                 #        [5, 6]])              rows 0–1, columns 1–2
m[-1, -1]        # np.int64(11)                 last row, last column
m[::2, ::3]      # array([[ 0,  3],
                 #        [ 8, 11]])            every other row, every third column
```

**How it works underneath: an integer removes an axis, a slice keeps it.** This is the rule that decides the shape of every indexing result. An integer index picks *one* position, so that axis disappears. A slice picks a *range*, even a range of length 1, so the axis stays:

```python
m[1].shape       # (4,)      integer on axis 0: axis 0 is gone
m[1:2].shape     # (1, 4)    slice on axis 0: still 2-D, one row
m[:, 1].shape    # (3,)      integer on axis 1: a 1-D array, not a column
m[:, 1:2].shape  # (3, 1)    slice on axis 1: a real column, 3 rows × 1
m[1:2, 1:2]      # array([[5]])   a 1 × 1 array, not the number 5
```

You'll need this from Day 8 on: `(3,)` and `(3, 1)` look alike and behave differently in arithmetic (Day 9's first classic bug).

!!! warning "Common mistake: `m[1][2]` works, `m[1:][2:]` doesn't mean what you think"
    `m[1][2]` gives 6 like `m[1, 2]`, but it first builds the row `m[1]` and then indexes that. With slices the two-step form is wrong, because the second bracket slices the *result's rows* again:

    ```python
    m[1:, 2:]        # array([[ 6,  7],
                     #        [10, 11]])    rows from 1, columns from 2: what you meant
    m[1:][2:]        # array([], shape=(0, 4), dtype=int64)    rows from 1, then rows 2+ of those: none left (int32 in the browser)
    ```

    **Fix:** always use one bracket with commas, `m[rows, cols]`.

## 3-D and more: images and batches

**What it is.** The same rule extends to any number of axes: one index per axis. An RGB image in **HWC** layout (height, width, channels) has shape `(H, W, 3)`; a batch adds a first axis, `(N, H, W, 3)`. `...` (**Ellipsis**) means "all the axes I haven't written", so `x[..., 0]` is "index 0 on the last axis, everything else in full".

**Why it exists.** Cropping images, picking colour channels, taking one example from a batch: this is preprocessing code.

**Kotlin comparison.** In Kotlin an image like this would be `Array<Array<IntArray>>`, read as `img[r][c][ch]`, three hops through three levels of objects. NumPy writes `img[r, c, ch]` and does one address calculation into one buffer.

**Example.** A tiny 2 × 3 "image" with 3 channels, built so each value tells you where it lives (hundreds = row, tens = column, units = channel):

```python
tiny = np.array([[[  0,   1,   2], [ 10,  11,  12], [ 20,  21,  22]],
                 [[100, 101, 102], [110, 111, 112], [120, 121, 122]]])
tiny.shape          # (2, 3, 3)          H=2, W=3, C=3
tiny[1, 2]          # array([120, 121, 122])   the pixel at row 1, column 2: its R, G, B
tiny[1, 2, 0]       # np.int64(120)            its red value
tiny[:, :, 0]       # array([[  0,  10,  20],
                    #        [100, 110, 120]])  the red channel: shape (2, 3)
tiny[..., 0].shape  # (2, 3)                    the same thing, with ...
tiny[:, 1:]         # array([[[ 10,  11,  12],
                    #         [ 20,  21,  22]],
                    #
                    #        [[110, 111, 112],
                    #         [120, 121, 122]]])   a crop: columns 1 to the end, all rows and channels
```

A 3-D repr prints each block of the first axis separately, with a blank line between them.

And a batch of images, one more axis in front:

```python
batch = np.zeros((32, 224, 224, 3), dtype=np.uint8)   # 32 photos, NHWC
batch[0].shape                  # (224, 224, 3)   the first photo
batch[:8].shape                 # (8, 224, 224, 3)   the first 8 photos
batch[:, 112, 112].shape        # (32, 3)            the centre pixel of every photo
batch[..., 2].shape             # (32, 224, 224)     the blue channel of every photo
```

**How it works underneath.** The integer-removes, slice-keeps rule applied axis by axis: `batch[:, 112, 112]` keeps axis 0 (slice), removes axes 1 and 2 (integers), and the unwritten last axis is kept in full, leaving `(32, 3)`.

!!! warning "Common mistake: indexing channels as if they came first"
    In HWC layout the channel is the **last** index. Code written for PyTorch's channels-first layout (CHW, Day 8) reads `img[0]` as "the red channel", but here it's row 0, and no error tells you:

    ```python
    tiny[0].shape            # (3, 3)   row 0: 3 pixels × 3 channels, not a colour plane
    tiny[..., 0].shape       # (2, 3)   the red channel
    try:
        tiny[0, 0, 0, 0]
    except IndexError as e:
        print("IndexError:", e)
    # IndexError: too many indices for array: array is 3-dimensional, but 4 were indexed
    ```

    The second mistake, one index too many (for example, writing batch-style indexing on a single image), at least fails loudly. **Fix:** print `.shape` and say out loud what each axis means before indexing.

## Picking by position: integer-array ("fancy") indexing

**What it is.** Instead of one integer or a slice, you can index with a **list or array of integer positions**. The result has one element per position, in the order you gave them, and repeats are allowed. NumPy calls this **integer-array indexing**; most people call it **fancy indexing**. It always returns a **copy**.

**Why it exists.** A slice can only pick a regular range. Real code needs arbitrary picks: rows in a shuffled order (Day 4c's train/test split), the top-k items found by sorting (Day 4b's `argsort`), a label id turned into its name, or "for each example, the probability of its true class".

**Kotlin comparison.** `IntArray.slice(listOf(3, 0, 0))` returns a new `List<Int>` of those elements, in that order. NumPy's version is `a[[3, 0, 0]]`, returns an array, and works on any axis.

**Example.**

```python
a = np.array([10, 20, 30, 40, 50])
a[[3, 0, 0]]                 # array([40, 10, 10])   any order, repeats allowed
idx = np.array([4, 1])
a[idx]                       # array([50, 20])       an index array works the same as a list

X = np.arange(6).reshape(3, 2)
X[[2, 0]]                    # array([[4, 5],
                             #        [0, 1]])       whole rows 2 and 0, reordered: shape (2, 2)

names = np.array(["cat", "dog", "fox"])
labels = np.array([2, 0, 0, 1])
names[labels]                # array(['fox', 'cat', 'cat', 'dog'], dtype='<U3')   a lookup table
```

**One index array per axis pairs them up.** With an index array on *two* axes, NumPy pairs the arrays element by element, `M[rows[i], cols[i]]`, and returns one value per pair; it doesn't make a block:

```python
M = np.arange(12).reshape(3, 4)
M[[0, 1, 2], [3, 2, 1]]      # array([3, 6, 9])      the pairs (0, 3), (1, 2), (2, 1)

probs = np.array([[0.1, 0.2, 0.7],
                  [0.6, 0.3, 0.1],
                  [0.2, 0.5, 0.3]])       # 3 examples × 3 classes
true_class = np.array([2, 0, 1])
probs[np.arange(3), true_class]           # array([0.7, 0.6, 0.5])   each example's probability of its true class
```

That last line, "row `i`, column `true_class[i]`", is the heart of the cross-entropy loss you'll meet in Phase 3.

**How it works underneath.** A slice can be described by a start and a stride, so it becomes a view. A list like `[3, 0, 0]` can't: there's no single stride that visits element 3, then 0, then 0 again. So NumPy allocates a new buffer with one slot per index and copies each selected element into it. The result's shape is the shape of the index array (plus any axes you didn't index):

```python
picked = X[[2, 0]]
np.shares_memory(X, picked)  # False   a copy
picked[0, 0] = 99
X[2, 0]                      # np.int64(4)   X is unchanged
```

!!! warning "Common mistake: expecting a block from two index lists"
    `M[[0, 2], [1, 3]]` looks like "rows 0 and 2, columns 1 and 3", a 2 × 2 block. It's two *pairs*, (0, 1) and (2, 3), so you get two numbers:

    ```python
    M[[0, 2], [1, 3]]        # array([ 1, 11])
    M[[0, 2]][:, [1, 3]]     # array([[ 1,  3],
                             #        [ 9, 11]])    rows first, then columns: the block
    ```

    **Fix:** pick rows and columns in two steps, as on the second line (Day 8 shows `np.ix_`, which does it in one). And an index past the end fails like a single index does: `M[[0, 3]]` raises `IndexError: index 3 is out of bounds for axis 0 with size 3`.

## Assigning through an index

**What it is.** Any index or slice can go on the left of `=` to write into the array. A single value is written into every selected position; a list or array must match the selected shape.

```python
canvas = np.zeros((3, 4), dtype=int)   # int means NumPy's default integer
canvas[0, 0] = 5          # one element
canvas[1] = 7             # a whole row gets 7
canvas[:, 3] = [1, 2, 3]  # a column, from a list of the right length
canvas
# array([[5, 0, 0, 1],
#        [7, 7, 7, 2],
#        [0, 0, 0, 3]])
```

Writing one value into many positions is the simplest case of **broadcasting**, NumPy's rule for combining different shapes; Day 9 covers it properly.

!!! warning "Common mistake: writing a float into an integer array"
    The array's dtype wins. Assigning `9.9` into an integer array truncates it, with no warning:

    ```python
    canvas[2, 0] = 9.9
    canvas[2, 0]              # np.int64(9)
    ```

    **Fix:** create the array with the dtype you'll need (`np.zeros((3, 4))` is float by default).

!!! warning "Common mistake: assigning the wrong number of values"
    A list on the right must match the shape of what you selected. Column 3 has 3 elements, so 2 values don't fit:

    ```python
    try:
        canvas[:, 3] = [1, 2]
    except ValueError as e:
        print("ValueError:", e)
    # ValueError: could not broadcast input array from shape (2,) into shape (3,)
    ```

    Read it as "the right-hand side has shape `(2,)`, the selection has shape `(3,)`". **Fix:** give one value (written everywhere) or exactly as many as the selection holds; check with `canvas[:, 3].shape`.

---

**Part 5: Boolean masks**

## Comparisons give arrays of booleans

**What it is.** A comparison between an array and a value (or two arrays of the same shape) compares **element by element** and returns a `bool` array of the same shape. That boolean array is called a **mask**. Indexing an array with a mask, `a[mask]`, keeps exactly the elements where the mask is `True`, as a new 1-D array.

**Why it exists.** It replaces "loop, test, append" with one expression. "Keep the confident predictions", "drop negative prices", "rows where the label is 1": all masks. pandas uses the same idea (Day 16).

**Kotlin comparison.** `scores.filter { it > 0.5f }` is `scores[scores > 0.5]`. Kotlin's version runs a lambda per element; NumPy's compares in a C loop and then copies the selected values in another.

**Example.**

```python
scores = np.array([0.91, 0.12, 0.55, 0.30, 0.78])
confident = scores > 0.5
confident                 # array([ True, False,  True, False,  True])
scores[confident]         # array([0.91, 0.55, 0.78])
scores[scores > 0.5]      # array([0.91, 0.55, 0.78])   usually written in one go
confident.sum()           # np.int64(3)      how many: True counts as 1
confident.mean()          # np.float64(0.6)  what fraction
```

`==`, `!=`, `<`, `<=`, `>`, `>=` all work this way.

**How it works underneath.** `scores > 0.5` calls `scores.__gt__(0.5)` (Day 3a's hook), which runs a C loop writing one byte per element into a new `bool` array. `scores[mask]` then checks the mask's shape matches, counts the `True`s (3), allocates a new 3-element array and copies those values in. Because the result can pick *any* subset, it can't be described as a regular step through the old buffer, so it's always a **copy** (Part 6).

On a 2-D array, a 2-D mask selects individual elements, and the result is 1-D (there's no way to keep rows and columns when you pick scattered cells):

```python
grid = np.array([[1, -2, 3],
                 [-4, 5, -6]])
grid[grid > 0]            # array([1, 3, 5])
```

## Combining conditions: `&`, `|`, `~`

**What it is.** To combine masks element by element, use `&` (and), `|` (or) and `~` (not). Each condition goes in **parentheses**.

**Why it exists.** Python's `and`, `or` and `not` can't be used here. They ask a single object "are you true?", and an array of many booleans has no single answer.

**Kotlin comparison.** Kotlin already has this split for single booleans: `&&` and `||` short-circuit, while the infix functions `and`, `or` and `inv()` evaluate both sides. Python's `and`/`or` are like `&&`/`||`; NumPy's `&`, `|` and `~` are like Kotlin's `and`, `or` and `inv()`, applied to every element.

**Example.**

```python
x = np.array([-3, -1, 0, 2, 4, 7])
(x > 0) & (x < 5)          # array([False, False, False,  True,  True, False])
x[(x > 0) & (x < 5)]       # array([2, 4])
x[(x < 0) | (x > 5)]       # array([-3, -1,  7])
x[~(x > 0)]                # array([-3, -1,  0])   ~ flips every boolean
```

**How it works underneath.** `&`, `|` and `~` are the bitwise operators, which NumPy implements element by element through `__and__`, `__or__` and `__invert__`. They bind **tighter** than comparisons in Python's grammar, the opposite of `and`/`or`, which is why the parentheses are compulsory.

!!! warning "Common mistake: `and`, or missing parentheses"
    Both give the same confusing error:

    ```python
    try:
        x > 0 and x < 5
    except ValueError as e:
        print("ValueError:", e)
    # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

    try:
        x > 0 & x < 5
    except ValueError as e:
        print("ValueError:", e)
    # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
    ```

    The first: `and` calls `bool()` on the array `x > 0`, and NumPy refuses to pick one truth value for six booleans. The second: without parentheses, `&` binds first, so Python reads `x > (0 & x) < 5`, a *chained* comparison (Day 2's `0 < x < 1`), which also uses `and` underneath. **Fix:** `(x > 0) & (x < 5)`. The same error appears for `if mask:` and `while arr:`; there you want `mask.any()` ("is any element true?") or `mask.all()` ("are all true?"), from Part 8.

## Masks on rows: picking samples by label

**What it is.** A 1-D mask on the first axis of a 2-D array selects whole **rows**.

**Why it exists.** "All the samples of class 1", "every row whose price is positive": you filter a dataset by a condition on one column or a separate label array.

```python
X = np.array([[1.0, 200.0],
              [2.0, 150.0],
              [3.0, 300.0],
              [4.0, 120.0]])       # 4 samples × 2 features
y = np.array([0, 1, 1, 0])         # one label per sample

X[y == 1]                  # array([[  2., 150.],
                           #        [  3., 300.]])   the rows of class 1
X[X[:, 1] > 160]           # array([[  1., 200.],
                           #        [  3., 300.]])   rows whose feature 1 is above 160
X[y == 1].shape            # (2, 2)
```

`X[:, 1] > 160` builds a 4-element mask from column 1; using it on axis 0 keeps the matching rows.

!!! warning "Common mistake: a mask of the wrong length"
    The mask must have the same length as the axis it selects from. A mask built from a different array, or one that lost an element, fails loudly:

    ```python
    try:
        X[np.array([True, False, True])]
    except IndexError as e:
        print("IndexError:", e)
    # IndexError: boolean index did not match indexed array along axis 0; size of axis is 4 but size of corresponding boolean axis is 3
    ```

    That's good: it's caught. The silent version is a mask from the wrong array of the *same* length, such as labels shuffled separately from features. Keep features and labels together, and shuffle them with the same permutation (Day 4c).

## Assigning through a mask

**What it is.** A mask on the left of `=` writes into the original array, at every `True` position.

**Why it exists.** Capping, zeroing, replacing bad values: one line each. It's the NumPy way to write `if` inside a loop.

**Example: ReLU.** ReLU ("rectified linear unit"), the most common activation function in neural networks (Phase 3), replaces every negative value with 0:

```python
activations = np.array([0.5, -1.2, 3.0, -0.1])
activations[activations < 0] = 0
activations               # array([0.5, 0. , 3. , 0. ])
```

Compare the plain-Python loop (in Kotlin it's `values.map { if (it > 0f) it else 0f }`, one lambda call per element):

=== "Python loop"

    ```python
    def relu_loop(values):
        out = []
        for v in values:
            out.append(v if v > 0 else 0.0)
        return out

    relu_loop([0.5, -1.2, 3.0, -0.1])   # [0.5, 0.0, 3.0, 0.0]
    ```

=== "NumPy"

    ```python
    def relu(x):
        out = x.copy()          # don't change the caller's array (Part 6)
        out[out < 0] = 0
        return out

    relu(np.array([0.5, -1.2, 3.0, -0.1]))   # array([0.5, 0. , 3. , 0. ])
    ```

A third, shorter version with `np.maximum` comes in Part 7, and Day 4b adds `np.where` and `np.clip`.

---

**Part 6: Views and copies**

## Slices are views

**What it is.** A **view** is a new array object with its own header (shape, strides, a starting offset) that reads the **same data buffer** as another array. A **copy** has its own new buffer. Basic slicing (`a[1:4]`, `m[:, 2]`, `m[::2]`) always returns a view. Writing through a view changes the original, and the other way round.

**Why it exists.** Speed and memory. Slicing a 2 GB dataset into training and validation halves shouldn't copy 2 GB. A view costs a header of a hundred-odd bytes, whatever the slice size.

**Kotlin comparison.** This is the opposite of Kotlin's collections. `list.subList(1, 4)` is a view, but `array.copyOfRange(1, 4)` and `array.sliceArray(1..3)` copy. A Python *list* slice copies too (Day 2b). NumPy slices **don't**. The closest Android match is `ByteBuffer.slice()` or `duplicate()`: a new buffer object over the same bytes, so a `put` through one is visible through the other.

**Example.** `np.shares_memory(a, b)` answers "do these two arrays use any of the same bytes?". A view's `.base` attribute is the array that owns the buffer (`None` for an array that owns its own).

```python
a = np.array([10, 20, 30, 40, 50, 60])
v = a[1:4]                 # a view
v                          # array([20, 30, 40])
np.shares_memory(a, v)     # True
v.base is a                # True    v borrows a's buffer
v[0] = 99                  # write through the view...
a                          # array([10, 99, 30, 40, 50, 60])   ...and a changed
a[3] = -5
v                          # array([99, 30, -5])               and the other way round
```

<figure class="diagram" markdown>
![Animation: the array a holds 10, 20, 30, 40, 50, 60 in one buffer with byte offsets 0 to 40. A dashed blue frame labelled "v = a[1:4], view" sits over the cells at bytes 8 to 24 of that same buffer. Step 2 writes 99 into v[0], and the cell for a[1] in the shared buffer turns yellow and shows 99. Step 3 shows "c = a[a > 30], copy": dashed orange arrows copy 40, 50 and 60 down into a separate new buffer. Step 4 writes -1 into c[0]; only the copy changes, and a still reads 10, 99, 30, 40, 50, 60](../../assets/img/phase-1/day04-view-vs-copy.svg)
<figcaption>Animated: a slice reads the original buffer through a new header; a mask copies the chosen values into a new buffer.</figcaption>
</figure>

**How it works underneath.** `v = a[1:4]` creates a header that says "start 8 bytes into `a`'s buffer, shape `(3,)`, stride 8". No values move. A slice with a step, `a[::2]`, is still a view: its stride is 16, "skip every other element". A column `m[:, 1]` is a view whose stride is a whole row's width. Any regular pattern "start here, step this much along each axis" can be a view.

## Which operations copy

**What it is.** A result is a **copy** whenever its elements can't be described as a regular stride pattern over the old buffer, or when the operation computes new values:

- **Mask indexing**, `a[a > 30]`: picks an arbitrary subset → copy.
- **Integer-array ("fancy") indexing**, `a[[0, 3, 4]]`: a list of positions, in any order, repeats allowed → copy. Part 4 taught it; Day 8 adds the advanced forms.
- **Arithmetic** and ufuncs, `a + 1`, `np.sqrt(a)`: new values → new array.
- **`astype`**: new dtype → copy.
- **`np.array(a)`** and **`a.copy()`**: explicit copies. Its non-copying cousin is **`np.asarray(a)`**: it converts a list to a new array, but hands back an existing array unchanged. Library functions start with `np.asarray(x)` so they accept lists and arrays without copying arrays.

And views, besides slices: `reshape` (usually), `.T` (transpose) and adding an axis with `None` (Day 8).

**Example.**

```python
a = np.array([10, 20, 30, 40, 50, 60])
c = a[a > 30]              # a mask: copy
np.shares_memory(a, c)     # False
c[0] = -1
a                          # array([10, 20, 30, 40, 50, 60])   unchanged

f = a[[0, 2]]              # fancy index: copy
np.shares_memory(a, f)     # False
np.shares_memory(a, a + 0) # False   arithmetic makes a new array

np.shares_memory(a, np.array(a))     # False   np.array always copies
np.shares_memory(a, np.asarray(a))   # True    np.asarray reuses an existing array

s = a[:3].copy()           # a slice, explicitly copied
s[0] = 0
a[0]                       # np.int64(10)   safe
```

The summary, now that each case has been shown:

| Operation | Result |
|---|---|
| basic slice `a[1:4]`, `m[:, 0]`, `a[::2]` | view |
| `reshape` (usually), `.T`, `a[None]` | view (Day 8) |
| mask `a[a > 0]` | copy |
| integer list `a[[0, 2]]` | copy |
| `a + 1`, `np.sqrt(a)`, `astype` | new array |
| `a.copy()`, `np.array(a)` | copy |
| `np.asarray(a)` on an array | the same array |

## Views in practice: the bugs

!!! warning "Common mistake: normalising a slice changes the dataset"
    You take the first rows as a validation set and scale them in place. The original dataset is scaled too, because the slice is a view:

    ```python
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    val = data[:2]             # a view of rows 0 and 1
    val /= 10                  # in place: writes into data's buffer
    data
    # array([[0.1, 0.2],
    #        [0.3, 0.4],
    #        [5. , 6. ]])
    ```

    No error; the next experiment silently trains on rescaled data. **Fix:** `val = data[:2].copy()` when you intend to modify it, or write `val = val / 10`, which makes a new array instead of writing in place.

!!! warning "Common mistake: assigning through a chained mask does nothing"
    `x[mask] = value` writes into `x`. But `x[mask][0] = value` first *copies* the masked elements, then writes into that temporary copy, which is thrown away:

    ```python
    x = np.array([5, -3, 8, -1])
    x[x < 0][0] = 100          # writes into a temporary copy
    x                          # array([ 5, -3,  8, -1])   unchanged
    x[x < 0] = 100             # one indexing step: writes into x
    x                          # array([  5, 100,   8, 100])
    ```

    pandas warns about the same pattern ("chained assignment", Day 16). **Fix:** do the selection and assignment in one indexing step.

!!! warning "Common mistake: `+=` inside a function changes the caller's array"
    Day 3 showed that a function receives the caller's object, not a copy. `x += 1` writes into that object's buffer; `x = x + 1` creates a new array and rebinds the local name only:

    ```python
    def add_noise_in_place(x):
        x += 1                 # modifies the caller's array
        return x

    def add_noise_new(x):
        x = x + 1              # a new array; the caller's is untouched
        return x

    weights = np.array([1.0, 2.0])
    add_noise_new(weights)     # array([2., 3.])
    weights                    # array([1., 2.])
    add_noise_in_place(weights)   # array([2., 3.])
    weights                    # array([2., 3.])   changed
    ```

    This is the same names-and-objects rule as Day 2's lists, so in-place updates are a feature when you want them (optimisers update weights in place to save memory). **Fix:** decide which you mean, and copy at the top of the function (`x = x.copy()`) if callers shouldn't see changes.

---

**Part 7: Element-wise maths and ufuncs**

## Arithmetic on whole arrays

**What it is.** The operators `+ - * / // % **` on two arrays of the same shape work **element by element**: `a + b` adds `a[i]` and `b[i]` for every `i`. An array with a single number applies that number to every element. This style, whole arrays instead of loops over elements, is called **vectorised** code.

**Why it exists.** Most ML maths is element-wise: scaling, adding a bias, squaring errors, applying an activation.

**Kotlin comparison.** Kotlin has no arithmetic operators on `FloatArray`. You'd write `FloatArray(a.size) { a[it] + b[it] }`, or define `operator fun plus` yourself. NumPy defines `__add__`, `__mul__` and the rest (Day 3a) to loop in C.

**Example.**

```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])
a + b          # array([11, 22, 33, 44])
b - a          # array([ 9, 18, 27, 36])
a * b          # array([ 10,  40,  90, 160])   element-wise, not a dot product (that's Day 5)
b / a          # array([10., 10., 10., 10.])
a ** 2         # array([ 1,  4,  9, 16])
a * 2          # array([2, 4, 6, 8])
1 / a          # array([1.        , 0.5       , 0.33333333, 0.25      ])
```

An ML example: the **mean squared error** between predictions and targets, the loss you'll minimise in Phase 2, is the average of the squared differences:

```python
pred = np.array([2.5, 0.0, 2.0, 8.0])
target = np.array([3.0, -0.5, 2.0, 7.0])
errors = pred - target       # array([-0.5,  0.5,  0. ,  1. ])
(errors ** 2).mean()         # np.float64(0.375)   (0.25 + 0.25 + 0 + 1) / 4
```

!!! warning "Common mistake: `*` on a list repeats it"
    The same operator means different things on a list and an array, and the list version doesn't fail:

    ```python
    [1, 2, 3] * 2              # [1, 2, 3, 1, 2, 3]          list: repetition (Day 2b)
    np.array([1, 2, 3]) * 2    # array([2, 4, 6])            array: multiplication
    [1, 2] + [3, 4]            # [1, 2, 3, 4]                list: concatenation
    np.array([1, 2]) + np.array([3, 4])   # array([4, 6])   array: addition
    ```

    **Fix:** convert with `np.array(...)` before doing maths. If a result is unexpectedly long, check whether something was still a list.

!!! warning "Common mistake: shapes that don't match"
    Two arrays of different lengths can't be paired up element by element:

    ```python
    try:
        np.array([1, 2, 3]) + np.array([1, 2, 3, 4])
    except ValueError as e:
        print("ValueError:", e)
    # ValueError: operands could not be broadcast together with shapes (3,) (4,)
    ```

    "broadcast" is NumPy's name for the rules that pair up shapes; Day 9 explains exactly which different shapes *are* compatible (a `(4, 3)` matrix plus a `(3,)` row is fine, for example). Today's rule: same shape, or one side is a single number. Read the two shapes in the message; they tell you what was mismatched.

## Universal functions (ufuncs)

**What it is.** A **ufunc** ("universal function") is a NumPy function that applies an operation element by element, in C, to whole arrays: `np.sqrt`, `np.exp`, `np.log`, `np.abs`, `np.sin`, `np.maximum`, `np.minimum` and dozens more. The operators above are ufuncs too: `a + b` calls `np.add(a, b)`.

**Why it exists.** The `math` module (Day 2) works on one Python number at a time. ML formulas need these functions on whole arrays: `exp` for softmax and sigmoid, `log` for losses, `sqrt` for norms.

**Example.**

```python
v = np.array([1.0, 4.0, 9.0])
np.sqrt(v)                 # array([1., 2., 3.])
np.exp(np.array([0.0, 1.0]))   # array([1.        , 2.71828183])
np.log(np.array([1.0, np.e]))  # array([0., 1.])       natural log; np.e is Euler's number
np.abs(np.array([-2, 3]))      # array([2, 3])

logits = np.array([-2.0, 0.0, 3.0])
1 / (1 + np.exp(-logits))      # array([0.11920292, 0.5       , 0.95257413])   the sigmoid
```

The **sigmoid**, \( 1 / (1 + e^{-x}) \), squashes any number into 0–1; it turns a model's raw score (a **logit**) into a probability. One line, any number of scores.

`np.maximum(a, b)` and `np.minimum(a, b)` compare **two** arrays (or an array and a number) element by element and keep the larger or smaller of each pair. That gives ReLU in one line:

```python
np.maximum(0, np.array([0.5, -1.2, 3.0, -0.1]))   # array([0.5, 0. , 3. , 0. ])
np.minimum(np.array([300, 12]), 255)              # array([255,  12])
```

**How it works underneath.** A ufunc checks the inputs' shapes and dtypes, picks the compiled inner loop for that dtype (a `float64` square root, say), allocates the output and runs the loop. Invalid inputs don't raise; they produce IEEE special values and a warning: `np.log(0.0)` is `-inf` ("divide by zero encountered in log") and `np.sqrt(-1.0)` is `nan` ("invalid value encountered in sqrt"). Day 4b covers `nan` and `inf` and how to control those warnings.

!!! warning "Common mistake: `np.max` versus `np.maximum`"
    The names are one letter apart and do different jobs. `np.max(a)` is a **reduction** (Part 8): the single largest element of one array. `np.maximum(a, b)` is a ufunc: element-wise, two inputs, same-shape output. Using `max` for ReLU gives one number instead of an array:

    ```python
    acts = np.array([0.5, -1.2, 3.0, -0.1])
    np.max(acts)          # np.float64(3.0)                 the largest element
    np.maximum(acts, 0)   # array([0.5, 0. , 3. , 0. ])     element-wise against 0
    ```

    (`np.max(acts, 0)` doesn't fail either: the `0` is taken as `axis=0`, and you still get `3.0`.) **Fix:** `maximum`/`minimum` for "clamp each element", `max`/`min` for "the biggest one".

!!! warning "Common mistake: `math` functions on arrays"
    `math.sqrt` only accepts one number:

    ```python
    import math
    try:
        math.sqrt(np.array([1.0, 4.0]))
    except TypeError as e:
        print("TypeError:", e)
    # TypeError: only 0-dimensional arrays can be converted to Python scalars
    ```

    "0-dimensional" means a single number. Older NumPy versions word the same error "only length-1 arrays can be converted to Python scalars", and the browser's NumPy may too. **Fix:** use the `np.` version (`np.sqrt`, `np.exp`, `np.log`), which takes arrays and single numbers alike.

## Comparing whole arrays

Because `==` is element-wise, `a == b` is an array, not a yes/no answer. To ask "are these two arrays the same?", reduce the mask with `.all()` (Part 8), or use `np.array_equal`, which also checks the shapes match:

```python
p = np.array([1, 2, 3])
q = np.array([1, 2, 4])
p == q                     # array([ True,  True, False])
(p == q).all()             # np.False_
np.array_equal(p, p.copy())   # True
```

`if p == q:` raises the "truth value ... is ambiguous" error from Part 5. For floats, compare with a tolerance (`np.allclose`, Day 4b).

---

**Part 8: Reductions and `axis`**

## Reducing a whole array

**What it is.** A **reduction** collapses many values into fewer, often one: `sum`, `mean`, `std` (standard deviation, the typical distance from the mean), `var` (variance, `std` squared), `min`, `max`, `argmin` and `argmax` (the *index* of the smallest or largest), `prod`, `any` and `all`. Each exists as a method (`a.sum()`) and as a function (`np.sum(a)`); they're the same.

**Why it exists.** Losses are means, accuracies are means of masks, predictions are argmaxes, normalisation needs means and standard deviations.

**Kotlin comparison.** `floatArray.sum()`, `average()`, `max()`, `indexOfMax`... Kotlin has a few of these on arrays. NumPy has them all, and, as you'll see next, along any axis.

**Example.**

```python
r = np.array([4.0, 1.0, 7.0, 3.0])
r.sum()        # np.float64(15.0)
r.mean()       # np.float64(3.75)
r.min(), r.max()     # (np.float64(1.0), np.float64(7.0))
r.argmin(), r.argmax()   # (np.int64(1), np.int64(2))    positions, not values
r.std()        # np.float64(2.165063509461097)
r.var()        # np.float64(4.6875)
(r > 5).any()  # np.True_     is any element above 5?
(r > 0).all()  # np.True_     are all of them positive?
```

`argmax` is the one you'll use most in ML: given scores for each class, the index of the highest score *is* the predicted class. When several elements tie for the maximum, `argmax` returns the **first** one, so a tie always goes to the lowest class index:

```python
np.array([0.3, 0.3, 0.4, 0.4]).argmax()   # np.int64(2)   0.4 appears at 2 and 3: the first wins
np.array([1, 2, 3, 4]).prod()             # np.int64(24)  prod multiplies: 1 × 2 × 3 × 4
```

**How it works underneath.** A reduction runs one C loop that keeps a running result: a total for `sum`, the best-so-far and its position for `argmax`. `mean` is `sum / size`. `std` computes the mean, then the square root of the average squared distance from it: here the distances from 3.75 are 0.25, −2.75, 3.25, −0.75; squared 0.0625, 7.5625, 10.5625, 0.5625; their mean is 4.6875 (the variance), and its square root is 2.165…

!!! warning "Common mistake: NumPy's `std` and pandas' `std` disagree"
    NumPy divides by `n` by default; pandas (Week 3) and most statistics textbooks divide by `n − 1` for a *sample* standard deviation (Day 43 explains why). Same data, different numbers:

    ```python
    r.std()            # np.float64(2.165063509461097)    divides by n = 4
    r.std(ddof=1)      # np.float64(2.5)                  divides by n − 1 = 3
    ```

    `ddof` means "delta degrees of freedom": the divisor is `n − ddof`. **Fix:** pass `ddof=1` when you want to match pandas or a textbook formula; for normalising features in ML the default is fine, as long as you're consistent.

!!! warning "Common mistake: reducing an empty selection"
    `sum` of nothing is 0 and `mean` of nothing is `nan` with a warning, but `max`, `min`, `argmax` and `argmin` have no sensible answer, so they raise. This happens in real code when a mask selects nothing:

    ```python
    probs_seen = np.array([0.42, 0.87, 0.91])
    very_sure = probs_seen[probs_seen > 0.99]     # nothing passes: an empty array
    very_sure.size                                # 0
    try:
        very_sure.max()
    except ValueError as e:
        print("ValueError:", e)
    # ValueError: zero-size array to reduction operation maximum which has no identity
    very_sure.max(initial=0.0)                    # np.float64(0.0)
    ```

    "No identity" means there's no neutral starting value for `max`, the way 0 is for `sum`. **Fix:** check `.size` before reducing a filtered array, or pass `initial=` to say what the answer should be when there's nothing to compare.

## `axis` on a 2-D array

**What it is.** A reduction with `axis=k` reduces **along axis `k` only**, giving one result for every position of the other axes. The axis you name is the one that **disappears** from the shape.

**Why it exists.** You rarely want one number for a whole dataset. You want one mean *per feature* (per column), or one prediction *per example* (per row).

**Intuition first.** Picture the `(3, 4)` grid `m` from Part 4. Axis 0 runs **down** the rows; axis 1 runs **across** the columns. `sum(axis=0)` walks down axis 0, adding the three rows together, so axis 0 disappears and one total per column is left: shape `(4,)`. `sum(axis=1)` walks across each row, so axis 1 disappears: one total per row, shape `(3,)`.

**Example.** By hand first. With

```text
m = [[ 0,  1,  2,  3],
     [ 4,  5,  6,  7],
     [ 8,  9, 10, 11]]
```

the column totals are 0 + 4 + 8 = 12, 1 + 5 + 9 = 15, 18 and 21, and the row totals are 0 + 1 + 2 + 3 = 6, 22 and 38. Now the code:

```python
m = np.arange(12).reshape(3, 4)
m.sum()               # np.int64(66)                 everything
m.sum(axis=0)         # array([12, 15, 18, 21])      axis 0 (rows) disappears: one per column
m.sum(axis=1)         # array([ 6, 22, 38])          axis 1 (columns) disappears: one per row
m.sum(axis=0).shape   # (4,)
m.mean(axis=0)        # array([4., 5., 6., 7.])
m.max(axis=1)         # array([ 3,  7, 11])
```

The two places you'll use this most:

```python
# a dataset: rows are samples, columns are features → (samples, features)
houses = np.array([[3, 120.0, 10],
                   [2,  80.0, 25],
                   [4, 200.0,  5]])        # bedrooms, area in m², age in years
houses.mean(axis=0)        # array([  3.        , 133.33333333,  13.33333333])   one mean per feature

# a model's output: rows are examples, columns are classes → (batch, classes)
class_scores = np.array([[0.1, 0.7, 0.2],
                         [0.8, 0.1, 0.1],
                         [0.3, 0.3, 0.4]])
class_scores.argmax(axis=1)   # array([1, 0, 2])   the predicted class of each example
```

The feature means are the first step of standardising features (Day 9 finishes it). The per-row argmax turns a batch of class scores into predicted labels.

!!! warning "Common mistake: `argmax` without `axis` on a 2-D array"
    Without `axis`, `argmax` treats the array as one long flat row and returns a position in that **flattened** order, not a row or a column:

    ```python
    class_scores.argmax()          # np.int64(3)   the 0.8, at flat position 3 (row 1, column 0)
    np.unravel_index(3, class_scores.shape)   # (np.int64(1), np.int64(0))
    ```

    One number instead of one per example, and no error. **Fix:** `argmax(axis=1)` for per-row predictions. When you do want the location of the overall maximum, `np.unravel_index(flat_index, shape)` converts the flat position back into `(row, column)`.

## `axis` on a 3-D array, traced

**What it is.** The same rule for any number of axes: name an axis, it disappears, and everything else stays. You can also pass a **tuple** of axes to reduce several at once.

**Why it exists.** Image batches are 4-D `(N, H, W, C)`. "The mean brightness of each image" reduces axes 1, 2 and 3; "the mean of each colour channel over the whole batch" reduces axes 0, 1 and 2.

**Example.** `x` has 2 blocks of 3 rows of 4 columns. Element `x[b, j, k]` is `12b + 4j + k`.

```python
x = np.arange(24).reshape(2, 3, 4)
x
# array([[[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]],
#
#        [[12, 13, 14, 15],
#         [16, 17, 18, 19],
#         [20, 21, 22, 23]]])
```

Trace each axis by hand, then check:

- **`axis=0`** (which block): add the two blocks cell by cell, `x[0, j, k] + x[1, j, k]`. The first cell is 0 + 12 = 12, the next 1 + 13 = 14. Shape `(2, 3, 4)` loses the 2: `(3, 4)`.
- **`axis=1`** (which row): inside each block, add down each column, `x[b, 0, k] + x[b, 1, k] + x[b, 2, k]`. Block 0, column 0: 0 + 4 + 8 = 12. Block 1, column 0: 12 + 16 + 20 = 48. Shape loses the 3: `(2, 4)`.
- **`axis=2`** (which column): inside each block, add along each row. Block 0, row 0: 0 + 1 + 2 + 3 = 6. Shape loses the 4: `(2, 3)`.

<figure class="diagram" markdown>
![Animation of x = np.arange(24).reshape(2, 3, 4) drawn as two 3 by 4 grids, x[0] holding 0 to 11 and x[1] holding 12 to 23, with arrows marking axis 0 between the blocks, axis 1 down the rows and axis 2 across the columns. Three phases cycle. Phase 1: the top-left cell of both blocks is outlined and the result x.sum(axis=0), shape (3, 4), appears with 12 outlined, explained as 0 + 12. Phase 2: column 0 of block 0 is outlined and x.sum(axis=1), shape (2, 4), appears with rows 12 15 18 21 and 48 51 54 57, the first 12 explained as 0 + 4 + 8. Phase 3: row 0 of block 0 is outlined and x.sum(axis=2), shape (2, 3), appears with rows 6 22 38 and 54 70 86, the first 6 explained as 0 + 1 + 2 + 3](../../assets/img/phase-1/day04-axis-3d.svg)
<figcaption>Animated: for each axis, the outlined cells are added into the outlined result cell. The named axis disappears from the shape.</figcaption>
</figure>

```python
x.sum(axis=0)
# array([[12, 14, 16, 18],
#        [20, 22, 24, 26],
#        [28, 30, 32, 34]])
x.sum(axis=1)
# array([[12, 15, 18, 21],
#        [48, 51, 54, 57]])
x.sum(axis=2)
# array([[ 6, 22, 38],
#        [54, 70, 86]])
x.sum(axis=(1, 2))         # array([ 66, 210])   one total per block: both inner axes disappear
x.sum(axis=-1).shape       # (2, 3)              -1 means the last axis, as in indexing
```

**How it works underneath.** NumPy loops over every position of the axes you *didn't* name, and for each one runs the reduction along the named axis. For `axis=1` on `(2, 3, 4)`, that's 2 × 4 = 8 positions, each adding 3 numbers, giving 8 results laid out as `(2, 4)`.

The image versions:

```python
rng = np.random.default_rng(0)
imgs = rng.integers(0, 256, size=(8, 32, 32, 3)).astype(np.float32)   # 8 tiny RGB images
imgs.mean(axis=(1, 2, 3)).shape    # (8,)   one brightness per image
imgs.mean(axis=(0, 1, 2)).shape    # (3,)   one mean per colour channel, over the whole batch
```

The per-channel means are exactly what image preprocessing subtracts before training (Day 9's third example).

## `keepdims`: reduce but keep the axis

**What it is.** `keepdims=True` keeps the reduced axis in the result with **length 1**, instead of removing it. `m.sum(axis=1)` has shape `(3,)`; `m.sum(axis=1, keepdims=True)` has shape `(3, 1)`.

**Why it exists.** You often reduce in order to combine the result with the original: divide each row by its total, subtract each row's mean. NumPy can combine a `(3, 4)` array with a `(3, 1)` column (each row gets its own value), but not with a `(3,)` row. Day 9 explains the rule; today, see it work.

**Example.** Turning each row of counts into proportions that add up to 1:

```python
counts = np.array([[1, 1, 2],
                   [3, 0, 1]])
totals = counts.sum(axis=1, keepdims=True)
totals                     # array([[4],
                           #        [4]])    shape (2, 1): one total per row, as a column
counts / totals            # array([[0.25, 0.25, 0.5 ],
                           #        [0.75, 0.  , 0.25]])
(counts / totals).sum(axis=1)   # array([1., 1.])   every row now adds up to 1
```

This "divide each row by its sum" is the last step of **softmax**, which turns scores into probabilities in every classifier (Phase 3).

!!! warning "Common mistake: forgetting `keepdims`"
    Without it, the row totals have shape `(2,)`, which NumPy can't line up with `(2, 3)`:

    ```python
    try:
        counts / counts.sum(axis=1)
    except ValueError as e:
        print("ValueError:", e)
    # ValueError: operands could not be broadcast together with shapes (2,3) (2,)
    ```

    Worse, on a *square* array the shapes do line up, the wrong way: each *column* gets divided by a row's total, with no error. **Fix:** `keepdims=True` whenever you reduce to combine with the original. Day 9 walks through this bug again with pictures.

## `any` and `all` along an axis

`any` and `all` reduce booleans, and take `axis` like the others. "Which samples have at least one negative feature?" is one line:

```python
readings = np.array([[0.5, 1.2, 0.3],
                     [0.7, -0.1, 0.9],
                     [-2.0, 0.4, -0.3]])
(readings < 0).any(axis=1)      # array([False,  True,  True])   per row: any negative?
(readings > 0).all(axis=0)      # array([False, False, False])   per column: all positive?
(readings < 0).sum(axis=1)      # array([0, 1, 2])               per row: how many negatives
```

---

**Part 9: Loops versus vectorised code, measured**

## Timing it yourself

**What it is.** The proof of Part 1, on a realistic calculation: the sum of squares of a million numbers, three ways.

**Why it matters.** You'll be tempted to write loops because they're familiar. Seeing the numbers on your own machine is what changes the habit.

```python
values = list(range(1_000_000))
varr = np.arange(1_000_000, dtype=np.float64)

def sum_squares_loop(xs):
    total = 0.0
    for x in xs:
        total += x * x
    return total

%timeit sum_squares_loop(values)      # Python loop over a list
%timeit sum_squares_loop(varr)        # Python loop over an array: slower still
%timeit (varr ** 2).sum()             # vectorised
%timeit varr @ varr                   # a dot product: Day 5 and Day 10
# 54 ms ± 3.95 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 106 ms ± 3.25 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 917 μs ± 28.4 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
# 42.2 μs ± 1.59 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
```

The output lines are from IPython on the machine this lesson was written on; yours will differ, and the browser's ▶ Run prints its own `[browser timing]` lines, slower again. On mine, the vectorised version was about 59× faster than the list loop, and the dot product (`@`, which computes the same sum of products in one call; you'll meet it on Day 5) about 1,300×. `(varr ** 2).sum()` is slower than `varr @ varr` because it allocates a whole temporary array of squares first, then sums it; `@` does both in one pass.

**When a loop is fine.** Loops over a handful of items (epochs, files, a few models) cost nothing. The rule is: **no Python loop over the elements of a big array**. And `np.vectorize` doesn't change that: it wraps a Python function so it accepts arrays, but still calls it once per element, at loop speed.

---

## Going further

These are real and worth knowing exist, but aren't needed yet:

- **The rest of the NumPy toolkit** (`where`, `clip`, sorting, `unique`, NaN handling, `allclose`): [Day 4b](day-04b-numpy-toolkit.md). Random numbers, saving arrays and statistics: [Day 4c](day-04c-random-io-stats.md).
- **Strides, reshape, transpose, adding axes, fancy indexing and stacking**: Day 8, and the [indexing guide](https://numpy.org/doc/stable/user/basics.indexing.html).
- **Broadcasting**, the full rules for combining different shapes: Day 9, and [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html).
- **Copies and views** in full, including `ravel` and when `reshape` has to copy: [Copies and views](https://numpy.org/doc/stable/user/basics.copies.html).
- **Type promotion rules** in detail (NEP 50, the "weak Python scalars" rule): [Data type promotion](https://numpy.org/doc/stable/reference/arrays.promotion.html).
- **The `out=` argument and ufunc methods** (`np.add.reduce`, `np.add.accumulate`, `np.add.at`): [Universal functions](https://numpy.org/doc/stable/user/basics.ufuncs.html).
- **Structured arrays and memory-mapped files** (`np.memmap`, arrays bigger than RAM): [Structured arrays](https://numpy.org/doc/stable/user/basics.rec.html) and [numpy.memmap](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html).
- **`bfloat16` and 8-bit float types** in NumPy, via the `ml_dtypes` package: [ml_dtypes on GitHub](https://github.com/jax-ml/ml_dtypes).
- **Making loops fast when you can't vectorise**, with Numba's compiler: [Numba](https://numba.pydata.org/).
- **Complex numbers and `object` arrays** (`complex128`, and arrays that hold Python objects, losing NumPy's speed): [Data types](https://numpy.org/doc/stable/user/basics.types.html).
- **Print options** (`np.set_printoptions` for precision and line width): [numpy.set_printoptions](https://numpy.org/doc/stable/reference/generated/numpy.set_printoptions.html).

---

## Exercises

**Warm-ups**

1. **Build and inspect.** Without running anything, write down the `shape`, `ndim`, `size`, `dtype` and `nbytes` of each array. Then check. (a) `np.zeros((4, 5))` (b) `np.arange(6, dtype=np.int8)` (c) `np.ones((2, 3, 4), dtype=np.float32)` (d) `np.array([[True, False]])` (e) `np.linspace(0, 1, 11)`.
2. **Crop and count.** Make a fake grayscale image with `rng = np.random.default_rng(7)` and `img = rng.integers(0, 256, size=(28, 28), dtype=np.uint8)`. Crop the centre 14 × 14 with one slice, confirm its shape, and count how many of its pixels are brighter than 200. Then show the crop is a view.

**Core**

3. **Scores to accuracy.** A classifier gave `scores = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1], [0.3, 0.3, 0.4], [0.2, 0.2, 0.6], [0.5, 0.4, 0.1]])` for 5 images × 3 classes, and the true labels are `labels = np.array([1, 0, 1, 2, 1])`. In three lines with no loop: the predicted class of each image, a mask of which predictions were right, and the accuracy as a fraction. Then print the scores of the images it got wrong.
4. **Preprocess an image for a model.** Make `photo = rng.integers(0, 256, size=(4, 4, 3), dtype=np.uint8)` with `rng = np.random.default_rng(3)`. (a) Convert it to `float32` in 0–1 and show its dtype, its min and max, and how many bytes it now takes. (b) Compute the mean of each colour channel. (c) Min-max normalise the red channel alone to 0–1: `(r − r.min()) / (r.max() − r.min())`. (d) Show the original `photo` is unchanged.
5. **Model memory.** A model has 7 billion weights. Using `np.dtype(...).itemsize`, compute its size in GB (10⁹ bytes) in `float32`, `float16` and `int8`, and in 4-bit (half a byte per weight). Print one line each with an f-string.

**Predict the output**

6. Write down what each `print` shows, then run it.

    ```python
    import numpy as np
    a = np.array([10, 20, 30, 40], dtype=np.uint8)
    b = a[1:3]
    b[0] = 255
    print(a)
    c = a[a > 25]
    c[0] = 0
    print(a)
    print(a + 10)
    print((a / 2).dtype)
    m = np.arange(6).reshape(2, 3)
    print(m[:, 1].shape, m[:, 1:2].shape)
    print(m.sum(axis=0), m.max(axis=1))
    ```

**Find and fix the bug**

7. This function should brighten a `uint8` image by `amount`, cap it at 255, and leave the caller's image alone. It does none of those three things correctly. Find all three bugs, explain each, and fix it.

    ```python
    def brighten(img, amount):
        bright = img
        bright += amount
        bright[bright > 255] = 255
        return bright

    photo = np.array([[100, 200], [240, 30]], dtype=np.uint8)
    out = brighten(photo, 50)
    print(out)
    print(photo)
    # [[150 250]
    #  [ 34  80]]     240 + 50 should be capped at 255, not 34
    # [[150 250]
    #  [ 34  80]]     and photo changed too
    ```

**Challenge**

8. **Standardise a batch per channel.** Make a batch of 6 RGB images, `batch = rng.integers(0, 256, size=(6, 8, 8, 3), dtype=np.uint8)` with `rng = np.random.default_rng(11)`. Without any Python loop: (a) convert to `float32` in 0–1; (b) compute the mean and standard deviation of each channel over the whole batch, keeping dimensions so the result has shape `(1, 1, 1, 3)`; (c) standardise: subtract the mean and divide by the standard deviation; (d) prove it worked by showing each channel of the result now has mean within 10⁻⁶ of 0 and standard deviation within 10⁻⁶ of 1; (e) find which image in the batch has the brightest average pixel. Explain every `axis` you used.

??? success "Solutions"
    **1. Build and inspect.**

    ```python
    import numpy as np

    for arr in [np.zeros((4, 5)), np.arange(6, dtype=np.int8), np.ones((2, 3, 4), dtype=np.float32),
                np.array([[True, False]]), np.linspace(0, 1, 11)]:
        print(arr.shape, arr.ndim, arr.size, arr.dtype, arr.nbytes)
    # (4, 5) 2 20 float64 160
    # (6,) 1 6 int8 6
    # (2, 3, 4) 3 24 float32 96
    # (1, 2) 2 2 bool 2
    # (11,) 1 11 float64 88
    ```

    `nbytes` is always `size × itemsize`: 20 × 8, 6 × 1, 24 × 4, 2 × 1, 11 × 8. `zeros`, `ones` and `linspace` default to `float64`; (d) is 2-D because of the double brackets; (e) has 11 points because `linspace` includes both ends: 0, 0.1, …, 1.

    **2. Crop and count.**

    ```python
    import numpy as np

    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, size=(28, 28), dtype=np.uint8)
    centre = img[7:21, 7:21]
    print(centre.shape)                    # (14, 14)
    print((centre > 200).sum())            # 34
    print(np.shares_memory(img, centre))   # True
    ```

    28 − 14 = 14, so the crop starts 7 in from each edge and stops 7 before the end: `7:21`. `centre > 200` is a 14 × 14 boolean mask and `.sum()` counts its `True` values. A basic slice is a view, so `centre` reads `img`'s buffer: changing `centre` would change `img`.

    **3. Scores to accuracy.**

    ```python
    import numpy as np

    scores = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1], [0.3, 0.3, 0.4],
                       [0.2, 0.2, 0.6], [0.5, 0.4, 0.1]])
    labels = np.array([1, 0, 1, 2, 1])
    pred = scores.argmax(axis=1)
    correct = pred == labels
    print(pred, correct, correct.mean())
    print(scores[~correct])
    # [1 0 2 2 0] [ True  True False  True False] 0.6
    # [[0.3 0.3 0.4]
    #  [0.5 0.4 0.1]]
    ```

    `argmax(axis=1)` makes axis 1 (classes) disappear, leaving one class per image. `pred == labels` compares element by element. The mean of a boolean mask is the fraction of `True`s: 3 right out of 5. `~correct` flips the mask, and using it on axis 0 keeps the rows (images) that were wrong.

    **4. Preprocess an image for a model.**

    ```python
    import numpy as np

    rng = np.random.default_rng(3)
    photo = rng.integers(0, 256, size=(4, 4, 3), dtype=np.uint8)
    original = photo.copy()                        # to check (d) later

    x = photo.astype(np.float32) / 255             # (a)
    print(x.dtype, x.min(), x.max(), x.nbytes, photo.nbytes)
    print(x.mean(axis=(0, 1)))                     # (b)
    r = x[:, :, 0]                                 # (c) the red channel, shape (4, 4)
    r_norm = (r - r.min()) / (r.max() - r.min())
    print(r_norm.min(), r_norm.max())
    print(np.array_equal(photo, original))         # (d)
    # float32 0.019607844 0.972549 192 48
    # [0.42230392 0.40441182 0.45171568]
    # 0.0 1.0
    # True
    ```

    `astype(np.float32)` before dividing keeps the result `float32` (4 bytes × 48 values = 192, four times the `uint8` image's 48). `axis=(0, 1)` removes height and width and leaves one mean per channel. The red-channel slice `x[:, :, 0]` is a view of `x`, but the min-max formula builds new arrays and doesn't write into it. `photo` is unchanged because `astype` copies.

    **5. Model memory.**

    ```python
    import numpy as np

    n_weights = 7e9
    for name, nbytes in [("float32", np.dtype(np.float32).itemsize),
                         ("float16", np.dtype(np.float16).itemsize),
                         ("int8", np.dtype(np.int8).itemsize),
                         ("4-bit", 0.5)]:
        print(f"{name:>8}: {n_weights * nbytes / 1e9:5.1f} GB")
    #  float32:  28.0 GB
    #  float16:  14.0 GB
    #     int8:   7.0 GB
    #    4-bit:   3.5 GB
    ```

    Memory is weights × bytes per weight. `:>8` right-aligns the name in 8 characters and `:5.1f` prints one decimal in 5 characters (Day 2a). 4-bit has no NumPy dtype; libraries pack two weights into each byte. This is why a 7-billion-parameter model needs a big GPU in `float32` but runs on a laptop, or a high-end phone, at 4 bits.

    **6. Predict the output.**

    ```python
    import numpy as np
    a = np.array([10, 20, 30, 40], dtype=np.uint8)
    b = a[1:3]
    b[0] = 255
    print(a)
    c = a[a > 25]
    c[0] = 0
    print(a)
    print(a + 10)
    print((a / 2).dtype)
    m = np.arange(6).reshape(2, 3)
    print(m[:, 1].shape, m[:, 1:2].shape)
    print(m.sum(axis=0), m.max(axis=1))
    # [ 10 255  30  40]
    # [ 10 255  30  40]
    # [20  9 40 50]
    # float64
    # (2,) (2, 1)
    # [3 5 7] [2 5]
    ```

    `b` is a slice, so `b[0] = 255` writes `a[1]`. `c` is a mask result, a copy, so `c[0] = 0` leaves `a` alone. `a + 10` stays `uint8`, and 255 + 10 = 265 wraps to 9. `/` always gives a float, and a `uint8` array with a Python number gives `float64`. An integer index removes axis 1, a slice keeps it with length 1. Column sums are 0 + 3, 1 + 4, 2 + 5; row maxima are 2 and 5.

    **7. Find and fix the bug.**

    ```python
    import numpy as np

    def brighten(img, amount):
        bright = img.astype(np.int16)          # a copy, in a type that holds 0 + 255 + amount
        bright += amount
        bright[bright > 255] = 255             # now the test can be true
        return bright.astype(np.uint8)

    photo = np.array([[100, 200], [240, 30]], dtype=np.uint8)
    out = brighten(photo, 50)
    print(out)
    print(photo)
    # [[150 250]
    #  [255  80]]
    # [[100 200]
    #  [240  30]]
    ```

    Bug 1: `bright = img` copies nothing; it's a second name for the same array (Day 2), so `+=` modified the caller's `photo`. Bug 2: in `uint8`, 240 + 50 = 290 wraps to 34 before any check runs. Bug 3: as a result, `bright > 255` can never be true for a `uint8` array, whose maximum is 255, so the cap did nothing. `astype(np.int16)` fixes all three: it makes a new array, and `int16` holds 290, so the mask finds it and caps it. Converting back to `uint8` at the end is safe because every value is now 0–255.

    **8. Standardise a batch per channel.**

    ```python
    import numpy as np

    rng = np.random.default_rng(11)
    batch = rng.integers(0, 256, size=(6, 8, 8, 3), dtype=np.uint8)

    x = batch.astype(np.float32) / 255                       # (a)
    mean = x.mean(axis=(0, 1, 2), keepdims=True)             # (b)
    std = x.std(axis=(0, 1, 2), keepdims=True)
    print(mean.shape, std.shape)
    z = (x - mean) / std                                     # (c)
    print(np.abs(z.mean(axis=(0, 1, 2))).max() < 1e-6)      # (d)
    print(np.abs(z.std(axis=(0, 1, 2)) - 1).max() < 1e-6)
    brightness = x.mean(axis=(1, 2, 3))                      # (e)
    print(brightness.argmax(), brightness.shape)
    # (1, 1, 1, 3) (1, 1, 1, 3)
    # True
    # True
    # 0 (6,)
    ```

    The axes of `x` are (image, height, width, channel). A per-channel statistic must combine every image and every pixel but keep the channels apart, so it reduces axes 0, 1 and 2 and keeps axis 3. `keepdims=True` leaves those as length-1 axes, `(1, 1, 1, 3)`, which NumPy can line up with `(6, 8, 8, 3)` so each channel gets its own mean and standard deviation (Day 9's rule). For (d), `np.abs(...).max() < 1e-6` checks the worst channel is within tolerance, instead of testing floats with `==`. For (e), the brightness of one image combines its height, width and channels, so it reduces axes 1, 2 and 3 and keeps axis 0, one value per image, and `argmax` gives the brightest one's index.

## Resources

- [NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html): the official guide, with diagrams of arrays, indexing and axes.
- [NumPy user guide: fundamentals](https://numpy.org/doc/stable/user/basics.html): array creation, indexing, data types, copies and views, and ufuncs, one page each.
- [Python Data Science Handbook, chapter 2: Introduction to NumPy](https://jakevdp.github.io/PythonDataScienceHandbook/02.00-introduction-to-numpy.html): free online; sections 2.1–2.6 cover today in a different voice, including why lists are slow.
- [Jay Alammar: A Visual Intro to NumPy and Data Representation](https://jalammar.github.io/visual-numpy/): shapes, images and text as arrays, drawn.
- [From Python to NumPy (free book)](https://www.labri.fr/perso/nrougier/from-python-to-numpy/): vectorisation in depth, including how views and strides work.
- [100 NumPy exercises](https://github.com/rougier/numpy-100): do the first 25 over the next two weeks.
- [CS231n Python/NumPy tutorial](https://cs231n.github.io/python-numpy-tutorial/): Stanford's quick tour, used by many ML courses.

## Checklist

- [ ] I can explain, with real byte counts, why an array of numbers is smaller and faster than a list of them.
- [ ] I can describe an `ndarray` as a buffer plus `dtype`, `shape` and `strides`, and I read `shape` and `dtype` by habit.
- [ ] I can create arrays with `array`, `zeros`, `ones`, `full`, `empty`, `arange`, `linspace`, `eye` and `*_like`, and I use `linspace` for float ranges.
- [ ] I can choose a dtype, convert with `astype`, and predict overflow, float32 rounding and promotion.
- [ ] I can index and slice 1-D, 2-D and 3-D arrays and predict the resulting shape.
- [ ] I can pick and reorder with integer-array indexing, I know `M[rows, cols]` picks pairs, and I know the result is a copy.
- [ ] I can filter and assign with masks, with parentheses and `&`, `|`, `~`.
- [ ] I know slices are views and masks are copies, and I check with `np.shares_memory`.
- [ ] I write element-wise maths and ufuncs instead of loops, and I know `np.maximum` from `np.max`.
- [ ] I can reduce along any axis of a 3-D array, with or without `keepdims`.
- [ ] I've timed a loop against vectorised code on my own machine.

**Next:** [Day 4b: The NumPy toolkit](day-04b-numpy-toolkit.md)
