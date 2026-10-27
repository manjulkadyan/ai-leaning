# Day 9: Broadcasting

**Phase 1 · Week 2 · ~120 minutes**

## Objectives

- [ ] Explain what broadcasting is, and write the Kotlin loop that a broadcast replaces.
- [ ] State the broadcasting rules precisely and use them to work out a result shape, or spot an error, by hand and with `np.broadcast_shapes`.
- [ ] Explain why broadcasting does not copy memory (stride 0), why a broadcast view is read-only, and which side of an in-place operation may stretch.
- [ ] Use broadcasting to standardise features, add a bias to a batch and normalise an image per channel, in both HWC and CHW layouts.
- [ ] Line a 1-D array up with any axis of a bigger array (`mean[:, None, None]`, `s[:, None, None, None]`).
- [ ] Use broadcasting in comparisons (one-hot labels), in assignment and in `np.where`, and build a pairwise distance table.
- [ ] Recognise and fix the classic broadcasting bugs: `(n,)` vs `(n, 1)`, a missing `keepdims=True`, a `uint8` wrap, a zero standard deviation and a huge intermediate.

**Prerequisites.**

- [Day 3b](day-03b-errors-files-json.md): `try` / `except ... as e`, and `assert condition, message`.
- [Day 4](day-04-numpy-arrays.md): arrays, `dtype` and `astype`, `uint8` overflow, `strides`, element-wise arithmetic, reductions with `axis=` (including a tuple of axes), and the first look at `keepdims=True`.
- [Day 4b](day-04b-numpy-toolkit.md): `np.where`, `np.isnan` and `np.nanmean`, `np.clip`, `np.allclose` and `np.errstate`.
- [Day 4c](day-04c-random-io-stats.md): `np.round`, and `rng.normal` / `rng.integers` from `np.random.default_rng`.
- [Day 5](day-05-vectors.md): the norm and the distance between two points, the neuron \( \mathbf{w} \cdot \mathbf{x} + b \), and `@` between two 1-D vectors (the dot product).
- [Day 8](day-08-shapes-indexing.md): `reshape`, adding an axis with `None` / `np.newaxis` / `np.expand_dims`, `squeeze`, `transpose`, NHWC vs NCHW, fancy indexing and `np.stack`.

!!! info "How to read the code on this page"
    Every Python block has a **▶ Run** button, and blocks share their variables from top to bottom like notebook cells, so run them in order. A bare expression on its own line shows its `repr()`, exactly as the `# comment` beside or below it says. Lines that fail on purpose are wrapped in `try` / `except` so the rest of the block still runs, and the comment shows the real error message. The outputs on this page come from 64-bit NumPy on a laptop; in the browser NumPy's default integer is `int32`, so where the page shows `np.int64(56)`, ▶ Run shows `np.int32(56)` ([How lessons work](../../how-lessons-work.md) explains why).

---

## The big idea

On Day 4 you added two arrays of the *same* shape element by element, and saw that `np.array([1, 2, 3]) + np.array([1, 2, 3, 4])` fails with a message that mentions "broadcast". But ML code constantly combines arrays of *different* shapes: subtract one average per column from a whole table, add one bias per neuron to a whole batch, divide every pixel of an image by one number per colour channel.

**Broadcasting** is NumPy's rule for doing that without writing a loop and without copying data: when two shapes differ in a compatible way, NumPy *treats* the smaller array as if it were repeated along the missing or size-1 axes until the shapes match, then does the normal element-by-element operation.

**Analogy.** A teacher writes one correction on the whiteboard ("add 2 bonus points to question 3") instead of writing it on every student's paper. Everyone applies the same line to their own paper. The correction exists once; it is *applied* to every row. That's a broadcast: one row of numbers, applied to every row of a table.

PyTorch, TensorFlow and JAX follow the same rules, and so does LiteRT in the operations that support broadcasting, so this lesson pays off in every framework you'll touch.

## Step 1: a scalar and an array

**What it is.** A **scalar** is a single number. Combining a scalar with an array applies the scalar to every element. You've been doing this since Day 4; it is the most basic broadcast.

**Why it exists.** Scaling pixels by `1 / 255`, shifting by a constant, comparing every element with a threshold: one number, a whole array.

**Kotlin comparison.** In Kotlin you'd write `a.map { it + 10 }` or a `for` loop. In NumPy the scalar goes straight into the expression:

```python
import numpy as np

a = np.array([1, 2, 3])
a + 10          # array([11, 12, 13])
a * 2           # array([2, 4, 6])
a > 1           # array([False,  True,  True])
```

**Underneath.** NumPy treats `10` as an array of shape `()`, an array with no axes at all, and the rules below stretch it to `(3,)`: `[10, 10, 10]`. Nothing is built; the C loop reads the same `10` for every element.

**How it goes wrong: the scalar takes the array's dtype.** A Python integer that fits the array's dtype is converted *to* that dtype (Day 4's type promotion). On a `uint8` image, subtracting 128 to centre the pixels wraps round instead of going negative:

```python
px = np.array([[200, 100]], dtype=np.uint8)
px - 128                        # array([[ 72, 228]], dtype=uint8)    100 − 128 wrapped to 228
px.astype(np.int16) - 128       # array([[ 72, -28]], dtype=int16)
px / 255.0 - 0.5                # array([[ 0.28431373, -0.10784314]])
```

No error and no warning: 100 − 128 = −28, and −28 + 256 = 228 (Day 4's overflow section). **Fix:** convert to a wider integer or to float before the arithmetic, which ML preprocessing does anyway.

## Step 2: a vector and a matrix

**What it is.** Combining a matrix of shape `(4, 3)` (4 rows, 3 columns) with a vector of shape `(3,)` applies the vector to **every row**.

**Why it exists.** "One value per column, applied to every row" is the shape of most per-feature operations: subtract each feature's mean, multiply each feature by a weight, add one bias per output.

**Kotlin comparison.** Here is the loop you'd write in Kotlin for a 4 × 3 matrix `m` and a 3-element vector `v`:

```kotlin
val r = Array(4) { IntArray(3) }
for (i in 0 until 4) {
    for (j in 0 until 3) {
        r[i][j] = m[i][j] + v[j]   // v has no i: the same v[j] for every row
    }
}
```

Look at the index on `v`: it has `j` but no `i`. That missing index *is* the broadcast. NumPy writes the same loop for you, in C, from the shapes alone. Two variations you'll meet below: `m[i][j] + c[i]` (one value per *row*, a column of shape `(4, 1)`) and `c[i] + v[j]` (every row value with every column value, an outer grid).

**The formula, one symbol at a time.** If \( M \) is the matrix and \( \mathbf{v} \) the vector, the result \( R \) is

\[
R_{ij} = M_{ij} + v_j
\]

\( i \) is the row number and \( j \) the column number. \( M_{ij} \) is the matrix element in row \( i \), column \( j \). The vector has only a column index \( j \), so the **same** \( v_j \) is used for every row \( i \), the maths version of the missing `i` in the Kotlin loop.

**By hand first.** With rows of `M` equal to 0s, 10s, 20s and 30s, and \( \mathbf{v} = [1, 2, 3] \): row 2, column 3 (counting from 1) is \( R_{23} = M_{23} + v_3 = 10 + 3 = 13 \). In code, which counts from 0, that's `R[1, 2]`.

```python
M = np.array([[ 0,  0,  0],
              [10, 10, 10],
              [20, 20, 20],
              [30, 30, 30]])
v = np.array([1, 2, 3])

R = M + v
R
# array([[ 1,  2,  3],
#        [11, 12, 13],
#        [21, 22, 23],
#        [31, 32, 33]])
R.shape         # (4, 3)
R[1, 2]         # np.int64(13)     in the browser: np.int32(13)
```

`R[1, 2]` is one element, so it comes back as a **NumPy scalar**, which displays as `np.int64(13)` or, for floats, `np.float64(...)` (Day 4). The code reproduces the hand result.

<figure class="diagram" markdown>
![Animation: a 4 by 3 matrix M with rows 0, 10, 20, 30 sits next to a row vector v = [1, 2, 3]; dashed copies of v slide down to fill 4 rows, then the 4 by 3 result fills in row by row with 1 2 3, 11 12 13, 21 22 23, 31 32 33](../../assets/img/phase-1/day09-broadcast-rows.svg)
<figcaption>Animated: the (3,) row behaves as if copied down all 4 rows, then the add is element by element.</figcaption>
</figure>

The underneath and goes-wrong parts of this step are the rules themselves, next.

## The rules, precisely

**What they are.** NumPy decides whether two shapes are compatible like this:

1. **Align the shapes from the right.** Write them one above the other, right-justified. If one shape has fewer dimensions, pad it on the **left** with 1s (missing leading dimensions count as 1).
2. **Compare each pair of dimensions.** For every column, the two sizes must be **equal**, or **one of them must be 1**. A size-1 dimension is stretched to match the other. If any pair fails, you get a `ValueError`.

The result shape takes, in each column, the larger of the two sizes. More than two arrays (as in `np.where(cond, a, b)`) follow the same rules, all at once.

**Why from the right?** Because the last axis is the one that varies fastest in memory (Day 4's row-major order) and, in ML data, it is usually the "feature" or "channel" axis. Aligning from the right makes `(rows, features)` combine with `(features,)` and `(H, W, 3)` combine with `(3,)` without any reshaping.

<figure class="diagram" markdown>
![Animated step diagram: for (4, 3) plus (3,), the missing axis is treated as 1, axis -1 compares 3 with 3 and axis -2 compares 4 with 1, result (4, 3); for (4, 3) plus (4,), axis -1 compares 3 with 4 and fails with ValueError; for (4, 1) plus (1, 3), both size-1 axes stretch, result (4, 3)](../../assets/img/phase-1/day09-rule-check.svg)
<figcaption>Animated: right-align the shapes, then check each axis from the right. One failing pair means an error.</figcaption>
</figure>

**By hand: the case that surprises everyone.**

```text
(4, 3)
   (4,)  → pad left →  (1, 4)

axis -1:  3 vs 4   neither is 1  →  ValueError
```

A vector of length 4 does **not** line up with the 4 rows. It lines up with the *last* axis, the 3 columns. NumPy never guesses which axis you meant; it only ever aligns from the right.

```python
try:
    M + np.array([1, 2, 3, 4])
except ValueError as e:
    print("ValueError:", e)
# ValueError: operands could not be broadcast together with shapes (4,3) (4,)
```

**Recognise it:** the message lists both shapes. Right-align them in your head and find the column that fails. **Fix:** to add one value per **row**, make the vector a column of shape `(4, 1)` (Day 8's `None`):

```python
per_row = np.array([1, 2, 3, 4])[:, None]
per_row.shape           # (4, 1)
M + per_row
# array([[ 1,  1,  1],
#        [12, 12, 12],
#        [23, 23, 23],
#        [34, 34, 34]])
```

Now the alignment is `(4, 3)` against `(4, 1)`: 3 vs 1 stretches, 4 vs 4 matches. In the Kotlin loop, this is `m[i][j] + c[i]`.

### A table to check yourself against

You can ask NumPy to apply the rules without making any arrays, using `np.broadcast_shapes(shape_a, shape_b)`: it returns the result shape, or raises the error. Every row below is real output:

| Shape A | Shape B | Result | Why |
|---|---|---|---|
| `(4, 3)` | `(3,)` | `(4, 3)` | B becomes `(1, 3)`, the 1 stretches |
| `(4, 3)` | `(4,)` | **error** | last axis: 3 vs 4 |
| `(4, 3)` | `(4, 1)` | `(4, 3)` | one value per row |
| `(4, 1)` | `(1, 3)` | `(4, 3)` | **both** stretch: an "outer" combination |
| `(2, 3, 4)` | `(3, 4)` | `(2, 3, 4)` | B repeated for each of the 2 blocks |
| `(2, 3, 4)` | `(3, 1)` | `(2, 3, 4)` | B becomes `(1, 3, 1)` |
| `(8, 1, 6, 1)` | `(7, 1, 5)` | `(8, 7, 6, 5)` | every axis has a 1 or a match |
| `(3,)` | `(4,)` | **error** | 3 vs 4 |
| `(5,)` | `(5, 1)` | `(5, 5)` | no error, but probably a bug (Bug 1 below) |
| `(256, 256, 3)` | `(3,)` | `(256, 256, 3)` | one value per colour channel |
| `(3, 224, 224)` | `(3,)` | **error** | channels first: 224 vs 3 (Real use 4) |
| `(2, 1)` | `(8, 4, 3)` | **error** | axis -2: 2 vs 4 |

```python
np.broadcast_shapes((8, 1, 6, 1), (7, 1, 5))      # (8, 7, 6, 5)
np.broadcast_shapes((2, 3, 4), (3, 1))            # (2, 3, 4)
try:
    np.broadcast_shapes((4, 3), (4,))
except ValueError as e:
    print("ValueError:", e)
# ValueError: shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (4, 3) and arg 1 with shape (4,).
```

### Both sides stretch: the outer grid

The `(4, 1)` + `(1, 3)` row of the table deserves its own example. A column plus a row gives a full grid, every row value combined with every column value, the `c[i] + v[j]` loop:

```python
col = np.array([[0], [10], [20], [30]])   # (4, 1)
row = np.array([[1, 2, 3]])               # (1, 3)
col + row
# array([[ 1,  2,  3],
#        [11, 12, 13],
#        [21, 22, 23],
#        [31, 32, 33]])
```

Neither input has 12 numbers; the result does. This "outer" pattern is how you build times tables, distance tables (Real use 6) and one-hot labels (Real use 5).

## It doesn't actually copy anything

**What it is.** The animations show copies, but that's only the mental model. NumPy never builds the stretched array. Day 4 showed that every array has **strides**, the number of bytes to jump to move one step along each axis, and Day 8 used them to explain why `reshape` is free. Broadcasting uses the same trick: a stretched axis gets a stride of **0**, "to go to the next row, move 0 bytes", so the same memory is read again.

**Why it matters.** `X - mean` on a million-row table does not allocate a million-row copy of `mean`. The loop runs in C and reads `mean` over and over. Only the **result** is a new full-size array.

**Example.** `np.broadcast_to(array, shape)` returns the stretched array explicitly, as a **view** (an array that reads another array's memory, Day 4):

```python
v = np.array([1.0, 2.0, 3.0])
big = np.broadcast_to(v, (1_000_000, 3))

big.shape                  # (1000000, 3)
big.strides                # (0, 8)    0 bytes to the next row, 8 bytes to the next column
v.nbytes, big.nbytes       # (24, 24000000)   nbytes reports the logical size...
np.shares_memory(big, v)   # True             ...but it is v's 24 bytes underneath
big.flags.writeable        # False
```

`big.flags` is the array's set of yes/no properties; `flags.writeable` says whether you may assign into it. A broadcast view is read-only, because every one of its million rows is the *same* 24 bytes.

**How it goes wrong: writing into a broadcast view.** If writing were allowed, `big[0, 0] = 99` would change every row at once, and `v` too. NumPy refuses:

```python
try:
    big[0, 0] = 99.0
except ValueError as e:
    print("ValueError:", e)
# ValueError: assignment destination is read-only

own = np.broadcast_to(v, (2, 3)).copy()    # a real (2, 3) array with its own memory
own[0, 0] = 99.0
own
# array([[99.,  2.,  3.],
#        [ 1.,  2.,  3.]])
v                          # array([1., 2., 3.])   unchanged
```

**Fix:** call `.copy()` when you need a stretched array you can write into.

### When you really want the copy: `np.tile` and `np.repeat`

`np.tile(a, reps)` builds a real array by repeating `a` `reps` times along each axis; `np.repeat(a, n, axis=k)` repeats each slice along axis `k`. Both allocate and copy:

```python
np.tile(v, (2, 1))                       # array([[1., 2., 3.],
                                         #        [1., 2., 3.]])
np.repeat(v[None, :], 2, axis=0)         # array([[1., 2., 3.],
                                         #        [1., 2., 3.]])
np.tile(v, (1_000_000, 1)).nbytes        # 24000000   24 MB of real memory this time
```

Use them only when you need to modify the copies independently or a library wants a real array. For arithmetic, let broadcasting do the stretching.

### In-place operations can't stretch the left side

An in-place operation such as `b += a` writes the result into `b`'s existing memory. So the broadcast result must have `b`'s shape. The right-hand side may stretch; the left-hand side may not:

```python
a = np.array([1.0, 2.0, 3.0])     # (3,)
b = np.zeros((2, 3))              # (2, 3)

b += a                            # fine: (2, 3) += (3,) gives (2, 3), b's own shape
b
# array([[1., 2., 3.],
#        [1., 2., 3.]])

try:
    a += b                        # the result would be (2, 3), but a only has room for (3,)
except ValueError as e:
    print("ValueError:", e)
# ValueError: non-broadcastable output operand with shape (3,) doesn't match the broadcast shape (2,3)
```

**Fix:** write `a = a + b` if you want a new `(2, 3)` array, or reduce `b` first (`a += b.sum(axis=0)`) if you meant to accumulate into `a`.

### Broadcasting in assignment

Assignment broadcasts too: the value on the right is stretched to the shape of the selection on the left.

```python
G = np.zeros((2, 3))
G[:] = np.array([7.0, 8.0, 9.0])      # (3,) stretched into the (2, 3) selection
G[:, 0] = -1.0                        # a scalar into a (2,) column selection
G
# array([[-1.,  8.,  9.],
#        [-1.,  8.,  9.]])

try:
    G[:] = np.array([1.0, 2.0])       # (2,) can't stretch to (2, 3)
except ValueError as e:
    print("ValueError:", e)
# ValueError: could not broadcast input array from shape (2,) into shape (2,3)
```

The error has the same cause as before: right-aligned, 2 vs 3 fails. To fill one value per row, assign a column: `G[:] = np.array([1.0, 2.0])[:, None]`.

## Real use 1: standardising features

**What it is.** Most ML models train better when each feature (column) has mean 0 and standard deviation 1, so that "area in m²" (around 100) doesn't drown out "bedrooms" (around 3). This is called **standardisation**:

\[
z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}
\]

One symbol at a time:

- \( x_{ij} \): the value of feature \( j \) for sample (row) \( i \).
- \( \mu_j \) ("mu j"): the **mean** of column \( j \), the average over all rows.
- \( \sigma_j \) ("sigma j"): the **standard deviation** of column \( j \), the square root of the average squared distance from the mean. NumPy's `std` divides by \( n \), not \( n - 1 \) (Day 4's `ddof` box).
- \( z_{ij} \): the standardised value, "how many standard deviations above average".

\( \mu \) and \( \sigma \) have only a column index: one number per feature, broadcast over every row, exactly like \( v_j \) in Step 2.

**By hand first**, for the area column of the table below (120, 70, 160, 110):

- mean: \( \mu = (120 + 70 + 160 + 110) / 4 = 115 \);
- distances from the mean: 5, −45, 45, −5; squared: 25, 2025, 2025, 25; their average: \( 4100 / 4 = 1025 \);
- standard deviation: \( \sigma = \sqrt{1025} = 32.016 \);
- house 2's area: \( z = (70 - 115) / 32.016 = -1.406 \).

Now the code. The first line changes how NumPy *prints* arrays for the rest of the page: `precision=3` shows at most 3 digits after the point, and `suppress=True` avoids scientific notation such as `1.2e-17` for tiny values. It doesn't change any stored value, and scalars such as `np.float64(...)` still print in full.

```python
np.set_printoptions(precision=3, suppress=True)

# rows = houses, columns = bedrooms, area_m2, age_years   → shape (4, 3)
X = np.array([[3, 120, 15],
              [2,  70, 40],
              [4, 160,  5],
              [3, 110, 20]], dtype=float)

mu = X.mean(axis=0)          # axis 0 disappears: one mean per column, shape (3,)
sigma = X.std(axis=0)
mu
# array([  3., 115.,  20.])
sigma
# array([ 0.707, 32.016, 12.748])

Z = (X - mu) / sigma         # (4, 3) with (3,), then (4, 3) with (3,)
Z
# array([[ 0.   ,  0.156, -0.392],
#        [-1.414, -1.406,  1.569],
#        [ 1.414,  1.406, -1.177],
#        [ 0.   , -0.156,  0.   ]])
Z[1, 1]                      # np.float64(-1.4055638569974547)   the hand value, in full
Z.mean(axis=0)               # array([0., 0., 0.])
Z.std(axis=0)                # array([1., 1., 1.])
```

**Underneath.** `X.mean(axis=0)` removes axis 0 and leaves `(3,)`, which aligns from the right with the 3 columns. No reshaping is needed: the reduced axis was the *leading* one, so the result already lines up (Bug 2 below says when that stops being true).

**How it goes wrong: a constant column.** If a feature has the same value in every row (a sensor that never changed, a category that's always "free"), its standard deviation is 0, and \( 0 / 0 \) is `nan` (Day 4b):

```python
K = np.array([[1.0, 5.0],
              [1.0, 7.0]])            # column 0 never changes
k_sigma = K.std(axis=0)
k_sigma                               # array([0., 1.])
(K - K.mean(axis=0)) / k_sigma
# RuntimeWarning: invalid value encountered in divide
# array([[nan, -1.],
#        [nan,  1.]])
```

A whole column of `nan`, and every later calculation that touches it becomes `nan` too. **Recognise it** by that `RuntimeWarning` (Day 4b's `errstate` section). **Fix:** drop the column, or divide by 1 where the deviation is 0, so the column becomes all zeros:

```python
safe = np.where(k_sigma == 0, 1.0, k_sigma)    # (2,): 1 where sigma is 0
(K - K.mean(axis=0)) / safe
# array([[ 0., -1.],
#        [ 0.,  1.]])
```

**And the habit that matters in real projects:** compute `mu` and `sigma` on the **training** data only, save them, and apply those same numbers to test data and to live inputs in your app. Recomputing them on each new batch gives every batch a different scale.

## Real use 2: adding a bias to a batch

On Day 5 a neuron computed \( \mathbf{w} \cdot \mathbf{x} + b \). A layer of 4 neurons has 4 biases, one per neuron. Run it on a **batch** of 32 inputs at once and the layer's raw outputs have shape `(32, 4)`: one row per input, one column per neuron. Adding the bias vector of shape `(4,)` broadcasts it across all 32 rows.

The pretend outputs below come from `rng.normal` (Day 4b). `.round(2)`, the method form of `np.round(..., 2)`, keeps them short.

```python
rng = np.random.default_rng(0)
out = rng.normal(size=(32, 4)).round(2)   # pretend layer outputs, before the bias
b = np.array([0.1, -0.2, 0.0, 0.5])       # one bias per neuron

y = out + b
out.shape, b.shape, y.shape    # ((32, 4), (4,), (32, 4))
out[0]                         # array([ 0.13, -0.13,  0.64,  0.1 ])
y[0]                           # array([ 0.23, -0.33,  0.64,  0.6 ])
```

Check the first row by hand: 0.13 + 0.1 = 0.23, −0.13 − 0.2 = −0.33, 0.64 + 0 = 0.64, 0.1 + 0.5 = 0.6.

This is what PyTorch's `nn.Linear` does: it computes `x @ W.T + b`, a matrix multiply (Day 10) followed by exactly this bias broadcast. You'll build it yourself on Day 12.

## Real use 3: normalising an image per channel (HWC)

Image models trained on the ImageNet dataset expect every pixel to be normalised per colour channel with these widely used statistics:

- mean = `[0.485, 0.456, 0.406]` (red, green, blue);
- std = `[0.229, 0.224, 0.225]`.

An RGB image in the layout height × width × channels (**HWC**) has shape `(H, W, 3)`. The mean has shape `(3,)`. Align from the right: `(H, W, 3)` against `(1, 1, 3)`. The channel axes match, the rest stretch. One line does it:

```python
rng = np.random.default_rng(1)
img = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)   # a fake 224×224 photo

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

x = (img.astype(np.float32) / 255.0 - mean) / std
x.shape, x.dtype     # ((224, 224, 3), dtype('float32'))
img[0, 0]            # array([255, 228,  34], dtype=uint8)
x[0, 0]              # array([ 2.249,  1.956, -1.212], dtype=float32)

batch = x[None]      # add a batch axis (Day 8)
batch.shape          # (1, 224, 224, 3)   NHWC, ready for a LiteRT model
```

By hand for the red value of pixel `(0, 0)`: \( (255 / 255 - 0.485) / 0.229 = 0.515 / 0.229 = 2.249 \).

!!! android "Android bridge"
    If you've shipped an image classifier on Android, you've written this broadcast by hand. The typical preprocessing loop reads the `Bitmap` into an `IntArray` of ARGB pixels and fills a direct `FloatBuffer` in NHWC order `[1, H, W, 3]`:

    ```kotlin
    val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    val std  = floatArrayOf(0.229f, 0.224f, 0.225f)
    val pixels = IntArray(w * h)
    bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
    for (p in pixels) {
        val r = ((p shr 16) and 0xFF) / 255f
        val g = ((p shr 8) and 0xFF) / 255f
        val b = (p and 0xFF) / 255f
        floatBuffer.put((r - mean[0]) / std[0])
        floatBuffer.put((g - mean[1]) / std[1])
        floatBuffer.put((b - mean[2]) / std[2])
    }
    ```

    The `mean[0]`, `mean[1]`, `mean[2]` cycling for every pixel *is* the `(3,)` array being broadcast over `(H, W, 3)`. The NumPy one-liner and your Kotlin loop must agree exactly: if training used these statistics and your app doesn't (or swaps the channel order to BGR), the model still runs but its accuracy quietly collapses. The LiteRT Support Library's `NormalizeOp` wraps the same per-channel computation. The older PyTorch Android library's `TensorImageUtils.bitmapToFloat32Tensor` (PyTorch Mobile, now superseded by ExecuTorch) also normalises with a per-channel mean and std, but by default writes the buffer in **NCHW** order: all red values, then all green, then all blue. That is the layout of the next section.

## Real use 4: when the axis you want isn't last

**What it is.** Right-alignment only lines a 1-D array up with the **last** axis. To apply one value per position of any other axis, you give the small array size-1 axes on its right, so that its length sits under the axis you mean.

**Why it exists.** Two everyday cases: PyTorch images are **channels first**, `(C, H, W)` or `(N, C, H, W)` (Day 8's NCHW), so the channel axis isn't last; and per-sample operations on a batch (one scale, one weight, one mask value per image) act on the **first** axis.

**Intuition first.** Write the shapes right-aligned, then slide the small array's length left until it sits under the right axis, filling the gap with 1s:

```text
chw              (3, 224, 224)
mean                      (3,)    → 224 vs 3: error
mean[:, None, None] (3,   1,   1) → one mean per channel plane
```

<figure class="diagram" markdown>
![Animated diagram: on the left, three stacked channel planes R, G and B, labelled chw, shape (3, H, W). On the right, the shape table chw (3, 224, 224) over mean, padded to (1, 1, 3); the last axis is outlined in red with the text 224 vs 3, ValueError, and the note that if W were 3 it would subtract along the width silently. Then a second table shows mean[:, None, None] as (3, 1, 1) under (3, 224, 224), with axis 0 outlined in green as 3 equals 3 and the two 1s stretching, result (3, 224, 224); finally the labels R plane minus 0.485, G plane minus 0.456, B plane minus 0.406 appear](../../assets/img/phase-1/day09-chw-mean.svg)
<figcaption>Animated: a (3,) mean aligns with the last axis, the width. Adding two size-1 axes moves it under the channel axis.</figcaption>
</figure>

**Example.** Take the photo from Real use 3, scale it to 0–1, and transpose it to CHW (Day 8):

```python
chw = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))
chw.shape                                  # (3, 224, 224)

try:
    chw - mean
except ValueError as e:
    print("ValueError:", e)
# ValueError: operands could not be broadcast together with shapes (3,224,224) (3,)

mean_c = mean[:, None, None]               # (3,) → (3, 1, 1)
std_c = std[:, None, None]
x_chw = (chw - mean_c) / std_c
x_chw.shape                                # (3, 224, 224)
x_chw[:, 0, 0]                             # array([ 2.249,  1.956, -1.212], dtype=float32)
np.array_equal(x_chw, np.transpose(x, (2, 0, 1)))   # True   the same numbers as the HWC version
```

`mean[:, None, None]` keeps the 3 values on axis 0 and adds two size-1 axes after them (Day 8). `mean.reshape(3, 1, 1)` and `np.expand_dims(mean, axis=(1, 2))` are other spellings of the same thing. torchvision's `Normalize` transform does exactly this internally: it reshapes its mean and std to `(C, 1, 1)`.

The same `(3, 1, 1)` mean also works on a whole **NCHW batch**, because `(N, 3, H, W)` and `(3, 1, 1)` align from the right with the channel axis in the right place:

```python
batch_nchw = np.stack([chw, chw])          # (2, 3, 224, 224), Day 8's stack
((batch_nchw - mean_c) / std_c).shape      # (2, 3, 224, 224)
```

**How it goes wrong (1): the silent case.** The error above saved you. If the width happens to equal the number of channels, `chw - mean` runs, and subtracts one mean per **column of pixels** instead of one per channel. A tiny image with 3 channel planes, all 10s, 20s and 30s, and a width of 3:

```python
tiny = np.zeros((3, 2, 3))         # C=3, H=2, W=3
tiny[0], tiny[1], tiny[2] = 10.0, 20.0, 30.0
m3 = np.array([1.0, 2.0, 3.0])

(tiny - m3)[0]                     # red plane, wrong: one value per column
# array([[9., 8., 7.],
#        [9., 8., 7.]])
(tiny - m3[:, None, None])[0]      # red plane, right: 10 − 1 everywhere
# array([[9., 9., 9.],
#        [9., 9., 9.]])
```

**Recognise it:** results that vary across a direction where they shouldn't. **Fix:** never rely on a size coincidence; say which axis you mean with `None`s, and assert shapes when you write the preprocessing.

**How it goes wrong (2): one value per sample.** Scaling each image in an NHWC batch by its own factor (a per-image brightness, or a per-sample loss weight) needs the `(N,)` array on the **first** axis:

```python
imgs = np.ones((4, 224, 224, 3))
s = np.array([0.5, 1.0, 2.0, 4.0])         # one factor per image

try:
    imgs * s
except ValueError as e:
    print("ValueError:", e)
# ValueError: operands could not be broadcast together with shapes (4,224,224,3) (4,)

scaled = imgs * s[:, None, None, None]     # (4,) → (4, 1, 1, 1)
scaled.shape                               # (4, 224, 224, 3)
scaled[:, 0, 0, 0]                         # array([0.5, 1. , 2. , 4. ])
```

The general rule: to line a 1-D array of length `n` up with axis `k` of an array, give it the shape `(n, 1, 1, ...)` with one `1` for every axis after `k`.

## Real use 5: comparisons, one-hot labels and `np.where`

Comparison operators broadcast like arithmetic, which gives a one-line **one-hot encoding**: turning integer class labels into rows with a single 1 in the label's column, the format classification losses expect (Phase 3).

```python
labels = np.array([2, 0, 1])               # 3 examples, classes 0..2
hits = labels[:, None] == np.arange(3)     # (3, 1) vs (3,) → (3, 3) booleans
hits
# array([[False, False,  True],
#        [ True, False, False],
#        [False,  True, False]])
hits.astype(int)
# array([[0, 0, 1],
#        [1, 0, 0],
#        [0, 1, 0]])
```

Row `i`, column `j` asks "is example `i`'s label equal to `j`?": an outer grid, like `col + row`. It's the reverse of Day 8's `probs[np.arange(3), labels]`, which *picked* each example's label column; multiplying `probs` by the one-hot grid and summing each row gives the same numbers.

`np.where(cond, a, b)` (Day 4b) broadcasts its three arguments together. That makes per-column filling of missing values one line: wherever a value is `nan`, take that column's mean.

```python
F = np.array([[1.0, np.nan, 3.0],
              [np.nan, 5.0, 6.0],
              [7.0, 8.0, np.nan]])
col_means = np.nanmean(F, axis=0)          # skip the gaps: one mean per column, (3,)
col_means                                  # array([4. , 6.5, 4.5])
np.where(np.isnan(F), col_means, F)        # (3, 3), (3,), (3, 3) → (3, 3)
# array([[1. , 6.5, 3. ],
#        [4. , 5. , 6. ],
#        [7. , 8. , 4.5]])
```

The `(3,)` means broadcast across the rows, so each gap gets its own column's mean.

## Real use 6: a pairwise distance table

Given 3 points `a` and 2 points `b` (each a 2-D vector), compute the distance from **every** `a` to **every** `b`: a `(3, 2)` table. This is the heart of k-nearest-neighbours and k-means (Phase 2), of brute-force vector search (Phase 4), and it is the "vector quantisation" example in the NumPy broadcasting guide.

**By hand first.** `a[1] = [3, 4]` and `b[0] = [0, 0]`: the difference is `[3, 4]`, and its length (Day 5's norm) is \( \sqrt{3^2 + 4^2} = \sqrt{25} = 5 \). That will be `D[1, 0]`.

The trick is to give each array a size-1 axis in a *different* place, exactly like the column + row grid, but with a coordinate axis on the end:

```python
a = np.array([[0., 0.], [3., 4.], [1., 1.]])   # (3, 2): 3 points
b = np.array([[0., 0.], [4., 4.]])             # (2, 2): 2 points

diff = a[:, None, :] - b[None, :, :]
#      (3, 1, 2)       (1, 2, 2)      →  (3, 2, 2)
diff.shape                     # (3, 2, 2)
diff[1, 0]                     # array([3., 4.])   a[1] - b[0]

D = np.sqrt((diff ** 2).sum(axis=-1))          # sum over the coordinate axis → (3, 2)
D
# array([[0.   , 5.657],
#        [5.   , 1.   ],
#        [1.414, 4.243]])
D.argmin(axis=1)               # array([0, 1, 0])   nearest b for each a
```

Line by line: `a[:, None, :]` turns `a` into 3 blocks of 1 row; `b[None, :, :]` turns `b` into 1 block of 2 rows. Subtracting stretches both, so `diff[i, j]` is the vector `a[i] - b[j]` for every pair. Squaring, summing the last axis and taking the square root is the norm of each of those vectors. `argmin(axis=1)` gives, for each row (each `a` point), the column of the smallest distance.

In maths, with \( \mathbf{a}_i \) the \( i \)-th point of `a` and \( \mathbf{b}_j \) the \( j \)-th point of `b`, we computed \( D_{ij} = \|\mathbf{a}_i - \mathbf{b}_j\| \) for every \( i, j \) at once.

## The classic bugs

### Bug 1: `(n,)` vs `(n, 1)` silently gives `(n, n)`

Many models return predictions shaped `(n, 1)` (one column), while your labels are shaped `(n,)`. Subtract them and broadcasting does **not** fail: right-aligned, `(3, 1)` against `(1, 3)` is the outer grid.

```python
y_true = np.array([1.0, 2.0, 3.0])          # (3,)
y_pred = np.array([[1.1], [1.9], [3.2]])    # (3, 1)

err = y_pred - y_true
err.shape        # (3, 3)    !!
err
# array([[ 0.1, -0.9, -1.9],
#        [ 0.9, -0.1, -1.1],
#        [ 2.2,  1.2,  0.2]])

((y_pred - y_true) ** 2).mean()           # np.float64(1.4200000000000004)    wrong
((y_pred.ravel() - y_true) ** 2).mean()   # np.float64(0.020000000000000035)  correct MSE
```

The long tails on `1.4200000000000004` and `0.020000000000000035` are float rounding (Day 2): 1.1, 1.9 and 3.2 have no exact binary form. Compare such results with `np.isclose`, never `==` (Day 4b). The real problem is the first number: it averages all 9 pairs, "every prediction minus every target", instead of the 3 matching ones.

<figure class="diagram" markdown>
![A (3, 1) column of predictions minus a (3,) row of targets produces a 3 by 3 grid; only the diagonal 0.1, -0.1, 0.2 are the intended errors, the 6 off-diagonal cells are every prediction minus every other target; the mean of squares is 1.42 instead of the correct 0.02](../../assets/img/phase-1/day09-n-vs-n1-bug.svg)
<figcaption>Rules applied correctly, wrong answer: (3, 1) and (3,) broadcast to (3, 3), and no error is raised.</figcaption>
</figure>

!!! warning "Common mistake: a loss that is wrong but finite"
    This bug doesn't crash. The loss is a meaningless number, and the model trains badly. PyTorch's `nn.MSELoss` emits a `UserWarning` for exactly this case ("Using a target size ... that is different to the input size ... This will likely lead to incorrect results due to broadcasting"). Defend yourself by **asserting shapes** before combining arrays (Day 3b's `assert`), and flatten with `.ravel()`, `[:, 0]` or `squeeze(axis=1)` (Day 8):

    ```python
    try:
        assert y_pred.shape == y_true.shape, f"shapes differ: {y_pred.shape} vs {y_true.shape}"
    except AssertionError as e:
        print("AssertionError:", e)
    # AssertionError: shapes differ: (3, 1) vs (3,)
    ```

### Bug 2: forgetting `keepdims=True`

Day 4 introduced `keepdims=True` and showed the error you get without it when you divide each row by its total. Here is *why*, in the terms of the rules: `S.sum(axis=1)` removes axis 1, leaving `(2,)`. Right-aligned against `(2, 3)`, those 2 row totals land under the **3 columns**, not the 2 rows. With `keepdims=True` they stay a `(2, 1)` column, and the 1 stretches across each row.

<figure class="diagram" markdown>
![Animated diagram in two panels. Left, S / S.sum(axis=1): S is a 2 by 3 grid with rows 1 2 1 and 4 4 2; under it the row totals 4 and 10 sit under the last two columns, with a dashed question mark under the first, outlined in red; text explains (2, 3) against (1, 2), last axis 3 vs 2, ValueError, and that a square matrix would run and divide columns. Right, S / S.sum(axis=1, keepdims=True): the totals form a 2 by 1 column, 4 over 10, beside S; three dashed copies of the column slide in under S to form a stretched 2 by 3 grid of 4s and 10s; then the result P appears, rows 0.25 0.5 0.25 and 0.4 0.4 0.2, with the shape check (2, 3) against (2, 1): 3 vs 1 stretch, 2 vs 2 equal](../../assets/img/phase-1/day09-keepdims.svg)
<figcaption>Animated: without keepdims the row totals align with the columns; with it they stay a column and stretch across each row.</figcaption>
</figure>

The same example as code, with the failing line wrapped so the fix runs:

```python
S = np.array([[1., 2., 1.],
              [4., 4., 2.]])
S.sum(axis=1).shape                      # (2,)
try:
    S / S.sum(axis=1)
except ValueError as e:
    print("ValueError:", e)
# ValueError: operands could not be broadcast together with shapes (2,3) (2,)

tot = S.sum(axis=1, keepdims=True)
tot
# array([[ 4.],
#        [10.]])
P = S / tot
P
# array([[0.25, 0.5 , 0.25],
#        [0.4 , 0.4 , 0.2 ]])
P.sum(axis=1)                            # array([1., 1.])
```

!!! warning "Common mistake: square matrices hide the bug"
    With a **square** matrix the shapes line up the wrong way, and the division runs, dividing each *column* by a row's total:

    ```python
    Q = np.array([[1., 3.],
                  [2., 8.]])
    (Q / Q.sum(axis=1)).sum(axis=1)                  # array([0.55, 1.3 ])   rows don't sum to 1
    (Q / Q.sum(axis=1, keepdims=True)).sum(axis=1)   # array([1., 1.])
    ```

    Row totals are 4 and 10. Without `keepdims`, row 0 becomes `[1/4, 3/10]`, which sums to 0.55. No error, wrong answer.

**The general rule.** Without `keepdims`, the reduced axes disappear and what's left is aligned from the right. That only lands in the right place when every reduced axis was a **leading** one: `axis=0` on a `(rows, features)` table (standardisation), or `axis=(0, 1, 2)` on an `(N, H, W, C)` batch (per-channel means). Reduce any later axis and you need `keepdims=True` to combine the result with the original. On a 4-D image batch:

```python
rng = np.random.default_rng(2)
imgs4 = rng.random((4, 8, 8, 3))           # 4 small NHWC images, values in [0, 1)

imgs4.mean(axis=(1, 2, 3)).shape                 # (4,)          one brightness per image
imgs4.mean(axis=(1, 2, 3), keepdims=True).shape  # (4, 1, 1, 1)
centred = imgs4 - imgs4.mean(axis=(1, 2, 3), keepdims=True)
np.allclose(centred.mean(axis=(1, 2, 3)), 0)     # True    every image now has mean 0

imgs4.mean(axis=(1, 2), keepdims=True).shape     # (4, 1, 1, 3)  per image AND per channel
imgs4.mean(axis=(0, 1, 2)).shape                 # (3,)          leading axes: no keepdims needed
try:
    imgs4 - imgs4.mean(axis=(1, 2, 3))
except ValueError as e:
    print("ValueError:", e)
# ValueError: operands could not be broadcast together with shapes (4,8,8,3) (4,)
```

`rng.random(shape)` draws floats from 0 up to (not including) 1 (Day 4b).

### Using it: a numerically safe softmax

**Softmax** turns a row of scores into probabilities: \( p_j = e^{z_j} / \sum_k e^{z_k} \), where \( z_j \) is the score of class \( j \) and \( e^{z} \) is `np.exp` (Day 4's ufuncs). Every classifier ends with it (Phase 3), and it uses `keepdims` twice. The naive version overflows on large scores: \( e^{1000} \) is far beyond the largest `float64`, so `np.exp` returns `inf`, and `inf / inf` is `nan`.

```python
scores = np.array([[2.0, 1.0, 0.1],
                   [1000.0, 1001.0, 1002.0]])
np.exp(scores[1])
# RuntimeWarning: overflow encountered in exp
# array([inf, inf, inf])

shifted = scores - scores.max(axis=1, keepdims=True)   # subtract each row's max: (2, 3) - (2, 1)
shifted
# array([[ 0. , -1. , -1.9],
#        [-2. , -1. ,  0. ]])
e = np.exp(shifted)
probs = e / e.sum(axis=1, keepdims=True)               # divide each row by its total: (2, 3) / (2, 1)
probs
# array([[0.659, 0.242, 0.099],
#        [0.09 , 0.245, 0.665]])
probs.sum(axis=1)                                      # array([1., 1.])
```

Subtracting the same number \( c \) from every score in a row doesn't change the softmax, because \( e^{z_j - c} / \sum_k e^{z_k - c} = e^{-c} e^{z_j} / (e^{-c} \sum_k e^{z_k}) \) and the \( e^{-c} \) cancels. Choosing \( c \) as the row's maximum makes the largest exponent \( e^0 = 1 \), so nothing overflows. Both the max and the sum are reductions along the **last** axis, so both need `keepdims=True`.

### Bug 3: huge broadcast intermediates

Broadcasting doesn't copy its inputs, but the **result** is full size. The pairwise trick above builds `diff` of shape `(n, m, d)`. For 10,000 queries against 10,000 database vectors with 128 dimensions, in `float64` (8 bytes each):

\[
10{,}000 \times 10{,}000 \times 128 \times 8 \text{ bytes} = 102.4 \text{ GB}
\]

Your laptop will freeze or the process will be killed. Two fixes:

- **Process in chunks**: loop over a few hundred queries at a time, so each `diff` is a manageable size. The challenge exercise builds this.
- **Use algebra to skip the 3-D intermediate.** Expanding the square, \( \|\mathbf{a} - \mathbf{b}\|^2 = \|\mathbf{a}\|^2 + \|\mathbf{b}\|^2 - 2\,\mathbf{a} \cdot \mathbf{b} \). The last term for all pairs is one matrix multiply: `a @ b.T` is the table of every dot product of a row of `a` with a row of `b`, shape `(n, m)` (Day 5 did `@` for two vectors; Day 10 explains it for matrices). The largest array is then only `(n, m)`.

A small version that fits in the browser, checked against the direct method:

```python
rng = np.random.default_rng(0)
qa = rng.normal(size=(250, 32))
qb = np.random.default_rng(1).normal(size=(400, 32))

250 * 400 * 32 * 8 / 1e6                   # 25.6   MB for the (250, 400, 32) intermediate
D1 = np.sqrt(((qa[:, None, :] - qb[None, :, :]) ** 2).sum(axis=-1))
sq = (qa**2).sum(axis=1)[:, None] + (qb**2).sum(axis=1)[None, :] - 2 * qa @ qb.T
D2 = np.sqrt(np.maximum(sq, 0))            # rounding can make sq slightly negative → clip at 0
np.allclose(D1, D2), D1.shape              # (True, (250, 400))
```

Read the `sq` line in three parts: `(qa**2).sum(axis=1)[:, None]` is each query's squared length as a `(250, 1)` column; `(qb**2).sum(axis=1)[None, :]` is each database vector's squared length as a `(1, 400)` row; `qa @ qb.T` is the `(250, 400)` table of dot products. `np.maximum(sq, 0)` (Day 4) replaces any tiny negative value, caused by rounding, with 0 before the square root.

The second version is *still* broadcasting: a `(250, 1)` column plus a `(1, 400)` row. Broadcasting isn't the problem; the size of what it produces is. Before a broadcast, multiply out the result shape and the bytes per element.

## Going further

These are real and worth knowing exist, but aren't needed yet:

- **`np.broadcast_arrays`**: returns all inputs stretched to their common shape, as read-only views. [numpy.broadcast_arrays](https://numpy.org/doc/stable/reference/generated/numpy.broadcast_arrays.html)
- **`np.outer` and `ufunc.outer`**: the outer grid without writing the `None`s (`np.add.outer(c, v)`). [numpy.ufunc.outer](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.outer.html)
- **`np.einsum`**: one string that describes which axes to pair, broadcast and sum; it replaces many broadcast-then-sum patterns. [numpy.einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)
- **PyTorch `expand` vs `repeat`**: `expand` is PyTorch's `broadcast_to` (a stride-0 view); `repeat` copies, like `np.tile`. [torch.Tensor.expand](https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html)
- **Broadcasting in batched matrix multiply**: `@` / `np.matmul` broadcasts every axis except the last two (Day 10). [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)
- **Large pairwise distances**: SciPy's `cdist` computes the table without the 3-D intermediate, and `numexpr` evaluates array expressions in chunks. [scipy.spatial.distance.cdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html) and [NumExpr](https://numexpr.readthedocs.io/)
- **`np.lib.stride_tricks.as_strided`**: build any view from hand-written strides, including stride 0; powerful and easy to misuse. [as_strided](https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.as_strided.html)

---

## Exercises

1. **Warm-up: shapes by hand.** Without running code, write the result shape (or "error") for: `(5, 4)` and `(4,)`; `(5, 4)` and `(5,)`; `(5, 1)` and `(1, 4)`; `(2, 3, 4)` and `(4,)`; `(2, 3, 4)` and `(2, 1, 4)`; `(3,)` and `(3, 1)`. Then check yourself with `np.broadcast_shapes`.
2. **Warm-up: multiplication table.** Build the 9 × 9 times table (1 to 9) with a single broadcast, no loops. Check that `table[6, 7]` is 56.
3. **Core: min-max scaling.** Using the house table `X` from this lesson, rescale each column to the range 0 to 1: \( (x - \min) / (\max - \min) \), with the min and max taken per column. One line of NumPy.
4. **Core: centre each row.** For `S = np.array([[1., 2., 6.], [4., 4., 10.]])`, subtract each row's own mean so every row averages 0. Verify with `.mean(axis=1)`.
5. **Core: undo the normalisation.** Take the normalised image `x` from Real use 3 and convert it back to a `uint8` image. Check that you get exactly the original `img` back. (You'd need this on Android to display what a model "sees".)
6. **Predict the output.** Write down what each line shows before you run it:
    ```python
    import numpy as np
    r = np.arange(3)
    r[:, None] - r
    (r[:, None] > r).sum()
    np.ones((2, 1)) * np.array([1, 2, 3])
    ```
7. **Find and fix the bug.** This softmax is supposed to return rows that sum to 1. It runs without an error, but they don't. Find the bug, explain why NumPy didn't complain, and fix it.
    ```python
    import numpy as np

    def softmax_rows(z):
        e = np.exp(z - z.max(axis=1))
        return e / e.sum(axis=1)

    scores = np.array([[1.0, 2.0, 3.0],
                       [1.0, 1.0, 1.0],
                       [0.0, 0.0, 10.0]])
    softmax_rows(scores).sum(axis=1)       # array([2.442, 0.928, 1.047])
    ```
8. **Challenge: nearest centroid, within a memory budget.** First, assign each of these 6 points to the nearest of 3 centroids with a pairwise distance table: `pts = [[0,0],[1,0],[9,9],[10,8],[0,9],[1,10]]`, `cent = [[0,0],[10,10],[0,10]]`. Expected answer: `[0 0 1 1 2 2]` (this is the assignment step of k-means, Phase 2). Then write `nearest(points, centroids, max_bytes)` that does the same for any number of points, processing them in chunks so that no `diff` array is ever bigger than `max_bytes`. Test it on 5,000 random points and 40 random centroids in 16 dimensions with a 2 MB budget, and check it matches the all-at-once answer.

??? success "Solutions"
    **1. Shapes by hand.**

    ```python
    import numpy as np
    for s1, s2 in [((5, 4), (4,)), ((5, 4), (5,)), ((5, 1), (1, 4)),
                   ((2, 3, 4), (4,)), ((2, 3, 4), (2, 1, 4)), ((3,), (3, 1))]:
        try:
            print(s1, s2, "->", np.broadcast_shapes(s1, s2))
        except ValueError:
            print(s1, s2, "-> error")
    # (5, 4) (4,) -> (5, 4)
    # (5, 4) (5,) -> error
    # (5, 1) (1, 4) -> (5, 4)
    # (2, 3, 4) (4,) -> (2, 3, 4)
    # (2, 3, 4) (2, 1, 4) -> (2, 3, 4)
    # (3,) (3, 1) -> (3, 3)
    ```

    Why: `(5, 4)` and `(5,)` fails because the 5 is right-aligned under the 4 columns, and 4 vs 5 has no 1. `(3,)` with `(3, 1)` doesn't fail, but becomes `(1, 3)` against `(3, 1)`, the outer grid of Bug 1.

    **2. Multiplication table.**

    ```python
    import numpy as np
    n = np.arange(1, 10)
    table = n[:, None] * n[None, :]   # (9, 1) * (1, 9) → (9, 9)
    table[6, 7]                       # np.int64(56)   (row for 7, column for 8); browser: np.int32(56)
    table[:3, :5]
    # array([[ 1,  2,  3,  4,  5],
    #        [ 2,  4,  6,  8, 10],
    #        [ 3,  6,  9, 12, 15]])
    ```

    Why: `n[:, None]` is a column and `n[None, :]` a row, so both size-1 axes stretch and cell `[i, j]` is `n[i] * n[j]`: every row value times every column value, the `c[i] * v[j]` loop.

    **3. Min-max scaling.**

    ```python
    import numpy as np
    np.set_printoptions(precision=3, suppress=True)
    X = np.array([[3, 120, 15],
                  [2,  70, 40],
                  [4, 160,  5],
                  [3, 110, 20]], dtype=float)
    lo, hi = X.min(axis=0), X.max(axis=0)
    X01 = (X - lo) / (hi - lo)
    X01
    # array([[0.5  , 0.556, 0.286],
    #        [0.   , 0.   , 1.   ],
    #        [1.   , 1.   , 0.   ],
    #        [0.5  , 0.444, 0.429]])
    X01.min(axis=0), X01.max(axis=0)   # (array([0., 0., 0.]), array([1., 1., 1.]))
    ```

    Why: `min(axis=0)` and `max(axis=0)` reduce the **leading** axis, so `lo` and `hi` are `(3,)` and already right-align with the 3 columns. No `keepdims` is needed. (A constant column would give `hi - lo = 0`, the same trap as a zero standard deviation.)

    **4. Centre each row.**

    ```python
    import numpy as np
    np.set_printoptions(precision=3, suppress=True)
    S = np.array([[1., 2., 6.],
                  [4., 4., 10.]])
    C = S - S.mean(axis=1, keepdims=True)    # (2, 3) - (2, 1)
    C
    # array([[-2., -1.,  3.],
    #        [-2., -2.,  4.]])
    C.mean(axis=1)                           # array([0., 0.])
    ```

    Why: the row means come from `axis=1`, which is not a leading axis. Without `keepdims` they'd be `(2,)` and land under the 3 columns, giving a `ValueError`. With it they're a `(2, 1)` column, and each row gets its own mean (3 and 6).

    **5. Undo the normalisation.**

    ```python
    import numpy as np
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (img.astype(np.float32) / 255.0 - mean) / std

    back = np.clip(np.round((x * std + mean) * 255.0), 0, 255).astype(np.uint8)
    back.dtype, np.array_equal(back, img)    # (dtype('uint8'), True)

    trunc = ((x * std + mean) * 255.0).astype(np.uint8)     # without the round
    np.count_nonzero(trunc != img), img.size                # (np.int64(26744), 150528)   browser: np.int32
    ```

    Why: the inverse is the same broadcast backwards, `(224, 224, 3) * (3,) + (3,)`. `np.round` matters: float arithmetic lands on values like 5.999992, and `astype(np.uint8)` truncates, turning it into 5. Without the round, 26,744 of the 150,528 values come back off by one. `np.clip` keeps any value that rounding pushed past 255 or below 0 from wrapping (Day 4b's order: round, clip, then `astype`).

    **6. Predict the output.**

    ```python
    import numpy as np
    r = np.arange(3)
    r[:, None] - r
    # array([[ 0, -1, -2],
    #        [ 1,  0, -1],
    #        [ 2,  1,  0]])
    (r[:, None] > r).sum()                     # np.int64(3)    browser: np.int32(3)
    np.ones((2, 1)) * np.array([1, 2, 3])
    # array([[1., 2., 3.],
    #        [1., 2., 3.]])
    ```

    Why: `r[:, None] - r` is `(3, 1)` against `(3,)`, so cell `[i, j]` is `i - j`. The comparison builds the same grid of `i > j`, which is `True` in the 3 cells below the diagonal, and `sum` counts `True` as 1. The last line is `(2, 1)` times `(3,)`: both stretch to `(2, 3)`, and the result is float because `np.ones` is float.

    **7. Find and fix the bug.**

    ```python
    import numpy as np
    np.set_printoptions(precision=3, suppress=True)

    def softmax_rows(z):
        e = np.exp(z - z.max(axis=1, keepdims=True))     # (3, 3) - (3, 1)
        return e / e.sum(axis=1, keepdims=True)          # (3, 3) / (3, 1)

    scores = np.array([[1.0, 2.0, 3.0],
                       [1.0, 1.0, 1.0],
                       [0.0, 0.0, 10.0]])
    softmax_rows(scores)
    # array([[0.09 , 0.245, 0.665],
    #        [0.333, 0.333, 0.333],
    #        [0.   , 0.   , 1.   ]])
    softmax_rows(scores).sum(axis=1)                     # array([1., 1., 1.])
    ```

    Why: `z.max(axis=1)` and `e.sum(axis=1)` reduce the last axis and return `(3,)`. Because `scores` is square, `(3, 3)` against `(3,)` is a valid broadcast, so each **column** `j` was shifted by row `j`'s max and divided by row `j`'s total: no error, wrong numbers. `keepdims=True` keeps them as `(3, 1)` columns so each row uses its own max and total. (In row 3, the two zeros show as `0.` because they're about 0.0000454, below the 3 printed digits.)

    **8. Challenge: nearest centroid, within a memory budget.**

    ```python
    import numpy as np

    pts = np.array([[0., 0.], [1., 0.], [9., 9.], [10., 8.], [0., 9.], [1., 10.]])
    cent = np.array([[0., 0.], [10., 10.], [0., 10.]])
    D = np.sqrt(((pts[:, None, :] - cent[None, :, :]) ** 2).sum(axis=-1))
    D.shape                        # (6, 3)    one row per point, one column per centroid
    D.argmin(axis=1)               # array([0, 0, 1, 1, 2, 2])

    def nearest(points, centroids, max_bytes):
        k, d = centroids.shape
        row_bytes = k * d * points.itemsize          # bytes of diff for one point
        chunk = max(1, max_bytes // row_bytes)       # points per chunk
        out = np.empty(len(points), dtype=np.int64)
        for start in range(0, len(points), chunk):
            p = points[start:start + chunk]
            diff = p[:, None, :] - centroids[None, :, :]       # (chunk, k, d)
            assert diff.nbytes <= max_bytes
            out[start:start + chunk] = (diff ** 2).sum(axis=-1).argmin(axis=1)
        return out, chunk

    rng = np.random.default_rng(7)
    P = rng.normal(size=(5000, 16))
    Cs = rng.normal(size=(40, 16))

    labels, chunk = nearest(P, Cs, max_bytes=2_000_000)
    chunk                                            # 390
    5000 * 40 * 16 * 8                               # 25600000   bytes if done all at once
    full = ((P[:, None, :] - Cs[None, :, :]) ** 2).sum(axis=-1).argmin(axis=1)
    np.array_equal(labels, full)                     # True
    np.array_equal(nearest(pts, cent, 100)[0], D.argmin(axis=1))   # True
    ```

    Why: `D[i, j]` is the distance from point `i` (axis 0) to centroid `j` (axis 1), so `argmin(axis=1)` picks the closest centroid for each point. In `nearest`, one point's slice of `diff` is `k × d` numbers, so dividing the budget by that size gives the largest chunk that fits: 2,000,000 / (40 × 16 × 8) = 390 points. Each chunk is the same broadcast on fewer rows, and slicing `points[start:start + chunk]` gives the last, shorter chunk for free. The square root is skipped because it doesn't change which distance is smallest. The full version needed 25.6 MB at once; the chunked one never more than 2 MB.

## Resources

- [NumPy: Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html): the official explanation, with diagrams, the same rules, and the vector-quantisation example.
- [NumPy: `numpy.broadcast_to`](https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html): the explicit, read-only broadcast view.
- [PyTorch: Broadcasting semantics](https://pytorch.org/docs/stable/notes/broadcasting.html): the same rules in PyTorch, including the in-place caveat.
- [Python Data Science Handbook: Computation on Arrays: Broadcasting](https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html): free book chapter with good visual examples.
- [CS231n Python/NumPy tutorial](https://cs231n.github.io/python-numpy-tutorial/): see its broadcasting section.
- [Dive into Deep Learning: 2.1 Data Manipulation](https://d2l.ai/chapter_preliminaries/ndarray.html): broadcasting in the context of deep learning code.
- [torchvision transforms](https://pytorch.org/vision/stable/transforms.html): where `Normalize(mean, std)` with the ImageNet statistics comes from.

## Checklist

- [ ] I can write the Kotlin loop a broadcast replaces, and point at the missing index.
- [ ] I can state the rules (align from the right, pad with 1s on the left, each pair equal or containing a 1) and work out a result shape or an error by hand.
- [ ] I can explain stride 0, why a broadcast view is read-only, and why `a += b` can't grow `a`.
- [ ] I can standardise a `(samples, features)` table, add a bias to a batch, and normalise an image in HWC and in CHW.
- [ ] I can line a 1-D array up with any axis using `None`s, for channels-first means and per-sample factors.
- [ ] I can build one-hot labels with a comparison, fill gaps with `np.where`, and build a pairwise distance table.
- [ ] I assert shapes before combining predictions and labels, I use `keepdims=True` whenever I reduce a non-leading axis, I watch for `uint8` wraps and zero deviations, and I check the size of a broadcast result before building it.

**Next:** [Day 10: Matrices and matrix multiplication](day-10-matrices-matmul.md)
