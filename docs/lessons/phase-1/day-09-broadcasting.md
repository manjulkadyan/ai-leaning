# Day 9: Broadcasting

**Phase 1 · Week 2 · ~90 minutes**

## Objectives

- [ ] Explain what broadcasting is and predict when NumPy will apply it.
- [ ] State the two broadcasting rules precisely and use them to work out a result shape, or spot an error, by hand.
- [ ] Explain why broadcasting does not copy memory.
- [ ] Use broadcasting for feature standardization, adding a bias, per-channel image normalization and pairwise distances.
- [ ] Recognize and fix the three classic broadcasting bugs: `(n,)` vs `(n, 1)`, missing `keepdims=True`, and memory blowups.

**Prerequisites.** [Day 4](day-04-numpy-arrays.md) (arrays, `axis=`) and [Day 8](day-08-shapes-indexing.md) (`reshape`, `None`/`np.newaxis`, `expand_dims`, `squeeze`).

---

## The big idea

On Day 4 you added two arrays of the *same* shape element by element. But ML code constantly combines arrays of *different* shapes: subtract one average per column from a whole table, add one bias per neuron to a whole batch, divide every pixel of an image by one number per color channel.

**Broadcasting** is NumPy's rule for doing that without writing a loop and without copying data: when two shapes differ, NumPy *pretends* the smaller array is repeated along the missing or size-1 axes until the shapes match, then does the normal element-by-element operation.

**Analogy.** A teacher writes one correction on the whiteboard ("add 2 bonus points to question 3") instead of writing it on every student's paper. Everyone applies the same line to their own paper. The correction exists once; it is *applied* to every row. That's a broadcast: one row of numbers, applied to every row of a table.

PyTorch, TensorFlow, JAX and LiteRT all follow the same rules, so this lesson pays off in every framework you'll touch.

## Step 1: a scalar and an array

You've already used broadcasting. A **scalar** (a single number) combined with an array is applied to every element:

```python
import numpy as np

a = np.array([1, 2, 3])
a + 10      # array([11, 12, 13])
a * 2       # array([2, 4, 6])
```

Think of `10` as an array of shape `()` (no axes at all) that NumPy stretches to `(3,)`: `[10, 10, 10]`.

## Step 2: a vector and a matrix

Now combine a matrix of shape `(4, 3)` (4 rows, 3 columns) with a vector of shape `(3,)`:

```python
M = np.array([[ 0,  0,  0],
              [10, 10, 10],
              [20, 20, 20],
              [30, 30, 30]])
v = np.array([1, 2, 3])

M + v
# array([[ 1,  2,  3],
#        [11, 12, 13],
#        [21, 22, 23],
#        [31, 32, 33]])
(M + v).shape   # (4, 3)
```

The vector's 3 elements match the matrix's 3 columns, so NumPy adds `v` to **every row**.

<figure class="diagram" markdown>
![Animation: a 4 by 3 matrix M with rows 0, 10, 20, 30 sits next to a row vector v = [1, 2, 3]; dashed copies of v slide down to fill 4 rows, then the 4 by 3 result fills in row by row with 1 2 3, 11 12 13, 21 22 23, 31 32 33](../../assets/img/phase-1/day09-broadcast-rows.svg)
<figcaption>Animated: the (3,) row behaves as if copied down all 4 rows, then the add is element by element.</figcaption>
</figure>

In math notation, if \( M \) is the matrix and \( \mathbf{v} \) the vector, the result \( R \) is

\[
R_{ij} = M_{ij} + v_j
\]

Read it symbol by symbol: \( i \) is the row number, \( j \) the column number, \( M_{ij} \) the matrix element in row \( i \), column \( j \). The vector has only a column index \( j \), so the **same** \( v_j \) is used for every row \( i \). Worked example for row 2, column 3 (counting from 1): \( R_{23} = M_{23} + v_3 = 10 + 3 = 13 \). Broadcasting is simply "the index that's missing is ignored".

## The rules, precisely

NumPy decides whether two shapes are compatible like this:

1. **Align the shapes from the right.** Write them one above the other, right-justified. If one shape has fewer dimensions, pad it on the **left** with 1s (missing leading dimensions count as 1).
2. **Compare each pair of dimensions.** For every column, the two sizes must be **equal**, or **one of them must be 1**. A size-1 dimension is stretched to match the other. If any pair fails, you get a `ValueError`.

The result shape takes, in each column, the larger of the two sizes.

<figure class="diagram" markdown>
![Animated step diagram: for (4, 3) plus (3,), the missing axis is treated as 1, axis -1 compares 3 with 3 and axis -2 compares 4 with 1, result (4, 3); for (4, 3) plus (4,), axis -1 compares 3 with 4 and fails with ValueError; for (4, 1) plus (1, 3), both size-1 axes stretch, result (4, 3)](../../assets/img/phase-1/day09-rule-check.svg)
<figcaption>Animated: right-align the shapes, then check each axis from the right. One failing pair means an error.</figcaption>
</figure>

Working the middle case by hand, because it surprises everyone:

```
(4, 3)
   (4,)  → pad left →  (1, 4)

axis -1:  3 vs 4   neither is 1  →  ValueError
```

A vector of length 4 does **not** line up with the 4 rows. It lines up with the *last* axis, the 3 columns. NumPy never guesses; it only ever aligns from the right.

```python
M + np.array([1, 2, 3, 4])
# ValueError: operands could not be broadcast together with shapes (4,3) (4,)
```

To add one value per **row**, make the vector a column of shape `(4, 1)` first (Day 8): `M + np.array([1, 2, 3, 4])[:, None]` works, giving `(4, 3)`.

### A table to check yourself against

You can ask NumPy to apply the rules without making any arrays, using `np.broadcast_shapes`. Every row below is real output:

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
| `(5,)` | `(5, 1)` | `(5, 5)` | no error, but probably a bug (see below) |
| `(256, 256, 3)` | `(3,)` | `(256, 256, 3)` | one value per color channel |
| `(2, 1)` | `(8, 4, 3)` | **error** | axis -2: 2 vs 4 |

```python
np.broadcast_shapes((8, 1, 6, 1), (7, 1, 5))   # (8, 7, 6, 5)
np.broadcast_shapes((4, 3), (4,))
# ValueError: shape mismatch: objects cannot be broadcast to a single shape.
#   Mismatch is between arg 0 with shape (4, 3) and arg 1 with shape (4,).
```

The `(4, 1)` + `(1, 3)` case deserves its own example: a column plus a row gives a full grid, every row value combined with every column value.

```python
col = np.array([[0], [10], [20], [30]])   # (4, 1)
row = np.array([[1, 2, 3]])               # (1, 3)
col + row
# array([[ 1,  2,  3],
#        [11, 12, 13],
#        [21, 22, 23],
#        [31, 32, 33]])
```

## It doesn't actually copy anything

The animation shows copies, but that's only the mental model. NumPy never builds the stretched array. Internally every array has **strides**: how many bytes to jump to move one step along each axis. A stretched axis gets a stride of **0**: "to go to the next row, move 0 bytes", so the same memory is read again.

You can see this with `np.broadcast_to`, which returns the stretched *view* explicitly:

```python
v = np.array([1.0, 2.0, 3.0])
big = np.broadcast_to(v, (1_000_000, 3))

big.shape                  # (1000000, 3)
big.strides                # (0, 8)   0 bytes to the next row, 8 bytes to the next column
v.nbytes, big.nbytes       # (24, 24000000)   nbytes reports the logical size...
np.shares_memory(big, v)   # True             ...but it is v's 24 bytes underneath
big.flags.writeable        # False            read-only, since every row is the same memory
```

So `X - mean` on a million-row table does not allocate a million-row copy of `mean`. The loop runs in C and reads `mean` over and over. Only the **result** is a new full-size array.

!!! warning "Common mistake: in-place ops can't grow the left side"
    `b += a` works when the result has `b`'s shape. But with `a` of shape `(3,)` and `b` of shape `(2, 3)`, `a += b` fails with `ValueError: non-broadcastable output operand with shape (3,) doesn't match the broadcast shape (2,3)`. An in-place operation writes into existing memory, so that array can't be the one that stretches.

## Real use 1: standardizing features

Most ML models train better when each feature (column) has mean 0 and standard deviation 1, so that "area in m²" (around 100) doesn't drown out "bedrooms" (around 3). This is called **standardization**:

\[
z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}
\]

- \( x_{ij} \): the value of feature \( j \) for sample (row) \( i \).
- \( \mu_j \) ("mu j"): the **mean** of column \( j \), the average over all rows.
- \( \sigma_j \) ("sigma j"): the **standard deviation** of column \( j \), roughly the typical distance of a value from that mean.
- \( z_{ij} \): the standardized value, "how many standard deviations above average".

Again, \( \mu \) and \( \sigma \) have only a column index: one number per feature, broadcast over every row.

```python
np.set_printoptions(precision=3, suppress=True)

# rows = houses, columns = bedrooms, area_m2, age_years   → shape (4, 3)
X = np.array([[3, 120, 15],
              [2,  70, 40],
              [4, 160,  5],
              [3, 110, 20]], dtype=float)

mu = X.mean(axis=0)      # [  3. 115.  20.]   shape (3,): one per column
sigma = X.std(axis=0)    # [ 0.707 32.016 12.748]

Z = (X - mu) / sigma     # (4, 3) with (3,), then (4, 3) with (3,)
# [[ 0.     0.156 -0.392]
#  [-1.414 -1.406  1.569]
#  [ 1.414  1.406 -1.177]
#  [ 0.    -0.156  0.   ]]
Z.mean(axis=0)           # [0. 0. 0.]
Z.std(axis=0)            # [1. 1. 1.]
```

Worked by hand for house 2's area: \( (70 - 115) / 32.016 = -1.406 \). This works with no reshaping because `X.mean(axis=0)` removes axis 0 and leaves `(3,)`, which aligns from the right with the 3 columns.

## Real use 2: adding a bias to a batch

On Day 5 a neuron computed \( \mathbf{w} \cdot \mathbf{x} + b \). A layer of 4 neurons has 4 biases, one per neuron. Run it on a **batch** of 32 inputs at once and the layer's raw outputs have shape `(32, 4)`: one row per input, one column per neuron. Adding the bias vector of shape `(4,)` broadcasts it across all 32 rows:

```python
rng = np.random.default_rng(0)
out = rng.normal(size=(32, 4)).round(2)   # pretend layer outputs, before the bias
b = np.array([0.1, -0.2, 0.0, 0.5])       # one bias per neuron

y = out + b
out.shape, b.shape, y.shape    # ((32, 4), (4,), (32, 4))
out[0]                         # [ 0.13 -0.13  0.64  0.1 ]
y[0]                           # [ 0.23 -0.33  0.64  0.6 ]
```

This is literally what `nn.Linear` in PyTorch does after its matrix multiply. You'll build it yourself on Day 12.

## Real use 3: normalizing an image per channel

Image models trained on the ImageNet dataset expect every pixel to be normalized per color channel with these well-known statistics:

- mean = `[0.485, 0.456, 0.406]` (red, green, blue)
- std = `[0.229, 0.224, 0.225]`

An RGB image in the layout height × width × channels has shape `(H, W, 3)`. The mean has shape `(3,)`. Align from the right: `(H, W, 3)` against `(1, 1, 3)`. The channel axes match, the rest stretch. One line does it:

```python
rng = np.random.default_rng(1)
img = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)   # a fake 224×224 photo

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

x = (img.astype(np.float32) / 255.0 - mean) / std
x.shape, x.dtype     # ((224, 224, 3), dtype('float32'))
img[0, 0]            # [255 228  34]
x[0, 0]              # [ 2.249  1.956 -1.212]    (255/255 - 0.485) / 0.229 = 2.249

batch = x[None]      # add a batch axis (Day 8)
batch.shape          # (1, 224, 224, 3)   NHWC, ready for a LiteRT model
```

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

    The `mean[0]`, `mean[1]`, `mean[2]` cycling for every pixel *is* the `(3,)` array being broadcast over `(H, W, 3)`. The NumPy one-liner and your Kotlin loop must agree exactly: if training used these statistics and your app doesn't (or swaps the channel order to BGR), the model still runs but its accuracy quietly collapses. The LiteRT Support Library's `NormalizeOp` and PyTorch Android's `TensorImageUtils` wrap the same computation.

## Real use 4: a pairwise distance matrix

Given 3 points `a` and 2 points `b` (each a 2-D vector), compute the distance from **every** `a` to **every** `b`: a `(3, 2)` table. This is the heart of k-nearest-neighbors (Phase 2) and of brute-force vector search (Phase 4).

The trick is to give each array a size-1 axis in a *different* place, exactly like the column + row example:

```python
a = np.array([[0., 0.], [3., 4.], [1., 1.]])   # (3, 2): 3 points
b = np.array([[0., 0.], [4., 4.]])             # (2, 2): 2 points

diff = a[:, None, :] - b[None, :, :]
#      (3, 1, 2)       (1, 2, 2)      →  (3, 2, 2)
# diff[i, j] is the vector a[i] - b[j]

D = np.sqrt((diff ** 2).sum(axis=-1))          # sum over the coordinate axis → (3, 2)
# [[0.    5.657]
#  [5.    1.   ]
#  [1.414 4.243]]
D.argmin(axis=1)     # [0 1 0]   nearest b for each a
```

Check one entry by hand: `a[1] = [3, 4]` and `b[0] = [0, 0]` give \( \sqrt{3^2 + 4^2} = 5 \). That's `D[1, 0]`.

In math, with \( \mathbf{a}_i \) the \( i \)-th point of `a` and \( \mathbf{b}_j \) the \( j \)-th point of `b`, we computed \( D_{ij} = \|\mathbf{a}_i - \mathbf{b}_j\| \) (the norm from Day 5) for every \( i, j \) at once.

## The classic bugs

### Bug 1: `(n,)` vs `(n, 1)` silently gives `(n, n)`

Many models return predictions shaped `(n, 1)` (one column), while your labels are shaped `(n,)`. Subtract them and broadcasting does **not** fail:

```python
y_true = np.array([1.0, 2.0, 3.0])          # (3,)
y_pred = np.array([[1.1], [1.9], [3.2]])    # (3, 1)

err = y_pred - y_true
err.shape        # (3, 3)    !!
err
# [[ 0.1 -0.9 -1.9]
#  [ 0.9 -0.1 -1.1]
#  [ 2.2  1.2  0.2]]

((y_pred - y_true) ** 2).mean()           # 1.42   wrong
((y_pred.ravel() - y_true) ** 2).mean()   # 0.02   correct mean squared error
```

<figure class="diagram" markdown>
![A (3, 1) column of predictions minus a (3,) row of targets produces a 3 by 3 grid; only the diagonal 0.1, -0.1, 0.2 are the intended errors, the 6 off-diagonal cells are every prediction minus every other target; the mean of squares is 1.42 instead of the correct 0.02](../../assets/img/phase-1/day09-n-vs-n1-bug.svg)
<figcaption>Rules applied correctly, wrong answer: (3, 1) and (3,) broadcast to (3, 3), and no error is raised.</figcaption>
</figure>

!!! warning "Common mistake: a loss that is wrong but finite"
    This bug doesn't crash. Your loss is just a meaningless number, and the model trains badly. PyTorch's `nn.MSELoss` even emits a `UserWarning` for exactly this case ("Using a target size ... that is different to the input size ... This will likely lead to incorrect results due to broadcasting"). Defend yourself: **assert shapes** before combining arrays, e.g. `assert y_pred.shape == y_true.shape`, and flatten with `.ravel()`, `[:, 0]` or `squeeze` (Day 8).

### Bug 2: forgetting `keepdims=True`

To make each **row** sum to 1 (turning scores into proportions), divide each row by its own total. But `sum(axis=1)` removes axis 1, leaving `(n,)`, which aligns with the *columns*:

```python
S = np.array([[1., 2., 1.],
              [4., 4., 2.]])
tot = S.sum(axis=1)                      # [ 4. 10.]   shape (2,)
S / tot
# ValueError: operands could not be broadcast together with shapes (2,3) (2,)

tot = S.sum(axis=1, keepdims=True)       # [[ 4.], [10.]]   shape (2, 1)
P = S / tot
# [[0.25 0.5  0.25]
#  [0.4  0.4  0.2 ]]
P.sum(axis=1)                            # [1. 1.]
```

`keepdims=True` keeps the reduced axis with size 1, so the result is a column that broadcasts across each row.

!!! warning "Common mistake: square matrices hide the bug"
    Above, the error saved you. With a **square** matrix, the wrong version runs and divides by the wrong totals:

    ```python
    Q = np.array([[1., 3.],
                  [2., 8.]])
    (Q / Q.sum(axis=1)).sum(axis=1)                  # [0.55 1.3 ]   rows don't sum to 1
    (Q / Q.sum(axis=1, keepdims=True)).sum(axis=1)   # [1. 1.]
    ```

    Rule of thumb: reducing along the **last** axis and combining with the original? Use `keepdims=True`. Reducing along axis 0 (per-feature statistics, as in standardization) works without it.

### Bug 3: huge broadcast intermediates

Broadcasting doesn't copy its inputs, but the **result** is full size. The pairwise trick above builds `diff` of shape `(n, m, d)`. For 10,000 queries against 10,000 database vectors with 128 dimensions, in `float64`:

\[
10{,}000 \times 10{,}000 \times 128 \times 8 \text{ bytes} = 102.4 \text{ GB}
\]

Your laptop will freeze or the process will be killed. Two fixes:

- **Process in chunks**: loop over 500 queries at a time, so each `diff` is a manageable size.
- **Use algebra to skip the 3-D intermediate.** Expanding the square, \( \|\mathbf{a} - \mathbf{b}\|^2 = \|\mathbf{a}\|^2 + \|\mathbf{b}\|^2 - 2\,\mathbf{a} \cdot \mathbf{b} \). The last term for all pairs is one matrix multiply (tomorrow's topic), and the largest array is only `(n, m)`:

```python
rng = np.random.default_rng(0)
a = rng.normal(size=(1000, 32))
b = np.random.default_rng(1).normal(size=(2000, 32))

D1 = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))       # builds a 0.51 GB intermediate
sq = (a**2).sum(1)[:, None] + (b**2).sum(1)[None, :] - 2 * a @ b.T  # largest array: (1000, 2000)
D2 = np.sqrt(np.maximum(sq, 0))      # rounding can make sq slightly negative → clip at 0
np.allclose(D1, D2), D1.shape        # (True, (1000, 2000))
```

Notice the second version is *still* broadcasting: a `(1000, 1)` column plus a `(1, 2000)` row. Broadcasting is great; just be aware of the size of what it produces.

---

## Exercises

1. Without running code, write the result shape (or "error") for: `(5, 4)` and `(4,)`; `(5, 4)` and `(5,)`; `(5, 1)` and `(1, 4)`; `(2, 3, 4)` and `(4,)`; `(2, 3, 4)` and `(2, 1, 4)`; `(3,)` and `(3, 1)`. Then check yourself with `np.broadcast_shapes`.
2. **Min-max scaling.** Using the house table `X` from this lesson, rescale each column to the range 0 to 1: \( (x - \min) / (\max - \min) \), with the min and max taken per column. One line of NumPy.
3. **Center each row.** For `S = np.array([[1., 2., 6.], [4., 4., 10.]])`, subtract each row's own mean so every row averages 0. Verify with `.mean(axis=1)`.
4. **Multiplication table.** Build the 9×9 times table (1 to 9) with a single broadcast, no loops. Check that `table[6, 7]` is 56.
5. **Nearest centroid.** Assign each of these 6 points to the nearest of 3 centroids using a pairwise distance matrix: `pts = [[0,0],[1,0],[9,9],[10,8],[0,9],[1,10]]`, `cent = [[0,0],[10,10],[0,10]]`. Expected answer: `[0 0 1 1 2 2]`. (This is the assignment step of k-means, Phase 2.)
6. **Undo the normalization.** Take the normalized image `x` from Real use 3 and convert it back to a `uint8` image. Check that you get exactly the original `img` back. (You'd need this on Android to display what a model "sees".)

??? success "Solutions"
    ```python
    import numpy as np
    np.set_printoptions(precision=3, suppress=True)

    # 1
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

    # 2
    X = np.array([[3, 120, 15],
                  [2,  70, 40],
                  [4, 160,  5],
                  [3, 110, 20]], dtype=float)
    lo, hi = X.min(axis=0), X.max(axis=0)
    X01 = (X - lo) / (hi - lo)
    # [[0.5   0.556 0.286]
    #  [0.    0.    1.   ]
    #  [1.    1.    0.   ]
    #  [0.5   0.444 0.429]]

    # 3
    S = np.array([[1., 2., 6.],
                  [4., 4., 10.]])
    C = S - S.mean(axis=1, keepdims=True)
    # [[-2. -1.  3.]
    #  [-2. -2.  4.]]
    C.mean(axis=1)       # [0. 0.]

    # 4
    n = np.arange(1, 10)
    table = n[:, None] * n[None, :]   # (9, 1) * (1, 9) → (9, 9)
    table[6, 7]          # 56   (row for 7, column for 8)
    table[:3, :5]
    # [[ 1  2  3  4  5]
    #  [ 2  4  6  8 10]
    #  [ 3  6  9 12 15]]

    # 5
    pts = np.array([[0., 0.], [1., 0.], [9., 9.], [10., 8.], [0., 9.], [1., 10.]])
    cent = np.array([[0., 0.], [10., 10.], [0., 10.]])
    D = np.sqrt(((pts[:, None, :] - cent[None, :, :]) ** 2).sum(axis=-1))
    D.shape              # (6, 3)
    D.argmin(axis=1)     # [0 0 1 1 2 2]

    # 6
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (img.astype(np.float32) / 255.0 - mean) / std

    back = np.clip(np.round((x * std + mean) * 255.0), 0, 255).astype(np.uint8)
    back.dtype, np.array_equal(back, img)    # (dtype('uint8'), True)
    ```
    In 6, `np.round` matters: float arithmetic lands on values like 5.999992, and `astype(np.uint8)` truncates, turning it into 5. Without the round, 26,744 of the 150,528 values come back off by one.

## Resources

- [NumPy: Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html): the official explanation, with diagrams and the same rules.
- [NumPy: `numpy.broadcast_to`](https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html): the explicit, read-only broadcast view.
- [PyTorch: Broadcasting semantics](https://pytorch.org/docs/stable/notes/broadcasting.html): the same rules in PyTorch, including the in-place caveat.
- [Python Data Science Handbook: Computation on Arrays: Broadcasting](https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html): free book chapter with good visual examples.
- [CS231n Python/NumPy tutorial](https://cs231n.github.io/python-numpy-tutorial/): see its broadcasting section.
- [Dive into Deep Learning: 2.1 Data Manipulation](https://d2l.ai/chapter_preliminaries/ndarray.html): broadcasting in the context of deep learning code.
- [torchvision transforms](https://pytorch.org/vision/stable/transforms.html): where `Normalize(mean, std)` with the ImageNet statistics comes from.

## Checklist

- [ ] I can state the two rules: align from the right (missing leading dims are 1), and each pair must be equal or contain a 1.
- [ ] I can work out a result shape, or predict an error, by hand.
- [ ] I can explain why broadcasting doesn't copy memory (stride 0).
- [ ] I can standardize a `(samples, features)` table and normalize an `(H, W, 3)` image in one line each.
- [ ] I can build a pairwise distance matrix with `[:, None, :]` and `[None, :, :]`.
- [ ] I check shapes before combining predictions and labels, and I reach for `keepdims=True` on row-wise reductions.
- [ ] I know broadcast results can be huge and how to avoid it.

**Next:** [Day 10: Matrices and matrix multiplication](day-10-matrices-matmul.md)
