# Day 11: Transpose, Identity and Inverse

**Phase 1 · Week 2 · ~150 minutes**

## Objectives

- [ ] Transpose a matrix by hand and in NumPy, predict the new shape, and know when `.T` is a view, when it does nothing (1-D) and what it does to a 3-D array.
- [ ] Use the transpose rules, including \( (AB)^\top = B^\top A^\top \), and read \( \mathbf{x}^\top\mathbf{y} \) and \( \mathbf{x}\mathbf{y}^\top \) as the dot and outer products.
- [ ] Recognise a **symmetric** matrix, prove that \( X^\top X \) is always symmetric, and test symmetry safely on floats.
- [ ] Read a 2×2 matrix as a **transformation** of the plane: its columns are where the basis vectors land, and it obeys the linearity rule.
- [ ] Explain what the identity, diagonal and inverse matrices do, compute a 2×2 inverse by hand, and use \( (AB)^{-1} = B^{-1}A^{-1} \).
- [ ] Use the determinant to tell how a matrix scales area, whether it flips the plane, and whether it has an inverse.
- [ ] Recognise an **orthogonal** matrix and use the fact that its inverse is its transpose.
- [ ] Read a **condition number**, and explain why `det(M) == 0` is the wrong singularity test.
- [ ] Solve a linear system with `np.linalg.solve` (and an over-determined one with `np.linalg.lstsq`), and explain why it beats `inv(A) @ b`.
- [ ] Explain why ML training almost never inverts a matrix.

**Prerequisites.** What this lesson uses from earlier days:

- [Day 2a](day-02a-strings.md): the f-string format specs `:.1e` (scientific notation, one decimal) and `:.0f` (no decimals).
- [Day 2b](day-02b-collections.md): unpacking a tuple into several names (`a, b = ...`).
- [Day 3a](day-03a-classes-modules.md): `time.perf_counter()` for timing a piece of code.
- [Day 3b](day-03b-errors-files-json.md): `try` / `except ... as e`, and the exception hierarchy (a subclass is caught by its parent's `except`).
- [Day 4](day-04-numpy-arrays.md): `np.eye`, `np.linspace`, `np.ones_like`, `rng.normal`; NumPy scalars displaying as `np.float64(...)`; which operations make views.
- [Day 4b](day-04b-numpy-toolkit.md): `np.allclose` and `np.array_equal`, and why `==` on floats is fragile.
- [Day 4c](day-04c-random-io-stats.md): `.round()`.
- [Day 5](day-05-vectors.md): the dot product, the norm `np.linalg.norm`, unit vectors and "orthogonal means dot product zero".
- [Day 8](day-08-shapes-indexing.md): `reshape(-1, 1)`, `x[None, :]` and `x[:, None]`, strides, transposing axes, `np.swapaxes` and `np.column_stack`.
- [Day 9](day-09-broadcasting.md): broadcasting, used once to scale rows.
- [Day 10](day-10-matrices-matmul.md): matrices, `A @ B`, the shape rule `(m, n) @ (n, p) → (m, p)`, \( AB \neq BA \), and the math-vs-NumPy index off-by-one.

---

## The big idea

Yesterday a matrix was a grid of numbers you multiply. Today it gets a second, more useful personality: **a matrix is a machine that moves space**. Feed it a vector, and it hands back a moved vector: stretched, rotated, sheared or flipped.

Once you see matrices that way, a handful of questions come naturally, and each one is a section of this lesson:

- Which machine does nothing? The **identity** matrix \( I \).
- Which machine undoes this one? The **inverse** \( A^{-1} \).
- How much does this machine stretch areas, and does it flip the plane over? The **determinant** \( \det(A) \).
- Which machines move things without distorting them? **Orthogonal** matrices (rotations and mirrors), whose undo button costs nothing.
- How close is this machine to being impossible to undo? The **condition number**.

Plus one bookkeeping tool, the **transpose**, which shows up in every ML formula because it makes shapes line up, and the special matrices it creates: **symmetric** ones, which are their own transpose. Day 23 (rank), Day 24 (eigenvectors) and Day 25 (PCA) all lean on them.

**Analogy.** Think of a photo-editing app. "Rotate 90°" is a matrix. "Do nothing" is the identity. "Undo" is the inverse. "Mirror" is a matrix with a negative determinant. And if you squash the photo to zero height, there's no undo button: the information is gone. That last case, a matrix with determinant 0, is called **singular**. A photo squashed to 0.0000001 of its height can be undone in theory, but every speck of noise gets blown up ten million times: that's a huge condition number.

**A note on how outputs are shown.** As on Days 4, 4b and 5, a bare expression's comment shows exactly what ▶ Run echoes: arrays with commas, and single numbers as NumPy scalars such as `np.float64(5.0)` (Day 4 explains why). Integer scalars show as `np.int64(...)` here; the browser's Python is 32-bit, so ▶ Run shows `np.int32(...)` for those, with the same value ([How lessons work](../../how-lessons-work.md) explains it).

**Part 1: The transpose**

## The transpose

**What it is.** The **transpose** of a matrix \( A \), written \( A^\top \) ("A transpose"), turns its rows into columns. Row 0 of \( A \) becomes column 0 of \( A^\top \), row 1 becomes column 1, and so on. One symbol at a time:

- \( A_{ij} \) is the element in row \( i \), column \( j \) of \( A \) (Day 10).
- \( (A^\top)_{ji} \) is the element in row \( j \), column \( i \) of the transpose.
- The rule says they are the same number: \( (A^\top)_{ji} = A_{ij} \).

A matrix of shape \( (m, n) \) becomes shape \( (n, m) \).

**Why it exists.** Shapes. A matrix multiply only works when the inner sizes match (Day 10's shape rule), and data rarely arrives in the orientation a formula wants. Tomorrow's neural-network layer stores its weights as `(out, in)` and computes `X @ W.T` to get `(batch, in) @ (in, out)`. Papers write the dot product as \( \mathbf{w}^\top\mathbf{x} \). The normal equation for linear regression is built from \( X^\top X \). And on Day 33 you'll see that training a network (the "backward pass") multiplies by \( W^\top \) at every layer. The transpose is everywhere.

**Intuition first.** Picture the matrix printed on a transparent sheet. Hold the top-left and bottom-right corners of the square part and flip the sheet over along that line, the **main diagonal**. The diagonal elements stay where they are; everything else swaps sides.

<figure class="diagram" markdown>
![A 2 by 3 matrix with rows 1 2 3 and 4 5 6 is flipped across a dashed red diagonal line into a 3 by 2 matrix whose columns are 1 2 3 and 4 5 6; the diagonal entries 1 and 5 stay in place](../../assets/img/phase-1/day11-transpose.svg)
<figcaption>The transpose mirrors a matrix across its main diagonal: row 0 becomes column 0.</figcaption>
</figure>

**By hand.**

\[
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}
\;\Rightarrow\;
A^\top = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}
\]

Check the rule on one element: \( A_{13} = 3 \) (row 1, column 3, counting from 1 as math does), and \( (A^\top)_{31} \), row 3 column 1 of the transpose, is also 3.

**In NumPy**, `A.T` is the transpose. The same check in 0-based indices is `A[0, 2]` against `A.T[2, 0]` (Day 10's off-by-one: subtract 1 from each math index).

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])
A.shape          # (2, 3)
A.T
# array([[1, 4],
#        [2, 5],
#        [3, 6]])
A.T.shape        # (3, 2)
A[0, 2], A.T[2, 0]
# (np.int64(3), np.int64(3))
```

That last line is the first NumPy integer scalar of the page: in the browser it shows as `(np.int32(3), np.int32(3))`.

`A.T` is an attribute, not a method, so it has no parentheses. `A.transpose()` and `np.transpose(A)` do the same thing for a 2-D array.

**How it works underneath.** `A.T` copies no numbers. Day 4 listed it among the operations that return a **view** (a new array header over the same memory buffer), and Day 8 showed why: a transpose keeps the buffer and swaps the **strides**, the number of bytes to jump to reach the next element along each axis.

```python
np.shares_memory(A, A.T)     # True    same buffer
A.strides                    # (24, 8)  next row: +24 bytes (3 int64s), next column: +8
A.T.strides                  # (8, 24)  the same two numbers, swapped
```

In the browser, `A` holds 4-byte `int32` values, so ▶ Run shows `(12, 4)` and `(4, 12)`; the swap is the same. The buffer holds `1 2 3 4 5 6`. Walking `A.T` along a row means jumping 24 bytes (from 1 to 4); walking down a column means jumping 8 bytes (from 1 to 2). Transposing a million-element matrix is as cheap as transposing a 2×2, because only the two-number header changes.

Being a view has a consequence you met on Day 4: writing through `A.T` changes `A`.

```python
At = A.T
At[0, 1] = 40      # row 0, column 1 of the transpose is A[1, 0]
A
# array([[ 1,  2,  3],
#        [40,  5,  6]])
A[1, 0] = 4        # put it back
```

If you need an independent transposed matrix, write `A.T.copy()`.

**How it goes wrong: `.T` on a 1-D array.** A NumPy 1-D array has one axis, so there is nothing to swap. Day 8 pointed this out; here is where it bites. A textbook vector is a column, and its transpose is a row, but NumPy's `(3,)` array is neither:

```python
x = np.array([1, 2, 3])
x.T.shape          # (3,)      unchanged, and no error
try:
    x.mT           # .mT, "matrix transpose", refuses instead of silently doing nothing
except ValueError as e:
    print("ValueError:", e)
# ValueError: matrix transpose with ndim < 2 is undefined

xc = x.reshape(-1, 1)    # a real column, shape (3, 1)
xc.T.shape               # (1, 3)    now the transpose is a row
x[None, :].shape         # (1, 3)    Day 8's other way to make a row
```

!!! warning "Common mistake: expecting `.T` to turn a 1-D vector into a column"
    `x.T` on a `(3,)` array returns the same `(3,)` array with no error and no warning. Code that "transposes" a vector to make shapes fit then fails later with a shape error, or worse, broadcasts into something you didn't intend (Day 9). Recognise it by printing `.shape` and seeing `(n,)`. **Fix:** make the vector 2-D on purpose with `x.reshape(-1, 1)` (column) or `x.reshape(1, -1)` (row), or `x[:, None]` / `x[None, :]`.

**`.T` on 3-D arrays.** On an array with more than two axes, `.T` reverses *all* of them, which is rarely what you want (Day 8 showed it scrambling an image's height and width). Attention code (Phase 4) keeps a batch of matrices in one array of shape `(batch, rows, cols)` and needs each matrix transposed, keeping the batch axis first. Two ways to say "swap only the last two axes":

```python
batch = np.arange(24).reshape(2, 3, 4)   # two 3×4 matrices
batch.T.shape                            # (4, 3, 2)   every axis reversed: the batch axis moved
np.swapaxes(batch, -1, -2).shape         # (2, 4, 3)   two 4×3 matrices
batch.mT.shape                           # (2, 4, 3)   the same, as a NumPy 2 attribute
np.array_equal(batch.mT[1], batch[1].T)  # True        each matrix transposed on its own
```

You'll see `k.transpose(-2, -1)` in PyTorch attention code; it is the same move.

## Transpose rules

Four rules cover almost every formula you'll read. The first three follow straight from "rows become columns":

- \( (A^\top)^\top = A \): flipping twice puts everything back.
- \( (A + B)^\top = A^\top + B^\top \): adding cell by cell, then flipping, is the same as flipping each, then adding.
- \( (cA)^\top = c\,A^\top \) for a number \( c \).

The fourth is the one that surprises people: **transposing a product reverses the order.**

\[
(AB)^\top = B^\top A^\top
\]

**Intuition first: socks and shoes.** You put on socks, then shoes. To undo it, you take off shoes first, then socks. Whatever was applied last comes off first. You'll meet the same reversal again for inverses later today.

**Why the order must reverse: shapes.** Take \( A \) of shape \( (2, 3) \) and \( C \) of shape \( (3, 4) \). Then \( AC \) is \( (2, 4) \), so \( (AC)^\top \) must be \( (4, 2) \).

- \( C^\top A^\top \) is \( (4, 3) \) times \( (3, 2) \): inner sizes match, result \( (4, 2) \). Correct shape.
- \( A^\top C^\top \) is \( (3, 2) \) times \( (4, 3) \): inner sizes 2 and 4 don't match. It can't even be computed.

**Why it's true for every element: the index proof.** Look at the element in row \( j \), column \( i \) of \( (AB)^\top \):

\[
\big((AB)^\top\big)_{ji} = (AB)_{ij} = \sum_k A_{ik}B_{kj} = \sum_k (B^\top)_{jk}(A^\top)_{ki} = (B^\top A^\top)_{ji}
\]

Step by step: the first equals sign is the definition of transpose; the second is Day 10's matrix-product formula; the third rewrites each factor with the transpose rule (\( A_{ik} = (A^\top)_{ki} \), \( B_{kj} = (B^\top)_{jk} \)) and swaps the two numbers in each product, which is allowed because ordinary multiplication commutes; and the last is the product formula again, read backwards. Every element agrees, so the matrices are equal.

**By hand.** With \( A \) from above and \( B = \begin{bmatrix} 1 & 0 \\ 2 & 1 \\ 0 & 3 \end{bmatrix} \):

\[
AB = \begin{bmatrix} 1 + 4 + 0 & 0 + 2 + 9 \\ 4 + 10 + 0 & 0 + 5 + 18 \end{bmatrix} = \begin{bmatrix} 5 & 11 \\ 14 & 23 \end{bmatrix}
\qquad
(AB)^\top = \begin{bmatrix} 5 & 14 \\ 11 & 23 \end{bmatrix}
\]

And \( B^\top A^\top \): its top-left cell is row 0 of \( B^\top \), \( [1, 2, 0] \), dotted with column 0 of \( A^\top \), \( [1, 2, 3] \): \( 1 + 4 + 0 = 5 \). The same.

```python
B = np.array([[1, 0],
              [2, 1],
              [0, 3]])
(A @ B).T
# array([[ 5, 14],
#        [11, 23]])
B.T @ A.T
# array([[ 5, 14],
#        [11, 23]])
np.array_equal((A @ B).T, B.T @ A.T)   # True
np.array_equal(A.T.T, A)               # True
```

**How it goes wrong: keeping the original order.** Writing `A.T @ B.T` for \( (AB)^\top \) is a very common slip. Sometimes NumPy catches it, and sometimes it doesn't:

```python
C = np.ones((3, 4))
try:
    A.T @ C.T          # (3, 2) @ (4, 3)
except ValueError as e:
    print("ValueError:", e)
# ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 4 is different from 2)

A.T @ B.T              # (3, 2) @ (2, 3): runs, but it is a (3, 3) matrix, not (AB)^T
# array([[ 1,  6, 12],
#        [ 2,  9, 15],
#        [ 3, 12, 18]])
```

With these shapes, the wrong order happens to be computable and gives a matrix of a different size, with no error. With square matrices it's worse: the shape is right and only the numbers are wrong. When you transpose a product, reverse the order, and check the result's shape against what the formula promises.

## Row vectors, column vectors, and \( \mathbf{x}^\top\mathbf{y} \)

**What it is.** Maths books treat every vector as a **column**: a matrix of shape \( (n, 1) \). Then \( \mathbf{x}^\top \) is a **row** of shape \( (1, n) \), and two products appear:

- \( \mathbf{x}^\top\mathbf{y} \): a row times a column, \( (1, n)(n, 1) \to (1, 1) \). One number: the **dot product** from Day 5.
- \( \mathbf{x}\mathbf{y}^\top \): a column times a row, \( (n, 1)(1, n) \to (n, n) \). A whole matrix, called the **outer product**, whose cell \( (i, j) \) is \( x_i y_j \).

**Why it matters.** You'll see \( \mathbf{w}^\top\mathbf{x} \) in every ML paper: it means "the dot product of \( \mathbf{w} \) and \( \mathbf{x} \)", exactly the neuron from Day 5. The outer product shows up in gradients (Day 33: the gradient of a layer's weights is an outer product of two vectors) and in low-rank updates such as LoRA (Day 23).

**By hand.**

\[
\begin{bmatrix} 1 & 2 & 3 \end{bmatrix}
\begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix} = 1 \cdot 4 + 2 \cdot 5 + 3 \cdot 6 = 32
\qquad
\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}
\begin{bmatrix} 4 & 5 & 6 \end{bmatrix} = \begin{bmatrix} 4 & 5 & 6 \\ 8 & 10 & 12 \\ 12 & 15 & 18 \end{bmatrix}
\]

In NumPy, `x @ y` on two 1-D arrays is already the dot product (Day 5); nothing needs transposing. To get the matrix versions, make the vectors 2-D first:

```python
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
x @ y                # np.int64(32)      Day 5's dot product
xc = x.reshape(-1, 1)
yc = y.reshape(-1, 1)
xc.T @ yc            # array([[32]])     the textbook x^T y: a 1×1 matrix
xc @ yc.T            # the outer product x y^T
# array([[ 4,  5,  6],
#        [ 8, 10, 12],
#        [12, 15, 18]])
np.outer(x, y)       # the same matrix, straight from 1-D arrays
# array([[ 4,  5,  6],
#        [ 8, 10, 12],
#        [12, 15, 18]])
```

**How it goes wrong.** Translating \( \mathbf{x}\mathbf{y}^\top \) literally as `x @ y.T` on 1-D arrays gives `np.int64(32)`, the dot product, because `.T` did nothing to either array. You wanted a 3×3 matrix and got one number, with no error. If a formula has a transpose on the *right* of a vector, you need a 2-D column, or `np.outer`.

## Symmetric matrices

**What it is.** A square matrix is **symmetric** when it equals its own transpose, \( S = S^\top \). Element by element, \( S_{ij} = S_{ji} \): the entry in row \( i \), column \( j \) matches the entry in row \( j \), column \( i \). In the picture of the transpose, flipping across the diagonal changes nothing.

\[
S = \begin{bmatrix} 0 & 3 & 5 \\ 3 & 0 & 4 \\ 5 & 4 & 0 \end{bmatrix}
\qquad S_{12} = S_{21} = 3,\quad S_{13} = S_{31} = 5,\quad S_{23} = S_{32} = 4
\]

**Why it matters.** Symmetric matrices are the most important special matrices in ML:

- a **distance table** (the distance from \( i \) to \( j \) equals the distance from \( j \) to \( i \)), and a **similarity matrix** of embeddings (Day 22);
- a **covariance matrix**, which records how each pair of features varies together (Days 24 and 25: PCA is built on it);
- the matrix \( X^\top X \) in linear regression (below and Day 23);
- on Day 24 you'll see their superpower: a symmetric matrix always has real eigenvalues and **perpendicular** eigenvectors, and NumPy has a faster, more accurate function, `np.linalg.eigh`, reserved for them.

**Intuition first.** Any table where the relationship between \( i \) and \( j \) is mutual is symmetric. "How far is Pune from Delhi" has the same answer both ways. "How many messages did \( i \) send \( j \)" does not, so that table isn't symmetric. The matrix above is a distance table for three cities.

**Example.**

```python
S = np.array([[0, 3, 5],
              [3, 0, 4],
              [5, 4, 0]])        # distances between three cities
np.array_equal(S, S.T)           # True    symmetric

P = np.array([[1, 2],
              [3, 4]])
np.array_equal(P, P.T)           # False   P[0, 1] is 2 but P[1, 0] is 3
P + P.T                          # adding a square matrix to its transpose
# array([[2, 5],
#        [5, 8]])
```

That last result is symmetric for any square matrix, because \( (P + P^\top)^\top = P^\top + P \) by the sum rule. Half of it, \( \tfrac{1}{2}(P + P^\top) \), is called the **symmetric part** of \( P \); you'll use it below to repair a matrix that should be symmetric but isn't, quite.

### \( X^\top X \) is always symmetric

This is the fact Days 23 and 24 rely on, so here it is in full.

**The claim.** For *any* matrix \( X \), of any shape \( (n, k) \), the product \( X^\top X \) is a square \( (k, k) \) matrix, and it is symmetric.

**The proof, using the product rule.** Transpose it and see what you get:

\[
(X^\top X)^\top = X^\top (X^\top)^\top = X^\top X
\]

The first step is \( (AB)^\top = B^\top A^\top \) with \( A = X^\top \) and \( B = X \); the second is "transposing twice puts everything back". The transpose of \( X^\top X \) is \( X^\top X \) itself: symmetric. The same argument shows \( XX^\top \), an \( (n, n) \) matrix, is symmetric too.

**What its entries mean.** Cell \( (i, j) \) of \( X^\top X \) is row \( i \) of \( X^\top \) dotted with column \( j \) of \( X \). But row \( i \) of \( X^\top \) *is* column \( i \) of \( X \). So

\[
(X^\top X)_{ij} = (\text{column } i \text{ of } X) \cdot (\text{column } j \text{ of } X)
\]

That is a second, more intuitive reason for the symmetry: the dot product of column \( i \) with column \( j \) is the same as column \( j \) with column \( i \) (Day 5: the dot product doesn't care about order). And on the diagonal, \( (X^\top X)_{ii} \) is a column dotted with itself, its squared length, so the diagonal is never negative. Statisticians call \( X^\top X \) the **Gram matrix** of the columns: a table of every column dotted with every other.

**By hand.** Take \( X \) with three rows (examples) and two columns (features):

\[
X = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 7 \end{bmatrix}
\qquad
\text{column 1} = [1, 3, 5], \quad \text{column 2} = [2, 4, 7]
\]

- \( (X^\top X)_{11} = 1 + 9 + 25 = 35 \)
- \( (X^\top X)_{12} = 2 + 12 + 35 = 49 \)
- \( (X^\top X)_{21} = 49 \), the same dot product in the other order
- \( (X^\top X)_{22} = 4 + 16 + 49 = 69 \)

\[
X^\top X = \begin{bmatrix} 35 & 49 \\ 49 & 69 \end{bmatrix}
\]

```python
X = np.array([[1, 2],
              [3, 4],
              [5, 7]])
G = X.T @ X                      # (2, 3) @ (3, 2) → (2, 2)
G
# array([[35, 49],
#        [49, 69]])
np.array_equal(G, G.T)           # True
X[:, 0] @ X[:, 1]                # np.int64(49)   column 0 · column 1, the off-diagonal cell
(X ** 2).sum(axis=0)             # array([35, 69])   squared column lengths: the diagonal
(X @ X.T).shape, np.array_equal(X @ X.T, (X @ X.T).T)
# ((3, 3), True)
```

**How it works underneath.** NumPy has no "symmetric matrix" type and no flag for it: a symmetric matrix is an ordinary array that happens to hold the same number in both mirrored cells. It stores all \( k^2 \) numbers. Functions that exploit symmetry, such as `np.linalg.eigh` on Day 24, read only one triangle and assume the other mirrors it, and they don't check.

**How it goes wrong (1): exact equality on floats.** A matrix that is symmetric on paper can come out of float arithmetic very slightly unsymmetric, because the two mirrored cells are computed by different sequences of roundings:

```python
rng = np.random.default_rng(0)
W = rng.normal(size=(4, 3))
C = rng.normal(size=(3, 3))
Ssym = C + C.T                   # symmetric, exactly
M = W @ Ssym @ W.T               # symmetric in theory: (W S W^T)^T = W S^T W^T = W S W^T
np.array_equal(M, M.T)           # False
np.abs(M - M.T).max()            # np.float64(1.1102230246251565e-16)   a rounding error
np.allclose(M, M.T)              # True    the right test for floats (Day 4b)
M_sym = (M + M.T) / 2            # its symmetric part
np.array_equal(M_sym, M_sym.T)   # True    now exactly symmetric
```

Test symmetry with `np.allclose(M, M.T)`, and if a function needs exact symmetry (such as `eigh`), pass it `(M + M.T) / 2`. (Plain `X.T @ X` usually comes out exactly symmetric, because NumPy's BLAS library recognises that pattern, but don't depend on it.)

**How it goes wrong (2): assuming products of symmetric matrices are symmetric.** They usually aren't. \( (PQ)^\top = Q^\top P^\top = QP \), which is a different matrix whenever \( P \) and \( Q \) don't commute (Day 10):

```python
P2 = np.array([[2., 1.], [1., 3.]])     # symmetric
Q2 = np.array([[1., 0.], [0., 4.]])     # symmetric
P2 @ Q2
# array([[ 2.,  4.],
#        [ 1., 12.]])                    not symmetric: 4 vs 1
```

The forms that *are* always symmetric are \( X^\top X \), \( XX^\top \), \( P + P^\top \), and \( W S W^\top \) when \( S \) is symmetric.

**Part 2: Matrices move space**

## Matrices move space

**What it is.** In 2-D, every vector is built from two **basis vectors**:

- \( \hat{\imath} \) ("i-hat") \( = [1, 0] \), one step right;
- \( \hat{\jmath} \) ("j-hat") \( = [0, 1] \), one step up.

The vector \( [2, 3] \) is "2 of \( \hat{\imath} \) plus 3 of \( \hat{\jmath} \)". Now multiply a 2×2 matrix by each basis vector:

\[
\begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} a \\ c \end{bmatrix}
\qquad
\begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} b \\ d \end{bmatrix}
\]

**The first column is where \( \hat{\imath} \) lands. The second column is where \( \hat{\jmath} \) lands.** Every other vector is a mix of \( \hat{\imath} \) and \( \hat{\jmath} \), so it lands at the same mix of the new columns: \( [2, 3] \) lands at 2 × (column 1) + 3 × (column 2). That is Day 10's "matrix times vector is a weighted sum of the columns", seen as geometry.

**Why it matters.** This picture is how you'll *read* matrices from now on. A weight matrix in a network moves its input vectors into a new space; PCA (Day 25) finds the rotation that lines data up with its main directions; image augmentation rotates and shears; Android's `Matrix` positions every bitmap you draw.

<figure class="diagram" markdown>
![Animation: the unit square with i-hat in red and j-hat in blue starts as the identity, morphs into a parallelogram under the shear matrix with columns [1, 0] and [1, 1], returns, then rotates 90 degrees under the matrix with columns [0, 1] and [-1, 0]; the matching matrix is shown on the right with each column colored like the basis vector it moves](../../assets/img/phase-1/day11-transform-morph.svg)
<figcaption>Animated: the matrix's columns are exactly where î (red) and ĵ (blue) end up; the square follows them.</figcaption>
</figure>

### Linearity: the rule every matrix obeys

**What it is.** A transformation \( T \) (a function from vectors to vectors) is **linear** when, for any vectors \( \mathbf{u}, \mathbf{v} \) and any numbers \( a, b \),

\[
T(a\mathbf{u} + b\mathbf{v}) = a\,T(\mathbf{u}) + b\,T(\mathbf{v})
\]

In words: transforming a mix gives the same mix of the transformed vectors. Every matrix obeys it, because \( A(a\mathbf{u} + b\mathbf{v}) = aA\mathbf{u} + bA\mathbf{v} \) is the distributive law of matrix multiplication. Geometrically, it means grid lines stay straight, parallel and evenly spaced.

**Why the origin must stay put.** Set \( a = b = 0 \): the rule says \( T(\mathbf{0}) = \mathbf{0} \). And indeed \( A\mathbf{0} = \mathbf{0} \) for every matrix \( A \). So a **translation** (shift everything by a fixed amount, \( \mathbf{v} \mapsto \mathbf{v} + \mathbf{t} \)) is *not* linear, and no 2×2 matrix can do it. That is why Android's `Matrix` is 3×3: the Android bridge below explains the trick.

```python
A = np.array([[2., 1.],
              [1., 3.]])        # A now names this 2×2 matrix for the rest of the lesson
u = np.array([1., 0.])
v = np.array([1., 2.])
A @ (2 * u + 3 * v)              # array([16., 23.])
2 * (A @ u) + 3 * (A @ v)        # array([16., 23.])   linearity: the same
A @ np.zeros(2)                  # array([0., 0.])     the origin never moves

shift = np.array([1., 0.])
def translate(p):
    return p + shift
translate(np.zeros(2))           # array([1., 0.])     the origin moved: not linear
```

### Four transformations with real numbers

Each of these is defined by where it sends \( \hat{\imath} \) and \( \hat{\jmath} \); write those as columns and you have the matrix.

- **Scale** \( S = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix} \): \( \hat{\imath} \to [2, 0] \), \( \hat{\jmath} \to [0, 3] \). Stretches x by 2 and y by 3.
- **Rotate 90°** \( R = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} \): \( \hat{\imath} \to [0, 1] \) (now pointing up), \( \hat{\jmath} \to [-1, 0] \) (now pointing left). A quarter turn counterclockwise.
- **Shear** \( H = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix} \): \( \hat{\imath} \) stays, \( \hat{\jmath} \to [1, 1] \). Each point slides right by its height, like italic text.
- **Reflect** \( F = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} \): \( \hat{\imath} \) stays, \( \hat{\jmath} \to [0, -1] \). A mirror across the x-axis.

**Rotation by any angle.** Rotate by \( \theta \) (theta, an angle measured counterclockwise). On the unit circle, \( \hat{\imath} \) goes to \( [\cos\theta, \sin\theta] \); \( \hat{\jmath} \) starts a quarter turn ahead of \( \hat{\imath} \) and stays that way, so it goes to \( [-\sin\theta, \cos\theta] \). Those are the columns:

\[
R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
\qquad
R(90°) = \begin{bmatrix} \cos 90° & -\sin 90° \\ \sin 90° & \cos 90° \end{bmatrix} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}
\]

By hand, \( R(90°) \) sends \( [2, 1] \) to \( 2 \cdot [0, 1] + 1 \cdot [-1, 0] = [-1, 2] \).

```python
S = np.array([[2, 0], [0, 3]])
S @ np.array([1, 0]), S @ np.array([0, 1]), S @ np.array([1, 1])
# (array([2, 0]), array([0, 3]), array([2, 3]))

theta = np.pi / 2                      # 90 degrees in radians
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
R @ np.array([1, 0])
# array([6.123234e-17, 1.000000e+00])
np.cos(np.pi / 2)                      # np.float64(6.123233995736766e-17)
(R @ np.array([2, 1])).round(3)
# array([-1.,  2.])                     [2, 1] turned a quarter to [-1, 2]

H = np.array([[1, 1], [0, 1]])         # shear
square = np.array([[0, 1, 1, 0],       # the 4 corners of the unit square,
                   [0, 0, 1, 1]])      # one corner per column
H @ square                             # the top corners slid right: a parallelogram
# array([[0, 1, 2, 1],
#        [0, 0, 1, 1]])

F = np.array([[1, 0], [0, -1]])        # reflect
F @ np.array([2, 3])                   # array([ 2, -3])
```

**Why `6.123234e-17` and not 0?** `np.pi` is the float64 closest to π, not π itself (it's off by about \( 10^{-16} \)), so `np.pi / 2` is a hair away from a true right angle, and its cosine is a hair away from 0. The `.round(3)` call (Day 4c) hides that noise when you want to read the result.

Notice the trick in `H @ square`: putting many points as the columns of one matrix transforms them all in one multiply. That's how a neural-network layer processes a whole batch (Day 12).

### Doing two transformations in a row is a product

"Rotate, then scale" means \( S(R\mathbf{v}) \), and by associativity (Day 10) that is \( (SR)\mathbf{v} \). Read right to left: **the matrix nearest the vector acts first.** Order matters, as Day 10 warned:

```python
R = np.array([[0, -1], [1, 0]])        # R now names the exact integer 90° rotation
M = S @ R                              # rotate first, then scale
M @ np.array([1, 0])                   # array([0, 3])   î -> [0, 1] -> y stretched by 3
R @ S @ np.array([1, 0])               # array([0, 2])   scale first (î -> [2, 0]), then rotate
```

## The identity matrix

**What it is.** You met `np.eye(n)` on Day 4 as "the identity matrix". In today's picture, the **identity** \( I \) is the transformation that does nothing: \( \hat{\imath} \) stays at \( [1, 0] \) and \( \hat{\jmath} \) at \( [0, 1] \). So its columns are the basis vectors themselves: 1s on the diagonal, 0s everywhere else.

\[
I_2 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
\qquad
I\mathbf{v} = \mathbf{v}
\qquad
AI = IA = A
\]

**Why it exists.** It is the matrix version of the number 1 (\( 5 \times 1 = 5 \)), and it's the target the inverse aims for: "\( A^{-1} \) is the matrix that turns \( A \) into \( I \)". It is always square, \( n \times n \); the subscript, often left out, gives \( n \). It also appears on its own in ML: ridge regression adds \( \lambda I \) to \( X^\top X \) (Day 23 mentions it), and residual connections in transformers compute \( \mathbf{x} + f(\mathbf{x}) \), which is \( (I + f)\mathbf{x} \) in disguise.

```python
I = np.eye(2)
I
# array([[1., 0.],
#        [0., 1.]])
np.array_equal(A @ I, A), np.array_equal(I @ A, A)   # (True, True)
np.eye(3, dtype=int)
# array([[1, 0, 0],
#        [0, 1, 0],
#        [0, 0, 1]])
```

**How it goes wrong.** `np.eye` is float by default (Day 4), so `np.eye(2) @ S` turns an integer matrix into floats. And `np.identity(n)` is an older alias of the same thing; you'll see both. The more serious slip is `np.ones((2, 2))` when you meant the identity: it is a matrix of all ones, which squashes the plane onto a line (you'll see why that's fatal in a moment).

## Diagonal matrices and `np.diag`

**What it is.** A **diagonal matrix** has non-zero entries only on its main diagonal. The identity is one; the scale matrix \( S \) is another. Geometrically, it stretches each axis by its own factor and never mixes them.

**Why it matters.** Diagonal matrices are the easy case of everything in this lesson: you can multiply, invert and take determinants of them in your head. They appear as per-feature scaling (standardising data), as the eigenvalue matrix on Day 24, and in Day 10's Exercise 4 (diagonal matrices commute with each other).

`np.diag` works in two directions:

- given a **1-D** array, it **builds** a diagonal matrix with those values;
- given a **2-D** array, it **extracts** the diagonal as a 1-D array.

```python
D = np.diag([2., 3.])            # build
D
# array([[2., 0.],
#        [0., 3.]])
np.diag(D)                       # extract: array([2., 3.])
np.diag(G)                       # array([35, 69])   the diagonal of X^T X from earlier

D @ P                            # D on the LEFT scales P's rows: row 0 × 2, row 1 × 3
# array([[ 2.,  4.],
#        [ 9., 12.]])
P * np.array([[2.], [3.]])       # the same, by broadcasting a column (Day 9), and cheaper
# array([[ 2.,  4.],
#        [ 9., 12.]])
P @ D                            # D on the RIGHT scales P's columns
# array([[ 2.,  6.],
#        [ 6., 12.]])
```

The inverse of a diagonal matrix undoes each stretch separately, so it is the diagonal of **reciprocals**: \( \operatorname{diag}(2, 3)^{-1} = \operatorname{diag}(\tfrac{1}{2}, \tfrac{1}{3}) \). Its determinant (coming up) is the product of the diagonal, \( 2 \times 3 = 6 \).

```python
np.diag(1 / np.diag(D))          # reciprocals on the diagonal
# array([[0.5       , 0.        ],
#        [0.        , 0.33333333]])
np.allclose(np.diag(1 / np.diag(D)), np.linalg.inv(D))   # True
```

**How it goes wrong: `np.diag` of a matrix doesn't give a diagonal matrix.** People reach for `np.diag(G)` expecting "G with its off-diagonal entries zeroed" and get a 1-D array instead, which then broadcasts strangely. To keep only the diagonal *as a matrix*, apply it twice:

```python
np.diag(G).shape                 # (2,)    extracted, 1-D
np.diag(np.diag(G))              # extract, then build
# array([[35,  0],
#        [ 0, 69]])
```

**Part 3: Undoing a transformation**

## The inverse: undo

**What it is.** The **inverse** of a square matrix \( A \), written \( A^{-1} \) ("A inverse"), is the transformation that puts everything back. Apply \( A \), then \( A^{-1} \), and you've done nothing:

\[
A^{-1} A = A A^{-1} = I
\]

A matrix that has an inverse is called **invertible** (or non-singular).

**Why it exists.** "Given where a point ended up, where did it start?" A tap on a zoomed photo needs mapping back to the original pixel (the Android bridge below). Solving equations \( A\mathbf{x} = \mathbf{b} \) is the same question. And the formula for linear regression is written with an inverse.

**Intuition first.** It's the matrix version of \( 5 \times \tfrac{1}{5} = 1 \). If \( A \) rotates 90° counterclockwise, \( A^{-1} \) rotates 90° clockwise. If \( A \) doubles x, \( A^{-1} \) halves it. If \( A \) slides points right by their height, \( A^{-1} \) slides them left by their height.

**The 2×2 formula.** For \( A = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \): swap \( a \) and \( d \), negate \( b \) and \( c \), and divide everything by \( ad - bc \):

\[
A^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}
\]

You can confirm it by multiplying: the top-left cell of \( A \cdot \begin{bmatrix} d & -b \\ -c & a \end{bmatrix} \) is \( ad - bc \), the top-right is \( -ab + ba = 0 \), and so on, giving \( (ad - bc) I \). Dividing by \( ad - bc \) leaves \( I \).

**Worked example** with \( A = \begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix} \): \( ad - bc = 2 \cdot 3 - 1 \cdot 1 = 5 \), so

\[
A^{-1} = \frac{1}{5} \begin{bmatrix} 3 & -1 \\ -1 & 2 \end{bmatrix} = \begin{bmatrix} 0.6 & -0.2 \\ -0.2 & 0.4 \end{bmatrix}
\]

Check one entry of \( AA^{-1} \): row 1 of \( A \) dotted with column 1 of \( A^{-1} \) is \( 2(0.6) + 1(-0.2) = 1 \).

```python
A_inv = np.linalg.inv(A)
A_inv
# array([[ 0.6, -0.2],
#        [-0.2,  0.4]])
A @ A_inv                                # the identity, up to rounding
# array([[ 1.00000000e+00,  0.00000000e+00],
#        [-5.55111512e-17,  1.00000000e+00]])
np.allclose(A @ A_inv, np.eye(2))        # True

v = np.array([1., 2.])
w = A @ v
w                  # array([4., 7.])   move it...
A_inv @ w          # array([1., 2.])   ...and move it back
```

`np.linalg` is NumPy's linear-algebra module: `inv`, `solve`, `det`, `norm` (Day 5) and the rest of today's functions live there.

**How it works underneath.** For anything bigger than 2×2 there's no handy formula; `inv` runs **Gaussian elimination**, the same row-by-row elimination you'd do by hand to solve equations (you'll do one in the solving section), organised as an **LU decomposition** (named here; see Going further). It does this with LAPACK, a compiled Fortran library, the way `@` uses BLAS (Day 10). Notice the tiny `-5.55111512e-17` in `A @ A_inv`: \( 0.6 \), \( 0.2 \) and \( 0.4 \) aren't exact in binary (Day 2), so the product is the identity only up to rounding. Always compare with `np.allclose`, never `np.array_equal`.

**How it goes wrong (1): only square matrices have inverses.** A `(3, 2)` matrix maps 2-D vectors into 3-D space; it can't reach most of 3-D, so there's no way back from an arbitrary 3-D point. NumPy refuses:

```python
try:
    np.linalg.inv(np.ones((3, 2)))
except np.linalg.LinAlgError as e:
    print("LinAlgError:", e)
# LinAlgError: Last 2 dimensions of the array must be square
```

`np.linalg.LinAlgError` is NumPy's own exception class for linear-algebra failures, defined inside the `np.linalg` module, so you catch it by its full name. Catching an exception that a *library* defines works exactly like catching a built-in one (Day 3b). (For non-square problems there's a different tool, the least-squares solve, at the end of this lesson.)

!!! warning "Common mistake: `A ** -1` is not the inverse"
    `**` is element-wise, like `*` versus `@` on Day 10. On a float matrix, `A ** -1` gives the reciprocal of **each element**, which is not the inverse at all. On an integer matrix it doesn't even run:

    ```python
    Pf = np.array([[1., 2.], [3., 4.]])
    Pf ** -1                     # element reciprocals, NOT the inverse
    # array([[1.        , 0.5       ],
    #        [0.33333333, 0.25      ]])
    np.linalg.inv(Pf)            # the real inverse
    # array([[-2. ,  1. ],
    #        [ 1.5, -0.5]])
    try:
        np.array([[1, 2], [3, 4]]) ** -1
    except ValueError as e:
        print("ValueError:", e)
    # ValueError: Integers to negative integer powers are not allowed.
    ```

    The same goes for `1 / A`. **Fix:** `np.linalg.inv(A)`, or better still `np.linalg.solve` (Part 4).

### Rules for inverses

**The product rule: socks and shoes again.** To undo "first \( B \), then \( A \)" (the product \( AB \), read right to left), undo \( A \) first, then \( B \):

\[
(AB)^{-1} = B^{-1} A^{-1}
\]

Check it by multiplying: \( (AB)(B^{-1}A^{-1}) = A(BB^{-1})A^{-1} = AIA^{-1} = AA^{-1} = I \). The order reverses for the same reason it did for transposes.

Two more rules, each one line to check:

- \( (A^{-1})^{-1} = A \): undoing the undo is the original.
- \( (A^\top)^{-1} = (A^{-1})^\top \): transpose \( AA^{-1} = I \) with the product rule to get \( (A^{-1})^\top A^\top = I^\top = I \).

```python
Si, Ri = np.linalg.inv(S), np.linalg.inv(R)
np.allclose(np.linalg.inv(S @ R), Ri @ Si)     # True    reversed order
np.allclose(np.linalg.inv(S @ R), Si @ Ri)     # False   same order: wrong
np.allclose(np.linalg.inv(np.linalg.inv(A)), A)            # True
np.allclose(np.linalg.inv(A.T), np.linalg.inv(A).T)        # True
```

## The determinant: how much area changes

**What it is.** That number \( ad - bc \) in the inverse formula has a name: the **determinant**, \( \det(A) \). It tells you **how much the transformation scales area**. The unit square (area 1) lands on a parallelogram whose area is \( |\det(A)| \), and every other shape scales by the same factor. Its sign says whether the plane was flipped over (next section).

\[
\det \begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc
\]

**Why it exists.** It answers "does this matrix have an inverse?" in one number (yes exactly when \( \det \neq 0 \)), it's the denominator of the inverse formula, and on Day 24 it's the tool that finds eigenvalues. In probability (Month 2 and beyond), it measures how a change of variables stretches density.

**Intuition first: why \( ad - bc \) is an area.** Take the parallelogram spanned by the columns \( [a, c] \) and \( [b, d] \), with \( b = c = 0 \): it's an \( a \times d \) rectangle, area \( ad \). Tilting the columns (non-zero \( b \) and \( c \)) slides part of that area away, and the amount lost works out to \( bc \). 3Blue1Brown's chapter 6 (Resources) animates the full argument.

<figure class="diagram" markdown>
![Animation in three panels: the unit square with area 1; under A with columns [2, 1] and [1, 3] it grows into a parallelogram of area 5; under the matrix with columns [1, 2] and [2, 4] both basis vectors land on the same line and the square collapses into a line segment of area 0](../../assets/img/phase-1/day11-determinant.svg)
<figcaption>Animated: det = 5 means every area grows 5×; det = 0 means the plane is squashed onto a line.</figcaption>
</figure>

**By hand.** \( \det(A) = 2 \cdot 3 - 1 \cdot 1 = 5 \). The scale \( S \): \( 2 \cdot 3 - 0 = 6 \). The rotation \( R \): \( 0 \cdot 0 - (-1)(1) = 1 \). The shear \( H \): \( 1 \cdot 1 - 1 \cdot 0 = 1 \) (sliding the top of a square sideways doesn't change its area). The reflection \( F \): \( 1 \cdot (-1) - 0 = -1 \).

```python
np.linalg.det(A)          # np.float64(5.000000000000001)   rounding again; it's 5
np.linalg.det(S), np.linalg.det(R), np.linalg.det(H), np.linalg.det(F)
# (np.float64(6.0), np.float64(1.0), np.float64(1.0), np.float64(-1.0))
```

`np.linalg.det` always returns a float, even for an integer matrix, because LAPACK computes it from the same elimination `inv` uses, not from the \( ad - bc \) formula.

### Negative determinants: the plane is flipped

**What it is.** A negative determinant means the transformation **reverses orientation**: it turns the plane over, like a mirror. Before, \( \hat{\jmath} \) is a quarter turn *counterclockwise* from \( \hat{\imath} \). After a transformation with negative determinant, the image of \( \hat{\jmath} \) is *clockwise* from the image of \( \hat{\imath} \). The size \( |\det| \) is still the area factor.

**Why it matters.** It's the difference between a rotation and a mirror image, which no amount of rotating can fix: a left glove turned any way you like is still a left glove. In graphics, orientation decides which side of a triangle faces the camera; on Day 24, a matrix with negative determinant has an eigenvalue that flips a direction.

<figure class="diagram" markdown>
![Two panels. Left: the identity, with i-hat pointing right, j-hat pointing up, an orange arc going counterclockwise from i-hat to j-hat, and an upright letter F; det(I) = +1. Right: after the reflection F = [[1, 0], [0, -1]], i-hat still points right but j-hat points down, the arc goes clockwise, and the letter F is mirrored upside down; det(F) = -1](../../assets/img/phase-1/day11-orientation-flip.svg)
<figcaption>A negative determinant flips the plane: ĵ ends up clockwise from î, and the shape becomes its mirror image.</figcaption>
</figure>

**The determinant as an orientation test.** Put two vectors \( \mathbf{u}, \mathbf{v} \) side by side as the columns of a matrix. Its determinant is positive when \( \mathbf{v} \) is counterclockwise from \( \mathbf{u} \) (turning the short way), negative when it's clockwise, and zero when they're on the same line. By hand: \( \mathbf{u} = [1, 0] \), \( \mathbf{v} = [1, 1] \) gives \( 1 \cdot 1 - 1 \cdot 0 = 1 \); \( \mathbf{v} = [1, -1] \) gives \( 1 \cdot (-1) - 1 \cdot 0 = -1 \).

```python
u = np.array([1, 0])
np.linalg.det(np.column_stack([u, np.array([1, 1])]))    # np.float64(1.0)    [1, 1] is counterclockwise of u
np.linalg.det(np.column_stack([u, np.array([1, -1])]))   # np.float64(-1.0)   [1, -1] is clockwise of u
np.linalg.det(R @ F)      # np.float64(-1.0)   a flip followed by a rotation is still a flip
```

`np.column_stack` (Day 8) places 1-D arrays side by side as columns.

**How it goes wrong.** Two flips cancel: \( \det(FF) = (-1)(-1) = 1 \), so composing an even number of mirrors gives something with no flip at all. Code that checks "is this transform a mirror?" must look at the sign of the determinant of the *whole* product, not count the mirror steps it applied. And on screens, the y-axis points **down** (the Android bridge below), so "counterclockwise" in maths looks clockwise on a phone: the sign flips again.

### Determinant 0: no inverse

Look at \( \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix} \) (called `Sg`, for "singular", in the code). Its second column \( [2, 4] \) is twice the first \( [1, 2] \), so \( \hat{\imath} \) and \( \hat{\jmath} \) land on the same line, and the whole plane collapses onto that line. Area becomes 0: \( \det = 1 \cdot 4 - 2 \cdot 2 = 0 \).

Many different inputs now give the same output: for example `[2, -1]` lands on `[0, 0]`, the same place as the origin. Given an output, you can't tell which input it came from, so no undo can exist. In the formula, you'd be dividing by \( ad - bc = 0 \). A matrix like this is called **singular**.

```python
Sg = np.array([[1., 2.], [2., 4.]])
np.linalg.det(Sg)         # np.float64(0.0)
Sg @ np.array([2., -1.])  # array([0., 0.])   a non-zero input that vanishes
try:
    np.linalg.inv(Sg)
except np.linalg.LinAlgError as e:
    print("LinAlgError:", e)
# LinAlgError: Singular matrix
issubclass(np.linalg.LinAlgError, ValueError)   # True
```

The last line matters when you read other people's code: `LinAlgError` is a subclass of `ValueError`, so an `except ValueError:` (Day 3b's hierarchy) also catches it. Catch `np.linalg.LinAlgError` when you mean "the matrix was singular", so that unrelated `ValueError`s still surface.

### Determinants of products and inverses

Areas scale, then scale again: **\( \det(AB) = \det(A)\det(B) \)**. And the undo must shrink by exactly the factor the original grew: **\( \det(A^{-1}) = 1/\det(A) \)**. By hand: \( \det(SR) = 6 \times 1 = 6 \), and \( \det(A^{-1}) = 1/5 = 0.2 \).

```python
np.linalg.det(S @ R)                 # np.float64(6.0)    6 × 1
np.linalg.det(np.linalg.inv(A))      # np.float64(0.2)    1 / 5
1 / np.linalg.det(A)                 # np.float64(0.19999999999999996)   the same, up to rounding
```

The product rule also explains why the two flips above cancel, and why a product containing one singular matrix is singular: one factor of 0 makes the whole product 0.

## Orthogonal matrices: the inverse is the transpose

**What it is.** A square matrix \( Q \) is **orthogonal** when

\[
Q^\top Q = I
\]

That says, one cell at a time, that column \( i \) dotted with column \( j \) is 1 when \( i = j \) and 0 otherwise (the Gram-matrix reading of \( X^\top X \) from Part 1). So: **every column is a unit vector, and every pair of columns is perpendicular** (Day 5's "orthogonal means dot product zero"). Multiply both sides by \( Q^{-1} \) on the right and you get the headline fact:

\[
Q^{-1} = Q^\top
\]

**Why it matters.** Rotations and reflections are orthogonal: they move things without stretching or squashing them, so lengths, angles and distances are preserved (\( \|Q\mathbf{v}\| = \|\mathbf{v}\| \)). Their inverse costs nothing: a transpose is a view, no computation at all. PCA (Day 25) produces an orthogonal matrix of directions; the eigenvectors of a symmetric matrix (Day 24) form one; numerically stable algorithms (QR, SVD) are built from them because they never amplify rounding errors.

**Intuition first.** Undoing a 30° counterclockwise turn is a 30° clockwise turn, \( R(-\theta) \). Since \( \cos(-\theta) = \cos\theta \) and \( \sin(-\theta) = -\sin\theta \), \( R(-\theta) \) is \( R(\theta) \) with the two sine entries swapped: exactly \( R(\theta) \) mirrored across its diagonal, its transpose.

**By hand** with \( R(90°) \):

\[
R^\top R = \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix} \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}
= \begin{bmatrix} 0 + 1 & 0 + 0 \\ 0 + 0 & 1 + 0 \end{bmatrix} = I
\]

```python
th = np.pi / 6                         # 30 degrees
Q = np.array([[np.cos(th), -np.sin(th)],
              [np.sin(th),  np.cos(th)]])
Q.T @ Q                                # the identity, up to rounding
# array([[1.00000000e+00, 7.43708407e-18],
#        [7.43708407e-18, 1.00000000e+00]])
np.allclose(Q.T @ Q, np.eye(2))        # True
np.allclose(np.linalg.inv(Q), Q.T)     # True    the inverse IS the transpose
np.linalg.norm(Q @ np.array([3., 4.])) # np.float64(5.0)   length 5 before and after
np.linalg.det(Q)                       # np.float64(1.0)
F.T @ F                                # the mirror is orthogonal too
# array([[1, 0],
#        [0, 1]])
```

**How it works underneath.** Because \( \det(Q^\top) = \det(Q) \) (true for every square matrix) and \( \det(Q^\top Q) = \det(I) = 1 \), the product rule gives \( \det(Q)^2 = 1 \): an orthogonal matrix has determinant \( +1 \) (a rotation) or \( -1 \) (a reflection, possibly combined with a rotation). For a square \( Q \), \( Q^\top Q = I \) also forces \( QQ^\top = I \), so the *rows* are perpendicular unit vectors too.

**How it goes wrong: perpendicular is not enough.** The columns must also have length 1. Here the columns \( [1, 1] \) and \( [-2, 2] \) are perpendicular (\( -2 + 2 = 0 \)) but have lengths \( \sqrt{2} \) and \( \sqrt{8} \):

```python
Bq = np.array([[1., -2.],
               [1.,  2.]])
Bq.T @ Bq                    # perpendicular columns give zeros off the diagonal, but not 1s on it
# array([[2., 0.],
#        [0., 8.]])
np.allclose(np.linalg.inv(Bq), Bq.T)          # False
lengths = np.linalg.norm(Bq, axis=0)          # the length of each column
lengths                      # array([1.41421356, 2.82842712])
Qn = Bq / lengths            # broadcasting divides each column by its length (Day 9)
np.allclose(Qn.T @ Qn, np.eye(2))             # True    now orthogonal
```

`np.linalg.norm(M, axis=0)` takes the norm down each column (Day 5's norm, with Day 4's `axis` rule). The confusing naming is maths' fault: a matrix with perpendicular but non-unit columns has "orthogonal columns" yet is *not* an "orthogonal matrix". Some books say "orthonormal" to make the unit length explicit.

## The condition number: how close to singular?

**What it is.** The **condition number** of a matrix, \( \kappa(A) \) (kappa), measures how much it can amplify small errors when you undo it. Precisely, for the common 2-norm version that NumPy computes, it is the ratio of the most a matrix stretches any vector to the least it stretches any vector:

\[
\kappa(A) = \frac{\text{largest stretch}}{\text{smallest stretch}} \;\ge 1
\]

**Why it exists.** Real data and floats are never exact. If you solve \( A\mathbf{x} = \mathbf{b} \) and \( \mathbf{b} \) carries a tiny relative error, the answer \( \mathbf{x} \) can carry up to \( \kappa(A) \) times that relative error. Rule of thumb: **you lose about \( \log_{10}\kappa \) of your significant digits.** float64 holds about 16 significant digits, so:

- \( \kappa \approx 1 \) to \( 100 \): perfectly safe (orthogonal matrices have \( \kappa = 1 \));
- \( \kappa \approx 10^{8} \): about half your digits are noise;
- \( \kappa \approx 10^{16} \) or more: no digits left; for float64 the matrix is **effectively singular**, even if its determinant isn't 0. A truly singular matrix has \( \kappa = \infty \) in exact arithmetic.

Day 23 uses this when nearly duplicated features make \( X^\top X \) badly conditioned.

**Intuition first.** Picture a machine that stretches x by 10 and squashes y to 0.1. Undoing it must multiply the y-direction by 10 and divide x by 10. A small wobble in the y-part of the output becomes a 10× bigger wobble in the answer, while the x-part shrinks 10×. Relative to each other, that's a factor of \( 10 / 0.1 = 100 \).

**By hand.** For a diagonal matrix the stretches are the diagonal entries (their sizes): \( \kappa(\operatorname{diag}(10, 0.1)) = 10 / 0.1 = 100 \). For the identity or any rotation, every vector keeps its length: \( \kappa = 1 / 1 = 1 \).

```python
np.linalg.cond(np.diag([10., 0.1]))   # np.float64(100.0)
np.linalg.cond(Q)                     # np.float64(1.0)    rotations never amplify errors
np.linalg.cond(A)                     # np.float64(2.6180339887498953)   comfortably safe
np.linalg.cond(Sg)                    # np.float64(4.804857307547117e+16)   singular: rounding keeps it from being inf
```

**What a huge condition number does.** Here is a matrix whose determinant looks completely ordinary, but whose two rows are almost identical:

```python
N = 1e5 * np.array([[1., 1.],
                    [1., 1. + 1e-10]])
np.linalg.det(N)                  # np.float64(1.0000003385357563)   looks harmless
np.linalg.cond(N)                 # np.float64(39999923889.78555)    about 4e10: lose ~10 of 16 digits
b = N @ np.array([1., 1.])        # so the exact answer is [1, 1]
np.linalg.solve(N, b)             # array([0.99999854, 1.00000146])   only ~6 digits right
np.linalg.solve(N, b + np.array([0., 1e-5]))   # nudge b in its 11th digit...
# array([0., 2.])                                ...and the answer changes completely
```

A change of \( 10^{-5} \) in a number about \( 2 \times 10^{5} \), a relative change of \( 5 \times 10^{-11} \), moved the answer from \( [1, 1] \) to \( [0, 2] \). That's what \( \kappa \approx 4 \times 10^{10} \) means.

!!! warning "Common mistake: testing `det == 0`"
    It's tempting to guard an inverse with `if np.linalg.det(M) == 0:`. The determinant is the wrong tool, in both directions. It also measures *scale*, not closeness to singular: shrink a perfectly good matrix and its determinant collapses. And the matrix `N` above is nearly singular while its determinant is a comfortable 1.0.

    ```python
    np.linalg.det(0.1 * np.eye(30))   # np.float64(1.0000000000000165e-30)   tiny...
    np.linalg.cond(0.1 * np.eye(30))  # np.float64(1.0)    ...yet perfectly conditioned
    ```

    `0.1 * np.eye(30)` shrinks each of 30 axes by 0.1, so the volume shrinks by \( 0.1^{30} = 10^{-30} \), but undoing it is trivial and exact. **Fix:** use `np.linalg.cond(M)` (compare it with about \( 10^{12} \) or so, depending on how many digits you need), or call `np.linalg.solve` and catch `np.linalg.LinAlgError`.

**Part 4: Solving equations**

## Solving \( A\mathbf{x} = \mathbf{b} \)

**What it is.** The most common practical use of all this: **solve a system of linear equations**. Say

\[
\begin{aligned}
2x_1 + 1x_2 &= 5 \\
1x_1 + 3x_2 &= 10
\end{aligned}
\qquad\Longleftrightarrow\qquad
\underbrace{\begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix}}_{A}
\underbrace{\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}}_{\mathbf{x}}
=
\underbrace{\begin{bmatrix} 5 \\ 10 \end{bmatrix}}_{\mathbf{b}}
\]

In transformation language: "which vector \( \mathbf{x} \) does \( A \) move onto \( \mathbf{b} \)?" On paper the answer is \( \mathbf{x} = A^{-1}\mathbf{b} \).

**Why it matters.** Fitting a line, finding the coefficients of a linear combination (Day 5 promised this), calibrating a sensor, the least-squares weights of a linear model: all are linear systems.

**By hand, by elimination.** The first equation gives \( x_2 = 5 - 2x_1 \). Put that into the second: \( x_1 + 3(5 - 2x_1) = 10 \), so \( x_1 + 15 - 6x_1 = 10 \), so \( -5x_1 = -5 \) and \( x_1 = 1 \). Then \( x_2 = 5 - 2 = 3 \). Answer: \( \mathbf{x} = [1, 3] \). Check: \( 2 + 3 = 5 \) and \( 1 + 9 = 10 \). This elimination, organised for a computer, is what `np.linalg.solve` does.

```python
b = np.array([5., 10.])
x = np.linalg.solve(A, b)
x                       # array([1., 3.])
A @ x                   # array([ 5., 10.])   check: it really is a solution
np.linalg.inv(A) @ b    # array([1., 3.])     the same answer here, but see below
try:
    np.linalg.solve(Sg, np.array([1., 2.]))
except np.linalg.LinAlgError as e:
    print("LinAlgError:", e)
# LinAlgError: Singular matrix
```

`solve(A, b)` needs a square, invertible `A`; with a singular one it raises the same `LinAlgError` as `inv`.

### Why `solve` beats `inv(A) @ b`

**1. Speed.** `solve` factors \( A \) once (the LU decomposition) and uses the factors directly on \( \mathbf{b} \). `inv` does the same factoring, then extra work to build all \( n^2 \) entries of the inverse, which you then still multiply by \( \mathbf{b} \). Counting operations, the factoring costs about \( \tfrac{2}{3}n^3 \) and building the full inverse about \( 2n^3 \), so expect `inv` to take roughly three times as long. `time.perf_counter()` (Day 3a) reads a clock in seconds; the difference between two readings is the elapsed time:

```python
import time
rng = np.random.default_rng(0)
Mbig = rng.normal(size=(1000, 1000))
bbig = rng.normal(size=1000)

t = time.perf_counter(); x1 = np.linalg.solve(Mbig, bbig); t_solve = time.perf_counter() - t
t = time.perf_counter(); x2 = np.linalg.inv(Mbig) @ bbig; t_inv = time.perf_counter() - t
print(f"solve: {t_solve*1000:.0f} ms, inv then @: {t_inv*1000:.0f} ms")
# solve: 10 ms, inv then @: 32 ms      on my machine; your timings will differ, the ratio is the point
np.allclose(x1, x2)    # True
```

`:.0f` (Day 2a) prints with no decimal places. The browser's timings are slower than a laptop's.

**2. Numerical stability.** Every float operation rounds a little. Building the inverse adds more operations, and for badly conditioned matrices the rounding errors get amplified. A classic badly behaved matrix is the **Hilbert matrix**. Textbooks write its entries with 1-based indices as \( 1/(i + j - 1) \); in 0-based NumPy indices (Day 10's off-by-one) that becomes `1 / (i + j + 1)`. Both give the same matrix, starting with 1, 1/2, 1/3 across the top row.

```python
n = 12
Hm = np.array([[1 / (i + j + 1) for j in range(n)] for i in range(n)])
b2 = Hm @ np.ones(n)             # so the true answer is all ones
x_solve = np.linalg.solve(Hm, b2)
x_inv = np.linalg.inv(Hm) @ b2
print(f"{np.linalg.cond(Hm):.1e}")                       # 1.6e+16  effectively singular
print(f"{np.linalg.norm(Hm @ x_solve - b2):.1e}")        # 6.0e-16  solve: tiny residual
print(f"{np.linalg.norm(Hm @ x_inv - b2):.1e}")          # 2.4e-01  inv: way off
```

`:.1e` (Day 2a) prints in scientific notation with one decimal. The **residual** \( \|A\mathbf{x} - \mathbf{b}\| \) measures how badly a candidate \( \mathbf{x} \) fails the equations. `solve` gives an answer that satisfies the equations to machine precision; the inverse route misses by 0.24. (With \( \kappa \approx 10^{16} \), neither \( \mathbf{x} \) is close to all ones: no method can recover digits the condition number has destroyed. But `solve`'s answer at least fits the data.)

!!! warning "Common mistake: `inv` in production code"
    A rule numerical people repeat: *if you're writing `inv(A) @ b`, you want `solve(A, b)`.* The same goes for `B @ inv(A)`-style code: rewrite it as a solve (\( BA^{-1} \) is the transpose of \( (A^\top)^{-1}B^\top \), which is `np.linalg.solve(A.T, B.T).T`). Explicit inverses are for learning, for tiny fixed matrices (2×2 or 3×3 graphics transforms), and for orthogonal matrices, where the inverse is a free transpose.

!!! android "Android bridge"
    Day 10 showed that `android.graphics.Matrix` is a 3×3 matrix and that `preConcat` / `postConcat` choose the multiplication order. Today's new pieces are the 3×3 shape and the inverse.

    **Why 3×3 for 2-D.** A translation isn't linear (it moves the origin), so no 2×2 matrix can do it. The trick, called **homogeneous coordinates**, writes each point \( (x, y) \) as \( (x, y, 1) \). Then a 3×3 matrix whose top-left 2×2 block is today's scale, rotate or shear, and whose third column holds the shift \( (t_x, t_y) \), does both at once: the third column gets multiplied by that 1 and added on.

    **The classic use of the inverse.** The user taps a zoomed and rotated image, and you need to know which pixel of the *original* image they hit. The screen point is \( M\mathbf{p} \); you want \( \mathbf{p} \), so you apply \( M^{-1} \):

    ```kotlin
    val m = Matrix().apply {
        setRotate(30f)        // rotate about (0, 0)
        postScale(2f, 2f)     // then zoom 2x: m becomes S × R
    }
    val inverse = Matrix()
    if (m.invert(inverse)) {                 // false if m is singular (det 0)
        val pts = floatArrayOf(event.x, event.y)
        inverse.mapPoints(pts)               // pts now in image coordinates
    }
    ```

    `invert()` returns a `Boolean`: it is the Android API telling you about singular matrices. A `setScale(0f, 1f)` matrix squashes everything onto a line and can't be inverted. `mapPoints` is matrix-times-vector for each point, like `H @ square` above.

    **The sign trap.** Android's screen y-axis points **down**. In this lesson's maths, y points up and \( R(\theta) \) turns counterclockwise. With y down, the very same matrix looks *clockwise* on screen, which is why `setRotate(30f)` rotates clockwise as you look at the phone. If you port a formula from a paper to `Canvas` code and everything spins or mirrors the wrong way, this flip is the first suspect: it's a reflection, determinant −1, hiding in the coordinate system.

## Why ML rarely inverts matrices

You might expect ML to solve big equations all day. Mostly it doesn't, for three reasons:

1. **Models aren't linear.** A neural network has non-linear activations (ReLU, Day 12) between its matrix multiplies, so there's no single \( A\mathbf{x} = \mathbf{b} \) to solve for the weights.
2. **Weight matrices usually aren't square.** A layer mapping 784 pixels to 128 features has a \( (128, 784) \) weight matrix: no inverse exists. And none is needed. You'll soon meet the **backward pass**, where training sends error signals backwards through the network; it multiplies by the *transpose* \( W^\top \), not the inverse (Day 33). Today's cheapest operation does the work.
3. **Inverting big matrices is costly in memory and accuracy.** Inverting an \( n \times n \) matrix takes about \( n^3 \) operations. For \( n = 100{,}000 \) that's \( 10^{15} \), which a modern GPU could grind through in minutes. The real blockers are memory and stability: the matrix alone is \( 100{,}000^2 \times 8 \) bytes = 80 GB in float64, and with that many features the condition number is usually enormous.

Instead, ML uses **gradient descent** (Month 2): start with a guess, measure the error, nudge the weights downhill, repeat. It uses only matrix multiplies and vector updates, which GPUs and phone NPUs are built for.

## The one place you'll meet an inverse: the normal equation

**What it is.** Linear regression (fitting a straight line or plane to data, Phase 2) has an exact formula, the **normal equation**:

\[
\mathbf{w} = (X^\top X)^{-1} X^\top \mathbf{y}
\]

Here \( X \) is the data (one row per example, one column per feature, Day 5's sense of "feature"), \( \mathbf{y} \) the targets, and \( \mathbf{w} \) the weights to learn. Every piece is from today: transposes, a product, an inverse. And \( X^\top X \) is the symmetric Gram matrix from Part 1. Following today's rule, you'd compute it with `solve`, never `inv`: solve \( (X^\top X)\mathbf{w} = X^\top\mathbf{y} \). A preview (Phase 2 derives it):

```python
rng = np.random.default_rng(42)
xs = np.linspace(0, 10, 50)
ys = 3 * xs + 2 + rng.normal(0, 1, size=50)     # a noisy line: slope 3, intercept 2
X = np.column_stack([xs, np.ones_like(xs)])     # shape (50, 2): [x, 1] per row
XtX = X.T @ X
XtX.shape, np.allclose(XtX, XtX.T)              # ((2, 2), True)   small and symmetric
w = np.linalg.solve(XtX, X.T @ ys)              # solves (X^T X) w = X^T y
w.round(3)                      # array([3.045, 1.865])   close to the true 3 and 2
np.polyfit(xs, ys, 1).round(3)  # array([3.045, 1.865])   NumPy's own line fit agrees
```

`X` now names this data matrix (it was the 3×2 example earlier). The column of ones is what lets the model learn an intercept: \( w_0 x + w_1 \cdot 1 \). `np.polyfit(x, y, deg)` fits a polynomial of degree `deg`; degree 1 is a straight line, and it returns the coefficients **highest power first**, so `[slope, intercept]`.

`X.T @ X` has shape `(2, 50) @ (50, 2) = (2, 2)`: a tiny square matrix, no matter how many data points. That's why the normal equation works for small feature counts, and why gradient descent takes over when features number in the thousands or millions.

### `np.linalg.lstsq`: what practitioners call

**What it is.** `np.linalg.lstsq(X, y)` ("least squares") solves an **over-determined** system directly: more equations (rows) than unknowns (columns), where no exact solution exists, so it finds the \( \mathbf{w} \) that makes \( \|X\mathbf{w} - \mathbf{y}\| \) as small as possible. That's the same \( \mathbf{w} \) as the normal equation, but computed from \( X \) itself without ever forming \( X^\top X \).

**Why it's preferred.** Forming \( X^\top X \) **squares** the condition number: \( \kappa(X^\top X) = \kappa(X)^2 \). A data matrix with \( \kappa = 10^8 \), merely shaky, becomes \( 10^{16} \), effectively singular. `lstsq` works on \( X \) and avoids that. It also copes with rank-deficient \( X \) (duplicated features, Day 23) by returning *a* best solution instead of raising.

```python
w_ls, residuals, rank, sing_vals = np.linalg.lstsq(X, ys, rcond=None)
w_ls.round(3)                   # array([3.045, 1.865])   the same weights
rank                            # np.int32(2)    both columns independent (lstsq returns int32 even on a laptop)
np.linalg.cond(X) ** 2, np.linalg.cond(XtX)
# (np.float64(136.604972539428), np.float64(136.6049725394279))   squared, as promised
```

`lstsq` returns four things, unpacked here into four names (Day 2b's tuple unpacking): the weights; the sum of squared residuals; the rank of `X` (Day 23); and its **singular values** (Day 23 and Going further). `rcond=None` picks the modern cut-off for treating tiny singular values as zero, and silences a warning in older NumPy versions.

---

## Going further

- **LU decomposition**: how `solve`, `inv` and `det` work inside, as \( A = LU \) with triangular factors. [Wikipedia: LU decomposition](https://en.wikipedia.org/wiki/LU_decomposition).
- **The pseudo-inverse** `np.linalg.pinv`: a best-possible "inverse" for non-square or singular matrices, built from the SVD. [NumPy: `numpy.linalg.pinv`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html).
- **3×3 and \( n \times n \) determinants**: cofactor expansion, the determinant as a signed **volume**, and \( \det(A^\top) = \det(A) \). [3Blue1Brown: The determinant](https://www.youtube.com/watch?v=Ip3X9LOh2dk) covers the 3-D case; [MML book, section 4.1](https://mml-book.github.io/) proves the rules.
- **Homogeneous coordinates** in depth: projective transforms and the 4×4 matrices of 3-D graphics. [Wikipedia: Homogeneous coordinates](https://en.wikipedia.org/wiki/Homogeneous_coordinates).
- **Batched linear algebra**: `np.linalg.solve`, `inv` and `det` accept stacks of shape `(..., n, n)` and work on every matrix at once, as `@` does. [NumPy linear algebra reference](https://numpy.org/doc/stable/reference/routines.linalg.html), section "Linear algebra on several matrices at once".
- **Cholesky decomposition** `np.linalg.cholesky`: a faster solve for symmetric positive-definite matrices such as a well-conditioned \( X^\top X \). [NumPy: `numpy.linalg.cholesky`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.cholesky.html).
- **`slogdet`**: the logarithm of the determinant without overflow or underflow, for the \( 10^{-30} \) cases above; used in Gaussian likelihoods. [NumPy: `numpy.linalg.slogdet`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.slogdet.html).
- **Singular values and the SVD**, which define the condition number precisely (largest over smallest singular value): Day 23 introduces them, and [MIT 18.06 (Gilbert Strang)](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) covers them in full.

---

## Exercises

1. **Warm-up: transpose.** By hand, write the transpose of \( A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix} \) and give its shape. Then, with \( B = \begin{bmatrix} 1 & 0 & 2 \\ 0 & 1 & 1 \end{bmatrix} \), check in NumPy that \( (AB)^\top = B^\top A^\top \). What shape is \( AB \), and why?
2. **Warm-up: an inverse by hand.** (a) Using the 2×2 formula, invert \( M = \begin{bmatrix} 4 & 7 \\ 2 & 6 \end{bmatrix} \) by hand, and verify with `np.allclose(M @ M_inv, np.eye(2))`. (b) The shear \( H = \begin{bmatrix} 1 & 1.5 \\ 0 & 1 \end{bmatrix} \) slides each point right by 1.5 times its height. Predict \( H^{-1} \) without the formula, check it with `np.linalg.inv`, and show that it moves `H @ [2, 1]` back to `[2, 1]`.
3. **Core: rotations are orthogonal.** For \( R = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} \): where does it send \( [3, 1] \)? What is \( \det(R) \)? Compute \( R^{-1} \) and compare it with \( R^\top \). Then build \( R(45°) \) with `np.cos` and `np.sin`, and check that it keeps the length of \( [3, 4] \).
4. **Core: solving and spotting singular matrices.** (a) Solve \( 3x + 2y = 12 \) and \( x - y = -1 \) with `np.linalg.solve`, and check the answer by multiplying back. (b) Which of these are singular? Decide from the columns first (is one a multiple of the other?), then confirm with `np.linalg.det` and `np.linalg.cond`: \( \begin{bmatrix} 2 & 4 \\ 1 & 2 \end{bmatrix} \), \( \begin{bmatrix} 3 & 1 \\ 6 & 2 \end{bmatrix} \), \( \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \).
5. **Core: the Gram matrix.** Make `X = rng.normal(size=(5, 3))` with `rng = np.random.default_rng(7)`. Show that `X.T @ X` is symmetric with `np.allclose`, that its diagonal equals the squared length of each column of `X`, and that its cell `[0, 2]` is column 0 dotted with column 2. What shape is `X @ X.T`, and is it symmetric?
6. **Predict the output.** With `S = np.array([[2, 0], [0, 3]])`, `R = np.array([[0, -1], [1, 0]])` and `v = np.array([1, 1])`, write down on paper, before running anything: `R @ S @ v`, `S @ R @ v`, `np.linalg.det(S @ R)` and `np.linalg.det(R @ S)`. Then run them.
7. **Find and fix the bugs.** This function is supposed to map a tapped screen point back to image coordinates, refusing matrices that can't be undone. It runs without errors on the example, and the answer is wrong. Find both bugs.

    ```python
    def screen_to_image(M, tap):
        if np.linalg.det(M) == 0:
            raise ValueError("cannot invert")
        return M ** -1 @ tap

    M7 = np.array([[2., 0.], [0., 4.]])       # zoom 2x horizontally, 4x vertically
    screen_to_image(M7, np.array([6., 8.]))   # should be [3, 2]
    ```

8. **Challenge: Android's 3×3 matrix in NumPy.** In homogeneous coordinates (Android bridge), a point \( (x, y) \) is the vector \( [x, y, 1] \). Build three 3×3 matrices: a rotation by 30° (today's \( R(\theta) \) in the top-left 2×2 block, a 1 in the bottom-right corner), a 2× zoom, and a translation by \( (100, 50) \) (the identity with 100 and 50 in the top two cells of the last column). Combine them as "rotate, then zoom, then translate", map the image point \( (10, 20) \) to the screen, then use `np.linalg.solve` to map the screen point back. Check that you get \( (10, 20) \) and that the combined matrix's determinant is the zoom's area factor.

??? success "Solutions"
    **1.** The transpose swaps rows and columns, so the (3, 2) matrix becomes (2, 3):

    ```python
    import numpy as np

    A = np.array([[1, 2], [3, 4], [5, 6]])
    A.T
    # array([[1, 3, 5],
    #        [2, 4, 6]])
    A.T.shape                # (2, 3)
    B = np.array([[1, 0, 2], [0, 1, 1]])
    (A @ B).shape            # (3, 3)
    np.array_equal((A @ B).T, B.T @ A.T)   # True
    ```

    \( AB \) is \( (3, 2) @ (2, 3) \): the inner 2s match and are summed away, and the outer sizes, 3 and 3, survive (Day 10's shape rule). So \( (AB)^\top \) is (3, 3) too, and \( B^\top A^\top \) is \( (3, 2) @ (2, 3) \), also (3, 3). The equality holds by the index proof in the lesson.

    **2.** (a) \( ad - bc = 4 \cdot 6 - 7 \cdot 2 = 10 \); swap 4 and 6, negate 7 and 2, divide by 10. (b) Undoing "slide right by 1.5 per unit of height" means sliding *left* by 1.5 per unit of height, and the height itself never changed, so \( H^{-1} = \begin{bmatrix} 1 & -1.5 \\ 0 & 1 \end{bmatrix} \).

    ```python
    import numpy as np

    M = np.array([[4., 7.], [2., 6.]])
    M_inv = np.array([[6., -7.], [-2., 4.]]) / 10
    M_inv
    # array([[ 0.6, -0.7],
    #        [-0.2,  0.4]])
    np.allclose(M @ M_inv, np.eye(2)), np.allclose(M_inv, np.linalg.inv(M))   # (True, True)

    H = np.array([[1., 1.5], [0., 1.]])
    np.linalg.inv(H)
    # array([[ 1. , -1.5],
    #        [ 0. ,  1. ]])
    p = np.array([2., 1.])
    H @ p                       # array([3.5, 1. ])
    np.linalg.inv(H) @ (H @ p)  # array([2., 1.])
    ```

    Dividing by 10 works because multiplying \( M \) by the swapped-and-negated matrix gives \( (ad - bc)I = 10I \); dividing by 10 turns that into \( I \). For the shear, the formula agrees with the guess: \( \det(H) = 1 \), so \( H^{-1} \) is the swapped-and-negated matrix itself, with \( -1.5 \) where \( 1.5 \) was.

    **3.** Rotations are orthogonal, so the inverse is the transpose and lengths don't change:

    ```python
    import numpy as np

    R = np.array([[0, -1], [1, 0]])
    R @ np.array([3, 1])     # array([-1,  3])
    np.linalg.det(R)         # np.float64(1.0)   rotations don't change area
    np.linalg.inv(R)
    # array([[ 0.,  1.],
    #        [-1., -0.]])
    R.T
    # array([[ 0,  1],
    #        [-1,  0]])
    np.allclose(np.linalg.inv(R), R.T)   # True

    th = np.pi / 4
    R45 = np.array([[np.cos(th), -np.sin(th)],
                    [np.sin(th),  np.cos(th)]])
    np.linalg.norm(np.array([3., 4.])), np.linalg.norm(R45 @ np.array([3., 4.]))
    # (np.float64(5.0), np.float64(5.0))
    ```

    \( [3, 1] \) is 3 of \( \hat{\imath} \) plus 1 of \( \hat{\jmath} \), and they land on \( [0, 1] \) and \( [-1, 0] \): \( 3[0, 1] + 1[-1, 0] = [-1, 3] \). The `-0.` in the inverse is a negative floating-point zero, which compares equal to 0. The inverse equals the transpose because the columns of \( R \) are perpendicular unit vectors, so \( R^\top R = I \); and \( \|R\mathbf{v}\|^2 = \mathbf{v}^\top R^\top R\,\mathbf{v} = \mathbf{v}^\top\mathbf{v} \), which is why the length stays 5.

    **4.** (a) Two equations, two unknowns, an invertible matrix: one answer. (b) Look at the columns first.

    ```python
    import numpy as np

    A4 = np.array([[3., 2.], [1., -1.]])
    b4 = np.array([12., -1.])
    sol = np.linalg.solve(A4, b4)
    sol                      # array([2., 3.])    x = 2, y = 3
    A4 @ sol                 # array([12., -1.])  multiplying back gives b

    for m in ([[2., 4.], [1., 2.]], [[3., 1.], [6., 2.]], [[1., 2.], [3., 4.]]):
        m = np.array(m)
        print(np.linalg.det(m), f"{np.linalg.cond(m):.1e}")
    # 0.0 3.6e+17
    # 0.0 6.4e+16
    # -2.0000000000000004 1.5e+01
    ```

    By hand, \( 3 \cdot 2 + 2 \cdot 3 = 12 \) and \( 2 - 3 = -1 \). In the first matrix, column 2 \( [4, 2] \) is 2 × column 1 \( [2, 1] \); in the second, column 1 \( [3, 6] \) is 3 × column 2 \( [1, 2] \). When one column is a multiple of the other, \( \hat{\imath} \) and \( \hat{\jmath} \) land on the same line, the plane collapses, the area is 0, and the condition number is effectively infinite: in exact arithmetic it would be \( \infty \), and rounding inside `cond` leaves a number above \( 10^{16} \), which means the same thing. The third matrix is invertible (\( 1 \cdot 4 - 2 \cdot 3 = -2 \)), well conditioned, and its negative determinant means it flips orientation.

    **5.** Every property comes from "cell \( (i, j) \) is column \( i \) dotted with column \( j \)":

    ```python
    import numpy as np

    rng = np.random.default_rng(7)
    X = rng.normal(size=(5, 3))
    G = X.T @ X
    G.shape, np.allclose(G, G.T)                       # ((3, 3), True)
    np.allclose(np.diag(G), (X ** 2).sum(axis=0))      # True    diagonal = squared column lengths
    np.allclose(G[0, 2], X[:, 0] @ X[:, 2])            # True
    (X @ X.T).shape, np.allclose(X @ X.T, (X @ X.T).T) # ((5, 5), True)
    ```

    \( X^\top X \) is symmetric because \( (X^\top X)^\top = X^\top X^{\top\top} = X^\top X \), or, in words, because column 0 · column 2 equals column 2 · column 0. Its diagonal holds each column dotted with itself, the squared length. \( XX^\top \) is \( (5, 5) \), one cell per pair of *rows* (examples), and symmetric by the same argument.

    **6.** Read each product right to left. `R @ S @ v`: scale first, \( [1, 1] \to [2, 3] \), then rotate, \( [2, 3] \to [-3, 2] \). `S @ R @ v`: rotate first, \( [1, 1] \to [-1, 1] \), then scale, \( \to [-2, 3] \). Both determinants are \( 6 \times 1 = 6 \).

    ```python
    import numpy as np

    S = np.array([[2, 0], [0, 3]])
    R = np.array([[0, -1], [1, 0]])
    v = np.array([1, 1])
    R @ S @ v                # array([-3,  2])
    S @ R @ v                # array([-2,  3])
    np.linalg.det(S @ R)     # np.float64(6.0)
    np.linalg.det(R @ S)     # np.float64(6.0)
    ```

    The vectors differ because matrix products don't commute (Day 10), but the determinants agree because \( \det(SR) = \det(S)\det(R) = \det(R)\det(S) \): the area factors are plain numbers, and numbers do commute.

    **7.** Bug 1: `M ** -1` takes the reciprocal of each element. On this diagonal matrix it turns the zeros into `inf` (with a `RuntimeWarning: divide by zero`), so the "answer" is `[inf, inf]`; on a matrix with no zeros it would quietly return wrong finite numbers. Bug 2: `det(M) == 0` misses nearly singular matrices and can reject well-conditioned tiny ones. Run the buggy version inside `np.errstate` (Day 4b) to see the damage without the warnings:

    ```python
    import numpy as np

    def screen_to_image_buggy(M, tap):
        if np.linalg.det(M) == 0:
            raise ValueError("cannot invert")
        return M ** -1 @ tap

    M7 = np.array([[2., 0.], [0., 4.]])
    with np.errstate(divide="ignore"):
        print(screen_to_image_buggy(M7, np.array([6., 8.])))
    # [inf inf]

    def screen_to_image(M, tap):
        if np.linalg.cond(M) > 1e12:
            raise ValueError("matrix is singular or nearly so")
        return np.linalg.solve(M, tap)

    screen_to_image(M7, np.array([6., 8.]))      # array([3., 2.])
    try:
        screen_to_image(np.array([[1., 2.], [2., 4.]]), np.array([1., 1.]))
    except ValueError as e:
        print("ValueError:", e)
    # ValueError: matrix is singular or nearly so
    ```

    `**` is element-wise, like `*`; the matrix inverse is a different operation entirely. `solve` computes \( M^{-1}\,\text{tap} \) without building the inverse, and the condition number, not the determinant, measures how close a matrix is to having no inverse.

    **8.** Build each 3×3 matrix, then multiply right to left in the order the steps happen: \( T \cdot Z \cdot R \).

    ```python
    import numpy as np

    th = np.pi / 6
    R3 = np.array([[np.cos(th), -np.sin(th), 0],
                   [np.sin(th),  np.cos(th), 0],
                   [0,           0,          1]])
    Z3 = np.diag([2., 2., 1.])                   # zoom x and y by 2, keep the 1
    T3 = np.array([[1., 0., 100.],
                   [0., 1.,  50.],
                   [0., 0.,   1.]])
    M3 = T3 @ Z3 @ R3                            # rotate, then zoom, then translate
    p = np.array([10., 20., 1.])
    screen = M3 @ p
    screen.round(4)                              # array([97.3205, 94.641 ,  1.    ])
    back = np.linalg.solve(M3, screen)
    back.round(10)                               # array([10., 20.,  1.])
    np.linalg.det(M3)                            # np.float64(4.0)
    ```

    The 1 in the last slot of \( \mathbf{p} \) picks up the third column, so the translation gets added after the 2×2 block has rotated and zoomed. Undoing it with `solve` reverses the whole chain at once, exactly as \( (TZR)^{-1} = R^{-1}Z^{-1}T^{-1} \) says. The determinant is \( \det(T)\det(Z)\det(R) = 1 \times 4 \times 1 = 4 \): translations and rotations don't change area, and a 2× zoom in both directions makes every area 4× bigger. This is what `Matrix.invert` and `mapPoints` do for you on Android.

## Resources

- [3Blue1Brown: Linear transformations and matrices (Essence of Linear Algebra, ch. 3)](https://www.youtube.com/watch?v=kYB8IZa5AuE): the "columns are where î and ĵ land" idea, animated. Watch this first.
- [3Blue1Brown: The determinant (ch. 6)](https://www.youtube.com/watch?v=Ip3X9LOh2dk) and [Inverse matrices, column space and null space (ch. 7)](https://www.youtube.com/watch?v=uQhTuRlWMxw): today's second half, beautifully visualised.
- [Khan Academy: Matrix transformations](https://www.khanacademy.org/math/linear-algebra/matrix-transformations): practice problems on transformations, inverses, transposes and determinants.
- [NumPy linear algebra (`numpy.linalg`) reference](https://numpy.org/doc/stable/reference/routines.linalg.html): every function used today, including [`solve`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html), [`inv`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html) and [`lstsq`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html).
- [Mathematics for Machine Learning (free book), ch. 2](https://mml-book.github.io/): matrices, transposes, inverses and linear systems, formally.
- [Android reference: `android.graphics.Matrix`](https://developer.android.com/reference/android/graphics/Matrix): `setRotate`, `postScale`, `invert`, `mapPoints`.

## Checklist

- [ ] I can transpose a matrix and say what happens to its shape; I know `.T` is a view, does nothing to a 1-D array, and reverses every axis of a 3-D one (and I'd use `.mT` or `swapaxes(-1, -2)` there).
- [ ] I can explain why \( (AB)^\top = B^\top A^\top \), and read \( \mathbf{x}^\top\mathbf{y} \) as a dot product and \( \mathbf{x}\mathbf{y}^\top \) as an outer product.
- [ ] I can say what makes a matrix symmetric, prove that \( X^\top X \) always is, and test symmetry of floats with `np.allclose`.
- [ ] Given a 2×2 matrix, I can sketch where \( \hat{\imath} \) and \( \hat{\jmath} \) land, and I know why a translation isn't linear.
- [ ] I can write the identity, build and invert a diagonal matrix with `np.diag`, invert a 2×2 matrix by hand, and use \( (AB)^{-1} = B^{-1}A^{-1} \).
- [ ] I can say what det = 5, det = 0 and det = −1 mean geometrically, and use \( \det(AB) = \det(A)\det(B) \).
- [ ] I know an orthogonal matrix has perpendicular unit columns and that its inverse is its transpose.
- [ ] I can read a condition number, and I never test `det(M) == 0`.
- [ ] I use `np.linalg.solve(A, b)`, not `np.linalg.inv(A) @ b`, and `np.linalg.lstsq` for over-determined systems, and can say why.
- [ ] I can explain why neural networks are trained with gradient descent instead of matrix inversion.

**Next:** [Day 12: A neural-network layer is a matrix multiply](day-12-layer-is-matmul.md)
