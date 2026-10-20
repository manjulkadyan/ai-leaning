# Day 10: Matrices and Matrix Multiplication

**Phase 1 · Week 2 · ~120 minutes**

## Objectives

- [ ] Explain what a matrix is, read its shape \( m \times n \), and find an element \( A_{ij} \) by row and column.
- [ ] Compute a matrix-vector product two ways: as dot products of rows, and as a weighted sum of columns.
- [ ] Compute a matrix-matrix product by hand using \( C_{ij} = \sum_k A_{ik} B_{kj} \).
- [ ] Predict the shape of `A @ B`, or that it fails, using the rule "inner dimensions match, outer dimensions survive".
- [ ] Tell `@` from `*`, and explain why \( AB \neq BA \) but \( (AB)C = A(BC) \).
- [ ] Count the cost of a matrix multiply and explain why GPUs and phone NPUs are built around it.

**Prerequisites.** [Day 5](day-05-vectors.md) (vectors and the dot product) and [Day 9](day-09-broadcasting.md) (broadcasting, element-wise operations).

---

## The big idea

A **matrix** is a grid of numbers with rows and columns. **Matrix multiplication** is a very organised way of doing *lots of dot products at once*: every row of the first matrix is dotted with every column of the second, and each result lands in its own cell.

That is the whole trick, and it is worth learning really well, because it is most of what a neural network does. A layer of a network is a matrix multiply (Day 12). An LLM generating one token is, by compute, mostly matrix multiplies. The GPU in your laptop and the NPU in your phone exist mainly to do this one operation fast.

**Analogy.** Picture a till at a shop. A row of the first matrix is one customer's basket ("6 apples, 1 loaf, 2 milks"). A column of the second matrix is one shop's price list ("apples 0.50, bread 2.00, milk 1.20"). Multiply matching items and add them up and you get one number: what that customer pays at that shop. Matrix multiplication does this for *every customer at every shop* in one go, and writes the answers in a table: customers down the side, shops across the top. We build exactly this example later in the lesson.

## What a matrix is

A matrix with \( m \) rows and \( n \) columns is called an **\( m \times n \) matrix** ("m by n"). Rows always come first.

\[
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}
\qquad A \text{ is } 2 \times 3
\]

- Matrices are written as **uppercase** letters: \( A, B, W \).
- \( A_{ij} \) is the element in **row \( i \), column \( j \)**. In math, counting starts at 1, so \( A_{13} = 3 \) (row 1, column 3) and \( A_{21} = 4 \).
- \( \mathbb{R}^{m \times n} \) means "the set of all \( m \times n \) grids of real numbers". \( W \in \mathbb{R}^{128 \times 784} \) just says "\( W \) is a 128-by-784 grid of floats".

In NumPy, a matrix is a 2-D array, and counting starts at 0:

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])
A.shape      # (2, 3)      (rows, columns)
A[0, 2]      # 3           math's A_13: NumPy counts from 0
A[1, 0]      # 4           math's A_21
A[1]         # array([4, 5, 6])   row 1, a vector
A[:, 1]      # array([2, 5])      column 1, a vector
```

!!! warning "Common mistake: off-by-one between math and code"
    Math writes the top-left element as \( A_{11} \); NumPy writes it as `A[0, 0]`. Papers and textbooks use the first, code uses the second. When you translate a formula into code, subtract 1 from every index, or (better) just loop over `range(m)` and stop thinking about it.

You can read a matrix in two ways, and both are useful: as a **stack of row vectors** (each row is one data point, one basket, one neuron's weights), or as a **row of column vectors** (each column is one feature, one shop, one direction).

## Matrix times vector, two ways

Let \( W \) be \( 2 \times 3 \) and \( \mathbf{x} \) a vector of 3 numbers. The product \( W\mathbf{x} \) is a vector of 2 numbers.

\[
W = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}
\qquad
\mathbf{x} = \begin{bmatrix} 1 \\ 0 \\ 2 \end{bmatrix}
\]

**Way 1: each output is a dot product of a row with \( \mathbf{x} \).** Output number \( i \) is row \( i \) of \( W \) dotted with \( \mathbf{x} \) (Day 5):

\[
y_i = \sum_{j=1}^{n} W_{ij} \, x_j
\qquad
W\mathbf{x} = \begin{bmatrix} 1\cdot1 + 2\cdot0 + 3\cdot2 \\ 4\cdot1 + 5\cdot0 + 6\cdot2 \end{bmatrix} = \begin{bmatrix} 7 \\ 16 \end{bmatrix}
\]

Read \( \sum_{j=1}^{n} \) as "add up, for \( j \) going from 1 to \( n \)". This is the **neuron view**: each row is one neuron's weight vector, and each output is how strongly the input matches that neuron's pattern.

**Way 2: the output is a weighted sum of the columns.** The numbers in \( \mathbf{x} \) say *how much of each column* to take:

\[
W\mathbf{x} = 1 \cdot \begin{bmatrix} 1 \\ 4 \end{bmatrix} + 0 \cdot \begin{bmatrix} 2 \\ 5 \end{bmatrix} + 2 \cdot \begin{bmatrix} 3 \\ 6 \end{bmatrix} = \begin{bmatrix} 7 \\ 16 \end{bmatrix}
\]

Same answer, different story. This is the **recipe view**: the columns are ingredients and \( \mathbf{x} \) is the recipe. It is the view you need tomorrow, when a matrix becomes a transformation of space (each column says where one axis ends up).

```python
W = np.array([[1, 2, 3],
              [4, 5, 6]])
x = np.array([1, 0, 2])

W @ x                                     # array([ 7, 16])

# Way 1: dot each row with x
np.array([W[0] @ x, W[1] @ x])            # array([ 7, 16])

# Way 2: weighted sum of the columns
x[0] * W[:, 0] + x[1] * W[:, 1] + x[2] * W[:, 2]   # array([ 7, 16])
```

NumPy treats the 1-D `x` of shape `(3,)` as a column here, and the result has shape `(2,)`.

## Matrix times matrix

Now let the second operand have several columns. Each column of \( B \) is its own vector, so \( AB \) is just "\( A \) times each column of \( B \)", with the answers placed side by side. Cell by cell:

\[
C = AB \qquad C_{ij} = \sum_{k=1}^{n} A_{ik} \, B_{kj}
\]

One symbol at a time:

- \( C_{ij} \) is the cell in row \( i \), column \( j \) of the result.
- \( k \) walks along **row \( i \) of \( A \)** (\( A_{i1}, A_{i2}, \dots \)) and at the same time down **column \( j \) of \( B \)** (\( B_{1j}, B_{2j}, \dots \)).
- Multiply the pairs, add them up. So \( C_{ij} \) = (row \( i \) of \( A \)) · (column \( j \) of \( B \)).

### Worked example by hand: 2×3 times 3×2

\[
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}
\qquad
B = \begin{bmatrix} 7 & 8 \\ 9 & 10 \\ 11 & 12 \end{bmatrix}
\]

\( A \) has 2 rows and \( B \) has 2 columns, so there are \( 2 \times 2 = 4 \) dot products to do:

| Cell | Row of A | Column of B | Dot product |
|---|---|---|---|
| \( C_{11} \) | [1, 2, 3] | [7, 9, 11] | 1·7 + 2·9 + 3·11 = 7 + 18 + 33 = **58** |
| \( C_{12} \) | [1, 2, 3] | [8, 10, 12] | 1·8 + 2·10 + 3·12 = 8 + 20 + 36 = **64** |
| \( C_{21} \) | [4, 5, 6] | [7, 9, 11] | 4·7 + 5·9 + 6·11 = 28 + 45 + 66 = **139** |
| \( C_{22} \) | [4, 5, 6] | [8, 10, 12] | 4·8 + 5·10 + 6·12 = 32 + 50 + 72 = **154** |

\[
C = AB = \begin{bmatrix} 58 & 64 \\ 139 & 154 \end{bmatrix}
\]

<figure class="diagram" markdown>
![Animation of C = A @ B for A = [[1,2,3],[4,5,6]] and B = [[7,8],[9,10],[11,12]]: row 0 of A and column 0 of B light up and C[0,0] fills in with 58, then row 0 and column 1 give 64, row 1 and column 0 give 139, and row 1 and column 1 give 154, with the sum written out under each step](../../assets/img/phase-1/day10-matmul-steps.svg)
<figcaption>Animated: each cell of C is one row of A dotted with one column of B (labels use NumPy's 0-based indices).</figcaption>
</figure>

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])
B = np.array([[7, 8],
              [9, 10],
              [11, 12]])

C = A @ B
print(C)
# [[ 58  64]
#  [139 154]]
C.shape            # (2, 2)

A[0] @ B[:, 1]     # 64   C[0, 1] is row 0 of A · column 1 of B
np.matmul(A, B)    # same as A @ B
```

### Try it: step through a matrix multiply

Press **Step** to fill in C one cell at a time, or hover over a cell of C to see where it comes from. The numbers in A and B are editable: change one and watch which cells of C change.

<div class="matmul-stepper"></div>

!!! example "Experiments to try"
    1. Change `A[0, 0]` from 1 to 10. Only the top row of C changes. Why? (Row 0 of A only takes part in row 0 of C.)
    2. Set the whole first column of B to zeros. The first column of C becomes 0.
    3. Make B's three rows `[1, 0]`, `[0, 1]`, `[0, 0]`. Now C is just the first two columns of A. You've built a "column picker".

### The same thing as a triple loop

Here is matrix multiplication written out with no NumPy, so you can see every step. Three nested loops: over rows of the result (`i`), columns of the result (`j`), and the shared inner dimension (`k`).

```python
def matmul(A, B):
    m, n = len(A), len(A[0])
    n2, p = len(B), len(B[0])
    assert n == n2, f"inner dimensions differ: {n} vs {n2}"
    C = [[0] * p for _ in range(m)]
    for i in range(m):              # each row of A
        for j in range(p):          # each column of B
            total = 0
            for k in range(n):      # walk along the row and down the column
                total += A[i][k] * B[k][j]
            C[i][j] = total
    return C

matmul([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]])
# [[58, 64], [139, 154]]
```

## The shape rule

A row of \( A \) has as many numbers as \( A \) has columns (\( n \)). A column of \( B \) has as many numbers as \( B \) has rows. To dot them, those must be equal. And the result has one cell per (row of \( A \), column of \( B \)) pair. So:

\[
\underbrace{A}_{m \times n} \; \underbrace{B}_{n \times p} = \underbrace{C}_{m \times p}
\]

Write the shapes next to each other: `(m, n) @ (n, p)`. **The inner dimensions must match, the outer dimensions survive.** The inner \( n \) is "used up" by the sum over \( k \).

<figure class="diagram" markdown>
![Diagram of the shape rule: (m, n) @ (n, p) → (m, p). The two inner n's are joined by a red bracket labelled "must be equal", and arcs carry the outer m and p to the result shape. Examples: (2, 3) @ (3, 2) → (2, 2); (3,) @ (3,) → a single number; (32, 784) @ (784, 128) → (32, 128); (2, 3) @ (2, 3) → ValueError](../../assets/img/phase-1/day10-shape-rule.svg)
<figcaption>Animated: the inner dimensions cancel; the outer ones become the shape of the result.</figcaption>
</figure>

Get in the habit of checking shapes *before* you run anything. In ML code, most bugs are shape bugs, and this rule finds them on paper.

## `@` is not `*`

From Day 9 you know `*` is **element-wise**: it multiplies matching cells and broadcasts if the shapes allow. `@` is **matrix multiplication**: rows times columns, summed. They are completely different operations.

```python
A * A               # element-wise: each number squared
# array([[ 1,  4,  9],
#        [16, 25, 36]])

A * B               # (2, 3) * (3, 2): no broadcasting rule fits
# ValueError: operands could not be broadcast together with shapes (2,3) (3,2)

A @ A               # (2, 3) @ (2, 3): inner 3 vs 2 don't match
# ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0,
# with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 2 is different from 3)
```

!!! warning "Common mistake: `*` on square matrices runs silently"
    With two square matrices of the same shape, both `P * Q` and `P @ Q` work and both return the same shape. You get no error, just wrong numbers. If you mean a matrix multiply, write `@`. (`np.dot` also does matrix multiplication for 2-D arrays, and you will see it in older code; prefer `@`.)

## Order matters: AB ≠ BA

With ordinary numbers \( 3 \times 5 = 5 \times 3 \). With matrices, swapping the order usually changes the answer, and often the swapped version doesn't even have a valid shape (try `B @ A` vs `A @ B` above: one is 2×2, the other 3×3).

Even for two square matrices of the same size:

```python
P = np.array([[1, 2],
              [3, 4]])
Q = np.array([[0, 1],
              [1, 0]])

P @ Q        # Q on the right swaps P's COLUMNS
# array([[2, 1],
#        [4, 3]])
Q @ P        # Q on the left swaps P's ROWS
# array([[3, 4],
#        [1, 2]])
P * Q        # and element-wise is something else again
# array([[0, 2],
#        [3, 0]])
```

Matrix multiplication is **not commutative**. It is, however, **associative**: \( (AB)C = A(BC) \). You can choose where to put the brackets, but not change the order.

```python
rng = np.random.default_rng(0)
X = rng.normal(size=(4, 3))
Y = rng.normal(size=(3, 5))
Z = rng.normal(size=(5, 2))
np.allclose((X @ Y) @ Z, X @ (Y @ Z))   # True   (equal up to float rounding)
```

Associativity is what lets a chain of layers or transformations be multiplied together in any grouping, and (as Exercise 5 shows) the grouping can change the cost enormously.

!!! android "Android bridge: `android.graphics.Matrix` and concat order"
    You have used matrix multiplication every time you combined transforms on a `Canvas` or an `ImageView`. `android.graphics.Matrix` is a 3×3 matrix for 2-D affine transforms (scale, rotate, translate, skew). Combining two transforms is a matrix multiply, and that is why the API has both `preConcat(other)` (the matrix becomes M × other) and `postConcat(other)` (the matrix becomes other × M), and why `setConcat(a, b)` spells out the order. Rotate-then-translate and translate-then-rotate put your bitmap in different places: that's \( AB \neq BA \) on screen. The same goes for `Canvas.translate()` and `Canvas.rotate()` calls, where swapping two lines moves your drawing.

## A tiny real example: shopping baskets

Four customers, three items. Each row of `baskets` is one customer's quantities; `prices` is the price of each item.

```python
#                  apples bread milk
baskets = np.array([[ 6,   1,    2],    # customer 0
                    [ 0,   2,    1],    # customer 1
                    [12,   0,    4],    # customer 2
                    [ 3,   1,    1]])   # customer 3
prices = np.array([0.50, 2.00, 1.20])   # apples, bread, milk

baskets.shape, prices.shape   # ((4, 3), (3,))
baskets @ prices              # (4, 3) @ (3,) → (4,)
# array([ 7.4,  5.2, 10.8,  4.7])
```

Customer 0 pays \( 6 \cdot 0.50 + 1 \cdot 2.00 + 2 \cdot 1.20 = 3.00 + 2.00 + 2.40 = 7.40 \). One dot product per basket, four baskets, one `@`.

Now add a second shop. Each **column** of `store_prices` is one shop's price list:

```python
#                        shop 0  shop 1
store_prices = np.array([[0.50,  0.40],    # apples
                         [2.00,  2.30],    # bread
                         [1.20,  1.00]])   # milk

totals = baskets @ store_prices            # (4, 3) @ (3, 2) → (4, 2)
print(totals)
# [[ 7.4  6.7]
#  [ 5.2  5.6]
#  [10.8  8.8]
#  [ 4.7  4.5]]

totals.argmin(axis=1)    # cheapest shop for each customer
# array([1, 0, 1, 1])
totals.sum(axis=0)       # what all four customers would spend at each shop
# array([28.1, 25.6])
```

The shape rule told you the answer before you ran it: customers (4) × items (3) times items (3) × shops (2) gives customers × shops. The shared dimension, *items*, is what gets multiplied and summed away. In a neural network the same picture holds: inputs × features times features × neurons gives inputs × neurons.

## What it costs, and why GPUs exist

Count the work. \( C \) has \( m \times p \) cells, and each is a dot product of length \( n \): \( n \) multiplications and (about) \( n \) additions. So

\[
\text{cost of } (m \times n)(n \times p) = m \cdot n \cdot p \text{ multiply-adds}
\]

Our 2×3 times 3×2 example: \( 2 \cdot 3 \cdot 2 = 12 \) multiply-adds. Two 1000×1000 matrices: \( 10^9 \), a billion. People often quote **FLOPs** (floating-point operations) instead, counting the multiply and the add separately: \( 2mnp \).

This is why hardware matters:

- **Every cell of \( C \) is independent.** No cell needs another cell's answer, so thousands of them can be computed at the same time. A **GPU** is a chip with thousands of small cores built for exactly this kind of parallel arithmetic.
- **Matrix multiplication is most of the compute in a neural network.** Fully connected layers, the attention and feed-forward blocks of a transformer, and (after some rearranging) convolutions are all matrix multiplies. A common rule of thumb is that an LLM with \( N \) parameters needs about \( 2N \) FLOPs per generated token, almost all of it in matmuls.
- **NPUs** (neural processing units, like Google's TPU and the AI accelerators in phone chips) go further: they contain hardware blocks that multiply whole tiles of matrices per clock cycle. When LiteRT (TensorFlow Lite) hands your model to the GPU delegate or a vendor NPU delegate on Android, the big win is running its `FULLY_CONNECTED` and convolution ops, which are matmuls under the hood, on that hardware.

### Pure Python vs NumPy

Let's time the triple loop from earlier against NumPy on two 200×200 matrices (8 million multiply-adds):

```python
import time
import numpy as np

rng = np.random.default_rng(42)
N = 200
A = rng.random((N, N))
B = rng.random((N, N))
A_list, B_list = A.tolist(), B.tolist()   # plain Python lists for the loop version

t0 = time.perf_counter()
C_loop = matmul(A_list, B_list)           # the triple loop from above
t_loop = time.perf_counter() - t0

t0 = time.perf_counter()
C_np = A @ B
t_np = time.perf_counter() - t0

print(f"multiply-adds:   {N**3:,}")
print(f"pure Python:     {t_loop:.3f} s")
print(f"NumPy A @ B:     {t_np * 1000:.3f} ms")
print(f"speed-up:        ~{t_loop / t_np:,.0f}x")
print("same answer?    ", np.allclose(C_loop, C_np))
```

One run on a 4-core cloud machine printed:

```text
multiply-adds:   8,000,000
pure Python:     0.681 s
NumPy A @ B:     0.881 ms
speed-up:        ~774x
same answer?     True
```

Your numbers will differ (on that machine, repeated runs gave 0.6 to 0.8 s for the loop, about 1 ms for NumPy, so a speed-up of a few hundred to roughly a thousand times). Same arithmetic, same answer. NumPy hands `@` to a **BLAS** library (Basic Linear Algebra Subprograms, here OpenBLAS): compiled code that uses SIMD instructions, multiple cores and cache-friendly tiling. PyTorch does the same on CPU and calls cuBLAS on NVIDIA GPUs. You will never write a matmul loop in real code; you write `@` and let the library pick the fastest path.

!!! warning "Common mistake: timing the first call"
    The first call to a library function can include one-time setup (loading code, spinning up threads). For a fair benchmark, run it a few times, or use `%timeit A @ B` in Jupyter, which repeats it and reports a typical time.

---

## Exercises

1. **By hand, then check.** Compute \( MN \) and \( NM \) for \( M = \begin{bmatrix} 2 & 0 \\ 1 & 3 \end{bmatrix} \) and \( N = \begin{bmatrix} 1 & 4 \\ 2 & 5 \end{bmatrix} \). Write each cell as a dot product first. Then check with NumPy. Are they equal?
2. **Shapes on paper.** Without running anything, give the shape of the result or say "error": `(5, 3) @ (3, 7)`, `(3, 5) @ (3, 7)`, `(10, 1) @ (1, 10)`, `(1, 10) @ (10, 1)`, `(64, 512) @ (512,)`, `(4,) @ (4, 2)`. Then check each with `np.ones(shape)`.
3. **Two views of `W @ x`.** Write `matvec_rows(W, x)` (one dot product per row) and `matvec_cols(W, x)` (a weighted sum of columns, using a loop over `j`). Test both against `W @ x` for a random 4×3 `W` and a 3-vector `x` using `np.allclose`.
4. **When does order not matter?** Find two 2×2 matrices that *do* commute (\( AB = BA \)). Hint: try matrices that are zero everywhere except the diagonal, or try `np.eye(2)` (the identity, Day 11).
5. **Brackets change the bill.** \( A \) and \( B \) are 1000×1000 and \( \mathbf{v} \) is a 1000-vector. Count the multiply-adds for \( (AB)\mathbf{v} \) and for \( A(B\mathbf{v}) \). Then time both in NumPy and check the results are equal.
6. **Playground.** In the stepper above, choose numbers for A and B so that C is all zeros, while neither A nor B is all zeros.

??? success "Solutions"
    ```python
    # 1
    M = np.array([[2, 0], [1, 3]])
    N = np.array([[1, 4], [2, 5]])
    # MN: [2,0]·[1,2]=2   [2,0]·[4,5]=8   [1,3]·[1,2]=7   [1,3]·[4,5]=19
    M @ N    # array([[ 2,  8],
             #        [ 7, 19]])
    N @ M    # array([[ 6, 12],
             #        [ 9, 15]])   not equal: order matters

    # 2
    # (5, 3) @ (3, 7)     → (5, 7)
    # (3, 5) @ (3, 7)     → error: inner 5 vs 3
    # (10, 1) @ (1, 10)   → (10, 10)  an "outer product": every pair multiplied
    # (1, 10) @ (10, 1)   → (1, 1)    a dot product wrapped in a 1×1 matrix
    # (64, 512) @ (512,)  → (64,)     a layer of 64 neurons applied to one input
    # (4,) @ (4, 2)       → (2,)      a 1-D vector on the left acts as a row
    for s1, s2 in [((5, 3), (3, 7)), ((3, 5), (3, 7)), ((10, 1), (1, 10)),
                   ((1, 10), (10, 1)), ((64, 512), (512,)), ((4,), (4, 2))]:
        try:
            print(s1, s2, (np.ones(s1) @ np.ones(s2)).shape)
        except ValueError:
            print(s1, s2, "ValueError")
    # (5, 3) (3, 7) (5, 7)
    # (3, 5) (3, 7) ValueError
    # (10, 1) (1, 10) (10, 10)
    # (1, 10) (10, 1) (1, 1)
    # (64, 512) (512,) (64,)
    # (4,) (4, 2) (2,)

    # 3
    def matvec_rows(W, x):
        return np.array([row @ x for row in W])

    def matvec_cols(W, x):
        out = np.zeros(W.shape[0])
        for j in range(W.shape[1]):
            out += x[j] * W[:, j]
        return out

    rng = np.random.default_rng(1)
    W = rng.normal(size=(4, 3))
    x = rng.normal(size=3)
    np.allclose(matvec_rows(W, x), W @ x)   # True
    np.allclose(matvec_cols(W, x), W @ x)   # True

    # 4: diagonal matrices commute with each other, and the identity commutes with everything
    D1 = np.array([[2, 0], [0, 3]])
    D2 = np.array([[5, 0], [0, 7]])
    D1 @ D2                              # array([[10,  0],
                                         #        [ 0, 21]])
    np.array_equal(D1 @ D2, D2 @ D1)     # True
    P = np.array([[1., 2.], [3., 4.]])
    np.array_equal(np.eye(2) @ P, P @ np.eye(2))   # True
    ```

    **5.** \( (AB)\mathbf{v} \): \( AB \) costs \( 1000^3 = 10^9 \) multiply-adds, then times \( \mathbf{v} \) adds \( 10^6 \): about a billion. \( A(B\mathbf{v}) \): two matrix-vector products of \( 10^6 \) each: two million. Same answer, about 500 times less work.

    ```python
    import time
    rng = np.random.default_rng(0)
    A = rng.random((1000, 1000))
    B = rng.random((1000, 1000))
    v = rng.random(1000)

    t0 = time.perf_counter(); r1 = (A @ B) @ v; t1 = time.perf_counter() - t0
    t0 = time.perf_counter(); r2 = A @ (B @ v); t2 = time.perf_counter() - t0
    print(f"(A @ B) @ v: {t1*1000:.1f} ms")
    print(f"A @ (B @ v): {t2*1000:.2f} ms")
    print(np.allclose(r1, r2))
    # one run printed (yours will differ):
    # (A @ B) @ v: 29.3 ms
    # A @ (B @ v): 0.42 ms
    # True
    ```

    **6.** For example, A = `[[1, 1, 0], [2, 2, 0]]` and B = `[[1, 1], [-1, -1], [5, 5]]`. Every row of A is `[a, a, 0]`, and every column of B is `[b, -b, anything]`, so each dot product is \( ab - ab + 0 = 0 \). With ordinary numbers, \( xy = 0 \) means one of them is 0; with matrices it doesn't.

## Resources

- [3Blue1Brown: Essence of Linear Algebra, ch. 3 "Linear transformations and matrices"](https://www.youtube.com/watch?v=kYB8IZa5AuE) and [ch. 4 "Matrix multiplication as composition"](https://www.youtube.com/watch?v=XkY2DOUCWMU): why matrix multiplication is defined this way, and why order matters. Watch ch. 3 before Day 11.
- [NumPy: `numpy.matmul` reference](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html): the exact rules for `@`, including 1-D and stacked (3-D) inputs.
- [Dive into Deep Learning, 2.3 Linear Algebra](https://d2l.ai/chapter_preliminaries/linear-algebra.html): matrices, matrix-vector and matrix-matrix products in code, with the ML angle.
- [CS231n: Python NumPy tutorial](https://cs231n.github.io/python-numpy-tutorial/): short, practical, covers `@`/`dot` vs `*`.
- [Khan Academy: Linear algebra](https://www.khanacademy.org/math/linear-algebra): practice problems on matrix products, with hints.
- [Mathematics for Machine Learning (free book), ch. 2](https://mml-book.github.io/): section 2.2 defines matrices and their products formally.
- [Immersive Linear Algebra](http://immersivemath.com/ila/index.html): an interactive book; its matrix chapters let you poke at products like the stepper above.

## Checklist

- [ ] I can read the shape of a matrix and find \( A_{ij} \), and I know NumPy's `A[i-1, j-1]` is math's \( A_{ij} \).
- [ ] I can compute \( W\mathbf{x} \) both as row dot products and as a weighted sum of columns.
- [ ] I computed the 2×3 times 3×2 example by hand and got [[58, 64], [139, 154]].
- [ ] I can predict the result shape of `A @ B` (or the error) from "inner match, outer survive".
- [ ] I know `@` is matrix multiplication and `*` is element-wise.
- [ ] I can show \( AB \neq BA \) with an example, and I know \( (AB)C = A(BC) \).
- [ ] I can count the cost \( m \cdot n \cdot p \) and explain why GPUs and NPUs are built for matmul.
- [ ] I ran the pure-Python vs NumPy timing and played with the stepper.

**Next:** Day 11: Transpose, identity and inverse (coming soon)
