# Day 10: Matrices and Matrix Multiplication

**Phase 1 · Week 2 · ~150 minutes**

Day 5 ended with a preview: a batch of vectors stored as the rows of a 2-D array, and one `@` that dots a weight vector with every row. Today makes that precise. A matrix is a 2-D grid of numbers, and matrix multiplication is a grid of dot products. It's the operation a neural network spends most of its time on. A good break point is after Part 3, "Matrix times matrix".

## Objectives

- [ ] Read a matrix's shape \( m \times n \), find the element \( A_{ij} \), translate it to NumPy's `A[i-1, j-1]`, and recognise square, zero and diagonal matrices.
- [ ] Add two matrices and multiply a matrix by a number, and say why both are element-wise.
- [ ] Compute a matrix-vector product \( W\mathbf{x} \) by hand in two ways (row dot products, and a weighted sum of columns), and reproduce it with `@`.
- [ ] Explain what `@` does with a 1-D operand on the left and on the right, and avoid the `(n, 1)` column broadcasting bug.
- [ ] Compute a matrix-matrix product by hand with \( C_{ij} = \sum_k A_{ik} B_{kj} \), write it as a triple loop, and read it column by column.
- [ ] Predict the shape of `A @ B` (or that it fails) with "inner dimensions match, outer dimensions survive", and read NumPy's `matmul` error message, gufunc signature included.
- [ ] Tell `@` from `*`, `np.dot` and the old `np.matrix` class, and replace `A @ 2` with `2 * A`.
- [ ] Use the rules of matrix algebra (not commutative; associative; distributive; scalars pull out) and explain why checking them needs `np.allclose`, not `==`.
- [ ] Predict the dtype of a matrix product, and spot silent `int32` overflow in an integer `@`.
- [ ] Multiply stacks of matrices with batched `@`, predict the shape, and explain why `np.dot` gives a different answer on 3-D input.
- [ ] Count the cost of a matrix multiply (\( m \cdot n \cdot p \) multiply-adds, \( 2mnp \) FLOPs), time it against a Python loop, and explain why GPUs and phone NPUs are built around it.

**Prerequisites.**

- [Day 2](day-02-python-types.md): floats round, so `0.1 + 0.2 != 0.3`.
- [Day 2a](day-02a-strings.md): f-strings and the `:,` format spec (thousands separators).
- [Day 2b](day-02b-collections.md): nested lists, list comprehensions, and why `[[0] * p] * m` shares one row.
- [Day 3a](day-03a-classes-modules.md): operators call dunder methods (`a @ b` calls `a.__matmul__(b)`).
- [Day 3b](day-03b-errors-files-json.md): reading an error message, `try` / `except ... as e`, and `assert condition, message`.
- [Day 4](day-04-numpy-arrays.md): 2-D arrays and `shape`, `np.zeros`, `np.ones`, `np.eye`, NumPy scalars (`np.int64(3)`), integer overflow and type promotion, `astype`, `argmin` / `sum` with `axis=`, `rng.random` / `rng.normal` / `rng.integers`, `.tolist()`, `%timeit`, and why NumPy beats a Python loop (compiled loops, contiguous memory, SIMD).
- [Day 4b](day-04b-numpy-toolkit.md): `np.allclose` (and why floats need a tolerance) and `np.array_equal`.
- [Day 5](day-05-vectors.md): vectors, linear combinations, the dot product (its Σ formula, `@` on two 1-D arrays, `*` as the element-wise or Hadamard product, `np.dot` as the older name), and the `matmul` "core dimension" error.
- [Day 8](day-08-shapes-indexing.md): indexing a row `A[1]` and a column `A[:, 1]`, adding an axis with `None`, `.T` on a 2-D array, and `np.stack`.
- [Day 9](day-09-broadcasting.md): broadcasting, and how `(n, 1)` and `(n,)` combine.

!!! info "How to read the code on this page"
    Every Python block has a **▶ Run** button, and blocks share their variables from top to bottom like notebook cells, so run them in order. A bare expression on its own line shows its `repr()`, exactly as the `# comment` beside or below it says. Lines that fail on purpose are wrapped in `try` / `except` so the rest of the block still runs, and the comment shows the real error message. The outputs come from 64-bit NumPy on a laptop; in the browser NumPy's default integer is `int32`, so where this page shows `np.int64(3)` or `dtype('int64')`, ▶ Run shows `np.int32(3)` or `dtype('int32')` ([How lessons work](../../how-lessons-work.md) explains why). Part 7 shows the one place in this lesson where that difference changes the numbers.

---

## The big idea

A **matrix** is a grid of numbers with rows and columns. **Matrix multiplication** is an organised way of doing *many dot products at once*: every row of the first matrix is dotted with every column of the second, and each result lands in its own cell.

It's worth learning thoroughly, because it's most of what a neural network does. A layer of a network is a matrix multiply (Day 12). An LLM generating one token is, by compute, mostly matrix multiplies. The GPU in your laptop and the NPU in your phone exist mainly to do this one operation fast.

**Analogy: a nutrition label.** A food app keeps a table: for each ingredient (oats, milk, banana), how many kilocalories and grams of protein are in 100 g. You log a breakfast: 50 g of oats, 200 g of milk, one 100 g banana. The app's job is to turn "how much of each ingredient" into "how much of each nutrient". For the calorie line, it multiplies each amount by that ingredient's calories and adds up: a dot product. For the protein line, it does the same with the protein numbers: a second dot product. A whole week of meals is the same work for every meal. That table of numbers is a matrix, and "every nutrient for every meal" is one matrix multiplication. This example runs through the whole lesson.

---

**Part 1: Matrices**

## What a matrix is

**What it is.** A matrix with \( m \) rows and \( n \) columns is called an **\( m \times n \) matrix** ("m by n"), and \( m \times n \) is its **shape**. Rows always come first.

\[
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}
\qquad A \text{ is } 2 \times 3
\]

One symbol at a time:

- Matrices are written as **uppercase** letters: \( A, B, W \). Vectors stay bold lowercase (\( \mathbf{x} \)), as on Day 5.
- \( A_{ij} \) is the **element** (or **entry**) in **row \( i \), column \( j \)**. In maths, counting starts at 1, so \( A_{13} = 3 \) (row 1, column 3) and \( A_{21} = 4 \).
- \( \mathbb{R}^{m \times n} \) means "the set of all \( m \times n \) grids of real numbers". \( W \in \mathbb{R}^{128 \times 784} \) reads "\( W \) is a 128-by-784 grid of real numbers".
- A matrix with as many rows as columns (\( n \times n \)) is a **square** matrix. Square matrices matter because they're the only ones that can be multiplied by themselves, and (Part 5) the only ones where a wrong operator gives no error.

**Why it exists.** A matrix holds anything with two indices: a dataset (rows are samples, columns are features), a layer's weights (rows are neurons, columns are inputs), a grayscale image (rows and columns of pixels), the nutrition table (rows are nutrients, columns are ingredients).

**Intuition first.** You can read a matrix in two ways, and both matter today:

- as a **stack of row vectors**: each row is one thing (one nutrient, one neuron, one data point);
- as a **row of column vectors**: each column is one thing (one ingredient, one feature, one direction).

**Example.** In NumPy, a matrix is a 2-D array (Day 4), and counting starts at 0:

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])
A.shape      # (2, 3)              (rows, columns)
A[0, 2]      # np.int64(3)         maths' A_13: NumPy counts from 0
A[1, 0]      # np.int64(4)         maths' A_21
A[1]         # array([4, 5, 6])    row 1, a vector
A[:, 1]      # array([2, 5])       column 1, a vector
```

One element comes back as a NumPy scalar, which displays as `np.int64(3)` (Day 4); in the browser ▶ Run shows `np.int32(3)`, because its default integer is `int32`. `A[1]` and `A[:, 1]` are Day 8's row and column indexing.

**Underneath.** The array is one block of six numbers in row order, `1 2 3 4 5 6`, plus a header saying "shape `(2, 3)`" (Day 4). Element `A[i, j]` lives at position \( i \cdot 3 + j \) of the block (Day 8). Nothing in memory says "matrix": it's a 2-D array that we choose to treat with matrix rules.

**How it goes wrong: copying a 1-based index into code.** Papers write the bottom-left element of a 2-row matrix as \( A_{21} \). Typed straight into NumPy:

```python
try:
    A[2, 0]
except IndexError as e:
    print("IndexError:", e)
# IndexError: index 2 is out of bounds for axis 0 with size 2
```

"axis 0 with size 2" means there are only rows 0 and 1. **Recognise it**: an `IndexError` that names the size of the axis you were indexing, right after you translated a formula. **Fix**: subtract 1 from every index (\( A_{21} \) is `A[1, 0]`), or better, loop over `range(m)` and let the code do the counting. The worse case is when the index *doesn't* run off the end: `A[1, 1]` for \( A_{11} \) silently reads the wrong element.

### Special matrices: zero, diagonal, identity

Three shapes of matrix get names, because they come up constantly:

- The **zero matrix** has every element 0. Multiplying by it gives zeros, like multiplying a number by 0. `np.zeros((2, 3))` (Day 4) makes one.
- A **diagonal matrix** is square, and zero everywhere except the **main diagonal**, the elements \( D_{11}, D_{22}, \dots \) running from top-left to bottom-right. `np.diag` builds one from a list of the diagonal values.
- The **identity matrix** \( I \) is the diagonal matrix with ones on the diagonal. `np.eye(n)` from Day 4 makes it. It plays the part of the number 1: \( IA = A \). Day 11 is about it.

```python
np.zeros((2, 3))      # array([[0., 0., 0.],
                      #        [0., 0., 0.]])
np.diag([2, 3])       # array([[2, 0],
                      #        [0, 3]])
np.eye(2)             # array([[1., 0.],
                      #        [0., 1.]])
```

Multiplying by a diagonal matrix scales each axis by its own number: `np.diag([2, 3]) @ v` doubles `v`'s first entry and triples its second. That's why two diagonal matrices give the same answer in either order (Exercise 4).

## Adding matrices and multiplying by a number

**What it is.** Two matrices of the **same shape** are added element by element: \( (A + B)_{ij} = A_{ij} + B_{ij} \). Multiplying a matrix by a number \( c \) (a **scalar**) multiplies every element: \( (cA)_{ij} = c\,A_{ij} \). These are the vector rules of Day 5, applied to a grid.

**Why it exists.** Day 12's layer is \( W\mathbf{x} + \mathbf{b} \): a product, then an addition. Every training step does \( W - \eta \, G \): subtract a scaled gradient matrix \( G \) (\( \eta \), "eta", is the learning rate, a small number). And the distributive law in Part 6 is a statement about adding matrices.

**Example.** Two people log the same two meals (rows) of oats, milk and banana (columns), in units of 100 g:

```python
alex = np.array([[0.5, 2.0, 1.0],     # breakfast
                 [0.0, 3.0, 2.0]])    # smoothie
sam  = np.array([[1.0, 2.5, 0.0],
                 [0.0, 1.0, 1.0]])
alex + sam            # array([[1.5, 4.5, 1. ],
                      #        [0. , 4. , 3. ]])
2 * alex              # array([[1., 4., 2.],
                      #        [0., 6., 4.]])
```

In NumPy these are Day 4's element-wise `+` and `*`; nothing new runs underneath. **How it goes wrong**: different shapes don't fail as maths says they should. NumPy broadcasts (Day 9), so `alex + np.array([1.0, 0.0, 0.0])` adds the row to every meal, and a `(2, 1)` column would be added across. Maths has no such rule. If two matrices you meant to add have different shapes, broadcasting can hide the bug, so check `a.shape == b.shape` first.

---

**Part 2: Matrix times vector**

## Matrix times vector, two ways

**What it is.** Multiplying an \( m \times n \) matrix \( W \) by a vector \( \mathbf{x} \) of \( n \) numbers gives a vector of \( m \) numbers, written \( W\mathbf{x} \). Each output number is one row of \( W \) dotted with \( \mathbf{x} \).

**Why it exists.** It turns "how much of each input" into "how much of each output" in one operation. In the nutrition app, inputs are ingredient amounts and outputs are nutrients. In a network layer (Day 12), inputs are features and outputs are neurons. Tomorrow (Day 11), it moves points in space.

**Intuition first.** Here is the app's table. Each **row** is one nutrient, each **column** is one ingredient, and each number is "per 100 g":

| | oats | milk | banana |
|---|---|---|---|
| kcal | 380 | 60 | 90 |
| protein (g) | 13 | 3 | 1 |

Breakfast, in units of 100 g, is \( \mathbf{x} = [0.5, 2, 1] \): half a unit of oats, two of milk, one banana. The calories are the kcal row dotted with \( \mathbf{x} \); the protein is the protein row dotted with \( \mathbf{x} \). Two rows, two dot products, two answers.

**The formula, one symbol at a time.** Write the table as a \( 2 \times 3 \) matrix \( N \). Then

\[
y_i = \sum_{j=1}^{n} N_{ij} \, x_j
\]

- \( \mathbf{y} = N\mathbf{x} \) is the output vector, and \( y_i \) is its entry number \( i \) (here \( y_1 \) is kcal, \( y_2 \) is protein).
- \( n \) is the number of **columns** of \( N \), which must equal the length of \( \mathbf{x} \): one amount per ingredient. Here \( n = 3 \).
- \( j \) walks along row \( i \) of \( N \) and down \( \mathbf{x} \) at the same time. The Σ is Day 5's "add up for \( j \) from 1 to \( n \)", so \( y_i \) is a dot product.

**Way 1, by hand: each output is a row dotted with \( \mathbf{x} \).**

\[
N\mathbf{x} =
\begin{bmatrix} 380 & 60 & 90 \\ 13 & 3 & 1 \end{bmatrix}
\begin{bmatrix} 0.5 \\ 2 \\ 1 \end{bmatrix}
= \begin{bmatrix} 380 \cdot 0.5 + 60 \cdot 2 + 90 \cdot 1 \\ 13 \cdot 0.5 + 3 \cdot 2 + 1 \cdot 1 \end{bmatrix}
= \begin{bmatrix} 190 + 120 + 90 \\ 6.5 + 6 + 1 \end{bmatrix}
= \begin{bmatrix} 400 \\ 13.5 \end{bmatrix}
\]

Breakfast is 400 kcal and 13.5 g of protein. This is the **neuron view**: in a layer, each row is one neuron's weight vector, and each output says how strongly the input matches that neuron's pattern (Day 5's "a neuron is a dot product").

**Way 2, by hand: the output is a weighted sum of the columns.** Group the same six products by column instead of by row. Each column is one ingredient's nutrients, and \( \mathbf{x} \) says *how much of each column* to take:

\[
N\mathbf{x} = 0.5 \begin{bmatrix} 380 \\ 13 \end{bmatrix} + 2 \begin{bmatrix} 60 \\ 3 \end{bmatrix} + 1 \begin{bmatrix} 90 \\ 1 \end{bmatrix}
= \begin{bmatrix} 190 \\ 6.5 \end{bmatrix} + \begin{bmatrix} 120 \\ 6 \end{bmatrix} + \begin{bmatrix} 90 \\ 1 \end{bmatrix}
= \begin{bmatrix} 400 \\ 13.5 \end{bmatrix}
\]

Same answer, different story. This is the **recipe view**: \( N\mathbf{x} \) is a linear combination (Day 5) of the columns of \( N \), with the entries of \( \mathbf{x} \) as the coefficients. It's the view Day 11 needs, where each column says where one axis of space ends up, and Day 23 builds "span" and "rank" on it.

<figure class="diagram" markdown>
![Animation in two panels. Left, "Row view": the 2 by 3 table N with rows kcal 380, 60, 90 and protein 13, 3, 1, next to x = 0.5, 2, 1. Row 0 lights up and the result 400 appears with the sum 380·0.5 + 60·2 + 90·1 = 400; then row 1 lights up and 13.5 appears with 13·0.5 + 3·2 + 1·1 = 13.5. Right, "Column view": 0.5 times column [380, 13] plus 2 times [60, 3] plus 1 times [90, 1], each column lighting up in turn, then = [190, 6.5] + [120, 6] + [90, 1] and = [400, 13.5]](../../assets/img/phase-1/day10-two-views.svg)
<figcaption>Animated: the same six products, grouped by rows of N (one dot product per output) or by columns of N (a mix of the columns).</figcaption>
</figure>

**The code reproduces both.** `@` is Python's matrix-multiplication operator; Day 5 used it on two 1-D arrays, where it means the dot product.

```python
N = np.array([[380, 60, 90],     # kcal per 100 g of oats, milk, banana
              [ 13,  3,  1]])    # protein (g) per 100 g
x = np.array([0.5, 2.0, 1.0])    # breakfast, in units of 100 g

N @ x                                     # array([400. ,  13.5])

# Way 1: dot each row with x
np.array([N[0] @ x, N[1] @ x])            # array([400. ,  13.5])

# Way 2: weighted sum of the columns
x[0] * N[:, 0] + x[1] * N[:, 1] + x[2] * N[:, 2]   # array([400. ,  13.5])
```

All three agree with the hand result. (The result is `float64` although `N` holds integers, because `x` is float: Part 7.)

**Underneath: a trace.** Way 1 is two of Day 5's accumulator loops, one per row:

| row \( i \) | step \( j \) | \( N_{ij} \) | \( x_j \) | product | running total |
|---|---|---|---|---|---|
| 0 (kcal) | 1 | 380 | 0.5 | 190 | 190 |
| 0 | 2 | 60 | 2 | 120 | 310 |
| 0 | 3 | 90 | 1 | 90 | **400** |
| 1 (protein) | 1 | 13 | 0.5 | 6.5 | 6.5 |
| 1 | 2 | 3 | 2 | 6 | 12.5 |
| 1 | 3 | 1 | 1 | 1 | **13.5** |

Way 2 does the same six multiplications and the same additions, in a different order: it keeps a running total *vector* of length 2 and adds one scaled column at a time.

!!! warning "Common mistake: the vector has the wrong length"
    `x` needs one entry per **column** of the matrix. A meal logged with only two ingredients fails, with Day 5's message:

    ```python
    try:
        N @ np.array([0.5, 2.0])
    except ValueError as e:
        print("ValueError:", e)
    # ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 2 is different from 3)
    ```

    "size 2 is different from 3": the matrix has 3 columns, the vector 2 entries. Part 4 explains the whole message. **Fix:** make the vector as long as the matrix is wide (here, log `0.0` for the missing banana).

## How `@` treats a 1-D array

**What it is.** A 1-D array of shape `(3,)` has no rows or columns, only one axis. So `@` decides by position:

- **On the right** (`N @ x`), the 1-D array is treated as a **column**, shape `(3, 1)`. The product has shape `(2, 1)`, and then the added 1 is **removed**, so the result is `(2,)`.
- **On the left** (`x @ M`), the 1-D array is treated as a **row**, shape `(1, 3)`. The product has shape `(1, p)`, and the added 1 is removed, giving `(p,)`.
- **Both 1-D** (`u @ v`): row times column, `(1, n) @ (n, 1)` is `(1, 1)`, and both added axes are removed: a single number, Day 5's dot product.

That's the rule from the [`numpy.matmul` reference](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html), word for word in spirit: a 1-D argument is promoted to a matrix by adding a 1 to its shape (before if it's first, after if it's second), and the added 1 is removed afterwards.

**Why it exists.** It lets you write `W @ x` for one input and `x @ v` for a dot product without reshaping anything, and the results come back 1-D, the same shape as the input.

**Intuition first.** A 1-D array is "a list of numbers with no orientation". `@` gives it whichever orientation makes the product make sense in that position, does the matrix product, and then drops the orientation again.

**Example.** You can do the promotion yourself with Day 8's `None` and watch the extra axis survive:

```python
x[:, None].shape              # (3, 1)   x as an explicit column
N @ x[:, None]                # array([[400. ],
                              #        [ 13.5]])     shape (2, 1): nothing removed
N @ x                         # array([400. ,  13.5])   shape (2,): the 1 removed
```

**How it goes wrong (1): swapping the operands.** `x @ N` puts the vector on the left, so it's treated as a `(1, 3)` row and multiplied by a `(2, 3)` matrix. The inner sizes are 3 and 2:

```python
try:
    x @ N
except ValueError as e:
    print("ValueError:", e)
# ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 2 is different from 3)
```

**Recognise it**: the error appears although you "only" changed the order. **Fix**: `N @ x`. (`x @ N.T` would also work: Day 11 explains why.)

**How it goes wrong (2): a column vector meets a 1-D bias.** Code that keeps inputs as explicit `(n, 1)` columns, as many textbooks do, then adds a 1-D bias, gets broadcasting (Day 9) instead of addition, with no error:

```python
b = np.array([10.0, 20.0])            # one bias per output
N @ x + b                             # array([410. ,  33.5])    (2,) + (2,): right
N @ x[:, None] + b                    # array([[410. , 420. ],
                                      #        [ 23.5,  33.5]])  (2, 1) + (2,) → (2, 2): wrong
```

A `(2, 1)` column plus a `(2,)` row stretches into a `(2, 2)` grid: Day 9's outer-grid broadcast. **Recognise it**: a result with one more axis than you expected, often noticed much later as a shape error somewhere else. **Fix**: keep everything 1-D (`N @ x`), or keep everything 2-D (a bias of shape `(2, 1)`), never a mix.

---

**Part 3: Matrix times matrix**

## Matrix times matrix

**What it is.** If the second operand has several columns, each column is its own vector, so \( AB \) is "\( A \) times each column of \( B \)", with the answers placed side by side. Cell by cell:

\[
C = AB \qquad C_{ij} = \sum_{k=1}^{n} A_{ik} \, B_{kj}
\]

One symbol at a time:

- \( C_{ij} \) is the cell in row \( i \), column \( j \) of the result.
- \( n \) is the number of columns of \( A \), which must equal the number of rows of \( B \).
- \( k \) walks along **row \( i \) of \( A \)** (\( A_{i1}, A_{i2}, \dots \)) and at the same time down **column \( j \) of \( B \)** (\( B_{1j}, B_{2j}, \dots \)).
- Multiply the pairs and add them up. So \( C_{ij} \) = (row \( i \) of \( A \)) · (column \( j \) of \( B \)).

**Why it exists.** It does many matrix-vector products in one operation: the nutrients of every meal in a week, a layer applied to a whole batch of inputs (Day 12), or two transformations combined into one (Day 11).

**Intuition first.** Put a week of meals side by side as the columns of one matrix. Each column of the answer is one meal's nutrients, computed exactly as in Part 2. Matrix multiplication is "Part 2, once per column".

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
C                  # array([[ 58,  64],
                   #        [139, 154]])
C.shape            # (2, 2)

A[0] @ B[:, 1]     # np.int64(64)   C[0, 1] is row 0 of A · column 1 of B
np.matmul(A, B)    # array([[ 58,  64],
                   #        [139, 154]])   np.matmul is the function behind @
```

The code reproduces the table. `A @ B` calls `A.__matmul__(B)` (Day 3a's operator hooks), which runs `np.matmul(A, B)`: the operator and the function are the same thing.

### Try it: step through a matrix multiply

Press **Step** to fill in C one cell at a time, or hover over a cell of C to see where it comes from. The numbers in A and B are editable: change one and watch which cells of C change.

<div class="matmul-stepper"></div>

!!! example "Experiments to try"
    1. Change `A[0, 0]` from 1 to 10. Only the top row of C changes. Why? (Row 0 of A only takes part in row 0 of C.)
    2. Set the whole first column of B to zeros. The first column of C becomes 0.
    3. Make B's three rows `[1, 0]`, `[0, 1]`, `[0, 0]`. Now C equals the first two columns of A. You've built a "column picker".
    4. Choose numbers so that C is all zeros while neither A nor B is all zeros. (Exercise 4 has an answer.)

### The same thing as a triple loop

**Kotlin comparison.** Here's how you'd write it in Kotlin with `Array<DoubleArray>`:

```kotlin
fun matmul(a: Array<DoubleArray>, b: Array<DoubleArray>): Array<DoubleArray> {
    val m = a.size; val n = a[0].size; val p = b[0].size
    require(b.size == n) { "inner dimensions differ: $n vs ${b.size}" }
    val c = Array(m) { DoubleArray(p) }
    for (i in 0 until m)            // each row of a
        for (j in 0 until p)        // each column of b
            for (k in 0 until n)    // walk along the row and down the column
                c[i][j] += a[i][k] * b[k][j]
    return c
}
```

And the same in Python, on nested lists:

```python
def matmul(A, B):
    m, n = len(A), len(A[0])
    n2, p = len(B), len(B[0])
    assert n == n2, f"inner dimensions differ: {n} vs {n2}"
    C = [[0] * p for _ in range(m)]   # m rows of p zeros (Day 2b: not [[0] * p] * m)
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

What's the same: three nested loops, `i` over rows of the result, `j` over its columns, and `k` over the shared inner dimension, which is the Σ in the formula. What's different: `C` is built with a comprehension so each row is a separate list, and the result is a plain Python list. You'll never write this in real code (Part 9 times it), but it's the definition, and it's what the stepper animates.

### Reading AB column by column, and row by row

**Column view.** Write \( \mathbf{b}_1, \mathbf{b}_2, \dots \) for the columns of \( B \). Then

\[
AB = \begin{bmatrix} A\mathbf{b}_1 & A\mathbf{b}_2 & \cdots & A\mathbf{b}_p \end{bmatrix}
\]

Column \( j \) of the result is \( A \) times column \( j \) of \( B \): a matrix-vector product, so (by Part 2's recipe view) a mix of \( A \)'s columns.

**Row view.** In the same way, row \( i \) of the result is row \( i \) of \( A \) times \( B \): a mix of \( B \)'s **rows**, with row \( i \) of \( A \) as the coefficients.

```python
np.array_equal((A @ B)[:, 1], A @ B[:, 1])    # True   column 1 of AB = A times column 1 of B
np.array_equal((A @ B)[1], A[1] @ B)          # True   row 1 of AB = row 1 of A times B
```

(`np.array_equal` is Day 4b's exact comparison; integers need no tolerance.)

### The nutrition table, for a whole week of meals

Put four meals side by side as the **columns** of one matrix, ingredients down the side:

```python
#                    breakfast smoothie porridge snack
meals = np.array([[0.5,      0.0,     1.0,     0.0],    # oats   (100 g)
                  [2.0,      3.0,     2.5,     0.0],    # milk
                  [1.0,      2.0,     0.0,     1.5]])   # banana

N.shape, meals.shape       # ((2, 3), (3, 4))
totals = N @ meals         # (2, 3) @ (3, 4) → (2, 4)
totals
# array([[400. , 360. , 530. , 135. ],
#        [ 13.5,  11. ,  20.5,   1.5]])
```

Column 0 is breakfast's `[400, 13.5]` from Part 2: the column view at work. Row 0 is the calories of every meal, row 1 the protein.

**Rows are samples.** Datasets are usually stored the other way round: one **row** per sample (Day 5's "many vectors at once" preview, and Day 12's batches). Then the meals are the rows, and the table must be turned on its side so ingredients run down it. Day 8's `.T` does that (Day 11 explains what it means for a matrix):

```python
plan = meals.T             # (4, 3): one row per meal
per100 = N.T               # (3, 2): ingredients down, nutrients across
daily = plan @ per100      # (4, 3) @ (3, 2) → (4, 2): one row per meal
daily
# array([[400. ,  13.5],
#        [360. ,  11. ],
#        [530. ,  20.5],
#        [135. ,   1.5]])
daily.sum(axis=0)          # array([1425. ,   46.5])    the whole day, per nutrient
daily[:, 1].argmax()       # np.int64(2)                porridge has the most protein
```

`sum(axis=0)` and `argmax` are Day 4's reductions. The shape rule told you the answer's shape before you ran it: meals (4) × ingredients (3) times ingredients (3) × nutrients (2) gives meals × nutrients. The shared dimension, *ingredients*, is multiplied and summed away. In a network the same picture holds: samples × features times features × neurons gives samples × neurons.

### An outer product: a column times a row

The shape rule allows an inner size of 1. A `(3, 1)` column times a `(1, 2)` row is a `(3, 2)` matrix, and each cell is a "dot product" of length 1, a single multiplication:

```python
u = np.array([1, 2, 3])
v = np.array([1, 10])
u[:, None] @ v[None, :]    # array([[ 1, 10],
                           #        [ 2, 20],
                           #        [ 3, 30]])
```

Every row is a multiple of `v`, every column a multiple of `u`. This is the **outer product**, the opposite of the dot product: two vectors in, a whole matrix out. Day 23 builds low-rank matrices and LoRA from it, and shows that any \( AB \) is a sum of outer products.

---

**Part 4: The shape rule**

## Inner dimensions match, outer dimensions survive

**What it is.** A row of \( A \) has as many numbers as \( A \) has columns (\( n \)). A column of \( B \) has as many numbers as \( B \) has rows. To dot them, those must be equal. And the result has one cell per (row of \( A \), column of \( B \)) pair. So:

\[
\underbrace{A}_{m \times n} \; \underbrace{B}_{n \times p} = \underbrace{C}_{m \times p}
\]

Write the shapes next to each other: `(m, n) @ (n, p)`. **The inner dimensions must match, the outer dimensions survive.** The inner \( n \) is "used up" by the sum over \( k \).

**Why it matters.** In ML code, most bugs are shape bugs, and this rule finds them on paper before you run anything.

<figure class="diagram" markdown>
![Diagram of the shape rule: (m, n) @ (n, p) → (m, p). The two inner n's are joined by a red bracket labelled "must be equal", and arcs carry the outer m and p to the result shape. Examples: (2, 3) @ (3, 2) → (2, 2), the worked example above; (3,) @ (3,) → a single number; (32, 784) @ (784, 128) → (32, 128); (2, 3) @ (2, 3) → ValueError](../../assets/img/phase-1/day10-shape-rule.svg)
<figcaption>Animated: the inner dimensions cancel; the outer ones become the shape of the result.</figcaption>
</figure>

**Example.** Check a few with `np.ones(shape)` (Day 4), which makes an array of the right shape without caring about the values:

```python
(np.ones((2, 3)) @ np.ones((3, 2))).shape       # (2, 2)
(np.ones((32, 784)) @ np.ones((784, 128))).shape  # (32, 128)   32 images through 128 neurons
(np.ones((3, 2)) @ np.ones((2, 3))).shape       # (3, 3)      same matrices, other order
```

### Reading the `matmul` error message

**Underneath.** `np.matmul` is a **generalised ufunc** ("gufunc"): a function that works on small blocks of an array (here, matrices) instead of single elements, and loops over any extra leading axes (Part 8). Its error messages quote its **signature**, a one-line description of the shapes it accepts:

```text
(n?,k),(k,m?)->(n?,m?)
```

Read it left to right:

- `(n?,k)` is the core shape of **operand 0**, the left one: `n` rows, `k` columns.
- `(k,m?)` is the core shape of **operand 1**, the right one: `k` rows, `m` columns. The same letter `k` in both places means "these must be equal": it's the inner dimension.
- `->(n?,m?)` is the result: the outer sizes survive, `k` is gone.
- `?` means "this axis may be missing": that's the 1-D case from Part 2, where NumPy adds the axis and removes it afterwards.

Now a full message:

```python
try:
    A @ A               # (2, 3) @ (2, 3): inner 3 vs 2
except ValueError as e:
    print("ValueError:", e)
# ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 2 is different from 3)
```

As Day 5 said, read it from the end. "size 2 is different from 3": the two sizes that should match. "Input operand 1 ... core dimension 0": the right operand's first core axis, its rows, which is the `k` of `(k,m?)`. Operands are counted from 0, so operand 1 is the right-hand one. Translated: "the right matrix has 2 rows, but the left has 3 columns."

!!! warning "Common mistake: `A @ 2` when you meant to scale"
    `@` is only for arrays with at least one axis; a plain number has none (shape `()`). The signature needs at least the `k` axis, so:

    ```python
    try:
        A @ 2
    except ValueError as e:
        print("ValueError:", e)
    # ValueError: matmul: Input operand 1 does not have enough dimensions (has 0, gufunc core with signature (n?,k),(k,m?)->(n?,m?) requires 1)
    ```

    "has 0 ... requires 1": the right operand has 0 axes, and `(k,m?)` needs at least 1. **Fix:** scaling a matrix is element-wise, so write `2 * A` (Part 1). (`np.dot(A, 2)` does accept a scalar, one of the differences between the two functions.)

A summary of the shapes you've now seen, after the teaching:

| Left | Right | Result | Why |
|---|---|---|---|
| `(m, n)` | `(n, p)` | `(m, p)` | inner `n` summed away |
| `(m, n)` | `(n,)` | `(m,)` | 1-D on the right: a column, 1 removed |
| `(n,)` | `(n, p)` | `(p,)` | 1-D on the left: a row, 1 removed |
| `(n,)` | `(n,)` | `()` | a dot product: one number |
| `(m, 1)` | `(1, p)` | `(m, p)` | outer product |
| `(m, n)` | `(k, p)`, `k ≠ n` | `ValueError` | inner sizes differ |
| `(m, n)` | a number | `ValueError` | use `*` to scale |

---

**Part 5: `@` and its look-alikes**

## `@` is not `*`

**What it is.** Day 5 showed this for vectors: `*` is **element-wise** (the Hadamard product, \( A \odot B \)), multiplying matching cells, while `@` is the dot product. For matrices the same two operators mean element-wise multiplication and matrix multiplication. They're completely different operations, and for matrices the mix-up matters more, because square matrices give no error.

```python
A * A               # element-wise: each number squared
# array([[ 1,  4,  9],
#        [16, 25, 36]])

try:
    A * B           # (2, 3) * (3, 2): no broadcasting rule fits (Day 9)
except ValueError as e:
    print("ValueError:", e)
# ValueError: operands could not be broadcast together with shapes (2,3) (3,2)
```

With rectangular matrices, the wrong operator usually fails and tells you. With two **square** matrices of the same shape, both work and both return the same shape:

```python
P = np.array([[1, 2],
              [3, 4]])
Q = np.array([[0, 1],
              [1, 0]])
P @ Q        # array([[2, 1],
             #        [4, 3]])     row · column, summed
P * Q        # array([[0, 2],
             #        [3, 0]])     cell × cell
```

Check the top-left cell by hand: `P @ Q` gives row `[1, 2]` · column `[0, 1]` = 0 + 2 = 2; `P * Q` gives 1 × 0 = 0.

!!! warning "Common mistake: `*` on square matrices runs silently"
    `P * Q` and `P @ Q` above are both `(2, 2)`, and nothing warns you that the numbers are wrong. Transform matrices (2×2, 3×3, 4×4 in graphics), attention scores (sequence × sequence) and covariance matrices are all square, so this bug hides in exactly the code that matters. **Recognise it**: results that are plausible-looking but don't match a hand-worked cell. **Fix**: if you mean a matrix product, write `@`, and test one cell by hand. Exercise 7 is this bug.

### `np.dot`, `np.matmul` and `@`

Day 5 met `np.dot` as the older name for the dot product. On 1-D and 2-D arrays, `np.dot(A, B)`, `np.matmul(A, B)` and `A @ B` give the same answer. They differ in two places:

- **scalars**: `np.dot(A, 2)` scales `A`; `A @ 2` is an error (the box above);
- **3-D and higher arrays**: `@` treats them as stacks of matrices, while `np.dot` does something else entirely. Part 8 shows the difference.

Prefer `@`: it's shorter, it reads like the maths, and its rules for stacks are the ones PyTorch, TensorFlow and JAX use.

!!! warning "Common mistake: `np.matrix`, where `*` *is* matrix multiplication"
    Old tutorials and Stack Overflow answers sometimes use `np.matrix`, a subclass of the array that NumPy no longer recommends. On it, `*` means matrix multiplication, so copied code behaves differently on ordinary arrays:

    ```python
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)   # np.matrix warns on creation
        Pm = np.matrix([[1, 2], [3, 4]])
        Qm = np.matrix([[0, 1], [1, 0]])
    Pm * Qm      # matrix([[2, 1],
                 #         [4, 3]])    the matrix product: same as P @ Q
    ```

    The suppressed warning says "the matrix subclass is not the recommended way to represent matrices or deal with linear algebra ... Please adjust your code to use regular ndarray." **Fix:** use `np.array` and `@`. When reading old code, `np.matrix(...) * ...` translates to `@`.

---

**Part 6: The rules of matrix algebra**

## Order matters: AB ≠ BA

**What it is.** With ordinary numbers \( 3 \times 5 = 5 \times 3 \): multiplication is **commutative**. Matrix multiplication isn't. Swapping the order usually changes the answer, and often the swapped version doesn't even have a valid shape: `A @ B` above is 2×2, while `B @ A` is 3×3.

**Intuition first.** From the row and column views: in \( PQ \), \( Q \) on the right mixes \( P \)'s **columns**; in \( QP \), \( Q \) on the left mixes \( P \)'s **rows**. Mixing columns and mixing rows are different operations.

```python
P @ Q        # Q on the right swaps P's COLUMNS
# array([[2, 1],
#        [4, 3]])
Q @ P        # Q on the left swaps P's ROWS
# array([[3, 4],
#        [1, 2]])
```

`Q` is the identity with its rows swapped, a **permutation matrix**, which is why its effect is so easy to see.

!!! android "Android bridge: `android.graphics.Matrix` and concat order"
    You have used matrix multiplication every time you combined transforms on a `Canvas` or an `ImageView`. `android.graphics.Matrix` is a 3×3 matrix for 2-D affine transforms (scale, rotate, translate, skew). Combining two transforms is a matrix multiply, and that's why the API has both `preConcat(other)` (the matrix becomes M × other) and `postConcat(other)` (the matrix becomes other × M), and why `setConcat(a, b)` spells out the order (a × b). Rotate-then-translate and translate-then-rotate put your bitmap in different places: that's \( AB \neq BA \) on screen. The same goes for `Canvas.translate()` and `Canvas.rotate()` calls, where swapping two lines moves your drawing. Day 11 shows why "the matrix nearest the vector acts first".

## Brackets don't matter: (AB)C = A(BC)

**What it is.** Matrix multiplication is **associative**: \( (AB)C = A(BC) \). You may choose where to put the brackets, but not change the order of the matrices.

**Why it exists.** It's what lets a chain of layers or transformations be combined in any grouping. Day 11 multiplies two transforms into one matrix before applying it; and the grouping can change the cost enormously (Exercise 8).

**By hand, on small matrices.** Take \( X = [1 \;\; 2] \) (1×2), \( Y = \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix} \), \( \mathbf{z} = \begin{bmatrix} 2 \\ 3 \end{bmatrix} \). Left grouping: \( XY = [1 + 2 \;\; 0 + 2] = [3 \;\; 2] \), then \( [3 \;\; 2]\,\mathbf{z} = 6 + 6 = 12 \). Right grouping: \( Y\mathbf{z} = [2 \;\; 5]^\top \) (2 and 2 + 3), then \( X(Y\mathbf{z}) = 2 + 10 = 12 \). Same number, different intermediate results.

**Underneath: why `==` can fail.** On integers both groupings give exactly the same numbers. On floats, the two groupings do the same multiplications but add them up in a different order, and float addition rounds at each step (Day 2's `0.1 + 0.2`), so the last digit can differ:

```python
rng = np.random.default_rng(0)
X = rng.normal(size=(4, 3))
Y = rng.normal(size=(3, 5))
Z = rng.normal(size=(5, 2))
left, right = (X @ Y) @ Z, X @ (Y @ Z)
np.array_equal(left, right)        # False  exact comparison fails
np.abs(left - right).max()         # np.float64(4.440892098500626e-16)
np.allclose(left, right)           # True   equal up to float rounding
```

The biggest difference is \( 4.4 \times 10^{-16} \), the size of a last-digit rounding error. That's why every float check of a matrix identity on this page uses Day 4b's `np.allclose`: the maths is exact, the arithmetic isn't. **How it goes wrong**: `assert (X @ Y) @ Z == X @ (Y @ Z)` doesn't even work, because `==` on arrays gives an array of booleans, and `assert` refuses an array of more than one element (the "truth value is ambiguous" error from Day 4). Use `assert np.allclose(...)`.

## Sums and scalars: the distributive law

**What it is.** Two rules that let you expand and factor matrix expressions, as with numbers:

\[
A(B + C) = AB + AC \qquad (A + B)C = AC + BC \qquad (cA)B = c(AB) = A(cB)
\]

**Why it exists.** Day 12's layer \( W\mathbf{x} + \mathbf{b} \), and every gradient formula in Month 2, are rearranged with these rules. In the nutrition app, they say that totals are additive: the nutrients of "breakfast plus an extra half banana" are breakfast's nutrients plus the extra's; a double portion has double the nutrients.

```python
extra = np.array([0.0, 0.0, 0.5])                 # half a banana
N @ (x + extra)                                    # array([445.,  14.])
N @ x + N @ extra                                  # array([445.,  14.])
N @ (2 * x), 2 * (N @ x)                           # (array([800.,  27.]), array([800.,  27.]))
np.allclose(A @ (B + B), A @ B + A @ B)            # True
```

By hand, the half banana adds \( 0.5 \cdot 90 = 45 \) kcal and \( 0.5 \cdot 1 = 0.5 \) g of protein: 445 and 14. **How it goes wrong**: the order rule still applies inside: \( A(B + C) = AB + AC \), never \( BA + CA \). Expanding \( (A + B)^2 \) gives \( A^2 + AB + BA + B^2 \), not \( A^2 + 2AB + B^2 \), because \( AB \neq BA \).

---

**Part 7: What dtype comes out**

## The dtype of a product

**What it is.** `@` follows Day 4's type promotion: both operands are converted to a common dtype first, and the result has that dtype.

- integer `@` float → float: `N @ x` above was `float64` because `x` held floats;
- integer `@` integer → integer, with **no** promotion to a bigger type: `A @ B` is `int64` on a laptop and `int32` in the browser.

```python
(A @ B).dtype               # dtype('int64')    int32 in the browser
(N @ x).dtype               # dtype('float64')
(A @ B.astype(float)).dtype # dtype('float64')
```

**Why it matters.** A matrix product adds up many products, so its results are much bigger than its inputs. Integer products of counts (word co-occurrence, click matrices, adjacency matrices of graphs) can overflow, and Day 4 showed that array integer overflow is **silent**.

!!! warning "Common mistake: silent `int32` overflow in an integer `@`"
    `int32` holds values up to 2,147,483,647. The top-left cell of this product is \( 100000 \cdot 100000 + 1 \cdot 1 = 10{,}000{,}000{,}001 \), far too big:

    ```python
    big = np.array([[100000, 1],
                    [1,      1]], dtype=np.int32)
    big @ big
    # array([[1410065409,     100001],
    #        [    100001,          2]], dtype=int32)
    big.astype(np.int64) @ big.astype(np.int64)
    # array([[10000000001,      100001],
    #        [     100001,           2]])
    ```

    No error and no warning: the extra bits were dropped, and \( 10{,}000{,}000{,}001 - 2 \times 2^{32} = 1{,}410{,}065{,}409 \). The `dtype=np.int32` here makes the laptop behave like the browser. In ▶ Run, the *default* integer is `int32`, so `np.array([[100000, 1], [1, 1]])` with no dtype overflows in the browser but not on your laptop: the same page gives different numbers. **Recognise it**: impossible values (negative counts, totals smaller than one of their terms) from an integer product. **Fix**: convert to `int64` or `float64` before multiplying; ML code almost always uses floats anyway.

---

**Part 8: Batched matrix multiplication**

## Stacks of matrices: `@` on 3-D arrays

**What it is.** When an operand has more than two axes, `@` treats it as a **stack** of matrices: the **last two axes** are the matrices, and every axis before them is a **batch** axis. `@` multiplies matching matrices from the two stacks, and the batch axes follow Day 9's broadcasting rules. Two common cases:

- `(B, n, k) @ (k, p) → (B, n, p)`: one matrix on the right, reused for every matrix in the stack (it broadcasts along the batch axis);
- `(B, n, k) @ (B, k, p) → (B, n, p)`: two stacks, multiplied pair by pair, slice 0 with slice 0, slice 1 with slice 1.

**Why it exists.** It's how frameworks run a layer on a batch of sequences, and it's the heart of attention: a transformer computes "queries times keys" for every sequence in the batch and every attention head at once, as one batched matmul with shapes like `(batch, heads, seq, dim) @ (batch, heads, dim, seq)`. Without it, you'd write a Python loop over the batch, and Day 4 showed what that costs.

**Intuition first.** Three days of meal plans are three `(4, 3)` matrices stacked into a `(3, 4, 3)` array. The nutrition table is the same every day. "Every day's nutrients" is three separate `plan @ per100` products, and batched `@` does all three in one call.

<figure class="diagram" markdown>
![Animation: a stack of three 4 by 3 grids labelled days, shape (3, 4, 3), then @, then one 3 by 2 grid labelled per100, shape (3, 2), reused for every day, then = and a stack of three 4 by 2 grids labelled out, shape (3, 4, 2). Day 0, day 1 and day 2 light up in turn, each with its matching output slice, while the text reads days[k] @ per100, (4, 3) @ (3, 2) → (4, 2), is out[k]. Below: (3, 4, 3) @ (3, 2) → (3, 4, 2), one matrix on the right reused for every slice; and (2, 3, 4) @ (2, 4, 3) → (2, 3, 3), two stacks paired slice by slice](../../assets/img/phase-1/day10-batched.svg)
<figcaption>Animated: the leading axis is a batch; each slice is an ordinary matrix product.</figcaption>
</figure>

**Example.** Random plans (amounts from 0 to 3 units of 100 g), with Day 4's `rng.integers`:

```python
rng = np.random.default_rng(7)
days = rng.integers(0, 4, size=(3, 4, 3))   # 3 days × 4 meals × 3 ingredients
out = days @ per100                         # (3, 4, 3) @ (3, 2) → (3, 4, 2)
out.shape                                   # (3, 4, 2)

loop = np.stack([days[d] @ per100 for d in range(3)])   # the same, one day at a time
np.array_equal(out, loop)                   # True
out.sum(axis=1)                             # one row per day: total kcal, total protein
# array([[4820,  153],
#        [3500,  105],
#        [4360,  135]])
```

`np.stack` (Day 8) glues the three per-day results into a new leading axis, and the comparison shows batched `@` is exactly "one ordinary `@` per slice". `sum(axis=1)` adds over the meals axis, leaving days × nutrients. The results are integers because both `days` and `per100` are (Part 7).

Two stacks, pair by pair, as in attention with two heads:

```python
Q = rng.normal(size=(2, 3, 4))      # 2 heads, 3 positions, 4 numbers each
K = rng.normal(size=(2, 4, 3))
S = Q @ K                           # (2, 3, 4) @ (2, 4, 3) → (2, 3, 3)
S.shape                             # (2, 3, 3)
np.allclose(S[1], Q[1] @ K[1])      # True   head 1 only meets head 1
```

**Underneath.** NumPy first splits each shape into batch axes and core axes: `(2, 3, 4)` is batch `(2,)` and core `(3, 4)`. It broadcasts the batch shapes (Day 9's rules), checks the core shapes with the signature `(n?,k),(k,m?)->(n?,m?)`, and runs one matrix product per batch position. The result's shape is the broadcast batch shape followed by `(n, m)`. That's what "generalised ufunc" means: an element-wise loop whose "elements" are whole matrices. Broadcasting also works with size-1 batch axes: `(2, 1, 3, 4) @ (5, 4, 2)` gives `(2, 5, 3, 2)`.

**How it goes wrong (1): batch sizes that don't broadcast.**

```python
try:
    np.ones((2, 3, 4)) @ np.ones((3, 4, 5))   # batch 2 vs 3
except ValueError as e:
    print("ValueError:", e)
# ValueError: operands could not be broadcast together with remapped shapes [original->remapped]: (2,3,4)->(2,newaxis,newaxis) (3,4,5)->(3,newaxis,newaxis)  and requested shape (3,5)
```

The core shapes `(3, 4)` and `(4, 5)` are fine; the batch axes 2 and 3 aren't. "remapped shapes ... (2,newaxis,newaxis)" is NumPy setting the core axes aside, leaving the batch shapes `(2,)` and `(3,)`, which don't broadcast. **Fix**: make the batch axes equal, or 1 on one side.

**How it goes wrong (2): `np.dot` on 3-D arrays.** `np.dot` doesn't treat 3-D arrays as stacks. It dots the last axis of the first array with the second-to-last axis of the second, for **every** combination of the other axes, so the batch axes multiply out instead of pairing up:

```python
np.dot(np.ones((2, 3, 4)), np.ones((2, 4, 5))).shape    # (2, 3, 2, 5)   every slice with every slice
(np.ones((2, 3, 4)) @ np.ones((2, 4, 5))).shape         # (2, 3, 5)      slice 0 with 0, 1 with 1
```

`np.dot` computed all four (slice, slice) pairs, including the unwanted 0-with-1 and 1-with-0, and laid them out as `(2, 3, 2, 5)`. With a batch of 64 that's 64 times the work and a shape nobody downstream expects. **Fix**: use `@` (or `np.matmul`) for stacks. This is the real reason to prefer `@` over `np.dot`.

---

**Part 9: Cost and hardware**

## What it costs, and why GPUs exist

**What it is.** Count the work. \( C \) has \( m \times p \) cells, and each is a dot product of length \( n \): \( n \) multiplications and \( n - 1 \) additions, so about \( n \) of each. A multiplication followed by adding the result to a running total is called a **multiply-add**, and

\[
\text{cost of } (m \times n)(n \times p) = m \cdot n \cdot p \text{ multiply-adds}
\]

**By hand.** The 2×3 times 3×2 example: \( 2 \cdot 3 \cdot 2 = 12 \) multiply-adds (count them in the table: 4 cells × 3 products). The nutrition week, \( (2 \times 3)(3 \times 4) \): 24. Two 1000×1000 matrices: \( 10^9 \), a billion.

People often quote **FLOPs** (floating-point operations) instead, counting the multiply and the add separately: \( 2mnp \). Why count them separately? Because chip makers rate their hardware in FLOPs per second, so counting the same way lets you compare a model's cost with a chip's speed. Modern CPUs and GPUs have an instruction for the pair, the **fused multiply-add (FMA)**, which computes \( a \cdot b + c \) in one step with one rounding; one FMA is counted as 2 FLOPs. Doubling \( n \) for square matrices multiplies the cost by 8 (\( n^3 \)), which is why matrix size dominates a model's speed.

**Why the hardware is built around it:**

- **Every cell of \( C \) is independent.** No cell needs another cell's answer, so thousands of them can be computed at the same time. A **GPU** is a chip with thousands of small cores built for exactly this kind of parallel arithmetic.
- **Matrix multiplication is most of the compute in a neural network.** Fully connected layers, the attention and feed-forward blocks of a transformer, and (after some rearranging) convolutions are all matrix multiplies. A common rule of thumb is that an LLM with \( N \) parameters needs about \( 2N \) FLOPs per generated token, almost all of it in matmuls.
- **Accelerators** such as Google's TPU, and the NPUs (neural processing units) in phone chips, go further: they contain hardware blocks that multiply whole tiles of matrices per clock cycle.

!!! android "Android bridge: LiteRT delegates"
    In LiteRT (formerly TensorFlow Lite), a **delegate** is a plug-in that takes over the supported operations of your model and runs them on other hardware, such as the GPU delegate, or a vendor's delegate for the phone's NPU. The ops that matter most are `FULLY_CONNECTED` and the convolutions, which are matmuls underneath: moving those off the CPU is where most of the speed-up comes from. When a delegate can't run an op, LiteRT runs it on the CPU, which is why a model with one unsupported op in the middle can be slower than expected.

### Pure Python vs NumPy

Day 4 timed a Python loop against NumPy and explained the gap (compiled loops, contiguous memory, SIMD). Matrix multiplication widens it further. Time the triple loop from Part 3 against `@` on two 100×100 matrices (a million multiply-adds) with Day 4's `%timeit`:

```python
rng = np.random.default_rng(42)
n = 100
A_big = rng.random((n, n))
B_big = rng.random((n, n))
A_list, B_list = A_big.tolist(), B_big.tolist()   # plain Python lists for the loop version

print(f"multiply-adds: {n**3:,}")
print("same answer?", np.allclose(matmul(A_list, B_list), A_big @ B_big))
%timeit matmul(A_list, B_list)
%timeit A_big @ B_big
```

On my laptop this printed the following; yours will differ, and the browser's ▶ Run is slower again and ends its timing lines with `[browser timing]`:

```text
multiply-adds: 1,000,000
same answer? True
49 ms per loop (best of 3, 5 loops each)
31.6 µs per loop (best of 3, 5000 loops each)
```

That's a ratio of about 1,500 on this machine, more than Day 4's element-wise gap. `%timeit` runs the statement many times and keeps the best run, so one-time start-up costs on the first call (loading code, starting threads) don't distort it. Don't time a single first call with a stopwatch: it can be several times slower than the steady state.

**Underneath: BLAS.** NumPy hands `@` on float arrays to a **BLAS** library (Basic Linear Algebra Subprograms): usually OpenBLAS in the pip wheels for Linux and Windows, and Apple's Accelerate on Apple Silicon Macs (`np.show_config()` tells you which). On top of Day 4's SIMD, BLAS uses several CPU cores and **cache tiling**: it splits the matrices into small blocks that fit in the CPU's fast cache memory and finishes all the work on one block before moving on, so each number is fetched from slow main memory far fewer times. In the browser, NumPy runs as WebAssembly on one thread, so the ratio there is different. PyTorch does the same on CPU and calls cuBLAS on NVIDIA GPUs. You'll never write a matmul loop in real code; you write `@` and let the library pick the fastest path.

---

## Going further

These are real and useful, but not needed yet:

- **`np.einsum`**: one notation for dot products, matrix products, batched products, transposes and traces, written as index strings like `"bik,bkj->bij"`: [numpy.einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html).
- **`np.linalg.matrix_power(A, k)`**: \( A \) multiplied by itself \( k \) times, with repeated squaring: [numpy.linalg.matrix_power](https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_power.html).
- **`np.linalg.multi_dot`**: multiplies a chain of matrices in the cheapest bracket order for you (Exercise 8 does this by hand): [numpy.linalg.multi_dot](https://numpy.org/doc/stable/reference/generated/numpy.linalg.multi_dot.html).
- **`np.matvec` and `np.vecmat`** (NumPy 2.2+): batched matrix-vector products, with the vector always treated as a vector: [linear algebra routines](https://numpy.org/doc/stable/reference/routines.linalg.html).
- **The outer-product view of \( AB \)**, as a sum of rank-1 matrices (column \( k \) of \( A \) times row \( k \) of \( B \)): [Day 23](day-23-rank-span.md), which builds low-rank matrices and LoRA from it.
- **Fast matrix multiplication**: Strassen's algorithm does a 2×2 block product with 7 multiplications instead of 8, giving about \( n^{2.81} \) instead of \( n^3 \): [Strassen algorithm (Wikipedia)](https://en.wikipedia.org/wiki/Strassen_algorithm).
- **Matmul precision on GPUs**: training and inference often use `float16`, `bfloat16` or NVIDIA's TF32 inside matmuls for speed, accumulating in `float32`: [PyTorch: TF32 on Ampere and later](https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices).
- **BLAS, cache tiling and how fast matmul is engineered**: the [`numpy.matmul` reference](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html) for NumPy's rules, and [OpenBLAS](https://www.openblas.net/) for the library underneath.

---

## Exercises

1. **Warm-up: by hand, then check.** Compute \( MN \) and \( NM \) for \( M = \begin{bmatrix} 2 & 0 \\ 1 & 3 \end{bmatrix} \) and \( N = \begin{bmatrix} 1 & 4 \\ 2 & 5 \end{bmatrix} \). Write each cell as a dot product first. Then check with NumPy. Are they equal, and why not?
2. **Warm-up: shapes on paper.** Without running anything, give the shape of the result or say "error": `(5, 3) @ (3, 7)`, `(3, 5) @ (3, 7)`, `(10, 1) @ (1, 10)`, `(1, 10) @ (10, 1)`, `(64, 512) @ (512,)`, `(4,) @ (4, 2)`, `(8, 2, 3) @ (3, 4)`. Then check each with `np.ones(shape)`.
3. **Core: two views of `W @ x`.** Write `matvec_rows(W, x)` (one dot product per row) and `matvec_cols(W, x)` (a weighted sum of columns, using a loop over `j`). Test both against `W @ x` for a random 4×3 `W` and a 3-vector `x` using `np.allclose`.
4. **Core: when does order not matter, and when is a product zero?** (a) Find two 2×2 matrices that *do* commute (\( AB = BA \)). Hint: try diagonal matrices (Part 1), or `np.eye(2)` (Day 4). (b) Find a 2×3 `A` and a 3×2 `B`, neither all zeros, with `A @ B` all zeros (you can use the stepper).
5. **Core: a week of meals, batched.** Using the nutrition table `N` from Part 2, build a `(7, 4, 3)` stack of random meal plans (7 days, 4 meals, 3 ingredients, amounts from `rng.integers(0, 3, ...)` with seed 3). Compute every meal's nutrients with one `@`, then the total kcal of each day, and find the day with the most protein. Check one day against an ordinary 2-D `@`.
6. **Predict the output.** Write down what each `print` shows, then run it.

    ```python
    import numpy as np
    v = np.array([1, 2, 3])
    M = np.array([[1, 0, 0],
                  [0, 2, 0]])
    print(M @ v)
    print(v @ v)
    print((v[:, None] @ v[None, :]).shape)
    print((M @ np.ones(3)).dtype)
    print(np.ones((4, 2, 3)) @ np.ones(3))
    ```

7. **Find and fix the bug.** This code should rotate the point `[1, 0]` by 90° twice (a half turn), which must land on `[-1, 0]`. It runs without an error, but prints the wrong answer. Find the bug, explain why there's no error, and fix it.

    ```python
    import numpy as np
    R = np.array([[0, -1],
                  [1,  0]])        # rotate 90° anticlockwise
    half_turn = R * R              # rotate twice
    print(half_turn @ np.array([1, 0]))
    ```

8. **Challenge: brackets change the bill.** \( A \) and \( B \) are 500×500 and \( \mathbf{v} \) is a 500-vector. Count the multiply-adds for \( (AB)\mathbf{v} \) and for \( A(B\mathbf{v}) \). Then time both with `%timeit` and check the results are equal.

??? success "Solutions"
    **1.** By hand, \( MN \): row `[2, 0]` · column `[1, 2]` = 2, · `[4, 5]` = 8; row `[1, 3]` · `[1, 2]` = 7, · `[4, 5]` = 19. \( NM \): `[1, 4]` · `[2, 1]` = 6, · `[0, 3]` = 12; `[2, 5]` · `[2, 1]` = 9, · `[0, 3]` = 15.

    ```python
    import numpy as np
    M = np.array([[2, 0], [1, 3]])
    N = np.array([[1, 4], [2, 5]])
    M @ N    # array([[ 2,  8],
             #        [ 7, 19]])
    N @ M    # array([[ 6, 12],
             #        [ 9, 15]])
    np.array_equal(M @ N, N @ M)   # False
    ```

    They differ because the two products do different jobs: in \( MN \), \( M \) on the left mixes the **rows** of \( N \) (row 0 of the result is 2 × row 0 of \( N \)); in \( NM \), \( M \) on the right mixes the **columns** of \( N \). Mixing rows and mixing columns are different operations.

    **2.**

    ```python
    import numpy as np
    # (5, 3) @ (3, 7)     → (5, 7)
    # (3, 5) @ (3, 7)     → error: inner 5 vs 3
    # (10, 1) @ (1, 10)   → (10, 10)  an outer product: every pair multiplied
    # (1, 10) @ (10, 1)   → (1, 1)    a dot product wrapped in a 1×1 matrix
    # (64, 512) @ (512,)  → (64,)     1-D on the right: a column, the 1 removed
    # (4,) @ (4, 2)       → (2,)      1-D on the left: a row, the 1 removed
    # (8, 2, 3) @ (3, 4)  → (8, 2, 4) a stack of 8, the right matrix reused
    for s1, s2 in [((5, 3), (3, 7)), ((3, 5), (3, 7)), ((10, 1), (1, 10)),
                   ((1, 10), (10, 1)), ((64, 512), (512,)), ((4,), (4, 2)),
                   ((8, 2, 3), (3, 4))]:
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
    # (8, 2, 3) (3, 4) (8, 2, 4)
    ```

    Every answer comes from "inner match, outer survive", plus Part 2's 1-D rule and Part 8's batch rule. `(1, 10) @ (10, 1)` keeps both 1s because the inputs were 2-D: only axes that `@` added itself are removed.

    **3.**

    ```python
    import numpy as np

    def matvec_rows(W, x):
        return np.array([row @ x for row in W])   # iterating a 2-D array gives its rows

    def matvec_cols(W, x):
        out = np.zeros(W.shape[0])
        for j in range(W.shape[1]):
            out += x[j] * W[:, j]                 # add x_j times column j
        return out

    rng = np.random.default_rng(1)
    W = rng.normal(size=(4, 3))
    x = rng.normal(size=3)
    np.allclose(matvec_rows(W, x), W @ x)   # True
    np.allclose(matvec_cols(W, x), W @ x)   # True
    ```

    `matvec_rows` is the row view (one dot product per output) and `matvec_cols` the column view (a weighted sum of the columns). They use `np.allclose`, not `==`, because each does the same additions in a different order from BLAS, so the last digit of a float can differ (Part 6).

    **4.**

    ```python
    import numpy as np
    # (a) diagonal matrices commute with each other; the identity commutes with everything
    D1 = np.array([[2, 0], [0, 3]])
    D2 = np.array([[5, 0], [0, 7]])
    D1 @ D2                              # array([[10,  0],
                                         #        [ 0, 21]])
    np.array_equal(D1 @ D2, D2 @ D1)     # True
    P = np.array([[1., 2.], [3., 4.]])
    np.array_equal(np.eye(2) @ P, P @ np.eye(2))   # True

    # (b) a zero product from non-zero matrices
    A = np.array([[1, 1, 0],
                  [2, 2, 0]])
    B = np.array([[1, 1],
                  [-1, -1],
                  [5, 5]])
    A @ B                                # array([[0, 0],
                                         #        [0, 0]])
    ```

    (a) A diagonal matrix scales each axis by its own number and leaves the others alone. Scaling axis 0 by 2 and then by 5, or by 5 and then by 2, gives 10 either way, and the same for axis 1, so the order can't matter. The identity scales every axis by 1. (b) Every row of A is `[a, a, 0]` and every column of B is `[b, -b, anything]`, so each dot product is \( ab - ab + 0 = 0 \). With numbers, \( xy = 0 \) means one of them is 0; with matrices it doesn't.

    **5.**

    ```python
    import numpy as np
    N = np.array([[380, 60, 90],
                  [ 13,  3,  1]])
    rng = np.random.default_rng(3)
    week = rng.integers(0, 3, size=(7, 4, 3))    # 7 days × 4 meals × 3 ingredients
    per_meal = week @ N.T                         # (7, 4, 3) @ (3, 2) → (7, 4, 2)
    per_meal.shape                                # (7, 4, 2)
    per_day = per_meal.sum(axis=1)                # (7, 2): kcal and protein per day
    per_day[:, 0]                                 # array([1850, 1420, 2240, 1100, 1070, 1420, 2470])
    per_day[:, 1].argmax()                        # np.int64(6)
    np.array_equal(per_meal[2], week[2] @ N.T)    # True
    ```

    `N.T` is the table with ingredients down the side, `(3, 2)`, so each day's `(4, 3)` plan meets it with inner sizes 3 and 3. Batched `@` reuses that one matrix for all 7 days, and `sum(axis=1)` adds over the meals axis. Everything is an integer here (in the browser, `int32`, which is fine at these sizes).

    **6.** The predictions, then the run:

    - `M @ v`: row `[1, 0, 0]` · `[1, 2, 3]` = 1, row `[0, 2, 0]` · `[1, 2, 3]` = 4, so `[1 4]`.
    - `v @ v`: a dot product, 1 + 4 + 9 = `14`.
    - a `(3, 1)` column times a `(1, 3)` row: an outer product, `(3, 3)`.
    - an integer matrix times a float vector: `float64`.
    - a stack of four `(2, 3)` matrices of ones times a 1-D vector of three ones: each row sums to 3, giving four `[3., 3.]` rows, shape `(4, 2)`.

    ```python
    import numpy as np
    v = np.array([1, 2, 3])
    M = np.array([[1, 0, 0],
                  [0, 2, 0]])
    print(M @ v)
    print(v @ v)
    print((v[:, None] @ v[None, :]).shape)
    print((M @ np.ones(3)).dtype)
    print(np.ones((4, 2, 3)) @ np.ones(3))
    # [1 4]
    # 14
    # (3, 3)
    # float64
    # [[3. 3.]
    #  [3. 3.]
    #  [3. 3.]
    #  [3. 3.]]
    ```

    The last line combines two rules: the 1-D vector on the right becomes a `(3, 1)` column, the `(4, 2, 3)` stack multiplies it slice by slice into `(4, 2, 1)`, and the added 1 is removed, giving `(4, 2)`. `print` shows arrays without commas (Day 4's `str` form).

    **7.** `R * R` is element-wise, not "rotate twice". Both are `(2, 2)`, so nothing fails: Part 5's silent square-matrix bug.

    ```python
    import numpy as np
    R = np.array([[0, -1],
                  [1,  0]])
    R * R                          # array([[0, 1],
                                   #        [1, 0]])   squares of each entry: a swap, not a turn
    half_turn = R @ R
    half_turn                      # array([[-1,  0],
                                   #        [ 0, -1]])
    print(half_turn @ np.array([1, 0]))
    # [-1  0]
    ```

    With `*`, the "half turn" was `[[0, 1], [1, 0]]`, which swaps the coordinates and sent `[1, 0]` to `[0, 1]`: plausible, wrong. `R @ R` is "apply R, then apply R again" (associativity: \( (RR)\mathbf{p} = R(R\mathbf{p}) \)), and a half turn is \( -I \), which sends every point to its opposite. The hand check of one known point is what catches this bug.

    **8.** \( (AB)\mathbf{v} \): \( AB \) costs \( 500^3 = 125{,}000{,}000 \) multiply-adds, then times \( \mathbf{v} \) adds \( 500^2 = 250{,}000 \). \( A(B\mathbf{v}) \): two matrix-vector products of \( 250{,}000 \) each, 500,000 in all. Same answer, about 250 times less work.

    ```python
    import numpy as np
    rng = np.random.default_rng(0)
    A = rng.random((500, 500))
    B = rng.random((500, 500))
    v = rng.random(500)

    print(np.allclose((A @ B) @ v, A @ (B @ v)))
    %timeit (A @ B) @ v
    %timeit A @ (B @ v)
    # True
    # 1.56 ms per loop (best of 3, 200 loops each)      on my laptop; yours, and the browser's, will differ
    # 108 µs per loop (best of 3, 2000 loops each)
    ```

    Associativity guarantees the same answer (up to rounding, hence `allclose`), but the cost depends on the brackets: grouping so that every product involves the vector keeps each step a cheap matrix-vector product, \( n^2 \) instead of \( n^3 \). The measured ratio (about 14 on my laptop) is much smaller than 250, because BLAS runs the big matrix product far more efficiently per multiply-add (several cores, cache tiling) than the small matrix-vector products, which mostly wait on memory, and each call has a fixed overhead. In a chain of layers, frameworks apply the layers to the input one at a time for exactly this reason.

## Resources

- [3Blue1Brown: Essence of Linear Algebra, ch. 3 "Linear transformations and matrices"](https://www.youtube.com/watch?v=kYB8IZa5AuE) and [ch. 4 "Matrix multiplication as composition"](https://www.youtube.com/watch?v=XkY2DOUCWMU): why matrix multiplication is defined this way, and why order matters. Watch ch. 3 before Day 11.
- [NumPy: `numpy.matmul` reference](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html): the exact rules for `@`, including 1-D and stacked (3-D) inputs, and how it differs from `np.dot`.
- [Dive into Deep Learning, 2.3 Linear Algebra](https://d2l.ai/chapter_preliminaries/linear-algebra.html): matrices, matrix-vector and matrix-matrix products in code, with the ML angle.
- [CS231n: Python NumPy tutorial](https://cs231n.github.io/python-numpy-tutorial/): short, practical, covers `@`/`dot` vs `*`.
- [Khan Academy: Linear algebra](https://www.khanacademy.org/math/linear-algebra): practice problems on matrix products, with hints.
- [Mathematics for Machine Learning (free book), ch. 2](https://mml-book.github.io/): section 2.2 defines matrices, their products and the algebra rules formally.
- [Immersive Linear Algebra](https://immersivemath.com/ila/index.html): an interactive book; its matrix chapters let you poke at products like the stepper above.

## Checklist

- [ ] I can read the shape of a matrix and find \( A_{ij} \), I know NumPy's `A[i-1, j-1]` is maths' \( A_{ij} \), and I can name square, zero and diagonal matrices.
- [ ] I can add two matrices and scale one, and I know both are element-wise.
- [ ] I computed breakfast's `[400, 13.5]` by hand both as row dot products and as a weighted sum of columns.
- [ ] I know what `@` does with a 1-D array on each side, and why `(2, 1) + (2,)` gives `(2, 2)`.
- [ ] I computed the 2×3 times 3×2 example by hand, got [[58, 64], [139, 154]], and can write the triple loop.
- [ ] I can predict the result shape of `A @ B` (or the error) and read the `(n?,k),(k,m?)->(n?,m?)` message.
- [ ] I know `@` from `*`, `np.dot` and `np.matrix`, and I write `2 * A`, not `A @ 2`.
- [ ] I can show \( AB \neq BA \), use associativity and the distributive law, and explain why the checks need `np.allclose`.
- [ ] I can predict a product's dtype and I know integer `@` overflows silently in `int32`.
- [ ] I can run batched `@` on a stack of matrices and explain the `np.dot` difference on 3-D input.
- [ ] I can count the cost \( m \cdot n \cdot p \), explain FLOPs and FMA, and say why GPUs and NPUs are built for matmul.

**Next:** [Day 11: Transpose, identity and inverse](day-11-transpose-identity-inverse.md)
