# Day 23: Rank, Span and Linear Independence

**Phase 1 · Week 4 · ~100 minutes**

## Objectives

- [ ] Build a **linear combination** of vectors by hand and in NumPy, and see that `A @ c` is one.
- [ ] Describe the **span** of a set of vectors, and tell when two vectors fill the plane and when they only reach a line.
- [ ] Test vectors for **linear independence**, and explain what a **basis** and **coordinates** are.
- [ ] Compute a matrix's **rank** with `np.linalg.matrix_rank`, and use the **column space** to decide whether \( A\mathbf{x} = \mathbf{b} \) has a solution.
- [ ] Explain why duplicated or collinear features make \( X^\top X \) singular, and catch it in real data.
- [ ] Build low-rank matrices from outer products, preview low-rank approximation, and do the parameter arithmetic of **LoRA**.

**Prerequisites.** [Day 10](day-10-matrices-matmul.md) (a matrix times a vector is a weighted sum of its columns), [Day 11](day-11-transpose-identity-inverse.md) (singular matrices, `np.linalg.solve`, the normal equation) and [Day 22](day-22-norms-distances.md) (norms, `np.linalg.norm`). The pandas part uses [Day 17](day-17-pandas-cleaning.md)'s category cleaning.

---

## The big idea

Yesterday you measured vectors. Today you ask a different question: **given a handful of vectors, where can you get to by combining them?**

Think of a drone with thrusters. Each thruster pushes in one fixed direction, and you choose how hard (or backwards) each one fires. Two thrusters pointing in *different* directions can take the drone anywhere on a flat field. But if the second thruster points along the same line as the first, it adds nothing new: the drone is stuck on rails, no matter how you throttle. A third thruster on that flat field is always redundant.

That one picture holds today's whole vocabulary:

| Word | Drone version | Math version |
|---|---|---|
| **linear combination** | one throttle setting | \( a\mathbf{v} + b\mathbf{w} \) |
| **span** | every place the drone can reach | all linear combinations |
| **linearly dependent** | a thruster that adds no new direction | one vector is a combination of the others |
| **basis** | a minimal set of thrusters that still reaches everywhere | independent vectors that span the space |
| **rank** | how many genuinely different directions you have | number of independent columns of a matrix |

And it matters in ML for a very practical reason: the columns of your data matrix are your features. If one feature is secretly a combination of others (temperature in °C *and* °F, or a full set of one-hot columns plus an intercept), you have a redundant thruster, the rank drops, and Day 11's \( X^\top X \) becomes singular.

## Linear combinations

A **linear combination** of vectors is: scale each one by a number, then add them up. With two vectors \( \mathbf{v} \) and \( \mathbf{w} \) and two numbers \( a \) and \( b \) (called the **coefficients** or **weights**):

\[
a\mathbf{v} + b\mathbf{w}
\]

Worked example with \( \mathbf{v} = [2, 1] \) and \( \mathbf{w} = [1, 3] \):

\[
2\mathbf{v} - 1\mathbf{w} = [4, 2] - [1, 3] = [3, -1]
\qquad
0.5\mathbf{v} + 0.5\mathbf{w} = [1, 0.5] + [0.5, 1.5] = [1.5, 2]
\]

"Linear" means only these two operations are allowed: multiply by a number, add. No squaring, no multiplying vectors together.

You've already met this on Day 10: **a matrix times a vector is a linear combination of the matrix's columns**, with the vector's entries as the coefficients.

\[
\begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix} \begin{bmatrix} 2 \\ -1 \end{bmatrix}
= 2 \begin{bmatrix} 2 \\ 1 \end{bmatrix} - 1 \begin{bmatrix} 1 \\ 3 \end{bmatrix}
= \begin{bmatrix} 3 \\ -1 \end{bmatrix}
\]

```python
import numpy as np

v = np.array([2., 1.])
w = np.array([1., 3.])
2 * v - 1 * w          # array([ 3., -1.])
0.5 * v + 0.5 * w      # array([1.5, 2. ])

V = np.column_stack([v, w])   # v and w become the COLUMNS of V
V
# [[2. 1.]
#  [1. 3.]]
V @ np.array([2., -1.])       # array([ 3., -1.])   the same combination
```

Hold on to this. Every question today ("can I reach this point?", "are these vectors redundant?") turns into a question about a matrix whose columns are the vectors.

## Span: everywhere you can reach

The **span** of a set of vectors is the set of *all* their linear combinations, for every possible choice of coefficients:

\[
\operatorname{span}(\mathbf{v}, \mathbf{w}) = \{\, a\mathbf{v} + b\mathbf{w} \;:\; a, b \in \mathbb{R} \,\}
\]

Read the notation left to right: "the set of all \( a\mathbf{v} + b\mathbf{w} \), such that (the colon) \( a \) and \( b \) are any real numbers (\( \mathbb{R} \))". Curly braces mean "the set of".

<figure class="diagram" markdown>
![Animation in two panels. Left: vectors v = [2, 1] and w = [1, 3] generate a slanted grid that covers the whole plane; an orange point jumps between combinations 1v + 1w, 2v - 1w, -1v + 1.5w and 0.5v - 1w, each drawn as a dashed red a-times-v segment followed by a dashed blue b-times-w segment. Right: p = [1, 2] and q = [2, 4] lie on the same line, and every combination, including 2p - 1q = [0, 0], lands on that line, while the point [3, 1] is marked unreachable](../../assets/img/phase-1/day23-span-sweep.svg)
<figcaption>Animated: two independent vectors reach every point of the plane; two parallel vectors only ever reach one line.</figcaption>
</figure>

Three cases in 2-D:

- **Two vectors pointing in different directions** span the whole plane. Every point is \( a\mathbf{v} + b\mathbf{w} \) for exactly one pair \( (a, b) \).
- **Two parallel vectors** (one is a multiple of the other) span only a **line** through the origin.
- **The zero vector alone** spans just the origin.

"Can I reach the point \( \mathbf{t} \)?" is the question "do coefficients exist with \( V\mathbf{c} = \mathbf{t} \)?", which is a linear system. Day 11's `solve` answers it:

```python
target = np.array([4., 7.])
c = np.linalg.solve(V, target)
c                          # array([1., 2.])    so target = 1·v + 2·w
c[0] * v + c[1] * w        # array([4., 7.])

p = np.array([1., 2.])
q = np.array([2., 4.])     # q = 2p: parallel
rng = np.random.default_rng(0)
coefs = rng.uniform(-3, 3, size=(1000, 2))   # 1000 random (a, b) pairs
pts = coefs @ np.vstack([p, q])              # each row: a·p + b·q
pts.shape                                     # (1000, 2)
np.allclose(pts[:, 1], 2 * pts[:, 0])        # True: every point is on the line y = 2x

P = np.column_stack([p, q])
try:
    np.linalg.solve(P, target)
except np.linalg.LinAlgError as e:
    print("LinAlgError:", e)                 # LinAlgError: Singular matrix
```

One thousand random throttle settings, and every single point lands on the line \( y = 2x \). The matrix with columns \( \mathbf{p}, \mathbf{q} \) is exactly Day 11's singular matrix: it squashes the plane onto that line, so `solve` refuses.

The same idea in 3-D: two independent vectors span a **plane** through the origin (a flat sheet inside 3-D space), and you need a third vector that sticks out of that sheet to reach everything.

## Linear independence

A set of vectors is **linearly dependent** if at least one of them is a linear combination of the others: it adds no new direction. Otherwise the set is **linearly independent**.

There's an equivalent test that's easier to check, and that treats all the vectors equally:

\[
c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \dots + c_k\mathbf{v}_k = \mathbf{0}
\quad\text{only when}\quad c_1 = c_2 = \dots = c_k = 0
\]

In words: the only recipe that brings you back to the origin is "fire nothing". Here \( k \) is how many vectors you have and \( \mathbf{0} \) is the zero vector. If some recipe with a non-zero coefficient lands on zero, the vectors are dependent. In the animation, \( 2\mathbf{p} - 1\mathbf{q} = [0, 0] \) is that tell-tale recipe.

Worked example in 3-D: \( \mathbf{u} = [1, 0, 1] \), \( \mathbf{v} = [0, 1, 1] \), \( \mathbf{w} = [1, 1, 2] \). Spot it? \( \mathbf{w} = \mathbf{u} + \mathbf{v} \), so \( 1\mathbf{u} + 1\mathbf{v} - 1\mathbf{w} = \mathbf{0} \) with non-zero coefficients: dependent. All three lie in one plane.

```python
u3 = np.array([1., 0., 1.])
v3 = np.array([0., 1., 1.])
w3 = np.array([1., 1., 2.])
u3 + v3 - w3                       # array([0., 0., 0.])   a non-trivial recipe for zero

np.linalg.matrix_rank(np.column_stack([u3, v3, w3]))                  # 2: only 2 independent
np.linalg.matrix_rank(np.column_stack([u3, v3, [0., 0., 1.]]))        # 3: independent

three_in_2d = np.column_stack([[1., 0.], [0., 1.], [3., -2.]])
np.linalg.matrix_rank(three_in_2d)    # 2: three vectors in 2-D are ALWAYS dependent
```

`matrix_rank` (properly introduced below) counts independent columns. (In Jupyter its result displays as `np.int64(2)`; the comments show the plain value, as on Day 11.) A fact worth memorizing: **in \( n \)-dimensional space you can never have more than \( n \) independent vectors.** Three thrusters on a flat field: one is always redundant (here \( [3, -2] = 3[1, 0] - 2[0, 1] \)).

!!! warning "Common mistake: \"no two are parallel\" is not enough"
    With two vectors, dependent means parallel. With three or more, it doesn't: \( \mathbf{u}, \mathbf{v}, \mathbf{w} \) above are pairwise non-parallel, yet dependent, because \( \mathbf{w} \) is a combination of *both* others. Checking pairs (say, with cosine similarity from Day 5) can't detect this; the rank can.

## Basis and coordinates

A **basis** of a space is a set of vectors that is **independent** (no redundancy) and **spans** the space (reaches everything). The number of vectors in any basis is always the same; it's the **dimension** of the space. The plane has dimension 2, so every basis of it has exactly 2 vectors.

You've been using one basis all along without naming it: the **standard basis** \( \mathbf{e}_1 = [1, 0] \), \( \mathbf{e}_2 = [0, 1] \) (Day 11 called them \( \hat{\imath} \) and \( \hat{\jmath} \)). When you write \( \mathbf{x} = [3, 1] \), you mean \( 3\mathbf{e}_1 + 1\mathbf{e}_2 \).

But any two independent vectors work. Take \( \mathbf{b}_1 = [1, 1] \), \( \mathbf{b}_2 = [1, -1] \). The **coordinates** of \( \mathbf{x} \) in that basis are the coefficients \( c_1, c_2 \) with \( c_1\mathbf{b}_1 + c_2\mathbf{b}_2 = \mathbf{x} \). Because the basis vectors are independent, there's exactly one answer:

\[
c_1 \begin{bmatrix} 1 \\ 1 \end{bmatrix} + c_2 \begin{bmatrix} 1 \\ -1 \end{bmatrix} = \begin{bmatrix} 3 \\ 1 \end{bmatrix}
\;\Rightarrow\;
\begin{aligned} c_1 + c_2 &= 3 \\ c_1 - c_2 &= 1 \end{aligned}
\;\Rightarrow\;
c_1 = 2,\; c_2 = 1
\]

(Add the two equations: \( 2c_1 = 4 \).)

<figure class="diagram" markdown>
![Two panels showing the same point x = [3, 1]. Left: on the standard square grid, x is reached by 3 steps along e1 then 1 step along e2, coordinates [3, 1]. Right: on a grid tilted 45 degrees made from b1 = [1, 1] and b2 = [1, -1], the same point is reached by 2 steps along b1 then 1 step along b2, coordinates [2, 1]](../../assets/img/phase-1/day23-basis-coordinates.svg)
<figcaption>The arrow doesn't change; only the grid you measure it on does. Coordinates are always relative to a basis.</figcaption>
</figure>

```python
B = np.column_stack([[1., 1.], [1., -1.]])   # basis vectors as columns
x = np.array([3., 1.])
coords = np.linalg.solve(B, x)
coords          # array([2., 1.])   x = 2·b1 + 1·b2
B @ coords      # array([3., 1.])   and back again
```

Why should an ML engineer care? Because **changing basis is what a lot of ML does**. PCA (Day 25) finds a basis aligned with the directions your data actually varies along, then describes each data point by its coordinates in that basis, keeping only the first few. An embedding model is often described as learning a useful set of directions in which to describe text or images.

## Rank and column space

The **column space** of a matrix \( A \) is the span of its columns. By the Day 10 rule, \( A\mathbf{x} \) is always a combination of the columns, so the column space is exactly **the set of all possible outputs** \( A\mathbf{x} \).

The **rank** of \( A \) is the dimension of its column space: the number of independent columns.

Two facts that are surprising the first time:

- **Row rank equals column rank.** The number of independent rows always equals the number of independent columns. So `rank(A) == rank(A.T)`.
- So for an \( m \times n \) matrix, \( \operatorname{rank}(A) \le \min(m, n) \). If it equals \( \min(m, n) \), \( A \) is **full rank**; otherwise it is **rank-deficient**.

Worked example:

\[
A = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 1 & 0 & 1 \end{bmatrix}
\]

Look at the columns: column 3 \( = [3, 6, 1] \) is column 1 \( + \) column 2 \( = [1, 2, 1] + [2, 4, 0] \). Columns 1 and 2 are not parallel. So there are 2 independent columns: rank 2. Rows agree: row 2 is twice row 1, and row 3 is something new. Rank 2 again.

**Is \( A\mathbf{x} = \mathbf{b} \) solvable?** Only if \( \mathbf{b} \) is in the column space. A clean test: glue \( \mathbf{b} \) on as an extra column. If the rank stays the same, \( \mathbf{b} \) added no new direction, so it was already reachable. If the rank goes up, it wasn't.

```python
A = np.array([[1., 2., 3.],
              [2., 4., 6.],
              [1., 0., 1.]])
np.linalg.matrix_rank(A)      # 2
np.linalg.matrix_rank(A.T)    # 2   row rank = column rank
np.linalg.det(A)              # 0.0  rank < 3 means singular (Day 11)

b_in = A @ np.array([1., 1., 1.])     # array([ 6., 12.,  2.])   reachable by construction
b_out = np.array([1., 0., 0.])
np.linalg.matrix_rank(np.column_stack([A, b_in]))    # 2  same rank: solvable
np.linalg.matrix_rank(np.column_stack([A, b_out]))   # 3  rank went up: no solution

np.linalg.matrix_rank(np.eye(4)), np.linalg.matrix_rank(np.zeros((3, 3)))   # (4, 0)
rng = np.random.default_rng(1)
np.linalg.matrix_rank(rng.normal(size=(5, 3)))   # 3: random data is full rank
```

This ties together three things from Day 11. For a **square** \( n \times n \) matrix, all of these say the same thing:

| Rank \( = n \) (full rank) | Rank \( < n \) (rank-deficient) |
|---|---|
| columns are independent | some column is a combination of others |
| column space is all of \( n \)-D space | column space is a flat sheet, a line... |
| \( \det(A) \neq 0 \) | \( \det(A) = 0 \) |
| invertible; \( A\mathbf{x} = \mathbf{b} \) has exactly one solution | singular; no solution or infinitely many |

The vectors that \( A \) squashes to zero, \( A\mathbf{n} = \mathbf{0} \), form the **null space**. Every lost dimension of rank shows up as a dimension of null space: \( \operatorname{rank}(A) + \dim(\text{null space}) = n \) (the number of columns). For our \( A \), \( [1, 1, -1] \) is in the null space (col 1 + col 2 − col 3 = 0), which is the "recipe for zero" from the independence test again.

### How `matrix_rank` decides, and floating point

On a computer, "exactly dependent" almost never survives rounding. `matrix_rank` uses the **singular values** of the matrix (from the SVD, which Day 25 explains; for now, think of them as "how much the matrix stretches along each of its independent directions", sorted largest first). The rank is the number of singular values above a small tolerance.

```python
np.linalg.svd(A, compute_uv=False)
# array([8.43544852e+00, 9.18263762e-01, 2.72823990e-16])   third one is rounding noise

rng = np.random.default_rng(1)
Ap = A + 1e-10 * rng.normal(size=A.shape)    # nudge every entry by about 1e-10
np.linalg.svd(Ap, compute_uv=False)
# array([8.43544852e+00, 9.18263762e-01, 6.50071423e-11])
np.linalg.matrix_rank(Ap)              # 3: technically full rank now
np.linalg.matrix_rank(Ap, tol=1e-6)    # 2: "rank 2, up to noise"
```

!!! warning "Common mistake: trusting the rank of noisy data"
    Real measurements are never *exactly* collinear, so `matrix_rank` will often say "full rank" while one singular value is tiny. That matrix is still effectively singular for solving (Day 11's huge condition number). When it matters, look at the singular values themselves, not just the count.

## Why collinear features break \( X^\top X \)

Day 11's normal equation fits linear-regression weights by solving

\[
(X^\top X)\,\mathbf{w} = X^\top \mathbf{y}
\]

where \( X \) has one row per example and one column per feature. \( X^\top X \) is a small square matrix, \( k \times k \) for \( k \) features. When is it singular?

**Exactly when the columns of \( X \) are dependent.** The argument is three lines. If some recipe \( \mathbf{n} \neq \mathbf{0} \) has \( X\mathbf{n} = \mathbf{0} \), then

\[
(X^\top X)\,\mathbf{n} = X^\top (X\mathbf{n}) = X^\top \mathbf{0} = \mathbf{0}
\]

so \( X^\top X \) also squashes a non-zero vector to zero: singular. (The converse holds too: \( \operatorname{rank}(X^\top X) = \operatorname{rank}(X) \) always.)

And here's what it means for the model. If \( X\mathbf{n} = \mathbf{0} \), then for any weights \( \mathbf{w} \), the weights \( \mathbf{w} + t\,\mathbf{n} \) give **identical predictions** for any \( t \). The data simply cannot tell those weights apart, so "the" best weights don't exist.

**Example: temperature twice.** You predict ice-cream sales from an intercept column of ones, the temperature in °C, and (by accident) the temperature in °F. Since \( F = 1.8\,C + 32 \), the °F column is \( 32 \times \text{ones} + 1.8 \times \text{°C} \): a combination of the other two. The null-space recipe is \( \mathbf{n} = [32, 1.8, -1] \).

```python
rng = np.random.default_rng(7)
temp_c = rng.uniform(10, 35, size=30)
temp_f = temp_c * 1.8 + 32
sales = 3 * temp_c + 10 + rng.normal(0, 2, size=30)     # truth: 3 per degree C, plus 10

X = np.column_stack([np.ones(30), temp_c, temp_f])
X.shape, np.linalg.matrix_rank(X)        # ((30, 3), 2)   3 columns, only 2 directions
G = X.T @ X
np.linalg.matrix_rank(G)                 # 2   the 3x3 matrix X^T X is singular
print(f"{np.linalg.cond(G):.1e}")        # 4.6e+18   "effectively singular"

try:
    w_bad = np.linalg.solve(G, X.T @ sales)
    print("solve returned:", w_bad.round(2))   # solve returned: [-9.88  1.97  0.59]
except np.linalg.LinAlgError as e:
    print("LinAlgError:", e)

n = np.array([32., 1.8, -1.])
np.abs(X @ n).max()           # 0.0   n really is a recipe for zero
w1 = np.array([10., 3., 0.])  # "10 + 3·C", ignoring F
w2 = w1 + 5 * n               # array([170.,  12.,  -5.])   wildly different weights...
np.allclose(X @ w1, X @ w2)   # True                        ...identical predictions
```

Look at what `solve` did on my machine: **no error**. Rounding kept the matrix a hair away from exactly singular, so it returned confident-looking nonsense (an intercept of −9.88 when the truth is 10). Depending on your platform's linear-algebra library you may get the `LinAlgError` instead; that's why the block catches both. Either way, the condition number of \( 10^{18} \) is the alarm bell. The fix is to remove the redundant direction:

```python
X2 = X[:, :2]                                  # drop the F column
np.linalg.matrix_rank(X2)                      # 2   full rank: 2 columns, 2 directions
np.linalg.solve(X2.T @ X2, X2.T @ sales).round(3)   # array([9.085, 3.039])   close to 10 and 3
```

### The same trap in real data: one-hot columns plus an intercept

A sneakier version shows up every time you encode a category. One-hot encoding (`pd.get_dummies`) makes one 0/1 column per category, and every row has exactly one 1. So the one-hot columns **always add up to a column of ones**, which is exactly the intercept column. Statisticians call this the **dummy-variable trap**. Here it is on the course dataset (tested with pandas 3.0; if you downloaded the CSV from the [Datasets](../../datasets.md) page, pass the local path instead):

```python
import pandas as pd

BASE = "https://manjulkadyan.github.io/ai-leaning/assets/data/"
apps = pd.read_csv(BASE + "play_store_apps.csv")
apps = apps.drop_duplicates()
apps["category"] = apps["category"].str.strip().str.title()   # Day 17's cleaning
rated = apps.dropna(subset=["rating"])
len(rated)                                     # 462

onehot = pd.get_dummies(rated["category"], dtype=float)
print(onehot.iloc[:3, :4].to_string())
#    Education  Finance  Games  Health & Fitness
# 0        0.0      0.0    0.0               0.0
# 1        0.0      0.0    0.0               0.0
# 2        0.0      0.0    0.0               0.0
onehot.sum(axis=1).unique()                    # array([1.])   every row sums to exactly 1

X = np.column_stack([np.ones(len(onehot)), onehot.to_numpy()])
y = rated["rating"].to_numpy()
X.shape, np.linalg.matrix_rank(X)              # ((462, 9), 8)   9 columns, rank 8
try:
    w_trap = np.linalg.solve(X.T @ X, X.T @ y)
    print("solve returned:", w_trap.round(2))
    # solve returned: [-13.84  17.89  17.95  18.13  17.92  18.06  18.02  17.98  17.99]
except np.linalg.LinAlgError as e:
    print("LinAlgError:", e)
```

An intercept of −13.84 stars and category effects of +18 for a rating that lives between 1 and 5: nonsense again, produced without complaint. The standard fix is `drop_first=True`: drop one category's column, and that category becomes the baseline that the intercept describes.

```python
onehot_ok = pd.get_dummies(rated["category"], dtype=float, drop_first=True)
list(onehot_ok.columns)      # Education is gone: it's the baseline now
# ['Finance', 'Games', 'Health & Fitness', 'Music & Audio', 'Photography', 'Productivity', 'Tools']
X_ok = np.column_stack([np.ones(len(onehot_ok)), onehot_ok.to_numpy()])
np.linalg.matrix_rank(X_ok)  # 8   8 columns, full rank
w_ok = np.linalg.solve(X_ok.T @ X_ok, X_ok.T @ y)
w_ok.round(3)
# array([4.043, 0.064, 0.24 , 0.034, 0.177, 0.137, 0.091, 0.108])

means = rated.groupby("category")["rating"].mean()
(means - means["Education"]).round(3)["Games"]   # 0.24
```

Now the weights mean something. The intercept 4.043 is the mean Education rating, and each other weight is "how much higher that category's mean is than Education's": Games is +0.24, matching a Day 18 `groupby` exactly.

!!! warning "Common mistake: thinking more features can't hurt"
    Adding a column that is a combination of others (a total next to its parts, a unit conversion, the log of a ratio next to the logs of its numerator and denominator, all one-hot columns plus an intercept) adds zero information and makes the weights unidentifiable. Libraries hide it: `np.linalg.lstsq` and scikit-learn's `LinearRegression` return *a* solution instead of crashing, so the model still predicts fine, but you can no longer read meaning into individual weights. Check `np.linalg.matrix_rank(X)` against `X.shape[1]` before interpreting coefficients. (Ridge regularization, which adds \( \lambda I \) to \( X^\top X \), is another cure; it's Month 2 material.)

## Low-rank matrices are outer products

Now flip the story. Rank deficiency is a bug in a feature matrix, but it's a **feature** when you want to store or learn a big matrix cheaply.

The simplest non-zero matrix has rank 1. It's an **outer product**: a column vector times a row vector. For \( \mathbf{u} \) of length \( m \) and \( \mathbf{v} \) of length \( n \):

\[
\mathbf{u}\mathbf{v}^\top =
\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}
\begin{bmatrix} 4 & 5 & 6 & 7 \end{bmatrix}
=
\begin{bmatrix} 4 & 5 & 6 & 7 \\ 8 & 10 & 12 & 14 \\ 12 & 15 & 18 & 21 \end{bmatrix}
\qquad
(\mathbf{u}\mathbf{v}^\top)_{ij} = u_i\, v_j
\]

Shape check from Day 10: \( (3, 1) @ (1, 4) = (3, 4) \). Every column is a multiple of \( \mathbf{u} \) (column \( j \) is \( v_j \mathbf{u} \)) and every row is a multiple of \( \mathbf{v} \), so there is one direction only: rank 1. Twelve numbers, but only \( 3 + 4 = 7 \) of them are really free.

Add up \( r \) outer products and you get a matrix of rank at most \( r \). Stacking the \( \mathbf{u} \)'s as columns of \( B \) (shape \( m \times r \)) and the \( \mathbf{v} \)'s as rows of \( A \) (shape \( r \times n \)), that sum is just one matrix product:

\[
BA = \mathbf{b}_1\mathbf{a}_1^\top + \mathbf{b}_2\mathbf{a}_2^\top + \dots + \mathbf{b}_r\mathbf{a}_r^\top
\qquad
\operatorname{rank}(BA) \le r
\]

Storage: the full \( m \times n \) matrix needs \( mn \) numbers; the factors need \( r(m + n) \). For \( m = 1000 \), \( n = 800 \), \( r = 10 \): 800,000 vs 18,000.

```python
u = np.array([1., 2., 3.])
v = np.array([4., 5., 6., 7.])
M = np.outer(u, v)
M
# [[ 4.  5.  6.  7.]
#  [ 8. 10. 12. 14.]
#  [12. 15. 18. 21.]]
np.linalg.matrix_rank(M)                 # 1
M[:, 2] / u                              # array([6., 6., 6.])   column 2 is 6·u
np.allclose(M, u[:, None] @ v[None, :])  # True   outer product = (3,1) @ (1,4)

rng = np.random.default_rng(3)
B = rng.normal(size=(6, 2))
A = rng.normal(size=(2, 5))
M2 = B @ A
M2.shape, np.linalg.matrix_rank(M2)      # ((6, 5), 2)
np.allclose(M2, np.outer(B[:, 0], A[0]) + np.outer(B[:, 1], A[1]))   # True: a sum of 2 outer products

m, n, r = 1000, 800, 10
m * n, r * (m + n)                       # (800000, 18000)
```

### Preview: low-rank approximation

Real matrices are rarely *exactly* low-rank, but many are *nearly* low-rank: a few strong patterns plus noise. A users × apps ratings table is a classic example: people's tastes are driven by a handful of factors (likes games, prefers free apps...). The SVD splits any matrix into a sum of rank-1 pieces sorted by importance (singular values). Keeping only the first \( k \) pieces gives the **best possible rank-\( k \) approximation** (a theorem called Eckart–Young). Day 25 shows where the pieces come from; for now, watch what they buy you:

```python
rng = np.random.default_rng(42)
users, items, true_r = 60, 40, 3
taste = rng.normal(size=(users, true_r))     # each user: 3 hidden taste numbers
traits = rng.normal(size=(true_r, items))    # each app: 3 hidden trait numbers
clean = taste @ traits                       # (60, 40) matrix of rank 3
noisy = clean + 0.1 * rng.normal(size=(users, items))
np.linalg.matrix_rank(clean), np.linalg.matrix_rank(noisy)   # (3, 40)  noise makes it "full rank"

U, s, Vt = np.linalg.svd(noisy, full_matrices=False)
s[:6].round(2)          # array([58.6 , 38.79, 30.58,  1.33,  1.27,  1.21])  3 big, then noise
for k in (1, 2, 3, 5, 10):
    approx = U[:, :k] @ np.diag(s[:k]) @ Vt[:k]      # keep the first k rank-1 pieces
    err = np.linalg.norm(noisy - approx) / np.linalg.norm(noisy)
    store = k * (users + items + 1)
    print(f"rank {k:2d}: relative error {err:.3f}, numbers stored {store:5d} of {users*items}")
# rank  1: relative error 0.646, numbers stored   101 of 2400
# rank  2: relative error 0.403, numbers stored   202 of 2400
# rank  3: relative error 0.061, numbers stored   303 of 2400
# rank  5: relative error 0.056, numbers stored   505 of 2400
# rank 10: relative error 0.046, numbers stored  1010 of 2400

approx3 = U[:, :3] @ np.diag(s[:3]) @ Vt[:3]
print(f"{np.linalg.norm(approx3 - clean) / np.linalg.norm(clean):.4f}")   # 0.0223
print(f"{np.linalg.norm(noisy - clean) / np.linalg.norm(clean):.4f}")     # 0.0648
```

The singular values fall off a cliff after 3, revealing the true rank hiding under the noise. Rank 3 reproduces the matrix to within 6.1% using 303 numbers instead of 2,400 (the error is a Day 22 norm of the difference, as a fraction of the norm of the whole). And the last two lines show a bonus: the rank-3 approximation is **closer to the clean truth than the noisy data itself** (2.2% vs 6.5% error). Throwing away the small pieces threw away mostly noise. This is the idea behind classic recommender systems, image compression and PCA.

## LoRA: fine-tuning with a low-rank update

Here is low rank at work in today's LLMs. Fine-tuning means taking a trained model and adjusting its weights for a new task. A large model has many weight matrices like \( W \) of shape \( 4096 \times 4096 \): 16.8 million numbers *each*. Updating all of them needs memory for every gradient and optimizer state, and produces a full new copy of the model per task.

**LoRA** (Low-Rank Adaptation, Hu et al., 2021) bets that the *change* a fine-tune needs, \( \Delta W \) ("delta W"), is nearly low-rank, like the ratings matrix above. So it never learns \( \Delta W \) directly. It freezes \( W \) and learns two thin factors:

\[
W' = W + \Delta W = W + BA
\qquad
B \in \mathbb{R}^{d \times r},\;\; A \in \mathbb{R}^{r \times k},\;\; r \ll \min(d, k)
\]

New symbols: \( W \) is \( d \times k \) (\( d \) outputs, \( k \) inputs, as on Day 12), \( r \) is the chosen **rank**, often 4 to 64, and \( \ll \) means "much smaller than". \( \mathbb{R}^{d \times r} \) just means "a \( d \times r \) matrix of real numbers". By the outer-product rule, \( BA \) has rank at most \( r \).

<figure class="diagram" markdown>
![Animation: a large square frozen matrix W of size 4096 by 4096, plus a tall thin matrix B of size 4096 by r times a short wide matrix A of size r by 4096, equals a full-size update Delta W of rank at most r. The rank r steps through 1, 4, 8 and 16, B and A thicken, and the panel below updates the trainable count: 8,192, 32,768, 65,536 and 131,072 numbers, which is 0.05%, 0.20%, 0.39% and 0.78% of the 16,777,216 entries of W](../../assets/img/phase-1/day23-lora.svg)
<figcaption>Animated: the update has W's full shape but only r·(d + k) trainable numbers; at r = 8 that's 0.39% of W.</figcaption>
</figure>

**The parameter arithmetic.** With \( d = k = 4096 \) and \( r = 8 \):

\[
\underbrace{d \cdot k}_{\text{full fine-tune}} = 16{,}777{,}216
\qquad
\underbrace{d \cdot r + r \cdot k}_{\text{LoRA}} = 32{,}768 + 32{,}768 = 65{,}536
\qquad
\frac{65{,}536}{16{,}777{,}216} \approx 0.39\%
\]

Three details from the paper that make it work in practice:

1. **\( B \) starts at zero** (and \( A \) random), so \( BA = 0 \) and training begins exactly at the original model.
2. **The forward pass never builds \( BA \).** Compute \( W\mathbf{x} + B(A\mathbf{x}) \): \( A\mathbf{x} \) is only \( r \) numbers, so the extra cost is tiny. (The paper also scales the update by a constant \( \alpha / r \); we leave it out here.)
3. **After training, merge:** \( W' = W + BA \) is a normal matrix of the original shape, so inference is exactly as fast as before. Or keep \( A, B \) separate and swap small adapters per task on one shared base model.

```python
d, k, r = 4096, 4096, 8
full = d * k
lora = d * r + r * k
full, lora, f"{lora / full:.2%}"     # (16777216, 65536, '0.39%')

# a small LoRA layer in NumPy
rng = np.random.default_rng(0)
d, k, r = 512, 512, 4
W = rng.normal(0, 0.02, size=(d, k))      # pretend this is pre-trained and frozen
A = rng.normal(0, 0.02, size=(r, k))      # random start
B = np.zeros((d, r))                      # zero start
x = rng.normal(size=k)
np.allclose(W @ x + B @ (A @ x), W @ x)   # True: at step 0, LoRA changes nothing

B = rng.normal(0, 0.02, size=(d, r))      # pretend training has updated B
delta = B @ A
delta.shape, np.linalg.matrix_rank(delta) # ((512, 512), 4)   full shape, rank 4
y_lora = W @ x + B @ (A @ x)              # cheap path: never builds the 512x512 delta
W_merged = W + B @ A                      # merge once, after training
np.allclose(y_lora, W_merged @ x)         # True: identical outputs
W.size, A.size + B.size                   # (262144, 4096)   trainable: 1/64 of W
```

The paper reports that, on GPT-3 175B, LoRA cut the number of trainable parameters by about 10,000× and GPU memory during training by about 3×, while matching full fine-tuning quality on its benchmarks.

!!! android "Android bridge"
    You already ship software this way. A Play Feature Delivery app has one **base module** everybody installs and small **dynamic feature modules** downloaded on demand. LoRA gives models the same shape: one large frozen base model on the device, plus a small adapter per feature ("summarize", "reply suggestions", "your app's tone of voice"). With \( r = 8 \) on one \( 4096 \times 4096 \) matrix in float16, an adapter is \( 65{,}536 \times 2 \) bytes \( = \) 128 KiB instead of 32 MiB for a full copy of that matrix, multiplied over every matrix you adapt.

    On-device LLM runtimes have started supporting this directly: Google's MediaPipe LLM Inference API, for example, documents loading LoRA weights alongside a supported base model (check the current [Google AI Edge](https://ai.google.dev/edge) docs for which models and backends support it). And a merged adapter (\( W' = W + BA \)) is just a regular model file: your LiteRT interpreter code wouldn't know the difference.

---

## Exercises

1. Is \( [7, 4] \) in the span of \( [1, 2] \) and \( [3, 1] \)? Write the two equations for the coefficients, solve them by hand, then check with `np.linalg.solve`.
2. Without code first, decide which sets are linearly independent, then confirm with `np.linalg.matrix_rank`: (a) \( [1, 2, 3], [2, 4, 6] \); (b) \( [1, 0, 0], [1, 1, 0], [1, 1, 1] \); (c) \( [1, 2], [3, 4], [5, 6] \).
3. Find the coordinates of \( \mathbf{x} = [5, 1] \) in the basis \( \mathbf{b}_1 = [1, 1] \), \( \mathbf{b}_2 = [1, -1] \), by hand and with `solve`. Verify by rebuilding \( \mathbf{x} \).
4. Let \( A = \begin{bmatrix} 1 & 2 \\ 2 & 4 \\ 3 & 6 \end{bmatrix} \). What is its rank, and what does its column space look like geometrically? Use the rank test to decide whether \( A\mathbf{x} = [1, 2, 3] \) and \( A\mathbf{x} = [1, 0, 0] \) have solutions.
5. With `rng = np.random.default_rng(5)`, make 40 people with `height_cm = rng.uniform(150, 195, 40)`, `height_in = height_cm / 2.54` and `weight = rng.uniform(50, 100, 40)`, and build `X` with columns `[ones, height_cm, height_in, weight]`. Find its rank, write down a null-space vector \( \mathbf{n} \) and check `X @ n` is (nearly) zero, then fix `X` so it's full rank.
6. A GPT-2-small-sized model has 12 layers, each with \( 768 \times 768 \) query and value matrices. You apply LoRA with \( r = 4 \) to both of them in every layer. How many trainable parameters is that, what percentage of those 24 matrices' parameters, and how many bytes is the adapter in float16?

??? success "Solutions"
    ```python
    import numpy as np

    # 1: a·[1, 2] + b·[3, 1] = [7, 4]  ->  a + 3b = 7 and 2a + b = 4
    #    second: b = 4 - 2a; first: a + 12 - 6a = 7, so a = 1, b = 2
    V1 = np.column_stack([[1., 2.], [3., 1.]])
    np.linalg.solve(V1, np.array([7., 4.]))     # array([1., 2.])   yes: 1·[1,2] + 2·[3,1]

    # 2
    np.linalg.matrix_rank(np.column_stack([[1, 2, 3], [2, 4, 6]]))              # 1: dependent (2nd = 2 × 1st)
    np.linalg.matrix_rank(np.column_stack([[1, 0, 0], [1, 1, 0], [1, 1, 1]]))   # 3: independent
    np.linalg.matrix_rank(np.column_stack([[1, 2], [3, 4], [5, 6]]))            # 2: dependent (3 vectors in 2-D)

    # 3: c1 + c2 = 5 and c1 - c2 = 1  ->  c1 = 3, c2 = 2
    B3 = np.column_stack([[1., 1.], [1., -1.]])
    c3 = np.linalg.solve(B3, np.array([5., 1.]))
    c3          # array([3., 2.])
    B3 @ c3     # array([5., 1.])

    # 4: column 2 = 2 × column 1, so rank 1: the column space is the LINE through [1, 2, 3]
    A4 = np.array([[1., 2.], [2., 4.], [3., 6.]])
    np.linalg.matrix_rank(A4)                                      # 1
    np.linalg.matrix_rank(np.column_stack([A4, [1., 2., 3.]]))     # 1  same: solvable (e.g. x = [1, 0])
    np.linalg.matrix_rank(np.column_stack([A4, [1., 0., 0.]]))     # 2  went up: no solution

    # 5
    rng = np.random.default_rng(5)
    height_cm = rng.uniform(150, 195, 40)
    height_in = height_cm / 2.54
    weight = rng.uniform(50, 100, 40)
    X5 = np.column_stack([np.ones(40), height_cm, height_in, weight])
    np.linalg.matrix_rank(X5)            # 3   4 columns, rank 3
    n5 = np.array([0., 1., -2.54, 0.])   # 1·height_cm - 2.54·height_in = 0
    np.abs(X5 @ n5).max() < 1e-10        # True  (the max is about 3e-14: rounding)
    X5_ok = np.delete(X5, 2, axis=1)     # drop height_in
    np.linalg.matrix_rank(X5_ok), X5_ok.shape[1]    # (3, 3)   full rank

    # 6
    layers, mats, d, r = 12, 2, 768, 4
    per_matrix = d * r + r * d
    trainable = layers * mats * per_matrix
    full = layers * mats * d * d
    per_matrix, trainable, full, f"{trainable / full:.2%}"   # (6144, 147456, 14155776, '1.04%')
    trainable * 2, trainable * 2 / 1024    # (294912, 288.0)   bytes in float16: 288 KiB
    ```
    In 2(c) the rank is 2, which is the most any set of 2-D vectors can have; with three vectors, one must be redundant. In 4, `[1, 2, 3]` is column 1 itself, so `x = [1, 0]` works (and so does `[-1, 1]`, and infinitely many others, because the null space contains `[2, -1]`).

## Resources

- [3Blue1Brown: Linear combinations, span, and basis vectors (Essence of Linear Algebra, ch. 2)](https://www.youtube.com/watch?v=k7RM-ot2NWY): today's first half, animated. Watch it first.
- [3Blue1Brown: Inverse matrices, column space and null space (ch. 7)](https://www.youtube.com/watch?v=uQhTuRlWMxw): rank, column space and null space as pictures.
- [Khan Academy: Vectors and spaces](https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces): linear combinations, span, independence and basis, with practice problems.
- [MIT OpenCourseWare 18.06 Linear Algebra (Gilbert Strang)](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/): the classic course; lectures 5–10 cover column space, null space, independence, basis and rank.
- [NumPy docs: `numpy.linalg.matrix_rank`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html): how the tolerance is chosen.
- [Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)](https://arxiv.org/abs/2106.09685): the original paper; sections 1 and 4 are readable now.
- [Hugging Face PEFT docs](https://huggingface.co/docs/peft): the library most people use to apply LoRA in practice.

## Checklist

- [ ] I can compute a linear combination by hand and explain why `A @ c` is one.
- [ ] I can say what the span of two vectors is when they're independent and when they're parallel.
- [ ] I can test independence with the "only the zero recipe gives zero" rule and with `matrix_rank`.
- [ ] I found the coordinates of a vector in a non-standard basis with `solve`.
- [ ] I can explain rank, column space, and the rank test for whether \( A\mathbf{x} = \mathbf{b} \) is solvable.
- [ ] I can explain why a redundant feature makes \( X^\top X \) singular, and I know `drop_first=True` avoids the dummy-variable trap.
- [ ] I can build a rank-\( r \) matrix as \( BA \), and compute LoRA's parameter count for given \( d, k, r \).

**Next:** Day 24: Eigenvectors, intuitively (coming soon)
