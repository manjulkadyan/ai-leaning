# Day 25: Projections and a First Look at PCA

**Phase 1 · Week 4 · ~120 minutes**

## Objectives

- [ ] Project one vector onto another by hand and in NumPy, and explain the result as a "shadow".
- [ ] Show that what is left over after a projection is perpendicular to the line you projected onto.
- [ ] Explain least squares (fitting a line) as projecting the data onto a subspace, and fit one with `np.linalg.lstsq`.
- [ ] Run PCA from scratch: center, covariance, `eigh`, sort, project, explained variance ratio, reconstruct.
- [ ] Apply PCA to the Play Store dataset and read what the components mean.
- [ ] Explain why libraries compute PCA with the SVD, and why PCA is useful for shrinking embeddings on a phone.

**Prerequisites.** [Day 5](day-05-vectors.md) (dot product, length), [Day 11](day-11-transpose-identity-inverse.md) (transpose, `np.linalg.solve`, the normal equation), [Day 22](day-22-norms-distances.md) (norms, standardizing features), [Day 23](day-23-rank-span.md) (span, basis, rank) and [Day 24](day-24-eigenvectors.md) (eigenvectors, `eigh`, symmetric matrices).

---

## The big idea

Hold a pencil under a desk lamp and look at its shadow on the desk. Tilt the pencil upright and the shadow shrinks to a dot; lay it flat and the shadow is as long as the pencil. A **projection** is that shadow: the part of one vector that points along some line (or plane), with everything else thrown away.

Two things make shadows the workhorse of ML:

| Idea | The shadow question |
|---|---|
| **Least squares** (fitting a line) | Your data can't be fitted exactly. What is the *closest* thing you *can* fit? Its shadow on the set of all possible fits. |
| **PCA** (principal component analysis) | Your data has 4, 256 or 1,000 columns. Which few directions keep most of the information? The ones where the data's shadows spread out the most. |

**Analogy.** You have to photograph a 3-D sculpture for a catalogue, and you get exactly one photo. A photo is a projection: 3-D squashed onto 2-D. Shoot it end-on and it looks like a blob; shoot it from the side where it is widest and people can tell what it is. PCA is an algorithm for finding that best camera angle, for data with any number of dimensions.

## Projecting a vector onto another

Take two vectors in the plane: \( \mathbf{a} = [3, 4] \) and \( \mathbf{b} = [2, 1] \). We want \( \mathbf{p} \), the **projection of \( \mathbf{a} \) onto \( \mathbf{b} \)**: the point on the line through \( \mathbf{b} \) that sits directly "below" \( \mathbf{a} \) when light shines at a right angle to that line.

<figure class="diagram" markdown>
![Animation: vector a = [3, 4] in blue and b = [2, 1] in orange with a dashed line through b; a dashed grey drop line falls from the tip of a onto the line at right angles, a red shadow p = [4, 2] grows along the line, then the teal leftover e = [-1, 2] appears with a right-angle mark while the panel on the right fills in a·b = 10, b·b = 5, 10/5 = 2, p = 2b, e = a - p and e·b = 0](../../assets/img/phase-1/day25-projection-shadow.svg)
<figcaption>Animated: the shadow p is 2 copies of b, and the leftover e meets b's line at a right angle.</figcaption>
</figure>

### The math, one symbol at a time

Since \( \mathbf{p} \) lies on the line through \( \mathbf{b} \), it is some number of copies of \( \mathbf{b} \): \( \mathbf{p} = c\,\mathbf{b} \). The only question is the number \( c \). The answer is

\[
c = \frac{\mathbf{a} \cdot \mathbf{b}}{\mathbf{b} \cdot \mathbf{b}}
\qquad\qquad
\mathbf{p} = \frac{\mathbf{a} \cdot \mathbf{b}}{\mathbf{b} \cdot \mathbf{b}}\, \mathbf{b}
\]

- \( \mathbf{a} \cdot \mathbf{b} \) is the dot product from Day 5: how much \( \mathbf{a} \) and \( \mathbf{b} \) agree in direction, scaled by both lengths.
- \( \mathbf{b} \cdot \mathbf{b} = \|\mathbf{b}\|^2 \) is the squared length of \( \mathbf{b} \). Dividing by it cancels out "how long \( \mathbf{b} \) happens to be", so only its direction matters.

**Worked example.** \( \mathbf{a} \cdot \mathbf{b} = 3 \cdot 2 + 4 \cdot 1 = 10 \), and \( \mathbf{b} \cdot \mathbf{b} = 2 \cdot 2 + 1 \cdot 1 = 5 \). So \( c = 10/5 = 2 \) and \( \mathbf{p} = 2\,[2, 1] = [4, 2] \).

**Why that formula?** Call the leftover \( \mathbf{e} = \mathbf{a} - \mathbf{p} \) ("e" for error). The shadow is the point where \( \mathbf{e} \) is **perpendicular** to \( \mathbf{b} \), and perpendicular means dot product zero:

\[
\mathbf{b} \cdot (\mathbf{a} - c\,\mathbf{b}) = 0
\;\Longrightarrow\;
\mathbf{a} \cdot \mathbf{b} - c\,(\mathbf{b} \cdot \mathbf{b}) = 0
\;\Longrightarrow\;
c = \frac{\mathbf{a} \cdot \mathbf{b}}{\mathbf{b} \cdot \mathbf{b}}
\]

Check: \( \mathbf{e} = [3, 4] - [4, 2] = [-1, 2] \), and \( \mathbf{e} \cdot \mathbf{b} = -2 + 2 = 0 \). Perpendicular, as promised. And because it meets the line at a right angle, \( \mathbf{p} \) is also the **closest point on the line to \( \mathbf{a} \)** (the shortest path from a point to a line is the perpendicular one). Keep that sentence in mind; it is the whole of least squares.

If \( \mathbf{b} \) is a **unit vector** \( \mathbf{u} \) (length 1, Day 22), then \( \mathbf{u} \cdot \mathbf{u} = 1 \) and everything simplifies:

\[
\mathbf{p} = (\mathbf{a} \cdot \mathbf{u})\, \mathbf{u}
\qquad
\text{shadow length} = \mathbf{a} \cdot \mathbf{u}
\]

That is the geometric meaning of a dot product with a unit vector: **the signed length of the shadow**. Here \( \mathbf{u} = \mathbf{b} / \|\mathbf{b}\| = [2, 1]/\sqrt{5} \), and the shadow length is \( 10/\sqrt{5} \approx 4.472 \). Negative means the shadow points the opposite way to \( \mathbf{u} \).

```python
import numpy as np

a = np.array([3., 4.])
b = np.array([2., 1.])

coef = (a @ b) / (b @ b)   # how many b's long the shadow is
p = coef * b               # the projection (the shadow)
e = a - p                  # the leftover part
coef, p, e
# (2.0, array([4., 2.]), array([-1.,  2.]))
e @ b                      # 0.0   perpendicular
np.linalg.norm(p), (a @ b) / np.linalg.norm(b)
# (4.47213595499958, 4.47213595499958)   shadow length = a · (unit b)
```

Wrap it in a function and check that only the *direction* of \( \mathbf{b} \) matters:

```python
def project(a, b):
    """Projection of vector a onto the line through vector b."""
    return (a @ b) / (b @ b) * b

project(a, b)        # array([4., 2.])
project(a, 10 * b)   # array([4., 2.])   a longer b: same line, same shadow
project(a, -b)       # array([4., 2.])   pointing backwards: still the same line
```

### Projection as a matrix

Projecting is a linear transformation (Day 11), so it has a matrix. For a line through \( \mathbf{b} \):

\[
P = \frac{\mathbf{b}\,\mathbf{b}^\top}{\mathbf{b}^\top \mathbf{b}}
\qquad
\mathbf{p} = P\mathbf{a}
\]

\( \mathbf{b}\,\mathbf{b}^\top \) is an **outer product** (Day 23): a column times a row, giving a 2×2 matrix. Two properties are worth seeing in code. Projecting twice is the same as projecting once (\( P^2 = P \): the shadow of a shadow is itself). And, tying back to Day 24, \( P \)'s eigenvalues are 1 (vectors along the line are left alone) and 0 (vectors perpendicular to it are squashed to nothing).

```python
P = np.outer(b, b) / (b @ b)
P
# [[0.8 0.4]
#  [0.4 0.2]]
P @ a                     # array([4., 2.])   same shadow
np.allclose(P @ P, P)     # True: projecting again changes nothing
np.linalg.eigvalsh(P)     # array([0., 1.])  squash or keep, nothing in between
```

!!! warning "Common mistake: dividing by the length instead of the squared length"
    \( \frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{b}\|} \) is the shadow's **length** (a number). \( \frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{b}\|^2}\,\mathbf{b} \) is the shadow **vector**. Mixing them up gives a vector that points the right way but is \( \|\mathbf{b}\| \) times too long. Here that is \( \sqrt{5} \approx 2.24 \) times. If you normalize \( \mathbf{b} \) to a unit vector first, the two denominators are both 1 and the problem disappears.

## Projecting onto a subspace: least squares

A line through the origin is the simplest **subspace**. A plane through the origin is the next one, and on Day 23 you met the most important one: the **column space** of a matrix \( X \), the span of its columns, which is every vector you can write as \( X\mathbf{w} \) for some weights \( \mathbf{w} \).

Projecting onto a subspace works the same way: find the point in it closest to your vector, and the leftover is perpendicular to the *whole* subspace (to every column of \( X \)).

### Fitting a line is a projection

Three data points: \( (0, 1), (1, 2), (2, 2) \). We want a line \( \hat{y} = w_1 x + w_2 \) (slope \( w_1 \), intercept \( w_2 \)). Written as matrices, one row per point:

\[
\underbrace{\begin{bmatrix} 0 & 1 \\ 1 & 1 \\ 2 & 1 \end{bmatrix}}_{X}
\underbrace{\begin{bmatrix} w_1 \\ w_2 \end{bmatrix}}_{\mathbf{w}}
\approx
\underbrace{\begin{bmatrix} 1 \\ 2 \\ 2 \end{bmatrix}}_{\mathbf{y}}
\]

The first column holds the \( x \) values; the column of ones carries the intercept. There is no exact solution: three equations, two unknowns, and the points aren't on one line. In geometric terms: \( \mathbf{y} \) is a vector in 3-D, the columns of \( X \) span only a plane, and \( \mathbf{y} \) sticks out of it.

So we settle for the **closest** \( X\mathbf{w} \): the one that makes the miss \( \|\mathbf{y} - X\mathbf{w}\| \) as short as possible. Its squared length is the sum of squared errors, which is why this is called **least squares**. And the closest point is the projection of \( \mathbf{y} \) onto the plane.

<figure class="diagram" markdown>
![Animation: a teal parallelogram represents the plane of all X w, spanned by two teal column arrows; a blue vector y rises out of the plane; an orange dot tries three guesses in the plane with a dashed line to the tip of y, then settles directly below y where a red arrow y-hat = X w-hat lies in the plane and a teal residual r = y - y-hat rises perpendicular to it; the right panel adds X transpose times (y - X w-hat) = 0, so X transpose X w-hat = X transpose y, the normal equation from Day 11](../../assets/img/phase-1/day25-least-squares-plane.svg)
<figcaption>Animated: every guess w is a point in the plane; the best one is straight below y, so the miss is perpendicular to the plane.</figcaption>
</figure>

The residual \( \mathbf{r} = \mathbf{y} - X\hat{\mathbf{w}} \) (the hat means "best estimate") must be perpendicular to every column of \( X \). "Dot product with every column is zero" is one matrix product:

\[
X^\top (\mathbf{y} - X\hat{\mathbf{w}}) = \mathbf{0}
\quad\Longrightarrow\quad
X^\top X\, \hat{\mathbf{w}} = X^\top \mathbf{y}
\]

That is exactly Day 11's **normal equation**, now with a reason behind it. Worked by hand: \( X^\top X = \begin{bmatrix} 5 & 3 \\ 3 & 3 \end{bmatrix} \) and \( X^\top \mathbf{y} = \begin{bmatrix} 0 + 2 + 4 \\ 1 + 2 + 2 \end{bmatrix} = \begin{bmatrix} 6 \\ 5 \end{bmatrix} \). With the Day 11 inverse formula (\( \det = 15 - 9 = 6 \)):

\[
\hat{\mathbf{w}} = \frac{1}{6}\begin{bmatrix} 3 & -3 \\ -3 & 5 \end{bmatrix}\begin{bmatrix} 6 \\ 5 \end{bmatrix} = \frac{1}{6}\begin{bmatrix} 3 \\ 7 \end{bmatrix} \approx \begin{bmatrix} 0.5 \\ 1.167 \end{bmatrix}
\]

So the best line is \( \hat{y} = 0.5x + 1.167 \). NumPy's `lstsq` ("least squares") does this for you, more stably than forming \( X^\top X \):

```python
X = np.array([[0., 1.],
              [1., 1.],
              [2., 1.]])
y = np.array([1., 2., 2.])

w, res, rank, sv = np.linalg.lstsq(X, y)   # also returns residual sum, rank, singular values
w                  # array([0.5       , 1.16666667])   slope 0.5, intercept 7/6
y_hat = X @ w
y_hat              # array([1.16666667, 1.66666667, 2.16666667])   the shadow of y
r = y - y_hat
r                  # array([-0.16666667,  0.33333333, -0.16666667])
X.T @ r            # array([-4.4408921e-16,  4.4408921e-16])   r is perpendicular to both columns
np.linalg.solve(X.T @ X, X.T @ y)   # array([0.5       , 1.16666667])   Day 11's way agrees
```

The projection matrix for a column space generalizes the one-line formula: \( P = X (X^\top X)^{-1} X^\top \). Computed with `solve` instead of an inverse, as Day 11 taught:

```python
P_col = X @ np.linalg.solve(X.T @ X, X.T)   # (3, 3): projects any 3-vector onto the plane
P_col.round(3)
# [[ 0.833  0.333 -0.167]
#  [ 0.333  0.333  0.333]
#  [-0.167  0.333  0.833]]
P_col @ y                         # array([1.16666667, 1.66666667, 2.16666667])   = y_hat
np.allclose(P_col @ P_col, P_col) # True
```

On a realistic amount of data it is the same one call. Forty noisy points around the line \( y = 1.5x + 4 \):

```python
rng = np.random.default_rng(25)
x = rng.uniform(0, 10, size=40)
y = 1.5 * x + 4 + rng.normal(0, 2, size=40)
X = np.column_stack([x, np.ones_like(x)])     # shape (40, 2)
w, res, rank, sv = np.linalg.lstsq(X, y)
w.round(3)                       # array([1.487, 4.721])   close to the true 1.5 and 4
np.polyfit(x, y, 1).round(3)     # array([1.487, 4.721])   NumPy's line fit is the same projection
np.abs(X.T @ (y - X @ w)).max()  # about 1e-12: the residual is perpendicular to both columns
```

Now \( \mathbf{y} \) lives in 40-D space and the column space is a plane inside it. You can't draw that, but the algebra doesn't care: it is still a shadow on a plane.

!!! note "Least squares is regression"
    This is **linear regression**, the first model of Phase 2. The "learning" is a projection. When features number in the millions, the same answer is approached step by step with gradient descent instead (Month 2), but the target is still this shadow.

## PCA: finding the directions that matter

Now the second use of shadows. Here are 200 points in 2-D that clearly lie along a slanted band:

```python
rng = np.random.default_rng(25)
n = 200
raw = rng.normal(size=(n, 2)) * [3.0, 0.8]          # long in x, thin in y
theta = np.deg2rad(30)
R = np.array([[np.cos(theta), -np.sin(theta)],      # Day 11's rotation matrix
              [np.sin(theta),  np.cos(theta)]])
X = raw @ R.T + [5.0, 3.0]                            # rotate 30 degrees, move the centre
X.shape                   # (200, 2)
X.mean(axis=0).round(3)   # array([5.106, 3.042])
```

(`raw @ R.T` rotates every row at once: each row is a point, and \( (R\mathbf{x})^\top = \mathbf{x}^\top R^\top \).)

You *built* this data, so you know it is "really" one long direction at 30° plus a bit of thin noise. PCA's job is to discover that from the numbers alone: **find the unit direction \( \mathbf{u} \) where the shadows of the points spread out the most**. That direction is **principal component 1** (PC1). The next one, perpendicular to it, with the most remaining spread, is PC2, and so on.

<figure class="diagram" markdown>
![Animation: 36 blue data points in a slanted band around the origin; an orange line through the centre turns to 0, 30, 60, 90, 120 and 150 degrees, each point dropping a grey line onto it and leaving a red shadow dot; the panel on the right highlights the matching row of a bar chart of shadow variances 7.08, 9.30, 7.20, 2.87, 0.65 and 2.75, then returns to 30 degrees and announces it as PC1 with lambda 1 = 9.30, while 120 degrees, the smallest, is PC2 with lambda 2 = 0.65](../../assets/img/phase-1/day25-pca-rotate.svg)
<figcaption>Animated: the shadows spread out most at about 30° (PC1) and least at 120° (PC2), perpendicular to it.</figcaption>
</figure>

### Why eigenvectors give the answer

Measure the spread of the shadows with their **variance** (average squared distance from the mean). Three steps turn "try every angle" into one line of linear algebra.

**Step 1: center the data.** Subtract the mean of each column, so the cloud sits around the origin: \( X_c = X - \bar{\mathbf{x}} \) (\( \bar{\mathbf{x}} \), "x-bar", is the vector of column means; broadcasting from Day 9 does the subtraction).

**Step 2: the covariance matrix.** For \( n \) points and \( d \) columns,

\[
C = \frac{1}{n-1} X_c^\top X_c
\qquad\text{shape } (d, d)
\]

Entry \( C_{jj} \) on the diagonal is the variance of column \( j \). Entry \( C_{jk} \) off the diagonal is the **covariance** of columns \( j \) and \( k \): positive when they tend to be big together, negative when one is big as the other is small, near zero when unrelated. (The \( n - 1 \) instead of \( n \) is a statistics convention you'll meet in Month 2; it doesn't change any directions.) \( C \) is symmetric, since \( C_{jk} = C_{kj} \).

**Step 3: the variance of the shadows on any unit direction \( \mathbf{u} \).** The shadow lengths are \( X_c\mathbf{u} \) (one dot product per point), and their variance works out to

\[
\operatorname{var}(X_c\mathbf{u}) = \frac{1}{n-1}(X_c\mathbf{u})^\top(X_c\mathbf{u}) = \mathbf{u}^\top C\, \mathbf{u}
\]

So "the direction with the most spread" means: find the unit vector \( \mathbf{u} \) that makes \( \mathbf{u}^\top C \mathbf{u} \) as large as possible. A standard linear-algebra result answers it: the maximum is the **largest eigenvalue** \( \lambda_1 \) of \( C \) (lambda one), reached at its **eigenvector**. The second eigenvector gives the best direction perpendicular to the first, and so on. Day 24 promised that a symmetric matrix has perpendicular eigenvectors; here is why that matters: the principal components automatically come out at right angles to each other.

Each eigenvalue *is* the variance along its direction. The quick check: if \( C\mathbf{u} = \lambda\mathbf{u} \) and \( \|\mathbf{u}\| = 1 \), then \( \mathbf{u}^\top C \mathbf{u} = \lambda\,\mathbf{u}^\top\mathbf{u} = \lambda \).

### A tiny example by hand

Four centered points: \( (2, 1), (1, 2), (-1, -2), (-2, -1) \). Their column means are already 0. Column \( x \) has squares \( 4 + 1 + 1 + 4 = 10 \), so variance \( 10/3 \); column \( y \) is the same. The cross products \( x \cdot y \) are \( 2 + 2 + 2 + 2 = 8 \), so covariance \( 8/3 \):

\[
C = \frac{1}{3}\begin{bmatrix} 10 & 8 \\ 8 & 10 \end{bmatrix}
\]

A symmetric 2×2 matrix of the form \( \begin{bmatrix} p & q \\ q & p \end{bmatrix} \) has eigenvectors \( [1, 1] \) and \( [1, -1] \), with eigenvalues \( p + q \) and \( p - q \) (multiply it out to check). So \( \lambda_1 = 18/3 = 6 \) along \( [1, 1]/\sqrt{2} \), the 45° diagonal the points cluster around, and \( \lambda_2 = 2/3 \) along \( [1, -1]/\sqrt{2} \). PC1 holds \( 6 / (6 + 2/3) = 90\% \) of the total variance.

```python
T = np.array([[ 2.,  1.],
              [ 1.,  2.],
              [-1., -2.],
              [-2., -1.]])
T.mean(axis=0)                 # array([0., 0.])   already centered
C = T.T @ T / (len(T) - 1)
C
# [[3.33333333 2.66666667]
#  [2.66666667 3.33333333]]
evals, evecs = np.linalg.eigh(C)   # eigh: for symmetric matrices (Day 24)
evals          # array([0.66666667, 6.        ])   ascending order!
evecs
# [[-0.70710678  0.70710678]
#  [ 0.70710678  0.70710678]]      columns are eigenvectors: [-1, 1]/√2 and [1, 1]/√2
T @ evecs[:, ::-1]                 # project onto PC1 and PC2 (largest first)
# [[ 2.12132034 -0.70710678]
#  [ 2.12132034  0.70710678]
#  [-2.12132034 -0.70710678]
#  [-2.12132034  0.70710678]]
```

The first column of that last result is each point's coordinate along PC1: \( (2 + 1)/\sqrt{2} \approx 2.121 \). Four 2-D points became four PC1 numbers that keep 90% of the variance.

## PCA from scratch, step by step

Back to the 200 points. The full recipe: **center, covariance, eigh, sort, project**.

```python
mean = X.mean(axis=0)
Xc = X - mean                         # 1. center (broadcasting)
C = Xc.T @ Xc / (n - 1)               # 2. covariance, shape (2, 2)
C.round(3)
# [[7.081 3.781]
#  [3.781 2.868]]
np.allclose(C, np.cov(X, rowvar=False))   # True: NumPy's built-in agrees
```

`np.cov` treats rows as variables by default, so `rowvar=False` tells it "my variables are columns". The positive covariance 3.781 says x and y rise together: the band slopes upwards.

```python
evals, evecs = np.linalg.eigh(C)      # 3. eigen-decomposition
evals.round(3)                        # array([0.647, 9.303])   ascending
order = np.argsort(evals)[::-1]       # 4. sort: biggest first
evals, evecs = evals[order], evecs[:, order]
evals.round(3)                        # array([9.303, 0.647])
evecs.round(3)
# [[-0.862  0.507]
#  [-0.507 -0.862]]
np.rad2deg(np.arctan2(evecs[1, 0], evecs[0, 0])).round(1)   # -149.6
evecs.T @ evecs
# [[1.00000000e+00 1.08346656e-17]
#  [1.08346656e-17 1.00000000e+00]]   the identity: perpendicular unit vectors
```

PC1 points at −149.6°, which is the 30.4° line pointing the other way (−149.6 + 180 = 30.4). An eigenvector's sign is arbitrary: if \( \mathbf{u} \) is one, so is \( -\mathbf{u} \), and the line is the same. Our hidden 30° direction is recovered from the numbers alone, and \( \lambda_1 = 9.303 \) is the variance along it.

Check the "try every direction" claim from the animation, with real numbers:

```python
for deg in range(0, 180, 15):
    u = np.array([np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))])
    print(f"{deg:4d} deg  variance of shadows = {np.var(Xc @ u, ddof=1):.3f}")
#    0 deg  variance of shadows = 7.081
#   15 deg  variance of shadows = 8.689
#   30 deg  variance of shadows = 9.302
#   45 deg  variance of shadows = 8.755
#   60 deg  variance of shadows = 7.196
#   75 deg  variance of shadows = 5.041
#   90 deg  variance of shadows = 2.868
#  105 deg  variance of shadows = 1.260
#  120 deg  variance of shadows = 0.647
#  135 deg  variance of shadows = 1.194
#  150 deg  variance of shadows = 2.754
#  165 deg  variance of shadows = 4.909
```

(`ddof=1` makes `np.var` divide by \( n - 1 \), matching \( C \).) The biggest value in the sweep, 9.302 at 30°, is just under \( \lambda_1 = 9.303 \) at 30.4°; the smallest, at 120°, matches \( \lambda_2 = 0.647 \). At 0° and 90° you get the plain column variances 7.081 and 2.868, the diagonal of \( C \).

**Step 5: project.** Multiply the centered data by the eigenvector matrix. Each row of the result, the **scores**, is that point's coordinates in the new PC axes:

```python
Z = Xc @ evecs                        # 5. project: shape (200, 2)
Z[:3].round(3)
# [[-0.949 -0.006]
#  [ 1.721  1.791]
#  [ 0.051 -0.759]]
np.cov(Z, rowvar=False).round(3)
# [[ 9.303 -0.   ]
#  [-0.     0.647]]     variances = eigenvalues; covariance 0 (-0. is a signed zero)
```

In the new axes the two columns are **uncorrelated**: PCA rotated the cloud so its spread lines up with the axes. Nothing was lost yet; it is just a rotation.

### Explained variance ratio

The **explained variance ratio** of component \( i \) is its share of the total:

\[
\text{ratio}_i = \frac{\lambda_i}{\lambda_1 + \lambda_2 + \dots + \lambda_d}
\]

```python
ratio = evals / evals.sum()
ratio.round(3)       # array([0.935, 0.065])
```

PC1 alone carries 93.5% of the variance. If you keep only PC1, you describe each point with 1 number instead of 2 and lose 6.5% of the variance.

### Reconstruct: going back

Keep \( k \) components, the first \( k \) columns of the eigenvector matrix, called \( V_k \) (shape \( (d, k) \)). Compress, then map back to the original space:

\[
Z_k = X_c V_k
\qquad
\hat{X} = Z_k V_k^\top + \bar{\mathbf{x}}
\]

Going back works with a plain transpose, no inverse, because the columns of \( V_k \) are perpendicular unit vectors (Day 11: for such matrices "inverting" is free). \( \hat{X} \) is each point's shadow on the PC1 line, put back where the data lives.

```python
k = 1
V1 = evecs[:, :k]                    # (2, 1)
z1 = Xc @ V1                         # (200, 1): one number per point
X_hat = z1 @ V1.T + mean             # (200, 2): back in the original space
X[:2].round(3)
# [[5.922 3.528]
#  [4.529 0.625]]
X_hat[:2].round(3)
# [[5.925 3.523]
#  [3.622 2.17 ]]
err = ((X - X_hat) ** 2).sum() / (n - 1)
err.round(4), evals[1].round(4)      # (0.6468, 0.6468)
```

The first point was almost on the line and barely moved; the second was far off it and moved more. And the reconstruction error (summed squared distance, divided by \( n - 1 \)) equals exactly the variance you threw away, \( \lambda_2 \). That is PCA's guarantee: **among all ways to squash the data onto \( k \) directions, the top \( k \) principal components lose the least**.

Here is the whole thing as a figure. The code also builds the complete PCA as a reusable function, used for the rest of the lesson:

```python
import matplotlib
matplotlib.use("Agg")            # draw into files, no window (leave this out in Jupyter)
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def pca(X, k):
    """PCA from scratch. Returns scores Z (n, k), directions V (d, k), mean (d,), all eigenvalues."""
    mean = X.mean(axis=0)
    Xc = X - mean                                   # 1. center
    C = Xc.T @ Xc / (len(X) - 1)                    # 2. covariance (d, d)
    evals, evecs = np.linalg.eigh(C)                # 3. eigen-decomposition (ascending!)
    order = np.argsort(evals)[::-1]                 # 4. sort, biggest first
    evals, evecs = evals[order], evecs[:, order]
    big = np.abs(evecs).argmax(axis=0)              #    sign convention: flip each column so
    evecs = evecs * np.sign(evecs[big, np.arange(len(evals))])   # its largest entry is positive
    V = evecs[:, :k]
    Z = Xc @ V                                      # 5. project
    return Z, V, mean, evals

Z2, V2, mu2, ev2 = pca(X, k=2)
V2.round(3)
# [[ 0.862 -0.507]
#  [ 0.507  0.862]]       PC1 now points at +30.4 degrees
(ev2 / ev2.sum()).round(3)     # array([0.935, 0.065])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
ax1.scatter(X[:, 0], X[:, 1], s=12, color="#3949ab", alpha=0.6)
for j, (col, name) in enumerate([("#e53935", "PC1"), ("#00897b", "PC2")]):
    tip = mu2 + 2 * np.sqrt(ev2[j]) * V2[:, j]     # arrow length: 2 standard deviations
    ax1.annotate("", xy=tip, xytext=mu2, arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5))
    ax1.text(*(tip + 0.3 * V2[:, j]), f"{name} ({ev2[j] / ev2.sum():.1%})", color=col,
             ha="right" if j == 1 else "left", bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
ax1.set_title("The data and its principal components")

Z1, V1, mu1, _ = pca(X, k=1)
X_back = Z1 @ V1.T + mu1                           # reconstruct from PC1 only
for p_orig, p_back in zip(X, X_back):
    ax2.plot([p_orig[0], p_back[0]], [p_orig[1], p_back[1]], color="#bdbdbd", lw=0.7)
ax2.scatter(X[:, 0], X[:, 1], s=12, color="#3949ab", alpha=0.35, label="original")
ax2.scatter(X_back[:, 0], X_back[:, 1], s=12, color="#e53935", label="kept: PC1 only")
ax2.set_title("Reconstructed from 1 number per point")
ax2.legend(loc="upper left", frameon=False)
for ax in (ax1, ax2):
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.grid(alpha=0.3)
ax1.set_ylabel("y")
fig.savefig("day25-pca-2d.svg", format="svg", bbox_inches="tight")
plt.close(fig)
```

<figure class="diagram" markdown>
![Two scatter plots side by side of the same 200 blue points in a band slanting up at 30 degrees; on the left a long red arrow labelled PC1 93.5% points along the band from its centre and a short teal arrow labelled PC2 6.5% points across it; on the right every point is joined by a short grey line to its red reconstruction, and all the red points lie on one straight line along the band](../../assets/img/phase-1/day25-pca-2d.svg)
<figcaption>Left: PC1 runs along the band, PC2 across it. Right: keeping only PC1 moves every point onto that line; the grey segments are the 6.5% that was thrown away.</figcaption>
</figure>

The sign convention in `pca` (largest entry of each direction positive) doesn't change any result; it only makes the output repeatable, so a plot or a saved model doesn't flip from run to run.

!!! warning "Common mistake: forgetting that `eigh` sorts ascending"
    `np.linalg.eigh` returns eigenvalues from **smallest to largest**. Take `evecs[:, 0]` without sorting and you've picked the *least* important direction, the one you meant to throw away. Always sort with `np.argsort(evals)[::-1]` and reorder the eigenvector **columns** to match.

!!! warning "Common mistake: forgetting to center"
    PCA is about spread *around the mean*. If you skip `X - X.mean(axis=0)`, the "top direction" mostly points from the origin to the middle of the cloud, which says where the data is, not how it varies. Here the mean happens to sit at about 31°, close to the true 30°, which would hide the bug; on other data it gives nonsense. Center every time, and keep the mean: you need it to project new points and to reconstruct.

## The SVD: how libraries actually do it

Computing \( X_c^\top X_c \) squares every number, which can lose precision when the data has very large and very small variances. Real libraries skip the covariance matrix and use the **singular value decomposition** (SVD) of the centered data directly:

\[
X_c = U\,S\,V^\top
\]

- \( V^\top \) (shape \( (d, d) \)): its **rows** are the principal directions, the same eigenvectors as before.
- \( S \): the **singular values** \( s_1 \ge s_2 \ge \dots \), already sorted, largest first. Each eigenvalue is \( \lambda_i = s_i^2 / (n - 1) \).
- \( U \) (shape \( (n, d) \) in the "thin" version): \( U S \) gives the scores, the same as \( X_c V \).

```python
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)    # "thin" SVD: no huge U
U.shape, s.shape, Vt.shape        # ((200, 2), (2,), (2, 2))
(s**2 / (n - 1)).round(3)         # array([9.303, 0.647])   = the eigenvalues
np.allclose(np.abs(Vt.T), np.abs(V2))    # True: same directions, signs may differ
np.allclose(np.abs(U * s), np.abs(Z2))   # True: same scores
```

Same answer, no covariance matrix, no sorting step. scikit-learn's `PCA` class (Phase 2) works this way: on your machine with scikit-learn installed, `PCA(n_components=2).fit_transform(X)` gives these scores, up to sign. You built the five-step version so you know what that one line does.

## PCA on real data: the Play Store apps

Now four numeric columns from the course dataset. Load it and apply the short cleaning from Days 17–19:

```python
import pandas as pd

BASE = "https://manjulkadyan.github.io/ai-leaning/assets/data/"
apps = pd.read_csv(BASE + "play_store_apps.csv")
apps = apps.drop_duplicates().assign(
    category=lambda d: d["category"].str.strip().str.title(),
    installs=lambda d: d["installs"].str.replace(",", "").str.rstrip("+").astype(int),
    size_mb=lambda d: d["size_mb"].replace("Varies with device", np.nan).astype(float),
)
apps.shape     # (480, 12)
```

No internet? Download the file from the [Datasets page](../../datasets.md) and pass the local path to `read_csv`. (Tested on pandas 3.0.)

Installs and reviews are hugely skewed (Day 19), so take their `log10`: 1,000 installs becomes 3, 1,000,000 becomes 6. Every app has at least one review, so the log is safe. PCA can't handle missing values, so drop the rows with a missing rating or size:

```python
feats = pd.DataFrame({
    "log_installs": np.log10(apps["installs"]),
    "log_reviews": np.log10(apps["reviews"]),
    "rating": apps["rating"],
    "size_mb": apps["size_mb"],
}).dropna()
feats.shape    # (442, 4)
print(feats.describe().loc[["mean", "std", "min", "max"]].round(2).to_string())
#       log_installs  log_reviews  rating  size_mb
# mean          4.45         3.20    4.17    43.70
# std           0.93         0.94    0.46    37.68
# min           3.00         0.30    2.90     4.40
# max           8.00         6.40    5.00   205.20
print(feats.corr().round(2).to_string())
#               log_installs  log_reviews  rating  size_mb
# log_installs          1.00         0.94    0.02     0.13
# log_reviews           0.94         1.00    0.00     0.11
# rating                0.02         0.00    1.00     0.12
# size_mb               0.13         0.11    0.12     1.00
```

`size_mb` has a standard deviation of 37.68; `rating` has 0.46. Raw PCA would chase the biggest variance, so it would mostly rediscover "size", only because megabytes are bigger numbers than stars (Exercise 5 shows it). So **standardize** first, as on Day 22: subtract each column's mean and divide by its standard deviation, so every column has variance 1 and counts equally.

```python
F = feats.to_numpy()
F_std = (F - F.mean(axis=0)) / F.std(axis=0, ddof=1)     # every column: mean 0, std 1
Zp, Vp, mup, evp = pca(F_std, k=2)
ratio = evp / evp.sum()
ratio.round(3)              # array([0.491, 0.275, 0.217, 0.016])
ratio.cumsum().round(3)     # array([0.491, 0.767, 0.984, 1.   ])
print(pd.DataFrame(Vp, index=feats.columns, columns=["PC1", "PC2"]).round(2).to_string())
#                PC1   PC2
# log_installs  0.70 -0.09
# log_reviews   0.69 -0.12
# rating        0.04  0.75
# size_mb       0.17  0.65
```

How to read this:

- The **loadings** (the entries of each direction) say what a component is made of. **PC1** is almost entirely `log_installs` and `log_reviews` with equal weight: call it "popularity". It carries 49.1% of the variance on its own, because those two columns are 0.94 correlated and so move as one.
- **PC2** mixes `rating` and `size_mb` (27.5%). Their correlation is only 0.12, so this is a weak pairing; PC3 (21.7%) holds most of the rest of those two columns. Don't over-interpret a component that is barely bigger than the next one.
- **PC4 explains only 1.6%.** The four columns behave like three: installs and reviews nearly duplicate each other. This is Day 23's "almost collinear columns" seen through PCA: a near-zero eigenvalue means a direction the data hardly uses.

Which apps sit at the two ends of PC1?

```python
ends = apps.loc[feats.index, ["app_name"]].join(feats).assign(PC1=Zp[:, 0]).sort_values("PC1")
print(ends.iloc[[0, -1]].round(2).to_string())
```

```text
            app_name  log_installs  log_reviews  rating  size_mb   PC1
279       Calm Radio           3.0         0.48     4.0     48.5 -3.09
399  Orbit Lens Lite           8.0         6.40     3.7     83.2  5.16
```

The lowest PC1 is an app with 1,000 installs and 3 reviews; the highest has 100,000,000 installs and about 2.5 million reviews. "Popularity" it is. Now the picture: a **scree plot** (explained variance per component, named after the rubble at the foot of a cliff) and the apps in the PC1–PC2 plane, with arrows showing how each original column maps into it:

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1, 1.5]})
pcs = ["PC1", "PC2", "PC3", "PC4"]
ax1.bar(pcs, ratio, color="#3949ab")
ax1.plot(pcs, ratio.cumsum(), color="#ef6c00", marker="o", label="cumulative")
for i, r in enumerate(ratio):
    ax1.text(i, r + 0.02, f"{r:.1%}", ha="center", fontsize=10)
ax1.set_ylim(0, 1.08)
ax1.set_ylabel("share of total variance")
ax1.set_title("Scree plot")
ax1.legend(frameon=False, loc="center right")

ax2.scatter(Zp[:, 0], Zp[:, 1], s=12, color="#3949ab", alpha=0.45)
nudge = {"log_installs": (0.1, 0.15, "left"), "log_reviews": (0.1, -0.45, "left"),
         "rating": (-0.1, 0.1, "right"), "size_mb": (0.12, -0.1, "left")}   # keep labels apart
for name, (l1, l2) in zip(feats.columns, Vp):       # one arrow per original column
    ax2.annotate("", xy=(3 * l1, 3 * l2), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="#e53935", lw=1.8))
    dx, dy, ha = nudge[name]
    ax2.text(3 * l1 + dx, 3 * l2 + dy, name, color="#e53935", fontsize=10, ha=ha,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
ax2.axhline(0, color="#9e9e9e", lw=0.8)
ax2.axvline(0, color="#9e9e9e", lw=0.8)
ax2.set_xlabel(f"PC1: popularity ({ratio[0]:.1%})")
ax2.set_ylabel(f"PC2: rating and size ({ratio[1]:.1%})")
ax2.set_title("442 apps in the first two components")
ax2.grid(alpha=0.3)
fig.savefig("day25-playstore-pca.svg", format="svg", bbox_inches="tight")
plt.close(fig)
```

<figure class="diagram" markdown>
![Left: a bar chart of explained variance for PC1 to PC4, 49.1%, 27.5%, 21.7% and 1.6%, with an orange cumulative line rising to 100%; right: a scatter of 442 apps with PC1 on the horizontal axis and PC2 on the vertical axis, and red arrows from the origin: log_installs and log_reviews point almost straight right along PC1, while rating and size_mb point up along PC2](../../assets/img/phase-1/day25-playstore-pca.svg)
<figcaption>Four columns, three real dimensions: PC4 is almost empty. The arrows show installs and reviews collapsing into one "popularity" axis.</figcaption>
</figure>

!!! warning "Common mistake: reading components as causes"
    PCA describes how columns vary *together in this dataset*. "PC1 = popularity" is a label you chose after looking at the loadings, not something the algorithm knows, and PC2 being "rating and size" does not mean bigger apps get better ratings. Like Day 20's findings, a component is a description, not an explanation.

!!! warning "Common mistake: fitting PCA on all your data before a train/test split"
    The mean, the standard deviations and the directions are all learned from data. In Phase 2 you'll split data into training and test sets; compute all three on the **training set only**, then apply the same `mean`, scale and `V` to the test set. Fitting on everything leaks information about the test set into the model, the same data-leakage warning as Day 17's fill values.

## PCA as compression

The same tool shrinks big vectors. An **embedding** model (you met them on Day 22) turns a sentence or an image into a long vector of floats, say 256 of them, and similar items get similar vectors. Those 256 numbers are rarely 256 independent things. Simulate that: each item is really a mix of 24 hidden "topics", spread over 256 dimensions plus noise, then L2-normalized as embedding models usually do.

```python
rng = np.random.default_rng(7)
n_items, d, hidden = 2000, 256, 24
topics = rng.normal(size=(n_items, hidden))            # each item: a mix of 24 hidden topics
mixing = rng.normal(size=(hidden, d))
E = topics @ mixing + 0.5 * rng.normal(size=(n_items, d))   # plus noise in every dimension
E = E / np.linalg.norm(E, axis=1, keepdims=True)       # unit length, like real embeddings
E.shape                                                # (2000, 256)

Ze, Ve, mue, eve = pca(E, k=32)
cum = np.cumsum(eve) / eve.sum()
cum[[7, 15, 23, 31, 63]].round(3)   # array([0.446, 0.766, 0.99 , 0.991, 0.993])
```

The cumulative share climbs to 99% at exactly 24 components and then hardly moves: PCA found the 24 hidden topics without being told. With real embeddings the curve is smoother and you choose \( k \) by testing; the method is the same. Keep 32 and compare similarities before and after:

```python
def cosine_rows(A, B):
    """Cosine similarity between matching rows of A and B."""
    return (A * B).sum(axis=1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))

np.linalg.norm(mue).round(3)          # 0.015: the mean is nearly zero, so centering barely moves anything
i = rng.integers(0, n_items, size=500)
j = rng.integers(0, n_items, size=500)
cos_full = cosine_rows(E[i], E[j])    # 500 random pairs, 256 floats each
cos_small = cosine_rows(Ze[i], Ze[j]) # the same pairs, 32 floats each
cos_full[:4].round(3)                 # array([ 0.108, -0.094,  0.421,  0.017])
cos_small[:4].round(3)                # array([ 0.111, -0.103,  0.423,  0.019])
np.abs(cos_full - cos_small).max().round(3)   # 0.015: the worst pair is off by 0.015

nn_full = np.argsort(-(E @ E[0]))[1:6]           # 5 nearest neighbours of item 0, full size
Zn = Ze / np.linalg.norm(Ze, axis=1, keepdims=True)
nn_small = np.argsort(-(Zn @ Zn[0]))[1:6]        # the same search on 32 floats
nn_full, nn_small
# (array([1430,  757, 1085,  242, 1701]), array([1430,  757, 1085,  242, 1701]))

full_bytes = n_items * d * 4                       # float32 = 4 bytes
small_bytes = n_items * 32 * 4 + (d * 32 + d) * 4  # scores, plus V and the mean to ship
full_bytes, small_bytes, round(full_bytes / small_bytes, 1)
# (2048000, 289792, 7.1)
```

Same nearest neighbours, similarities within 0.015, and about 7× less storage including the cost of shipping \( V \) and the mean. (PCA preserves similarities between *centered* vectors; this data's mean is nearly zero, so that's the same thing here. On real embeddings, judge the result by search quality on your own queries.)

!!! android "Android bridge"
    On-device semantic search (notes, photos, chat history) stores one embedding per item, often in a Room table as a `BLOB` or in a flat file. Shrinking each vector with PCA cuts storage and makes every brute-force similarity scan (Day 22) proportionally faster, which matters on a mid-range phone with thousands of items.

    The split of work is the classic one: **fit offline, apply on device.** In Python you compute `mean` (\( d \) floats) and `V` (\( d \times k \) floats) once, export them as little-endian `float32` bytes (for example `V.astype(np.float32).tobytes()`), and ship them in `assets/`. On the phone, compressing a new embedding is one subtraction and one small matrix-vector product, \( \mathbf{z} = (\mathbf{x} - \bar{\mathbf{x}})\,V \):

    ```kotlin
    class PcaCompressor(
        private val mean: FloatArray,   // size d
        private val v: FloatArray,      // d * k values, row-major: v[i * k + j] = V[i][j]
        private val d: Int,
        private val k: Int,
    ) {
        fun compress(x: FloatArray): FloatArray {
            require(x.size == d)
            val z = FloatArray(k)
            for (i in 0 until d) {
                val xi = x[i] - mean[i]               // center with the TRAINING mean
                val row = i * k
                for (j in 0 until k) z[j] += xi * v[row + j]
            }
            return z
        }
    }
    ```

    When reading those bytes with `ByteBuffer.wrap(bytes)`, call `.order(ByteOrder.LITTLE_ENDIAN)` before `.asFloatBuffer()`: a `ByteBuffer` is big-endian by default, and NumPy on x86 and ARM writes little-endian. Re-normalize the 32-float vectors before cosine similarity, as `Zn` did above. And if you change the embedding model, the old `mean` and `V` are meaningless: refit them.

---

## Exercises

1. By hand, project \( \mathbf{a} = [4, 2] \) onto \( \mathbf{b} = [1, 1] \). Give \( \mathbf{p} \), the leftover \( \mathbf{e} \) and the shadow length. Check with `project` and show \( \mathbf{e} \cdot \mathbf{b} = 0 \).
2. Project \( \mathbf{a} = [3, 4] \) onto \( [1, 0] \), and then onto \( \mathbf{u} = [0.6, 0.8] \). Explain both answers in one sentence each. What do the coordinates of a vector have to do with projections?
3. Fit a line \( y = w_1 x + w_2 \) to the points \( (1, 2), (2, 3), (3, 5), (4, 6) \) with `np.linalg.lstsq`. Verify that the residual is perpendicular to both columns of \( X \), and compare with `np.polyfit`.
4. By hand, find the covariance matrix and the principal components of the four points \( (3, 0), (-3, 0), (0, 1), (0, -1) \). What share of the variance does PC1 explain? Confirm with `pca`.
5. Run `pca` on the Play Store features **without** standardizing (`pca(F, k=2)`). What are the explained variance ratios and PC1's loadings? Why is the result misleading?
6. For the embeddings `E`, find the smallest \( k \) that keeps at least 95% of the variance (hint: `np.searchsorted` on the cumulative share). How many bytes would 2,000 items take at that \( k \), including `V` and the mean?

??? success "Solutions"
    ```python
    # 1: a·b = 4 + 2 = 6, b·b = 2, so c = 3 and p = [3, 3]; e = [1, -1]
    a1, b1 = np.array([4., 2.]), np.array([1., 1.])
    p1 = project(a1, b1)
    p1, a1 - p1                       # (array([3., 3.]), array([ 1., -1.]))
    (a1 - p1) @ b1                    # 0.0
    (a1 @ b1) / np.linalg.norm(b1)    # 4.242640687119285   = 6 / √2
    ```
    ```python
    # 2
    a2 = np.array([3., 4.])
    project(a2, np.array([1., 0.]))   # array([3., 0.])
    project(a2, np.array([0.6, 0.8])) # array([3., 4.])
    ```
    Onto \( [1, 0] \) the shadow is \( [3, 0] \): projecting onto an axis keeps just that coordinate. Onto \( [0.6, 0.8] \) the shadow is \( \mathbf{a} \) itself, because \( \mathbf{a} = 5\,[0.6, 0.8] \) already lies on that line, so nothing is left over. In general, **a vector's coordinates are its shadow lengths on the basis vectors**: \( 3 = \mathbf{a} \cdot [1, 0] \) and \( 4 = \mathbf{a} \cdot [0, 1] \). PCA scores are the same idea with the PC directions as the basis.

    ```python
    # 3
    x3 = np.array([1., 2., 3., 4.])
    y3 = np.array([2., 3., 5., 6.])
    X3 = np.column_stack([x3, np.ones_like(x3)])
    w3, *_ = np.linalg.lstsq(X3, y3)
    w3                              # array([1.4, 0.5])   (may print as 1.4000000000000001)
    X3.T @ (y3 - X3 @ w3)           # both entries are 0 up to rounding (about 1e-14)
    np.polyfit(x3, y3, 1)           # array([1.4, 0.5])
    ```
    By hand: \( X^\top X = \begin{bmatrix} 30 & 10 \\ 10 & 4 \end{bmatrix} \), \( X^\top\mathbf{y} = [51, 16] \), and solving gives slope 1.4 and intercept 0.5.

    ```python
    # 4: mean is 0; var(x) = (9 + 9) / 3 = 6, var(y) = (1 + 1) / 3 = 2/3, covariance 0
    T4 = np.array([[3., 0.], [-3., 0.], [0., 1.], [0., -1.]])
    Z4, V4, mu4, ev4 = pca(T4, k=2)
    np.cov(T4, rowvar=False)
    # [[6.         0.        ]
    #  [0.         0.66666667]]
    V4                       # [[1. 0.]  [0. 1.]]  PC1 = x-axis, PC2 = y-axis
    ev4 / ev4.sum()          # array([0.9, 0.1])
    ```
    The covariance is already diagonal (the columns are uncorrelated), so the eigenvectors are the axes themselves and the eigenvalues are the diagonal: 6 and 2/3. PC1 explains \( 6 / (20/3) = 90\% \).

    ```python
    # 5
    Zr, Vr, mur, evr = pca(F, k=2)
    (evr / evr.sum()).round(4)   # array([9.986e-01, 1.200e-03, 1.000e-04, 0.000e+00])
    Vr[:, 0].round(3)            # array([0.003, 0.003, 0.001, 1.   ])   PC1 = size_mb
    ```
    Without standardizing, PC1 is just `size_mb` and claims 99.86% of the variance, because size is measured in megabytes with a variance of about 1,420 (37.68²), against about 0.9 for the log columns and 0.2 for rating. The "finding" is an artifact of units: measure size in GB and the result changes completely. Standardize whenever the columns have different units.

    ```python
    # 6
    k95 = int(np.searchsorted(cum, 0.95) + 1)
    k95                                              # 23
    n_items * k95 * 4 + (d * k95 + d) * 4            # 208576 bytes, against 2048000 at full size
    ```
    `cum[22]` (23 components) is the first value at or above 0.95; `searchsorted` returns the index 22 and the `+ 1` turns it into a count.

## Resources

- [3Blue1Brown: Dot products and duality (Essence of Linear Algebra, ch. 9)](https://www.youtube.com/watch?v=LyGKycYT2v0): why a dot product is a projection, animated.
- [StatQuest: Principal Component Analysis (PCA), Step-by-Step](https://www.youtube.com/watch?v=FgakZw6K1QQ): a gentle, picture-first walk through the same five steps.
- [Mathematics for Machine Learning (free book)](https://mml-book.github.io/): chapter 3.8 covers orthogonal projections and chapter 10 derives PCA properly, including the "maximum variance" and "minimum reconstruction error" views.
- [Khan Academy: Linear algebra](https://www.khanacademy.org/math/linear-algebra): the "Alternate coordinate systems (bases)" unit has projections onto subspaces and least squares with practice.
- [NumPy docs: `numpy.linalg.lstsq`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html), [`numpy.linalg.eigh`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html) and [`numpy.linalg.svd`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html).
- [scikit-learn user guide: PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca): the library version you'll use in Phase 2, with notes on whitening and incremental PCA.
- [Android reference: `java.nio.ByteBuffer`](https://developer.android.com/reference/java/nio/ByteBuffer): `order(ByteOrder.LITTLE_ENDIAN)` and `asFloatBuffer()` for loading `mean` and `V`.

## Checklist

- [ ] I can project a vector onto another with \( \frac{\mathbf{a}\cdot\mathbf{b}}{\mathbf{b}\cdot\mathbf{b}}\,\mathbf{b} \) and check that the leftover is perpendicular.
- [ ] I know that a dot product with a unit vector is the signed length of a shadow.
- [ ] I can explain least squares as "the shadow of \( \mathbf{y} \) on the column space" and derive the normal equation from "the residual is perpendicular".
- [ ] I fitted a line with `np.linalg.lstsq` and got the same answer as `np.polyfit`.
- [ ] I can run PCA from scratch: center, covariance, `eigh`, sort descending, project, explained variance ratio, reconstruct.
- [ ] I know why to standardize columns with different units before PCA, and why PCA is fitted on training data only.
- [ ] I can say how the SVD gives the same components, and why libraries prefer it.
- [ ] I can describe how to fit PCA offline and apply it on device to compress embeddings.

**Next:** [Day 26: Month 1 recap and gaps](day-26-month1-recap.md)
