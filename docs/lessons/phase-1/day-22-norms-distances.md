# Day 22: Norms and Distances

**Phase 1 · Week 4 · ~120 minutes**

## Objectives

- [ ] Compute the L1, L2 and L∞ norms of a vector by hand and with `np.linalg.norm(x, ord=...)`.
- [ ] Sketch the "unit ball" of each norm and say what its shape means.
- [ ] Choose between Euclidean, Manhattan and cosine distance for a problem, and explain why.
- [ ] Normalize vectors to length 1, and explain why L2 distance and cosine similarity then rank neighbours the same way.
- [ ] Write a brute-force nearest-neighbour search with broadcasting.
- [ ] Show with real numbers why features must be standardized before measuring distance.
- [ ] Run an experiment that shows distances "concentrating" as the number of dimensions grows.

**Prerequisites.** [Day 5](day-05-vectors.md) (length, dot product, cosine similarity), [Day 9](day-09-broadcasting.md) (broadcasting, standardization, the pairwise-distance trick) and [Day 17](day-17-pandas-cleaning.md) (cleaning the Play Store dataset).

---

## The big idea

Machine learning asks two questions about vectors over and over:

- **How big is this vector?** How large are a model's weights? How large is an error?
- **How far apart are these two vectors?** Which saved photo looks most like this one? Which FAQ entry answers this question?

A **norm** answers the first question: it turns a vector into one non-negative number, its "size". A **distance** answers the second, and it is almost always just the norm of the difference, \( \|\mathbf{a} - \mathbf{b}\| \). On Day 5 you met one norm, Pythagoras' length. Today you'll see that it is one ruler among several, and that picking the ruler is a real design decision.

**Analogy.** You're in Manhattan and want to know how far the coffee shop is, three blocks east and four blocks north.

- A crow flies straight there: \( \sqrt{3^2 + 4^2} = 5 \) blocks. That's the **L2** (Euclidean) distance.
- A taxi has to follow the streets: \( 3 + 4 = 7 \) blocks, whichever route it takes. That's the **L1** (Manhattan) distance.
- A chess king, who may step diagonally at no extra cost, needs only \( \max(3, 4) = 4 \) moves. That's the **L∞** (Chebyshev) distance.

Same two points, three honest answers. Each one is the right answer to a different question.

## Three norms for one vector

Notation first. For a vector \( \mathbf{x} = [x_1, x_2, \dots, x_n] \):

- \( |x_i| \) is the **absolute value** of one element: drop its minus sign. \( |-4| = 4 \).
- \( \|\mathbf{x}\| \) (double bars) is a **norm** of the whole vector. A subscript says which one: \( \|\mathbf{x}\|_1 \), \( \|\mathbf{x}\|_2 \), \( \|\mathbf{x}\|_\infty \). With no subscript, people mean L2.
- \( \sum_{i=1}^{n} \) means "add up, for \( i \) from 1 to \( n \)", and \( \max_i \) means "the largest, over all \( i \)".

The three norms you'll meet constantly:

\[
\|\mathbf{x}\|_1 = \sum_{i=1}^{n} |x_i|
\qquad
\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2}
\qquad
\|\mathbf{x}\|_\infty = \max_i |x_i|
\]

In words: L1 adds up the sizes of the elements, L2 is Pythagoras' straight-line length, and L∞ ("L-infinity") keeps only the single biggest element.

**Worked example** with \( \mathbf{x} = [3, -4] \):

\[
\|\mathbf{x}\|_1 = |3| + |-4| = 7
\qquad
\|\mathbf{x}\|_2 = \sqrt{3^2 + (-4)^2} = \sqrt{25} = 5
\qquad
\|\mathbf{x}\|_\infty = \max(3, 4) = 4
\]

Notice the minus sign never matters: every norm measures size, not direction. And notice the order, which holds for every vector:

\[
\|\mathbf{x}\|_\infty \;\le\; \|\mathbf{x}\|_2 \;\le\; \|\mathbf{x}\|_1
\]

The biggest single element can't be more than the straight-line length, and the straight line can't be longer than walking along each axis in turn.

`np.linalg.norm` computes all three. The `ord=` argument picks the norm; the default is L2.

```python
import numpy as np

x = np.array([3., -4.])
np.linalg.norm(x, ord=1)        # 7.0
np.linalg.norm(x)               # 5.0    ord=2 is the default
np.linalg.norm(x, ord=np.inf)   # 4.0

# The same three, written out by hand
np.abs(x).sum(), np.sqrt((x ** 2).sum()), np.abs(x).max()
# (7.0, 5.0, 4.0)
```

(In Jupyter, NumPy 2 shows a single number as `np.float64(7.0)`; the comments show the plain value, which is what `print()` gives you.)

A 3-D example, to see L2 is still Pythagoras with one more term:

```python
y = np.array([1., -2., 2.])
[float(np.linalg.norm(y, ord=p)) for p in (1, 2, np.inf)]
# [5.0, 3.0, 2.0]      1+2+2,  sqrt(1+4+4),  max(1, 2, 2)
```

### Many vectors at once: `axis=`

Data usually arrives as a table, one vector per row. `axis=1` computes one norm per row, and `keepdims=True` keeps the result as a column so it can broadcast back against the table (the Day 9 rule):

```python
X = np.array([[3., -4.],
              [1.,  1.],
              [0.,  2.]])
np.linalg.norm(X, axis=1)                  # array([5.        , 1.41421356, 2.        ])
np.linalg.norm(X, ord=1, axis=1)           # array([7., 2., 2.])
np.linalg.norm(X, axis=1, keepdims=True).shape   # (3, 1)
```

!!! warning "Common mistake: `ord=` on a matrix means something else"
    Give `np.linalg.norm` a 2-D array **without** `axis=` and it computes a *matrix* norm, not one norm per row. `np.linalg.norm(X, ord=1)` returns the largest column sum of absolute values (here `7.0`, from the first column \( 3 + 1 + 0 \), which only coincidentally matches the first row), and `ord=2` returns something different again (the largest *singular value*, a relative of the PCA you'll meet on Day 25). With no `ord` and no `axis` you get the **Frobenius norm**: the L2 norm of all elements flattened together. If you mean "one norm per vector", always pass `axis=1`.

    ```python
    np.linalg.norm(X, ord=1)        # 7.0         max column sum: NOT per-row L1
    np.linalg.norm(X)               # 5.5677643628300215  sqrt(9+16+1+1+0+4) = sqrt(31)
    ```

### What makes something a norm

Any function that deserves the name "norm" follows three rules. You'll never have to prove them, but they explain why these three behave like "size":

1. **Zero only for zero:** \( \|\mathbf{x}\| \ge 0 \), and it equals 0 only for the all-zeros vector.
2. **Scaling:** \( \|c\,\mathbf{x}\| = |c|\,\|\mathbf{x}\| \). Double a vector and its size doubles.
3. **Triangle inequality:** \( \|\mathbf{a} + \mathbf{b}\| \le \|\mathbf{a}\| + \|\mathbf{b}\| \). A detour is never shorter than going direct.

L1, L2 and L∞ are all members of one family, the **p-norms** \( \|\mathbf{x}\|_p = \left(\sum_i |x_i|^p\right)^{1/p} \): raise each size to the power \( p \), add, take the \( p \)-th root. \( p = 1 \) and \( p = 2 \) give the first two; as \( p \) grows, the biggest element dominates more and more, and in the limit \( p \to \infty \) only the maximum is left. That's where the ∞ in the name comes from.

## Unit balls: what "length 1" looks like

A great way to *see* a norm is to draw every vector whose norm is exactly 1. For L2 that's the familiar circle. For the other two it isn't round at all:

<figure class="diagram" markdown>
![Animation: the set of 2-D vectors with norm 1 morphs from a diamond with corners on the axes (L1), into a circle (L2), into a square with corners on the diagonals (L-infinity) and back; a red arrow along the diagonal direction [1, 1] ends at [0.5, 0.5], then [0.71, 0.71], then [1, 1], and a side panel shows each norm's formula](../../assets/img/phase-1/day22-unit-balls.svg)
<figcaption>Animated: the same "length 1", drawn with three rulers. The red arrow points the same way each time, but reaches length 1 at a different place.</figcaption>
</figure>

- **L1: a diamond.** \( |x_1| + |x_2| = 1 \) is four straight edges: on the edge in the first quadrant, \( x_1 + x_2 = 1 \). Its corners sit **on the axes**, at \( [1, 0] \), \( [0, 1] \) and so on: points where one coordinate is exactly zero.
- **L2: a circle.** Every direction is treated the same. Rotating a vector never changes its L2 length, which is why it is the natural norm for geometry.
- **L∞: a square.** \( \max(|x_1|, |x_2|) = 1 \) means "neither coordinate is bigger than 1", and at least one equals 1.

The shapes are called **unit balls** (in 2-D they're flat, but the name comes from 3-D and beyond). They are nested: diamond inside circle inside square. That's the inequality \( \|\mathbf{x}\|_\infty \le \|\mathbf{x}\|_2 \le \|\mathbf{x}\|_1 \) drawn as a picture: the vector \( [0.5, 0.5] \) has L1 norm 1 but L2 norm only 0.71.

Remember the diamond's pointy corners on the axes; they come back at the end of the lesson, in regularization.

```python
d = np.array([0.5, 0.5])
np.linalg.norm(d, ord=1), np.linalg.norm(d), np.linalg.norm(d, ord=np.inf)
# (1.0, 0.7071067811865476, 0.5)
```

## From norms to distances

A **distance** between two vectors is the norm of their difference:

\[
d_1(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|_1
\qquad
d_2(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|_2
\qquad
d_\infty(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|_\infty
\]

They have names: **Manhattan** (or taxicab), **Euclidean** and **Chebyshev** distance. Take \( \mathbf{a} = [1, 1] \) and \( \mathbf{b} = [4, 5] \). The difference is \( \mathbf{b} - \mathbf{a} = [3, 4] \), and the three distances are the three norms of that one vector: 7, 5 and 4.

<figure class="diagram" markdown>
![Animation on a street grid from A at [1, 1] to B at [4, 5]: first a teal straight line of length 5 is drawn, then a blue staircase route of 7 one-block steps, then an orange chess-king route of 4 moves (one step up, then three diagonal steps); a panel lists Euclidean 5, Manhattan 7 and Chebyshev 4](../../assets/img/phase-1/day22-three-distances.svg)
<figcaption>Animated: the crow, the taxi and the chess king. All three are norms of the same difference vector [3, 4].</figcaption>
</figure>

```python
a = np.array([1., 1.])
b = np.array([4., 5.])
diff = b - a                                      # array([3., 4.])
[float(np.linalg.norm(diff, ord=p)) for p in (2, 1, np.inf)]
# [5.0, 7.0, 4.0]
```

### Cosine distance: direction only

Day 5's **cosine similarity** compared directions and ignored lengths:

\[
\cos\theta = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2 \, \|\mathbf{b}\|_2}
\]

It's a *similarity* (1 means "same direction"), so libraries turn it into a distance-like number by flipping it: the **cosine distance** is

\[
d_{\cos}(\mathbf{a}, \mathbf{b}) = 1 - \cos\theta
\]

It's 0 for vectors pointing the same way, 1 at right angles and 2 for opposite directions. Strictly speaking it isn't the norm of anything (it can break the triangle inequality), but every vector library offers it, under exactly this name.

When does that difference matter? Here's a case where Euclidean and cosine distance disagree completely. Count words over a vocabulary of `["battery", "crash", "camera"]`. A short review mentions "battery" twice and "crash" once. A long review says the same thing three times over. A third review is about the camera:

```python
def cosine_distance(a, b):
    return 1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

short = np.array([2., 1., 0.])   # "battery battery crash"
long_ = np.array([6., 3., 0.])   # the same complaint, three times as long
other = np.array([1., 0., 2.])   # mostly about the camera

np.linalg.norm(short - long_), np.linalg.norm(short - other)
# (4.47213595499958, 2.449489742783178)    Euclidean: "other" is closer!
cosine_distance(short, long_), cosine_distance(short, other)
# (1.1102230246251565e-16, 0.6000000000000001)   cosine: same topic is closer
```

That `1.1e-16` is floating-point rounding: the true cosine distance between `short` and `long_` is exactly 0, because they point the same way.

Euclidean distance is fooled by **length**: the long review is "far away" just because it has bigger counts. Cosine distance only asks whether the vectors point the same way, which is what "same topic" means here.

### Which distance when

| Distance | Formula | Use it when | You'll meet it in |
|---|---|---|---|
| Euclidean (L2) | \( \sqrt{\sum (a_i - b_i)^2} \) | coordinates are comparable and length matters: positions, standardized features | k-nearest neighbours and k-means (Phase 2), `IndexFlatL2` in FAISS |
| Manhattan (L1) | \( \sum \lvert a_i - b_i \rvert \) | you want outliers to count less: one huge difference isn't squared | robust error measures (the *mean absolute error* in Phase 2), grid movement |
| Chebyshev (L∞) | \( \max \lvert a_i - b_i \rvert \) | only the worst single coordinate matters | "every pixel within ±2", tolerance checks, adversarial-robustness limits |
| Cosine | \( 1 - \cos\theta \) | direction carries the meaning and length is noise | text and image embeddings, semantic search, RAG (Phase 4) |

Why "outliers count less" for L1: L2 squares each difference first, so one coordinate that's off by 10 contributes 100, while ten coordinates off by 1 contribute only 10 in total. L1 adds them as 10 and 10. L2 punishes one big miss much harder than many small ones.

## Normalizing vectors

To **normalize** a vector means to divide it by its own L2 norm. The result points the same way and has length exactly 1, a **unit vector**:

\[
\hat{\mathbf{x}} = \frac{\mathbf{x}}{\|\mathbf{x}\|_2}
\]

(The hat, \( \hat{\mathbf{x}} \), is the usual mark for "unit-length version of".) For \( [3, -4] \): divide by 5 to get \( [0.6, -0.8] \), and \( \sqrt{0.36 + 0.64} = 1 \).

<figure class="diagram" markdown>
![Animation: four arrows a = [3, 1], b = [1, 2.5], c = [-1.5, 0.5] and d = [0.3, -0.4] of different lengths shrink or stretch until each tip lies on the dashed unit circle, keeping its direction; a dashed purple segment between the tips of a and b shrinks from length 2.50 to 0.84, which a panel explains equals the square root of 2 minus 2 times cos theta](../../assets/img/phase-1/day22-normalize.svg)
<figcaption>Animated: normalizing keeps every direction and throws every length away. Afterwards, distance depends only on the angle.</figcaption>
</figure>

For a whole table, divide each row by its own norm. `keepdims=True` makes the norms a `(n, 1)` column that broadcasts across each row:

```python
V = np.array([[3., 1.],
              [1., 2.5],
              [-1.5, 0.5],
              [0.3, -0.4]])
norms = np.linalg.norm(V, axis=1, keepdims=True)   # shape (4, 1)
V_unit = V / norms
V_unit.round(3)
# [[ 0.949  0.316]
#  [ 0.371  0.928]
#  [-0.949  0.316]
#  [ 0.6   -0.8  ]]
np.linalg.norm(V_unit, axis=1)                     # array([1., 1., 1., 1.])
```

**The key fact about unit vectors.** Expand the squared distance (Day 9 did this to save memory):

\[
\|\mathbf{a} - \mathbf{b}\|_2^2 = \|\mathbf{a}\|_2^2 + \|\mathbf{b}\|_2^2 - 2\,\mathbf{a} \cdot \mathbf{b}
\]

If both vectors have length 1, the first two terms are 1 each, and \( \mathbf{a} \cdot \mathbf{b} \) *is* \( \cos\theta \). So

\[
\|\hat{\mathbf{a}} - \hat{\mathbf{b}}\|_2^2 = 2 - 2\cos\theta
\]

The distance is now a function of the angle alone. A smaller angle means a bigger \( \cos\theta \) and a smaller distance, so **after normalizing, sorting by Euclidean distance and sorting by cosine similarity give the same order**. That's why vector databases normalize embeddings once, when they're stored, and then use whichever operation is fastest, usually a plain dot product.

```python
a_hat, b_hat = V_unit[0], V_unit[1]
cos = a_hat @ b_hat                       # for unit vectors, the dot product IS the cosine
np.linalg.norm(V[0] - V[1])               # 2.5                  before: lengths mixed in
np.linalg.norm(a_hat - b_hat)             # 0.841495999436511    after normalizing
np.sqrt(2 - 2 * cos)                      # 0.8414959994365109   the formula agrees
```

!!! warning "Common mistake: normalizing a zero vector"
    An all-zeros vector has norm 0, and `v / 0` gives `nan` (with a `RuntimeWarning`) that then spreads through every calculation it touches. Empty text and blank images produce such vectors. Guard against it: `V / np.maximum(norms, 1e-12)` leaves zero rows as zeros instead of `nan`, or drop them before you normalize.

## Nearest-neighbour search by brute force

**Nearest-neighbour search** means: given a query vector, find the stored vectors closest to it. "Photos similar to this one", "FAQ answers for this question" and k-nearest-neighbour classification (Phase 2) are all this. The simplest correct method is **brute force**: compute the distance to every stored vector, then sort.

With one query, broadcasting does it in one line. A tiny database of 6 points in 2-D, so you can check it on graph paper:

```python
db = np.array([[1., 1.],
               [4., 5.],
               [2., 0.],
               [3., 5.],
               [5., 4.],
               [3., 3.]])
q = np.array([3., 2.])

dist = np.linalg.norm(db - q, axis=1)     # (6, 2) - (2,) -> (6, 2) -> norms (6,)
dist.round(3)                             # array([2.236, 3.162, 2.236, 3.   , 2.828, 1.   ])
order = np.argsort(dist, kind="stable")   # indices, nearest first
order                                     # array([5, 0, 2, 4, 3, 1])
db[order[:3]]
# [[3. 3.]
#  [1. 1.]
#  [2. 0.]]
```

`np.argsort` returns the *indices* that would sort the array, so `order[:k]` is "the k nearest". Point 5, `[3, 3]`, is one step above the query. Points 0 and 2 are tied at \( \sqrt{5} \approx 2.236 \). `kind="stable"` makes ties come out in index order; the default sort makes no such promise, so without it the order of tied points can differ between machines.

The metric matters. The same query with Manhattan distance picks different neighbours:

```python
dist_l1 = np.abs(db - q).sum(axis=1)
dist_l1                                   # array([3., 4., 3., 3., 4., 1.])
np.argsort(dist_l1, kind="stable")        # array([5, 0, 2, 3, 1, 4])
dist_inf = np.abs(db - q).max(axis=1)
dist_inf                                  # array([2., 3., 2., 3., 2., 1.])
```

Watch points 3, `[3, 5]`, and 4, `[5, 4]`. Their differences from the query are `[0, 3]` and `[2, 2]`. L2 calls point 4 closer (2.83 vs 3), and so does L∞ (2 vs 3). L1 says the opposite (4 vs 3): to a taxi, two blocks across plus two up is farther than three straight up. Change the ruler and you change who your neighbours are.

**Many queries at once.** Day 9's pairwise trick gives every query against every stored vector in one array: put the size-1 axis in a different place for each.

```python
def pairwise(Q, DB, metric="l2"):
    diff = Q[:, None, :] - DB[None, :, :]      # (m, 1, d) - (1, n, d) -> (m, n, d)
    if metric == "l2":
        return np.sqrt((diff ** 2).sum(axis=-1))
    if metric == "l1":
        return np.abs(diff).sum(axis=-1)
    if metric == "linf":
        return np.abs(diff).max(axis=-1)
    raise ValueError(f"unknown metric {metric!r}")

def knn(Q, DB, k=3, metric="l2"):
    D = pairwise(Q, DB, metric)                # (m, n)
    return np.argsort(D, axis=1, kind="stable")[:, :k]   # (m, k): k nearest per query

queries = np.array([[3., 2.], [0., 0.]])
knn(queries, db, k=2)
# [[5 0]
#  [0 2]]
```

**At embedding scale.** Real embeddings have hundreds of dimensions and databases have many thousands of rows. The `(m, n, d)` intermediate would be huge (Day 9's memory warning), so use normalized vectors and a matrix multiply instead: cosine similarity for all pairs is one `@`.

```python
rng = np.random.default_rng(22)
emb = rng.normal(size=(20_000, 64))                         # 20,000 stored "embeddings", 64-D
emb /= np.linalg.norm(emb, axis=1, keepdims=True)           # normalize once, at insert time

query = emb[123] + 0.05 * rng.normal(size=64)               # a slightly noisy copy of item 123
query /= np.linalg.norm(query)

sims = emb @ query                                          # (20000,) cosine similarities
top5 = np.argpartition(-sims, 5)[:5]                        # the 5 best, in no particular order
top5 = top5[np.argsort(-sims[top5])]                        # now sort just those 5
top5, sims[top5].round(3)
# (array([  123, 15147, 16562,  8729,  2477]), array([0.912, 0.468, 0.447, 0.421, 0.401]))
```

Item 123 comes first with similarity 0.91, far ahead of the runner-up at 0.47: the search found the original behind the noise.

`np.argpartition(a, k)` moves the `k` smallest values to the front without fully sorting the rest, which is much cheaper than `argsort` when you only need a handful out of millions. We pass `-sims` because we want the *largest* similarities.

Brute force costs one dot product per stored vector: \( n \times d \) multiply-adds per query, here \( 20{,}000 \times 64 \approx 1.3 \) million. That is fast on a laptop, and on a phone, for tens of thousands of items. For millions or billions of vectors, libraries like FAISS use *approximate* nearest-neighbour indexes that skip most of the comparisons (Phase 4).

## Why scale matters

Distances add up differences from every coordinate. If one feature is measured in big units and another in small ones, the big one silently decides everything.

Three apps, described by `[rating, size_mb]`:

```python
A = np.array([4.5, 20.])   # 4.5 stars, 20 MB
B = np.array([4.5, 26.])   # same rating, 6 MB bigger
C = np.array([2.0, 21.])   # 2.5 stars worse, about the same size
np.linalg.norm(A - B), np.linalg.norm(A - C)
# (6.0, 2.692582403567252)
```

By raw Euclidean distance, C (a 2-star app) is the "closest" to A, because 6 MB is a bigger number than 2.5 stars. That's an accident of units: measure size in kilobytes and the size gap would be a thousand times bigger again.

The fix is Day 9's **standardization**: for each feature (column), subtract its mean \( \mu \) ("mu") and divide by its standard deviation \( \sigma \) ("sigma"), so every feature is measured in "standard deviations from typical":

\[
z = \frac{x - \mu}{\sigma}
\]

### On the real dataset

Let's search the Play Store data for "apps like this one", using three numeric features: rating, size, and reviews. Reviews are wildly skewed (Day 19), so we use \( \log_{10} \) of reviews: "how many digits", 2581 reviews becomes about 3.41. The loading and cleaning steps are the short version from Day 17 ([download the CSV from the Datasets page](../../datasets.md) to work offline and replace `BASE + ...` with your local path). This lesson was tested with NumPy 2.5 and pandas 3.0.

```python
import pandas as pd

BASE = "https://manjulkadyan.github.io/ai-leaning/assets/data/"
apps = pd.read_csv(BASE + "play_store_apps.csv")

apps = apps.drop_duplicates().assign(
    size_mb=lambda d: d["size_mb"].replace("Varies with device", np.nan).astype(float),
    log_reviews=lambda d: np.log10(d["reviews"]),
)
apps = apps.dropna(subset=["rating", "size_mb"]).reset_index(drop=True)

cols = ["rating", "log_reviews", "size_mb"]
F = apps[cols].to_numpy()            # (442, 3): one row per app
print(F.shape)
print(F.std(axis=0).round(2))        # the spread of each feature
```

```text
(442, 3)
[ 0.46  0.94 37.64]
```

Size varies about 80 times more than rating, so size will dominate raw distances. Search for the apps nearest to the first app, Bright Quest X, first on the raw features, then on standardized ones:

```python
def nearest_apps(features, i, k=3):
    dist = np.linalg.norm(features - features[i], axis=1)
    idx = np.argsort(dist)[1:k + 1]          # [0] is the app itself, at distance 0
    return apps.loc[idx, ["app_name"] + cols].assign(dist=dist[idx]).round(2)

print(apps.loc[[0], ["app_name"] + cols].round(2).to_string())
print(nearest_apps(F, 0).to_string())
```

```text
         app_name  rating  log_reviews  size_mb
0  Bright Quest X     3.8         3.41     47.7
          app_name  rating  log_reviews  size_mb  dist
32   Nova Runner X     4.3         3.30     47.2  0.72
374   Bright Quest     3.6         3.04     48.6  0.99
62        Tiny Fit     4.0         2.42     47.5  1.03
```

Standardize with broadcasting (Day 9) and search again:

```python
mu, sigma = F.mean(axis=0), F.std(axis=0)
Z = (F - mu) / sigma                          # every column: mean 0, std 1
print(np.allclose(Z.mean(axis=0), 0), np.allclose(Z.std(axis=0), 1))
print(nearest_apps(Z, 0).to_string())
```

```text
True True
              app_name  rating  log_reviews  size_mb  dist
330  Pocket Coach Lite     3.9         3.29     39.3  0.34
437      Spark Cleaner     3.9         3.42     36.5  0.37
48         Lumen Tasks     3.8         3.65     58.6  0.39
```

(The `dist` column of the second table is in standard deviations, not in mixed units.)

On the raw features, all three "neighbours" match Bright Quest X's size to within a megabyte, while their ratings and review counts are all over the place: one has ten times fewer reviews. After standardizing, all three features get a fair say: the neighbours have ratings within 0.1 star, similar review counts, and sizes within about 11 MB. Same data, same formula, different answer, purely because of units.

!!! warning "Common mistake: computing \( \mu \) and \( \sigma \) on all the data"
    Here we standardized the whole table because we're only exploring. In a real model you compute `mu` and `sigma` on the **training set only** and reuse those numbers for new data, exactly the leakage rule from [Day 17](day-17-pandas-cleaning.md). A phone app does the same: the scaling constants are fixed at training time and shipped with the model.

!!! note "Standardization isn't the only option"
    *Min-max scaling* maps each feature to the range 0 to 1: \( (x - \min) / (\max - \min) \). It's common for pixel values (divide by 255). And sometimes you deliberately want one feature to count more; then multiply its column by a weight after standardizing. The point is to choose the scale on purpose, never by accident of units.

## The curse of dimensionality

Here's a strange fact that matters for every embedding you'll ever search. **In high dimensions, random points are almost all the same distance from each other.** The nearest neighbour and the farthest one end up nearly equally far away, so "nearest" carries less and less information.

Don't take that on trust; measure it. Scatter 500 random points uniformly in a cube with side 1 and \( d \) dimensions, pick one more random point as a query, and compare its nearest and farthest neighbours:

```python
rng = np.random.default_rng(0)
for d in [2, 10, 100, 1000, 10000]:
    pts = rng.random((500, d))          # 500 random points in the cube [0, 1]^d
    qd = rng.random(d)                  # one random query point
    dist = np.linalg.norm(pts - qd, axis=1)
    print(f"d={d:>5}  nearest {dist.min():6.2f}  farthest {dist.max():6.2f}  "
          f"farthest/nearest {dist.max() / dist.min():6.2f}")
```

```text
d=    2  nearest   0.03  farthest   1.27  farthest/nearest  41.73
d=   10  nearest   0.34  farthest   1.91  farthest/nearest   5.65
d=  100  nearest   3.39  farthest   4.60  farthest/nearest   1.36
d= 1000  nearest  12.32  farthest  13.63  farthest/nearest   1.11
d=10000  nearest  40.19  farthest  41.38  farthest/nearest   1.03
```

In 2-D the farthest point is dozens of times farther away than the nearest. In 10,000-D the ratio is barely above 1: every point is at almost the same distance. The figure shows the whole picture, with distances divided by their average so the dimensions can share one axis:

```python
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "savefig.facecolor": "white",
                     "axes.spines.top": False, "axes.spines.right": False})

rng = np.random.default_rng(0)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
colors = ["#3949ab", "#00897b", "#ef6c00", "#e53935"]
for d, c in zip([2, 10, 100, 1000], colors):
    dist = np.linalg.norm(rng.random((500, d)) - rng.random(d), axis=1)
    ax1.hist(dist / dist.mean(), bins=40, range=(0, 2), histtype="step", lw=2, color=c, label=f"d = {d}")
ax1.set_xlabel("distance ÷ average distance")
ax1.set_ylabel("number of points")
ax1.set_title("Distances bunch up as d grows")
ax1.legend(frameon=False)

dims = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
contrast = []
for d in dims:
    dist = np.linalg.norm(rng.random((500, d)) - rng.random(d), axis=1)
    contrast.append((dist.max() - dist.min()) / dist.min())
ax2.loglog(dims, contrast, marker="o", color="#3949ab")
ax2.set_xlabel("dimensions d (log scale)")
ax2.set_ylabel("(farthest − nearest) ÷ nearest")
ax2.set_title("Nearest and farthest become alike")
ax2.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("day22-curse.svg", format="svg", bbox_inches="tight")
plt.close(fig)
```

<figure class="diagram" markdown>
![Two charts. Left: histograms of distance divided by average distance for 500 random points; for d = 2 the histogram is spread from near 0 to about 1.8, for d = 10 it is narrower, for d = 100 it is a tight peak around 1, and for d = 1000 an even tighter spike at 1. Right: on log-log axes, the relative gap between the farthest and nearest point falls steadily from several hundred at d = 1 to about 0.03 at d = 10000](../../assets/img/phase-1/day22-curse.svg)
<figcaption>A real experiment: as the number of dimensions grows, the distances from a query to random points concentrate around their average.</figcaption>
</figure>

**Why it happens.** The squared L2 distance is a sum of \( d \) independent squared differences. Add up many independent random numbers and the total grows like \( d \), but its wobble grows only like \( \sqrt{d} \), so the wobble *relative to the total* shrinks like \( 1/\sqrt{d} \). Every distance ends up close to the same typical value.

**Why embeddings still work.** Real embeddings aren't random points filling a cube. A model trains them so that related items cluster together along a few meaningful directions, so neighbours really are much closer than strangers. But the effect is real at the edges: useless or noisy features add "random" dimensions that wash out the useful ones, and this is part of why too many features can make nearest-neighbour methods worse (Phase 2), and why dimensionality reduction (PCA, Day 25) helps.

## Preview: norms as penalties (regularization)

Norms don't only measure data; they also measure **models**. In Phase 2 you'll train models by minimizing an error, and you'll often add a penalty on the size of the weight vector \( \mathbf{w} \): minimize "error \( + \lambda\|\mathbf{w}\|_2^2 \)" (called **L2 regularization**, ridge, or *weight decay* in deep learning) or "error \( + \lambda\|\mathbf{w}\|_1 \)" (**L1 regularization**, the lasso), where \( \lambda \) ("lambda") is a knob for how much the penalty counts. Both say "prefer small weights", which keeps a model from memorizing noise. They differ in style: L2 shrinks every weight a little, while L1 tends to push many weights to *exactly zero*, effectively switching features off. The unit balls explain why: the L1 diamond has its corners on the axes, where some coordinates are zero, and the best solution tends to land on a corner; the round L2 ball has no corners to land on.

!!! android "Android bridge: on-device semantic search"
    MediaPipe's **Text Embedder** turns a string into an embedding vector on the phone, and everything in this lesson applies to it directly. Its options include `l2Normalize`, which returns every embedding already divided by its L2 norm (length 1), and the task provides a static `TextEmbedder.cosineSimilarity(a, b)` helper:

    ```kotlin
    val options = TextEmbedder.TextEmbedderOptions.builder()
        .setBaseOptions(BaseOptions.builder().setModelAssetPath("text_embedder.tflite").build())
        .setL2Normalize(true)                 // every embedding comes back with length 1
        .build()
    val embedder = TextEmbedder.createFromOptions(context, options)

    fun embed(text: String) = embedder.embed(text).embeddingResult().embeddings().first()

    val similarity = TextEmbedder.cosineSimilarity(
        embed("my battery drains overnight"),
        embed("why does my phone lose charge?"),
    )   // a Double between -1 and 1
    ```

    For an offline FAQ search, embed each FAQ entry once (say at build time or first launch) and keep the normalized vectors in one `FloatArray` of size `n * d`, row after row, like a NumPy array's buffer (Day 8). A query is then exactly today's brute force: one dot product per row and keep the top k, which takes milliseconds for a few thousand entries. Because the vectors are normalized, the dot product *is* the cosine similarity, and ranking by it gives the same order as ranking by Euclidean distance.

---

## Exercises

1. By hand, compute the L1, L2 and L∞ norms of \( [2, -3, 6] \). Check with `np.linalg.norm` and the `ord=` argument.
2. Generate 1,000 random vectors with `rng = np.random.default_rng(1)` and `rng.normal(size=(1000, 5))`. Verify that \( \|\mathbf{x}\|_\infty \le \|\mathbf{x}\|_2 \le \|\mathbf{x}\|_1 \) holds for every row. Then find the vector \( [1, 1, 1, 1, 1] \)'s three norms: which inequality becomes "as unequal as possible"?
3. Take \( \mathbf{a} = [2, 1] \) and \( \mathbf{b} = [1, 3] \). Compute their Euclidean, Manhattan, Chebyshev and cosine distances. Then show that scaling \( \mathbf{b} \) by 10 changes the first three but not the cosine distance.
4. Normalize \( \mathbf{a} \) and \( \mathbf{b} \) from Exercise 3 and verify numerically that \( \|\hat{\mathbf{a}} - \hat{\mathbf{b}}\|_2^2 = 2 - 2\cos\theta \).
5. Using `pairwise` and `knn` from the lesson, find each point's nearest *other* point in `db` (not itself) with the L1 metric. Hint: the distance from a point to itself is 0; set the diagonal of the distance matrix to `np.inf` with `np.fill_diagonal`.
6. Repeat the curse-of-dimensionality experiment with **Gaussian** points (`rng.normal(size=(500, d))`) instead of the uniform cube, and with **L1** distance instead of L2. Does the farthest/nearest ratio still shrink toward 1? Use `rng = np.random.default_rng(0)`.

??? success "Solutions"
    ```python
    # 1: L1 = 2+3+6 = 11, L2 = sqrt(4+9+36) = sqrt(49) = 7, Linf = 6
    v = np.array([2., -3., 6.])
    [float(np.linalg.norm(v, ord=p)) for p in (1, 2, np.inf)]
    # [11.0, 7.0, 6.0]

    # 2
    rng = np.random.default_rng(1)
    R = rng.normal(size=(1000, 5))
    l1 = np.linalg.norm(R, ord=1, axis=1)
    l2 = np.linalg.norm(R, axis=1)
    linf = np.linalg.norm(R, ord=np.inf, axis=1)
    bool(np.all(linf <= l2)), bool(np.all(l2 <= l1))    # (True, True)
    ones = np.ones(5)
    [float(np.linalg.norm(ones, ord=p)) for p in (np.inf, 2, 1)]
    # [1.0, 2.23606797749979, 5.0]
    ```
    For a vector with all elements equal, L1 is as large as it can be compared with L∞: \( n \) times bigger (5 vs 1). The other extreme is a vector with a single non-zero element, like \( [0, 0, 7, 0, 0] \), where all three norms are equal (7).

    ```python
    # 3
    a3, b3 = np.array([2., 1.]), np.array([1., 3.])
    def four(a, b):
        return [round(float(np.linalg.norm(a - b, ord=p)), 3) for p in (2, 1, np.inf)] + \
               [round(float(cosine_distance(a, b)), 3)]
    four(a3, b3)          # [2.236, 3.0, 2.0, 0.293]
    four(a3, 10 * b3)     # [30.083, 37.0, 29.0, 0.293]   only cosine is unchanged

    # 4
    a_h = a3 / np.linalg.norm(a3)
    b_h = b3 / np.linalg.norm(b3)
    cos3 = a_h @ b_h
    float(np.linalg.norm(a_h - b_h) ** 2), float(2 - 2 * cos3)
    # (0.585786437626905, 0.5857864376269051)    equal up to rounding: 2 - sqrt(2)
    ```

    ```python
    # 5
    D1 = pairwise(db, db, metric="l1")       # (6, 6), zeros on the diagonal
    np.fill_diagonal(D1, np.inf)             # a point may not pick itself
    D1.argmin(axis=1)                        # array([2, 3, 0, 1, 1, 3])
    D1.min(axis=1)                           # array([2., 1., 2., 1., 2., 2.])
    ```
    Read it as pairs: `[1, 1]` and `[2, 0]` are each other's nearest (2 blocks apart), and so are `[4, 5]` and `[3, 5]` (1 block). `[5, 4]` is nearest to `[4, 5]` (1 + 1 = 2), and the query-like point `[3, 3]` is nearest to `[3, 5]` (2 blocks straight up). Without `fill_diagonal`, every `argmin` would just return the point itself. If two points had tied, `argmin` would return the first index with the minimum value.

    ```python
    # 6
    rng = np.random.default_rng(0)
    for d in [2, 10, 100, 1000, 10000]:
        pts = rng.normal(size=(500, d))
        qd = rng.normal(size=d)
        dist = np.abs(pts - qd).sum(axis=1)        # L1 distance
        print(f"d={d:>5}  farthest/nearest {dist.max() / dist.min():6.2f}")
    # d=    2  farthest/nearest 106.44
    # d=   10  farthest/nearest   4.01
    # d=  100  farthest/nearest   1.50
    # d= 1000  farthest/nearest   1.14
    # d=10000  farthest/nearest   1.04
    ```
    Yes: the effect doesn't depend on the cube or on L2. Any distance that adds up many independent per-coordinate differences concentrates the same way.

## Resources

- [NumPy docs: `numpy.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html): the full table of what each `ord=` means for vectors and for matrices.
- [CS231n: Image classification, the nearest-neighbour classifier](https://cs231n.github.io/classification/): Stanford's notes compare L1 and L2 distance on real images, with a k-NN classifier built in NumPy.
- [Dive into Deep Learning: Linear algebra, section on norms](https://d2l.ai/chapter_preliminaries/linear-algebra.html): the same norms, with code, from an ML angle.
- [Mathematics for Machine Learning (free book), ch. 3 "Analytic Geometry"](https://mml-book.github.io/): norms, inner products, lengths, distances and angles, formally.
- [scikit-learn user guide: Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html): what brute force, k-d trees and ball trees do, and which metrics they accept (on your machine with scikit-learn installed).
- [Wikipedia: Curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality): the distance-concentration effect and its other faces.
- [MediaPipe Text Embedder for Android](https://ai.google.dev/edge/mediapipe/solutions/text/text_embedder/android): the options (including L2 normalization) and the cosine-similarity helper used in the Android bridge.

## Checklist

- [ ] I can compute L1, L2 and L∞ norms of a vector by hand and with `np.linalg.norm(x, ord=...)`.
- [ ] I always pass `axis=1` when I want one norm per row.
- [ ] I can draw the unit ball of each norm and explain the nesting diamond ⊂ circle ⊂ square.
- [ ] I can say when to use Euclidean, Manhattan, Chebyshev and cosine distance.
- [ ] I can normalize the rows of a matrix safely, and explain why L2 and cosine rankings agree afterwards.
- [ ] I wrote a brute-force nearest-neighbour search with broadcasting, and know when to switch to `@` and `argpartition`.
- [ ] I can show with an example why features must be standardized before computing distances.
- [ ] I ran the curse-of-dimensionality experiment and can explain why distances concentrate.

**Next:** [Day 23: Rank, span and linear independence](day-23-rank-span.md)
