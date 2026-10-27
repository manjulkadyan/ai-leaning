# Day 4c: Random Numbers, Saving Arrays and Statistics

**Phase 1 · Week 1 · ~140 minutes**

This lesson finishes the NumPy toolkit that [Day 4b](day-04b-numpy-toolkit.md) started. Day 4b chose, sorted, counted and cleaned values; today you make data (random numbers and reproducible train/test splits), keep it (saving and loading arrays), and summarise it (rounding, percentiles and histograms). It ends with one cleaning pipeline that uses tools from Days 4, 4b and 4c together.

## Objectives

- [ ] Generate reproducible random numbers with `np.random.default_rng(seed)`, explain what a seed is and what happens without one, and say why the legacy `np.random.seed` is discouraged.
- [ ] Draw numbers with `random`, `uniform`, `integers` (and its excluded upper bound) and `normal`, and sample with `choice`, with and without replacement and with probabilities `p`.
- [ ] Tell `shuffle` from `permutation`, and shuffle features and labels together.
- [ ] Build a reproducible train/test split, prove the parts don't overlap, and check their label counts.
- [ ] Save and load arrays with `np.save` / `np.load`, `np.savez` / `np.savez_compressed`, and `np.savetxt` / `np.loadtxt`, and explain what's inside a `.npy` file.
- [ ] Round with `np.round`, say why `np.round(2.5)` is `2.0`, and tell rounding from truncation, `floor` and `ceil`.
- [ ] Quantise an array to `int8` with a scale and a zero point, the way LiteRT does, using `round`, `clip` and `astype`.
- [ ] Summarise data with `np.median`, `np.percentile` and `np.quantile`, and compute a percentile by hand.
- [ ] Count values into bins with `np.histogram`, and read its `counts` and `edges`.
- [ ] Test membership element by element with `np.isin`.

**Prerequisites.**

- [Day 2](day-02-python-types.md): banker's rounding in `round`, `%` (remainder), and `zip(..., strict=True)`.
- [Day 2a](day-02a-strings.md): f-string format specs such as `:.3f` and `:02d`, and repeating a string with `*`.
- [Day 2b](day-02b-collections.md): sets and intersection, `sorted`, and list comprehensions.
- [Day 3b](day-03b-errors-files-json.md): `try` / `except` and `raise ValueError(...)`.
- [Day 3c](day-03c-files-json.md): `with`, `tempfile.mkdtemp()`, and `pathlib.Path` (`/`, `read_text`, `read_bytes`, `iterdir`, `stat().st_size`).
- [Day 3d](day-03d-iterators-generators.md): generator functions with `yield`, and the mini `DataLoader` that shuffled with `random.Random(seed)`.
- [Day 4](day-04-numpy-arrays.md): `dtype` and `astype` (which truncates), `reshape`, fancy indexing (which makes a copy), reductions, and the preview of `np.random.default_rng`.
- [Day 4b](day-04b-numpy-toolkit.md): `np.where`, `np.clip`, `np.sort` and `argsort` (with `kind="stable"`), `np.unique(..., return_counts=True)`, `np.searchsorted`, `np.cumsum` and `np.diff`, `np.isnan` and the `nan*` functions, `np.count_nonzero`, and `np.allclose` / `np.array_equal`.

!!! info "How to read the code on this page"
    Every Python block has a **▶ Run** button, and the blocks share their variables from top to bottom like notebook cells, so run them in order, starting with the first. A bare expression on its own line shows its value, exactly as the `# comment` under it says. Lines that fail on purpose are wrapped in `try` / `except`, and the comment shows the real error message.

    The outputs on this page come from 64-bit Python. The browser's Python is 32-bit, so NumPy's default integer there is `int32`: where this page shows `np.int64(2)`, ▶ Run shows `np.int32(2)`. The reverse happens too: arrays from `rng.integers` are always `int64`, so in the browser their repr gains `, dtype=int64)` at the end, where this page shows none. The values are the same everywhere; only the type label differs. Random numbers from a seeded generator are identical in the browser and on your computer, and `print(...)` output looks the same everywhere.

---

## The big idea

Day 4b's tools worked on data you already had. Three jobs remain before that data can train a model. You need data to practise on, and randomness for weights, shuffling and test sets, **but reproducibly**, so that a better score means better code, not a luckier draw. You need to **keep** results, so that an hour of preprocessing doesn't run again every time you restart a notebook. And you need to **describe** data in a few numbers: what's typical, what's extreme, what the shape of the distribution is.

**Analogy.** A seeded random generator is a deck of cards shuffled once by a machine that always shuffles the same way for the same seed. Deal from it and the cards look random, but anyone with the same seed deals exactly the same hand. A `.npy` file is a photocopy of the array's memory with a label on the front saying what it is. A percentile is a ruler you lay along the sorted data: "90% of requests finished before this mark".

The lesson has three parts, then a pipeline:

1. Random numbers: generators and seeds, the sampling methods, shuffling, and a train/test split.
2. Saving and loading: `.npy`, `.npz` and text files.
3. Summaries: `round` (and int8 quantisation), percentiles, histograms and `isin`.

---

**Part 1: Random numbers**

## Generators and seeds: `np.random.default_rng`

Day 4 previewed this generator to make test data. Here is how it works, and how it goes wrong.

**What it is.** Computers generate "random" numbers with a **pseudo-random number generator** (PRNG): a small algorithm with some internal state that produces a long sequence of numbers that look random. The **seed** is the number that sets the starting state. The same seed always gives the same sequence. `np.random.default_rng(seed)` creates a `Generator` object; its methods (`random`, `integers`, `normal`, `choice`, ...) each draw numbers and move the state forward.

**Why it exists.** ML uses randomness everywhere: initial weights, the order of training samples, which samples go in the test set, dropout, data augmentation. **Reproducibility**, getting exactly the same result on the next run, is how you know that a change in accuracy came from your code change and not from luck. A fixed seed gives you that.

**Kotlin comparison.** It's Kotlin's `kotlin.random.Random(seed)`:

```kotlin
val rng = Random(42)          // an object with its own state
rng.nextDouble()              // advances the state
val rng2 = Random(42)         // same seed, same sequence
```

Kotlin also has `Random.Default`, one shared generator for the whole program. NumPy has the same split: the `Generator` objects you create, and an older global generator (below).

```python
import numpy as np

rng = np.random.default_rng(42)
rng.random(3)
# array([0.77395605, 0.43887844, 0.85859792])
rng.random(3)
# array([0.69736803, 0.09417735, 0.97562235])
```

```python
rng_a = np.random.default_rng(42)
rng_b = np.random.default_rng(42)
rng_a.random(3)
# array([0.77395605, 0.43887844, 0.85859792])
rng_b.random(3)
# array([0.77395605, 0.43887844, 0.85859792])
```

`rng.random(n)` draws `n` floats from \([0, 1)\): from 0 included up to 1 excluded. These exact values are the same on every computer and in the browser, because the algorithm is fully specified. One caveat: NumPy's policy allows a `Generator`'s output for a given seed to change between NumPy **versions** (when a method gets a better algorithm). The same seed and the same NumPy version always give the same numbers; to freeze something across versions, such as a test split, save the result (Part 2), not only the seed.

**Without a seed.** `np.random.default_rng()` with no argument takes a fresh seed from the operating system's entropy source, so every run is different:

```python
a = np.random.default_rng().random(3)
b = np.random.default_rng().random(3)
np.array_equal(a, b)
# False
```

Seed experiments, tests and data splits, so they can be repeated. Leave the seed out only when you want different results every time, such as a demo that should never repeat itself.

**How it works underneath.** The two calls on one `rng` give different numbers because each call moves the state forward. Two generators with the same seed start in the same state, so they produce the same sequence, step for step. The algorithm inside is called PCG64:

```python
type(rng).__name__, type(rng.bit_generator).__name__
# ('Generator', 'PCG64')
```

`bit_generator` is the part that produces raw random bits; the `Generator` turns those bits into floats, integers, normal samples and so on.

**The legacy API.** Older code (and many tutorials) write `np.random.seed(0)` and then `np.random.rand(3)`, `np.random.randint(...)` or `np.random.shuffle(...)`. Those functions all share **one hidden global generator**. NumPy's documentation recommends `default_rng` for new code because:

- any library you import can call `np.random.seed` or draw from the global generator, which silently changes your sequence;
- with one global state, two parts of a program can't have independent, reproducible streams;
- a `Generator` is an ordinary object: you pass it to the functions that need randomness, so it's visible in the code which randomness each function uses. (Its algorithm is also faster and statistically better than the legacy one.)

The legacy functions still work, so you'll read them, but write `rng = np.random.default_rng(seed)` and pass `rng` around.

**How it goes wrong: re-creating the generator inside a function.** It looks tidy to seed inside the function that needs randomness:

```python
def noisy(values):
    rng = np.random.default_rng(0)        # bug: a fresh generator on every call
    return values + rng.normal(size=len(values))

noisy(np.zeros(2))
# array([ 0.12573022, -0.13210486])
noisy(np.zeros(2))
# array([ 0.12573022, -0.13210486])
```

Every call creates a new generator in the same starting state, so every call adds the **same** "noise". In data augmentation, that means every image gets the identical "random" crop. **Recognise it** by "random" results that repeat between calls. **Fix it** by creating one generator once, at the top of the program, and passing it in:

```python
def noisy(values, rng):
    return values + rng.normal(size=len(values))

rng = np.random.default_rng(0)
noisy(np.zeros(2), rng)
# array([ 0.12573022, -0.13210486])
noisy(np.zeros(2), rng)
# array([0.64042265, 0.10490012])
```

The whole program is still reproducible (the same seed gives the same two results next run), but each call gets fresh numbers. `rng.normal` is explained in the next section.

## Drawing numbers: `random`, `uniform`, `integers`, `normal`

**What they are.** The `Generator` methods you'll use most. Each takes a `size=`, which can be one number or a shape tuple (Day 4), and returns an array of that shape.

- `rng.random(size)`: floats from \([0, 1)\), every value equally likely (a **uniform** distribution).
- `rng.uniform(low, high, size)`: uniform floats from \([\text{low}, \text{high})\).
- `rng.integers(low, high, size)`: integers from `low` up to **but not including** `high`, like `range`. `endpoint=True` includes `high`.
- `rng.normal(loc, scale, size)`: floats from a **normal** (bell-curve, or Gaussian) distribution, centred on `loc` (the mean) with spread `scale` (the standard deviation). Values near `loc` are common; about 68% land within one `scale` of it. Month 2 covers the normal distribution properly.

**Why.** `uniform` and `normal` are how weights are initialised (small random values), `integers` picks random indices and simulates dice or class labels, and `normal` makes realistic fake data and noise.

**Kotlin comparison.** `rng.nextDouble()`, `rng.nextDouble(from, until)` and `rng.nextInt(from, until)`, where `until` is also excluded. Kotlin's `Random` has no normal distribution (Java's `java.util.Random.nextGaussian()` does). The big difference: NumPy draws a whole array in one call, not one number at a time.

`rng.integers`, and `rng.choice` when you give it an integer, return `int64` by default on every platform, even where NumPy's default integer is `int32`. That's why, in the browser, their arrays show `, dtype=int64)` (the box at the top of the page).

**How they work underneath.** All four start from the same stream of random bits from the bit generator (PCG64). `random()` turns 53 of those bits into a float in \([0, 1)\). `uniform(low, high)` is that float stretched and shifted, `low + (high - low) * random()`, so with the same seed the two agree exactly:

```python
a = np.random.default_rng(7).random(4)
b = np.random.default_rng(7).uniform(-1, 1, size=4)
np.array_equal(-1 + 2 * a, b)
# True
```

`integers` maps the bits onto the integer range in a way that keeps every value exactly equally likely (it doesn't use `random() * n`, which would favour some values very slightly). `normal` turns uniform bits into bell-curve values with an algorithm called the ziggurat method; you don't need its details, only the picture: one bit stream, many shapes of distribution on top of it.

```python
rng = np.random.default_rng(7)
rng.random((2, 3))
# array([[0.62509547, 0.8972138 , 0.77568569],
#        [0.22520719, 0.30016628, 0.87355345]])
rng.uniform(-1, 1, size=4)
# array([-0.98946939,  0.64245684,  0.59413886, -0.06413009])
```

```python
rng = np.random.default_rng(7)
rng.integers(0, 10, size=8)
# array([9, 6, 6, 8, 5, 7, 8, 2])
rng.integers(1, 6, size=10)
# array([1, 2, 2, 5, 5, 1, 3, 5, 1, 4])
rng.integers(1, 6, size=10, endpoint=True)
# array([1, 3, 5, 2, 3, 2, 5, 2, 6, 3])
```

**Choosing the integer type.** `dtype=` picks the integer type of the result, which is how you make a fake `uint8` image (Day 8 does this). `high` must fit in that type, or NumPy refuses:

```python
rng = np.random.default_rng(7)
rng.integers(0, 256, size=(2, 3), dtype=np.uint8)
# array([[139,  74, 229],
#        [241, 169,  65]], dtype=uint8)
try:
    rng.integers(0, 300, dtype=np.uint8)
except ValueError as e:
    print(e)
# high is out of bounds for uint8
```

`high=256` is allowed for `uint8` because it's excluded: the largest value drawn is 255.

!!! warning "Common mistake: a dice roll that never rolls 6"
    `rng.integers(1, 6)` gives 1 to **5**: the upper bound is excluded, like `range(1, 6)` and Kotlin's `nextInt(1, 6)`. The middle line above has no 6 in it, and never will. There's no error; the simulation is wrong. **Fix:** `rng.integers(1, 7)`, or `endpoint=True` as in the last line, which does contain a 6. The same off-by-one hits class labels: for 10 classes numbered 0 to 9, use `rng.integers(0, 10)`.

```python
rng = np.random.default_rng(7)
heights = rng.normal(loc=170, scale=10, size=5)
heights
# array([170.01230153, 172.98745538, 167.25862145, 161.09408161,
#        165.45329215])
big = rng.normal(loc=170, scale=10, size=100_000)
big.mean(), big.std()
# (np.float64(169.9868652970921), np.float64(9.982915529217578))
```

Five samples scatter around 170. With 100,000 samples (`100_000` is how Python lets you write 100000 readably), the sample mean and standard deviation land very close to the `loc` and `scale` you asked for, which is a quick way to check you passed the arguments in the right order.

**dtype.** The generator methods return `float64` (or `int64`). Neural networks usually train in `float32`, so convert, as on Day 4:

```python
rng = np.random.default_rng(7)
w = rng.normal(0, 0.01, size=(3, 2)).astype(np.float32)
w
# array([[ 1.2301533e-05,  2.9874553e-03],
#        [-2.7413785e-03, -8.9059183e-03],
#        [-4.5467080e-03, -9.9164657e-03]], dtype=float32)
w.dtype
# dtype('float32')
```

That's a small layer's weights initialised the classic way: normal, mean 0, standard deviation 0.01. (Phase 3 explains better scales.)

## Sampling from a set: `rng.choice`

**What it is.** `rng.choice(a, size, replace=True, p=None)` picks `size` elements from `a`. `a` can be an array, a list, or an integer `n`, meaning `np.arange(n)`. With `replace=True` (the default), the same element can be picked again, like rolling a die; with `replace=False`, each element is picked at most once, like dealing cards. `p` is a list of probabilities, one per element, that must add up to 1.

**Why.** Picking a random mini-batch, sampling a subset of a large dataset to look at, simulating imbalanced labels, and, with `p`, sampling the next word from a language model's predicted probabilities (Phase 5).

**Kotlin comparison.** `list.random(rng)` picks one element; `list.shuffled(rng).take(3)` is sampling without replacement. Kotlin has no built-in weighted choice.

```python
rng = np.random.default_rng(3)
colors = np.array(["red", "green", "blue"])
rng.choice(colors, size=6)
# array(['blue', 'red', 'red', 'red', 'red', 'blue'], dtype='<U5')
rng.choice(colors, size=3, replace=False)
# array(['green', 'blue', 'red'], dtype='<U5')
rng.choice(5, size=3, replace=False)
# array([3, 0, 1])           in the browser: array([3, 0, 1], dtype=int64)
```

The first line repeats colours (with replacement); the second uses each exactly once; the third picks 3 different indices from 0 to 4.

With `p`, about 90% of 10,000 draws are `"ok"`. `np.unique` from Day 4b counts them:

```python
rng = np.random.default_rng(3)
draws = rng.choice(["ok", "spam"], size=10_000, p=[0.9, 0.1])
np.unique(draws, return_counts=True)
# (array(['ok', 'spam'], dtype='<U4'), array([8996, 1004]))
```

**How `p` works underneath.** It's Day 4b's `cumsum` and `searchsorted` together. The running total of the probabilities splits the interval from 0 to 1 into pieces, one per element, each as long as that element's probability. A uniform draw from `rng.random()` lands in one piece, and `searchsorted` finds which:

```python
p = np.array([0.5, 0.3, 0.2])
cdf = np.cumsum(p)
cdf
# array([0.5, 0.8, 1. ])
u = np.random.default_rng(3).random(5)
u
# array([0.08564917, 0.23681051, 0.80127447, 0.58216204, 0.09412864])
np.searchsorted(cdf, u, side="right")
# array([0, 0, 2, 1, 0])
np.random.default_rng(3).choice(3, size=5, p=p)
# array([0, 0, 2, 1, 0])     in the browser: array([0, 0, 2, 1, 0], dtype=int64)
```

<figure class="diagram" markdown>
![Animation: a bar from 0 to 1 is split into three pieces, element 0 from 0 to 0.5, element 1 from 0.5 to 0.8 and element 2 from 0.8 to 1. Five red markers for the uniform draws 0.0856, 0.2368, 0.8013, 0.5822 and 0.0941 drop onto the bar one by one, and the picked elements 0, 0, 2, 1, 0 appear below](../../assets/img/phase-1/day04c-choice-cdf.svg)
<figcaption>Animated: each uniform draw lands in one piece of the cumulative bar; the piece decides the element.</figcaption>
</figure>

Element 0 owns 0 to 0.5, element 1 owns 0.5 to 0.8, element 2 owns 0.8 to 1. 0.0856 falls in the first piece, 0.8013 in the third, 0.5822 in the second. With the same seed, `choice` gives exactly the same values, because this is what it does inside. (In the browser the two lines print differently only because `choice` always returns `int64` while `searchsorted` returns the platform's default integer.) (`cdf` is short for "cumulative distribution function", Month 2's name for this running total.)

**How it goes wrong.** Two real errors. Sampling more items than exist without replacement, and probabilities that don't add up to 1:

```python
try:
    rng.choice(colors, size=5, replace=False)
except ValueError as e:
    print(e)
# Cannot take a larger sample than population when replace is False
try:
    rng.choice(colors, p=[0.5, 0.3, 0.1])
except ValueError as e:
    print(e)
# Probabilities do not sum to 1. See Notes section of docstring for more information.
```

The second usually comes from raw scores that weren't normalised. **Fix it** by dividing by the total, `p = scores / scores.sum()`.

## `shuffle` vs `permutation`

**What they are.** Both put elements in a random order.

- `rng.shuffle(a)` shuffles the array **in place** and returns `None`.
- `rng.permutation(a)` returns a **shuffled copy** and leaves `a` alone. `rng.permutation(n)` with an integer returns `0` to `n - 1` in random order.

On a 2-D array, both shuffle **whole rows** (along axis 0), keeping each row intact.

**Underneath.** Both use the Fisher–Yates shuffle: for each position from the last one down to the second, pick a random position at or before it and swap the two. Every order is equally likely, and it takes one pass. Kotlin's `shuffle` uses the same algorithm.

**Why.** Shuffling the training data every epoch, so the model doesn't see the samples in the same order each time (Day 3d's `DataLoader`), and building train/test splits (next section).

**Kotlin comparison.** The same pair as `sort` and `sorted`: Kotlin's `mutableList.shuffle(rng)` (in place, returns `Unit`) and `list.shuffled(rng)` (new list).

```python
rng = np.random.default_rng(1)
deck = np.arange(6)
rng.permutation(deck)
# array([4, 0, 2, 1, 5, 3])
deck
# array([0, 1, 2, 3, 4, 5])
rng.shuffle(deck)
deck
# array([2, 3, 5, 4, 0, 1])
rng.permutation(5)
# array([1, 4, 0, 2, 3])
```

Rows stay together:

```python
rng = np.random.default_rng(2)
table = np.array([[1, 10],
                  [2, 20],
                  [3, 30]])
rng.permutation(table)
# array([[ 3, 30],
#        [ 1, 10],
#        [ 2, 20]])
```

**How it goes wrong (1): assigning the result of `shuffle`.**

```python
deck = np.arange(6)
deck = rng.shuffle(deck)      # bug: shuffle returns None
print(deck)
# None
```

Same mistake as `x = x.sort()`. **Fix:** `rng.shuffle(deck)` on its own line, or `deck = rng.permutation(deck)`.

**How it goes wrong (2): shuffling features and labels separately.** This is the most damaging bug in this lesson, because nothing fails. `X` holds one sample per row and `y` the label of each sample; row `i` of `X` belongs with `y[i]`. Shuffling each with its own call gives each its own random order:

```python
X = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
y = np.array([0, 0, 1, 1, 1])
rng = np.random.default_rng(5)
rng.shuffle(X)
rng.shuffle(y)
np.column_stack([X, y])
# array([[0.5, 0. ],
#        [0.4, 1. ],
#        [0.2, 1. ],
#        [0.3, 1. ],
#        [0.1, 0. ]])
```

`np.column_stack` puts 1-D arrays (or 2-D columns) side by side as columns, so you can see each sample next to its label. Originally 0.5 had label 1 and 0.2 had label 0; now 0.5 has label 0 and 0.2 has label 1. The model trains on wrong labels and gets worse, which looks like "this model doesn't work". **Recognise it** by printing a few `(X[i], y[i])` pairs before and after shuffling. **Fix it** with **one** permutation of the row positions, applied to both:

```python
X = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
y = np.array([0, 0, 1, 1, 1])
rng = np.random.default_rng(5)
idx = rng.permutation(len(X))
idx
# array([4, 3, 1, 2, 0])
np.column_stack([X[idx], y[idx]])
# array([[0.5, 1. ],
#        [0.4, 1. ],
#        [0.2, 0. ],
#        [0.3, 1. ],
#        [0.1, 0. ]])
```

Every sample still has its own label. Indexing with `idx` is fancy indexing, so `X[idx]` is a new, reordered copy (Day 4).

## A reproducible train/test split

**What it is.** Before training, you set some of your data aside as a **test set** that the model never sees during training. You measure the final accuracy on it, so the score tells you how the model does on new data rather than how well it memorised. A **train/test split** divides the samples at random into the two groups: commonly 80/20 or 70/30.

**Why at random, and why reproducible.** At random, because data files are often sorted (by date, by class, by user), and the last 20% of a sorted file is not like the first 80%. Reproducible (a fixed seed), because if the split changes every run, you can't tell whether a better score came from your change or from an easier test set.

**How it works.** Shuffle the **positions** once with `permutation`, cut that list in two, and index `X` and `y` with each part. One cut of one permutation means the two groups can't overlap and together contain every sample.

<figure class="diagram" markdown>
![Animation: ten samples numbered 0 to 9, each with its label 0 or 1 beneath it, slide into the shuffled order 4, 6, 2, 7, 3, 5, 9, 0, 8, 1, each label moving with its sample. Then the first three turn orange and are bracketed as test_idx = idx[:3], and the other seven turn teal as train_idx = idx[3:]](../../assets/img/phase-1/day04c-split.svg)
<figcaption>Animated: one permutation moves every sample with its label; one cut makes two groups that don't overlap.</figcaption>
</figure>

```python
rng = np.random.default_rng(0)
n = 10
X = np.arange(n * 2).reshape(n, 2)      # 10 samples, 2 features
y = np.arange(n) % 2                    # labels 0, 1, 0, 1, ...

idx = rng.permutation(n)
n_test = round(n * 0.3)
test_idx, train_idx = idx[:n_test], idx[n_test:]
idx
# array([4, 6, 2, 7, 3, 5, 9, 0, 8, 1])
test_idx, train_idx
# (array([4, 6, 2]), array([7, 3, 5, 9, 0, 8, 1]))
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]
X_train.shape, X_test.shape
# ((7, 2), (3, 2))
```

Line by line: `X` is a fake dataset of 10 rows; `y` gives alternating labels (`%` is the remainder, Day 2). `idx` is the 10 row positions in random order. `n_test = round(10 * 0.3)` is 3 (why `round` and not `int` is explained after the function below). The first 3 shuffled positions become the test set, the other 7 the training set. Then the same positions select rows of both `X` and `y`.

It's worth proving the two properties. `np.intersect1d(a, b)` returns the values present in both arrays, sorted (like a set intersection, Day 2b), and `np.concatenate` joins arrays end to end:

```python
len(np.intersect1d(train_idx, test_idx))
# 0
np.array_equal(np.sort(np.concatenate([train_idx, test_idx])), np.arange(n))
# True
```

No position is in both (the intersection is empty), and together they're exactly 0 to 9. As a function you can reuse:

```python
def train_test_split(X, y, test_frac=0.2, seed=0):
    """Shuffle rows of X and y together, then cut off the first test_frac."""
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = round(len(X) * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_frac=0.3, seed=0)
y_te
# array([0, 0, 0])
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X, y, test_frac=0.3, seed=0)
np.array_equal(X_te, X_te2)
# True
```

Here creating the generator inside the function is right, unlike `noisy` above: the whole point is that the same seed gives the same split every time. The length check catches the case where `X` and `y` came from different files. `round(...)` with one argument returns an `int` (Day 2), so `n_test` can slice directly. Why `round` and not `int`? `int` truncates, and a fraction times a count is often a hair below a whole number:

```python
100 * 0.29
# 28.999999999999996
int(100 * 0.29), round(100 * 0.29)
# (28, 29)
```

With `int`, a "29% test set" of 100 samples would get 28. That's why every split on this page, including the solutions, uses `round`.

Look at `y_te`: all three test labels are 0, by chance, although the data is half 0s and half 1s. With small datasets or rare classes, a purely random split can leave a class out of the test set. The fix is a **stratified** split, which splits each class separately; scikit-learn (Month 4) does this with `train_test_split(..., stratify=y)`. Always check the label counts of each part with `np.unique(y_te, return_counts=True)`.

!!! android "Android bridge: same seed, same numbers, on one platform"
    `kotlin.random.Random(42)` and `np.random.default_rng(42)` use different algorithms, so they give **different** sequences from the same seed. If your app must reproduce a split, an augmentation or a sample made in Python, don't re-create the randomness in Kotlin: compute the indices once in Python, save them (Part 2), and ship the saved indices. That also protects you from a NumPy upgrade changing the stream for a seed. The same goes for preprocessing in general: the app has to repeat exactly what training did, and data is easier to keep identical than two implementations.

---

**Part 2: Saving and loading arrays**

All the files in this part go into a fresh temporary folder, as on Day 3c, so the examples don't depend on what's in your current folder. In the browser, Python's files live in an in-memory filesystem that disappears when you close the tab; on your computer, the temporary folder is real and stays until you delete it (`shutil.rmtree(work)` removes it and everything in it, as on Day 3c).

## `.npy` files: `np.save` and `np.load`

**What it is.** `np.save(path, arr)` writes one array to a binary `.npy` file: a short header that records the dtype and shape, followed by the raw bytes of the data. `np.load(path)` reads it back as an identical array.

**Why.** Recomputing preprocessed data, embeddings or model weights takes time; saving them takes milliseconds. `.npy` round-trips exactly: same dtype, same shape, same bits, which text formats can't promise.

**Kotlin comparison.** It's like writing a `FloatArray` into a `ByteBuffer` and then to a file, but with the dimensions and type written in front so the reader doesn't need to know them in advance. Compared with JSON (Day 3c), it's smaller, faster, and keeps the exact float values, but only NumPy (and libraries that know the format) can read it.

```python
import tempfile
from pathlib import Path

work = Path(tempfile.mkdtemp())
weights = np.arange(6, dtype=np.float32).reshape(2, 3)
np.save(work / "weights.npy", weights)
loaded = np.load(work / "weights.npy")
loaded
# array([[0., 1., 2.],
#        [3., 4., 5.]], dtype=float32)
loaded.dtype, loaded.shape
# (dtype('float32'), (2, 3))
np.array_equal(weights, loaded)
# True
```

**How it works underneath.** Look at the file itself:

```python
path = work / "weights.npy"
path.stat().st_size
# 152
weights.nbytes
# 24
path.read_bytes()[:10]
# b'\x93NUMPY\x01\x00v\x00'
path.read_bytes()[10:80]
# b"{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }           "
int.from_bytes(path.read_bytes()[8:10], "little")
# 118
```

The file is 152 bytes: a 128-byte header, then the 24 bytes of data (6 values × 4 bytes each). Byte by byte:

- `\x93NUMPY` (6 bytes) is the "magic string" that identifies the format.
- `\x01\x00` is the format version, 1.0.
- `v\x00` is the length of the text that follows, as a 2-byte **little-endian** integer. Little-endian means the least significant byte comes first; it's the byte order of x86 and ARM CPUs, and Kotlin's `ByteOrder.LITTLE_ENDIAN`. The first byte is 0x76 = 118, which Python displays as the letter `v` because bytes objects show printable bytes as ASCII characters; the second byte is 0. So the text is 118 bytes, and 10 + 118 = 128.
- The text is a Python dict: `'descr': '<f4'` means little-endian (`<`) 4-byte float (`f4`), which is `float32`; `'shape': (2, 3)` is the shape; `fortran_order` says whether the data is stored row by row (`False`) or column by column.
- Spaces (and a final newline) pad the header to a multiple of 64 bytes, so the data starts at an aligned position. For small arrays that makes the header 128 bytes; an array with a very long shape needs more text, and the header grows to 192 bytes or more.

After the header come the raw bytes, exactly as they sit in memory. That's why loading is fast: there's nothing to parse.

**How it goes wrong (1): the `.npy` extension.** `np.save` adds `.npy` to the name if it doesn't end with it; `np.load` does not:

```python
np.save(work / "bias", np.zeros(3))
sorted(p.name for p in work.iterdir())
# ['bias.npy', 'weights.npy']
try:
    np.load(work / "bias")
except FileNotFoundError as e:
    print("FileNotFoundError:", e.strerror)
# FileNotFoundError: No such file or directory
np.load(work / "bias.npy")
# array([0., 0., 0.])
```

`sorted(p.name for p in work.iterdir())` lists the folder's file names (Day 3c). The file is called `bias.npy`, and `bias` doesn't exist. **Fix:** always write the extension yourself, in both calls.

**How it goes wrong (2): object arrays and `allow_pickle`.** An array of Python objects (`dtype=object`), such as a list of lists of different lengths, can't be stored as raw numbers. NumPy saves it with **pickle**, Python's general object serialisation. Loading a pickle can run arbitrary code, so `np.load` refuses by default:

```python
ragged = np.array([[1, 2], [3]], dtype=object)
np.save(work / "ragged.npy", ragged)
try:
    np.load(work / "ragged.npy")
except ValueError as e:
    print(e)
# Object arrays cannot be loaded when allow_pickle=False
```

**Fix:** pass `allow_pickle=True` **only** for files you created yourself. Never for a file downloaded from somewhere you don't trust; that's the same rule as for PyTorch `.pt` files later in the course. Better still, avoid object arrays: pad the rows to one length, or save each piece separately.

## Several arrays in one file: `np.savez` and `np.savez_compressed`

**What it is.** `np.savez(path, name1=arr1, name2=arr2, ...)` saves several arrays into one `.npz` file. An `.npz` is a zip archive with one `.npy` inside for each array. `np.load` on it returns an `NpzFile`, which works like a read-only dict: `data["name1"]` loads that array, and `data.files` lists the names. `np.savez_compressed` does the same but compresses the data.

**Why.** A dataset is several arrays that belong together (`X_train`, `y_train`, `X_test`, `y_test`, the class names); a checkpoint is a set of weight arrays. One file keeps them together and named.

**Kotlin comparison.** An `.npz` is a real zip file: `java.util.zip.ZipFile` on Android would list `X.npy` and `y.npy` inside it. Reading it with `np.load` is like opening the zip and reading one entry by name when you ask for it.

```python
X = np.arange(12, dtype=np.float32).reshape(4, 3)
y = np.array([0, 1, 1, 0])
np.savez(work / "dataset.npz", X=X, y=y)
with np.load(work / "dataset.npz") as data:
    print(data.files)
    X2 = data["X"]
    y2 = data["y"]
X2.shape, y2
# ['X', 'y']
# ((4, 3), array([0, 1, 1, 0]))
```

**Underneath.** The `.npz` really is a zip archive, which Python's standard `zipfile` module can open:

```python
import zipfile
with zipfile.ZipFile(work / "dataset.npz") as z:
    print(z.namelist())
# ['X.npy', 'y.npy']
```

Each entry is a complete `.npy` file, header included, named after its keyword. The keyword names (`X=`, `y=`) become the names inside the file. `np.load` on an `.npz` keeps the file open so it can load each array when you ask for it; the `with` block (Day 3c) closes it at the end. Arrays you took out inside the block, like `X2`, stay usable afterwards.

**How it goes wrong: forgetting the names.** Pass arrays without keywords and they get numbered names, which nobody will remember:

```python
np.savez(work / "unnamed.npz", X, y)
with np.load(work / "unnamed.npz") as data:
    print(data.files)
# ['arr_0', 'arr_1']
```

**Compression** helps a lot when the data has repetition (zeros, masks, small integer labels) and very little on random-looking floats such as trained weights:

```python
big = np.zeros((1000, 100))
np.savez(work / "plain.npz", big=big)
np.savez_compressed(work / "small.npz", big=big)
(work / "plain.npz").stat().st_size, (work / "small.npz").stat().st_size
# (800260, 1009)
```

800,000 bytes of zeros compress to about a kilobyte. (The exact compressed size depends on the zlib version, so yours may differ by a few bytes.) The cost is time: compressing and decompressing is slower than reading raw bytes.

## Text files: `np.savetxt` and `np.loadtxt`

**What it is.** `np.savetxt(path, arr)` writes a 1-D or 2-D array as text, one row per line. `np.loadtxt(path)` reads such a file back into an array. The important arguments: `delimiter=` (the separator between columns, a space by default), `fmt=` (how each number is written, using `%` format codes), `header=` (a first line), `comments=` (the prefix for the header, `"# "` by default) and, for `loadtxt`, `skiprows=` (lines to skip at the top).

**Why.** Text is for exchanging small numeric tables with people and other tools: a CSV a spreadsheet can open, a list of thresholds, a confusion matrix for a report. For your own data, prefer `.npy`; for mixed-type tables with column names, prefer pandas (Week 3).

**Kotlin comparison.** `savetxt` is `rows.joinToString("\n") { row -> row.joinToString(",") { "%.3f".format(it) } }` written to a file; `loadtxt` is the reverse, `lines.drop(1).map { it.split(",").map(String::toDouble) }`, with the parsing errors reported for you.

The `fmt` codes are the old-style cousins of Day 2a's f-string format specs: `"%.3f"` means "a float with 3 digits after the point", like `f"{x:.3f}"`. `"%d"` is an integer, `"%.4e"` scientific notation.

```python
scores = np.array([[1, 0.5, 0.25],
                   [2, 0.75, 0.125]])
np.savetxt(work / "scores.csv", scores, delimiter=",", fmt="%.3f",
           header="id,precision,recall", comments="")
print((work / "scores.csv").read_text())
# id,precision,recall
# 1.000,0.500,0.250
# 2.000,0.750,0.125
```

`comments=""` removes the `# ` that would otherwise go in front of the header, so the file is a normal CSV. Reading it back, skip the header line and give the same delimiter:

```python
np.loadtxt(work / "scores.csv", delimiter=",", skiprows=1)
# array([[1.   , 0.5  , 0.25 ],
#        [2.   , 0.75 , 0.125]])
```

**Underneath.** `loadtxt` reads the file line by line. It skips the first `skiprows` lines and any line that starts with the `comments` prefix (`#` by default), splits each remaining line at `delimiter`, and converts each piece with the parser for the target dtype (`float64` unless you say otherwise). All rows must have the same number of pieces, because the result is one rectangular array.

**How it goes wrong (1): the wrong delimiter.** `loadtxt` splits on whitespace by default, so a comma-separated line is one long "number":

```python
try:
    np.loadtxt(work / "scores.csv", skiprows=1)
except ValueError as e:
    print(e)
# could not convert string '1.000,0.500,0.250' to float64 at row 0, column 1.
```

**Recognise it** by a whole line inside the quotes: with no comma delimiter, the whole line was one piece, and the float parser was handed `'1.000,0.500,0.250'`. **Fix:** `delimiter=","`.

**Picking columns and a dtype.** `usecols=` reads only some columns (by position, from 0), and `dtype=` sets the result's dtype. The dtype has to fit the text: the integer parser accepts only digits (and a sign), so `"1.000"` can't be read as an integer.

```python
np.loadtxt(work / "scores.csv", delimiter=",", skiprows=1, usecols=(0, 2))
# array([[1.   , 0.25 ],
#        [2.   , 0.125]])
np.loadtxt(work / "scores.csv", delimiter=",", skiprows=1, usecols=1, dtype=np.float32)
# array([0.5 , 0.75], dtype=float32)
try:
    np.loadtxt(work / "scores.csv", delimiter=",", skiprows=1, usecols=0, dtype=int)
except ValueError as e:
    print(e)
# could not convert string '1.000' to int64 at row 0, column 1.
```

A single `usecols=1` gives a 1-D array, one value per row. (The error message names `int32` in the browser.)

**How it goes wrong (2): text loses precision.** The file holds only the digits that `fmt` wrote:

```python
np.savetxt(work / "rounded.csv", np.array([0.123456789]), fmt="%.2f")
np.loadtxt(work / "rounded.csv")
# array(0.12)
```

The value came back as 0.12, and the rest is gone. (`loadtxt` returns a 0-D array, `array(0.12)`, for a file with a single value.) Text is fine for reports; use `.npy` for anything you'll compute with again. `loadtxt` also fails on a missing value in a row; `np.genfromtxt` (in Going further) or pandas handle gaps.

!!! android "Android bridge: getting an array into your app"
    Android has no built-in reader for `.npy`. The simplest way to ship a NumPy array to an app (a lookup table, an embedding matrix, normalisation constants) is raw bytes in a known order: `arr.astype("<f4").tofile(path)` writes little-endian `float32` with no header. In Kotlin, read the file into a `ByteBuffer`, call `.order(ByteOrder.LITTLE_ENDIAN)`, and `.asFloatBuffer()`. Since there's no header, the app has to know the shape, so write it down next to the file (a small JSON, as on Day 3c). A `.tflite` model file also stores raw tensor bytes, but records each tensor's shape, dtype and quantisation parameters inside the file (in a FlatBuffer), so it's more like `.npy` than like bare `tofile` bytes.

---

**Part 3: Summaries**

## `np.round`, and rounding vs truncating

**What it is.** `np.round(a, decimals=0)` rounds every element to `decimals` digits after the point. A negative `decimals` rounds to tens, hundreds, and so on. Like Python's `round` (Day 2), values exactly halfway go to the **even** neighbour ("banker's rounding"). The result keeps the float dtype.

**Why.** For display, for comparing results at a fixed precision, and as one step of quantisation (the bridge below). Banker's rounding is the IEEE-754 default because always rounding halves up biases sums upwards; alternating avoids that.

```python
np.round(3.14159, 2)
# np.float64(3.14)
np.round(np.array([1.234, 5.678]), 1)
# array([1.2, 5.7])
np.round(np.array([0.5, 1.5, 2.5, 3.5, -0.5]))
# array([ 0.,  2.,  2.,  4., -0.])
np.round(1234.5, -2)
# np.float64(1200.0)
```

0.5 goes to 0 and 2.5 to 2 (the even neighbours), but 1.5 and 3.5 go up to 2 and 4. `-0.` is "negative zero", a float that compares equal to `0.0`; it shows up when a small negative number rounds to zero. Only values that are *exactly* halfway are affected, and many decimals aren't exact in binary (Day 2). Python's `round(2.675, 2)` gives 2.67, because the stored value is slightly below 2.675, but `np.round(2.675, 2)` gives `np.float64(2.68)`: NumPy multiplies by 100, rounds and divides back, and the multiplication rounds as well. Don't rely on either for exact halves of decimals.

**Kotlin comparison.** `kotlin.math.round(2.5)` is `2.0`: it also rounds halves to the even neighbour, so it matches `np.round`. But Java's `Math.round(2.5)` is `3`, and Kotlin's `2.5.roundToInt()` is `3` too: both round halves towards positive infinity (so `(-2.5).roundToInt()` is `-2`). If your app repeats Python preprocessing that rounds, use `kotlin.math.round`, or values that land exactly on .5 (pixel arithmetic often does) come out one higher in the app than in training, with no error anywhere.

**How it goes wrong: `astype(int)` doesn't round.** Day 4 showed that `astype` truncates (`-1.7` became `-1`) and promised the fix; here it is. Converting floats to integers **truncates** toward zero:

```python
vals = np.array([2.7, -2.7, 1.5])
vals.astype(int)
# array([ 2, -2,  1])
np.round(vals).astype(int)
# array([ 3, -3,  2])
pixels = np.array([0.2, 0.5, 0.9999]) * 255     # normalised pixels back to 0..255
pixels
# array([ 51.    , 127.5   , 254.9745])
pixels.astype(np.uint8), np.round(pixels).astype(np.uint8)
# (array([ 51, 127, 254], dtype=uint8), array([ 51, 128, 255], dtype=uint8))
np.floor(vals), np.ceil(vals), np.trunc(vals)
# (array([ 2., -3.,  1.]), array([ 3., -2.,  2.]), array([ 2., -2.,  1.]))
```

2.7 became 2, not 3, and the nearly white pixel 254.97 became 254. Converting normalised images back to bytes is where this bites in practice: every pixel ends up a little darker on average. **Fix:** round first, then convert. (127.5 rounds to the even 128.) The three related functions: `np.floor` goes down (towards minus infinity), `np.ceil` goes up, `np.trunc` goes towards zero, which is what `astype(int)` does.

!!! android "Android bridge: int8 quantisation is round, clip, astype"
    A LiteRT (TensorFlow Lite) model quantised to `int8` stores each float \(x\) as an integer \(q\), with a **scale** \(s\) and a **zero point** \(z\): \(x \approx s \times (q - z)\). The scale is the width of one integer step; the zero point is the integer that stands for 0.0. LiteRT uses two variants:

    - **Activations and model inputs** are **asymmetric**: the 256 values from -128 to 127 are spread over the observed range, so \(z\) can be any value in -128 to 127. The converter always widens that range to include 0.0 (min ≤ 0 ≤ max), so 0.0 is exactly representable and \(z\) stays inside the int8 range; for data that's all positive, the range starts at 0. This is what `getInputTensor(0).quantizationParams()` gives your Kotlin code as `scale` and `zeroPoint` when you feed a quantised model. (That's for an `int8` input. Some models, including older quantised image models, take `uint8` input with codes 0 to 255, so their zero point is often 128; `getInputTensor(0).dataType()` tells you which. For a float tensor that isn't quantised, both values are 0.)
    - **Weights** are **symmetric**: \(z = 0\), the integers run from -127 to 127, and \(s\) is the largest absolute weight divided by 127, usually with a separate \(s\) for each output channel.

    Both use three functions from today:

    ```python
    # activations or model inputs: asymmetric
    x = np.array([-1.0, -0.2, 0.0, 0.37, 2.0])
    lo, hi = min(x.min(), 0.0), max(x.max(), 0.0)      # the range must include 0
    scale = (hi - lo) / 255
    zero_point = int(np.round(-128 - lo / scale))
    q = np.clip(np.round(x / scale) + zero_point, -128, 127).astype(np.int8)
    scale, zero_point
    # (np.float64(0.011764705882352941), -43)
    q
    # array([-128,  -60,  -43,  -12,  127], dtype=int8)
    (q.astype(np.float32) - zero_point) * scale
    # array([-1.        , -0.2       ,  0.        ,  0.36470588,  2.        ])
    ```

    ```python
    # weights: symmetric, zero point 0
    w = np.array([-0.5, -0.1, 0.0, 0.2, 0.4])
    w_scale = np.abs(w).max() / 127
    q_w = np.clip(np.round(w / w_scale), -127, 127).astype(np.int8)
    w_scale
    # np.float64(0.003937007874015748)
    q_w
    # array([-127,  -25,    0,   51,  102], dtype=int8)
    q_w * w_scale
    # array([-0.5      , -0.0984252,  0.       ,  0.2007874,  0.4015748])
    ```

    The order is round, then clip, then cast: Day 4b's "clip before you cast", with rounding added. `np.round` picks the nearest step, `np.clip` keeps it inside the integer range, and only then `astype(np.int8)`. Converting back gives each value to within half a step: 0.37 became 0.3647, and -0.1 became -0.0984. In the asymmetric case the range's low end maps to -128 and its high end to 127, so none of the 256 values is wasted; in the symmetric case 0.0 is exactly representable as `q = 0` and the arithmetic inside the model is simpler, at the cost of unused codes when the weights aren't balanced around 0 (here, nothing maps above 102). That rounding error, spread over millions of values, is the accuracy you trade for a model four times smaller. The challenge exercise builds both round trips.

<figure class="diagram" markdown>
![Animation: a float axis from -1 to 2 above an int8 axis from -128 to 127. A dashed line links 0.0 to the zero point -43. One by one, the values -1.0, -0.2, 0.37 and 2.0 drop onto their codes -128, -60, -12 and 127, and a box shows the arithmetic for 0.37: divided by s it is 31.45, rounded to 31, plus z gives -12, and converting back gives 0.3647](../../assets/img/phase-1/day04c-quant.svg)
<figcaption>Animated: asymmetric quantisation lays 256 evenly spaced codes over the range and snaps each float to the nearest one.</figcaption>
</figure>

## Percentiles and the median: `np.median`, `np.percentile`, `np.quantile`

**What they are.** The **median** is the middle value of the sorted data: half the values are below it. A **percentile** generalises that: the 90th percentile is the value that 90% of the data lies below. `np.percentile(a, q)` takes `q` from 0 to 100; `np.quantile(a, q)` is the same thing with `q` from 0 to 1. Both accept a list of `q` values.

**Why.** They describe data with outliers better than the mean does. App start-up latency is reported as "p50" and "p90" (50th and 90th percentiles) for this reason, both in Android performance work and for ML inference. In data cleaning, clipping a feature to its 1st and 99th percentiles removes extreme outliers.

**Kotlin comparison.** Kotlin has no built-in percentile. `sorted()[((n - 1) * q / 100).toInt()]` is the simplest version: it picks the sorted value at or below the percentile's position, which NumPy calls `method="lower"` (below). NumPy's default goes one step further and interpolates.

**Worked by hand first.** Ten request latencies in milliseconds, with two slow ones:

```python
lat = np.array([12, 15, 11, 90, 14, 13, 16, 12, 200, 14])
np.sort(lat)
# array([ 11,  12,  12,  13,  14,  14,  15,  16,  90, 200])
```

With the default method, the \(q\)-th percentile sits at position \((n - 1) \times q / 100\) in the sorted array, counting from 0, and **interpolates** (draws a straight line) between neighbours when that position isn't a whole number. Here \(n = 10\):

- median (50th): position \(9 \times 0.5 = 4.5\), halfway between sorted[4] = 14 and sorted[5] = 14, so 14.
- 25th: position \(9 \times 0.25 = 2.25\), a quarter of the way from sorted[2] = 12 to sorted[3] = 13: 12.25. The 75th, position 6.75, is \(15 + 0.75 \times (16 - 15) = 15.75\).
- 90th: position \(9 \times 0.9 = 8.1\), a tenth of the way from sorted[8] = 90 to sorted[9] = 200: \(90 + 0.1 \times (200 - 90) = 101\).

NumPy reproduces the hand results:

```python
np.median(lat)
# np.float64(14.0)
np.percentile(lat, 50)
# np.float64(14.0)
np.percentile(lat, [25, 75])
# array([12.25, 15.75])
np.percentile(lat, 90)
# np.float64(100.99999999999996)
np.quantile(lat, 0.9)
# np.float64(100.99999999999996)
```

NumPy printed `100.99999999999996`, which is 101 with floating-point rounding in the last digit: the same effect as `0.1 + 0.2`, and the reason Day 4b's float comparisons exist (`np.isclose(np.percentile(lat, 90), 101)` is `True`). The mean, `lat.mean()`, is 39.7, dragged up by the two outliers; the median, 14, describes a typical request.

**Other methods.** Interpolating is the default, `method="linear"`. Other methods return an actual data value instead, which matters when the answer must be one of the observations (an integer count, a real latency someone saw):

```python
np.percentile(lat, 90, method="nearest")
# np.int64(90)
np.percentile(lat, 90, method="lower"), np.percentile(lat, 90, method="higher")
# (np.int64(90), np.int64(200))
```

`"lower"` and `"higher"` take the sorted values on either side of position 8.1; `"nearest"` takes the closer one. Like the other reductions, percentiles are poisoned by `nan`, and `np.nanpercentile` (with `np.nanmedian` and `np.nanquantile`) skips gaps, as on Day 4b:

```python
np.percentile(np.array([1.0, np.nan, 3.0]), 50)
# np.float64(nan)
np.nanpercentile(np.array([1.0, np.nan, 3.0]), 50)
# np.float64(2.0)
```

**How it goes wrong: a fraction passed to `percentile`.**

```python
np.percentile(lat, 0.95)
# np.float64(11.0855)
np.percentile(lat, 95)
# np.float64(150.4999999999999)
```

`np.percentile(lat, 0.95)` is the 0.95th percentile, almost the minimum, not the 95th. No error, a wrong number. **Recognise it** by a "p95" near the smallest value. **Fix it:** `percentile` takes 0 to 100, `quantile` takes 0 to 1; pick one and stick to it.

## Histograms: `np.histogram`

**What it is.** `np.histogram(a, bins=10)` splits the range of `a` into `bins` equal-width intervals and counts how many values fall into each. It returns two arrays: `counts`, with one number per bin, and `edges`, the bin boundaries, which has **one more** element than `counts`. `bins` can also be a list of edges, and `range=(lo, hi)` fixes the overall range instead of using the data's min and max.

**Why.** It's the first look at any feature's distribution: is it bell-shaped, skewed, has it got two peaks, are there outliers? Matplotlib's `plt.hist` (Week 3) calls `np.histogram` and draws the result.

**Kotlin comparison.** With equal-width bins, a histogram is `values.groupingBy { ((it - lo) / width).toInt().coerceAtMost(bins - 1) }.eachCount()`: each value's bin number is how many widths it sits above the low end. The `coerceAtMost` is there because the maximum value would otherwise get a bin of its own, one past the end; that's exactly why NumPy's last bin includes its right edge.

**Worked by hand first.** Take `vals = [1, 2, 2, 3, 3, 3, 4, 9]` and 4 bins. The data runs from 1 to 9, so 4 equal bins are each 2 wide, with edges 1, 3, 5, 7, 9 (five edges for four bins, like fence posts). Each bin includes its left edge and excludes its right edge, **except the last**, which includes both, so the maximum is counted. Bin [1, 3) holds 1, 2, 2: 3 values. Bin [3, 5) holds 3, 3, 3, 4: 4 values. Bin [5, 7) is empty. Bin [7, 9] holds the 9. NumPy gives the same counts and edges:

```python
vals = np.array([1, 2, 2, 3, 3, 3, 4, 9])
counts, edges = np.histogram(vals, bins=4)
counts
# array([3, 4, 0, 1])
edges
# array([1., 3., 5., 7., 9.])
```

A quick text plot shows the shape:

```python
for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
    print(f"{lo:4.1f} to {hi:4.1f}  {'#' * c}")
#  1.0 to  3.0  ###
#  3.0 to  5.0  ####
#  5.0 to  7.0
#  7.0 to  9.0  #
```

`edges[:-1]` is every edge but the last (the left edges) and `edges[1:]` every edge but the first (the right edges), so `zip` pairs each bin's two edges with its count. `{lo:4.1f}` formats a float at least 4 characters wide with 1 digit after the point, so the columns line up (Day 2a's width and precision specs, combined). `'#' * c` repeats the character `c` times (Day 2a). Custom edges make bins of any widths:

```python
counts, edges = np.histogram(vals, bins=[0, 2, 4, 10])
counts
# array([1, 5, 2])
```

[0, 2) holds the 1; [2, 4) holds 2, 2, 3, 3, 3; [4, 10] holds 4 and 9.

**How it goes wrong: zipping counts with edges.** There is always one more edge than count. `zip` stops at the shorter input without complaint, so zipping the two directly drops the last edge and pairs each count with its left edge only:

```python
list(zip(counts.tolist(), edges.tolist()))
# [(1, 0), (5, 2), (2, 4)]
try:
    list(zip(counts, edges, strict=True))
except ValueError as e:
    print(e)
# zip() argument 2 is longer than argument 1
```

The right edge 10 is gone, and a report built from these pairs would describe the last bin as "4 and up" instead of "4 to 10". **Fix:** use the `edges[:-1]`, `edges[1:]` pattern above, or pass `strict=True` (Day 2) so a length mismatch raises the error you see here.

**Densities.** `density=True` divides each count by (total count × bin width), so the bar areas add up to 1. That's what lets you draw a histogram on top of a probability curve, which Month 2 does; `plt.hist(..., density=True)` uses the same option.

```python
dens, edges = np.histogram(vals, bins=4, density=True)
dens
# array([0.1875, 0.25  , 0.    , 0.0625])
(dens * np.diff(edges)).sum()
# np.float64(1.0)
```

The first bin held 3 of the 8 values and is 2 wide, so its density is 3 / (8 × 2) = 0.1875. Multiplying each density by its bin's width (`np.diff(edges)`, Day 4b) gives back the share of values in that bin, and the shares add up to 1.

## `np.isin`: membership for every element

**What it is.** `np.isin(a, values)` returns a boolean array with `a`'s shape: `True` where an element of `a` appears anywhere in `values`. It's Python's `in` (Day 2b), applied to every element at once, and the result is a mask you can index with (Day 4).

**Why.** Filtering by a list of allowed (or banned) categories: keep only the samples whose label is one of the classes you're training on, drop the apps in a blocked category, mark the user ids that appear in a test list.

**Kotlin comparison.** `a.map { it in allowed }` with `allowed` as a `Set`, and `a.filter { it in allowed }` for the filtered list.

```python
np.isin(np.array(["cat", "dog", "fox"]), ["dog", "owl", "fox"])
# array([False,  True,  True])
labels = np.array(["cat", "bird", "dog", "cat", "fox"])
feats = np.arange(10).reshape(5, 2)
keep = np.isin(labels, ["cat", "dog"])
keep
# array([ True, False,  True,  True, False])
feats[keep]
# array([[0, 1],
#        [4, 5],
#        [6, 7]])
```

The mask keeps rows 0, 2 and 3, the samples labelled cat or dog, and drops the bird and the fox.

**Underneath.** For small `values`, `np.isin` is the same as comparing every element with every value through broadcasting (Day 9) and asking whether any matched: `(a[:, None] == values).any(axis=1)`. NumPy picks a strategy by size: element-by-element comparison against each value when `values` is short, and sorting (or, for integers, a lookup table) when it's long, so the cost doesn't grow with `len(a) × len(values)`. Either way, there's no Python loop over `a`.

```python
a = np.array([1, 2, 3])
(a[:, None] == np.array([3, 4])).any(axis=1)
# array([False, False,  True])
```

**How it goes wrong: passing a Python set.** A set is the natural Kotlin-style container for "allowed values", but NumPy doesn't treat it as a collection: it becomes a single object, which no number equals.

```python
np.isin(np.array([1, 2, 3]), {3, 4})
# array([False, False, False])
np.isin(np.array([1, 2, 3]), [3, 4])
# array([False, False,  True])
```

No error, and a mask that's all `False`, so the "filtered" data is empty. **Recognise it** by a filter that keeps nothing. **Fix it** by passing a list or an array: `np.isin(a, list(allowed))`. (NumPy's documentation warns about exactly this.)

---

## Putting it together: cleaning a feature column

Here is a small, realistic preprocessing script that uses tools from Days 4b and 4c together. The data is fake but shaped like the Play Store data you'll load in Week 3: an app name, a rating, and an install count. The export wrote `-1` for "no rating yet", and one rating is corrupt.

```python
rng = np.random.default_rng(42)
n = 12
names = np.array([f"app{i:02d}" for i in range(n)])
rating = np.round(rng.normal(4.0, 0.6, size=n), 1)
rating[[2, 7]] = -1.0          # the export wrote -1 for "no rating yet"
rating[5] = 7.3                # a corrupt value
installs = rng.integers(1_000, 60_000, size=n)   # int64 (shows dtype=int64 in the browser)
rating
# array([ 4.2,  3.4, -1. ,  4.6,  2.8,  7.3,  4.1, -1. ,  4. ,  3.5,  4.5,
#         4.5])
installs
# array([47112, 38988, 24742, 49542, 33180, 27161, 27577, 14407,  6436,
#        33720, 53385,  4765])
```

The list comprehension builds names `app00` to `app11` (`:02d` pads to two digits, Day 2a). `rating[[2, 7]] = -1.0` assigns through a fancy index, setting positions 2 and 7 at once. Now clean it in steps:

```python
rating = np.where(rating < 0, np.nan, rating)      # 1: sentinel -> NaN
np.count_nonzero(np.isnan(rating))
# np.int64(2)
median = np.nanmedian(rating)                       # 2: fill with the median
median
# np.float64(4.15)
rating = np.where(np.isnan(rating), median, rating)
rating = np.clip(rating, 1.0, 5.0)                  # 3: force into the valid range
rating
# array([4.2 , 3.4 , 4.15, 4.6 , 2.8 , 5.  , 4.1 , 4.15, 4.  , 3.5 , 4.5 ,
#        4.5 ])
```

Step 1 turns the sentinel into a real missing value, so it can't be averaged in as -1. Step 2 fills the gaps with the median of the ratings that exist (the median, not the mean, because it isn't pulled by the corrupt 7.3). Step 3 clips the corrupt value to 5.0. Then summarise and split:

```python
top = np.argsort(-installs, kind="stable")[:3]     # 4: top 3 apps by installs
names[top], installs[top], rating[top]
# (array(['app10', 'app03', 'app00'], dtype='<U5'), array([53385, 49542, 47112]), array([4.5, 4.6, 4.2]))
np.percentile(installs, [50, 90])
# array([30378.5, 49299. ])
counts, edges = np.histogram(rating, bins=4, range=(1, 5))   # 5: rating histogram
counts
# array([0, 1, 2, 9])
```

The median app has about 30,000 installs and the 90th percentile about 49,000. With `range=(1, 5)` the four bins are 1–2, 2–3, 3–4 and 4–5 stars, so 9 of the 12 apps sit in the top bin (the two filled values among them).

```python
idx = np.random.default_rng(0).permutation(n)               # 6: split
test_idx, train_idx = idx[:3], idx[3:]
features = np.column_stack([rating, np.log10(installs)])
X_train, X_test = features[train_idx], features[test_idx]
X_train.shape, X_test.shape
# ((9, 2), (3, 2))
```

`np.log10` is the base-10 logarithm, a ufunc like `np.log`: install counts span orders of magnitude, so models usually see their logarithm (Day 19 shows why on a plot). Finally, save the result so the next script starts from here:

```python
out_dir = Path(tempfile.mkdtemp())
np.savez(out_dir / "apps_split.npz", X_train=X_train, X_test=X_test,
         train_idx=train_idx, test_idx=test_idx)
with np.load(out_dir / "apps_split.npz") as d:
    print(d.files)
    print(np.allclose(d["X_test"], X_test))
# ['X_train', 'X_test', 'train_idx', 'test_idx']
# True
```

Saving the indices as well as the arrays means you can always find out which original apps ended up in the test set.


---

## Going further

These exist and are worth knowing about; you won't need them this month.

- [Parallel random streams](https://numpy.org/doc/stable/reference/random/parallel.html): `rng.spawn(n)` and `SeedSequence` give independent generators for parallel workers (one per data-loading worker in PyTorch, Month 4).
- [All the `Generator` distributions](https://numpy.org/doc/stable/reference/random/generator.html): `exponential`, `poisson`, `binomial`, `lognormal` and many more; Month 2 uses several.
- [`np.genfromtxt`](https://numpy.org/doc/stable/reference/generated/numpy.genfromtxt.html): text loading that handles missing values; pandas `read_csv` (Week 3) is usually the better tool.
- [`np.memmap` and `np.load(..., mmap_mode="r")`](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html): use an array on disk without loading it all into memory.
- [`ndarray.tofile` and `np.fromfile`](https://numpy.org/doc/stable/reference/generated/numpy.fromfile.html): raw bytes with no header, the format of the Android bridge in Part 2.
- [`np.histogram_bin_edges`](https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html): automatic bin choices such as `bins="auto"`.
- Stratified splits: scikit-learn's [`train_test_split(..., stratify=y)`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) (Month 4).
- [LiteRT quantization specification](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/quantization_spec): the exact int8 rules, including per-channel weight scales.

---

## Exercises

Each solution re-creates its own inputs, so you can run it on its own.

**1. Warm-up: a fair die.** With `rng = np.random.default_rng(2026)`, roll a six-sided die 6,000 times with `rng.integers`. Count how often each face came up with `np.unique`, check that all six faces (and only those) appear, and print the mean roll. What mean do you expect?

**2. Warm-up: save, load, compare.** Make `w = np.random.default_rng(1).normal(0, 1, size=(3, 4)).astype(np.float32)`. (a) Save it with `np.save` to a temporary folder, load it back, and check it's identical with `np.array_equal`. (b) Check that the file's size is the header (128 bytes for an array this small) plus `w.nbytes`. (c) Save it with `np.savetxt(..., fmt="%.2f")`, load that back, and show that it's no longer equal, but is equal within 0.005.

**3. Core: a three-way split.** You have 20 samples, `X = np.arange(40).reshape(20, 2)` and `y = np.array([0]*10 + [1]*10)`. Split them 70/15/15 into train, validation and test sets with one permutation from `np.random.default_rng(3)`. Prove that the three index sets don't overlap and together cover all 20 samples, and print the label counts of each set.

**4. Core: a latency report.** Make 1,000 fake request latencies in milliseconds: `rng = np.random.default_rng(8)`, 995 values from `rng.normal(40, 5, size=995)` plus five slow ones, `[250, 300, 180, 220, 400]`, joined with `np.concatenate`. Print the mean, the median, and the 90th and 99th percentiles, rounded to 1 decimal. Count the requests in the bins 0–20, 20–40, 40–60, 60–100 and 100–1000 ms with `np.histogram`, and print the share of requests slower than 100 ms. Which of the summary numbers would you put on a dashboard, and why?

**5. Core: an epoch of mini-batches.** Write a generator function `batches(X, y, batch_size, rng)` that shuffles the sample positions once with `rng.permutation`, then yields `(X_batch, y_batch)` pairs of `batch_size` samples (the last batch may be smaller). With `X = np.arange(10).reshape(10, 1)`, `y = np.arange(10) * 10` and one `rng = np.random.default_rng(0)`, run two epochs with `batch_size=4`. Print the `y` values of each batch, check that every batch keeps `y == X * 10`, and check that each epoch uses every sample exactly once, in a different order.

**6. Predict the output.** Write down what each line prints *before* running it. Then run it and explain every one you got wrong.

```python
import numpy as np

print(np.round(np.array([0.5, 1.5, 2.5])))
print(np.random.default_rng(1).integers(0, 1, size=3))
print(np.array([2.7, -2.7]).astype(int))
print(np.percentile(np.array([1, 2, 3, 4]), 50))
print(np.histogram(np.array([1, 2, 3]), bins=2)[0])
print(np.random.default_rng(9).random() == np.random.default_rng(9).random())
print(np.sort(np.random.default_rng(5).permutation(3)))
print(np.isin(np.array([1, 2, 3]), [3, 4]))
```

**7. Find and fix the bug.** This function should add a little random noise to every sample (a simple data augmentation), shuffle, and split off a test set. It runs without an error, but it has **two** bugs. Find them, explain what each one does to the data, and fix them. (It also changes the caller's `X`; fix that too.)

```python
import numpy as np

def augment_and_split(X, y, test_frac=0.25):
    for i in range(len(X)):
        rng = np.random.default_rng(0)
        X[i] = X[i] + rng.normal(0, 0.1, size=X.shape[1])
    rng = np.random.default_rng(0)
    rng.shuffle(X)
    rng.shuffle(y)
    n_test = round(len(X) * test_frac)
    return X[n_test:], X[:n_test], y[n_test:], y[:n_test]

X = np.zeros((4, 2))
y = np.array([0, 1, 2, 3])
X_tr, X_te, y_tr, y_te = augment_and_split(X, y)
print(X)
print(y)
# [[ 0.01257302 -0.01321049]
#  [ 0.01257302 -0.01321049]
#  [ 0.01257302 -0.01321049]
#  [ 0.01257302 -0.01321049]]
# [3 2 1 0]
```

**8. Challenge: int8 quantisation round trips.** Following the Android bridge in Part 3, write `quantize_symmetric(w)` (returning `q`, `scale`) and `quantize_asymmetric(x)` (returning `q`, `scale`, `zero_point`), and one `dequantize(q, scale, zero_point=0)` that returns `float32`. (a) Quantise 1,000 weights, `np.random.default_rng(0).normal(0, 0.05, size=1000).astype(np.float32)`, symmetrically, and check with `np.allclose` that every dequantised weight is within half a step of the original. (b) Make 1,000 ReLU activations, `np.maximum(0, np.random.default_rng(1).normal(0, 1, size=1000)).astype(np.float32)` (all ≥ 0), quantise them both ways, and compare the maximum error. Explain which scheme wins and why. (c) Save the asymmetric `q`, `scale` and `zero_point` to one `.npz` in a temporary folder, load them back, and check that dequantising gives exactly the same array.

??? success "Solutions"

    **1. A fair die.**

    ```python
    import numpy as np

    rng = np.random.default_rng(2026)
    rolls = rng.integers(1, 7, size=6000)
    faces, counts = np.unique(rolls, return_counts=True)
    print(faces)
    print(counts)
    print(np.array_equal(faces, np.arange(1, 7)))
    print(rolls.mean())
    # [1 2 3 4 5 6]
    # [ 994 1020 1024  985  976 1001]
    # True
    # 3.4886666666666666
    ```

    `integers(1, 7)` gives 1 to 6, because the upper bound is excluded; `integers(1, 6)` would never roll a 6, and `faces` would show it. Each face comes up about 1,000 times, but not exactly: that's what random means, and the counts wander by a few dozen. The expected mean is (1 + 2 + 3 + 4 + 5 + 6) / 6 = 3.5, and 6,000 rolls land close to it, at 3.489.

    **2. Save, load, compare.**

    ```python
    import numpy as np
    import tempfile
    from pathlib import Path

    w = np.random.default_rng(1).normal(0, 1, size=(3, 4)).astype(np.float32)
    folder = Path(tempfile.mkdtemp())
    np.save(folder / "w.npy", w)
    w2 = np.load(folder / "w.npy")
    print(np.array_equal(w, w2), w2.dtype)
    print((folder / "w.npy").stat().st_size == 128 + w.nbytes)
    np.savetxt(folder / "w.txt", w, fmt="%.2f")
    w3 = np.loadtxt(folder / "w.txt")
    print(np.array_equal(w, w3))
    print(np.allclose(w3, w, rtol=0, atol=0.005 + 1e-7))
    # True float32
    # True
    # False
    # True
    ```

    `.npy` stores the exact bits and the dtype, so the round trip is exact and stays `float32`. The file is the header, padded to 128 bytes for a short shape like `(3, 4)`, plus 12 values × 4 bytes. Text with `fmt="%.2f"` keeps two decimals, so each value can be off by up to half of 0.01 (and `loadtxt` returns `float64`). `rtol=0` makes the check purely absolute; the extra `1e-7` allows for the `float32` value itself being stored with a little rounding.

    **3. A three-way split.**

    ```python
    import numpy as np

    X = np.arange(40).reshape(20, 2)
    y = np.array([0] * 10 + [1] * 10)
    idx = np.random.default_rng(3).permutation(len(X))
    n_train = round(len(X) * 0.70)
    n_val = round(len(X) * 0.15)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    print(len(train_idx), len(val_idx), len(test_idx))
    # 14 3 3
    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    print(np.array_equal(np.sort(all_idx), np.arange(20)))
    print(len(np.unique(all_idx)) == len(all_idx))
    # True
    # True
    for name, part in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        print(name, np.unique(y[part], return_counts=True))
    # train (array([0, 1]), array([6, 8]))
    # val (array([0, 1]), array([2, 1]))
    # test (array([0, 1]), array([2, 1]))
    X_train, y_train = X[train_idx], y[train_idx]
    X_train.shape
    # (14, 2)
    ```

    One permutation cut at two points can't produce overlapping parts, and the checks prove it: the sorted union is exactly 0 to 19, and no index appears twice (the number of unique indices equals the total). The last part takes whatever is left, so rounding never loses a sample. The label counts show how balanced each part came out; with 3 samples in a set, a random split can easily be unbalanced, which is why stratified splits exist.

    **4. A latency report.**

    ```python
    import numpy as np

    rng = np.random.default_rng(8)
    lat = np.concatenate([rng.normal(40, 5, size=995), [250, 300, 180, 220, 400]])
    print(np.round(lat.mean(), 1), np.round(np.median(lat), 1))
    print(np.round(np.percentile(lat, [90, 99]), 1))
    counts, edges = np.histogram(lat, bins=[0, 20, 40, 60, 100, 1000])
    print(counts)
    print(np.count_nonzero(lat > 100) / len(lat))
    # 41.0 39.8
    # [46.5 53.3]
    # [  0 516 479   0   5]
    # 0.005
    ```

    Five slow requests out of 1,000 pull the mean (41.0) above the median (39.8), while the median and p90 barely notice them; p99 (53.3) sits below the slow group, because only 0.5% of requests are in it, fewer than the 1% above the 99th percentile. The histogram shows that almost everything is in 20–60 ms, with the five slow ones alone in the last bin. On a dashboard, p50 says what a typical user sees and p99 (or the share above 100 ms) says how often someone has a bad time; the mean mixes the two and describes nobody.

    **5. An epoch of mini-batches.**

    ```python
    import numpy as np

    def batches(X, y, batch_size, rng):
        idx = rng.permutation(len(X))
        for start in range(0, len(X), batch_size):
            part = idx[start:start + batch_size]
            yield X[part], y[part]

    X = np.arange(10).reshape(10, 1)
    y = np.arange(10) * 10
    rng = np.random.default_rng(0)
    for epoch in range(2):
        seen = []
        for X_b, y_b in batches(X, y, 4, rng):
            print(epoch, y_b, np.array_equal(y_b, X_b[:, 0] * 10))
            seen.extend(y_b.tolist())
        print("every sample once:", sorted(seen) == y.tolist())
    # 0 [40 60 20 70] True
    # 0 [30 50 90  0] True
    # 0 [80 10] True
    # every sample once: True
    # 1 [20 90 30 60] True
    # 1 [ 0 40 80 70] True
    # 1 [50 10] True
    # every sample once: True
    ```

    One permutation per epoch, applied to both arrays, keeps every label with its sample (the `True`s). Slicing the shuffled positions in steps of `batch_size` gives batches of 4, 4 and 2, and together they're each position exactly once. The generator calls `rng.permutation` each time it starts, and the same `rng` continues its sequence, so the second epoch gets a new order while the whole run stays reproducible. This is Day 3d's `DataLoader`, NumPy style.

    **6. Predict the output.**

    ```python
    import numpy as np

    print(np.round(np.array([0.5, 1.5, 2.5])))
    print(np.random.default_rng(1).integers(0, 1, size=3))
    print(np.array([2.7, -2.7]).astype(int))
    print(np.percentile(np.array([1, 2, 3, 4]), 50))
    print(np.histogram(np.array([1, 2, 3]), bins=2)[0])
    print(np.random.default_rng(9).random() == np.random.default_rng(9).random())
    print(np.sort(np.random.default_rng(5).permutation(3)))
    print(np.isin(np.array([1, 2, 3]), [3, 4]))
    # [0. 2. 2.]
    # [0 0 0]
    # [ 2 -2]
    # 2.5
    # [1 2]
    # True
    # [0 1 2]
    # [False False  True]
    ```

    Line by line: halves round to the even neighbour (0, 2, 2). `integers(0, 1)` excludes 1, so it can only produce 0. `astype(int)` truncates toward zero, so 2 and -2. The median of 1, 2, 3, 4 is halfway between 2 and 3, 2.5 (position 1.5 in the sorted array). Two bins over 1 to 3 are [1, 2) and [2, 3], so the counts are 1 and 2: the last bin includes its right edge. Two generators with the same seed give the same first number, so `True`. A permutation of 3 contains 0, 1 and 2 once each, so sorting it always gives `[0 1 2]`, whatever the seed. And only 3 is in `[3, 4]`.

    **7. Find and fix the bug.**

    ```python
    import numpy as np

    def augment_and_split(X, y, test_frac=0.25, seed=0):
        rng = np.random.default_rng(seed)                  # fix 1: one generator, created once
        X = X + rng.normal(0, 0.1, size=X.shape)           # new array: the caller's X is untouched
        idx = rng.permutation(len(X))                      # fix 2: one permutation for both
        X, y = X[idx], y[idx]
        n_test = round(len(X) * test_frac)
        return X[n_test:], X[:n_test], y[n_test:], y[:n_test]

    X = np.zeros((4, 2))
    y = np.array([0, 1, 2, 3])
    X_tr, X_te, y_tr, y_te = augment_and_split(X, y)
    print(X)
    print(y)
    print(X_te, y_te)
    # [[0. 0.]
    #  [0. 0.]
    #  [0. 0.]
    #  [0. 0.]]
    # [0 1 2 3]
    # [[ 0.01257302 -0.01321049]] [0]
    ```

    Bug 1: a new generator with seed 0 was created for every sample, so every sample got the **same** noise: all four rows of the output are identical, which isn't augmentation at all. One generator, created once, draws fresh noise for each row (drawing the whole `(4, 2)` array of noise in one call also removes the loop). With 4 samples and `test_frac=0.25`, the test set is one sample: the last printed line, which still has its own label. Bug 2: two `shuffle` calls gave `X` and `y` different orders, so samples were separated from their labels. One permutation applied to both keeps them together. The original also wrote into the caller's arrays (`X[i] = ...` and `shuffle` both modify in place, and the printout showed `X` and `y` changed); `X + noise` and fancy indexing make new arrays, so the caller's `X` stays zeros and `y` stays `[0 1 2 3]`.

    **8. Challenge: int8 quantisation round trips.**

    ```python
    import numpy as np
    import tempfile
    from pathlib import Path

    def quantize_symmetric(w):
        """Weights: zero point 0, integers -127..127."""
        scale = float(np.abs(w).max()) / 127
        q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
        return q, scale

    def quantize_asymmetric(x):
        """Activations: the full -128..127 range covers the data's range, widened to include 0."""
        lo, hi = min(float(x.min()), 0.0), max(float(x.max()), 0.0)   # include 0
        scale = (hi - lo) / 255
        zero_point = int(np.round(-128 - lo / scale))
        q = np.clip(np.round(x / scale) + zero_point, -128, 127).astype(np.int8)
        return q, scale, zero_point

    def dequantize(q, scale, zero_point=0):
        return ((q.astype(np.float32) - zero_point) * scale).astype(np.float32)

    w = np.random.default_rng(0).normal(0, 0.05, size=1000).astype(np.float32)
    q_w, s_w = quantize_symmetric(w)
    w_back = dequantize(q_w, s_w)
    print("weights within half a step:", np.allclose(w_back, w, rtol=0, atol=s_w / 2 + 1e-7))
    print("bytes:", w.nbytes, "->", q_w.nbytes)

    act = np.maximum(0, np.random.default_rng(1).normal(0, 1, size=1000)).astype(np.float32)
    q_s, s_s = quantize_symmetric(act)
    q_a, s_a, z_a = quantize_asymmetric(act)
    err_sym = np.abs(dequantize(q_s, s_s) - act).max()
    err_asym = np.abs(dequantize(q_a, s_a, z_a) - act).max()
    print(f"activations: symmetric max error {err_sym:.5f}, asymmetric {err_asym:.5f}")
    print("codes used:", np.unique(q_s).size, "vs", np.unique(q_a).size)

    folder = Path(tempfile.mkdtemp())
    np.savez(folder / "act_q.npz", q=q_a, scale=s_a, zero_point=z_a)
    with np.load(folder / "act_q.npz") as d:
        q2, s2, z2 = d["q"], float(d["scale"]), int(d["zero_point"])
    print("same after reload:", np.array_equal(dequantize(q2, s2, z2), dequantize(q_a, s_a, z_a)))
    # weights within half a step: True
    # bytes: 4000 -> 1000
    # activations: symmetric max error 0.01470, asymmetric 0.00734
    # codes used: 78 vs 137
    # same after reload: True
    ```

    (a) Rounding to the nearest step means no weight can be more than half a step from its original; `rtol=0` makes the check purely absolute, and the tiny `+ 1e-7` allows for `float32` rounding in the arithmetic itself. The weights shrink from 4,000 bytes to 1,000. (b) ReLU outputs are never negative, so the symmetric scheme wastes every code from -127 to -1 and can use at most 128 of the 256 (78 different codes actually occur here, against 137 for the asymmetric scheme): its step is twice as wide, and so is its worst error, 0.0147 against 0.0073. The asymmetric scheme maps 0.0 to -128 and the largest activation to 127, using the full range. That's why LiteRT quantises activations asymmetrically, while weights, which are roughly balanced around 0, use the simpler symmetric form. (c) `scale` and `zero_point` are saved as 0-D arrays next to `q`; `float(...)` and `int(...)` turn them back into plain numbers. Because `.npz` stores the exact bits, the reloaded values dequantise to exactly the same array, so `array_equal` (not `allclose`) is the right check.

## Resources

- [NumPy: Random sampling](https://numpy.org/doc/stable/reference/random/index.html): the `Generator` API, why it replaced the legacy functions, and the reproducibility policy.
- [NumPy: Reading and writing files](https://numpy.org/doc/stable/user/how-to-io.html): which format to choose for which job, including `.npy`, `.npz` and text.
- [NumPy: the `.npy` format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html): the header layout you looked at in Part 2.
- [NumPy reference: `numpy.percentile`](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html): every `method=`, with a diagram of how they differ.
- [Python for Data Analysis, Chapter 4](https://wesmckinney.com/book/numpy-basics): free online; pseudorandom number generation and file input and output with arrays.
- [LiteRT: Post-training quantization](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_quantization): how the `int8` schemes in the Android bridge are used for real models.

## Checklist

- [ ] I can make reproducible random numbers with `np.random.default_rng(seed)`, I know what happens without a seed, and I know why `np.random.seed` is discouraged.
- [ ] I can draw with `random`, `uniform`, `integers` (upper bound excluded) and `normal`, and sample with `choice`, with or without replacement and with `p`.
- [ ] I know `shuffle` from `permutation`, and I shuffle features and labels with one permutation.
- [ ] I can build a reproducible train/test split, prove the parts don't overlap, and check their label counts.
- [ ] I can save and load arrays with `.npy`, `.npz` and text files, and I know what's in a `.npy` header.
- [ ] I can round with `np.round`, explain why `np.round(2.5)` is `2.0`, and tell rounding from truncation, `floor` and `ceil`.
- [ ] I can quantise an array to `int8`, symmetrically for weights and asymmetrically for activations.
- [ ] I can compute the median and percentiles, by hand and with `np.percentile` / `np.quantile`.
- [ ] I can bin values with `np.histogram` and read its `counts` and `edges`.
- [ ] I can test membership element by element with `np.isin`.

**Next:** [Day 5: Vectors, the atoms of machine learning](day-05-vectors.md)
