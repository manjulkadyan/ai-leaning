# Days 55–56: Month 2 Milestone

**Phase 1 · Week 8 · ~420 minutes (two sessions of about 3½ hours)**

## Objectives

By the end of the weekend you can:

- [ ] Turn the Play Store table into a regression problem: choose a target and features, explain why, and spot a feature that leaks the answer.
- [ ] Encode a yes/no column and a category column as numbers (one-hot encoding), and say why one category gets no column.
- [ ] Split the data into train, validation and test sets with a fixed seed, and say what each set is for.
- [ ] Fit every preprocessing statistic (fill values, category list, means, standard deviations) on the training rows only, and apply it unchanged to the other sets.
- [ ] Set a pass mark from a baseline *before* training.
- [ ] Derive the gradient of the mean squared error by hand, one residual at a time, and prove it correct with `grad_check`.
- [ ] Write mini-batch gradient descent, trace one epoch of it, choose the learning rate on the validation set, and read learning curves.
- [ ] Look at the test set exactly once, report whether you beat the baseline, and confirm your weights against `np.linalg.lstsq`.
- [ ] Read a predicted-versus-actual chart for an error analysis, write a short report, and write the Month 2 retrospective.

**Prerequisites.** All of Month 2. The parts you lean on most:

- [Day 33](day-33-vector-matrix-gradients.md): the MSE gradient in matrix form, the shape rule, `numerical_grad` and `grad_check`, which this page reuses unchanged.
- [Day 31](day-31-chain-rule.md): the chain rule, used in the derivation.
- [Day 43](day-43-sampling.md): a sample is not the population, which is why a held-out set estimates performance on new data.
- [Day 45](day-45-hypothesis-tests.md): multiple comparisons, the reason for a separate validation set.
- [Day 47](day-47-maximum-likelihood.md): why minimizing MSE is a sensible goal (it is maximum likelihood under Gaussian noise).
- [Day 50](day-50-loss-functions.md): losses. [Day 51](day-51-gradient-descent.md): gradient descent and the learning rate. [Day 52](day-52-sgd-momentum.md): mini-batches, epochs, shuffling. [Day 53](day-53-convexity.md): why this loss has a single minimum.
- From Month 1: [Day 17](day-17-pandas-cleaning.md) (`clean_apps`), [Day 19](day-19-matplotlib.md) (charts), [Day 22](day-22-norms-distances.md) (standardizing features), [Day 23](day-23-rank-span.md) (collinear columns make \( X^\top X \) singular), [Day 24](day-24-eigenvectors.md) (eigenvalues) and [Day 25](day-25-projections-pca.md) (least squares with `np.linalg.lstsq`).

**Suggested time split.** Day 55 (about 3½ h): Part 1. 40 minutes choosing the target and features, 50 minutes on the split and the preprocessing, 20 minutes on the baseline and the pass mark, 90 minutes deriving the gradient on paper and getting `check_milestone()` green. Day 56 (about 3½ h): Part 2. 80 minutes on the training loop and the learning-rate sweep, 30 minutes on the single test evaluation, the `lstsq` comparison and the error analysis, 50 minutes on the exercises you choose, 50 minutes on the report and the Month 2 retrospective.

---

## The big idea

[Days 27–28](day-27-month1-milestone.md) explained what a milestone is: the month's exam, where nobody prompts the steps and you grade yourself against a published rubric. Month 1's exam asked whether you could compute. This one asks whether you can **learn from data**, which is what the whole of Month 2 was for.

The milestone, from the [Month 2 plan](../../plan/month-02.md): *train a linear regression on the Play Store dataset with gradient descent whose gradients you derived by hand, verify them against a numerical gradient check, and get the mean squared error on a held-out test set below a target you set from a baseline.*

Every week of the month has a job in that sentence:

- **"gradients you derived by hand"** comes from Week 5: derivatives, the chain rule, the gradient, and Day 33's formula \( \frac{2}{n}X^\top(X\mathbf{w} - \mathbf{y}) \).
- **"held-out test set"** and **"a target set from a baseline"** come from Weeks 6 and 7: a sample is not the population, and a number means little without a comparison.
- **"mean squared error"** comes from Day 47 (MSE is maximum likelihood when the noise is Gaussian) and Day 50 (losses).
- **"gradient descent"** comes from Week 8: Days 51 and 52, with Day 53 promising that this loss is a single bowl, so there is one bottom to find.

**Analogy.** Think of a release with a proper QA process. The training set is your dev build: you poke at it as much as you like. The validation set is the internal test track: you use it to choose between candidate builds (here, learning rates). The test set is production: you ship once, and the crash rate you see is the one you report. If you peek at production metrics and rebuild until they look good, the number stops meaning "how this does on new users" and starts meaning "how well I tuned to these particular users".

<figure class="diagram" markdown>
![Animated milestone flow. A row of six boxes: 1 Split (seed 55, 60/20/20), 2 Standardize (fit on train, apply to validation and test), 3 Derive ((2/n) X transpose (Xw minus y), by hand on paper), 4 grad_check (relative error below 1e-7), 5 Train (mini-batch gradient descent, learning rate chosen on validation), 6 Test once (against the baseline and against lstsq). The active box turns yellow and finished boxes get a green check mark. Below, a chart of MSE on a log scale over 100 epochs with a grey dashed line at 0.98 for the validation baseline and a red dotted target line at 0.49, half of it. During step 5 the blue training curve and orange validation curve draw themselves, falling from about 11 and flattening by about epoch 25 near 0.11 and 0.13; during step 6 a red dot appears at the end labelled test 0.1085, 11.9% of the test baseline. A caption under the chart explains each step](../../assets/img/phase-1/day55-milestone-flow.svg)
<figcaption>Animated: six steps, each ticked only when done. The curves are the reference solution's real training run from this page.</figcaption>
</figure>

The deliverable is a folder that runs from scratch:

```text
month2-milestone/
├── README.md        what this is, how to run it, the headline numbers
├── regression.py    split_indices, fit_preprocess, transform, mse_and_grad, train_minibatch
├── checks.py        numerical_grad, grad_check, check_milestone()
├── run.py           the driver: split → preprocess → baseline → train → test once
├── derivation.md    your hand derivation of the gradient (template below)
├── figures/         learning curves, test predictions
└── report.md        the written report (template below)
```

Every Python block on this page also runs in your browser with ▶ Run, top to bottom, like notebook cells: a name defined in one block is available in the blocks below it. Do the real work on your machine, in that folder. The code was tested with Python 3.13, NumPy 2.5 and pandas 3.0.

!!! android "Android bridge"
    The statistics you learn in `fit_preprocess` (the median size, each column's mean and standard deviation, the list of categories) are **part of the model**, exactly as much as the weights. If you ship this model in an app, the Kotlin code that builds the input vector must apply the *same* numbers in the *same* column order, or the model silently receives inputs it has never seen. That mismatch is called **training-serving skew**, and it is one of the most common on-device ML bugs. LiteRT (TensorFlow Lite) model metadata has a `NormalizationOptions` entry with `mean` and `std` for exactly this reason. For a hand-rolled model like this one, have the training script write the statistics to a JSON asset (or generate a Kotlin constants file), and never re-type them by hand.

---

## Part 1: the problem and the gradient (Day 55)

### The spec

You will write these functions. Inside them, use only NumPy and pandas: no scikit-learn, no autodiff.

| Function | Input → output | Rules |
|---|---|---|
| `split_indices(n, seed)` | → `(train_idx, val_idx, test_idx)` | one seeded shuffle, cut 60 / 20 / 20; every row in exactly one set |
| `fit_preprocess(train_df)` | → `stats` (a dict) | fill values, category list, column means and stds, all from `train_df` only |
| `transform(df, stats)` | → `X` of shape `(n, d + 1)` | uses `stats` only (never refits); last column is all ones (the intercept) |
| `mse_and_grad(w, X, y)` | → `(loss, grad)` | your hand-derived formula; `grad.shape == w.shape` |
| `train_minibatch(X, y, lr, batch_size, epochs, seed, X_val, y_val)` | → `(w, history)` | starts from \( \mathbf{w} = \mathbf{0} \); reshuffles every epoch; records full-set MSE before epoch 1 and after each epoch |
| `check_milestone()` | → prints `all milestone checks passed` | every check is an `assert` |

Each term in this table is taught in the sections below before you need it. Two pass marks:

1. **Test MSE at most half the baseline's validation MSE**, where the baseline predicts the training mean for every app. You'll compute the baseline on the validation set and write down this target *before* training, so the test set is not touched until the very end.
2. **Your gradient-descent weights match `np.linalg.lstsq`'s** to within about \( 10^{-3} \). Day 53 showed this loss has only one minimum, so two correct methods must arrive at the same place.

### Setup

The same loading and cleaning as [Days 27–28](day-27-month1-milestone.md). `clean_apps` removes duplicate rows, tidies the category spelling, turns the install text into integers, turns "Varies with device" into `NaN` (a missing value), and parses the dates. In the browser, `BASE` downloads the file; on your machine, after downloading it from the [Datasets page](../../datasets.md), you can set `BASE = ""` and keep the CSV next to your script.

```python
import numpy as np
import pandas as pd
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

BASE = "https://manjulkadyan.github.io/ai-leaning/assets/data/"
raw = pd.read_csv(BASE + "play_store_apps.csv")


def clean_apps(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the Play Store apps table (Day 17). The input is not modified."""
    out = df.drop_duplicates(ignore_index=True)
    return out.assign(
        category=out["category"].str.strip().str.title(),
        installs=(out["installs"]
                  .str.replace(",", "", regex=False)
                  .str.replace("+", "", regex=False)
                  .astype("int64")),
        size_mb=out["size_mb"].replace("Varies with device", np.nan).astype("float64"),
        last_updated=pd.to_datetime(out["last_updated"], format="%Y-%m-%d"),
        min_android=out["min_android"].map(
            lambda v: f"{v:.1f}" if v < 9 else str(int(v))
        ).astype("str"),
    )


apps = clean_apps(raw)
print(raw.shape, apps.shape)
# (486, 12) (480, 12)
```

Six duplicate rows are gone: 480 apps remain.

### Step 1: choose a target

**What it is.** The **target** is the number the model predicts, one per app; in formulas it's \( y \). The **features** are the numbers the model is allowed to look at to make that prediction.

**Why it matters.** The target decides whether the project can succeed at all. Linear regression can only find a relationship that exists in the data. Pick a target the columns don't carry information about, and no amount of careful gradient descent will beat the baseline.

**Intuition first.** A good target is one where, if you showed a colleague a few columns of an app, they could make a sensible guess. "This app has a million installs; how many reviews does it have?" Somewhere in the tens of thousands, probably. "This app has a million installs; what is its star rating?" No idea. That gut check is what the correlations say too (the block in Step 2 computes them): the correlation between `rating` and every other column in this dataset is below 0.13 in absolute value, while `log10(reviews)` and `log10(installs)` have a correlation of 0.935. (Day 33 already found that installs and size barely predict rating.)

So the reference solution predicts **`log10(reviews)`**. A real use: estimate how much feedback a release will collect. Predicting `rating` is Exercise 7, an honest negative result.

**Why `log10`?** Reviews range from 1 to more than 2.5 million. On the raw scale, the MSE would be ruled by a handful of huge apps: one error of a million reviews outweighs every other app combined. `np.log10(x)` returns the power of 10 that gives \( x \): `log10(1000)` is 3, `log10(1_000_000)` is 6. On the log scale an error of 1 means "off by a factor of 10" whatever the app's size, which is how people judge these numbers anyway. Every review count is at least 1, so the logarithm is always defined here. It is not defined at 0, and NumPy doesn't raise an error there; it warns and returns minus infinity:

```python
print(np.log10(np.array([1.0, 10.0, 0.0])))
print(np.mean((np.log10(np.array([1.0, 10.0, 0.0])) - 1.0) ** 2))
print(np.log10(np.array([0.0]) + 1))
# RuntimeWarning: divide by zero encountered in log10
# [  0.   1. -inf]
# RuntimeWarning: divide by zero encountered in log10
# inf
# [0.]
```

(Each of the first two lines takes a log of 0, so each prints the warning; in a terminal, a file name and line number come in front of it.) A single `-inf` in the target turns the whole MSE into `inf`, and its gradient into `inf` or `nan`, so training breaks for every app because of one. If a column can contain zeros, the usual fix is `np.log10(x + 1)`, which maps 0 to 0 (for natural logarithms, `np.log1p(x)` computes \( \ln(1 + x) \) accurately).

### Step 2: choose features, and encode them as numbers

A linear model computes \( \hat{y} = w_1 x_1 + w_2 x_2 + \dots + b \) (Day 12's layer with one output), so every feature has to be a number. The reference uses six columns:

- `log_installs`: `log10` of the install bucket, for the same reason as the target.
- `size_mb`: the download size. 20 apps are missing it. They'll be filled with the **training** median (Step 4 explains why "training").
- `paid` and `has_ads`: yes/no columns turned into 1.0 and 0.0. `(apps["price_usd"] > 0)` is a column of `True`/`False`, and `.astype(float)` turns `True` into 1.0 and `False` into 0.0.
- `days_since_update`: days between `last_updated` and a **fixed** date, 2026-10-01. Subtracting two dates gives a "timedelta" (a duration), and `.dt.days` reads its length in whole days. A fixed date, not "today", keeps the table identical tomorrow.
- `category`: 8 text values, encoded as **one-hot** columns (below).

```python
SNAPSHOT = pd.Timestamp("2026-10-01")      # a fixed "today", so the table never changes

table = pd.DataFrame({
    "log_installs": np.log10(apps["installs"]),
    "size_mb": apps["size_mb"],                                  # 20 NaN, filled later
    "paid": (apps["price_usd"] > 0).astype(float),
    "has_ads": (apps["has_ads"] == "Yes").astype(float),
    "days_since_update": (SNAPSHOT - apps["last_updated"]).dt.days.astype(float),
    "category": apps["category"],
    "log_reviews": np.log10(apps["reviews"]),                    # the target
})
print(table.shape)
print(table.head(3).round(2).to_string())
print("missing sizes:", table["size_mb"].isna().sum())
# (480, 7)
#    log_installs  size_mb  paid  has_ads  days_since_update      category  log_reviews
# 0           4.7     47.7   1.0      0.0              233.0  Productivity         3.41
# 1           6.0     11.4   0.0      0.0              651.0  Productivity         4.81
# 2           6.0     26.8   0.0      1.0              130.0         Tools         4.74
# missing sizes: 20
```

Now the correlations quoted in Step 1. `table.drop(columns="category")` leaves the numeric columns (correlation needs numbers), `.assign(rating=...)` adds the rating, and `.corr()` returns the table of Pearson correlations (Day 46) between every pair of columns, skipping missing values pair by pair:

```python
corr = table.drop(columns="category").assign(rating=apps["rating"]).corr()
print(corr["rating"].round(3).to_string())
print("installs vs reviews:", round(corr.loc["log_installs", "log_reviews"], 3))
# log_installs         0.013
# size_mb              0.120
# paid                -0.034
# has_ads              0.042
# days_since_update    0.050
# log_reviews         -0.007
# rating               1.000
# installs vs reviews: 0.935
```

Rating's correlation with itself is 1, by definition; with everything else it is between −0.034 and 0.120. Reviews and installs, both on the log scale, have 0.935.

Row 0 has 50,000 installs (\( \log_{10} 50{,}000 = 4.70 \)) and 2,581 reviews (\( \log_{10} 2{,}581 = 3.41 \)).

#### One-hot encoding

**What it is.** One-hot encoding turns a category column into several 0/1 columns, one per category: "is this app in Finance?", "is it in Games?", and so on. Each row has a 1 in exactly one of them (hence "one hot").

**Why.** You can't multiply a weight by the word "Games". And numbering the categories (Education = 0, Finance = 1, Games = 2, ...) would be worse than useless: it tells the model that Games is "twice Finance" and that one weight times that number captures every category, which is nonsense. One-hot columns give each category its own weight, its own "bonus" added to the prediction.

**Kotlin comparison.** Think of a `when` over an enum, written as arithmetic: `bonus = when (category) { FINANCE -> wFinance; GAMES -> wGames; ...; EDUCATION -> 0.0 }`. One-hot columns implement exactly that `when` as a dot product: all the 0 columns contribute nothing, and the single 1 picks out its category's weight.

**Example.** For three apps in Productivity, Productivity and Tools, the seven columns (Education has none, see below) are:

| app | Finance | Games | Health & Fitness | Music & Audio | Photography | Productivity | Tools |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |

In code, one column is `(df["category"] == "Tools").to_numpy(dtype=float)`: compare every row with the string, get `True`/`False`, convert to 1.0/0.0.

**Underneath: why one category gets no column.** The model also has an intercept, a column of all ones. If all eight categories had a column, the eight would add up, row by row, to exactly that column of ones (every app is in exactly one category). Then the columns are **collinear** (Day 23): one is a combination of others, \( X^\top X \) is singular, and there are infinitely many equally good weight vectors (add 1 to every category weight and subtract 1 from the intercept: same predictions). So one category, the **reference category**, gets no column; here it's Education, the first alphabetically. Its "bonus" is folded into the intercept, and every other category's weight means "difference from Education".

**How it goes wrong.** Build the one-hot list separately for each split, and a category missing from the validation rows changes the number of columns: the training matrix has 13 columns and the validation matrix 12, and `X_val @ w` fails with `ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 13 is different from 12)`. Read it from the end: the weight vector has 13 entries but the matrix only 12 columns. The fix is the same as for every statistic in this project: learn the category list once, from the training rows, and reuse it.

The opposite case fails silently. An app whose category is not in the stored list (a category that appears only after training, say "Weather") gets zeros in all seven columns, which is exactly the encoding of Education, the reference category. The model treats it as an Education app and gives no warning: a quiet case of the training-serving skew from the Android bridge. Libraries handle this with an explicit "unknown" policy; see the link on encoding categorical features in "Going further".

**Underneath, in code.** Day 23's `np.linalg.matrix_rank` counts how many columns of a matrix are truly independent. Build the matrix with all eight category columns plus the column of ones, then again without the first category (this uses the whole table and a median fill, only to count columns):

```python
filled = table["size_mb"].fillna(table["size_mb"].median())
num = table[["log_installs", "size_mb", "paid", "has_ads", "days_since_update"]].assign(size_mb=filled).to_numpy()
all8 = np.column_stack([(table["category"] == c).to_numpy(dtype=float)
                        for c in sorted(table["category"].unique())])
X_all8 = np.column_stack([num, all8, np.ones(len(table))])
X_drop1 = np.column_stack([num, all8[:, 1:], np.ones(len(table))])
print(all8.sum(axis=1)[:5], X_all8.shape, np.linalg.matrix_rank(X_all8))
print(X_drop1.shape, np.linalg.matrix_rank(X_drop1))
# [1. 1. 1. 1. 1.] (480, 14) 13
# (480, 13) 13
```

Every row of the eight category columns sums to 1 (the first five are shown), so 14 columns hold only 13 independent ones: \( X^\top X \) is singular. Dropping one category leaves 13 columns of rank 13, and the least-squares answer is unique again.

!!! warning "Common mistake: a feature that is really the answer"
    A feature computed *from* the target, or only known after it, makes the model look brilliant and makes it useless. Suppose you add "reviews per install". It looks innocent, but \( \log_{10}(\text{reviews}/\text{installs}) = \log_{10}\text{reviews} - \log_{10}\text{installs} \): together with `log_installs`, it *is* the target. The model finds that out at once:

    ```python
    leaky = table.assign(reviews_per_install=np.log10(apps["reviews"] / apps["installs"]))
    print(np.allclose(leaky["reviews_per_install"] + leaky["log_installs"], leaky["log_reviews"]))
    # True
    ```

    (The block further down, after the split, trains on it and gets a test MSE of \( 2.6 \times 10^{-29} \): a perfect score, and worthless, because you need the reviews to compute the feature.) Ask of every feature: "would I know this at the moment I need the prediction?" Installs pass that test when you predict reviews for an app that is already live; they would not for an app that hasn't been released.

### Step 3: split into train, validation and test

**What it is.** Three disjoint sets of rows, each with one job:

| Set | Share | Rows here | Its only job |
|---|---|---|---|
| **train** | 60% | 288 | compute the gradient; compute every preprocessing statistic |
| **validation** | 20% | 96 | choose between settings (learning rate, epochs, which features) |
| **test** | 20% | 96 | estimate the final error on new apps, **once**, at the very end |

**Why three and not two.** A model's score on the rows it was fitted to is optimistic: the weights were chosen to fit those rows, noise included. So you score on rows it hasn't seen. But each time you *choose a setting* because it scored best on some set, that set's score becomes a little optimistic too: you picked the winner partly for its luck on those particular rows. That is Day 45's multiple-comparisons problem in another costume. Validation absorbs that optimism, so the test set stays a fair sample of "apps the model has never influenced" (Day 43), and its score estimates what happens on genuinely new apps.

**Kotlin comparison.** In Kotlin you'd write `val shuffled = rows.shuffled(Random(55))` and then `take`/`drop` slices. NumPy's version shuffles **row numbers** instead of rows: `rng.permutation(n)` returns the integers `0 .. n-1` in a random order, and slicing that array gives index arrays you use with `.iloc[...]` (pandas' "select by position") or `X[idx]` (NumPy). Same idea, and the fixed seed plays the role of `Random(55)`: same seed, same order, same split, same numbers in your report.

**Example.** With 7 rows and seed 0, the reference `split_indices` (which you'll write below) returns `(array([2, 4, 3, 6]), array([5]), array([0, 1]))`: `int(0.6 * 7)` is 4 training rows, `int(0.2 * 7)` is 1 validation row, and the test set gets the remaining 2. `int(...)` rounds toward zero, so the test set picks up any leftover rows. Exercise 1 checks the sizes for other values of \( n \).

**Underneath.** `order = rng.permutation(n)` is one shuffled array. Slicing it at two cut points, `order[:n_train]`, `order[n_train:n_train + n_val]` and `order[n_train + n_val:]`, gives three pieces that can't overlap and together contain every row once: they are consecutive pieces of one array.

**How it goes wrong.** Two classic bugs. Shuffling twice (once for train, once for test) gives sets that overlap, so some test apps were also training apps and the test score is optimistic. And splitting *after* sorting by something (say, by install count) without shuffling gives a test set of only the biggest apps. Here is the first bug, with two separate shuffles:

```python
rng_a, rng_b = np.random.default_rng(1), np.random.default_rng(2)
tr_bad = rng_a.permutation(480)[:288]          # "train": first 288 of one shuffle
te_bad = rng_b.permutation(480)[384:]          # "test": last 96 of a DIFFERENT shuffle
print("overlap:", len(np.intersect1d(tr_bad, te_bad)), "of", len(te_bad), "test rows")
# overlap: 55 of 96 test rows
```

`np.intersect1d(a, b)` returns the values present in both arrays. 55 of the 96 "test" apps were also training apps, so the test score would partly measure memory rather than generalization. Nothing crashes; only a check finds it. `check_milestone()` below asserts that every row appears in exactly one set.

### Step 4: fit on train, transform everything

**What it is.** **Preprocessing** is everything that turns raw columns into the matrix \( X \): filling missing sizes, one-hot encoding, standardizing. Each step needs some numbers learned from data (the median size, the list of categories, each column's mean and standard deviation). **Fitting** learns those numbers; **transforming** applies them. The rule: fit on the training rows only, then transform every set with the same numbers.

**Standardizing**, from Day 22, rescales each feature column to mean 0 and standard deviation 1:

\[
z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}
\]

- \( x \) is the raw value, for example an app's size in MB.
- \( \mu_{\text{train}} \) ("mu") is that column's mean over the **training** rows.
- \( \sigma_{\text{train}} \) ("sigma") is that column's standard deviation over the same rows. In NumPy, `F.std(axis=0, ddof=1)` computes one per column; `ddof=1` divides by \( n - 1 \), Bessel's correction from Day 43.
- \( z \) is the standardized value: "how many standard deviations above the training average".

**Worked example by hand.** Suppose the training sizes had mean 40.0 and standard deviation 30.0. A 100 MB app becomes \( z = (100 - 40)/30 = 2.0 \): two standard deviations bigger than a typical training app. A 25 MB app becomes \( (25 - 40)/30 = -0.5 \). A *validation* app of 100 MB also becomes 2.0, because it is converted with the training mean and standard deviation, not its own set's.

**Why standardize at all?** For gradient descent, not for the final answer. Day 33 showed that a raw `size_mb` column (values around 40) makes its gradient entry thousands of times bigger than the others, and Day 52 showed why one learning rate can't then suit every weight. Standardized columns all have the same scale, so one learning rate works for all of them.

**Why fit on train only? Intuition.** The validation and test sets stand in for apps that don't exist yet. When the model is used for real, those future apps can't contribute to the median, the means or the category list: the statistics were frozen when you trained. To simulate that honestly, the held-out rows must not contribute either.

<figure class="diagram" markdown>
![Animated diagram in three steps. On the left, three boxes: train with 288 apps, validation with 96 apps, test with 96 apps. On the right, three boxes each labelled transform(df, stats), producing X_train, X_val and X_test, with dashed grey arrows carrying each set's rows across. Above them, in the middle, a yellow box named stats lists size median 31.35, 7 category columns, 12 means and 12 stds. Step 1: a blue arrow labelled fit runs from train into stats. Step 2: three arrows run from stats to the three transform boxes. Step 3: a red dashed arrow from test to stats appears with a red cross over it and the warning never fit on validation or test rows: that is leakage. A caption under the diagram explains each step](../../assets/img/phase-1/day55-fit-transform.svg)
<figcaption>Animated: statistics flow out of the training set only, then the same numbers transform all three sets.</figcaption>
</figure>

**Kotlin comparison.** It's a configuration object built once and injected everywhere. `fit_preprocess` is the builder: it returns a plain `dict` (Python's `Map<String, Any>`) holding everything learned. `transform` is a pure function of `(rows, stats)`: it never looks at anything else, so it gives the same answer for an app whether it is transformed alone or with 95 others.

**How it goes wrong.** If `transform` quietly refits (computes means and standard deviations from the rows it was given), it fails in production, where it is called with one app at a time: the standard deviation of a single row is undefined. Exercise 5 asks you to predict exactly what happens. That's why `check_milestone()` transforms a single row on purpose.

### Step 5: the baseline and the pass mark

**What it is.** The **baseline** is the simplest possible model: predict the training mean of the target for every app, ignoring every feature.

**Why.** A score means nothing alone. "Test MSE 0.11" is excellent if the trivial model gets 0.9, and terrible if it gets 0.12. The baseline is also exactly what linear regression returns when all its feature weights are zero, so beating it by a clear margin means "the features carry real information".

**Worked example by hand.** Training targets 2, 3, 4, 7 have mean \( 16/4 = 4 \), so the baseline predicts 4 for everything. On test targets 3 and 5, the errors are \( 4 - 3 = 1 \) and \( 4 - 5 = -1 \), and the MSE is \( (1^2 + (-1)^2)/2 = 1 \). Why the mean and not, say, the median? The mean is the single number with the lowest training MSE (set the derivative of \( \frac{1}{n}\sum_i (c - y_i)^2 \) with respect to \( c \) to zero and you get \( c = \bar{y} \)). Exercise 2 checks it numerically.

**The pass mark.** Decide it from the baseline *before* training, and write it in your report. Choosing it afterwards means choosing a target you already know you hit. The reference uses half the baseline's **validation** MSE: the features must remove at least half of the squared error that the mean leaves. Using the validation set for this keeps the test set closed until the single, final evaluation; the baseline's test MSE is computed only then, as the comparison for the result. That's a sensible bar when one feature is known to be strong; for `rating` (Exercise 7) you'd have to set a far humbler one.

### Step 6: derive the gradient by hand

**What it is.** The gradient \( \nabla L(\mathbf{w}) \) is the vector of partial derivatives of the loss with respect to each weight (Day 32): it points uphill, so gradient descent steps the other way.

**Why derive it again?** You derived this on [Day 33](day-33-vector-matrix-gradients.md). The milestone asks you to do it from a blank page, because a derivation you can reproduce is a derivation you own, and because every model you'll meet later (logistic regression in Month 3, neural networks in Month 4) is trained by exactly this chain of reasoning, done by a library.

**Intuition first.** Picture each app pulling on the weights. An app whose prediction is too high (positive residual) pulls the weights down, in proportion to how far off it is and how big its features are: a feature that was 0 for that app played no part in the error, so it gets no blame. The gradient is the average of all those pulls, times 2.

**The symbols, one at a time.**

- \( n \): the number of apps (rows).
- \( X \): the \( n \times (d+1) \) feature matrix, one row per app. Row \( i \) is the vector \( \mathbf{x}_i \), and the entry in row \( i \), column \( j \) is \( X_{ij} \). The last column is all ones.
- \( \mathbf{w} \): the \( d + 1 \) weights. \( w_j \) multiplies column \( j \); the last one is the intercept.
- \( \mathbf{y} \): the \( n \) targets.
- \( \hat{y}_i = \mathbf{x}_i \cdot \mathbf{w} \): the prediction for app \( i \) (the hat means "estimated").
- \( r_i = \hat{y}_i - y_i \): the **residual**, how far the prediction is from the truth, with a sign.

The loss is the mean of the squared residuals:

\[
L(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^{n} r_i^2 = \frac{1}{n}\lVert X\mathbf{w} - \mathbf{y} \rVert^2
\]

Copy this template into `derivation.md` and fill in every blank without looking at Day 33:

```markdown
# Gradient of the MSE for linear regression

1. One residual, written with indices:      r_i = ________________________
2. The loss in terms of the residuals:      L(w) = ________________________
3. Pick ONE weight w_j. How r_i moves:      ∂r_i/∂w_j = ________   (only one term of r_i contains w_j)
4. Chain rule (Day 31) for one term:        ∂(r_i²)/∂w_j = ____ · ____
5. Sum over the apps (sum rule, Day 30):    ∂L/∂w_j = (1/n) Σ_i ________ = (2/n) Σ_i ________
6. Recognize a matrix product:              Σ_i X_ij r_i is entry j of ________
7. Pack all j together:                     ∇L(w) = ________________________
8. Shape check (Day 33's shape rule):       X is (n, d+1), r is (n,), so ∇L is ________ = shape of w ✓
9. Hand-worked numbers:                     X = [[1, 1], [2, 1]], y = [1, 3], w = [0, 0]
                                            r = ____, L = ____, ∇L = ____
```

??? success "The completed derivation"
    1. \( r_i = \sum_{j} X_{ij} w_j - y_i \)
    2. \( L(\mathbf{w}) = \frac{1}{n}\sum_i r_i^2 \)
    3. \( \partial r_i / \partial w_j = X_{ij} \): every other term of the sum is constant in \( w_j \), and \( y_i \) doesn't depend on \( \mathbf{w} \).
    4. \( \partial (r_i^2)/\partial w_j = 2 r_i \cdot X_{ij} \): outer function \( u^2 \), slope \( 2u \); inner function \( r_i \), slope \( X_{ij} \).
    5. \( \partial L/\partial w_j = \frac{1}{n}\sum_i 2 r_i X_{ij} = \frac{2}{n}\sum_i X_{ij} r_i \)
    6. \( \sum_i X_{ij} r_i \) is the dot product of column \( j \) of \( X \) with \( \mathbf{r} \), which is entry \( j \) of \( X^\top \mathbf{r} \).
    7. \( \nabla L(\mathbf{w}) = \frac{2}{n} X^\top (X\mathbf{w} - \mathbf{y}) \)
    8. \( (d+1, n) \) times \( (n,) \) gives \( (d+1,) \), the shape of \( \mathbf{w} \).
    9. \( \mathbf{r} = [0 - 1,\; 0 - 3] = [-1, -3] \), \( L = (1 + 9)/2 = 5 \), \( X^\top\mathbf{r} = [1\cdot(-1) + 2\cdot(-3),\; -1 - 3] = [-7, -4] \), and \( \nabla L = \frac{2}{2}[-7, -4] = [-7, -4] \). Both entries are negative: raise the slope and the intercept, since the line \( y = 0 \) sits below both points.

Now reproduce step 9 in code. First, Day 33's two helpers, unchanged. `numerical_grad` nudges one weight at a time by \( \pm h \) and measures the slope with a central difference (Day 29); `np.ndindex(w.shape)` yields every index of an array of that shape, so it works for vectors and matrices alike. `grad_check` returns the **relative error** between an analytic gradient \( \mathbf{g}_a \) and the numerical one \( \mathbf{g}_n \), \( \lVert \mathbf{g}_a - \mathbf{g}_n \rVert / (\lVert \mathbf{g}_a \rVert + \lVert \mathbf{g}_n \rVert) \): 0 means identical, and below \( 10^{-7} \) means correct.

```python
def numerical_grad(f, w, h=1e-5):
    """Central-difference gradient of a scalar function f at w. Works for any shape of w."""
    w = np.array(w, dtype=float)          # a float64 copy we are free to poke
    grad = np.zeros_like(w)               # the shape rule: same shape as w
    for idx in np.ndindex(w.shape):       # (0,), (1,), ... or (i, j) for a matrix
        old = w[idx]
        w[idx] = old + h
        f_plus = f(w)
        w[idx] = old - h
        f_minus = f(w)
        w[idx] = old                      # put it back before the next entry
        grad[idx] = (f_plus - f_minus) / (2 * h)
    return grad


def grad_check(f, w, h=1e-5):
    """f(w) must return (loss, analytic_gradient).
    Returns the relative error between that gradient and a central-difference one."""
    _, g_analytic = f(w)
    g_numeric = numerical_grad(lambda v: f(v)[0], w, h)
    diff = np.linalg.norm(g_analytic - g_numeric)
    size = np.linalg.norm(g_analytic) + np.linalg.norm(g_numeric)
    return diff / size if size > 0 else 0.0
```

```python
X_tiny = np.array([[1.0, 1.0],
                   [2.0, 1.0]])           # one feature + the column of ones
y_tiny = np.array([1.0, 3.0])
w_tiny = np.zeros(2)

r = X_tiny @ w_tiny - y_tiny             # residuals
print(r, r @ r / 2, 2 / 2 * X_tiny.T @ r)
print(numerical_grad(lambda w: np.mean((X_tiny @ w - y_tiny) ** 2), w_tiny))
# [-1. -3.] 5.0 [-7. -4.]
# [-7. -4.]
```

The residuals \( [-1, -3] \), the loss 5 and the gradient \( [-7, -4] \) match the hand result, and the numerical gradient agrees. `r @ r` is the dot product of `r` with itself, the sum of squared residuals.

**How it goes wrong.** Remember Day 33's fingerprints when `grad_check` fails. If your gradient is the right one times a constant \( c \), the relative error is \( |c - 1|/(|c| + 1) \): a clean 0.333 means a factor of 2 (\( c = \tfrac{1}{2} \)), a clean 1.0 means a flipped sign (\( c = -1 \)). Exercise 6 has a bug whose fingerprint is 0.8.

### Starter code and the checks

Copy this skeleton into `regression.py` and fill in the `TODO`s. `NotImplementedError` is Python's `TODO()`: like Kotlin's `TODO()`, it throws when called, so you can't forget a stub.

```python
NUMERIC = ["log_installs", "size_mb", "paid", "has_ads", "days_since_update"]


def split_indices(n, seed, fractions=(0.6, 0.2, 0.2)):
    """Shuffle 0..n-1 with a seeded rng and cut into (train, val, test) index arrays."""
    # TODO: rng = np.random.default_rng(seed); order = rng.permutation(n); cut at 60% and 80%
    raise NotImplementedError


def fit_preprocess(train_df):
    """Learn EVERYTHING the features need from the training rows only:
    the median size (to fill NaN), the category list, and each column's mean and std."""
    # TODO: return a dict of those statistics
    raise NotImplementedError


def transform(df, stats):
    """Turn any rows (train, val or test) into X of shape (n, d + 1), using stats only.
    Columns: standardized features, then a final column of ones for the intercept."""
    # TODO: fill size NaN with stats' median, one-hot the categories, standardize, append ones
    raise NotImplementedError


def mse_and_grad(w, X, y):
    """Return (L, grad): L = (1/n)·||Xw − y||², and its gradient, shape like w."""
    # TODO: your hand-derived formula
    raise NotImplementedError


def train_minibatch(X, y, lr, batch_size, epochs, seed, X_val=None, y_val=None):
    """Mini-batch gradient descent from w = 0. Returns (w, history), where
    history["train"] (and "val", if given) hold the full-set MSE before epoch 1 and after each epoch."""
    # TODO: each epoch, shuffle with a seeded rng, walk the batches, step w -= lr * grad
    raise NotImplementedError
```

The checks are your spec in executable form, as on Day 27. Each one targets a mistake people actually make. A few pieces of syntax in it, explained before you read it:

- `set(tr) | set(va) | set(te)`: `set(...)` builds a set of the row numbers and `|` is set union (Kotlin's `union`). If the union has 100 members and the three sizes add up to 100, no row is in two sets.
- `np.array_equal(a, b)`: `True` when two arrays have the same shape and the same elements.
- `table.iloc[:200]` selects the first 200 rows by position; `table.iloc[[300]]`, with a list inside, selects row 300 as a one-row DataFrame.
- `np.isfinite(a).all()`: `True` when no entry is `NaN` or infinite.
- `try: ... except NotImplementedError: ...` is Kotlin's `try { } catch (e: NotImplementedError) { }`. It lets this block print a message instead of crashing while the stubs are unfinished.

```python
def check_milestone():
    # 1. the split: right sizes, no overlap, reproducible
    tr, va, te = split_indices(100, seed=1)
    assert (len(tr), len(va), len(te)) == (60, 20, 20)
    assert len(set(tr) | set(va) | set(te)) == 100                 # every row exactly once
    assert np.array_equal(tr, split_indices(100, seed=1)[0])       # same seed, same split
    # 2. preprocessing: statistics from the training rows, applied unchanged elsewhere
    stats = fit_preprocess(table.iloc[:200])
    X = transform(table.iloc[:200], stats)
    assert np.all(X[:, -1] == 1) and not np.isnan(X).any()
    assert np.allclose(X[:, :-1].mean(axis=0), 0)                  # centred on the TRAIN rows
    one_row = transform(table.iloc[[300]], stats)                  # a refit would divide by 0 here
    assert one_row.shape == (1, X.shape[1]) and np.isfinite(one_row).all()
    # 3. the gradient: right shape, and agrees with central differences
    rng = np.random.default_rng(0)
    Xs, ys = rng.normal(size=(30, 4)), rng.normal(size=30)
    for _ in range(3):
        w = rng.normal(size=4)
        assert mse_and_grad(w, Xs, ys)[1].shape == w.shape
        assert grad_check(lambda v: mse_and_grad(v, Xs, ys), w) < 1e-7
    # 4. training: the loss falls and ends near the least-squares minimum
    w_gd, hist = train_minibatch(Xs, ys, lr=0.05, batch_size=8, epochs=200, seed=0)
    L_gd = mse_and_grad(w_gd, Xs, ys)[0]
    L_best = mse_and_grad(np.linalg.lstsq(Xs, ys)[0], Xs, ys)[0]
    assert len(hist["train"]) == 201 and hist["train"][-1] < hist["train"][0]
    assert L_gd < 1.02 * L_best
    print("all milestone checks passed")


try:
    check_milestone()   # raises NotImplementedError until you fill in the TODOs: that's the point
except NotImplementedError:
    print("not implemented yet")
# not implemented yet
```

What each check catches:

1. **The split**: wrong sizes, overlapping sets, or a split that changes between runs.
2. **Preprocessing**: a missing intercept column, `NaN` left in the matrix, columns not centred on the *fitting* rows, and (the single-row test) a `transform` that secretly refits.
3. **The gradient**: a wrong shape, a wrong factor or a wrong sign, checked at three random points on random data.
4. **Training**: a loop that doesn't record history correctly, or never gets near the bottom. `L_best` is the lowest loss possible, found by `np.linalg.lstsq` (Day 25); mini-batch descent must come within 2% of it.

Make the checks pass. Then **break `mse_and_grad` on purpose** (drop the 2) and watch check 3 fail. If you're stuck on one function for more than 20 minutes, re-read the matching lesson before opening the solution.

??? success "Reference solution: the functions (get your checks green first)"
    ```python
    def split_indices(n, seed, fractions=(0.6, 0.2, 0.2)):
        """Shuffle 0..n-1 with a seeded rng and cut into (train, val, test) index arrays."""
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
        n_train = int(fractions[0] * n)
        n_val = int(fractions[1] * n)
        return order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]


    def _raw_features(df, stats):
        """Numeric columns (size NaN filled) + one-hot categories, before standardizing."""
        num = df[NUMERIC].assign(size_mb=df["size_mb"].fillna(stats["size_median"]))
        onehot = [(df["category"] == c).to_numpy(dtype=float) for c in stats["categories"]]
        return np.column_stack([num.to_numpy(dtype=float)] + onehot)


    def fit_preprocess(train_df):
        """Learn EVERYTHING the features need from the training rows only."""
        stats = {
            "size_median": train_df["size_mb"].median(),
            "categories": sorted(train_df["category"].unique())[1:],   # drop one: it's the baseline
        }
        F = _raw_features(train_df, stats)
        std = F.std(axis=0, ddof=1)
        stats["mean"] = F.mean(axis=0)
        stats["std"] = np.where(std > 0, std, 1.0)                     # a constant column stays 0
        stats["names"] = NUMERIC + [f"cat={c}" for c in stats["categories"]]
        return stats


    def transform(df, stats):
        """Turn any rows into X of shape (n, d + 1) using the stored statistics only."""
        Z = (_raw_features(df, stats) - stats["mean"]) / stats["std"]
        return np.column_stack([Z, np.ones(len(df))])


    def mse_and_grad(w, X, y):
        """Return (L, grad): L = (1/n)·||Xw − y||², grad = (2/n)·Xᵀ(Xw − y)."""
        r = X @ w - y
        n = len(y)
        return r @ r / n, 2 / n * X.T @ r


    def train_minibatch(X, y, lr, batch_size, epochs, seed, X_val=None, y_val=None):
        """Mini-batch gradient descent from w = 0. Returns (w, history)."""
        rng = np.random.default_rng(seed)
        w = np.zeros(X.shape[1])
        history = {"train": [], "val": []}

        def record():
            history["train"].append(mse_and_grad(w, X, y)[0])
            if X_val is not None:
                history["val"].append(mse_and_grad(w, X_val, y_val)[0])

        record()                                     # epoch 0: the starting point
        for _ in range(epochs):
            order = rng.permutation(len(y))          # reshuffle every epoch (Day 52)
            for start in range(0, len(y), batch_size):
                batch = order[start:start + batch_size]
                _, g = mse_and_grad(w, X[batch], y[batch])
                w = w - lr * g
            record()
        return w, history


    check_milestone()
    # all milestone checks passed
    ```

    **Line by line, the parts that are new.**

    - `_raw_features` builds the unstandardized matrix. `df[NUMERIC]` selects the five numeric columns; `.assign(size_mb=...fillna(stats["size_median"]))` returns a copy with the missing sizes replaced by the *stored* median. The list comprehension makes one 0/1 array per stored category. `np.column_stack([...])` glues 1-D arrays (and 2-D blocks) side by side as columns, so the result has 5 + 7 = 12 columns.
    - `sorted(train_df["category"].unique())[1:]`: `.unique()` lists the distinct categories, `sorted` puts them in alphabetical order, and `[1:]` drops the first (Education), the reference category.
    - `np.where(std > 0, std, 1.0)` is an element-wise `if`: keep `std` where it's positive, use 1.0 where it's 0. A category absent from the training rows would have a column of all zeros and a standard deviation of 0; dividing by 1.0 leaves that column at 0 instead of producing `NaN`.
    - `transform` subtracts the stored means and divides by the stored standard deviations, both arrays of 12 numbers, broadcast over every row (Day 9), then appends the column of ones with `np.ones(len(df))`.
    - `train_minibatch` defines `record()` inside itself. A nested function can read `w` from the enclosing function (a closure, like a Kotlin lambda capturing a local); because `w = w - lr * g` creates a new array each step, `record()` always sees the current `w`. The last batch of an epoch would be smaller when 32 doesn't divide the row count (288 = 9 × 32 does); `mse_and_grad` divides by `len(y)` of the batch, so a short batch still gives an average, not a sum.

    **Design notes.** `fit_preprocess` returns a plain dict, so everything the model needs besides `w` is visible, and easy to save as JSON for an app (the Android bridge above). `_raw_features` is shared by fitting and transforming, so the two can never build the columns in different orders.

    **The complete `run.py`** is the setup block, `table`, the two helpers, these functions, and the Part 2 driver blocks below, in page order. Every block on this page runs in that order, which is how the numbers below were produced.

---

## Part 2: train, evaluate, report (Day 56)

### Milestone 1: split, preprocess, baseline

```python
train_idx, val_idx, test_idx = split_indices(len(table), seed=55)
train, val, test = table.iloc[train_idx], table.iloc[val_idx], table.iloc[test_idx]

stats = fit_preprocess(train)                     # training rows ONLY
X_train, X_val, X_test = (transform(d, stats) for d in (train, val, test))
y_train, y_val, y_test = (d["log_reviews"].to_numpy() for d in (train, val, test))

print(X_train.shape, X_val.shape, X_test.shape)
print("median size from train:", stats["size_median"])
print("train feature means:", X_train[:, :5].mean(axis=0).round(3) + 0.0)   # + 0.0 turns -0.0 into 0.0
print("val   feature means:", X_val[:, :5].mean(axis=0).round(3))
# (288, 13) (96, 13) (96, 13)
# median size from train: 31.35
# train feature means: [0. 0. 0. 0. 0.]
# val   feature means: [-0.12  -0.083 -0.023  0.022 -0.024]
```

`X_train, X_val, X_test = (transform(d, stats) for d in (train, val, test))` is a **generator expression** (a lazy comprehension, like a Kotlin `Sequence`) unpacked into three names: it calls `transform` three times, once per DataFrame, in order. The `+ 0.0` is a small trick: rounding a tiny negative number like \( -10^{-17} \) gives `-0.0`, which prints as `-0.`; adding 0.0 turns it into `0.0`.

13 columns: 5 numeric features, 7 one-hot categories and the column of ones. The training means are exactly 0 and the validation means aren't: that's what "fit on train only" looks like, and it is correct. Validation apps happen to have slightly fewer installs than training apps (0.12 standard deviations fewer on average), as random samples do (Day 43).

**Saving the statistics for an app** (the Android bridge). `stats` holds NumPy arrays, and Python's `json` module refuses them; `.tolist()` converts an array (or a NumPy number) into plain Python lists and floats that JSON accepts:

```python
import json

try:
    json.dumps(stats)
except TypeError as e:
    print("TypeError:", e)
stats_json = json.dumps({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in stats.items()})
print(len(stats_json), stats_json[:90])
# TypeError: Object of type ndarray is not JSON serializable
# 806 {"size_median": 31.35, "categories": ["Finance", "Games", "Health & Fitness", "Music & Aud
```

`hasattr(v, "tolist")` asks whether `v` has a `tolist` method (arrays and NumPy numbers do, plain lists don't), and the dict comprehension builds a converted copy. The 806-character string is what you'd ship as an asset, next to the weights.

```python
TARGET_RATIO = 0.5                       # pass mark: at most half the baseline's validation MSE

baseline = y_train.mean()                # the "model" that ignores every feature
baseline_val = np.mean((baseline - y_val) ** 2)
TARGET = TARGET_RATIO * baseline_val     # written down now, before any training
print(f"baseline prediction: {baseline:.3f}  (10^{baseline:.3f} = {10**baseline:,.0f} reviews)")
print(f"baseline MSE on validation: {baseline_val:.4f}")
print(f"target: test MSE <= {TARGET:.4f}")
# baseline prediction: 3.237  (10^3.237 = 1,724 reviews)
# baseline MSE on validation: 0.9843
# target: test MSE <= 0.4922
```

The format specs inside the f-strings: `:.3f` prints a float with 3 decimals, `:.4f` with 4, and `:,.0f` with no decimals and a comma every three digits. `**` is Python's power operator, so `10**baseline` is \( 10^{3.237} \).

The baseline predicts 1,724 reviews for every app. That is the *geometric* mean of the training apps' review counts (the mean of the logs, converted back), far below their ordinary mean of 20,511, which a few huge apps drag up. Its validation MSE is 0.9843, so its root-mean-square miss is the square root of that, 0.992 in log10 units: off by a factor of 9.8 (\( 10^{0.992} \)).

!!! warning "Common mistake: standardizing before splitting"
    It's tempting to standardize the whole table and split afterwards. Then the test rows help compute the means, standard deviations and fill values the model is trained with: information about the test set has **leaked** into training. Here is the leaky version next to the correct one:

    ```python
    stats_all = fit_preprocess(table)                       # WRONG: every row, test included
    Xb_train, Xb_test = transform(train, stats_all), transform(test, stats_all)
    w_b = np.linalg.lstsq(Xb_train, y_train)[0]
    w_c = np.linalg.lstsq(X_train, y_train)[0]               # the correct, train-only version
    print("size median, all rows vs train:", stats_all["size_median"], stats["size_median"])
    print("train means of first 3 columns:", Xb_train[:, :3].mean(axis=0).round(3))
    print(f"test MSE leaky {mse_and_grad(w_b, Xb_test, y_test)[0]:.6f}  clean {mse_and_grad(w_c, X_test, y_test)[0]:.6f}")
    # size median, all rows vs train: 30.950000000000003 31.35
    # train means of first 3 columns: [ 0.048  0.01  -0.011]
    # test MSE leaky 0.108581  clean 0.108581
    ```

    Two lessons in that output. First, the **symptom**: the training columns' means are no longer exactly 0, and the fill value changed. That's how you spot the leak. Second, for *this* model the test score didn't move: linear regression with an intercept gives the same predictions whatever shift and scale you apply to a column (the weights absorb it), only 4 test apps had a missing size, and the size weight is almost zero (−0.003, Milestone 6), so changing 20 apps' filled size by 0.4 MB moves no prediction visibly. Don't conclude the rule doesn't matter. With models that compare distances (k-nearest neighbours), penalize weights (ridge, Exercise 3), or fill values computed from the target, the leak changes results, and always in the optimistic direction. Follow the rule every time, so you never have to work out whether this time it mattered.

Now the leaky-feature example from Step 2, trained properly on these splits:

```python
leaky = table.assign(reviews_per_install=np.log10(apps["reviews"] / apps["installs"]))
cols = NUMERIC + ["reviews_per_install"]
F_tr = leaky.iloc[train_idx][cols].fillna(stats["size_median"]).to_numpy()
F_te = leaky.iloc[test_idx][cols].fillna(stats["size_median"]).to_numpy()
mu, sd = F_tr.mean(axis=0), F_tr.std(axis=0, ddof=1)
A_tr = np.column_stack([(F_tr - mu) / sd, np.ones(len(F_tr))])
A_te = np.column_stack([(F_te - mu) / sd, np.ones(len(F_te))])
w_leak = np.linalg.lstsq(A_tr, y_train)[0]
print(f"test MSE with the leaky feature: {mse_and_grad(w_leak, A_te, y_test)[0]:.2e}")
# test MSE with the leaky feature: 2.59e-29
```

`:.2e` prints in scientific notation with 2 decimals: `2.59e-29` means \( 2.59 \times 10^{-29} \), zero up to rounding. A perfect test score is not a triumph: it is the loudest sign of a leak.

### Milestone 2: the gradient on the real data

Day 33 checked the formula on toy numbers. Check it again on the actual matrix you will train with, at the starting point and at a random point:

```python
rng = np.random.default_rng(55)
for w in (np.zeros(X_train.shape[1]), rng.normal(size=X_train.shape[1])):
    err = grad_check(lambda v: mse_and_grad(v, X_train, y_train), w)
    print(f"relative error {err:.1e}")
# relative error 1.9e-11
# relative error 3.0e-11
```

Both far below \( 10^{-7} \): correct. (The last digits can differ on your machine, because the exact rounding depends on the order the CPU adds numbers in.)

!!! warning "Common mistake: a target with shape (n, 1)"
    `train[["log_reviews"]]` with double brackets selects a one-column **DataFrame**, and `.to_numpy()` turns it into a 2-D array of shape `(288, 1)`, not the 1-D `(288,)` your functions expect. Broadcasting (Day 9) then turns `X @ w - y`, which is `(288,)` minus `(288, 1)`, into a `(288, 288)` matrix, and the "gradient" comes out with shape `(13, 32)` on a batch of 32. The first update fails:

    ```python
    y_col = train[["log_reviews"]].to_numpy()          # double brackets: a one-column DataFrame
    print(y_col.shape)
    try:
        train_minibatch(X_train, y_col, 0.01, 32, 1, seed=55)
    except ValueError as e:                            # fails on purpose
        print("ValueError:", e)
    # (288, 1)
    # ValueError: operands could not be broadcast together with shapes (13,) (13,32)
    ```

    Recognize it by a shape in the error that contains the batch size where you expected only the number of weights. Fix it with single brackets, `train["log_reviews"].to_numpy()`, or `y_col.ravel()`, and make the bug impossible to miss: add `assert grad.shape == w.shape` inside `mse_and_grad`.

### Milestone 3: train, and choose the learning rate on validation

**What it is.** **Mini-batch gradient descent** (Day 52) repeats one step, \( \mathbf{w} \leftarrow \mathbf{w} - \eta \nabla L_{\text{batch}}(\mathbf{w}) \), where \( \eta \) ("eta") is the **learning rate** and \( \nabla L_{\text{batch}} \) is the gradient computed on a small random group of rows, a **batch**. One pass over all training rows, batch by batch, is an **epoch**. Before each epoch the rows are reshuffled, so the batches differ every time.

**Why batches.** A batch gradient is a noisy but unbiased estimate of the full gradient (Day 39), and it's far cheaper: with a million rows you take many useful steps for the cost of one full-gradient step. Here, with 288 rows, the full gradient would also be cheap; you use batches because every model you train from Month 4 on is trained this way.

**Underneath: a trace of the first epoch.** `range(0, 288, 32)` yields the batch starts 0, 32, ..., 256: nine batches of 32. This block replays the first epoch of the training loop by hand, with the same seed, printing a few batches:

```python
EPOCHS, BATCH = 100, 32
rng_trace = np.random.default_rng(55)
order = rng_trace.permutation(len(y_train))
w_trace = np.zeros(X_train.shape[1])
for k, start in enumerate(range(0, len(y_train), BATCH)):
    batch = order[start:start + BATCH]
    L_b, g_b = mse_and_grad(w_trace, X_train[batch], y_train[batch])
    w_trace = w_trace - 0.01 * g_b
    if k < 3 or k == 8:
        print(f"batch {k}: {len(batch)} rows, batch loss {L_b:.3f}, intercept now {w_trace[-1]:.4f}")
print("full-train MSE after one epoch:", round(mse_and_grad(w_trace, X_train, y_train)[0], 3))
# batch 0: 32 rows, batch loss 9.908, intercept now 0.0602
# batch 1: 32 rows, batch loss 11.901, intercept now 0.1247
# batch 2: 32 rows, batch loss 10.261, intercept now 0.1854
# batch 8: 32 rows, batch loss 9.163, intercept now 0.5398
# full-train MSE after one epoch: 7.908
```

Read it step by step. Batch 0's loss is 9.908: the weights are all 0, so every prediction is 0 and the loss is the mean squared target of those 32 apps. The intercept's gradient entry is \( \frac{2}{32}\sum_i (0 - y_i) = -2\bar{y}_{\text{batch}} \), which here is \( -6.02 \) (twice the batch's mean target of 3.01), so the step \( -0.01 \times (-6.02) \) lifts the intercept from 0 to 0.0602. The batch losses jump around (11.901, then 10.261) because each batch holds different apps: that's the noise of mini-batches. After nine steps the intercept is 0.54 and the full training MSE has fallen from 11.359 to 7.908, which is exactly the value `train_minibatch` records after epoch 1 below (same seed, same shuffle).

**Choosing the learning rate.** Train once per candidate learning rate, all with the same seed, batches of 32 and 100 epochs, and compare the **validation** MSE at the end. The test set stays closed.

```python
learning_rates = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
runs = {}
for lr in learning_rates:
    with np.errstate(over="ignore", invalid="ignore"):   # lr = 1.0 blows up, on purpose
        w, hist = train_minibatch(X_train, y_train, lr, BATCH, EPOCHS, seed=55,
                                  X_val=X_val, y_val=y_val)
    runs[lr] = hist
    print(f"lr {lr:<6} train {hist['train'][-1]:10.4g}   val {hist['val'][-1]:10.4g}")

best_lr = min(learning_rates, key=lambda lr: np.nan_to_num(runs[lr]["val"][-1], nan=np.inf))
print("best on validation:", best_lr)
# lr 0.003  train     0.1074   val     0.1324
# lr 0.01   train     0.1071   val     0.1308
# lr 0.03   train     0.1072   val      0.131
# lr 0.1    train      0.109   val     0.1343
# lr 0.3    train      0.122   val     0.1641
# lr 1.0    train        nan   val        nan
# best on validation: 0.01
```

New pieces of code, one at a time:

- `with np.errstate(over="ignore", invalid="ignore"):` silences NumPy's floating-point warnings (`RuntimeWarning: overflow encountered ...` and `invalid value encountered ...`) inside the indented block only, for the one run that is meant to blow up. Whether those warnings appear at all depends on the NumPy build: the matrix products run inside a compiled library that may not report them.
- `{lr:<6}` left-aligns `lr` in a field 6 characters wide; `{x:10.4g}` right-aligns `x` in 10 characters with 4 significant digits, dropping trailing zeros (so `0.1310` prints as `0.131`).
- `min(learning_rates, key=...)` returns the learning rate whose `key` is smallest, like Kotlin's `minBy`. `np.nan_to_num(x, nan=np.inf)` replaces `NaN` by infinity, so the diverged run can never be "the best": comparisons with `NaN` are always `False`, which would make `min` unreliable.

The table reads like Days 51 and 52 in miniature:

- **0.003** is too small: it's still creeping down after 100 epochs.
- **0.01 and 0.03** land at the bottom. 0.01 wins on validation by a hair.
- **0.1 and 0.3** are fast but noisy: a big step turns each batch's noise into jitter around the minimum, so they finish a little above it.
- **1.0** diverges. Each step overshoots, the loss grows without limit, and it ends as `nan` ("not a number", which appears when the overflow reaches infinity and the code computes `inf - inf`).

How large is too large? For full-batch gradient descent, Days 51 and 52 gave the limit \( \eta < 2/\lambda_{\max} \), where \( \lambda_{\max} \) is the largest eigenvalue of the loss's curvature matrix, here \( \frac{2}{n}X^\top X \). `np.linalg.eigvalsh` computes the eigenvalues of a symmetric matrix (Day 24's `eigh`, eigenvalues only):

```python
curv = np.linalg.eigvalsh(2 / len(y_train) * X_train.T @ X_train)
print(f"largest eigenvalue {curv.max():.3f}, so lr must stay below {2 / curv.max():.3f}")
# largest eigenvalue 3.739, so lr must stay below 0.535
```

That is why 0.3 still converges (noisily) and 1.0 can't.

### Milestone 4: train the final model and plot learning curves

**What it is.** A **learning curve** plots the loss against the epoch, for the training set and for the validation set. It's the training run's heartbeat monitor.

```python
w_gd, hist = train_minibatch(X_train, y_train, best_lr, BATCH, EPOCHS, seed=55,
                             X_val=X_val, y_val=y_val)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
epochs = np.arange(EPOCHS + 1)
ax1.plot(epochs, hist["train"], color="#3949ab", label="train")
ax1.plot(epochs, hist["val"], color="#ef6c00", label="validation")
ax1.axhline(baseline_val, color="#555", linestyle="--", linewidth=1)
ax1.text(EPOCHS, baseline_val * 1.15, "baseline (val)", ha="right", color="#555", fontsize=10)
ax1.set_yscale("log")
ax1.set_xlabel("epoch")
ax1.set_ylabel("MSE (log10 reviews)², log scale")
ax1.set_title(f"Learning curves, lr = {best_lr}, batch {BATCH}")
ax1.legend(frameon=False)
colors = ["#9fa8da", "#3949ab", "#00897b", "#ef6c00", "#e53935"]
for lr, c in zip(learning_rates[:-1], colors):            # lr = 1.0 left out: it is NaN
    ax2.plot(epochs, runs[lr]["val"], color=c, label=f"lr = {lr}")
ax2.set_yscale("log")
ax2.set_ylim(0.1, 20)
ax2.set_xlabel("epoch")
ax2.set_ylabel("validation MSE, log scale")
ax2.set_title("Choosing the learning rate on validation")
ax2.legend(frameon=False, ncol=2)
fig.tight_layout()
fig.savefig("day55-learning-curves.svg", format="svg", bbox_inches="tight")
plt.close(fig)

print("first epochs, train:", np.round(hist["train"][:4], 3))
print("last epoch, train:", round(hist["train"][-1], 4), " val:", round(hist["val"][-1], 4))
print("start = mean² + variance:", round(baseline ** 2 + y_train.var(), 3))
# first epochs, train: [11.359  7.908  5.521  3.865]
# last epoch, train: 0.1071  val: 0.1308
# start = mean² + variance: 11.359
```

<figure class="diagram" markdown>
![Two panels. Left: learning curves for learning rate 0.01 with batches of 32, MSE on a log scale against epoch from 0 to 100. Training (blue) and validation (orange) both start near 11, cross the dashed validation baseline near 1 at about epoch 7, and flatten by epoch 25, training at about 0.107 and validation slightly higher at about 0.131. Right: validation MSE for five learning rates. 0.003 falls slowly and is still decreasing at epoch 100; 0.01 and 0.03 reach the floor of about 0.13 within 10 to 25 epochs; 0.1 drops almost immediately; 0.3 drops immediately but keeps jittering between about 0.13 and 0.2](../../assets/img/phase-1/day55-learning-curves.svg)
<figcaption>The reference run. Left: training and validation loss per epoch. Right: the learning-rate sweep that chose 0.01.</figcaption>
</figure>

**How to read them.**

- **The start, 11.359,** is the MSE of predicting 0 for every app, since \( \mathbf{w} = \mathbf{0} \). It equals the squared mean target plus the target's variance (`y_train.var()`, which divides by \( n \)), as the last printed line confirms: an MSE splits into "how far off on average, squared" plus "how spread out". The first thing gradient descent learns is the intercept, which removes the first part.
- **The curves flatten.** That's the bottom of the bowl. More epochs won't help; only better features would.
- **Validation sits slightly above training,** 0.1308 against 0.1071, and the gap stays constant. A small, steady gap is normal: the weights were fitted to the training rows. A gap that *widens* while the training loss keeps falling is **overfitting** (Month 3's big topic). With 12 features and 288 rows there is little room for it.

**How it goes wrong.** A learning curve that rises, or turns into `nan`, means the learning rate is too large (the 1.0 run). One that is still falling steeply at the last epoch means too small a rate or too few epochs (the 0.003 run). One that falls on training while rising on validation means overfitting. And one that is flat from the start, at the starting value, usually means the update line never changes `w`: for example `w - lr * g` computed but not assigned.

### Milestone 5: open the test set, once

```python
test_mse = mse_and_grad(w_gd, X_test, y_test)[0]
print(f"test MSE {test_mse:.4f}  target {TARGET:.4f}  ->", "PASS" if test_mse <= TARGET else "NOT YET")
baseline_test = np.mean((baseline - y_test) ** 2)      # the same constant, on the test apps
ratio = test_mse / baseline_test
print(f"baseline test MSE {baseline_test:.4f}  ->  model is {ratio:.1%} of it, test R² {1 - ratio:.3f}")
rmse = np.sqrt(test_mse)
print(f"test RMSE {rmse:.3f} in log10 units: a typical miss is a factor of {10**rmse:.2f}")
# test MSE 0.1085  target 0.4922  -> PASS
# baseline test MSE 0.9095  ->  model is 11.9% of it, test R² 0.881
# test RMSE 0.329 in log10 units: a typical miss is a factor of 2.13
```

`:.1%` multiplies by 100 and adds a percent sign: 0.119 prints as `11.9%`. **R²** (Day 50) is \( 1 - \text{MSE}_{\text{model}}/\text{MSE}_{\text{baseline}} \): the share of the baseline's squared error that the model removes, 1 for perfect predictions and 0 for no better than the mean. It is the ratio this page has been quoting, in its usual form, and the number Month 3's regression metrics start from. `"PASS" if ... else "NOT YET"` is Python's conditional expression, Kotlin's `if (...) "PASS" else "NOT YET"`. **RMSE**, the root mean squared error, is the square root of the MSE, back in the target's units.

The model's test MSE, 0.1085, is far below the target of 0.4922 written down before training, and 11.9% of the baseline's test MSE (R² 0.881). In plain words: where predicting the average review count misses by a factor of 9.0 (\( 10^{0.954} \)), the model's root-mean-square miss is a factor of 2.13 (\( 10^{0.329} \)). That is the honest headline for the README.

**How it goes wrong.** A tempting shortcut is to compute the baseline from the *test* targets' own mean. That baseline has peeked at the test answers: it is the best constant anyone could pick for those 96 apps, and nobody could have picked it before seeing them.

```python
print(f"{np.mean((y_test.mean() - y_test) ** 2):.4f} {np.mean((y_train.mean() - y_test) ** 2):.4f}")
# 0.8976 0.9095
```

Here the gap is small (0.8976 against 0.9095), but the comparison is unfair in principle, and with a small or shifted test set it can be large. The baseline is a *model*: it gets the training data only, like every other model.

The test MSE (0.1085) is a little lower than the validation MSE (0.1308): the two are separate samples of 96 apps, and a difference of that size between them is ordinary sampling variation (Day 44's confidence intervals would put a range on each).

!!! warning "Common mistake: tuning on the test set"
    "The test MSE is 0.1085; let me try another learning rate and see if it drops." The moment you change anything because of a test score, the test set has become a second validation set, and the number you report is no longer an unbiased estimate. How much can luck alone move a test score? Re-run the whole pipeline with 20 different split seeds (with `lstsq`, to keep it fast):

    ```python
    test_mses = []
    for seed in range(20):
        tr_s, _, te_s = split_indices(len(table), seed=seed)
        st_s = fit_preprocess(table.iloc[tr_s])
        Xs_tr, Xs_te = transform(table.iloc[tr_s], st_s), transform(table.iloc[te_s], st_s)
        ys_tr, ys_te = table["log_reviews"].to_numpy()[tr_s], table["log_reviews"].to_numpy()[te_s]
        w_s = np.linalg.lstsq(Xs_tr, ys_tr)[0]
        test_mses.append(mse_and_grad(w_s, Xs_te, ys_te)[0])
    test_mses = np.array(test_mses)
    print(f"test MSE over 20 splits: min {test_mses.min():.4f}  max {test_mses.max():.4f}  mean {test_mses.mean():.4f}")
    # test MSE over 20 splits: min 0.0897  max 0.1673  mean 0.1189
    ```

    The same model, on the same data, scores anywhere from 0.0897 to 0.1673 depending only on which 96 apps landed in the test set. If you tried 20 settings and kept the one with the best test score, you'd be reporting a lucky draw from a range like that. Choose on validation, report test once. Month 3's cross-validation goes further and averages over several splits.

### Milestone 6: compare with least squares

Day 53 showed this loss is convex, with a single minimum, and Day 25 found that minimum directly with `np.linalg.lstsq`, which solves the least-squares problem without any iterations. If your gradient descent is right, it should arrive at the same weights:

```python
w_ls = np.linalg.lstsq(X_train, y_train)[0]
print("max |w_gd - w_ls|:", np.abs(w_gd - w_ls).max().round(4))
for name, w in [("mini-batch GD", w_gd), ("lstsq", w_ls)]:
    mses = [mse_and_grad(w, X, y)[0] for X, y in
            [(X_train, y_train), (X_val, y_val), (X_test, y_test)]]
    print(f"{name:<14} train {mses[0]:.4f}  val {mses[1]:.4f}  test {mses[2]:.4f}")

weights = pd.Series(w_ls, index=stats["names"] + ["intercept"])
print(weights.round(3).to_string())
# max |w_gd - w_ls|: 0.0005
# mini-batch GD  train 0.1071  val 0.1308  test 0.1085
# lstsq          train 0.1071  val 0.1307  test 0.1086
# log_installs            0.875
# size_mb                -0.003
# paid                   -0.026
# has_ads                -0.038
# days_since_update      -0.017
# cat=Finance             0.009
# cat=Games               0.002
# cat=Health & Fitness   -0.032
# cat=Music & Audio       0.008
# cat=Photography         0.003
# cat=Productivity        0.034
# cat=Tools              -0.005
# intercept               3.237
```

`np.linalg.lstsq(X, y)` returns four things (the solution, residuals, rank and singular values); `[0]` takes the solution. `pd.Series(values, index=names)` labels each weight with its column name, so the printout is readable.

Same weights to within 0.0005, same losses to four decimals: the second pass mark is met. The remaining gap is mini-batch jitter: with a constant learning rate, mini-batch descent hovers *near* the minimum instead of sitting on it (Day 52's learning-rate schedules shrink that). Mini-batch GD even scores 0.0001 better on test, and that is luck: `lstsq` minimizes the *training* loss exactly, and nothing promises that the training optimum is the test optimum.

**Reading the weights.** Every feature is standardized, so each weight means "change in log10(reviews) for one standard deviation more of that feature, the others held fixed", and the weights are comparable with each other. `log_installs` (0.875) does almost all the work; everything else is within ±0.04, small next to the root-mean-square miss of 0.329. The intercept equals the training mean, 3.237, because every standardized column has mean 0 on the training rows. Don't over-read the small weights: they are associations in a synthetic sample, not causes (Day 46), and with 288 rows a weight of 0.03 could easily be noise.

### Milestone 7: look at the predictions

An **error analysis** asks *where* a model is wrong, not only how wrong on average. The simplest tool is a predicted-versus-actual scatter: a perfect model puts every point on the diagonal.

```python
pred = X_test @ w_gd
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(y_test, pred, s=18, alpha=0.6, color="#3949ab", label="model")
ax.axhline(baseline, color="#ef6c00", linestyle="--", linewidth=1.5,
           label="baseline (train mean)")
lims = [0.5, 5.5]
ax.plot(lims, lims, color="#555", linewidth=1, label="perfect prediction")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("actual log10(reviews), test apps (n = 96)")
ax.set_ylabel("predicted log10(reviews)")
ax.set_title("Test set: predictions follow the diagonal")
ax.legend(frameon=False, loc="upper left")
fig.savefig("day55-test-predictions.svg", format="svg", bbox_inches="tight")
plt.close(fig)
print(f"actual range {y_test.min():.2f} to {y_test.max():.2f}; predicted {pred.min():.2f} to {pred.max():.2f}")
low = pred < 2.2
print(f"lowest band: {low.sum()} apps, actual reviews {10**y_test[low].min():.0f} to {10**y_test[low].max():.0f}, "
      f"mean prediction {pred[low].mean():.2f} (= {10**pred[low].mean():.0f} reviews)")
# actual range 0.78 to 5.04; predicted 1.71 to 4.85
# lowest band: 18 apps, actual reviews 6 to 371, mean prediction 1.83 (= 68 reviews)
```

<figure class="diagram" markdown>
![Scatter of predicted against actual log10 reviews for the 96 test apps, with a grey diagonal line for perfect predictions and an orange dashed horizontal line at 3.24 for the baseline. The points follow the diagonal but sit in horizontal bands at predicted values near 1.8, 2.8, 3.7 and 4.5, one band per install bucket. In the lowest band, apps with actual values between about 0.8 and 2.6 are all predicted near 1.8, so the apps with the fewest reviews are over-predicted](../../assets/img/phase-1/day55-test-predictions.svg)
<figcaption>Each dot is a test app. The horizontal bands are the install buckets: the model can't tell apart apps in the same bucket.</figcaption>
</figure>

`low = pred < 2.2` is a boolean mask (Day 8) that selects the apps in the lowest band; `y_test[low]` keeps only those.

A single number hides patterns that one chart shows at once. The predictions form **horizontal bands**: installs come in buckets (`10,000+`, `100,000+`, ...), and since installs carry almost all the weight, every app in a bucket gets nearly the same prediction. The lowest band shows the biggest misses: 18 test apps with between 6 and 371 reviews are all predicted at around 68. That's your error analysis, and it points at the next improvement (a feature that separates apps within a bucket), which is exactly the habit Month 3's milestone asks for.

### How to verify the milestone

- `check_milestone()` prints `all milestone checks passed`, and you broke `mse_and_grad` once to see check 3 fail.
- `grad_check` on the real training matrix is below \( 10^{-7} \).
- `run.py` from a clean folder prints the same numbers twice in a row (fixed seeds everywhere: the split, the shuffles, the random check points).
- The pass mark was written in the report *before* the test MSE was computed, and the test set was evaluated once.
- Gradient-descent weights match `lstsq` to within about \( 10^{-3} \), and the train, validation and test MSEs match to about three decimals.

---

## Grading rubric

Score your own milestone honestly, then again in a week with fresh eyes. Each row is 0, 1 or 2 points.

| Criterion | 0: Not yet | 1: Meets | 2: Strong |
|---|---|---|---|
| **Problem framing** | Target or features chosen without a reason | Target and features stated, with why; no feature leaks the target | Plus: the log scale and the one-hot reference category explained |
| **Data discipline** | Statistics computed on all rows, or the test set used for tuning | Seeded 60/20/20 split; every statistic fitted on train only | Plus: the single-row transform check, and the pass mark written before training |
| **Gradient** | Formula copied, or no check | Derivation written out; `grad_check` below \( 10^{-7} \) on the real data | Plus: a deliberate bug shown to fail the check |
| **Training** | Loss doesn't fall, or the learning rate is a guess | Mini-batch GD with reshuffling; learning rate chosen on validation; learning-curve figure | Plus: the sweep explained (too slow, right, noisy, diverging) |
| **Evaluation** | No baseline, or the test set evaluated repeatedly | Test MSE vs baseline against the stated target; `lstsq` comparison | Plus: an error-analysis chart and what it suggests next |
| **Reproducibility and report** | Fails on a fresh run, or no report | Runs from a clean folder with README; report filled in | Plus: numbers identical across two runs; caveats and no causal claims |

**Pass:** 8 of 12 or more, with no zeros. **Strong:** 11 or 12. A zero in any row means fixing that row before calling the month done, whatever the total.

## Report template

Copy this into `report.md`:

```markdown
# Month 2 milestone: linear regression by gradient descent

**Question.** Predict ______ from ______ (one sentence on why someone would want this).

**Data.** play_store_apps.csv: 486 rows raw → ___ after clean_apps.
Split with seed ___: ___ train / ___ validation / ___ test.
Synthetic course dataset: results describe this sample only.

**Target and features.**
- Target: ______ (transformed with ______ because ______)
- Features: ______ (missing values filled with the TRAINING ______)
- Categoricals: one-hot, reference category ______

**Pass mark (written before training).** Baseline = predict the training mean.
Baseline validation MSE = ______. Target: test MSE ≤ ______ (= ___ × baseline), because ______.

**Gradient.** ∇L = ______ (derivation in derivation.md).
grad_check on the real training matrix: ______ (at w = 0) and ______ (at a random w).

**Training.** Mini-batch GD, batch ___, epochs ___, learning rates tried: ______.
Chosen on validation: lr = ___ (validation MSE ______).
![Learning curves](figures/learning_curves.svg)

**Result (test set, evaluated once).**
| model | train MSE | val MSE | test MSE |
|---|---|---|---|
| baseline (train mean) | | | |
| mini-batch GD | | | |
| np.linalg.lstsq | | | |
Pass mark met? yes / no. Test R² = 1 − test MSE / baseline test MSE = ______.
Max |w_GD − w_lstsq| = ______.
In plain words: ______ (e.g. "the root-mean-square miss is a factor of ___").

**Error analysis.** ![Predicted vs actual](figures/test_predictions.svg)
Where the model misses most: ______. One feature that might help: ______.

**Caveats.** sample size; synthetic data; associations, not causes; ______

## Self-score (rubric)
Framing _/2 · Data discipline _/2 · Gradient _/2 · Training _/2 · Evaluation _/2 · Reproducibility _/2 = __/12
```

---

## Month 2 retrospective

Same purpose as Month 1's: turn four weeks of effort into a short list of what to keep and what to change. Fill it in with your Month 1 retrospective open next to you, and check whether last month's adjustments actually happened.

```markdown
# Month 2 retrospective (Days 29–56)

## Milestone results
- Test MSE ______ vs baseline ______ (pass mark ______): pass / not yet
- grad_check on real data: ______   |w_GD − w_lstsq|: ______
- Rubric self-score __/12   weakest row:
- One thing the error-analysis chart showed me:

## Month 2 in one paragraph (as if explaining to a junior engineer)
(Hint: how does a model know which way to change its weights, and how do
we know whether the result will hold up on data it hasn't seen?)

## Can I do these without looking anything up? (yes / shaky / no, + lesson to revisit)
- Derive a derivative with the chain rule and check it numerically (Days 29–33):
- Explain the gradient and why −∇L points downhill (Day 32):
- Simulate a probability and compute it exactly (Days 36–40):
- Use Bayes' rule on a base-rate problem (Day 38):
- Build a confidence interval and run a permutation test (Days 44–45):
- Explain why correlation is not causation, with a confounder (Day 46):
- Say why MSE and cross-entropy are maximum-likelihood losses (Days 47, 54):
- Choose a learning rate and recognise divergence and SGD noise (Days 51–52):

## Did last month's adjustments happen?
- Adjustment: ______   happened? yes / partly / no   why:

## Time
Planned ~__ h, actual ~__ h. Days skipped or doubled up:

## What worked (keep doing)
-

## What didn't (stop or change)
-

## Kotlin/Android habit that helped:
## Kotlin/Android habit that got in the way:

## Adjustments for Month 3 (specific and checkable)
- Pace (keep / slow down / speed up):
- One study habit to add:
- One thing to drop:
```

!!! tip "Write the retro even though the milestone passed"
    A passing milestone is when a retro is cheapest and most useful: you still remember what was hard and why. "Passed, moving on" loses the one thing worth carrying forward, such as "I lost 2 hours to a `(n, 1)` versus `(n,)` shape bug: assert shapes in every function from now on". Write at least one specific, checkable adjustment.

---

## What's next in Month 3

Months 1 and 2 built the foundations: the tools (Python, NumPy, pandas, plots), the linear algebra, and the mathematics of learning. Month 3 is **classic machine learning**, Phase 2 of the [Roadmap](../../roadmap/index.md): the methods that still win on most tabular data, and the habits that keep results honest.

| Module | Topic | Where this weekend gave you a head start |
|---|---|---|
| 2.1 | What is learning? Supervised vs unsupervised, training vs inference | you did supervised learning this weekend |
| 2.2 | Data preparation: splits, leakage, scaling, encoding categories | `split_indices`, `fit_preprocess`, one-hot columns |
| 2.3 | Feature engineering | the install-bucket bands in the error analysis |
| 2.4 | Linear and logistic regression | this weekend's model; logistic adds a sigmoid and cross-entropy (Days 30, 54) |
| 2.5 | Trees and ensembles: decision trees, random forests, gradient boosting | (new) |
| 2.6 | Other classifiers: k-NN, SVMs, naive Bayes | Day 22's nearest neighbours, Day 41's naive Bayes |
| 2.7 | Clustering and dimensionality reduction: k-means, PCA, t-SNE/UMAP | Day 25's PCA |
| 2.8 | Evaluation: precision, recall, F1, ROC-AUC, regression metrics | MSE, RMSE and the baseline comparison |
| 2.9 | Overfitting and generalization: bias–variance, regularization, cross-validation | the train/validation gap; ridge in Exercise 3 |
| 2.10 | scikit-learn pipelines | `fit_preprocess` + `transform` is a hand-made pipeline |

**The Month 3 milestone** (from the [Monthly Plan](../../plan/monthly.md)): *a tabular project with three models compared with 5-fold cross-validation, the metric choice justified, and an error analysis written.* Cross-validation is this weekend's validation set done five times: split the training data into five folds, train on four and validate on the fifth, rotate, and average, so the choice between models doesn't hinge on one lucky split of 96 apps (the 20-seed experiment above showed how much a single split can vary). From Month 3 on you'll also use **scikit-learn**, whose `StandardScaler().fit(train)` followed by `.transform(test)` is exactly the discipline you built by hand. Now you know what it does inside.

## Going further

These are real and useful, but not needed for this milestone. Month 3 covers the first three.

- **k-fold cross-validation**: averaging over several validation splits. [scikit-learn: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html).
- **Pipelines**: bundling preprocessing and model into one object so fit/transform can't be mixed up. [scikit-learn: Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html).
- **Encoding categorical features**, including what to do with a category never seen in training. [scikit-learn: Encoding categorical features](https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features).
- **Early stopping**: stop training when the validation loss stops improving, instead of a fixed number of epochs. Covered with neural networks in [Dive into Deep Learning](https://d2l.ai/).
- **How `lstsq` solves the problem without iterating** (via the singular value decomposition, Day 25's preview). [NumPy: `numpy.linalg.lstsq`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html).

---

## Exercises

Do the milestone first; these deepen it. Each solution below runs on its own: it re-creates its own data and helpers.

1. **Warm-up: split sizes.** Write `split_indices` (or reuse yours) and print the three sizes for \( n = 480 \), \( n = 101 \) and \( n = 7 \), plus the number of distinct rows across the three sets. Which set gets the leftover rows, and why?
2. **Warm-up: the baseline by hand.** Training targets are 2, 3, 4 and 7; test targets are 3 and 5. By hand, find the baseline prediction and its test MSE. Then compute the *training* MSE of the constants 3.5 (the median), 4.0 (the mean) and 4.5, and confirm the mean wins.
3. **Core: ridge regression.** Add an L2 penalty: \( L(\mathbf{w}) = \frac{1}{n}\lVert X\mathbf{w} - \mathbf{y} \rVert^2 + \lambda \lVert \mathbf{w}_{\text{feat}} \rVert^2 \), where \( \mathbf{w}_{\text{feat}} \) is every weight except the intercept and \( \lambda \) ("lambda") sets the penalty's strength. Derive the gradient, `grad_check` it on random data with \( \lambda = 0.1 \), then solve "gradient = 0" with `np.linalg.solve` and compare the weights with plain least squares.
4. **Core: batch size.** On synthetic data with 288 rows, train with batches of 1, 32 and 288 at the same learning rate (0.01) for 20 epochs. Count the update steps of each and compare the final training MSE with the least-squares floor. Explain the ranking.
5. **Predict the output.** Before running, predict what this prints (values and any warnings), then run it:
    ```text
    one_app = np.array([[4.0, 25.0, 0.0]])
    mean = one_app.mean(axis=0)
    std = one_app.std(axis=0, ddof=1)
    print(mean, std)
    print((one_app - mean) / std)
    ```
    What does this tell you about a `transform` that refits on the rows it's given?
6. **Find and fix the bug.** This mini-batch trainer runs without errors, and its `grad_check` *on the full training set* passes, yet after 10 epochs its weights are far from the least-squares answer. Find the bug, predict the relative error between its gradient and the correct one on a batch of 32 (Day 33's fingerprint formula), explain why `grad_check` did not catch it, and fix it.
    ```text
    N = len(y)                                        # 288, the whole training set

    def mse_and_grad_buggy(w, Xb, yb):
        r = Xb @ w - yb
        return r @ r / N, 2 / N * Xb.T @ r
    ```
7. **Challenge: the honest negative result.** Run a self-contained version of the pipeline with `rating` as the target (drop unrated apps first, add `log_reviews` as a feature) and fit with `lstsq`. Report validation and test MSE against the baseline. Does it pass a 50% target? What target *would* be reasonable, and what does that say about the features?

??? success "Solutions"
    **1. Split sizes.**

    ```python
    import numpy as np

    def split_indices(n, seed, fractions=(0.6, 0.2, 0.2)):
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
        n_train = int(fractions[0] * n)
        n_val = int(fractions[1] * n)
        return order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]

    for n in (480, 101, 7):
        tr, va, te = split_indices(n, seed=0)
        print(n, len(tr), len(va), len(te), len(np.union1d(np.union1d(tr, va), te)))
    print(split_indices(7, seed=0))
    # 480 288 96 96 480
    # 101 60 20 21 101
    # 7 4 1 2 7
    # (array([2, 4, 3, 6]), array([5]), array([0, 1]))
    ```

    `int()` rounds toward zero: `int(0.6 * 101)` is 60 and `int(0.2 * 101)` is 20, so the test set, which takes "everything after the second cut", gets 21. `np.union1d` returns the sorted distinct values of two arrays; the union always has \( n \) members because the three sets are consecutive slices of one permutation.

    **2. The baseline by hand.** The training mean is \( (2 + 3 + 4 + 7)/4 = 4 \). Test errors are \( 4 - 3 = 1 \) and \( 4 - 5 = -1 \), so the test MSE is \( (1 + 1)/2 = 1 \). Training MSE of the constant 4: \( (4 + 1 + 0 + 9)/4 = 3.5 \).

    ```python
    import numpy as np

    y_train = np.array([2.0, 3.0, 4.0, 7.0])
    y_test = np.array([3.0, 5.0])
    baseline = y_train.mean()
    print(baseline)
    print(np.mean((baseline - y_test) ** 2))
    for c in [3.5, 4.0, 4.5]:                        # median, mean, and a bit above
        print(c, np.mean((c - y_train) ** 2))
    # 4.0
    # 1.0
    # 3.5 3.75
    # 4.0 3.5
    # 4.5 3.75
    ```

    The mean has the lowest training MSE, as calculus predicts: \( \frac{d}{dc}\frac{1}{n}\sum_i (c - y_i)^2 = \frac{2}{n}\sum_i (c - y_i) \), which is zero exactly when \( c = \bar{y} \). The median minimizes the mean *absolute* error instead (Day 50). Note also that this block redefines `y_train` and `baseline`; re-run the page from the top if you go back to the milestone blocks.

    **3. Ridge regression.** The penalty \( \lambda\sum_j w_j^2 \) adds \( 2\lambda w_j \) to each penalized gradient entry (Day 33's rule: the gradient of \( \lVert \mathbf{w} \rVert^2 \) is \( 2\mathbf{w} \)) and nothing to the intercept's. Setting the full gradient to zero: \( \frac{2}{n}X^\top(X\mathbf{w} - \mathbf{y}) + 2\lambda P\mathbf{w} = \mathbf{0} \), so \( \left(\frac{1}{n}X^\top X + \lambda P\right)\mathbf{w} = \frac{1}{n}X^\top\mathbf{y} \), where \( P \) is the identity matrix with a 0 in the intercept's corner.

    ```python
    import numpy as np

    def numerical_grad(f, w, h=1e-5):
        w = np.array(w, dtype=float)
        grad = np.zeros_like(w)
        for idx in np.ndindex(w.shape):
            old = w[idx]
            w[idx] = old + h
            f_plus = f(w)
            w[idx] = old - h
            f_minus = f(w)
            w[idx] = old
            grad[idx] = (f_plus - f_minus) / (2 * h)
        return grad

    def grad_check(f, w, h=1e-5):
        _, g_analytic = f(w)
        g_numeric = numerical_grad(lambda v: f(v)[0], w, h)
        diff = np.linalg.norm(g_analytic - g_numeric)
        size = np.linalg.norm(g_analytic) + np.linalg.norm(g_numeric)
        return diff / size if size > 0 else 0.0

    rng = np.random.default_rng(3)
    X = np.column_stack([rng.normal(size=(50, 3)), np.ones(50)])   # 3 features + intercept
    y = X @ np.array([1.0, -2.0, 0.5, 4.0]) + rng.normal(scale=0.3, size=50)
    n, lam = len(y), 0.1

    def ridge_and_grad(w):
        r = X @ w - y
        w_pen = w.copy()
        w_pen[-1] = 0.0                       # the intercept is not penalized
        return r @ r / n + lam * w_pen @ w_pen, 2 / n * X.T @ r + 2 * lam * w_pen

    print(f"{grad_check(ridge_and_grad, rng.normal(size=4)):.1e}")
    P = np.diag([1.0, 1.0, 1.0, 0.0])
    w_ridge = np.linalg.solve(X.T @ X / n + lam * P, X.T @ y / n)
    w_plain = np.linalg.lstsq(X, y)[0]
    print(w_plain.round(3))
    print(w_ridge.round(3))
    print(np.abs(ridge_and_grad(w_ridge)[1]).max() < 1e-12)
    # 1.4e-11
    # [ 0.976 -2.003  0.437  4.048]
    # [ 0.922 -1.861  0.398  4.029]
    # True
    ```

    The gradient checks out, and at the solved weights the gradient is zero to within \( 10^{-12} \). Every feature weight shrinks toward 0 (0.976 → 0.922, −2.003 → −1.861): that's what the penalty buys, at the price of a slightly higher training MSE. The intercept also moves a little even though it isn't penalized, because these random feature columns don't have exactly zero mean, so shrinking the slopes shifts the best intercept. Whether ridge helps on the Play Store split is a question for the validation set: with 288 rows, 12 features and one dominant signal there is little overfitting to cure, so expect little or no gain (try it: the same `solve` on the milestone's `X_train`, scored on `X_val`). Ridge earns its keep when features are many or correlated; Month 3 returns to it.

    **4. Batch size.**

    ```python
    import numpy as np

    rng = np.random.default_rng(4)
    X = np.column_stack([rng.normal(size=(288, 3)), np.ones(288)])
    y = X @ np.array([0.8, -0.3, 0.1, 3.0]) + rng.normal(scale=0.3, size=288)

    def mse_and_grad(w, X, y):
        r = X @ w - y
        return r @ r / len(y), 2 / len(y) * X.T @ r

    def train_sgd(X, y, lr, batch_size, epochs, seed):
        rng = np.random.default_rng(seed)
        w = np.zeros(X.shape[1])
        for _ in range(epochs):
            order = rng.permutation(len(y))
            for start in range(0, len(y), batch_size):
                batch = order[start:start + batch_size]
                w = w - lr * mse_and_grad(w, X[batch], y[batch])[1]
        return w

    best = mse_and_grad(np.linalg.lstsq(X, y)[0], X, y)[0]
    print(f"least-squares floor: {best:.4f}")
    for bs in [1, 32, 288]:
        w = train_sgd(X, y, lr=0.01, batch_size=bs, epochs=20, seed=0)
        steps = 20 * int(np.ceil(288 / bs))
        print(f"batch {bs:>3}: {steps:>5} steps, train MSE {mse_and_grad(w, X, y)[0]:.4f}")
    # least-squares floor: 0.1035
    # batch   1:  5760 steps, train MSE 0.1063
    # batch  32:   180 steps, train MSE 0.1098
    # batch 288:    20 steps, train MSE 4.4113
    ```

    `{bs:>3}` right-aligns in 3 characters, and `np.ceil` rounds up, so the step count includes a final short batch when the batch size doesn't divide 288. Every run visits each row 20 times; what differs is how many **updates** that buys. The full batch takes only 20 exact steps of size 0.01 and is nowhere near the bottom (4.41). Batch 1 takes 5,760 tiny noisy steps; at this small learning rate its noise is small, so it gets closest (0.1063, slightly above the 0.1035 floor). Batch 32 is in between on steps, and still a little short after 20 epochs. On a GPU, larger batches cost little more per step than small ones, which is why batches of 32–512 are the usual compromise (Day 52).

    **5. Predict the output.** The mean of a single row is the row itself. The standard deviation with `ddof=1` divides by \( n - 1 = 0 \), which NumPy reports with a warning and turns into `NaN`. Then every standardized value is \( 0/\text{NaN} \), which is `NaN`.

    ```python
    import numpy as np

    one_app = np.array([[4.0, 25.0, 0.0]])            # one row, three raw features
    mean = one_app.mean(axis=0)
    std = one_app.std(axis=0, ddof=1)
    print(mean, std)
    print((one_app - mean) / std)
    # RuntimeWarning: Degrees of freedom <= 0 for slice
    # RuntimeWarning: invalid value encountered in divide
    # [ 4. 25.  0.] [nan nan nan]
    # [[nan nan nan]]
    ```

    (The two warnings come with a file path and line number in front, which depend on your NumPy installation.) With `ddof=0` you'd get a standard deviation of 0 instead, and \( 0/0 \) is also `NaN`. Either way, a `transform` that refits returns garbage for exactly the call an app makes most: one input at a time. That is why `check_milestone()` transforms a single row.

    **6. Find and fix the bug.** The loss and gradient divide by `N`, the size of the whole training set, instead of `len(yb)`, the size of the batch. On the full set the two are equal, so a full-set `grad_check` passes. On a batch of 32 the gradient is \( 32/288 = 1/9 \) of the right one, so the effective learning rate is 9 times smaller and training is 9 times slower. The fingerprint formula \( |c - 1|/(|c| + 1) \) with \( c = 1/9 \) gives \( (8/9)/(10/9) = 0.8 \).

    ```python
    import numpy as np

    rng = np.random.default_rng(6)
    X = np.column_stack([rng.normal(size=(288, 2)), np.ones(288)])
    y = X @ np.array([1.5, -1.0, 2.0]) + rng.normal(scale=0.2, size=288)
    N = len(y)                                        # 288, the whole training set

    def mse_and_grad_buggy(w, Xb, yb):
        r = Xb @ w - yb
        return r @ r / N, 2 / N * Xb.T @ r

    def mse_and_grad_fixed(w, Xb, yb):
        r = Xb @ w - yb
        return r @ r / len(yb), 2 / len(yb) * Xb.T @ r

    def train_with(grad_fn, lr=0.05, batch_size=32, epochs=10, seed=0):
        rng = np.random.default_rng(seed)
        w = np.zeros(3)
        for _ in range(epochs):
            order = rng.permutation(N)
            for start in range(0, N, batch_size):
                batch = order[start:start + batch_size]
                w = w - lr * grad_fn(w, X[batch], y[batch])[1]
        return w

    print("buggy:", train_with(mse_and_grad_buggy).round(3))
    print("fixed:", train_with(mse_and_grad_fixed).round(3))
    print("lstsq:", np.linalg.lstsq(X, y)[0].round(3))

    def numerical_grad(f, w, h=1e-5):
        w = np.array(w, dtype=float)
        grad = np.zeros_like(w)
        for idx in np.ndindex(w.shape):
            old = w[idx]
            w[idx] = old + h
            f_plus = f(w)
            w[idx] = old - h
            f_minus = f(w)
            w[idx] = old
            grad[idx] = (f_plus - f_minus) / (2 * h)
        return grad

    def grad_check(f, w, h=1e-5):
        _, g_analytic = f(w)
        g_numeric = numerical_grad(lambda v: f(v)[0], w, h)
        diff = np.linalg.norm(g_analytic - g_numeric)
        size = np.linalg.norm(g_analytic) + np.linalg.norm(g_numeric)
        return diff / size if size > 0 else 0.0

    # grad_check passes on the full set AND on a batch: the gradient matches its own loss
    Xb, yb, w0 = X[:32], y[:32], np.zeros(3)
    print(f"grad_check, full set: {grad_check(lambda v: mse_and_grad_buggy(v, X, y), w0):.1e}")
    print(f"grad_check, batch of 32: {grad_check(lambda v: mse_and_grad_buggy(v, Xb, yb), w0):.1e}")
    # ...but compared with the CORRECT batch gradient, the fingerprint appears
    g_bug, g_ok = mse_and_grad_buggy(w0, Xb, yb)[1], mse_and_grad_fixed(w0, Xb, yb)[1]
    print((g_bug / g_ok).round(4))
    print(round(np.linalg.norm(g_bug - g_ok) / (np.linalg.norm(g_bug) + np.linalg.norm(g_ok)), 4))
    # buggy: [ 0.937 -0.583  1.262]
    # fixed: [ 1.513 -0.994  1.992]
    # lstsq: [ 1.518 -0.994  1.991]
    # grad_check, full set: 5.8e-12
    # grad_check, batch of 32: 8.1e-12
    # [0.1111 0.1111 0.1111]
    # 0.8
    ```

    `grad_check` passes everywhere, on the full set and on a batch, and that is the lesson of this bug. A gradient check only tests that the gradient matches the loss function *you wrote*; here the loss itself is wrong on batches (it divides a 32-row sum by 288), and the gradient faithfully matches the wrong loss. Comparing with the correct batch gradient shows the fingerprint: every entry is exactly \( 1/9 = 0.1111 \) of the right one, and the relative error is the predicted 0.8. The bug hides from a full-set check because the check runs on the one input size where it's harmless: always grad-check on the same kind of input the training loop uses, a batch.

    **7. The honest negative result.** A self-contained version, with its own loading, cleaning, features and split:

    ```python
    import numpy as np
    import pandas as pd

    BASE = "https://manjulkadyan.github.io/ai-leaning/assets/data/"
    raw7 = pd.read_csv(BASE + "play_store_apps.csv")
    apps7 = raw7.drop_duplicates(ignore_index=True)
    apps7 = apps7.assign(
        category=apps7["category"].str.strip().str.title(),
        installs=apps7["installs"].str.replace(",", "", regex=False)
                                  .str.replace("+", "", regex=False).astype("int64"),
        size_mb=apps7["size_mb"].replace("Varies with device", np.nan).astype("float64"),
        last_updated=pd.to_datetime(apps7["last_updated"], format="%Y-%m-%d"),
    ).dropna(subset=["rating"]).reset_index(drop=True)          # rating is the target: no NaN allowed

    numeric = pd.DataFrame({
        "log_installs": np.log10(apps7["installs"]),
        "log_reviews": np.log10(apps7["reviews"]),
        "size_mb": apps7["size_mb"],
        "paid": (apps7["price_usd"] > 0).astype(float),
        "has_ads": (apps7["has_ads"] == "Yes").astype(float),
        "days_since_update": (pd.Timestamp("2026-10-01") - apps7["last_updated"]).dt.days.astype(float),
    })
    y7 = apps7["rating"].to_numpy()

    order = np.random.default_rng(55).permutation(len(y7))
    n_tr, n_va = int(0.6 * len(y7)), int(0.2 * len(y7))
    tr, va, te = order[:n_tr], order[n_tr:n_tr + n_va], order[n_tr + n_va:]
    print(len(y7), len(tr), len(va), len(te))

    cats = sorted(apps7["category"].iloc[tr].unique())[1:]
    def features(rows, size_median):
        num = numeric.iloc[rows].assign(size_mb=numeric["size_mb"].iloc[rows].fillna(size_median))
        onehot = [(apps7["category"].iloc[rows] == c).to_numpy(dtype=float) for c in cats]
        return np.column_stack([num.to_numpy()] + onehot)

    size_median = numeric["size_mb"].iloc[tr].median()          # training rows only
    F_tr, F_va, F_te = features(tr, size_median), features(va, size_median), features(te, size_median)
    mu, sd = F_tr.mean(axis=0), F_tr.std(axis=0, ddof=1)
    def design(F):
        return np.column_stack([(F - mu) / sd, np.ones(len(F))])
    X_tr, X_va, X_te = design(F_tr), design(F_va), design(F_te)

    w7 = np.linalg.lstsq(X_tr, y7[tr])[0]
    mse = lambda w, X, y: np.mean((X @ w - y) ** 2)
    base = y7[tr].mean()
    for name, X, rows in [("val", X_va, va), ("test", X_te, te)]:
        m, b = mse(w7, X, y7[rows]), np.mean((base - y7[rows]) ** 2)
        print(f"{name}: model {m:.4f}  baseline {b:.4f}  ratio {m / b:.1%}")
    # 462 277 92 93
    # val: model 0.1652  baseline 0.1693  ratio 97.6%
    # test: model 0.2466  baseline 0.2508  ratio 98.3%
    ```

    `.dropna(subset=["rating"])` drops only the rows whose rating is missing (a target can't be `NaN`), and `.reset_index(drop=True)` renumbers the remaining 462 rows 0..461 so positions and labels agree. `mse = lambda w, X, y: ...` is a one-line anonymous function, Kotlin's `val mse = { w, X, y -> ... }`.

    The model is 97.6% of the baseline on validation and 98.3% on test: the features remove only 2.4% (validation) and 1.7% (test) of the squared error, so a 50% target fails by a mile. A reasonable target would be "below the baseline on both validation and test", and even that margin is small next to how much a test score varies between splits (the 20-seed experiment). The finding goes in a report as a result, not a failure: *in this dataset, installs, reviews, size, price, ads, update recency and category say almost nothing about an app's rating.*

## Resources

- [NumPy: `numpy.linalg.lstsq`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html): the least-squares solver you compared against, and what it returns.
- [NumPy: `Generator.permutation`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.permutation.html): the seeded shuffle behind the split and every epoch.
- [scikit-learn: Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html): data leakage and inconsistent preprocessing, explained with examples. Read it now, use the library next month.
- [Dive into Deep Learning: Linear Neural Networks for Regression](https://d2l.ai/chapter_linear-regression/index.html): linear regression trained with minibatch SGD from scratch, the same structure as this weekend.
- [CS231n: Optimization notes](https://cs231n.github.io/optimization-1/): gradient checking, analytic versus numerical gradients, and mini-batch gradient descent.
- [Google Machine Learning Crash Course: Linear regression](https://developers.google.com/machine-learning/crash-course/linear-regression): loss, gradient descent and hyperparameters, with interactive exercises.
- [Mathematics for Machine Learning (free book)](https://mml-book.github.io/): chapter 9 is linear regression from both the least-squares and the maximum-likelihood side.

## Checklist

- [ ] I chose a target and features, can say why (and why not `rating`), and can spot a leaky feature.
- [ ] I can one-hot encode a category column and explain why one category gets no column.
- [ ] My split is seeded, 60/20/20, and I can say what each set is for.
- [ ] Every preprocessing statistic comes from the training rows only, and my `transform` works on a single row.
- [ ] I wrote the pass mark from the baseline's validation MSE before training.
- [ ] I wrote the gradient derivation from a blank page, and `grad_check` on the real data is below 1e-7.
- [ ] `check_milestone()` passes, and I broke the gradient once to see it fail.
- [ ] I traced one epoch, chose the learning rate on validation, and can explain each curve in the sweep, including the one that diverged.
- [ ] I evaluated the test set once, beat the baseline by the stated margin, and my weights match `np.linalg.lstsq`.
- [ ] I made the learning-curve and predicted-versus-actual charts and wrote one error-analysis finding.
- [ ] `month2-milestone/` runs from a clean folder with a README, `report.md` is filled in, and I wrote my Month 2 retrospective with one specific adjustment for Month 3.

**Next:** Month 3: classic machine learning (coming soon)
