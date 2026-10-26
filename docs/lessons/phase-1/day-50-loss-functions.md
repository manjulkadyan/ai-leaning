# Day 50: Loss Functions: What "Wrong" Means

**Phase 1 · Week 8 · ~150 minutes**

## Objectives

- [ ] Say what a loss function is, and why training needs a single number that measures "how wrong".
- [ ] Compute mean squared error (MSE) and mean absolute error (MAE) by hand and in NumPy, and say which one an outlier can push around.
- [ ] Show that the best constant prediction is the **mean** under MSE and the **median** under MAE, on the Play Store data.
- [ ] Explain the Huber loss as "squared when small, absolute when big", and compare its gradient with the other two.
- [ ] Compute binary cross-entropy (log loss) for a classifier by hand, explain why it punishes confident mistakes so hard, and compute it safely from logits.
- [ ] Explain why accuracy can be reported but cannot be trained on: its gradient is zero almost everywhere.
- [ ] Read a loss as a function of the **parameters** (the data is fixed), and draw its loss curve for a 1-parameter model and its loss surface for a 2-parameter model.
- [ ] Recognise and fix the classic loss bugs from their symptoms: residuals that cancel, a column of targets that broadcasts, `nan` from missing data, `log(0)`, and overflow in `exp`.

**Prerequisites.** [Day 22](day-22-norms-distances.md) (L1 and L2 norms), [Day 29](day-29-derivatives.md) (the derivative, `central_diff`), [Day 30](day-30-derivative-rules.md) (the sigmoid and its derivative), [Day 32](day-32-partial-derivatives-gradient.md) and [Day 33](day-33-vector-matrix-gradients.md) (the gradient of the MSE), [Day 39](day-39-expectation-variance.md) (mean, variance) and [Day 47](day-47-maximum-likelihood.md) (maximum likelihood). The data cleaning is the short version from [Day 17](day-17-pandas-cleaning.md).

---

## The big idea

A model makes predictions. Some are good, some are bad. To *improve* the model automatically, a computer needs one thing above all: a **single number that says how bad the predictions are right now**, which goes down when they get better. That number is the **loss** (also called the **cost**, the **objective** or the **error**). Training a model is nothing more than turning its knobs to make that number small.

**Analogy: a sat-nav's "distance remaining".** A navigation app can't tell you "you are driving well". It can tell you "12.4 km to go", and every decision it makes is about shrinking that number. The loss is the model's distance remaining. And in the same way that "distance" could mean straight-line distance, road distance or travel time, there are several sensible ways to measure "wrong", which reward different behaviour. Choosing one is a real design decision, and it is today's topic.

The four pieces of every supervised model, and who owns what:

| Piece | What it is | Fixed or changing during training? |
|---|---|---|
| **Data** \( (x_i, y_i) \) | Inputs and the true answers | Fixed |
| **Model** \( \hat{y} = f(x; \mathbf{w}) \) | A formula with knobs \( \mathbf{w} \) (the **parameters**) | The formula is fixed, the knobs change |
| **Loss** \( L(\mathbf{w}) \) | One number: how wrong the model is on the data | Recomputed at every new \( \mathbf{w} \) |
| **Optimizer** | The rule that turns the knobs to lower \( L \) | Gradient descent, from Day 51 |

The animation shows that relationship for the simplest model there is: a line through the origin, \( \hat{y} = w\,x \), with one knob \( w \). The five points never move. Turning the knob tilts the line, and each tilt gets one score, a point on the curve on the right.

<figure class="diagram" markdown>
![Animation: on the left, five fixed blue data points and an orange line through the origin whose slope w steps through 0.2, 0.6, 1.0, 1.4, 1.8 and back; red vertical segments show each point's miss. On the right, a parabola of the loss L(w) against w, with an orange dot that moves to the matching point on the curve each time the line tilts; the curve is lowest at w = 1.01](../../assets/img/phase-1/day50-knob-and-curve.svg)
<figcaption>Animated: the data stays put; each setting of the knob w gives one loss value, and together they trace the loss curve.</figcaption>
</figure>

## Residuals: the raw material of a loss

Start with regression: predicting a number. Some notation, one symbol at a time:

- \( y_i \) is the **true value** for example \( i \) (the real star rating of app \( i \)).
- \( \hat{y}_i \) ("y-hat") is the model's **prediction** for that example.
- \( r_i = \hat{y}_i - y_i \) is the **residual**: how far off the prediction is, with a sign. Positive means the model guessed too high.
- \( n \) is the number of examples.

Four apps, their real ratings and what some model guessed:

```python
import numpy as np

y_true = np.array([4.5, 3.8, 4.1, 2.9])   # real ratings of four apps
y_pred = np.array([4.2, 4.0, 4.1, 4.0])   # what some model guessed
r = y_pred - y_true                        # residuals
r                                          # array([-0.3,  0.2,  0. ,  1.1])
```

The residuals are \( [-0.3, 0.2, 0, 1.1] \). The model was a little low on the first app, a little high on the second, exact on the third and badly high on the fourth. (NumPy prints `0.` for a float zero: the trailing dot marks a float, not an integer.)

Now we need to squash four numbers into one. The tempting first try is to add them up, and it goes wrong in a way worth seeing:

```python
r.sum()                         # np.float64(1.0000000000000004)
np.array([-5.0, 5.0]).sum()     # np.float64(0.0)   a 5-star miss each way looks "perfect"
```

(The `...0004` at the end is floating-point rounding: a float64 stores numbers as binary fractions, and 4.2, 4.5 and 0.3 have no exact binary form, in the same way that 1/3 has no exact decimal form, so each is stored as the nearest value it can hold and the tiny differences show up in the last digit.) Positive and negative misses **cancel**, so a sum (or a plain average) of residuals can be 0 for a terrible model. That number still has a use: the average residual measures **bias**, whether the model guesses too high or too low *on average*. It can't measure how wrong the model is. So every loss first gets rid of the sign, and the two classic ways are squaring and taking the absolute value.

## Mean squared error

\[
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n} (\hat{y}_i - y_i)^2 = \frac{1}{n}\sum_{i=1}^{n} r_i^2
\]

- \( \sum_{i=1}^{n} \) means "add up for \( i = 1, 2, \dots, n \)".
- Each residual is squared, which makes it positive and makes big misses count much more than small ones.
- \( \frac{1}{n} \) turns the total into an average, so the loss doesn't grow because you have more data. A total would double when the dataset doubles, even if every prediction stayed as good as before. Libraries let you choose: PyTorch's losses take `reduction="mean"` (the default) or `reduction="sum"`. Turning the per-example numbers into one number is called the **reduction**.

**Worked example.** The squares are \( 0.09, 0.04, 0, 1.21 \). Their sum is \( 1.34 \), and \( 1.34 / 4 = 0.335 \).

Look at where that 0.335 comes from: \( 1.21 / 1.34 = 90\% \) of it is the one bad prediction. That is the personality of MSE: **it is dominated by the biggest misses**.

MSE is in squared units ("squared stars"), which nobody can picture. Its square root, the **root mean squared error (RMSE)**, is back in stars: \( \sqrt{0.335} = 0.5788 \) to four decimals. RMSE is the \( L^2 \) norm of the residual vector from Day 22, divided by \( \sqrt{n} \). It is nicer to *report*; MSE is nicer to *train* on (no square root to differentiate) and has the same minimum.

```python
r ** 2                         # array([0.09, 0.04, 0.  , 1.21])
np.mean(r ** 2)                # np.float64(0.335)   MSE
np.sqrt(np.mean(r ** 2))       # np.float64(0.5787918451395113)   RMSE, in stars
np.sum(r ** 2)                 # np.float64(1.34)   the "sum" reduction: grows with n
```

MSE also splits neatly into the two things that can be wrong with a model: being off **on average** (the bias, the mean residual) and being **inconsistent** (how much the residuals scatter around their own mean, their variance from Day 39). The identity is \( \text{MSE} = (\text{mean residual})^2 + \text{variance of the residuals} \), where the variance uses \( n \) in the denominator, which is what `np.var` does by default:

```python
np.mean(r) ** 2, np.var(r)          # (np.float64(0.06250000000000006), np.float64(0.27249999999999996))
np.mean(r) ** 2 + np.var(r)         # np.float64(0.335)   the MSE again
```

Here the bias accounts for 0.0625 of the 0.335 and the scatter for the rest, so shifting every prediction by a constant would help only a little.

**What NumPy does underneath.** Each line runs as whole-array steps (Day 4), one new array per step, all of shape `(4,)`:

| Step | Code | Result |
|---|---|---|
| subtract, element by element | `y_pred - y_true` | `[-0.3, 0.2, 0, 1.1]` |
| square each element | `r ** 2` | `[0.09, 0.04, 0, 1.21]` |
| add up, divide by 4 | `np.mean(...)` | `0.335` |

The middle array is the **per-example loss**: one number per example. Every loss today has this shape: a per-example formula, then a mean. That will matter on Day 52, where training averages the loss over a small random *batch* of examples rather than over all of them.

!!! warning "Common mistake: a column of true values"
    The two arrays must have the **same shape**. Targets often arrive as a column, shape `(4, 1)`, for example from `df[["rating"]].to_numpy()` (double brackets keep a 2-D table). Then Day 9's broadcasting quietly pairs *every* prediction with *every* truth:

    ```python
    y_col = y_true.reshape(-1, 1)            # shape (4, 1): a column
    (y_pred - y_col).shape                   # (4, 4)   16 "residuals", not 4
    np.mean((y_pred - y_col) ** 2)           # np.float64(0.41625000000000006)   wrong, and no error
    np.mean((y_pred - y_col.ravel()) ** 2)   # np.float64(0.335)   fixed: ravel() flattens to shape (4,)
    ```

    There is no error message, only a wrong number, which is the worst kind of bug. Recognise it by checking shapes: put `assert y_pred.shape == y_true.shape` at the top of any loss function that receives two arrays.

## Mean absolute error

\[
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n} |\hat{y}_i - y_i| = \frac{1}{n}\sum_{i=1}^{n} |r_i|
\]

\( |r_i| \) is the absolute value: the residual with its sign dropped. **Worked example.** \( 0.3 + 0.2 + 0 + 1.1 = 1.6 \), and \( 1.6 / 4 = 0.4 \). MAE is in stars already: "on average the model is 0.4 stars off". It is the \( L^1 \) norm of Day 22 divided by \( n \).

```python
np.abs(r)                      # array([0.3, 0.2, 0. , 1.1])
np.mean(np.abs(r))             # np.float64(0.4)   MAE
np.mean(r)                     # np.float64(0.2500000000000001)   forgot the abs: this is the bias, not the MAE
```

The last line is the classic slip: without `np.abs`, the 1.1 and the −0.3 partly cancel and you get 0.25, which looks like a better model than the MAE says. If an "MAE" ever comes out negative, or smaller when you add a bad prediction, look for a missing `abs`.

Wrap both in functions, because you'll call them a lot:

```python
def mse(y_pred, y_true):
    r = y_pred - y_true
    return np.mean(r ** 2)

def mae(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))

mse(y_pred, y_true), mae(y_pred, y_true)    # (np.float64(0.335), np.float64(0.4))
```

### Squares versus bars: what an outlier does

Draw each residual as a vertical bar from the point to the line: MAE is the average bar length. Now draw a **square** on each bar: MSE is the average square area. When one point drifts away from the line, its bar grows in a straight line, but its square grows with the *square* of the distance.

<figure class="diagram" markdown>
![Animation: a fixed teal line and five blue points. Four sit close to the line with small red residual bars and small orange squares. The fifth point moves away in three steps, residual 1, then 3, then 5; its orange square grows from 1 to 9 to 25 square units, while the panel on the right updates MAE from 0.8 to 1.2 to 1.6 and MSE from 0.7 to 2.3 to 5.5, ending with the note that the one far point is 91% of the MSE](../../assets/img/phase-1/day50-squares-vs-bars.svg)
<figcaption>Animated: one point moving away from the line doubles the MAE but makes the MSE almost eight times bigger.</figcaption>
</figure>

The same thing in code: make the fourth prediction worse, 4.9 instead of 4.0, so its residual goes from 1.1 to 2.0.

```python
y_pred_worse = y_pred.copy()
y_pred_worse[3] = 4.9
mse(y_pred_worse, y_true), mae(y_pred_worse, y_true)   # (np.float64(1.0325000000000004), np.float64(0.6250000000000001))
```

MAE went up by about half (0.4 to 0.625). MSE tripled (0.335 to 1.03). Neither is "right". They encode different opinions:

- **MSE** says: a 2-star miss is *four* times as bad as a 1-star miss. Avoid big mistakes at almost any cost.
- **MAE** says: a 2-star miss is *twice* as bad as a 1-star miss. Every unit of error costs the same.

That opinion decides what the model does when the data contains points it can't fit, which real data always does.

## The best constant prediction: mean versus median

The simplest possible model ignores its input and predicts one number \( c \) for everyone: \( \hat{y}_i = c \). It has a single parameter. Which \( c \) is best? It depends on the loss, and the answer is a classic worth seeing on real data.

Load the Play Store data with the short cleaning from Day 17, keep the rated apps, and look at the `reviews` column:

```python
import pandas as pd

BASE = "https://manjulkadyan.github.io/ai-leaning/assets/data/"
apps = pd.read_csv(BASE + "play_store_apps.csv")
apps = apps.drop_duplicates().assign(
    installs=lambda d: d["installs"].str.replace(",", "").str.rstrip("+").astype(int),
)
rated = apps.dropna(subset=["rating"])
rated.shape                                      # (462, 12)

reviews = rated["reviews"].to_numpy().astype(float)   # int64 counts -> float64
reviews.mean(), np.median(reviews), reviews.max()   # (np.float64(17756.012987012986), np.float64(1658.5), np.float64(2525379.0))
```

No internet? Download the file from the [Datasets page](../../datasets.md) and pass the local path to `read_csv`. (Tested on pandas 3.0.)

`reviews` comes out of pandas as `int64` (whole counts). `.astype(float)` converts it to `float64` once, up front, so that squaring big counts can never hit integer overflow (Day 4's `uint8` wrap-around, with bigger numbers) and every loss below is computed in floating point.

The mean is 10.7 times the median, because a handful of hugely popular apps have millions of reviews. Now **search** for the best constant under each loss: try every \( c \) from 0 to 40,000 in steps of 5 and keep the one with the lowest loss.

```python
c_grid = np.arange(0, 40_001, 5.0)
mse_by_c = np.array([mse(c, reviews) for c in c_grid])    # c broadcasts against the data
mae_by_c = np.array([mae(c, reviews) for c in c_grid])
c_grid[mse_by_c.argmin()], c_grid[mae_by_c.argmin()]      # (np.float64(17755.0), np.float64(1640.0))
```

Line by line: `np.arange(0, 40_001, 5.0)` makes the candidates 0, 5, 10, ..., 40,000 (the underscore in `40_001` is a digit separator, like Kotlin's `40_001`). Inside the list comprehension, `c` is one plain number and `reviews` is an array of 462, so `c - reviews` broadcasts the number against every app (Day 9) and `mse` returns one loss per candidate. `.argmin()` returns the **position** of the smallest value (not the value itself), and indexing `c_grid` with that position gives the winning \( c \).

The search lands on the mean for MSE and on the median for MAE (within the grid's step of 5; more on the small gap below). That is no coincidence; both follow from Week 5 calculus:

- **MSE.** \( L(c) = \frac{1}{n}\sum_i (c - y_i)^2 \). Its derivative is \( L'(c) = \frac{2}{n}\sum_i (c - y_i) = 2(c - \bar{y}) \), where \( \bar{y} \) is the mean. It is zero exactly at \( c = \bar{y} \).
- **MAE.** \( L(c) = \frac{1}{n}\sum_i |c - y_i| \). The derivative of \( |c - y_i| \) is \( +1 \) when \( c \) is above \( y_i \) and \( -1 \) when below (Day 30's kink, like ReLU). So \( L'(c) = \frac{(\text{number of points below } c) - (\text{number above } c)}{n} \). It stops being negative when as many points are below \( c \) as above: the **median**.

With an even number of apps (462), any \( c \) between the two middle values has 231 points on each side, so the MAE curve has a flat bottom between them (1,636 and 1,681) and `argmin` reports the first grid point on it. That is why it says 1,640, not the median 1,658.5: both are equally good under MAE. (MSE's 17,755 is the grid point nearest the mean, 17,756.01.)

Now see how much each answer depends on the few giant apps. Drop the top 5 (1% of the apps) and recompute:

```python
trimmed = np.sort(reviews)[:-5]           # everything except the 5 most-reviewed apps
reviews.mean(), trimmed.mean()            # (np.float64(17756.012987012986), np.float64(8435.00875273523))
np.median(reviews), np.median(trimmed)    # (np.float64(1658.5), np.float64(1569.0))
```

Five apps out of 462 move the MSE-best guess from 17,756 to 8,435, more than halving it. The MAE-best guess barely moves. This is what "MAE is **robust** to outliers" means: a few extreme points can't drag it far.

!!! warning "Common mistake: a missing value turns the whole loss into nan"
    `apps` still contains the 18 unrated apps, whose rating is `NaN` ("not a number", pandas' marker for missing). NumPy arithmetic spreads it: any sum that includes a `NaN` is `NaN`. pandas' own `.mean()` skips missing values by default, so the two disagree:

    ```python
    np.mean(apps["rating"].to_numpy())       # np.float64(nan)
    apps["rating"].mean()                    # np.float64(4.165584415584416)   pandas skips NaN
    np.nanmean(apps["rating"].to_numpy())    # np.float64(4.165584415584416)   NumPy's NaN-skipping version
    ```

    A loss of `nan` during training usually means a `NaN` in the inputs or targets, or one produced on the way (you'll meet `log(0)` below). Recognise it by checking `np.isnan(y).sum()` before training, and fix it the Day 17 way: drop or impute the missing rows deliberately. That is why this lesson works with `rated`, which dropped them.

Here is the same experiment on `rating`, which has no extreme values, drawn as **loss curves**: the loss on the vertical axis, the parameter \( c \) on the horizontal.

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

ratings = rated["rating"].to_numpy()
cs = np.linspace(3.0, 5.0, 401)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, loss, name, best, label, color in [
        (axes[0], mse, "MSE", ratings.mean(), "mean", "#3949ab"),
        (axes[1], mae, "MAE", np.median(ratings), "median", "#00897b")]:
    ax.plot(cs, [loss(c, ratings) for c in cs], color=color, lw=2)
    ax.axvline(best, color="#e53935", ls="--", lw=1.5, label=f"{label} = {best:.3f}")
    ax.set_xlabel("constant prediction c (stars)")
    ax.set_ylabel(f"{name}(c)")
    ax.set_title(f"{name} of predicting c for every app")
    ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("day50-constant-model.svg", format="svg", bbox_inches="tight")
```

<figure class="diagram" markdown>
![Two loss curves over the constant prediction c from 3 to 5 stars. Left: the MSE is a smooth U-shaped parabola with its lowest point at the mean rating 4.166, marked by a dashed red line. Right: the MAE is a V-like curve made of short straight pieces, lowest at the median rating 4.2](../../assets/img/phase-1/day50-constant-model.svg)
<figcaption>A 1-parameter loss curve, on real ratings. MSE is a smooth bowl with its bottom at the mean; MAE is a bowl of straight segments with its bottom at the median.</figcaption>
</figure>

Two shapes to remember. The MSE curve is a **parabola**: smooth everywhere, with a slope that shrinks as you approach the bottom. The MAE curve is made of **straight pieces** with a kink at every data value: its slope never shrinks near the bottom, it only flips sign. You'll see on Day 51 why that matters for gradient descent: a shrinking slope means steps that shrink by themselves as you arrive.

!!! warning "Common mistake: comparing MSE and MAE numbers to each other"
    "The MSE is 0.335 but the MAE is 0.4, so MSE says the model is better." No: they are in different units (squared stars versus stars) and measure different things. Compare a loss only with the *same* loss of another model on the *same* data. Even then, the two losses can rank the same two models in opposite orders:

    ```python
    y_t = np.array([1.0, 2.0, 3.0])
    model_a = np.array([1.0, 2.0, 6.0])    # perfect twice, one 3-unit miss
    model_b = np.array([2.5, 3.5, 4.5])    # 1.5 units off every time
    mse(model_a, y_t), mse(model_b, y_t)   # (np.float64(3.0), np.float64(2.25))   MSE prefers B
    mae(model_a, y_t), mae(model_b, y_t)   # (np.float64(1.0), np.float64(1.5))    MAE prefers A
    ```

    Neither ranking is a bug: MSE hates the single big miss, MAE counts every unit of error the same. Decide which behaviour you want *before* comparing models, and report the loss you trained on. If you want one number to show people, report RMSE or MAE, which are in the target's units.

## The Huber loss: the best of both

MSE has a smooth, shrinking gradient but panics about outliers. MAE shrugs off outliers but has a kink and a slope that never shrinks. The **Huber loss** takes the good half of each: it is squared for small residuals and absolute for big ones. You choose the switch-over point \( \delta \) (delta). Per example:

\[
\ell_\delta(r) =
\begin{cases}
\tfrac{1}{2} r^2 & \text{if } |r| \le \delta \\[4pt]
\delta \left( |r| - \tfrac{1}{2}\delta \right) & \text{if } |r| > \delta
\end{cases}
\qquad\qquad
\text{Huber} = \frac{1}{n}\sum_{i=1}^{n} \ell_\delta(r_i)
\]

- \( \ell \) (a curly "l") is the loss of **one** example; the loss of the dataset is their average.
- Inside \( [-\delta, \delta] \) it is the parabola \( \tfrac{1}{2}r^2 \). The \( \tfrac{1}{2} \) is a convention that makes its derivative plain \( r \).
- Outside, it is a straight line with slope \( \delta \). The \( -\tfrac{1}{2}\delta \) shifts the line down so the two pieces meet without a jump: at \( r = \delta \) both give \( \tfrac{1}{2}\delta^2 \).

**Worked example**, the four residuals \( [-0.3, 0.2, 0, 1.1] \) with \( \delta = 0.5 \). The first three are inside, so \( \tfrac{1}{2}(0.09) = 0.045 \), \( \tfrac{1}{2}(0.04) = 0.02 \) and 0. The last is outside: \( 0.5 \times (1.1 - 0.25) = 0.425 \). The average is \( (0.045 + 0.02 + 0 + 0.425)/4 = 0.1225 \).

```python
def huber(y_pred, y_true, delta=1.0):
    r = y_pred - y_true
    a = np.abs(r)
    per_example = np.where(a <= delta, 0.5 * r ** 2, delta * (a - 0.5 * delta))
    return np.mean(per_example)

huber(y_pred, y_true, delta=0.5)      # np.float64(0.12250000000000001)
huber(y_pred, y_true, delta=100.0)    # np.float64(0.1675)   delta huge: every residual is "small"
0.5 * mse(y_pred, y_true)             # np.float64(0.1675)   ... so it is half the MSE
huber(y_pred, y_true, delta=1e-3) / 1e-3   # np.float64(0.39962500000000006)   delta tiny: (almost) the MAE
```

`np.where(condition, a, b)` picks, element by element, from `a` where the condition is `True` and from `b` where it is `False`: here the parabola for small residuals and the straight line for big ones. Both `a` and `b` are computed for every element; `np.where` only chooses between them.

The last two lines show the dial: a huge \( \delta \) turns Huber into (half) the MSE, and a tiny \( \delta \), divided by \( \delta \), turns it into the MAE. In between, \( \delta \) is the answer to "how big a miss do I still believe is honest noise, rather than a broken data point?". PyTorch's `nn.HuberLoss` uses this exact formula with `delta=1.0` by default.

!!! warning "Common mistake: a delta in the wrong units"
    \( \delta \) is measured in the **target's units**. The default 1.0 is sensible for ratings (1 star) and silly for review counts, where every residual is in the thousands. Rerun the best-constant search on `reviews` with several deltas:

    ```python
    def huber_const(c, y, delta):
        a = np.abs(c - y)
        return np.mean(np.where(a <= delta, 0.5 * (c - y) ** 2, delta * (a - 0.5 * delta)))

    for d in [1.0, 1_000.0, 10_000.0, 10_000_000.0]:
        losses_d = np.array([huber_const(c, reviews, d) for c in c_grid])
        print(f"delta = {d:>12,.0f}   best constant = {c_grid[losses_d.argmin()]:,.0f}")
    # delta =            1   best constant = 1,640
    # delta =        1,000   best constant = 1,760
    # delta =       10,000   best constant = 4,415
    # delta =   10,000,000   best constant = 17,755
    ```

    (The format spec `{d:>12,.0f}` means: right-align in 12 characters, `,` as thousands separator, 0 decimals.) With \( \delta = 1 \) every residual is "big", so Huber behaves exactly like MAE and lands on the MAE answer, 1,640. With \( \delta \) of ten million every residual is "small" and it lands on the mean. No error is raised in either case: you silently trained a different loss from the one you meant. Fix it by choosing \( \delta \) in the target's units (for example, a typical "honest" error size), or by standardizing the target first (Day 22) so that \( \delta = 1 \) means "one standard deviation".

### The gradient is the pull

Why does the shape matter so much? Because training (Day 51) moves the parameters along the **gradient** of the loss, and each example contributes its own derivative \( d\ell/dr \) to that gradient. Think of it as how hard that example **pulls** on the model:

| Loss | Per example \( \ell(r) \) | Derivative \( d\ell/dr \) | Pull of a point that is 10 off |
|---|---|---|---|
| Squared | \( r^2 \) | \( 2r \) | 20: grows without limit |
| Absolute | \( \lvert r \rvert \) | \( \operatorname{sign}(r) \): \( -1 \), \( +1 \) (and 0 at \( r = 0 \), by convention) | 1: same as any other point |
| Huber | as above | \( r \) inside, \( \pm\delta \) outside | \( \delta \): capped |

`np.sign(r)` returns \( -1 \) for negative numbers, \( +1 \) for positive ones and 0 for exactly 0. That 0 is a convention: \( |r| \) has a kink at 0, so it has no derivative there, and libraries pick 0, one of the slopes between \( -1 \) and \( +1 \) that "touch" the kink (such a stand-in is called a **subgradient**).

Under MSE, a point 10 off pulls with \( 2 \times 10 = 20 \) against \( 2 \times 1 = 2 \) for a point 1 off: ten times harder. A point 100 off pulls a hundred times harder. One broken data point can out-shout all the others. Under MAE and Huber the pull is capped.

```python
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
rr = np.linspace(-3, 3, 601)
delta = 1.0
inside = np.abs(rr) <= delta
shapes = [
    ("squared  r²", rr ** 2, 2 * rr, "#3949ab"),
    ("absolute  |r|", np.abs(rr), np.sign(rr), "#00897b"),
    ("Huber, δ = 1", np.where(inside, 0.5 * rr ** 2, delta * (np.abs(rr) - 0.5 * delta)),
     np.where(inside, rr, delta * np.sign(rr)), "#ef6c00"),
]
for name, loss_vals, grad_vals, color in shapes:
    axes[0].plot(rr, loss_vals, color=color, lw=2, label=name)
    axes[1].plot(rr, grad_vals, color=color, lw=2, label=name)
axes[0].set_ylim(-0.2, 5)
axes[0].set_title("loss of one example")
axes[0].set_ylabel("ℓ(r)")
axes[1].set_title("its derivative: how hard the example pulls")
axes[1].set_ylabel("dℓ/dr")
for ax in axes:
    ax.axvspan(-delta, delta, color="#fff59d", alpha=0.5, lw=0)   # shade the band |r| <= delta
    ax.set_xlabel("residual r = prediction − truth")
    ax.legend(frameon=False, loc="upper center")
fig.tight_layout()
fig.savefig("day50-loss-shapes.svg", format="svg", bbox_inches="tight")
```

<figure class="diagram" markdown>
![Left: the per-example loss against the residual from -3 to 3: a blue parabola r squared, a teal V shape |r| and an orange Huber curve that follows half r squared inside the yellow band from -1 to 1 and then continues as straight lines. Right: the derivatives: a blue straight line 2r through the origin, a teal step from -1 to +1 at zero, and an orange curve equal to r inside the band and flat at -1 and +1 outside it](../../assets/img/phase-1/day50-loss-shapes.svg)
<figcaption>Left: the three per-example losses. Right: their derivatives. Squared error's pull grows forever; absolute error's pull is constant and jumps at 0; Huber's is smooth near 0 and capped beyond δ (yellow band).</figcaption>
</figure>

## Robustness on real data: a line through corrupted rows

Time for a real model with two parameters. Apps with more installs have more reviews, and on a log scale (Day 25) the relationship is close to a straight line. Predict \( y = \log_{10}(\text{reviews}) \) from \( x = \log_{10}(\text{installs}) \) with \( \hat{y} = w\,x + b \): slope \( w \), intercept \( b \).

```python
x = np.log10(rated["installs"].to_numpy())
y = np.log10(rated["reviews"].to_numpy())
X = np.column_stack([x, np.ones_like(x)])     # Day 25: a column of ones for the intercept
w_ls, b_ls = np.linalg.lstsq(X, y)[0]         # least squares = the MSE-best line
w_ls.round(3), b_ls.round(3)                  # (np.float64(0.95), np.float64(-1.022))
```

`np.linalg.lstsq` returns four things (the solution, the residual sum of squares, the rank of `X` and its singular values, as on Day 25); `[0]` keeps only the solution \( [w, b] \), which the assignment unpacks into two names.

How good is that line? The fair yardstick is the **baseline** of Day 33 and of the constant model above: ignore \( x \) and predict the training mean of \( y \) for every app. Any regression model has to beat it to be worth anything, and the Month 2 milestone (Day 55) asks you to beat it by a stated margin. The usual way to report the comparison is \( R^2 \) ("R squared"), the fraction of the baseline's MSE that the model removes:

\[
R^2 = 1 - \frac{\text{MSE(model)}}{\text{MSE(predict the mean)}}
\]

```python
mse_line = mse(w_ls * x + b_ls, y)
mse_baseline = mse(np.full(len(y), y.mean()), y)
mse_line, mse_baseline                # (np.float64(0.10937640736253515), np.float64(0.8905311782840248))
1 - mse_line / mse_baseline           # np.float64(0.8771784637868673)   R²
```

\( R^2 = 0.877 \): the line removes 87.7% of the baseline's squared error. \( R^2 = 0 \) means no better than the mean, and a negative \( R^2 \) means worse than the mean, a sure sign of a bug or a badly broken model.

Least squares (Day 25) *is* the MSE-best line: it minimizes \( \frac{1}{n}\sum_i (w x_i + b - y_i)^2 \). A slope of 0.95 on log scales means reviews grow almost in proportion to installs, and \( 10^{-1.022} = 0.095 \) to three decimals: roughly one review per 10 installs.

Now simulate a broken pipeline. Suppose a scraper bug recorded **1 review** for 30 random apps (about 6.5% of them), so their \( y = \log_{10} 1 = 0 \). This corruption is ours, added on purpose; the dataset itself is fine. Fit the line three ways, on the clean and the corrupted data:

- **MSE**: `lstsq`, as above.
- **MAE**: there's no one-line formula, and gradient descent is tomorrow, so use a search. For a fixed slope \( w \), the best intercept under MAE is the median of \( y_i - w x_i \) (that's the constant-model result again, applied to what's left after the slope). So we only need to search over \( w \).
- **Huber** (\( \delta = 1 \), so a miss of more than a factor of 10 counts as "broken"): a plain grid search over both \( w \) and \( b \).

```python
def fit_mse(x, y):
    return np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y)[0]   # [0]: the solution [w, b]

def fit_mae(x, y, w_grid=np.linspace(0.5, 1.5, 1001)):
    best = (np.inf, 0.0, 0.0)
    for w in w_grid:
        b = np.median(y - w * x)                 # best intercept for this slope
        loss = np.mean(np.abs(w * x + b - y))
        if loss < best[0]:
            best = (loss, w, b)
    return np.array(best[1:])

def fit_huber(x, y, delta=1.0, w_grid=np.linspace(0.5, 1.5, 201),
              b_grid=np.linspace(-2, 0, 401)):
    best = (np.inf, 0.0, 0.0)
    for w in w_grid:
        r = w * x + b_grid[:, None] - y          # (401, n): every intercept at once
        a = np.abs(r)
        loss = np.where(a <= delta, 0.5 * r ** 2, delta * (a - 0.5 * delta)).mean(axis=1)
        i = loss.argmin()
        if loss[i] < best[0]:
            best = (loss[i], w, b_grid[i])
    return np.array(best[1:])

rng = np.random.default_rng(50)
bad = rng.choice(len(y), size=30, replace=False)   # 30 random apps
y_bad = y.copy()
y_bad[bad] = 0.0                                    # "1 review" -> log10(1) = 0

for name, fit in [("MSE", fit_mse), ("MAE", fit_mae), ("Huber", fit_huber)]:
    w_c, b_c = fit(x, y)
    w_b, b_b = fit(x, y_bad)
    shift = np.mean(np.abs((w_b * x + b_b) - (w_c * x + b_c)))
    print(f"{name:5s}  clean w={w_c:.3f} b={b_c:+.3f}   corrupted w={w_b:.3f} b={b_b:+.3f}"
          f"   predictions moved {shift:.3f}")
# MSE    clean w=0.950 b=-1.022   corrupted w=0.905 b=-1.020   predictions moved 0.201
# MAE    clean w=0.902 b=-0.826   corrupted w=0.908 b=-0.862   predictions moved 0.009
# Huber  clean w=0.945 b=-0.995   corrupted w=0.955 b=-1.110   predictions moved 0.070
```

How the code works, piece by piece:

- `rng.choice(len(y), size=30, replace=False)` draws 30 different row positions from 0 to 461 (`replace=False` means no position is drawn twice), and `y_bad[bad] = 0.0` overwrites those rows in a copy, so `y` itself stays clean.
- Each fitter keeps the best result so far in a tuple `(loss, w, b)`, starting from `np.inf` (infinity) so that the first real loss always beats it, and returns `[w, b]` as an array, which `w_c, b_c = fit(...)` unpacks.
- In `fit_huber`, `b_grid[:, None]` turns the 401 intercepts into a column, shape `(401, 1)` (Day 8's new axis). Against the 462 apps, shape `(462,)`, broadcasting builds a `(401, 462)` table of residuals: row \( j \) holds every app's residual for intercept \( j \). `.mean(axis=1)` averages along each row, giving one Huber loss per intercept, and `argmin` picks the best row. So the loop runs over slopes only, and all intercepts are tried at once.
- In the f-strings, `{name:5s}` pads the name to 5 characters and `{b_c:+.3f}` always prints the sign, so the columns line up.

The last column is the average distance, in \( \log_{10} \) units, between each fit's predictions before and after the corruption. The MSE line moved by 0.201, which is a factor of \( 10^{0.201} = 1.59 \) in reviews: 6.5% bad rows made every prediction substantially wrong. The MAE line moved by less than 0.01, Huber's by 0.07.

Notice also that the *clean* fits differ: MAE's line is a bit flatter than MSE's. That is not an error. MSE fits the **mean** of \( y \) at each \( x \), MAE fits the **median**, and when the noise around the line is lopsided those are different lines. Choosing a loss is choosing which question your model answers.

```python
fig, ax = plt.subplots(figsize=(8, 4.6))
ok = np.ones(len(y), dtype=bool)
ok[bad] = False
jitter = np.random.default_rng(0).uniform(-0.06, 0.06, size=len(x))   # spread the install buckets
ax.scatter(x[ok] + jitter[ok], y_bad[ok], s=10, color="#9fa8da", label="apps")
ax.scatter(x[bad] + jitter[bad], y_bad[bad], s=22, color="#e53935", marker="x",
           label="corrupted: 1 review")
xs = np.array([2.8, 8.2])
w_c, b_c = fit_mse(x, y)
ax.plot(xs, w_c * xs + b_c, color="#555", ls=":", lw=2, label="MSE line on clean data")
for name, fit, color in [("MSE", fit_mse, "#3949ab"), ("MAE", fit_mae, "#00897b"),
                         ("Huber", fit_huber, "#ef6c00")]:
    w_b, b_b = fit(x, y_bad)
    ax.plot(xs, w_b * xs + b_b, color=color, lw=2, label=f"{name} line on corrupted data")
ax.set_xlabel("log10(installs)")
ax.set_ylabel("log10(reviews)")
ax.legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig("day50-robust-fits.svg", format="svg", bbox_inches="tight")
```

<figure class="diagram" markdown>
![Scatter of log10 reviews against log10 installs for 462 apps rising along a straight band, with 30 red crosses on the bottom at zero for the corrupted rows. A grey dotted line shows the MSE fit on clean data. The blue MSE line fitted on corrupted data sits visibly below it and is flatter, pulled down towards the red crosses; the teal MAE and orange Huber lines stay inside the band of points](../../assets/img/phase-1/day50-robust-fits.svg)
<figcaption>Thirty corrupted rows (red) drag the MSE line (blue) down off the cloud. The MAE (teal) and Huber (orange) lines barely notice them.</figcaption>
</figure>

!!! tip "So should you always use MAE or Huber?"
    No. If the big residuals are *real* (a genuinely viral app, a real spike in crash reports), throwing away their pull means your model will underestimate exactly the cases you may care about most. Robust losses are for data you **distrust**; clean the data first (Day 17) when you can. And MSE has strong advantages: a smooth gradient, a **closed-form solution** (a formula you can evaluate directly, here the normal equation from Day 25 computed by `lstsq`, with no step-by-step search), and on Day 47's terms it is the maximum-likelihood loss when the noise is normal. It is the default for good reasons.

!!! android "Android bridge: you already choose losses, as metrics"
    Jetpack **Macrobenchmark** reports each metric, such as `timeToInitialDisplayMs`, as **min, median and max** over the iterations, not as a mean. That's the same reasoning as today: one iteration hit by a GC pause or a thermal throttle would drag a mean around, while the median shrugs it off. Android vitals goes the other way for jank and ANRs: it reports *rates of bad events* because the outliers are the user experience you need to fix. "Do the extreme cases matter to me, or are they noise?" is the question behind every choice between MSE, MAE and Huber.

## Classification: binary cross-entropy (log loss)

Now the other big family: **classification**, predicting a yes/no label. Will this user open the notification? Is this message spam? Labels are \( y_i \in \{0, 1\} \) (1 = "yes"), and a good classifier doesn't only say yes or no; it outputs a **probability** \( p_i \) that the answer is yes. On Day 30 you met the function that produces it: the sigmoid squashes any score into \( (0, 1) \).

How should we score a probability against a 0/1 label? The standard answer is **binary cross-entropy**, also called **log loss**:

\[
\ell(p, y) = -\big[\, y \ln p + (1 - y)\ln(1 - p) \,\big]
\qquad\qquad
\text{BCE} = \frac{1}{n}\sum_{i=1}^{n} \ell(p_i, y_i)
\]

It looks heavier than it is, because one of the two terms is always zero:

- If \( y = 1 \), the second term vanishes and \( \ell = -\ln p \): the minus log of the probability you gave to the **right answer**.
- If \( y = 0 \), the first term vanishes and \( \ell = -\ln(1 - p) \): again the minus log of the probability you gave to the right answer (which is "no", with probability \( 1 - p \)).

So log loss is one sentence: **minus the log of the probability the model gave to what actually happened**. \( \ln \) is the natural logarithm (`np.log`). Since a probability is at most 1, its log is at most 0, and the minus sign makes the loss positive: 0 for a perfectly confident correct answer, growing without limit as the probability of the truth goes to 0.

**Worked example.** Four predictions:

| Example | True label \( y \) | Model's \( p \) | Probability given to the truth | Loss \( -\ln(\cdot) \) |
|---|---|---|---|---|
| A | 1 | 0.9 | 0.9 | 0.105 |
| B | 1 | 0.6 | 0.6 | 0.511 |
| C | 0 | 0.2 | 0.8 | 0.223 |
| D | 1 | 0.01 | 0.01 | 4.605 |

The average is \( (0.105 + 0.511 + 0.223 + 4.605)/4 = 1.361 \) to three decimals. Example D, a **confident mistake** (1% sure, and wrong), costs more than the other three together.

```python
y_cls = np.array([1, 1, 0, 1])
p = np.array([0.9, 0.6, 0.2, 0.01])
per_example = -(y_cls * np.log(p) + (1 - y_cls) * np.log(1 - p))
per_example.round(3)          # array([0.105, 0.511, 0.223, 4.605])
per_example.mean()            # np.float64(1.3611249691815295)

def bce(p, y, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)          # keep log away from log(0) = -inf
    return np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))

bce(p, y_cls)                       # np.float64(1.3611249691815295)
bce(np.full(4, 0.5), y_cls)         # np.float64(0.6931471805599453)   the know-nothing model: always 50/50
np.log(2)                           # np.float64(0.6931471805599453)
```

Two new functions here. `np.clip(p, eps, 1 - eps)` returns a copy of `p` with every value below `eps` raised to `eps` and every value above `1 - eps` lowered to it, so the logs never see exactly 0 or 1 (why that matters is in the box below). `np.full(4, 0.5)` makes an array of four 0.5s.

!!! warning "Common mistake: log(0), and the inf and nan that end a training run"
    Compute the formula by hand on a model that is *certain*, once wrongly (says 0 for a true 1) and once rightly (says 1 for a true 1):

    ```python
    p_sure = np.array([0.0, 1.0])
    y_sure = np.array([1, 1])
    -(y_sure * np.log(p_sure) + (1 - y_sure) * np.log(1 - p_sure))
    # RuntimeWarning: divide by zero encountered in log
    # RuntimeWarning: invalid value encountered in multiply
    # array([inf, nan])
    bce(p_sure, y_sure)       # np.float64(13.815510557964773)   clipped: large, but a real number
    ```

    The first example is a real infinity: \( -\ln 0 = \infty \), the honest price of being certain and wrong. The second should cost 0, but the code still evaluates the unused term: \( (1 - y)\ln(1 - p) = 0 \times \ln 0 = 0 \times (-\infty) \), which is `nan` in floating point. One `nan` in the loss makes the gradient `nan`, and one update spreads it into every parameter. Recognise it by the `RuntimeWarning` lines and by a loss that prints `inf` or `nan`. Fix it by clipping (as `bce` does, with \( \varepsilon = 10^{-12} \), "epsilon", a tiny number) or, better, by computing the loss from logits, below.

A useful landmark: a model that always says 0.5 scores \( \ln 2 = 0.6931 \) (to four decimals) whatever the labels. A classifier with a log loss above 0.693 is doing worse than a coin flip at expressing its confidence.

### Why not use squared error on the probability?

You could score \( (p - y)^2 \) instead. It works, but it barely notices confident mistakes, because it can never exceed 1:

```python
for q in [0.9, 0.6, 0.3, 0.1, 0.01, 0.001]:
    print(f"p = {q:<6} -ln p = {-np.log(q):.3f}   (1 - p)^2 = {(1 - q) ** 2:.3f}")
# p = 0.9    -ln p = 0.105   (1 - p)^2 = 0.010
# p = 0.6    -ln p = 0.511   (1 - p)^2 = 0.160
# p = 0.3    -ln p = 1.204   (1 - p)^2 = 0.490
# p = 0.1    -ln p = 2.303   (1 - p)^2 = 0.810
# p = 0.01   -ln p = 4.605   (1 - p)^2 = 0.980
# p = 0.001  -ln p = 6.908   (1 - p)^2 = 0.998
```

(`{q:<6}` left-aligns `q` in a field 6 characters wide, so the columns line up; `<` means left, as `>` meant right earlier.)

Going from 1% to 0.1% on an example whose truth was "yes" barely moves the squared error (0.980 to 0.998), but adds 2.3 to the log loss. Log loss insists that a model which is sure and wrong must pay, and that pressure is what makes classifiers produce honest probabilities.

<figure class="diagram" markdown>
![Animation: a plot of loss against the probability p the model gave to the true answer, from 0 to 1. A red curve, -ln p, shoots up towards the left edge; a dashed blue curve, (1 - p) squared, rises gently and stays below 1. An orange dot slides along the red curve through p = 0.9, 0.6, 0.3, 0.1 and 0.01 while a panel shows the matching values: log loss 0.105, 0.511, 1.204, 2.303, 4.605 against squared error 0.010, 0.160, 0.490, 0.810, 0.980](../../assets/img/phase-1/day50-log-loss.svg)
<figcaption>Animated: as the probability given to the truth falls towards 0, log loss explodes while squared error stays below 1.</figcaption>
</figure>

### Where the formula comes from: maximum likelihood

On Day 47 you fitted a coin's \( p \) by **maximum likelihood**: pick the parameters that make the observed data most probable. Apply that to a classifier. Treat each label as a Bernoulli coin (Day 37) whose probability of "1" is the model's \( p_i \). The probability the model assigns to the label that actually happened is \( p_i \) when \( y_i = 1 \) and \( 1 - p_i \) when \( y_i = 0 \), which one formula covers: \( p_i^{\,y_i}(1 - p_i)^{1 - y_i} \). Assuming the examples are independent, the likelihood of the whole dataset is the product:

\[
\text{likelihood} = \prod_{i=1}^{n} p_i^{\,y_i}(1 - p_i)^{1 - y_i}
\]

(\( \prod \) is "multiply them all", the product version of \( \sum \).) Take the log. Two log rules from Day 47 do the work: \( \ln(ab) = \ln a + \ln b \) turns the product into a sum, and \( \ln(a^k) = k\ln a \) brings the exponents down:

\[
\ln(\text{likelihood}) = \sum_{i=1}^{n} \big[\, y_i \ln p_i + (1 - y_i)\ln(1 - p_i) \,\big]
\]

Maximizing this is the same as minimizing its negative, and dividing by \( n \) doesn't move the best parameters:

\[
-\frac{1}{n}\ln(\text{likelihood}) = \frac{1}{n}\sum_{i=1}^{n} -\big[\, y_i \ln p_i + (1 - y_i)\ln(1 - p_i) \,\big] = \text{BCE}
\]

That is exactly the BCE formula. **Minimizing log loss is maximum likelihood for a yes/no model**, in the same way that Day 47 showed that minimizing MSE is maximum likelihood when the noise is normal. Day 54 generalizes it to many classes and connects it to information theory.

### Logits and the numerically safe version

In practice the model doesn't output \( p \) directly. Its last layer outputs a raw score \( z \) (any real number, called a **logit**), and \( p = \sigma(z) = 1/(1 + e^{-z}) \). Computing `np.log(sigmoid(z))` is a numerical trap: for \( z = -800 \), \( e^{800} \) is bigger than the largest float64 (about \( 1.8 \times 10^{308} \)), NumPy warns `RuntimeWarning: overflow encountered in exp` and returns `inf`, so \( \sigma(z) = 1/(1 + \infty) \) becomes exactly 0 and its log is \( -\infty \) (with `RuntimeWarning: divide by zero encountered in log`). Writing the loss in terms of \( z \) avoids that. Two identities make it possible:

- From \( \sigma(z) = \dfrac{1}{1 + e^{-z}} \): \( -\ln\sigma(z) = -\ln 1 + \ln(1 + e^{-z}) = \ln(1 + e^{-z}) \), since \( \ln 1 = 0 \) and \( \ln(1/a) = -\ln a \).
- For the other term: \( 1 - \sigma(z) = \dfrac{1 + e^{-z} - 1}{1 + e^{-z}} = \dfrac{e^{-z}}{1 + e^{-z}} \). Multiply top and bottom by \( e^{z} \): \( \dfrac{1}{e^{z} + 1} \). So \( -\ln(1 - \sigma(z)) = \ln(1 + e^{z}) \).

Substituting both into the loss:

\[
\ell(z, y) = y\,\ln(1 + e^{-z}) + (1 - y)\ln(1 + e^{z})
\]

NumPy's `np.logaddexp(a, b)` computes \( \ln(e^a + e^b) \), so `np.logaddexp(0, t)` is \( \ln(1 + e^t) \). It never forms \( e^{800} \). Instead it pulls the bigger exponent out first, \( \ln(e^a + e^b) = \max(a, b) + \ln\big(1 + e^{-|a - b|}\big) \), so the exponent it does compute is never positive and \( e^{-|a-b|} \) lies between 0 and 1. The last step uses `np.log1p(u)`, which computes \( \ln(1 + u) \) accurately even when \( u \) is tiny (where `np.log(1 + u)` would round \( 1 + u \) to exactly 1 first). For \( t = 800 \):

```python
np.logaddexp(0, 800.0)                       # np.float64(800.0)
max(0.0, 800.0) + np.log1p(np.exp(-800.0))   # np.float64(800.0)   e^-800 underflows harmlessly to 0
np.log1p(1e-20), np.log(1 + 1e-20)           # (np.float64(1e-20), np.float64(0.0))   why log1p exists
```


```python
def central_diff(f, x, h=1e-5):            # Day 29
    return (f(x + h) - f(x - h)) / (2 * h)

def sigmoid(z):                             # Day 30
    return 1 / (1 + np.exp(-z))

def bce_logits(z, y):
    return np.mean(y * np.logaddexp(0, -z) + (1 - y) * np.logaddexp(0, z))

z = np.log(p / (1 - p))           # the logits that give our four probabilities
z.round(3)                        # array([ 2.197,  0.405, -1.386, -4.595])
bce_logits(z, y_cls)              # np.float64(1.3611249691815295)   same 1.361 as before

with np.errstate(over="ignore", divide="ignore"):   # silence the warnings we expect
    naive = -np.log(sigmoid(np.array([-800.0])))
naive, np.logaddexp(0, 800.0)     # (array([inf]), np.float64(800.0))   the naive way says inf; the safe way says 800
```

`with np.errstate(over="ignore", divide="ignore"):` switches off NumPy's overflow and divide-by-zero warnings only inside the `with` block, because here we *expect* them. Never wrap real training code in it: those warnings are your early alarm. `z = np.log(p / (1 - p))` is the inverse of the sigmoid (the "log-odds"), used here only to get logits that match our four probabilities.

This is why PyTorch has both `nn.BCELoss` (takes probabilities) and `nn.BCEWithLogitsLoss` (takes logits, and is the one its docs recommend for numerical stability).

There's a bonus. The derivative of this loss with respect to the logit is astonishingly simple: \( \dfrac{d\ell}{dz} = \sigma(z) - y = p - y \), "prediction minus truth", the same shape as MSE's residual. (The chain rule from Day 31 with \( \sigma' = \sigma(1 - \sigma) \) from Day 30 gives it in three lines; Exercise 8 walks you through it.) Check it numerically:

```python
for z0, y0 in [(2.0, 1), (2.0, 0), (-1.0, 1)]:
    numeric = central_diff(lambda t: bce_logits(np.array([t]), np.array([y0])), z0)
    print(f"z = {z0:+.1f}, y = {y0}:  numeric {numeric:+.6f}   p - y = {sigmoid(z0) - y0:+.6f}")
# z = +2.0, y = 1:  numeric -0.119203   p - y = -0.119203
# z = +2.0, y = 0:  numeric +0.880797   p - y = +0.880797
# z = -1.0, y = 1:  numeric -0.731059   p - y = -0.731059
```

## Why accuracy can't be the training loss

For classifiers, what you usually care about is **accuracy**: the fraction of examples where the predicted class (say, \( p \ge 0.5 \)) is right. So why not minimize "1 − accuracy" directly? Because its gradient is useless.

A tiny made-up dataset (clearly synthetic): 24 users, the feature \( x \) is how many sessions they had in their first week, and \( y = 1 \) if they were still active a month later. The model has one parameter, a threshold \( t \): \( p = \sigma(2(x - t)) \), so it predicts "active" when \( x \ge t \).

```python
rng = np.random.default_rng(50)
x_cls = np.concatenate([rng.normal(3.0, 1.2, 12), rng.normal(5.5, 1.2, 12)])   # sessions
y_users = np.concatenate([np.zeros(12), np.ones(12)])                          # 1 = active

def accuracy(t):
    return np.mean((x_cls >= t) == y_users)

def log_loss(t):
    return bce_logits(2 * (x_cls - t), y_users)

ts = np.linspace(1, 8, 1401)
acc = np.array([accuracy(t) for t in ts])
ll = np.array([log_loss(t) for t in ts])
ts[acc.argmax()], acc.max()       # (np.float64(3.935), np.float64(0.8333333333333334))
ts[ll.argmin()], ll.min()         # (np.float64(3.82), np.float64(0.6252607345066151))

central_diff(accuracy, 4.0), central_diff(log_loss, 4.0)   # (np.float64(0.0), np.float64(0.08507980081007281))
```

Reading the code: `rng.normal(3.0, 1.2, 12)` draws 12 numbers from a normal distribution with mean 3 and standard deviation 1.2 (Day 37), and `np.concatenate` glues the two groups of 12 into one array of 24. In `accuracy`, `x_cls >= t` is an array of 24 `True`/`False` predictions; `== y_users` compares each with its label (`True == 1.0` is `True` in Python, as is `False == 0.0`), and the mean of a boolean array is the fraction of `True`s, which is the accuracy.

Accuracy can only change when the threshold crosses a data point, so as a function of \( t \) it is a **staircase**: flat almost everywhere, with jumps. Its derivative is 0 on every flat part and undefined at the jumps, and that's exactly what `central_diff` reports: 0. A gradient of 0 tells the optimizer "nothing you do nearby makes any difference", which is false and useless. Log loss changes smoothly with \( t \), so its derivative always says which way is better and roughly by how much.

!!! warning "Common mistake: training on the metric you report"
    Try it. Start both from \( t = 2 \) and repeat Day 32's downhill step, \( t \leftarrow t - 0.5 \times \text{slope} \), 30 times: once on \( 1 - \text{accuracy} \), once on log loss.

    ```python
    t_acc = t_ll = 2.0
    for step in range(30):
        t_acc = t_acc - 0.5 * central_diff(lambda t: 1 - accuracy(t), t_acc)
        t_ll = t_ll - 0.5 * central_diff(log_loss, t_ll)
    t_acc, accuracy(t_acc)            # (np.float64(2.0), np.float64(0.5416666666666666))   never moved
    round(t_ll, 3), accuracy(t_ll)    # (np.float64(3.818), np.float64(0.7916666666666666))  walked to the bottom
    ```

    The accuracy run never moves: every slope it sees is exactly 0, so no error appears, the "training" finishes, and the model is as bad as it started. That silence is the symptom. The fix is the split below: train on a smooth loss, report the metric. Notice also that the log-loss run reaches 79.2% accuracy, not the best possible 83.3%: the two scores disagree slightly about where the best threshold is, which is normal.

```python
fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True)
axes[0].plot(ts, acc, color="#00897b", lw=2)
axes[0].set_ylabel("accuracy")
axes[0].set_title("Accuracy is a staircase: its slope is 0 almost everywhere")
axes[1].plot(ts, ll, color="#3949ab", lw=2)
axes[1].set_ylabel("log loss")
axes[1].set_title("Log loss is smooth: its slope always points somewhere useful")
axes[1].set_xlabel("threshold parameter t")
for ax in axes:
    for xi, yi in zip(x_cls, y_users):
        ax.axvline(xi, color="#e53935" if yi == 0 else "#ef6c00", lw=0.6, alpha=0.35)
fig.tight_layout()
fig.savefig("day50-accuracy-vs-logloss.svg", format="svg", bbox_inches="tight")
```

<figure class="diagram" markdown>
![Two stacked plots against the threshold t from 1 to 8, with faint vertical lines at each user's session count. Top: accuracy is a staircase that steps up from 0.5 to a peak of about 0.83 near t = 4 and steps back down, flat between data points. Bottom: log loss is a smooth U-shaped curve whose lowest point is near the same place](../../assets/img/phase-1/day50-accuracy-vs-logloss.svg)
<figcaption>Same model, same data, two scores. Accuracy only changes when the threshold crosses a data point (faint lines); log loss changes smoothly, so gradient descent can follow it.</figcaption>
</figure>

This gives the standard split you'll see in every training script from now on:

- the **training loss** (MSE, Huber, log loss, ...) is chosen so it's smooth and its gradient is informative;
- the **metrics** (accuracy, MAE in stars, RMSE, and others such as precision, the fraction of "yes" predictions that were right, which Month 3 covers) are what you report to humans. They needn't be differentiable, because nobody takes their gradient.

One more distinction matters as soon as you train for real. The average loss over the examples you train on has a name: the **empirical risk** ("empirical" because it is measured on the data you happen to have; "risk" is the expected loss on *new* data, which is what you actually care about). Minimizing the training loss is a means to an end. So you always keep some examples aside, never trained on, and compute the same loss on them: the **validation loss** (used while developing) and the **test loss** (measured once, at the end). A training loss that keeps falling while the validation loss rises means the model is memorizing its training examples. Day 51 plots loss curves step by step, and the Day 55 milestone plots training and validation curves side by side.

The training loss and the metrics usually move together, but not always. Exercise 5 builds a case where the model with the better accuracy has the worse log loss.

## The loss is a function of the parameters

This is the single most important shift in viewpoint today. When you write `mse(y_pred, y_true)`, it looks like a function of *data*. During training, though, the data is fixed: the same \( x_i, y_i \) every time. What changes is the parameters. So the loss is really a function of the parameters alone:

\[
L(w, b) = \frac{1}{n}\sum_{i=1}^{n} \big( w\,x_i + b - y_i \big)^2
\]

Read it carefully: \( x_i \) and \( y_i \) are fixed numbers baked into the formula (462 of each, for our apps), and \( w, b \) are the variables. In code, that is a function that **closes over** the data and takes only the parameters:

```python
def make_line_loss(x, y):
    def L(params):                       # the only input: the parameters
        w, b = params
        return np.mean((w * x + b - y) ** 2)
    return L

L = make_line_loss(x, y)                 # data baked in, once
L(np.array([0.0, 0.0]))                  # np.float64(11.192030857561296)   a terrible line
L(np.array([1.0, -1.0]))                 # np.float64(0.17094498580357298)   a decent guess
L(np.array([w_ls, b_ls]))                # np.float64(0.10937640736253515)   the least-squares line: the minimum
```

`make_line_loss` returns the inner function `L`, and `L` remembers the `x` and `y` it was created with: a **closure**. You've written the same thing in Kotlin many times:

```kotlin
// Kotlin: a function that returns a lambda capturing x and y
fun makeLineLoss(x: DoubleArray, y: DoubleArray): (DoubleArray) -> Double = { params ->
    val (w, b) = params                      // destructuring, like Python's  w, b = params
    x.indices.sumOf { i -> (w * x[i] + b - y[i]).let { it * it } } / x.size
}
```

Same idea in both languages: the data is captured once, and callers see only `(parameters) -> loss`. In both, the lambda captures the variables themselves rather than a copy of the arrays, so if code changed the array's contents later, the loss would change with it. Nothing here does, which is the point: during training the data stays put.

This is exactly how training code is organised: the training loop only ever gets `L`, a black box from "parameters" to "one number", and its gradient. Days 51 to 53 are entirely about how to walk around on that function.

With one parameter the loss is a **curve**, like the constant model's above and the knob animation at the top. With two parameters it is a **surface** over the \( (w, b) \) plane, which we draw as a contour map (Day 32): each line joins parameter settings with equal loss.

```python
W, B = np.meshgrid(np.linspace(0.55, 1.35, 81), np.linspace(-2.8, 0.8, 91))
mse_surface = np.mean((W[..., None] * x + B[..., None] - y) ** 2, axis=-1)
mae_surface = np.mean(np.abs(W[..., None] * x + B[..., None] - y), axis=-1)
mse_surface.shape                       # (91, 81)   one loss value per (b, w) pair

w_mae, b_mae = fit_mae(x, y)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, surf, name, (wm, bm), color in [
        (axes[0], mse_surface, "MSE", (w_ls, b_ls), "#3949ab"),
        (axes[1], mae_surface, "MAE", (w_mae, b_mae), "#00897b")]:
    levels = np.quantile(surf, [0.01, 0.05, 0.15, 0.3, 0.5, 0.7, 0.9])
    cs_ = ax.contour(W, B, surf, levels=levels, colors=color, linewidths=1)
    ax.clabel(cs_, fmt="%.2f", fontsize=8)
    ax.plot(wm, bm, marker="*", color="#e53935", markersize=14)
    ax.set_xlabel("slope w")
    ax.set_ylabel("intercept b")
    ax.set_title(f"{name}(w, b): minimum at w={wm:.3f}, b={bm:.3f}", fontsize=10)
fig.tight_layout()
fig.savefig("day50-loss-surface.svg", format="svg", bbox_inches="tight")
```

Line by line:

- `np.meshgrid(ws, bs)` takes 81 slopes and 91 intercepts and returns two `(91, 81)` arrays: `W[i, j]` is the slope and `B[i, j]` the intercept at grid point `(i, j)`. Every row of `W` repeats the 81 slopes; every column of `B` repeats the 91 intercepts. Together they list all 7,371 pairs.
- `W[..., None]` adds a last axis of length 1, making shape `(91, 81, 1)` (`...` means "all the existing axes"). Against `x`, shape `(462,)`, broadcasting (Day 9) builds a `(91, 81, 462)` array: every grid point's residual for every app. That is 3.4 million float64 values, 27 MB, which is why the grid is kept fairly coarse: a 181 × 161 grid would need 108 MB per temporary array, heavy for the in-browser Python.
- `np.mean(..., axis=-1)` averages over the last axis, the apps, leaving one loss per grid point: shape `(91, 81)`.
- `np.quantile(surf, [0.01, ...])` returns the values below which 1%, 5%, ... of the grid's losses fall. Using them as contour levels puts more lines near the bottom of the valley, where the interesting detail is. `ax.contour` draws one line per level, and `ax.clabel` writes the loss value on each line.

!!! warning "Common mistake: forgetting the new axis"
    Without `[..., None]`, NumPy lines up the shapes `(91, 81)` and `(462,)` from the right, finds 81 against 462, and refuses:

    ```python
    try:
        np.mean((W * x + B - y) ** 2)
    except ValueError as err:
        print(err)
    # operands could not be broadcast together with shapes (91,81) (462,)
    ```

    This time the error is loud, which is good. Read the two shapes in the message, decide which axis should pair with which (here: the apps must get their *own* axis), and add it with `None`. If, by bad luck, the grid had been 462 wide, there'd be no error at all, and you'd get a wrong surface, so check the result's shape too.

<figure class="diagram" markdown>
![Two contour maps over slope w from 0.55 to 1.35 and intercept b from -2.8 to 0.8. Left, MSE: nested smooth ellipses stretched into a long narrow valley running diagonally from top left to bottom right, with a red star at w = 0.950, b = -1.022. Right, MAE: a similar diagonal valley with its red star at w = 0.902, b = -0.826](../../assets/img/phase-1/day50-loss-surface.svg)
<figcaption>A 2-parameter loss surface, seen from above. Each contour joins (w, b) settings with equal loss. Both losses form a long diagonal valley, with their lowest points in different places.</figcaption>
</figure>

Three things to read off this picture, each of which becomes a lesson this week:

1. **There is a lowest point** (the red star). Finding it is the whole job of training. Here we found it with `lstsq` and a search; on Day 51 you'll walk to it with gradient descent.
2. **The valley is long, thin and diagonal.** Moving \( w \) up a little and \( b \) down a lot keeps the loss nearly the same, because \( x \) is around 4.5, never near 0, so slope and intercept can trade off against each other. Thin valleys make gradient descent zig-zag; Day 52 shows how momentum and feature scaling (standardizing \( x \), Day 22) fix it.
3. **MSE's surface is a smooth bowl with one bottom** (it is a parabola in every direction). Losses with a single bottom and no false dips are called **convex**: that's Day 53.

!!! android "Android bridge: the loss doesn't ship"
    The `.tflite` model you load into a LiteRT `Interpreter` contains the **forward pass only**: input tensor in, prediction out. The loss existed only during training, on a server, and it's gone; the frozen weights are the bottom of the valley it led to. The exception proves the rule: LiteRT's **on-device training** support works by exporting extra *signatures* from the model (for example a `train` function) that compute a loss and update the weights, so that personalization can run on the phone. If you ever see a model with a `train` signature, the loss function is baked in there.

## Choosing a loss: a cheat sheet

| Task | Loss | Gradient per example | Use when | PyTorch |
|---|---|---|---|---|
| Regression | MSE | \( 2r \) | The default; noise roughly normal; big errors really are bad | `nn.MSELoss` |
| Regression | MAE | \( \operatorname{sign}(r) \) | Outliers you don't trust; you want the median | `nn.L1Loss` |
| Regression | Huber | \( r \), capped at \( \pm\delta \) | Mostly MSE-like, but a few broken points | `nn.HuberLoss` |
| Yes/no classification | Binary cross-entropy | \( p - y \) (w.r.t. the logit) | Any model that outputs a probability | `nn.BCEWithLogitsLoss` |
| Many classes | Cross-entropy | \( \mathbf{p} - \mathbf{y} \) | Next to everything with a softmax (Day 54) | `nn.CrossEntropyLoss` |

## Where this shows up in ML

- **Every training loop** is "compute the loss, compute its gradient, update the parameters". The loss line is the one place where you tell the model what you want.
- **Linear regression** (the Month 2 milestone) is MSE plus gradient descent; **logistic regression** (Month 3) is log loss plus gradient descent.
- **Neural network classifiers**, from MNIST digits to the image classifier in an Android app, are trained with cross-entropy, and LLMs are trained with the same loss on "which token comes next".
- **Loss curves** (loss against training step) are the main thing you watch while training. You'll plot them from Day 51 on.
- **Robust losses** like Huber appear wherever labels are noisy: object-detection box regression (as "smooth L1") and reinforcement learning (DQN) both use them.

## Going further

These are real and useful, but not needed this month. Each line says where to read more.

- **Quantile (pinball) loss**: an MAE that is tilted, so the best constant is the 90th percentile instead of the median; used to predict ranges ("90% of downloads finish within X seconds"). [Wikipedia: Quantile regression](https://en.wikipedia.org/wiki/Quantile_regression).
- **Hinge loss**: the loss of support vector machines, which only cares whether an example is on the right side of the boundary by a margin. Month 3's classic ML covers it. [Wikipedia: Hinge loss](https://en.wikipedia.org/wiki/Hinge_loss).
- **Focal loss**: a cross-entropy that down-weights easy examples, for heavily imbalanced data such as object detection. [Lin et al., "Focal Loss for Dense Object Detection" (arXiv)](https://arxiv.org/abs/1708.02002).
- **Class weights** (`pos_weight` in PyTorch's `BCEWithLogitsLoss`): make mistakes on a rare class cost more. [PyTorch docs: BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html).
- **Brier score and calibration**: the squared error on probabilities you met above, used to check whether "70% sure" really means right 70% of the time. [scikit-learn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html).
- **Regularization terms**: adding \( \lambda\|\mathbf{w}\|^2 \) to the loss to keep weights small (Day 22's preview, Day 33's exercise); Phase 2 makes it a habit.
- **Multi-class cross-entropy and KL divergence**: Day 54, next week.

---

## Exercises

Two warm-ups, three core exercises, then a predict-the-output, a find-the-bug and a challenge. Every solution below re-creates its own inputs and helper functions, so you can run it on its own.

1. **Warm-up: the regression losses by hand.** True values \( [2, 4, 6, 8] \), predictions \( [3, 4, 4, 9] \). Compute MSE, RMSE, MAE and Huber with \( \delta = 1 \) on paper, then check in NumPy.
2. **Warm-up: log loss by hand.** Labels \( [1, 0, 1] \), predicted probabilities of "1" \( [0.8, 0.3, 0.5] \). Write down the probability each prediction gave to the *true* answer, take \( -\ln \) of each, and average. Check in NumPy.
3. **Core: the median's flat bottom.** The best constant under MAE is the median. Check it with a grid search for the data \( [1, 2, 3, 10] \), then for \( [1, 2, 3, 10, 50] \). In the first case, which values of \( c \) are all equally good, and why? What did adding the outlier 50 do?
4. **Core: the best constant probability.** Labels are 30% ones and 70% zeros (10 labels). A "model" predicts the same probability \( q \) for everyone. Find the \( q \) that minimizes log loss by grid search, and the loss there. Compare it with the coin-flip model \( q = 0.5 \), and connect the answer to Day 47.
5. **Core: accuracy and log loss disagree.** Build two sets of predicted probabilities for the labels \( [1, 1, 1, 0] \) so that model A has **higher accuracy** but **worse log loss** than model B. Explain in one sentence what A is doing wrong.
6. **Predict the output.** Before running it, write down the six numbers this prints. Then run it and explain any you got wrong.

    ```python
    import numpy as np

    def mse(y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def mae(y_pred, y_true):
        return np.mean(np.abs(y_pred - y_true))

    y6 = np.array([1.0, 2.0, 3.0])
    a6 = np.array([1.0, 2.0, 6.0])    # model A
    b6 = np.array([2.0, 3.0, 4.0])    # model B
    print(mse(a6, y6), mse(b6, y6))
    print(mae(a6, y6), mae(b6, y6))
    print(np.mean(a6 - y6), np.mean(b6 - y6))
    ```

7. **Find and fix the bug.** A teammate used to scikit-learn, whose `log_loss(y_true, y_pred)` takes the labels first, scores a classifier with this lesson's `bce`. The predictions look decent, but the loss comes out as 6.9, far worse than a coin flip. Find the bug, fix it, and explain where 6.9 comes from.

    ```python
    import numpy as np

    def bce(p, y, eps=1e-12):
        p = np.clip(p, eps, 1 - eps)
        return np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))

    labels = np.array([1, 0, 1, 1])
    probs = np.array([0.9, 0.2, 0.7, 0.6])
    bce(labels, probs)        # np.float64(6.9077597033758495)   ?!
    ```

8. **Challenge: the gradient of log loss with respect to the logit.** Show that \( \dfrac{d\ell}{dz} = \sigma(z) - y \) for \( \ell(z, y) = -[y\ln\sigma(z) + (1 - y)\ln(1 - \sigma(z))] \). (Hint: chain rule from Day 31 through \( s = \sigma(z) \), and \( \sigma'(z) = \sigma(z)(1 - \sigma(z)) \) from Day 30.) Then check it with `central_diff` at 5 random \( z \) for both labels, using the stable `bce_logits`, and report the largest disagreement.

??? success "Solutions"
    **1.** Residuals (prediction minus truth) \( [1, 0, -2, 1] \). MSE \( = (1 + 0 + 4 + 1)/4 = 1.5 \). RMSE \( = \sqrt{1.5} = 1.2247 \) to four decimals. MAE \( = (1 + 0 + 2 + 1)/4 = 1 \). Huber with \( \delta = 1 \): \( |1| \le 1 \) gives \( \tfrac12 \cdot 1 = 0.5 \); 0 gives 0; \( |{-2}| > 1 \) gives \( 1 \times (2 - 0.5) = 1.5 \); the last gives 0.5. Average \( 2.5/4 = 0.625 \).

    ```python
    import numpy as np

    def mse(y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def mae(y_pred, y_true):
        return np.mean(np.abs(y_pred - y_true))

    def huber(y_pred, y_true, delta=1.0):
        r = y_pred - y_true
        a = np.abs(r)
        return np.mean(np.where(a <= delta, 0.5 * r ** 2, delta * (a - 0.5 * delta)))

    t1 = np.array([2.0, 4.0, 6.0, 8.0])
    p1 = np.array([3.0, 4.0, 4.0, 9.0])
    mse(p1, t1), np.sqrt(mse(p1, t1)), mae(p1, t1), huber(p1, t1, delta=1.0)
    # (np.float64(1.5), np.float64(1.224744871391589), np.float64(1.0), np.float64(0.625))
    ```

    The one 2-unit miss is 4 of the 6 squared units in the MSE (67%) but only 2 of the 4 absolute units in the MAE (50%): squaring gives the biggest miss the biggest say. Huber treats it linearly because it lies outside \( \delta = 1 \).

    **2.** The probabilities given to the truth are 0.8 (label 1), \( 1 - 0.3 = 0.7 \) (label 0) and 0.5 (label 1). Their minus logs are \( 0.2231, 0.3567, 0.6931 \), and the average is \( 1.2729/3 = 0.4243 \).

    ```python
    import numpy as np

    y2 = np.array([1, 0, 1])
    p2 = np.array([0.8, 0.3, 0.5])
    per2 = -(y2 * np.log(p2) + (1 - y2) * np.log(1 - p2))
    per2            # array([0.22314355, 0.35667494, 0.69314718])
    per2.mean()     # np.float64(0.4243218919376292)
    ```

    The third prediction, a shrug at 0.5, costs \( \ln 2 \) whatever the label: it is the coin-flip landmark. The second was a fairly confident correct "no", so it is the cheapest after the first.

    **3.**

    ```python
    import numpy as np

    def mae(y_pred, y_true):
        return np.mean(np.abs(y_pred - y_true))

    grid = np.linspace(0, 60, 6001)                  # steps of 0.01
    for data in [np.array([1.0, 2, 3, 10]), np.array([1.0, 2, 3, 10, 50])]:
        losses = np.array([mae(c, data) for c in grid])
        best = grid[np.isclose(losses, losses.min())]
        print(data, "median", np.median(data), " best c from", best.min().round(2),
              "to", best.max().round(2), " MAE", losses.min().round(3))
    # [ 1.  2.  3. 10.] median 2.5  best c from 2.0 to 3.0  MAE 2.5
    # [ 1.  2.  3. 10. 50.] median 3.0  best c from 3.0 to 3.0  MAE 11.4
    ```

    `np.isclose(losses, losses.min())` marks every grid point whose loss equals the minimum up to floating-point rounding, so `best` lists all the tied winners. With four values, every \( c \) between the two middle values 2 and 3 has two points below and two above: moving \( c \) inside that range adds as much distance to two points as it removes from the other two, so the MAE curve is flat there. With five values the middle one, 3, is the single best. The outlier 50 raised the MAE value, but it moved the best \( c \) only from "anywhere in [2, 3]" to 3; under MSE it would have dragged the answer from 4 to 13.2.

    **4.**

    ```python
    import numpy as np

    def bce(p, y, eps=1e-12):
        p = np.clip(p, eps, 1 - eps)
        return np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))

    y4 = np.array([1] * 3 + [0] * 7)                  # 30% ones
    qs = np.linspace(0.01, 0.99, 99)
    losses4 = np.array([bce(np.full(10, q), y4) for q in qs])
    qs[losses4.argmin()].round(2), losses4.min().round(4)    # (np.float64(0.3), np.float64(0.6109))
    bce(np.full(10, 0.5), y4).round(4)                       # np.float64(0.6931)
    ```

    `[1] * 3 + [0] * 7` builds the Python list `[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]`. The best constant probability is the fraction of ones, 0.3: exactly Day 47's maximum-likelihood estimate of a coin, \( \hat{p} = k/n \), because minimizing log loss *is* maximizing the likelihood. Its loss, 0.6109, is the floor any useful classifier on this data must beat; the coin flip's 0.6931 is worse because it ignores even the base rate.

    **5.** Model A gets three right but is absurdly confident on the one it gets wrong. Model B gets only two right, sitting a little on the wrong side of 0.5 twice, but is never extreme.

    ```python
    import numpy as np

    def bce(p, y, eps=1e-12):
        p = np.clip(p, eps, 1 - eps)
        return np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))

    y5 = np.array([1, 1, 1, 0])
    p_A = np.array([0.9, 0.9, 0.001, 0.1])     # 3 of 4 correct, one very confident miss
    p_B = np.array([0.8, 0.8, 0.45, 0.55])     # 2 of 4 correct, never extreme
    acc5 = lambda q: np.mean((q >= 0.5) == y5)
    acc5(p_A), acc5(p_B), bce(p_A, y5).round(3), bce(p_B, y5).round(3)   # (np.float64(0.75), np.float64(0.5), np.float64(1.806), np.float64(0.511))
    ```

    A bets almost everything against the truth on one example, and log loss charges \( -\ln 0.001 = 6.908 \) for that single bet, which dominates its average. Accuracy only counts right versus wrong, so it cannot see the difference between "slightly wrong" and "absurdly overconfident".

    **6.** Predictions first. Model A's residuals are \( [0, 0, 3] \): MSE \( 9/3 = 3 \), MAE \( 3/3 = 1 \), mean residual \( 3/3 = 1 \). Model B's are \( [1, 1, 1] \): MSE 1, MAE 1, mean residual 1.

    ```python
    import numpy as np

    def mse(y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def mae(y_pred, y_true):
        return np.mean(np.abs(y_pred - y_true))

    y6 = np.array([1.0, 2.0, 3.0])
    a6 = np.array([1.0, 2.0, 6.0])
    b6 = np.array([2.0, 3.0, 4.0])
    print(mse(a6, y6), mse(b6, y6))
    print(mae(a6, y6), mae(b6, y6))
    print(np.mean(a6 - y6), np.mean(b6 - y6))
    # 3.0 1.0
    # 1.0 1.0
    # 1.0 1.0
    ```

    Same total error, spread differently. MAE and the mean residual (the bias) can't tell the models apart; MSE prefers B three to one, because A concentrates all its error in one big miss and squaring punishes concentration. If you predicted a smaller MSE for A, you averaged before squaring: the mean of the squares is not the square of the mean (Day 39's \( \mathbb{E}[g(X)] \ne g(\mathbb{E}[X]) \)).

    **7.** The function is fine; the **call** passes the arguments in scikit-learn's order, labels first, but `bce` expects `(p, y)`. Inside, `p` is the labels and `y` the probabilities. Clipping turns the label 0 into \( 10^{-12} \) and the label 1 into \( 1 - 10^{-12} \), and \( -\ln 10^{-12} = 27.63 \). Each example then costs \( 27.63 \times |\text{label} - \text{prob}| \): \( 27.63 \times 0.1 \), \( \times 0.2 \), \( \times 0.3 \) and \( \times 0.4 \), which is 2.763, 5.526, 8.289 and 11.052. They sum to \( 27.63 \times (0.1 + 0.2 + 0.3 + 0.4) = 27.63 \), and \( 27.63 / 4 = 6.908 \). No error is raised because both arrays have the same shape and the formula accepts any numbers.

    ```python
    import numpy as np

    def bce(p, y, eps=1e-12):
        p = np.clip(p, eps, 1 - eps)
        return np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))

    labels = np.array([1, 0, 1, 1])
    probs = np.array([0.9, 0.2, 0.7, 0.6])
    bce(labels, probs)            # np.float64(6.9077597033758495)   the bug: arguments swapped
    bce(probs, labels)            # np.float64(0.2990011586691898)   fixed
    bce(p=probs, y=labels)        # np.float64(0.2990011586691898)   better: keywords make the order irrelevant
    -np.log(1e-12)                # np.float64(27.631021115928547)
    p_sw = np.clip(labels, 1e-12, 1 - 1e-12)          # what bce did to the labels
    -(probs * np.log(p_sw) + (1 - probs) * np.log(1 - p_sw))
    # array([ 2.76310432,  5.52620422,  8.28931297, 11.0524173 ])
    ```

    Two habits prevent it: call loss functions with keyword arguments when the order isn't obvious, and sanity-check any log loss against the 0.693 coin-flip landmark. A "trained" classifier scoring 6.9 is a bug until proven otherwise.

    **8.** Let \( s = \sigma(z) \). The chain rule gives \( \dfrac{d\ell}{dz} = \dfrac{d\ell}{ds}\cdot\dfrac{ds}{dz} \).

    - First factor: \( \ell = -y\ln s - (1 - y)\ln(1 - s) \), and \( \frac{d}{ds}\ln s = \frac{1}{s} \), \( \frac{d}{ds}\ln(1 - s) = -\frac{1}{1 - s} \). So \( \dfrac{d\ell}{ds} = -\dfrac{y}{s} + \dfrac{1 - y}{1 - s} \).
    - Second factor: \( \dfrac{ds}{dz} = s(1 - s) \).
    - Multiply: \( \left(-\dfrac{y}{s} + \dfrac{1 - y}{1 - s}\right)s(1 - s) = -y(1 - s) + (1 - y)s = -y + ys + s - ys = s - y \).

    ```python
    import numpy as np

    def central_diff(f, x, h=1e-5):
        return (f(x + h) - f(x - h)) / (2 * h)

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def bce_logits(z, y):
        return np.mean(y * np.logaddexp(0, -z) + (1 - y) * np.logaddexp(0, z))

    rng8 = np.random.default_rng(5)
    zs = rng8.normal(0, 3, 5)
    worst = 0.0
    for z0 in zs:
        for y0 in (0, 1):
            numeric = central_diff(lambda t: bce_logits(np.array([t]), np.array([y0])), z0)
            worst = max(worst, abs(numeric - (sigmoid(z0) - y0)))
    worst          # np.float64(1.4303891404665592e-11)
    ```

    The largest disagreement is about \( 10^{-11} \), the size of `central_diff`'s own rounding error, so the formula is right. Everything that divided by \( s \) or \( 1 - s \) cancelled, which is why the gradient stays finite even when \( s \) is extremely close to 0 or 1: one more reason frameworks compute this loss from logits.

## Resources

- [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course): the "Linear regression" and "Logistic regression" modules cover loss (MSE, MAE, log loss) with interactive exercises.
- [PyTorch docs: loss functions](https://pytorch.org/docs/stable/nn.html#loss-functions): `MSELoss`, `L1Loss`, `HuberLoss`, `BCELoss`, `BCEWithLogitsLoss`, each with its exact formula.
- [Dive into Deep Learning: Linear Regression](https://d2l.ai/chapter_linear-regression/linear-regression.html): the squared loss, and why it is maximum likelihood under Gaussian noise.
- [CS231n: Linear classification](https://cs231n.github.io/linear-classify/): loss functions for classifiers, including the softmax cross-entropy you'll meet on Day 54.
- [Wikipedia: Huber loss](https://en.wikipedia.org/wiki/Huber_loss): the definition, plots, and the "pseudo-Huber" smooth variant.
- [Mathematics for Machine Learning (free book)](https://mml-book.github.io/): chapter 8 frames models, losses and empirical risk minimization; chapter 9 does linear regression as maximum likelihood.
- [Android developers: Macrobenchmark overview](https://developer.android.com/topic/performance/benchmarking/macrobenchmark-overview): the min/median/max reporting mentioned in today's Android bridge.

## Checklist

- [ ] I can say in one sentence what a loss is and why training needs one number.
- [ ] I computed MSE, RMSE and MAE by hand and in NumPy, and I know which units each is in.
- [ ] I can explain why the MSE-best constant is the mean and the MAE-best constant is the median, and I saw it on the reviews column.
- [ ] I can write the Huber loss, and explain its gradient as a "capped pull".
- [ ] I saw 6.5% corrupted rows drag an MSE line far more than MAE and Huber lines.
- [ ] I can compute log loss by hand, explain why a confident mistake costs so much, and compute it safely from logits.
- [ ] I can explain why accuracy is a metric and not a training loss.
- [ ] I can read a loss as a function of the parameters, and I drew a 1-parameter loss curve and a 2-parameter loss surface.
- [ ] I can recognise residuals that cancel, a broadcasting column of targets, `nan` from missing data, `log(0)` and `exp` overflow from their symptoms, and fix each one.

**Next:** [Day 51: Gradient descent](day-51-gradient-descent.md)
