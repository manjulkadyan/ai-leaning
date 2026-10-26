# Day 52: Learning Rates, Mini-Batches, SGD and Momentum

**Phase 1 · Week 8 · ~150 minutes**

## Objectives

- [ ] Explain batch, stochastic and mini-batch gradient descent, measure how a batch gradient's noise shrinks with the batch size, and weigh that against the cost of a step.
- [ ] Define an epoch, compute the number of steps per epoch, and explain why the data is reshuffled every epoch.
- [ ] Write a reusable `minibatch_gd` function in NumPy that covers all three variants.
- [ ] Pick a learning rate with a log-spaced sweep, and explain how the learning rate and the batch size set the size of SGD's noise band.
- [ ] Explain momentum with the rolling-ball picture, write its two update equations, work two steps by hand on a ravine, and lower the learning rate when you add momentum.
- [ ] Use Day 51's curvatures and condition number to explain why plain GD needs about \( \kappa \) steps and well-tuned momentum about \( \sqrt{\kappa} \), and show both on the Play Store data with unscaled features.
- [ ] Write Adam's update with its default settings, and say honestly what it does and does not fix.
- [ ] Use step-decay, cosine and warmup learning-rate schedules, and explain why decaying the learning rate helps noisy SGD.

**Prerequisites.** [Day 51](day-51-gradient-descent.md) (the gradient descent update, the learning rate and its limit \( 2/\text{curvature} \), stretched bowls, the Hessian, curvatures as eigenvalues, the condition number \( \kappa \), and standardizing on the Play Store data), [Day 50](day-50-loss-functions.md) (the MSE loss and loss surfaces), [Day 39](day-39-expectation-variance.md) (a mini-batch gradient is a noisy, unbiased average whose noise shrinks like \( 1/\sqrt{B} \)), [Day 33](day-33-vector-matrix-gradients.md) (the MSE gradient \( \frac{2}{n}X^\top(X\mathbf{w} - \mathbf{y}) \)), [Day 24](day-24-eigenvectors.md) (eigenvalues and eigenvectors of a symmetric matrix) and [Day 22](day-22-norms-distances.md) (standardizing features).

---

## The big idea

Yesterday's gradient descent was a careful hiker: before every single step, it surveyed the **whole** mountain (all \( n \) training examples) to find the exact downhill direction. That is fine for 1,000 examples. A real training set has millions of images or sentences, and surveying all of them for one step is absurdly slow.

Today's lesson is three upgrades that every real training loop uses:

| Upgrade | The idea in one line | What it fixes |
|---|---|---|
| **Mini-batches (SGD)** | Estimate the downhill direction from a small random sample of examples | Each step costs a few examples instead of all of them |
| **Momentum** | Keep a running velocity, so consistent directions build up speed and back-and-forth wobbles cancel | Slow zigzagging in long, narrow valleys |
| **Learning-rate schedules** (and **Adam**) | Change the step size during training, or per parameter | Noise near the end of training; parameters on very different scales |

**Analogy for an app developer.** You want to know whether a new release made your app slower. You could wait for every user's session report (exact, but you'd wait days), or look at a random 1% of sessions every few minutes (slightly noisy, but you can react 100 times as often). Training with mini-batches is the second option: many cheap, slightly wrong steps beat a few expensive, exact ones, because the errors average out (Day 39's law of large numbers).

Momentum is a heavy ball instead of a careful hiker. A ball rolling down a narrow gutter doesn't bounce from wall to wall forever: its sideways bounces cancel out while its speed along the gutter keeps building up. It is also exactly what a `RecyclerView` fling feels like: the list keeps moving after your finger lifts, carrying the velocity you gave it, and friction slowly bleeds it away.

## The setup: a problem we know the answer to

To see what each method does, we need a problem whose best answer we know exactly. Here's a small linear regression with two weights: 1,000 examples, two features that are correlated with each other, and targets made from a known line plus noise. We reuse Day 33's loss and gradient. Notice that `grad` works on **any subset of rows**: give it all 1,000 rows and it's the full gradient; give it 32 rows and it's a mini-batch gradient.

```python
import numpy as np

rng = np.random.default_rng(52)
n = 1_000
x1 = rng.normal(0, 1, size=n)
x2 = 0.6 * x1 + 0.8 * rng.normal(0, 1, size=n)    # a second feature, correlated with the first
X = np.column_stack([x1, x2])                      # (1000, 2): two standardized-ish features
w_true = np.array([2.0, -1.0])
y = X @ w_true + rng.normal(0, 0.5, size=n)        # a known line plus noise

def loss(w, X, y):
    """Mean squared error of the predictions X @ w (Day 50)."""
    r = X @ w - y
    return r @ r / len(y)

def grad(w, X, y):
    """Gradient of the MSE (Day 33): (2/n) X^T (Xw - y). Works for any subset of rows."""
    return 2 / len(y) * X.T @ (X @ w - y)

w_best = np.linalg.lstsq(X, y)[0]
L_best = loss(w_best, X, y)
w_best, L_best          # (array([ 2.02675518, -1.02494252]), np.float64(0.253458966949242))
```

(NumPy 2 shows a single number as `np.float64(0.253458966949242)` when it is echoed, as the ▶ Run button and Jupyter do; `print()` shows the plain `0.253458966949242`. The comments on this page show exactly what the Run button prints.)

`np.linalg.lstsq` gives the exact best weights (Day 25). It returns a tuple of four things (the solution, the leftover squared error, the rank of `X` and its singular values), and `[0]` keeps the first, the weights. It lets us always measure how far a method is from the finish: the **gap** \( L(\mathbf{w}) - L(\mathbf{w}^*) \), where \( \mathbf{w}^* \) ("w star") is the best weight vector. The best loss isn't 0 because of the noise in `y`: about \( 0.5^2 = 0.25 \) is the floor no line can beat.

## Batch, stochastic and mini-batch gradient descent

### The full gradient is an average

Day 33's gradient is an **average** over the examples:

\[
\nabla L(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^{n} \nabla \ell_i(\mathbf{w})
\qquad\text{where}\qquad
\ell_i(\mathbf{w}) = (\mathbf{x}_i \cdot \mathbf{w} - y_i)^2
\]

Symbol by symbol:

- \( \ell_i \) ("little L sub i") is the loss on example \( i \) alone: the squared error of one prediction.
- \( \nabla \ell_i(\mathbf{w}) = 2(\mathbf{x}_i \cdot \mathbf{w} - y_i)\,\mathbf{x}_i \) is that one example's gradient: "error times input" (Day 33's rule 1).
- \( \nabla L \) averages those \( n \) per-example gradients.

Three ways to compute the step direction follow from how many terms of that average you actually compute. Write \( B \) for the **batch size**, the number of examples used per step:

| Name | Batch size \( B \) | Each step uses | Steps per pass over the data |
|---|---|---|---|
| **Batch** (full-batch) gradient descent | \( n \) | every example | 1 |
| **Stochastic** gradient descent (SGD) | 1 | one random example | \( n \) |
| **Mini-batch** gradient descent | in between, e.g. 32 | a random handful | \( n / B \) |

"Stochastic" means "random". In practice almost everybody uses mini-batches and still calls it SGD; PyTorch's optimizer is named `torch.optim.SGD` and is used with batches of 32 or 256. The update is always Day 51's rule, with the gradient computed on the batch only. If \( \mathcal{B} \) is the set of example numbers in the batch,

\[
\mathbf{w} \leftarrow \mathbf{w} - \eta \cdot \frac{1}{B}\sum_{i \in \mathcal{B}} \nabla \ell_i(\mathbf{w})
\]

where \( \eta \) ("eta") is the learning rate.

"One random example" needs one clarification. Day 39's analysis drew each batch independently, *with replacement*, so the same example could turn up twice in a row. Real training loops instead shuffle the data once per epoch and walk through it in order (next section), which is sampling *without replacement*: every example is used exactly once per epoch. The batch gradients are still unbiased and their noise is very slightly smaller, so everything Day 39 showed carries over.

### Worked example: one SGD step by hand

Start from the guess \( \mathbf{w} = [-1, 2] \) and take one SGD step using only example 0:

```python
w0 = np.array([-1.0, 2.0])         # the starting guess used all lesson
grad(w0, X, y)                     # array([-2.16378682,  2.62807864])   the full gradient
grad(w0, X[:1], y[:1])             # array([-0.04130884, -0.0475873 ])   from example 0 alone
grad(w0, X[:32], y[:32])           # array([-1.08088259,  3.37989395])   average of the first 32
```

By hand: example 0 has \( \mathbf{x}_0 = [-0.826, -0.951] \) and \( y_0 = -1.102 \).

1. Prediction: \( \mathbf{x}_0 \cdot \mathbf{w} = (-0.826)(-1) + (-0.951)(2) = 0.826 - 1.903 = -1.077 \).
2. Error: \( -1.077 - (-1.102) = 0.025 \). This one example is already almost perfectly predicted.
3. Gradient: \( 2 \times 0.025 \times [-0.826, -0.951] = [-0.041, -0.048] \).
4. Step with \( \eta = 0.05 \): \( \mathbf{w} \leftarrow [-1, 2] - 0.05 \times [-0.041, -0.048] = [-0.998, 2.002] \).

```python
X[0], y[0]                  # (array([-0.82595023, -0.95148495]), np.float64(-1.1020265379722878))
pred = X[0] @ w0
pred                        # np.float64(-1.0770196803364291)
g0 = 2 * (pred - y[0]) * X[0]
g0                          # array([-0.04130884, -0.0475873 ])
w0 - 0.05 * g0              # array([-0.99793456,  2.00237936])
loss(w0, X, y), loss(w0 - 0.05 * g0, X, y)
# (np.float64(7.50297886883199), np.float64(7.5047790088730055))
```

Look at the last line: this step made the full loss slightly **worse** (7.5030 → 7.5048). The full gradient says "move \( w_2 \) down a lot" (+2.63 means uphill is towards bigger \( w_2 \)); example 0, which happens to be well predicted already, says "nudge \( w_2 \) up a hair". A single example's gradient can point the wrong way. That is fine: Day 39 showed that the average of these noisy gradients is the full gradient, so over many steps the wrong-way nudges cancel and the right-way pull wins. The 32-example batch, \( [-1.08, 3.38] \), already points roughly the right way.

### Epochs and shuffling

One full pass over the training data is called an **epoch**. With \( n \) examples and batch size \( B \), an epoch has

\[
\text{steps per epoch} = \left\lceil \frac{n}{B} \right\rceil
\]

where \( \lceil \cdot \rceil \) ("ceiling") rounds up, because the leftover examples form one last, smaller batch. **Worked example:** \( n = 1000 \), \( B = 32 \): \( 1000 / 32 = 31.25 \), so 31 full batches of 32 (992 examples) plus a last batch of 8, which makes 32 steps.

```python
n_ex, B_ex = 1_000, 32
-(-n_ex // B_ex)                 # 32     ceil(1000 / 32) with integer arithmetic
n_ex - (n_ex // B_ex) * B_ex     # 8      the last batch has only 8 examples
```

(`-(-a // b)` is a common Python idiom for "ceiling of a divided by b" without floats: `//` rounds down, and negating twice turns it into rounding up.)

At the start of every epoch the examples are **shuffled** into a new random order, then cut into consecutive batches:

<figure class="diagram" markdown>
![Animation: a row of 12 data cards numbered 1 to 12; below it, epoch 1 shows the cards shuffled into the order 5 7 11 1, 2 4 9 8, 3 6 10 12, split into three batches of four that light up one at a time, labelled step 1, step 2 and step 3; then epoch 2 shows a new shuffle 5 11 7 4, 1 12 6 3, 9 8 2 10, whose batches light up as steps 4, 5 and 6; a note says steps per epoch = ceil(12 / 4) = 3 and every example is used exactly once per epoch](../../assets/img/phase-1/day52-epochs-batches.svg)
<figcaption>Animated: each epoch reshuffles the data and cuts it into batches; each batch gives one gradient step, so 12 examples with B = 4 make 3 steps per epoch.</figcaption>
</figure>

Why shuffle? Two reasons:

1. **Data files are often sorted**: by date, by user, by label. Batches cut from sorted data are not random samples, so their gradients are no longer **unbiased** estimates of the full gradient (Day 39's word for "right on average": averaged over many random batches, an unbiased estimate equals the true value), and the model gets dragged towards whatever the last batches of each epoch contain.
2. **A new order every epoch** means the noise in epoch 2 is different from the noise in epoch 1, so it averages out instead of repeating the same mistakes.

### A reusable mini-batch gradient descent

Here's the function the rest of the lesson uses. `batch_size=len(y)` makes it batch GD and `batch_size=1` makes it pure SGD. It also records the full-data loss after every step, which a real training loop would *not* do (that costs a full pass every step); we do it only to draw the curves.

```python
def minibatch_gd(X, y, w0, lr, batch_size, epochs, rng, shuffle=True, schedule=None):
    """Mini-batch gradient descent. batch_size=len(y) is batch GD, batch_size=1 is SGD.
    Returns the final weights, every visited point, and the full-data loss after every step."""
    n = len(y)
    steps_per_epoch = -(-n // batch_size)          # ceil(n / batch_size)
    total_steps = epochs * steps_per_epoch
    w = np.array(w0, dtype=float)
    path, losses = [w.copy()], [loss(w, X, y)]
    step = 0
    for epoch in range(epochs):
        order = rng.permutation(n) if shuffle else np.arange(n)   # a fresh shuffle every epoch
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]                 # this mini-batch's rows
            eta = lr if schedule is None else schedule(step, total_steps, lr)
            w = w - eta * grad(w, X[idx], y[idx])                 # Day 51's update, on the batch
            step += 1
            path.append(w.copy())
            losses.append(loss(w, X, y))                          # tracked on ALL data, for plotting
    return w, np.array(path), np.array(losses)
```

`rng.permutation(n)` returns the numbers 0 to \( n - 1 \) in a random order: a shuffled list of row numbers. Slicing it in chunks of `batch_size` gives each batch's rows. (The `schedule` argument is for later in the lesson; ignore it for now.)

!!! warning "Common mistake: adding up the batch's gradients instead of averaging them"
    `grad` divides by `len(y)`, the number of rows **in this batch**. A version that forgets the division adds up the per-example gradients, so its size grows with the batch size: a batch of 1,000 gives a gradient 1,000 times too big, and the effective learning rate becomes \( 1000 \times 0.05 = 50 \). Here is what that does, as full-batch steps from `w0`:

    ```python
    def grad_sum(w, X, y):            # BUG: adds up the per-example gradients instead of averaging them
        return 2 * X.T @ (X @ w - y)

    w_sum = w0.copy()
    for step in range(1, 5):
        w_sum = w_sum - 0.05 * grad_sum(w_sum, X, y)
        print(step, w_sum.round(1), f"{loss(w_sum, X, y):.4g}")
    # 1 [ 107.2 -129.4] 1.145e+04
    # 2 [-2217.3  6882.7] 3.577e+07
    # 3 [-210376.5 -579808.2] 5.467e+11
    # 4 [55991379.3 73318185.5] 1.376e+16
    ```

    The loss is multiplied by thousands, then tens of thousands, at every step: the signature of a learning rate far past its limit. A learning rate tuned for one batch size then silently stops working when you change the batch size. Recognise it by a loss that explodes for large batches but not small ones; fix it by dividing by the batch's own length (the last batch of an epoch can be shorter, so never divide by the `batch_size` setting). PyTorch's built-in losses average over the batch by default (`reduction="mean"`) for the same reason.

Now give all three variants the same budget, 10 epochs (each example's gradient computed 10 times), and the same learning rate, 0.05:

```python
runs = {}
for B in (1000, 32, 1):
    w, path, losses = minibatch_gd(X, y, w0, lr=0.05, batch_size=B, epochs=10,
                                   rng=np.random.default_rng(1))
    runs[B] = (path, losses)
    spe = -(-n // B)                                   # steps per epoch
    gap = losses[::spe] - L_best                       # loss above the best, at each epoch's end
    print(f"B={B:5d}  steps={len(losses) - 1:6d}  gap after epochs 1, 2, 5, 10: "
          f"{gap[1]:.4f} {gap[2]:.4f} {gap[5]:.4f} {gap[10]:.4f}")
# B= 1000  steps=    10  gap after epochs 1, 2, 5, 10: 6.6821 6.1605 4.8319 3.2277
# B=   32  steps=   320  gap after epochs 1, 2, 5, 10: 0.5417 0.0441 0.0001 0.0001
# B=    1  steps= 10000  gap after epochs 1, 2, 5, 10: 0.1863 0.0106 0.0274 0.0218
```

Two pieces of syntax in that `print`. Inside an f-string, `{B:5d}` formats an integer right-aligned in 5 characters and `{gap[1]:.4f}` formats a float with 4 digits after the point; later blocks also use `:<6` (left-align in 6 characters), `:.4g` (4 significant digits, switching to scientific notation like `1.145e+04` for very big or small numbers), `:.2e` (always scientific, 2 digits after the point), `:,.1f` (a comma between thousands, as in `84,687.5`) and `:13s` (a string padded to 13 characters). Writing `!s` before the colon, as in `{shuffle!s:5}`, converts the value with `str()` first; without it, `f"{True:5}"` gives `'    1'`, because Python's `bool` is a subclass of `int` and a width spec formats it as the number 1. And `losses[::spe]` is a slice with a step (Day 8): every `spe`-th entry, so the loss at the end of each epoch.

Read the three rows:

- **Batch GD** took only 10 steps in 10 epochs: every step is exact, but there are too few of them. It's still far from the bottom.
- **SGD** took 10,000 steps and was close after a single epoch (gap 0.19), but then it stopped improving: the gap bounces between about 0.01 and 0.03. Every single-example gradient is noisy (Day 39: its typical error was larger than the gradient itself), so near the bottom the steps are mostly noise and the weights jitter around the minimum.
- **Mini-batch** \( B = 32 \) got the best of both: 32 steps per epoch, each with \( \sqrt{32} \approx 5.7 \) times less noise than SGD's, and a final gap of 0.0001.

<figure class="diagram" markdown>
![Left: contour ellipses of the loss with three paths from a black square at (-1, 2) towards a red star at (2.03, -1.02); the blue batch-GD path is one short step; the teal mini-batch path is a slightly wiggly curve reaching most of the way; the orange SGD path jumps around wildly in long strokes and ends in a dense scribble around the star. Right: loss minus best loss on a log scale against epochs from 0 to 10; the blue batch-GD line falls slowly from 7 to 3; the teal mini-batch line falls steeply to about 0.0001 and wiggles there; the faint orange SGD line drops fastest at first and then fills a noisy band between about 0.001 and 0.1](../../assets/img/phase-1/day52-batch-sizes.svg)
<figcaption>Same learning rate, same compute. SGD (orange) gets close fastest but then jitters in a noise band; batch GD (blue) is smooth but takes only one step per epoch; mini-batches (teal) combine many steps with little noise.</figcaption>
</figure>

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
styles = {1000: ("#3949ab", "batch GD (B = 1000)"),
          32: ("#00897b", "mini-batch (B = 32)"),
          1: ("#ef6c00", "SGD (B = 1)")}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
g1, g2 = np.meshgrid(np.linspace(-1.5, 3, 200), np.linspace(-2, 2.5, 200))   # a 200×200 grid of (w1, w2), as on Day 32
W = np.stack([g1.ravel(), g2.ravel()], axis=1)                 # every grid point as a weight vector
Lgrid = ((W @ X.T - y) ** 2).mean(axis=1).reshape(g1.shape)    # the loss at each grid point
ax1.contour(g1, g2, Lgrid, levels=L_best + np.array([0.05, 0.3, 1, 2.5, 5, 8]),
            colors="#9e9e9e", linewidths=0.8)
for B in (1, 32, 1000):
    path, _ = runs[B]
    first_epoch = path[: -(-n // B) + 1]
    color, label = styles[B]
    ax1.plot(first_epoch[:, 0], first_epoch[:, 1], "-", color=color, lw=1 if B == 1 else 1.8,
             marker="o" if B == 1000 else None, ms=4, label=label)
ax1.plot(*w0, "ks", ms=6)                  # *w0 unpacks [-1, 2] into x and y; "ks" = black (k) square (s) marker
ax1.plot(*w_best, "*", color="#e53935", ms=14, label="best weights")
ax1.set_xlabel("$w_1$")
ax1.set_ylabel("$w_2$")
ax1.set_title("The first epoch's path")
ax1.legend(frameon=False, loc="upper right", fontsize=9)

for B in (1, 32, 1000):
    _, losses = runs[B]
    epochs_axis = np.arange(len(losses)) / -(-n // B)          # step count in units of epochs
    every = 5 if B == 1 else 1                                 # thin out SGD's 10,000 points
    color, label = styles[B]
    ax2.plot(epochs_axis[::every], (losses - L_best)[::every], color=color,
             lw=0.5 if B == 1 else 1.6, alpha=0.6 if B == 1 else 1, label=label)
ax2.set_yscale("log")
ax2.set_xlabel("epochs (passes over the data)")
ax2.set_ylabel("loss − best loss (log scale)")
ax2.set_title("Same compute budget, same learning rate 0.05")
ax2.legend(frameon=False, loc="lower left", fontsize=9)
fig.tight_layout()
fig.savefig("day52-batch-sizes.svg", format="svg", bbox_inches="tight")
```

To be fair to batch GD: its steps are exact, so it can afford a much **bigger** learning rate than noisy SGD. With \( \eta = 0.5 \) it reaches a gap of 0.0003 in its 10 steps:

```python
w, path, losses = minibatch_gd(X, y, w0, lr=0.5, batch_size=1000, epochs=10,
                               rng=np.random.default_rng(1))
(losses - L_best)[[1, 2, 5, 10]].round(4)      # array([2.657e+00, 9.738e-01, 4.800e-02, 3.000e-04])
```

(NumPy switched to scientific notation, `2.657e+00`, because the array's largest and smallest entries differ by more than a factor of 1,000; it then writes every entry the same way so the column lines up.)

So on 1,000 examples batch GD is perfectly usable. The case for mini-batches grows with \( n \): with 10 million examples, one batch-GD step costs 10 million example-gradients, while a mini-batch step costs 32 and already points roughly the right way. Mini-batch SGD has typically made a lot of progress before batch GD has taken its first step.

**Noise against batch size, measured.** Day 39 derived that a mini-batch gradient's noise shrinks like \( 1/\sqrt{B} \). Check it on this problem: draw 4,000 random batches of each size at `w0`, and measure the typical distance between each batch gradient and the full gradient.

```python
g_full = grad(w0, X, y)
rng_b = np.random.default_rng(9)
for B in (1, 4, 16, 64):
    dists = []
    for _ in range(4000):
        idx = rng_b.choice(n, size=B, replace=False)                 # B different random rows
        dists.append(np.linalg.norm(grad(w0, X[idx], y[idx]) - g_full))   # distance to the full gradient (Day 22)
    typical = np.sqrt(np.mean(np.square(dists)))                     # root-mean-square distance
    print(f"B={B:3d}  typical error of a batch gradient = {typical:.3f}")
# B=  1  typical error of a batch gradient = 8.463
# B=  4  typical error of a batch gradient = 4.215
# B= 16  typical error of a batch gradient = 2.101
# B= 64  typical error of a batch gradient = 1.013
```

`rng_b.choice(n, size=B, replace=False)` picks `B` different row numbers out of 0 to \( n - 1 \). The root-mean-square (square, average, square root) is the natural "typical size" here, the same recipe as a standard deviation. Every 4 times bigger batch halves the error, as \( 1/\sqrt{B} \) predicts. Compare the error at \( B = 1 \) with the full gradient's own length, \( \|[-2.16, 2.63]\| = 3.40 \): a single example's gradient is mostly noise.

**How to pick \( B \).** The noise of a mini-batch gradient shrinks like \( 1/\sqrt{B} \), but the cost of a step grows like \( B \). Quadrupling \( B \) quadruples the cost and only halves the noise, so the payoff from bigger batches diminishes fast. One thing pulls the other way: GPUs and phone NPUs compute a batch of examples in parallel, so a batch of 32 often takes barely longer than a batch of 1. That is why typical batch sizes are powers of two from about 16 to 512, chosen as big as the memory allows and then with the learning rate tuned for it.

### What shuffling buys, measured

Sort the training data by its target (as if the CSV were sorted by price) and train with and without shuffling:

```python
order = np.argsort(y)                      # row numbers that sort y (Day 8): a file sorted by target
X_sorted, y_sorted = X[order], y[order]
for shuffle in (False, True):
    w, _, losses = minibatch_gd(X_sorted, y_sorted, w0, lr=0.05, batch_size=32, epochs=5,
                                rng=np.random.default_rng(2), shuffle=shuffle)
    print(f"shuffle={shuffle!s:5}  final w = {w.round(3)}  gap after each epoch: {(losses[::32] - L_best)[1:].round(4)}")
# shuffle=False  final w = [ 2.113 -0.986]  gap after each epoch: [0.5671 0.0549 0.0171 0.0135 0.013 ]
# shuffle=True   final w = [ 1.982 -1.048]  gap after each epoch: [0.5638 0.0388 0.0028 0.0011 0.0037]
```

Without shuffling the gap gets stuck at 0.013 from epoch 3 on. The shuffled run keeps improving, to 0.0011 after epoch 4 (12 times lower), and ends at 0.0037, still 3.5 times lower; its last value bounces because a constant learning rate leaves noise (the next section). Every epoch ends with the batches of the *largest* targets, so every epoch ends with the weights dragged towards fitting those examples. Shuffling costs one `rng.permutation` per epoch; never skip it.

!!! warning "Common mistake: shuffling X and y separately"
    Shuffle the **row numbers** once and index both arrays with them, as `X[idx], y[idx]` does above. `rng.shuffle(a)` shuffles the rows of `a` in place, so calling it once on `X` and once on `y` puts them in two *different* random orders and pairs every input with somebody else's target. Nothing crashes. The best any model can then do is predict the average:

    ```python
    rng_bug = np.random.default_rng(6)
    X_bad, y_bad = X.copy(), y.copy()
    rng_bug.shuffle(X_bad)           # shuffles the rows of X_bad in place
    rng_bug.shuffle(y_bad)           # ...and y_bad, in a DIFFERENT random order
    w_bad = np.linalg.lstsq(X_bad, y_bad)[0]       # the best weights for the scrambled pairs
    w_bad, loss(w_bad, X_bad, y_bad)     # (array([ 0.03185694, -0.02319468]), np.float64(2.794729428708777))
    y.var()                              # np.float64(2.7939645192600144)
    ```

    Even the *exact* best weights are about zero, and the loss equals the variance of `y`: the features have stopped carrying any information. Recognise it by a training loss that stalls at about `y.var()` from the very first epoch. In PyTorch, `DataLoader(dataset, batch_size=32, shuffle=True)` keeps inputs and targets together and reshuffles every epoch; `shuffle=False` is the default, so the flag matters.

!!! android "Android bridge: pages, not the whole table"
    You already never load a 100,000-row Room table into one list. The Paging library loads it in pages (`PagingConfig(pageSize = 50)`) as the user scrolls, and scrolling from the top to the bottom touches every row once. Mini-batch training has the same shape: a batch is a page, an epoch is one full scroll, and the reason is the same too: memory and latency. The big difference is the order. Your UI pages in a fixed, sorted order on purpose; a training loop must page in a **random** order, or the gradient estimates stop being fair samples.

### Picking a learning rate

How do you choose \( \eta \) for a new problem? You can't compute it from a formula in a real network, so you try a few values spread over **powers of ten** and keep the best. Each run is short: a few epochs are enough to see which values work.

```python
for lr in (0.001, 0.01, 0.1, 0.3, 1.0):
    _, _, losses = minibatch_gd(X, y, w0, lr=lr, batch_size=32, epochs=3, rng=np.random.default_rng(7))
    print(f"lr={lr:<6} gap after 3 epochs = {losses[-1] - L_best:.4g}")
# lr=0.001  gap after 3 epochs = 6.21
# lr=0.01   gap after 3 epochs = 1.547
# lr=0.1    gap after 3 epochs = 0.003741
# lr=0.3    gap after 3 epochs = 0.04766
# lr=1.0    gap after 3 epochs = 2.454e+61
```

The pattern is the one you'll see on every problem. Too small (0.001) and the run has barely moved. Too big (1.0) and it explodes: \( 10^{61} \). In between there's a best value (0.1 here), and the value next to it on the big side (0.3) is already worse, because each step is noisier. Practitioners start a little below the largest value that doesn't blow up, then refine with values like 0.03 or 0.3. The next sections explain where that "blows up" limit comes from.

### The learning rate sets the size of the noise

Look again at SGD's orange band in the figure: the weights never settle, they hover around the minimum. The size of that hover is set by the learning rate. Each step moves the weights by \( \eta \) times a noisy gradient, so the jitter scales with \( \eta \):

```python
for lr in (0.05, 0.005):
    _, _, losses = minibatch_gd(X, y, w0, lr=lr, batch_size=1, epochs=10, rng=np.random.default_rng(3))
    last = losses[-1000:] - L_best                    # the gap over the final epoch
    print(f"lr={lr:<6} mean gap over the last epoch = {last.mean():.5f}   worst = {last.max():.5f}")
# lr=0.05   mean gap over the last epoch = 0.03998   worst = 0.35974
# lr=0.005  mean gap over the last epoch = 0.00278   worst = 0.02277
```

Ten times smaller learning rate, about 14 times smaller mean gap (0.03998 against 0.00278; the worst values differ by 15.8 times). Here is why it is roughly proportional to \( \eta \) and not to \( \sqrt{\eta} \) or \( \eta^2 \). Near the minimum, each step adds a random kick of size \( \eta \) times the gradient noise, while the pull back towards the minimum removes a fraction proportional to \( \eta \) of the current distance. Those balance when the *squared* distance from the minimum is proportional to \( \eta \), so the typical distance grows like \( \sqrt{\eta} \). The loss gap of a bowl is proportional to the squared distance, so the gap grows like \( \eta \): about 10 times for a 10 times change. The measured 14 times is that, plus the noise of measuring a band from one 1,000-step window.

The batch size enters the same balance through the noise: averaging \( B \) examples divides the gradient's *variance* by \( B \) (Day 39), so the gap shrinks roughly like \( 1/B \). Same learning rate, three batch sizes, and the gap measured over each run's final epoch:

```python
for B in (1, 8, 32):
    _, _, losses = minibatch_gd(X, y, w0, lr=0.05, batch_size=B, epochs=10, rng=np.random.default_rng(3))
    spe = -(-n // B)                                  # steps per epoch
    last = losses[-spe:] - L_best                     # the gap over the final epoch
    print(f"B={B:3d}  mean gap over the last epoch = {last.mean():.6f}   worst = {last.max():.6f}")
# B=  1  mean gap over the last epoch = 0.039983   worst = 0.359741
# B=  8  mean gap over the last epoch = 0.004042   worst = 0.031916
# B= 32  mean gap over the last epoch = 0.000818   worst = 0.004103
```

Eight times bigger batch, about 10 times smaller band; four times bigger again, about 5 times smaller. So you can calm the noise two ways: a smaller learning rate, or a bigger batch. A small learning rate makes the early part of training slower, and a big batch makes every step more expensive. You want a **big** learning rate at the start, to cover ground, and a **small** one at the end, to settle. That's a learning-rate schedule, and we'll build two after momentum.

## Momentum: a ball, not a hiker

### The problem: ravines

Plain gradient descent has a weakness that has nothing to do with noise. Take a loss shaped like a long, narrow valley, a **ravine**:

\[
f(w_1, w_2) = \tfrac{1}{2}\left(w_1^2 + 25\,w_2^2\right)
\]

Its gradient is \( \nabla f = [\,w_1,\; 25\,w_2\,] \). Across the valley (the \( w_2 \) direction) the walls are 25 times steeper than along its floor (the \( w_1 \) direction). Two facts about a single learning rate follow:

- **The steep direction caps the learning rate.** Along \( w_2 \), one step does \( w_2 \leftarrow w_2 - \eta \cdot 25 w_2 = (1 - 25\eta)\,w_2 \). If \( |1 - 25\eta| > 1 \), that factor makes \( w_2 \) grow every step: divergence. So \( \eta \) must stay below \( 2/25 = 0.08 \).
- **The gentle direction is then painfully slow.** Along \( w_1 \), a step does \( w_1 \leftarrow (1 - \eta)\,w_1 \). With \( \eta = 0.07 \), each step removes only 7% of the remaining distance along the floor.

With \( \eta = 0.07 \) the \( w_2 \) factor is \( 1 - 1.75 = -0.75 \): \( w_2 \) flips sign every step, bouncing from wall to wall, while \( w_1 \) crawls. That's the zigzag.

### The momentum update

Momentum adds one new vector, the **velocity** \( \mathbf{v} \), which starts at zero. Each step:

\[
\mathbf{v} \leftarrow \beta\,\mathbf{v} + \nabla f(\mathbf{w})
\qquad\qquad
\mathbf{w} \leftarrow \mathbf{w} - \eta\,\mathbf{v}
\]

- \( \mathbf{v} \) is a running, decaying **sum of past gradients**.
- \( \beta \) ("beta"), the **momentum coefficient**, between 0 and 1, says how much of the old velocity survives each step. Think of \( 1 - \beta \) as friction. \( \beta = 0 \) is plain gradient descent; \( \beta = 0.9 \) is the most common default.
- The weights then move along the velocity, not along the raw gradient.

Why this helps, in the two directions of the ravine:

- **Along the floor** the gradient points the same way step after step, so the velocity keeps adding up. If the gradient were a constant \( g \), the velocity would grow to \( g + \beta g + \beta^2 g + \dots = g / (1 - \beta) \). For \( \beta = 0.9 \) that's \( 10g \): an effective learning rate ten times bigger, in exactly the direction that needed it.
- **Across the valley** the gradient flips sign every step, so consecutive gradients **cancel** inside the velocity. The bouncing dies down.

```python
v = 0.0
for step in range(1, 51):
    v = 0.9 * v + 1.0            # the same gradient, 1.0, every step
    if step in (1, 2, 3, 10, 50):
        print(step, round(v, 4))
# 1 1.0
# 2 1.9
# 3 2.71
# 10 6.5132
# 50 9.9485
```

The velocity approaches \( 1/(1 - 0.9) = 10 \), as promised.

!!! note "Two ways to write momentum"
    PyTorch's `torch.optim.SGD(params, lr=..., momentum=0.9)` uses the form above, \( \mathbf{v} \leftarrow \beta\mathbf{v} + \nabla f \), \( \mathbf{w} \leftarrow \mathbf{w} - \eta\mathbf{v} \). Many textbooks and Keras put the learning rate inside: \( \mathbf{v} \leftarrow \beta\mathbf{v} - \eta\nabla f \), \( \mathbf{w} \leftarrow \mathbf{w} + \mathbf{v} \). With a constant learning rate the two produce exactly the same weights (the second velocity equals \( -\eta \) times the first), so don't let the notation confuse you when you read papers.

### Worked example: two steps by hand

Start at \( \mathbf{w} = [-10, 1] \) with \( \eta = 0.07 \) and \( \beta = 0.7 \).

**Step 1.** The gradient is \( [w_1, 25w_2] = [-10, 25] \). The velocity was zero, so \( \mathbf{v} = [-10, 25] \).
New weights: \( [-10, 1] - 0.07 \times [-10, 25] = [-10 + 0.7,\; 1 - 1.75] = [-9.3, -0.75] \).
This is exactly what plain GD does on step 1, because there is no history yet.

**Step 2.** The gradient is \( [-9.3,\; 25 \times (-0.75)] = [-9.3, -18.75] \).
Velocity: \( 0.7 \times [-10, 25] + [-9.3, -18.75] = [-7 - 9.3,\; 17.5 - 18.75] = [-16.3, -1.25] \).

Look at the two entries. Along \( w_1 \), old and new gradients agreed, and the velocity grew to \( -16.3 \). Along \( w_2 \), they disagreed (\( +25 \) then \( -18.75 \)) and nearly cancelled, to \( -1.25 \).
New weights: \( [-9.3, -0.75] - 0.07 \times [-16.3, -1.25] = [-9.3 + 1.141,\; -0.75 + 0.0875] = [-8.159, -0.6625] \).

Plain GD's second step lands at \( [-8.65, 0.56] \) instead: less progress along the floor and a full bounce to the other wall.

```python
lam = np.array([1.0, 25.0])                  # curvature along w1 (gentle) and w2 (steep)

def f_ravine(w):
    return 0.5 * (lam * w ** 2).sum()

def grad_ravine(w):
    return lam * w

def run_gd(w0, lr, steps):
    w, path = np.array(w0, dtype=float), [np.array(w0, dtype=float)]
    for _ in range(steps):
        w = w - lr * grad_ravine(w)
        path.append(w)
    return np.array(path)

def run_momentum(w0, lr, beta, steps):
    w, v = np.array(w0, dtype=float), np.zeros(len(w0))
    path = [w.copy()]
    for _ in range(steps):
        v = beta * v + grad_ravine(w)        # velocity: a decaying sum of past gradients
        w = w - lr * v                       # move along the velocity, not the raw gradient
        path.append(w.copy())
    return np.array(path)

start = [-10.0, 1.0]
run_gd(start, 0.07, 4).round(2)
# array([[-10.  ,   1.  ],
#        [ -9.3 ,  -0.75],
#        [ -8.65,   0.56],
#        [ -8.04,  -0.42],
#        [ -7.48,   0.32]])
run_momentum(start, 0.07, 0.7, 4).round(2)
# array([[-10.  ,   1.  ],
#        [ -9.3 ,  -0.75],
#        [ -8.16,  -0.66],
#        [ -6.79,   0.56],
#        [ -5.36,   0.44]])
```

Here are both runs for 30 steps, drawn to scale (the ravine really is 5 times longer than it is wide at every height, because \( \sqrt{25} = 5 \)):

<figure class="diagram" markdown>
![Animation in two stacked panels, each showing the same narrow elliptical contours of the ravine with the minimum marked in red on the right. Top, plain gradient descent with learning rate 0.07: the orange point zigzags steeply up and down between the valley walls, the zigzags shrinking as it inches right; after 30 steps it is still well short of the minimum with loss 0.643. Bottom, momentum with learning rate 0.07 and beta 0.7: the point makes one bounce, then sweeps in long strides along the valley floor, overshoots the minimum slightly and settles, reaching loss 0.00154 after 30 steps. Side panels count the steps and show the loss; plain GD gets under 0.001 at step 75 with no bounces, momentum first dips under 0.001 at step 24 and stays under from step 32](../../assets/img/phase-1/day52-ravine-momentum.svg)
<figcaption>Animated: in a ravine, plain gradient descent spends its steps bouncing between the walls; momentum cancels the bounces and builds up speed along the floor.</figcaption>
</figure>

### How much momentum?

Count the steps until the loss drops below 0.001, for a few settings. Momentum's loss can dip below and then bounce back up, so count two things: the **first** step below 0.001, and the step from which it **stays** below:

```python
def steps_to(path, tol=1e-3):
    # returns (first step with loss < tol, step from which the loss stays < tol)
    vals = np.array([f_ravine(p) for p in path])
    below = np.flatnonzero(vals < tol)            # the step numbers where the loss is below tol
    above = np.flatnonzero(vals >= tol)           # ...and where it is not
    if below.size == 0:                           # .size is the number of entries in an array
        return None, None
    return int(below[0]), int(above[-1]) + 1      # first dip; one past the last step above tol

for lr in (0.02, 0.07, 0.08, 0.081):
    p = run_gd(start, lr, 300)
    print(f"GD        lr={lr:<5}            first below, stays below: {steps_to(p)}   final loss {f_ravine(p[-1]):.3g}")
for beta in (0.5, 0.7, 0.9):
    p = run_momentum(start, 0.07, beta, 300)
    print(f"momentum  lr=0.07  beta={beta}  first below, stays below: {steps_to(p)}")
# GD        lr=0.02             first below, stays below: (268, 268)   final loss 0.000272
# GD        lr=0.07             first below, stays below: (75, 75)   final loss 6.15e-18
# GD        lr=0.08             first below, stays below: (None, None)   final loss 12.5
# GD        lr=0.081            first below, stays below: (None, None)   final loss 3.4e+07
# momentum  lr=0.07  beta=0.5  first below, stays below: (30, 30)
# momentum  lr=0.07  beta=0.7  first below, stays below: (24, 32)
# momentum  lr=0.07  beta=0.9  first below, stays below: (87, 106)
```

`np.flatnonzero(condition)` returns the positions where a boolean array is `True`; here those positions are step numbers, because `path` holds the start (step 0) and then one point per step.

Four lessons in eight lines:

1. **Plain GD is fastest a little below the limit** \( 2/25 = 0.08 \): 75 steps at 0.07, but 268 at a timid 0.02.
2. **At exactly the limit it never converges.** At \( \eta = 0.08 \) the \( w_2 \) factor is exactly \( -1 \), so \( w_2 \) bounces between \( +1 \) and \( -1 \) forever, and the loss sticks at \( \frac{1}{2} \cdot 25 \cdot 1^2 = 12.5 \). A hair past it, at 0.081, the loss explodes to 34 million.
3. **Moderate momentum is more than twice as fast** as the best plain GD: with \( \beta = 0.7 \) the loss first dips below 0.001 at step 24, bounces back above it a few times (the ball overshoots), and stays below from step 32, against 75 for plain GD, which never bounces.
4. **Too much momentum overshoots.** With \( \beta = 0.9 \) and this learning rate, the ball has so little friction that it rolls past the minimum and back again for a long time: first below at 87, below for good only from 106. The best \( \beta \) depends on the learning rate and on how stretched the ravine is (the exercises explore a narrower one). In real networks people usually keep \( \beta = 0.9 \) and tune the learning rate around it.

<figure class="diagram" markdown>
![Loss on a log scale against step from 0 to 100 on the ravine. The blue plain-GD line is a straight diagonal falling steadily from about 60 to about 0.00003. The teal momentum line with beta 0.7 falls much faster, with small bumps, passing 0.001 near step 24 and 0.000001 before step 50. The orange momentum line with beta 0.9 falls quickly at first, then rises and falls in waves, staying above the blue line for most of the run](../../assets/img/phase-1/day52-momentum-loss.svg)
<figcaption>Momentum's loss is not monotone (the ball can roll uphill for a moment), but with a sensible β it gets far lower, far sooner. Too much momentum (β = 0.9 here) keeps sloshing.</figcaption>
</figure>

```python
fig, ax = plt.subplots(figsize=(7.5, 4.2))
curves = [(run_gd(start, 0.07, 100), "#3949ab", "plain GD, lr 0.07"),
          (run_momentum(start, 0.07, 0.7, 100), "#00897b", "momentum, lr 0.07, β = 0.7"),
          (run_momentum(start, 0.07, 0.9, 100), "#ef6c00", "momentum, lr 0.07, β = 0.9")]
for p, color, label in curves:
    ax.plot([f_ravine(w) for w in p], color=color, lw=1.6, label=label)
ax.set_yscale("log")
ax.set_ylim(1e-6, 200)
ax.axhline(1e-3, color="#9e9e9e", lw=0.8, ls=":")
ax.set_xlabel("step")
ax.set_ylabel("loss (log scale)")
ax.set_title("The ravine f = ½(w₁² + 25 w₂²)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("day52-momentum-loss.svg", format="svg", bbox_inches="tight")
```

!!! warning "Common mistake: expecting the loss to fall every step"
    With momentum (and with SGD noise) the loss can go **up** for a few steps: the ball rolls up the far side of the valley before coming back. Here is the \( \beta = 0.7 \) ravine run around its first dip:

    ```python
    p07 = run_momentum(start, 0.07, 0.7, 30)
    for step in range(22, 28):
        print(step, f"{f_ravine(p07[step]):.6f}")
    # 22 0.001870
    # 23 0.005826
    # 24 0.000674
    # 25 0.004584
    # 26 0.002494
    # 27 0.004190
    ```

    The loss drops below 0.001 at step 24 and is back above it at step 25. If you stopped training the first time the loss looked good, or panicked the first time it went up, you'd misread both. A training curve that wiggles is normal. A curve that trends upward over many steps, or jumps to `nan` or `inf`, means the learning rate is too high (Day 51). Judge training by the trend of the curve, ideally smoothed over many steps, not by any single step.

!!! warning "Common mistake: adding momentum without lowering the learning rate"
    Momentum multiplies the step in any steady direction by about \( 1/(1 - \beta) \): 10 times for \( \beta = 0.9 \). So a learning rate tuned for plain SGD is suddenly 10 times too big in effect. On a clean quadratic with the full gradient, momentum actually tolerates a larger \( \eta \) (it stays stable while \( \eta\lambda < 2(1 + \beta) \), against \( \eta\lambda < 2 \) without it). The damage comes from noise: momentum also adds up about 10 batches' worth of gradient noise into every step. Here is mini-batch SGD with momentum on the toy data:

    ```python
    def minibatch_momentum(lr, beta, batch_size=32, epochs=3, seed=7):
        r = np.random.default_rng(seed)
        w, v, losses = w0.copy(), np.zeros(2), []
        for epoch in range(epochs):
            order = r.permutation(n)
            for start_row in range(0, n, batch_size):
                idx = order[start_row:start_row + batch_size]
                v = beta * v + grad(w, X[idx], y[idx])
                w = w - lr * v
                losses.append(loss(w, X, y))
        last = np.array(losses[-32:]) - L_best        # the final epoch's 32 steps
        return f"mean gap {last.mean():.4f}   worst {last.max():.4f}"

    print("lr=0.5,  no momentum: ", minibatch_momentum(0.5, 0.0))
    print("lr=0.5,  beta=0.9:    ", minibatch_momentum(0.5, 0.9))
    print("lr=0.05, beta=0.9:    ", minibatch_momentum(0.05, 0.9))
    # lr=0.5,  no momentum:  mean gap 0.0313   worst 0.1957
    # lr=0.5,  beta=0.9:     mean gap 0.5634   worst 3.3062
    # lr=0.05, beta=0.9:     mean gap 0.0078   worst 0.0220
    ```

    Adding \( \beta = 0.9 \) at the same \( \eta = 0.5 \) makes the final epoch about 18 times worse. Lowering \( \eta \) by the factor \( 1 - \beta = 0.1 \) keeps the same effective step and gives the best result of the three. The rule of thumb: when you add momentum, divide the learning rate by about \( 1/(1 - \beta) \), then retune. (This is also why PyTorch's typical SGD settings pair `momentum=0.9` with learning rates like 0.01 to 0.1, not the larger values plain GD tolerates.)

!!! android "Android bridge: you've already written a momentum buffer"
    The Android reference page for `SensorEvent` shows how to separate gravity from an accelerometer reading with a low-pass filter:

    ```kotlin
    val alpha = 0.8f
    gravity[0] = alpha * gravity[0] + (1 - alpha) * event.values[0]
    linearAcceleration[0] = event.values[0] - gravity[0]
    ```

    That line is an **exponential moving average** (EMA): keep 80% of the old estimate, mix in 20% of the new reading. A momentum velocity is the same recipe applied to gradients: \( \mathbf{v} \leftarrow \beta\mathbf{v} + \nabla f \) keeps \( \beta \) of the old velocity and adds the new gradient. (PyTorch's version doesn't scale the new gradient by \( 1 - \beta \), so its velocity is \( 1/(1 - \beta) \) times an EMA. The direction is identical; only the size differs, and the learning rate absorbs that.) The trade-off is the same too. Your filter's `alpha` smooths out shaky readings but reacts late when the phone really turns; a large \( \beta \) smooths out zigzagging gradients but keeps rolling after the valley has turned. Adam, below, keeps **two** of these EMAs per parameter.

## Why momentum helps: a recap of Day 51, then √κ

Day 51 showed where ravines come from. For the MSE of a linear model the **Hessian**, the matrix of second derivatives, is \( H = \frac{2}{n}X^\top X \). Its eigenvectors are the axes of the bowl's elliptical contours and its eigenvalues \( \lambda \) are the curvatures along them. Plain GD is stable only while \( \eta < 2/\lambda_{\max} \), and Day 51 called the ratio \( \kappa = \lambda_{\max}/\lambda_{\min} \) the **condition number**. Our ravine has \( H = \begin{bmatrix} 1 & 0 \\ 0 & 25 \end{bmatrix} \), so \( \kappa = 25 \):

```python
H = np.diag(lam)                 # the ravine's Hessian
np.linalg.eigvalsh(H)            # array([ 1., 25.])   eigenvalues of a symmetric matrix, ascending (Day 51)
H_toy = 2 / n * X.T @ X          # the Hessian of this lesson's toy regression
eig_toy = np.linalg.eigvalsh(H_toy)
eig_toy, eig_toy[-1] / eig_toy[0]      # [-1] is the largest, [0] the smallest
# (array([0.78939813, 3.25901281]), np.float64(4.128477982948648))
```

The toy regression has \( \kappa = 4.1 \): its correlated features tilt and stretch the bowl a little.

**Why plain GD needs about \( \kappa \) steps.** With \( \eta = 1/\lambda_{\max} \), the flattest direction's distance is multiplied by \( 1 - \lambda_{\min}/\lambda_{\max} = 1 - 1/\kappa \) every step, so after \( k \) steps by \( (1 - 1/\kappa)^k \). For large \( \kappa \) that is very close to \( e^{-k/\kappa} \) (the same limit that defines \( e \): \( (1 - 1/m)^m \to 1/e \)). So every \( \kappa \) steps shrink the remaining distance by a factor of \( e \approx 2.72 \), and reaching a given precision takes a number of steps proportional to \( \kappa \). Day 51's balanced learning rate \( 2/(\lambda_{\min} + \lambda_{\max}) \) only halves that.

**Why momentum needs about \( \sqrt{\kappa} \).** With the best choice of both knobs (Polyak's "heavy ball" settings for a quadratic),

\[
\eta = \frac{4}{\left(\sqrt{\lambda_{\max}} + \sqrt{\lambda_{\min}}\right)^2}
\qquad
\beta = \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^2
\]

the distance shrinks by \( \sqrt{\beta} = (\sqrt{\kappa} - 1)/(\sqrt{\kappa} + 1) \) per step, which has the same shape as plain GD's best factor \( (\kappa - 1)/(\kappa + 1) \) with \( \kappa \) replaced by \( \sqrt{\kappa} \). **Worked example** for our ravine, \( \lambda = 1 \) and 25: \( \sqrt{\kappa} = 5 \), so \( \eta = 4/(5 + 1)^2 = 0.111 \) and \( \beta = (4/6)^2 = 0.444 \); the factor per step is \( 4/6 = 0.67 \), against \( 24/26 = 0.92 \) for the best plain GD. Count the steps to a loss below \( 10^{-6} \) for three ravines, each method at its best settings:

```python
def steps_below(curv, lr, beta, tol=1e-6, max_steps=100_000):
    # momentum (beta=0 is plain GD) on f = ½(w1² + curv·w2²) from (-10, 1)
    l = np.array([1.0, curv])
    w, v = np.array([-10.0, 1.0]), np.zeros(2)
    for step in range(1, max_steps + 1):
        v = beta * v + l * w
        w = w - lr * v
        if 0.5 * (l * w ** 2).sum() < tol:
            return step

for kappa in (25, 100, 400):
    r = np.sqrt(kappa)
    gd_steps = steps_below(kappa, 2 / (kappa + 1), 0.0)                        # GD's balanced lr (Day 51)
    hb_steps = steps_below(kappa, 4 / (r + 1) ** 2, ((r - 1) / (r + 1)) ** 2)  # tuned heavy ball
    print(f"kappa={kappa:4d}  plain GD: {gd_steps:5d} steps   tuned momentum: {hb_steps:4d} steps")
# kappa=  25  plain GD:   113 steps   tuned momentum:   31 steps
# kappa= 100  plain GD:   461 steps   tuned momentum:   69 steps
# kappa= 400  plain GD:  1934 steps   tuned momentum:  153 steps
```

Each time \( \kappa \) grows 4 times, plain GD needs about 4 times as many steps and tuned momentum only about twice as many. On a ravine with \( \kappa = 10{,}000 \) that is the difference between tens of thousands of steps and a few hundred. In a real network nobody knows \( \lambda_{\min} \) and \( \lambda_{\max} \), so \( \beta = 0.9 \) is a robust compromise rather than the optimum. Now a real dataset.

### The Play Store ravine, and what momentum does to it

Day 51 showed that raw Play Store features make a ravine, using four features to predict `log_reviews` (condition number 94,496). Here we use Day 33's smaller model instead: predict `rating` from `log_installs` and `size_mb` plus an intercept column. Different columns give a different \( X^\top X \), so the condition number differs too. Load it with the same short cleaning (tested on pandas 3.0):

```python
import pandas as pd

BASE = "https://manjulkadyan.github.io/ai-leaning/assets/data/"
apps = pd.read_csv(BASE + "play_store_apps.csv")
apps = apps.drop_duplicates().assign(
    installs=lambda d: d["installs"].str.replace(",", "").str.rstrip("+").astype(int),
    size_mb=lambda d: d["size_mb"].replace("Varies with device", np.nan).astype(float),
)
data = pd.DataFrame({
    "log_installs": np.log10(apps["installs"]),
    "size_mb": apps["size_mb"],
    "rating": apps["rating"],
}).dropna()
F = data[["log_installs", "size_mb"]].to_numpy()
yp = data["rating"].to_numpy()
F_std = (F - F.mean(axis=0)) / F.std(axis=0, ddof=1)
X_raw = np.column_stack([F, np.ones(len(F))])        # raw features + intercept column
X_std = np.column_stack([F_std, np.ones(len(F))])    # standardized features + intercept column
X_raw.shape                                          # (442, 3)

for name, Xp in (("raw", X_raw), ("standardized", X_std)):
    ev = np.linalg.eigvalsh(2 / len(yp) * Xp.T @ Xp)
    print(f"{name:13s} curvatures {ev.round(3)}   condition number {ev[-1] / ev[0]:,.1f}")
# raw           curvatures [7.900000e-02 1.822500e+01 6.678243e+03]   condition number 84,687.5
# standardized  curvatures [1.741 2.    2.25 ]   condition number 1.3
```

(No internet? Download the CSV from the [Datasets](../../datasets.md) page and pass its local path to `read_csv`.)

\( \kappa = 84{,}687 \) raw, 1.3 standardized: the same story as Day 51 with this model's numbers. Plain GD at \( \eta = 1/\lambda_{\max} \), stopping within \( 10^{-6} \) of the best loss:

```python
def gd_until(Xp, yp, tol=1e-6, max_steps=20_000):
    """Batch GD with lr = 1 / (largest curvature). Stops when the loss is within tol of the best."""
    lr = 1 / np.linalg.eigvalsh(2 / len(yp) * Xp.T @ Xp)[-1]
    L_star = loss(np.linalg.lstsq(Xp, yp)[0], Xp, yp)
    w = np.zeros(Xp.shape[1])
    for step in range(1, max_steps + 1):
        w = w - lr * grad(w, Xp, yp)
        if loss(w, Xp, yp) - L_star < tol:
            break
    return step, loss(w, Xp, yp) - L_star

gd_until(X_std, yp)            # (4, np.float64(4.1409820031312883e-07))
gd_until(X_raw, yp)            # (20000, np.float64(0.39334580466703584))   gave up, still far away
```

Four steps standardized; raw, 20,000 steps leave a gap of 0.39, about twice the best loss itself. Now the new part: give the raw problem tuned heavy-ball momentum, with \( \sqrt{\kappa} = 291 \):

```python
def momentum_until(Xp, yp, tol=1e-6, max_steps=20_000):
    # heavy-ball momentum with the tuned lr and beta from the curvatures
    ev = np.linalg.eigvalsh(2 / len(yp) * Xp.T @ Xp)
    r = np.sqrt(ev[-1] / ev[0])                       # sqrt(kappa)
    lr = 4 / (np.sqrt(ev[-1]) + np.sqrt(ev[0])) ** 2
    beta = ((r - 1) / (r + 1)) ** 2
    L_star = loss(np.linalg.lstsq(Xp, yp)[0], Xp, yp)
    w, v = np.zeros(Xp.shape[1]), np.zeros(Xp.shape[1])
    for step in range(1, max_steps + 1):
        v = beta * v + grad(w, Xp, yp)
        w = w - lr * v
        if loss(w, Xp, yp) - L_star < tol:
            break
    return step, round(float(beta), 4)

momentum_until(X_raw, yp)        # (2408, 0.9863)
```

Momentum reaches the same \( 10^{-6} \) in 2,408 steps, where plain GD had not arrived after 20,000, using \( \beta = 0.9863 \), far more momentum than the usual 0.9 because the ravine is so stretched. That is the \( \sqrt{\kappa} \) effect on real data. It is still 600 times slower than standardizing, which removes the ravine instead of rolling through it. "Standardize your inputs" (Day 22) remains the first rule of training.

!!! warning "Common mistake: tuning the learning rate to fix a scaling problem"
    When training on raw features is painfully slow, the tempting fix is a bigger learning rate. The raw problem's largest curvature is 6,678, so its limit is \( 2/6678 = 0.0003 \). At 0.001 the loss grows about 32 times per step, overflows to `inf` and turns into `nan`, exactly as in Day 51's divergence demo:

    ```python
    w_div = np.zeros(3)
    for step in range(1, 601):
        w_div = w_div - 0.001 * grad(w_div, X_raw, yp)
        if step in (1, 2, 3, 600):
            print(step, loss(w_div, X_raw, yp))
    # 1 337.67917747115746
    # 2 10665.811130668173
    # 3 343677.12102102255
    # 600 nan
    ```

    Between those lines NumPy prints warnings such as `RuntimeWarning: overflow encountered in matmul` and `RuntimeWarning: invalid value encountered in subtract`. And lowering the learning rate below 0.0003 brings back the 20,000-step crawl. No single learning rate suits directions whose curvatures differ by a factor of 84,687, so fix the scale, not the learning rate: check `X.std(axis=0)`, standardize with the **training** set's mean and standard deviation, and reuse those same numbers for validation, test and in the app.

## Adam, briefly and honestly

Standardizing inputs fixes the scale of the *inputs*. Inside a deep network the gradients of different weights can still differ by orders of magnitude, and nobody can standardize those by hand. **Adam** (Kingma and Ba, 2014; "adaptive moment estimation") gives **every parameter its own step size**, based on how big that parameter's gradients have recently been. It keeps two EMAs per parameter, like the Android bridge's filter:

\[
\begin{aligned}
\mathbf{m} &\leftarrow \beta_1\,\mathbf{m} + (1 - \beta_1)\,\mathbf{g} && \text{average gradient (momentum)}\\
\mathbf{s} &\leftarrow \beta_2\,\mathbf{s} + (1 - \beta_2)\,\mathbf{g}^2 && \text{average squared gradient (its typical size)}\\
\hat{\mathbf{m}} &= \frac{\mathbf{m}}{1 - \beta_1^{\,t}}, \qquad \hat{\mathbf{s}} = \frac{\mathbf{s}}{1 - \beta_2^{\,t}} && \text{bias correction}\\
\mathbf{w} &\leftarrow \mathbf{w} - \eta\,\frac{\hat{\mathbf{m}}}{\sqrt{\hat{\mathbf{s}}} + \epsilon}
\end{aligned}
\]

One symbol at a time:

- \( \mathbf{g} \) is the current (mini-batch) gradient, and \( \mathbf{g}^2 \) squares each entry separately; the division and square root in the last line are also entry by entry, so each parameter gets its own step.
- \( t \) is the step number, starting at 1.
- \( \mathbf{m} \) and \( \mathbf{s} \) start at zero, so early on they are too small; dividing by \( 1 - \beta^t \) corrects for that. At \( t = 1 \): \( 1 - \beta_1 = 0.1 \), and \( \mathbf{m} = 0.1\mathbf{g} \) divided by 0.1 gives back exactly \( \mathbf{g} \). As \( t \) grows, \( \beta^t \to 0 \) and the correction fades away.
- \( \epsilon \) ("epsilon") is a tiny number that prevents division by zero.
- **Defaults** (PyTorch's `torch.optim.Adam`): \( \eta = 0.001 \), \( \beta_1 = 0.9 \), \( \beta_2 = 0.999 \), \( \epsilon = 10^{-8} \).

**The key property.** Dividing \( \hat{\mathbf{m}} \) by \( \sqrt{\hat{\mathbf{s}}} \) cancels the gradient's size: a parameter whose gradients are always about 300 and one whose gradients are always about 0.01 both move by about \( \eta \) per step. **Worked example, step 1:** \( \hat{\mathbf{m}} = \mathbf{g} \) and \( \hat{\mathbf{s}} = \mathbf{g}^2 \), so the step is \( \eta\,\mathbf{g}/|\mathbf{g}| \): every parameter moves by exactly \( \eta \), in the downhill direction, whatever its gradient's size.

```python
def adam(grad_fn, w0, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, steps=1000):
    w = np.array(w0, dtype=float)
    m = np.zeros_like(w)                       # running average of gradients (momentum)
    s = np.zeros_like(w)                       # running average of squared gradients
    for t in range(1, steps + 1):
        g = grad_fn(w)
        m = beta1 * m + (1 - beta1) * g
        s = beta2 * s + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** t)           # bias correction: m and s start at 0
        s_hat = s / (1 - beta2 ** t)
        w = w - lr * m_hat / (np.sqrt(s_hat) + eps)
    return w

g_raw = lambda w: grad(w, X_raw, yp)
grad(np.zeros(3), X_raw, yp).round(2)       # array([ -37.1 , -368.43,   -8.34])   wildly different sizes
adam(g_raw, np.zeros(3), lr=0.1, steps=1)   # array([0.1, 0.1, 0.1])               yet every weight moved by 0.1
```

On the badly conditioned raw Play Store problem, where plain GD gave up after 20,000 steps:

```python
L_star_raw = loss(np.linalg.lstsq(X_raw, yp)[0], X_raw, yp)
for steps in (100, 300, 1000):
    w_adam = adam(g_raw, np.zeros(3), lr=0.1, steps=steps)
    print(f"Adam, raw features, {steps:5d} steps: gap {loss(w_adam, X_raw, yp) - L_star_raw:.1e}   w = {w_adam.round(4)}")
# Adam, raw features,   100 steps: gap 2.4e-01   w = [0.5214 0.0021 1.66  ]
# Adam, raw features,   300 steps: gap 1.2e-02   w = [1.2100e-01 1.7000e-03 3.5381e+00]
# Adam, raw features,  1000 steps: gap 5.9e-13   w = [3.6000e-03 1.4000e-03 4.0883e+00]
```

Adam solves in 1,000 steps what plain GD couldn't in 20,000. That is why it is the default first choice for deep networks, and why most transformer models are trained with its variant **AdamW** (which handles weight decay, the L2 penalty from Day 22, more cleanly). Now the honest part:

- **It is not magic, and the raw ravine is tilted.** Standardized features plus plain GD needed **4** steps. Adam effectively rescales each parameter's own axis, which can only undo the part of the stretch that lies *along* the axes. You can measure that part: rescale \( H \) so every diagonal entry becomes 1 (that is what a perfect per-parameter step size could do) and look at what's left:

    ```python
    H_raw = 2 / len(yp) * X_raw.T @ X_raw
    vals, vecs = np.linalg.eigh(H_raw)                # eigh: eigenvalues AND eigenvectors (Day 24)
    vecs[:, 0].round(3)          # array([-0.206, -0.001,  0.979])   flattest direction: (log_installs, size_mb, intercept)
    D = np.diag(1 / np.sqrt(np.diag(H_raw)))          # per-parameter rescaling
    ev_D = np.linalg.eigvalsh(D @ H_raw @ D)
    ev_D[-1] / ev_D[0]           # np.float64(126.65247902315062)   kappa after the best per-axis rescaling
    ```

    The flattest direction is a mix of the `log_installs` weight and the intercept, not one parameter: `log_installs` averages 4.45 and barely varies, so its column looks almost like a multiple of the column of ones, and the bowl is stretched along a diagonal between those two weights. The best per-parameter rescaling cuts \( \kappa \) from 84,687 to about 127, a big help but not 1. Only centring the columns (subtracting the means, part of standardizing) removes that diagonal stretch. Good input scaling still matters with Adam.
- **The step size doesn't shrink on its own near the minimum.** Each step is still about \( \eta \) per parameter, so with noisy mini-batch gradients Adam hovers too. It is used with a learning-rate schedule, the same as SGD.
- **It costs memory.** \( \mathbf{m} \) and \( \mathbf{s} \) are two extra numbers per parameter, so the optimizer state is twice the size of the model. That matters for training on GPUs, and even more for the rare case of training on a phone.
- **It doesn't always win.** For some image models, well-tuned SGD with momentum has been reported to generalize slightly better (Wilson et al., 2017). Practitioners usually start with Adam or AdamW at the defaults and only switch if they have a reason.

!!! warning "Common mistake: leaving out the bias correction"
    The two correction lines look optional, and some hand-written Adams skip them. Then at step 1, \( \mathbf{m} = 0.1\,\mathbf{g} \) and \( \sqrt{\mathbf{s}} = \sqrt{0.001}\,|\mathbf{g}| = 0.0316\,|\mathbf{g}| \), so the step is \( 0.1/0.0316 = 3.16 \) times bigger than intended:

    ```python
    def adam_no_correction(grad_fn, w0, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, steps=1):
        w = np.array(w0, dtype=float)
        m = np.zeros_like(w)
        s = np.zeros_like(w)
        for t in range(1, steps + 1):
            g = grad_fn(w)
            m = beta1 * m + (1 - beta1) * g
            s = beta2 * s + (1 - beta2) * g ** 2
            w = w - lr * m / (np.sqrt(s) + eps)      # BUG: no bias correction
        return w

    adam_no_correction(g_raw, np.zeros(3), lr=0.1, steps=1)   # array([0.31622776, 0.31622777, 0.31622775])
    ```

    Every weight moved by 0.316 instead of 0.1, because \( \mathbf{s} \) starts at zero and is still far too small after one step. The early steps are the ones taken with random weights and the biggest gradients, so this is when an oversized step does the most damage. Use the corrected version, or the library's.

## Learning-rate schedules

Remember the noise band: a big learning rate covers ground fast but jitters at the end; a small one settles but crawls. A **schedule** changes \( \eta \) during training, big early and small late. If \( \eta_0 \) is the starting learning rate, \( t \) the step number and \( T \) the total number of steps, two common choices are:

- **Step decay**: divide by 10 at fixed points, here at one third and two thirds of training: \( \eta_t = \eta_0 \cdot 0.1^{\lfloor 3t/T \rfloor} \) (\( \lfloor\cdot\rfloor \) rounds down). PyTorch: `torch.optim.lr_scheduler.StepLR`.
- **Cosine decay**: glide smoothly from \( \eta_0 \) to (near) zero along half a cosine wave:

\[
\eta_t = \eta_{\min} + \tfrac{1}{2}\left(\eta_0 - \eta_{\min}\right)\left(1 + \cos\frac{\pi t}{T}\right)
\]

**Worked example** with \( \eta_0 = 0.1 \), \( \eta_{\min} = 0 \), \( T = 1500 \): at \( t = 0 \), \( \cos 0 = 1 \), so \( \eta = 0.05 \times 2 = 0.1 \). Halfway, \( t = 750 \): \( \cos(\pi/2) = 0 \), so \( \eta = 0.05 \). At the end, \( \cos\pi = -1 \), so \( \eta = 0 \). PyTorch: `torch.optim.lr_scheduler.CosineAnnealingLR`.

```python
def constant_lr(step, total, lr0):
    return lr0

def step_decay_lr(step, total, lr0, drops=3):
    """Divide the learning rate by 10 at 1/3 and 2/3 of training."""
    return lr0 * 0.1 ** (step // (total // drops))

def cosine_lr(step, total, lr0, lr_min=0.0):
    return lr_min + 0.5 * (lr0 - lr_min) * (1 + np.cos(np.pi * step / total))

T = 1500
[round(float(cosine_lr(t, T, 0.1)), 4) for t in (0, 375, 750, 1125, 1500)]     # [0.1, 0.0854, 0.05, 0.0146, 0.0]
[round(step_decay_lr(t, T, 0.1), 4) for t in (0, 499, 500, 1000, 1499)]        # [0.1, 0.1, 0.01, 0.001, 0.001]
```

Our `minibatch_gd` already accepts a `schedule` function. Train with mini-batches of 8 for 12 epochs (1,500 steps), starting every schedule at 0.1:

```python
sched_runs = {}
for name, sched in (("constant", constant_lr), ("step decay", step_decay_lr), ("cosine", cosine_lr)):
    _, _, losses = minibatch_gd(X, y, w0, lr=0.1, batch_size=8, epochs=12,
                                rng=np.random.default_rng(4), schedule=sched)
    sched_runs[name] = losses
    print(f"{name:10s}  gap at the end: {losses[-1] - L_best:.2e}   mean gap, last epoch: {(losses[-125:] - L_best).mean():.2e}")
# constant    gap at the end: 8.12e-03   mean gap, last epoch: 6.27e-03
# step decay  gap at the end: 9.34e-07   mean gap, last epoch: 6.37e-06
# cosine      gap at the end: 3.13e-05   mean gap, last epoch: 3.31e-05
```

Measured by the mean gap over the last epoch (a fairer number than the single final step, which depends on where the noise happened to land), step decay ends about 1,000 times closer to the minimum than the constant learning rate and cosine about 190 times closer, using the same steps on the same batches. All three runs are identical for the first stretch, when big steps are what you want; the schedules only differ once there is noise to calm down. (Which of the two schedules wins here is down to this one seed and setup; on real networks cosine is a popular default because it has only one knob, the total length \( T \).)

<figure class="diagram" markdown>
![Left: three learning-rate schedules over 1,500 steps, all starting at 0.1: a flat blue constant line, an orange staircase that drops to 0.01 at step 500 and 0.001 at step 1000, and a teal cosine curve gliding smoothly from 0.1 to 0. Right: loss minus best loss on a log scale for mini-batch SGD under each schedule; all three fall together for the first 50 steps; the blue constant run then jitters in a band around 0.001 to 0.01 for the rest of training, the orange step-decay run drops to a lower band after step 500 and lower again after step 1000, reaching about 0.000001, and the teal cosine run slides down gradually to about 0.00003](../../assets/img/phase-1/day52-lr-schedules.svg)
<figcaption>Decaying the learning rate shrinks the noise band: the staircase drops it in two jumps, the cosine in a smooth glide; the constant learning rate keeps jittering to the end.</figcaption>
</figure>

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
T = 12 * 125                                          # 12 epochs × 125 steps
t = np.arange(T)
colors = {"constant": "#3949ab", "step decay": "#ef6c00", "cosine": "#00897b"}
for name, sched in (("constant", constant_lr), ("step decay", step_decay_lr), ("cosine", cosine_lr)):
    ax1.plot(t, [sched(s, T, 0.1) for s in t], color=colors[name], lw=2, label=name)
    ax2.plot(sched_runs[name] - L_best, color=colors[name], lw=0.9, label=name)
ax1.set_xlabel("step")
ax1.set_ylabel("learning rate")
ax1.set_title("Three schedules, all starting at 0.1")
ax1.legend(frameon=False)
ax2.set_yscale("log")
ax2.set_xlabel("step")
ax2.set_ylabel("loss − best loss (log scale)")
ax2.set_title("Mini-batch SGD (B = 8) under each schedule")
ax2.legend(frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig("day52-lr-schedules.svg", format="svg", bbox_inches="tight")
```

!!! warning "Common mistake: a schedule measured in epochs, stepped in steps"
    Schedules need the step counter and the total to use the same unit. If you pass the number of **epochs** (12) as `total` but call the schedule with the **step** number (0 to 1,499), the cosine keeps going round:

    ```python
    epochs = 12
    [round(float(cosine_lr(t, epochs, 0.1)), 4) for t in (0, 6, 12, 18, 24, 100)]   # [0.1, 0.05, 0.0, 0.05, 0.1, 0.075]
    ```

    After step 12 the "decay" climbs back up to 0.1 at step 24, then down again, oscillating for the whole run. No error appears; training is noisier than it should be and never settles. Recognise it by printing the learning rate at a few steps across the whole run (or plotting it, as in the left panel above). PyTorch's schedulers have the same trap: `CosineAnnealingLR(optimizer, T_max=...)` counts calls to `scheduler.step()`, so `T_max` must be in epochs if you call it once per epoch, and in steps if you call it once per batch.

One more piece you'll meet in every transformer training recipe: **warmup**. For the first few hundred or thousand steps the learning rate ramps *up* linearly from near zero, and only then decays. At the very start the weights are random, the gradients can be huge and Adam's \( \hat{\mathbf{s}} \) estimates are based on only a few steps, so starting gently avoids an early blow-up. Exercise 8 builds warmup plus cosine.

## Where this shows up in ML

| Today's idea | Where you'll meet it |
|---|---|
| Mini-batches, epochs, shuffling | Every training loop: PyTorch's `DataLoader(batch_size=..., shuffle=True)`, Keras's `model.fit(batch_size=..., epochs=...)` |
| Batch size vs noise | Choosing \( B \) to fit memory; why bigger batches usually need a bigger learning rate |
| Momentum | `torch.optim.SGD(..., momentum=0.9)`, the standard optimizer for many image models |
| Condition number | Why inputs are standardized, why networks use normalization layers, why some losses are slow to train |
| Adam / AdamW | The default optimizer for most deep learning, including large language models |
| Schedules and warmup | Cosine decay with warmup in nearly every modern training recipe |

## Going further

These are real and you will meet them, but they build on today's ideas rather than replacing them:

- **Nesterov momentum**: evaluates the gradient at the point the velocity is about to carry you to, which reduces overshoot. PyTorch: `SGD(..., momentum=0.9, nesterov=True)`. [CS231n notes, "Nesterov Momentum"](https://cs231n.github.io/neural-networks-3/).
- **AdaGrad and RMSProp**: Adam's ancestors. RMSProp is Adam's \( \mathbf{s} \) average on its own, without \( \mathbf{m} \). [Dive into Deep Learning: Optimization Algorithms](https://d2l.ai/chapter_optimization/index.html).
- **AdamW and weight decay**: how to combine Adam with L2 regularization correctly. [Loshchilov and Hutter, "Decoupled Weight Decay Regularization"](https://arxiv.org/abs/1711.05101).
- **Large-batch training and the linear scaling rule**: raise the learning rate in proportion to the batch size, with warmup. [Goyal et al., "Accurate, Large Minibatch SGD"](https://arxiv.org/abs/1706.02677).
- **Batch size and generalization**: very large batches can find "sharp" minima that generalize worse. [Keskar et al., "On Large-Batch Training for Deep Learning"](https://arxiv.org/abs/1609.04836).
- **The learning-rate range test and cyclical learning rates**: sweep the learning rate upward during one short run to find a good range. [Smith, "Cyclical Learning Rates for Training Neural Networks"](https://arxiv.org/abs/1506.01186).
- **Cosine schedules with restarts**: [Loshchilov and Hutter, "SGDR"](https://arxiv.org/abs/1608.03983).
- **Other schedulers and gradient clipping**: `ReduceLROnPlateau`, `OneCycleLR`, and `torch.nn.utils.clip_grad_norm_` for capping exploding gradients. [PyTorch docs: torch.optim](https://pytorch.org/docs/stable/optim.html).
- **Local minima, saddle points and plateaus**, and how SGD's noise helps with them: Day 53, next.

---

## Exercises

The exercises go from warm-ups to a challenge. Every solution below re-creates its own data, so you can run it on its own.

**Warm-ups**

1. **Steps per epoch.** A training set has 50,000 examples and you use a batch size of 128. How many steps is one epoch, how many examples are in the last batch, and how many steps are 20 epochs? Work it out by hand, then check with Python.
2. **How long momentum remembers.** With a constant gradient \( g = 1 \) and \( \beta = 0.99 \), the velocity tends to \( g/(1 - \beta) = 100 \). After \( k \) steps it equals \( (1 - \beta^k)/(1 - \beta) \). How many steps until it reaches 90% of its final size? Solve it with logarithms, then check with a loop.

**Core**

3. **Batch size at a fixed learning rate.** On the lesson's toy data, train for 2 epochs at `lr=0.05` with \( B \) = 1, 8, 32, 128 and 1000 (shuffle with `np.random.default_rng(5)`). Print the number of steps and the final gap for each. At a *fixed* learning rate and a fixed number of epochs, which batch size wins, and why? Why would the answer change if each batch size could use its own best learning rate?
4. **A narrower ravine.** Change the ravine to \( f = \tfrac{1}{2}(w_1^2 + 100\,w_2^2) \), starting from \( [-10, 1] \). What is the largest stable learning rate for plain GD? Show that 0.019 converges and 0.021 diverges, and count GD's steps to a loss below 0.001 at 0.019. Then, at `lr=0.019`, try momentum with \( \beta \) = 0.5, 0.7, 0.8 and 0.9. Which is best, and how does that compare with the lesson's ravine?
5. **Adam ignores units.** Multiply the toy problem's gradient by 1,000 (as if the loss were measured in different units). Run plain GD with `lr=0.05` on the original gradient for 50 steps and on the scaled one for 2 steps. Then run Adam with `lr=0.05` for 50 steps on both. What happens, and why?

**Predict the output**

6. Write down what each loop prints **before** running it. Then do the same for the Adam step: what is `step` for a gradient with one huge and one tiny entry? (Hint: the tiny entry is not exactly what you might first expect.)

    ```python
    beta = 0.5
    v = 0.0
    for g in [4.0, -4.0, 4.0, -4.0]:            # a gradient that flips sign every step
        v = beta * v + g
        print("flipping:", v)
    v = 0.0
    for g in [4.0, 4.0, 4.0, 4.0]:              # a gradient that keeps pointing the same way
        v = beta * v + g
        print("steady:  ", v)
    ```

    ```python
    import numpy as np

    g = np.array([-500.0, 0.002])               # one huge and one tiny gradient entry
    lr, eps = 0.01, 1e-8
    m_hat, s_hat = g, g ** 2                    # Adam's bias-corrected averages after step 1
    step = lr * m_hat / (np.sqrt(s_hat) + eps)
    step
    ```

**Find and fix the bug**

7. This epoch loop crashes on its second batch. Read the error, find the bug, explain why the **first** batch worked, and fix it.

    ```python
    import numpy as np

    rng = np.random.default_rng(52)
    X = rng.normal(0, 1, size=(1_000, 2))
    y = X @ np.array([2.0, -1.0]) + rng.normal(0, 0.5, size=1_000)

    def grad(w, X, y):
        return 2 / len(y) * X.T @ (X @ w - y)

    def one_epoch(X, y, w, lr, batch_size, rng):
        order = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            idx = order[start:batch_size]        # the bug is on this line
            w = w - lr * grad(w, X[idx], y[idx])
        return w

    try:
        one_epoch(X, y, np.zeros(2), 0.05, 32, np.random.default_rng(0))
    except ZeroDivisionError as e:               # caught so the page keeps running; the message is the real one
        print(f"{type(e).__name__}: {e}")   # ZeroDivisionError: division by zero
    ```

**Challenge**

8. **Warmup plus cosine.** Write `warmup_cosine_lr(step, total, lr0, warmup_frac=0.1)`: a linear ramp from \( \eta_0/\text{warm} \) up to \( \eta_0 \) over the first 10% of the steps (where \( \text{warm} \) is that number of steps), then the cosine schedule over the remaining steps. Print it at steps 0, 49, 99, 100, 550 and 999 of a 1,000-step run with \( \eta_0 = 0.1 \). Then train the toy problem with mini-batches of 8 for 12 epochs (shuffle seed 4) with plain cosine and with warmup plus cosine, and compare the final gaps.

??? success "Solutions"
    **1. Steps per epoch.** \( 50000/128 = 390.6 \), so 391 steps; 390 full batches hold \( 390 \times 128 = 49{,}920 \) examples, which leaves 80 for the last batch; 20 epochs are \( 20 \times 391 = 7{,}820 \) steps.

    ```python
    n, B = 50_000, 128
    steps = -(-n // B)                  # ceil(50000 / 128)
    last = n - (steps - 1) * B          # what is left for the final batch
    steps, last, 20 * steps             # (391, 80, 7820)
    ```

    `-(-n // B)` is integer ceiling division, and the last batch gets whatever the 390 full batches left over. If you pass `drop_last=True` to PyTorch's `DataLoader`, that 80-example batch is skipped and an epoch is 390 steps.

    **2. How long momentum remembers.** We need \( 1 - 0.99^k \ge 0.9 \), so \( 0.99^k \le 0.1 \). Taking logarithms (and flipping the inequality, because \( \ln 0.99 \) is negative): \( k \ge \ln 0.1 / \ln 0.99 = 229.1 \), so \( k = 230 \).

    ```python
    import numpy as np

    beta, g = 0.99, 1.0
    v, k = 0.0, 0
    while v < 0.9 * g / (1 - beta):     # until v reaches 90% of its limit g / (1 - beta) = 100
        v = beta * v + g
        k += 1
    k, round(v, 3)                                # (230, 90.09)
    round((1 - beta ** k) / (1 - beta), 3)        # 90.09   the closed form agrees
    float(np.log(0.1) / np.log(0.99))             # 229.1052882766942   round up to 230
    ```

    By the same arithmetic, a gradient from 230 steps ago still carries 10% of its weight in the velocity. A rule of thumb: momentum averages over roughly the last \( 1/(1 - \beta) \) steps, so about 10 steps for \( \beta = 0.9 \) and 100 for \( \beta = 0.99 \). High momentum is smooth but slow to turn, like the accelerometer filter with `alpha` close to 1.

    **3. Batch size at a fixed learning rate.**

    ```python
    import numpy as np

    rng = np.random.default_rng(52)                       # re-create the lesson's toy data exactly
    n = 1_000
    x1 = rng.normal(0, 1, size=n)
    x2 = 0.6 * x1 + 0.8 * rng.normal(0, 1, size=n)
    X = np.column_stack([x1, x2])
    y = X @ np.array([2.0, -1.0]) + rng.normal(0, 0.5, size=n)
    w0 = np.array([-1.0, 2.0])

    def loss(w, X, y):
        r = X @ w - y
        return r @ r / len(y)

    def grad(w, X, y):
        return 2 / len(y) * X.T @ (X @ w - y)

    def minibatch_gd(X, y, w0, lr, batch_size, epochs, rng):
        w = np.array(w0, dtype=float)
        steps = 0
        for epoch in range(epochs):
            order = rng.permutation(len(y))
            for start in range(0, len(y), batch_size):
                idx = order[start:start + batch_size]
                w = w - lr * grad(w, X[idx], y[idx])
                steps += 1
        return w, steps

    L_best = loss(np.linalg.lstsq(X, y)[0], X, y)
    for B in (1, 8, 32, 128, 1000):
        w, steps = minibatch_gd(X, y, w0, lr=0.05, batch_size=B, epochs=2, rng=np.random.default_rng(5))
        print(f"B={B:5d}  steps={steps:5d}  gap after 2 epochs = {loss(w, X, y) - L_best:.5f}")
    # B=    1  steps= 2000  gap after 2 epochs = 0.00063
    # B=    8  steps=  250  gap after 2 epochs = 0.00160
    # B=   32  steps=   64  gap after 2 epochs = 0.03798
    # B=  128  steps=   16  gap after 2 epochs = 1.98582
    # B= 1000  steps=    2  gap after 2 epochs = 6.16052
    ```

    At a fixed learning rate and a fixed number of epochs, the smallest batches win, because they take the most steps, and early in training the distance still to cover matters more than the noise. (B = 1 beats B = 8 here partly by luck of when it was stopped: it is already in its noise band.) Larger batches have less noisy gradients, so they can safely take **bigger** steps. With the learning rate raised along with \( B \), the large-batch runs close much of the gap (batch GD at `lr=0.5` reached 0.0003 in the lesson). The fair comparison is always "each batch size at its own best learning rate".

    **4. A narrower ravine.** The steep curvature is 100, so plain GD needs \( \eta < 2/100 = 0.02 \).

    ```python
    import numpy as np

    lam = np.array([1.0, 100.0])                 # a narrower ravine: curvatures 1 and 100
    start = np.array([-10.0, 1.0])

    def f(w):
        return 0.5 * (lam * w ** 2).sum()

    def steps_to(lr, beta, max_steps=2000, tol=1e-3):
        """Momentum with the given beta (beta=0 is plain GD). Returns the first step with f < tol."""
        w, v = start.copy(), np.zeros(2)
        for step in range(1, max_steps + 1):
            v = beta * v + lam * w
            w = w - lr * v
            if f(w) < tol:
                return step
        return None

    for lr in (0.019, 0.021):
        w = start.copy()
        for _ in range(300):
            w = w - lr * lam * w                 # plain GD on the ravine
        print(f"GD lr={lr}: loss after 300 steps = {f(w):.3g}")
    print("GD lr=0.019, steps to loss < 0.001:", steps_to(0.019, 0.0))
    for beta in (0.5, 0.7, 0.8, 0.9):
        print(f"momentum lr=0.019 beta={beta}: steps to loss < 0.001 = {steps_to(0.019, beta)}")
    # GD lr=0.019: loss after 300 steps = 0.000502
    # GD lr=0.021: loss after 300 steps = 3.42e+26
    # GD lr=0.019, steps to loss < 0.001: 283
    # momentum lr=0.019 beta=0.5: steps to loss < 0.001 = 135
    # momentum lr=0.019 beta=0.7: steps to loss < 0.001 = 70
    # momentum lr=0.019 beta=0.8: steps to loss < 0.001 = 51
    # momentum lr=0.019 beta=0.9: steps to loss < 0.001 = 106
    ```

    The narrower ravine (\( \kappa = 100 \) instead of 25) makes plain GD almost 4 times slower (283 steps instead of 75), and the best momentum is now higher, \( \beta = 0.8 \) instead of 0.7: the more stretched the valley, the more momentum pays. Momentum's advantage also grew, from about 3 times to about 5.5 times (counting the first step below 0.001, as this solution's `steps_to` does: 75/24 on the lesson's ravine, 283/51 here), in line with "GD needs on the order of \( \kappa \) steps, momentum on the order of \( \sqrt{\kappa} \)".

    **5. Adam ignores units.**

    ```python
    import numpy as np

    rng = np.random.default_rng(52)                       # re-create the lesson's toy data exactly
    n = 1_000
    x1 = rng.normal(0, 1, size=n)
    x2 = 0.6 * x1 + 0.8 * rng.normal(0, 1, size=n)
    X = np.column_stack([x1, x2])
    y = X @ np.array([2.0, -1.0]) + rng.normal(0, 0.5, size=n)
    w0 = np.array([-1.0, 2.0])

    def grad(w):
        return 2 / n * X.T @ (X @ w - y)

    def grad_1000(w):
        return 1000 * grad(w)                             # the same loss, measured in different units

    def gd(grad_fn, w, lr, steps):
        for _ in range(steps):
            w = w - lr * grad_fn(w)
        return w

    def adam(grad_fn, w, lr, steps, beta1=0.9, beta2=0.999, eps=1e-8):
        m, s = np.zeros_like(w), np.zeros_like(w)
        for t in range(1, steps + 1):
            g = grad_fn(w)
            m = beta1 * m + (1 - beta1) * g
            s = beta2 * s + (1 - beta2) * g ** 2
            w = w - lr * (m / (1 - beta1 ** t)) / (np.sqrt(s / (1 - beta2 ** t)) + eps)
        return w

    gd(grad, w0, 0.05, 50).round(4)           # array([ 1.6104, -0.634 ])
    gd(grad_1000, w0, 0.05, 2).round(1)       # array([-2217.3,  6882.7])   exploded after 2 steps
    adam(grad, w0, 0.05, 50).round(4)         # array([ 1.1525, -0.1517])
    adam(grad_1000, w0, 0.05, 50).round(4)    # array([ 1.1525, -0.1517])   identical
    ```

    Scaling the loss by 1,000 multiplies every curvature by 1,000, so `lr=0.05` is far beyond plain GD's limit of \( 2/\lambda_{\max} \), and it has exploded to thousands after 2 steps. Adam's path is identical for both, because \( \hat{\mathbf{m}} \) and \( \sqrt{\hat{\mathbf{s}}} \) both grow 1,000 times and cancel (only the tiny \( \epsilon \) could tell them apart). That is why Adam's default learning rate works across so many problems, while SGD's learning rate has to be tuned to each one.

    **6. Predict the output.** Flipping: \( v_1 = 4 \); \( v_2 = 0.5 \times 4 - 4 = -2 \); \( v_3 = 0.5 \times (-2) + 4 = 3 \); \( v_4 = 0.5 \times 3 - 4 = -2.5 \). Steady: 4, then \( 2 + 4 = 6 \), \( 3 + 4 = 7 \), \( 3.5 + 4 = 7.5 \), heading for \( 4/(1 - 0.5) = 8 \).

    ```python
    beta = 0.5
    v = 0.0
    for g in [4.0, -4.0, 4.0, -4.0]:            # a gradient that flips sign every step
        v = beta * v + g
        print("flipping:", v)
    v = 0.0
    for g in [4.0, 4.0, 4.0, 4.0]:              # a gradient that keeps pointing the same way
        v = beta * v + g
        print("steady:  ", v)
    # flipping: 4.0
    # flipping: -2.0
    # flipping: 3.0
    # flipping: -2.5
    # steady:   4.0
    # steady:   6.0
    # steady:   7.0
    # steady:   7.5
    ```

    The flipping gradient's velocity stays smaller than a single gradient (it shrinks from 4 towards about 2.7 in size), while the steady one nearly doubles. That is momentum's whole trick in eight numbers. For Adam, the first step divides each entry by its own size, so both entries move by `lr = 0.01` in size, whatever the gradient's size:

    ```python
    import numpy as np

    g = np.array([-500.0, 0.002])               # one huge and one tiny gradient entry
    lr, eps = 0.01, 1e-8
    m_hat, s_hat = g, g ** 2                    # Adam's bias-corrected averages after step 1
    step = lr * m_hat / (np.sqrt(s_hat) + eps)
    step                # array([-0.01      ,  0.00999995])
    ```

    The tiny entry moves by 0.00999995, not exactly 0.01: its \( \sqrt{\hat{s}} = 0.002 \) is small enough that \( \epsilon = 10^{-8} \) adds 0.0005% to the divisor. Remember that the update is \( \mathbf{w} \leftarrow \mathbf{w} - \text{step} \), so the first weight goes **up** by 0.01 (its gradient was negative) and the second goes down.

    **7. Find and fix the bug.** The slice `order[start:batch_size]` ends at the fixed position `batch_size` instead of `batch_size` rows after `start`. For the first batch `start = 0`, so `order[0:32]` is correct. For the second, `order[32:32]` is **empty**: a slice whose end is not after its start has no elements. `X[idx]` and `y[idx]` are then empty, `len(y)` inside `grad` is 0, and `2 / 0` raises `ZeroDivisionError`. The fix is `order[start:start + batch_size]`:

    ```python
    import numpy as np

    rng = np.random.default_rng(52)
    X = rng.normal(0, 1, size=(1_000, 2))
    y = X @ np.array([2.0, -1.0]) + rng.normal(0, 0.5, size=1_000)

    def grad(w, X, y):
        return 2 / len(y) * X.T @ (X @ w - y)

    order = np.arange(10)
    order[0:4], order[4:4], order[8:4]     # the buggy slice order[start:B] for start = 0, 4, 8 with B = 4
    # (array([0, 1, 2, 3]), array([], dtype=int64), array([], dtype=int64))

    def one_epoch(X, y, w, lr, batch_size, rng):
        order = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            idx = order[start:start + batch_size]   # fixed: the slice ends batch_size rows after start
            w = w - lr * grad(w, X[idx], y[idx])
        return w

    w = np.zeros(2)
    for epoch in range(3):
        w = one_epoch(X, y, w, 0.05, 32, np.random.default_rng(epoch))
    w.round(3)                             # array([ 1.99 , -0.989])   close to the true [2, -1]
    ```

    A wrong slice end is one of the commonest batching bugs. Here it crashed, which is lucky; with a gradient written as a NumPy `.mean(axis=0)` over the batch instead of `2 / len(y)`, an empty batch gives `nan` and a `RuntimeWarning: Mean of empty slice` instead, and the weights silently turn into `nan`. Printing `len(idx)` for the first few batches catches it at once.

    **8. Warmup plus cosine.**

    ```python
    import numpy as np

    rng = np.random.default_rng(52)                       # re-create the lesson's toy data exactly
    n = 1_000
    x1 = rng.normal(0, 1, size=n)
    x2 = 0.6 * x1 + 0.8 * rng.normal(0, 1, size=n)
    X = np.column_stack([x1, x2])
    y = X @ np.array([2.0, -1.0]) + rng.normal(0, 0.5, size=n)
    w0 = np.array([-1.0, 2.0])

    def loss(w):
        r = X @ w - y
        return r @ r / n

    def grad(w, idx):
        return 2 / len(idx) * X[idx].T @ (X[idx] @ w - y[idx])

    def cosine_lr(step, total, lr0):
        return 0.5 * lr0 * (1 + np.cos(np.pi * step / total))

    def warmup_cosine_lr(step, total, lr0, warmup_frac=0.1):
        warm = int(warmup_frac * total)                   # number of warmup steps
        if step < warm:
            return lr0 * (step + 1) / warm                # linear ramp: lr0/warm, 2*lr0/warm, ..., lr0
        return cosine_lr(step - warm, total - warm, lr0)  # then cosine over the remaining steps

    [round(float(warmup_cosine_lr(s, 1000, 0.1)), 4) for s in (0, 49, 99, 100, 550, 999)]
    # [0.001, 0.05, 0.1, 0.1, 0.05, 0.0]

    def train(schedule, lr0=0.1, B=8, epochs=12, seed=4):
        total = epochs * -(-n // B)
        rng = np.random.default_rng(seed)
        w, step = w0.copy(), 0
        for epoch in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, B):
                w = w - schedule(step, total, lr0) * grad(w, order[start:start + B])
                step += 1
        return loss(w) - loss(np.linalg.lstsq(X, y)[0])

    float(train(cosine_lr)), float(train(warmup_cosine_lr))
    # (3.1263938954917414e-05, 2.6822319682817675e-05)
    ```

    The ramp reaches 0.1 at step 99, the cosine starts there, is back to 0.05 halfway through the remaining 900 steps (step 550) and ends at about zero (\( 3 \times 10^{-7} \), which rounds to 0.0). The two final gaps are close (0.000031 and 0.000027): on this small, well-scaled problem nothing blows up at the start, so warmup has nothing to prevent. It earns its keep on deep networks with random initial weights and on Adam, whose \( \hat{\mathbf{s}} \) is based on very few gradients in the first steps.

## Resources

- [Why Momentum Really Works (Distill)](https://distill.pub/2017/momentum/): Gabriel Goh's interactive article on momentum, with sliders for the learning rate and \( \beta \), and the condition-number analysis behind the \( \sqrt{\kappa} \) speed-up.
- [CS231n: Neural Networks Part 3](https://cs231n.github.io/neural-networks-3/): Stanford's notes on SGD, momentum, Nesterov momentum, Adam and learning-rate annealing, with practical defaults.
- [Dive into Deep Learning: Optimization Algorithms](https://d2l.ai/chapter_optimization/index.html): chapters on mini-batch SGD, momentum, Adam and learning-rate scheduling, each with code.
- [An overview of gradient descent optimization algorithms (Sebastian Ruder)](https://arxiv.org/abs/1609.04747): a readable survey of every optimizer you're likely to meet, with their update equations side by side.
- [Adam: A Method for Stochastic Optimization (Kingma and Ba)](https://arxiv.org/abs/1412.6980): the original paper; Algorithm 1 on page 2 is exactly the update in this lesson.
- [PyTorch docs: torch.optim](https://pytorch.org/docs/stable/optim.html): `SGD` with `momentum`, `Adam`, `AdamW`, and the learning-rate schedulers `StepLR` and `CosineAnnealingLR`.
- [Android reference: SensorEvent](https://developer.android.com/reference/android/hardware/SensorEvent): the low-pass filter from the Android bridge, in the `TYPE_ACCELEROMETER` section.

## Checklist

- [ ] I can explain batch, stochastic and mini-batch gradient descent, and I measured a batch gradient's noise halving when the batch size quadruples.
- [ ] I can compute steps per epoch with a ceiling, and I shuffle row numbers (not X and y separately) every epoch.
- [ ] I wrote `minibatch_gd` and used it for batch GD, SGD and mini-batches.
- [ ] I can pick a learning rate with a sweep over powers of ten, and I know the noise band grows with \( \eta \) and shrinks with \( B \).
- [ ] I can write the two momentum equations, explain why sign-flipping gradients cancel, and I know to lower \( \eta \) by about \( 1 - \beta \) when adding momentum.
- [ ] I can say why GD needs about \( \kappa \) steps and tuned momentum about \( \sqrt{\kappa} \), and I saw both on the raw Play Store features.
- [ ] I can write Adam's update with its defaults, and name two things it doesn't fix (including a tilted ravine from uncentred features).
- [ ] I can implement step-decay, cosine and warmup schedules and plug them into a training loop.

**Next:** Day 53: Convexity, local minima and saddle points (coming soon)
