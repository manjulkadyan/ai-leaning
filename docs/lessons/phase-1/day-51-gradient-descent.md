# Day 51: Gradient Descent

**Phase 1 · Week 8 · ~150 minutes**

## Objectives

- [ ] State the gradient-descent update \( \mathbf{w} \leftarrow \mathbf{w} - \eta \nabla L(\mathbf{w}) \), say what every symbol means, and explain why there is a minus sign.
- [ ] Run gradient descent by hand for a few steps in one dimension, and in code for any number of parameters.
- [ ] Explain, with the "the gap shrinks by a fixed factor each step" argument, why a learning rate can be too small, about right, too large, or so large that the loss explodes, and compute the limit \( 2/\text{curvature} \).
- [ ] Recognise divergence in the output (`inf`, `nan`, overflow warnings) and fix it.
- [ ] Explain why a stretched bowl makes gradient descent zig-zag and crawl, define the condition number, and connect the curvatures to the eigenvalues of Day 24.
- [ ] Show that gradient descent finds *a* minimum, the one whose valley you start in, not necessarily the lowest one.
- [ ] Write a gradient-descent loop with sensible stopping rules (a step budget, a small gradient, a divergence alarm), and say why "the loss stopped improving" is a weak rule.
- [ ] Fit a linear regression to the Play Store data with gradient descent, track the loss per step, and match `np.linalg.lstsq`.
- [ ] Show on real data why standardizing features makes gradient descent thousands of times faster.

**Prerequisites.**

- [Day 22](day-22-norms-distances.md): the L2 norm (used as "the length of the gradient") and standardizing features.
- [Day 24](day-24-eigenvectors.md): eigenvalues and eigenvectors of a symmetric matrix, `np.linalg.eigh`.
- [Day 25](day-25-projections-pca.md): least squares and `np.linalg.lstsq`, the direct answer we compare against.
- [Day 29](day-29-derivatives.md) and [Day 30](day-30-derivative-rules.md): derivatives as slopes, the power rule, central differences.
- [Day 31](day-31-chain-rule.md): the chain rule, used to differentiate the two-valley loss.
- [Day 32](day-32-partial-derivatives-gradient.md): the gradient points uphill, contour maps, and a first walk downhill on a line-fit loss.
- [Day 33](day-33-vector-matrix-gradients.md): the MSE gradient \( \frac{2}{n}X^\top(X\mathbf{w} - \mathbf{y}) \) and the `(loss, gradient)` function contract.
- [Day 50](day-50-loss-functions.md): what a loss is, and the loss as a function of the parameters.

---

## The big idea

You are on a hillside in thick fog, and you want to get to the bottom of the valley. You can't see the valley. All you can feel is the slope of the ground under your boots. So you do the natural thing: feel which way is downhill, take a step that way, and repeat. Steep ground, big step; flat ground, small step. When the ground under you is flat, you're at the bottom (or at least at *a* bottom).

That is **gradient descent**, and it is how almost every model in this course gets trained, from a two-parameter line to a large language model. Day 50 gave you the hillside: the loss \( L(\mathbf{w}) \), a function of the model's parameters. Days 29–33 gave you the slope under your boots: the gradient \( \nabla L(\mathbf{w}) \). Today puts them together into an algorithm, and studies the decision that makes or breaks it: **how big a step to take**.

| Fog-walk | Gradient descent |
|---|---|
| Where you stand | The current parameters \( \mathbf{w} \) |
| Height above sea level | The loss \( L(\mathbf{w}) \) |
| Slope under your boots | The gradient \( \nabla L(\mathbf{w}) \) (points uphill) |
| Stride length | The learning rate \( \eta \) |
| "Ground feels flat, I'll stop" | A stopping rule |

Why not solve for the bottom directly, the way Day 25's `lstsq` did? For linear regression you can, and we'll use that answer to check our work. But almost every other model (logistic regression with Day 50's cross-entropy loss, every neural network) has no formula for the bottom. The only thing you can compute is the slope where you stand. Gradient descent needs nothing more, which is why it is the workhorse.

## The update rule

### What it is

Here is the whole algorithm:

\[
\mathbf{w} \leftarrow \mathbf{w} - \eta\, \nabla L(\mathbf{w})
\]

One symbol at a time:

- \( \mathbf{w} \) is the vector of all the model's parameters (for a line: slope and intercept). A **parameter** is a number the training is allowed to change.
- \( \nabla L(\mathbf{w}) \) is the gradient of the loss at the current \( \mathbf{w} \): one entry per parameter, pointing in the direction where the loss rises fastest (Day 32).
- \( \eta \) ("eta") is the **learning rate**, a small positive number you choose. It scales the step. A number you choose rather than one training learns is called a **hyperparameter**.
- The minus sign turns "uphill" into "downhill".
- \( \leftarrow \) means "replace with", like `w = w - lr * grad` in code. You do it again and again, recomputing the gradient at each new point. One application of the rule is a **step** (or **iteration**).

### Why it works

Near any point, a smooth function looks like a straight line (in 2-D, a tilted plane). That's what a derivative is (Day 29): for a small change \( \Delta w \), the loss changes by about \( L'(w)\,\Delta w \). If you choose \( \Delta w = -\eta\,L'(w) \), the change is about \( -\eta\,L'(w)^2 \), which is negative whenever the slope isn't zero. So **a small enough step against the slope always lowers the loss**. "Small enough" is doing a lot of work in that sentence; the next section is about it.

### A worked example by hand

Take the loss \( L(w) = (w - 3)^2 + 1 \). Its lowest point is at \( w = 3 \), where \( L = 1 \). Its derivative is \( L'(w) = 2(w - 3) \) (Day 30's power rule and sum rule). Start at \( w_0 = 0 \) with \( \eta = 0.1 \):

- Step 0: slope \( 2(0 - 3) = -6 \). A negative slope means "the loss falls to the right", so move right: \( w_1 = 0 - 0.1 \times (-6) = 0.6 \). Loss \( (0.6 - 3)^2 + 1 = 6.76 \).
- Step 1: slope \( 2(0.6 - 3) = -4.8 \), so \( w_2 = 0.6 - 0.1 \times (-4.8) = 1.08 \). Loss \( 4.6864 \).
- Step 2: slope \( 2(1.08 - 3) = -3.84 \), so \( w_3 = 1.08 + 0.384 = 1.464 \). Loss \( 3.359296 \).

The steps get shorter on their own (0.6, then 0.48, then 0.384) because the slope gets gentler as you approach the bottom. You never told the algorithm where the bottom is.

<figure class="diagram" markdown>
![Animation: the parabola L(w) = (w - 3) squared plus 1 in blue with its minimum at w = 3 marked in teal; an orange ball starts at w = 0, a dashed red tangent line shows the slope there, and a teal arrow along the w axis shows the step; the ball hops to w = 0.6, 1.08, 1.464, 1.771 and 2.017, leaving orange dots behind, each hop shorter than the last; a panel on the right lists each step: w, the slope, and the new w, ending with after 5 steps w = 2.017, loss = 1.966, and the note each step closes 20% of the gap to w = 3](../../assets/img/phase-1/day51-1d-descent.svg)
<figcaption>Animated: feel the slope (red tangent), step against it (teal arrow), repeat. The hops shrink because the slope shrinks.</figcaption>
</figure>

### The same steps in code

The code reproduces the hand calculation. The format spec `{w:.4f}` inside an f-string prints `w` as a fixed-point number with 4 digits after the decimal point.

```python
import numpy as np

def loss(w):
    return (w - 3) ** 2 + 1          # lowest point: w = 3, loss = 1

def grad(w):
    return 2 * (w - 3)               # its derivative (Day 30's power and sum rules)

lr = 0.1                             # the learning rate, eta
w = 0.0                              # the starting guess
for step in range(5):
    g = grad(w)
    print(f"step {step}: w = {w:.4f}  loss = {loss(w):.4f}  slope = {g:.4f}")
    w = w - lr * g                   # the whole algorithm
print(f"step 5: w = {w:.4f}  loss = {loss(w):.4f}")
# step 0: w = 0.0000  loss = 10.0000  slope = -6.0000
# step 1: w = 0.6000  loss = 6.7600  slope = -4.8000
# step 2: w = 1.0800  loss = 4.6864  slope = -3.8400
# step 3: w = 1.4640  loss = 3.3593  slope = -3.0720
# step 4: w = 1.7712  loss = 2.5099  slope = -2.4576
# step 5: w = 2.0170  loss = 1.9664
```

The same loop works for any number of parameters, because NumPy computes `w - lr * grad(w)` entry by entry when `w` is an array. Package it as a function you'll reuse all day. It records every point visited, so you can plot the walk:

```python
def gradient_descent(grad, w0, lr, steps):
    """Plain gradient descent. Returns every point visited, the start included."""
    w = np.array(w0, dtype=float)        # works for a number or a vector of any length
    path = [w.copy()]
    for _ in range(steps):
        w = w - lr * grad(w)
        path.append(w.copy())
    return np.array(path)

path = gradient_descent(grad, 0.0, lr=0.1, steps=30)
path.shape                               # (31,)
path[[0, 1, 2, 10, 20, 30]].round(4)     # array([0.    , 0.6   , 1.08  , 2.6779, 2.9654, 2.9963])
dist = 3 - path                          # how far from the bottom, step by step
(dist[1:6] / dist[0:5]).round(4)         # array([0.8, 0.8, 0.8, 0.8, 0.8])
```

Line by line:

- `np.array(w0, dtype=float)` makes a float copy of the start. For a number like `0.0` it is a **0-dimensional array**: an array with shape `()`, no axes, holding exactly one number (like a `FloatArray` that can only ever have one element, but still a NumPy array with a `dtype`). For a list like `[-4.0, 1.5]` it is a vector of shape `(2,)`. Every later line works for both.
- `path.append(w.copy())` stores a snapshot of the current values. In this loop the copy isn't strictly needed: `w = w - lr * grad(w)` builds a brand-new array each step and rebinds the name `w` to it, so the old array in `path` is never touched. The copy is defensive: it keeps the function correct if someone later "optimizes" the update into the in-place `w -= lr * grad(w)`, which modifies the one existing array. The box below shows what that does.
- `np.array(path)` stacks the snapshots: shape `(steps + 1,)` for one parameter, `(steps + 1, d)` for \( d \) parameters.
- `path[[0, 1, 2, 10, 20, 30]]` is fancy indexing (Day 8): it picks those six rows.
- `dist[1:6] / dist[0:5]` divides each gap by the gap one step earlier.

One display detail before more output: when you press ▶ Run, a bare expression is echoed with `repr()`, the way a notebook does. A NumPy number therefore shows as `np.float64(0.5)` rather than `0.5`, and a NumPy boolean as `np.True_`. It is the same number; the comments in this lesson show what Run shows.

!!! warning "Common mistake: storing the same array over and over"
    Python lists store references (Day 2), and `-=` on a NumPy array changes it in place instead of making a new one. Combine the two without `.copy()` and every entry of the path is the same object, so every row shows the *final* value:

    ```python
    def gd_inplace_no_copy(grad, w0, lr, steps):
        w = np.array(w0, dtype=float)
        path = [w]                           # BUG: stores the array itself, not a snapshot
        for _ in range(steps):
            w -= lr * grad(w)                # in place: changes the one array every entry refers to
            path.append(w)
        return np.array(path)

    gd_inplace_no_copy(grad, 0.0, lr=0.1, steps=3)   # array([1.464, 1.464, 1.464, 1.464])
    ```

    The right answer is `[0, 0.6, 1.08, 1.464]`. The tell-tale sign is a recorded history that is constant, or a loss curve that is flat at the final value. The fix is `path.append(w.copy())` (or the non-in-place `w = w - lr * grad(w)`).

### Underneath: every step multiplies the gap by the same number

That last line is the key to the whole lesson: **every step shrinks the distance to the bottom by the same factor, 0.8.** Here is why. Write \( e = w - 3 \) for the **gap** between where you are and the bottom. One step does:

\[
w_{\text{new}} - 3 = (w - 3) - \eta \cdot 2(w - 3) = (1 - 2\eta)\,(w - 3)
\qquad\text{so}\qquad
e_{\text{new}} = (1 - 2\eta)\, e
\]

With \( \eta = 0.1 \), \( 1 - 2\eta = 0.8 \). The gaps in the hand example are 3, 2.4, 1.92, 1.536: each one is 0.8 times the one before. After \( k \) steps the gap is \( 0.8^k \times 3 \); after 30 steps that's \( 0.8^{30} \times 3 = 0.0037 \), which matches `path[30]` = 2.9963.

### How it goes wrong: adding the gradient

The gradient points **uphill**. Flip the sign by accident and every step climbs. The bug on purpose, so you recognise the output:

```python
w = 0.0
for step in range(4):
    w = w + 0.1 * grad(w)            # BUG: + instead of -
    print(f"step {step + 1}: w = {w:.4f}  loss = {loss(w):.4f}")
# step 1: w = -0.6000  loss = 13.9600
# step 2: w = -1.3200  loss = 19.6624
# step 3: w = -2.1840  loss = 27.8739
# step 4: w = -3.2208  loss = 39.6984
```

!!! warning "Common mistake: adding the gradient"
    The symptom is a loss that **rises from the very first step** and keeps rising faster, while `w` walks away from the minimum. (A too-large learning rate, next section, can also make the loss rise, but it bounces from side to side first.) The fix is the minus sign: `w = w - lr * grad(w)`. In PyTorch (Phase 3) the built-in optimizers handle the sign for you, but any hand-written update is where this bug lives.

## How big a step? The learning rate

### What it is and why it matters

The learning rate \( \eta \) decides how far each step goes in the direction the gradient gives. It exists because the gradient only tells you the *direction* and the *steepness*, not how far away the bottom is. Too small and training takes forever; too large and it never settles, or blows up. Choosing it is the most common tuning job in all of deep learning.

**Intuition first.** Picture walking down into a narrow ditch in the dark. Tiny shuffles get you there, slowly. Normal strides get you there fast. A huge leap carries you across the ditch and up the other side, possibly higher than where you started. Then you leap back, even higher. That last case is divergence.

### Worked example: the factor \( 1 - 2\eta \) for four learning rates

From the formula above, each step multiplies the gap by \( 1 - 2\eta \) on this loss. By hand, from \( w_0 = 0 \) (gap \( -3 \), so \( w = 3 + e \)):

| \( \eta \) | Factor \( 1 - 2\eta \) | Gaps after 1, 2, 3 steps | What happens |
|---|---|---|---|
| 0.02 | +0.96 | −2.88, −2.765, −2.654 | Converges, but each step closes only 4% of the gap. Too small: it crawls. |
| 0.5 | 0 | 0, 0, 0 | Lands on the bottom in one step. Perfect, for this loss. |
| 0.9 | −0.80 | +2.4, −1.92, +1.536 | The sign flips every step: it jumps over the bottom, lands on the far side, jumps back, and still converges because the size of the factor, 0.8, is below 1. |
| 1.05 | −1.10 | +3.3, −3.63, +3.993 | Each jump overshoots by more than the last. The gap grows by 10% per step: **divergence**. |

So gradient descent converges exactly when \( |1 - 2\eta| < 1 \), that is, when \( 0 < \eta < 1 \) for this loss. The code reproduces the table (add 3 to each gap to get \( w \)). In `{lr:<4}` the `<4` means "left-align in a field 4 characters wide", and in `{1 - 2 * lr:+.2f}` the `+` means "always print the sign":

```python
for lr in [0.02, 0.5, 0.9, 1.05]:
    p = gradient_descent(grad, 0.0, lr, steps=20)
    print(f"lr = {lr:<4}  factor 1 - 2*lr = {1 - 2 * lr:+.2f}   w after 1, 2, 3, 20 steps:",
          p[[1, 2, 3, 20]].round(3))
# lr = 0.02  factor 1 - 2*lr = +0.96   w after 1, 2, 3, 20 steps: [0.12  0.235 0.346 1.674]
# lr = 0.5   factor 1 - 2*lr = +0.00   w after 1, 2, 3, 20 steps: [3. 3. 3. 3.]
# lr = 0.9   factor 1 - 2*lr = -0.80   w after 1, 2, 3, 20 steps: [5.4   1.08  4.536 2.965]
# lr = 1.05  factor 1 - 2*lr = -1.10   w after 1, 2, 3, 20 steps: [  6.3    -0.63    6.993 -17.182]
```

### Underneath: curvature sets the speed limit

Where did the "2" in \( 1 - 2\eta \) come from? From how sharply the bowl curves. For a general bowl \( L(w) = a(w - c)^2 \), the derivative is \( 2a(w - c) \), and the same algebra gives the factor \( 1 - 2a\eta \). The number \( 2a \) is the **second derivative** \( L''(w) \): the derivative of the derivative. It measures how fast the slope changes as you move, in other words how sharply the bowl curves. We'll call it the **curvature**. (Day 53 treats curvature properly; here you only need the number.) So:

\[
\text{gap factor} = 1 - \eta\, L''
\qquad\Longrightarrow\qquad
\text{converges when } 0 < \eta < \frac{2}{L''},
\qquad \text{fastest at } \eta = \frac{1}{L''}
\]

Check it against our loss: \( L'' = 2 \), so the limit is \( 2/2 = 1 \) and the one-step learning rate is \( 1/2 = 0.5 \). Both match the table. In words: **a sharply curved bowl needs small steps; a gently curved one tolerates big ones.** Real losses aren't perfect bowls, but near a minimum they look like one (zoom in on any smooth valley floor and it looks like a parabola), so this rule describes the end of every training run.

Here are four learning rates as walks on the parabola. The matplotlib setup is the same as Day 29's. New arguments: `sharey=True` makes the four panels share one vertical scale; `alpha=0.7` makes a line 70% opaque; `s=28` is a marker's area in points²; and `zorder` sets drawing order (higher numbers are drawn on top), so the start and the minimum aren't hidden under the path:

```python
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "savefig.facecolor": "white"})

ws = np.linspace(-1.5, 7.5, 300)
cases = [(0.02, "too small: crawls"), (0.3, "about right"),
         (0.9, "too large: zig-zags in"), (1.05, "far too large: diverges")]
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
for ax, (lr, title) in zip(axes, cases):
    p = gradient_descent(grad, 0.0, lr, steps=8)
    ax.plot(ws, loss(ws), color="#3949ab", lw=2)
    ax.plot(p, loss(p), "-", color="#ef6c00", lw=1, alpha=0.7)
    ax.scatter(p, loss(p), color="#ef6c00", s=28, zorder=3)
    ax.scatter([p[0]], [loss(p[0])], color="#e53935", s=60, zorder=4, label="start")
    ax.plot(3, 1, "*", color="#00897b", ms=14, zorder=5, label="minimum")
    ax.set_title(f"lr = {lr}: {title}", fontsize=11)
    ax.set_xlabel("w")
    ax.set_xlim(-1.5, 7.5); ax.set_ylim(0, 22)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("loss  (w - 3)² + 1")
axes[0].legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig("day51-learning-rates.svg", format="svg", bbox_inches="tight")
plt.close(fig)
```

<figure class="diagram" markdown>
![Four panels, each the parabola (w - 3) squared plus 1 with eight gradient-descent steps from w = 0 drawn as orange dots joined by lines: with lr = 0.02 the dots bunch up close to the start on the left wall; with lr = 0.3 they drop to the minimum at w = 3 in three or four steps; with lr = 0.9 they bounce from wall to wall, alternating sides of the minimum, and settle in; with lr = 1.05 they bounce outward, climbing higher up the walls with every step and leaving the plot](../../assets/img/phase-1/day51-learning-rates.svg)
<figcaption>Eight steps from w = 0 with four learning rates. Too small crawls; too large overshoots, and past the limit of 2/curvature every overshoot is bigger than the last.</figcaption>
</figure>

### Reading a loss curve

A **loss curve** is the loss plotted against the step number. It's what you'll look at when training a real model, because you can't draw a million-dimensional valley. Plot the loss **above its minimum** on a log scale (`ax.semilogy` makes the vertical axis logarithmic): a constant gap factor then shows up as a straight line, because multiplying by the same number each step means subtracting the same amount from the logarithm. The steeper the line, the faster the training. In the code, `ms` is the marker size, `mfc="none"` ("marker face colour") draws hollow markers so the 0.1 curve stays visible under the 0.9 one, and `6 if lr == 0.1 else 3` is Python's inline conditional, like Kotlin's `if (lr == 0.1) 6 else 3`.

```python
fig, ax = plt.subplots(figsize=(7, 3.8))
for lr, col, style in [(0.02, "#9e9e9e", "o-"), (0.1, "#3949ab", "s-"), (0.3, "#00897b", "o-"),
                       (0.9, "#ef6c00", "o--"), (1.05, "#e53935", "o-")]:
    p = gradient_descent(grad, 0.0, lr, steps=40)
    ax.semilogy(loss(p) - 1, style, ms=6 if lr == 0.1 else 3, lw=1.5, color=col,
                mfc="none" if lr == 0.1 else col, label=f"lr = {lr}")
ax.set_xlabel("step")
ax.set_ylabel("loss above the minimum (log scale)")
ax.set_ylim(1e-12, 1e5)                  # returns the new limits: (1e-12, 100000.0)
ax.grid(alpha=0.3, which="both")
ax.legend(frameon=False, ncol=2, loc="lower left")
fig.savefig("day51-loss-curves.svg", format="svg", bbox_inches="tight")
plt.close(fig)
```

<figure class="diagram" markdown>
![Log-scale loss curves over 40 steps for five learning rates: lr = 1.05 in red climbs steadily from 10 to about 10 to the 4; lr = 0.02 in grey falls very slowly to about 0.4; lr = 0.1 (blue squares) and lr = 0.9 (orange dashed) fall along exactly the same straight line to about 10 to the minus 7; lr = 0.3 in teal falls steeply, below 10 to the minus 12 by step 16](../../assets/img/phase-1/day51-loss-curves.svg)
<figcaption>Straight lines on a log scale: a constant factor per step. Note that lr = 0.1 and lr = 0.9 lie exactly on top of each other: their factors are +0.8 and −0.8, the same size.</figcaption>
</figure>

Once you've seen these shapes, they summarise like this:

| Loss curve shape | Diagnosis |
|---|---|
| Falls steadily, straight on a log scale | Healthy. The slope tells you the speed. |
| Falls, but barely | Learning rate too small, or a stretched bowl (next section). |
| Falls while bouncing up and down | Learning rate near the limit; lowering it may be faster. |
| Rises every step | Learning rate above the limit, or the sign bug above. |
| Becomes `inf`, then `nan` | Divergence went all the way (next). |

### How it goes wrong: divergence all the way to `nan`

Push the learning rate well past the limit and let it run. The factor is \( 1 - 2 \times 1.5 = -2 \), so the gap doubles every step:

```python
p = gradient_descent(grad, 0.0, lr=1.5, steps=1100)
# <cell>:6: RuntimeWarning: overflow encountered in scalar multiply
# <cell>:6: RuntimeWarning: invalid value encountered in scalar subtract
p[[1, 2, 3, 10, 100]]
# array([ 9.0000000e+00, -9.0000000e+00,  2.7000000e+01, -3.0690000e+03,
#        -3.8029518e+30])
p[-3:]                                   # array([nan, nan, nan])
```

Read the warnings in order. After about a thousand doublings the numbers pass the largest value a float64 can hold (about \( 1.8 \times 10^{308} \)), so `2 * (w - 3)` becomes `inf` (infinity) and NumPy warns about **overflow**. On the next step the update computes `inf - inf`, which has no meaningful value, so the result is `nan` ("not a number") and NumPy warns about an **invalid value**. From then on everything is `nan`, because any arithmetic with `nan` gives `nan`.

!!! warning "Common mistake: \"bigger learning rate, faster training\""
    Only up to a point. Past \( 1/L'' \) a bigger \( \eta \) makes each step *overshoot*, and the speed gets **worse** (0.9 is exactly as slow as 0.1 above). Past \( 2/L'' \) it diverges. The usual symptom in a real training log is a loss that falls for a few steps, starts oscillating, then becomes `inf` and finally `nan`, with the two `RuntimeWarning`s above. When you see `nan`, divide the learning rate by 10 before you suspect anything else. A practical habit: try learning rates on a log grid (0.001, 0.01, 0.1, 1) first, then refine between the best two.

## Two dimensions: round bowls and stretched bowls

### What changes with more parameters

With two parameters the walk happens on a contour map (Day 32). Take two bowls with their bottom at the origin:

- **Round bowl** \( L = w_1^2 + w_2^2 \), gradient \( [2w_1, 2w_2] \): curvature 2 in every direction.
- **Stretched bowl** \( L = w_1^2 + 10\,w_2^2 \), gradient \( [2w_1, 20w_2] \): curvature 2 along \( w_1 \), but 20 along \( w_2 \). Its contours are ellipses, ten times steeper across than along.

**Why this matters.** Real models have parameters that affect the loss very differently: a weight on a feature measured in megabytes and a weight on a 0/1 flag, or a weight early in a network and one at the end. Their curvatures differ, and one learning rate must serve all of them. This section shows what that costs.

**Intuition first.** Picture a long, narrow canyon with a gentle slope along its floor. Standing on a canyon wall, "steepest downhill" points almost straight across the canyon, not along it towards the exit. So you hop across, land on the other wall, hop back, and make slow progress along the floor.

### Worked example by hand

Because each gradient entry only involves its own coordinate, gradient descent on these bowls is two independent one-dimensional walks, and each coordinate gets its own gap factor \( 1 - \eta \times \text{its curvature} \):

| | Factor along \( w_1 \) | Factor along \( w_2 \) | Converges if |
|---|---|---|---|
| Round bowl | \( 1 - 2\eta \) | \( 1 - 2\eta \) | \( \eta < 2/2 = 1 \) |
| Stretched bowl | \( 1 - 2\eta \) | \( 1 - 20\eta \) | \( \eta < 2/20 = 0.1 \) |

The steep direction sets the speed limit, \( \eta < 0.1 \). At \( \eta = 0.09 \), from \( (-4, 1.5) \):

- along \( w_1 \) the factor is \( 1 - 0.18 = 0.82 \): \( -4 \to -3.28 \to -2.6896 \), closing only 18% of the gap per step;
- along \( w_2 \) it is \( 1 - 1.8 = -0.8 \): \( 1.5 \to -1.2 \to 0.96 \), overshooting and flipping sign every step.

The result is the classic zig-zag:

<figure class="diagram" markdown>
![Animation, two contour maps side by side, both starting at (-4, 1.5) with the minimum marked by a red dot: on the left, the round bowl with circular contours and lr = 0.3, the teal path runs straight to the centre, every step closing 60% of the gap, and the loss after 10 steps is 2.0e-07; on the right, the stretched bowl with long flat elliptical contours and lr = 0.09, the orange path bounces up and down across the narrow direction while inching to the right, and after 10 steps the loss is still 0.5617](../../assets/img/phase-1/day51-zigzag.svg)
<figcaption>Animated: the same start, ten steps each. On the round bowl the gradient points straight at the bottom; on the stretched bowl it points mostly across the valley, so the walk zig-zags across while crawling along.</figcaption>
</figure>

The code reproduces the hand numbers:

```python
def round_grad(w):
    return np.array([2 * w[0], 2 * w[1]])            # L = w1² + w2²

def stretched_grad(w):
    return np.array([2 * w[0], 20 * w[1]])           # L = w1² + 10 w2²

def stretched_loss(w):
    return w[0] ** 2 + 10 * w[1] ** 2

start = [-4.0, 1.5]
gradient_descent(round_grad, start, lr=0.5, steps=1)       # one step lands on the bottom
# array([[-4. ,  1.5],
#        [ 0. ,  0. ]])
p = gradient_descent(stretched_grad, start, lr=0.09, steps=6)
p.round(3)                     # w2 flips sign every step; w1 creeps towards 0
# array([[-4.   ,  1.5  ],
#        [-3.28 , -1.2  ],
#        [-2.69 ,  0.96 ],
#        [-2.205, -0.768],
#        [-1.808,  0.614],
#        [-1.483, -0.492],
#        [-1.216,  0.393]])
```

How many steps does each bowl need to get the loss below \( 10^{-6} \)? The helper below counts them. `np.isfinite(w)` is `True` for ordinary numbers and `False` for `inf` and `nan`; `.all()` asks whether every entry is finite.

```python
def steps_needed(grad, loss_fn, w0, lr, target=1e-6, max_steps=10_000):
    """How many steps until the loss drops below target (None if it never does)."""
    w = np.array(w0, dtype=float)
    for k in range(max_steps):
        if loss_fn(w) < target:
            return k
        w = w - lr * grad(w)
        if not np.isfinite(w).all():
            return None                  # blew up
    return None

for lr in [0.01, 0.05, 0.09, 0.0909, 0.099, 0.101]:
    print(f"lr = {lr:<7} stretched bowl: {steps_needed(stretched_grad, stretched_loss, start, lr)} steps")
# lr = 0.01    stretched bowl: 411 steps
# lr = 0.05    stretched bowl: 79 steps
# lr = 0.09    stretched bowl: 43 steps
# lr = 0.0909  stretched bowl: 44 steps
# lr = 0.099   stretched bowl: 419 steps
# lr = 0.101   stretched bowl: None steps
round_loss = lambda w: w[0] ** 2 + w[1] ** 2
steps_needed(round_grad, round_loss, start, lr=0.5)      # 1
```

(`lambda w: ...` defines a small unnamed function in one line, like a Kotlin lambda `{ w -> ... }`.)

### Underneath: the condition number

The round bowl needs **one** step, at \( \eta = 1/\text{curvature} = 0.5 \). The stretched bowl needs at least about 43, however you tune \( \eta \), and slightly past 0.1 it never arrives. Raising \( \eta \) speeds up the flat direction (its factor \( 1 - \eta\lambda_{\min} \) falls towards 0) but slows the steep one (its factor \( 1 - \eta\lambda_{\max} \) heads towards −1). The best compromise makes the two factors the same size, with opposite signs. Write \( \lambda_{\min} \) and \( \lambda_{\max} \) for the smallest and largest curvature ("lambda", the letter Day 24 used for eigenvalues; the reason for the name comes in a moment):

\[
1 - \eta\lambda_{\min} = -(1 - \eta\lambda_{\max})
\quad\Longrightarrow\quad
\eta_{\text{balanced}} = \frac{2}{\lambda_{\min} + \lambda_{\max}},
\qquad
\text{factor} = \frac{\lambda_{\max} - \lambda_{\min}}{\lambda_{\max} + \lambda_{\min}}
\]

For the stretched bowl, \( \lambda_{\min} = 2 \) and \( \lambda_{\max} = 20 \): \( \eta = 2/22 \approx 0.0909 \) and a factor of \( 18/22 = 9/11 \approx 0.82 \) in both directions (Exercise 5 checks it).

The ratio of the largest curvature to the smallest, \( \kappa = \lambda_{\max}/\lambda_{\min} \) (here \( 20/2 = 10 \)), is called the **condition number**, written \( \kappa \) ("kappa"). Divide the top and bottom of the balanced factor by \( \lambda_{\min} \) and it becomes \( (\kappa - 1)/(\kappa + 1) \): 0 for a round bowl (\( \kappa = 1 \)), 0.82 for \( \kappa = 10 \), 0.98 for \( \kappa = 100 \).

A fixed factor per step has a name: **linear convergence** (the gap falls geometrically, which looks like a straight line on a log-scale loss curve). When \( \kappa \) is large, the factor is close to \( 1 - 2/\kappa \), so the loss (the gap squared) shrinks by about \( 1 - 4/\kappa \) per step, and reaching a loss \( \varepsilon \) from a starting loss \( L_0 \) takes about \( \tfrac{\kappa}{4}\ln(L_0/\varepsilon) \) steps: **proportional to the condition number**. Check it on bowls \( w_1^2 + \kappa\,w_2^2 \), each with its balanced rate (`k=kappa` in the lambdas freezes the current value of `kappa` into each function, because a Python lambda otherwise looks the variable up when it is called, not when it is made):

```python
for kappa in [1, 10, 100, 1000]:
    g_k = lambda w, k=kappa: np.array([2 * w[0], 2 * k * w[1]])
    L_k = lambda w, k=kappa: w[0] ** 2 + k * w[1] ** 2
    lr_k = 2 / (2 + 2 * kappa)                       # the balanced rate, 2 / (smallest + largest curvature)
    print(f"kappa = {kappa:4}: factor {(kappa - 1) / (kappa + 1):.4f}, "
          f"{steps_needed(g_k, L_k, start, lr_k)} steps to loss < 1e-6")
# kappa =    1: factor 0.0000, 1 steps to loss < 1e-6
# kappa =   10: factor 0.8182, 44 steps to loss < 1e-6
# kappa =  100: factor 0.9802, 483 steps to loss < 1e-6
# kappa = 1000: factor 0.9980, 5386 steps to loss < 1e-6
```

Ten times the condition number, about ten times the steps. For \( \kappa = 100 \) the estimate is \( \tfrac{100}{4}\ln(241/10^{-6}) = 25 \times 19.3 = 482 \), next to the measured 483 (the starting loss is \( 16 + 100 \times 2.25 = 241 \)). **The bigger the condition number, the slower plain gradient descent goes, whatever learning rate you pick.**

### How it goes wrong: tuning for the flat direction

The flat direction is the slow one, so it is tempting to pick a learning rate that suits it: \( 1/\text{curvature} = 1/2 = 0.5 \) would solve \( w_1 \) in one step. But the steep direction's factor at 0.5 is \( 1 - 20 \times 0.5 = -9 \):

```python
gradient_descent(stretched_grad, start, lr=0.5, steps=4).round(1)
# array([[-4.0000e+00,  1.5000e+00],
#        [ 0.0000e+00, -1.3500e+01],
#        [ 0.0000e+00,  1.2150e+02],
#        [ 0.0000e+00, -1.0935e+03],
#        [ 0.0000e+00,  9.8415e+03]])
```

\( w_1 \) is solved after one step, exactly as planned, while \( w_2 \) is multiplied by −9 every step. The rule is the one from one dimension, applied to every direction: **the learning rate must be below \( 2/\text{curvature} \) for the steepest direction**, and the flat directions then take as long as they take.

### Tilted bowls: the curvatures are eigenvalues

Real loss bowls are stretched in *tilted* directions, not along the axes. Day 32's line-fit valley was like that, and its Exercise 6 found that gradient descent diverged slightly past \( \eta \approx 0.12 \), with the promise that you'd compute the exact limit today. Here it is.

For the MSE loss, Day 33's gradient is \( \nabla L = \frac{2}{n}X^\top(X\mathbf{w} - \mathbf{y}) = \frac{2}{n}X^\top X\,\mathbf{w} - \frac{2}{n}X^\top\mathbf{y} \). It is a matrix times \( \mathbf{w} \), plus a constant. So if you move \( \mathbf{w} \) by \( \Delta\mathbf{w} \), the gradient changes by

\[
H\,\Delta\mathbf{w}, \qquad H = \frac{2}{n}X^\top X
\]

- \( H \) plays the role that the curvature \( L'' \) played in one dimension: "how much the slope changes per unit you move". It is called the **Hessian**: the matrix of second derivatives, whose entry \( (j, k) \) is \( \partial^2 L / \partial w_j \partial w_k \), "differentiate the loss by \( w_k \), then differentiate that by \( w_j \)". For the MSE you can see it directly: entry \( j \) of the gradient is \( \frac{2}{n}\sum_i x_{ij}(\mathbf{x}_i \cdot \mathbf{w} - y_i) \), and differentiating that by \( w_k \) leaves \( \frac{2}{n}\sum_i x_{ij}x_{ik} \), which is entry \( (j, k) \) of \( \frac{2}{n}X^\top X \). For linear regression \( H \) is the same everywhere, because the loss is an exact bowl. For a neural network \( H \) changes with \( \mathbf{w} \), so the safe learning rate changes as training moves through the landscape; Day 53 meets the Hessian again.
- \( H \) is symmetric, so by Day 24 it has perpendicular eigenvectors, and along each eigenvector \( \mathbf{v} \) it only stretches: \( H\mathbf{v} = \lambda\mathbf{v} \). Along that direction the bowl is a one-dimensional parabola with curvature \( \lambda \) ("lambda", the eigenvalue).
- So **the eigenvectors of \( H \) are the axes of the tilted ellipses, and the eigenvalues are the curvatures along them.** Gradient descent converges when \( \eta < 2/\lambda_{\max} \); the flattest direction, \( \lambda_{\min} \), sets how slowly it crawls; and \( \kappa = \lambda_{\max}/\lambda_{\min} \).

**By hand** for Day 32's four points \( x = 1, 2, 3, 4 \): \( X \) has rows \( (x_i, 1) \), so \( X^\top X = \begin{bmatrix} \sum x_i^2 & \sum x_i \\ \sum x_i & 4 \end{bmatrix} = \begin{bmatrix} 30 & 10 \\ 10 & 4 \end{bmatrix} \), and \( H = \frac{2}{4}X^\top X = \begin{bmatrix} 15 & 5 \\ 5 & 2 \end{bmatrix} \). Its eigenvalues solve \( \lambda^2 - 17\lambda + 5 = 0 \) (trace 17, determinant \( 30 - 25 = 5 \), Day 24's characteristic polynomial), so \( \lambda = (17 \pm \sqrt{269})/2 \), about 16.70 and 0.30. The limit is \( 2/16.70 \approx 0.1198 \). The code (`np.linalg.eigvalsh` returns the eigenvalues of a symmetric matrix in ascending order) agrees:

```python
x_data = np.array([1., 2., 3., 4.])              # Day 32's four points
y_data = np.array([2., 3., 5., 6.])
X32 = np.column_stack([x_data, np.ones(4)])      # columns: x and the intercept's ones
H = 2 / len(x_data) * X32.T @ X32                # how the gradient changes per unit of w
H
# array([[15.,  5.],
#        [ 5.,  2.]])
curv = np.linalg.eigvalsh(H)                     # Day 24: curvatures along the eigen-directions
curv                                             # array([ 0.29939027, 16.70060973])
2 / curv.max()                                   # np.float64(0.11975610662865498)   the largest safe lr
curv.max() / curv.min()                          # np.float64(55.78207309365642)     the condition number
```

The limit is 0.1198: slightly under the 0.12 that Day 32 saw diverge. And a condition number of 56 explains the slow crawl along that valley's floor that Day 32 noticed.

## A minimum, not the minimum

### What it is

Every bowl so far had one bottom. Many losses have several low points. A **local minimum** is a point lower than everything *near* it; the **global minimum** is the lowest point of all. Gradient descent only uses the slope where it stands, so it rolls into whichever valley it starts in and stops at that valley's bottom.

**Why you need to know this.** Neural-network losses are not bowls. The starting point (the random initial weights) can change where training ends, which is one reason the course fixes random seeds. Day 53 explains why this is less of a problem in practice than it sounds.

**Example.** \( L(w) = (w^2 - 1)^2 + 0.3w \) has two valleys, near \( w = -1 \) and \( w = +1 \); the \( +0.3w \) term tilts the landscape so the left valley is deeper. Its derivative is \( L'(w) = 4w(w^2 - 1) + 0.3 \) (chain rule, Day 31). Run gradient descent from six starts. `{w0:5}` gives a width with no alignment character: numbers are then right-aligned in a field 5 characters wide (strings would be left-aligned), which keeps the column tidy.

```python
g2 = lambda w: 4 * w * (w ** 2 - 1) + 0.3
L2 = lambda w: (w ** 2 - 1) ** 2 + 0.3 * w
for w0 in [-2.0, -0.5, 0.0, 0.1, 0.5, 2.0]:
    w_end = gradient_descent(g2, w0, lr=0.05, steps=200)[-1]
    print(f"start {w0:5}: ends at w = {w_end:.4f}, loss = {L2(w_end):.4f}")
# start  -2.0: ends at w = -1.0356, loss = -0.3054
# start  -0.5: ends at w = -1.0356, loss = -0.3054
# start   0.0: ends at w = -1.0356, loss = -0.3054
# start   0.1: ends at w = 0.9601, loss = 0.2941
# start   0.5: ends at w = 0.9601, loss = 0.2941
# start   2.0: ends at w = 0.9601, loss = 0.2941
```

**Underneath.** Between the valleys there is a hilltop at about \( w = 0.0754 \), where \( L'(w) = 0 \) too. Starts to its left roll left into the deeper valley (loss −0.305); starts to its right roll right and get stuck in the shallower one (loss 0.294). Both endpoints have a zero gradient, so a "gradient is small" stopping rule happily accepts either.

**How it goes wrong.** The mistake is concluding "the model can't do better" from one run. Here, a run started at 0.5 reports loss 0.294 and a zero gradient, when −0.305 was available. The fix in practice: try a few starting points (different seeds) and compare, and prefer losses known to have a single minimum where you can (Day 53: convex losses, which include linear regression's MSE).

!!! example "Try it: the gradient-descent playground"
    Pick a loss, **click** (or tap) anywhere on the map to choose a starting point, set the learning rate with the slider (it moves through round values such as 0.09, 0.1, 0.105, 0.5) or type any value into the box next to it, then press **Step** (one step) or **Run**. The orange trail is the walk, the teal dots are the minima, and the readout shows the loss and the length of the gradient. Things to try:

    - **Round bowl:** set the learning rate to 0.5 (one step to the bottom), then 0.9 (it jumps across and back), then 1.1 (it diverges). The limit is 2 / 2 = 1.
    - **Stretched bowl:** start at the default 0.09 and watch the zig-zag. Try 0.02: no zig-zag, but count the steps. Then try 0.105, slightly past the limit 2 / 20 = 0.1.
    - **Two minima:** this is the loss above with a \( y^2 \) added. The hilltop between the valleys is at \( x \approx 0.075 \). Click a start with \( x \) around 0.07, then one around 0.08 (right after a click, at step 0, the readout shows the start's coordinates), and watch them end in different valleys.
    - On any loss, find the learning rate where the trail first starts to zig-zag, and the one where it starts to blow up.

<div class="gd-playground"></div>

## Knowing when to stop

### What it is and why

`gradient_descent` above ran for a fixed number of steps. A real training loop has to decide by itself when to stop: too early and the model is worse than it could be; too late and you waste time, battery or money. A **stopping rule** (or **stopping criterion**) is the test the loop runs after each step. There are three standard rules, and you use all three together:

- **Rule 1, the budget:** stop after `max_steps`. Time is finite, and on a stretched bowl or a diverging run it may never converge.
- **Rule 2, flat ground:** stop when the length of the gradient \( \|\nabla L\| \) (its L2 norm, Day 22, computed with `np.linalg.norm`) drops below a tolerance `grad_tol`. At a minimum the gradient is zero, so a tiny gradient means you are nearly there.
- **Rule 3, the divergence alarm:** stop with an error if the loss is `inf` or `nan`, or has grown hugely. Fail loudly instead of returning garbage.

**Intuition.** Rule 2 is the fog-walker noticing the ground is flat under their boots. Rule 1 is their watch. Rule 3 is noticing they've climbed a thousand metres and turning back.

### Example

The function below takes `loss_and_grad`, a function returning both the loss and the gradient (Day 33's contract for `grad_check`, so those functions plug straight in). Two Python features appear for the first time in this course:

- `raise FloatingPointError(...)` stops the function and signals an error, like `throw` in Kotlin. `FloatingPointError` is a built-in Python exception class meant for numerical failures.
- `try: ... except FloatingPointError as err: ...` catches that error, like Kotlin's `try { } catch (e: ...) { }`, and `err` holds the exception, whose text `print(err)` shows.

`{L:.3g}` formats a number with 3 significant digits, switching to scientific notation for very large or small values.

```python
def minimize(loss_and_grad, w0, lr, max_steps=10_000, grad_tol=1e-6):
    """Gradient descent with stopping rules. loss_and_grad(w) returns (loss, gradient)."""
    w = np.array(w0, dtype=float)
    losses = []
    for step in range(max_steps):
        L, g = loss_and_grad(w)
        losses.append(L)
        if not np.isfinite(L) or abs(L) > 1e6 * (abs(losses[0]) + 1):   # rule 3: diverging
            raise FloatingPointError(f"loss {L:.3g} at step {step}: lower the learning rate")
        if np.linalg.norm(g) < grad_tol:             # rule 2: the ground is flat enough
            return w, np.array(losses), "converged"
        w = w - lr * g
    return w, np.array(losses), "hit max_steps"      # rule 1: out of budget

def stretched(w):
    return stretched_loss(w), stretched_grad(w)

w_end, losses, why = minimize(stretched, start, lr=0.09)
why, len(losses), w_end
# ('converged', 82, array([-4.17813082e-07, -2.12021648e-08]))
try:
    minimize(stretched, start, lr=0.11)
except FloatingPointError as err:
    print(err)
# loss 4.86e+07 at step 40: lower the learning rate
```

**Underneath, step by step.** Each pass through the loop: compute the loss and gradient at the current `w`; record the loss; check rule 3 (is the loss `inf`/`nan`, or has its size grown more than a million-fold compared with the starting loss? The `abs(...)` and the `+ 1` are explained in the box below); check rule 2 (is the gradient shorter than `grad_tol`?); only then take the step. If the loop ends without returning, the budget ran out (rule 1). The function returns three things: the final `w`, the whole loss history as an array (your loss curve, for free), and a string saying *why* it stopped. Always look at that string: "hit max_steps" means the answer may not be a minimum. On the stretched bowl at 0.09, `len(losses)` is 82 (the start plus 81 steps) by the time the gradient is shorter than \( 10^{-6} \), and at 0.11 the alarm went off after 40 steps, instead of the loss reaching `inf` after 1,938 steps and the weights turning into `nan` after about 3,900.

!!! warning "Common mistake: a divergence alarm that assumes the loss is positive"
    A first version of rule 3 is often `L > 1e6 * losses[0]`. That works for MSE, which is never negative, but some losses can be zero or negative: the two-valley loss above is −0.234 at \( w = -0.9 \). Then `1e6 * losses[0]` is −234,000, every loss is bigger than that, and the alarm fires on step 0 of a perfectly healthy run:

    ```python
    def minimize_naive_alarm(loss_and_grad, w0, lr, max_steps=10_000):
        w = np.array(w0, dtype=float)
        losses = []
        for step in range(max_steps):
            L, g = loss_and_grad(w)
            losses.append(L)
            if L > 1e6 * losses[0]:                  # BUG: assumes losses[0] > 0
                raise FloatingPointError(f"loss {L:.3g} at step {step}: lower the learning rate")
            w = w - lr * g
        return w

    two_valleys = lambda w: ((w ** 2 - 1) ** 2 + 0.3 * w, 4 * w * (w ** 2 - 1) + 0.3)
    try:
        minimize_naive_alarm(two_valleys, -0.9, lr=0.05)
    except FloatingPointError as err:
        print(err)
    # loss -0.234 at step 0: lower the learning rate
    w_tv, L_tv, why_tv = minimize(two_valleys, -0.9, lr=0.05)   # the guarded version
    why_tv, len(L_tv), w_tv
    # ('converged', 26, np.float64(-1.0355786297694902))
    ```

    The fix in `minimize` compares sizes, `abs(L)` against `abs(losses[0])`, and adds 1 so that a starting loss of exactly 0 doesn't make the threshold 0. The recognisable symptom is an alarm on step 0 or 1, before any step could have gone wrong.

### How it goes wrong: "stop when the loss stops improving"

A popular alternative rule is "stop when one step improves the loss by less than some tolerance". It sounds equivalent to rule 2, and on a round bowl it nearly is. On a long, nearly flat valley floor (a big condition number) the loss improves by a tiny amount per step **for thousands of steps** while you are still far from the bottom, so the rule stops early with a mediocre model and a reassuringly flat loss curve. You'll see it happen on real data at the end of the next section. Prefer the gradient-length rule, and when you can, compare against a known answer (as with `lstsq` below).

## Linear regression on the Play Store data

### The model

Now a real model. Predict how many **reviews** an app gets from four features: its installs, its download size, its price, and whether it shows ads. Reviews and installs both span several orders of magnitude (from tens to millions), so take \( \log_{10} \) of both, as on Days 25 and 33: the model then predicts roughly "how many digits the review count has". The model is linear regression (Day 25): \( \hat{y} = X\mathbf{w} \), with one weight per feature plus an intercept, trained to minimise the MSE.

**Why use gradient descent when `lstsq` exists?** Because this is the smallest real model where you can check every number against an exact answer. The loop you write here is, line for line, the loop that trains a neural network; only the gradient function changes.

Load and clean the data the Day 33 way. `(apps["has_ads"] == "Yes").astype(float)` turns the text column into 1.0 for "Yes" and 0.0 for "No", and `.dropna()` drops the rows where `size_mb` is missing:

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
    "price_usd": apps["price_usd"],
    "has_ads": (apps["has_ads"] == "Yes").astype(float),
    "log_reviews": np.log10(apps["reviews"]),
}).dropna()
data.shape                 # (460, 5)
print(data.describe().loc[["mean", "std"]].round(2).to_string())
#       log_installs  size_mb  price_usd  has_ads  log_reviews
# mean          4.44    43.49       0.50     0.63         3.19
# std           0.93    37.24       1.83     0.48         0.95
```

No internet? Download the file from the [Datasets page](../../datasets.md) and pass the local path to `read_csv`. (Tested on pandas 3.0. Remember the dataset is synthetic, so the "findings" describe it, not the real Play Store.)

### Step 1: standardize and build X

Each feature becomes "standard deviations away from its mean" (Day 22), and a column of ones carries the intercept. Keep `mu` and `sd`: a new app must be scaled with the *same* numbers before you can predict for it. (`ddof=1` divides by \( n - 1 \), the sample standard deviation, as Day 33 did.)

```python
features = ["log_installs", "size_mb", "price_usd", "has_ads"]
F = data[features].to_numpy()                       # (460, 4)
mu, sd = F.mean(axis=0), F.std(axis=0, ddof=1)      # remember these: new apps need them too
X = np.column_stack([(F - mu) / sd, np.ones(len(F))])   # standardized features + intercept column
y = data["log_reviews"].to_numpy()
n = len(y)
X.shape, y.shape                                    # ((460, 5), (460,))
```

### Step 2: the loss and its gradient, checked

The loss and gradient come straight from Day 33. Before trusting the gradient, check it: nudge \( \mathbf{w} \) along each axis in turn and take Day 29's central difference of the loss. `np.eye(5)` is the 5×5 identity matrix, so `e[j]` is a vector with a 1 in position `j` and zeros elsewhere: `w_test + t * e[j]` moves only parameter `j`, by `t`.

```python
def mse_and_grad(w):
    r = X @ w - y                                   # residuals
    return r @ r / n, 2 / n * X.T @ r               # Day 33's loss and gradient

def central_diff(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)          # Day 29

w_test = np.array([0.3, -0.2, 0.1, 0.5, 1.0])
e = np.eye(5)                                       # e[j] is the unit step along axis j
numeric = [central_diff(lambda t: mse_and_grad(w_test + t * e[j])[0], 0.0) for j in range(5)]
np.allclose(numeric, mse_and_grad(w_test)[1])       # True: the gradient is right
```

### Step 3: pick a learning rate from the curvatures

You could guess and watch the loss curve, which is what people do with neural networks. For linear regression you can compute the speed limit exactly, as on Day 32's data:

```python
H = 2 / n * X.T @ X
curv = np.linalg.eigvalsh(H)
curv.round(3)                    # array([1.254, 1.757, 2.   , 2.186, 2.785])
2 / curv.max()                   # np.float64(0.7180739824763505)   the speed limit
```

The curvatures are all between 1.25 and 2.8: a nearly round bowl, condition number \( 2.785 / 1.254 = 2.2 \). That's what standardizing bought; you'll see the alternative in a moment. A rate of 0.5 is safely under the limit of 0.72, and close to the balanced choice \( 2/(\lambda_{\min} + \lambda_{\max}) = 2/4.039 = 0.495 \).

### Step 4: train, and compare with the direct answer

`losses_gd[[0, 1, 2, 5, 10, -1]]` picks the loss at steps 0, 1, 2, 5, 10 and the last one (`-1` counts from the end).

```python
w_gd, losses_gd, why = minimize(mse_and_grad, np.zeros(5), lr=0.5)
why, len(losses_gd)                      # ('converged', 16): 16 losses recorded, so 15 steps
losses_gd[[0, 1, 2, 5, 10, -1]].round(5)
# array([11.11021,  0.13404,  0.11665,  0.11491,  0.11491,  0.11491])
w_ls = np.linalg.lstsq(X, y)[0]          # Day 25's direct answer
w_gd.round(4)                            # array([ 0.8896, -0.0103, -0.0221, -0.0323,  3.1945])
w_ls.round(4)                            # array([ 0.8896, -0.0103, -0.0221, -0.0323,  3.1945])
np.abs(w_gd - w_ls).max()                # np.float64(1.315027879783548e-07)
```

Fifteen steps, and gradient descent agrees with least squares to 7 decimal places. The first step does almost all the work (11.1 down to 0.134), because the loudest message from \( \mathbf{w} = \mathbf{0} \) is "raise the intercept to the mean", as Day 33 found. 

**Why not always use `lstsq`, then?** Cost. `lstsq` works with \( X \) as a whole: its cost grows like \( nd^2 + d^3 \) operations for \( n \) examples and \( d \) parameters, and it needs all of \( X \) in memory at once. One gradient step costs about \( nd \) (two matrix-vector products). With \( d = 5 \) both are instant. With \( d = 10^6 \) parameters, \( d^3 = 10^{18} \) operations and a \( d \times d \) matrix of \( 10^{12} \) numbers (8 TB in float64) rule out the direct solve, while a gradient step is still affordable, and Day 52's mini-batches make each step cheaper again. And for every model that isn't linear regression there is no direct solve at all.

The learning rate behaves as the theory says:

```python
for lr in [0.05, 0.5, 0.7, 0.75]:
    try:
        _, L_hist, why = minimize(mse_and_grad, np.zeros(5), lr=lr)
        print(f"lr = {lr:<5} {why:9} after {len(L_hist) - 1:4} steps, loss = {L_hist[-1]:.5f}")
    except FloatingPointError as err:
        print(f"lr = {lr:<5} {err}")
# lr = 0.05  converged after  172 steps, loss = 0.11491
# lr = 0.5   converged after   15 steps, loss = 0.11491
# lr = 0.7   converged after  259 steps, loss = 0.11491
# lr = 0.75  loss 1.31e+07 at step 112: lower the learning rate
```

In the print, `{why:9}` pads the string to width 9 (strings left-align by default) and `{len(L_hist) - 1:4}` right-aligns the number in width 4. 0.7 is slightly under the limit and slow (the steep direction's factor is \( 1 - 0.7 \times 2.785 = -0.95 \), nearly a pure bounce); 0.75 is slightly over it and diverges.

### Reading the model

Dividing a standardized weight by its feature's standard deviation converts it back to raw units (a weight per standard deviation, divided by "units per standard deviation", gives a weight per unit). The fraction of variance explained, \( 1 - \text{MSE}/\text{Var}(y) \), says how much of the spread in \( y \) the model accounts for (Day 25). Here `y.var()` uses NumPy's default `ddof=0`, dividing by \( n \), on purpose: the MSE also divides by \( n \), so the ratio compares like with like. (Standardizing used `ddof=1`; there either choice works, as long as new data is scaled with the same numbers.)

```python
w_gd[0] / sd[0]                    # np.float64(0.9544385417661584)   slope per unit of log10(installs)
1 - losses_gd[-1] / y.var()        # np.float64(0.8731000126289848)   share of the variance explained
```

One more unit of \( \log_{10}(\text{installs}) \), ten times the installs, predicts 0.954 more units of \( \log_{10}(\text{reviews}) \): \( 10^{0.954} = 9.0 \) times the reviews. Size, price and ads barely matter once installs are known. And the model explains 87% of the variance in log reviews, far more than the rating model of Day 33 managed: in this dataset, reviews track installs closely.

**Predicting for a new app.** A trained model is only useful if you can feed it new inputs, and new inputs must go through the *same* scaling as the training data: subtract the training `mu`, divide by the training `sd`, append the 1 for the intercept. `np.append(a, 1.0)` returns a new array with 1.0 added at the end. Take a free app with 100,000 installs (\( \log_{10} = 5 \)), 25 MB and ads:

```python
new_app = np.array([5.0, 25.0, 0.0, 1.0])            # log10(installs), size_mb, price_usd, has_ads
x_new = np.append((new_app - mu) / sd, 1.0)          # scale with the TRAINING mu and sd, add the 1
pred = x_new @ w_gd
pred, 10 ** pred                                     # predicted log10(reviews), and reviews
# (np.float64(3.7150677427583996), np.float64(5188.809694687611))
```

The model expects about \( 10^{3.715} = 5{,}189 \) reviews for such an app.

### What standardizing bought: the same fit on raw features

**What and why.** Standardizing is a change of units, so it can't change the best predictions; it changes the *shape of the bowl* that gradient descent walks on. Skip step 1 and use the raw columns. Same model, same minimum, but look at the curvatures:

```python
X_raw = np.column_stack([F, np.ones(n)])             # the same model, features NOT standardized

def raw_mse_and_grad(w):
    r = X_raw @ w - y
    return r @ r / n, 2 / n * X_raw.T @ r

curv_raw = np.linalg.eigvalsh(2 / n * X_raw.T @ X_raw)
curv_raw.round(3)
# array([7.000000e-02, 4.480000e-01, 6.644000e+00, 1.847900e+01, 6.576581e+03])
2 / curv_raw.max()                                   # np.float64(0.00030410936415322206)   the new speed limit
curv_raw.max() / curv_raw.min(), curv.max() / curv.min()   # condition numbers: raw vs standardized
# (np.float64(94496.4322045903), np.float64(2.220479910231076))

w_raw, losses_raw, why_raw = minimize(raw_mse_and_grad, np.zeros(5), lr=1.8 / curv_raw.max())
why_raw, losses_raw[-1].round(4)                     # ('hit max_steps', np.float64(0.147))
```

**Underneath.** Day 33's rule 1 is \( \nabla_{\mathbf{w}}(\mathbf{w} \cdot \mathbf{x}) = \mathbf{x} \): each weight's gradient is proportional to the input it multiplies. A feature with values around 43 therefore gives its weight a gradient about 43 times larger than a feature with values around 1, and the curvature, which is the gradient's own rate of change, picks up the factor twice: the diagonal entry of \( H \) for `size_mb` is \( \frac{2}{n}\sum_i x_{i,\text{size}}^2 = 2 \times \text{mean}(\text{size}^2) \):

```python
2 * np.mean(F[:, 1] ** 2)                            # np.float64(6550.613913043478)
```

That one entry is most of the largest eigenvalue, 6,577. It forces the learning rate below 0.0003. Along the flattest direction, curvature 0.07, a step at that rate closes only \( 0.000274 \times 0.07 = 0.0019\% \) of the gap. The condition number went from 2.2 to about 94,000. After the full budget of 10,000 steps the loss is still 0.147 instead of 0.115, and `why_raw` honestly says "hit max_steps".

The loss curves side by side. `ax.set_xscale("symlog", linthresh=10)` makes the step axis linear from 0 to 10 and logarithmic after that, so 16 steps and 10,000 steps fit on one plot; the `+ 1e-16` keeps the log scale from meeting an exact zero:

```python
fig, ax = plt.subplots(figsize=(7, 3.8))
best = losses_gd[-1]
ax.semilogy(losses_gd - best + 1e-16, "o-", ms=3, color="#00897b", label="standardized, lr = 0.5")
_, L005, _ = minimize(mse_and_grad, np.zeros(5), lr=0.05)
ax.semilogy(L005 - best + 1e-16, "-", color="#3949ab", label="standardized, lr = 0.05")
ax.semilogy(losses_raw - best, "-", color="#e53935", label="raw features, lr = 0.9 × limit")
ax.set_xscale("symlog", linthresh=10)
ax.set_xlabel("step (log scale after 10)")
ax.set_ylabel("loss above the best (log scale)")
ax.set_ylim(1e-12, 1e2)                  # returns the new limits: (1e-12, 100.0)
ax.grid(alpha=0.3, which="both")
ax.legend(frameon=False, loc="center right")
fig.savefig("day51-playstore-loss.svg", format="svg", bbox_inches="tight")
plt.close(fig)
```

<figure class="diagram" markdown>
![Loss above the best possible value, on a log scale, against step number (log scale after step 10) for the Play Store regression: standardized features with lr = 0.5 in teal plunge below 10 to the minus 12 within about 14 steps; standardized with lr = 0.05 in blue reaches the same level by about 170 steps; raw unstandardized features in red, at 0.9 times their much smaller limit, fall to about 0.03 above the best and are still there after 10,000 steps](../../assets/img/phase-1/day51-playstore-loss.svg)
<figcaption>The same model and the same data. Standardized features: a round bowl, solved in 15 steps. Raw features: a bowl stretched 94,000 to 1, still unsolved after 10,000 steps.</figcaption>
</figure>

This is the practical reason behind "always standardize your features" (Day 22) and the reason neural networks use normalization layers: **gradient descent is fast on round bowls and hopeless on stretched ones.** Day 52 adds momentum and Adam, which help on stretched bowls, but making the bowl round in the first place is cheaper.

### How it goes wrong: the stopping-rule trap, on real data

Back to the rule "stop when the loss stops improving". `losses_raw[:-1] - losses_raw[1:]` subtracts each loss from the one before it, giving the improvement made by each step. `drops < 1e-4` is a boolean array, and `np.argmax` of a boolean array returns the index of the first `True` (careful: it also returns 0 if there is no `True` at all, so check that one exists).

```python
drops = losses_raw[:-1] - losses_raw[1:]       # how much each step improved the loss
(drops < 1e-4).any()                           # np.True_: the rule does fire somewhere
k_stop = np.argmax(drops < 1e-4)               # the first step that improved it by less than 1e-4
k_stop, losses_raw[k_stop].round(4), losses_gd[-1].round(4)
# (np.int64(622), np.float64(0.1916), np.float64(0.1149))
np.linalg.norm(raw_mse_and_grad(w_raw)[1])     # np.float64(0.07931644101485624)  even after 10,000 steps
```

The rule fires at step 622 (the step from 622 to 623 is the first to improve by less than \( 10^{-4} \)), when the loss is 0.192, two-thirds worse than the best 0.115: each step was still helping, only slowly. The gradient length at the end of the full run (0.079) is nowhere near a tolerance like \( 10^{-6} \), so the gradient rule correctly refuses to declare victory.

!!! warning "Common mistake: trusting a flat loss curve"
    A loss curve that has gone flat means "the steps stopped helping much", which can be a minimum **or** a long, flat valley floor. Check the gradient length, compare against a known answer when one exists, and fix the conditioning (standardize) rather than waiting longer.

!!! warning "Common mistake: standardizing with statistics from the wrong data"
    `mu` and `sd` are part of the model. Predictions for a new app must use `(f_new - mu) / sd` with the **training** `mu` and `sd`, never statistics recomputed on the new data. And when you split data into training and test sets (Day 55's milestone, and Phase 2), compute `mu` and `sd` on the training rows only; otherwise information about the test set leaks into training.

!!! android "Android bridge: a training loop that runs on the phone"
    Most models are trained in Python and only *run* on the device, but small models can learn on the phone, for example a ranking model that adapts to one user's taps. LiteRT (TensorFlow Lite) supports this with a model exported with a `train` signature, called through `Interpreter.runSignature(inputs, outputs, "train")`. For a linear model you can write the whole loop yourself. Here is today's `minimize` in Kotlin, using Day 33's gradient:

    ```kotlin
    import kotlin.math.sqrt

    /** Linear regression by gradient descent. x is n-by-d, row-major: X[i][j] = x[i * d + j].
     *  Features must already be standardized, with a column of ones for the intercept. */
    fun fitLinear(x: FloatArray, y: FloatArray, n: Int, d: Int,
                  lr: Float = 0.5f, maxSteps: Int = 1000, gradTol: Float = 1e-4f): FloatArray {
        val w = FloatArray(d)
        val grad = FloatArray(d)
        for (step in 0 until maxSteps) {                       // rule 1: budget
            grad.fill(0f)
            var loss = 0f
            for (i in 0 until n) {
                var pred = 0f
                for (j in 0 until d) pred += x[i * d + j] * w[j]
                val r = pred - y[i]
                loss += r * r / n
                for (j in 0 until d) grad[j] += 2f / n * r * x[i * d + j]
            }
            check(loss.isFinite()) { "diverged at step $step: lower the learning rate" }   // rule 3
            if (sqrt(grad.sumOf { (it * it).toDouble() }) < gradTol) break                // rule 2
            for (j in 0 until d) w[j] -= lr * grad[j]          // the update
        }
        return w
    }
    ```

    Three device-specific notes. The tolerance is `1e-4f` rather than today's `1e-6`, because `Float` keeps only about 7 significant digits, so a gradient computed in float32 can't reliably get much smaller than that. `check(...)` throws an `IllegalStateException` with that message, the Kotlin twin of today's `raise FloatingPointError`. And run it off the main thread (a coroutine on `Dispatchers.Default`, or WorkManager for a longer job while the phone is charging): even 1,000 passes over a few thousand examples is too much work for a 16 ms frame.

## Going further

These are real and useful, but beyond today. Days 52–55 pick up the first three.

- **Mini-batch and stochastic gradient descent:** compute the gradient on a small random batch instead of all the data. Day 52. ([Dive into Deep Learning: SGD](https://d2l.ai/chapter_optimization/sgd.html))
- **Momentum and Adam:** remember previous steps to roll through stretched valleys. Day 52. ([Distill: Why Momentum Really Works](https://distill.pub/2017/momentum/))
- **Learning-rate schedules:** shrink \( \eta \) as training goes on. Day 52. ([PyTorch docs: torch.optim](https://pytorch.org/docs/stable/optim.html))
- **Convexity and convergence guarantees:** when one minimum is guaranteed, and saddle points. Day 53. ([Dive into Deep Learning: Convexity](https://d2l.ai/chapter_optimization/convexity.html))
- **Line search (backtracking, the Armijo rule):** choose the step size at every step by trying a big one and halving until the loss drops enough. Exercise 8 builds a simple version. ([Boyd and Vandenberghe, Convex Optimization, free book](https://web.stanford.edu/~boyd/cvxbook/), section 9.2)
- **L-smooth functions:** the general version of today's speed limit. If the gradient never changes faster than \( L \) times the distance moved (the largest curvature anywhere is at most \( L \)), gradient descent with \( \eta < 2/L \) is guaranteed to decrease the loss. (Here \( L \) is a curvature bound, not the loss; the clash of letters is standard, sorry.) ([Dive into Deep Learning: Gradient Descent](https://d2l.ai/chapter_optimization/gd.html) and [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/), section 9.1)
- **Newton's method:** divide the gradient by the curvature (the Hessian) to jump straight to the bottom of a bowl; too expensive for big networks. ([Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/), section 9.5)
- **Gradient clipping:** cap the gradient's length to stop single huge steps, common in training recurrent networks and transformers. ([PyTorch docs: clip_grad_norm_](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html))
- **Early stopping on a validation set:** stop when the loss on held-out data starts rising, to prevent overfitting. Phase 2.

---

## Exercises

**Warm-ups**

1. By hand, run three steps of gradient descent on \( L(w) = 2(w + 1)^2 \) from \( w_0 = 1 \) with \( \eta = 0.1 \). What is the gap factor? What is the largest learning rate that converges, and which one reaches the minimum in one step? Check with code, including a rate slightly above and slightly below your limit.
2. The bowl \( L(\mathbf{w}) = 3w_1^2 + 0.5\,w_2^2 \) has gradient \( [6w_1, w_2] \). Write down its two curvatures, its condition number, its learning-rate limit and its balanced learning rate. Then run 500 steps from \( (1, 1) \) at \( \eta = 0.33 \) and \( \eta = 0.34 \) and report the final loss of each.

**Core**

3. Confirm Day 32's threshold with code: build \( H \) for Day 32's four points, compute \( 2/\lambda_{\max} \), then run 1,000 steps on Day 32's line-fit loss from \( (0, 0) \) with \( \eta = 0.119 \) and with \( \eta = 0.1205 \). Report the final parameters and loss of each.
4. Add `min_android` as a fifth feature to the Play Store regression (standardized like the others) and fit it with gradient descent at \( \eta = 0.5 \). Does it converge as fast? Compare with `lstsq`. What weight does `min_android` get, and what does that tell you?
5. On the stretched bowl \( L = w_1^2 + 10w_2^2 \), show that \( \eta = 1/11 \) makes both coordinates shrink by the same factor every step, and find that factor in theory and in code. Then compare the number of steps to reach a loss below \( 10^{-12} \) from \( (-4, 1.5) \) at \( \eta = 0.08 \), \( 1/11 \) and \( 0.095 \).

**Predict the output**

6. Before running it, write down the four lines this prints. Then run it and explain.

    ```python
    w = 0.0
    for step in range(4):
        w = w - 1.0 * 2 * (w - 3)        # gradient descent on (w - 3)² + 1 with lr = 1.0
        print(step + 1, w, (w - 3) ** 2 + 1)
    ```

**Find and fix the bug**

7. This training loop for a small linear regression diverges at \( \eta = 0.5 \), although a correct MSE gradient on standardized data would converge comfortably there. Find the bug, explain why it causes divergence (in terms of curvature), and fix it.

    ```python
    import numpy as np

    rng = np.random.default_rng(51)
    n = 200
    Xb = np.column_stack([rng.normal(size=(n, 2)), np.ones(n)])   # 2 standardized-ish features + ones
    yb = Xb @ np.array([2.0, -1.0, 0.5]) + rng.normal(scale=0.1, size=n)

    def mse_and_grad_buggy(w):
        r = Xb @ w - yb
        return r @ r / n, 2 * Xb.T @ r

    w = np.zeros(3)
    for step in range(30):
        L, g = mse_and_grad_buggy(w)
        w = w - 0.5 * g
    print(step, L)
    ```

**Challenge**

8. **Backtracking line search.** Instead of a fixed learning rate, choose it at every step: start with \( \eta = 1 \); while the step would not lower the loss by at least \( \tfrac{1}{2}\eta\|\nabla L\|^2 \), halve \( \eta \) and try again; then take the step. (This is the Armijo rule with constant \( \tfrac{1}{2} \).) Implement `minimize_backtracking(loss_and_grad, w0, grad_tol=1e-6, max_steps=10_000)`, run it on the stretched bowl from \( (-4, 1.5) \) and on Day 32's line-fit loss from \( (0, 0) \), and compare the number of steps with a fixed learning rate. What does it cost per step that a fixed rate doesn't?

??? success "Solutions"
    **1.** \( L'(w) = 4(w + 1) \). \( w_1 = 1 - 0.1 \times 8 = 0.2 \), \( w_2 = 0.2 - 0.1 \times 4.8 = -0.28 \), \( w_3 = -0.28 - 0.1 \times 2.88 = -0.568 \). The curvature is \( L'' = 4 \), so the gap to \( -1 \) shrinks by \( 1 - 0.1 \times 4 = 0.6 \) per step. It converges for \( \eta < 2/4 = 0.5 \) and lands in one step at \( \eta = 1/4 \).

    ```python
    import numpy as np

    def gd_1d(grad, w0, lr, steps):
        path = [w0]
        for _ in range(steps):
            path.append(path[-1] - lr * grad(path[-1]))
        return np.array(path)

    g1 = lambda w: 4 * (w + 1)
    p1 = gd_1d(g1, 1.0, lr=0.1, steps=3)
    p1.round(4)                                    # array([ 1.   ,  0.2  , -0.28 , -0.568])
    ((p1[1:] + 1) / (p1[:-1] + 1)).round(4)        # array([0.6, 0.6, 0.6])
    for lr in [0.25, 0.49, 0.51]:
        print(lr, gd_1d(g1, 1.0, lr, steps=50)[-1].round(4))
    # 0.25 -1.0
    # 0.49 -0.7402
    # 0.51 13.2134
    ```

    At 0.49 the factor is \( 1 - 0.49 \times 4 = -0.96 \): it converges, but after 50 steps the gap is \( 2 \times 0.96^{50} = 0.26 \), so it is still bouncing. At 0.51 the factor is \( -1.04 \) and the gap grows by 4% per step. The limit \( 2/L'' \) is sharp.

    **2.** The curvatures are the second derivatives along each axis: \( 6 \) for \( w_1 \) and \( 1 \) for \( w_2 \). Condition number \( 6/1 = 6 \). Limit \( 2/6 = 0.333 \). Balanced rate \( 2/(6 + 1) = 0.286 \), with factor \( (6 - 1)/(6 + 1) = 0.714 \).

    ```python
    import numpy as np

    L_b = lambda w: 3 * w[0] ** 2 + 0.5 * w[1] ** 2
    g_b = lambda w: np.array([6 * w[0], w[1]])
    for lr in [0.33, 0.34]:
        w = np.array([1.0, 1.0])
        for _ in range(500):
            w = w - lr * g_b(w)
        print(lr, L_b(w))
    # 0.33 5.0489020716483684e-09
    # 0.34 3.2393699825001274e+17
    ```

    At 0.33 the steep factor is \( 1 - 6 \times 0.33 = -0.98 \), so it converges, slowly bouncing; at 0.34 it is \( -1.04 \), and 500 steps of 4% growth is a factor of \( 1.04^{500} \), about \( 3 \times 10^8 \), in \( w_1 \), which the loss squares. The steep direction decides, even though the flat one (\( w_2 \)) converges at both rates.

    **3.**

    ```python
    import numpy as np

    x_data = np.array([1., 2., 3., 4.])
    y_data = np.array([2., 3., 5., 6.])
    X32 = np.column_stack([x_data, np.ones(4)])
    H = 2 / 4 * X32.T @ X32
    2 / np.linalg.eigvalsh(H).max()                  # np.float64(0.11975610662865498)

    for lr in [0.119, 0.1205]:
        w = np.zeros(2)
        for _ in range(1000):
            w = w - lr * 2 / 4 * X32.T @ (X32 @ w - y_data)
        r = X32 @ w - y_data
        print(lr, w.round(3), (r @ r / 4).round(4))
    # 0.119 [1.4 0.5] 0.05
    # 0.1205 [-324025.733 -110208.239] 978150949305.7365
    ```

    0.119 is slightly under \( 2/\lambda_{\max} = 0.11976 \) and reaches Day 25's line \( (1.4, 0.5) \) with the best loss 0.05. 0.1205 is slightly over: the steep direction's factor is \( 1 - 0.1205 \times 16.7006 = -1.0124 \), so its gap grows by 1.24% per step, and after 1,000 steps that is a factor of \( 1.0124^{1000} \), about \( 2 \times 10^5 \).

    **4.**

    ```python
    import numpy as np
    import pandas as pd

    BASE = "https://manjulkadyan.github.io/ai-leaning/assets/data/"
    apps4 = pd.read_csv(BASE + "play_store_apps.csv")
    apps4 = apps4.drop_duplicates().assign(
        installs=lambda d: d["installs"].str.replace(",", "").str.rstrip("+").astype(int),
        size_mb=lambda d: d["size_mb"].replace("Varies with device", np.nan).astype(float),
    )
    data4 = pd.DataFrame({
        "log_installs": np.log10(apps4["installs"]),
        "size_mb": apps4["size_mb"],
        "price_usd": apps4["price_usd"],
        "has_ads": (apps4["has_ads"] == "Yes").astype(float),
        "min_android": apps4["min_android"],
        "log_reviews": np.log10(apps4["reviews"]),
    }).dropna()
    F4 = data4.drop(columns="log_reviews").to_numpy()        # 5 features
    X4 = np.column_stack([(F4 - F4.mean(axis=0)) / F4.std(axis=0, ddof=1), np.ones(len(F4))])
    y4 = data4["log_reviews"].to_numpy()
    n4 = len(y4)

    w4 = np.zeros(6)
    for step in range(10_000):
        g = 2 / n4 * X4.T @ (X4 @ w4 - y4)
        if np.linalg.norm(g) < 1e-6:
            break
        w4 = w4 - 0.5 * g
    r4 = X4 @ w4 - y4
    step, (r4 @ r4 / n4).round(5)                   # (15, np.float64(0.11491))
    w4.round(4)
    # array([ 8.8970e-01, -1.0300e-02, -2.2100e-02, -3.2300e-02,  8.0000e-04,
    #         3.1945e+00])
    np.abs(w4 - np.linalg.lstsq(X4, y4)[0]).max()   # np.float64(2.3171261260418463e-07)
    ```

    It stops after 15 steps, the same as the four-feature model, because a standardized extra feature keeps the bowl nearly round. `min_android` (the fifth weight) gets 0.0008 standard deviations of log reviews, essentially zero, and the loss doesn't move: once installs are known, the minimum Android version tells you nothing more about reviews in this dataset. Gradient descent agrees with `lstsq` to 7 decimal places again.

    **5.** Balance the factors: \( 1 - 2\eta = -(1 - 20\eta) \) gives \( 22\eta = 2 \), so \( \eta = 1/11 \). Then \( 1 - 2/11 = 9/11 = 0.818 \) along \( w_1 \) and \( 1 - 20/11 = -9/11 \) along \( w_2 \): the same size, opposite signs.

    ```python
    import numpy as np

    g5 = lambda w: np.array([2 * w[0], 20 * w[1]])
    L5 = lambda w: w[0] ** 2 + 10 * w[1] ** 2

    w = np.array([-4.0, 1.5])
    for _ in range(3):
        w_next = w - (1 / 11) * g5(w)
        print((w_next / w).round(4))             # each coordinate's ratio for one step
        w = w_next
    # [ 0.8182 -0.8182]
    # [ 0.8182 -0.8182]
    # [ 0.8182 -0.8182]

    for lr in [0.08, 1 / 11, 0.095]:
        w, k = np.array([-4.0, 1.5]), 0
        while L5(w) >= 1e-12:
            w, k = w - lr * g5(w), k + 1
        print(round(lr, 4), k)
    # 0.08 88
    # 0.0909 78
    # 0.095 146
    ```

    \( 1/11 \) is the fastest of the three: below it, the flat direction's factor \( 1 - 2\eta \) is closer to 1; above it, the steep direction's factor \( 1 - 20\eta \) is closer to −1 and bounces longer.

    **6.** Prediction: the factor is \( 1 - 2 \times 1.0 = -1 \), so the gap \( -3 \) becomes \( +3, -3, +3, -3 \): \( w \) alternates between 6 and 0 forever, and the loss stays at 10.

    ```python
    w = 0.0
    for step in range(4):
        w = w - 1.0 * 2 * (w - 3)        # gradient descent on (w - 3)² + 1 with lr = 1.0
        print(step + 1, w, (w - 3) ** 2 + 1)
    # 1 6.0 10.0
    # 2 0.0 10.0
    # 3 6.0 10.0
    # 4 0.0 10.0
    ```

    \( \eta = 1 \) is exactly the limit \( 2/L'' \): every step jumps from one wall to the same height on the opposite wall. It neither converges nor diverges, which is why the convergence condition is a strict "less than".

    **7.** The gradient is missing the division by \( n \): `2 * Xb.T @ r` instead of `2 / n * Xb.T @ r`. That makes it the gradient of the *sum* of squared errors, which is \( n = 200 \) times bigger. A gradient 200 times bigger means a curvature 200 times bigger, so the safe learning rate is 200 times smaller (the correct curvatures, printed below, go up to 2.477, so the buggy limit is \( 2/(200 \times 2.477) = 0.004 \)). At 0.5 it is more than 100 times over the limit, and after 30 steps the loss has reached \( 2.3 \times 10^{139} \).

    ```python
    import numpy as np

    rng = np.random.default_rng(51)
    n = 200
    Xb = np.column_stack([rng.normal(size=(n, 2)), np.ones(n)])
    yb = Xb @ np.array([2.0, -1.0, 0.5]) + rng.normal(scale=0.1, size=n)

    def mse_and_grad_buggy(w):
        r = Xb @ w - yb
        return r @ r / n, 2 * Xb.T @ r                # BUG: missing / n

    w = np.zeros(3)
    for step in range(30):
        L, g = mse_and_grad_buggy(w)
        w = w - 0.5 * g
    print(step, L)
    # 29 2.2939198888030636e+139

    def mse_and_grad_fixed(w):
        r = Xb @ w - yb
        return r @ r / n, 2 / n * Xb.T @ r            # the mean's gradient

    w = np.zeros(3)
    for step in range(30):
        L, g = mse_and_grad_fixed(w)
        w = w - 0.5 * g
    print(step, L, w.round(3))
    # 29 0.011395896217861745 [ 1.988 -1.002  0.511]
    np.linalg.eigvalsh(2 / n * Xb.T @ Xb).round(3)  # array([1.965, 2.011, 2.477])   the correct curvatures
    ```

    After the fix the weights land on the true \( [2, -1, 0.5] \) up to the noise, and the loss settles at the noise level (0.1² = 0.01). A gradient that is off by a constant factor is invisible in a quick look at its direction, and shows up as a learning rate that "has to be" tiny; Day 33's `grad_check` would catch it as a relative error near 1.

    **8.**

    ```python
    import numpy as np

    def minimize_backtracking(loss_and_grad, w0, grad_tol=1e-6, max_steps=10_000):
        w = np.array(w0, dtype=float)
        evals = 0                                        # how many times we computed the loss
        for step in range(max_steps):
            L, g = loss_and_grad(w)
            evals += 1
            gg = g @ g                                   # squared gradient length
            if np.sqrt(gg) < grad_tol:
                return w, step, evals
            lr = 1.0
            while loss_and_grad(w - lr * g)[0] > L - 0.5 * lr * gg:
                lr /= 2                                  # too big: halve and try again
                evals += 1
            evals += 1                                   # the trial that was accepted
            w = w - lr * g
        return w, max_steps, evals

    def fixed_lr(loss_and_grad, w0, lr, grad_tol=1e-6, max_steps=10_000):
        w = np.array(w0, dtype=float)
        for step in range(max_steps):
            L, g = loss_and_grad(w)
            if np.linalg.norm(g) < grad_tol:
                return w, step
            w = w - lr * g
        return w, max_steps

    def stretched8(w):
        return w[0] ** 2 + 10 * w[1] ** 2, np.array([2 * w[0], 20 * w[1]])

    x8 = np.array([1., 2., 3., 4.])
    y8 = np.array([2., 3., 5., 6.])
    X8 = np.column_stack([x8, np.ones(4)])

    def line8(w):
        r = X8 @ w - y8
        return r @ r / 4, 2 / 4 * X8.T @ r

    for name, f, w0, lr in [("stretched", stretched8, [-4.0, 1.5], 1 / 11),
                            ("day 32 line", line8, [0.0, 0.0], 0.1)]:
        w_bt, steps_bt, evals_bt = minimize_backtracking(f, w0)
        w_fx, steps_fx = fixed_lr(f, w0, lr)
        print(f"{name:12} backtracking: {steps_bt} steps, {evals_bt} loss evaluations;"
              f"  fixed lr {lr:.4f}: {steps_fx} steps;  w = {w_bt.round(4)}")
    # stretched    backtracking: 55 steps, 307 loss evaluations;  fixed lr 0.0909: 86 steps;  w = [-0.  0.]
    # day 32 line  backtracking: 59 steps, 269 loss evaluations;  fixed lr 0.1000: 291 steps;  w = [1.4 0.5]
    ```

    Line by line: each step starts optimistic with \( \eta = 1 \), and the `while` loop halves it until the new loss is at least \( \tfrac{1}{2}\eta\|\nabla L\|^2 \) below the old one. That target comes from the "why it works" argument at the start of the lesson: a small step of \( -\eta\nabla L \) should lower the loss by about \( \eta\|\nabla L\|^2 \), and asking for half of that rejects steps that overshoot. Because the step size adapts to the local curvature, it never diverges and you never pick \( \eta \).

    It needs fewer steps on both losses (55 against 86, and 59 against 291), but each step costs extra loss evaluations for the trials: 307 evaluations on the stretched bowl, where a fixed, well-chosen rate needs 87 loss-and-gradient computations in total (86 steps plus the final check). That trade is why line search is popular for small, full-batch problems and rare in deep learning, where one loss evaluation means a pass over a batch of data and the loss is noisy from batch to batch (Day 52). The `w = [-0.  0.]` is a tiny negative number rounded to zero; NumPy keeps its sign.

## Resources

- [3Blue1Brown: Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w): the walk downhill on a real network's loss, animated, and what the gradient's entries mean.
- [Dive into Deep Learning: Gradient Descent](https://d2l.ai/chapter_optimization/gd.html): one-dimensional and multivariate gradient descent, learning rates and divergence, with code.
- [CS231n: Optimization (stochastic gradient descent)](https://cs231n.github.io/optimization-1/): loss landscapes, the numerical vs analytic gradient, and the update loop, from Stanford's course notes.
- [Distill: Why Momentum Really Works (Gabriel Goh)](https://distill.pub/2017/momentum/): interactive figures of gradient descent on stretched bowls and the eigenvalue analysis from today, extended to momentum (Day 52).
- [Mathematics for Machine Learning (free book)](https://mml-book.github.io/): chapter 7, "Continuous Optimization", covers gradient descent, step sizes and the condition number.
- [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course): the linear-regression module covers gradient descent, learning rates and loss curves with interactive exercises.
- [NumPy docs: `numpy.linalg.eigvalsh`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvalsh.html) and [`numpy.linalg.lstsq`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html): the eigenvalues of a symmetric matrix, and the direct solver you compared against.

## Checklist

- [ ] I can write \( \mathbf{w} \leftarrow \mathbf{w} - \eta\nabla L(\mathbf{w}) \), explain each symbol, and say why there is a minus sign.
- [ ] I ran gradient descent by hand for a few steps and saw the gap shrink by a constant factor.
- [ ] I can explain too small, about right, too large and diverging learning rates with the factor \( 1 - \eta L'' \), and compute the limit \( 2/L'' \).
- [ ] I can recognise divergence from `inf`, `nan` and NumPy's overflow and invalid-value warnings, and I know the first fix to try.
- [ ] I can explain why a stretched bowl zig-zags, what the condition number is, and why the curvatures are the eigenvalues of \( \frac{2}{n}X^\top X \).
- [ ] I saw gradient descent land in different minima from different starts, in code and in the playground.
- [ ] I wrote `minimize` with a step budget, a gradient-length rule and a divergence alarm, and I know why "the loss stopped improving" is a weak rule.
- [ ] I fitted the Play Store regression with gradient descent and matched `lstsq`.
- [ ] I saw raw features make the same fit thousands of times slower, and I can explain why standardizing fixes it.

**Next:** [Day 52: Learning rates, mini-batches, SGD and momentum](day-52-sgd-momentum.md)
