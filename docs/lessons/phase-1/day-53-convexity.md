# Day 53: Convexity, Local Minima and Saddle Points

**Phase 1 · Week 8 · ~150 minutes**

## Objectives

- [ ] Say what a convex set and a convex function are, and test a function with the "chord above the curve" check, by picture and in code.
- [ ] Use the second derivative as curvature: \( f''(x) \ge 0 \) everywhere means convex, and compute it numerically.
- [ ] Explain why a convex loss has no bad valleys: any point where the slope is zero is a global minimum.
- [ ] Build the Hessian (the matrix of second derivatives) and read convexity, minima and saddles off its eigenvalues.
- [ ] Show that the losses of linear regression (MSE) and logistic regression (cross-entropy) are convex, on the Play Store data.
- [ ] Tell local and global minima, saddle points and plateaus apart, and predict how gradient descent behaves near each.
- [ ] Explain, briefly and honestly, why deep learning works even though its losses are not convex.

**Prerequisites.** [Day 22](day-22-norms-distances.md) (the L1/L2 unit balls and L2 regularization), [Day 23](day-23-rank-span.md) (collinear features and a singular \( X^\top X \)), [Day 24](day-24-eigenvectors.md) (eigenvalues, `eigh`, symmetric matrices), [Day 29](day-29-derivatives.md) and [Day 30](day-30-derivative-rules.md) (derivatives, `central_diff`, the sigmoid), [Day 31](day-31-chain-rule.md) (the chain rule, used to explain plateaus and to derive the logistic Hessian), [Day 32](day-32-partial-derivatives-gradient.md) (contour maps and the gradient), [Day 33](day-33-vector-matrix-gradients.md) (the MSE gradient in matrix form, on the Play Store data), [Day 47](day-47-maximum-likelihood.md) (maximum likelihood: why cross-entropy is the classification loss, and why maximizing a concave log-likelihood is a convex problem), and this week's [Day 50](day-50-loss-functions.md) (loss functions), [Day 51](day-51-gradient-descent.md) (gradient descent) and [Day 52](day-52-sgd-momentum.md) (SGD and momentum).

---

## The big idea

Gradient descent (Day 51) only ever looks at the ground under its feet. It feels the slope, takes a step downhill, and repeats. It has no map. So the obvious worry is: **what if it walks into the wrong valley?** What if it stops somewhere that only *looks* like the bottom?

The answer depends entirely on the **shape** of the loss landscape.

**Analogy.** Drop a marble into a salad bowl. Wherever you drop it, it rolls to the same single lowest point. Now drop it onto an egg carton: it settles into whichever cup is nearest, and most cups are not the lowest one. Now put it on a horse saddle, right in the middle: it can balance there for a moment (the surface is flat at that point), but the slightest nudge sideways sends it sliding off.

- The salad bowl is a **convex** function: one valley, and gradient descent always finds its bottom.
- The egg carton has **local minima**: bottoms of small valleys that are not the lowest point overall.
- The saddle has a **saddle point**: flat, but downhill in some directions and uphill in others.
- Add one more: a huge, almost flat field, a **plateau**, where the slope is so small that the marble barely moves.

| Landscape | Where GD ends up | Where you meet it in ML |
|---|---|---|
| Convex bowl | The global minimum, from any start | Linear regression (MSE), logistic regression (cross-entropy) |
| Several valleys | The valley you started in | Neural networks, k-means clustering |
| Saddle point | Slows to a crawl, then escapes | Everywhere in high-dimensional neural network losses |
| Plateau | Crawls for a long time | Saturated sigmoids, bad initialization |

Today you'll learn to tell which landscape you are in, with two tools: the **chord test** and the **second derivative** (curvature).

## Convex sets

Start with shapes, because the definition of a convex function is built on them. A set of points (a region of the plane, or of any \( n \)-dimensional space) is **convex** if, for **any** two points in it, the whole straight segment between them is also in it. No dents, no holes, no arms.

<figure class="diagram" markdown>
![Four shapes side by side: a teal disk labelled L2 ball and a teal diamond labelled L1 ball, each with a green segment between two of its points lying fully inside, marked convex; then an orange crescent with a red vertical segment joining its two tips that crosses the empty bite, and an orange four-pointed star labelled L one-half ball with a red segment between two of its arms that leaves the shape, both marked not convex](../../assets/img/phase-1/day53-convex-sets.svg)
<figcaption>A set is convex when every segment between two of its points stays inside. One escaping segment is enough to fail.</figcaption>
</figure>

You've met two of these before. On Day 22, the "unit ball" of a norm was the set of all vectors with norm at most 1. The L2 ball (a disk) and the L1 ball (a diamond) are both convex; so is the L∞ ball (a square). If you plug \( p = \tfrac12 \) into the norm formula you get a pinched star that is **not** convex, which is one reason \( p < 1 \) isn't called a norm.

Why care about sets? Two reasons. First, a convex function is defined by a convex shape (next section). Second, when training has **constraints** ("keep every weight between −1 and 1", "keep the weight vector inside the L2 ball"), the allowed region is a convex set, and convex constraints keep the problem easy: if you stand at two allowed settings, every setting in between is allowed too, so a step from one towards the other never leaves the allowed region.

**Intuition first.** Imagine the set is a room and you are holding a laser pointer. The room is convex if, from any spot in it, you can shine the laser at any other spot without the beam hitting a wall. An L-shaped room fails: from one arm you can't see into the other.

**The segment between two points.** To test this in code you need the points of the segment from \( \mathbf{p} \) to \( \mathbf{q} \). Every one of them can be written with a single number \( t \) between 0 and 1:

\[
(1 - t)\,\mathbf{p} + t\,\mathbf{q}
\]

At \( t = 0 \) this is \( \mathbf{p} \), at \( t = 1 \) it is \( \mathbf{q} \), and at \( t = 0.5 \) it is the midpoint. By hand, with \( \mathbf{p} = [-0.6, 0.7] \), \( \mathbf{q} = [0.8, -0.5] \) and \( t = 0.5 \): \( 0.5 \cdot [-0.6, 0.7] + 0.5 \cdot [0.8, -0.5] = [0.1, 0.1] \). Its length is \( \sqrt{0.01 + 0.01} \approx 0.14 \le 1 \), so the midpoint is inside the unit disk. The code walks 101 such points along the segment and asks a `contains` function about each one:

```python
import numpy as np

def in_unit_disk(p):
    return bool(np.linalg.norm(p) <= 1)          # the L2 ball of Day 22

def in_two_disks(p):
    """Inside a disk of radius 1 around (-1.5, 0) OR one around (1.5, 0)."""
    p = np.asarray(p, dtype=float)
    return bool(np.linalg.norm(p - [-1.5, 0]) <= 1 or np.linalg.norm(p - [1.5, 0]) <= 1)

def segment_stays_inside(contains, p, q, steps=101):
    """Walk from p to q in small steps; report the first point that falls outside."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    for t in np.linspace(0, 1, steps):           # t = 0.00, 0.01, ..., 1.00
        point = (1 - t) * p + t * q
        if not contains(point):
            return False, point.round(3)
    return True, None

print(segment_stays_inside(in_unit_disk, [-0.6, 0.7], [0.8, -0.5]))
print(in_two_disks([-1.5, 0]), in_two_disks([1.5, 0]))
print(segment_stays_inside(in_two_disks, [-1.5, 0], [1.5, 0]))
# (True, None)
# True True
# (False, array([-0.48,  0.  ]))
```

Line by line: `np.linalg.norm(p)` is the L2 length of Day 22; `bool(...)` turns NumPy's `np.True_` into a plain Python `True` so the printout is clean; `np.linspace(0, 1, steps)` makes 101 evenly spaced values of \( t \); the function returns a pair (a tuple), `(True, None)` when every point was inside, or `(False, the first outside point)`.

**Underneath.** A single segment test can only *disprove* convexity, by finding one escaping point. Proving a set convex needs an argument for **every** pair of points (for the disk: the length of a mix is at most the mix of the lengths, by the triangle inequality of Day 22, so a mix of two points of length \( \le 1 \) has length \( \le 1 \)). Code checks many pairs; maths covers all of them.

!!! warning "Common mistake: assuming that gluing convex sets together keeps them convex"
    Each disk on its own is convex, but their **union** (points in one disk *or* the other) is not: the segment between the two centres leaves the shape at \( [-0.48, 0] \), the first point past the left disk's edge, as the output above shows. The rule that does hold is for the **intersection** (points in both sets at once): if a segment stays in each set, it stays in their overlap. That's why stacking several convex constraints ("inside the L2 ball **and** every weight \( \ge 0 \)") still gives a convex region, while allowing "this region **or** that one" usually doesn't.

## Convex functions: the chord test

Take a function \( f \) of one variable and pick any two points on its graph, at \( x = a \) and \( x = b \). Join them with a straight line: a **chord**. The function is **convex** if every such chord lies **on or above** the curve between \( a \) and \( b \).

To write that down we need a way to name "a point between \( a \) and \( b \)". Use a number \( t \) between 0 and 1:

\[
x_t = (1 - t)\,a + t\,b
\]

- \( t \) is "how far along" you are: \( t = 0 \) gives \( a \), \( t = 1 \) gives \( b \), \( t = 0.5 \) gives the midpoint.
- The chord's height at that point mixes the two end heights with the same weights: \( (1 - t)\,f(a) + t\,f(b) \).
- The curve's height there is \( f(x_t) \).

So \( f \) is convex when, for every \( a \), every \( b \) and every \( t \) between 0 and 1:

\[
f\big((1 - t)\,a + t\,b\big) \;\le\; (1 - t)\,f(a) + t\,f(b)
\]

In words: **the function at a mix of two points is at most the same mix of the function values.** (Equivalently: colour in the whole region on and above the graph, the "inside of the bowl". The function is convex exactly when that region is a convex set in the sense of the previous section; a hump in the graph is a dent in the region. This region is called the **epigraph**.)

**Worked example.** \( f(x) = x^2 \), \( a = -1 \), \( b = 3 \), \( t = 0.5 \). The midpoint is \( x = 0.5 \cdot (-1) + 0.5 \cdot 3 = 1 \), where the curve is at \( f(1) = 1 \). The chord there is at \( 0.5 \cdot f(-1) + 0.5 \cdot f(3) = 0.5 \cdot 1 + 0.5 \cdot 9 = 5 \). The chord is 4 above the curve: the test passes for this chord.

Now a function with two valleys that we'll use all lesson, \( f(x) = x^4 - 3x^2 + x \). Take \( a = -1.3 \) and \( b = 1.1 \), near the bottoms of its two valleys, and \( t = 0.5 \): the midpoint is \( x = -0.1 \). The curve there is at \( f(-0.1) = -0.1299 \), the chord at \( 0.5 \cdot (-3.5139) + 0.5 \cdot (-1.0659) = -2.2899 \). The chord is **2.16 below** the curve, so this function is not convex.

**Why it matters.** The chord test is the precise version of "bowl-shaped". A loss that passes it everywhere has a single valley (you'll prove that below), so gradient descent can't get trapped. A loss that fails it somewhere has a hump, and a hump can separate two valleys.

```python
def square(x):
    return x ** 2

def wavy(x):
    return x**4 - 3 * x**2 + x       # a curve with two valleys; we'll use it all lesson

def chord_gap(f, a, b, t):
    """Chord height minus curve height, at the point x = (1 - t)·a + t·b."""
    x = (1 - t) * a + t * b
    return (1 - t) * f(a) + t * f(b) - f(x)

print(chord_gap(square, -1, 3, 0.5))
print(chord_gap(wavy, -1.3, 1.1, 0.5))
# 4.0
# -2.16
```

**Underneath: a trace along one chord.** Slide \( t \) from 0 to 1 along the failing chord and watch the gap. Both ends are on the curve (gap 0, by construction); in between, the hump of the curve rises above the straight line:

```python
for t in [0, 0.25, 0.5, 0.75, 1.0]:
    x = (1 - t) * -1.3 + t * 1.1
    chord = (1 - t) * wavy(-1.3) + t * wavy(1.1)
    print(f"t = {t:4.2f}   x = {x:5.2f}   chord {chord:7.4f}   curve {wavy(x):7.4f}   gap {chord - wavy(x):7.4f}")
# t = 0.00   x = -1.30   chord -3.5139   curve -3.5139   gap  0.0000
# t = 0.25   x = -0.70   chord -2.9019   curve -1.9299   gap -0.9720
# t = 0.50   x = -0.10   chord -2.2899   curve -0.1299   gap -2.1600
# t = 0.75   x =  0.50   chord -1.6779   curve -0.1875   gap -1.4904
# t = 1.00   x =  1.10   chord -1.0659   curve -1.0659   gap  0.0000
```

(The format spec `{t:4.2f}` means "a float, 2 digits after the point, padded to 4 characters wide", so the columns line up; `{chord:7.4f}` is the same with 4 decimals in 7 characters.) The worst gap along this chord is not exactly at the midpoint: checking every \( t \) finely, it reaches 2.19, slightly to the right of \( t = 0.5 \). That's the "by up to 2.19" in the animation below, which draws the gap at the worst point of each chord (its "0.37" frame is a different, shorter chord, from \( x = -0.2 \) to \( x = 1.5 \), across the hump). Notice also that the chord's height moves in equal steps (−3.5139, −2.9019, −2.2899, ...: \( +0.612 \) each time), because it is a straight line, while the curve does not.

<figure class="diagram" markdown>
![Animation in two panels: on the left, the parabola x squared, with a chord between two orange points that jumps to four different positions and is always teal and above the curve, labelled chord stays above the curve; on the right, the two-valley curve x to the fourth minus 3 x squared plus x, where some chords lie above the curve and others turn red where they pass below the hump between the valleys, labelled chord dips below the curve by up to 2.19 and by up to 0.37](../../assets/img/phase-1/day53-chord-test.svg)
<figcaption>Animated: on x² every chord stays above the curve; on the two-valley curve some chords dip below the hump in the middle, so it is not convex.</figcaption>
</figure>

One chord can only *disprove* convexity. To gain confidence that a function *is* convex you'd have to check every chord, which code can't do; but it can check a hundred thousand random ones in a blink:

```python
def looks_convex(f, lo, hi, trials=100_000, seed=53):
    """Random chord test on [lo, hi]. Returns (passed, most negative gap found)."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(lo, hi, trials)
    b = rng.uniform(lo, hi, trials)
    t = rng.uniform(0, 1, trials)
    worst = chord_gap(f, a, b, t).min()
    return bool(worst >= -1e-9), round(float(worst), 4)

for name, fn in [("x²", square), ("|x|", np.abs), ("exp", np.exp), ("wavy", wavy), ("sin", np.sin)]:
    print(f"{name:5s}", looks_convex(fn, -2, 2))
print("sin on [π, 2π]", looks_convex(np.sin, np.pi, 2 * np.pi))
# x²    (True, 0.0)
# |x|   (True, 0.0)
# exp   (True, 0.0)
# wavy  (False, -2.2451)
# sin   (False, -0.5132)
# sin on [π, 2π] (True, 0.0)
```

How it works: `rng.uniform(lo, hi, trials)` draws 100,000 random numbers between `lo` and `hi` in one array, so `a`, `b` and `t` each hold 100,000 values, and `chord_gap` (written with NumPy operations only) computes all 100,000 gaps at once by elementwise arithmetic (Day 4). `.min()` picks the worst one. The `f"{name:5s}"` spec pads a string to 5 characters.

Notice three things. \( |x| \) is convex even though it has a sharp corner at 0: convexity doesn't need smoothness (MAE and ReLU-style kinks are fine). The `-1e-9` tolerance is there because floating-point rounding can make a gap of exactly 0 come out as \( -10^{-16} \). And convexity depends on the **interval**: \( \sin x \) is not convex on \( [-2, 2] \) but is convex on \( [\pi, 2\pi] \), where it is one smooth valley.

!!! warning "Common mistake: testing only part of the range"
    A function can pass the chord test on every interval you happen to try and still not be convex. Test the two-valley function on each valley separately:

    ```python
    print(looks_convex(wavy, -2, -0.8), looks_convex(wavy, 0.8, 2), looks_convex(wavy, -2, 2))
    # (True, 0.0) (True, 0.0) (False, -2.2451)
    ```

    Each half passes; the whole range fails, because the failing chords are the ones that **cross** the hump. When you check a loss, test the full range of parameters that training can actually visit, and remember that a passing random test is evidence, not proof.

!!! warning "Common mistake: convex vs concave, and ‘concave up’"
    A **concave** function is the opposite: every chord lies on or *below* the curve (like \( -x^2 \) or \( \log x \)), so it has a single *peak*. The chord gap is then negative:

    ```python
    print(chord_gap(lambda x: -x**2, -1, 3, 0.5))     # chord −5, curve −1
    # -4.0
    ```

    (`lambda x: -x**2` is a one-line anonymous function, like a Kotlin lambda `{ x -> -x * x }`.) School textbooks often call convex "concave up" and concave "concave down". In ML, "convex" always means **bowl-shaped**, and \( f \) is concave exactly when \( -f \) is convex. Maximizing a concave function (like the log-likelihood of Day 47) is the same problem as minimizing a convex one.

## Curvature: the second derivative

The chord test is a definition, not a convenient tool. For smooth functions there is a much quicker test, using something you already know how to compute.

The derivative \( f'(x) \) is the slope (Day 29). The derivative **of the derivative**, written \( f''(x) \) ("f double-prime") or \( \frac{d^2 f}{dx^2} \), says how fast the slope is changing:

- \( f''(x) > 0 \): the slope is **increasing** as you move right. The curve bends upward, like the inside of a bowl.
- \( f''(x) < 0 \): the slope is decreasing. The curve bends downward, like the top of a hill.
- \( f''(x) = 0 \): no bending at that point (for a moment, the curve is straight).

So \( f'' \) measures **curvature**, and the test is:

\[
f''(x) \ge 0 \text{ for every } x \text{ in an interval} \quad\Longleftrightarrow\quad f \text{ is convex on that interval}
\]

This test needs \( f'' \) to exist everywhere on the interval (\( f \) "twice differentiable"); for functions with corners, such as \( |x| \), which has no \( f'' \) at 0, fall back on the chord test. Why it works: \( f'' \ge 0 \) means the slope \( f' \) never decreases as you move right. A curve whose slope only ever increases keeps turning upward, so it can never come back down below a tangent line drawn at any of its points, and that "above every tangent" shape is exactly convexity (you'll use the tangent form in a moment).

**Worked example.** For \( f(x) = x^2 \): \( f'(x) = 2x \), so \( f''(x) = 2 \). Positive everywhere: convex everywhere, with the same bend at every point. For \( f(x) = x^4 - 3x^2 + x \), apply the power rule (Day 30) twice: \( f'(x) = 4x^3 - 6x + 1 \), then \( f''(x) = 12x^2 - 6 \). At \( x = 0 \), \( f''(0) = -6 \): the curve bends down there (the hump). \( f''(x) < 0 \) whenever \( 12x^2 < 6 \), that is for \( |x| < \sqrt{0.5} \approx 0.707 \). So the function is convex on \( (-\infty, -0.707] \) and on \( [0.707, \infty) \), but not on any interval that crosses the hump.

To compute \( f'' \) numerically, apply the "nudge left, nudge right" idea once more. The slope a little to the right of \( x \) is about \( \frac{f(x+h) - f(x)}{h} \), a little to the left about \( \frac{f(x) - f(x-h)}{h} \); their difference divided by \( h \) is the rate of change of the slope:

\[
f''(x) \approx \frac{\frac{f(x+h) - f(x)}{h} - \frac{f(x) - f(x-h)}{h}}{h} = \frac{f(x + h) - 2f(x) + f(x - h)}{h^2}
\]

**By hand first**, for \( f(x) = x^2 \) at \( x = 1 \) with \( h = 0.1 \): \( f(0.9) = 0.81 \), \( f(1) = 1 \), \( f(1.1) = 1.21 \). The slope on the left is \( (1 - 0.81)/0.1 = 1.9 \); on the right, \( (1.21 - 1)/0.1 = 2.1 \). The slope grew by \( 0.2 \) over a distance of \( 0.1 \), so \( f''(1) \approx 0.2 / 0.1 = 2 \). The one-line formula gives the same: \( (1.21 - 2 + 0.81)/0.01 = 2 \). That is the exact answer, \( f'' = 2 \), which the code reproduces:

```python
def central_diff(f, x, h=1e-5):          # Day 29
    return (f(x + h) - f(x - h)) / (2 * h)

def second_diff(f, x, h=1e-4):
    return (f(x + h) - 2 * f(x) + f(x - h)) / h**2

xs = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
print(second_diff(square, xs).round(4))
print(second_diff(wavy, xs).round(4))
print(12 * xs**2 - 6)                    # the formula, for comparison
# [2. 2. 2. 2. 2. 2. 2.]
# [21.  6. -3. -6. -3.  6. 21.]
# [21.  6. -3. -6. -3.  6. 21.]
```

(The step is \( h = 10^{-4} \), bigger than `central_diff`'s \( 10^{-5} \), because dividing by \( h^2 \) magnifies floating-point error much more; Day 29's trade-off, squared.)

**Where you'll use it.** In ML the second derivative answers "how sharply does the loss bend here?". A sharp bend means a small step size is needed (you'll see the exact rule in the Hessian section); a flat region (\( f'' \approx 0 \)) means the slope barely changes, so gradient descent can creep for a long time.

!!! warning "Common mistake: a step size that is too small"
    Smaller \( h \) is not always more accurate. The three function values in the numerator agree in almost all their digits, and subtracting them leaves only rounding noise, which the division by \( h^2 \) then blows up:

    ```python
    for h in [1e-2, 1e-4, 1e-6, 1e-8]:
        print(h, second_diff(square, 1.0, h), second_diff(wavy, 1.0, h))
    # 0.01 1.9999999999997797 6.000200000004785
    # 0.0001 1.9999999989472883 6.000000007944095
    # 1e-06 1.9998447342572945 5.9999782919817335
    # 1e-08 -1.1102230246251563 12.21245327087672
    ```

    The true values are 2 and 6. At \( h = 10^{-8} \) the curvature of \( x^2 \) comes out **negative**, which would wrongly call a bowl "not convex". If a numerical second derivative looks strange, try \( h \) between \( 10^{-4} \) and \( 10^{-3} \) and check that the answer stays the same.

### Why convex means "no bad valleys"

Here is the payoff, and it takes three lines. For a convex function, every **tangent line lies below the curve** (it's the chord test in the limit where the chord shrinks to one point). In symbols, at any point \( x^* \) and for every \( x \):

\[
f(x) \;\ge\; f(x^*) + f'(x^*)\,(x - x^*)
\]

The right side is the tangent line at \( x^* \): its height plus slope times distance.

**By hand.** For \( f(x) = x^2 \) at \( x^* = 1 \): \( f(1) = 1 \) and \( f'(1) = 2 \), so the tangent is \( 1 + 2(x - 1) \). At \( x = 3 \) the tangent is at \( 5 \), the curve at \( 9 \): the curve is 4 above. At \( x = -1 \) the tangent is at \( -3 \), the curve at \( 1 \): again 4 above. The code checks a few more points:

```python
xs_t = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
tangent = 1 + 2 * (xs_t - 1)                 # tangent line of x² at x* = 1
print(xs_t**2, tangent, xs_t**2 - tangent)
# [1. 0. 1. 4. 9.] [-3. -1.  1.  3.  5.] [4. 1. 0. 1. 4.]
```

The gap is never negative, and it is 0 only at the touching point. (For \( x^2 \) the gap is \( (x - 1)^2 \), a square.) You'll see this property fail for the two-valley function in the next section. Now suppose gradient descent stops at \( x^* \) because the slope there is zero, \( f'(x^*) = 0 \). The inequality becomes

\[
f(x) \;\ge\; f(x^*) \quad \text{for every } x
\]

which says \( x^* \) is the **global** minimum. For a convex function, **"the slope is zero" and "this is the best possible point" are the same thing.** Gradient descent stops only where the slope is zero, so on a convex loss (with a sensible learning rate) it can only ever stop at the global minimum. There are no traps.

!!! note "One minimum value, but maybe many minimizers"
    Convexity guarantees one minimum **value**. The best points can still form a flat valley floor (a whole line of equally good parameters, as you'll see with duplicated features below), and that floor is itself a convex set. A **strictly convex** function is one where every chord lies strictly above the curve between its ends (no straight pieces). It has **at most one** minimizer: maybe one point, maybe none at all. \( e^x \) is strictly convex and has no minimum (it keeps falling towards 0 as \( x \to -\infty \) without reaching it). \( f'' > 0 \) everywhere is enough for strict convexity but not needed: \( x^4 \) is strictly convex even though \( f''(0) = 0 \).

## Local vs global minima

Back to the two-valley function. Its slope is zero where \( 4x^3 - 6x + 1 = 0 \). That cubic has three solutions, and the second derivative classifies each one:

- \( f'' > 0 \) at a flat point: the curve bends up, so it's the bottom of a valley, a **minimum**.
- \( f'' < 0 \): it bends down, a **maximum** (a hilltop).

```python
crit = np.sort(np.roots([4, 0, -6, 1]).real)    # solve 4x³ − 6x + 1 = 0
print(crit.round(4))
print(wavy(crit).round(4))
print(second_diff(wavy, crit).round(2))
# [-1.3008  0.1699  1.1309]
# [-3.5139  0.0841 -1.0702]
# [14.31 -5.65  9.35]
```

`np.roots([4, 0, -6, 1])` finds the solutions of the polynomial whose coefficients you list from the highest power down: \( 4x^3 + 0x^2 - 6x + 1 \). It may return complex numbers in general, so `.real` keeps the real part (here all three solutions are real), and `np.sort` puts them in order.

**Why the local minimum is not global: the tangent test fails.** At the local minimum \( x^* = 1.1309 \) the slope is 0, so its tangent line is the flat line at height \( -1.0702 \). For a convex function that flat line would lie below the whole curve. Here it doesn't: at the global minimum the curve is down at \( -3.5139 \), far **below** the tangent.

```python
print(round(wavy(crit[2]), 4), round(central_diff(wavy, crit[2]), 6), round(wavy(crit[0]), 4))
# -1.0702 0.0 -3.5139
```

That's the whole difference between the two cases: on a convex function the flat tangent at a zero-slope point is a floor under everything; on a non-convex one, part of the curve can sit beneath it.

!!! warning "Common mistake: trusting the second-derivative test when f'' = 0"
    The rule "\( f' = 0 \) and \( f'' > 0 \) means minimum" says nothing when \( f'' = 0 \). Three functions are all flat with zero curvature at 0, and they are three different things:

    ```python
    for name, f in [("x³", lambda x: x**3), ("x⁴", lambda x: x**4), ("−x⁴", lambda x: -x**4)]:
        print(name, second_diff(f, 0.0), [f(-0.1), f(0.0), f(0.1)])
    # x³ 0.0 [-0.0010000000000000002, 0.0, 0.0010000000000000002]
    # x⁴ 2.0000000000000004e-08 [0.00010000000000000002, 0.0, 0.00010000000000000002]
    # −x⁴ -2.0000000000000004e-08 [-0.00010000000000000002, -0.0, -0.00010000000000000002]
    ```

    (The tiny \( \pm 2 \times 10^{-8} \) is the numerical estimate of an exact 0.) Look at the neighbours instead: \( x^3 \) goes down on the left and up on the right, so 0 is neither a minimum nor a maximum (a flat **inflection point**); \( x^4 \) is higher on both sides (a minimum); \( -x^4 \) lower on both sides (a maximum). When the curvature is zero, check the function's values around the point, or higher derivatives.

So the function has a **global minimum** at \( x \approx -1.30 \) (height −3.51, the lowest point anywhere), a **local minimum** at \( x \approx 1.13 \) (height −1.07, the lowest point *nearby*, but not overall) and a local maximum at \( x \approx 0.17 \) between them. Gradient descent can't tell the two minima apart: at both, the slope is zero and every small step goes uphill.

```python
def gd_1d(f, x0, lr=0.05, steps=200):
    x = x0
    for _ in range(steps):
        x = x - lr * central_diff(f, x)
    return x

for x0 in [-2.0, -0.5, 0.0, 0.3, 2.0]:
    x_end = gd_1d(wavy, x0)
    print(f"start {x0:5.1f} -> x = {x_end:7.4f}, f = {wavy(x_end):.4f}")
# start  -2.0 -> x = -1.3008, f = -3.5139
# start  -0.5 -> x = -1.3008, f = -3.5139
# start   0.0 -> x = -1.3008, f = -3.5139
# start   0.3 -> x =  1.1309, f = -1.0702
# start   2.0 -> x =  1.1309, f = -1.0702
```

The local maximum at 0.17 is a **watershed**: start left of it and you end in the good valley, start right of it and you're stuck in the worse one, with a loss 2.44 higher (−1.07 instead of −3.51). Starts at 0.0 and 0.3, only 0.3 apart, end up in different valleys. The set of starting points that lead to a given minimum is called its **basin of attraction**.

On the convex \( x^2 \), by contrast, the start doesn't matter:

```python
for x0 in [-2.0, 0.3, 2.0]:
    print(f"start {x0:5.1f} -> x = {gd_1d(square, x0):.1e}")
# start  -2.0 -> x = -1.4e-09
# start   0.3 -> x = 2.1e-10
# start   2.0 -> x = 1.4e-09
```

(The format spec `:.1e` prints a float in **scientific notation** with 1 digit after the point: `-1.4e-09` means \( -1.4 \times 10^{-9} \). All three runs end at 0 up to tiny rounding leftovers.)

<figure class="diagram" markdown>
![Line plot of f(x) = x^4 − 3x^2 + x from x = −2 to 2: a deep valley on the left at x ≈ −1.30 with f = −3.51, a small hump at x ≈ 0.17 marked as the watershed, and a shallower valley on the right at x ≈ 1.13 with f = −1.07; orange dots of gradient descent started at x = 0 roll left into the deep valley, teal dots started at x = 0.3 roll right into the shallow one](../../assets/img/phase-1/day53-two-valleys.svg)
<figcaption>Gradient descent on a non-convex function: which valley you reach depends on which side of the watershed you start.</figcaption>
</figure>

Here is the code that drew it:

```python
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "savefig.facecolor": "white"})
grid = np.linspace(-2.0, 2.0, 400)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(grid, wavy(grid), color="#3949ab", lw=2, label="f(x) = x⁴ − 3x² + x")
for x0, color in [(0.0, "#ef6c00"), (0.3, "#00897b")]:
    path = [x0]
    for _ in range(12):
        path.append(path[-1] - 0.05 * central_diff(wavy, path[-1]))
    path = np.array(path)
    ax.plot(path, wavy(path), "o-", color=color, ms=5, lw=1, label=f"GD from x = {x0}")
ax.axvline(crit[1], color="#555", ls=":", lw=1)
ax.annotate("local max: the watershed\nx ≈ 0.17", xy=(crit[1], wavy(crit[1])), xytext=(0.5, 1.4),
            arrowprops=dict(arrowstyle="->", color="#555"), color="#555", fontsize=10)
ax.annotate(f"global min  f = {wavy(crit[0]):.2f}", xy=(crit[0], wavy(crit[0])), xytext=(-1.75, -4.6),
            color="#ef6c00", fontsize=10)
ax.annotate(f"local min  f = {wavy(crit[2]):.2f}", xy=(crit[2], wavy(crit[2])), xytext=(0.65, -2.4),
            color="#00897b", fontsize=10)
ax.set_ylim(-5, 5)
ax.set_xlabel("x (the parameter)")
ax.set_ylabel("f(x) (the loss)")
ax.set_title("Two starts 0.3 apart, two different valleys")
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=9)
fig.savefig("day53-two-valleys.svg", format="svg", bbox_inches="tight")
plt.close(fig)
```

!!! warning "Common mistake: ‘the loss stopped falling, so I found the best model’"
    A flat loss curve tells you the gradient is (nearly) zero, nothing more. On a non-convex loss that can be a local minimum, a saddle point or a plateau. Practical checks: train again from a few different random seeds and compare the final losses; if they differ a lot, you are landing in different basins.

## Many parameters: the Hessian

**Intuition first.** Stand on a hillside (Day 32's picture). The gradient tells you which way is steepest. The **curvature** tells you how the ground bends under your feet, and on a surface it can bend differently in different directions: across a mountain pass the ground curves up on both sides of you (a valley across), while along the path it curves down in front of and behind you (a ridge along). One number can't describe that; you need one curvature per direction, and a matrix holds all of them at once.

A real loss has many parameters, so "the second derivative" becomes a whole table of them. For a function \( F \) of a vector \( \mathbf{p} = [p_1, \dots, p_k] \), take the partial derivative (Day 32) twice, once with respect to \( p_i \) and once with respect to \( p_j \). Put the results in a \( k \times k \) matrix, the **Hessian** \( H \):

\[
H_{ij} = \frac{\partial^2 F}{\partial p_i \, \partial p_j}
\]

- The diagonal entry \( H_{ii} \) is the curvature along the axis of \( p_i \), with every other parameter frozen.
- An off-diagonal entry \( H_{ij} \) says how the slope in direction \( i \) changes as you move in direction \( j \): a "twist".
- For smooth functions the order of the two derivatives doesn't matter, so \( H_{ij} = H_{ji} \): **the Hessian is symmetric**.

**Worked example.** \( F(x, y) = x^2 + 3y^2 \), the bowl from Day 32. \( \partial F/\partial x = 2x \), and differentiating that again by \( x \) gives 2, by \( y \) gives 0. \( \partial F/\partial y = 6y \), which gives 0 by \( x \) and 6 by \( y \). So

\[
H = \begin{bmatrix} 2 & 0 \\ 0 & 6 \end{bmatrix}
\]

the same at every point: curvature 2 along \( x \), 6 along \( y \) (that's why its contours were stretched ellipses).

Symmetric means you can use Day 24: \( H \) has real eigenvalues and perpendicular eigenvectors. **Each eigenvector is a direction, and its eigenvalue is the curvature along that direction.** That gives the many-dimensional versions of today's tests:

| Hessian eigenvalues | Shape | At a point with zero gradient |
|---|---|---|
| All \( \ge 0 \) at every point | Convex | Global minimum |
| All \( > 0 \) at that point | Bowl locally | (Local) minimum |
| All \( < 0 \) at that point | Dome locally | (Local) maximum |
| Some \( > 0 \), some \( < 0 \) | Saddle locally | **Saddle point** |
| Some \( = 0 \), the rest all \( \ge 0 \) (or all \( \le 0 \)) at that point | Flat in some direction | Inconclusive: look closer |

The last row is the many-variable version of the \( f'' = 0 \) case (see the warning box in the next section): a zero eigenvalue means zero curvature in that direction, and second derivatives alone can't say what happens along it. If the Hessian is like that at **every** point, as for the duplicated-feature regression below, the function is convex with a flat valley floor of minimizers; at a single point it could also be a flat inflection, like \( x^3 \) at 0.

A symmetric matrix with all eigenvalues \( \ge 0 \) is called **positive semi-definite** (PSD); with all \( > 0 \), **positive definite**. "A function is convex exactly when its Hessian is PSD everywhere" is the many-variable version of \( f'' \ge 0 \).

The numerical Hessian is the second-difference idea applied to each pair of coordinates:

```python
def hessian(F, p, h=1e-4):
    """Numerical Hessian: H[i, j] ≈ ∂²F / ∂p_i ∂p_j, by central differences."""
    p = np.asarray(p, dtype=float)
    k = p.size
    E = np.eye(k) * h
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            H[i, j] = (F(p + E[i] + E[j]) - F(p + E[i] - E[j])
                       - F(p - E[i] + E[j]) + F(p - E[i] - E[j])) / (4 * h * h)
    return H

def bowl(p):
    return p[0]**2 + 3 * p[1]**2

def saddle(p):
    return p[0]**2 - p[1]**2

print(hessian(bowl, [0.7, -1.2]).round(4))
print(np.linalg.eigvalsh(hessian(bowl, [0.7, -1.2])).round(4))
print(hessian(saddle, [0.0, 0.0]).round(4))
print(np.linalg.eigvalsh(hessian(saddle, [0.0, 0.0])).round(4))
# [[2. 0.]
#  [0. 6.]]
# [2. 6.]
# [[ 2.  0.]
#  [ 0. -2.]]
# [-2.  2.]
```

(`np.linalg.eigvalsh` is `eigh` that returns only the eigenvalues, ascending; the Hessian is symmetric, so `eigh` is the right tool, Day 24.)

**Underneath: the four-point formula.** `hessian` estimates the mixed derivative \( \partial^2 F / \partial p_i \partial p_j \) as "the change in the \( p_i \)-slope when you move along \( p_j \)". Write \( \mathbf{e}_i \) for the unit vector along coordinate \( i \), so `E[i]` is \( h\mathbf{e}_i \). The \( p_i \)-slope one step forward along \( p_j \), and one step back, are central differences:

\[
\text{slope}_+ = \frac{F(\mathbf{p} + h\mathbf{e}_i + h\mathbf{e}_j) - F(\mathbf{p} - h\mathbf{e}_i + h\mathbf{e}_j)}{2h}
\qquad
\text{slope}_- = \frac{F(\mathbf{p} + h\mathbf{e}_i - h\mathbf{e}_j) - F(\mathbf{p} - h\mathbf{e}_i - h\mathbf{e}_j)}{2h}
\]

In \( \text{slope}_- \) **both** terms have \( -h\mathbf{e}_j \). Subtract the two slopes and divide by the distance \( 2h \) between them: \( (\text{slope}_+ - \text{slope}_-)/(2h) \), four function values over \( 2h \cdot 2h = 4h^2 \). That's exactly the expression in the loop. For the saddle at the origin with \( h = 0.1 \), entry \( H_{00} \) uses \( F(0.2, 0) = 0.04 \), \( F(0, 0) \) twice (\( = 0 \)) and \( F(-0.2, 0) = 0.04 \): \( (0.04 - 0 - 0 + 0.04)/0.04 = 2 \). It costs \( 4k^2 \) loss evaluations, fine for 3 parameters and hopeless for a million, which is why big models never build the full Hessian.

!!! warning "Common mistake: reading curvature off the diagonal only"
    The diagonal of \( H \) is the curvature along each **axis**. A function can curve up along both axes and still be a saddle, because the steepest downhill curvature lies along a diagonal direction. Take \( g(x, y) = x^2 + 3xy + y^2 \), whose Hessian is \( \begin{bmatrix} 2 & 3 \\ 3 & 2 \end{bmatrix} \):

    ```python
    H_g = np.array([[2., 3.], [3., 2.]])
    g = lambda x, y: x**2 + 3 * x * y + y**2
    print(np.diag(H_g), np.linalg.eigvalsh(H_g), g(0., 0.), g(1., -1.))
    # [2. 2.] [-1.  5.] 0.0 -1.0
    ```

    Both diagonal entries are positive, which suggests a bowl, but one eigenvalue is −1: along the direction \( [1, -1] \) the function goes **down** (\( g(1, -1) = -1 < g(0, 0) \)). The origin is a saddle. Always decide with the eigenvalues, not the diagonal.

### The eigenvalue is the curvature along its eigenvector

Why do the eigenvalues, and not the diagonal, decide? Walk along a straight line through the origin in the direction of a unit vector \( \mathbf{v} \): the points are \( t\mathbf{v} \), and the function along that line is the one-variable function \( t \mapsto g(t\mathbf{v}) \). Its second derivative at \( t = 0 \) is the curvature in direction \( \mathbf{v} \), and it equals \( \mathbf{v}^\top H \mathbf{v} \) (you'll see why in the next subsection). If \( \mathbf{v} \) is a unit eigenvector, \( H\mathbf{v} = \lambda\mathbf{v} \), so \( \mathbf{v}^\top H \mathbf{v} = \lambda\, \mathbf{v} \cdot \mathbf{v} = \lambda \) (Day 24's Rayleigh quotient). Check it by measuring the curvature along each eigenvector of \( g \) with `second_diff`:

```python
lam_g, Q_g = np.linalg.eigh(H_g)
for i in range(2):
    v = Q_g[:, i]                                      # unit eigenvector i (a column)
    along = lambda t: g(*(t * v))                      # g on the line through 0 along v
    print(round(lam_g[i], 4), v.round(4), round(second_diff(along, 0.0), 4), round(v @ H_g @ v, 4))
# -1.0 [-0.7071  0.7071] -1.0 -1.0
# 5.0 [0.7071 0.7071] 5.0 5.0
```

(`g(*(t * v))` unpacks the two entries of the vector `t * v` into `g`'s two arguments `x` and `y`, like Kotlin's spread operator `*array`.) The measured curvature along \( [-0.707, 0.707] \) is −1 and along \( [0.707, 0.707] \) it is 5: exactly the eigenvalues. Every other direction is a mix of these two, so its curvature lies between −1 and 5; the axis directions give 2, the diagonal entries.

### The quadratic picture: second-order Taylor

**Intuition first.** Zoom in on any smooth curve and it looks like a straight line (that's the tangent, Day 29). Zoom in a little less and a straight line is no longer good enough, but a parabola is. The parabola that matches the function's height, slope and curvature at a point is its **second-order Taylor approximation**:

\[
f(x + \delta) \approx f(x) + f'(x)\,\delta + \tfrac12 f''(x)\,\delta^2
\]

- \( \delta \) ("delta") is a small step away from \( x \).
- \( f(x) \) is where you are; \( f'(x)\,\delta \) is the tangent line's change (slope times step); \( \tfrac12 f''(x)\,\delta^2 \) is the bend.
- The \( \tfrac12 \) makes the parabola's curvature come out right: the second derivative of \( \tfrac12 c\,\delta^2 \) is \( c \).

**By hand**, for the two-valley function at \( x = 1 \): \( f(1) = 1 - 3 + 1 = -1 \), \( f'(1) = 4 - 6 + 1 = -1 \), \( f''(1) = 12 - 6 = 6 \). For \( \delta = 0.1 \): linear \( -1 + (-1)(0.1) = -1.1 \), quadratic \( -1.1 + \tfrac12 \cdot 6 \cdot 0.01 = -1.07 \). The true value is \( f(1.1) = -1.0659 \), so the quadratic is much closer:

```python
f0, f1, f2 = wavy(1.0), central_diff(wavy, 1.0), second_diff(wavy, 1.0)
print(f0, round(f1, 6), round(f2, 4))
for d in [0.5, 0.1, 0.01]:
    lin = f0 + f1 * d
    quad = lin + 0.5 * f2 * d**2
    print(f"δ = {d:4}   true {wavy(1.0 + d):.6f}   linear {lin:.6f}   quadratic {quad:.6f}   "
          f"errors {wavy(1.0 + d) - lin:.2e} {wavy(1.0 + d) - quad:.2e}")
# -1.0 -1.0 6.0
# δ =  0.5   true -0.187500   linear -1.500000   quadratic -0.750000   errors 1.31e+00 5.62e-01
# δ =  0.1   true -1.065900   linear -1.100000   quadratic -1.070000   errors 3.41e-02 4.10e-03
# δ = 0.01   true -1.009696   linear -1.010000   quadratic -1.009700   errors 3.04e-04 4.01e-06
```

Each time \( \delta \) shrinks 10×, the linear error shrinks about 100× and the quadratic error about 1000×. For a big step (\( \delta = 0.5 \)) both are poor: the picture is local.

**Many variables.** The same idea with the gradient and the Hessian, for a step \( \mathbf{v} \):

\[
F(\mathbf{p} + \mathbf{v}) \approx F(\mathbf{p}) + \nabla F(\mathbf{p}) \cdot \mathbf{v} + \tfrac12\, \mathbf{v}^\top H \mathbf{v}
\]

Put \( \mathbf{v} = t\mathbf{u} \) for a unit direction \( \mathbf{u} \): the function along that line is about \( F(\mathbf{p}) + t\,\nabla F \cdot \mathbf{u} + \tfrac12 t^2\, \mathbf{u}^\top H \mathbf{u} \), whose second derivative in \( t \) is \( \mathbf{u}^\top H \mathbf{u} \). That is the "curvature in direction \( \mathbf{u} \)" promised in the previous subsection.

This is why today's tests work. At a flat point \( \nabla F = \mathbf{0} \), so near it **only** \( \tfrac12\mathbf{v}^\top H\mathbf{v} \) is left: if that's positive for every step direction (all eigenvalues positive), every nearby point is higher, a minimum; if it's negative for some \( \mathbf{v} \), there's a way down, a saddle or a maximum. Check it near the origin for a saddle with an extra cubic term, \( F(x, y) = x^2 - y^2 + x^3 \), whose gradient at the origin is \( \mathbf{0} \) and whose Hessian there is \( \operatorname{diag}(2, -2) \):

```python
def F_cubic(p):
    return p[0]**2 - p[1]**2 + p[0]**3

H0 = np.array([[2., 0.], [0., -2.]])
for v in [np.array([0.1, 0.05]), np.array([0.01, 0.005])]:
    print(v, F_cubic(v), 0.5 * v @ H0 @ v)
# [0.1  0.05] 0.008500000000000002 0.0075000000000000015
# [0.01  0.005] 7.6e-05 7.500000000000001e-05
```

The quadratic term gives 0.0075 against the true 0.0085 for the bigger step, and 0.000075 against 0.000076 for the smaller one: close to a flat point, \( \tfrac12\mathbf{v}^\top H\mathbf{v} \) is the whole story.

**The gradient of a quadratic.** For a symmetric \( H \), the function \( q(\mathbf{p}) = \tfrac12 \mathbf{p}^\top H \mathbf{p} \) has gradient \( \nabla q = H\mathbf{p} \) and Hessian \( H \). (In one variable this is \( \tfrac12 c p^2 \to cp \); the \( \tfrac12 \) cancels the 2 from the power rule.) A numerical check with Day 29's central difference, one coordinate at a time:

```python
q = lambda p: 0.5 * p @ H_g @ p
p_q = np.array([0.7, -0.4])
numeric = np.array([(q(p_q + e) - q(p_q - e)) / 2e-5 for e in np.eye(2) * 1e-5])
print(numeric.round(6), H_g @ p_q)
# [0.2 1.3] [0.2 1.3]
```

(`np.eye(2) * 1e-5` is a 2×2 matrix whose rows are the two nudge vectors \( [10^{-5}, 0] \) and \( [0, 10^{-5}] \); iterating over a matrix gives its rows.) That's the form behind Exercise 6, and it's why the MSE Hessian below is a fixed matrix: a quadratic loss has the same curvature everywhere.

!!! note "Curvature and the learning rate"
    The largest Hessian eigenvalue is the steepest curvature, and it sets the speed limit for gradient descent: on a quadratic, GD converges only if the learning rate is below \( 2/\lambda_{\max} \). The ratio \( \lambda_{\max} / \lambda_{\min} \) (the **condition number**) says how stretched the valley is, which is why Day 52's ravine was slow and why standardizing features helps: it makes the Hessian's eigenvalues more alike.

### Building convex functions from convex pieces

Checking convexity from scratch every time is tiring. In practice you build losses from pieces whose convexity you already know, using four rules that each preserve convexity:

1. **Sums**: if \( f \) and \( g \) are convex, so is \( f + g \). (Add the two chord inequalities.) A loss that is a sum over training examples of convex per-example losses is convex.
2. **Non-negative multiples**: \( c\,f \) with \( c \ge 0 \) is convex. (Multiply the chord inequality by \( c \); a negative \( c \) would flip it.) So the mean, which multiplies a sum by \( 1/n \), is convex, and so is adding an L2 penalty \( \lambda \|\mathbf{w}\|^2 \) with \( \lambda \ge 0 \).
3. **Composition with an affine map**: if \( f \) is convex, so is \( \mathbf{w} \mapsto f(A\mathbf{w} + \mathbf{b}) \). (A straight line in \( \mathbf{w} \) maps to a straight line in \( A\mathbf{w} + \mathbf{b} \), so chords map to chords.) This is the step from "\( r^2 \) is convex in the residual \( r \)" to "\( (\mathbf{x}_i \cdot \mathbf{w} - y_i)^2 \) is convex in \( \mathbf{w} \)".
4. **Maximum**: the pointwise maximum \( \max(f, g) \) of convex functions is convex. (ReLU, \( \max(0, x) \), and the hinge loss are built this way.)

The minimum, the difference and composition with a non-linear map (such as a sigmoid inside a square) do **not** preserve convexity in general:

```python
print(looks_convex(lambda x: (2 * x - 1) ** 2 + 3 * np.abs(x), -3, 3))       # rule 3, then rules 2 and 1
print(looks_convex(lambda x: np.maximum(x ** 2, np.exp(x)), -3, 3))          # rule 4
print(looks_convex(lambda x: x ** 2 - 3 * np.abs(x), -3, 3))                 # a difference
print(looks_convex(lambda x: np.minimum(x ** 2, (x - 2) ** 2), -3, 3))       # a minimum
# (True, 0.0)
# (True, 0.0)
# (False, -2.2326)
# (False, -0.9869)
```

(`np.maximum` and `np.minimum` take the larger or smaller of two arrays entry by entry; they differ from `np.max`, which reduces one array to its largest value.) The minimum of two bowls is two valleys with a ridge between them: the shape of our two-valley function. With these rules, MSE (rule 3 applied to a square, then rules 1 and 2) and logistic cross-entropy (rule 3 applied to the convex \( \log(1 + e^{z}) \), then 1 and 2) are convex before you compute a single Hessian. The next section confirms it numerically.

## The convex losses of classic ML

Now the reason this matters. The two most important classic models have convex losses, so gradient descent on them is guaranteed to find the best parameters.

### Linear regression: MSE

On Day 33 you derived the MSE gradient \( \nabla L = \frac{2}{n} X^\top (X\mathbf{w} - \mathbf{y}) \). Differentiate once more: the gradient is \( \frac{2}{n} X^\top X \mathbf{w} \) minus a constant, and \( \mathbf{w} \) appears only multiplied by a fixed matrix, so the Hessian is that matrix (the same rule as \( \frac{d}{dx}(5x) = 5 \)):

\[
H = \frac{2}{n} X^\top X
\]

It doesn't depend on \( \mathbf{w} \) at all: the MSE is an exact quadratic bowl. And it is always PSD, for a one-line reason. For any direction \( \mathbf{v} \):

\[
\mathbf{v}^\top H \mathbf{v} = \frac{2}{n} \mathbf{v}^\top X^\top X \mathbf{v} = \frac{2}{n} (X\mathbf{v}) \cdot (X\mathbf{v}) = \frac{2}{n} \|X\mathbf{v}\|^2 \;\ge\; 0
\]

(using \( (X\mathbf{v})^\top = \mathbf{v}^\top X^\top \), Day 11). \( \mathbf{v}^\top H \mathbf{v} \) is the curvature in direction \( \mathbf{v} \) (for a unit \( \mathbf{v} \)), and a squared length can't be negative. Check it on the same Play Store setup as Day 33 (predict rating from standardized log-installs and size, plus an intercept):

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
F_std = (F - F.mean(axis=0)) / F.std(axis=0, ddof=1)
X = np.column_stack([F_std, np.ones(len(F_std))])     # (442, 3): two features + intercept
y = data["rating"].to_numpy()
n = len(y)

def mse(w):
    r = X @ w - y
    return r @ r / n

H = 2 / n * X.T @ X
print(H.round(4))
print(np.linalg.eigvalsh(H).round(4))
print(np.abs(hessian(mse, [0.5, -1.0, 3.0]) - H).max())
# [[ 1.9955  0.2548 -0.    ]
#  [ 0.2548  1.9955  0.    ]
#  [ 0.      0.      2.    ]]
# [1.7407 2.     2.2502]
# 6.661338115832027e-08
```

No internet? Download the file from the [Datasets page](../../datasets.md) and pass the local path to `read_csv`. (Tested on pandas 3.0.)

All three eigenvalues are positive, so this loss is a strictly convex bowl with exactly one best \( \mathbf{w} \), the one `np.linalg.lstsq` found on Day 33. The numerical Hessian, taken at an arbitrary point, matches the formula to \( 10^{-7} \). (That last value is a NumPy scalar. This lesson always uses `print(...)`, which shows it as `6.661338115832027e-08`; if you type the bare expression `np.abs(hessian(mse, [0.5, -1.0, 3.0]) - H).max()` as the last line of a cell, the ▶ Run button echoes its `repr()` instead, `np.float64(6.661338115832027e-08)`. Same number, different display.) (The `-0.` entries are tiny negative rounding errors: standardized columns have mean 0, so their products with the column of ones are 0.) The eigenvalues are also close to each other (condition number \( 2.25 / 1.74 \approx 1.3 \)): standardization made this bowl almost round, which is why GD on it is easy.

**Duplicate features make a flat valley.** Add a fourth column that is 2 × the size column, a perfectly collinear feature (Day 23):

```python
Xd = np.column_stack([X, 2 * X[:, 1]])            # a 4th column: 2 × the size column
Hd = 2 / n * Xd.T @ Xd
print(np.linalg.eigvalsh(Hd).round(4))

w_a = np.linalg.lstsq(Xd, y)[0]
d = np.array([0., 2., 0., -1.])                   # Xd @ d = 2·size − 2·size = 0
for s in [0, 1, 10, 100]:
    w_s = w_a + s * d
    print(s, np.round(w_s, 3).tolist(), round(float(np.mean((Xd @ w_s - y) ** 2)), 6))
# [-0.      1.955   2.     10.0178]
# 0 [0.003, 0.011, 4.168, 0.022] 0.207129
# 1 [0.003, 2.011, 4.168, -0.978] 0.207129
# 10 [0.003, 20.011, 4.168, -9.978] 0.207129
# 100 [0.003, 200.011, 4.168, -99.978] 0.207129
```

One eigenvalue is now 0: zero curvature along the direction \( \mathbf{d} \). The loss is still convex (no eigenvalue is negative), but its bottom is a **flat line**: moving along \( \mathbf{d} \) doesn't change any prediction, so weights of 200 and −100 fit exactly as well as 0.011 and 0.022. Every point on that line is a global minimum, and which one GD reaches depends on the start. This is the same singular \( X^\top X \) you met on Day 23, seen as a landscape, and it's one reason for L2 regularization (Day 22): adding \( \lambda\|\mathbf{w}\|^2 \) adds \( 2\lambda \) to every Hessian eigenvalue and makes the bowl strictly convex again.

### Logistic regression: cross-entropy

Logistic regression (Day 50's binary cross-entropy on top of a sigmoid) predicts a probability \( p = \sigma(\mathbf{x} \cdot \mathbf{w}) \). Its Hessian works out to

\[
H = \frac{1}{n} X^\top S X, \qquad S = \operatorname{diag}\big(p_i (1 - p_i)\big)
\]

Where it comes from, in two lines. The gradient of the mean cross-entropy is \( \nabla L = \frac1n X^\top (\mathbf{p} - \mathbf{y}) \) (the sigmoid and the log cancel neatly, the same "prediction minus target" shape as MSE's gradient; Day 54 derives it for the multi-class version). Differentiate once more: \( \mathbf{y} \) is fixed, and by the chain rule (Day 31) with \( \sigma' = \sigma(1 - \sigma) \), each \( p_i = \sigma(\mathbf{x}_i \cdot \mathbf{w}) \) changes as \( \partial p_i / \partial \mathbf{w} = p_i(1 - p_i)\,\mathbf{x}_i \). Stacking those rows gives \( SX \), so the Hessian is \( \frac1n X^\top S X \):
the same shape as MSE's, but with each example weighted by \( p_i(1 - p_i) \), which is between 0 and 0.25 (Day 30's \( \sigma' = \sigma(1-\sigma) \) again). Since the weights are never negative, \( \mathbf{v}^\top H \mathbf{v} = \frac{1}{n}\sum_i p_i(1-p_i)\,(\mathbf{x}_i \cdot \mathbf{v})^2 \ge 0 \): convex at **every** \( \mathbf{w} \). Unlike MSE, it changes from point to point, so test it at several random points. Make a yes/no target, "rated 4.5 or more":

```python
target = (y >= 4.5).astype(float)       # 1 = "rated 4.5 or more"; 29% of apps

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_loss(w):
    z = X @ w
    # −log σ(z) = log(1 + e^(−z)) and −log(1 − σ(z)) = log(1 + e^z), written stably
    return np.mean(target * np.logaddexp(0, -z) + (1 - target) * np.logaddexp(0, z))

def log_loss_hessian(w):
    p = sigmoid(X @ w)
    return X.T @ (X * (p * (1 - p))[:, None]) / n

rng = np.random.default_rng(53)
for _ in range(4):
    w = rng.normal(0, 2, size=3)
    H_w = log_loss_hessian(w)
    print(w.round(2), np.linalg.eigvalsh(H_w).round(4), np.abs(hessian(log_loss, w) - H_w).max() < 1e-5)
# [ 0.21 -1.61  1.78] [0.0557 0.1059 0.125 ] True
# [-1.86 -2.27  1.03] [0.0262 0.0683 0.1337] True
# [2.64 2.55 2.71] [0.0089 0.0347 0.1343] True
# [ 0.81 -0.7   1.16] [0.0902 0.1934 0.2192] True
```

Positive everywhere we look, and the formula matches the numerical Hessian (`True`). Two pieces of new code: `X * (p * (1 - p))[:, None]` multiplies each row of `X` by its own weight \( p_i(1 - p_i) \) (the `[:, None]` turns the length-442 weight vector into a column of shape `(442, 1)` so it broadcasts across the 3 columns, Day 9), which computes \( S X \) without building the \( 442 \times 442 \) diagonal matrix \( S \). And `np.logaddexp(0, z)` computes \( \log(e^0 + e^z) = \log(1 + e^z) \) in a way that never overflows.

!!! warning "Common mistake: writing the log loss the textbook way"
    The formula \( -[y \log p + (1-y)\log(1-p)] \) translated literally breaks as soon as the model is confident:

    ```python
    print(np.log(1 - sigmoid(40.0)), np.logaddexp(0, 40.0))
    # RuntimeWarning: divide by zero encountered in log
    # -inf 40.0
    ```

    \( \sigma(40) \) is so close to 1 that `1 - sigmoid(40.0)` rounds to exactly `0.0` in float64, and \( \log 0 = -\infty \). One such example makes the whole mean loss `inf` and its gradient `nan`. The true value is \( -\log(1 + e^{40}) \approx -40 \), which `logaddexp` returns. Even before it reaches `-inf`, the literal version loses digits, and that rounding noise can make a numerical second derivative look negative, so a perfectly convex loss would fail your convexity check. (The warning's exact first line, which shows the file and line number, depends on where you run it.)

### A convex loss with no minimum

Convexity promises that any minimum is global; it does not promise that a minimum **exists**. If the two classes can be split perfectly by the model (**separable** data), logistic regression can always make its predictions more confident by making the weights bigger, so the loss keeps falling towards 0 without ever reaching it, like \( e^{-x} \). Four points, one weight:

```python
xs4 = np.array([-2., -1., 1., 2.])
ys4 = np.array([0., 0., 1., 1.])          # every negative x is class 0, every positive x class 1

def ce4(w, lam=0.0):
    z = w * xs4
    return np.mean(ys4 * np.logaddexp(0, -z) + (1 - ys4) * np.logaddexp(0, z)) + lam * w**2

for lam in [0.0, 0.01]:
    w = 0.0
    for k in range(1, 10001):
        w = w - 1.0 * central_diff(lambda v: ce4(v, lam), w)
        if k in (10, 100, 1000, 10000):
            print(f"λ = {lam}   step {k:5d}   w = {w:6.3f}   loss = {ce4(w, lam):.6f}")
# λ = 0.0   step    10   w =  2.103   loss = 0.064994
# λ = 0.0   step   100   w =  4.018   loss = 0.009076
# λ = 0.0   step  1000   w =  6.231   loss = 0.000985
# λ = 0.0   step 10000   w =  8.519   loss = 0.000100
# λ = 0.01   step    10   w =  1.923   loss = 0.115780
# λ = 0.01   step   100   w =  2.423   loss = 0.105095
# λ = 0.01   step  1000   w =  2.424   loss = 0.105095
# λ = 0.01   step 10000   w =  2.424   loss = 0.105095
```

Without a penalty the weight keeps growing (2.1, 4.0, 6.2, 8.5, ...) and would grow forever: every step lowers the loss a little, so there is no point where the gradient is zero. Adding the L2 penalty \( \lambda w^2 \) (rule 2 above: still convex) makes the loss strictly convex with a real minimum, here at \( w = 2.424 \): already 2.423 at step 100, and settled by step 1000. That's the standard fix, and the reason libraries apply a small L2 penalty to logistic regression by default.

### The same sigmoid with the wrong loss is not convex

Why not use squared error for classification too, \( (\sigma(\mathbf{x}\cdot\mathbf{w}) - y)^2 \)? Take a one-weight version on six points and compare the curvature of the two losses:

```python
xs1 = np.array([-2., -1., -0.5, 0.5, 1., 2.])
ys1 = np.array([0., 0., 1., 0., 1., 1.])

def ce_1d(w):            # cross-entropy, one weight, no bias
    z = w * xs1
    return np.mean(ys1 * np.logaddexp(0, -z) + (1 - ys1) * np.logaddexp(0, z))

def sq_1d(w):            # squared error on the same sigmoid outputs
    return np.mean((sigmoid(w * xs1) - ys1) ** 2)

for w in [-6.0, -3.0, 0.0, 1.0, 3.0]:
    print(f"w = {w:4.1f}   CE'' = {second_diff(ce_1d, w):7.4f}   SQ'' = {second_diff(sq_1d, w):7.4f}")
print(looks_convex(np.vectorize(ce_1d), -8, 8, trials=20_000))
print(looks_convex(np.vectorize(sq_1d), -8, 8, trials=20_000))
# w = -6.0   CE'' =  0.0046   SQ'' = -0.0010
# w = -3.0   CE'' =  0.0308   SQ'' = -0.0245
# w =  0.0   CE'' =  0.4375   SQ'' =  0.2187
# w =  1.0   CE'' =  0.2251   SQ'' =  0.1001
# w =  3.0   CE'' =  0.0308   SQ'' = -0.0065
# (True, 0.0)
# (False, -0.2738)
```

(`looks_convex` passes whole arrays of 20,000 weights at once, but `ce_1d` expects one number, because `w * xs1` must be a weight times the six data points. `np.vectorize(ce_1d)` wraps it so that it calls `ce_1d` once per weight in a Python loop and collects the results into an array. It's a convenience, not a speed-up.) Cross-entropy curves up everywhere; squared error on a sigmoid curves **down** far from the middle. That is one of the two big reasons classification uses cross-entropy (the other, from Day 47, is that it's the maximum-likelihood loss). The next section shows what that downward bend does to gradient descent.

## Plateaus: where the slope almost vanishes

Look at the slopes of both losses at a bad starting weight, \( w = -6 \) (the model is confidently wrong on every point):

```python
for w in [-6.0, -3.0, 0.0]:
    print(f"w = {w:4.1f}   CE slope = {central_diff(ce_1d, w):8.5f}   SQ slope = {central_diff(sq_1d, w):8.5f}")
# w = -6.0   CE slope = -0.99127   SQ slope = -0.00093
# w = -3.0   CE slope = -0.95214   SQ slope = -0.02290
# w =  0.0   CE slope = -0.41667   SQ slope = -0.20833
```

At \( w = -6 \) the squared-error slope is a thousand times smaller. The sigmoid is **saturated** there (its output is pinned near 0 or 1, and \( \sigma' \approx 0 \)), and the squared error multiplies by that tiny \( \sigma' \) through the chain rule (Day 31). Cross-entropy's \( \log \) cancels it, so its slope stays large exactly when the model is badly wrong. The squared-error loss is on a **plateau**: not a minimum, only nearly flat. Run gradient descent on both from \( w = -6 \):

```python
def gd_history(L, w0, lr=0.5, steps=3000):
    w, ws = w0, []
    for _ in range(steps):
        w = w - lr * central_diff(L, w)
        ws.append(w)
    return np.array(ws)

ws_ce = gd_history(ce_1d, -6.0)
ws_sq = gd_history(sq_1d, -6.0)
for k in [10, 100, 1000, 2000, 2500, 3000]:
    w_c, w_s = ws_ce[k - 1], ws_sq[k - 1]
    print(f"step {k:4d}   CE: w = {w_c:6.3f}, loss {ce_1d(w_c):.4f}   SQ: w = {w_s:6.3f}, loss {sq_1d(w_s):.4f}")
# step   10   CE: w = -1.220, loss 1.4787   SQ: w = -5.995, loss 0.6658
# step  100   CE: w =  1.350, loss 0.4607   SQ: w = -5.952, loss 0.6657
# step 1000   CE: w =  1.351, loss 0.4607   SQ: w = -5.357, loss 0.6649
# step 2000   CE: w =  1.351, loss 0.4607   SQ: w = -1.682, loss 0.5786
# step 2500   CE: w =  1.351, loss 0.4607   SQ: w =  1.033, loss 0.1579
# step 3000   CE: w =  1.351, loss 0.4607   SQ: w =  1.033, loss 0.1579
```

Cross-entropy reaches its minimum (\( w \approx 1.35 \)) in under 100 steps. Squared error spends about **2000 steps** on the plateau, its loss stuck at 0.665 so that it looks converged, then rushes down to its own minimum (\( w \approx 1.03 \)). (The two losses measure different things, so compare the steps, not the loss values.) A plateau is the most deceptive spot in training: the loss curve is flat, but you are nowhere near a minimum.

**Underneath.** Each GD step moves the weight by \( \eta \cdot |\text{slope}| \). At \( w = -6 \) that's \( 0.5 \times 0.00093 = 0.00047 \): it would take over 10,000 steps of that size to cover the distance to the minimum. Steps grow only as the slope slowly grows, which is why the orange curve in the figure below creeps first and then shoots up.

!!! warning "Common mistake: a stopping rule that a plateau fools"
    A common stopping rule (Day 51) is "stop when the loss changes by less than a tolerance". On a plateau it fires immediately:

    ```python
    w, prev = -6.0, sq_1d(-6.0)
    for step in range(1, 3001):
        w = w - 0.5 * central_diff(sq_1d, w)
        loss_now = sq_1d(w)
        if abs(prev - loss_now) < 1e-6:       # "the loss stopped changing"
            break
        prev = loss_now
    print(step, round(w, 4), round(loss_now, 4))
    # 1 -5.9995 0.6658
    ```

    It stopped after **one** step, at \( w = -6.0 \), with the loss at 0.666 when 0.158 was reachable. Fixes: require the condition to hold for many steps in a row, also look at the gradient size and the validation metric, and above all plot the loss curve: a plateau followed by a drop is a familiar shape once you've seen it.

## Saddle points

The textbook saddle is

\[
z = x^2 - y^2
\]

Along the \( x \)-axis (\( y = 0 \)) it's \( z = x^2 \), a valley. Along the \( y \)-axis it's \( z = -y^2 \), a hill. At the origin the gradient \( [2x, -2y] \) is exactly zero, so it's a flat point, but it is neither a minimum nor a maximum. The Hessian you computed above, eigenvalues \( +2 \) and \( -2 \), says exactly that: curving up one way, down the other.

What does gradient descent do there? The update \( \mathbf{p} \leftarrow \mathbf{p} - \eta \nabla z \) with \( \eta = 0.1 \) works on each coordinate separately:

- \( x \leftarrow x - 0.1 \cdot 2x = 0.8\,x \): \( x \) shrinks by 20% per step, towards the saddle.
- \( y \leftarrow y - 0.1 \cdot (-2y) = 1.2\,y \): \( y \) **grows** by 20% per step, away from it.

Start at \( (1, 0) \), exactly on the valley line, and \( y \) stays 0 forever: GD walks straight into the saddle and stops, with a perfectly zero gradient. Start at \( (1, 0.000001) \) and the tiny \( y \) grows by \( 1.2^k \):

```python
def saddle_grad(p):
    return np.array([2 * p[0], -2 * p[1]])

def walk(p0, lr=0.1, steps=80):
    p = np.array(p0, dtype=float)
    path = [p]
    for _ in range(steps):
        p = p - lr * saddle_grad(p)
        path.append(p)
    return np.array(path)

for p0 in [(1.0, 0.0), (1.0, 1e-6)]:
    path = walk(p0)
    print(f"start {p0}")
    for k in [0, 10, 30, 50, 70, 76]:
        g = np.linalg.norm(saddle_grad(path[k]))
        print(f"  step {k:2d}: point ({path[k][0]:.2e}, {path[k][1]:.2e})   |grad| = {g:.2e}")
# start (1.0, 0.0)
#   step  0: point (1.00e+00, 0.00e+00)   |grad| = 2.00e+00
#   step 10: point (1.07e-01, 0.00e+00)   |grad| = 2.15e-01
#   step 30: point (1.24e-03, 0.00e+00)   |grad| = 2.48e-03
#   step 50: point (1.43e-05, 0.00e+00)   |grad| = 2.85e-05
#   step 70: point (1.65e-07, 0.00e+00)   |grad| = 3.29e-07
#   step 76: point (4.31e-08, 0.00e+00)   |grad| = 8.63e-08
# start (1.0, 1e-06)
#   step  0: point (1.00e+00, 1.00e-06)   |grad| = 2.00e+00
#   step 10: point (1.07e-01, 6.19e-06)   |grad| = 2.15e-01
#   step 30: point (1.24e-03, 2.37e-04)   |grad| = 2.52e-03
#   step 50: point (1.43e-05, 9.10e-03)   |grad| = 1.82e-02
#   step 70: point (1.65e-07, 3.49e-01)   |grad| = 6.98e-01
#   step 76: point (4.31e-08, 1.04e+00)   |grad| = 2.08e+00
```

The two runs are almost identical for the first 30 steps (at step 30 the gradient sizes are 2.48e-03 and 2.52e-03): the gradient shrinks to about \( 1/800 \) of its starting size, and anyone watching would say "converged". Then the second run's \( y \), which grew from \( 10^{-6} \) by \( 1.2 \) per step, takes over. You can predict when: \( 10^{-6} \cdot 1.2^k > 1 \) when \( k > \log(10^6) / \log(1.2) \approx 75.8 \), so at step 76:

```python
print(round(1e-6 * 1.2**76, 4), round(np.log(1e6) / np.log(1.2), 1))
# 1.0418 75.8
```

<figure class="diagram" markdown>
![Animation: a contour map of z = x^2 − y^2 with blue hyperbolas opening left and right where z is positive and orange hyperbolas opening up and down where z is negative, crossed by two dashed diagonals; a red ball starts at (1, 0) on the right, steps left along the x-axis leaving teal dots that bunch up near the centre, then turns and climbs straight up the y-axis to the top edge; a table on the right fills in step, x, y and gradient size for steps 0 to 76, with the tiny gradients at steps 30 and 40 highlighted in red](../../assets/img/phase-1/day53-saddle-walk.svg)
<figcaption>Animated: gradient descent near a saddle. It rolls in along the valley direction, nearly stops at the flat centre, then escapes along the downhill direction.</figcaption>
</figure>

!!! warning "Common mistake: stopping on a small gradient and calling it a minimum"
    Use the rule "stop when the gradient's length drops below 0.005" on the run from \( (1, 10^{-6}) \), then ask the Hessian what kind of point you stopped at:

    ```python
    p = np.array([1.0, 1e-6])
    for step in range(1, 1001):
        p = p - 0.1 * saddle_grad(p)
        if np.linalg.norm(saddle_grad(p)) < 0.005:
            break
    print(step, p, saddle(p))
    print(np.linalg.eigvalsh(hessian(saddle, p)).round(4))
    # 27 [0.00241785 0.00013737] 5.82713588077577e-06
    # [-2.  2.]
    ```

    The rule stopped at step 27, near the saddle, and one Hessian eigenvalue is −2: there is a direction in which the loss still goes down. A small gradient means "flat", and flat can be a saddle. For small models you can check the Hessian's smallest eigenvalue; for big ones, keep training (noise and momentum carry you past) and judge by the loss curve over many steps.

Two lessons. A saddle is not a trap for gradient descent in practice: the tiniest component along a downhill direction grows exponentially, and real training always has such a component (random initialization, mini-batch noise from Day 52). But it is a **slowdown**: the closer you pass to the saddle, the longer you crawl. With \( y_0 = 10^{-12} \) the escape would take 152 steps instead of 76. Momentum (Day 52) helps, because the velocity it builds up rolling in keeps it moving across the flat region, and a little noise rescues even a start that sits exactly on the valley line:

```python
def steps_to_escape(p0, lr=0.1, beta=0.0, noise=0.0, seed=0, max_steps=1000):
    """GD with momentum beta (0 = plain GD) and optional gradient noise; steps until |y| > 1."""
    rng = np.random.default_rng(seed)
    p, v = np.array(p0, dtype=float), np.zeros(2)
    for k in range(1, max_steps + 1):
        g = saddle_grad(p) + noise * rng.normal(size=2)   # noise imitates mini-batch gradients
        v = beta * v - lr * g                              # Day 52's velocity update
        p = p + v
        if abs(p[1]) > 1:
            return k
    return None                                            # never escaped

print(steps_to_escape((1, 1e-6)), steps_to_escape((1, 1e-6), beta=0.5), steps_to_escape((1, 1e-6), beta=0.9))
print(steps_to_escape((1, 0)), steps_to_escape((1, 0), noise=1e-6, seed=53))
# 76 51 36
# None 94
```

Plain GD needs 76 steps from \( (1, 10^{-6}) \); momentum 0.5 needs 51 and momentum 0.9 needs 36, because the growing \( y \)-velocity compounds on top of the gradient. From exactly \( (1, 0) \) plain GD never escapes (`None`, the function's way of saying "not within 1000 steps"), but gradient noise of size \( 10^{-6} \), far smaller than anything you'd notice, gets it out in 94 steps.

<figure class="diagram" markdown>
![Two side-by-side charts. Left: the size of the gradient on a log scale against step for the saddle run, falling in a straight line from 2 to about 0.001 around step 35, inside a yellow band labelled near the saddle, then rising in a straight line to about 4 by step 80. Right: the weight w against step for two losses started at w = −6: the teal cross-entropy line jumps to 1.35 within a few dozen steps, while the orange squared-error line creeps slowly upward for about 2000 steps, then shoots up to 1.03 and stays there](../../assets/img/phase-1/day53-stalls.svg)
<figcaption>Two ways to stall: near a saddle the gradient becomes tiny for a while (left); on a plateau the weight barely moves for 2000 steps before it finally finds the slope (right).</figcaption>
</figure>

Here is the code that drew it. Two matplotlib calls are new since Day 19: `ax.semilogy` is `plot` with a logarithmic y-axis (each grid line is 10× the one below, so an exponential decay becomes a straight line), and `ax.axvspan(25, 45, ...)` shades the vertical band between x = 25 and x = 45.

```python
grad_norms = np.array([np.linalg.norm(saddle_grad(p)) for p in walk((1.0, 1e-6))])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
ax1.semilogy(grad_norms, color="#e53935", lw=2)
ax1.set_xlabel("step")
ax1.set_ylabel("|gradient| (log scale)")
ax1.set_title("Saddle x² − y²: slows down, then escapes")
ax1.axvspan(25, 45, color="#fff59d", alpha=0.6)
ax1.text(35, 1.2e-2, "near the\nsaddle", ha="center", fontsize=9, color="#555")
ax1.grid(alpha=0.3, which="both")
steps = np.arange(1, len(ws_ce) + 1)
ax2.plot(steps, ws_ce, color="#00897b", lw=2, label="cross-entropy")
ax2.plot(steps, ws_sq, color="#ef6c00", lw=2, label="squared error on σ")
ax2.set_xlabel("step")
ax2.set_ylabel("the weight w")
ax2.set_title("Plateau: same start w = −6, two losses")
ax2.grid(alpha=0.3, which="both")
ax2.legend(fontsize=9, loc="center right")
fig.tight_layout()
fig.savefig("day53-stalls.svg", format="svg", bbox_inches="tight")
plt.close(fig)
```

### Why saddles dominate in high dimensions

In 2-D, a flat point is a minimum, a maximum or a saddle, depending on the signs of two eigenvalues. A neural network with a million parameters has a million Hessian eigenvalues at every flat point, and it's a minimum only if **all** of them are positive. As a rough intuition: if each sign were a coin flip, the chance of all \( k \) being positive would be \( 0.5^k \), which for \( k = 1{,}000{,}000 \) is zero for every practical purpose. Almost every flat point would be a saddle.

The signs aren't really independent coin flips, so here is a slightly better experiment: generate random symmetric matrices as stand-in Hessians and count how often all eigenvalues are positive.

```python
rng = np.random.default_rng(53)

def fraction_all_positive(dim, trials=20_000):
    A = rng.normal(size=(trials, dim, dim))
    S = (A + A.transpose(0, 2, 1)) / 2          # 20,000 random symmetric "Hessians"
    eigs = np.linalg.eigvalsh(S)                # shape (trials, dim)
    return (eigs > 0).all(axis=1).mean()

for dim in [1, 2, 3, 4, 5, 6, 8]:
    print(f"dim {dim}: minima {fraction_all_positive(dim):.5f}   coin-flip guess {0.5**dim:.5f}")
# dim 1: minima 0.50425   coin-flip guess 0.50000
# dim 2: minima 0.14500   coin-flip guess 0.25000
# dim 3: minima 0.02515   coin-flip guess 0.12500
# dim 4: minima 0.00235   coin-flip guess 0.06250
# dim 5: minima 0.00010   coin-flip guess 0.03125
# dim 6: minima 0.00000   coin-flip guess 0.01562
# dim 8: minima 0.00000   coin-flip guess 0.00391
```

(`A` has shape `(20000, dim, dim)`: a stack of 20,000 square matrices. `A.transpose(0, 2, 1)` keeps axis 0, the stack, and swaps the last two, which transposes each matrix; averaging a matrix with its transpose makes it symmetric. `eigvalsh` works on the whole stack at once and returns one row of `dim` eigenvalues per matrix. `(eigs > 0)` is a True/False array of the same shape, `.all(axis=1)` is True for the rows where every entry is True, and `.mean()` of True/False values counts True as 1 and False as 0, so it gives the fraction of matrices that would be minima.) For random symmetric matrices, "all positive" becomes rare even faster than the coin-flip guess: by 6 dimensions not one of 20,000 qualified.

Be careful with what this shows. Real Hessians are not random matrices, so this is an intuition pump, not a proof. Research on real networks (Dauphin et al., 2014, in the resources) supports the picture: in high dimensions, flat points with **high** loss are overwhelmingly saddles, and the flat points that are true minima tend to have low loss. So the classic fear, "GD gets stuck in a bad local minimum", turns out to be less of a problem in big networks than "GD slows down near saddles and on plateaus".

!!! warning "Common mistake: assuming a zero gradient means a minimum"
    \( \nabla L = \mathbf{0} \) only says the point is flat: minimum, maximum or saddle. To tell which, you need second-order information (the Hessian's eigenvalue signs). In 1,000+ dimensions, a random flat point is almost certainly a saddle. Libraries never compute full Hessians for big networks (a million parameters means \( 10^{12} \) entries), which is another reason to rely on noise and momentum rather than on careful analysis of each flat point.

!!! android "Android bridge"
    You already pick convex and concave curves every day: animation **interpolators** map time \( t \in [0, 1] \) to progress. `AccelerateInterpolator()` (default factor 1) returns \( t^2 \), a convex curve; `DecelerateInterpolator()` (default factor 1) returns \( 1 - (1 - t)^2 \), a concave one.

    ```kotlin
    AccelerateInterpolator().getInterpolation(0.5f)        // 0.25: below the straight line
    DecelerateInterpolator().getInterpolation(0.5f)        // 0.75: above it
    AccelerateDecelerateInterpolator().getInterpolation(0.25f)  // cos(1.25π)/2 + 0.5 = 0.146 (3 decimals)
    ```

    The chord test is visible on screen. The chord from \( (0, 0) \) to \( (1, 1) \) is `LinearInterpolator`. A convex easing curve lies on or below that chord, so an accelerating animation is **behind** a linear one at every instant; a concave one is ahead. `AccelerateDecelerateInterpolator` is convex for the first half and concave for the second: its second derivative changes sign at \( t = 0.5 \), an **inflection point**, exactly like the two-valley function's curvature changing sign at \( \pm 0.707 \).

    The basin idea shows up in on-device learning too. When you personalize a model on the phone (LiteRT, formerly TensorFlow Lite, supports on-device training through model signatures), you start from pretrained weights rather than random ones. On a non-convex loss the start decides which valley you end in, and a pretrained start is already in a good one, so a few cheap steps on the device stay there.

## Why deep learning works anyway

Neural network losses are **not** convex, and you can prove it in a few lines. Take a tiny network, 2 inputs → 3 tanh units → 1 output, and make its targets with its own weights, so those weights have loss exactly 0. Now relabel the hidden units (unit 0 becomes unit 1, and so on, moving each unit's incoming and outgoing weights together). The network computes exactly the same function, so the loss is still 0, but the weight vector is different. A convex function that is 0 at two points and never negative must be 0 everywhere on the segment between them. Check the midpoint:

```python
rng = np.random.default_rng(53)
Xn = rng.normal(size=(200, 2))
W1, b1, w2 = rng.normal(size=(2, 3)), rng.normal(size=3), rng.normal(size=3)

def net(W1, b1, w2):
    return np.tanh(Xn @ W1 + b1) @ w2           # 2 inputs → 3 tanh units → 1 output

y_teacher = net(W1, b1, w2)                     # targets made by these very weights

def net_loss(W1, b1, w2):
    return np.mean((net(W1, b1, w2) - y_teacher) ** 2)

perm = [2, 0, 1]                                # relabel the hidden units
W1p, b1p, w2p = W1[:, perm], b1[perm], w2[perm]
print(net_loss(W1, b1, w2), net_loss(W1p, b1p, w2p))
print(np.abs(W1 - W1p).max())
for t in [0, 0.25, 0.5, 0.75, 1]:
    mix = net_loss((1 - t) * W1 + t * W1p, (1 - t) * b1 + t * b1p, (1 - t) * w2 + t * w2p)
    print(f"t = {t:4.2f}   loss = {mix:.4f}")
# 0.0 5.687271125775508e-32
# 1.5356329187011726
# t = 0.00   loss = 0.0000
# t = 0.25   loss = 1.5681
# t = 0.50   loss = 2.6399
# t = 0.75   loss = 1.0741
# t = 1.00   loss = 0.0000
```

`W1[:, perm]` is fancy indexing (Day 8): it builds a new matrix whose columns are columns 2, 0 and 1 of `W1`. Column \( j \) of `W1` holds the incoming weights of hidden unit \( j \), so reordering the columns relabels the units; using the same `perm` on the biases `b1` and the outgoing weights `w2` keeps each unit's three sets of weights together. (The \( 5.7 \times 10^{-32} \) is 0 up to rounding: the sums happen in a different order.) Two perfect minima, a hill of height 2.64 between them: the chord test fails, so the loss is not convex. And this isn't a rare accident: with \( h \) hidden units there are \( h! \) such relabellings (\( 16! \approx 2 \times 10^{13} \) for a layer of 16), so every minimum comes with a huge family of copies, separated by hills.

So why does training work at all? The honest answer is that there is no complete theory yet, but several things are known:

- **Saddles, not bad minima, are the main obstacle**, and GD escapes saddles (the experiment above), especially with mini-batch noise and momentum (Day 52).
- **Many minima are equally good.** The permutation copies have *identical* loss, so landing in any of them is fine. Experiments on large networks find that different runs from different seeds usually reach different weights with very similar loss.
- **Over-parameterization helps.** Networks with far more parameters than training examples tend to have big, connected regions of low loss that are easy to fall into.
- **Design choices shape the landscape.** Good initialization, ReLU instead of saturating sigmoids (avoiding the plateaus above), normalization layers and residual connections all make the loss surface smoother; visualizations of real networks (Li et al., 2018, in the resources) show residual connections turning a chaotic surface into a much more bowl-like one.
- **We don't want the global minimum anyway.** The global minimum of the *training* loss often memorizes noise. What matters is performance on new data, which is Month 3's topic.

None of this is a guarantee. Sometimes training does get stuck, diverges or lands somewhere worse, which is why practitioners try several seeds and learning rates and watch the loss curves.

## Going further

These are real and useful, but beyond what the next month needs. Each is one line here so you know it exists.

- **Strong convexity and convergence rates**: if \( f'' \ge m > 0 \) everywhere, gradient descent's error shrinks by a fixed factor every step, and the factor depends on the condition number. See the strong convexity part of [Wikipedia: Convex function](https://en.wikipedia.org/wiki/Convex_function).
- **Jensen's inequality**: the chord test for averages, \( f(\mathbb{E}[X]) \le \mathbb{E}[f(X)] \) for convex \( f \); it's behind several proofs in information theory (Day 54) and variational methods. [Wikipedia: Jensen's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality).
- **Subgradients**: how to do gradient descent on convex functions with corners, like \( |x| \) (MAE) or the hinge loss. [Wikipedia: Subderivative](https://en.wikipedia.org/wiki/Subderivative).
- **Newton's method**: use the Hessian itself to jump towards the minimum, \( \mathbf{w} \leftarrow \mathbf{w} - H^{-1}\nabla L \); fast for small convex problems such as logistic regression. [Wikipedia: Newton's method in optimization](https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization).
- **Hessian-vector products**: computing \( H\mathbf{v} \) without building \( H \), which is how people measure the curvature of big networks. [PyTorch docs: `torch.autograd.functional.hvp`](https://pytorch.org/docs/stable/generated/torch.autograd.functional.hvp.html).
- **Convex optimization solvers**: for problems you can state as convex (portfolio weights, constrained regression), a solver finds the guaranteed optimum. [CVXPY](https://www.cvxpy.org/).
- **Mode connectivity**: the minima found by different training runs of a big network are often joined by curved paths of low loss. [Garipov et al. (2018): Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs](https://arxiv.org/abs/1802.10026).

---

## Exercises

Every solution below is self-contained: it imports NumPy and re-creates its own helpers and data, so you can paste it into a fresh Python session.

**Warm-ups**

1. Classify each function as convex or not on the given interval. First find \( f'' \) by hand and look at its sign; then confirm with the chord test and `second_diff`: softplus \( \log(1 + e^x) \) on \( [-3, 3] \); \( x^3 \) on \( [-2, 2] \); \( x^3 \) on \( [0, 2] \); \( -\log x \) on \( [0.1, 3] \).
2. By hand, write the Hessians of \( g_1(x, y) = x^2 + xy + y^2 \) and \( g_2(x, y) = x^2 + 3xy + y^2 \). Find their eigenvalues with `np.linalg.eigvalsh`. Both have a zero gradient at the origin: what kind of point is it for each? Confirm \( g_2 \)'s answer by evaluating it at \( (0, 0) \), \( (1, 1) \) and \( (1, -1) \).

**Core**

3. Run gradient descent (\( \eta = 0.05 \), 200 steps) on \( f(x) = x^4 - 3x^2 + x \) from 41 starts evenly spaced in \( [-2, 2] \), using the exact slope \( f'(x) = 4x^3 - 6x + 1 \). How many starts reach the global minimum? Between which two neighbouring starts is the watershed, and does that match the local maximum at \( x = 0.1699 \)?
4. Logistic regression's convexity on fresh data: make a seeded synthetic dataset of 300 points with 2 features plus an intercept column and a 0/1 target. At 200 random weight vectors, compute the smallest eigenvalue of the numerical Hessian of (a) the cross-entropy loss and (b) squared error on the sigmoid. What do the smallest values you saw tell you?
5. On the saddle \( x^2 - y^2 \), each GD step multiplies \( y \) by \( 1 + 2\eta \). Predict, with a logarithm, how many steps it takes to go from \( y = 10^{-9} \) to \( |y| > 1 \) with \( \eta = 0.1 \) and with \( \eta = 0.2 \). Then check by running GD from \( (1, 10^{-9}) \).

**Predict the output**

6. The bowl \( F(\mathbf{p}) = \tfrac12 \mathbf{p}^\top H \mathbf{p} \) with \( H = \begin{bmatrix} 0.5 & 0 \\ 0 & 2 \end{bmatrix} \) has gradient \( H\mathbf{p} \), so one GD step multiplies the first coordinate by \( 1 - 0.5\eta \) and the second by \( 1 - 2\eta \). Before running it, predict for \( \eta = 0.5 \), \( 0.9 \) and \( 1.1 \): after 20 steps from \( (1, 1) \), which coordinates are near 0, and does any of them blow up? Write your guesses down, then run:

    ```python
    import numpy as np

    H = np.array([[0.5, 0.0], [0.0, 2.0]])
    print(np.linalg.eigvalsh(H))

    def grad(p):
        return H @ p

    for lr in [0.5, 0.9, 1.1]:
        p = np.array([1.0, 1.0])
        for _ in range(20):
            p = p - lr * grad(p)
        print(lr, p.round(4))
    ```

**Find and fix the bug**

7. This convexity checker says that \( x^2 \), the textbook convex function, is **not** convex. Find the bug, explain the wrong number it prints for \( a = -1, b = 3, t = 0.25 \), and fix it.

    ```python
    import numpy as np

    def chord_gap(f, a, b, t):
        x = t * a + (1 - t) * b
        return (1 - t) * f(a) + t * f(b) - f(x)

    def looks_convex(f, lo, hi, trials=100_000, seed=53):
        rng = np.random.default_rng(seed)
        a = rng.uniform(lo, hi, trials)
        b = rng.uniform(lo, hi, trials)
        t = rng.uniform(0, 1, trials)
        worst = chord_gap(f, a, b, t).min()
        return bool(worst >= -1e-9), round(float(worst), 4)

    print(looks_convex(lambda x: x**2, -2, 2))
    print(chord_gap(lambda x: x**2, -1, 3, 0.25))
    # (False, -3.9942)
    # -1.0
    ```

**Challenge**

8. tanh is an **odd** function: \( \tanh(-z) = -\tanh(z) \). Use that to find a second symmetry of the tiny 2 → 3 → 1 tanh network from "Why deep learning works anyway": flip the signs of one hidden unit's incoming weights, its bias and its outgoing weight. Show that the loss is unchanged, show that the midpoint between the original and the flipped weights has a higher loss, and count how many equivalent weight settings a layer of \( h \) tanh units has (relabellings and sign flips together), for \( h = 3 \), 16 and 128.

??? success "Solutions"
    **1.** Softplus: \( f' = \sigma(x) \), so \( f'' = \sigma(x)(1 - \sigma(x)) > 0 \): convex everywhere. \( x^3 \): \( f'' = 6x \), negative for \( x < 0 \), so not convex on \( [-2, 2] \) but convex on \( [0, 2] \). \( -\log x \): \( f' = -1/x \), \( f'' = 1/x^2 > 0 \): convex.

    ```python
    import numpy as np

    def second_diff(f, x, h=1e-4):
        return (f(x + h) - 2 * f(x) + f(x - h)) / h**2

    def chord_gap(f, a, b, t):
        x = (1 - t) * a + t * b
        return (1 - t) * f(a) + t * f(b) - f(x)

    def looks_convex(f, lo, hi, trials=100_000, seed=53):
        rng = np.random.default_rng(seed)
        a = rng.uniform(lo, hi, trials)
        b = rng.uniform(lo, hi, trials)
        t = rng.uniform(0, 1, trials)
        worst = chord_gap(f, a, b, t).min()
        return bool(worst >= -1e-9), round(float(worst), 4)

    tests = [("softplus", lambda x: np.logaddexp(0, x), -3, 3),
             ("x³", lambda x: x**3, -2, 2),
             ("x³ on [0, 2]", lambda x: x**3, 0, 2),
             ("−log x", lambda x: -np.log(x), 0.1, 3)]
    for name, fn, lo, hi in tests:
        pts = np.linspace(lo + 0.01, hi - 0.01, 5)
        print(f"{name:12s} chord test {looks_convex(fn, lo, hi)}   f'' at 5 points {second_diff(fn, pts).round(2)}")
    # softplus     chord test (True, 0.0)   f'' at 5 points [0.05 0.15 0.25 0.15 0.05]
    # x³           chord test (False, -3.9477)   f'' at 5 points [-11.94  -5.97   0.     5.97  11.94]
    # x³ on [0, 2] chord test (True, 0.0)   f'' at 5 points [ 0.06  3.03  6.    8.97 11.94]
    # −log x       chord test (True, 0.0)   f'' at 5 points [82.64  1.45  0.42  0.19  0.11]
    ```

    The two tests agree every time, as they must for smooth functions: a negative \( f'' \) somewhere (the left half for \( x^3 \)) is exactly where a chord can dip below the curve. Note the softplus curvature, at most 0.25 at \( x = 0 \): it is the smooth version of ReLU, and its \( f'' \) is the sigmoid derivative of Day 30. \( -\log x \) has huge curvature near 0 (82.6 at \( x = 0.11 \)), so \( \log x \) is concave.

    **2.** \( g_1 \): \( \partial g_1/\partial x = 2x + y \) and \( \partial g_1/\partial y = x + 2y \); differentiating each again gives \( H_1 = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} \). In the same way \( H_2 = \begin{bmatrix} 2 & 3 \\ 3 & 2 \end{bmatrix} \).

    ```python
    import numpy as np

    H1 = np.array([[2., 1.], [1., 2.]])
    H2 = np.array([[2., 3.], [3., 2.]])
    print(np.linalg.eigvalsh(H1), np.linalg.eigvalsh(H2))

    def g2(x, y):
        return x**2 + 3 * x * y + y**2

    print(g2(0, 0), g2(1, 1), g2(1, -1))
    # [1. 3.] [-1.  5.]
    # 0 5 -1
    ```

    \( H_1 \)'s eigenvalues are both positive at every point (the Hessian is constant), so \( g_1 \) is convex and the origin is its global minimum. \( H_2 \) has one negative eigenvalue, so the origin is a saddle of \( g_2 \): it goes up along \( [1, 1] \) (\( g_2(1, 1) = 5 \)) and down along \( [1, -1] \) (\( g_2(1, -1) = -1 \)), even though both diagonal entries are positive.

    **3.**

    ```python
    import numpy as np

    def wavy_slope(x):
        return 4 * x**3 - 6 * x + 1          # f'(x), by the power rule

    def gd_1d(slope, x0, lr=0.05, steps=200):
        x = x0
        for _ in range(steps):
            x = x - lr * slope(x)
        return x

    starts = np.linspace(-2, 2, 41)          # -2.0, -1.9, ..., 2.0
    ends = np.array([gd_1d(wavy_slope, x0) for x0 in starts])
    left = starts[ends < 0]                  # starts that reached the global minimum
    right = starts[ends > 0]
    print(left.max().round(2), right.min().round(2))
    print(len(left), len(right))
    print(np.unique(ends.round(4)))
    # 0.1 0.2
    # 22 19
    # [-1.3008  1.1309]
    ```

    22 of the 41 starts (every start up to 0.1) reach the global minimum at −1.3008; the other 19 (from 0.2 up) end at the local minimum 1.1309, and `np.unique` confirms there is nowhere else to end. The watershed lies between 0.1 and 0.2, where the local maximum 0.1699 sits: a start's slope sign decides which way it first moves, and on each side of the hilltop the slope points away from it.

    **4.**

    ```python
    import numpy as np

    rng = np.random.default_rng(8)
    X = np.column_stack([rng.normal(size=(300, 2)), np.ones(300)])   # 2 features + intercept
    target = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 1, 300) > 0).astype(float)

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def hessian(F, p, h=1e-4):
        p = np.asarray(p, dtype=float)
        k = p.size
        E = np.eye(k) * h
        H = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                H[i, j] = (F(p + E[i] + E[j]) - F(p + E[i] - E[j])
                           - F(p - E[i] + E[j]) + F(p - E[i] - E[j])) / (4 * h * h)
        return H

    def log_loss(w):
        z = X @ w
        return np.mean(target * np.logaddexp(0, -z) + (1 - target) * np.logaddexp(0, z))

    def sq_sigmoid(w):
        return np.mean((sigmoid(X @ w) - target) ** 2)

    lowest_ce, lowest_sq = np.inf, np.inf
    for _ in range(200):
        w = rng.normal(0, 3, size=3)
        lowest_ce = min(lowest_ce, np.linalg.eigvalsh(hessian(log_loss, w))[0])
        lowest_sq = min(lowest_sq, np.linalg.eigvalsh(hessian(sq_sigmoid, w))[0])
    print(f"smallest eigenvalue seen, cross-entropy: {lowest_ce:.5f}")
    print(f"smallest eigenvalue seen, squared error: {lowest_sq:.5f}")
    # smallest eigenvalue seen, cross-entropy: 0.00015
    # smallest eigenvalue seen, squared error: -0.07288
    ```

    The target is "a noisy weighted sum of the features is positive", a typical yes/no label. Cross-entropy's smallest eigenvalue never went below zero (it gets close to 0 where the model is very confident and \( p(1 - p) \) is tiny, but \( \frac{1}{n}X^\top S X \) can't be negative). Squared error on a sigmoid reached −0.073: a direction of downward curvature, so that loss is not convex on this data, and gradient descent on it can meet plateaus and humps that cross-entropy doesn't have.

    **5.** We need \( (1 + 2\eta)^k \cdot 10^{-9} > 1 \), so \( k > \log(10^9) / \log(1 + 2\eta) \), rounded up. For \( \eta = 0.1 \): \( 20.72 / 0.1823 = 113.7 \), so 114. For \( \eta = 0.2 \): \( 20.72 / 0.3365 = 61.6 \), so 62.

    ```python
    import math
    import numpy as np

    def saddle_grad(p):
        return np.array([2 * p[0], -2 * p[1]])

    for lr in [0.1, 0.2]:
        growth = 1 + 2 * lr
        predicted = math.ceil(math.log(1e9) / math.log(growth))   # ceil: round up
        p, k = np.array([1.0, 1e-9]), 0
        while abs(p[1]) <= 1:
            p = p - lr * saddle_grad(p)
            k += 1
        print(lr, growth, predicted, k)
    # 0.1 1.2 114 114
    # 0.2 1.4 62 62
    ```

    The prediction is exact because the \( y \)-update never depends on \( x \): it is pure repeated multiplication. The escape time grows with the **logarithm** of how close to the ridge you start: \( 10^{-9} \) is 1000 times closer than the lesson's \( 10^{-6} \), yet costs only 38 more steps (114 against 76).

    **6.** Prediction: the factors are \( (0.75, 0) \) for \( \eta = 0.5 \), \( (0.55, -0.8) \) for \( \eta = 0.9 \) and \( (0.45, -1.2) \) for \( \eta = 1.1 \). After 20 steps: \( 0.75^{20} = 0.0032 \) and exactly 0 for the first; about 0 and \( 0.8^{20} = 0.0115 \) for the second; about 0 and \( 1.2^{20} = 38.3 \), blown up, for the third.

    ```python
    import numpy as np

    H = np.array([[0.5, 0.0], [0.0, 2.0]])
    print(np.linalg.eigvalsh(H))

    def grad(p):
        return H @ p

    for lr in [0.5, 0.9, 1.1]:
        p = np.array([1.0, 1.0])
        for _ in range(20):
            p = p - lr * grad(p)
        print(lr, p.round(4))
    # [0.5 2. ]
    # 0.5 [0.0032 0.    ]
    # 0.9 [0.     0.0115]
    # 1.1 [ 0.     38.3376]
    ```

    Each coordinate is multiplied by \( 1 - \eta\lambda \), where \( \lambda \) is its curvature (Hessian eigenvalue). It shrinks only if \( |1 - \eta\lambda| < 1 \), that is \( \eta < 2/\lambda \). The steep direction (\( \lambda = 2 \)) sets the limit \( \eta < 1 \): at 0.9 it overshoots back and forth (factor −0.8) but still settles, at 1.1 it overshoots further every step and diverges, while the gentle direction is happy at every rate tried. That is the "\( 2/\lambda_{\max} \)" rule from the Hessian section.

    **7.** The bug is in the first line of `chord_gap`: the point on the \( x \)-axis uses the weights \( (t, 1 - t) \) but the chord's height uses \( (1 - t, t) \), so the code compares the chord's height at one place with the curve's height at another. For \( a = -1, b = 3, t = 0.25 \): the chord height is \( 0.75 \cdot 1 + 0.25 \cdot 9 = 3 \), the correct point is \( x = 0.75 \cdot (-1) + 0.25 \cdot 3 = 0 \) (curve height 0, gap 3), but the buggy \( x = 0.25 \cdot (-1) + 0.75 \cdot 3 = 2 \) has curve height 4, so it reports a gap of \( 3 - 4 = -1 \).

    ```python
    import numpy as np

    def chord_gap(f, a, b, t):
        x = (1 - t) * a + t * b                       # fixed: same weights as the chord
        return (1 - t) * f(a) + t * f(b) - f(x)

    def looks_convex(f, lo, hi, trials=100_000, seed=53):
        rng = np.random.default_rng(seed)
        a = rng.uniform(lo, hi, trials)
        b = rng.uniform(lo, hi, trials)
        t = rng.uniform(0, 1, trials)
        worst = chord_gap(f, a, b, t).min()
        return bool(worst >= -1e-9), round(float(worst), 4)

    print(looks_convex(lambda x: x**2, -2, 2))
    print(chord_gap(lambda x: x**2, -1, 3, 0.25))
    print(looks_convex(lambda x: x**4 - 3 * x**2 + x, -2, 2))
    # (True, 0.0)
    # 3.0
    # (False, -2.2451)
    ```

    After the fix \( x^2 \) passes and the two-valley function still fails, as it should. The bug only showed up because we tested the checker on a function whose answer we knew; always try a new checker on a known yes and a known no.

    **8.** For one hidden unit, flipping its incoming weights and bias flips its pre-activation \( z \) to \( -z \), so its output becomes \( \tanh(-z) = -\tanh(z) \); flipping its outgoing weight multiplies by −1 again, and the network's output is unchanged.

    ```python
    import math
    import numpy as np

    rng = np.random.default_rng(7)
    Xn = rng.normal(size=(200, 2))
    W1, b1, w2 = rng.normal(size=(2, 3)), rng.normal(size=3), rng.normal(size=3)

    def net(W1, b1, w2):
        return np.tanh(Xn @ W1 + b1) @ w2

    y_teacher = net(W1, b1, w2)

    def net_loss(W1, b1, w2):
        return np.mean((net(W1, b1, w2) - y_teacher) ** 2)

    signs = np.array([1., -1., 1.])                   # flip hidden unit 1
    W1f, b1f, w2f = W1 * signs, b1 * signs, w2 * signs
    print(net_loss(W1, b1, w2), net_loss(W1f, b1f, w2f))
    print(net_loss((W1 + W1f) / 2, (b1 + b1f) / 2, (w2 + w2f) / 2))
    for h in [3, 16, 128]:
        print(h, math.factorial(h) * 2**h if h < 20 else f"{math.factorial(h) * 2**h:.3e}")
    # 0.0 0.0
    # 0.7258981155750354
    # 3 48
    # 16 1371195958099968000
    # 128 1.312e+254
    ```

    `W1 * signs` broadcasts the sign vector across the rows of `W1` (Day 9), so it flips column 1, the incoming weights of hidden unit 1. Both settings have loss exactly 0; their midpoint has unit 1's weights all equal to 0, so that unit outputs nothing and the loss rises to 0.726: another failed chord test. Each unit can be flipped or not (\( 2^h \) choices) and the units can be relabelled (\( h! \) orders), giving \( 2^h \cdot h! \) equivalent settings, about \( 10^{254} \) for a layer of 128. (`math.factorial(h)` is \( h! \); Python integers never overflow, so the exact 19-digit count for \( h = 16 \) is printed in full, and the `:.3e` spec shows the huge one in scientific notation.) Every minimum of a neural network comes with this enormous family of identical copies.

## Resources

- [Boyd & Vandenberghe: Convex Optimization (free book PDF and course)](https://web.stanford.edu/~boyd/cvxbook/): chapters 2 and 3 define convex sets and functions with many pictures; the classic reference.
- [Mathematics for Machine Learning (free book)](https://mml-book.github.io/): chapter 7, "Continuous Optimization", covers gradient descent, convex sets and functions, and the Hessian test.
- [Dive into Deep Learning: Optimization and Deep Learning](https://d2l.ai/chapter_optimization/optimization-intro.html): local minima, saddle points and vanishing gradients with plots; the next page, [Convexity](https://d2l.ai/chapter_optimization/convexity.html), proves the "local minimum is global" fact.
- [Khan Academy: Multivariable calculus](https://www.khanacademy.org/math/multivariable-calculus): the "Applications of multivariable derivatives" unit covers the second partial derivative test and saddle points.
- [Dauphin et al. (2014): Identifying and attacking the saddle point problem in high-dimensional non-convex optimization](https://arxiv.org/abs/1406.2572): the paper behind "saddles, not local minima, are the problem".
- [Li et al. (2018): Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913): pictures of real loss surfaces, and how residual connections smooth them.
- [Android reference: `TimeInterpolator`](https://developer.android.com/reference/android/animation/TimeInterpolator) and [`AccelerateInterpolator`](https://developer.android.com/reference/android/view/animation/AccelerateInterpolator): the easing curves from the Android bridge.

## Checklist

- [ ] I can state the chord test, \( f((1-t)a + tb) \le (1-t)f(a) + t f(b) \), and use `chord_gap` to disprove convexity with one chord.
- [ ] I can compute \( f'' \) by hand and with `second_diff`, and use \( f'' \ge 0 \) to find where a function is convex.
- [ ] I can explain why, for a convex function, zero slope means global minimum (the tangent line lies below the curve).
- [ ] I can build a Hessian and read minimum / maximum / saddle from the signs of its eigenvalues.
- [ ] I showed that MSE (\( H = \frac{2}{n}X^\top X \)) and logistic cross-entropy are convex on the Play Store data, and saw how a duplicated feature makes a flat valley.
- [ ] I can tell a local minimum, a saddle point and a plateau apart, and describe what gradient descent does near each.
- [ ] I can explain why neural network losses are not convex, and give honest reasons why training them works anyway.

**Next:** [Day 54: Entropy, cross-entropy and KL divergence](day-54-information-theory.md)
