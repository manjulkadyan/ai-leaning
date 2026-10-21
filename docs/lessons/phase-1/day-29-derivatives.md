# Day 29: Derivatives as Slopes and Rates of Change

**Phase 1 · Week 5 · ~120 minutes**

## Objectives

- [ ] Compute the slope of a straight line from two points, and say what it means in words.
- [ ] Explain how secant lines through two nearby points of a curve close in on the tangent line, and why that limit is the derivative.
- [ ] Read the notations \( f'(x) \), \( \frac{df}{dx} \) and \( \frac{d}{dx} \), and tell the derivative *function* from its value at one point.
- [ ] Derive the derivative of \( x^2 \) from the definition, by hand.
- [ ] Use the derivative as a sensitivity: "nudge the input by \( \Delta x \), the output moves by about \( f'(x)\,\Delta x \)", and connect it to how a model's weight is changed during training.
- [ ] Write forward and central difference functions, and explain with a real log-log plot why the step \( h \) can be too big *and* too small.
- [ ] Estimate velocity from sampled positions with `np.gradient`, and relate it to Android's `VelocityTracker`.

**Prerequisites.** [Day 4](day-04-numpy-arrays.md) (NumPy arrays and vectorized math), [Day 19](day-19-matplotlib.md) (plotting with Matplotlib) and [Day 25](day-25-projections-pca.md) (least squares: a model "fits" by making an error as small as possible). No calculus is assumed: today starts from zero.

---

## The big idea

Month 1 gave you data, vectors and matrices. Month 2 is about how a model **learns**, and learning is a loop: make a prediction, measure how wrong it is with one number (the **loss**), then change each weight a little in the direction that makes the loss smaller. The question at the heart of that loop is:

> If I change this one weight a tiny bit, how much does the loss change, and in which direction?

That number is a **derivative**. Everything in the next five days (rules, the chain rule, gradients, backpropagation) is machinery for computing it quickly for millions of weights at once.

**Analogy.** A car's odometer tells you *where* you are: 12.0 km, 12.4 km, 12.8 km. Its speedometer tells you *how fast that number is changing right now*: 72 km/h. The odometer reading is a function of time; the speedometer reading is its derivative. Notice what the speedometer shows: not an average over the whole trip, but the rate **at this instant**. Today's job is to make "rate at an instant" precise, because an instant has no duration, and "distance divided by zero time" makes no sense. The trick is to take shorter and shorter time windows and see what the average speed settles on.

| Question | Derivative answer |
|---|---|
| How fast is the car going right now? | Derivative of position with respect to time |
| How fast is the finger moving across the screen? | Derivative of touch position with respect to time (`VelocityTracker`) |
| If I increase weight \( w \) a little, does the loss go up or down? | The sign of the derivative of the loss with respect to \( w \) |
| How strongly does the output react to this input? | The size of the derivative: the **sensitivity** |

## Slope of a straight line

Start with something you already know. A straight line rises at a constant rate. Its **slope** is "how much \( y \) goes up for each 1 that \( x \) goes right":

\[
m = \frac{\Delta y}{\Delta x} = \frac{y_2 - y_1}{x_2 - x_1}
\]

Symbol by symbol:

- \( (x_1, y_1) \) and \( (x_2, y_2) \) are any two points on the line.
- \( \Delta \) (the Greek capital "delta") means "change in". \( \Delta x = x_2 - x_1 \) is the **run**, \( \Delta y = y_2 - y_1 \) is the **rise**.
- \( m \) is the slope: rise divided by run.

**Worked example.** The line \( y = 3x + 1 \) passes through \( (0, 1) \) and \( (2, 7) \). Rise \( = 7 - 1 = 6 \), run \( = 2 - 0 = 2 \), slope \( = 6 / 2 = 3 \). Pick any other two points, say \( (5, 16) \) and \( (10, 31) \): \( 15 / 5 = 3 \) again. For a line the answer never depends on which points you pick, and it equals the number in front of \( x \).

```python
import numpy as np

def slope(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1)

slope(0, 1, 2, 7)       # 3.0    the line y = 3x + 1
slope(5, 16, 10, 31)    # 3.0    same line, other points, same slope
slope(0, 10, 4, 2)      # -2.0   negative: the line goes down as x grows
```

Slope has **units**: units of \( y \) per unit of \( x \). Metres against seconds gives metres per second, a speed. Loss against a weight gives "loss per unit of weight". Keep the units in mind; they tell you what a derivative means.

## Slope of a curve: secants close in on a tangent

A curve like \( f(x) = x^2 \) has no single slope: it is flat near 0 and steep further out. So ask a sharper question: **what is the slope at one point**, say at \( P = (1, 1) \)?

You can't do rise over run with a single point. But you can take a second point \( Q \) a little to the right, at \( x = 1 + h \), where \( h \) is a small step. The straight line through \( P \) and \( Q \) is called a **secant** line, and its slope is ordinary rise over run:

\[
\text{slope of } PQ = \frac{f(1 + h) - f(1)}{h}
\]

(The run is \( (1 + h) - 1 = h \); the rise is the change in \( f \).)

**Worked example** with \( f(x) = x^2 \):

| \( h \) | \( Q = (1 + h,\ (1 + h)^2) \) | rise \( f(1+h) - f(1) \) | slope \( = \) rise / \( h \) |
|---|---|---|---|
| 2 | (3, 9) | 8 | 4 |
| 1 | (2, 4) | 3 | 3 |
| 0.5 | (1.5, 2.25) | 1.25 | 2.5 |
| 0.1 | (1.1, 1.21) | 0.21 | 2.1 |
| 0.01 | (1.01, 1.0201) | 0.0201 | 2.01 |

As \( Q \) slides towards \( P \), the slopes settle on **2**. The line they settle on, the one that just touches the curve at \( P \) and points the way the curve is heading there, is the **tangent line**. Its slope is what we mean by "the slope of the curve at \( P \)".

<figure class="diagram" markdown>
![Animation: the parabola y = x squared with a fixed blue point P at (1, 1); an orange point Q starts at (3, 9) and jumps closer to P in steps h = 2, 1, 0.5, 0.25 and 0.1, each time drawing the orange secant line through P and Q and a dashed rise-and-run triangle, while a table on the right lists the slopes 4, 3, 2.5, 2.25 and 2.1; finally a red tangent line with slope 2 appears and the table reads h to 0: the tangent, slope 2](../../assets/img/phase-1/day29-secant-to-tangent.svg)
<figcaption>Animated: as h shrinks, Q slides towards P and the secant's slope (4, 3, 2.5, 2.25, 2.1, …) closes in on the tangent's slope, 2.</figcaption>
</figure>

```python
def f(x):
    return x ** 2

for h in [1, 0.1, 0.01, 0.001, 0.0001]:
    s = (f(1 + h) - f(1)) / h
    print(f"h = {h:<6}  secant slope = {s:.6f}")
# h = 1       secant slope = 3.000000
# h = 0.1     secant slope = 2.100000
# h = 0.01    secant slope = 2.010000
# h = 0.001   secant slope = 2.001000
# h = 0.0001  secant slope = 2.000100
```

Look at the pattern: the slope is always exactly \( 2 + h \). The next section shows why.

## The derivative, as a limit

**Definition.** The **derivative** of \( f \) at \( x \) is the number the secant slopes settle on as \( h \) shrinks towards 0:

\[
f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
\]

Reading it:

- \( f'(x) \), said "f prime of x", is the derivative of \( f \) at the point \( x \): the slope of the tangent there.
- \( \frac{f(x + h) - f(x)}{h} \) is the secant slope you just computed: rise over a run of \( h \).
- \( \lim_{h \to 0} \), said "the limit as h goes to 0", means: *don't* put \( h = 0 \) in (that would be \( \frac{0}{0} \), which means nothing). Instead, watch what the expression gets closer and closer to as \( h \) gets smaller and smaller, from either side. That target value is the limit.

That's all a limit is, for this course. (Mathematicians have a fully rigorous version, the "epsilon-delta" definition; you won't need it to train models.)

### The derivative of \( x^2 \), from the definition

Now do the algebra once, for any \( x \), not just \( x = 1 \). Expand \( (x + h)^2 = x^2 + 2xh + h^2 \):

\[
\frac{f(x + h) - f(x)}{h}
= \frac{(x + h)^2 - x^2}{h}
= \frac{x^2 + 2xh + h^2 - x^2}{h}
= \frac{2xh + h^2}{h}
= 2x + h
\]

The division by \( h \) is allowed because \( h \) is small but **not** zero. Now let \( h \) shrink: \( 2x + h \) heads to \( 2x \). So

\[
\text{if } f(x) = x^2, \text{ then } f'(x) = 2x
\]

At \( x = 1 \) that's \( 2 \), matching the table. And it explains the pattern above: at \( x = 1 \) the secant slope was \( 2 \cdot 1 + h = 2 + h \), exactly.

```python
h = 1e-6
for x in [-2, 0, 1, 3]:
    print(x, round((f(x + h) - f(x)) / h, 4), 2 * x)    # tiny-h secant vs the formula 2x
# -2 -4.0 -4
# 0 0.0 0
# 1 2.0 2
# 3 6.0 6
```

### Notation you'll see everywhere

The same idea has several spellings. For \( y = f(x) = x^2 \), all of these mean "the derivative is \( 2x \)":

| Notation | Read as | Who uses it |
|---|---|---|
| \( f'(x) = 2x \) | "f prime of x" | Lagrange's notation; short, good for one variable |
| \( \frac{df}{dx} = 2x \) or \( \frac{dy}{dx} = 2x \) | "d f d x" | Leibniz's notation; looks like rise over run, \( \frac{\Delta y}{\Delta x} \), with an infinitely small run |
| \( \frac{d}{dx}\, x^2 = 2x \) | "d by d x of x squared" | \( \frac{d}{dx} \) is an *operation*: "take the derivative of what follows" |
| \( \frac{dL}{dw} \) | "d L d w" | ML papers: derivative of the loss \( L \) with respect to a weight \( w \) |

The \( \frac{d}{dx} \) style is the most useful in ML, because it names the variable you're changing (\( x \), or \( w \)). Note that \( \frac{df}{dx} \) is **not** a fraction you can cancel; it's one symbol for one number (although Day 31's chain rule will make it behave suspiciously like one).

!!! warning "Common mistake: \( f'(x) \) the function versus \( f'(1) \) the number"
    \( f'(x) = 2x \) is a **function**: give it any \( x \) and it returns the slope there. \( f'(1) = 2 \) is **one number**: the slope at \( x = 1 \). When someone says "the derivative is 2", ask "at which point?". Also, \( f'(1) \) means "differentiate, *then* put in 1", never "put in 1, then differentiate": \( f(1) = 1 \) is a constant, whose derivative is 0.

## The derivative is a function too

Since \( f'(x) \) gives a slope for every \( x \), you can plot it as its own curve. Below, \( f(x) = x^3 - x \) and its derivative \( f'(x) = 3x^2 - 1 \) (tomorrow's rules give this in one line; you can check it with the tiny-\( h \) secant trick above). Each red segment on top is a tangent; its slope is the height of the orange curve below at the same \( x \).

<figure class="diagram" markdown>
![Two stacked plots sharing an x-axis from -1.6 to 1.6: on top, the S-shaped curve x cubed minus x with short red tangent segments at five points, steep at the ends and flat at the hump near x = -0.577 and the dip near x = 0.577; below, the U-shaped derivative 3x squared minus 1, which is positive where the top curve rises, negative where it falls, and crosses zero at exactly the two flat points, marked by teal dashed lines](../../assets/img/phase-1/day29-function-and-derivative.svg)
<figcaption>Where f rises, f′ is positive; where f falls, f′ is negative; at the hump and the dip the tangent is flat and f′ = 0.</figcaption>
</figure>

Read three facts off the picture, because training models depends on all three:

1. **The sign of \( f' \) says which way is uphill.** \( f' > 0 \): increasing \( x \) increases \( f \). \( f' < 0 \): increasing \( x \) decreases \( f \).
2. **The size of \( f' \) says how steep.** Near \( x = \pm 1.6 \) the slope is about 6.7; near 0 it's about −1.
3. **At a lowest or highest point the slope is 0.** The dip at \( x = 1/\sqrt{3} \approx 0.577 \) is where \( f' \) crosses zero. A model's best weights sit at the bottom of a loss curve, where the derivative is 0, so "find where the derivative is zero" (or "walk downhill until the slope is flat") is how you find them.

Here is the code that drew the figure:

```python
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "savefig.facecolor": "white"})

xs = np.linspace(-1.6, 1.6, 200)
def g(x):
    return x ** 3 - x
def dg(x):
    return 3 * x ** 2 - 1

fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.5, 6), sharex=True)
top.plot(xs, g(xs), color="#3949ab", lw=2.5, label="f(x) = x³ − x")
for a in [-1.2, -0.577, 0.0, 0.577, 1.2]:
    seg = np.linspace(a - 0.35, a + 0.35, 2)
    top.plot(seg, g(a) + dg(a) * (seg - a), color="#e53935", lw=2)   # tangent: f(a) + f'(a)(x − a)
    top.plot(a, g(a), "o", color="#e53935", ms=5)
top.set_ylabel("f(x)")
top.set_title("A function (top) and its derivative (bottom)")
top.legend(loc="upper left")
bottom.plot(xs, dg(xs), color="#ef6c00", lw=2.5, label="f′(x) = 3x² − 1  (slope of f)")
bottom.axhline(0, color="#555", lw=1)
for a in [-0.577, 0.577]:
    for ax in (top, bottom):
        ax.axvline(a, color="#00897b", ls="--", lw=1)
bottom.plot([-0.577, 0.577], [0, 0], "o", color="#00897b", ms=6)
bottom.text(0, 0.25, "f′ = 0: flat tangent\n(peak or valley of f)", ha="center", color="#00897b")
bottom.set_xlabel("x")
bottom.set_ylabel("f′(x)")
bottom.legend(loc="upper center")
for ax in (top, bottom):
    ax.grid(alpha=0.3)
fig.savefig("day29-function-and-derivative.svg", format="svg", bbox_inches="tight")
plt.close(fig)
```

The tangent at \( x = a \) is the line through \( (a, f(a)) \) with slope \( f'(a) \), which is the point-slope form \( y = f(a) + f'(a)\,(x - a) \). That one formula drew all five red segments.

!!! note "Where there is no derivative"
    The secant slopes have to settle on **one** number from both sides. For \( |x| \) (absolute value) at \( x = 0 \), secants from the right all have slope \( +1 \) and secants from the left all have slope \( -1 \): there's a corner, and no single tangent. The ReLU activation, \( \max(0, x) \), has the same kind of corner at 0. Tomorrow you'll see what deep-learning libraries do about it (they just pick a value).

## The derivative as sensitivity: nudge the input

Here is the reading of the derivative you will use most. Zoom in on any smooth curve far enough and it looks like a straight line, its tangent:

<figure class="diagram" markdown>
![Animation in three panels: the parabola y = x squared around x = 1 with its dashed red tangent line; the first panel spans x from 0 to 2 and the curve clearly bends away from the tangent; an orange box zooms in five times to x from 0.8 to 1.2, where the gap is small; another box zooms in five times more, to x from 0.96 to 1.04, where curve and tangent are indistinguishable; a yellow box then explains that close up f behaves like a line of slope 2, so a nudge of 0.01 in x moves y by about 0.02 (exactly 0.0201)](../../assets/img/phase-1/day29-zoom-to-line.svg)
<figcaption>Animated: zoom in on a smooth curve and it becomes its tangent line. That is why a small nudge Δx moves the output by about f′(x)·Δx.</figcaption>
</figure>

Close to \( x \), the function behaves like a line of slope \( f'(x) \). On a line, the rise is slope times run. So for a small nudge \( \Delta x \):

\[
\Delta f = f(x + \Delta x) - f(x) \;\approx\; f'(x)\,\Delta x
\]

The \( \approx \) ("approximately equal") is there because the curve isn't *exactly* the line; for \( x^2 \) the leftover is \( \Delta x^2 \) (from the algebra above: the change is \( 2x\,\Delta x + \Delta x^2 \)), which for small \( \Delta x \) is tiny compared with \( \Delta x \) itself.

**Worked example.** \( f(x) = x^2 \), nudge by \( \Delta x = 0.01 \).

- At \( x = 1 \): \( f'(1) = 2 \), so the output should move by about \( 2 \times 0.01 = 0.02 \). Actually: \( 1.01^2 - 1 = 0.0201 \).
- At \( x = 3 \): \( f'(3) = 6 \), so about \( 0.06 \). Actually: \( 3.01^2 - 9 = 0.0601 \).

Same nudge, three times the response. The output is three times as **sensitive** to its input at \( x = 3 \) as at \( x = 1 \).

```python
f(1.01) - f(1), 2 * 1 * 0.01     # (0.020100000000000007, 0.02)
f(3.01) - f(3), 2 * 3 * 0.01     # (0.06009999999999849, 0.06)
```

(The `...007` and `...849` endings are floating-point rounding, the same thing you met on Day 4. They will matter a lot in a moment.)

### Why ML cares: one weight, one loss

Take the smallest possible "model": a prediction \( \hat{y} = w \cdot x \) with a single weight \( w \) (the hat on \( \hat{y} \) means "predicted"). One training example has input \( x = 2 \) and correct answer \( y = 6 \), so the perfect weight is \( w = 3 \). The model's error is the squared difference, the same squared error that least squares minimized on Day 25:

\[
L(w) = (w \cdot x - y)^2 = (2w - 6)^2
\]

Our current guess is \( w = 1 \): prediction 2, loss \( (2 - 6)^2 = 16 \). Which way should \( w \) move, and how far? Ask the derivative:

```python
x_in, y_true = 2.0, 6.0

def loss(w):
    return (w * x_in - y_true) ** 2

w = 1.0
h = 1e-6
dL = (loss(w + h) - loss(w)) / h    # slope of the loss at w = 1, with a tiny secant
loss(w), round(dL, 3)               # (16.0, -16.0)

w_new = w - 0.05 * dL               # step against the slope
round(w_new, 4), round(loss(w_new), 4)   # (1.8, 5.76)
```

\( \frac{dL}{dw} = -16 \) at \( w = 1 \). The minus sign says: increasing \( w \) **decreases** the loss. Its size says the loss is quite sensitive here: a nudge of 0.01 in \( w \) would lower it by about \( 16 \times 0.01 = 0.16 \). So we moved \( w \) a little *against* the slope (\( w - 0.05 \times (-16) = 1.8 \)) and the loss dropped from 16 to 5.76.

Repeat that step and you have **gradient descent**, the algorithm that trains almost every neural network. Its full treatment comes on Day 51; this week builds the derivatives it runs on. The 0.05 is the **learning rate**, the size of the step.

!!! example "Try it: the tangent explorer"
    Pick a function, then **drag** the blue point along the curve (or focus it and use the arrow keys). The red line is the tangent, with slope \( f'(x) \) from the exact formula. The dashed orange line is the secant from \( x \) to \( x + h \): shrink \( h \) with the slider and watch it fold onto the tangent and the error column fall. Things to look for:

    - On \( x^3 - x \), find the two points where the tangent is flat. How close to 0 does the slope get near \( x = \pm 0.58 \)? (It is exactly 0 at \( \pm 1/\sqrt{3} \approx \pm 0.577 \).)
    - On \( \sin x \), compare the slope with the height of \( \cos x \): at \( x = 0 \) the slope is 1, at \( x = 1.57 \) (about \( \pi/2 \)) it's about 0.
    - On the sigmoid, the steepest point is \( x = 0 \), with slope 0.25. Far from 0 the curve is almost flat and the slope is nearly 0. (Tomorrow: why that flatness makes deep sigmoid networks hard to train.)
    - Compare the **forward** (secant) error with the **central** error for the same \( h \). The next section explains the difference.

<div class="tangent-explorer"></div>

## Numerical derivatives: forward and central differences

The tiny-\( h \) secant is a perfectly usable way to compute a derivative with a computer, called a **finite difference**. It needs nothing but the ability to evaluate \( f \), which is why it is the standard way to **check** derivatives you've worked out by hand or that a library computed (Day 33 builds a full gradient checker on it).

There are two common versions:

\[
\text{forward difference: } \frac{f(x + h) - f(x)}{h}
\qquad\qquad
\text{central difference: } \frac{f(x + h) - f(x - h)}{2h}
\]

The forward difference is the secant from \( x \) to \( x + h \). The central difference is the secant from \( x - h \) to \( x + h \): it straddles the point symmetrically, so the run is \( 2h \).

**Worked example, with exact algebra.** For \( f(x) = x^3 \), whose true derivative is \( 3x^2 \) (you'll prove that in Exercise 1). Expanding \( (x \pm h)^3 = x^3 \pm 3x^2 h + 3x h^2 \pm h^3 \):

\[
\text{forward: } \frac{3x^2 h + 3x h^2 + h^3}{h} = 3x^2 + \underbrace{3xh + h^2}_{\text{error}}
\qquad
\text{central: } \frac{6x^2 h + 2h^3}{2h} = 3x^2 + \underbrace{h^2}_{\text{error}}
\]

In the central version, the \( 3xh^2 \) terms cancel because they appear with the same sign on both sides. At \( x = 1 \) with \( h = 0.1 \): forward gives \( 3 + 0.3 + 0.01 = 3.31 \), central gives \( 3 + 0.01 = 3.01 \). The forward error is proportional to \( h \); the central error is proportional to \( h^2 \). Make \( h \) ten times smaller and the forward error shrinks about 10×, the central error about 100×. This holds for any smooth function, not just \( x^3 \).

```python
def forward_diff(f, x, h):
    return (f(x + h) - f(x)) / h

def central_diff(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

def cube(x):
    return x ** 3

forward_diff(cube, 1.0, 0.1), central_diff(cube, 1.0, 0.1)    # (3.310000000000004, 3.0100000000000016)
```

The same experiment on \( \sin x \), whose derivative is \( \cos x \) (tomorrow's table; for now, trust it or check it in the widget):

```python
true = np.cos(1.0)
for h in [0.1, 0.01, 0.001, 0.0001]:
    e_f = abs(forward_diff(np.sin, 1.0, h) - true)
    e_c = abs(central_diff(np.sin, 1.0, h) - true)
    print(f"h = {h:<7} forward error {e_f:.2e}   central error {e_c:.2e}")
# h = 0.1     forward error 4.29e-02   central error 9.00e-04
# h = 0.01    forward error 4.22e-03   central error 9.00e-06
# h = 0.001   forward error 4.21e-04   central error 9.01e-08
# h = 0.0001  forward error 4.21e-05   central error 9.00e-10
```

Each row: forward error ÷10, central error ÷100. This error, which comes from using a secant instead of the true tangent, is called the **truncation error**. Central costs two evaluations of \( f \), like forward (which needs \( f(x) \) and \( f(x + h) \)), and is far more accurate, so it is the default choice for checking.

### How small should \( h \) be?

The obvious idea is "as small as possible". Try it:

```python
for h in [1e-8, 1e-11, 1e-13, 1e-15, 1e-17]:
    print(f"h = {h:.0e}  forward estimate {forward_diff(np.sin, 1.0, h):.10f}")
print(f"true            {true:.10f}")
# h = 1e-08  forward estimate 0.5403023029
# h = 1e-11  forward estimate 0.5403011372
# h = 1e-13  forward estimate 0.5395683900
# h = 1e-15  forward estimate 0.5551115123
# h = 1e-17  forward estimate 0.0000000000
# true            0.5403023059
```

Below about \( 10^{-8} \) the estimate gets **worse**, and at \( 10^{-17} \) it's exactly 0. The culprit is floating point. A Python `float` (NumPy `float64`) keeps about 16 significant digits. The smallest relative gap it can represent, **machine epsilon**, is about \( 2.2 \times 10^{-16} \):

```python
np.finfo(np.float64).eps      # np.float64(2.220446049250313e-16)
1.0 + 1e-16 == 1.0            # True: 1e-16 is below the gap, so it's lost entirely
(1.0 + 1e-15) - 1.0           # 1.1102230246251565e-15   not 1e-15: only ~1 digit survived
```

Here's what goes wrong. With a tiny \( h \), \( f(x + h) \) and \( f(x) \) agree in almost all their 16 digits. Each is rounded in its last digit, an error of about \( 10^{-16} \). Subtracting them cancels the matching digits and leaves mostly that rounding noise (this is called **catastrophic cancellation**), and then dividing by a tiny \( h \) blows the noise up. At \( h = 10^{-17} \), \( 1 + h \) rounds to exactly 1, so the rise is 0.

So there are **two** errors pulling in opposite directions:

- **Truncation error** (secant isn't the tangent): grows with \( h \). About \( h \) for forward, \( h^2 \) for central.
- **Rounding error** (float noise divided by \( h \)): grows as \( h \) shrinks. About \( \varepsilon / h \), with \( \varepsilon \) ("epsilon") the machine epsilon.

Their sum is smallest somewhere in the middle. Measure the whole curve:

```python
hs = np.logspace(-16, 0, 161)          # h from 1e-16 to 1, evenly spaced on a log scale
err_fwd = np.abs((np.sin(1.0 + hs) - np.sin(1.0)) / hs - np.cos(1.0))
err_ctr = np.abs((np.sin(1.0 + hs) - np.sin(1.0 - hs)) / (2 * hs) - np.cos(1.0))
best_f, best_c = hs[np.argmin(err_fwd)], hs[np.argmin(err_ctr)]
print(f"forward: best h {best_f:.1e}, error {err_fwd.min():.1e}")
print(f"central: best h {best_c:.1e}, error {err_ctr.min():.1e}")
# forward: best h 2.5e-09, error 2.2e-09
# central: best h 3.2e-06, error 6.0e-13
```

<figure class="diagram" markdown>
![Log-log plot of the error of the numerical derivative of sin x at x = 1 against the step size h from 1e-16 to 1: both the forward (orange) and central (blue) curves are jagged and high on the left, where rounding error wins, and fall as h grows; the forward curve bottoms out near h = 1e-8 at an error of about 1e-9 and then rises as a straight line of slope 1; the central curve bottoms out near h = 1e-5 to 1e-6 at an error near 1e-12 and then rises as a steeper straight line of slope 2](../../assets/img/phase-1/day29-step-size-error.svg)
<figcaption>Real measurement: a V-shaped error. Left of each minimum, floating-point rounding wins; right of it, the secant-versus-tangent truncation error wins. Central differences reach a ~1000× smaller error, at a larger h.</figcaption>
</figure>

```python
fig, ax = plt.subplots(figsize=(6.5, 4.3))
ax.loglog(hs, err_fwd, color="#ef6c00", lw=2, label="forward difference")
ax.loglog(hs, err_ctr, color="#3949ab", lw=2, label="central difference")
ax.axvline(best_f, color="#ef6c00", ls=":", lw=1.2)
ax.axvline(best_c, color="#3949ab", ls=":", lw=1.2)
ax.text(1e-13, 1e-12, "rounding error wins\n(h too small)", ha="center", color="#555")
ax.text(5e-3, 1e-13, "truncation error wins\n(h too big)", ha="center", color="#555")
ax.set_xlabel("step size h (log scale)")
ax.set_ylabel("|estimate − cos(1)| (log scale)")
ax.set_title("Error of the numerical derivative of sin x at x = 1")
ax.set_ylim(1e-14, 10)
ax.grid(alpha=0.3, which="major")
ax.legend(loc="upper center")
fig.savefig("day29-step-size-error.svg", format="svg", bbox_inches="tight")
plt.close(fig)
```

On a log-log plot a power law is a straight line, and its steepness is the power: the right-hand sides rise with slope 1 (forward, error \( \propto h \)) and slope 2 (central, error \( \propto h^2 \)), exactly the algebra above. Setting the two errors equal gives the rule of thumb for where the bottom of the V is:

| Method | Best \( h \) (about) | Best error (about) |
|---|---|---|
| Forward, float64 | \( \sqrt{\varepsilon} \approx 1.5 \times 10^{-8} \) | \( 10^{-8} \) to \( 10^{-9} \) |
| Central, float64 | \( \varepsilon^{1/3} \approx 6 \times 10^{-6} \) | \( 10^{-11} \) to \( 10^{-12} \) |

The exact best \( h \) depends on the function and on \( x \), but the orders of magnitude hold. This is why this course's `central_diff` uses **\( h = 10^{-5} \)** by default, and why its results agree with exact derivatives to about 10 digits, not 16. (If your inputs are huge, say \( x = 10^6 \), scale \( h \) with them: \( h = 10^{-5} \cdot \max(1, |x|) \).)

!!! warning "Common mistake: numerical derivatives in float32"
    Neural networks usually run in `float32`, whose epsilon is about \( 1.2 \times 10^{-7} \), a billion times coarser. The same formulas then lose most of their digits:

    ```python
    np.finfo(np.float32).eps      # np.float32(1.1920929e-07)
    x32 = np.float32(1.0)
    for h in [1e-2, 1e-3, 1e-5]:
        h32 = np.float32(h)
        est = (np.sin(x32 + h32) - np.sin(x32 - h32)) / (2 * h32)
        print(f"h = {h:.0e}  float32 central estimate {est:.7f}  error {abs(float(est) - np.cos(1.0)):.1e}")
    # h = 1e-02  float32 central estimate 0.5402923  error 1.0e-05
    # h = 1e-03  float32 central estimate 0.5403161  error 1.4e-05
    # h = 1e-05  float32 central estimate 0.5394220  error 8.8e-04
    ```

    The best you get is about 5 correct digits, and the "safe" float64 step \( 10^{-5} \) is now one of the worst. Rule: **do gradient checks in float64.** (PyTorch's `torch.autograd.gradcheck` warns you if its inputs aren't double precision, for this reason.)

!!! warning "Common mistake: thinking libraries train with finite differences"
    They don't. A finite difference needs one or two extra evaluations of the whole model **per weight**. A model with a million weights would need a million forward passes for a single training step, and every estimate would carry the errors above. Real frameworks use **automatic differentiation** (Day 31's chain rule, built into a tiny engine on Day 34): exact derivatives for all weights in about the cost of one extra pass. Finite differences are for *checking* those derivatives on small cases.

## Velocity from sampled positions

Real data rarely comes as a formula. A touch screen or a sensor gives you **samples**: a position at each timestamp. The derivative of position with respect to time is velocity, and you estimate it with finite differences on neighbouring samples, where \( h \) is now the time between samples, fixed by the hardware.

`np.gradient(values, times)` does this for a whole array: central differences in the interior, and one-sided (forward or backward) differences at the two ends, so the output has the same length as the input. `np.diff(values) / dt` gives forward differences, one fewer value.

Simulate a finger accelerating across the screen, sampled every 16 ms (a 60 Hz screen), with half a pixel of sensor noise:

```python
dt = 0.016                                   # a sample every 16 ms
t = np.arange(0, 0.2, dt)                    # 13 timestamps
x_true = 300 * t + 2000 * t ** 2             # finger position in px: speeding up
v_true = 300 + 4000 * t                      # its true velocity in px/s
rng = np.random.default_rng(29)
x_meas = x_true + rng.normal(0, 0.5, size=t.size)   # sensor noise: about ±0.5 px

v_diff = np.diff(x_meas) / dt                # forward differences: one fewer value
v_grad = np.gradient(x_meas, t)              # central inside, one-sided at the ends
t.size, v_diff.size, v_grad.size             # (13, 12, 13)
v_true[5:9].round(1)                         # array([620., 684., 748., 812.])
v_grad[5:9].round(1)                         # array([623.7, 686.8, 752.2, 802.5])
np.abs(v_grad - v_true)[1:-1].max().round(1) # np.float64(20.5)   worst interior error, px/s
```

(For a quadratic like this one, central differences have zero truncation error, as you saw with \( h^2 \) terms cancelling; all the error here is noise.)

Now the step-size trade-off shows up in a new costume. Half a pixel of noise, divided by the time step, becomes velocity noise. A faster sensor gives *smaller* \( h \), so the noise gets *amplified more*:

```python
for dt_ in [0.016, 0.004, 0.001]:
    t_ = np.arange(0, 0.2, dt_)
    xm = 300 * t_ + 2000 * t_ ** 2 + rng.normal(0, 0.5, size=t_.size)
    err = np.abs(np.gradient(xm, t_) - (300 + 4000 * t_))[1:-1]
    print(f"dt = {dt_ * 1000:>4.0f} ms   typical velocity error {np.median(err):6.1f} px/s")
# dt =   16 ms   typical velocity error   17.9 px/s
# dt =    4 ms   typical velocity error   72.7 px/s
# dt =    1 ms   typical velocity error  222.4 px/s
```

Same shape as the floating-point story: noise in the values, divided by a small step, dominates. Measurement noise plays the role that rounding played above. The fix is the same idea too: don't trust the difference of just two neighbouring samples; use more of them.

!!! android "Android bridge: velocity from touch samples"
    Every `MotionEvent` carries a position (`getX()`, `getY()`) and a timestamp (`getEventTime()`, in milliseconds). The naive fling speed is a forward difference over the last two events:

    ```kotlin
    private val tracker = VelocityTracker.obtain()   // call tracker.recycle() when done
    private var lastX = 0f
    private var lastT = 0L

    override fun onTouchEvent(e: MotionEvent): Boolean {
        if (e.actionMasked == MotionEvent.ACTION_DOWN) tracker.clear()
        tracker.addMovement(e)
        if (e.actionMasked == MotionEvent.ACTION_MOVE && e.eventTime > lastT) {
            val naive = (e.x - lastX) / ((e.eventTime - lastT) / 1000f)   // Δx / Δt, px per second
            tracker.computeCurrentVelocity(1000)                          // 1000: report px per 1000 ms
            Log.d("vel", "naive=$naive  tracker=${tracker.xVelocity}")
        }
        lastX = e.x
        lastT = e.eventTime
        return true
    }
    ```

    Log both while you swipe and the naive value jumps around, exactly like the 1 ms row above: a 1-pixel jitter over an 8 ms gap is 125 px/s of fake velocity. (The `eventTime > lastT` check avoids dividing by a zero time gap.) `VelocityTracker` is the grown-up version: it keeps a short window of recent samples, including the batched "historical" ones (`getHistoricalX(i)`, `getHistoricalEventTime(i)`), and fits a smooth curve through them rather than differencing two points, which is why the fling velocity you hand to `OverScroller.fling` feels stable. Sensors play the same game in reverse: the accelerometer and gyroscope from `SensorManager` measure **derivatives** directly (acceleration is the derivative of velocity; the gyroscope reports angular velocity, the derivative of an angle), and turning them back into position means undoing a derivative, which lets errors pile up.

## Where this goes next

You now have the one idea the rest of the week is built on: **the derivative is the slope of the tangent, which is the sensitivity of the output to a nudge of the input**, and you can always check one with `central_diff`. Next:

- Day 30: rules that give derivatives in one line (no limits), including the ones ML uses daily (sigmoid, tanh, ReLU).
- Day 31: the chain rule, for functions inside functions, which is what backpropagation is.
- Day 32: functions of many inputs, partial derivatives and the gradient (all the slopes at once).
- Day 33: gradients for vectors and matrices, checked numerically with the central difference from today.

---

## Exercises

1. **By hand, from the definition.** Find the derivative of \( f(x) = 3x + 2 \) and of \( g(x) = x^3 \), using \( \lim_{h \to 0} \frac{f(x + h) - f(x)}{h} \). (Hint: \( (x + h)^3 = x^3 + 3x^2 h + 3x h^2 + h^3 \).) Does the first answer agree with what you know about straight lines?
2. **Guess a derivative from secants.** Compute forward-difference slopes of \( f(x) = 1/x \) at \( x = 2 \) for \( h = 0.1, 0.01, 0.001 \). What number are they closing in on? Is the curve rising or falling there?
3. **Check known values.** With `central_diff`, compute the slope of the sigmoid \( \sigma(x) = \frac{1}{1 + e^{-x}} \) at 0, and of \( e^x \) (`np.exp`) at 0 and at 1. What do you notice about \( e^x \)? Compare with the tangent explorer's sigmoid readout.
4. **Find the bottom of the V yourself.** For \( f = e^x \) at \( x = 1 \) (true derivative \( e \approx 2.71828 \)), evaluate the central-difference error for 111 values of \( h \) from \( 10^{-12} \) to \( 10^{-1} \) (`np.logspace(-12, -1, 111)`). Which \( h \) is best, and how does it compare with \( \varepsilon^{1/3} \)?
5. **Sensitivity and steps.** Using the one-weight `loss` from the lesson (\( x = 2, y = 6 \)), start from \( w = 4 \) instead. Before computing, predict the sign of \( \frac{dL}{dw} \) and which way \( w \) should move. Then take three steps of `w = w - 0.05 * central_diff(loss, w)` and print the loss after each. Why do the steps get smaller on their own?
6. **Velocity from a table.** A car's position is logged every 0.5 s: `[0, 4, 15, 33, 60]` metres at `t = [0, 0.5, 1, 1.5, 2]`. Compute velocities with `np.diff(x) / 0.5` and with `np.gradient(x, t)`. What is each method's estimate of the speed at \( t = 1 \) s, and why is the `np.gradient` one the better guess?

??? success "Solutions"
    **1.** For \( f(x) = 3x + 2 \):

    \[
    \frac{3(x + h) + 2 - (3x + 2)}{h} = \frac{3h}{h} = 3
    \]

    There's no \( h \) left, so the limit is just 3: \( f'(x) = 3 \) everywhere. A straight line's derivative is its slope, the same at every point, as in the first section. For \( g(x) = x^3 \):

    \[
    \frac{x^3 + 3x^2 h + 3x h^2 + h^3 - x^3}{h} = 3x^2 + 3xh + h^2 \;\longrightarrow\; 3x^2 \text{ as } h \to 0
    \]

    So \( g'(x) = 3x^2 \), and \( 3xh + h^2 \) is exactly the forward-difference error you met in the lesson.

    ```python
    # 2
    def recip(x):
        return 1 / x
    for h in [0.1, 0.01, 0.001]:
        print(h, forward_diff(recip, 2.0, h))
    # 0.1 -0.23809523809523836
    # 0.01 -0.24875621890546595
    # 0.001 -0.2498750624687629
    ```
    They close in on \( -0.25 = -1/4 \) (tomorrow's power rule gives \( \frac{d}{dx}\frac{1}{x} = -\frac{1}{x^2} = -\frac{1}{4} \) at \( x = 2 \)). Negative: \( 1/x \) is falling at \( x = 2 \).

    ```python
    # 3
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    central_diff(sigmoid, 0.0)     # np.float64(0.2499999999933111)   the sigmoid's steepest slope, 0.25
    central_diff(np.exp, 0.0)      # np.float64(1.0000000000121023)   = e⁰ = 1
    central_diff(np.exp, 1.0)      # np.float64(2.718281828517632)    ≈ e = 2.718281828459045
    ```
    The slope of \( e^x \) at each point equals \( e^x \) itself: \( e^x \) is its own derivative (Day 30 says why that makes it so useful). Notice the answers agree with the truth to about 10 digits, not 16: that is the bottom of the central-difference V.

    ```python
    # 4
    hs4 = np.logspace(-12, -1, 111)
    errs4 = np.array([abs(central_diff(np.exp, 1.0, h) - np.e) for h in hs4])
    print(f"best h {hs4[np.argmin(errs4)]:.1e}, error {errs4.min():.1e}")
    print(f"eps ** (1/3) = {np.finfo(float).eps ** (1/3):.1e}")
    # best h 7.9e-06, error 3.2e-12
    # eps ** (1/3) = 6.1e-06
    ```
    The best \( h \) is within a factor of 2 of \( \varepsilon^{1/3} \), as the rule of thumb predicts. (The exact winner wobbles from run to run of \( h \) values because rounding noise is jagged, as the plot showed.)

    ```python
    # 5: at w = 4 the prediction is 8, too big, so increasing w makes the loss worse:
    # dL/dw > 0, and w should move down, towards 3.
    w = 4.0
    for step in range(1, 4):
        d = central_diff(loss, w)
        w = w - 0.05 * d
        print(f"step {step}: slope {d:.4f} -> w = {w:.4f}, loss {loss(w):.4f}")
    # step 1: slope 8.0000 -> w = 3.6000, loss 1.4400
    # step 2: slope 4.8000 -> w = 3.3600, loss 0.5184
    # step 3: slope 2.8800 -> w = 3.2160, loss 0.1866
    ```
    The step is learning rate × slope, and the slope shrinks as \( w \) approaches the bottom of the loss curve at \( w = 3 \), where the slope is 0. So the steps shrink automatically: \( 0.4, 0.24, 0.144 \).

    ```python
    # 6
    t6 = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    x6 = np.array([0.0, 4.0, 15.0, 33.0, 60.0])
    np.diff(x6) / 0.5          # array([ 8., 22., 36., 54.])
    np.gradient(x6, t6)        # array([ 8., 15., 29., 45., 54.])
    ```
    `np.diff` gives 4 average speeds, one per *interval*: the value 22 m/s is the average over 0.5–1 s, and 36 over 1–1.5 s. Read as "the speed at \( t = 1 \)" a forward difference says 36 and a backward difference says 22. `np.gradient` gives 29 m/s at \( t = 1 \): the central difference \( (33 - 4) / 1.0 \), which is the average of 22 and 36. It straddles the moment symmetrically, so (as with \( h^2 \) versus \( h \)) it's the more accurate estimate at that instant.

## Resources

- [3Blue1Brown: The essence of calculus (Essence of Calculus, ch. 1)](https://www.youtube.com/watch?v=WUvTyaaNkzM) and [The paradox of the derivative (ch. 2)](https://www.youtube.com/watch?v=9vKqVkMQHKk): the clearest visual explanation of "rate at an instant". Watch chapter 2 after today's lesson.
- [Khan Academy: Differential calculus](https://www.khanacademy.org/math/differential-calculus): the "Limits" and "Derivatives: definition and basic rules" units, with practice problems and worked videos.
- [Dive into Deep Learning, 2.4 Calculus](https://d2l.ai/chapter_preliminaries/calculus.html): derivatives, then gradients, written for ML with runnable code.
- [Mathematics for Machine Learning (free book), ch. 5 "Vector Calculus"](https://mml-book.github.io/): section 5.1 defines the derivative as a limit of difference quotients, exactly as here, then generalizes it.
- [NumPy docs: `numpy.gradient`](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html): central differences on sampled data, including uneven spacing.
- [Wikipedia: Numerical differentiation](https://en.wikipedia.org/wiki/Numerical_differentiation): forward and central differences and the step-size trade-off, with the \( \sqrt{\varepsilon} \) rule.
- [Android reference: `VelocityTracker`](https://developer.android.com/reference/android/view/VelocityTracker): `addMovement`, `computeCurrentVelocity`, `getXVelocity`.

## Checklist

- [ ] I can compute a slope as rise over run and say its units.
- [ ] I can explain how secant lines close in on the tangent, and why the derivative is a limit and not "plug in \( h = 0 \)".
- [ ] I can derive \( \frac{d}{dx} x^2 = 2x \) from the definition without looking.
- [ ] I can read \( f'(x) \), \( \frac{df}{dx} \) and \( \frac{dL}{dw} \), and I know \( f'(x) \) is a function while \( f'(1) \) is a number.
- [ ] I can use \( \Delta f \approx f'(x)\,\Delta x \) to predict how much an output moves, and explain why a negative \( \frac{dL}{dw} \) means "increase \( w \)".
- [ ] I can write `forward_diff` and `central_diff`, and explain why \( h \) can be too big and too small, using the V-shaped error plot.
- [ ] I can estimate a velocity from samples with `np.gradient`, and explain why noisy samples with a tiny time step give noisy velocities.

**Next:** [Day 30: Derivative rules you'll actually use](day-30-derivative-rules.md)
